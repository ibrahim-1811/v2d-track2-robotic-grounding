"""Load a retargeted motion_v1 parquet through FlashCHORD's own scene builder.

This is the physics-scene compatibility gate for a reference produced by
``soma_to_g1.py``: it proves FlashCHORD can parse the schema, resolve the object
mesh/URDF, build a Newton model with the G1+Dex3 embodiment and the support
surface, and snap the scene onto reference frame 0 — without training anything.

Run inside the flash_chord container with the reference's asset tree mounted at
the same path recorded in the parquet (see docs/EPISODE_0_G1_RETARGET.md).
"""

from __future__ import annotations

import argparse
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--expect-frames", type=int, default=None)
    parser.add_argument("--control-fps", type=float, default=50.0)
    parser.add_argument(
        "--expect-support-shapes",
        type=int,
        default=None,
        help=(
            "Fail unless the support stage declares exactly this many "
            "Cylinder/Cube collision shapes. The orchestrator passes the count "
            "its own host-side text scan produced, so the scene.support "
            "decision is backed by two independent implementations instead of "
            "one. A disagreement means the decision is unsound and is reported, "
            "not reconciled."
        ),
    )
    parser.add_argument(
        "--include-support",
        choices=("auto", "true", "false"),
        default="auto",
        help=(
            "auto (default) enables the support surface only when the sequence's "
            "reconstructed_stage USDA actually declares Cylinder/Cube collision "
            "shapes. NVIDIA ships empty stubs for sequences with no distinct "
            "support (e.g. mid-air handovers), and FlashCHORD raises on those."
        ),
    )
    return parser.parse_args()


def _resolve_include_support(args, reference) -> tuple[bool, int]:
    """Decide whether this sequence has a usable support surface.

    ``add_support_surfaces`` raises when a reconstructed_stage USDA declares no
    Cylinder/Cube collision shapes. NVIDIA's own corpus contains such stubs
    (2 of 3 whole_body sequences), so 'auto' inspects the stage instead of
    assuming every sequence has a support surface.
    """
    from flash_chord.assets.registry import support_usda_for_reference

    path = support_usda_for_reference(reference)
    if path is None:
        print("  support usda    : none found")
        print("SUPPORT_SHAPES=0")
        return (args.include_support == "true"), 0
    try:
        from pxr import Usd

        stage = Usd.Stage.Open(str(path))
        shapes = [
            prim.GetPath()
            for prim in stage.Traverse()
            if prim.GetTypeName() in ("Cylinder", "Cube")
        ]
    except Exception as error:  # noqa: BLE001 - an unreadable stage is a failure,
        # not a quiet "no support": the count is unknown, so say so.
        print("  support usda    : {} (unreadable: {})".format(path, error))
        print("SUPPORT_SHAPES=-1")
        return (args.include_support == "true"), -1
    count = len(shapes)
    print("  support usda    : {} ({} collision shapes)".format(path, count))
    # Parsed by scripts/run_tier_pipeline.py; keep the exact spelling.
    print("SUPPORT_SHAPES={}".format(count))
    if args.include_support != "auto":
        return args.include_support == "true", count
    return bool(shapes), count


def main() -> int:
    args = parse_args()

    from flash_chord.data import load_reference

    print("== load_reference ==", flush=True)
    reference = load_reference(
        args.parquet,
        control_fps=args.control_fps,
        motion_speed=1.0,
        source_frame_playback=True,
    )
    print("  num_frames      : {}".format(reference.num_frames))
    print("  fps             : {}".format(reference.fps))
    # These are methods on the Reference protocol, not properties.
    print("  object name     : {}".format(reference.object_name()))
    print("  object bodies   : {}".format(list(reference.object_body_names())))
    print("  mesh paths      : {}".format(list(reference.object_mesh_paths())))
    print("  urdf paths      : {}".format(list(reference.object_urdf_paths())))
    print("  mesh radius     : {}".format(reference.object_mesh_radius().tolist()))

    if args.expect_frames is not None and reference.num_frames != args.expect_frames:
        print(
            "FAIL: expected {} frames, got {}".format(
                args.expect_frames, reference.num_frames
            )
        )
        return 1

    # source_frame_playback=True keeps one reference frame per policy step, so
    # the loaded frame count must equal the retargeted frame count exactly.
    print("== build the physics scene ==", flush=True)
    from flash_chord.configuration import instantiate_typed
    from flash_chord.embodiments.base import Embodiment
    from flash_chord.embodiments.g1_dex3 import G1Dex3, G1Dex3Config
    from flash_chord.scene.collision import CollisionPolicy
    from flash_chord.scene.setup import setup_scene

    del instantiate_typed, Embodiment  # imported to mirror the official entrypoints

    include_support, support_shapes = _resolve_include_support(args, reference)
    print("  include_support : {}".format(include_support))
    print("INCLUDE_SUPPORT={}".format(str(include_support).lower()))
    if args.expect_support_shapes is not None and support_shapes != args.expect_support_shapes:
        print(
            "FAIL: the stage declares {} Cylinder/Cube collision shapes, the "
            "caller expected {}. The host-side scan and this USD parse disagree, "
            "so the scene.support decision is not trustworthy.".format(
                support_shapes, args.expect_support_shapes
            )
        )
        return 1

    embodiment = G1Dex3(config=G1Dex3Config())
    collision = CollisionPolicy(
        robot_scope="all",
        hand_self_collision=True,
        robot_object_collision=True,
        robot_support_collision=True,
        ground=True,
    )
    setup = setup_scene(
        parquet=args.parquet,
        control_fps=args.control_fps,
        motion_speed=1.0,
        embodiment=embodiment,
        collision=collision,
        world_count=1,
        include_support=include_support,
        decompose_objects=True,
        source_frame_playback=True,
    )
    scene = setup.scene
    print("  world_count     : {}".format(scene.world_count))
    print("  model bodies    : {}".format(scene.model.body_count))
    print("  model shapes    : {}".format(scene.model.shape_count))
    print("  joint dof       : {}".format(scene.model.joint_dof_count))
    print("  objects         : {}".format([o.name for o in scene.objects]))
    print("  support bound   : {}".format(scene.support is not None))

    print("== snap to reference frame 0 ==", flush=True)
    from flash_chord.runtime.actions import SonicJointResidualActionConfig
    from flash_chord.runtime.object_control import VOCParams
    from flash_chord.runtime.replay import ReplayConfig, ReplayRunner
    from flash_chord.runtime.sim import SimConfig

    # ReplayConfig's defaults target Sharpa's floating-wrist action, which
    # rejects G1 ("requires a 3-DOF position and 3-DOF orientation wrist").
    # Mirror configs/sonic_replay.yaml instead: the G1 SONIC joint residual,
    # the replay sim profile, and ReconBody's direct-gain VOC.
    replay_config = ReplayConfig(
        sim=SimConfig(
            fps=args.control_fps,
            substeps=4,
            solver="newton",
            integrator="implicitfast",
            cone="pyramidal",
            njmax=16384,
            nconmax=4096,
            impratio=1.0,
            iterations=100,
            ls_iterations=50,
            use_mujoco_contacts=True,
        ),
        action=SonicJointResidualActionConfig(),
        voc_scale=1.0,
        voc_params=VOCParams(
            k_lin=50.0, d_lin=10.0, k_ang=10.0, d_ang=0.1,
            max_force=60.0, max_torque=60.0, gravity=9.81,
            mass_normalized_gains=False,
        ),
    )
    runner = ReplayRunner(scene, setup.reference, replay_config, device="cuda:0")
    runner.reset(0)
    runner.command.gather(runner.timestep)
    result = runner.check_termination()
    print("  wrist pos err   : {:.6f} m".format(result.max_wrist_pos))
    print("  object pos err  : {:.6f} m".format(result.max_object_pos))
    print("  object ori err  : {:.6f} rad".format(result.max_object_ori))

    import numpy as np

    body_q = runner.state_0.body_q.numpy()
    nonfinite = int(np.count_nonzero(~np.isfinite(body_q.view(np.float32))))
    print("  non-finite body_q entries after reset: {}".format(nonfinite))
    if nonfinite:
        print("FAIL: scene state contains non-finite values")
        return 1

    print("FLASHCHORD_LOAD=PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
