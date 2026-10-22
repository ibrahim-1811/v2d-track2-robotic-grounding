# Headless FlashCHORD replay harness for the official G1 snack-box example.
#
# Mirrors scripts/debug/replay.py's two Viser buttons exactly, with no browser:
#   --mode reference  == "Show kinematic reference"  (ReplayRunner.reset(f) per frame)
#   --mode replay     == "Run configured replay"     (ReplayRunner.step(f) per frame)
#
# Only official APIs are used (setup_scene, ReplayRunner, check_termination). NVIDIA's
# pinned source is never imported-around, patched, or modified; no integrity check is
# disabled. Run it inside the flash_chord container with /workspace on sys.path.
#
# Beyond the frame loop it records what a viewer cannot: the exact number of frames
# completed, per-frame finiteness of the simulation state, and the official tracking
# termination peaks (wrist/object position + orientation error) for every frame.

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", default="/workspace/src/flash_chord/configs")
    parser.add_argument("--config-name", default="sonic_replay")
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--mode", choices=("reference", "replay"), required=True)
    parser.add_argument(
        "--frames",
        type=int,
        default=-1,
        help="frames to run; -1 (default) runs the whole reference",
    )
    parser.add_argument(
        "--check-every",
        type=int,
        default=1,
        help="run the finiteness + termination readback every N frames (1 = every frame)",
    )
    parser.add_argument("--json-out", default=None)
    parser.add_argument(
        "--trace-csv",
        default=None,
        help="per-frame CSV of pelvis/object height and tracking error (fall diagnosis)",
    )
    return parser.parse_args()


def finite_report(state) -> dict[str, int]:
    """Count non-finite entries in the arrays that define the simulated state."""
    counts = {}
    for name in ("body_q", "body_qd", "joint_q", "joint_qd"):
        array = getattr(state, name, None)
        if array is None:
            continue
        values = array.numpy()
        counts[name] = int(np.count_nonzero(~np.isfinite(values.view(np.float32))))
    return counts


def main() -> int:
    args = parse_args()

    from hydra import compose, initialize_config_dir

    from flash_chord.configuration import instantiate_typed
    from flash_chord.embodiments.base import Embodiment
    from flash_chord.runtime.replay import ReplayConfig, ReplayRunner
    from flash_chord.scene.collision import CollisionPolicy
    from flash_chord.scene.setup import setup_scene

    # The parquet path contains '=' characters (Hive-style sequence_id=... partitions),
    # which Hydra's override grammar would mis-parse, so it is assigned after compose
    # rather than passed as an override string.
    with initialize_config_dir(version_base="1.3", config_dir=args.config_dir):
        cfg = compose(config_name=args.config_name, overrides=["markers.enabled=false"])
    cfg.task.parquet = args.parquet

    embodiment = instantiate_typed(cfg.embodiment, Embodiment)
    collision = instantiate_typed(cfg.collision, CollisionPolicy)
    replay_cfg = instantiate_typed(cfg.replay, ReplayConfig)

    setup = setup_scene(
        parquet=cfg.task.parquet,
        control_fps=replay_cfg.sim.fps,
        motion_speed=cfg.task.motion_speed,
        embodiment=embodiment,
        collision=collision,
        world_count=cfg.scene.world_count,
        include_support=cfg.scene.support,
        decompose_objects=cfg.scene.decompose_objects,
        object_scale_min=cfg.scene.object_scale_min,
        object_scale_max=cfg.scene.object_scale_max,
        object_scale_seed=cfg.scene.object_scale_seed,
        contact_friction=cfg.scene.get("contact_friction", 1.0),
        object_free_joint_damping=cfg.scene.get("object_free_joint_damping", 0.0),
        source_frame_playback=bool(cfg.task.get("source_frame_playback", False)),
        motion_start_frame=int(cfg.task.get("motion_start_frame", 0)),
        motion_end_frame=int(cfg.task.get("motion_end_frame", -1)),
    )
    reference = setup.reference
    runner = ReplayRunner(setup.scene, reference, replay_cfg, device=cfg.device)

    num_frames = reference.num_frames
    requested = num_frames if args.frames < 0 else min(args.frames, num_frames)

    # Same warm-up the Viser path performs before enabling playback, so the timed
    # loop below excludes Warp/MuJoCo kernel compilation and CUDA graph capture.
    print("Compiling replay kernels before the headless loop...", flush=True)
    compile_start = time.perf_counter()
    runner.step(0)
    runner.reset(0)
    compile_seconds = time.perf_counter() - compile_start
    print(
        "Replay kernels are ready; starting headless {} of {} frames.".format(args.mode, requested),
        flush=True,
    )

    voc = float(replay_cfg.voc_scale)
    print(
        "harness: mode={}, frames={}/{}, world_count={}, control={:g} Hz, physics={:g} Hz, "
        "VOC={:g}, action={}, action_dim={}".format(
            args.mode,
            requested,
            num_frames,
            setup.scene.world_count,
            replay_cfg.sim.fps,
            replay_cfg.sim.physics_fps,
            voc,
            type(runner.action).__name__,
            runner.action.action_dim,
        ),
        flush=True,
    )

    # Diagnostic body ids: the robot pelvis (semantic frame) and the first object root.
    pelvis_body_id = None
    for frame_spec in setup.scene.layout.semantic_frames:
        if "pelvis" in frame_spec.name:
            pelvis_body_id = int(frame_spec.body_id)
            break
    object_body_ids = runner.command.body_ids_w.numpy()
    object_body_id = int(object_body_ids[0]) if object_body_ids.size else None
    print(
        "diagnostics: pelvis_body_id={}, object_body_id={}".format(pelvis_body_id, object_body_id),
        flush=True,
    )

    trace_handle = None
    if args.trace_csv:
        trace_handle = open(args.trace_csv, "w")
        trace_handle.write("frame,pelvis_z_m,object_z_m,wrist_pos_err_m,object_pos_err_m\n")

    runner.reset(0)
    completed = 0
    nonfinite_frames = []
    terminated_frames = []
    peaks = {"wrist_pos": 0.0, "wrist_ori": 0.0, "object_pos": 0.0, "object_ori": 0.0}
    object_pos_err = []
    loop_start = time.perf_counter()

    for frame in range(requested):
        if args.mode == "replay":
            runner.step(frame)
        else:
            runner.reset(frame)
            # ReplayRunner.step() gathers the per-frame command targets; reset() only
            # sets runner.timestep and snaps the state. Without this gather the
            # termination readback below would compare frame f's state against the
            # targets last gathered (frame 0) and report the object's travel distance
            # as a tracking error. gather() is the same official call step() makes.
            runner.command.gather(runner.timestep)
        completed = frame + 1

        if args.check_every > 0 and (frame % args.check_every == 0 or completed == requested):
            counts = finite_report(runner.state_0)
            if any(counts.values()):
                nonfinite_frames.append({"frame": frame, "counts": counts})
            result = runner.check_termination()
            peaks["wrist_pos"] = max(peaks["wrist_pos"], result.max_wrist_pos)
            peaks["wrist_ori"] = max(peaks["wrist_ori"], result.max_wrist_ori)
            peaks["object_pos"] = max(peaks["object_pos"], result.max_object_pos)
            peaks["object_ori"] = max(peaks["object_ori"], result.max_object_ori)
            object_pos_err.append(result.max_object_pos)
            if trace_handle is not None:
                body_q = runner.state_0.body_q.numpy()
                pelvis_z = float(body_q[pelvis_body_id][2]) if pelvis_body_id is not None else float("nan")
                object_z = float(body_q[object_body_id][2]) if object_body_id is not None else float("nan")
                trace_handle.write(
                    "{},{:.6f},{:.6f},{:.6f},{:.6f}\n".format(
                        frame, pelvis_z, object_z, result.max_wrist_pos, result.max_object_pos
                    )
                )
            if result.any:
                terminated_frames.append(
                    {
                        "frame": frame,
                        "wrist_done": result.wrist_done,
                        "object_done": result.object_done,
                        "object_pos_err_m": result.max_object_pos,
                        "wrist_pos_err_m": result.max_wrist_pos,
                    }
                )

    loop_seconds = time.perf_counter() - loop_start
    if trace_handle is not None:
        trace_handle.close()
        print("wrote trace " + args.trace_csv, flush=True)

    summary = {
        "mode": args.mode,
        "parquet": str(cfg.task.parquet),
        "reference_frames": int(num_frames),
        "frames_requested": int(requested),
        "frames_completed": int(completed),
        "world_count": int(setup.scene.world_count),
        "control_fps": float(replay_cfg.sim.fps),
        "physics_fps": float(replay_cfg.sim.physics_fps),
        "substeps": int(replay_cfg.sim.substeps),
        "voc_scale": voc,
        "voc_mass_normalized_gains": bool(replay_cfg.voc_params.mass_normalized_gains),
        "action": type(runner.action).__name__,
        "action_dim": int(runner.action.action_dim),
        "kernel_compile_seconds": round(compile_seconds, 3),
        "loop_seconds": round(loop_seconds, 3),
        "frames_per_second": round(completed / loop_seconds, 2) if loop_seconds > 0 else None,
        "nonfinite_frame_count": len(nonfinite_frames),
        "nonfinite_frames": nonfinite_frames[:10],
        "termination_threshold_frame_count": len(terminated_frames),
        "first_termination_frames": terminated_frames[:5],
        "peak_tracking_error": {
            "wrist_position_m": round(peaks["wrist_pos"], 6),
            "wrist_orientation_rad": round(peaks["wrist_ori"], 6),
            "object_position_m": round(peaks["object_pos"], 6),
            "object_orientation_rad": round(peaks["object_ori"], 6),
        },
        "mean_object_position_error_m": (
            round(float(np.mean(object_pos_err)), 6) if object_pos_err else None
        ),
        "final_object_position_error_m": (
            round(float(object_pos_err[-1]), 6) if object_pos_err else None
        ),
    }

    print("=== HARNESS SUMMARY ===", flush=True)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    if args.json_out:
        with open(args.json_out, "w") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
        print("wrote " + args.json_out, flush=True)

    ok = completed == requested and not nonfinite_frames
    print("HARNESS_RESULT=" + ("PASS" if ok else "FAIL"), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
