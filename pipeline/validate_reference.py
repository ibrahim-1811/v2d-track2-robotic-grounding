#!/usr/bin/env python3
"""Validate a retargeted G1 ``motion_v1`` parquet (Step B output).

Originally the body of ``scripts/data/validate_g1_reference.py``, moved here so the
30-episode orchestrator can call the same checks in-process and collect their
results instead of scraping stdout. The script is now a thin CLI over
``run_checks``; its command line and printed output are unchanged.

What it checks, and why each one would otherwise bite later:

1. identity/schema      - a column set that differs from NVIDIA's own reference
                          makes FlashCHORD's reader fail at load time
2. frame counts         - columns that disagree about T desynchronise the policy
                          from the object trajectory
3. joint dimensions     - a width mismatch silently misassigns joint targets
4. finiteness           - one NaN poisons the whole physics step
5. quaternion norms     - a non-unit quaternion is not a rotation
6. grounding/continuity - a teleporting or floating root is not trackable
7. root alignment       - the G1 pelvis must follow the SOMA hips it was
                          retargeted from; a frame-chain mistake shows up here
                          as metres of offset (see the coordinate rule in
                          CLAUDE.md)
8. hand/object          - hands that never reach the object mean the object
                          trajectory and the body are in different frames
9. IK residuals         - compared against NVIDIA's shipped demo, not an
                          absolute threshold we invented, plus a per-episode
                          spike test that catches one broken frame hiding
                          inside an otherwise healthy median
10. kinematic feasibility - a reference is only trainable if the G1 can
                          actually execute it, so every joint step is
                          checked against that joint's own URDF velocity
                          limit
11. mesh/urdf paths     - the parquet bakes absolute paths; they must resolve

Deliberately NOT gated: the raw IK iteration count. Section 9 carries the
measurement that retired that gate.
"""
from __future__ import annotations

import pickle
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pyarrow.parquet as pq

from pipeline.checks import Checker

# Per-frame columns whose outer list length must equal the frame count.
PER_FRAME_COLUMNS = (
    "robot_root_position",
    "robot_root_wxyz",
    "robot_joint_positions",
    "object_root_position",
    "object_root_axis_angle",
    "ik_error_per_frame",
    "ik_num_iterations",
    "frame_task_errors",
)

#: ``world_axis_swap_matrix`` from
#: ``retarget/configs/g1/frame_alignment.json`` - the fixed source(SOMA) ->
#: robot(G1) world rotation ``WholeBodyKinematics`` applies as
#: ``config.r_world``. Mirrored here as data so the validator never imports
#: NVIDIA code (it would need torch + Isaac Sim).
R_SOURCE_TO_ROBOT = np.array(
    [[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64
)

#: The SOMA rig's root joint index. ``SOMA_JOINTS_ORDER.index("Hips") == 0``
#: upstream, and ``collapse_joint_names`` asserts joint 0 is ``Hips`` on the way
#: in, so index 0 is the hips in both directions.
SOMA_HIPS_INDEX = 0

# Root-alignment tolerances. Measured on Episode 0 (the only reference NVIDIA's
# own demo lets us compare against): XY max 0.088 m, dz std 0.014 m. The gates
# below are an order of magnitude looser, because the failure they exist to
# catch - a double-applied or missing frame conversion - shows up as METRES,
# not centimetres.
ROOT_FRAME0_XY_TOLERANCE_M = 0.25
ROOT_TRACKING_XY_TOLERANCE_M = 0.50
ROOT_VERTICAL_OFFSET_STD_TOLERANCE_M = 0.20

#: ``WholeBodyKinematics.max_iter`` for the G1 retargeter config. Mirrored as
#: data rather than imported, for the same reason as ``R_SOURCE_TO_ROBOT``.
IK_MAX_ITER = 200

# A single catastrophically bad frame can hide inside a healthy median, so the
# worst frame is also gated against this episode's OWN median. Measured over all
# 30 Tier 1 episodes that ratio is at most 1.63x, and NVIDIA's own demo
# reference sits at 1.65x - our spread is already no worse than theirs. 3.0
# leaves about a factor of two of headroom over both, so it flags a frame that
# genuinely diverged without flagging ordinary within-episode variation.
RESIDUAL_SPIKE_TOLERANCE = 3.0

#: G1's URDF: the source of the per-joint velocity limits used in section 10.
#: Read as data with stdlib XML, never imported - the validator must not need
#: torch or Isaac Sim (same reason as ``R_SOURCE_TO_ROBOT``).
G1_URDF = (
    Path(__file__).resolve().parent.parent
    / "official/video_to_data/robotic_grounding/source/robotic_grounding"
    / "robotic_grounding/assets/urdfs/g1/main_with_hand.urdf"
)

#: Fraction of a joint's URDF velocity limit the reference may demand. 1.0 is
#: the robot's own published number rather than a threshold we chose: a
#: reference asking for more than this cannot be tracked on real hardware.
#: Measured worst case across Tier 1 is 0.57x; NVIDIA's demo is 0.12x.
JOINT_VELOCITY_LIMIT_FRACTION = 1.0


@dataclass
class ReferenceReport:
    parquet: Path
    frames_expected: int
    checker: Checker

    @property
    def ok(self) -> bool:
        return self.checker.failed == 0

    def as_dict(self) -> dict:
        payload = self.checker.as_dict()
        payload["parquet"] = str(self.parquet)
        payload["frames_expected"] = self.frames_expected
        payload["ok"] = self.ok
        return payload


def load_row(path: Path) -> dict:
    """``motion_v1`` packs an entire sequence into a single parquet row."""
    table = pq.read_table(path)
    if table.num_rows != 1:
        raise ValueError(f"expected exactly 1 motion_v1 row, got {table.num_rows}")
    return {name: table.column(name)[0].as_py() for name in table.schema.names}


def finite_count(values) -> tuple[int, int]:
    """Return (non_finite, total) over an arbitrarily nested numeric structure."""
    flat = np.asarray(_flatten(values), dtype=np.float64)
    if flat.size == 0:
        return 0, 0
    return int(np.count_nonzero(~np.isfinite(flat))), int(flat.size)


def _flatten(values) -> list:
    out: list = []
    stack = [values]
    while stack:
        item = stack.pop()
        if isinstance(item, (list, tuple)):
            stack.extend(item)
        elif isinstance(item, (int, float)) and not isinstance(item, bool):
            out.append(item)
    return out


def quaternion_norm_error(quaternions) -> float:
    """Max |‖q‖ − 1| over a list of wxyz quaternions."""
    array = np.asarray(quaternions, dtype=np.float64)
    if array.size == 0:
        return 0.0
    norms = np.linalg.norm(array.reshape(-1, array.shape[-1]), axis=1)
    return float(np.max(np.abs(norms - 1.0)))


def load_urdf_joint_limits(urdf: Path) -> dict[str, dict[str, float]]:
    """Map joint name -> {lower, upper, velocity} from a URDF's ``<limit>`` tags.

    Stdlib XML on purpose: the limits are data, and importing NVIDIA's robot
    model would drag in torch and Isaac Sim. Returns an empty dict when the
    file is missing, so the caller skips the check with a note rather than
    failing for lack of an input.
    """
    if not urdf.is_file():
        return {}
    limits: dict[str, dict[str, float]] = {}
    for joint in ElementTree.parse(urdf).getroot().iter("joint"):
        name = joint.get("name")
        limit = joint.find("limit")
        if name is None or limit is None:
            continue
        entry = {
            key: float(limit.get(key))
            for key in ("lower", "upper", "velocity")
            if limit.get(key) is not None
        }
        if entry:
            limits[name] = entry
    return limits


def source_hips_in_robot_frame(row: dict) -> np.ndarray | None:
    """The plane-leveled SOMA hips trajectory, rotated into the robot frame.

    ``soma_to_g1.py`` stores the post-leveling source joints in the parquet's
    ``source_payload`` blob as opaque provenance. That makes a strong,
    self-contained alignment check possible: the G1 free-flyer is solved to
    track exactly this joint, so ``r_world @ soma_joints[:, Hips]`` and
    ``robot_root_position`` must stay within centimetres of each other.

    Returns None when the payload predates that field, as NVIDIA's own shipped
    demo parquet does - the check is then skipped for lack of data rather than
    passed by default.
    """
    payload = row.get("source_payload")
    if not payload:
        return None
    try:
        decoded = pickle.loads(payload)
    except Exception:  # noqa: BLE001 - an unreadable blob just means no check
        return None
    joints = decoded.get("soma_joints") if isinstance(decoded, dict) else None
    if joints is None:
        return None
    array = np.asarray(joints, dtype=np.float64)
    if array.ndim != 3 or array.shape[1] <= SOMA_HIPS_INDEX:
        return None
    return array[:, SOMA_HIPS_INDEX, :] @ R_SOURCE_TO_ROBOT.T


def run_checks(
    parquet: Path,
    *,
    frames: int,
    reference: Path | None = None,
    mesh_root: Path | None = None,
    urdf: Path | None = None,
    dropped_frames: tuple[int, ...] | list[int] | None = None,
    summary: bool = True,
    emit: Callable[[str], None] = print,
) -> ReferenceReport:
    """Run every reference check. Returns the report; never raises on a failure.

    ``dropped_frames`` are source frames deliberately excluded from this
    reference (NVIDIA zero-filled them; see ``CLAUDE.md``). They are recorded
    as a note so the exclusion travels with the gate output instead of living
    only in the manifest.

    ``summary=False`` suppresses the closing ``ALL CHECKS PASSED`` banner. The
    batch orchestrator sets it, because a per-episode banner printed 30 times
    reads like a verdict on the whole tier.
    """
    checker = Checker(emit=emit)
    row = load_row(parquet)
    expected = int(frames)
    if dropped_frames:
        dropped = list(dropped_frames)
        checker.note(
            "{} source frame(s) are deliberately absent from this reference: {}. "
            "observation.object.visible == False there, so NVIDIA zero-filled the "
            "pose and it was never measured; the retarget window stops at the last "
            "measured frame rather than interpolating. The {} checked frames are "
            "the measured ones.".format(
                len(dropped),
                dropped if len(dropped) <= 10 else f"{dropped[:10]} ...",
                expected,
            )
        )
        checker.metric("dropped_frames", len(dropped))

    checker.emit("=== 1. identity and schema ===")
    checker.info(f"file: {parquet}")
    checker.info(
        "schema_version={} motion_kind={} coord_frame={}".format(
            row["schema_version"], row["motion_kind"], row["coord_frame"]
        )
    )
    checker.info(f"source_dataset={row['source_dataset']} fps={row['fps']}")
    checker.metric("fps", float(row["fps"]))
    checker.check(
        row["schema_version"] == "motion_v1",
        f"schema_version is motion_v1 (got {row['schema_version']})",
    )
    checker.check(
        abs(float(row["fps"]) - 30.0) < 1e-6, f"fps = 30 (got {row['fps']})"
    )

    reference_row = None
    if reference is not None:
        reference_row = load_row(reference)
        ours = set(pq.read_table(parquet).schema.names)
        theirs = set(pq.read_table(reference).schema.names)
        checker.check(
            ours == theirs,
            f"column set matches NVIDIA's demo reference ({len(ours)} columns)",
        )
        checker.check(
            row["coord_frame"] == reference_row["coord_frame"],
            f"coord_frame matches demo ({reference_row['coord_frame']})",
        )
        checker.check(
            row["robot_joint_names"] == reference_row["robot_joint_names"],
            "robot_joint_names identical to demo ({} joints)".format(
                len(reference_row["robot_joint_names"])
            ),
        )
        checker.check(
            row["ee_link_names"] == reference_row["ee_link_names"],
            "ee_link_names identical to demo",
        )
        checker.check(
            row["hand_sides"] == reference_row["hand_sides"],
            f"hand_sides identical to demo ({reference_row['hand_sides']})",
        )

    checker.section("2. frame counts agree across every per-frame column")
    for column in PER_FRAME_COLUMNS:
        length = len(row[column])
        checker.check(length == expected, f"{column:28s} has {length} frames")
    checker.metric("frames", len(row["robot_root_position"]))
    checker.check(
        all(len(side) == expected for side in row["hand_contact_active"]),
        "hand_contact_active        has {} frames per hand".format(
            [len(s) for s in row["hand_contact_active"]]
        ),
    )
    # soma_to_g1 leaves the detailed hand-frame/finger columns unpopulated for
    # G1 (NVIDIA's own shipped demo reference is empty here too), so these are
    # compared against the reference rather than assumed per-frame.
    for column in ("hand_frames_w", "hand_finger_joints"):
        ours_lengths = [len(side) for side in row[column]]
        if reference_row is None:
            checker.info(
                f"{column:26s} inner lengths {ours_lengths} (no reference to compare)"
            )
            continue
        ref_lengths = [len(side) for side in reference_row[column]]
        checker.check(
            ours_lengths == ref_lengths,
            f"{column:26s} population matches demo (ours {ours_lengths}, demo {ref_lengths})",
        )

    checker.section("3. robot joint dimensions")
    joint_names = row["robot_joint_names"]
    widths = {len(frame) for frame in row["robot_joint_positions"]}
    checker.info(f"robot_joint_names: {len(joint_names)} entries")
    checker.check(
        widths == {len(joint_names)},
        "every frame's robot_joint_positions width == {} (got {})".format(
            len(joint_names), sorted(widths)
        ),
    )
    for side, names, values in zip(
        row["hand_sides"], row["hand_finger_joint_names"], row["hand_finger_joints"]
    ):
        hand_widths = {len(frame) for frame in values}
        checker.check(
            hand_widths == {len(names)},
            "{} hand finger joints width == {} (got {})".format(
                side, len(names), sorted(hand_widths)
            ),
        )

    checker.section("4. all poses finite")
    for column in (
        "robot_root_position",
        "robot_root_wxyz",
        "robot_joint_positions",
        "object_root_position",
        "object_root_axis_angle",
        "object_body_position",
        "object_body_wxyz",
        "ee_pose_w",
        "hand_frames_w",
        "hand_finger_joints",
    ):
        bad, total = finite_count(row[column])
        checker.check(
            bad == 0, f"{column:28s} {bad} non-finite of {total} values"
        )

    checker.section("5. quaternion normalization")
    for column in ("robot_root_wxyz", "object_body_wxyz"):
        error = quaternion_norm_error(row[column])
        checker.check(error < 1e-3, f"{column:28s} max |norm-1| = {error:.2e}")

    checker.section("6. grounding and physical plausibility")
    root = np.asarray(row["robot_root_position"], dtype=np.float64)
    object_root = np.asarray(row["object_root_position"], dtype=np.float64)
    checker.info(
        "robot root z: min={:+.3f} max={:+.3f} mean={:+.3f} m".format(
            root[:, 2].min(), root[:, 2].max(), root[:, 2].mean()
        )
    )
    checker.info(
        "object root z: min={:+.3f} max={:+.3f} m".format(
            object_root[:, 2].min(), object_root[:, 2].max()
        )
    )
    checker.metric("pelvis_z_mean_m", round(float(root[:, 2].mean()), 4))
    checker.metric("object_z_min_m", round(float(object_root[:, 2].min()), 4))
    checker.check(
        0.3 < root[:, 2].mean() < 1.2,
        "robot pelvis height is humanoid-plausible (mean {:+.3f} m)".format(
            root[:, 2].mean()
        ),
    )
    checker.check(
        object_root[:, 2].min() > -0.25,
        "object never sinks far below the ground plane (min z {:+.3f} m)".format(
            object_root[:, 2].min()
        ),
    )
    step = np.linalg.norm(np.diff(root, axis=0), axis=1)
    checker.info(
        "robot root per-frame step: max={:.4f} m mean={:.4f} m".format(
            step.max(), step.mean()
        )
    )
    checker.metric("root_step_max_m", round(float(step.max()), 4))
    checker.check(
        step.max() < 0.25, f"no root teleport between frames (max step {step.max():.4f} m)"
    )
    object_step = np.linalg.norm(np.diff(object_root, axis=0), axis=1)
    checker.metric("object_step_max_m", round(float(object_step.max()), 4))
    checker.check(
        object_step.max() < 0.5,
        f"no object teleport between frames (max step {object_step.max():.4f} m)",
    )

    checker.section("7. root alignment against the SOMA source it came from")
    source_hips = source_hips_in_robot_frame(row)
    frame0_xy = float(np.linalg.norm(root[0, :2]))
    checker.info(
        "frame-0 robot root: [{:+.4f} {:+.4f} {:+.4f}] m (|XY| = {:.4f} m)".format(
            root[0, 0], root[0, 1], root[0, 2], frame0_xy
        )
    )
    checker.metric("root_frame0_xy_m", round(frame0_xy, 4))
    # soma_to_g1 anchors the body to the source root's frame-0 pose, so the
    # retargeted pelvis starts over the origin by construction.
    checker.check(
        frame0_xy < ROOT_FRAME0_XY_TOLERANCE_M,
        "frame-0 robot root sits on the anchored origin "
        f"(|XY| {frame0_xy:.4f} m < {ROOT_FRAME0_XY_TOLERANCE_M} m)",
    )
    if source_hips is None:
        checker.note(
            "source_payload carries no soma_joints; cannot compare the robot root "
            "against the SOMA hips for this file"
        )
        checker.metric("root_align_xy_max_m", None)
        checker.metric("root_align_dz_std_m", None)
    elif source_hips.shape[0] != root.shape[0]:
        checker.check(
            False,
            "source_payload soma_joints has {} frames, robot root has {}".format(
                source_hips.shape[0], root.shape[0]
            ),
        )
    else:
        delta = root - source_hips
        horizontal = np.linalg.norm(delta[:, :2], axis=1)
        dz = delta[:, 2]
        checker.info(
            "robot pelvis vs SOMA hips, XY: mean={:.4f} max={:.4f} m".format(
                horizontal.mean(), horizontal.max()
            )
        )
        checker.info(
            "robot pelvis vs SOMA hips, dz: mean={:+.4f} std={:.4f} m "
            "(a stable G1-vs-human morphology offset)".format(dz.mean(), dz.std())
        )
        checker.metric("root_align_xy_mean_m", round(float(horizontal.mean()), 4))
        checker.metric("root_align_xy_max_m", round(float(horizontal.max()), 4))
        checker.metric("root_align_dz_mean_m", round(float(dz.mean()), 4))
        checker.metric("root_align_dz_std_m", round(float(dz.std()), 4))
        checker.check(
            horizontal.max() < ROOT_TRACKING_XY_TOLERANCE_M,
            "robot root tracks the SOMA hips horizontally "
            f"(max {horizontal.max():.4f} m < {ROOT_TRACKING_XY_TOLERANCE_M} m)",
        )
        checker.check(
            float(dz.std()) < ROOT_VERTICAL_OFFSET_STD_TOLERANCE_M,
            "the pelvis-to-hips height offset is stable, not drifting "
            f"(std {dz.std():.4f} m < {ROOT_VERTICAL_OFFSET_STD_TOLERANCE_M} m)",
        )

    checker.section("8. hand/object relationship")
    # ee_pose_w is [frame][ee][7]; use the declared palm/end-effector links.
    ee_names = row["ee_link_names"]
    ee = np.asarray(row["ee_pose_w"], dtype=np.float64)  # (T, E, 7)
    checker.info(f"ee_link_names: {ee_names}")
    distances = np.linalg.norm(ee[:, :, :3] - object_root[:, None, :], axis=2)
    nearest = distances.min(axis=1)
    checker.info(
        "nearest-EE-to-object-origin distance: "
        "min={:.3f} median={:.3f} max={:.3f} m".format(
            nearest.min(), float(np.median(nearest)), nearest.max()
        )
    )
    radius = float(np.max(row["object_mesh_radius"]))
    checker.info(f"object_mesh_radius = {radius:.3f} m")
    checker.metric("nearest_ee_min_m", round(float(nearest.min()), 4))
    checker.metric("object_mesh_radius_m", round(radius, 4))
    checker.check(
        nearest.min() < radius + 0.30,
        "hands reach the object at some point "
        f"(min {nearest.min():.3f} m vs radius {radius:.3f} m)",
    )
    active = np.asarray(row["hand_contact_active"], dtype=np.float64)
    contact_frames = int(np.count_nonzero(active.max(axis=0) > 0.5))
    checker.info(
        f"frames with at least one hand flagged in contact: {contact_frames} / {expected}"
    )
    checker.metric("contact_frames", contact_frames)

    checker.section("9. IK residuals vs NVIDIA's own shipped reference")
    ours_ik = np.asarray(row["ik_error_per_frame"], dtype=np.float64)
    iterations = np.asarray(row["ik_num_iterations"], dtype=np.int64)
    checker.info(
        "ours   ik_error_per_frame: min={:.4f} median={:.4f} mean={:.4f} max={:.4f}".format(
            ours_ik.min(), float(np.median(ours_ik)), ours_ik.mean(), ours_ik.max()
        )
    )
    checker.info(
        "ours   iterations: median={} max={}".format(
            int(np.median(iterations)), int(iterations.max())
        )
    )
    checker.metric("ik_median", round(float(np.median(ours_ik)), 4))
    checker.metric("ik_max", round(float(ours_ik.max()), 4))
    checker.metric("ik_iter_median", int(np.median(iterations)))
    checker.metric("ik_iter_max", int(iterations.max()))
    bad, _ = finite_count(row["ik_error_per_frame"])
    checker.check(bad == 0, "ik_error_per_frame is all finite")

    # The iteration count is REPORTED, NOT GATED. An earlier version of this
    # validator failed any episode whose solve hit the cap after frame 0; that
    # gate was wrong, and the measurements that retired it are:
    #
    # 1. For G1 the cap is not a failure signal at all. The retargeter config
    #    sets ``posture_task.q_prev_cost = 0.05``, so ``_posture_qprev_task``
    #    exists and ``has_extra_tasks`` is True on every frame. That makes the
    #    plateau exit at ``whole_body_kinematics.py:564`` unreachable, leaving
    #    ``vel_norm < 1e-6`` (line 566) as the solve's ONLY early exit.
    #    Reaching ``max_iter`` therefore means "did not drive the whole-body QP
    #    to a 1e-6 velocity norm", which is an asymptotic condition, not a
    #    divergence.
    # 2. Residual quality does not break at the cap. Over all 30 Tier 1
    #    episodes the median residual on capped frames is 1.07x the same
    #    episode's uncapped median (Episode 11: 0.91x, i.e. better). No
    #    residual blow-up, and the root moves millimetres across those frames.
    # 3. The cap is a pile-up on a continuum, not a cliff. Binned over 22,959
    #    frames the median residual climbs smoothly 0.96 -> 1.04 across
    #    iterations 150..199 and reads 1.13 at the cap; frames at exactly 199
    #    sit at 1.08. There is no discontinuity for a gate to sit on.
    # 4. NVIDIA's demo reference cannot be used as evidence here: its
    #    iterations are median 8 / max 34, which is unreachable under a
    #    1e-6 velocity-norm exit, so it was produced under the plateau exit and
    #    its iteration counts are not comparable to ours.
    # 5. The failure the old gate was reaching for - the solver genuinely
    #    giving up mid-trajectory - is the early ``break`` on the Pink/eigenpy
    #    flake, and it announces itself on stdout. No Tier 1 retarget log
    #    contains that message.
    # 6. NVIDIA's own ``--diagnose-ik`` treats iteration saturation as a
    #    printed diagnostic at 0.9*max_iter, never as an error.
    #
    # What replaces it: the residual spike test below, and the per-joint
    # velocity-limit check in section 10, both of which measure the reference
    # rather than the solver's effort.
    saturated = np.flatnonzero(iterations >= IK_MAX_ITER)
    later = [int(i) for i in saturated if i != 0]
    checker.metric("ik_saturated_frames", int(saturated.size))
    checker.metric("ik_saturated_after_frame0", len(later))
    checker.metric(
        "ik_saturated_fraction", round(float(saturated.size) / max(len(iterations), 1), 4)
    )
    if saturated.size:
        checker.info(
            "frames at the {}-iteration cap: {} of {} ({:.1%}) ({})".format(
                IK_MAX_ITER, saturated.size, len(iterations),
                saturated.size / len(iterations),
                list(saturated[:10]) + (["..."] if saturated.size > 10 else []),
            )
        )
        residuals = ours_ik[saturated]
        uncapped = np.ones(len(iterations), dtype=bool)
        uncapped[saturated] = False
        uncapped[0] = False
        checker.info(
            "their residuals: min={:.4f} median={:.4f} max={:.4f} "
            "(uncapped frames of this episode: median {:.4f})".format(
                residuals.min(), float(np.median(residuals)), residuals.max(),
                float(np.median(ours_ik[uncapped])) if uncapped.any() else float("nan"),
            )
        )
    if later:
        checker.note(
            "{} of {} frames reached the {}-iteration cap. Reported, not failed: "
            "G1's nonzero posture q_prev_cost makes the plateau exit unreachable, "
            "so the cap means the whole-body QP did not reach a 1e-6 velocity "
            "norm - not that the solve diverged. Residual quality is gated "
            "separately, just below and in section 10.".format(
                len(later), len(iterations), IK_MAX_ITER
            )
        )
    if 0 in saturated:
        checker.note(
            "frame 0 used the full {}-iteration budget: it is the only frame "
            "solved from the robot's neutral pose with no warm start. Its "
            "residual is {:.4f} against an episode median of {:.4f}.".format(
                IK_MAX_ITER, float(ours_ik[0]), float(np.median(ours_ik))
            )
        )

    # A healthy median can hide one diverged frame, so gate the worst frame
    # against this episode's own median.
    spike = float(ours_ik.max() / max(np.median(ours_ik), 1e-9))
    worst_frame = int(ours_ik.argmax())
    checker.metric("ik_spike_ratio", round(spike, 3))
    checker.metric("ik_worst_frame", worst_frame)
    checker.info(
        "worst frame is {} at residual {:.4f} = {:.2f}x this episode's median".format(
            worst_frame, float(ours_ik[worst_frame]), spike
        )
    )
    checker.check(
        spike < RESIDUAL_SPIKE_TOLERANCE,
        "no single frame's IK residual spikes away from the episode median "
        "({:.2f}x < {}x)".format(spike, RESIDUAL_SPIKE_TOLERANCE),
    )
    if reference_row is not None:
        ref_ik = np.asarray(reference_row["ik_error_per_frame"], dtype=np.float64)
        ref_iter = np.asarray(reference_row["ik_num_iterations"], dtype=np.int64)
        checker.info(
            "demo   ik_error_per_frame: min={:.4f} median={:.4f} mean={:.4f} max={:.4f}".format(
                ref_ik.min(), float(np.median(ref_ik)), ref_ik.mean(), ref_ik.max()
            )
        )
        checker.info(
            "demo   iterations: median={} max={}".format(
                int(np.median(ref_iter)), int(ref_iter.max())
            )
        )
        ratio = float(np.median(ours_ik) / max(np.median(ref_ik), 1e-9))
        checker.info(f"median IK residual ratio ours/demo = {ratio:.2f}x")
        checker.metric("ik_ratio_vs_demo", round(ratio, 3))
        checker.check(
            ratio < 3.0,
            f"our IK residual is within 3x NVIDIA's own demo ({ratio:.2f}x)",
        )
        # Iteration count is reported, not gated: a harder QP (tipping a heavy
        # stool, dragging a ladder) legitimately needs more iterations than the
        # demo's standing pick-and-place. What matters is that the solve
        # converged below max_iter with a comparable final residual.
        checker.info(
            "iteration ratio ours/demo = {:.1f}x median (reported, not gated)".format(
                float(np.median(iterations)) / max(float(np.median(ref_iter)), 1e-9)
            )
        )

    checker.section("10. kinematic feasibility against G1's own URDF limits")
    # This is the check that actually matters for the next step: a reference is
    # only usable as a BC target or an RL tracking target if the robot can
    # execute it. The limits come from G1's URDF, so the bar is the robot's own
    # published number rather than one we invented.
    joint_names = row["robot_joint_names"]
    joint_positions = np.asarray(row["robot_joint_positions"], dtype=np.float64)
    limits = load_urdf_joint_limits(urdf or G1_URDF)
    if not limits:
        checker.note(
            "G1 URDF not found at {}; the joint velocity and joint-limit "
            "measurements are skipped for lack of the limits, not passed "
            "by default".format(urdf or G1_URDF)
        )
        checker.metric("joint_velocity_worst_fraction", None)
        checker.metric("joint_limit_pinned_fraction", None)
    elif joint_positions.shape[0] < 2:
        checker.note("fewer than 2 frames; no joint velocity to measure")
    else:
        fps = float(row["fps"])
        # soma_to_g1 leaves G1's 14 hand joints undriven (they are constant
        # here and in NVIDIA's demo), so they carry no velocity information and
        # sit at a limit for the whole sequence. Measuring them would report
        # the rig's rest pose, not this reference, so restrict to the joints
        # the whole-body IK actually solves.
        driven = [
            i for i, name in enumerate(joint_names)
            if name in limits and not np.allclose(
                joint_positions[:, i], joint_positions[0, i]
            )
        ]
        undriven = len(joint_names) - len(driven)
        checker.info(
            "{} of {} joints are actually driven by the solve "
            "({} constant across the sequence, e.g. G1's hand joints)".format(
                len(driven), len(joint_names), undriven
            )
        )
        if not driven:
            checker.check(False, "at least one joint varies over the sequence")
        else:
            velocity_limit = np.array(
                [limits[joint_names[i]].get("velocity", np.inf) for i in driven]
            )
            speed = np.abs(np.diff(joint_positions[:, driven], axis=0)) * fps
            fraction = speed / np.where(velocity_limit > 0, velocity_limit, np.inf)
            worst = float(fraction.max())
            frame, column = np.unravel_index(int(fraction.argmax()), fraction.shape)
            over = int(np.count_nonzero(
                (fraction > JOINT_VELOCITY_LIMIT_FRACTION).any(axis=1)
            ))
            checker.metric("joint_velocity_worst_fraction", round(worst, 4))
            checker.metric("joint_velocity_frames_over_limit", over)
            checker.info(
                "fastest joint step: {} at frame {} = {:.3f} rad/s, "
                "{:.2f}x its {:.1f} rad/s URDF limit".format(
                    joint_names[driven[int(column)]], int(frame) + 1,
                    float(speed[frame, column]), worst,
                    float(velocity_limit[int(column)]),
                )
            )
            checker.check(
                over == 0,
                "every joint step is within that joint's URDF velocity limit "
                "(worst {:.2f}x, {} frame(s) over)".format(worst, over),
            )

            # Reported, not gated. A joint resting on its limit is a real
            # G1-vs-human morphology fact, not a defect: waist_pitch_joint
            # dominates here, the same stable torso offset already documented
            # for torso_link, and NVIDIA's own demo reference also pins a joint
            # on 11.4% of its frames.
            lower = np.array([limits[joint_names[i]]["lower"] for i in driven])
            upper = np.array([limits[joint_names[i]]["upper"] for i in driven])
            margin = np.maximum(1e-3, 0.01 * (upper - lower))
            pinned = (joint_positions[:, driven] <= lower + margin) | (
                joint_positions[:, driven] >= upper - margin
            )
            pinned_fraction = float(pinned.any(axis=1).mean())
            checker.metric("joint_limit_pinned_fraction", round(pinned_fraction, 4))
            if pinned.any():
                dominant = joint_names[driven[int(pinned.sum(axis=0).argmax())]]
                checker.metric("joint_limit_pinned_dominant", dominant)
                checker.info(
                    "a driven joint rests on its position limit on {:.1%} of "
                    "frames; most often {} (reported, not gated: NVIDIA's own "
                    "demo does this on 11.4% of its frames)".format(
                        pinned_fraction, dominant
                    )
                )
            else:
                checker.info("no driven joint ever rests on its position limit")

    checker.section("11. mesh and urdf paths")
    for column in ("object_mesh_paths", "object_urdf_paths"):
        checker.info(f"{column}: {row[column]}")
    if mesh_root is not None:
        for column in ("object_mesh_paths", "object_urdf_paths"):
            for recorded in row[column]:
                name = Path(recorded).name
                local = mesh_root / name
                checker.check(local.is_file(), f"{name} resolves on host: {local}")
    else:
        checker.info("(--mesh-root not given; path existence not checked)")

    if summary:
        checker.summary()
    return ReferenceReport(parquet=parquet, frames_expected=expected, checker=checker)
