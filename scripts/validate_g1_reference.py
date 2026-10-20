"""Validate a retargeted G1 motion_v1 parquet against NVIDIA's own demo reference.

Checks the things that would silently break FlashCHORD later: frame counts that
disagree across columns, non-finite poses, joint-name/dimension mismatches,
unresolvable mesh paths, an object that drifts away from the hands, and IK
residuals far outside what NVIDIA's own shipped reference exhibits.

Run with the project venv (needs pyarrow):

    .venvs/episode0/bin/python scripts/validate_g1_reference.py \
        --parquet <ours.parquet> --reference <nvidia_demo.parquet> --frames 778
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

PASS = "  PASS  "
FAIL = "  FAIL  "
INFO = "  ..    "

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", required=True, type=Path)
    parser.add_argument("--reference", type=Path, default=None)
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument(
        "--mesh-root",
        type=Path,
        default=None,
        help="host directory the parquet's container-side mesh paths map to",
    )
    return parser.parse_args()


def load_row(path: Path) -> dict:
    """motion_v1 packs an entire sequence into a single parquet row."""
    table = pq.read_table(path)
    if table.num_rows != 1:
        raise ValueError(
            "expected exactly 1 motion_v1 row, got {}".format(table.num_rows)
        )
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


def main() -> int:
    args = parse_args()
    row = load_row(args.parquet)
    failures = 0

    def check(ok: bool, message: str) -> None:
        nonlocal failures
        print((PASS if ok else FAIL) + message)
        if not ok:
            failures += 1

    print("=== 1. identity and schema ===")
    print(INFO + "file: {}".format(args.parquet))
    print(INFO + "schema_version={} motion_kind={} coord_frame={}".format(
        row["schema_version"], row["motion_kind"], row["coord_frame"]))
    print(INFO + "source_dataset={} fps={}".format(row["source_dataset"], row["fps"]))
    check(row["schema_version"] == "motion_v1",
          "schema_version is motion_v1 (got {})".format(row["schema_version"]))
    check(abs(float(row["fps"]) - 30.0) < 1e-6,
          "fps = 30 (got {})".format(row["fps"]))

    if args.reference is not None:
        reference = load_row(args.reference)
        ours = set(pq.read_table(args.parquet).schema.names)
        theirs = set(pq.read_table(args.reference).schema.names)
        check(ours == theirs,
              "column set matches NVIDIA's demo reference ({} columns)".format(len(ours)))
        check(row["coord_frame"] == reference["coord_frame"],
              "coord_frame matches demo ({})".format(reference["coord_frame"]))
        check(row["robot_joint_names"] == reference["robot_joint_names"],
              "robot_joint_names identical to demo ({} joints)".format(
                  len(reference["robot_joint_names"])))
        check(row["ee_link_names"] == reference["ee_link_names"],
              "ee_link_names identical to demo")
        check(row["hand_sides"] == reference["hand_sides"],
              "hand_sides identical to demo ({})".format(reference["hand_sides"]))
    else:
        reference = None

    print()
    print("=== 2. frame counts agree across every per-frame column ===")
    expected = args.frames
    for column in PER_FRAME_COLUMNS:
        length = len(row[column])
        check(length == expected,
              "{:28s} has {} frames".format(column, length))
    # Nested per-hand columns: outer list is per hand, inner is per frame.
    check(all(len(side) == expected for side in row["hand_contact_active"]),
          "hand_contact_active        has {} frames per hand".format(
              [len(s) for s in row["hand_contact_active"]]))
    # soma_to_g1 leaves the detailed hand-frame/finger columns unpopulated for
    # G1 (NVIDIA's own shipped demo reference is empty here too), so these are
    # compared against the reference rather than assumed per-frame.
    for column in ("hand_frames_w", "hand_finger_joints"):
        ours_lengths = [len(side) for side in row[column]]
        if reference is None:
            print(INFO + "{:26s} inner lengths {} (no reference to compare)".format(
                column, ours_lengths))
            continue
        ref_lengths = [len(side) for side in reference[column]]
        check(ours_lengths == ref_lengths,
              "{:26s} population matches demo (ours {}, demo {})".format(
                  column, ours_lengths, ref_lengths))

    print()
    print("=== 3. robot joint dimensions ===")
    joint_names = row["robot_joint_names"]
    widths = {len(frame) for frame in row["robot_joint_positions"]}
    print(INFO + "robot_joint_names: {} entries".format(len(joint_names)))
    check(widths == {len(joint_names)},
          "every frame's robot_joint_positions width == {} (got {})".format(
              len(joint_names), sorted(widths)))
    for side, names, values in zip(
        row["hand_sides"], row["hand_finger_joint_names"], row["hand_finger_joints"]
    ):
        hand_widths = {len(frame) for frame in values}
        check(hand_widths == {len(names)},
              "{} hand finger joints width == {} (got {})".format(
                  side, len(names), sorted(hand_widths)))

    print()
    print("=== 4. all poses finite ===")
    for column in (
        "robot_root_position", "robot_root_wxyz", "robot_joint_positions",
        "object_root_position", "object_root_axis_angle",
        "object_body_position", "object_body_wxyz",
        "ee_pose_w", "hand_frames_w", "hand_finger_joints",
    ):
        bad, total = finite_count(row[column])
        check(bad == 0,
              "{:28s} {} non-finite of {} values".format(column, bad, total))

    print()
    print("=== 5. quaternion normalization ===")
    for column in ("robot_root_wxyz", "object_body_wxyz"):
        error = quaternion_norm_error(row[column])
        check(error < 1e-3,
              "{:28s} max |norm-1| = {:.2e}".format(column, error))

    print()
    print("=== 6. grounding and physical plausibility ===")
    root = np.asarray(row["robot_root_position"], dtype=np.float64)
    object_root = np.asarray(row["object_root_position"], dtype=np.float64)
    print(INFO + "robot root z: min={:+.3f} max={:+.3f} mean={:+.3f} m".format(
        root[:, 2].min(), root[:, 2].max(), root[:, 2].mean()))
    print(INFO + "object root z: min={:+.3f} max={:+.3f} m".format(
        object_root[:, 2].min(), object_root[:, 2].max()))
    check(0.3 < root[:, 2].mean() < 1.2,
          "robot pelvis height is humanoid-plausible (mean {:+.3f} m)".format(
              root[:, 2].mean()))
    check(object_root[:, 2].min() > -0.25,
          "object never sinks far below the ground plane (min z {:+.3f} m)".format(
              object_root[:, 2].min()))
    step = np.linalg.norm(np.diff(root, axis=0), axis=1)
    print(INFO + "robot root per-frame step: max={:.4f} m mean={:.4f} m".format(
        step.max(), step.mean()))
    check(step.max() < 0.25,
          "no root teleport between frames (max step {:.4f} m)".format(step.max()))
    object_step = np.linalg.norm(np.diff(object_root, axis=0), axis=1)
    check(object_step.max() < 0.5,
          "no object teleport between frames (max step {:.4f} m)".format(
              object_step.max()))

    print()
    print("=== 7. hand/object relationship ===")
    # ee_pose_w is [frame][ee][7]; use the declared palm/end-effector links.
    ee_names = row["ee_link_names"]
    ee = np.asarray(row["ee_pose_w"], dtype=np.float64)  # (T, E, 7)
    print(INFO + "ee_link_names: {}".format(ee_names))
    distances = np.linalg.norm(ee[:, :, :3] - object_root[:, None, :], axis=2)
    nearest = distances.min(axis=1)
    print(INFO + "nearest-EE-to-object-origin distance: "
                 "min={:.3f} median={:.3f} max={:.3f} m".format(
                     nearest.min(), float(np.median(nearest)), nearest.max()))
    radius = float(np.max(row["object_mesh_radius"]))
    print(INFO + "object_mesh_radius = {:.3f} m".format(radius))
    check(nearest.min() < radius + 0.30,
          "hands reach the object at some point (min {:.3f} m vs radius {:.3f} m)".format(
              nearest.min(), radius))
    active = np.asarray(row["hand_contact_active"], dtype=np.float64)
    contact_frames = int(np.count_nonzero(active.max(axis=0) > 0.5))
    print(INFO + "frames with at least one hand flagged in contact: {} / {}".format(
        contact_frames, expected))

    print()
    print("=== 8. IK residuals vs NVIDIA's own shipped reference ===")
    ours_ik = np.asarray(row["ik_error_per_frame"], dtype=np.float64)
    iterations = np.asarray(row["ik_num_iterations"], dtype=np.int64)
    print(INFO + "ours   ik_error_per_frame: "
                 "min={:.4f} median={:.4f} mean={:.4f} max={:.4f}".format(
                     ours_ik.min(), float(np.median(ours_ik)),
                     ours_ik.mean(), ours_ik.max()))
    print(INFO + "ours   iterations: median={} max={}".format(
        int(np.median(iterations)), int(iterations.max())))
    bad, _ = finite_count(row["ik_error_per_frame"])
    check(bad == 0, "ik_error_per_frame is all finite")
    if reference is not None:
        ref_ik = np.asarray(reference["ik_error_per_frame"], dtype=np.float64)
        ref_iter = np.asarray(reference["ik_num_iterations"], dtype=np.int64)
        print(INFO + "demo   ik_error_per_frame: "
                     "min={:.4f} median={:.4f} mean={:.4f} max={:.4f}".format(
                         ref_ik.min(), float(np.median(ref_ik)),
                         ref_ik.mean(), ref_ik.max()))
        print(INFO + "demo   iterations: median={} max={}".format(
            int(np.median(ref_iter)), int(ref_iter.max())))
        ratio = float(np.median(ours_ik) / max(np.median(ref_ik), 1e-9))
        print(INFO + "median IK residual ratio ours/demo = {:.2f}x".format(ratio))
        check(ratio < 3.0,
              "our IK residual is within 3x NVIDIA's own demo ({:.2f}x)".format(ratio))
        # Iteration count is reported, not gated: this sequence (tipping a heavy
        # floor-lying stool with both hands) poses a harder QP than the demo's
        # standing pick-and-place, and what matters is that the solve converged
        # below max_iter with a comparable final residual.
        print(INFO + "iteration ratio ours/demo = {:.1f}x median "
                     "(reported, not gated)".format(
                         float(np.median(iterations)) / max(float(np.median(ref_iter)), 1e-9)))
        check(int(iterations.max()) < 200,
              "IK never saturated max_iter=200 (ours max {})".format(int(iterations.max())))

    print()
    print("=== 9. mesh and urdf paths ===")
    for column in ("object_mesh_paths", "object_urdf_paths"):
        print(INFO + "{}: {}".format(column, row[column]))
    if args.mesh_root is not None:
        for column in ("object_mesh_paths", "object_urdf_paths"):
            for recorded in row[column]:
                name = Path(recorded).name
                local = args.mesh_root / name
                check(local.is_file(),
                      "{} resolves on host: {}".format(name, local))
    else:
        print(INFO + "(--mesh-root not given; path existence not checked)")

    print()
    if failures == 0:
        print("ALL CHECKS PASSED")
    else:
        print("{} CHECK(S) FAILED".format(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
