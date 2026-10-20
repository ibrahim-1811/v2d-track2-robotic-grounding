#!/usr/bin/env python3
"""LeRobot -> SOMA adapter for the NVIDIA V2D Track 2 challenge dataset.

WHAT THIS IS FOR
----------------
The challenge publishes each episode as a LeRobot v2.1 Parquet file. NVIDIA's
retargeter ``scripts/retarget/soma_to_g1.py`` does not read that layout - it reads
a folder containing ``soma_params.npz`` and ``poses.npy``. Nothing official
bridges the two. This module is that bridge.

It is a FORMAT translator, not a FRAME translator. See the coordinate contract
below - getting this wrong is the single easiest way to silently corrupt the
whole pipeline.

THE COORDINATE CONTRACT  (verified in docs/EPISODE_0_FRAME_AUDIT.md)
--------------------------------------------------------------------
* ``observation.human.*``  is ALREADY in the SOMA source frame. Copy verbatim.
* ``observation.object.pose`` is in the source OpenCV world frame (X right,
  Y down, Z forward).
* ``soma_to_g1.py`` itself converts the object OpenCV -> SOMA using
  ``_R_CV_TO_SOMA = np.diag([1.0, -1.0, -1.0, 1.0])`` (soma_to_g1.py:308).

  => This adapter must NOT apply that matrix. It is its own inverse, so
     applying it here would double-apply it downstream and silently restore the
     broken frame, where the human sits 5-7 m from the object it is holding.

WHAT IT REFUSES TO DO
---------------------
It never interpolates, back-fills or invents a measurement. Frames the dataset
marks invalid are zero-filled by NVIDIA, and a zero quaternion is not a
rotation; by default the adapter stops rather than guess. See ``--on-invalid``.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# The contract that upstream's read_soma.py enforces on soma_params.npz.
# Mirrored here as data so our tests can assert against it WITHOUT importing
# NVIDIA code (importing it would need torch + the license-gated SOMA assets).
# Source: read_soma.py:608-618 (_load_npz).
# ---------------------------------------------------------------------------
REQUIRED_NPZ_KEYS: frozenset[str] = frozenset({
    "poses",
    "transl",
    "joint_names",
    "identity_model_type",
    "identity_coeffs",
    "scale_params",
    "unit",
    "keep_root",
})

NUM_SOMA_JOINTS = 77
POSE_VALUES_PER_FRAME = NUM_SOMA_JOINTS * 3          # 231
ROTATION_SUFFIXES = (".rx", ".ry", ".rz")


class AdapterError(RuntimeError):
    """Raised when the input violates an assumption we refuse to paper over."""


@dataclass(frozen=True)
class Episode:
    """One episode's measurements, straight from the Parquet - no transforms."""

    sequence_id: str
    object_name: str
    task: str
    fps: int
    identity_model_type: str
    joint_names: list[str]
    poses: np.ndarray            # (T, 77, 3) float32, radians, local rotations
    transl: np.ndarray           # (T, 3)     float32, metres, SOMA frame
    identity_coeffs: np.ndarray  # (T, 45)    float32
    scale_params: np.ndarray     # (T, 68)    float32
    bone_length_flexibles: np.ndarray  # (T, 6) float32
    object_pose: np.ndarray      # (T, 7)     float32, OpenCV frame, [xyz, wxyz]
    object_visible: np.ndarray   # (T,)       bool
    timestamp: np.ndarray        # (T,)       float32, seconds


# ---------------------------------------------------------------------------
# Step 1 - joint names
# ---------------------------------------------------------------------------
def collapse_joint_names(pose_feature_names: list[str]) -> list[str]:
    """Turn 231 ``['Hips.rx','Hips.ry','Hips.rz', ...]`` into 77 joint names.

    WHY: ``meta/info.json`` names the pose column one SCALAR at a time, but
    ``soma_params.npz`` wants one name per JOINT. Upstream compares this list
    against ``SOMALayer.rig_data['joint_names']`` and raises if it disagrees
    (read_soma.py:647-653), so the order matters exactly.

    This also VERIFIES the triplet layout rather than trusting it: the reshape
    in ``reshape_human_pose`` is only correct if the names really do come in
    consecutive ``.rx/.ry/.rz`` groups per joint.
    """
    if len(pose_feature_names) != POSE_VALUES_PER_FRAME:
        raise AdapterError(
            f"expected {POSE_VALUES_PER_FRAME} pose feature names, "
            f"got {len(pose_feature_names)}"
        )
    joints: list[str] = []
    for j in range(NUM_SOMA_JOINTS):
        triplet = pose_feature_names[3 * j : 3 * j + 3]
        bases = {name.rsplit(".", 1)[0] for name in triplet}
        if len(bases) != 1:
            raise AdapterError(
                f"pose names {triplet} at joint {j} do not share one joint name; "
                "the 231 values are not laid out as consecutive per-joint triplets"
            )
        suffixes = tuple(name[len(name.rsplit('.', 1)[0]):] for name in triplet)
        if suffixes != ROTATION_SUFFIXES:
            raise AdapterError(
                f"joint {j} ({bases}) has suffixes {suffixes}, expected {ROTATION_SUFFIXES}"
            )
        joints.append(bases.pop())
    return joints


# ---------------------------------------------------------------------------
# Step 2 - the human pose reshape
# ---------------------------------------------------------------------------
def reshape_human_pose(flat_pose: np.ndarray) -> np.ndarray:
    """(T, 231) -> (T, 77, 3) axis-angle rotation vectors.

    WHY: the Parquet stores one flat row of 231 floats per frame because LeRobot
    columns are flat. SOMA wants ``(T, J, 3)``. Since the 231 names are verified
    to be consecutive per-joint triplets, a plain C-order reshape is exactly the
    right regrouping - value ``3*j + k`` becomes ``[j, k]``.

    Each ``[j]`` is an axis-angle vector in RADIANS: direction = axis of
    rotation, length = angle. It is the joint's rotation relative to its PARENT.
    """
    if flat_pose.ndim != 2 or flat_pose.shape[1] != POSE_VALUES_PER_FRAME:
        raise AdapterError(
            f"human pose must be (T, {POSE_VALUES_PER_FRAME}); got {flat_pose.shape}"
        )
    return np.ascontiguousarray(
        flat_pose.reshape(flat_pose.shape[0], NUM_SOMA_JOINTS, 3), dtype=np.float32
    )


# ---------------------------------------------------------------------------
# Step 3 - the object trajectory
# ---------------------------------------------------------------------------
def quaternion_wxyz_to_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
    """(T, 4) w-first unit quaternions -> (T, 3, 3) rotation matrices.

    WHY W-FIRST MATTERS: the dataset stores ``[qw, qx, qy, qz]`` (verified three
    ways in docs/EPISODE_0_WALKTHROUGH.md section 5). SciPy, ROS and most
    libraries use w-LAST. Mixing them up still produces a perfectly valid unit
    quaternion - just the WRONG rotation - so nothing downstream would complain.

    We reorder the components by hand instead of using SciPy's ``scalar_first=``
    because that argument only exists in SciPy >= 1.14 and this repo pins 1.11.4.
    Doing it explicitly is also simply easier to audit.

    The formula below is the standard quaternion-to-matrix expansion, written
    out so the adapter has no hidden dependency on a rotation library.
    """
    q = np.asarray(quat_wxyz, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 4:
        raise AdapterError(f"quaternions must be (T, 4); got {q.shape}")

    norms = np.linalg.norm(q, axis=1)
    if not np.all(np.isfinite(norms)):
        raise AdapterError("non-finite quaternion encountered")
    if np.any(norms == 0.0):
        bad = int(np.flatnonzero(norms == 0.0)[0])
        raise AdapterError(
            f"zero quaternion at frame {bad}: this is NVIDIA's zero-fill for an "
            "invalid pose and cannot be converted to a rotation. Handle it via "
            "observation.object.visible rather than normalising it away."
        )
    # Normalise only to absorb float32 storage error (norms are ~1 +/- 1e-7).
    q = q / norms[:, None]
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    out = np.empty((q.shape[0], 3, 3), dtype=np.float64)
    out[:, 0, 0] = 1 - 2 * (y * y + z * z)
    out[:, 0, 1] = 2 * (x * y - z * w)
    out[:, 0, 2] = 2 * (x * z + y * w)
    out[:, 1, 0] = 2 * (x * y + z * w)
    out[:, 1, 1] = 1 - 2 * (x * x + z * z)
    out[:, 1, 2] = 2 * (y * z - x * w)
    out[:, 2, 0] = 2 * (x * z - y * w)
    out[:, 2, 1] = 2 * (y * z + x * w)
    out[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return out


def object_poses_to_matrices(
    object_pose: np.ndarray,
    object_visible: np.ndarray,
    *,
    on_invalid: str = "fail",
) -> np.ndarray:
    """(T, 7) ``[x,y,z,qw,qx,qy,qz]`` -> (T, 4, 4) homogeneous transforms.

    The result is ``world_T_object``: it maps a point in the stool mesh's own
    local frame into the world. ``poses.npy`` is exactly this array.

    FRAME: left completely alone. These stay in the source OpenCV world frame,
    because ``soma_to_g1.load_data()`` applies the OpenCV->SOMA conversion
    itself. Applying it here too would cancel out and silently break everything.

    INVALID FRAMES: where ``observation.object.visible`` is False the dataset
    zero-fills the row. We do not interpolate across those frames and we do not
    substitute identity - both would be inventing measurements. ``on_invalid``
    decides between stopping and writing the zeros through untouched.
    """
    if on_invalid not in ("fail", "passthrough"):
        raise AdapterError(f"on_invalid must be 'fail' or 'passthrough'; got {on_invalid!r}")
    if object_pose.ndim != 2 or object_pose.shape[1] != 7:
        raise AdapterError(f"object pose must be (T, 7); got {object_pose.shape}")
    if object_visible.shape != (object_pose.shape[0],):
        raise AdapterError("object_visible length does not match object_pose")

    invalid = np.flatnonzero(~object_visible)
    if invalid.size and on_invalid == "fail":
        raise AdapterError(
            f"{invalid.size} of {object_pose.shape[0]} frames have "
            f"observation.object.visible == False (first at index {int(invalid[0])}). "
            "Their poses are zero-filled, not measured. Re-run with "
            "--on-invalid passthrough to write the zero rows through unchanged, "
            "and handle them explicitly downstream. The adapter will not "
            "interpolate or substitute a pose for you."
        )

    n = object_pose.shape[0]
    matrices = np.tile(np.eye(4, dtype=np.float64), (n, 1, 1))
    valid = np.flatnonzero(object_visible)
    if valid.size:
        matrices[valid, :3, :3] = quaternion_wxyz_to_matrix(object_pose[valid, 3:7])
        matrices[valid, :3, 3] = object_pose[valid, :3].astype(np.float64)
    if invalid.size:
        # Write the dataset's own zero-fill through verbatim: an all-zero 4x4.
        # It is obviously not a transform, which is the point - it stays
        # detectable downstream instead of masquerading as a real pose.
        matrices[invalid] = 0.0
    return matrices


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_episode(tier_root: Path, episode_index: int = 0) -> Episode:
    """Read one episode's Parquet plus the dataset-level metadata."""
    parquet = tier_root / "data/chunk-000" / f"episode_{episode_index:06d}.parquet"
    info_path = tier_root / "meta/info.json"
    meta_path = tier_root / "meta/episodes_metadata.jsonl"
    for p in (parquet, info_path, meta_path):
        if not p.is_file():
            raise AdapterError(f"missing required input: {p}")

    info = json.loads(info_path.read_text())
    meta = None
    for line in meta_path.read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            if rec.get("episode_index") == episode_index:
                meta = rec
                break
    if meta is None:
        raise AdapterError(f"episode {episode_index} not found in {meta_path}")

    table = pq.read_table(parquet)
    col = lambda name: np.stack(table[name].to_numpy(zero_copy_only=False))

    # Cross-check the Parquet against the metadata instead of trusting either.
    n_rows = table.num_rows
    if n_rows != meta["frames"]:
        raise AdapterError(
            f"row count {n_rows} disagrees with metadata frames {meta['frames']}"
        )

    joint_names = collapse_joint_names(
        list(info["features"]["observation.human.pose"]["names"])
    )

    # 'soma_x_mhr' -> 'mhr'. Upstream rejects the npz if this does not match the
    # SOMALayer it was constructed with (read_soma.py:640-646).
    robot_type = str(info["robot_type"])
    if not robot_type.startswith("soma_x_"):
        raise AdapterError(f"unexpected robot_type {robot_type!r}; cannot derive identity model")
    identity_model_type = robot_type[len("soma_x_"):]

    return Episode(
        sequence_id=meta["sequence_id"],
        object_name=meta["object"],
        task=meta["task"],
        fps=int(info["fps"]),
        identity_model_type=identity_model_type,
        joint_names=joint_names,
        poses=reshape_human_pose(col("observation.human.pose")),
        transl=col("observation.human.translation").astype(np.float32),
        identity_coeffs=col("observation.human.identity_coeffs").astype(np.float32),
        scale_params=col("observation.human.scale_params").astype(np.float32),
        bone_length_flexibles=col("observation.human.bone_length_flexibles").astype(np.float32),
        object_pose=col("observation.object.pose").astype(np.float32),
        object_visible=table["observation.object.visible"].to_numpy().astype(bool),
        timestamp=table["timestamp"].to_numpy().astype(np.float32),
    )


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------
def build_soma_params(ep: Episode) -> dict[str, np.ndarray]:
    """Assemble the dict that becomes ``soma_params.npz``.

    Every scalar is wrapped in ``np.array`` so it survives the npz round-trip in
    the exact form upstream parses it:

    * ``unit`` / ``identity_model_type`` are read with ``.item()`` when
      ``dtype.kind == 'U'`` (read_soma.py:631-639), so they must be numpy
      unicode scalars, not Python ``str``.
    * ``keep_root`` is read with ``bool(np.asarray(...).item())``.

    ``joint_orient`` is deliberately ABSENT: it is optional, and read_soma.py
    fills it from the live SOMA model, which its docstring calls authoritative.
    Fabricating one would be inventing data.
    """
    return {
        # --- the motion itself -------------------------------------------
        "poses": ep.poses,                       # (T, 77, 3) local rotations, rad
        "transl": ep.transl,                     # (T, 3) root position, metres
        # --- who the body is ---------------------------------------------
        "identity_coeffs": ep.identity_coeffs,   # (T, 45); upstream reads [0]
        "scale_params": ep.scale_params,         # (T, 68); upstream reads [0]
        # --- rig identification ------------------------------------------
        "joint_names": np.array(ep.joint_names),
        "identity_model_type": np.array(ep.identity_model_type),
        # --- interpretation flags ----------------------------------------
        "unit": np.array("meters"),
        # keep_root=False means "poses[:, 0] is the root rotation", which is
        # true here: joint 0 is Hips and the root TRANSLATION is a separate
        # column. See docs/CHALLENGE_ADAPTER.md for why the value is also inert.
        "keep_root": np.array(False),
        # Challenge poses are LOCAL (per meta/info.json's
        # human_pose_convention). absolute_pose=True is explicitly unsupported
        # upstream (read_soma.py:390-393), so state the correct value.
        "absolute_pose": np.array(False),
        "rotation_repr": np.array("rotvec"),
        # --- preserved, though upstream ignores it ------------------------
        # Challenge-only measurement with no npz counterpart. Carried along so
        # the adapter is not lossy; read_soma.py simply never reads it.
        "bone_length_flexibles": ep.bone_length_flexibles,
    }


def write_outputs(
    ep: Episode,
    out_dir: Path,
    *,
    on_invalid: str = "fail",
) -> dict[str, Path]:
    """Write ``soma_params.npz`` and ``poses.npy`` into the layout soma_to_g1 wants."""
    out_dir.mkdir(parents=True, exist_ok=True)
    params = build_soma_params(ep)
    missing = REQUIRED_NPZ_KEYS - set(params)
    if missing:
        raise AdapterError(f"refusing to write: missing mandatory keys {sorted(missing)}")

    npz_path = out_dir / "soma_params.npz"
    poses_path = out_dir / "poses.npy"
    np.savez(npz_path, **params)
    np.save(poses_path, object_poses_to_matrices(
        ep.object_pose, ep.object_visible, on_invalid=on_invalid))

    manifest = {
        "sequence_id": ep.sequence_id,
        "object": ep.object_name,
        "task": ep.task,
        "frames": int(ep.poses.shape[0]),
        "fps": ep.fps,
        "identity_model_type": ep.identity_model_type,
        "object_visible_frames": int(ep.object_visible.sum()),
        "object_frame": "source OpenCV world (UNCHANGED - soma_to_g1.py applies diag(1,-1,-1))",
        "human_frame": "SOMA source frame (UNCHANGED)",
        "generated_files": ["soma_params.npz", "poses.npy"],
        "still_required_by_soma_to_g1": [
            "reconstructed_mesh/output_aligned.glb",
            "ground_plane.json",
        ],
    }
    manifest_path = out_dir / "adapter_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return {"soma_params": npz_path, "poses": poses_path, "manifest": manifest_path}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    repo = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tier-root", type=Path,
                    default=repo / "data/v2d_challenge/track_2/tier_1_multiview_caption")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--out-root", type=Path, default=repo / "data/adapter_out")
    ap.add_argument("--on-invalid", choices=("fail", "passthrough"), default="fail",
                    help="what to do when observation.object.visible is False")
    args = ap.parse_args()

    ep = load_episode(args.tier_root, args.episode)
    out_dir = args.out_root / "whole_body/soma" / ep.sequence_id
    written = write_outputs(ep, out_dir, on_invalid=args.on_invalid)

    print(f"episode {args.episode}: {ep.sequence_id}")
    print(f"  object        : {ep.object_name}")
    print(f"  frames        : {ep.poses.shape[0]} @ {ep.fps} fps")
    print(f"  visible       : {int(ep.object_visible.sum())}/{ep.object_visible.size}")
    print(f"  poses         : {ep.poses.shape} {ep.poses.dtype}")
    print(f"  transl        : {ep.transl.shape} {ep.transl.dtype}")
    print(f"  identity      : {ep.identity_coeffs.shape}   scale: {ep.scale_params.shape}")
    print(f"  joint_names   : {len(ep.joint_names)} (first={ep.joint_names[0]!r}, "
          f"last={ep.joint_names[-1]!r})")
    obj = np.load(written["poses"])
    print(f"  poses.npy     : {obj.shape} {obj.dtype}")
    for name, path in written.items():
        print(f"  wrote {name:13s} {path}")


if __name__ == "__main__":
    main()
