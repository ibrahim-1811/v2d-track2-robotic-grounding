"""Tier 3 object cleanup from Tier 3 data and physics only: undo symmetry flips, drop outliers, hold untouched.

CARI4D's object track flips between visually equivalent orientations of symmetric objects and spins
the free axis of near-cylindrical ones. 14 of 30 references have 77-422 frames with a rotation step
above 10 deg (single steps up to 179 deg, median 2-10 deg/frame), and their policies lose the object
(mean evaluation AUC 0.10 vs 0.65 on the other 16; training episodes end after 56-92 steps). No
policy can follow a target that turns 5-10 deg every frame. Nothing here reads Tier 1 data.

Which episodes: pre-declared, from the Tier 3 reference alone -- at least ``FLAG_FRAMES`` = 20 frames
with a rotation step above ``FLAG_STEP_DEG`` = 10 deg (the clean 16 have at most 10). Others are
copied byte for byte. Per flagged episode, in order:

1. **symmetry unwrap.** The mesh's approximate self-symmetries are found once: the 23 non-identity
   rotations that map its principal axes onto themselves, plus continuous rotation about a principal
   axis, accepted when the median nearest-vertex distance after the rotation is below
   ``SYM_TOL`` = 5% of the bounding radius (median, so a mug's handle does not veto its body). Frame
   by frame, the equivalent pose closest to the previous one is kept, starting from frame 0 (the
   aligned pose, which is never changed). Rotations are taken about the mesh centroid.
2. **outliers.** A frame more than ``ROT_OUTLIER_DEG`` = 20 deg or ``POS_OUTLIER_M`` = 0.08 m from the
   robust centre of its +-``WINDOW`` = 7 frames (0.23 s) is replaced by interpolation.
3. **contact hold** (``recover_tier2_contact.py``'s rule): the object moves only while a palm or foot
   is within its bounding radius + ``REACH_M`` = 0.15 m (hand ~7 cm plus ~8 cm for CARI4D's human and
   the G1 retarget offset), with the same 5-frame release.
4. **rate limit** to ``V_MAX`` = 4 m/s and ``W_MAX`` = 4 pi rad/s (human hand speeds), so hold
   transitions and residual jumps become motion a policy can follow.
5. **smoothing**, Gaussian sigma ``SMOOTH_FRAMES`` = 1.5 frames (~3 Hz). Frame 0 stays exact.

All constants are fixed here from physics before anything is trained or scored.

Outputs: ``data/adapter_out_tier_3c`` (G1 references, all 30), ``outputs/tier3c_frame_references``
(the reconstruction's raw frame, for the packer's ``--frame-references``) and
``training_manifest_tier3c.json``. Run inside containers/flash_chord.sif::

    python scripts/tier3/clean_tier3_object.py           # report only
    python scripts/tier3/clean_tier3_object.py --write
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.data.recover_tier2_contact import URDF, Kinematics, contact_hold_index, touched_mask  # noqa: E402
from scripts.tier3.make_tier3_manifest import read_obj_vertices, stationary_auc  # noqa: E402

FLAG_STEP_DEG, FLAG_FRAMES = 10.0, 20
SYM_TOL = 0.05
WINDOW, ROT_OUTLIER_DEG, POS_OUTLIER_M = 7, 20.0, 0.08
REACH_M = 0.15
V_MAX, W_MAX = 4.0, 4 * np.pi
SMOOTH_FRAMES = 1.5
FPS = 30.0
COLUMNS = ("object_body_position", "object_body_wxyz", "object_root_position", "object_root_axis_angle")


def steps(position: np.ndarray, rotation: Rotation) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame position step (m) and rotation step (deg)."""
    return (np.linalg.norm(np.diff(position, axis=0), axis=1),
            np.degrees((rotation[1:] * rotation[:-1].inv()).magnitude()))


def symmetries(vertices: np.ndarray) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """Centroid, accepted discrete self-rotations and continuous symmetry axes (body frame)."""
    centre = vertices.mean(axis=0)
    local = vertices - centre
    radius = np.linalg.norm(local, axis=1).max()
    axes = np.linalg.eigh(local.T @ local)[1]  # columns = principal axes
    tree = cKDTree(local)

    def fits(rotation: np.ndarray) -> bool:
        return float(np.median(tree.query(local @ rotation.T)[0])) < SYM_TOL * radius

    discrete = []
    for perm in ([0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]):
        for signs in np.ndindex(2, 2, 2):
            p = np.zeros((3, 3))
            p[range(3), perm] = np.where(np.array(signs) == 0, 1.0, -1.0)
            if np.linalg.det(p) < 0 or np.allclose(p, np.eye(3)):
                continue
            rotation = axes @ p @ axes.T
            if fits(rotation):
                discrete.append(rotation)
    continuous = [axes[:, k] for k in range(3)
                  if all(fits(Rotation.from_rotvec(np.radians(a) * axes[:, k]).as_matrix()) for a in (20, 45, 70, 135))]
    return centre, discrete, continuous


def unwrap(position: np.ndarray, rotation: Rotation, centre: np.ndarray, discrete: list[np.ndarray],
           continuous: list[np.ndarray]) -> tuple[np.ndarray, Rotation, int]:
    """Replace each pose by its symmetric equivalent closest to the previous output pose."""
    candidates = [np.eye(3)] + discrete
    out_r = rotation.as_matrix().copy()
    out_p = position.copy()
    switched = 0
    for t in range(1, len(position)):
        previous = out_r[t - 1]
        best, best_angle = None, np.inf
        for sym in candidates:
            r = rotation[t].as_matrix() @ sym
            for axis in continuous:  # twist about the free axis that best matches the previous pose
                q = Rotation.from_matrix(r.T @ previous).as_quat()  # xyzw, body-frame residual
                r = r @ Rotation.from_rotvec(2 * np.arctan2(q[:3] @ axis, q[3]) * axis).as_matrix()
            angle = Rotation.from_matrix(r.T @ previous).magnitude()
            if angle < best_angle:
                best, best_angle = r, angle
        switched += int(not np.allclose(best, rotation[t].as_matrix(), atol=1e-6))
        out_p[t] = position[t] + (rotation[t].as_matrix() - best) @ centre  # centroid stays put
        out_r[t] = best
    return out_p, Rotation.from_matrix(out_r), switched


def interpolate(position: np.ndarray, rotation: Rotation, bad: np.ndarray) -> tuple[np.ndarray, Rotation]:
    good = np.flatnonzero(~bad)
    frames = np.arange(len(position))
    position = np.stack([np.interp(frames, good, position[good, k]) for k in range(3)], axis=1)
    rotation = Slerp(good, rotation[good])(np.clip(frames, good[0], good[-1]))
    return position, rotation


def outliers(position: np.ndarray, rotation: Rotation) -> np.ndarray:
    frames = len(position)
    bad = np.zeros(frames, dtype=bool)
    for t in range(1, frames):
        window = slice(max(0, t - WINDOW), min(frames, t + WINDOW + 1))
        if np.linalg.norm(position[t] - np.median(position[window], axis=0)) > POS_OUTLIER_M:
            bad[t] = True
            continue
        centre = rotation[window].mean()
        bad[t] = np.degrees((rotation[t] * centre.inv()).magnitude()) > ROT_OUTLIER_DEG
    return bad


def rate_limit(position: np.ndarray, rotation: Rotation) -> tuple[np.ndarray, Rotation]:
    out_p, out_r = position.copy(), [rotation[0]]
    max_step, max_turn = V_MAX / FPS, W_MAX / FPS
    for t in range(1, len(position)):
        delta = position[t] - out_p[t - 1]
        norm = np.linalg.norm(delta)
        out_p[t] = out_p[t - 1] + (delta * min(1.0, max_step / norm) if norm > 0 else 0)
        turn = (rotation[t] * out_r[-1].inv()).as_rotvec()
        angle = np.linalg.norm(turn)
        out_r.append(Rotation.from_rotvec(turn * min(1.0, max_turn / angle) if angle > 0 else turn) * out_r[-1])
    return out_p, Rotation.concatenate(out_r)


def smooth(position: np.ndarray, rotation: Rotation) -> tuple[np.ndarray, Rotation]:
    quat = rotation.as_quat()
    for t in range(1, len(quat)):  # one hemisphere, so averaging is meaningful
        if quat[t] @ quat[t - 1] < 0:
            quat[t] = -quat[t]
    position = gaussian_filter1d(position, SMOOTH_FRAMES, axis=0, mode="nearest")
    quat = gaussian_filter1d(quat, SMOOTH_FRAMES, axis=0, mode="nearest")
    return position, Rotation.from_quat(quat / np.linalg.norm(quat, axis=1, keepdims=True))


def clean(table: pa.Table, kin: Kinematics, mesh: Path) -> tuple[np.ndarray, Rotation, dict]:
    position = np.asarray(table.column("object_body_position")[0].as_py(), float)[:, 0]
    wxyz = np.asarray(table.column("object_body_wxyz")[0].as_py(), float)[:, 0]
    rotation = Rotation.from_quat(wxyz[:, [1, 2, 3, 0]])
    centre, discrete, continuous = symmetries(read_obj_vertices(mesh, limit=4000))
    p, r, switched = unwrap(position, rotation, centre, discrete, continuous)
    bad = outliers(p, r)
    p, r = interpolate(p, r, bad)
    touched, fk_error = touched_mask(table, kin, REACH_M, centre=p)
    source = contact_hold_index(touched)
    p, r = rate_limit(p[source], r[source])
    p, r = smooth(p, r)
    p[0], r = position[0], Rotation.concatenate([rotation[0], r[1:]])
    info = {"symmetries_discrete": len(discrete), "symmetry_axes_continuous": len(continuous),
            "unwrapped_frames": switched, "outlier_frames": int(bad.sum()),
            "touched_fraction": round(float(touched.mean()), 4), "held_frames": int((source != np.arange(len(source))).sum()),
            "fk_palm_error_cm": round(fk_error * 100, 3)}
    return p, r, info


def summary(position: np.ndarray, rotation: Rotation) -> str:
    dp, dr = steps(position, rotation)
    return (f"rot>10deg {int((dr > FLAG_STEP_DEG).sum()):4d}  rot max {dr.max():6.1f}  rot med {np.median(dr):5.2f}  "
            f"pos>5cm {int((dp > 0.05).sum()):4d}  pos max {dp.max() * 100:6.1f} cm")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write", action="store_true", help="write references, frame references and the manifest")
    parser.add_argument("--src", default="adapter_out_tier_3_cari4d_ours")
    parser.add_argument("--dst", default="adapter_out_tier_3c")
    parser.add_argument("--raw", default="data/reconstructed/track_2/tier_3_cari4d_ours")
    parser.add_argument("--frame-refs", default="outputs/tier3c_frame_references")
    parser.add_argument("--manifest-in", default="training_manifest_tier3.json")
    parser.add_argument("--manifest-out", default="training_manifest_tier3c.json")
    args = parser.parse_args()

    kin = Kinematics(URDF)
    manifest = json.loads((REPO / args.manifest_in).read_text())
    tree = REPO / "data" / args.dst / "whole_body"
    refs = REPO / args.frame_refs
    if args.write:
        shutil.copytree(REPO / "data" / args.src / "whole_body/reconstructed_stage", tree / "reconstructed_stage", dirs_exist_ok=True)
        refs.mkdir(parents=True, exist_ok=True)
    flagged = []
    for episode in manifest["episodes"]:
        ep, seq = episode["episode_index"], episode["sequence_id"]
        src = REPO / f"data/{args.src}/whole_body/soma/sequence_id={seq}/robot_name=g1/data.parquet"
        table = pq.read_table(src)
        position = np.asarray(table.column("object_body_position")[0].as_py(), float)[:, 0]
        wxyz = np.asarray(table.column("object_body_wxyz")[0].as_py(), float)[:, 0]
        rotation = Rotation.from_quat(wxyz[:, [1, 2, 3, 0]])
        frames = len(position)
        raw = pq.read_table(REPO / args.raw / f"data/chunk-000/episode_{ep:06d}.parquet",
                            columns=["observation.object.pose", "frame_index"])
        raw_pose = np.stack(raw.column("observation.object.pose").to_numpy(zero_copy_only=False)).astype(float)[:frames]
        frame_index = np.asarray(raw.column("frame_index").to_pylist())[:frames]
        # One left rigid transform maps the G1 reference's object pose into the raw frame (the packer
        # relies on exactly that); derive it from frame 0 and prove it on every original frame.
        raw_rot = Rotation.from_quat(raw_pose[:, [4, 5, 6, 3]])
        world = raw_rot[0] * rotation[0].inv()
        offset = raw_pose[0, :3] - world.apply(position[0])
        residual = np.linalg.norm(world.apply(position) + offset - raw_pose[:, :3], axis=1).max()
        if residual > 1e-5:
            raise SystemExit(f"ep{ep:02d}: G1 and raw object poses are not one rigid transform apart ({residual:.2e} m)")
        dr = steps(position, rotation)[1]
        out = tree / "soma" / f"sequence_id={seq}" / "robot_name=g1" / "data.parquet"
        if (dr > FLAG_STEP_DEG).sum() < FLAG_FRAMES:
            print(f"ep{ep:02d} keep   {summary(position, rotation)}")
            pose_xyzw = raw_pose[:, [0, 1, 2, 4, 5, 6, 3]]
            if args.write:
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, out)
        else:
            flagged.append(ep)
            mesh = REPO / "data" / table.column("object_mesh_paths")[0].as_py()[0].split("/work/", 1)[1]
            p, r, info = clean(table, kin, mesh)
            print(f"ep{ep:02d} before {summary(position, rotation)}")
            print(f"     after  {summary(p, r)}  {info}")
            episode["cleanup"] = {"method": "symmetry_unwrap+outliers+contact_hold+rate_limit+smooth", **info}
            pose_xyzw = np.concatenate([world.apply(p) + offset, (world * r).as_quat()], axis=1)
            if args.write:
                wxyz_new = r.as_quat()[:, [3, 0, 1, 2]]
                values = {"object_body_position": p[:, None, :].tolist(), "object_body_wxyz": wxyz_new[:, None, :].tolist(),
                          "object_root_position": p.tolist(), "object_root_axis_angle": r.as_rotvec().tolist()}
                for name in COLUMNS:
                    table = table.set_column(table.schema.get_field_index(name), name,
                                             pa.array([values[name]], type=table.schema.field(name).type))
                out.parent.mkdir(parents=True, exist_ok=True)
                pq.write_table(table, out)
                episode["stationary_auc"] = stationary_auc(out, mesh)
                episode["stationary_auc_source"] = f"{args.dst} input, frame-0 pose frozen (no Tier 1 data)"
        if args.write:
            np.savez(refs / f"episode_{ep:06d}.npz", episode_index=np.int64(ep), coordinate_frame="released_track2_world",
                     frame_index=frame_index.astype(np.int64), pose_xyzw=pose_xyzw[:, None, :])
            episode["task_parquet_container"] = f"/work/{args.dst}/whole_body/soma/sequence_id={seq}/robot_name=g1/data.parquet"
    print(f"cleaned {len(flagged)} episodes: {flagged}")
    if args.write:
        manifest["adapter_root"] = args.dst
        manifest["cleaned_episodes"] = flagged
        manifest["note"] = (manifest.get("note", "") + " Object track cleaned by scripts/tier3/clean_tier3_object.py "
                            "(symmetry unwrap, outliers, contact hold, rate limit, smoothing; Tier 3 data and physics only).")
        (REPO / args.manifest_out).write_text(json.dumps(manifest, indent=1) + "\n")
        print(f"wrote data/{args.dst}, {args.frame_refs}, {args.manifest_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
