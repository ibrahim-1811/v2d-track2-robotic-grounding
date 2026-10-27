"""Score a CARI4D object trajectory against the clean Tier 1 reference, as Tier 3 would see it.

ONE-OFF FEASIBILITY CHECK (Episode 0, 2026-10-27). Do not use it to select or tune anything for
Tier 3: the FAQ allows Tier 1's first frame for alignment "provided the Tier 1 caption isn't used
in Tier 3 otherwise". See CLAUDE.md, "Rule discipline for Tiers 2 and 3".

Tier 3 may use Tier 1's FIRST frame for alignment and nothing else (FAQ, upstream d9d6bf04).
So the only alignment here is one rigid transform, fixed by frame 0:

    T_align = P_ref(0) @ inv(P_cari(0)),   P_hat(t) = T_align @ P_cari(t)

which is exact at frame 0 and leaves every later frame as CARI4D reconstructed it (the camera
is static, so one transform is the whole camera-to-world map). The result is the AUC/MPPE of
the Tier 3 *policy input* -- an upper bound on what a policy tracking it can score, comparable
with Tier 2's input AUC (0.422 noisy, 0.480 recovered).

CARI4D poses live in its aligned copy of the source GLB; the reference poses refer to the
adapter's textured_mesh.obj. The GLB->OBJ mesh transform is measured from the vertices
(Procrustes on matched vertices), never assumed, and its residual is printed.

Two stages, one per container (no image has both pyarrow and torch):

    # flash_chord.sif (pyarrow): reference pose + OBJ vertices -> npz
    python scripts/tier3/score_cari4d_object.py dump-reference --episode 0 --out ref_ep00.npz
    # cari4d.sif (torch, trimesh): score
    python scripts/tier3/score_cari4d_object.py score --reference ref_ep00.npz \
        --cari4d outputs/tier3/cari4d/<seq>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ADD_THRESHOLDS_M = np.arange(0.01, 0.10, 0.01)  # the kit's metric_code/track_2/AUC.py
MPPE_RADIUS_M = 0.05
MPPE_DIRECTIONS = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], float)


def read_obj_vertices(path: Path) -> np.ndarray:
    rows = [line.split()[1:4] for line in path.open() if line.startswith("v ")]
    return np.asarray(rows, dtype=np.float64)


def dump_reference(args: argparse.Namespace) -> int:
    import pyarrow.parquet as pq

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from pipeline.episodes import load_tier

    spec = [s for s in load_tier("tier_1_multiview_caption") if s.episode_index == args.episode][0]
    table = pq.read_table(spec.reference_parquet, columns=["object_root_position", "object_root_axis_angle"])
    np.savez(
        args.out,
        episode_index=args.episode,
        sequence_id=spec.sequence_id,
        position=np.asarray(table.column("object_root_position")[0].as_py(), dtype=np.float64),
        rotvec=np.asarray(table.column("object_root_axis_angle")[0].as_py(), dtype=np.float64),
        obj_vertices=read_obj_vertices(spec.object_dir / "textured_mesh.obj"),
    )
    print(f"wrote {args.out}: {spec.sequence_id}")
    return 0


def rigid_fit(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float]:
    """Least-squares rigid transform (no scale) mapping source points onto target points."""
    mu_s, mu_t = source.mean(0), target.mean(0)
    u, _, vt = np.linalg.svd((source - mu_s).T @ (target - mu_t))
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = mu_t - rotation @ mu_s
    residual = np.linalg.norm(source @ rotation.T + transform[:3, 3] - target, axis=1)
    return transform, float(residual.max())


def add_auc(add: np.ndarray) -> float:
    curve = np.array([float(np.mean(add < t)) for t in ADD_THRESHOLDS_M])
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    return float(trapezoid(curve, x=np.linspace(0.0, 1.0, ADD_THRESHOLDS_M.size)))


def metrics(predicted: np.ndarray, reference: np.ndarray, vertices: np.ndarray) -> dict:
    """predicted/reference: (T, 4, 4) world-from-mesh poses in the OBJ mesh frame."""
    p_points = predicted[:, None, :3, 3] + np.einsum("tij,kj->tki", predicted[:, :3, :3], vertices)
    r_points = reference[:, None, :3, 3] + np.einsum("tij,kj->tki", reference[:, :3, :3], vertices)
    add = np.linalg.norm(p_points - r_points, axis=-1).mean(axis=1)
    offsets = MPPE_DIRECTIONS * MPPE_RADIUS_M
    p_key = predicted[:, None, :3, 3] + np.einsum("tij,kj->tki", predicted[:, :3, :3], offsets)
    r_key = reference[:, None, :3, 3] + np.einsum("tij,kj->tki", reference[:, :3, :3], offsets)
    relative = np.einsum("tji,tjk->tik", reference[:, :3, :3], predicted[:, :3, :3])
    angle = np.degrees(np.arccos(np.clip((np.trace(relative, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)))
    position = np.linalg.norm(predicted[:, :3, 3] - reference[:, :3, 3], axis=1)
    return {
        "auc": round(add_auc(add), 4),
        "mppe_cm": round(float(np.linalg.norm(p_key - r_key, axis=-1).mean() * 100.0), 2),
        "mean_position_error_cm": round(float(position.mean() * 100.0), 2),
        "mean_rotation_error_deg": round(float(angle.mean()), 2),
        "sp_success": bool(position.mean() <= 0.10 and np.radians(angle.mean()) <= 0.5),
        "frames_add_under_5cm": round(float(np.mean(add < 0.05)), 3),
    }


def score(args: argparse.Namespace) -> int:
    import torch
    import trimesh
    from scipy.spatial.transform import Rotation

    reference = np.load(args.reference)
    sequence = str(reference["sequence_id"])
    root = Path(args.cari4d)
    export = json.loads((root / "export" / sequence / "wild_export.json").read_text())
    refined = torch.load(root / "inference" / "refined.pth", map_location="cpu", weights_only=False)

    reference_pose = np.tile(np.eye(4), (len(reference["position"]), 1, 1))
    reference_pose[:, :3, :3] = Rotation.from_rotvec(reference["rotvec"]).as_matrix()
    reference_pose[:, :3, 3] = reference["position"]

    # GLB (the source CARI4D was given) -> OBJ (the frame the reference poses refer to).
    glb = trimesh.load(export["source_object_mesh"]["path"], force="mesh")
    obj_vertices = reference["obj_vertices"]
    glb_vertices = np.asarray(glb.vertices, dtype=np.float64)
    if glb_vertices.shape != obj_vertices.shape:
        print(f"vertex counts differ (GLB {glb_vertices.shape[0]}, OBJ {obj_vertices.shape[0]}); refusing to guess")
        return 2
    obj_from_glb, residual = rigid_fit(glb_vertices, obj_vertices)
    print(f"GLB->OBJ mesh transform: max vertex residual {residual * 1000:.3f} mm")
    if residual > 1e-3:
        print("mesh frames do not match vertex-for-vertex; refusing to score")
        return 2
    glb_from_obj = np.linalg.inv(obj_from_glb)
    aligned_from_source = np.asarray(export["source_object_mesh_to_aligned_transform"], dtype=np.float64)

    step = max(1, obj_vertices.shape[0] // args.max_vertices)
    vertices = obj_vertices[::step][: args.max_vertices]
    frames = min(len(reference_pose), len(refined["frames"]))
    if frames != len(reference_pose):
        print(f"note: CARI4D has {len(refined['frames'])} frames, reference {len(reference_pose)}; scoring {frames}")
    reference_pose = reference_pose[:frames]

    results = {"episode_index": int(reference["episode_index"]), "sequence": sequence, "frames": frames}
    stages = {
        "foundationpose_only": refined["pr_initial"]["pose_abs"],
        "coconet": refined["pr"]["pose_abs"],
        "refined_postopt": refined["pr"]["pose_abs_postopt"],
    }
    for name, camera_from_aligned in stages.items():
        camera_from_obj = np.asarray(camera_from_aligned, dtype=np.float64)[:frames] @ aligned_from_source @ glb_from_obj
        world_from_camera = reference_pose[0] @ np.linalg.inv(camera_from_obj[0])  # Tier 1 frame 0 only
        results[name] = metrics(world_from_camera[None] @ camera_from_obj, reference_pose, vertices)
    frozen = np.repeat(reference_pose[:1], frames, axis=0)
    results["stationary_floor"] = metrics(frozen, reference_pose, vertices)
    print(json.dumps(results, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=1))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    dump = sub.add_parser("dump-reference")
    dump.add_argument("--episode", type=int, required=True)
    dump.add_argument("--out", required=True)
    run = sub.add_parser("score")
    run.add_argument("--reference", required=True)
    run.add_argument("--cari4d", required=True)
    run.add_argument("--max-vertices", type=int, default=2000)
    run.add_argument("--out")
    args = parser.parse_args()
    return dump_reference(args) if args.command == "dump-reference" else score(args)


if __name__ == "__main__":
    raise SystemExit(main())
