#!/usr/bin/env python3
"""Score a do-nothing policy with the OFFICIAL Track 2 metrics, over a whole tier.

The V2D leaderboard ranks Track 2 by **AUC**, and AUC is the area under the
"fraction of frames with ADD below t" curve for t in 1..9 cm
(``_ADD_THRESHOLDS_M = np.arange(0.01, 0.10, 0.01)`` in the kit's
``metric_code/track_2/AUC.py``). That is a tight tolerance, and it has a
consequence worth measuring before optimising anything:

    a policy that never touches the object still scores well early in the
    episode, because the object STARTS in the right pose. It only leaves the
    1-9 cm band as the reference moves away from it.

So "do nothing" is not a zero on the leaderboard - it is a floor that a
half-trained policy can fall BELOW, by pushing the object out of the band
sooner than gravity would. This script computes that floor per episode, using
the kit's own formulas re-implemented from its published source, so we know
which side of it our training is on.

MPPE is computed the same way the kit does: six keypoints at +/-5 cm along each
body axis, so it penalises orientation as well as position.

Runs on the host with the project venv (needs pyarrow + scipy).

    .venvs/episode0/bin/python scripts/evaluate/leaderboard_floor.py
    .venvs/episode0/bin/python scripts/evaluate/leaderboard_floor.py --episodes 0,21,22
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipeline.episodes import load_tier  # noqa: E402

# Verbatim from the kit's metric_code/track_2/AUC.py.
ADD_THRESHOLDS_M = np.arange(0.01, 0.10, 0.01)
MPPE_KEYPOINT_RADIUS_M = 0.05
MPPE_KEYPOINT_DIRECTIONS = np.array(
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
     [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
    dtype=np.float64,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", default="tier_1_multiview_caption")
    parser.add_argument("--episodes", default="all", help="'all' or '0,21,22'")
    parser.add_argument("--max-vertices", type=int, default=2000,
                        help="vertices sampled from the object mesh for ADD")
    parser.add_argument("--json-out", default=None)
    return parser.parse_args()


def read_obj_vertices(path: Path, limit: int) -> np.ndarray:
    """Read 'v x y z' lines from a Wavefront OBJ. Stdlib parsing - no trimesh."""
    vertices = []
    with path.open() as handle:
        for line in handle:
            if line.startswith("v "):
                parts = line.split()
                vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
    array = np.asarray(vertices, dtype=np.float64)
    if array.shape[0] > limit:
        step = array.shape[0] // limit
        array = array[::step][:limit]
    return array


def add_auc(add: np.ndarray) -> float:
    """The kit's _add_auc for a single body: trapezoid over the 1-9 cm thresholds."""
    axis = np.linspace(0.0, 1.0, ADD_THRESHOLDS_M.size)
    curve = np.array([float(np.mean(add < t)) for t in ADD_THRESHOLDS_M])
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    return float(trapezoid(curve, x=axis))


def mppe_cm(achieved_p, achieved_R, reference_p, reference_R) -> float:
    """The kit's _mean_per_point_error_cm: six +/-5 cm keypoints, position AND orientation."""
    offsets = MPPE_KEYPOINT_DIRECTIONS * MPPE_KEYPOINT_RADIUS_M
    keypoints_a = achieved_p[:, None, :] + np.einsum("tij,kj->tki", achieved_R, offsets)
    keypoints_r = reference_p[:, None, :] + np.einsum("tij,kj->tki", reference_R, offsets)
    return float(np.linalg.norm(keypoints_a - keypoints_r, axis=-1).mean() * 100.0)


def score_episode(spec, max_vertices: int) -> dict | None:
    if not spec.reference_parquet.is_file():
        return None
    table = pq.read_table(spec.reference_parquet)
    row = {name: table.column(name)[0].as_py() for name in table.schema.names}
    position = np.asarray(row["object_root_position"], dtype=np.float64)
    rotation = Rotation.from_rotvec(
        np.asarray(row["object_root_axis_angle"], dtype=np.float64)
    ).as_matrix()

    mesh = spec.object_dir / "textured_mesh.obj"
    if not mesh.is_file():
        return None
    vertices = read_obj_vertices(mesh, max_vertices)
    if vertices.size == 0:
        return None

    # The do-nothing policy: the object stays in its frame-0 pose forever.
    frozen_p = np.repeat(position[0:1], position.shape[0], axis=0)
    frozen_R = np.repeat(rotation[0:1], rotation.shape[0], axis=0)

    # ADD: mean distance between corresponding model points under the two poses.
    points_frozen = frozen_p[:, None, :] + np.einsum("tij,kj->tki", frozen_R, vertices)
    points_reference = position[:, None, :] + np.einsum("tij,kj->tki", rotation, vertices)
    add = np.linalg.norm(points_frozen - points_reference, axis=-1).mean(axis=1)

    return {
        "episode_index": spec.episode_index,
        "object": spec.object_name,
        "frames": int(position.shape[0]),
        "vertices_sampled": int(vertices.shape[0]),
        "stationary_auc": round(add_auc(add), 4),
        "stationary_mppe_cm": round(mppe_cm(frozen_p, frozen_R, position, rotation), 4),
        "reference_travel_m": round(float(np.linalg.norm(position - position[0], axis=1).mean()), 4),
        "mean_add_m": round(float(add.mean()), 4),
    }


def main() -> int:
    args = parse_args()
    specs = load_tier(args.tier)
    if args.episodes != "all":
        wanted = {int(x) for x in args.episodes.split(",")}
        specs = [s for s in specs if s.episode_index in wanted]

    rows = []
    for spec in specs:
        result = score_episode(spec, args.max_vertices)
        if result is not None:
            rows.append(result)

    if not rows:
        print("no episodes scored", file=sys.stderr)
        return 1

    print("Do-nothing (object frozen at frame 0) scored with the OFFICIAL formulas")
    print(f"tier: {args.tier}   episodes: {len(rows)}")
    print()
    print("{:>3}  {:<24} {:>8} {:>10} {:>10}".format("ep", "object", "AUC", "MPPE cm", "ref move m"))
    print("-" * 60)
    for row in rows:
        print("{:>3}  {:<24} {:>8.4f} {:>10.2f} {:>10.4f}".format(
            row["episode_index"], row["object"][:24],
            row["stationary_auc"], row["stationary_mppe_cm"], row["reference_travel_m"]))

    auc = np.array([r["stationary_auc"] for r in rows])
    mppe = np.array([r["stationary_mppe_cm"] for r in rows])
    print("-" * 60)
    print("{:>3}  {:<24} {:>8.4f} {:>10.2f}".format("", "TIER MEAN (do nothing)", auc.mean(), mppe.mean()))
    print()
    print("  leaderboard context (Track 2 - Tier 1, 2026-10-20):")
    print("    CHORD Baseline        AUC 0.5104   MPPE 17.74")
    print("    best complete entry   AUC 0.6003   MPPE 12.66")
    print()
    print("  A trained policy must beat the do-nothing AUC above, not just a low MPPE.")

    if args.json_out:
        payload = {
            "tier": args.tier,
            "episodes": rows,
            "tier_mean_stationary_auc": round(float(auc.mean()), 4),
            "tier_mean_stationary_mppe_cm": round(float(mppe.mean()), 4),
        }
        with open(args.json_out, "w") as handle:
            json.dump(payload, handle, indent=2)
        print(f"\n  wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
