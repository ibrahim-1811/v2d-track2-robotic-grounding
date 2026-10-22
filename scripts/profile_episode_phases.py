#!/usr/bin/env python3
"""Phase-profile every Tier 1 episode: how much is static, how much is manipulation.

Episode 0's diagnosis found that its stool is motionless for the first 4.7 s of
15.5 s, and that the 350M policy tracks that prefix almost perfectly then loses
the object the moment the tip-up starts. That motivated a reset-distribution
experiment -- but only if the phase structure generalises. This measures it.

Per episode, from the retargeted G1 reference parquet (CPU only, no container):

* first frame of significant object motion, and the static prefix before it
* the main manipulation window (where the object is actually moving)
* net displacement, path length and total orientation change
* the static / manipulating split
* the stationary-baseline AUC and MPPE from scripts/leaderboard_floor.py
* an "intervention opportunity" score: how much of the episode a
  manipulation-focused reset distribution would stop wasting

    .venvs/episode0/bin/python scripts/profile_episode_phases.py
    .venvs/episode0/bin/python scripts/profile_episode_phases.py --json outputs/.../phases.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parent.parent
SOMA = REPO / "data/adapter_out/whole_body/soma"
FLOOR = REPO / "outputs/step_D/leaderboard_floor.json"

#: "Displaced" = this far from the frame-0 pose. Used only for the static PREFIX,
#: i.e. when the object first leaves its initial pose.
MOVE_POSITION_M = 0.02
MOVE_ANGLE_RAD = math.radians(5.0)

#: "Actively moving" = instantaneous speed above these, smoothed over 11 frames.
#: Displacement-from-start is the wrong test for this: once an object is moved
#: and set down, every later frame would still count as manipulation.
SPEED_M_S = 0.03
ANGULAR_RATE_DEG_S = 5.0
SMOOTH = 11


def quat_angle_wxyz(q: np.ndarray, q0: np.ndarray) -> np.ndarray:
    """Geodesic angle between each wxyz quaternion and a reference, in radians."""
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    q0 = q0 / np.linalg.norm(q0)
    dot = np.abs(q @ q0)
    return 2.0 * np.arccos(np.clip(dot, 0.0, 1.0))


def load_object_track(parquet: Path) -> tuple[np.ndarray, np.ndarray]:
    row = pq.read_table(parquet).to_pylist()[0]
    position = np.asarray(row["object_body_position"], dtype=float)
    quaternion = np.asarray(row["object_body_wxyz"], dtype=float)
    # Shapes are (frames, bodies, 3/4) or already flat for a single body.
    if position.ndim == 3:
        position = position[:, 0, :]
    if quaternion.ndim == 3:
        quaternion = quaternion[:, 0, :]
    return position, quaternion


def profile(parquet: Path) -> dict:
    position, quaternion = load_object_track(parquet)
    frames = position.shape[0]
    displacement = np.linalg.norm(position - position[0], axis=1)
    angle = quat_angle_wxyz(quaternion, quaternion[0])
    displaced = (displacement > MOVE_POSITION_M) | (angle > MOVE_ANGLE_RAD)
    indices = np.flatnonzero(displaced)
    first_motion = int(indices[0]) if indices.size else -1
    path_length = float(np.linalg.norm(np.diff(position, axis=0), axis=1).sum())

    # Instantaneous motion, smoothed, is what "actively manipulating" means.
    kernel = np.ones(SMOOTH) / SMOOTH
    speed = np.zeros(frames)
    speed[1:] = np.linalg.norm(np.diff(position, axis=0), axis=1) * 50.0
    speed = np.convolve(speed, kernel, mode="same")
    rate = np.zeros(frames)
    rate[1:] = np.degrees(
        np.array([
            quat_angle_wxyz(quaternion[i : i + 1], quaternion[i - 1])[0] for i in range(1, frames)
        ])
    ) * 50.0
    rate = np.convolve(rate, kernel, mode="same")
    moving = (speed > SPEED_M_S) | (rate > ANGULAR_RATE_DEG_S)
    active = np.flatnonzero(moving)
    last_motion = int(active[-1]) if active.size else -1

    return {
        "frames": frames,
        "seconds": frames / 50.0,
        "first_motion_frame": first_motion,
        "last_motion_frame": last_motion,
        "static_prefix_frames": first_motion if first_motion >= 0 else frames,
        "static_prefix_s": (first_motion if first_motion >= 0 else frames) / 50.0,
        "active_start_frame": int(np.flatnonzero(moving)[0]) if moving.any() else -1,
        "active_end_frame": last_motion,
        "manipulation_frames": int(moving.sum()),
        "manipulation_fraction": float(moving.mean()),
        "static_fraction": float(1.0 - moving.mean()),
        "net_displacement_m": float(displacement[-1]),
        "max_displacement_m": float(displacement.max()),
        "path_length_m": path_length,
        "total_rotation_deg": float(np.degrees(angle.max())),
        "final_rotation_deg": float(np.degrees(angle[-1])),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    floors = {}
    if FLOOR.exists():
        payload = json.loads(FLOOR.read_text())
        rows = payload["episodes"] if isinstance(payload, dict) and "episodes" in payload else payload
        floors = {int(row["episode_index"]): row for row in rows}

    results = []
    for directory in sorted(SOMA.glob("sequence_id=*")):
        parquet = directory / "robot_name=g1" / "data.parquet"
        if not parquet.exists():
            continue
        name = directory.name.replace("sequence_id=", "")
        try:
            record = profile(parquet)
        except Exception as error:  # noqa: BLE001 - report and continue over 30 episodes
            print(f"  SKIP {name}: {error}")
            continue
        record["sequence"] = name
        results.append(record)

    # Match episode indices through the floor file's object names where possible.
    by_object: dict[str, list[dict]] = {}
    for index, row in floors.items():
        by_object.setdefault(row["object"], []).append({**row, "episode_index": index})
    used: set[int] = set()
    for record in results:
        record["episode_index"] = None
        for index, row in floors.items():
            if index in used:
                continue
            if row["object"] in record["sequence"] and row["frames"] == record["frames"]:
                record["episode_index"] = index
                record["stationary_auc"] = row["stationary_auc"]
                record["stationary_mppe_cm"] = row["stationary_mppe_cm"]
                used.add(index)
                break

    for record in results:
        # How much of the episode a manipulation-focused reset would stop wasting,
        # weighted by how much the object actually has to move.
        record["intervention_score"] = round(
            record["static_fraction"] * min(record["path_length_m"] / 0.5, 2.0), 3
        )

    results.sort(key=lambda r: -r["intervention_score"])

    header = (
        f"{'ep':>3} {'object / sequence':<34}{'frames':>7}{'static s':>9}{'1st mv':>7}"
        f"{'manip %':>8}{'disp m':>8}{'path m':>8}{'rot deg':>8}{'floorAUC':>9}{'score':>7}"
    )
    print(header)
    print("-" * len(header))
    for record in results:
        index = record.get("episode_index")
        print(
            f"{index if index is not None else '?':>3} {record['sequence'][:34]:<34}"
            f"{record['frames']:>7}{record['static_prefix_s']:>9.1f}{record['first_motion_frame']:>7}"
            f"{record['manipulation_fraction'] * 100:>8.0f}{record['net_displacement_m']:>8.3f}"
            f"{record['path_length_m']:>8.3f}{record['total_rotation_deg']:>8.0f}"
            f"{record.get('stationary_auc', float('nan')):>9.4f}{record['intervention_score']:>7.2f}"
        )

    static = np.array([r["static_fraction"] for r in results])
    prefix = np.array([r["static_prefix_s"] for r in results])
    print()
    print(f"episodes profiled              : {len(results)}")
    print(f"static fraction  median / mean : {np.median(static):.2f} / {static.mean():.2f}")
    print(f"static prefix s  median / max  : {np.median(prefix):.1f} / {prefix.max():.1f}")
    print(f"episodes with >2 s static prefix: {(prefix > 2.0).sum()} of {len(results)}")
    print(f"episodes with >50% static       : {(static > 0.5).sum()} of {len(results)}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2) + "\n")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
