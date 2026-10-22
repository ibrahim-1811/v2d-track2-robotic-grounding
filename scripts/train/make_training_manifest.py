#!/usr/bin/env python3
"""Produce the per-episode facts a trainer needs, so nobody has to rediscover them.

Two of these will stop a run dead if you get them wrong:

* **``scene.support``** must be ``false`` for episodes whose support-surface USDA
  declares no collision shapes. FlashCHORD raises on an empty stub. We decide it
  per episode by counting ``def Cylinder`` / ``def Cube`` in the tier's USDA.
* **``task.parquet``** is an absolute *container* path and contains ``=``
  characters, so Hydra needs it wrapped in inner single quotes.

Also carries each episode's stationary-baseline AUC, because a policy that does
not beat that floor is worse than submitting a frozen object.

    .venvs/episode0/bin/python scripts/train/make_training_manifest.py \
        -o outputs/step_D/training_manifest.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SOMA = REPO / "data/adapter_out/whole_body/soma"
STAGE = REPO / "data/adapter_out/whole_body/reconstructed_stage"
FLOOR = REPO / "outputs/step_D/leaderboard_floor.json"
CONTAINER_ROOT = "/work/adapter_out/whole_body/soma"

SHAPE = re.compile(r"\bdef\s+(Cylinder|Cube|Sphere|Capsule|Mesh)\b")


def pq_frames(parquet: Path) -> int:
    """Frame count of a retargeted reference, for disambiguating floor records."""
    import pyarrow.parquet as pq

    return len(pq.read_table(parquet, columns=["object_body_position"]).to_pylist()[0]["object_body_position"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-o", "--output", type=Path, default=REPO / "outputs/step_D/training_manifest.json")
    args = parser.parse_args()

    floor_rows = []
    if FLOOR.exists():
        payload = json.loads(FLOOR.read_text())
        floor_rows = payload["episodes"] if isinstance(payload, dict) and "episodes" in payload else payload
    # Object names are ambiguous substrings of each other -- "skinny_wood_chair"
    # also occurs inside "big_red_bowl_under_skinny_wood_chair". Match on the
    # frame count too, and consume each floor record only once.
    used: set[int] = set()

    episodes = []
    for directory in sorted(SOMA.glob("sequence_id=*")):
        sequence = directory.name.replace("sequence_id=", "")
        parquet = directory / "robot_name=g1" / "data.parquet"
        if not parquet.exists():
            continue
        usda = STAGE / f"{sequence}_support.usda"
        shapes = len(SHAPE.findall(usda.read_text())) if usda.exists() else 0
        frames = pq_frames(parquet)
        floor = next(
            (
                row
                for row in floor_rows
                if row["episode_index"] not in used
                and row["object"] in sequence
                and row["frames"] == frames
            ),
            None,
        )
        if floor is not None:
            used.add(floor["episode_index"])
        episodes.append(
            {
                "sequence_id": sequence,
                "episode_index": floor["episode_index"] if floor else None,
                "task_parquet_container": f"{CONTAINER_ROOT}/sequence_id={sequence}/robot_name=g1/data.parquet",
                "scene_support": shapes > 0,
                "support_collision_shapes": shapes,
                "stationary_auc": floor["stationary_auc"] if floor else None,
                "stationary_mppe_cm": floor["stationary_mppe_cm"] if floor else None,
                "frames": frames,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"episodes": episodes}, indent=2) + "\n")

    with_support = sum(1 for e in episodes if e["scene_support"])
    try:
        shown = args.output.resolve().relative_to(REPO)
    except ValueError:
        shown = args.output
    print(f"{len(episodes)} episodes -> {shown}")
    print(f"  scene.support=true  : {with_support}")
    print(f"  scene.support=false : {len(episodes) - with_support}  (empty support stub; FlashCHORD raises otherwise)")
    print()
    print(f"  {'ep':>3}  {'support':<8}{'floorAUC':>9}  sequence")
    for episode in sorted(episodes, key=lambda e: (e["episode_index"] is None, e["episode_index"])):
        index = episode["episode_index"]
        floor = episode["stationary_auc"]
        print(
            f"  {index if index is not None else '?':>3}  {str(episode['scene_support']).lower():<8}"
            f"{floor if floor is not None else float('nan'):>9.4f}  {episode['sequence_id'][:46]}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
