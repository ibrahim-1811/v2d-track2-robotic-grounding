#!/usr/bin/env python3
"""Training manifest for any tier, from the staged ``episode_manifest.json`` files.

``make_training_manifest.py`` is Tier 1's original and needs pyarrow plus the
retargeted parquets. Everything a trainer needs is already recorded per episode
by ``run_tier_pipeline.py stage`` -- sequence, episode index, the
``scene_support`` decision (collision shapes counted in the tier's own support
USDA), frame counts and dropped frames -- so this is stdlib only and runs on the
cluster's Python 3.9.

``stationary_auc`` is carried over from the Tier 1 manifest for the same
sequence and labelled as such: every tier is scored against the clean Tier 1
reference, so it is the right floor to read evaluations against, but it was
computed from the clean frame-0 pose, not this tier's.

    python3 scripts/train/make_tier_manifest.py --tier tier_2_synthetic_noise \
        -o training_manifest_tier2.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ADAPTER_ROOT = {"tier_1_multiview_caption": "adapter_out", "tier_2_synthetic_noise": "adapter_out_tier_2"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tier", required=True)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--floors", type=Path, default=REPO / "training_manifest_verified.json")
    parser.add_argument("--require-references", action="store_true",
                        help="fail unless every episode's retargeted G1 parquet exists")
    args = parser.parse_args()

    adapter = ADAPTER_ROOT.get(args.tier, f"adapter_out_{args.tier}")
    soma = REPO / "data" / adapter / "whole_body" / "soma"
    floors = {e["sequence_id"]: e["stationary_auc"] for e in json.loads(args.floors.read_text())["episodes"]}

    episodes, missing = [], []
    for manifest in sorted(soma.glob("*/episode_manifest.json")):
        staged = json.loads(manifest.read_text())
        if staged["tier"] != args.tier:
            sys.exit(f"{manifest} belongs to {staged['tier']}, not {args.tier}")
        sequence = staged["sequence_id"]
        parquet = soma / f"sequence_id={sequence}" / "robot_name=g1" / "data.parquet"
        if not parquet.is_file():
            missing.append(staged["episode_index"])
        episodes.append({
            "sequence_id": sequence,
            "episode_index": staged["episode_index"],
            "task_parquet_container": f"/work/{adapter}/whole_body/soma/sequence_id={sequence}/robot_name=g1/data.parquet",
            "scene_support": staged["scene_support"],
            "support_collision_shapes": staged["support_shape_count"],
            "frames": staged["retarget_frames"],
            "dropped_frames": staged["dropped_frames"],
            "stationary_auc": floors[sequence],
            "stationary_auc_source": "tier_1_multiview_caption clean frame-0 pose",
            "reference_present": parquet.is_file(),
        })
    episodes.sort(key=lambda e: e["episode_index"])
    if len(episodes) != 30 or [e["episode_index"] for e in episodes] != list(range(30)):
        sys.exit(f"expected episodes 0..29 under {soma}, found {[e['episode_index'] for e in episodes]}")
    if args.require_references and missing:
        sys.exit(f"{len(missing)} episodes have no retargeted G1 parquet yet: {missing}")

    args.output.write_text(json.dumps({"tier": args.tier, "adapter_root": adapter, "episodes": episodes}, indent=1) + "\n")
    print(f"wrote {args.output}: {len(episodes)} episodes, {sum(e['scene_support'] for e in episodes)} with support, "
          f"{30 - len(missing)}/30 G1 references present (adapter root data/{adapter})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
