"""Training manifest for our Tier 3 references (data/adapter_out_<tier>), Tier 1 free.

Same fields cluster/train_tier_episode.sbatch and submit_tier_lanes.sh read from the Tier 1/2
manifests, for every episode whose G1 reference has come back from the retarget. Two
differences, both to keep the Tier 1 caption out of Tier 3:

* ``stationary_auc`` (only printed next to evaluation results) is the do-nothing score against
  OUR Tier 3 input -- frame-0 pose frozen vs the reconstructed trajectory -- not against Tier 1.
* ``scene_support`` counts collision shapes in the retargeted stage, as for the other tiers.

Run inside containers/flash_chord.sif (pyarrow)::

    python scripts/tier3/make_tier3_manifest.py [--tier tier_3_cari4d_ours]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from pipeline.episodes import count_support_collision_shapes, out_root_for_tier, tier_root_for  # noqa: E402

ADD_THRESHOLDS_M = np.arange(0.01, 0.10, 0.01)


def read_obj_vertices(path: Path, limit: int = 2000) -> np.ndarray:
    v = np.asarray([line.split()[1:4] for line in path.open() if line.startswith("v ")], dtype=np.float64)
    return v[:: max(1, len(v) // limit)][:limit]


def stationary_auc(reference: Path, mesh: Path) -> float:
    table = pq.read_table(reference, columns=["object_root_position", "object_root_axis_angle"])
    position = np.asarray(table.column(0)[0].as_py(), dtype=np.float64)
    rotation = Rotation.from_rotvec(np.asarray(table.column(1)[0].as_py(), dtype=np.float64)).as_matrix()
    vertices = read_obj_vertices(mesh)
    moving = position[:, None] + np.einsum("tij,kj->tki", rotation, vertices)
    frozen = position[0] + vertices @ rotation[0].T
    add = np.linalg.norm(moving - frozen[None], axis=-1).mean(axis=1)
    curve = np.array([np.mean(add < t) for t in ADD_THRESHOLDS_M])
    return round(float(np.trapezoid(curve, x=np.linspace(0.0, 1.0, ADD_THRESHOLDS_M.size))), 4)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tier", default="tier_3_cari4d_ours")
    parser.add_argument("--out", default=str(REPO / "training_manifest_tier3.json"))
    args = parser.parse_args()

    tier_root, out_root = tier_root_for(args.tier), out_root_for_tier(args.tier)
    rows = [json.loads(line) for line in (tier_root / "meta/episodes_metadata.jsonl").read_text().splitlines() if line.strip()]
    episodes, skipped = [], []
    for row in sorted(rows, key=lambda r: r["episode_index"]):
        sequence = row["sequence_id"]
        retarget = out_root / "whole_body/soma" / f"sequence_id={sequence}"
        reference = retarget / "robot_name=g1/data.parquet"
        if not reference.is_file():
            continue
        # Only references that passed run_tier_pipeline.py validate (run here, on the files we
        # train from) are trainable; a failed reconstruction must not reach the GPUs.
        record = REPO / "outputs" / args.tier / "records" / f"{row['episode_index']:06d}_validate.json"
        verdict = json.loads(record.read_text()) if record.is_file() else {}
        if verdict.get("checks_failed", 1) != 0 or verdict.get("checks_passed", 0) == 0:
            skipped.append(row["episode_index"])
            continue
        # Validation cannot see a scene that is self-consistent but turned (episodes 12, 14: 178 and
        # 73 deg), so the frame-0 human alignment check must have passed too: either the converter
        # placed the scene by the human, or the object-only alignment agreed with it within 20 deg.
        conversion = tier_root / "conversion" / f"episode_{row['episode_index']:06d}.json"
        report = json.loads(conversion.read_text()) if conversion.is_file() else {}
        align = report.get("alignment") or report.get("alignment_check")
        if not align or not (align.get("method") == "human" or align.get("rotation_deg", 999.0) <= 20.0):
            skipped.append(row["episode_index"])
            continue
        shapes = count_support_collision_shapes(out_root / "whole_body/reconstructed_stage" / f"{sequence}_support.usda")
        frames = pq.ParquetFile(reference).metadata.row_group(0).num_rows and len(
            pq.read_table(reference, columns=["robot_root_position"]).column(0)[0].as_py())
        episodes.append({
            "sequence_id": sequence,
            "episode_index": row["episode_index"],
            "task_parquet_container": f"/work/{out_root.name}/whole_body/soma/sequence_id={sequence}/robot_name=g1/data.parquet",
            "scene_support": shapes > 0,
            "support_collision_shapes": shapes,
            "frames": frames,
            "dropped_frames": [],
            "stationary_auc": stationary_auc(reference, retarget / "object/textured_mesh.obj"),
            "stationary_auc_source": f"{args.tier} input, frame-0 pose frozen (no Tier 1 data)",
            "reference_present": True,
        })
    manifest = {"tier": args.tier, "adapter_root": out_root.name,
                "note": "Tier 3: CARI4D reconstruction aligned with Tier 1's frame-0 object pose only "
                        "(scripts/tier3/cari4d_to_lerobot.py), retargeted with the unchanged pipeline.",
                "episodes": episodes}
    Path(args.out).write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"{len(episodes)} episode(s) -> {args.out}; not validated or failed, left out: {skipped}")
    for e in episodes:
        print(f"  ep{e['episode_index']:02d} support={str(e['scene_support']).lower():5s} frames={e['frames']} "
              f"stationary_auc={e['stationary_auc']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
