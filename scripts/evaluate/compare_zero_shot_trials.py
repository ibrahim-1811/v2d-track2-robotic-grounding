#!/usr/bin/env python3
"""Compare two zero-shot tier trials against each other and the stationary floor.

The Track 2 leaderboard ranks by AUC, and ``leaderboard_floor.py`` showed that a
frozen object already scores a tier mean of 0.2661. So a trial's result is only
meaningful next to two numbers: the previous trial (did more training help?) and
that episode's own do-nothing floor (did we beat not touching the object?).

Reads each trial's ``run_manifest.json`` - the evaluator's own metrics - plus
``outputs/step_D/leaderboard_floor.json``. Note that ``pack_submission.py``
re-scores the TRAJECTORY against the released meshes and lands ~0.001 away from
the evaluator's AUC (the kit's ``v2dlb/vertices.py`` documents why); the packer's
CSV is the submitted number, this table is for reading the per-episode shape.

    .venvs/episode0/bin/python scripts/evaluate/compare_zero_shot_trials.py \
        --baseline outputs/step_D/submission_trial_20261021/run_manifest.json \
        --candidate outputs/step_D/submission_trial2_orient350M/run_manifest.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--floor", default=str(ROOT / "outputs/step_D/leaderboard_floor.json"))
    parser.add_argument("--json-out", default=None)
    return parser.parse_args()


def load(manifest_path: str) -> tuple[dict, dict]:
    manifest = json.loads(Path(manifest_path).read_text())
    by_episode = {record["episode_index"]: record["metrics"] for record in manifest["runs"]}
    return manifest, by_episode


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def main() -> int:
    args = parse_args()
    base_manifest, base = load(args.baseline)
    cand_manifest, cand = load(args.candidate)
    floor = {item["episode_index"]: item for item in json.loads(Path(args.floor).read_text())["episodes"]}

    shared = sorted(set(base) & set(cand))
    print("baseline : {}".format(base_manifest["method"]))
    print("candidate: {}".format(cand_manifest["method"]))
    print("episodes compared: {} (baseline has {}, candidate has {})".format(
        len(shared), len(base), len(cand)))
    print()
    header = "{:>3}  {:<24} {:>8} {:>8} {:>8}   {:>8} {:>8} {:>8}  {:>5}"
    print(header.format("ep", "object", "AUCbase", "AUCcand", "AUCfloor",
                        "MPPEbase", "MPPEcand", "MPPEfloor", "vs fl"))
    print("-" * 104)
    rows = []
    for idx in shared:
        b, c, f = base[idx], cand[idx], floor.get(idx, {})
        auc_floor = f.get("stationary_auc", float("nan"))
        beats = "yes" if c["add_auc"] > auc_floor else "no"
        rows.append({"episode_index": idx, "object": f.get("object", "?"),
                     "auc_baseline": b["add_auc"], "auc_candidate": c["add_auc"],
                     "auc_floor": auc_floor,
                     "mppe_baseline_cm": b["mppe_cm"], "mppe_candidate_cm": c["mppe_cm"],
                     "mppe_floor_cm": f.get("stationary_mppe_cm", float("nan")),
                     "candidate_beats_floor_auc": c["add_auc"] > auc_floor})
        print(header.format(idx, f.get("object", "?")[:24], "{:.4f}".format(b["add_auc"]),
                            "{:.4f}".format(c["add_auc"]), "{:.4f}".format(auc_floor),
                            "{:.2f}".format(b["mppe_cm"]), "{:.2f}".format(c["mppe_cm"]),
                            "{:.2f}".format(f.get("stationary_mppe_cm", float("nan"))), beats))

    print("-" * 104)
    summary = {
        "episodes": len(shared),
        "auc_baseline": mean([r["auc_baseline"] for r in rows]),
        "auc_candidate": mean([r["auc_candidate"] for r in rows]),
        "auc_floor": mean([r["auc_floor"] for r in rows]),
        "mppe_baseline_cm": mean([r["mppe_baseline_cm"] for r in rows]),
        "mppe_candidate_cm": mean([r["mppe_candidate_cm"] for r in rows]),
        "mppe_floor_cm": mean([r["mppe_floor_cm"] for r in rows]),
        "episodes_candidate_beats_floor_auc": sum(r["candidate_beats_floor_auc"] for r in rows),
        "episodes_candidate_beats_baseline_auc": sum(
            r["auc_candidate"] > r["auc_baseline"] for r in rows),
    }
    print(header.format("", "TIER MEAN",
                        "{:.4f}".format(summary["auc_baseline"]),
                        "{:.4f}".format(summary["auc_candidate"]),
                        "{:.4f}".format(summary["auc_floor"]),
                        "{:.2f}".format(summary["mppe_baseline_cm"]),
                        "{:.2f}".format(summary["mppe_candidate_cm"]),
                        "{:.2f}".format(summary["mppe_floor_cm"]),
                        "{}/{}".format(summary["episodes_candidate_beats_floor_auc"], len(shared))))
    print()
    delta = summary["auc_candidate"] - summary["auc_baseline"]
    print("AUC  candidate - baseline : {:+.4f}  ({:+.1f}%)".format(
        delta, 100.0 * delta / summary["auc_baseline"]))
    gap = summary["auc_candidate"] - summary["auc_floor"]
    print("AUC  candidate - floor    : {:+.4f}  ({:+.1f}%)".format(
        gap, 100.0 * gap / summary["auc_floor"]))
    print("MPPE candidate - baseline : {:+.2f} cm".format(
        summary["mppe_candidate_cm"] - summary["mppe_baseline_cm"]))
    print("episodes where candidate beats baseline AUC: {}/{}".format(
        summary["episodes_candidate_beats_baseline_auc"], len(shared)))
    print("episodes where candidate beats do-nothing  : {}/{}".format(
        summary["episodes_candidate_beats_floor_auc"], len(shared)))

    if args.json_out:
        Path(args.json_out).write_text(json.dumps({"summary": summary, "episodes": rows}, indent=2) + "\n")
        print("\nwrote {}".format(args.json_out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
