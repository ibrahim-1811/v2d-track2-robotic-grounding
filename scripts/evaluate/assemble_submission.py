#!/usr/bin/env python3
"""Assemble one tier's ``results/`` directory for ``tools/pack_submission.py``.

For every episode, every trained run under ``outputs/cluster/`` is a candidate
(stock, keypoint arm, Episode 7's resumes, ...). Each candidate checkpoint has
been evaluated by ``cluster/evaluate_policy.sbatch`` as N fresh, unassisted,
world-1 runs of NVIDIA's unchanged evaluator. Selection is pre-declared:

1. pick the candidate with the highest **mean** ``add_auc`` over its runs
   (ties: lower mean MPPE). Choosing among learned checkpoints is allowed; a
   frozen-object trajectory is not (FAQ: trajectories from learned policies).
   ``--select balanced`` instead takes the lowest mean MPPE among candidates
   within ``--auc-margin`` (0.02) of that best mean AUC: the 75th-percentile
   standard error of a 5-run mean AUC is 0.016 (154 Tier 1 evaluations,
   2026-10-29), so those AUCs are indistinguishable and MPPE decides. The kit
   requires the same CSV in all four metric competitions of a tier, so the
   trade-off has to live inside one CSV, never in a per-metric upload;
2. submit **rep 1** of that candidate's evaluation, whatever its score -- never
   the best of the N rollouts, which would not reproduce.

Writes ``<out>/episode_XXXXXX.parquet`` and ``<out>/episode_XXXXXX.json`` (the
evaluator's own trajectory and metrics, copied byte for byte; the packer
cross-checks the metrics) and ``<out>/../selection.json`` -- outside ``<out>``,
because the packer reads every JSON in the results tree as an evaluator report.
Stdlib only.

    python3 scripts/evaluate/assemble_submission.py --prefix t1 -o outputs/submission/tier1/results
    python3 scripts/evaluate/assemble_submission.py --prefix t2 -o outputs/submission/tier2/results
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CLUSTER = REPO / "outputs" / "cluster"


def episode_of(run: str, prefix: str) -> int | None:
    match = re.match(rf"^{prefix}_ep(\d\d)_", run)
    if match:
        return int(match.group(1))
    if prefix == "t1" and run.startswith("ep0_"):  # Episode 0's first cluster run predates the t1_ prefix
        return 0
    return None


def candidates(prefix: str, min_reps: int) -> dict[int, list[dict]]:
    found: dict[int, list[dict]] = {}
    for run_dir in sorted(p for p in CLUSTER.iterdir() if p.is_dir()):
        episode = episode_of(run_dir.name, prefix)
        if episode is None:
            continue
        for eval_dir in sorted(run_dir.glob("eval_*")):
            if not re.fullmatch(r"eval_\d+", eval_dir.name):  # EVAL_TAG'd test evaluations are not candidates
                continue
            reps = []
            for metrics_file in sorted(eval_dir.glob("rep*.json"), key=lambda p: int(p.stem[3:])):
                trajectory = eval_dir / metrics_file.stem / f"episode_{episode:06d}.parquet"
                if trajectory.is_file():
                    report = json.loads(metrics_file.read_text())
                    reps.append({"rep": int(metrics_file.stem[3:]), "metrics": report["metrics"],
                                 "checkpoint_sha256": report.get("checkpoint_sha256"), "trajectory": trajectory,
                                 "report": metrics_file})
            if len(reps) < min_reps or reps[0]["rep"] != 1:
                continue
            found.setdefault(episode, []).append({
                "run": run_dir.name,
                "step": int(eval_dir.name[5:]),
                "reps": reps,
                "mean_auc": statistics.fmean(r["metrics"]["add_auc"] for r in reps),
                "mean_mppe_cm": statistics.fmean(r["metrics"]["mppe_cm"] for r in reps),
            })
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prefix", required=True, help="t1 or t2: the run-name prefix of the tier")
    parser.add_argument("-o", "--out", type=Path, required=True)
    parser.add_argument("--min-reps", type=int, default=3, help="ignore evaluations with fewer runs")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--select", choices=("auc", "balanced"), default="auc",
                        help="auc: highest mean AUC; balanced: lowest mean MPPE within --auc-margin of it")
    parser.add_argument("--auc-margin", type=float, default=0.02, help="balanced only: mean-AUC margin")
    args = parser.parse_args()

    found = candidates(args.prefix, args.min_reps)
    missing = [e for e in range(args.episodes) if e not in found]
    if missing:
        print(f"error: no evaluated candidate for episodes {missing}", file=sys.stderr)
        return 1

    if args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True)
    selection = []
    print(f"{'ep':>2}  {'chosen run':34s} {'step':>10s} {'mean AUC':>8s} {'rep1 AUC':>8s} {'rep1 MPPE':>9s}  others")
    for episode in range(args.episodes):
        ranked = sorted(found[episode], key=lambda c: (-c["mean_auc"], c["mean_mppe_cm"]))
        best = ranked[0]
        if args.select == "balanced":
            best = min((c for c in ranked if c["mean_auc"] >= ranked[0]["mean_auc"] - args.auc_margin),
                       key=lambda c: (c["mean_mppe_cm"], -c["mean_auc"]))
        rep1 = best["reps"][0]
        shutil.copy2(rep1["trajectory"], args.out / f"episode_{episode:06d}.parquet")
        shutil.copy2(rep1["report"], args.out / f"episode_{episode:06d}.json")
        others = "  ".join(f"{c['run']}@{c['step'] // 1_000_000}M={c['mean_auc']:.3f}/{c['mean_mppe_cm']:.1f}c"
                           for c in ranked if c is not best)
        print(f"{episode:2d}  {best['run']:34s} {best['step']:10d} {best['mean_auc']:8.4f} "
              f"{rep1['metrics']['add_auc']:8.4f} {rep1['metrics']['mppe_cm']:8.2f}c  {others}")
        selection.append({
            "episode_index": episode, "run": best["run"], "step": best["step"], "submitted_rep": 1,
            "checkpoint_sha256": rep1["checkpoint_sha256"], "trajectory": str(rep1["trajectory"].relative_to(REPO)),
            "mean_auc": best["mean_auc"], "mean_mppe_cm": best["mean_mppe_cm"], "reps": len(best["reps"]),
            "rep1_metrics": {k: rep1["metrics"][k] for k in ("add_auc", "mppe_cm", "spider_sr_uncentered", "maniptrans_sr")},
            "select": args.select,
            "candidates": [{"run": c["run"], "step": c["step"], "mean_auc": c["mean_auc"], "mean_mppe_cm": c["mean_mppe_cm"],
                            "reps": len(c["reps"])} for c in ranked],
        })
    (args.out.parent / "selection.json").write_text(json.dumps(selection, indent=1) + "\n")
    mean = statistics.fmean(s["rep1_metrics"]["add_auc"] for s in selection)
    print(f"\n{len(selection)} episodes -> {args.out}  (evaluator mean AUC of the submitted rollouts: {mean:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
