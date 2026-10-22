#!/usr/bin/env python3
"""Aggregate a repeated evaluation sweep and compare it with the baselines.

Three comparisons decide whether the orientation arm is worth continuing:

1. **The stationary floor.** Doing nothing scores Episode 0 at AUC 0.3769 /
   MPPE 32.591 cm. A policy below that floor is worse than not touching the
   object, whatever its MPPE looks like. This is the comparison that matters
   for the leaderboard, because Track 2 ranks by AUC.
2. **The previous Episode 0 policy**, the 10M checkpoint whose five repeats
   gave AUC 0.1127-0.1290 and MPPE 47.6-189.6 cm.
3. **Achieved object travel.** The reference object moves 0.3254 m. A policy
   that barely moves it can post a flattering MPPE while having done nothing,
   so travel is reported alongside and a near-zero value invalidates the rest.

Spread is reported, never a best repeat.

    .venvs/episode0/bin/python scripts/summarize_orientation_sweep.py outputs/step_D/sweep_ep0_orient_350M
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

#: scripts/leaderboard_floor.py, Episode 0, verified against NVIDIA's evaluator.
FLOOR = {"add_auc": 0.3769, "mppe_cm": 32.591, "reference_travel_m": 0.3254}

#: Five fresh-process evaluations of outputs/step_D/source_ep0_w1024/policy_10000384.safetensors.
PREVIOUS_10M = {
    "add_auc": [0.112709, 0.119296, 0.119055, 0.129017, 0.115119],
    "mppe_cm": [48.844, 81.313, 102.180, 189.555, 47.574],
}

METRICS = ("add_auc", "mppe_cm", "spider_sr_uncentered", "maniptrans_sr", "chord_sr")


def describe(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    quartiles = statistics.quantiles(ordered, n=4) if len(ordered) >= 2 else [ordered[0]] * 3
    return {
        "n": len(ordered),
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "std": statistics.stdev(ordered) if len(ordered) > 1 else 0.0,
        "iqr": quartiles[2] - quartiles[0],
        "min": ordered[0],
        "max": ordered[-1],
    }


def object_travel(trajectory: Path) -> float | None:
    """Straight-line distance the achieved object moved from its first frame."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return None
    try:
        table = pq.read_table(trajectory)
    except Exception:
        return None
    names = [n for n in table.column_names if "position" in n.lower() and "ref" not in n.lower()]
    if not names:
        return None
    import numpy as np

    data = np.asarray(table.column(names[0]).to_pylist(), dtype=float)
    if data.ndim != 2 or data.shape[0] < 2:
        return None
    return float(np.linalg.norm(data[-1] - data[0]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sweep_dir", type=Path)
    parser.add_argument("--run-dir", type=Path, default=None)
    args = parser.parse_args()

    eval_dir = REPO / "outputs/step_D/eval"
    by_step: dict[int, list[tuple[str, dict]]] = {}
    for report in sorted(eval_dir.glob("*.json")):
        match = re.match(r".*_(\d+)_r(\d+)$", report.stem)
        if not match or args.sweep_dir.name.replace("sweep_", "") not in report.stem:
            continue
        payload = json.loads(report.read_text())
        by_step.setdefault(int(match.group(1)), []).append((report.stem, payload.get("metrics", payload)))

    if not by_step:
        print(f"no sweep results found for {args.sweep_dir.name} in {eval_dir}")
        return 1

    previous = {name: describe(values) for name, values in PREVIOUS_10M.items()}
    print("=" * 78)
    print("BASELINES (Episode 0)")
    print(f"  stationary floor        AUC {FLOOR['add_auc']:.4f}   MPPE {FLOOR['mppe_cm']:.2f} cm")
    print(
        f"  previous 10M policy     AUC {previous['add_auc']['mean']:.4f}"
        f" +/- {previous['add_auc']['std']:.4f}   MPPE {previous['mppe_cm']['mean']:.2f}"
        f" +/- {previous['mppe_cm']['std']:.2f} cm   (n=5)"
    )

    for step in sorted(by_step):
        runs = by_step[step]
        print("=" * 78)
        print(f"CHECKPOINT step {step:,}   ({len(runs)} repeats, world_count=1, unassisted)")
        print(f"  {'metric':<24}{'mean':>10}{'median':>10}{'std':>10}{'IQR':>10}{'min':>10}{'max':>10}")
        summary: dict[str, dict] = {}
        for metric in METRICS:
            values = [m[metric] for _, m in runs if metric in m]
            if not values:
                continue
            stats = describe(values)
            summary[metric] = stats
            print(
                f"  {metric:<24}{stats['mean']:>10.4f}{stats['median']:>10.4f}"
                f"{stats['std']:>10.4f}{stats['iqr']:>10.4f}{stats['min']:>10.4f}{stats['max']:>10.4f}"
            )

        travels = []
        for tag, _ in runs:
            path = REPO / "outputs/step_D/trajectories" / tag / "episode_000000.parquet"
            value = object_travel(path)
            if value is not None:
                travels.append(value)
        if travels:
            stats = describe(travels)
            print(f"  {'object travel (m)':<24}{stats['mean']:>10.4f}{stats['median']:>10.4f}"
                  f"{stats['std']:>10.4f}{stats['iqr']:>10.4f}{stats['min']:>10.4f}{stats['max']:>10.4f}")
            print(f"       reference travels {FLOOR['reference_travel_m']:.4f} m")
            if stats["mean"] < 0.05:
                print("       WARNING: the object barely moved. Any MPPE here is the inert floor, not a score.")

        print("  --- verdict ---")
        if "add_auc" in summary:
            auc = summary["add_auc"]["mean"]
            delta = (auc - FLOOR["add_auc"]) / FLOOR["add_auc"] * 100.0
            verdict = "BEATS" if auc > FLOOR["add_auc"] else "WORSE THAN"
            print(f"  AUC {auc:.4f} {verdict} the stationary floor {FLOOR['add_auc']:.4f}  ({delta:+.1f}%)")
            previous_auc = previous["add_auc"]["mean"]
            print(
                f"  AUC {auc:.4f} vs the previous 10M policy {previous_auc:.4f}"
                f"  ({(auc - previous_auc) / previous_auc * 100.0:+.1f}%)"
            )
            overlap = summary["add_auc"]["min"] <= previous["add_auc"]["max"] and (
                previous["add_auc"]["min"] <= summary["add_auc"]["max"]
            )
            if overlap:
                print("  NOTE: the two repeat ranges OVERLAP. Treat the difference as unresolved")
                print("        and extend both to 20 repeats before deciding.")
        if "mppe_cm" in summary:
            mppe = summary["mppe_cm"]["mean"]
            print(
                f"  MPPE {mppe:.2f} cm vs floor {FLOOR['mppe_cm']:.2f} cm "
                f"({(mppe - FLOOR['mppe_cm']) / FLOOR['mppe_cm'] * 100.0:+.1f}%, lower is better)"
            )

    print("=" * 78)
    print("Track 2 ranks by AUC. An MPPE gain with an AUC loss is not progress.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
