#!/usr/bin/env python3
"""Decide whether a policy is actually manipulating the object, or just standing there.

Step D. MPPE alone cannot answer this. On Episode 21 a policy that never touches
the object scores ~55 cm, and a policy that flings it across the room scores
~109 cm - so a "good" MPPE near the floor and a "bad" one far above it can both
mean "no useful manipulation". The distinguishing measurement is how far the
object actually moved, compared against two counterfactuals:

  * STATIONARY  - the object frozen at its first achieved pose. This is what a
    policy that never makes contact produces, and it is the real baseline to
    beat.
  * REFERENCE   - how far the object is supposed to travel.

A policy is manipulating when its achieved object displacement approaches the
reference's and its MPPE drops below the stationary baseline. A policy whose
displacement greatly EXCEEDS the reference is destabilising the object, which
is the failure mode the actor-only transfer smoke test exhibited.

Reads the parquet that ``evaluate_policy.py --object-trajectories-output``
writes (``flash_chord_object_trajectories_v1``). Host-side: needs only numpy and
pyarrow, so run it with the project venv.

    .venvs/episode0/bin/python scripts/analyze_manipulation.py \
        outputs/step_D/trajectories/*/episode_*.parquet
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

AXES = ("x", "y", "z")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectories", nargs="+", type=Path)
    parser.add_argument("--world", type=int, default=0,
                        help="which evaluation world to analyse (the packer scores world 0)")
    parser.add_argument("--json-out", default=None)
    return parser.parse_args()


def load(path: Path, world: int) -> dict:
    table = pq.read_table(path)
    columns = {name: table.column(name).to_numpy() for name in table.schema.names}
    mask = columns["environment_id"] == world
    if not mask.any():
        raise ValueError(f"{path}: no rows for world {world}")
    achieved = np.stack([columns["achieved_position_" + a][mask] for a in AXES], axis=1)
    reference = np.stack([columns["reference_position_" + a][mask] for a in AXES], axis=1)
    return {"achieved": achieved.astype(np.float64), "reference": reference.astype(np.float64)}


def analyse(path: Path, world: int) -> dict:
    data = load(path, world)
    achieved, reference = data["achieved"], data["reference"]
    frames = achieved.shape[0]

    # Displacement from the first frame: "did the object go anywhere?"
    achieved_travel = np.linalg.norm(achieved - achieved[0], axis=1)
    reference_travel = np.linalg.norm(reference - reference[0], axis=1)

    # Tracking error, and the counterfactual error of never moving the object.
    error = np.linalg.norm(achieved - reference, axis=1)
    stationary = np.linalg.norm(achieved[0][None, :] - reference, axis=1)

    mppe_cm = 100.0 * error.mean()
    stationary_cm = 100.0 * stationary.mean()
    # Fraction of the stationary baseline's error that the policy removed.
    # 0 = no better than standing still; 1 = perfect tracking; <0 = worse.
    improvement = (stationary_cm - mppe_cm) / stationary_cm if stationary_cm > 0 else float("nan")
    travel_ratio = achieved_travel.mean() / reference_travel.mean() if reference_travel.mean() > 0 else float("nan")

    if improvement > 0.05 and 0.3 < travel_ratio < 2.0:
        verdict = "MANIPULATING"
    elif travel_ratio > 2.0:
        verdict = "DESTABILISING (object moves far more than the reference)"
    elif abs(improvement) <= 0.05:
        verdict = "INERT (indistinguishable from leaving the object alone)"
    else:
        verdict = "WORSE THAN INERT"

    return {
        "trajectory": str(path),
        "frames": int(frames),
        "achieved_travel_mean_m": round(float(achieved_travel.mean()), 4),
        "achieved_travel_max_m": round(float(achieved_travel.max()), 4),
        "reference_travel_mean_m": round(float(reference_travel.mean()), 4),
        "reference_travel_max_m": round(float(reference_travel.max()), 4),
        "travel_ratio_achieved_over_reference": round(float(travel_ratio), 4),
        "mppe_cm": round(float(mppe_cm), 4),
        "stationary_baseline_cm": round(float(stationary_cm), 4),
        "improvement_over_stationary": round(float(improvement), 4),
        "verdict": verdict,
    }


def main() -> int:
    args = parse_args()
    results = []
    for path in args.trajectories:
        try:
            results.append(analyse(path, args.world))
        except Exception as error:  # noqa: BLE001
            print(f"  SKIP {path}: {type(error).__name__}: {error}", file=sys.stderr)

    if not results:
        return 1

    header = "{:<34} {:>9} {:>9} {:>7} {:>9} {:>9} {:>7}".format(
        "trajectory", "obj move", "ref move", "ratio", "MPPE cm", "static cm", "gain")
    print(header)
    print("-" * len(header))
    for row in results:
        print("{:<34} {:>9.4f} {:>9.4f} {:>7.2f} {:>9.2f} {:>9.2f} {:>6.1%}".format(
            Path(row["trajectory"]).parent.name[:34],
            row["achieved_travel_mean_m"], row["reference_travel_mean_m"],
            row["travel_ratio_achieved_over_reference"],
            row["mppe_cm"], row["stationary_baseline_cm"],
            row["improvement_over_stationary"]))
    print()
    for row in results:
        print("  {:<32} {}".format(Path(row["trajectory"]).parent.name[:32], row["verdict"]))
    print()
    print("  'gain' is the fraction of the stationary baseline's error removed.")
    print("  <=5% means the policy is doing nothing a frozen object would not do.")

    if args.json_out:
        with open(args.json_out, "w") as handle:
            json.dump(results, handle, indent=2)
        print(f"\n  wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
