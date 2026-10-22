#!/usr/bin/env python3
"""Watch disk during a long FlashSAC run and prune only that run's learner states.

A 350M-step Episode 0 run writes 34 checkpoint pairs: 34 x 4.6 MiB actors
(cheap, always kept) and 34 x 152 MiB learner states (5.2 GiB). That fits
today, but the run is six hours long and a full disk loses the whole job.

Safety, because this deletes files:

* It only ever touches ``state_*.safetensors`` inside the ONE run directory
  named on the command line. Actors, replays, manifests, logs and every other
  directory are never considered.
* :data:`PROTECTED` directories are refused outright, so the submitted trial
  and the original 10M source run cannot be passed in even by mistake.
* The newest ``--keep`` states are always retained, and so is the single
  largest step number, so a resume is always possible.
* It is a **dry run unless ``--apply`` is given**.

    # report only
    python3 scripts/train/monitor_training_disk.py outputs/step_D/ep0_orient_350M
    # prune, keeping the newest 3 learner states
    python3 scripts/train/monitor_training_disk.py outputs/step_D/ep0_orient_350M --apply --keep 3
    # follow a running job
    python3 scripts/train/monitor_training_disk.py outputs/step_D/ep0_orient_350M --apply --watch 300
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

#: Never prune inside these, whatever is asked. Completed work and submissions.
PROTECTED = (
    REPO / "outputs/step_D/source_ep0_w1024",
    REPO / "outputs/step_D/submission_trial_20261021",
    REPO / "outputs/step_D/orientation_plan_20261021",
    REPO / "outputs/step_D/cluster_handoff_20261021",
    REPO / "outputs/step_D/transfer",
    REPO / "outputs/step_D/trajectories",
    REPO / "outputs/step_D/eval",
    REPO / "data",
    REPO / "official",
)

STATE_PATTERN = re.compile(r"^state_(\d+)\.safetensors$")


def gib(value: float) -> float:
    return value / 1024**3


def resolve_run(path: Path) -> Path:
    run = path.resolve()
    if not run.is_dir():
        raise SystemExit(f"{run} is not a directory")
    for protected in PROTECTED:
        protected = protected.resolve()
        if run == protected or protected in run.parents or run in protected.parents:
            raise SystemExit(
                f"refusing to prune {run}: it is inside, or contains, the protected path {protected}"
            )
    if REPO / "outputs" not in run.parents:
        raise SystemExit(f"refusing to prune {run}: only directories under outputs/ may be pruned")
    return run


def report(run: Path) -> tuple[list[tuple[int, Path]], int, int]:
    states: list[tuple[int, Path]] = []
    policy_bytes = 0
    for entry in run.iterdir():
        if not entry.is_file():
            continue
        match = STATE_PATTERN.match(entry.name)
        if match:
            states.append((int(match.group(1)), entry))
        elif entry.name.startswith("policy_"):
            policy_bytes += entry.stat().st_size
    states.sort()
    state_bytes = sum(path.stat().st_size for _, path in states)
    return states, state_bytes, policy_bytes


def prune(run: Path, keep: int, apply: bool) -> int:
    states, state_bytes, policy_bytes = report(run)
    usage = shutil.disk_usage(REPO)
    print(f"[{time.strftime('%H:%M:%S')}] {run.relative_to(REPO)}")
    print(f"  disk free        : {gib(usage.free):6.2f} GiB of {gib(usage.total):.0f} GiB")
    print(f"  actors kept      : {gib(policy_bytes):6.3f} GiB (never pruned)")
    print(f"  learner states   : {len(states)} files, {gib(state_bytes):.2f} GiB")
    if not states:
        return 0

    # Always keep the newest `keep`, which includes the largest step number.
    doomed = states[:-keep] if keep > 0 else []
    if not doomed:
        print(f"  nothing to prune : newest {min(keep, len(states))} states retained")
        return 0

    freed = sum(path.stat().st_size for _, path in doomed)
    kept = [step for step, _ in states[-keep:]]
    print(f"  would free       : {gib(freed):6.2f} GiB from {len(doomed)} states")
    print(f"  retaining steps  : {kept}")
    for step, path in doomed:
        if apply:
            path.unlink()
            print(f"    deleted {path.name} (step {step:,})")
        else:
            print(f"    DRY RUN would delete {path.name} (step {step:,})")
    return freed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path, help="the ONE run directory to prune, e.g. outputs/step_D/ep0_orient_350M")
    parser.add_argument("--keep", type=int, default=3, help="newest learner states to retain (default 3)")
    parser.add_argument("--apply", action="store_true", help="actually delete; otherwise dry run")
    parser.add_argument("--watch", type=int, default=0, help="repeat every N seconds (0 = once)")
    parser.add_argument("--min-free-gib", type=float, default=5.0, help="warn below this much free space")
    args = parser.parse_args()

    if args.keep < 1:
        raise SystemExit("--keep must be at least 1 so a resume stays possible")
    run = resolve_run(args.run)

    while True:
        prune(run, args.keep, args.apply)
        free = gib(shutil.disk_usage(REPO).free)
        if free < args.min_free_gib:
            print(f"  WARNING: only {free:.2f} GiB free, below the {args.min_free_gib} GiB floor")
        if args.watch <= 0:
            return 0
        time.sleep(args.watch)


if __name__ == "__main__":
    sys.exit(main())
