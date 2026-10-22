#!/usr/bin/env python3
"""Summarise or plot a FlashSAC run's metrics.jsonl.

``pipeline/flash_sac_metrics_logger.py`` writes one JSON record per log
boundary. This reads them back on the host -- no container, no W&B.

    # table of the key series, head and tail
    .venvs/episode0/bin/python scripts/show_training_metrics.py outputs/step_D/ep0_orient_350M
    # one named series in full
    .venvs/episode0/bin/python scripts/show_training_metrics.py <dir> --series sac/critic_loss_mean_per_update
    # every reward term's contribution, latest record
    .venvs/episode0/bin/python scripts/show_training_metrics.py <dir> --terms
    # PNG curves
    .venvs/episode0/bin/python scripts/show_training_metrics.py <dir> --plot metrics.png
    # follow a live run
    .venvs/episode0/bin/python scripts/show_training_metrics.py <dir> --watch 60
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

TABLE = (
    ("environment_steps", "steps", "{:>13,.0f}"),
    ("curriculum/stage_index", "stg", "{:>3.0f}"),
    ("curriculum/voc_scale", "voc", "{:>5.2f}"),
    ("reward/total_mean_per_env_step", "reward", "{:>+8.4f}"),
    ("episode/return_mean_per_completed_episode", "return", "{:>+8.2f}"),
    ("episode/length_mean_per_completed_episode", "eplen", "{:>6.0f}"),
    ("sac/actor_loss_mean_per_update", "actor", "{:>+8.3f}"),
    ("sac/critic_loss_mean_per_update", "critic", "{:>8.3f}"),
    ("policy/entropy_mean_per_actor_update", "entropy", "{:>+8.2f}"),
    ("tracking_error/object_position_m_mean_env_max_per_step", "objerr_m", "{:>8.4f}"),
    ("tracking_error/object_orientation_rad_mean_env_max_per_step", "objerr_rad", "{:>10.4f}"),
    ("replay/fill_fraction", "replay", "{:>6.2f}"),
    ("throughput/env_steps_per_second", "steps/s", "{:>8,.0f}"),
)

PLOT_PANELS = (
    ("losses", ("sac/actor_loss_mean_per_update", "sac/critic_loss_mean_per_update")),
    ("reward per env step", ("reward/total_mean_per_env_step", "reward/dense_mean_per_env_step")),
    ("episode return", ("episode/return_mean_per_completed_episode",)),
    ("episode length", ("episode/length_mean_per_completed_episode",)),
    (
        "object tracking error",
        ("tracking_error/object_position_m_mean_env_max_per_step", "tracking_error/object_orientation_rad_mean_env_max_per_step"),
    ),
    (
        "orientation vs position reward",
        (
            "reward_components/object_position_weighted_mean_per_env_step",
            "reward_components/object_orientation_weighted_mean_per_env_step",
        ),
    ),
    ("curriculum", ("curriculum/voc_scale", "curriculum/stage_index")),
    ("replay fill", ("replay/fill_fraction",)),
)


def load(run: Path) -> list[dict]:
    path = run / "metrics.jsonl" if run.is_dir() else run
    if not path.exists():
        raise SystemExit(f"no metrics at {path}; the run may predate the local metric logger")
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # a torn final line from a run still in progress
    if not rows:
        raise SystemExit(f"{path} has no complete records yet")
    return rows


def print_table(rows: list[dict], limit: int) -> None:
    header = "  ".join(label for _, label, _ in TABLE)
    print(header)
    print("-" * len(header))
    shown = rows if len(rows) <= 2 * limit else rows[:limit] + [None] + rows[-limit:]
    for row in shown:
        if row is None:
            print("   ...")
            continue
        cells = []
        for key, label, spec in TABLE:
            value = row.get(key)
            cells.append(" " * len(label) if value is None else spec.format(value).rjust(len(label)))
        print("  ".join(cells))


def print_terms(rows: list[dict]) -> None:
    last = rows[-1]
    print(f"reward term contributions at step {last['environment_steps']:,} (weighted, per env step)")
    terms = {
        key.replace("reward_components/", "").replace("_weighted_mean_per_env_step", ""): value
        for key, value in last.items()
        if key.startswith("reward_components/")
    }
    width = max((len(name) for name in terms), default=0)
    for name, value in sorted(terms.items(), key=lambda item: -abs(item[1])):
        print(f"  {name:<{width}}  {value:+.6f}")
    print(f"  {'TOTAL (dense+term)':<{width}}  {last.get('reward/total_mean_per_env_step', float('nan')):+.6f}")


def plot(rows: list[dict], destination: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [row["environment_steps"] for row in rows]
    panels = [(title, keys) for title, keys in PLOT_PANELS if any(key in rows[-1] for key in keys)]
    columns = 2
    figure, axes = plt.subplots((len(panels) + 1) // columns, columns, figsize=(13, 2.6 * len(panels) / columns + 2))
    for axis, (title, keys) in zip(axes.flat, panels, strict=False):
        for key in keys:
            if key not in rows[-1]:
                continue
            axis.plot(steps, [row.get(key) for row in rows], linewidth=1.2, label=key.rsplit("/", 1)[-1])
        axis.set_title(title, fontsize=9)
        axis.legend(fontsize=6)
        axis.grid(alpha=0.3)
        axis.tick_params(labelsize=7)
    for axis in list(axes.flat)[len(panels) :]:
        axis.axis("off")
    figure.suptitle(f"FlashSAC: {len(rows)} log boundaries to {steps[-1]:,} environment steps", fontsize=10)
    figure.tight_layout()
    figure.savefig(destination, dpi=130)
    print(f"wrote {destination}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path, help="run directory, or a metrics.jsonl path")
    parser.add_argument("--limit", type=int, default=12, help="rows of head and tail to show")
    parser.add_argument("--series", help="print one metric in full")
    parser.add_argument("--terms", action="store_true", help="per-reward-term contributions, latest record")
    parser.add_argument("--plot", type=Path, help="write a PNG of the key curves")
    parser.add_argument("--watch", type=int, default=0, help="redraw every N seconds")
    args = parser.parse_args()

    while True:
        rows = load(args.run)
        if args.watch:
            print("\033[2J\033[H", end="")
        print(f"{len(rows)} records, to {rows[-1]['environment_steps']:,} env steps, "
              f"{rows[-1].get('wall_seconds', 0) / 60:.1f} min wall\n")
        if args.series:
            if args.series not in rows[-1]:
                raise SystemExit(f"no such metric: {args.series}\navailable:\n  " + "\n  ".join(sorted(rows[-1])))
            for row in rows:
                print(f"{row['environment_steps']:>13,}  {row[args.series]:+.6f}")
        elif args.terms:
            print_terms(rows)
        else:
            print_table(rows, args.limit)
        if args.plot:
            plot(rows, args.plot)
        if not args.watch:
            return 0
        time.sleep(args.watch)


if __name__ == "__main__":
    sys.exit(main())
