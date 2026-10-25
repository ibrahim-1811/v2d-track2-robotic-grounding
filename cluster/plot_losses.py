"""Plot FlashSAC actor/critic loss against environment steps for several runs.

Reads each run's metrics.jsonl (written by pipeline/flash_sac_metrics_logger.py) and
draws a smoothed curve per run. Host Python (3.9) with matplotlib; no repo imports.

    python3 cluster/plot_losses.py            # writes outputs/cluster/plots/loss_comparison.png
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

R = Path(__file__).resolve().parents[1]
OUT = R / "outputs/cluster/plots/loss_comparison.png"

# (label, metrics file); colours follow the fixed categorical order, never cycled.
RUNS = [
    ("Ep 0, 8 GB laptop (1024 worlds, 13 terms)", R / "reference_policy/metrics.jsonl"),
    ("Ep 0, cluster A100 (NVIDIA recipe)", R / "outputs/cluster/ep0_stock_nvidia_recipe/metrics.jsonl"),
    ("Ep 7 retrain, seed 43 (running)", R / "outputs/cluster/t1_ep07_seed43/metrics.jsonl"),
    ("Ep 27 (running)", R / "outputs/cluster/t1_ep27_stock/metrics.jsonl"),
    ("Ep 29 (running)", R / "outputs/cluster/t1_ep29_stock/metrics.jsonl"),
]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
METRICS = [
    ("sac/actor_loss_mean_per_update", "Actor loss"),
    ("sac/critic_loss_mean_per_update", "Critic loss"),
]
SMOOTH_STEPS = 2_000_000  # rolling window, in environment steps
TEXT, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"


def load(path: Path) -> list[dict]:
    rows = []
    with path.open() as stream:
        for line in stream:
            rows.append(json.loads(line))
    return rows


def smooth(xs: list[float], ys: list[float]) -> tuple[list[float], list[float]]:
    """Trailing mean over SMOOTH_STEPS of environment steps; NaN rows are dropped."""
    pairs = [(x, y) for x, y in zip(xs, ys) if y is not None and not math.isnan(y)]
    out_x, out_y, total, start = [], [], 0.0, 0
    for i, (x, y) in enumerate(pairs):
        total += y
        while pairs[start][0] < x - SMOOTH_STEPS:
            total -= pairs[start][1]
            start += 1
        out_x.append(x / 1e6)
        out_y.append(total / (i - start + 1))
    return out_x, out_y


def main() -> None:
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED})
    fig, axes = plt.subplots(1, len(METRICS), figsize=(14, 5.2), facecolor=SURFACE)
    loaded = [(label, color, load(path)) for (label, path), color in zip(RUNS, COLORS) if path.exists()]
    for ax, (key, title) in zip(axes, METRICS):
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for label, color, rows in loaded:
            x, y = smooth([r["environment_steps"] for r in rows], [r.get(key) for r in rows])
            if not x:
                continue
            ax.plot(x, y, color=color, linewidth=2, label=label, solid_capstyle="round")
            ax.plot(x[-1], y[-1], "o", color=color, markersize=6, markeredgecolor=SURFACE, markeredgewidth=2)
        ax.set_title(title, loc="left", color=TEXT, fontsize=12, fontweight="semibold")
        ax.set_xlabel("environment steps (millions)")
        ax.set_xlim(0, 360)
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.08, -0.16), ncol=3, frameon=False,
                   labelcolor=TEXT, fontsize=9.5)
    fig.suptitle("FlashSAC training loss vs environment steps (trailing 2M-step mean)",
                 x=0.06, ha="left", color=TEXT, fontsize=13, fontweight="semibold")
    fig.text(0.06, 0.905, "Losses are not comparable across episodes or objectives: each run normalises "
             "its own rewards. Curve shape, stability and NaNs are what to compare.",
             color=MUTED, fontsize=9.5)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.83, bottom=0.27, wspace=0.18)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140, facecolor=SURFACE)
    for label, _, rows in loaded:
        last = rows[-1]
        print("{:44s} {:>12,} steps  actor {:+.3f}  critic {:.3f}".format(
            label, last["environment_steps"], last.get("sac/actor_loss_mean_per_update", float("nan")),
            last.get("sac/critic_loss_mean_per_update", float("nan"))))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
