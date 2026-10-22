"""A local metric sink for FlashSAC, so a run is readable without W&B.

``run_training`` publishes a rich metric record at every ``log_interval``
collections, but only to an attached :class:`MetricLogger`. Upstream's only
implementation is ``WandbLogger``, and ``train_flash_sac.py`` builds it solely
when ``logging.mode != "disabled"`` -- so a local, offline run currently prints
environment steps per second and nothing else: no losses, no returns, no
per-reward-term breakdown.

This satisfies the same two-method protocol and writes:

* ``metrics.jsonl``  -- one JSON record per log boundary, every metric
* ``metrics.csv``    -- the same data, flat, for a spreadsheet or pandas
* stdout             -- a compact line per boundary, so ``tail -f`` on the
  training log shows losses as they happen

It is deliberately append-only and flushes every record: a six-hour run that
dies at hour five must still leave a complete history.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from pathlib import Path

#: Printed on the console line, in this order. Everything else still lands in
#: the JSONL and CSV; this is only what is worth watching live.
CONSOLE_KEYS: tuple[tuple[str, str, str], ...] = (
    ("reward/total_mean_per_env_step", "rew", "{:+.4f}"),
    ("episode/return_mean_per_completed_episode", "ret", "{:+.1f}"),
    ("episode/length_mean_per_completed_episode", "len", "{:.0f}"),
    ("sac/actor_loss_mean_per_update", "actor", "{:+.3f}"),
    ("sac/critic_loss_mean_per_update", "critic", "{:.3f}"),
    ("policy/entropy_mean_per_actor_update", "H", "{:+.2f}"),
    ("reward_components/object_position_weighted_mean_per_env_step", "obj_pos", "{:+.5f}"),
    ("reward_components/object_orientation_weighted_mean_per_env_step", "obj_rot", "{:+.5f}"),
    ("replay/fill_fraction", "replay", "{:.2f}"),
    ("curriculum/stage_index", "stage", "{:.0f}"),
    ("curriculum/voc_scale", "voc", "{:.2f}"),
    ("throughput/env_steps_per_second", "steps/s", "{:,.0f}"),
)


class JsonlMetricLogger:
    """Write every FlashSAC metric record to disk and a digest to stdout."""

    def __init__(self, output_dir: str | Path, *, console: bool = True) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.output_dir / "metrics.jsonl"
        self.csv_path = self.output_dir / "metrics.csv"
        self.console = console
        self._started = time.time()
        self._columns: list[str] | None = None
        self._jsonl = self.jsonl_path.open("a", buffering=1)
        self._csv = self.csv_path.open("a", buffering=1)

    def log(self, metrics: Mapping[str, float], step: int) -> None:
        record = {"environment_steps": int(step), "wall_seconds": round(time.time() - self._started, 3)}
        record.update({name: float(value) for name, value in sorted(metrics.items())})
        self._jsonl.write(json.dumps(record) + "\n")

        if self._columns is None:
            # Fix the column order from the first record; later records are
            # written against it so the CSV stays rectangular.
            self._columns = list(record)
            self._csv.write(",".join(self._columns) + "\n")
        self._csv.write(",".join(_format_csv(record.get(name)) for name in self._columns) + "\n")

        if self.console:
            parts = [f"metrics step={step:,}"]
            for key, label, spec in CONSOLE_KEYS:
                if key in metrics:
                    parts.append(f"{label}=" + spec.format(metrics[key]))
            print("  ".join(parts), flush=True)

    def close(self) -> None:
        for handle in (self._jsonl, self._csv):
            if not handle.closed:
                handle.flush()
                handle.close()


def _format_csv(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return repr(value)
    return str(value)
