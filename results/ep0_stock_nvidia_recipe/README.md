# Episode 0 — NVIDIA's exact shipped recipe, on the cluster

Task 2 of `docs/guides/ONBOARDING.md`: the one run that tests whether the 8 GB
configuration was the problem.

## Run

- Episode 0, `2026-06-04_14-16-23_tall_bar_stool_tip_up_04`, Tier 1
- `experiment=g1_recon_body_flash_sac`, stock 12-term objective, nothing of ours
- `scene.world_count=2048`, `training.replay.capacity=10000000`,
  `training.updates_per_collection=4.0`, `training.total_environment_steps=349999104`,
  `scene.support=false`, seed 42
- 1x NVIDIA A100-SXM4-80GB (driver 610.57.04), Slurm partition `gpu4`
- Apptainer image built from upstream's Dockerfile: `cluster/flash_chord.def`
- Upstream `193382a`, this repo `252c3e8`
- Wall clock **3 h 22 min**, ~29,300 env steps/s during updates
- Launched with `cluster/train_flash_sac.sbatch` (defaults)

## Evaluation

5 fresh unassisted processes, `evaluation.world_count=1`, NVIDIA's unchanged
`evaluate_policy.py` (`cluster/evaluate_policy.sbatch`).

| repeat | AUC | MPPE cm |
|---|---|---|
| 1 | 0.4018 | 17.82 |
| 2 | 0.5131 | 3.36 |
| 3 | 0.3897 | 8.06 |
| 4 | 0.4943 | 3.59 |
| 5 | 0.5123 | 2.98 |
| **mean** | **0.4622 +/- 0.0613** | **7.16 +/- 6.31** |
| stationary floor | 0.3769 | 32.59 |
| 8 GB orientation run, 350M | 0.3491 +/- 0.0185 | 26.84 |

CHORD SR 0.6, SPIDER SR 0.8, ManipTrans SR 0.2 (means over 5).

Gate: **0.38–0.50 — helps but insufficient.** Two of five rollouts fall to
~0.39–0.40; the other three are ~0.49–0.51.

## Files

- `policy_349999104.safetensors` — final actor, stock 12-term, accepted by the
  official evaluator as-is
- `metrics.csv` — training metrics (`pipeline/flash_sac_metrics_logger.py`)
- `eval/repN.json` — evaluator reports; `eval/repN_episode_000000.parquet` —
  exported object trajectories for `tools/pack_submission.py`
- `eval/summary.txt` — the evaluation job's aggregate output

Intermediate checkpoints (every 10M steps) and the 160 MB learner states stay on
the cluster under `outputs/cluster/ep0_stock_nvidia_recipe/`.
