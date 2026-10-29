#!/usr/bin/env bash
# Train, then evaluate, one episode on a Colab runtime prepared by colab/setup_colab.sh. It runs
# the cluster's own cluster/train_tier_episode.sbatch and cluster/evaluate_policy.sbatch in
# V2D_NATIVE=1 mode, so recipe, run name and layout are the cluster's:
# outputs/cluster/<prefix>_epNN_<arm>[_s<seed>]/ with eval_<step>/rep*.json + rep*/episode_*.parquet,
# which scripts/evaluate/assemble_submission.py picks up once copied back to the cluster.
#
#   PREFIX=t2c MANIFEST=training_manifest_tier2c.json ARM=keypoints SEED=43 EP=4 bash colab/run_colab.sh
#   SMOKE=1 PREFIX=smoke MANIFEST=training_manifest_tier2c.json ARM=keypoints EP=4 bash colab/run_colab.sh
set -euo pipefail
export V2D_ROOT=/content/v2d V2D_NATIVE=1 V2D_WARP_CACHE=/content/warp_cache
export PATH=/content/fcvenv/bin:$PATH
R=$V2D_ROOT
export PREFIX=${PREFIX:?} MANIFEST=${MANIFEST:?} ARM=${ARM:?} EP=${EP:?} SEED=${SEED:-42}
STEP=349999104
case "$ARM" in *slowvoc) STEP=507500544 ;; esac
REPEATS=${REPEATS:-5}
if [ "${SMOKE:-0}" = 1 ]; then
  STEP=2048000 REPEATS=1
  export EXTRA="training.total_environment_steps=$STEP"
fi
SUFFIX=""
[ "$SEED" != 42 ] && SUFFIX="_s$SEED"
TAG=$(printf '%s_ep%02d_%s%s' "$PREFIX" "$EP" "$ARM" "$SUFFIX")

# train_flash_sac.sbatch's NaN watchdog reads logs/train_<job name>_<job id>.out.
export SLURM_JOB_NAME=colab-$TAG SLURM_JOB_ID=$(date +%Y%m%d%H%M%S)_$$ SLURM_CPUS_PER_TASK=$(nproc)
LOG=$R/logs/train_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out
echo "$(date +%H:%M) training $TAG -> $LOG"
bash "$R/cluster/train_tier_episode.sbatch" >"$LOG" 2>&1 || { echo "training failed:"; tail -20 "$LOG"; exit 1; }

ELOG=$R/logs/eval_${TAG}_${SLURM_JOB_ID}.out
echo "$(date +%H:%M) evaluating $TAG step $STEP x $REPEATS -> $ELOG"
RUN=$TAG STEP=$STEP EPISODE=$EP REPEATS=$REPEATS bash "$R/cluster/evaluate_policy.sbatch" >"$ELOG" 2>&1 || true
tail -8 "$ELOG"
