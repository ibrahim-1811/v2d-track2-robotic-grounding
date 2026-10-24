#!/bin/bash
#SBATCH --job-name=v2d-train
#SBATCH --partition=gpu4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/work/mnidgu2s/project/v2d-track2-robotic-grounding/logs/train_%x_%j.out

# FlashSAC training in the flash_chord Apptainer image -- the cluster equivalent of the
# `docker run` in docs/guides/ORIENTATION_RUN_COMMANDS.md (appendix), same mounts.
#
# Defaults = Task 2 of docs/guides/ONBOARDING.md: Episode 0, NVIDIA's exact shipped
# g1_recon_body_flash_sac recipe (2048 worlds, 10M replay, 4.0 updates/collection,
# 349,999,104 steps, stock 12-term objective), scene.support=false, seed 42.
#
#   sbatch cluster/train_flash_sac.sbatch
#   TAG=ep21_stock SEQ=2026-04-29_11-53-57_tall_bar_stool_pull_08 SUPPORT=false \
#       sbatch cluster/train_flash_sac.sbatch
#   EXTRA="training.total_environment_steps=2048000" TAG=smoke sbatch ...   # short test
set -euo pipefail

R=/work/mnidgu2s/project/v2d-track2-robotic-grounding
SIF=$R/containers/flash_chord.sif
FC=$R/official/video_to_data/robotic_grounding/flash_chord
ASSETS=$R/official/video_to_data/robotic_grounding/source/robotic_grounding/robotic_grounding/assets

TAG=${TAG:-ep0_stock_nvidia_recipe}
SEQ=${SEQ:-2026-06-04_14-16-23_tall_bar_stool_tip_up_04}
SUPPORT=${SUPPORT:-false}
EXTRA=${EXTRA:-}

OUT=$R/outputs/cluster
mkdir -p "$OUT/$TAG" "$OUT/warp_cache" "$OUT/tmp/$SLURM_JOB_ID"
PARQUET="/work/adapter_out/whole_body/soma/sequence_id=${SEQ}/robot_name=g1/data.parquet"
test -f "$R/data/adapter_out/whole_body/soma/sequence_id=${SEQ}/robot_name=g1/data.parquet"

echo "job $SLURM_JOB_ID on $(hostname), tag=$TAG seq=$SEQ support=$SUPPORT extra='$EXTRA'"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
git -C "$R" rev-parse --short HEAD
git -C "$R/official/video_to_data" rev-parse --short HEAD

# Site bind paths are switched off so /work inside the container is only what we mount:
# the parquets bake absolute /work/adapter_out/... paths (CLAUDE.md, Step B).
apptainer exec --nv --cleanenv \
  --no-mount bind-paths,home,cwd \
  -B "$FC":/workspace \
  -B "$ASSETS":/v2d/assets:ro \
  -B "$R/data/adapter_out":/work/adapter_out:ro \
  -B "$R/pipeline":/v2d/repo/pipeline:ro \
  -B "$OUT":/out \
  -B "$OUT/tmp/$SLURM_JOB_ID":/tmp \
  --env HOME=/tmp \
  --env PYTHONPATH=/v2d/repo:/workspace/src \
  --env XLA_PYTHON_CLIENT_PREALLOCATE=false \
  --env WARP_CACHE_PATH=/out/warp_cache \
  --pwd /workspace \
  "$SIF" \
  python -m pipeline.flash_sac_train_entry \
    experiment=g1_recon_body_flash_sac \
    "task.parquet='${PARQUET}'" \
    scene.world_count=2048 \
    scene.support="$SUPPORT" \
    training.total_environment_steps=349999104 \
    training.updates_per_collection=4.0 \
    training.replay.capacity=10000000 \
    training.checkpoint.save_replay=false \
    training.seed=42 \
    reset.seed=42 \
    logging.mode=disabled \
    output_dir="/out/$TAG" \
    hydra.run.dir="/out/$TAG/hydra" \
    $EXTRA
