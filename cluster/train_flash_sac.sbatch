#!/bin/bash
#SBATCH --job-name=v2d-train
#SBATCH --partition=gpu4
#SBATCH --exclude=wr25   # cap: at most 5 GPU nodes, wr20-24 (user, 2026-10-29)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G   # peak RSS is 7.6-8.2 GB; 128G let only 3 jobs share a 4-GPU, 510 GB node
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
#
# CONFIGS=<dir under configs/> mounts our own Hydra groups (objective/, curriculum/) at
# /v2d/configs, for the thirteen-term objectives; select them through EXTRA:
#   CONFIGS=keypoints EXTRA="objective=keypoints_recon_body curriculum=keypoints_recon_body_flash_sac" ...
#
# A NaN watchdog stops the job at the first non-finite metrics line: a diverged FlashSAC
# run never recovers (Episode 7's seed-42 run trained 6.5 h on NaN). Resume from the last
# state_*.safetensors before the NaN with training.resume=true training.checkpoint_path=...
#
# V2D_NATIVE=1 (Colab, colab/run_colab.sh) runs the same command without Apptainer, in a machine
# whose Python is flash_chord's own uv environment; cluster/native_paths.sh recreates the
# container's paths as symlinks. V2D_ROOT points at the checkout there.
set -euo pipefail

R=${V2D_ROOT:-/work/mnidgu2s/project/v2d-track2-robotic-grounding}
SIF=$R/containers/flash_chord.sif
FC=$R/official/video_to_data/robotic_grounding/flash_chord
ASSETS=$R/official/video_to_data/robotic_grounding/source/robotic_grounding/robotic_grounding/assets
NATIVE=${V2D_NATIVE:-0}
if [ "$NATIVE" = 1 ]; then
  : "${SLURM_JOB_NAME:=v2d-native}" "${SLURM_JOB_ID:=$$}"
fi

TAG=${TAG:-ep0_stock_nvidia_recipe}
SEQ=${SEQ:-2026-06-04_14-16-23_tall_bar_stool_tip_up_04}
SUPPORT=${SUPPORT:-false}
SEED=${SEED:-42}
EXTRA=${EXTRA:-}
CONFIGS=${CONFIGS:-}
# Staged tree under data/, mounted at the same name under /work: every tier bakes its own
# absolute object-mesh paths (/work/adapter_out, /work/adapter_out_tier_2, ...).
ADAPTER=${ADAPTER:-adapter_out}

CONFIG_ARGS=()
if [ -n "$CONFIGS" ]; then
  test -d "$R/configs/$CONFIGS"
  CONFIG_ARGS=(-B "$R/configs/$CONFIGS":/v2d/configs:ro)
fi
CONFIG_DIR_ARGS=()
[ -n "$CONFIGS" ] && CONFIG_DIR_ARGS=(--config-dir /v2d/configs)
LOG="$R/logs/train_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out"
# Every staged tree, read-only, at its own /work/<name>: recovered references (adapter_out_tier_2r)
# keep the object meshes of the tree they were derived from.
DATA_BINDS=()
for tree in "$R"/data/adapter_out*; do
  [ -d "$tree" ] && DATA_BINDS+=(-B "$tree:/work/$(basename "$tree"):ro")
done

OUT=$R/outputs/cluster
mkdir -p "$OUT/$TAG" "$OUT/warp_cache" "$OUT/tmp/$SLURM_JOB_ID"
PARQUET="/work/${ADAPTER}/whole_body/soma/sequence_id=${SEQ}/robot_name=g1/data.parquet"
test -f "$R/data/${ADAPTER}/whole_body/soma/sequence_id=${SEQ}/robot_name=g1/data.parquet"

echo "job $SLURM_JOB_ID on $(hostname), tag=$TAG seq=$SEQ support=$SUPPORT seed=$SEED adapter=$ADAPTER configs='$CONFIGS' extra='$EXTRA'"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
# A Colab bundle is a git archive: no .git, its commit is in BUNDLE_COMMIT.
git -C "$R" rev-parse --short HEAD 2>/dev/null || cat "$R/BUNDLE_COMMIT"
git -C "$R/official/video_to_data" rev-parse --short HEAD 2>/dev/null || echo "official: bundled flash_chord"

TRAIN_CMD=(python -m pipeline.flash_sac_train_entry
  "${CONFIG_DIR_ARGS[@]}"
  experiment=g1_recon_body_flash_sac
  "task.parquet='${PARQUET}'"
  scene.world_count=2048
  scene.support="$SUPPORT"
  training.total_environment_steps=349999104
  training.updates_per_collection=4.0
  training.replay.capacity=10000000
  training.checkpoint.save_replay=false
  training.seed="$SEED"
  reset.seed="$SEED"
  logging.mode=disabled
  output_dir="/out/$TAG"
  hydra.run.dir="/out/$TAG/hydra")

# $EXTRA stays unquoted: it is a space-separated list of Hydra overrides.
if [ "$NATIVE" = 1 ]; then
  source "$R/cluster/native_paths.sh"
  [ -n "$CONFIGS" ] && native_link "$R/configs/$CONFIGS" /v2d/configs
  (cd /workspace && PYTHONPATH=/v2d/repo:/workspace/src XLA_PYTHON_CLIENT_PREALLOCATE=false \
     WARP_CACHE_PATH="${V2D_WARP_CACHE:-/out/warp_cache}" exec "${TRAIN_CMD[@]}" $EXTRA) &
else
  # Site bind paths are switched off so /work inside the container is only what we mount:
  # the parquets bake absolute /work/adapter_out/... paths (CLAUDE.md, Step B).
  apptainer exec --nv --cleanenv \
    --no-mount bind-paths,home,cwd \
    -B "$FC":/workspace \
    -B "$ASSETS":/v2d/assets:ro \
    "${DATA_BINDS[@]}" \
    -B "$R/pipeline":/v2d/repo/pipeline:ro \
    "${CONFIG_ARGS[@]}" \
    -B "$OUT":/out \
    -B "$OUT/tmp/$SLURM_JOB_ID":/tmp \
    --env HOME=/tmp \
    --env PYTHONPATH=/v2d/repo:/workspace/src \
    --env XLA_PYTHON_CLIENT_PREALLOCATE=false \
    --env WARP_CACHE_PATH=/out/warp_cache \
    --pwd /workspace \
    "$SIF" "${TRAIN_CMD[@]}" $EXTRA &
fi
TRAIN=$!

# AUTO_RESUME=N lets the watchdog resubmit this run up to N times from the newest
# state_*.safetensors saved before the NaN, with seed+1 (Episode 7 diverges every ~75M steps).
AUTO_RESUME=${AUTO_RESUME:-0}
while kill -0 "$TRAIN" 2>/dev/null; do
  if grep -q 'rew=+*nan' "$LOG" 2>/dev/null; then
    nan_step=$(grep -m1 -oP 'step=\K[0-9,]+(?=\s+rew=\+?nan)' "$LOG" | tr -d ,)
    echo "NaN watchdog: metrics non-finite since step $nan_step; stopping"
    kill "$TRAIN"
    wait "$TRAIN" || true
    state=$(ls "$OUT/$TAG"/state_*.safetensors 2>/dev/null | sed -E 's/.*state_([0-9]+)\.safetensors/\1/' \
            | awk -v n="$nan_step" '$1 < n' | sort -n | tail -1)
    if [ "$NATIVE" = 1 ] && [ -n "$state" ]; then
      echo "NaN watchdog: resume with SEED=$((SEED + 1)) EXTRA='training.resume=true training.checkpoint_path=/out/$TAG/state_${state}.safetensors'"
    elif [ "$AUTO_RESUME" -gt 0 ] && [ -n "$state" ]; then
      base=$(sed -E 's/training\.(resume|checkpoint_path)=[^ ]*//g' <<<"$EXTRA")
      jid=$(TAG="$TAG" SEQ="$SEQ" SUPPORT="$SUPPORT" ADAPTER="$ADAPTER" CONFIGS="$CONFIGS" \
            SEED=$((SEED + 1)) AUTO_RESUME=$((AUTO_RESUME - 1)) \
            EXTRA="$base training.resume=true training.checkpoint_path=/out/$TAG/state_${state}.safetensors" \
            sbatch --parsable --job-name="$SLURM_JOB_NAME" --time=12:00:00 "$R/cluster/train_flash_sac.sbatch")
      echo "NaN watchdog: resubmitted from state_${state} with seed $((SEED + 1)) as job $jid ($((AUTO_RESUME - 1)) resumes left)"
    fi
    exit 3
  fi
  sleep 60
done
wait "$TRAIN"
