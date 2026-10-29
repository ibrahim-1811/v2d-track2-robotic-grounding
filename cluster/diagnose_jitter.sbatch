#!/bin/bash
#SBATCH --job-name=v2d-jitter
#SBATCH --partition=gpu
# V100 nodes wr14-19 cannot run the SONIC controller (see evaluate_policy.sbatch).
#SBATCH --exclude=wr[14-19,25]   # cap: at most 5 GPU nodes, wr20-24 (user, 2026-10-29)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/work/mnidgu2s/project/v2d-track2-robotic-grounding/logs/diag_%x_%j.out

# How much does each policy shake the robot? scripts/diagnose/diagnose_jitter_rollout.py on the
# same unassisted world-1 rollout the evaluator uses: step-to-step joint vibration of the
# reference, of SONIC's decoded output, of the PD target and of the simulated joints.
#
#   RUNS="t1_ep02_keypoints t1_ep02_keypoints_smooth" sbatch cluster/diagnose_jitter.sbatch
#   ZERO=1 RUNS="t1_ep02_keypoints" sbatch cluster/diagnose_jitter.sbatch    # SONIC alone
#
# Output: outputs/diag/jitter/<run>_{policy,zero}.{npz,json}
set -uo pipefail
R=/work/mnidgu2s/project/v2d-track2-robotic-grounding
SIF=$R/containers/flash_chord.sif
FC=$R/official/video_to_data/robotic_grounding/flash_chord
ASSETS=$R/official/video_to_data/robotic_grounding/source/robotic_grounding/robotic_grounding/assets
OUT=$R/outputs/cluster
mkdir -p "$R/outputs/diag/jitter" "$OUT/warp_cache" "$OUT/tmp/$SLURM_JOB_ID"
DATA_BINDS=()
for tree in "$R"/data/adapter_out*; do
  [ -d "$tree" ] && DATA_BINDS+=(-B "$tree:/work/$(basename "$tree"):ro")
done
MODE=policy; ZERO_ARG=()
[ "${ZERO:-0}" = 1 ] && { MODE=zero; ZERO_ARG=(--zero-action); }
failed=0
for RUN in ${RUNS:?set RUNS}; do
  step=$(ls "$OUT/$RUN" | sed -n 's/^policy_\([0-9]*\)\.safetensors$/\1/p' | sort -n | tail -1)
  [ -n "$step" ] || { echo "== $RUN: no policy"; failed=1; continue; }
  ckpt=policy_$step.safetensors
  if [ ! -f "$OUT/$RUN/portable_$ckpt" ]; then
    (cd "$R" && python3 -m pipeline.flash_sac_actor_export "$OUT/$RUN/$ckpt" "$OUT/$RUN/portable_$ckpt") || true
  fi
  [ -f "$OUT/$RUN/portable_$ckpt" ] && ckpt=portable_$ckpt
  echo "== $RUN $MODE ($ckpt)"
  apptainer exec --nv --cleanenv --no-mount bind-paths,home,cwd \
    -B "$FC":/workspace -B "$ASSETS":/v2d/assets:ro "${DATA_BINDS[@]}" -B "$OUT":/out \
    -B "$R/outputs/diag":/diag -B "$R/scripts":/v2d/scripts:ro -B "$OUT/tmp/$SLURM_JOB_ID":/tmp \
    --env PYTHONPATH=/workspace/src --env XLA_PYTHON_CLIENT_PREALLOCATE=false --env WARP_CACHE_PATH=/out/warp_cache --pwd /workspace "$SIF" \
    python /v2d/scripts/diagnose/diagnose_jitter_rollout.py --checkpoint "/out/$RUN/$ckpt" \
      --output "/diag/jitter/${RUN}_${MODE}.npz" "${ZERO_ARG[@]}" 2>&1 | grep -v -i "warn" | tail -4 || failed=1
done
exit $failed
