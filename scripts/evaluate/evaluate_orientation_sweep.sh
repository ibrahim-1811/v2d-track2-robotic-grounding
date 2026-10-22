#!/usr/bin/env bash
# Repeated, unassisted, world-1 evaluation of one or more FlashSAC checkpoints.
#
# Single-rollout evaluation on this machine is NOT repeatable: five runs of one
# fixed checkpoint gave AUC 0.1127-0.1290 and MPPE 47.6-189.6 cm. One number is
# not a result. This runs N fresh processes per checkpoint, keeps every
# trajectory and JSON, and reports mean/median/IQR/std/min/max.
#
# world_count=1 because the kit's track_2_tier1 sets worlds_per_episode: 1 and
# pack_submission.py scores world 0 only. A 64-world number disagrees with the
# packer and must never be submitted.
#
#   bash scripts/evaluate/evaluate_orientation_sweep.sh <run_dir> <repeats> <step> [step...]
#
# e.g. bash scripts/evaluate/evaluate_orientation_sweep.sh outputs/step_D/ep0_orient_350M 5 349999104 230008832
set -uo pipefail
cd "$(dirname "$0")/../.."
REPO=$PWD

RUN_DIR=${1:?usage: evaluate_orientation_sweep.sh <run_dir> <repeats> <step> [step...]}
REPEATS=${2:?missing repeat count}
shift 2
STEPS=("$@")
[ ${#STEPS[@]} -gt 0 ] || { echo "give at least one checkpoint step"; exit 2; }

# --- guard: never contend with a training job for 8 GB of VRAM -------------
RUNNING=$(docker ps --format '{{.Names}}' | grep '^v2d-train-' || true)
if [ -n "$RUNNING" ]; then
  echo "REFUSING: a training job is running ($RUNNING)."
  echo "  Evaluation needs ~1.8 GiB and training holds ~7.2 GiB of 8.1 GiB."
  echo "  Running both would OOM, most likely killing the training job."
  echo "  Wait for it to finish, or stop it deliberately first."
  exit 1
fi
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
if [ "$FREE" -lt 2200 ]; then
  echo "REFUSING: only ${FREE} MiB of GPU memory free; evaluation needs ~1800 MiB plus headroom."
  exit 1
fi

TAG_BASE=$(basename "$RUN_DIR")
OUT=outputs/step_D/sweep_${TAG_BASE}
mkdir -p "$OUT"
echo "=== evaluation sweep: ${#STEPS[@]} checkpoint(s) x $REPEATS repeats, world_count=1 ==="
echo "    GPU free: ${FREE} MiB      results -> $OUT"

for STEP in "${STEPS[@]}"; do
  NATIVE="$RUN_DIR/policy_${STEP}.safetensors"
  if [ ! -f "$NATIVE" ]; then
    echo "  SKIP step $STEP: no $NATIVE"; continue
  fi
  PORTABLE="$OUT/portable_${STEP}.safetensors"
  if [ ! -f "$PORTABLE" ]; then
    .venvs/episode0/bin/python -c "
import sys; sys.path.insert(0, '.')
from pathlib import Path
from pipeline.flash_sac_actor_export import export_official_actor
export_official_actor(Path('$NATIVE'), Path('$PORTABLE'),
                      provenance={'arm': 'orientation', 'episode': '0', 'step': '$STEP'})
" || { echo "  FAIL export at step $STEP"; continue; }
  fi
  echo
  echo "--- step $STEP ---"
  for R in $(seq 1 "$REPEATS"); do
    TAG="${TAG_BASE}_${STEP}_r${R}"
    printf '  repeat %d/%d ... ' "$R" "$REPEATS"
    # A fresh container per repeat: execution variance is the thing being measured.
    bash scripts/evaluate/evaluate_step_d.sh \
      "/out/sweep_${TAG_BASE}/portable_${STEP}.safetensors" "$TAG" 1 0 \
      > "$OUT/${TAG}.console" 2>&1 \
      && echo "ok" || echo "FAILED (see $OUT/${TAG}.console)"
  done
done

echo
.venvs/episode0/bin/python scripts/evaluate/summarize_orientation_sweep.py "$OUT" --run-dir "$RUN_DIR"
