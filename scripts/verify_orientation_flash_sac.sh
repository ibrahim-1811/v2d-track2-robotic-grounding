#!/usr/bin/env bash
# Verify the orientation-aware FlashSAC implementation end to end.
#
#   bash scripts/verify_orientation_flash_sac.sh          # everything (~4 min)
#   bash scripts/verify_orientation_flash_sac.sh --fast   # skip the GPU evaluator round trip
#
# exit 0 = the thirteenth reward term is implemented, tested, reaches the
# learner's replay, and its actors load in the UNCHANGED official evaluator.
set -uo pipefail
cd "$(dirname "$0")/.."
REPO=$PWD
FAST=0
[ "${1:-}" = "--fast" ] && FAST=1

PASS=0; FAIL=0
step() { printf '\n=== %s ===\n' "$1"; }
ok()   { echo "  PASS  $1"; PASS=$((PASS+1)); }
bad()  { echo "  FAIL  $1"; FAIL=$((FAIL+1)); }
check(){ if [ "$1" -eq 0 ]; then ok "$2"; else bad "$2"; fi; }

FC_ROOT=$REPO/official/video_to_data/robotic_grounding/flash_chord
FC_ASSETS=$REPO/official/video_to_data/robotic_grounding/source/robotic_grounding/robotic_grounding/assets

step "1. NVIDIA upstream is unmodified"
git -C "$REPO/official/video_to_data" status --porcelain > /tmp/v2d_upstream_status.txt 2>&1
if [ -s /tmp/v2d_upstream_status.txt ]; then
  bad "official/video_to_data has local changes:"; cat /tmp/v2d_upstream_status.txt
else
  ok "official/video_to_data is clean"
fi

step "2. Generated configs match what the generator produces"
# Regenerating must be a no-op: that is what proves the twelve original weights,
# the nine thresholds and every shaping scale are still upstream's.
cp configs/orientation/curriculum/orientation_recon_body_flash_sac.yaml /tmp/v2d_curr_before.yaml
cp configs/orientation/objective/orientation_recon_body.yaml /tmp/v2d_obj_before.yaml
.venvs/episode0/bin/python scripts/make_orientation_curriculum.py > /tmp/v2d_gen.log 2>&1
check $? "generator ran (it refuses if upstream's weights or thresholds moved)"
diff -q /tmp/v2d_curr_before.yaml configs/orientation/curriculum/orientation_recon_body_flash_sac.yaml >/dev/null
check $? "curriculum config is reproducible"
diff -q /tmp/v2d_obj_before.yaml configs/orientation/objective/orientation_recon_body.yaml >/dev/null
check $? "objective config is reproducible"
test "$(grep -c '^      [a-z_0-9]*:' configs/orientation/curriculum/orientation_recon_body_flash_sac.yaml)" = 13
check $? "curriculum stage carries exactly 13 weights"

step "3. Host tests: reward math and portable export"
.venvs/episode0/bin/python -m pytest \
  pipeline/tests/test_orientation_reward.py \
  pipeline/tests/test_flash_sac_actor_export.py -q 2>&1 | tail -3
check ${PIPESTATUS[0]} "host test suite"

step "4. Container tests: Warp kernel and FlashSAC replay"
docker run --rm --gpus all \
  -v "$FC_ROOT:/workspace" \
  -v "$REPO/pipeline:/v2d/repo/pipeline:ro" \
  -v "$REPO/outputs/step_D/warp_cache:/root/.cache/warp" \
  -e PYTHONPATH=/v2d/repo -e XLA_PYTHON_CLIENT_PREALLOCATE=false \
  -w /v2d/repo flash_chord:latest \
  python -m pytest \
    pipeline/tests/container_flash_sac_orientation_kernel.py \
    pipeline/tests/container_flash_sac_orientation_replay.py \
    -q -p no:cacheprovider 2>&1 | tail -3
check ${PIPESTATUS[0]} "container test suite"

step "5. Launch plan preflight"
# A throwaway tag: the real one's output directory exists once a run has
# completed, and preflight correctly refuses to reuse it.
python3 scripts/train_orientation_flash_sac.py --dry-run \
  --tag preflight_check_unused > /tmp/v2d_plan.txt 2>&1
check $? "350M launch plan passes preflight"
grep -q "optimizer updates    : 683,592" /tmp/v2d_plan.txt
check $? "LR horizon is the full-recipe 683,592 optimizer updates"
grep -q "environment steps    : 349,999,104" /tmp/v2d_plan.txt
check $? "budget is NVIDIA's terminal 349,999,104"

step "6. Disk monitor refuses protected directories"
python3 scripts/monitor_training_disk.py outputs/step_D/source_ep0_w1024 >/dev/null 2>&1
[ $? -ne 0 ]; check $? "refuses to prune the original 10M source run"
python3 scripts/monitor_training_disk.py outputs/step_D/submission_trial_20261021 >/dev/null 2>&1
[ $? -ne 0 ]; check $? "refuses to prune the submitted trial"

if [ $FAST -eq 1 ]; then
  echo; echo "--fast: skipped the evaluator round trip"
else
  step "7. Portable actor loads in the UNCHANGED official evaluator"
  # This needs a real thirteen-term checkpoint. Rather than depend on a
  # leftover directory, make one: two collections is enough, because a short
  # run always writes a terminal checkpoint. Costs about a minute, nearly all
  # of it Warp/JAX startup.
  SMOKE=outputs/step_D/verify_orient_train/policy_2048.safetensors
  if [ ! -f "$SMOKE" ]; then
    echo "  (no checkpoint yet; training 2,048 steps to produce one, ~1 min)"
    rm -rf outputs/step_D/verify_orient_train
    python3 scripts/train_orientation_flash_sac.py \
      --tag verify_orient_train --total-steps 2048 --save-interval 2048 \
      --replay-minimum 1024 > outputs/step_D/verify_orientation_train.log 2>&1
    check $? "produced a thirteen-term checkpoint to project"
  fi
  if [ ! -f "$SMOKE" ]; then
    bad "no checkpoint at $SMOKE; see outputs/step_D/verify_orientation_train.log"
  else
    rm -f /tmp/v2d_portable.safetensors
    .venvs/episode0/bin/python -c "
import sys; sys.path.insert(0, '.')
from pathlib import Path
from pipeline.flash_sac_actor_export import export_official_actor
export_official_actor(Path('$SMOKE'), Path('/tmp/v2d_portable.safetensors'), provenance={'check': 'verify_script'})
" >/dev/null 2>&1
    check $? "export produced a portable actor"
    mkdir -p outputs/step_D/verify_orientation
    cp /tmp/v2d_portable.safetensors outputs/step_D/verify_orientation/policy_portable.safetensors
    # The pipeline package is deliberately NOT mounted here.
    docker run --rm --gpus all \
      -v "$FC_ROOT:/workspace" -v "$FC_ASSETS:/v2d/assets:ro" \
      -v "$REPO/data/adapter_out:/work/adapter_out:ro" \
      -v "$REPO/outputs/step_D:/out" \
      -v "$REPO/outputs/step_D/warp_cache:/root/.cache/warp" \
      -e XLA_PYTHON_CLIENT_PREALLOCATE=false \
      -w /workspace flash_chord:latest \
      python scripts/evaluate_policy.py \
        evaluation.checkpoint=/out/verify_orientation/policy_portable.safetensors \
        evaluation.world_count=1 \
        evaluation.metrics_output=/out/verify_orientation/metrics.json \
        > outputs/step_D/verify_orientation/eval.log 2>&1
    check $? "official evaluator ran it with no local module on PYTHONPATH"

    # And the control: the native thirteen-term checkpoint must NOT load.
    docker run --rm --gpus all \
      -v "$FC_ROOT:/workspace" -v "$FC_ASSETS:/v2d/assets:ro" \
      -v "$REPO/data/adapter_out:/work/adapter_out:ro" \
      -v "$REPO/outputs/step_D:/out" \
      -v "$REPO/outputs/step_D/warp_cache:/root/.cache/warp" \
      -e XLA_PYTHON_CLIENT_PREALLOCATE=false \
      -w /workspace flash_chord:latest \
      python scripts/evaluate_policy.py \
        evaluation.checkpoint=/out/verify_orient_train/policy_2048.safetensors \
        evaluation.world_count=1 \
        evaluation.metrics_output=/out/verify_orientation/native_metrics.json \
        > outputs/step_D/verify_orientation/native_eval.log 2>&1
    [ $? -ne 0 ]; check $? "control: the NATIVE 13-term checkpoint is correctly rejected"
    docker run --rm -v "$REPO/outputs/step_D:/out" flash_chord:latest \
      chown -R 1000:1000 /out >/dev/null 2>&1
  fi
fi

printf '\n=== verify_orientation_flash_sac: %d passed, %d failed ===\n' "$PASS" "$FAIL"
[ "$FAIL" -eq 0 ] || exit 1
