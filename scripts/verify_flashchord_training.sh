#!/usr/bin/env bash
# Verify the FlashCHORD G1 PPO training feasibility test. Run from the repo root:
#     bash scripts/verify_flashchord_training.sh          # re-trains + re-evaluates (~2 min)
#     bash scripts/verify_flashchord_training.sh --fast    # checks existing artifacts (~30 s)
# Every check prints PASS or FAIL. Exit code 0 means everything passed.
#
# Self-contained: every container step is a throwaway `docker run --rm`, so no
# persistent flash_chord-latest-gpu0 is needed (docker/run.sh starts it with
# --rm, so it vanishes on stop). Inline Python is avoided - helper scripts are
# bind-mounted read-only, per this repo's container rules.
set -uo pipefail
cd "$(dirname "$0")/.."

FC=official/video_to_data/robotic_grounding/flash_chord
ASSETS=official/video_to_data/robotic_grounding/source/robotic_grounding/robotic_grounding/assets
CKPT_DIR=outputs/g1_ppo_smoke
G1_PARQUET=/v2d/assets/human_motion_data/whole_body/soma/sequence_id=2026-03-06_10-24-18_snack_box_pick_and_place_01/robot_name=g1/data.parquet
FAST=0
[ "${1:-}" = "--fast" ] && FAST=1

FAILED=0
ok(){ echo "  PASS  $1"; }
no(){ echo "  FAIL  $1"; FAILED=1; }
chk(){ if eval "$2" >/dev/null 2>&1; then ok "$1"; else no "$1"; fi; }

# fc <extra docker args...> -- <shell command>
fc() {
  local extra=()
  while [ $# -gt 0 ] && [ "$1" != "--" ]; do extra+=("$1"); shift; done
  shift || true
  docker run --rm --gpus all \
    -v "$PWD/$FC:/workspace" \
    -v "$PWD/$ASSETS:/v2d/assets:ro" \
    -v "$PWD/scripts/inspect_checkpoint.py:/driver/inspect_checkpoint.py:ro" \
    -v "$PWD/scripts/check_unassisted.py:/driver/check_unassisted.py:ro" \
    ${extra[0]+"${extra[@]}"} \
    -w /workspace flash_chord:latest bash -lc "$1"
}

echo "=== 1. image and recipe present ==="
chk "flash_chord:latest image present" "docker image inspect flash_chord:latest"
chk "g1_recon_body_ppo recipe exists" "[ -f $FC/src/flash_chord/configs/experiment/g1_recon_body_ppo.yaml ]"
chk "PPO entrypoint scripts/train_rl.py exists" "[ -f $FC/scripts/train_rl.py ]"
chk "recipe targets 4096 worlds / 20000 iterations" \
    "grep -q 'world_count: 4096' $FC/src/flash_chord/configs/experiment/g1_recon_body_ppo.yaml && grep -q 'max_iterations: 20000' $FC/src/flash_chord/configs/experiment/g1_recon_body_ppo.yaml"

echo
echo "=== 2. train_rl.py is the one entrypoint missing the JAX prealloc guard ==="
for s in train_flash_sac evaluate_policy view_policy; do
  chk "scripts/$s.py sets XLA_PYTHON_CLIENT_PREALLOCATE" \
      "grep -q XLA_PYTHON_CLIENT_PREALLOCATE $FC/scripts/$s.py"
done
if grep -q XLA_PYTHON_CLIENT_PREALLOCATE "$FC/scripts/train_rl.py" 2>/dev/null; then
  no "train_rl.py now sets it too - upstream changed; re-read docs/FLASHCHORD_TRAINING_FEASIBILITY.md section 3"
else
  ok "scripts/train_rl.py does NOT set it (pass -e XLA_PYTHON_CLIENT_PREALLOCATE=false)"
fi

if [ "$FAST" -eq 0 ]; then
  echo
  echo "=== 3. 2-iteration PPO training smoke test ==="
  tr=$(fc -e XLA_PYTHON_CLIENT_PREALLOCATE=false -- "python scripts/train_rl.py experiment=g1_recon_body_ppo \"task.parquet='$G1_PARQUET'\" scene.world_count=32 training.max_iterations=2 training.save_interval=1 logging.mode=disabled reset.seed=42 output_dir=$CKPT_DIR" 2>&1)
  printf '%s' "$tr" | grep -q "completed 2 iterations" && ok "completed 2 iterations" || no "training did not complete"
  printf '%s' "$tr" | grep -q "1,536/1,536" && ok "1,536 environment steps" || no "unexpected step count"
  printf '%s' "$tr" | grep -qiE "out of memory|RESOURCE_EXHAUSTED" && no "OOM during training" || ok "no CUDA OOM"
fi

echo
echo "=== 4. checkpoint is an authentic FlashCHORD PPO checkpoint ==="
chk "model_2.safetensors exists" "[ -f $FC/$CKPT_DIR/model_2.safetensors ]"
if [ -f "$FC/$CKPT_DIR/model_2.safetensors" ]; then
  meta=$(fc -- "python /driver/inspect_checkpoint.py $CKPT_DIR/model_2.safetensors" 2>&1)
  printf '%s' "$meta" | grep -q "^ALGO ppo"   && ok "NVIDIA's reader reports algorithm = ppo" || no "not recognised as a PPO checkpoint"
  printf '%s' "$meta" | grep -q "^ITER 2"     && ok "metadata iteration = 2" || no "iteration metadata wrong"
  printf '%s' "$meta" | grep -q "^STEPS 1536" && ok "metadata environment_steps = 1536" || no "step metadata wrong"
  printf '%s' "$meta" | grep -q "g1_dex3.G1Dex3" && ok "embodiment = G1Dex3" || no "wrong embodiment"
  printf '%s' "$meta" | grep -E "^(WORLD_COUNT|PARQUET)" | sed 's/^/  info: /'
else
  no "cannot inspect: checkpoint missing (run without --fast to produce it)"
fi

echo
echo "=== 5. evaluation strips object assistance ==="
if [ -f "$FC/$CKPT_DIR/model_2.safetensors" ]; then
  un=$(fc -- "python /driver/check_unassisted.py $CKPT_DIR/model_2.safetensors" 2>&1)
  printf '%s' "$un" | grep -q "^TRAINED 1.0"      && ok "trained WITH assistance (voc_scale 1.0, curriculum stage 0)" || no "unexpected trained voc_scale"
  printf '%s' "$un" | grep -q "^EVAL 0.0 0.0 0.0" && ok "evaluated UNASSISTED (voc_scale and both limits forced to 0)" || no "assistance was not removed"
  printf '%s' "$un" | grep -q "^UNASSISTED YES"   && ok "UNASSISTED=YES" || no "evaluator did not report UNASSISTED"
else
  no "cannot check assistance: checkpoint missing"
fi

if [ "$FAST" -eq 0 ]; then
  echo
  echo "=== 6. evaluator runs on one world ==="
  ev=$(fc -e XLA_PYTHON_CLIENT_PREALLOCATE=false -- "python scripts/evaluate_policy.py \"evaluation.checkpoint=$CKPT_DIR/model_2.safetensors\" evaluation.world_count=1 \"evaluation.metrics_output=$CKPT_DIR/model_2.safetensors.metrics.json\" logging.mode=disabled" 2>&1)
  printf '%s' "$ev" | grep -q "evaluation step 800/800" && ok "evaluator completed 800/800 steps" || no "evaluator did not finish"
  printf '%s' "$ev" | grep -q "mppe_cm" && ok "MPPE reported" || no "no MPPE in output"
fi
chk "metrics JSON written" "[ -f $FC/$CKPT_DIR/model_2.safetensors.metrics.json ]"

echo
echo "=== 7. this checkpoint is NOT a competitive baseline ==="
# The report nests metrics under "metrics", without the "evaluation/" prefix the
# stdout scalars carry.
if [ -f "$FC/$CKPT_DIR/model_2.safetensors.metrics.json" ]; then
  if python3 - "$FC/$CKPT_DIR/model_2.safetensors.metrics.json" <<'PY'
import json, sys
report = json.load(open(sys.argv[1]))
m = report["metrics"]
mppe = m["mppe_cm"]
srs = {k: v for k, v in m.items() if k.endswith("_sr") or k.endswith("_sr_uncentered")}
print("  MPPE = {:.2f} cm   (a trained policy targets single-digit cm)".format(mppe))
print("  success rates: " + ", ".join("{}={}".format(k, v) for k, v in sorted(srs.items())))
print("  world_count = {}, environment_steps = {}".format(
    report["world_count"], report["environment_steps"]))
sys.exit(0 if mppe > 10 and all(v == 0.0 for v in srs.values()) else 1)
PY
  then
    ok "metrics are those of an untrained policy - a mechanism check, not a Track 2 result"
  else
    no "metrics no longer look like an untrained policy - re-read docs/FLASHCHORD_TRAINING_FEASIBILITY.md"
  fi
else
  no "no metrics JSON to inspect"
fi

echo
echo "=== 8. upstream untouched; outputs are gitignored by NVIDIA ==="
if [ -z "$(git -C official/video_to_data status --short)" ]; then
  ok "NVIDIA upstream unmodified"; else no "NVIDIA upstream has local changes"; fi
chk "flash_chord/outputs/ is in NVIDIA's own .gitignore" \
    "git -C official/video_to_data check-ignore -q robotic_grounding/flash_chord/outputs/x"

echo
echo "=== 9. headroom ==="
echo -n "  disk free: "; df -h / | awk 'NR==2{print $4" ("$5" used)"}'
echo -n "  GPU:       "; nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
echo "  measured ceiling: 1024 worlds comfortable (5,437 MiB); 2048 OOMs"

echo
if [ "$FAILED" -eq 0 ]; then echo "ALL CHECKS PASSED"; else echo "SOME CHECKS FAILED"; fi
exit $FAILED
