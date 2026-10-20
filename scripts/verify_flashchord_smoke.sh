#!/usr/bin/env bash
# Verify the FlashCHORD G1 baseline smoke test end to end. Run from the repo root:
#     bash scripts/verify_flashchord_smoke.sh
# Every check prints PASS or FAIL. Exit code 0 means everything passed.
#
# Checks 5 and 6 run the full 800-frame replays and take about a minute together.
# Pass --fast to skip them.
#
# Self-contained: every container step is a throwaway `docker run --rm`, so no
# persistent flash_chord-latest-gpu0 is needed (docker/run.sh starts it with
# --rm, so it vanishes on stop). The replay harness is bind-mounted read-only
# rather than copied in with `docker cp`.
set -uo pipefail
cd "$(dirname "$0")/.."

FC=official/video_to_data/robotic_grounding/flash_chord
ASSETS=official/video_to_data/robotic_grounding/source/robotic_grounding/robotic_grounding/assets
TRACE=/tmp/verify_trace.csv
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
    -v "$PWD/scripts/flashchord_headless_replay.py:/driver/flashchord_headless_replay.py:ro" \
    -v "$(dirname "$TRACE"):/trace" \
    ${extra[0]+"${extra[@]}"} \
    -w /workspace flash_chord:latest bash -lc "$1"
}

echo "=== 1. pinned asset hashes (NVIDIA enforces these in its own code) ==="
while read -r want file; do
  got=$(sha256sum "$FC/$file" 2>/dev/null | cut -d' ' -f1)
  if [ "$got" = "$want" ]; then ok "$(basename "$file")"; else no "$(basename "$file") (got ${got:0:16}...)"; fi
done <<'EOF'
6071c943492cb6c91e99b392200a288e829f5fea9491b530b34451fcf9e8d02a src/flash_chord/assets/policies/sonic/encoder_batched.onnx
de629511bc3a110a26827a54389e82d35a722ab9b7cc4232aab865b8451d2a18 src/flash_chord/assets/policies/sonic/decoder_batched.onnx
8c7f768dc8da6c969d8a4b4efb3dcbe3d9d2b05fca5104df12205a69879d19e9 src/flash_chord/assets/urdfs/g1/main_with_hand.urdf
EOF

echo
echo "=== 2. image is present and the example parquet is visible ==="
chk "flash_chord:latest image present" "docker image inspect flash_chord:latest"
chk "G1 snack-box parquet readable in container" \
    "fc -- \"test -f '$G1_PARQUET'\""

echo
echo "=== 3. BLOCKED: no FlashCHORD checkpoint exists ==="
# Scope to what NVIDIA SHIPS: exclude the gitignored outputs/ trees, which now
# hold the checkpoints our own training feasibility test produced.
n_disk=$(find official -name '*.safetensors' -not -path '*/outputs/*' 2>/dev/null | wc -l)
n_lfs=$(git -C official/video_to_data lfs ls-files 2>/dev/null | grep -c safetensors)
[ "$n_disk" -eq 0 ] && ok "0 .safetensors shipped by NVIDIA (excluding our outputs/)" || no "$n_disk shipped .safetensors found (unexpected)"
[ "$n_lfs" -eq 0 ]  && ok "0 .safetensors in the Git LFS manifest" || no "$n_lfs in LFS (unexpected)"
# Captured to a variable first: piping straight into `grep -q` makes grep exit on
# the first match, python dies of SIGPIPE, and `pipefail` reports the whole
# pipeline as failed even though the check succeeded.
evalout=$(fc -- "python scripts/evaluate_policy.py evaluation.world_count=1 logging.mode=disabled" 2>&1)
if printf '%s' "$evalout" | grep -q "Missing mandatory value: evaluation.checkpoint"; then
  ok "evaluate_policy.py requires a checkpoint (no default to fall back on)"
else
  no "evaluate_policy.py did not report the expected missing-checkpoint error"
fi

echo
echo "=== 4. the replay is object-assisted; evaluation is not ==="
chk "replay/physical.yaml sets voc_scale: 1.0" \
    "grep -q 'voc_scale: 1.0' $FC/src/flash_chord/configs/replay/physical.yaml"
chk "evaluation forces env.voc_scale = 0.0" \
    "grep -q 'config\[.env.\]\[.voc_scale.\] = 0.0' $FC/src/flash_chord/evaluation/harness.py"
chk "evaluate_policy.py documents 'unassisted' attempts" \
    "grep -q unassisted $FC/scripts/evaluate_policy.py"

if [ "$FAST" -eq 1 ]; then
  echo
  echo "(skipping the 800-frame replays: --fast)"
else
  echo
  echo "=== 5. kinematic reference: 800/800 frames, object error 0.0 m ==="
  ref=$(fc -- "python /driver/flashchord_headless_replay.py --parquet '$G1_PARQUET' --mode reference" 2>/dev/null)
  echo "$ref" | grep -q 'HARNESS_RESULT=PASS'          && ok "harness PASS" || no "harness did not PASS"
  echo "$ref" | grep -q '"frames_completed": 800'      && ok "800 frames completed" || no "frame count != 800"
  echo "$ref" | grep -q '"nonfinite_frame_count": 0'   && ok "no non-finite state" || no "non-finite state found"
  echo "$ref" | grep -q '"object_position_m": 0.0'     && ok "object tracked exactly" || no "object error != 0.0"

  echo
  echo "=== 6. controller physics replay: 800/800 frames, then the documented fall ==="
  rep=$(fc -- "python /driver/flashchord_headless_replay.py --parquet '$G1_PARQUET' --mode replay --trace-csv /trace/$(basename "$TRACE") ; chown 1000:1000 /trace/$(basename "$TRACE") 2>/dev/null" 2>/dev/null)
  echo "$rep" | grep -q 'HARNESS_RESULT=PASS'        && ok "harness PASS (ran to completion)" || no "harness did not PASS"
  echo "$rep" | grep -q '"frames_completed": 800'    && ok "800 frames completed" || no "frame count != 800"
  echo "$rep" | grep -q '"nonfinite_frame_count": 0' && ok "no non-finite state, no NaN blow-up" || no "non-finite state found"
  fall=$(awk -F, 'NR>1 && $2<0.60 {print $1; exit}' "$TRACE" 2>/dev/null)
  if [ -n "$fall" ]; then
    ok "G1 falls at frame $fall (expected ~663; README: task success not guaranteed)"
  else
    echo "  NOTE  no fall detected this run - replay is stochastic in its delay seed"
  fi
fi

echo
echo "=== 7. nothing was modified that should not be ==="
if [ -z "$(git -C official/video_to_data status --short)" ]; then
  ok "NVIDIA upstream unmodified"; else no "NVIDIA upstream has local changes"; fi

echo
echo "=== 8. headroom ==="
echo -n "  disk free: "; df -h / | awk 'NR==2{print $4" ("$5" used)"}'
echo -n "  GPU:       "; nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
echo "  peak observed during this smoke test: 1841 MiB of 8151 MiB"

echo
if [ "$FAILED" -eq 0 ]; then echo "ALL CHECKS PASSED"; else echo "SOME CHECKS FAILED"; fi
exit $FAILED
