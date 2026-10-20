#!/usr/bin/env bash
# Verify Step B (Episode 0 SOMA -> G1 retargeting) end to end. Run from the repo root:
#     bash scripts/verify_step_B.sh            # re-retargets all 778 frames (~3 min)
#     bash scripts/verify_step_B.sh --fast     # validates the existing parquet only (~15 s)
# Every check prints PASS or FAIL. Exit code 0 means everything passed.
set -uo pipefail
cd "$(dirname "$0")/.."

SEQ=2026-06-04_14-16-23_tall_bar_stool_tip_up_04
FRAMES=778
IN_DIR="data/adapter_out/whole_body/soma/$SEQ"
OUT_DIR="data/adapter_out/whole_body/soma/sequence_id=$SEQ"
PARQUET="$OUT_DIR/robot_name=g1/data.parquet"
DEMO="official/video_to_data/robotic_grounding/source/robotic_grounding/robotic_grounding/assets/human_motion_data/whole_body/soma/sequence_id=2026-03-06_10-24-18_snack_box_pick_and_place_01/robot_name=g1/data.parquet"
FC="$PWD/official/video_to_data/robotic_grounding/flash_chord"
ASSETS="$PWD/official/video_to_data/robotic_grounding/source/robotic_grounding/robotic_grounding/assets"
FAST=0
[ "${1:-}" = "--fast" ] && FAST=1

FAILED=0
ok(){ echo "  PASS  $1"; }
no(){ echo "  FAIL  $1"; FAILED=1; }
chk(){ if eval "$2" >/dev/null 2>&1; then ok "$1"; else no "$1"; fi; }

echo "=== 1. Step A inputs still present ==="
for f in soma_params.npz poses.npy ground_plane.json reconstructed_mesh/output_aligned.glb; do
  chk "$f" "[ -f '$IN_DIR/$f' ]"
done
chk "SOMA-X bundle (SOMA_neutral.npz)" "[ -f data/body_models/soma/SOMA_neutral.npz ]"
chk "support stage USDA present" \
    "[ -f 'data/adapter_out/whole_body/reconstructed_stage/${SEQ}_support.usda' ]"
chk "robotic-grounding image available" \
    "docker image inspect robotic-grounding:latest"

echo
echo "=== 2. the support USDA is an empty stub, as NVIDIA's own corpus also contains ==="
shapes=$(grep -cE '^[[:space:]]*def (Cylinder|Cube) ' \
         "data/adapter_out/whole_body/reconstructed_stage/${SEQ}_support.usda" 2>/dev/null)
nv_empty=$(cd "$ASSETS/human_motion_data/whole_body/reconstructed_stage" 2>/dev/null && \
           for f in *_support.usda; do
             [ "$(grep -cE '^[[:space:]]*def (Cylinder|Cube) ' "$f")" = "0" ] && echo "$f";
           done | wc -l)
[ "${shapes:-0}" = "0" ] && ok "ours declares 0 collision shapes -> needs scene.support=false" \
                         || no "ours declares $shapes shapes (expected 0)"
[ "${nv_empty:-0}" -ge 1 ] && ok "NVIDIA ships $nv_empty empty support stub(s) too (same convention)" \
                           || no "no empty NVIDIA stub found; re-read docs/EPISODE_0_G1_RETARGET.md section 6.1"

if [ "$FAST" -eq 0 ]; then
  echo
  echo "=== 3. retarget all $FRAMES frames with the memory-bounded driver ==="
  log=$(mktemp)
  docker run --rm --gpus all \
    -v "$PWD/data/adapter_out:/work/adapter_out" \
    -v "$PWD/data/body_models/soma:/soma:ro" \
    -v "$PWD/scripts/retarget_episode_g1.py:/driver/retarget_episode_g1.py:ro" \
    --entrypoint bash robotic-grounding:latest -lc "
python /driver/retarget_episode_g1.py --chunk-frames 64 -- \
  /work/adapter_out/whole_body/soma/$SEQ \
  --save --soma-data-root /soma --robot-name g1 \
  --motion-root /work/adapter_out --soma-subdir soma --fps 30 ;
rc=\$? ; chown -R 1000:1000 /work/adapter_out ; exit \$rc
" > "$log" 2>&1
  rc=$?
  [ "$rc" -eq 0 ] && ok "retarget exited 0" || no "retarget exited $rc"
  grep -q "Processed $FRAMES frames" "$log" \
    && ok "processed all $FRAMES frames" || no "did not report $FRAMES frames"
  grep -q "grounding plane: reconstructed" "$log" \
    && ok "used the reconstructed ground plane (not the fallback)" \
    || no "did not use the reconstructed ground plane"
  grep -q "0/$FRAMES frames" "$log" \
    && ok "zero anti-penetration lift needed on any frame" \
    || echo "  NOTE  anti-penetration lift was applied on some frames (see the log)"
  grep -qiE "out of memory|Failed to allocate" "$log" \
    && no "an out-of-memory error occurred" || ok "no CUDA OOM"
  rm -f "$log"
else
  echo
  echo "(skipping the retarget run: --fast)"
fi

echo
echo "=== 4. the parquet exists and matches NVIDIA's schema + residuals ==="
chk "data.parquet written" "[ -f '$PARQUET' ]"
chk "object mesh written"  "[ -f '$OUT_DIR/object/textured_mesh.obj' ]"
chk "object urdf written"  "[ -f '$OUT_DIR/object/textured_mesh.urdf' ]"
if [ -f "$PARQUET" ] && [ -f "$DEMO" ]; then
  if .venvs/episode0/bin/python scripts/validate_g1_reference.py \
       --parquet "$PARQUET" --reference "$DEMO" --frames "$FRAMES" \
       --mesh-root "$OUT_DIR/object" > /tmp/step_b_validate.log 2>&1; then
    ok "validate_g1_reference.py: ALL CHECKS PASSED"
  else
    no "validate_g1_reference.py reported failures (see /tmp/step_b_validate.log)"
  fi
  grep -E "median IK residual ratio|robot pelvis height|nearest-EE" /tmp/step_b_validate.log \
    | sed 's/^  \.\.    /  info: /'
else
  no "cannot validate: parquet or NVIDIA demo reference missing"
fi

if [ "$FAST" -eq 0 ]; then
  echo
  echo "=== 5. FlashCHORD loads it and builds a physics scene ==="
  if docker image inspect flash_chord:latest >/dev/null 2>&1; then
    fclog=$(mktemp)
    docker run --rm --gpus all \
      -v "$FC:/workspace" -v "$ASSETS:/v2d/assets:ro" \
      -v "$PWD/data/adapter_out:/work/adapter_out:ro" \
      -v "$PWD/scripts/check_reference_in_flashchord.py:/driver/check_reference_in_flashchord.py:ro" \
      -e XLA_PYTHON_CLIENT_PREALLOCATE=false \
      -w /workspace flash_chord:latest \
      python /driver/check_reference_in_flashchord.py \
        --parquet "/work/adapter_out/whole_body/soma/sequence_id=$SEQ/robot_name=g1/data.parquet" \
        --expect-frames "$FRAMES" > "$fclog" 2>&1
    grep -q "FLASHCHORD_LOAD=PASS" "$fclog" \
      && ok "FLASHCHORD_LOAD=PASS (scene built, exact frame-0 snap)" \
      || no "FlashCHORD load failed (see $fclog)"
    grep -E "model bodies|include_support" "$fclog" | sed 's/^  /  info: /'
    grep -q "FLASHCHORD_LOAD=PASS" "$fclog" && rm -f "$fclog"
  else
    echo "  NOTE  flash_chord:latest not present; skipping the load check"
  fi
fi

echo
echo "=== 6. upstream untouched ==="
if [ -z "$(git -C official/video_to_data status --short)" ]; then
  ok "NVIDIA upstream unmodified"; else no "NVIDIA upstream has local changes"; fi
chk "upstream still pinned at 193382a" \
    "git -C official/video_to_data log --oneline -1 | grep -q 193382a"

echo
echo "=== 7. headroom ==="
echo -n "  disk free: "; df -h / | awk 'NR==2{print $4" ("$5" used)"}'
echo -n "  GPU:       "; nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
echo "  retarget peak measured this episode: 3,411 MiB VRAM / 7,025 MiB host RAM"

echo
if [ "$FAILED" -eq 0 ]; then echo "ALL CHECKS PASSED"; else echo "SOME CHECKS FAILED"; fi
exit $FAILED
