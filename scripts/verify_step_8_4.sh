#!/usr/bin/env bash
# Verify Step 8.4 end to end. Run from the repo root:
#     bash scripts/verify_step_8_4.sh
# Every check prints PASS or FAIL. Exit code 0 means everything passed.
set -uo pipefail
cd "$(dirname "$0")/.."
SEQ=2026-06-04_14-16-23_tall_bar_stool_tip_up_04
FAILED=0
ok(){ echo "  PASS  $1"; }
no(){ echo "  FAIL  $1"; FAILED=1; }
chk(){ if eval "$2" >/dev/null 2>&1; then ok "$1"; else no "$1"; fi; }

echo "=== 1. adapter outputs from Step 8.3 still valid ==="
.venvs/episode0/bin/python -m pytest pipeline/tests/ -q 2>&1 | tail -1
chk "soma_params.npz exists" "[ -f data/adapter_out/whole_body/soma/$SEQ/soma_params.npz ]"
chk "poses.npy exists"       "[ -f data/adapter_out/whole_body/soma/$SEQ/poses.npy ]"

echo
echo "=== 2. Episode 0 assets: hashes + staging ==="
.venvs/episode0/bin/python scripts/stage_episode_0_assets.py >/dev/null 2>&1 \
  && ok "stage_episode_0_assets.py (hash-verifies all 3 sources, re-stages)" \
  || no "stage_episode_0_assets.py"
echo -n "  GLB sha256: "
if sha256sum data/v2d_challenge/track_2/tier_1_multiview_caption/mesh/tall_bar_stool/tall_bar_stool.glb \
   | grep -q cc214961423d674d55239077043e670c0696bf9113e225e11c1c2623c7ed26ad; then
  echo "matches published LFS oid  PASS"; else echo "MISMATCH  FAIL"; FAILED=1; fi

echo
echo "=== 3. data_folder has everything soma_to_g1.py needs ==="
for f in soma_params.npz poses.npy ground_plane.json reconstructed_mesh/output_aligned.glb; do
  chk "data_folder/$f" "[ -f data/adapter_out/whole_body/soma/$SEQ/$f ]"
done
chk "support USDA at <motion-root>/whole_body/reconstructed_stage/" \
    "[ -f data/adapter_out/whole_body/reconstructed_stage/${SEQ}_support.usda ]"

echo
echo "=== 4. SOMA-X bundle: all 5 files + published hashes ==="
for f in SOMA_neutral.npz correctives_model.pt MHR/mhr_model_lod1.pt \
         MHR/base_body_lod1.obj MHR/SOMA_wrap_lod1.obj; do
  chk "data/body_models/soma/$f" "[ -f data/body_models/soma/$f ]"
done
echo "  --- sha256 vs published ---"
while read -r want file; do
  got=$(sha256sum "data/body_models/soma/$file" 2>/dev/null | cut -d' ' -f1)
  if [ "$got" = "$want" ]; then ok "$file"; else no "$file (got ${got:0:16}...)"; fi
done <<EOF
515f7d5bb74be4e370e9adf5e779760ec3581556374c0b33212a32d13ab3b53f SOMA_neutral.npz
067237a235f626eedc217bf25ab71a2000d29e9a4801c9b93bf300917977e12f correctives_model.pt
93f1b6d7fbcfb10ea7b2390ba7d544cede3f30ede413170fec89bb04557caaf0 MHR/mhr_model_lod1.pt
EOF

echo
echo "=== 5. upstream's own validator + our npz against the real read_soma ==="
docker run --rm -v "$PWD/data/body_models/soma:/staging/soma:ro" \
  -v "$PWD/data/adapter_out/whole_body/soma/$SEQ:/staging/episode0:ro" \
  -v "$PWD/scripts/validate_soma_in_container.py:/staging/validate.py:ro" \
  --entrypoint bash robotic-grounding:latest \
  -lc 'python /staging/validate.py /staging/soma /staging/episode0' 2>&1 \
  | grep -E "^  (PASS|FAIL)" || { echo "  FAIL  container validation did not run"; FAILED=1; }

echo
echo "=== 6. nothing was modified that should not be ==="
if [ -z "$(git -C official/video_to_data status --short)" ]; then
  ok "NVIDIA upstream unmodified"; else no "NVIDIA upstream has local changes"; fi
if git status --short | grep -q "^ M data/"; then no "source dataset modified"; else ok "source dataset intact"; fi

echo
echo "=== 7. disk usage ==="
du -sh data/v2d_challenge data/adapter_out data/body_models | sed 's/^/  /'

echo
if [ "$FAILED" -eq 0 ]; then echo "ALL CHECKS PASSED"; else echo "SOME CHECKS FAILED"; fi
exit $FAILED
