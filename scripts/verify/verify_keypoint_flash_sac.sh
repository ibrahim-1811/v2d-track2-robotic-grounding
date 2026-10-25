#!/usr/bin/env bash
# Verify the ADD-shaped object_keypoints objective (arm 3) on the cluster.
#
#   bash scripts/verify/verify_keypoint_flash_sac.sh     # login node, CPU only, ~1 min
#
# exit 0 = upstream is untouched, the generated configs are reproducible, the reward
# math and the portable export pass their host tests, the Warp kernel matches the NumPy
# reference, and -- once the smoke run has finished -- a real GPU run trained the
# thirteenth term and its actor exports for the unchanged official evaluator.
set -uo pipefail
cd "$(dirname "$0")/../.."
R=$PWD
SIF=$R/containers/flash_chord.sif
FC=$R/official/video_to_data/robotic_grounding/flash_chord
SCRATCH=$(mktemp -d)
trap 'rm -rf "$SCRATCH"' EXIT

PASS=0; FAIL=0
step() { printf '\n=== %s ===\n' "$1"; }
ok()   { echo "  PASS  $1"; PASS=$((PASS+1)); }
bad()  { echo "  FAIL  $1"; FAIL=$((FAIL+1)); }
check(){ if [ "$1" -eq 0 ]; then ok "$2"; else bad "$2"; fi; }
sif()  { apptainer exec --cleanenv --no-mount home,cwd -B "$R":/r -B "$FC":/workspace \
           -B "$SCRATCH":/wc --env WARP_CACHE_PATH=/wc --env PYTHONPATH=/r:/workspace/src --pwd /r \
           "$SIF" "$@" 2>&1 | grep -v pthread_setaffinity; return "${PIPESTATUS[0]}"; }

step "1. NVIDIA upstream is unmodified"
status=$(git -C "$R/official/video_to_data" status --porcelain)
[ -z "$status" ]; check $? "official/video_to_data has no local changes"

step "2. Generated configs are reproducible"
cp -r configs/keypoints "$SCRATCH/before"
python3 scripts/train/make_orientation_curriculum.py --arm keypoints > "$SCRATCH/gen.log" 2>&1
check $? "generator ran (it refuses if upstream's weights or thresholds moved)"
diff -rq "$SCRATCH/before" configs/keypoints >/dev/null; check $? "configs/keypoints regenerates byte-identical"
test "$(grep -c '^      [a-z_0-9]*:' configs/keypoints/curriculum/keypoints_recon_body_flash_sac.yaml)" = 13
check $? "curriculum stage carries exactly 13 weights"
grep -q '^      object_keypoints: 5.0$' configs/keypoints/curriculum/keypoints_recon_body_flash_sac.yaml
check $? "object_keypoints weight is 5.0"

step "3. Host tests: reward math and portable export"
sif python -m pytest -q -p no:cacheprovider pipeline/tests/test_keypoint_reward.py \
  pipeline/tests/test_orientation_reward.py pipeline/tests/test_flash_sac_actor_export.py | tail -2
check "${PIPESTATUS[0]}" "host test suite"

step "4. Warp kernel matches the NumPy reference (Warp CPU)"
sif python -m pytest -q -p no:cacheprovider pipeline/tests/container_flash_sac_keypoints_kernel.py | tail -2
check "${PIPESTATUS[0]}" "kernel test suite"

step "5. GPU smoke run (outputs/cluster/smoke_kp_ep03)"
SMOKE=$R/outputs/cluster/smoke_kp_ep03
policy=$(ls "$SMOKE"/policy_*.safetensors 2>/dev/null | grep -v portable | sort -V | tail -1)
if [ -z "$policy" ]; then
  echo "  SKIP  smoke run has not finished yet (job v2d-smoke-kp)"
else
  head -1 "$SMOKE/metrics.csv" | grep -q 'reward_components/object_keypoints_weighted_mean_per_env_step'
  check $? "the learner logged object_keypoints as a reward component"
  python3 -m pipeline.flash_sac_actor_export "$policy" "$SCRATCH/portable.safetensors" > /dev/null
  check $? "the smoke actor projects to an upstream-only portable checkpoint"
fi

printf '\n%d passed, %d failed\n' "$PASS" "$FAIL"
[ "$FAIL" -eq 0 ]
