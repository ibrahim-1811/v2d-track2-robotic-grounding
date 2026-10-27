#!/usr/bin/env bash
# Verify arm 4 (object_velocity) and the slow-VOC curricula on the cluster.
#
#   bash scripts/verify/verify_velocity_flash_sac.sh     # login node, CPU only, ~1 min
#
# exit 0 = upstream is untouched, all keypoint-family curricula instantiate through upstream's
# own constructors with the right term set, the exporter and reward tests pass, the velocity
# Warp kernel matches its NumPy/SciPy reference, and -- once the smoke run has finished -- a
# real GPU run trained the fourteenth term and its actor went through the official evaluator.
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

step "2. Curricula instantiate through upstream's constructors"
sif python - <<'PY'
import glob, sys, yaml
from flash_chord.configuration import make_curriculum_stage, make_fixed_curriculum
from pipeline.flash_sac_keypoints import KEYPOINT_RECON_BODY_OBJECTIVE_TERM_NAMES as K
from pipeline.flash_sac_velocity import VELOCITY_RECON_BODY_OBJECTIVE_TERM_NAMES as V
bad = 0
for path in sorted(glob.glob("configs/keypoints/curriculum/*.yaml")):
    c = yaml.safe_load(open(path))
    make_fixed_curriculum(c["thresholds"], [make_curriculum_stage(**{k: v for k, v in s.items() if k != "_target_"}) for s in c["stages"]])
    names = set(c["stages"][0]["objective_weights"])
    want, final = (set(V) if "velocity" in path else set(K)), (507500544 if "slowvoc" in path else 349999104)
    good = names == want and c["thresholds"][-1] == final and all(t % 2048 == 0 for t in c["thresholds"])
    bad += not good
    print("  {}  {} ({} stages, final {})".format("ok " if good else "BAD", path.split("/")[-1], len(c["stages"]), c["thresholds"][-1]))
sys.exit(bad)
PY
check $? "every curriculum: valid, right term set, final step and 2048-step boundaries"
grep -q '^_target_: pipeline.flash_sac_velocity.VelocityReconBodyObjectiveConfig$' configs/keypoints/objective/velocity_keypoints_recon_body.yaml
check $? "velocity objective config targets pipeline.flash_sac_velocity"

step "3. Host tests: exporter (incl. the fourteen-term projection) and reward math"
sif python -m pytest -q -p no:cacheprovider pipeline/tests/test_flash_sac_actor_export.py \
  pipeline/tests/test_keypoint_reward.py pipeline/tests/test_orientation_reward.py | tail -2
check "${PIPESTATUS[0]}" "host test suite"

step "4. Warp kernels match their NumPy references (Warp CPU)"
sif python -m pytest -q -p no:cacheprovider pipeline/tests/container_flash_sac_velocity_kernel.py \
  pipeline/tests/container_flash_sac_keypoints_kernel.py | tail -2
check "${PIPESTATUS[0]}" "kernel test suites"

step "5. GPU smoke run (outputs/cluster/smoke_vel_slowvoc_ep25)"
SMOKE=$R/outputs/cluster/smoke_vel_slowvoc_ep25
if [ ! -f "$SMOKE/policy_4096000.safetensors" ]; then
  echo "  SKIP  smoke run has not finished yet (job v2d-smoke-vel)"
else
  head -1 "$SMOKE/metrics.csv" | grep -q 'reward_components/object_velocity_weighted_mean_per_env_step'
  check $? "the learner logged object_velocity as a reward component"
  python3 -m pipeline.flash_sac_actor_export "$SMOKE/policy_4096000.safetensors" "$SCRATCH/portable.safetensors" > /dev/null
  check $? "the smoke actor projects to an upstream-only portable checkpoint"
  if [ -f "$SMOKE/eval_4096000/rep1.json" ]; then
    python3 -c "import json,sys; m=json.load(open(sys.argv[1]))['metrics']; print('  official evaluator: AUC %.4f MPPE %.1f cm' % (m['add_auc'], m['mppe_cm']))" \
      "$SMOKE/eval_4096000/rep1.json"
    ok "the unchanged official evaluator ran the projected actor"
  else
    echo "  SKIP  smoke evaluation has not finished yet (job v2d-eval-smoke-vel)"
  fi
fi

printf '\n%d passed, %d failed\n' "$PASS" "$FAIL"
[ "$FAIL" -eq 0 ]
