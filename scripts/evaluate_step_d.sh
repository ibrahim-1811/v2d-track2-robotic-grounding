#!/usr/bin/env bash
# Evaluate one FlashSAC policy checkpoint, unassisted, and export its object
# trajectory for the official submission packer.
#
# Step D. Evaluation always forces voc_scale=0.0 and zero force/torque limits
# (see CLAUDE.md), so this is never an assisted replay. The trajectory export is
# what tools/pack_submission.py consumes - the metrics JSON is not.
#
#   bash scripts/evaluate_step_d.sh <policy.safetensors> <tag> [world_count] [episode_index]
#
# Paths are CONTAINER paths for the checkpoint (under /out), host-relative for
# the outputs. Example:
#
#   bash scripts/evaluate_step_d.sh /out/smokeB_ep21_transfer/policy_65536.safetensors smokeB 64 21
set -uo pipefail
cd "$(dirname "$0")/.."
REPO=$PWD

CKPT=${1:?usage: evaluate_step_d.sh <container policy path> <tag> [world_count] [episode_index]}
TAG=${2:?missing tag}
WORLDS=${3:-64}
EPISODE=${4:-21}

FC_ROOT=$REPO/official/video_to_data/robotic_grounding/flash_chord
FC_ASSETS=$REPO/official/video_to_data/robotic_grounding/source/robotic_grounding/robotic_grounding/assets
OUT=$REPO/outputs/step_D/eval
mkdir -p "$OUT" "$REPO/outputs/step_D/warp_cache" "$REPO/outputs/step_D/trajectories/$TAG"

TRAJ=$(printf "/out/trajectories/%s/episode_%06d.parquet" "$TAG" "$EPISODE")

echo "=== evaluating $TAG ==="
echo "  checkpoint : $CKPT"
echo "  worlds     : $WORLDS"
echo "  trajectory : $TRAJ"

docker run --rm --gpus all --name "v2d-eval-$TAG" \
  -v "$FC_ROOT:/workspace" -v "$FC_ASSETS:/v2d/assets:ro" \
  -v "$REPO/data/adapter_out:/work/adapter_out:ro" \
  -v "$REPO/outputs/step_D:/out" \
  -v "$REPO/outputs/step_D/warp_cache:/root/.cache/warp" \
  -e XLA_PYTHON_CLIENT_PREALLOCATE=false \
  -w /workspace flash_chord:latest \
  python scripts/evaluate_policy.py \
    "evaluation.checkpoint=$CKPT" \
    "evaluation.world_count=$WORLDS" \
    "evaluation.metrics_output=/out/eval/${TAG}.json" \
    "evaluation.object_trajectories_output=$TRAJ" \
  > "$OUT/${TAG}.log" 2>&1
RC=$?

# The image runs as root; hand the artefacts back to the host user.
docker run --rm -v "$REPO/outputs/step_D:/out" flash_chord:latest \
  chown -R 1000:1000 /out >/dev/null 2>&1

if [ $RC -ne 0 ]; then
  echo "  FAIL  evaluation exited $RC (see outputs/step_D/eval/${TAG}.log)"
  tail -20 "$OUT/${TAG}.log"
  exit $RC
fi

echo "  PASS  evaluation completed"
.venvs/episode0/bin/python - "$OUT/${TAG}.json" <<'PY'
import json, sys
report = json.load(open(sys.argv[1]))
metrics = report.get("metrics", report)
for key in ("mppe_cm", "add_auc", "spider_sr_uncentered", "maniptrans_sr", "chord_sr"):
    if key in metrics:
        print("  {:24s} {}".format(key, metrics[key]))
PY
echo "  wrote outputs/step_D/eval/${TAG}.json"
ls -la "$REPO/outputs/step_D/trajectories/$TAG/" 2>/dev/null | tail -2
