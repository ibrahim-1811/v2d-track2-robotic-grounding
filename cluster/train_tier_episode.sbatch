#!/bin/bash
#SBATCH --job-name=v2d-tier
#SBATCH --partition=gpu4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/work/mnidgu2s/project/v2d-track2-robotic-grounding/logs/train_%x_%j.out

# One policy for one episode. Looks up the episode's sequence and scene.support flag in
# the training manifest, then runs cluster/train_flash_sac.sbatch unchanged -- NVIDIA's
# exact shipped g1_recon_body_flash_sac recipe, the one that took Episode 0 to AUC 0.4622.
#
# Normally queued by cluster/submit_tier_lanes.sh. By hand:
#   EP=8 sbatch --job-name=v2d-t1-ep08 cluster/train_tier_episode.sbatch
#   PREFIX=t2 MANIFEST=training_manifest_tier2.json EP=8 sbatch cluster/train_tier_episode.sbatch
#
# Checkpoints land in outputs/cluster/<PREFIX>_epNN_stock/. A run whose terminal policy
# already exists exits at once, so re-queueing a lane resumes it.
set -euo pipefail

R=/work/mnidgu2s/project/v2d-track2-robotic-grounding
MANIFEST=${MANIFEST:-training_manifest_verified.json}
PREFIX=${PREFIX:-t1}
EP=${EP:?set EP=<episode_index>}
FINAL_STEP=349999104

# The login and compute nodes ship Python 3.9: no nested same-quote f-strings.
read -r SEQ SUPPORT ADAPTER < <(python3 - "$R/$MANIFEST" "$EP" <<'PY'
import json, sys
manifest, episode = sys.argv[1], int(sys.argv[2])
payload = json.load(open(manifest))
rows = [e for e in payload["episodes"] if e["episode_index"] == episode]
if len(rows) != 1:
    sys.exit("episode {} appears {} times in {}".format(episode, len(rows), manifest))
# Tier 1's original manifest predates adapter_root; it lives in data/adapter_out.
print(rows[0]["sequence_id"], "true" if rows[0]["scene_support"] else "false", payload.get("adapter_root", "adapter_out"))
PY
)
export ADAPTER

# ARM=stock (default) is NVIDIA's twelve-term objective; ARM=keypoints adds the ADD-shaped
# object_keypoints term (pipeline/flash_sac_keypoints.py) through configs/keypoints/.
ARM=${ARM:-stock}
# Extra seeds get their own run: <prefix>_epNN_<arm>_s<seed>. Seed 42 keeps the bare name.
SEED_SUFFIX=""
[ -n "${SEED:-}" ] && [ "$SEED" != 42 ] && SEED_SUFFIX="_s$SEED"
TAG=$(printf '%s_ep%02d_%s%s' "$PREFIX" "$EP" "$ARM" "$SEED_SUFFIX")
echo "episode $EP -> tag=$TAG seq=$SEQ support=$SUPPORT arm=$ARM manifest=$MANIFEST"
if [ -z "${EXTRA:-}" ] && [ -f "$R/outputs/cluster/$TAG/policy_${FINAL_STEP}.safetensors" ]; then
  echo "already trained: outputs/cluster/$TAG/policy_${FINAL_STEP}.safetensors -- nothing to do"
  exit 0
fi
case "$ARM" in
  stock) ;;
  keypoints)
    export CONFIGS=keypoints
    EXTRA="objective=keypoints_recon_body curriculum=keypoints_recon_body_flash_sac ${EXTRA:-}"
    ;;
  *) echo "unknown ARM=$ARM"; exit 2 ;;
esac
export TAG SEQ SUPPORT EXTRA
exec bash "$R/cluster/train_flash_sac.sbatch"
