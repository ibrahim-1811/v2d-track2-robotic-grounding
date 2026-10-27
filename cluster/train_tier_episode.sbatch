#!/bin/bash
#SBATCH --job-name=v2d-tier
#SBATCH --partition=gpu4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
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
# ARM=keypoints_vel adds object_velocity on top (pipeline/flash_sac_velocity.py);
# *_slowvoc fades the VOC assist 2.5x more slowly over 507,500,544 steps.
ARM=${ARM:-stock}
case "$ARM" in *slowvoc) FINAL_STEP=507500544 ;; esac
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
  keypoints|keypoints_closegrip)
    # closegrip = the keypoint recipe on references with a scripted Dex3 grip
    # (scripts/train/make_grip_references.py; pass MANIFEST=training_manifest_t1grip.json).
    export CONFIGS=keypoints
    EXTRA="objective=keypoints_recon_body curriculum=keypoints_recon_body_flash_sac ${EXTRA:-}"
    ;;
  keypoints_tight)
    # keypoint Laplacian at 3 cm instead of 5 cm, for episodes that hold the object but tilt it
    # a few degrees (Tier 1 AUC 0.6-0.8); env.objective follows via Hydra's ${objective}.
    export CONFIGS=keypoints
    EXTRA="objective=keypoints_recon_body curriculum=keypoints_recon_body_flash_sac objective.object_keypoints_scale=0.03 ${EXTRA:-}"
    ;;
  keypoints_smooth)
    # 100x action_rate_l2 / action_l2 penalties: same AUC or better on held-object episodes and
    # 19-69% less joint shake (cluster/diagnose_jitter.sbatch, CLAUDE.md "Arm 4" / "smooth").
    export CONFIGS=keypoints
    EXTRA="objective=keypoints_recon_body curriculum=keypoints_smooth_recon_body_flash_sac objective.action_rate_l2_weight=-0.01 objective.action_l2_weight=-0.0001 ${EXTRA:-}"
    ;;
  keypoints_gentle)
    # smooth + half the body residual (0.15 -> 0.075; fingers keep 0.15), so GEAR-SONIC's own gait
    # and posture dominate: the residual, not SONIC, is what shakes the pelvis and torso.
    export CONFIGS=keypoints
    EXTRA="objective=keypoints_recon_body curriculum=keypoints_smooth_recon_body_flash_sac objective.action_rate_l2_weight=-0.01 objective.action_l2_weight=-0.0001 action.residual_scale=0.075 ${EXTRA:-}"
    ;;
  keypoints_vel)
    export CONFIGS=keypoints
    EXTRA="objective=velocity_keypoints_recon_body curriculum=velocity_keypoints_recon_body_flash_sac ${EXTRA:-}"
    ;;
  keypoints_slowvoc)
    export CONFIGS=keypoints
    EXTRA="objective=keypoints_recon_body curriculum=keypoints_slowvoc_recon_body_flash_sac training.total_environment_steps=$FINAL_STEP ${EXTRA:-}"
    ;;
  keypoints_vel_slowvoc)
    export CONFIGS=keypoints
    EXTRA="objective=velocity_keypoints_recon_body curriculum=velocity_keypoints_slowvoc_recon_body_flash_sac training.total_environment_steps=$FINAL_STEP ${EXTRA:-}"
    ;;
  *) echo "unknown ARM=$ARM"; exit 2 ;;
esac
export TAG SEQ SUPPORT EXTRA
exec bash "$R/cluster/train_flash_sac.sbatch"
