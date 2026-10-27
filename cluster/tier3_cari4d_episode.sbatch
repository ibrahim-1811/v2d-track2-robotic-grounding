#!/bin/bash
#SBATCH --job-name=v2d-t3-cari4d
#SBATCH --partition=gpu
#SBATCH --exclude=wr[14-19]
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=/work/mnidgu2s/project/v2d-track2-robotic-grounding/logs/tier3_%x_%j.out

# Tier 3 reconstruction of one episode from its monocular exo-camera video, all inside
# containers/cari4d.sif:
#   1. SAM3 text-prompt masks: actor + object (scripts/tier3/sam3_text_masks.py)
#   2. CARI4D's own pack_masks -> <seq>_masks_k0.h5
#   3. CARI4D's own run_inference (MoGe 2, SAM 3D Body, FoundationPose, CoCoNet, refinement)
# Object mesh: the released Track 2 mesh (tier_2_synthetic_noise/mesh), as decided for Tier 3.
# Nothing here reads a Tier 1 trajectory.
#
#   EP=0 sbatch cluster/tier3_cari4d_episode.sbatch
#
# Output: outputs/tier3/{inputs,masks,cari4d}/<seq>...
set -euo pipefail

R=/work/mnidgu2s/project/v2d-track2-robotic-grounding
SIF=$R/containers/cari4d.sif
T1=$R/data/v2d_challenge/track_2/tier_1_multiview_caption
T2=$R/data/v2d_challenge/track_2/tier_2_synthetic_noise
W=$R/data/weights/cari4d
O=$R/outputs/tier3
EP=${EP:?set EP=<episode_index>}

read -r SEQ OBJ FRAMES MESH < <(python3 - "$T1/meta/episodes_metadata.jsonl" "$EP" <<'PY'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if json.loads(l)["episode_index"] == int(sys.argv[2])]
r = rows[0]
print(r["sequence_id"], r["object"], r["frames"], r["mesh"])
PY
)
VIDEO_SRC=$(printf '%s/videos/chunk-000/observation.images.exo_camera/episode_%06d.mp4' "$T1" "$EP")
mkdir -p "$O/inputs" "$O/masks" "$O/cari4d" "$O/tmp/$SLURM_JOB_ID"
# CARI4D names the sequence after the video: <sequence>.0.color.mp4
VIDEO=$O/inputs/$SEQ.0.color.mp4
[ -e "$VIDEO" ] || cp "$VIDEO_SRC" "$VIDEO"
echo "job $SLURM_JOB_ID on $(hostname -s): episode $EP $SEQ ($OBJ, $FRAMES frames), mesh $T2/$MESH"

c4d() {
  apptainer exec --nv --cleanenv -B "$R" -B "$O/tmp/$SLURM_JOB_ID":/tmp \
    --env HOME=/tmp --env HF_HOME="$W/hf_home" --env HF_HUB_OFFLINE=1 \
    "$SIF" "$@"
}

if [ ! -f "$O/masks/$SEQ/selection.json" ]; then
  echo "== 1. SAM3 masks"
  c4d python "$R/scripts/tier3/sam3_text_masks.py" --video_path "$VIDEO" --object_name "$OBJ" \
    --masks_dir "$O/masks/$SEQ" --sam3_path "$W/sam3"
fi
if [ ! -f "$O/masks/${SEQ}_masks_k0.h5" ]; then
  echo "== 2. pack masks"
  c4d python -m v2d.cari4d.lib.pack_masks --video_path "$VIDEO" \
    --human_masks_path "$O/masks/$SEQ/0" --object_masks_path "$O/masks/$SEQ/1" \
    --output_path "$O/masks/${SEQ}_masks_k0.h5"
fi
echo "== 3. CARI4D"
c4d python -m v2d.cari4d.lib.run_inference --video_path "$VIDEO" \
  --mask_h5_path "$O/masks/${SEQ}_masks_k0.h5" --object_mesh_path "$T2/$MESH" \
  --weights_path "$W" --output_dir "$O/cari4d" --skip_weight_download --expected_frames "$FRAMES"
echo "done: $O/cari4d/$SEQ"
