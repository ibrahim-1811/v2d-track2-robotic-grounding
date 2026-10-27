#!/bin/bash
#SBATCH --job-name=v2d-t3-masks
#SBATCH --partition=gpu
# SAM3 needs only PyTorch, so it runs on the V100 nodes the FlashCHORD and CARI4D jobs cannot use.
#SBATCH --exclude=wr[15-16,19-25]
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/work/mnidgu2s/project/v2d-track2-robotic-grounding/logs/tier3_%x_%j.out

# Step 1 of cluster/tier3_cari4d_episode.sbatch on its own: SAM3 text-prompt masks for the
# episodes in EPS (default all 30), in float32 (V100s have no bfloat16). The A100 job skips
# this step for any episode whose masks/<seq>/selection.json exists.
#
#   EPS="0" sbatch cluster/tier3_masks.sbatch
#   EPS="7" PROMPT="metallic iron with blue accents" sbatch cluster/tier3_masks.sbatch   # when the bare name finds nothing
set -uo pipefail
R=/work/mnidgu2s/project/v2d-track2-robotic-grounding
SIF=$R/containers/cari4d.sif
T1=$R/data/v2d_challenge/track_2/tier_1_multiview_caption
W=$R/data/weights/cari4d
O=$R/outputs/tier3
M=${MASKS:-$O/masks}   # MASKS=<dir> writes a separate set, e.g. to compare mask versions
mkdir -p "$O/inputs" "$M" "$O/tmp/$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
failed=0
for EP in ${EPS:-$(seq 0 29)}; do
  read -r SEQ OBJ < <(python3 - "$T1/meta/episodes_metadata.jsonl" "$EP" <<'PY'
import json, sys
r = [json.loads(l) for l in open(sys.argv[1]) if json.loads(l)["episode_index"] == int(sys.argv[2])][0]
print(r["sequence_id"], r["object"])
PY
)
  [ -f "$M/$SEQ/selection.json" ] && { echo "== ep$EP: masks exist"; continue; }
  VIDEO=$O/inputs/$SEQ.0.color.mp4
  [ -e "$VIDEO" ] || cp "$(printf '%s/videos/chunk-000/observation.images.exo_camera/episode_%06d.mp4' "$T1" "$EP")" "$VIDEO"
  echo "== ep$EP $SEQ ($OBJ) $(date +%H:%M)"
  apptainer exec --nv --cleanenv -B "$R" -B "$O/tmp/$SLURM_JOB_ID":/tmp --env HOME=/tmp \
    --env HF_HOME="$W/hf_home" --env HF_HUB_OFFLINE=1 "$SIF" \
    python "$R/scripts/tier3/sam3_text_masks.py" --video_path "$VIDEO" --object_name "${PROMPT:-$OBJ}" \
      --masks_dir "$M/$SEQ" --sam3_path "$W/sam3" --dtype float32 --state_on_cpu < /dev/null || failed=1
done
exit $failed
