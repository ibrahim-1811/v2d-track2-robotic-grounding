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
  c4d python "$R/scripts/tier3/sam3_text_masks.py" --video_path "$VIDEO" --object_name "${PROMPT:-$OBJ}" \
    --masks_dir "$O/masks/$SEQ" --sam3_path "$W/sam3"
fi
if [ ! -f "$O/masks/${SEQ}_masks_k0.h5" ]; then
  echo "== 2. pack masks"
  c4d python -m v2d.cari4d.lib.pack_masks --video_path "$VIDEO" \
    --human_masks_path "$O/masks/$SEQ/0" --object_masks_path "$O/masks/$SEQ/1" \
    --output_path "$O/masks/${SEQ}_masks_k0.h5"
fi
echo "== 3. CARI4D"
# One-line fix to upstream's FoundationPose export, applied to a copy and bind-mounted over the
# image's file (official/ and the image stay untouched). When the hand completely covers a small
# object, its rendered-visible area is 0 and measure_visible_silhouettes leaves centroid_distance
# at +inf; the export records that diagnostic verbatim and its own validator then rejects the
# whole run ("fp_centroid_distance contains infinity" -- mug, bat, book, sign, cone, cane, bottle).
# Upstream already writes NaN for an undefined value (_selected_metric_values) and the validator
# accepts NaN; nothing downstream reads this field. So: undefined -> NaN, nothing else changes.
FP_EXPORT=/workspace/v2d_cari4d/lib/cari4d/prep/run_foundationpose_mhr_export.py
PATCHED=$O/tmp/$SLURM_JOB_ID/run_foundationpose_mhr_export.py
apptainer exec --cleanenv "$SIF" cat "$FP_EXPORT" > "$PATCHED"
OLD='selection_diagnostics["fp_centroid_distance"].append(float(metric_values["centroid_distance"]))'
NEW='selection_diagnostics["fp_centroid_distance"].append(float(metric_values["centroid_distance"]) if np.isfinite(metric_values["centroid_distance"]) else float("nan"))'
python3 - "$PATCHED" "$OLD" "$NEW" <<'PY'
import sys
path, old, new = sys.argv[1:]
text = open(path).read()
if text.count(old) != 1 or "import numpy as np" not in text:
    sys.exit("FoundationPose export patch does not apply cleanly; refusing to run a guessed fix")
open(path, "w").write(text.replace(old, new))
PY
# Second one-line fix, same mechanism: CoCoNet's geometry-guided crop rasterizes the tracked object
# for every frame whose FoundationPose pose is "valid", and raises when that pose projects entirely
# outside the image (blue easter basket, episode 2). Such a pose gives no crop support, so it is
# treated exactly like an invalid one -- upstream's own object_pose_valid=False path (person mask
# only). The flag is local to the crop; nothing downstream reads it.
CROP=/workspace/v2d_cari4d/lib/cari4d/prep/mhr_geometry_crop.py
CROP_PATCHED=$O/tmp/$SLURM_JOB_ID/mhr_geometry_crop.py
apptainer exec --cleanenv "$SIF" cat "$CROP" > "$CROP_PATCHED"
OLD='    object_visible_silhouette = None
    if bool(object_pose_valid):'
NEW='    object_visible_silhouette = None
    if bool(object_pose_valid) and object_vertices_camera is not None and object_faces is not None:
        try:
            rasterize_mesh_silhouette(object_vertices_camera, object_faces, K, image_shape)
        except ValueError:
            object_pose_valid = False  # v2d: an off-image object pose gives no crop support
    if bool(object_pose_valid):'
python3 - "$CROP_PATCHED" "$OLD" "$NEW" <<'PY'
import sys
path, old, new = sys.argv[1:]
text = open(path).read()
if text.count(old) != 1:
    sys.exit("geometry-crop patch does not apply cleanly; refusing to run a guessed fix")
open(path, "w").write(text.replace(old, new))
PY
c4d() {  # same as above, plus the two patched files
  apptainer exec --nv --cleanenv -B "$R" -B "$O/tmp/$SLURM_JOB_ID":/tmp -B "$PATCHED:$FP_EXPORT:ro" -B "$CROP_PATCHED:$CROP:ro" \
    --env HOME=/tmp --env HF_HOME="$W/hf_home" --env HF_HUB_OFFLINE=1 \
    "$SIF" "$@"
}
c4d python -m v2d.cari4d.lib.run_inference --video_path "$VIDEO" \
  --mask_h5_path "$O/masks/${SEQ}_masks_k0.h5" --object_mesh_path "$T2/$MESH" \
  --weights_path "$W" --output_dir "$O/cari4d" --skip_weight_download --expected_frames "$FRAMES"
echo "done: $O/cari4d/$SEQ"
