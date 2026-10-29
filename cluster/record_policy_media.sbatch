#!/bin/bash
#SBATCH --job-name=v2d-media
#SBATCH --partition=gpu
# V100 nodes wr14-19 cannot run the SONIC controller (see evaluate_policy.sbatch).
#SBATCH --exclude=wr[14-19,25]   # cap: at most 5 GPU nodes, wr20-24 (user, 2026-10-29)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/work/mnidgu2s/project/v2d-track2-robotic-grounding/logs/media_%x_%j.out

# Save policies as files you can watch on any PC, without the cluster or a GPU:
#   epNN_<run>.mp4     a video of the whole episode
#   epNN_<run>.viser   the interactive 3D scene (orbit, zoom, timeline), played in a browser
# Same unassisted, deterministic world-1 rollout as cluster/view_policy_viser.sbatch, through
# NVIDIA's unchanged scripts/view_policy.py (MP4 = its own mp4 backend; .viser = Newton's
# record_to_viser, see scripts/diagnose/record_policy_viser.py).
#
#   sbatch cluster/record_policy_media.sbatch                     # all 30 episodes of $SELECTION
#   EPS="4 24" sbatch cluster/record_policy_media.sbatch          # some episodes
#   RUNS="t1_ep04_keypoints_grip10" sbatch cluster/record_policy_media.sbatch   # any evaluated run
#   FORMATS=mp4 / FORMATS=viser, SET=<folder name>, STEP=<checkpoint step>, FRAMES=<limit>
#
# Output: outputs/media/$SET/ (default: the selection's folder name, or "runs"), plus
# outputs/media/viser_client/ -- the Viser web player from the image, needed for .viser files.
set -uo pipefail

R=/work/mnidgu2s/project/v2d-track2-robotic-grounding
SIF=$R/containers/flash_chord.sif
FC=$R/official/video_to_data/robotic_grounding/flash_chord
ASSETS=$R/official/video_to_data/robotic_grounding/source/robotic_grounding/robotic_grounding/assets
OUT=$R/outputs/cluster
SELECTION=${SELECTION:-outputs/submission/t1/selection.json}
FORMATS=${FORMATS:-mp4 viser}
if [ -n "${RUNS:-}" ]; then SET=${SET:-runs}; else SET=${SET:-$(basename "$(dirname "$SELECTION")")}; fi
MEDIA=$R/outputs/media
mkdir -p "$MEDIA/$SET" "$OUT/warp_cache" "$OUT/tmp/$SLURM_JOB_ID"
echo "job $SLURM_JOB_ID on $(hostname -s): $SET -> $MEDIA/$SET  [$FORMATS]"

# One line per policy: name run checkpoint frames eye target (camera, MP4 only). Frames = rows of the evaluator's trajectory
# (one per dataset frame), else the reference parquet's rows.
PLAN=$(apptainer exec --cleanenv -B "$R" "$SIF" python3 - "$R" "$SELECTION" "${EPS:-}" "${RUNS:-}" "${STEP:-349999104}" <<'PY' 2>/dev/null
import glob, json, struct, sys
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

root, selection, eps, runs, step = Path(sys.argv[1]), sys.argv[2], sys.argv[3].split(), sys.argv[4].split(), sys.argv[5]
out = root / "outputs/cluster"

def reference(run, step):
    with open(out / run / f"policy_{step}.safetensors", "rb") as f:
        meta = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))["__metadata__"]
    parquet = json.loads(meta["user.resolved_config_json"])["task"]["parquet"]
    return pq.read_table(str(parquet).replace("/work/adapter_out", str(root / "data/adapter_out"), 1))

def rows(run, step, ref):
    traj = sorted(glob.glob(str(out / run / f"eval_{step}" / "rep1" / "episode_*.parquet")))
    if traj:
        return pq.ParquetFile(traj[0]).metadata.num_rows
    return len(ref.column("robot_root_position")[0])

def camera(ref):
    """Look at the robot and object from 2.8 m, 40 deg off the robot's frame-0 facing, 1.0 m up."""
    root = np.asarray(ref.column("robot_root_position")[0].as_py()[0])
    w, x, y, z = ref.column("robot_root_wxyz")[0].as_py()[0]
    obj = np.asarray(ref.column("object_root_position")[0].as_py()[0])
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)) + np.radians(40)
    target = np.array([(root[0] + obj[0]) / 2, (root[1] + obj[1]) / 2, 0.7])
    eye = target + np.array([2.8 * np.cos(yaw), 2.8 * np.sin(yaw), 1.0])
    return "[{:.3f},{:.3f},{:.3f}] [{:.3f},{:.3f},{:.3f}]".format(*eye, *target)

def ckpt(run, step):
    portable = out / run / f"portable_policy_{step}.safetensors"
    return f"{run}/{portable.name}" if portable.exists() else f"{run}/policy_{step}.safetensors"

if runs:
    for run in runs:
        ref = reference(run, step)
        print(f"{run} {run} {ckpt(run, step)} {rows(run, step, ref)} {camera(ref)}")
else:
    for s in json.load(open(root / selection)):
        if eps and str(s["episode_index"]) not in eps:
            continue
        run, step = s["run"], s["step"]
        ref = reference(run, step)
        print(f"ep{s['episode_index']:02d}_{run} {run} {ckpt(run, step)} {rows(run, step, ref)} {camera(ref)}")
PY
)
[ -n "$PLAN" ] || { echo "nothing to record (check EPS/RUNS/SELECTION)"; exit 2; }

if [[ " $FORMATS " == *" viser "* ]] && [ ! -f "$MEDIA/viser_client/index.html" ]; then
  apptainer exec --cleanenv -B "$MEDIA" "$SIF" cp -r /opt/venv/lib/python3.11/site-packages/viser/client/build "$MEDIA/viser_client"
fi
mkdir -p "$MEDIA/viser_client/recordings"
LOGS=$R/logs/media/$SET  # full output per policy and format
mkdir -p "$LOGS"

DATA_BINDS=()
for tree in "$R"/data/adapter_out*; do
  [ -d "$tree" ] && DATA_BINDS+=(-B "$tree:/work/$(basename "$tree"):ro")
done

fc() {  # fc <python script> <overrides...>
  apptainer exec --nv --cleanenv \
    --no-mount bind-paths,home,cwd \
    -B "$FC":/workspace \
    -B "$ASSETS":/v2d/assets:ro \
    "${DATA_BINDS[@]}" \
    -B "$OUT":/out:ro \
    -B "$OUT/warp_cache":/warp_cache \
    -B "$OUT/tmp/$SLURM_JOB_ID":/tmp \
    -B "$R/pipeline":/v2d/repo/pipeline:ro \
    -B "$R/scripts/diagnose":/v2d/tools:ro \
    -B "$MEDIA":/media \
    --env HOME=/tmp \
    --env PYTHONPATH=/v2d/repo:/workspace/src \
    --env XLA_PYTHON_CLIENT_PREALLOCATE=false \
    --env WARP_CACHE_PATH=/warp_cache \
    --env PYOPENGL_PLATFORM=egl \
    --env PYGLET_HEADLESS=true \
    --env V2D_RECORD_VISER="${VISER_OUT:-}" \
    --pwd /workspace \
    "$SIF" python "$@" < /dev/null
}

failed=0
while read -r NAME RUN CKPT N EYE TARGET; do
  N=${FRAMES:-$N}
  echo "== $NAME: $CKPT, $N frames"
  common=("evaluation.checkpoint=/out/$CKPT" scene.world_count=1 hydra.run.dir=/tmp/view_policy "viewer.num_frames=$N")
  if [[ " $FORMATS " == *" mp4 "* ]]; then
    # 30 fps = one dataset frame per video frame, i.e. the speed of the original human video.
    # markers.keypoints/contacts=false: the GL renderer rejects the scalar radius their spheres pass
    # (warp kernel 'update_vbo_transforms_from_points' wants an array); Viser accepts it.
    fc scripts/view_policy.py "${common[@]}" viewer.backend=mp4 viewer.headless=true markers.keypoints=false markers.contacts=false \
      "viewer.camera_position=$EYE" "viewer.camera_target=$TARGET" \
      viewer.width=1280 viewer.height=720 viewer.video_fps=30 viewer.realtime=false \
      "viewer.output_path=/media/$SET/$NAME.mp4" > "$LOGS/$NAME.mp4.log" 2>&1 \
      && grep -h "Video:" "$LOGS/$NAME.mp4.log" || { failed=1; tail -15 "$LOGS/$NAME.mp4.log"; }
  fi
  if [[ " $FORMATS " == *" viser "* ]]; then
    VISER_OUT=/media/viser_client/recordings/$SET/$NAME.viser \
      fc /v2d/tools/record_policy_viser.py "${common[@]}" viewer.backend=viser viewer.realtime=false \
      > "$LOGS/$NAME.viser.log" 2>&1 \
      && grep -h "Recording saved" "$LOGS/$NAME.viser.log" || { failed=1; tail -15 "$LOGS/$NAME.viser.log"; }
  fi
done <<< "$PLAN"

echo
ls -la "$MEDIA/$SET" "$MEDIA/viser_client/recordings/$SET" 2>/dev/null
python3 "$R/scripts/diagnose/make_media_index.py"
echo "done: $MEDIA -- copy the whole folder to a PC, then:"
echo "  cd media && python3 -m http.server 8000    # then open http://localhost:8000 (index of every episode)"
exit $failed
