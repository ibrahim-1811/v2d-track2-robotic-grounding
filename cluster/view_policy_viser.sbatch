#!/bin/bash
#SBATCH --job-name=v2d-viser
#SBATCH --partition=gpu
# V100 nodes wr14-19 cannot run the SONIC controller (see evaluate_policy.sbatch).
#SBATCH --exclude=wr[14-19]
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/work/mnidgu2s/project/v2d-track2-robotic-grounding/logs/viser_%x_%j.out

# Watch a policy drive the G1 in a browser: NVIDIA's unchanged scripts/view_policy.py with the
# Viser backend, on a GPU node -- the cluster equivalent of scripts/diagnose/view_policy_viser.sh.
# What you see is the unassisted policy (view_policy.yaml forces voc_scale=0), deterministic,
# world 1: the same rollout the evaluator records.
#
#   EP=4 sbatch cluster/view_policy_viser.sbatch                      # the policy submitted for episode 4
#   EP=4 SELECTION=outputs/submission/tier1_v2/selection.json sbatch ...   # another submission's pick
#   RUN=t1_ep04_keypoints_grip10 sbatch cluster/view_policy_viser.sbatch   # any evaluated run
#
# Viser always listens on port 8080 of the compute node. The log prints the node and the SSH
# tunnel to open on your PC; then browse to http://localhost:8080. scancel the job when done.
set -uo pipefail

R=/work/mnidgu2s/project/v2d-track2-robotic-grounding
SIF=$R/containers/flash_chord.sif
FC=$R/official/video_to_data/robotic_grounding/flash_chord
ASSETS=$R/official/video_to_data/robotic_grounding/source/robotic_grounding/robotic_grounding/assets
OUT=$R/outputs/cluster
STEP=${STEP:-349999104}
SELECTION=${SELECTION:-outputs/submission/t1/selection.json}

if [ -z "${RUN:-}" ]; then
  EP=${EP:?set EP=<episode> (looked up in \$SELECTION) or RUN=<run tag>}
  read -r RUN STEP < <(python3 - "$R/$SELECTION" "$EP" <<'PY'
import json, sys
rows = [s for s in json.load(open(sys.argv[1])) if s["episode_index"] == int(sys.argv[2])]
print(rows[0]["run"], rows[0]["step"])
PY
)
fi
# Thirteen-term checkpoints name our training module; the portable projection (written by
# evaluate_policy.sbatch) is what the evaluator ran, so view that when it exists.
CKPT=$RUN/policy_${STEP}.safetensors
[ -f "$OUT/$RUN/portable_policy_${STEP}.safetensors" ] && CKPT=$RUN/portable_policy_${STEP}.safetensors
test -f "$OUT/$CKPT" || { echo "missing $OUT/$CKPT"; exit 2; }

NODE=$(hostname -s)
echo "job $SLURM_JOB_ID on $NODE: $CKPT"
echo "on your PC:  ssh -N -L 8080:$NODE:8080 $USER@wr0.wr.inf.h-brs.de"
echo "then open:   http://localhost:8080   (blank until 'Starting viser server' appears below)"

DATA_BINDS=()
for tree in "$R"/data/adapter_out*; do
  [ -d "$tree" ] && DATA_BINDS+=(-B "$tree:/work/$(basename "$tree"):ro")
done
mkdir -p "$OUT/warp_cache" "$OUT/tmp/$SLURM_JOB_ID"

apptainer exec --nv --cleanenv \
  --no-mount bind-paths,home,cwd \
  -B "$FC":/workspace \
  -B "$ASSETS":/v2d/assets:ro \
  "${DATA_BINDS[@]}" \
  -B "$OUT":/out:ro \
  -B "$OUT/warp_cache":/warp_cache \
  -B "$OUT/tmp/$SLURM_JOB_ID":/tmp \
  -B "$R/pipeline":/v2d/repo/pipeline:ro \
  --env HOME=/tmp \
  --env PYTHONPATH=/v2d/repo:/workspace/src \
  --env XLA_PYTHON_CLIENT_PREALLOCATE=false \
  --env WARP_CACHE_PATH=/warp_cache \
  --pwd /workspace \
  "$SIF" python scripts/view_policy.py \
    "evaluation.checkpoint=/out/$CKPT" \
    viewer.backend=viser \
    scene.world_count=1 \
    hydra.run.dir=/tmp/view_policy
