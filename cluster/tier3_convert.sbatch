#!/bin/bash
#SBATCH --job-name=v2d-t3-convert
#SBATCH --partition=gpu
# cari4d.sif's Warp is a CUDA 13 build: sm_75+ only, so not the V100 (sm_70) nodes wr14-19.
#SBATCH --exclude=wr[14-19,25]   # cap: at most 5 GPU nodes, wr20-24 (user, 2026-10-29)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/work/mnidgu2s/project/v2d-track2-robotic-grounding/logs/tier3_%x_%j.out

# CARI4D reconstruction -> Tier-1-shaped LeRobot episode (scripts/tier3/cari4d_to_lerobot.py),
# for every episode in EPS whose CARI4D run passed. Writes data/reconstructed/track_2/$TIER/.
#
#   EPS="0" sbatch cluster/tier3_convert.sbatch                 # one episode
#   EPS="$(seq 0 29)" sbatch cluster/tier3_convert.sbatch
set -uo pipefail
R=/work/mnidgu2s/project/v2d-track2-robotic-grounding
SIF=$R/containers/cari4d.sif
TIER=${TIER:-tier_3_cari4d_ours}
mkdir -p "$R/outputs/tier3/tmp/$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
failed=0
for EP in ${EPS:?set EPS}; do
  echo "== ep$EP $(date +%H:%M)"
  apptainer exec --nv --cleanenv -B "$R" -B "$R/outputs/tier3/tmp/$SLURM_JOB_ID":/tmp --env HOME=/tmp \
    --env PYTHONPATH="/workspace/v2d_cari4d/lib/cari4d:/workspace/v2d_sam3d_body/lib:$R:$R/data/pydeps_cari4d" \
    --env MOMENTUM_ENABLED=0 --env HF_HUB_OFFLINE=1 --env HF_HOME="$R/data/weights/cari4d/hf_home" \
    --pwd "$R" "$SIF" \
    python scripts/tier3/cari4d_to_lerobot.py --episode "$EP" --tier "$TIER" || failed=1
done
exit $failed
