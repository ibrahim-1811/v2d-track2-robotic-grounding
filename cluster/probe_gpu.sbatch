#!/bin/bash
#SBATCH --job-name=v2d-probe
#SBATCH --partition=gpu
#SBATCH --exclude=wr[14-19,25]   # cap: at most 5 GPU nodes, wr20-24 (user, 2026-10-29)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/work/mnidgu2s/project/v2d-track2-robotic-grounding/logs/probe_%N_%j.out

# Read-only GPU probe: model, VRAM, driver, and whether a CUDA 13 container runs here.
R=/work/mnidgu2s/project/v2d-track2-robotic-grounding
export APPTAINER_CACHEDIR=$R/.cache/apptainer
nvidia-smi --query-gpu=index,name,memory.total,driver_version,compute_cap --format=csv
bash $R/scripts/verify/check_cluster.sh
echo "=== CUDA 13 container test ==="
if [ -f $R/containers/cuda13-base.sif ]; then
  apptainer exec --nv $R/containers/cuda13-base.sif nvidia-smi
else
  echo "cuda13-base.sif not present"
fi
