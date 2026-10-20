#!/usr/bin/env bash
# Read-only feasibility probe for running FlashCHORD training on a university cluster.
# Installs nothing, downloads nothing, writes nothing outside $TMPDIR.
#
# Copy it over and run it ON A GPU COMPUTE NODE (not the login node — login nodes
# usually have no GPU and a different driver):
#     scp scripts/check_cluster.sh you@cluster:~/
#     # then inside an interactive GPU job, e.g.
#     srun --gres=gpu:1 --pty bash -lc 'bash ~/check_cluster.sh'
#
# Paste the whole output back. Every line is a go/no-go input for the training plan.
set -uo pipefail

say(){ printf '\n=== %s ===\n' "$1"; }
have(){ command -v "$1" >/dev/null 2>&1 && echo "yes" || echo "NO"; }

say "0. where am I"
echo "host            : $(hostname)"
echo "os              : $(. /etc/os-release 2>/dev/null && echo "$PRETTY_NAME" || uname -sr)"
echo "kernel          : $(uname -r)"
echo "user            : $(id -un) (uid $(id -u))"
echo "in a slurm job  : ${SLURM_JOB_ID:-no}"

say "1. GPU and driver  <-- THE critical gate"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total,memory.free,driver_version,compute_cap \
             --format=csv 2>&1
  DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1)
  echo
  echo "driver major    : ${DRV:-unknown}"
  if [ -n "${DRV:-}" ] && [ "${DRV}" -ge 580 ] 2>/dev/null; then
    echo "VERDICT         : OK - meets FlashCHORD's documented 'driver 580 or newer'"
  elif [ -n "${DRV:-}" ] && [ "${DRV}" -ge 570 ] 2>/dev/null; then
    echo "VERDICT         : MARGINAL - below the documented 580; the CUDA 13 image may still"
    echo "                  run since its own gate accepts >=570. Needs an empirical test (step 5)."
  else
    echo "VERDICT         : LIKELY BLOCKER - the image ships CUDA 13.0 and needs a recent driver."
    echo "                  Ask the admins whether a newer driver or a CUDA 13 capable node exists."
  fi
else
  echo "nvidia-smi      : NOT FOUND - are you on a GPU node?"
fi

say "2. container runtime (clusters rarely allow Docker)"
echo "docker          : $(have docker)"
echo "podman          : $(have podman)"
echo "singularity     : $(have singularity)"
echo "apptainer       : $(have apptainer)"
for c in singularity apptainer; do
  command -v $c >/dev/null 2>&1 && echo "$c version : $($c --version 2>&1 | head -1)"
done
command -v docker >/dev/null 2>&1 && { echo "docker usable   : $(docker info >/dev/null 2>&1 && echo yes || echo 'NO (no permission / no daemon)')"; }

say "3. scheduler limits (training needs hours, and may be chained)"
echo "sbatch          : $(have sbatch)"
if command -v sinfo >/dev/null 2>&1; then
  echo "--- partitions (name, timelimit, gres) ---"
  sinfo -o "%20P %12l %8D %G" 2>&1 | head -15
fi
if command -v sacctmgr >/dev/null 2>&1; then
  echo "--- your associations ---"
  sacctmgr -n -p show assoc user="$(id -un)" format=Account,Partition,MaxWall,GrpTRES,MaxTRES 2>&1 | head -10
fi
if command -v scontrol >/dev/null 2>&1 && [ -n "${SLURM_JOB_ID:-}" ]; then
  echo "--- this job ---"
  scontrol show job "$SLURM_JOB_ID" 2>&1 | grep -E "TimeLimit|TRES|Partition" | head -5
fi

say "4. disk and quota (image ~21 GB; .sif adds ~10-20 GB; outputs grow)"
for d in "$HOME" "${SCRATCH:-}" "${TMPDIR:-/tmp}" /tmp; do
  [ -n "$d" ] && [ -d "$d" ] && printf '%-24s %s\n' "$d" "$(df -h "$d" 2>/dev/null | awk 'NR==2{print $4" free of "$2}')"
done
command -v quota >/dev/null 2>&1 && { echo "--- quota ---"; quota -s 2>&1 | head -8; }
echo "SCRATCH env     : ${SCRATCH:-unset}"

say "5. outbound network (needed to pull the image; NOT needed to train)"
for url in https://ghcr.io https://nvcr.io https://pypi.org; do
  code=$(curl -s -o /dev/null -m 8 -w '%{http_code}' "$url" 2>/dev/null)
  echo "$(printf '%-24s' "$url") HTTP ${code:-timeout/blocked}"
done

say "6. empirical CUDA 13 test (only if a container runtime exists)"
echo "Run ONE of these by hand; it is the real answer to step 1:"
echo
echo "  # if apptainer/singularity is available (no root needed):"
echo "  apptainer exec --nv docker://nvidia/cuda:13.0.0-base-ubuntu24.04 nvidia-smi"
echo
echo "  # if docker is usable:"
echo "  docker run --rm --gpus all nvidia/cuda:13.0.0-base-ubuntu24.04 nvidia-smi"
echo
echo "If that prints the GPU table, a CUDA 13 image runs here and we are in business."

say "SUMMARY - paste everything above back"
echo "Go/no-go hinges on: (1) driver >= 580 (or a passing step-6 test),"
echo "(2) apptainer or docker present, (3) a partition whose timelimit allows"
echo "multi-hour jobs, (4) ~50 GB of scratch."
