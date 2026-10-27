#!/usr/bin/env bash
# One-screen status of the current training waves: latest checkpoint per run (of 349,999,104
# steps) and how many of the 5 evaluation repeats exist, then the Slurm queue summary.
#
#   bash cluster/train_status.sh                 # Tier 1 grip runs + Tier 2 t2r runs
#   bash cluster/train_status.sh 't1_*_grip*'    # any run glob under outputs/cluster
#   bash cluster/train_status.sh 't1_*_vel*' 't1_*_slowvoc'   # the object-velocity / slow-VOC arms
set -uo pipefail
R=/work/mnidgu2s/project/v2d-track2-robotic-grounding
cd "$R/outputs/cluster"
GLOBS=("$@")
[ ${#GLOBS[@]} -eq 0 ] && GLOBS=('t1_ep*_grip*' 't2r_ep*_keypoints')
for g in "${GLOBS[@]}"; do
  for d in $g; do
    [ -d "$d" ] || continue
    final=349999104
    case "$d" in *slowvoc*) final=507500544 ;; esac  # the slow-VOC arms train 1.45x longer
    step=$(ls "$d" | sed -n 's/^policy_\([0-9]*\)\.safetensors$/\1/p' | sort -n | tail -1)
    reps=$(ls "$d"/eval_$final/rep*.json 2>/dev/null | wc -l)
    printf "%-32s %5s%%  eval %d/5\n" "$d" "$(( ${step:-0} * 100 / final ))" "$reps"
  done
done
echo
squeue -u "$USER" -h -o "%j %T" | sed -E 's/-ep[0-9]+.*//' | sort | uniq -c
