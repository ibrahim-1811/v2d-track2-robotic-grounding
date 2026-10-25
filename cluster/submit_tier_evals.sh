#!/usr/bin/env bash
# Queue cluster/evaluate_policy.sbatch (5 unassisted world-1 repeats) for each trained
# episode, in LANES chains like submit_tier_lanes.sh. Each episode is scored against its
# own stationary floor from the manifest. Episodes without a terminal policy are skipped.
#
#   DRY=1 bash cluster/submit_tier_evals.sh
#   EPISODES="1 2 3" bash cluster/submit_tier_evals.sh
#   EPISODES=7 RUN_FMT=t1_ep%02d_seed43 bash cluster/submit_tier_evals.sh
set -euo pipefail

R=/work/mnidgu2s/project/v2d-track2-robotic-grounding
LANES=${LANES:-4}
EPISODES=${EPISODES:-$(seq 1 29)}
MANIFEST=${MANIFEST:-training_manifest_verified.json}
RUN_FMT=${RUN_FMT:-t1_ep%02d_stock}
STEP=349999104
REPEATS=${REPEATS:-5}
DRY=${DRY:-0}

declare -a last
i=0
for ep in $EPISODES; do
  run=$(printf "$RUN_FMT" "$ep")
  if [ ! -f "$R/outputs/cluster/$run/policy_${STEP}.safetensors" ]; then
    echo "skip episode $ep: no outputs/cluster/$run/policy_${STEP}.safetensors"
    continue
  fi
  floor=$(python3 - "$R/$MANIFEST" "$ep" <<'PY'
import json, sys
rows = [e for e in json.load(open(sys.argv[1]))["episodes"] if e["episode_index"] == int(sys.argv[2])]
print(rows[0]["stationary_auc"])
PY
)
  lane=$((i % LANES))
  dep_arg=()
  [ -n "${last[$lane]:-}" ] && dep_arg=(--dependency="afterany:${last[$lane]}")
  name=$(printf 'v2d-eval-ep%02d' "$ep")
  cmd=(sbatch --parsable --job-name="$name" -t 01:00:00 "${dep_arg[@]}"
       --export=ALL,RUN="$run",STEP="$STEP",REPEATS="$REPEATS",EPISODE="$ep",FLOOR="$floor"
       "$R/cluster/evaluate_policy.sbatch")
  if [ "$DRY" = 1 ]; then
    echo "lane $lane: ${cmd[*]}"
    last[$lane]="<eval ep $ep>"
  else
    jid=$("${cmd[@]}")
    echo "lane $lane: episode $ep ($run, floor $floor) -> job $jid"
    last[$lane]=$jid
  fi
  i=$((i + 1))
done
