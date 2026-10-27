#!/usr/bin/env bash
# Queue one training job per episode, spread over LANES chains that each run one after
# another. At most LANES A100s are ever in use, which matters on a shared partition --
# and this cluster caps job arrays at 5 tasks (MaxArraySize=5), so an array cannot do it.
#
# Inside a lane each job waits `afterany` on the previous one, so a failed episode never
# blocks the rest. With AFTER=<jobid>, every job also waits `afterok` on that job (a smoke
# test) and is cancelled if it fails.
#
#   DRY=1 bash cluster/submit_tier_lanes.sh                 # print the plan, submit nothing
#   AFTER=349322 bash cluster/submit_tier_lanes.sh          # Tier 1, episodes 1-29, 4 lanes
#   LANES=2 EPISODES="3 8" bash cluster/submit_tier_lanes.sh
#   ARM=keypoints LANES=6 EPISODES="0 3 5 17 21 25" bash cluster/submit_tier_lanes.sh
#   PREFIX=t2 MANIFEST=training_manifest_tier2.json EPISODES="$(seq 0 29)" \
#       bash cluster/submit_tier_lanes.sh
set -euo pipefail

R=/work/mnidgu2s/project/v2d-track2-robotic-grounding
LANES=${LANES:-4}
EPISODES=${EPISODES:-$(seq 1 29)}
PREFIX=${PREFIX:-t1}
AFTER=${AFTER:-}
DRY=${DRY:-0}
# EVAL=1 also queues cluster/evaluate_policy.sbatch (5 repeats, the episode's own floor)
# behind each training job, afterok, so a finished policy is scored without a second step.
EVAL=${EVAL:-0}
MANIFEST=${MANIFEST:-training_manifest_verified.json}
# NICE=<n> queues the whole wave behind other pending jobs of ours (sbatch --nice).
NICE_ARG=()
[ -n "${NICE:-}" ] && NICE_ARG=(--nice="$NICE")

declare -a last
i=0
for ep in $EPISODES; do
  lane=$((i % LANES))
  deps=()
  [ -n "${last[$lane]:-}" ] && deps+=("afterany:${last[$lane]}")
  [ -n "$AFTER" ] && deps+=("afterok:$AFTER")
  dep_arg=()
  if [ ${#deps[@]} -gt 0 ]; then
    dep_arg=(--dependency="$(IFS=,; echo "${deps[*]}")" --kill-on-invalid-dep=yes)
  fi
  name=$(printf 'v2d-%s-ep%02d' "$PREFIX" "$ep")
  case "${ARM:-stock}" in
    stock) ;;
    keypoints_*) name="$name-${ARM#keypoints_}" ;;
    *) name="$name-${ARM:0:2}" ;;
  esac
  seed_suffix=""
  [ -n "${SEED:-}" ] && [ "$SEED" != 42 ] && { seed_suffix="_s$SEED"; name="$name-s$SEED"; }
  cmd=(sbatch --parsable --job-name="$name" "${NICE_ARG[@]}" "${dep_arg[@]}" --export=ALL,EP="$ep",PREFIX="$PREFIX"
       "$R/cluster/train_tier_episode.sbatch")
  if [ "$DRY" = 1 ]; then
    echo "lane $lane: ${cmd[*]}"
    last[$lane]="<job of ep $ep>"
  else
    jid=$("${cmd[@]}")
    echo "lane $lane: episode $ep -> job $jid ${dep_arg[0]:-}"
    last[$lane]=$jid
    if [ "$EVAL" = 1 ]; then
      floor=$(python3 - "$R/$MANIFEST" "$ep" <<'PY'
import json, sys
rows = [e for e in json.load(open(sys.argv[1]))["episodes"] if e["episode_index"] == int(sys.argv[2])]
print(rows[0]["stationary_auc"])
PY
)
      run=$(printf '%s_ep%02d_%s%s' "$PREFIX" "$ep" "${ARM:-stock}" "$seed_suffix")
      final_step=349999104
      case "${ARM:-stock}" in *slowvoc) final_step=507500544 ;; esac
      eid=$(sbatch --parsable --job-name="${name/v2d-/v2d-eval-}" -t 01:00:00 "${NICE_ARG[@]}" \
            --dependency="afterok:$jid" --kill-on-invalid-dep=yes \
            --export=ALL,RUN="$run",STEP="$final_step",REPEATS=5,EPISODE="$ep",FLOOR="$floor" \
            "$R/cluster/evaluate_policy.sbatch")
      echo "         eval of $run (floor $floor) -> job $eid"
    fi
  fi
  i=$((i + 1))
done
