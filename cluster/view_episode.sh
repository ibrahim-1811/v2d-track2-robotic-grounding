#!/usr/bin/env bash
# Switch the browser viewer to another episode: cancels any running v2d-viser job, starts
# cluster/view_policy_viser.sbatch for the episode, waits until the policy is playing and
# prints the node to forward (VS Code PORTS tab: "<node>:8080").
#
#   bash cluster/view_episode.sh 4                       # policy submitted for episode 4 (live Tier 1)
#   SELECTION=outputs/submission/tier1_v2/selection.json bash cluster/view_episode.sh 4
#   RUN=t1_ep04_keypoints_grip10 bash cluster/view_episode.sh   # any evaluated run
set -euo pipefail
R=/work/mnidgu2s/project/v2d-track2-robotic-grounding
cd "$R"
old=$(squeue -u "$USER" -h -n v2d-viser -o %i)
[ -n "$old" ] && scancel $old && echo "stopped viewer job(s): $old"
if [ -n "${RUN:-}" ]; then
  jid=$(sbatch --parsable --export=ALL cluster/view_policy_viser.sbatch)
else
  jid=$(EP=${1:?usage: bash cluster/view_episode.sh <episode>} sbatch --parsable --export=ALL cluster/view_policy_viser.sbatch)
fi
log=logs/viser_v2d-viser_$jid.out
echo "job $jid queued; waiting for the policy to load (1-3 min)..."
until grep -qE "policy view:|Traceback|missing " "$log" 2>/dev/null; do
  [ -z "$(squeue -h -j "$jid")" ] && { echo "job $jid ended early:"; tail -20 "$log"; exit 1; }
  sleep 5
done
grep -E "Traceback|missing " "$log" && exit 1
node=$(squeue -h -j "$jid" -o %N)
echo
echo "playing: $(grep -oP 'checkpoint=/out/\K[^/]+' "$log" | head -1)"
echo "forward in VS Code PORTS:  $node:8080   (remove the old forward if the node changed)"
echo "then open http://localhost:8080 (reload the page)"
