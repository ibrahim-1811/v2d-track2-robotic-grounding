#!/bin/bash
# One-screen status of the tier training lanes: Slurm queue, then per running job the
# latest step, % of the 350M budget, curriculum stage, rate and ETA, then finished policies.
#   bash cluster/status.sh            # once
#   watch -n 60 bash cluster/status.sh
R=/work/mnidgu2s/project/v2d-track2-robotic-grounding
FINAL=349999104

echo "== queue ($(squeue -u "$USER" -h -t R | wc -l) running, $(squeue -u "$USER" -h -t PD | wc -l) pending)"
squeue -u "$USER" -t R -o "%.8i %.14j %.10M %.10l %R"

echo
echo "== progress"
printf "%-14s %8s %6s %5s %9s %8s\n" job step pct stage steps/s eta
for id in $(squeue -u "$USER" -h -t R -o "%i"); do
  name=$(squeue -j "$id" -h -o "%j")
  log="$R/logs/train_${name}_${id}.out"
  [ -f "$log" ] || { printf "%-14s %s\n" "$name" "(no log yet)"; continue; }
  line=$(grep '^metrics step=' "$log" | tail -1)
  [ -n "$line" ] || { printf "%-14s %s\n" "$name" "(starting up)"; continue; }
  step=$(sed -E 's/.*step=([0-9,]+).*/\1/' <<<"$line" | tr -d ,)
  stage=$(sed -E 's/.*stage=([0-9]+).*/\1/' <<<"$line")
  rate=$(sed -E 's/.*steps\/s=([0-9,]+).*/\1/' <<<"$line" | tr -d ,)
  grep -q 'rew=+*nan' <<<"$line" && { printf "%-14s %s\n" "$name" "DIVERGED: metrics are NaN since $(grep -m1 -oP 'step=\S+(?=\s+rew=\+?nan)' "$log")"; continue; }
  awk -v n="$name" -v s="$step" -v st="$stage" -v r="$rate" -v f="$FINAL" 'BEGIN{
    eta = r > 0 ? (f - s) / r : 0
    printf "%-14s %7.1fM %5.1f%% %5s %9d %4dh%02dm\n", n, s/1e6, 100*s/f, st, r, eta/3600, (eta%3600)/60 }'
done

echo
echo "== errors in running logs"
for id in $(squeue -u "$USER" -h -t R -o "%i"); do
  name=$(squeue -j "$id" -h -o "%j")
  grep -HiE 'Traceback|Error|OOM|CANCELLED|nan' "$R/logs/train_${name}_${id}.out" 2>/dev/null | grep -vE 'metrics step=|pthread_setaffinity_np' | tail -3
done

echo
echo "== finished policies"
ls "$R"/outputs/cluster/*/policy_${FINAL}.safetensors 2>/dev/null | sed -E 's#.*/outputs/cluster/([^/]+)/.*#  \1#' || echo "  none yet"
