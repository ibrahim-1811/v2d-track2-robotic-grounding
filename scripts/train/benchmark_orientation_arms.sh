#!/usr/bin/env bash
# A/B throughput and VRAM benchmark: the original twelve-term objective against
# the thirteen-term orientation objective, at the exact 350M-run settings.
#
# The progress bar's "env steps/s" is a CUMULATIVE average (total/elapsed,
# including the ~40 s JAX/Warp compile), so it climbs monotonically and must
# never be read as the steady rate. This script differences the log instead.
#
#   bash scripts/train/benchmark_orientation_arms.sh [steps]
set -uo pipefail
cd "$(dirname "$0")/../.."
STEPS=${1:-2097152}
OUT=outputs/step_D

for ARM in original orientation; do
  TAG="bench_${ARM}_$(date +%H%M%S)"
  FLAG=""
  [ "$ARM" = "original" ] && FLAG="--no-orientation"
  rm -rf "$OUT/$TAG"
  echo "=== $ARM arm: $STEPS steps ==="
  ( for _ in $(seq 1 900); do nvidia-smi --query-gpu=memory.used \
      --format=csv,noheader,nounits; sleep 2; done ) > "$OUT/${TAG}.vram" 2>&1 &
  SAMPLER=$!
  python3 scripts/train/train_orientation_flash_sac.py --tag "$TAG" \
    --total-steps "$STEPS" --save-interval "$STEPS" --replay-minimum 50000 $FLAG \
    > "$OUT/${TAG}.launch" 2>&1
  RC=$?
  kill $SAMPLER 2>/dev/null
  if [ $RC -ne 0 ]; then echo "  FAIL rc=$RC"; tail -5 "$OUT/${TAG}.launch"; continue; fi
  python3 - "$OUT/${TAG}.log" "$OUT/${TAG}.vram" "$ARM" <<'PY'
import re, sys
log, vram, arm = sys.argv[1], sys.argv[2], sys.argv[3]
rows = []
for line in open(log, errors="ignore"):
    m = re.search(r"progress: ([\d,]+)/[\d,]+ environment steps.*elapsed (\d+):(\d+):(\d+)", line)
    if m:
        steps = int(m.group(1).replace(",", ""))
        secs = int(m.group(2)) * 3600 + int(m.group(3)) * 60 + int(m.group(4))
        rows.append((steps, secs))
if len(rows) < 3:
    print("  not enough progress lines"); raise SystemExit
# Steady rate: ignore the first quarter, which still carries compile time.
start = rows[len(rows) // 4]
end = rows[-1]
d_steps, d_secs = end[0] - start[0], end[1] - start[1]
peak = max(int(v) for v in open(vram) if v.strip().isdigit())
print("  startup (first collection) : {:d} s".format(rows[1][1] if len(rows) > 1 else -1))
print("  steady throughput          : {:,.0f} env steps/s".format(d_steps / max(d_secs, 1)))
print("  total wall clock           : {:d} s for {:,d} steps".format(end[1], end[0]))
print("  peak GPU memory            : {:,d} MiB".format(peak))
print("  projected 350M run         : {:.2f} h".format(349999104 / max(d_steps / max(d_secs, 1), 1) / 3600))
PY
done
