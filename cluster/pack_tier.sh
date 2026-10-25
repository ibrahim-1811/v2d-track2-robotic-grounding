#!/usr/bin/env bash
# Assemble and pack one tier's Kaggle submission CSV on the cluster (login node, ~3 min).
#
#   COMMIT_URL=https://github.com/ibrahim-1811/v2d-track2-robotic-grounding/commit/<sha> \
#       bash cluster/pack_tier.sh 1            # -> outputs/submission/tier1/track2_tier1.csv
#   COMMIT_URL=... bash cluster/pack_tier.sh 2
#
# Picks the best evaluated learned policy per episode (scripts/evaluate/assemble_submission.py,
# rep 1 of the winner -- pre-declared), then runs the kit's unchanged tools/pack_submission.py,
# which validates every frame against the clean reference and prints the four local scores.
# Upload is a separate, deliberate step (submit/v2d_submit.py needs Kaggle credentials).
set -euo pipefail
R=/work/mnidgu2s/project/v2d-track2-robotic-grounding
TIER=${1:?usage: COMMIT_URL=... bash cluster/pack_tier.sh <1|2>}
COMMIT_URL=${COMMIT_URL:?set COMMIT_URL to the GitHub /commit/<sha> that trained these policies}
OUT=$R/outputs/submission/tier$TIER

cd "$R"
python3 scripts/evaluate/assemble_submission.py --prefix "t$TIER" -o "$OUT/results"
apptainer exec --cleanenv --no-mount home,cwd -B "$R":/r --env PYTHONPATH=/r/.venvs/kit_site \
  --pwd /r/outputs/kit/v2d_submission_kit "$R/containers/flash_chord.sif" \
  python tools/pack_submission.py --track "track_2_tier$TIER" --results "/r/outputs/submission/tier$TIER/results" \
    --dataset-root /r/data/v2d_challenge --code-commit "$COMMIT_URL" \
    -o "/r/outputs/submission/tier$TIER/track2_tier$TIER.csv" 2>&1 | grep -v pthread_setaffinity
