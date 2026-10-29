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
#
# Tier 2 trained on the rules-clean recovered trajectory (scripts/data/recover_tier2_contact.py) packs with its
# own run prefix and the matching frame references:
#   PREFIX=t2c FRAME_REFS=outputs/tier2c_frame_references COMMIT_URL=... bash cluster/pack_tier.sh 2
# Tier 3 packs against our own reconstruction (a tier dataset root):
#   PREFIX=t3 FRAME_REFS=data/reconstructed/track_2/tier_3_cari4d_ours COMMIT_URL=... bash cluster/pack_tier.sh 3
set -euo pipefail
R=/work/mnidgu2s/project/v2d-track2-robotic-grounding
TIER=${1:?usage: COMMIT_URL=... bash cluster/pack_tier.sh <1|2|3>}
COMMIT_URL=${COMMIT_URL:?set COMMIT_URL to the GitHub /commit/<sha> that trained these policies}
PREFIX=${PREFIX:-t$TIER}
FRAME_REFS=${FRAME_REFS:-}
# OUT_NAME keeps packs of the same prefix apart (default: the prefix; that folder is overwritten).
OUT_NAME=${OUT_NAME:-$PREFIX}
# SELECT=balanced: lowest mean MPPE within 0.02 mean AUC of the best (assemble_submission.py --select).
SELECT=${SELECT:-auc}
OUT=$R/outputs/submission/$OUT_NAME
REF_ARGS=()
[ -n "$FRAME_REFS" ] && REF_ARGS=(--frame-references "/r/$FRAME_REFS")

cd "$R"
python3 scripts/evaluate/assemble_submission.py --prefix "$PREFIX" --select "$SELECT" -o "$OUT/results"
apptainer exec --cleanenv --no-mount home,cwd -B "$R":/r --env PYTHONPATH=/r/.venvs/kit_site \
  --pwd /r/outputs/kit/v2d_submission_kit "$R/containers/flash_chord.sif" \
  python tools/pack_submission.py --track "track_2_tier$TIER" --results "/r/outputs/submission/$OUT_NAME/results" \
    --dataset-root /r/data/v2d_challenge "${REF_ARGS[@]}" --code-commit "$COMMIT_URL" \
    -o "/r/outputs/submission/$OUT_NAME/track2_tier$TIER.csv" 2>&1 | grep -v pthread_setaffinity
