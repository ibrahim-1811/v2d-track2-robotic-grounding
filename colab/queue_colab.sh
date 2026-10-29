#!/usr/bin/env bash
# Run a queue of colab/run_colab.sh jobs, LANES at a time on the one Colab GPU (each run of the
# 2048-world recipe is a separate process; an A100-80GB has room for two).
#
#   PREFIX=t2c MANIFEST=training_manifest_tier2c.json LANES=2 \
#       QUEUE="4:keypoints:43 25:keypoints:43 28:keypoints:43 1:keypoints:43" bash colab/queue_colab.sh
set -euo pipefail
: "${PREFIX:?}" "${MANIFEST:?}" "${QUEUE:?EP:ARM:SEED entries, space-separated}"
LANES=${LANES:-2}
export PREFIX MANIFEST
printf '%s\n' $QUEUE | xargs -P "$LANES" -I{} bash -c '
  IFS=: read -r ep arm seed <<<"{}"
  EP=$ep ARM=$arm SEED=$seed bash /content/v2d/colab/run_colab.sh \
    >"/content/v2d/logs/queue_${PREFIX}_ep${ep}_${arm}_s${seed}.out" 2>&1 \
    && echo "$(date +%H:%M) done  ep$ep $arm s$seed" || echo "$(date +%H:%M) FAILED ep$ep $arm s$seed"'
