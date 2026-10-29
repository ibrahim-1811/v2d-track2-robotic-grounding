#!/usr/bin/env bash
# Pack what a Colab runtime needs to train and evaluate one tier with the cluster's own scripts
# (V2D_NATIVE=1): the committed repo (git archive HEAD, commit in BUNDLE_COMMIT), NVIDIA's
# flash_chord source and assets, and only that tier's data trees -- a Tier 2 bundle holds no
# Tier 1 data. No container: colab/setup_colab.sh rebuilds flash_chord's uv.lock environment.
#
#   PREFIX=t2c bash colab/make_bundle.sh     # -> outputs/colab/v2d_bundle_t2c.tar (~2.9 GB)
#   PREFIX=t3c bash colab/make_bundle.sh
#   PREFIX=t2c TREES="adapter_out_tier_2c adapter_out_tier_2" bash colab/make_bundle.sh
set -euo pipefail
R=$(cd "$(dirname "$0")/.." && pwd)
PREFIX=${PREFIX:?set PREFIX=t2c, t2, t3c or t1}
if [ -z "${TREES:-}" ]; then
  # Recovered trees keep their object meshes in the tree they were derived from.
  case "$PREFIX" in
    t2c) TREES="adapter_out_tier_2c adapter_out_tier_2" ;;
    t2) TREES="adapter_out_tier_2" ;;
    t3c) TREES="adapter_out_tier_3c adapter_out_tier_3_cari4d_ours" ;;
    t1) TREES="adapter_out adapter_out_grip" ;;
    *) echo "unknown PREFIX=$PREFIX -- set TREES to the data trees it needs"; exit 2 ;;
  esac
fi
# The bundle is HEAD, not the working tree: refuse to ship code that differs from what runs.
git -C "$R" diff --quiet HEAD || { echo "uncommitted changes to tracked files -- commit first"; exit 2; }

FCREL=official/video_to_data/robotic_grounding/flash_chord
ASREL=official/video_to_data/robotic_grounding/source/robotic_grounding/robotic_grounding/assets
OUT=$R/outputs/colab
TAR=$OUT/v2d_bundle_$PREFIX.tar
SHA=$(git -C "$R" rev-parse HEAD)
mkdir -p "$OUT"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
echo "$SHA" >"$tmp/BUNDLE_COMMIT"

git -C "$R" archive --format=tar --prefix=v2d/ -o "$TAR" HEAD
# ,S: prefix member names but leave symlink targets alone.
tar -rf "$TAR" -C "$tmp" --transform 's,^,v2d/,S' BUNDLE_COMMIT
tar -rf "$TAR" -C "$R" --transform 's,^,v2d/,S' --exclude=.git --exclude=__pycache__ \
  --exclude="$FCREL/outputs" --exclude="$FCREL/logs" "$FCREL" "$ASREL"
for tree in $TREES; do
  test -d "$R/data/$tree" || { echo "missing data/$tree"; exit 2; }
  tar -rf "$TAR" -C "$R" --transform 's,^,v2d/,S' "data/$tree"
done
echo "commit $SHA, trees: $TREES"
ls -lh "$TAR"
sha256sum "$TAR"
