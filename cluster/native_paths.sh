# Sourced by the cluster scripts in V2D_NATIVE=1 mode (Colab): recreate, as symlinks, the paths
# the Apptainer binds give the container. The parquets bake absolute /work/<tree>/... paths and
# the training command runs from /workspace with /v2d/repo and /v2d/assets, so these must be the
# real absolute paths. Needs root (a Colab runtime is). Expects R, FC, ASSETS and OUT to be set.
mkdir -p /v2d /work
ln -sfn "$FC" /workspace
ln -sfn "$ASSETS" /v2d/assets
ln -sfn "$R" /v2d/repo
ln -sfn "$OUT" /out
for tree in "$R"/data/adapter_out*; do
  [ -d "$tree" ] && ln -sfn "$tree" "/work/$(basename "$tree")"
done
true
