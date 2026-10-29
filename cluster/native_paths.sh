# Sourced by the cluster scripts in V2D_NATIVE=1 mode (Colab): recreate, as symlinks, the paths
# the Apptainer binds give the container. The parquets bake absolute /work/<tree>/... paths and
# the training command runs from /workspace with /v2d/repo and /v2d/assets, so these must be the
# real absolute paths. Needs root (a Colab runtime is). Expects R, FC, ASSETS and OUT to be set.

# A real directory already at the path is accepted only if it IS the target (e.g. a bind mount):
# `ln -sfn` onto a directory would create the link inside it -- inside flash_chord, for /workspace.
native_link() {
  if [ -L "$2" ] || [ ! -e "$2" ]; then
    ln -sfn "$1" "$2"
  elif [ ! "$2" -ef "$1" ]; then
    echo "native_paths: $2 exists and is not $1" >&2
    return 1
  fi
}
mkdir -p /v2d /work
native_link "$FC" /workspace
native_link "$ASSETS" /v2d/assets
native_link "$R" /v2d/repo
native_link "$OUT" /out
for tree in "$R"/data/adapter_out*; do
  [ -d "$tree" ] && native_link "$tree" "/work/$(basename "$tree")"
done
true
