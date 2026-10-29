#!/usr/bin/env bash
# Turn a fresh Colab GPU runtime into a FlashCHORD worker for colab/run_colab.sh.
# Measured 2026-10-29 on Colab Pro: A100-SXM4-80GB, driver 580.82 (FlashCHORD needs >= 580),
# 12 vCPU, 167 GB RAM, 8.88 compute units/hour. Colab's Python is 3.13 and flash_chord wants
# 3.11, so uv installs the exact CPython 3.11.15 + uv.lock environment the cluster image is
# built from (docker/Dockerfile: uv sync --frozen --extra train). Outputs and logs live on Drive
# so a disconnect loses nothing that was written.
#
#   BUNDLE=/content/drive/MyDrive/v2d/v2d_bundle_t2c.tar DRIVE_DIR=/content/drive/MyDrive/v2d \
#       bash /content/v2d/colab/setup_colab.sh
set -euo pipefail
BUNDLE=${BUNDLE:?path of v2d_bundle_<prefix>.tar on Drive}
DRIVE_DIR=${DRIVE_DIR:?Drive folder that receives outputs/cluster and logs}
R=/content/v2d
VENV=/content/fcvenv

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
[ -f "$R/BUNDLE_COMMIT" ] || tar -xf "$BUNDLE" -C /content
echo "bundle commit $(cat "$R/BUNDLE_COMMIT")"

# The image's GL/EGL client libraries (newton and mujoco load them at import time).
apt-get -qq update
apt-get -qq install -y --no-install-recommends libgl1 libglx0 libegl1 libgles2 libopengl0 \
  libglu1-mesa libosmesa6 libglfw3 libxrender1 libxi6 libxrandr2 libxcursor1 libxinerama1 \
  libxxf86vm1 libxkbcommon0 >/dev/null

pip install -q uv==0.11.21
uv python install 3.11.15
[ -x "$VENV/bin/python" ] || uv venv --python 3.11.15 "$VENV"
(cd "$R/official/video_to_data/robotic_grounding/flash_chord" &&
  VIRTUAL_ENV="$VENV" uv sync --active --frozen --extra train)

mkdir -p "$DRIVE_DIR/outputs/cluster" "$DRIVE_DIR/logs" "$R/outputs"
for link in outputs/cluster:outputs/cluster logs:logs; do
  src=$DRIVE_DIR/${link%%:*} dst=$R/${link##*:}
  [ -d "$dst" ] && [ ! -L "$dst" ] && rmdir "$dst"
  ln -sfn "$src" "$dst"
done

"$VENV/bin/python" -c "import jax, warp, mujoco_warp, newton, flash_chord; print(jax.devices())"
echo "ready: checkout $R, python $VENV/bin/python, outputs -> $DRIVE_DIR/outputs/cluster"
