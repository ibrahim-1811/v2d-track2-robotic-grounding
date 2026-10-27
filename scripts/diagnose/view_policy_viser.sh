#!/usr/bin/env bash
# Watch a trained FlashSAC/PPO policy drive the G1 in a browser, via Viser.
#
# Step D. NVIDIA's scripts/view_policy.py rebuilds the whole experiment from the
# checkpoint's own baked config (same as evaluate_policy.py), so the only thing
# you have to pass is the checkpoint - the episode, embodiment, action space and
# scene all come from the file.
#
#   bash scripts/diagnose/view_policy_viser.sh <container-policy-path> [port]
#
# /out is OUT (default outputs/cluster, the cluster's run tree; rsync a run's policy file into
# it). Every data/adapter_out* tree is mounted at /work/<name>, as on the cluster, so Tier 1,
# 2 and 3 policies all resolve their references. e.g. a cluster run copied to this PC:
#   bash scripts/diagnose/view_policy_viser.sh /out/t1_ep24_keypoints/portable_policy_349999104.safetensors
#   OUT=outputs/step_D bash scripts/diagnose/view_policy_viser.sh /out/source_ep0_w1024/policy_10000384.safetensors
#
# Then open   http://localhost:8080   in a browser.
#
# Notes that cost time to learn:
#  * viser binds 0.0.0.0:8080 INSIDE the container, so the port must be
#    published with -p; newton.viewer.ViewerViser() takes the default 8080.
#  * env.voc_scale=0.0 is forced by view_policy.yaml, so what you see is the
#    UNASSISTED policy - not an object-assisted replay (see CLAUDE.md).
#  * scene.world_count is forced to 1: the viewer draws one world, and more just
#    costs VRAM.
#  * the first launch compiles Warp kernels (~1-2 min even with the cache warm,
#    because the viewer pulls in extra render kernels). The browser page will
#    not respond until "Starting viser server" appears in the terminal.
#  * Ctrl-C in this terminal stops the container and the server.
#  * a NATIVE orientation-arm checkpoint bakes _target_:
#    pipeline.flash_sac_orientation..., so pipeline/ is mounted and PYTHONPATH
#    set. Without that the viewer dies with "Error locating target". A portable
#    actor from flash_sac_actor_export.py needs neither.
#  * the checkpoint decides the EPISODE. Viewing a different episode needs a
#    policy trained on it; there is no flag to repoint one (see CLAUDE.md on
#    evaluation.source_root). To watch a reference with no policy, replay it
#    instead: scripts/diagnose/flashchord_headless_replay.py.
set -uo pipefail
cd "$(dirname "$0")/../.."
REPO=$PWD

CKPT=${1:?usage: view_policy_viser.sh <container policy path under /out> [port]}
PORT=${2:-8080}

FC_ROOT=$REPO/official/video_to_data/robotic_grounding/flash_chord
FC_ASSETS=$REPO/official/video_to_data/robotic_grounding/source/robotic_grounding/robotic_grounding/assets
OUT=$(cd "$REPO" && realpath -m "${OUT:-outputs/cluster}")
mkdir -p "$OUT" "$REPO/outputs/warp_cache"
DATA_MOUNTS=()
for tree in "$REPO"/data/adapter_out*; do
  [ -d "$tree" ] && DATA_MOUNTS+=(-v "$tree:/work/$(basename "$tree"):ro")
done

echo "=== Viser policy viewer ==="
echo "  checkpoint : $CKPT"
echo "  url        : http://localhost:$PORT"
echo "  (compiling kernels first - the page is blank until viser starts)"
echo

exec docker run --rm -it --gpus all --name v2d-viser \
  -p "${PORT}:8080" \
  -v "$FC_ROOT:/workspace" \
  -v "$FC_ASSETS:/v2d/assets:ro" \
  "${DATA_MOUNTS[@]}" \
  -v "$OUT:/out" \
  -v "$REPO/outputs/warp_cache:/root/.cache/warp" \
  -v "$REPO/pipeline:/v2d/repo/pipeline:ro" \
  -e PYTHONPATH=/v2d/repo \
  -e XLA_PYTHON_CLIENT_PREALLOCATE=false \
  -w /workspace flash_chord:latest \
  python scripts/view_policy.py \
    "evaluation.checkpoint=$CKPT" \
    viewer.backend=viser \
    scene.world_count=1
