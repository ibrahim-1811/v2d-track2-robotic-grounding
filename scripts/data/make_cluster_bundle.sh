#!/usr/bin/env bash
# Build the one archive a collaborator needs in order to train on a cluster.
#
# The git repo carries code and docs. This carries the things git cannot:
# the 30 retargeted G1 references (which are what training actually reads),
# the per-episode training manifest, and our best policy as a reference point.
#
# NOT included, on purpose:
#   * official/        -- a public clone, pinned at 193382a (1.5 GB)
#   * docker images    -- built from upstream's own Dockerfile (21 GB)
#   * data/v2d_challenge -- only the submission packer needs it, and packing
#                           happens on Ibrahim's machine
#   * SOMA-X body model  -- only needed to RE-retarget, which the bundle avoids
#
#   bash scripts/data/make_cluster_bundle.sh [output_dir]
set -uo pipefail
cd "$(dirname "$0")/../.."
REPO=$PWD
OUT=${1:-outputs/cluster_bundle}
STAMP=$(date +%Y%m%d)
NAME="v2d_cluster_bundle_${STAMP}"
STAGE="$OUT/$NAME"

command -v tar >/dev/null || { echo "tar not found"; exit 1; }
[ -d data/adapter_out ] || { echo "missing data/adapter_out -- nothing to ship"; exit 1; }

echo "=== building $NAME ==="
rm -rf "$STAGE"; mkdir -p "$STAGE"

# 1. the references: the irreplaceable part
echo "  [1/4] G1 references (2.0 GB) ..."
mkdir -p "$STAGE/data"
cp -r data/adapter_out "$STAGE/data/adapter_out"

# 2. the per-episode facts a trainer needs
echo "  [2/4] training manifest ..."
.venvs/episode0/bin/python scripts/train/make_training_manifest.py \
  -o "$STAGE/training_manifest.json" >/dev/null
cp outputs/step_D/leaderboard_floor.json "$STAGE/leaderboard_floor.json" 2>/dev/null || true

# 3. our best policy, as a baseline to beat (not to warm-start from)
echo "  [3/4] reference policy ..."
mkdir -p "$STAGE/reference_policy"
cp outputs/step_D/ep0_orient_350M/policy_349999104.safetensors "$STAGE/reference_policy/" 2>/dev/null || true
cp outputs/step_D/ep0_orient_350M/metrics.jsonl "$STAGE/reference_policy/" 2>/dev/null || true

cat > "$STAGE/README.md" <<'INNER'
# Cluster bundle

Unpack this **inside your clone of the repo**, so `data/adapter_out` lands
beside `pipeline/` and `scripts/`:

```bash
git clone git@github.com:ibrahim-1811/v2d-track2-robotic-grounding.git
cd v2d-track2-robotic-grounding
tar xzf ../v2d_cluster_bundle_*.tar.gz --strip-components=1
```

## What is in here

| | |
|---|---|
| `data/adapter_out/` | **the 30 retargeted G1 references.** This is what training reads. Self-contained: the mesh and URDF paths baked into each parquet resolve to `/work/adapter_out/...`, which is where you mount this directory. |
| `training_manifest.json` | per episode: the container parquet path, whether `scene.support` must be false, and the stationary-baseline AUC |
| `leaderboard_floor.json` | the "do nothing" score for every episode |
| `reference_policy/` | our best Episode 0 policy (AUC 0.3491) and its training metrics — a baseline to beat, **not** something to warm-start from |

## What you still need

1. **NVIDIA upstream**, pinned:
   ```bash
   mkdir -p official && cd official
   git clone https://github.com/nvidia-isaac/video_to_data.git
   cd video_to_data && git checkout 193382a && git lfs pull && cd ../..
   ```
2. **The `flash_chord` container image**, built from upstream's own Dockerfile.
3. **A Python env** for analysis:
   ```bash
   python3 -m venv --system-site-packages .venvs/episode0
   .venvs/episode0/bin/pip install pyarrow==15.0.2
   ```

## Then read, in order

1. `docs/architecture/ARCHITECTURE.md` — what this is and how it works
2. `CLAUDE.md` — the traps. Every line was learned by something breaking.
3. `docs/guides/ONBOARDING.md` — setup, and the task list

## The first thing to do

Probe the hardware and report back **VRAM per GPU**. The whole plan needs
>= 48 GiB; below that it changes.

```bash
bash scripts/verify/check_cluster.sh
```

Then the one run that decides everything — Episode 0 on NVIDIA's *exact*
shipped recipe, which we have never been able to run on 8 GB:

`scene.world_count=2048`, `training.replay.capacity=10000000`,
`training.updates_per_collection=4.0`,
`training.total_environment_steps=349999104`, stock 12-term objective,
`scene.support=false`, seed 42. Expect ~2 h on an L40S.

Episode 0's "do nothing" floor is **0.3769**; our crippled 8 GB run reached
**0.3491**. If the real recipe clears **0.50**, the configuration was the
problem and we scale to all 30 episodes. If it comes in under **0.38**, stop
and tell us — do not spend 60 GPU-hours.
INNER

# 4. archive + checksum
echo "  [4/4] compressing ..."
tar czf "$OUT/$NAME.tar.gz" -C "$OUT" "$NAME"
rm -rf "$STAGE"
( cd "$OUT" && sha256sum "$NAME.tar.gz" > "$NAME.tar.gz.sha256" )

echo
echo "  $OUT/$NAME.tar.gz  ($(du -h "$OUT/$NAME.tar.gz" | cut -f1))"
cat "$OUT/$NAME.tar.gz.sha256"
