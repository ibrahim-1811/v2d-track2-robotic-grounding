#!/bin/bash
#SBATCH --job-name=v2d-eval
#SBATCH --partition=gpu
# V100 nodes wr14-19 cannot run the SONIC controller (onnxruntime CUDA: "CUBLAS failure 8:
# architectural feature absent", sm_70), so gpu-partition evaluations use its A100 nodes.
#SBATCH --exclude=wr[14-19,25]   # cap: at most 5 GPU nodes, wr20-24 (user, 2026-10-29)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/work/mnidgu2s/project/v2d-track2-robotic-grounding/logs/eval_%x_%j.out

# Repeated, unassisted, world-1 evaluation of one checkpoint with NVIDIA's unchanged
# evaluate_policy.py -- the cluster equivalent of the laptop's former evaluate_step_d.sh,
# run REPEATS times as fresh processes (single rollouts are not repeatable, see
# docs/guides/ONBOARDING.md §5). world_count=1 because the packer scores world 0 only.
#
# Stock 12-term checkpoints go in as-is; 13-term orientation checkpoints must first be
# projected with pipeline/flash_sac_actor_export.py.
#
#   RUN=ep0_stock_nvidia_recipe STEP=349999104 sbatch cluster/evaluate_policy.sbatch
#   RUN=... STEP=... REPEATS=5 FLOOR=0.3769 sbatch cluster/evaluate_policy.sbatch
#
# V2D_NATIVE=1 (Colab, colab/run_colab.sh) runs the evaluator without Apptainer, like
# cluster/train_flash_sac.sbatch.
set -uo pipefail

R=${V2D_ROOT:-/work/mnidgu2s/project/v2d-track2-robotic-grounding}
SIF=$R/containers/flash_chord.sif
FC=$R/official/video_to_data/robotic_grounding/flash_chord
ASSETS=$R/official/video_to_data/robotic_grounding/source/robotic_grounding/robotic_grounding/assets
OUT=$R/outputs/cluster
NATIVE=${V2D_NATIVE:-0}
if [ "$NATIVE" = 1 ]; then
  : "${SLURM_JOB_ID:=$$}" "${SLURM_CPUS_PER_TASK:=$(nproc)}"
fi

RUN=${RUN:?set RUN=<run tag under outputs/cluster>}
STEP=${STEP:?set STEP=<checkpoint step>}
REPEATS=${REPEATS:-5}
EPISODE=${EPISODE:-0}
FLOOR=${FLOOR:-0.3769}

CKPT=/out/$RUN/policy_${STEP}.safetensors
test -f "$OUT/$RUN/policy_${STEP}.safetensors" || { echo "missing $OUT/$RUN/policy_${STEP}.safetensors"; exit 2; }
# A thirteen-term checkpoint names our training module, which the unchanged evaluator
# cannot import: project it once (actor tensors byte-identical) and evaluate that copy.
PORTABLE=portable_policy_${STEP}.safetensors
if [ ! -f "$OUT/$RUN/$PORTABLE" ]; then
  (cd "$R" && python3 -m pipeline.flash_sac_actor_export "$OUT/$RUN/policy_${STEP}.safetensors" "$OUT/$RUN/$PORTABLE")
  rc=$?
  [ "$rc" = 0 ] || [ "$rc" = 10 ] || { echo "actor export failed (exit $rc)"; exit 2; }
fi
[ -f "$OUT/$RUN/$PORTABLE" ] && CKPT=/out/$RUN/$PORTABLE
# EVAL_TAG keeps test or alternative evaluations out of the canonical eval_<step> folder.
EVAL=$RUN/eval_${STEP}${EVAL_TAG:+_$EVAL_TAG}
[ -e "$OUT/$EVAL/rep1.json" ] && { echo "refusing to overwrite existing $OUT/$EVAL (set EVAL_TAG)"; exit 3; }
mkdir -p "$OUT/$EVAL" "$OUT/warp_cache" "$OUT/tmp/$SLURM_JOB_ID"
echo "job $SLURM_JOB_ID on $(hostname): $CKPT x $REPEATS, world_count=1, episode $EPISODE -> $EVAL"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Every staged tier tree, each at its own /work/<name>: a checkpoint bakes its tier's
# absolute parquet path, so Tier 2 policies need /work/adapter_out_tier_2.
DATA_BINDS=()
for tree in "$R"/data/adapter_out*; do
  [ -d "$tree" ] && DATA_BINDS+=(-B "$tree:/work/$(basename "$tree"):ro")
done

[ "$NATIVE" = 1 ] && source "$R/cluster/native_paths.sh"

run() {
  if [ "$NATIVE" = 1 ]; then
    (cd /workspace && PYTHONPATH=/workspace/src XLA_PYTHON_CLIENT_PREALLOCATE=false \
       WARP_CACHE_PATH="${V2D_WARP_CACHE:-/out/warp_cache}" OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" "$@")
    return
  fi
  apptainer exec --nv --cleanenv \
    --no-mount bind-paths,home,cwd \
    -B "$FC":/workspace \
    -B "$ASSETS":/v2d/assets:ro \
    "${DATA_BINDS[@]}" \
    -B "$OUT":/out \
    -B "$OUT/tmp/$SLURM_JOB_ID":/tmp \
    --env XLA_PYTHON_CLIENT_PREALLOCATE=false \
    --env WARP_CACHE_PATH=/out/warp_cache \
    --env OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" \
    --pwd /workspace \
    "$SIF" "$@"
}

FAILS=0
for i in $(seq 1 "$REPEATS"); do
  TRAJ=$(printf "/out/%s/rep%d/episode_%06d.parquet" "$EVAL" "$i" "$EPISODE")
  mkdir -p "$OUT/$EVAL/rep$i"
  echo "--- repeat $i/$REPEATS"
  run python scripts/evaluate_policy.py \
      "evaluation.checkpoint=$CKPT" \
      "evaluation.world_count=1" \
      "evaluation.metrics_output=/out/$EVAL/rep$i.json" \
      "evaluation.object_trajectories_output=$TRAJ" \
      hydra.run.dir="/out/$EVAL/rep$i/hydra" \
    > "$OUT/$EVAL/rep$i.log" 2>&1 || { echo "    FAIL (see $OUT/$EVAL/rep$i.log)"; tail -20 "$OUT/$EVAL/rep$i.log"; FAILS=$((FAILS+1)); }
done

python3 - "$OUT/$EVAL" "$REPEATS" "$FLOOR" <<'PY'
import json, statistics, sys
from pathlib import Path
d, n, floor = Path(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
rows = []
for i in range(1, n + 1):
    p = d / f"rep{i}.json"
    if p.exists():
        m = json.loads(p.read_text()).get("metrics", {})
        rows.append(m)
        print(f"rep{i}: " + "  ".join(f"{k}={m[k]:.4f}" for k in ("add_auc", "mppe_cm") if k in m))
for k in ("add_auc", "mppe_cm", "spider_sr_uncentered", "maniptrans_sr", "chord_sr"):
    v = [r[k] for r in rows if k in r]
    if v:
        sd = statistics.stdev(v) if len(v) > 1 else 0.0
        print(f"{k:22s} mean {statistics.mean(v):.4f} +/- {sd:.4f}  min {min(v):.4f}  max {max(v):.4f}  (n={len(v)})")
auc = [r["add_auc"] for r in rows if "add_auc" in r]
if auc:
    a = statistics.mean(auc)
    gate = ">= 0.50: config was the problem, scale up" if a >= 0.50 else (
        "0.38-0.50: helps but insufficient" if a >= 0.38 else "< 0.38: STOP, re-diagnose")
    print(f"AUC {a:.4f} vs stationary floor {floor:.4f}  ->  {gate}")
PY
exit $FAILS
