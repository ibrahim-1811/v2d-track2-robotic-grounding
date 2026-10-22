# Running the 350M orientation run

Episode 0, `tall_bar_stool_tip_up_04`, 349,999,104 environment steps, about
**6 hours**. Run every command from `/home/ibrahim/v2d-track2-robotic-grounding`.

> **Do not paste this whole page into a terminal.** Step 5 stops the run. Copy
> one block at a time.

---

## Step 1 — check the machine is ready

```bash
cd /home/ibrahim/v2d-track2-robotic-grounding
docker ps
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv
df -h .
```

Expect: no containers, GPU near 0 % and ~1.1 GiB used (just the desktop), and at
least 11 GiB free. The run needs 5.22 GiB of checkpoints plus headroom.

## Step 2 — check the implementation (about 4 minutes)

```bash
bash scripts/verify/verify_orientation_flash_sac.sh
```

Expect `=== verify_orientation_flash_sac: 15 passed, 0 failed ===`. Stop if not.

## Step 3 — see the plan without launching

```bash
python3 scripts/train/train_orientation_flash_sac.py --dry-run
```

Expect 1,024 worlds, 349,999,104 steps, 683,592 optimizer updates, 34
checkpoints, 5.22 GiB, seed 42, orientation weight 2.0, and no preflight
problems.

## Step 4 — start it, in tmux

```bash
tmux new -s v2d
```

Then, **inside** the tmux session, paste only this one line:

```bash
python3 scripts/train/train_orientation_flash_sac.py --tag ep0_orient_350M
```

Training output now streams to your terminal. You will see:

- `training progress: ...` — echoed at most every 30 s (every line still goes
  to the log file);
- `metrics step=... actor=... critic=... obj_rot=...` — every 25,600 steps,
  about every 1.6 s of wall clock at the start.

Leave it running and get your shell back with **Ctrl-b** then **d**.
Come back with `tmux attach -t v2d`.

`Ctrl-C` inside the session stops the run. Checkpoints already written survive.

## Step 5 — watch it from any other terminal

```bash
cd /home/ibrahim/v2d-track2-robotic-grounding

# losses, returns and object tracking error, refreshed every minute
.venvs/episode0/bin/python scripts/train/show_training_metrics.py \
    outputs/step_D/ep0_orient_350M --watch 60
```

```bash
# all thirteen reward-term contributions, latest record
.venvs/episode0/bin/python scripts/train/show_training_metrics.py \
    outputs/step_D/ep0_orient_350M --terms
```

```bash
# raw log: just the metric lines
tail -f outputs/step_D/ep0_orient_350M.log | grep --line-buffered '^metrics step'
```

```bash
# raw log: just progress and ETA
tail -f outputs/step_D/ep0_orient_350M.log | grep --line-buffered 'training progress'
```

```bash
# is it still alive?
docker ps --filter name=v2d-train-ep0_orient_350M
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv
```

```bash
# PNG curves at any point
MPLCONFIGDIR=/tmp/mpl .venvs/episode0/bin/python scripts/train/show_training_metrics.py \
    outputs/step_D/ep0_orient_350M --plot outputs/step_D/ep0_orient_350M/metrics.png
```

## Step 6 — what to expect, and what would be wrong

| When | What you should see |
|---|---|
| first ~40 s | Warp modules loading, then `objective terms (13): (...)` ending in `object_orientation` |
| after ~40 s | `training progress` lines, settling at **~16,300 env steps/s** |
| throughout | `stage=0 voc=1.00` until 35,000,320 steps, then `stage=1 voc=0.75` |
| throughout | peak GPU memory ~7,276 MiB of 8,151 |
| every 10,000,384 steps | a new `policy_*.safetensors` and `state_*.safetensors` |

`obj_rot` will sit near **−0.00004** for the whole of stage 0. That is expected,
not a bug: the stage-0 VOC controller holds the object to about 4° of its
target, and the term is a penalty that is zero on target. It only starts to
matter as VOC decays after 35M steps.

Wrong, and worth stopping for: throughput far below 16,000 steps/s, GPU memory
near 8,100 MiB, `objective terms (12)`, or `actor=nan` / `critic=nan`.

## Step 7 — disk, optional

34 checkpoint pairs cost 5.22 GiB and you have 63 GiB, so this is insurance.

```bash
python3 scripts/train/monitor_training_disk.py outputs/step_D/ep0_orient_350M
```

```bash
# prune, keeping the newest 3 learner states; every actor is always kept
python3 scripts/train/monitor_training_disk.py outputs/step_D/ep0_orient_350M --apply --keep 3
```

It hard-refuses `source_ep0_w1024`, `submission_trial_20261021` and the other
protected directories, and never deletes an actor.

---

## Stopping it

Only when you mean it. Either `Ctrl-C` in the tmux session, or:

```bash
docker stop v2d-train-ep0_orient_350M
```

A previous attempt died at 62 s with exit 137 because this command reached it
by accident. `save_replay=false`, so resuming would refill the replay from
scratch — ask before resuming rather than guessing.

## Evaluating a checkpoint

**Stop training first** — evaluation needs the GPU.

```bash
# 1. project a native checkpoint into one the official evaluator accepts
.venvs/episode0/bin/python -c "
import sys; sys.path.insert(0, '.')
from pathlib import Path
from pipeline.flash_sac_actor_export import export_official_actor
export_official_actor(
    Path('outputs/step_D/ep0_orient_350M/policy_100003840.safetensors'),
    Path('outputs/step_D/ep0_orient_350M/portable_100003840.safetensors'),
    provenance={'arm': 'orientation', 'episode': '0'})
"
```

```bash
# 2. the UNCHANGED official evaluator, one world (the packer scores world 0 only)
bash scripts/evaluate/evaluate_step_d.sh \
    /out/ep0_orient_350M/portable_100003840.safetensors orient_10M 1 0
```

Repeat step 2 at least five times with different tags. Single-rollout
evaluation on this machine is **not** repeatable: five runs of one fixed
checkpoint gave AUC 0.1127–0.1290 and MPPE 47.6–189.6 cm. Report the spread,
never the best repeat. Compare AUC against Episode 0's stationary floor
(AUC 0.3769 / MPPE 32.59 cm), not against zero.

---

## Appendix — the raw docker command

The launcher builds this for you and runs preflight first; prefer it. If you do
use this, **paste it whole** — an abbreviated `-v .../pipeline:...` makes docker
create empty mount directories and the run dies with
`No module named pipeline.flash_sac_train_entry`.

```bash
docker \
  run \
  --rm \
  --gpus all \
  --name v2d-train-ep0_orient_350M \
  -v /home/ibrahim/v2d-track2-robotic-grounding/official/video_to_data/robotic_grounding/flash_chord:/workspace \
  -v /home/ibrahim/v2d-track2-robotic-grounding/official/video_to_data/robotic_grounding/source/robotic_grounding/robotic_grounding/assets:/v2d/assets:ro \
  -v /home/ibrahim/v2d-track2-robotic-grounding/data/adapter_out:/work/adapter_out:ro \
  -v /home/ibrahim/v2d-track2-robotic-grounding/pipeline:/v2d/repo/pipeline:ro \
  -v /home/ibrahim/v2d-track2-robotic-grounding/configs/orientation:/v2d/configs:ro \
  -v /home/ibrahim/v2d-track2-robotic-grounding/outputs/step_D:/out \
  -v /home/ibrahim/v2d-track2-robotic-grounding/outputs/step_D/warp_cache:/root/.cache/warp \
  -e PYTHONPATH=/v2d/repo \
  -e XLA_PYTHON_CLIENT_PREALLOCATE=false \
  -w /workspace \
  flash_chord:latest \
  python \
  -m pipeline.flash_sac_train_entry \
  --config-dir /v2d/configs \
  experiment=g1_recon_body_flash_sac \
  objective=orientation_recon_body \
  curriculum=orientation_recon_body_flash_sac \
  'task.parquet='"'"'/work/adapter_out/whole_body/soma/sequence_id=2026-06-04_14-16-23_tall_bar_stool_tip_up_04/robot_name=g1/data.parquet'"'"'' \
  scene.world_count=1024 \
  scene.support=false \
  training.total_environment_steps=349999104 \
  training.updates_per_collection=2.0 \
  training.replay.capacity=500000 \
  training.replay.minimum_size=50000 \
  training.checkpoint.save_interval_environment_steps=10000384 \
  training.checkpoint.save_replay=false \
  training.seed=42 \
  reset.seed=42 \
  logging.mode=disabled \
  output_dir=/out/ep0_orient_350M \
  objective.object_orientation_weight=2.0
```
