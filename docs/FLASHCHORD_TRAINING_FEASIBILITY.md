# FlashCHORD G1 PPO Training Feasibility — Minimal Smoke Test

Can we train a FlashCHORD G1 PPO policy on an 8 GB laptop GPU, produce an
authentic checkpoint, and score it with NVIDIA's official evaluator?

Companion documents: [FLASHCHORD_BASELINE_SMOKE.md](FLASHCHORD_BASELINE_SMOKE.md)
(replay verification) · [CHALLENGE_BASELINE.md](CHALLENGE_BASELINE.md) (dataset and
submission gap) · [NVIDIA_UPSTREAM.md](NVIDIA_UPSTREAM.md) (code map).

- **Run:** 2026-10-20
- **Upstream code:** commit `193382a184308f6800331f3db829f76dc4294cfe` (unmodified)
- **Image:** `flash_chord:latest`, container `flash_chord-latest-gpu0` (reused, not rebuilt)
- **GPU:** RTX 5070 Laptop, 8151 MiB, driver 580.178.04 · **Host RAM:** 32 GB
- **Recipe:** `experiment=g1_recon_body_ppo`, entrypoint `scripts/train_rl.py`
- **Reference:** NVIDIA's supplied `snack_box_pick_and_place_01`, `robot_name=g1`

Everything below is marked **[VERIFIED]** (observed directly), **[DOC]** (stated
by NVIDIA, not independently checked) or **[GAP]/[UNRESOLVED]**.

---

## 1. Result: PASS

| Milestone | Verdict | Evidence |
| --- | --- | --- |
| Official PPO config composes and validates | ✅ **PASS** | `--cfg job --resolve` clean, all overrides propagate |
| Minimal 2-iteration training smoke test | ✅ **PASS** | 2 iterations, 1,536 env steps, **45 s**, peak **1,966 MiB** |
| Authentic FlashCHORD PPO checkpoint saved | ✅ **PASS** | `model_2.safetensors`, 28.67 MB, NVIDIA's reader reports `algorithm = ppo` |
| Official evaluator at `world_count=1`, unassisted | ✅ **PASS** | 800/800 steps, **18 s**, peak **1,867 MiB**, MPPE 60.48 cm |
| Object assistance removed during evaluation | ✅ **PASS** | `voc_scale` 1.0 → **0.0**, `max_force`/`max_torque` 60 → **0.0** |
| Useful world counts on 8 GB | ✅ up to **1,024** comfortably, 1,536 at the edge | 2,048 OOMs |
| Full baseline training on this machine | ❌ **NOT FEASIBLE** | **25.3 days** at best measured throughput |

**The entire train → checkpoint → evaluate pipeline works on this 8 GB GPU.**
No upstream file was modified and nothing was downloaded.

> **The checkpoint produced here is NOT a competitive baseline.** It is 2 PPO
> iterations — 1,536 environment steps against the recipe's terminal
> 1,966,080,000, i.e. **0.00008 %** of the schedule. Its MPPE of 60.48 cm is
> approximately the error of doing nothing. It exists to prove the mechanism, and
> must never be quoted as a Track 2 result.

---

## 2. What NVIDIA's pinned PPO recipe specifies **[VERIFIED]**

`src/flash_chord/configs/experiment/g1_recon_body_ppo.yaml` — "Canonical
G1+Dex3/SONIC ReconBody PPO recipe", selecting `sim: recon_body`,
`observation: recon_body`, `objective: recon_body`, `termination: recon_body`,
`reset: recon_body`, `curriculum: recon_body_ppo`, `action: sonic_joint_residual`.

| Setting | Recipe value | Our smoke override |
| --- | ---: | ---: |
| `scene.world_count` | 4096 | **32** |
| `training.max_iterations` | 20000 | **2** |
| `training.save_interval` | 500 | **1** |
| `training.rollout_steps` | 24 | 24 (unchanged) |
| `reset.seed` | 42 | 42 (unchanged) |
| Network (actor & critic) | `[1024, 512, 256, 128]`, elu | unchanged |
| `ppo.learning_rate` / schedule | 5e-4 / adaptive | unchanged |
| `ppo.mini_batches` / `learning_epochs` | 4 / 5 | unchanged |
| `logging` | wandb | `mode=disabled` |

Terminal budget: 20000 × 4096 × 24 = **1,966,080,000 environment steps**.

**Checkpoints** are written by `training/ppo/runner.py:329` as
`model_<iteration>.safetensors` when
`(iteration+1) % save_interval == 0 or iteration+1 == max_iterations`. The second
clause means **a 2-iteration run saves a terminal checkpoint even without
touching `save_interval`**; we set `save_interval=1` anyway so both iterations
were captured.

**Curriculum, stage 0** (`curriculum/recon_body_ppo.yaml`, thresholds start at
iteration 2000) holds `voc_scale: 1.0`. So iterations 1–2 train **with full
object assistance**, by NVIDIA's design — assistance only decays from iteration
2000 (0.75) through iteration 12000 (0.0). Our 2-iteration run never leaves
stage 0. Evaluation removes assistance regardless (§5).

---

## 3. The one blocker worth reporting: JAX preallocation **[VERIFIED]**

`scripts/train_rl.py` is **the only entrypoint of the four that does not disable
JAX's GPU preallocation**:

| Entrypoint | Sets `XLA_PYTHON_CLIENT_PREALLOCATE=false`? |
| --- | --- |
| `scripts/train_flash_sac.py:34` | ✅ yes |
| `scripts/evaluate_policy.py:12` | ✅ yes |
| `scripts/view_policy.py:11` | ✅ yes |
| **`scripts/train_rl.py` (PPO)** | ❌ **no** |

By default JAX reserves ~75 % of the device, leaving too little for Warp/MuJoCo's
CUDA graph capture. Measured directly:

| World count | Without the env var | With `XLA_PYTHON_CLIENT_PREALLOCATE=false` |
| ---: | --- | --- |
| 32 | ✅ completes 1 iteration | ✅ completes |
| 1024 | ❌ **OOM** — `RESOURCE_EXHAUSTED`, `Graph creation error` | ✅ completes, peak 5,437 MiB |

All three scripts use `os.environ.setdefault`, so supplying the variable
externally is the intended escape hatch. We pass it with
`docker exec -e XLA_PYTHON_CLIENT_PREALLOCATE=false`, which changes no file.

This looks like an upstream oversight rather than a deliberate choice, since the
sibling trainer sets it. It is invisible on a 48 GB GPU and fatal on 8 GB above
~512 worlds. **[VERIFIED]** that it matters here; **[UNRESOLVED]** whether NVIDIA
intends it.

---

## 4. Training smoke test **[VERIFIED]**

```
training progress: 0/1,536 environment steps (0.0%); checkpoints: /workspace/outputs/g1_ppo_smoke
training progress: 768/1,536 environment steps (50.0%); 43 env steps/s; elapsed 00:00:18; ETA 00:00:18
training progress: 1,536/1,536 environment steps (100.0%); 71 env steps/s; elapsed 00:00:22; ETA 00:00:00
completed 2 iterations; checkpoints: /workspace/outputs/g1_ppo_smoke
```

| Metric | Value |
| --- | --- |
| Iterations completed | **2 / 2** |
| Environment steps | **1,536** (= 2 × 32 worlds × 24 rollout steps) |
| Wall time (incl. compile) | **45 s** |
| Steady-state throughput | 71 env steps/s |
| **Peak GPU memory** | **1,966 MiB of 8,151 (24.1 %)** |
| Exit code | **0** |
| CUDA OOM / NaN / crash | **none** |

Warnings only (all benign and also present in the verified replay): Newton
inertia correction on 2 + 64 bodies, an onnxruntime note about shape-massaging
nodes on CPU, and `Unable to initialize backend 'tpu'`. One new line appeared
that the replay did not show — `No registered plugin EP device found for
'CUDAExecutionProvider' with device_id=0` — emitted while JAX holds the CUDA
context; training completed normally regardless. **[UNRESOLVED]** whether SONIC
silently fell back to CPU execution here; it does not affect this feasibility
verdict but is worth checking before any long run.

### 4.1 Checkpoint authenticity **[VERIFIED]**

Read back through NVIDIA's own `read_checkpoint_metadata` +
`checkpoint_training_algorithm`:

```
metadata keys: ['environment_steps', 'iteration', 'policy_schema_json',
                'resolved_config_json', 'voc_scale']
  iteration = 2
  environment_steps = 1536
training algorithm: ppo
training._target_: flash_chord.training.ppo.config.TrainingConfig
scene.world_count: 32
embodiment._target_: flash_chord.embodiments.g1_dex3.G1Dex3
task.parquet: /v2d/assets/.../snack_box_pick_and_place_01/robot_name=g1/data.parquet
env.voc_scale (as trained): 1.0
```

Both `model_1.safetensors` and `model_2.safetensors` are 28.67 MB and carry the
full resolved experiment config plus a policy schema in the safetensors header —
which is exactly what [FLASHCHORD_BASELINE_SMOKE.md](FLASHCHORD_BASELINE_SMOKE.md)
found no file in the checkout could supply. **The "no compatible checkpoint"
blocker from the previous milestone is now mechanically resolved**, though not
usefully so until real training happens.

---

## 5. Official evaluation at `world_count=1`, unassisted **[VERIFIED]**

`scripts/evaluate_policy.py` ran to completion over all 800 reference frames:
**18 s**, peak **1,867 MiB**, exit 0.

### 5.1 Assistance really was off

`apply_evaluation_composition` was applied to our checkpoint's own embedded
config and the result read back:

| Setting | As trained | As evaluated |
| --- | ---: | ---: |
| `env.voc_scale` | 1.0 | **0.0** |
| `env.voc.max_force` | 60.0 N | **0.0** |
| `env.voc.max_torque` | 60.0 N·m | **0.0** |
| `env.auto_reset` | true | false |
| `scene.world_count` | 32 | 1 |

`UNASSISTED=YES`. No Virtual Rigid Object Control, no gravity compensation on the
box — unlike the replay in the previous milestone.

### 5.2 Metrics (2-iteration checkpoint — a floor, not a baseline)

```json
{
  "evaluation/mppe_cm": 60.48383521018228,
  "evaluation/mean_per_point_error_cm": 60.48383521018229,
  "evaluation/object_position_error_m": 0.6043218788919152,
  "evaluation/object_orientation_error_deg": 62.24026974239888,
  "evaluation/mean_add_m": 0.619963470255388,
  "evaluation/add_auc": 0.563984375,
  "evaluation/add_std_m": 0.7977535684570366,
  "evaluation/chord_sr": 0.0,
  "evaluation/maniptrans_sr": 0.0,
  "evaluation/spider_sr_uncentered": 0.0,
  "evaluation/spider_position_error_centered_m": 0.7395273012011327,
  "evaluation/relative_position_error_cm": 0.0,
  "evaluation/cause_fraction/object": 1.0,
  "evaluation/cause_fraction/wrist": 1.0,
  "evaluation/environment_steps": 1536.0,
  "evaluation/world_count": 1.0
}
```

Interpretation: **every success rate is 0.0**, and `cause_fraction/object` and
`cause_fraction/wrist` are both **1.0** — the single world failed on both object
and wrist tracking. MPPE 60.48 cm means the box ends up roughly 0.6 m from where
it should be. After 1,536 environment steps that is the expected outcome; the
actor output gain starts at 0.0 and the policy has effectively not learned
anything. **This number's only use is as a sanity floor.** **[VERIFIED]**

Two notes on reading these numbers:

- **The JSON report nests everything under a `metrics` key and drops the
  `evaluation/` prefix** that the stdout scalars carry, alongside provenance
  fields (`checkpoint_sha256`, `parquet`, `reset_mode`, thresholds). Scripts
  should read `report["metrics"]["mppe_cm"]`, not `report["evaluation/mppe_cm"]`.
- **Re-running is not bit-exact.** A repeat of the same train+evaluate pair gave
  MPPE 60.41 cm against 60.48 cm here — about 0.1 % drift, from GPU reduction
  ordering rather than configuration (`reset.seed=42` and the action's
  `body_delay.seed=0` are both fixed). Irrelevant at this error scale; worth
  knowing before treating any single digit as reproducible. **[VERIFIED]**

---

## 6. How far 8 GB actually goes **[VERIFIED]**

One PPO iteration per world count, kernels warm, `XLA_PYTHON_CLIENT_PREALLOCATE=false`:

| `scene.world_count` | Peak GPU (MiB) | % of 8,151 | Throughput (env steps/s) | Result |
| ---: | ---: | ---: | ---: | --- |
| 32 | 1,966 | 24 % | 71 | ✅ |
| 128 | 2,277 | 28 % | 147 | ✅ |
| 512 | 3,637 | 45 % | 518 | ✅ |
| **1024** | **5,437** | **67 %** | **899** | ✅ **best throughput** |
| 1536 | 7,517 | 92 % | 311 | ⚠️ runs, but throughput *collapses* |
| 2048 | — | — | — | ❌ **OOM** |

Memory scales almost linearly at ≈3.4 MiB per world above a ≈1.9 GB floor. The
2,048-world failure is clean and reproducible:

```
Warp CUDA error 2: out of memory (in function wp_cuda_graph_create_exec,
  /builds/omniverse/warp/warp/native/warp.cu:3430)
RuntimeError: Graph creation error: Warp CUDA error 2: out of memory
```

**1,024 worlds is the sweet spot**: 67 % of VRAM and peak throughput. 1,536 fits
but throughput drops 65 % (899 → 311 steps/s) — memory pressure costs more than
the extra parallelism gains, so there is no reason to go above 1,024 here.

NVIDIA's stated requirement for full training is **48 GiB with ≥46,000 MiB
free**, validated in their EVT-07. **[DOC]** Our measurements are consistent with
that: 4,096 worlds would need roughly 1.9 GB + 4096 × 3.4 MiB ≈ **16 GB** for the
physics state alone, before PPO's own buffers and any safety margin.

---

## 7. Why 32 GB of host RAM does not rescue this **[VERIFIED]**

A reasonable question, and the answer is no — for two independent reasons, the
second of which is decisive.

### 7.1 Offloading is not available, and would not help if it were

- **FlashCHORD exposes no offload knob.** A search of `src/` for
  `mempool|managed_memory|unified memory|offload|pinned` returns nothing
  relevant. There is no configuration path to place physics state in host memory.
- **The OOM is in CUDA graph creation**, not in a tensor allocation:
  `wp_cuda_graph_create_exec`. A captured CUDA graph requires its buffers
  resident in device memory for the whole launch; you cannot page them to the
  host mid-graph. Warp's memory pool is already enabled on this device.
- **The access direction is wrong for zero-copy.** This device reports
  `is_cpu_memory_access_from_gpu_supported: True` but
  `is_gpu_memory_access_from_cpu_supported: False`. Even using pinned host memory
  the physics state is touched every one of 4 substeps at 200 Hz, so each access
  becomes a PCIe round trip — orders of magnitude slower, not marginally.
- **Memory pressure already hurts before any offloading.** Throughput fell from
  899 to 311 env steps/s going from 1,024 to 1,536 worlds, entirely inside VRAM.
  Spilling to host RAM would extend that curve in the wrong direction.

### 7.2 VRAM is not the binding constraint — wall-clock is

This is the part that settles it. Using our **best measured** throughput:

| Target | Environment steps | At 899 steps/s (1024 worlds) |
| --- | ---: | ---: |
| **PPO terminal** (`g1_recon_body_ppo`, 20000 iters) | 1,966,080,000 | **25.3 days** |
| FlashSAC G1 terminal (README) | 349,999,104 | **4.5 days** |

At 512 worlds the PPO figure becomes 43.9 days; at 1,536 worlds, 73.2 days.

So even a hypothetical perfect offloading scheme that let us run 4,096 worlds on
8 GB would not deliver a competitive policy here — it would raise throughput at
best to the point where the run still takes weeks of uninterrupted laptop GPU
time. **Fitting more worlds into memory is not the lever; total compute is.**
**[VERIFIED]** arithmetic from measured rates.

The practical consequences: a real baseline needs either a larger GPU (NVIDIA's
48 GiB target), a rented cloud GPU, or a deliberately reduced training budget
whose result we would have to report honestly as under-trained. **[GAP]**

---

## 8. Errors encountered and fixes

| # | Symptom | Cause | Fix | Upstream touched? |
| --- | --- | --- | --- | --- |
| 1 | OOM at 1024 worlds: `RESOURCE_EXHAUSTED`, `Graph creation error` | `scripts/train_rl.py` does not set `XLA_PYTHON_CLIENT_PREALLOCATE=false`, unlike the other three entrypoints; JAX reserves ~75 % of the device | Pass the variable via `docker exec -e` (the scripts use `setdefault`, so an external value wins) | No |
| 2 | OOM at 2048 worlds in `wp_cuda_graph_create_exec` | Genuine capacity limit of an 8 GB device | None — recorded as the measured ceiling; stay at ≤1024 | No |
| 3 | Throughput collapse at 1536 worlds (899 → 311 steps/s) | VRAM pressure at 92 % occupancy | Use 1024 worlds | No |

No error required modifying NVIDIA's source or configuration. Both fixes are a
CLI/environment override.

**Writes stayed outside tracked source.** Checkpoints went to
`/workspace/outputs/`, which NVIDIA's own `.gitignore:23` excludes (`logs/` on
line 22 likewise). `git -C official/video_to_data status --short` is empty after
every run. **[VERIFIED]**

---

## 9. Housekeeping

- Probe checkpoints from the six world-count runs were deleted; only
  `outputs/g1_ppo_smoke/` (55 MB) is kept. Disk: **30 GB free**, above the 15 GB
  floor. **[VERIFIED]**
- No Docker rebuild, no downloads. The container was the one already running.
- Nothing committed or pushed.

---

## 10. Remaining blockers

1. **No competitive checkpoint.** The pipeline works; the compute does not exist
   locally. 25.3 days for the PPO schedule on this GPU. **[GAP]**
2. **Our 30 challenge episodes are still not wired into FlashCHORD.** This used
   NVIDIA's snack-box parquet, not our adapter output. **[GAP]**
3. **SONIC execution provider under JAX is unconfirmed.** The
   `No registered plugin EP device found for 'CUDAExecutionProvider'` warning
   appears only in the training path. Worth resolving before committing to a long
   run, since a CPU fallback would throttle throughput. **[UNRESOLVED]**

---

## 11. Commands you can run yourself

### 11.0 Everything at once

```bash
cd /home/ibrahim/v2d-track2-robotic-grounding
bash scripts/verify_flashchord_training.sh          # ~2 min, includes train + evaluate
bash scripts/verify_flashchord_training.sh --fast   # ~20 s, checks existing artifacts only
```

Exit 0 and `ALL CHECKS PASSED` means every claim here still holds.

### 11.1 Validate the PPO config without training

```bash
G1_PARQUET=/v2d/assets/human_motion_data/whole_body/soma/sequence_id=2026-03-06_10-24-18_snack_box_pick_and_place_01/robot_name=g1/data.parquet

docker exec -w /workspace flash_chord-latest-gpu0 python scripts/train_rl.py \
  experiment=g1_recon_body_ppo "task.parquet='${G1_PARQUET}'" \
  scene.world_count=32 training.max_iterations=2 training.save_interval=1 \
  logging.mode=disabled reset.seed=42 output_dir=outputs/g1_ppo_smoke \
  --cfg job --resolve | grep -E "world_count|max_iterations|save_interval|voc_scale"
```

Note the inner single quotes around the parquet path — its Hive `key=value`
partitions otherwise break Hydra's override grammar.

### 11.2 Run the 2-iteration training smoke test

```bash
G1_PARQUET=/v2d/assets/human_motion_data/whole_body/soma/sequence_id=2026-03-06_10-24-18_snack_box_pick_and_place_01/robot_name=g1/data.parquet

docker exec -w /workspace -e XLA_PYTHON_CLIENT_PREALLOCATE=false \
  flash_chord-latest-gpu0 python scripts/train_rl.py \
  experiment=g1_recon_body_ppo "task.parquet='${G1_PARQUET}'" \
  scene.world_count=32 training.max_iterations=2 training.save_interval=1 \
  logging.mode=disabled reset.seed=42 output_dir=outputs/g1_ppo_smoke
# expect: "completed 2 iterations; checkpoints: /workspace/outputs/g1_ppo_smoke"
```

The `-e XLA_PYTHON_CLIENT_PREALLOCATE=false` is required above ~512 worlds (§3).

### 11.3 Confirm the checkpoint is a real PPO checkpoint

```bash
ls -l official/video_to_data/robotic_grounding/flash_chord/outputs/g1_ppo_smoke/
# model_1.safetensors, model_2.safetensors (28.67 MB each), model_2...metrics.json

docker exec -w /workspace flash_chord-latest-gpu0 python -c "
from flash_chord.training.checkpoint import read_checkpoint_metadata
from flash_chord.evaluation.harness import checkpoint_config, checkpoint_training_algorithm
m = read_checkpoint_metadata('outputs/g1_ppo_smoke/model_2.safetensors')
c = checkpoint_config(m)
print('algorithm:', checkpoint_training_algorithm(c))
print('iteration:', m['iteration'], 'env steps:', m['environment_steps'])
"
# expect: algorithm: ppo / iteration: 2 / env steps: 1536
```

### 11.4 Evaluate it, unassisted, on one world

```bash
docker exec -w /workspace -e XLA_PYTHON_CLIENT_PREALLOCATE=false \
  flash_chord-latest-gpu0 python scripts/evaluate_policy.py \
  "evaluation.checkpoint=outputs/g1_ppo_smoke/model_2.safetensors" \
  evaluation.world_count=1 \
  "evaluation.metrics_output=outputs/g1_ppo_smoke/model_2.safetensors.metrics.json" \
  logging.mode=disabled
# expect 800/800 evaluation steps, then mppe_cm ~60.5 and all *_sr = 0.0
```

### 11.5 Prove the evaluation removed object assistance

```bash
docker exec -w /workspace flash_chord-latest-gpu0 python -c "
from flash_chord.training.checkpoint import read_checkpoint_metadata
from flash_chord.evaluation.harness import apply_evaluation_composition, checkpoint_config
c = checkpoint_config(read_checkpoint_metadata('outputs/g1_ppo_smoke/model_2.safetensors'))
print('trained  voc_scale =', c['env']['voc_scale'])
apply_evaluation_composition(c, world_count=1, reset_mode='explicit')
print('evaluated voc_scale =', c['env']['voc_scale'],
      'max_force =', c['env']['voc']['max_force'])
"
# expect: trained 1.0 -> evaluated 0.0, max_force 0.0
```

### 11.6 Reproduce the memory ceiling (optional, ~1 min each)

```bash
G1_PARQUET=/v2d/assets/human_motion_data/whole_body/soma/sequence_id=2026-03-06_10-24-18_snack_box_pick_and_place_01/robot_name=g1/data.parquet

for W in 1024 2048; do
  echo "--- world_count=$W ---"
  docker exec -w /workspace -e XLA_PYTHON_CLIENT_PREALLOCATE=false \
    flash_chord-latest-gpu0 python scripts/train_rl.py \
    experiment=g1_recon_body_ppo "task.parquet='${G1_PARQUET}'" \
    scene.world_count=$W training.max_iterations=1 training.save_interval=1000 \
    logging.mode=disabled reset.seed=42 output_dir=outputs/g1_ppo_probe_$W 2>&1 \
    | grep -E "completed|out of memory" | tail -2
done
# expect: 1024 completes; 2048 prints "Warp CUDA error 2: out of memory"
# clean up afterwards:
rm -rf official/video_to_data/robotic_grounding/flash_chord/outputs/g1_ppo_probe_*
```

### 11.7 Watch GPU memory during any run

```bash
nvidia-smi --query-gpu=memory.used,memory.total --format=csv -l 2
# training at 32 worlds peaks near 1,966 MiB; at 1024 worlds near 5,437 MiB
```

### 11.8 Confirm upstream is untouched

```bash
git -C official/video_to_data status --short    # expect empty
git -C official/video_to_data log --oneline -1  # expect 193382a
```
