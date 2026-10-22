# A100 80 GB — compatibility and benchmark plan (read-only)

A plan, not a result. Nothing in this document has been run: there is no A100 on
this machine, so every number marked **[ESTIMATE]** is arithmetic from our
measured RTX 5070 Laptop throughput and must be replaced by a measurement before
any multi-day commitment is made.

Companion: [STEP_D_FLASHSAC_FEASIBILITY.md](STEP_D_FLASHSAC_FEASIBILITY.md) ·
[STEP_D_TRANSFER_SMOKE.md](STEP_D_TRANSFER_SMOKE.md).

---

## 1. Why an A100 changes the problem

On 8 GiB we cannot run NVIDIA's shipped recipe at all. Two hard limits, both
measured:

| | shipped | our 8 GiB ceiling | A100 80 GB |
| --- | --- | --- | --- |
| `replay.capacity` | 10,000,000 = **20.92 GiB** | 1,000,000 (2.09 GiB) | **fits with ~59 GiB spare** |
| `scene.world_count` | 2048 | 1024 (2048 OOMs in `wp_cuda_graph_create_exec`) | **2048+ expected** |

So the A100 is not merely "faster" — it is the first configuration where the
**published recipe runs unmodified**, which is what makes any result comparable
to NVIDIA's own. Everything we produce on the laptop carries an asterisk
(10–20× smaller replay, half the worlds); A100 results would not.

VRAM budget at the shipped recipe, from our measured per-transition cost of
2,246 B (float16, obs 486 / action 43 / critic ctx 10 / objective terms 12):

| component | bytes |
| --- | --- |
| replay ring, 10,000,000 transitions | 20.92 GiB |
| Warp scene @ 2048 worlds | ~6.9 GiB **[ESTIMATE]**, extrapolated from 1024 → 6,271 MiB delta |
| JAX executables, params, optimizer | ~1.5 GiB **[ESTIMATE]** |
| **total** | **~29 GiB of 80** |

Comfortable. The A100 could instead run 4096 worlds, or several episodes
concurrently on one card — see §4.

---

## 2. Compatibility checks to run first, in order

Each is cheap and each one, if it fails, invalidates everything after it. Run
them before any training.

| # | check | command | pass condition |
| --- | --- | --- | --- |
| 1 | image runs at all | `docker run --rm --gpus all flash_chord:latest python -c "import warp, jax; print(jax.devices())"` | prints one `cuda:0` device |
| 2 | **sm_80 kernels compile** | same as above, then `nvidia-smi --query-gpu=compute_cap --format=csv` | Warp reports `sm_80`; our card is `sm_120`, so **every kernel recompiles** — the cache is not portable |
| 3 | scene builds at 2048 | `scripts/probe_flash_sac_memory.py --world-count 2048` | prints the dims; no OOM |
| 4 | replay math at shipped capacity | same probe | `10,000,000 float16` row shows **"yes"** at 80 GiB |
| 5 | reference loads | `scripts/check_reference_in_flashchord.py` on 2–3 episodes | `load_pass`, support counts agree |
| 6 | 2-minute training smoke | `train_flash_sac.py ... total_environment_steps=<2048*8>` | terminal checkpoint written |
| 7 | checkpoint portability | §3 | our laptop checkpoint evaluates on A100 |

**Check 2 is the one most likely to surprise.** The Warp kernel cache is keyed
by architecture, so the ~130 s cold compile we measured will be paid again on
the A100, and the host cache we bind-mount must be a *separate directory per
architecture* or the two will thrash. Budget one cold compile per node.

---

## 3. Checkpoint portability

FlashSAC checkpoints are **safetensors with no pickle** — an 8-byte header
length, a JSON header with a pytree `manifest`, and `user.*` string metadata.
There is nothing architecture- or CUDA-specific in the file. Portability is
therefore expected to be total, but it is cheap to prove:

```bash
# on the A100, against a checkpoint produced on the laptop
python3 scripts/read_checkpoint_metadata.py <laptop>/policy_65536.safetensors
bash scripts/evaluate_step_d.sh /out/<laptop>/policy_65536.safetensors portability 1 21
```

Pass condition: the metrics match the laptop's numbers to within GPU reduction
noise (our Episode 0 regression measured 7.6e-06 m worst-case drift for the same
class of comparison).

**Three portability constraints that are NOT about hardware** and will bite on a
cluster:

1. **Baked absolute paths.** `soma_to_g1.py` writes `object_mesh_paths` as
   absolute `/work/adapter_out/...`, and `task.parquet` is baked into every
   checkpoint's metadata. The cluster **must mount the staged tree at the same
   `/work/adapter_out`**, or `evaluation.source_root` must be used — and note
   `_resolve_parquet` only rebases paths that are relative or under
   `/workspace`, so an absolute `/work/...` path is *not* remapped by it.
2. **World-count-dependent state.** `RewardNormalizerState.discounted_return`
   and `ExplorationState.noise` are per-world arrays. A `state_*.safetensors`
   written at 1024 worlds cannot be resumed at 2048. Policy-only checkpoints are
   world-count independent and port freely.
3. **Transfer checkpoints must be rebuilt per world count**, for the same
   reason — `transfer_flash_sac_policy.py --world-count` must equal the
   consuming run's.

---

## 4. Benchmark plan on the A100

Same structure as the laptop benchmark in `outputs/step_D/bench/`, extended
upward, with the **update ratio held constant** so configurations are
comparable:

```
updates_per_collection = 4.0 * world_count / 2048     # == 0.001953 updates/env-step
```

| run | worlds | upd/coll | steps | purpose |
| --- | --- | --- | --- | --- |
| A1 | 1024 | 2.0 | 307,200 | direct comparison to our laptop number |
| A2 | 2048 | 4.0 | 307,200 | **the shipped recipe's world count** |
| A3 | 4096 | 8.0 | 307,200 | does throughput still scale? |
| A4 | 2048 | 4.0 | 307,200 | repeat of A2 with `replay.capacity=10000000` — isolates the replay's cost |

Record per run, exactly as the laptop benchmark does: instantaneous throughput
(differenced from the log — **never** the progress bar's cumulative average),
peak VRAM, wall clock, and updates per environment step.

**Decision rule:** pick the largest world count whose instantaneous throughput
is still rising. On the laptop, throughput *collapsed* above the ceiling rather
than degrading gracefully (PPO: 1024 → 899 steps/s, 1536 → 311), so the knee is
sharp and worth finding before committing.

### What the numbers would have to be

Our measured laptop rate is the anchor. The full recipe is 349,999,104
environment steps per episode:

| A100 throughput | h/episode | 30 episodes (one tier) | 90 policies (3 tiers) |
| --- | --- | --- | --- |
| 5,000 steps/s | 19.4 | 583 GPU-h | 1,750 GPU-h |
| 10,000 steps/s | 9.7 | 292 GPU-h | 875 GPU-h |
| 20,000 steps/s | 4.9 | 146 GPU-h | 437 GPU-h |
| 30,000 steps/s | 3.2 | 97 GPU-h | 292 GPU-h |

All **[ESTIMATE]**. The honest planning range is **300–900 A100-hours** for a
full three-tier submission, and the first benchmark run collapses it to one
number. Note this is for independent per-episode policies; if transfer works,
the source-policy cost amortizes across episodes and the total drops — which is
exactly what the Stage 2 experiment is for.

---

## 5. Reproducible cluster training

### Resume, and why it matters more on a cluster

Preemption is the normal case on shared clusters, and FlashSAC resume is
well-behaved **as long as the world count does not change**:

- `training.resume=true training.checkpoint_path=<state_N.safetensors>`
- `environment_steps` restores, so the curriculum stage resumes correctly and
  `total_environment_steps` is an absolute budget, not a per-job one
- the runner refuses a resume whose remaining budget is not divisible by
  `world_count`, and refuses one already past the budget — both are good
- the replay does **not** restore unless `checkpoint.load_replay_path` is set,
  so a preempted job loses its replay. For long runs, set
  `checkpoint.save_replay=true` and pass `load_replay_path` on resume, at a cost
  of `capacity * 2,246 B` per checkpoint (**21 GiB** at the shipped capacity —
  budget the disk, or accept the replay refill)

### Job shape

One episode per job, `save_interval_environment_steps` set so a preemption loses
at most ~20 minutes:

```
save_interval = round(throughput * 1200)     # ~20 min of steps
```

Write checkpoints to `outputs/…` only — NVIDIA's own `.gitignore` excludes
`outputs/` and `logs/`, so upstream `git status` stays clean (`CLAUDE.md`).

### Determinism and provenance

`reset.seed` is the single seed. For the A/B transfer comparison, **run ≥3 seeds
per arm**: our smoke test showed a 2× MPPE swing between arms, and with no
seed replication we cannot distinguish a transfer effect from seed noise. This
is the single biggest methodological gap in the current plan.

Every trajectory export already carries `checkpoint_sha256`,
`eval_code_sha256`, `eval_script_sha256` and `reference_parquet` in its parquet
metadata, so a cluster result can be traced to the exact weights and code that
produced it without trusting the directory name.

---

## 6. What this plan deliberately does not claim

- No A100 has been tested. Checks in §2 are proposed, not passed. **[GAP]**
- Throughput scaling from 1024 → 2048 → 4096 worlds is assumed monotonic up to
  a knee; the laptop shows the knee can be sharp and *downward*. **[GAP]**
- The 300–900 GPU-hour range is arithmetic on our hardware's rate, not a
  measurement on theirs. **[ESTIMATE]**
- Whether transfer reduces the total budget is unknown — that is the open
  experimental question, and the smoke test's only finding so far was negative
  (see [STEP_D_TRANSFER_SMOKE.md](STEP_D_TRANSFER_SMOKE.md) §4). **[GAP]**
