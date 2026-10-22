# Step D, Stage 1 — is FlashSAC cross-episode transfer feasible here?

Feasibility findings for the proposed experiment: train Episode 0, transfer its
policy to Episode 21, fine-tune, and compare against training Episode 21 from
scratch. Everything below was measured on this machine (RTX 5070 Laptop, 8 GiB)
against our own Step C references, or read out of NVIDIA's code at `193382a`.

Companion documents: [TIER_REFERENCES.md](TIER_REFERENCES.md) (the 30 references
this builds on) · [FLASHCHORD_TRAINING_FEASIBILITY.md](FLASHCHORD_TRAINING_FEASIBILITY.md)
(the PPO path) · [FLASHCHORD_BASELINE_SMOKE.md](FLASHCHORD_BASELINE_SMOKE.md).

Conclusions are marked **[VERIFIED]** (observed directly), **[DOC]** (stated by
NVIDIA, not independently checked) or **[GAP]/[UNRESOLVED]**.

---

## 1. The recipe, and what it asks for

`experiment=g1_recon_body_flash_sac` is the canonical G1+Dex3/SONIC FlashSAC
recipe, the direct analogue of the `g1_recon_body_ppo` recipe we already
validated end-to-end. It selects `sim/observation/objective/termination/reset:
recon_body` and `action: sonic_joint_residual` — the action config G1 requires
(`ReplayConfig()`'s floating-wrist default rejects G1; see `CLAUDE.md`).

| | shipped value | runnable on 8 GiB? |
| --- | --- | --- |
| `scene.world_count` | 2048 | no — PPO already OOM'd at 2048 |
| `training.replay.capacity` | 10,000,000 | **no — 20.92 GiB, 2.6× the whole card** |
| `training.replay.observation_storage_dtype` | float16 | yes |
| `training.replay.minimum_size` | 409,600 | yes, but see §3 |
| `training.total_environment_steps` | 349,999,104 | see §5 |
| `training.updates_per_collection` | 4.0 | yes, but see §4 |

**[VERIFIED]** — `scripts/probe_flash_sac_memory.py`, run against Episode 0.

---

## 2. The replay buffer is the binding constraint, not the world count

`replay.create_replay` allocates the ring with `jnp.zeros`, so it lives in VRAM
alongside Warp's scene and the JAX executables. NVIDIA ships an exact byte
accounting (`replay.replay_memory`), which our probe calls rather than inventing
its own arithmetic.

Measured dimensions for this recipe on our Episode 0 reference:

| | value |
| --- | --- |
| `observation_dim` | **486** |
| `action_dim` | **43** |
| `critic_context_dim` | **10** |
| `objective_term_count` | **12** |
| bytes per transition (float16 obs) | **2,246** |
| bytes per transition (float32 obs) | 4,190 |

Ring size at float16:

| capacity | ring | verdict |
| --- | --- | --- |
| 10,000,000 (shipped) | **20.92 GiB** | impossible |
| 4,000,000 | 8.37 GiB | impossible |
| 2,000,000 | 4.18 GiB | no headroom for scene + executables |
| **1,000,000** | **2.09 GiB** | **workable ceiling** |
| 500,000 | 1.05 GiB | comfortable |
| 250,000 | 0.52 GiB | used for the smoke test |

So the single largest deviation we must make from NVIDIA's recipe is a **10×
reduction in replay capacity**. That is a change to the algorithm's memory of its
own past, not a cosmetic knob, and it must be stated in any result we report.
**[VERIFIED]**

---

## 3. `minimum_size` interacts with the world count

Learning does not start until the replay holds `minimum_size` transitions. The
shipped 409,600 at 2048 worlds is 200 collections; at 256 worlds it is 1,600
collections of pure, un-learned exploration. For the smoke test we used 4,096.
For real runs this has to be set deliberately — it is a warmup length, and
leaving it at the shipped value while cutting the world count silently changes
how long the policy wanders before any gradient is taken. **[VERIFIED]**

---

## 4. Cutting the world count changes the replay ratio

```python
collection_count       = total_environment_steps // world_count
total_optimizer_updates = collection_count * updates_per_collection
```
(`flash_sac/config.py:359-366`)

Gradient updates per environment step are therefore
`updates_per_collection / world_count`. Holding `updates_per_collection = 4.0`
while dropping 2048 → 256 worlds multiplies the optimizer work per environment
step by **8×**:

| | world_count | upd/coll | optimizer updates for 350M steps | updates per env step |
| --- | --- | --- | --- | --- |
| NVIDIA reference | 2048 | 4.0 | 683,592 | 0.00195 |
| ours, naive | 256 | 4.0 | 5,468,736 | 0.0156 |
| ours, ratio-matched | 256 | **0.5** | 683,592 | 0.00195 |

This matters twice. For **absolute** quality claims, the ratio-matched setting is
the honest comparison to NVIDIA's recipe. For the **A-vs-B transfer comparison**,
what matters is that both arms use identical settings — the confound is only
fatal if the arms differ. **[VERIFIED]**

---

## 5. Throughput, measured

Smoke test: Episode 0, 256 worlds, replay capacity 250,000, `minimum_size`
4,096, 65,536 environment steps.

| | value |
| --- | --- |
| Warp kernel compile (cold cache) | ~130 s |
| JAX executable compile | **56.9 s** |
| wall clock for 65,536 steps | **77 s** |
| rate as the progress bar reports it | 852 env steps/s |
| **true steady-state rate** | **~2,900–3,450 env steps/s** |

**The progress bar's number is a cumulative average** (`total / elapsed`,
including a fixed startup), which is why it climbs monotonically from 13 to 852
and never plateaus. Differencing the last half of the run gives 2,909 env
steps/s; the earliest ≥8 s window gives 3,456. For budgeting, use **~3,000 env
steps/s at 256 worlds**.

For scale, PPO's best measured rate was **899 env steps/s at 1024 worlds**
(`FLASHCHORD_TRAINING_FEASIBILITY.md`). FlashSAC at a quarter of the worlds is
already **3.3× faster**. **[VERIFIED]**

### Kernel cache

`Kernel cache: /root/.cache/warp/1.14.0` is inside the container, so every
`docker run` recompiles from scratch (~130 s, one module alone took 25.8 s).
Bind-mounting a host directory there removes that from every subsequent run. The
CPU saturation visible during this phase is the CUDA toolchain compiling, not
training — the GPU is nearly idle until `compiled FlashSAC JAX executables`
appears. **[VERIFIED]**

---

## 6. Checkpoint anatomy, and what "transfer" would actually carry

`save_learner_checkpoint` writes **three separate files** per boundary:

| file | contents | size (measured) |
| --- | --- | --- |
| `policy_<step>.safetensors` | `ActorInferenceState` — **`params` + `batch_stats` only**, documented as "world-count-independent" | **4.6 MiB** |
| `state_<step>.safetensors` | `TrainingCheckpoint` — actor, critic, target critic, temperature, reward normalizer, exploration, loss scale, RNG key, `update_credit`, **`environment_steps`**, `global_update_step` | **152 MiB** |
| `replay_<step>.safetensors` | the ring — only when `save_replay=true` (default **false**) | not written |

Format is safetensors, **no pickle**: an 8-byte length, a JSON header with a
`manifest` of pytree-flattened leaves, and `user.*` string metadata. That makes
it inspectable from the host with the standard library —
`scripts/read_checkpoint_metadata.py`. **[VERIFIED]**

### What `resume` restores

`restore_learner_checkpoint` (`flash_sac/checkpoint.py`):

| state | carried on resume? | task-specific? |
| --- | --- | --- |
| actor params / batch stats | yes | the thing we want |
| critic + target critic | yes | partly — value scale is task-specific |
| **`environment_steps`** | **yes** | **yes — drives the curriculum stage and the budget** |
| `global_update_step`, LR schedule position | yes (unless `load_optimizer=false`) | yes |
| Adam optimizer state | yes (unless `load_optimizer=false`) | yes |
| reward normalizer running moments | yes (unless `load_reward_normalizer=false`); only `discounted_return` is zeroed | **yes — Episode 0's reward scale** |
| exploration state (temperature, sigma) | yes | yes |
| **replay buffer** | **no, unless `load_replay_path` is set** (default `null`) | — |

So the good news first: **stale replay data cannot leak across episodes by
accident.** It is opt-in via an explicit path. **[VERIFIED]**

The bad news is everything else in that table. A plain
`training.resume=true training.checkpoint_path=<ep0 state>` pointed at Episode
21 is a **resume, not a transfer**: it would start Episode 21 at Episode 0's
environment-step count, hence at the wrong curriculum stage and with that much
of the budget already consumed, carrying Episode 0's reward normalization and
optimizer moments.

### The guard that does not guard

`runner.py:470` calls `validate_resume_reference_config`, which looks like it
would stop exactly that mistake. It does not. It compares **only** the object
articulation contract:

```
object_articulations  []
```

Our Episode 0 checkpoint's own metadata, read back from the file. Every
rigid-object episode in the tier has an **empty** articulation list, so the
contract is `{"version": …, "entries": []}` for all of them and compares equal.
**The guard cannot tell two rigid-object episodes apart and will happily let
Episode 0's learner state resume onto Episode 21.** The baked
`task.parquet` is recorded in the metadata but is *not* compared.
**[VERIFIED]**

### There is no supported weight-only warm start

`training.checkpoint_path` is fed to `restore_learner_checkpoint`, which calls
`_validate_metadata(path, "training_state")` and **rejects a `policy` checkpoint
outright**. So the one file that contains exactly what we want to transfer — the
4.6 MiB actor — cannot be used to initialize training through any config knob
upstream provides. **[VERIFIED]**

---

## 7. Three ways to do the transfer, and which to pick

| | approach | carries | verdict |
| --- | --- | --- | --- |
| **A** | `resume=true` with Episode 0's `state_*.safetensors` | weights **+ step counter + curriculum position + optimizer + reward normalizer** | **scientifically invalid** — confounds "transfer" with "started mid-curriculum" |
| **B** | A, plus `load_optimizer=false load_reward_normalizer=false` | weights + target critic + exploration + **still the step counter** | better, but the step counter alone still misplaces the curriculum |
| **C** | **checkpoint surgery**: build a fresh Episode 21 learner, copy in only Episode 0's actor (and optionally critic) weights, reset everything else, write a valid `training_state` file | **weights only** | **the honest experiment** |

**C is the recommendation.** It is implementable without touching upstream: a
script in our `scripts/` that imports NVIDIA's own `training_checkpoint`,
`load_checkpoint` and `save_checkpoint`, instantiates the Episode 21 learner to
get the correct pytree template, replaces only the weight leaves, and re-emits
metadata describing Episode 21. Because the policy/critic schemas are identical
across episodes (486/43/10/12 for every episode in the tier), `_validate_resume_schemas`
will pass legitimately rather than by accident.

Two variants worth separating in the results, since they answer different
questions:
- **C-actor** — transfer the actor only; critic re-learns Episode 21's value
  scale from scratch. The conservative, more defensible claim.
- **C-actor+critic** — transfer both. Faster if the value functions are
  compatible, riskier if Episode 21's reward scale differs.

**Zero-shot** is simpler: it needs no surgery, only `evaluate_policy.py` with
Episode 0's **policy** checkpoint. But note `use_checkpoint_config: true` rebuilds
the environment from the checkpoint's own baked config, and `source_root` remaps
only a path *root*, not the `sequence_id` — so pointing Episode 0's policy at
Episode 21's parquet needs an explicit override, which must be verified before
the number is trusted. **[UNRESOLVED]** — see §11.

---

## 8. Why Episodes 0 and 21 are the right pair, and what they are not

Episodes 0 and 21 are **the only two episodes in the entire 30-episode tier that
share an object** — both are `tall_bar_stool`, both `scene.support=false`:

| | ep 0 | ep 21 |
| --- | --- | --- |
| task | tip the stool from face-down back to upright | pull the stool away from the desk |
| frames | 778 | 678 |
| IK residual vs demo | 1.07× | 1.44× |

That makes the proposed A/B a **same-object, different-task** transfer — the
easiest and most favourable case available in this tier. It is a good first
experiment precisely because it maximises the chance of seeing an effect at all,
but a positive result there **does not** license a claim about the other 28
episodes, which have 29 distinct objects between them.

For the **cross-object** check, the natural targets from Episode 0 (large
furniture, `support=false`, clean references) are:

| candidate | object | task | IK ratio |
| --- | --- | --- | --- |
| **ep 22** | `step_ladder` | drag with right hand | 1.01× |
| ep 8 | `skinny_wood_chair` | push | 0.92× |

Both are among the cleanest references in the tier, which keeps reference quality
from confounding the transfer result. **[VERIFIED]** (episode facts) /
**[GAP]** (no transfer result yet)

---

## 9. World-count ceiling on 8 GiB, measured

Eight collections per world count, replay capacity 500,000 (1.05 GiB), Episode 0:

| world_count | peak GPU | delta over baseline | result |
| --- | --- | --- | --- |
| 256 | 4,296 MiB | 3,691 MiB | ok |
| 512 | 4,575 MiB | 3,978 MiB | ok |
| **1024** | **6,826 MiB** | **6,271 MiB** | **ok — the ceiling** |
| 2048 | 7,607 MiB | 6,868 MiB | **OOM** |

2048 fails in exactly the place PPO failed:

```
Warp CUDA error 2: out of memory (in function wp_cuda_graph_create_exec, warp.cu:3430)
RuntimeError: Graph creation error
```

So FlashSAC's ceiling is **the same 1024 worlds as PPO**, and the failure is
again CUDA graph creation rather than steady-state allocation — which is why it
cannot be traded away with host RAM (see `CLAUDE.md`).

Note how little of the budget the worlds themselves cost: 256 → 1024 worlds adds
only 2,580 MiB, while raising replay capacity from 500,000 to 10,000,000 would
add 19.9 GiB. **The replay ring, not the world count, is what makes this recipe
not fit.**

Two workable operating points:

| | world_count | replay capacity | est. peak | replay ratio vs NVIDIA |
| --- | --- | --- | --- | --- |
| **throughput-first (recommended)** | 1024 | 500,000 | ~6.8 GiB | 2× theirs |
| memory-first | 512 | 1,000,000 | ~5.1 GiB | 4× theirs |

**1024 worlds is the better choice on both counts**: more throughput, and its
replay ratio is only 2× NVIDIA's reference rather than 8× at 256 worlds. At 1024
worlds, `updates_per_collection=2.0` matches their ratio exactly.
**[VERIFIED]**

---

## 10. Proposed Stage 2 budget

At the measured 256-world rate (~3,000 env steps/s), the full recipe is
349,999,104 steps = **32.4 h for one episode**. The first curriculum threshold is
35,000,320 steps = **3.2 h**, so one stage-0 arm is an affordable unit of
experiment.

| run | steps | hours |
| --- | --- | --- |
| A — ep21 from scratch, to 70M, checkpointed at 35M | 70.0M | 6.5 |
| B1 — ep0 source policy (**part of the transfer's cost**) | 35.0M | 3.2 |
| B2 — ep21 fine-tuned from B1 | 35.0M | 3.2 |
| C1 — ep22 `step_ladder` from scratch (cross-object control) | 35.0M | 3.2 |
| C2 — ep22 fine-tuned from B1 (cross-object transfer) | 35.0M | 3.2 |
| **total** | **210M** | **~19.4 h** (+0.7 h for 7 evaluations) |

These are upper bounds: the budget uses the 256-world rate, while the
recommended operating point is 1024 worlds (§9), which should be faster.

**Both comparisons get reported**, so the source-policy cost is never hidden:

- equal fine-tune budget — A@35M vs B2@35M
- equal **total** budget — A@70M vs (B1 + B2) = 70M

Metrics per run, all from `evaluate_policy.py` with evaluation's own
`voc_scale=0.0` and zero force/torque limits (never an assisted replay):
`mppe_cm`, `add_auc`, `spider_sr_uncentered` (SP-SR), `maniptrans_sr` (MP-SR),
plus `chord_sr`, wall clock, peak VRAM and environment steps.

### A100 80 GB **[GAP — arithmetic, not measured]**

An A100 holds the full 10,000,000 replay (20.92 GiB) and 2048+ worlds, so the
recipe runs unmodified there. Throughput is unknown; at 10k/20k/30k env steps/s
the full recipe is 9.7/4.9/3.2 h per episode, i.e. **875/437/292 GPU-hours** for
all 90 policies (30 episodes x 3 tiers). One measured A100 run is needed before
trusting the narrow end of that range.

---

## 11. What is still unverified

- **Zero-shot cross-episode evaluation has not been demonstrated.** The
  mechanism for pointing a checkpoint at a different episode's parquet is not
  yet confirmed to work. **[UNRESOLVED]**
- **Throughput at 1024 worlds is unmeasured.** The §9 sweep runs were 8
  collections each, so compile dominated and their rates say nothing. The
  budget in §10 uses the measured 256-world rate, which is conservative.
  One ~10-minute run at 1024 worlds would tighten it. **[GAP]**
- **No A100 measurement is possible here.** Any A100 projection is arithmetic
  from our own throughput, not a measurement. **[GAP]**
- **No transfer result exists.** Nothing in this document claims one.
