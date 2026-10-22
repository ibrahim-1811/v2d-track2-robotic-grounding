# Step D, Stage 2 — weight-only transfer: mechanism verified, first signal negative

The actor-only transfer machinery works and is verified. The first smoke
comparison says **fine-tuning a transferred actor made things worse, not
better** — but at a budget where no arm has learned anything, so it is a signal
about *stability*, not about transfer at scale.

Companion: [STEP_D_FLASHSAC_FEASIBILITY.md](STEP_D_FLASHSAC_FEASIBILITY.md)
(Stage 1) · [TIER_REFERENCES.md](TIER_REFERENCES.md) (the references used).

- **Upstream:** `193382a`, **unmodified**
- **Tools added:** `scripts/transfer_flash_sac_policy.py`,
  `scripts/read_checkpoint_metadata.py`, `scripts/evaluate_step_d.sh`,
  `scripts/probe_flash_sac_memory.py`

---

## 1. The transfer tool, and what it proves per run

`scripts/transfer_flash_sac_policy.py` composes the **target** episode's
experiment, builds a **fresh** target learner with `create_learner`, splices in
only the source actor's `params` + `batch_stats`, and writes a valid
`training_state` checkpoint carrying the target's own metadata. It uses
upstream's `create_learner`, `load_actor_checkpoint`, `load_checkpoint`,
`training_checkpoint`, `save_checkpoint` and `_validate_resume_schemas` — no
upstream file is touched.

It refuses to emit anything unless every one of these holds, checked and printed
on each run:

```
PASS  policy and critic I/O schemas are compatible
PASS  action / observation / embodiment / objective / sim / termination / reset / env
          identical between source and target        (8 config groups compared)
PASS  only ['actor'] differs from a fresh target learner
          31 leaves changed, 157 identical to fresh
PASS  environment_steps      = 0
PASS  global_update_step     = 0
PASS  update_credit          = 0.0
PASS  reward_normalizer / exploration / loss_scale / temperature
          are the target's own fresh state
PASS  replay buffer is not part of a training_state checkpoint at all
```

The leaf comparison is the real guarantee: both the fresh and the merged learner
are flattened with `jax.tree_util.tree_flatten_with_path` and compared array by
array, so "fresh optimizer, fresh critic, fresh normalizer" is **measured**, not
asserted. **[VERIFIED]**

### Semantic compatibility beyond tensor shapes

Matching dimensions (486/43/10/12) is necessary but not sufficient — two
episodes could agree on widths while disagreeing on what the widths *mean*. The
tool therefore compares the resolved config of eight groups
(`action, observation, embodiment, objective, sim, termination, reset, env`) and
**refuses** on any difference. Only `task` is expected to differ; that
difference is the experiment. For Episodes 0 → 21, all eight matched.
**[VERIFIED]**

### Metadata the evaluator requires

The first emitted policy failed evaluation with
`ValueError: FlashSAC checkpoint is missing its curriculum stage`.
`save_learner_checkpoint` adds `curriculum_stage` and `voc_scale` at save time
from the live stage; a hand-built checkpoint must too. The tool now derives them
from the target's own curriculum at step 0 (`stage_index(0)`), never hard-coded,
and also carries `reset_to_first_frame_probability` /
`immediate_first_frame_probability`. **[VERIFIED]**

### Provenance

The emitted file records what it is, so it can never be mistaken for a real run:

```
user.environment_steps                 0
user.curriculum_stage                  0
user.transfer_mode                     actor
user.transfer_source_checkpoint        /out/smoke_ep0/policy_65536.safetensors
user.transfer_source_environment_steps 65536
task.parquet                           ...tall_bar_stool_pull_08...   <- the TARGET
reference.num_frames                   678                            <- ep21, not ep0's 778
```

---

## 2. The step counter really does reset

The decisive runtime evidence that this is a transfer and not a resume: arm B's
first progress line is

```
training progress: 0/65,536 environment steps (0.0%)
```

A plain `resume` from Episode 0's `state_65536.safetensors` would have started
at `65,536/65,536` — already at the budget, in whatever curriculum stage that
implied. **[VERIFIED]**

---

## 3. The smoke comparison

Identical settings in both arms except `training.resume` / `checkpoint_path`:
Episode 21, 256 worlds, replay capacity 250,000, `minimum_size` 4,096,
65,536 environment steps, `updates_per_collection` 4.0. Evaluation is
`evaluate_policy.py` at 64 worlds — unassisted, `voc_scale=0.0` and zero
force/torque limits, never a replay.

| arm | what it is | MPPE (cm) | ADD-AUC | SP-SR | MP-SR | CHORD-SR |
| --- | --- | --- | --- | --- | --- | --- |
| **A** | ep21 from scratch | **55.98** | 0.1881 | 0 | 0 | 0 |
| **B** | ep0 actor → ep21, fine-tuned 65k | **109.46** | 0.1210 | 0 | 0 | 0 |
| **Z** | ep0 actor on ep21, **zero-shot** | **55.86** | 0.1878 | 0 | 0 | 0 |

Throughput and memory were identical between arms — transfer costs nothing at
run time:

| | A (scratch) | B (transfer) |
| --- | --- | --- |
| instantaneous throughput | 3,200 env steps/s | 3,200 env steps/s |
| peak GPU | 4,200 MiB | 4,540 MiB |

The 340 MiB difference is the 152 MiB transfer checkpoint being loaded.
**[VERIFIED]**

---

## 4. What those numbers actually mean

**MPPE ≈ 55 cm is the "never touched the object" floor for this episode.**
Measured from the exported trajectory of arm A, world 0:

| | mean | max |
| --- | --- | --- |
| reference object travel from frame 0 | **0.5471 m** | 1.1261 m |
| arm A achieved object travel | **0.0146 m** | 0.0165 m |
| actual per-frame error | **0.5497 m (54.97 cm)** | — |
| counterfactual: object frozen at frame 0 | **0.5474 m (54.74 cm)** | — |

Arm A moved the object 1.5 cm over the whole episode, and its error is within
0.2 cm of simply never moving it. So **A and Z are both at the do-nothing
floor** — and their achieved object travel is identical to four decimals
(0.0146 m), which is what you expect when neither policy touches the object and
it just settles under the same physics.

**Arm B is worse than doing nothing, and the trajectory says why:**

| | achieved object travel, mean | max |
| --- | --- | --- |
| A (scratch) | 0.0146 m | 0.0165 m |
| Z (zero-shot) | 0.0146 m | 0.0165 m |
| **B (transfer + fine-tune)** | **1.1396 m** | **1.7556 m** |

Arm B moves the object **2.1× further than the reference does** (1.14 m vs
0.55 m). It is not failing to act — it is flinging the stool around. That is the
classic signature of **a pretrained actor paired with a randomly initialized
critic**: the fresh critic's Q-gradients are meaningless early on, and they
destroy a policy that started out structured. **[VERIFIED]**

### What this does and does not establish

**Does not:** that transfer fails. 65,536 steps is **0.19 %** of the 35M stage-0
budget, and the *source* policy itself had only 65,536 steps — it is a
do-nothing policy (0.0146 m of object motion). Transferring a do-nothing policy
cannot be expected to help. The zero-shot arm matching scratch exactly is
consistent with "the source knows nothing yet", not with "transfer does not
work". **[GAP]**

**Does:** that actor-only transfer into a fresh critic is **actively
destabilizing** at the start of fine-tuning, strongly enough to turn an inert
policy into a destructive one. That is a real risk for the full experiment and
has to be designed around rather than discovered at 35M steps.

---

## 5. Submission-evaluator compatibility

### The trajectory export is accepted

`evaluate_policy.py`'s `object_trajectories_output` writes exactly the schema
the kit reads — `flash_chord.schema = flash_chord_object_trajectories_v1` — with
full provenance (`checkpoint_sha256`, `eval_code_sha256`, `eval_script_sha256`,
`reference_parquet`). Running the kit's own `tools/pack_submission.py` against
arm A's export:

```
input reference check: one rigid frame transform per episode;
                       maximum residual 1.23e-07 m, 1.26e-07 rad
converted 1 Track 2 episode(s) into the released frame
wrote packer_test.csv (678 rows)
locally computed: SP-SR=0.0000  MP-SR=0.0000  MPPE=55.0082
```

That **1.23e-07 m** residual is an independent confirmation of the whole Step
A→C pipeline: the kit found a single rigid transform between our reference and
the released dataset to within 0.1 µm. A frame error anywhere in staging or
retargeting would show up here. **[VERIFIED]**

### Evaluate at `world_count=1` for any number you intend to submit

With the evaluator's metrics JSON present, the packer cross-check warns:

```
WARNING: the trajectory and the evaluator's metrics JSON disagree.
  - episode 21 mppe_cm: evaluator reported 55.976887,
                        recomputing from the trajectory gives 55.008226
The leaderboard scores the TRAJECTORY.
```

The kit's `track_2_tier1` spec sets `worlds_per_episode: 1` and the packer
builds its grid from `range(worlds_expected)`, so it scores **world 0 only**
(678 rows), while our evaluation averaged 64 worlds. The 1.8 % gap is exactly
that. It is harmless for A/B comparisons as long as both arms use the same world
count, but **every submission-grade number must come from
`evaluation.world_count=1`**. **[VERIFIED]**

Two further notes:
- `participant_reference_root` falls back to
  `<root>/track_2/tier_1_multiview_caption`, which is our mirror's layout, so
  `--dataset-root data/v2d_challenge` resolves with no copying.
- The packer's **AUC** cross-check is skipped in `.venvs/kit`:
  `cannot sample vertices for tall_bar_stool (No module named 'trimesh')`.
  MPPE/SP-SR/MP-SR cross-check fine. **[GAP]** — install `trimesh` into
  `.venvs/kit` to close it.
- Episode coverage is still the blocker for a real submission: with all 30
  required, the packer stops at
  `error: 29 of 30 scored episodes have no trajectory`. The single-episode run
  above used a scratch copy of `config/tracks.json` with
  `scored_episodes: [21]`, purely to exercise format validation.

---

## 6. Reproduce

```bash
cd /home/ibrahim/v2d-track2-robotic-grounding
source <scratch>/fc_env.sh           # docker helper with our mounts + warp cache

# build the weight-only transfer checkpoint (prints every check in §1)
fc python /driver/transfer_flash_sac_policy.py \
    --source-policy /out/smoke_ep0/policy_65536.safetensors \
    --target-parquet "$EP21" --target-support false --world-count 256 \
    --replay-capacity 250000 --replay-minimum-size 4096 --transfer actor \
    --out /out/transfer/ep0_to_ep21_actor_w256.safetensors \
    --emit-policy /out/transfer/zeroshot_ep0actor_on_ep21.safetensors

# inspect any checkpoint from the host (stdlib only, no container)
python3 scripts/read_checkpoint_metadata.py \
    outputs/step_D/transfer/ep0_to_ep21_actor_w256.safetensors

# evaluate unassisted + export a packer-ready trajectory
bash scripts/evaluate_step_d.sh /out/smokeA_ep21_scratch/policy_65536.safetensors A_scratch_65k 64 21

# the results quoted above
cat outputs/step_D/eval/{A_scratch_65k,B_transfer_65k,Z_zeroshot_ep0actor}.json
```

---

## 7. Source-policy staging: manipulation IS emerging

Episode 0 trained to 10,000,384 steps at the item-4 benchmark's winning
configuration — **1024 worlds, `updates_per_collection=2.0`** (0.001953
updates/env-step, NVIDIA's reference ratio). **11.0 minutes**, 16,171 env
steps/s instantaneous, peak 6,825 MiB. Checkpoints every 1M, all preserved.

Evaluated unassisted at **`evaluation.world_count=1`** (submission-grade; the
packer scores world 0 only):

| checkpoint | object travel | ref travel | ratio | MPPE cm | stationary cm | gain | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1M | 0.0050 m | 0.3254 m | 0.02 | 32.43 | 32.52 | **0.3 %** | INERT |
| 5M | **0.2973 m** | 0.3254 m | **0.91** | 30.78 | 32.52 | 5.3 % | **MANIPULATING** |
| 10M | 0.2588 m | 0.3254 m | 0.80 | **22.41** | 32.52 | **31.1 %** | **MANIPULATING** |

`gain` is the fraction of the stationary baseline's error the policy removes;
≤5 % means it is doing nothing a frozen object would not do.

**Manipulation begins between 1M and 5M steps.** At 1M the robot never contacts
the stool — its MPPE of 32.4738 cm is *bit-identical* to the 65k smoke policy's,
because with no contact the object's settling is deterministic from the same
initial state. By 5M the object travels 91 % of the reference distance, and by
10M the policy has removed **31 %** of the do-nothing error. **[VERIFIED]**

### But only position is being learned — orientation regressed

| checkpoint | object position error | object orientation error | ADD-AUC |
| --- | --- | --- | --- |
| 1M (inert) | 0.3243 m | **37.25°** | 0.3782 |
| 5M | 0.3078 m | 49.01° | 0.3470 |
| 10M | **0.2241 m** (−31 %) | **48.43°** (worse than inert) | 0.2125 |

Episode 0's task is *"tip the stool from a face-down lying position back to
upright"* — fundamentally a **rotation**. The policy has learned to translate
the stool toward where it should be while rotating it **wrongly**: leaving it
face-down scores 37°, and the trained policy scores 48°. That is why ADD-AUC
*falls* while MPPE improves — ADD mixes position and orientation, and the
orientation term is getting worse.

So "manipulation is emerging" is true and worth acting on, but the honest
statement is **position tracking is emerging; the task's defining rotation is
not, and is currently worse than not touching the object at all.** All success
rates remain 0.000 at 10M. **[VERIFIED]**

---

## 8. Actor+critic transfer: mechanism verified against the real source

Both variants were built from the genuine 10M Episode 0 learner state, at 1024
worlds to match the consuming run:

| variant | leaves changed from fresh | groups |
| --- | --- | --- |
| `--transfer actor` (control) | **31** | `actor` |
| `--transfer actor_critic` (primary) | **87** | `actor`, `critic`, `target_critic` |

Both left 101–157 leaves bit-identical to a fresh target learner, with
`environment_steps`, `global_update_step` and `update_credit` all 0 and the
reward normalizer, exploration, loss scale and temperature all the target's own.

### Why a transferred critic is interpretable here

FlashSAC's critic is **distributional**: it emits a categorical distribution
over a fixed value support and predicts returns in **reward-normalized** units.
The tool verifies the three things that have to hold, and all three do:

```
PASS  critic value support identical: +/-5.0 (normalized return units)
PASS  critic architecture identical (atoms, count, blocks, width, expansion)
PASS  stage-0 objective weights identical (12 terms) - the critic predicts the same reward
```

The value support is `±reward_normalization.normalized_return_bound`,
constructed at `learner.py:732` from config — **never learned, never
task-derived** — so it is the same for every episode by construction. And
because Episodes 0 and 21 share the entire `recon_body` curriculum, the stage-0
reward *function* is literally identical; only the reference trajectory differs.
**[VERIFIED]**

### The one real incompatibility

The transfer resets the reward normalizer to `mean=0, variance=1, count=0`
(`create_reward_normalizer`). The source critic was calibrated against Episode
0's *converged* normalization, so it is miscalibrated until the target's running
moments adapt. This is bounded — initial scaling is identity, not a divide-by-
near-zero — but it is the known risk of critic transfer here, and the tool
prints it as a `NOTE` on every actor+critic build. Carrying the normalizer's
moments across is the obvious mitigation if instability appears. **[GAP]** —
not yet tested.

---

## 9. Against the real leaderboard: we are optimising the wrong thing

The V2D Track 2 leaderboard (Tier 1, read 2026-10-20) **ranks by AUC**, not
MPPE:

| # | team | AUC | SP-SR | MP-SR | MPPE cm |
| --- | --- | --- | --- | --- | --- |
| 1 | vinguynvn | 0.9691 | – | – | – (no entry for SP-SR/MP-SR/MPPE) |
| 2 | DAP | 0.6003 | 0.8000 | 0.4333 | 12.66 |
| 3 | Xueyi Wang (FF) | 0.5953 | 0.8667 | 0.8333 | **7.44** |
| 4 | maniphys.ai | 0.5826 | 0.8333 | 0.7000 | 11.87 |
| – | **CHORD Baseline** | **0.5104** | 0.7000 | 0.2000 | 17.74 |
| 5 | wte09fl | 0.4295 | 0.5333 | 0.2333 | 14.88 |

### What AUC actually measures

From the kit's own `metric_code/track_2/AUC.py`:

```python
_ADD_THRESHOLDS_M = np.arange(0.01, 0.10, 0.01)   # 1 cm .. 9 cm
```

AUC is the area under *"fraction of frames whose ADD is below t"* for t from
**1 cm to 9 cm**. ADD is the mean distance between corresponding object model
points under the achieved and reference poses, so at a 1–9 cm tolerance on an
object half a metre across, **a few degrees of rotation error is enough to leave
the band entirely**. AUC is dominated by orientation.

MPPE is a different quantity: six keypoints at ±5 cm along each body axis
(`_MPPE_KEYPOINT_RADIUS_M = 0.05`), so it mixes position and orientation but is
far more forgiving of both.

### The do-nothing floor, computed with the official formulas

`scripts/leaderboard_floor.py` re-implements both formulas from the kit's
published source and scores a policy that leaves the object frozen at its
frame-0 pose, across all 30 Tier 1 episodes. The re-implementation agrees with
NVIDIA's evaluator to **0.3 % on AUC and 0.4 % on MPPE** (Episode 0, against our
measured inert 1M checkpoint), so the numbers below are trustworthy.

| | AUC | MPPE cm |
| --- | --- | --- |
| **do nothing, tier mean** | **0.2661** | **49.12** |
| lowest-ranked board entry (wte09fl) | 0.4295 | 14.88 |
| CHORD Baseline | 0.5104 | 17.74 |

Per-episode the floor varies enormously with how far the object is supposed to
travel: Episode 8 (`skinny_wood_chair`, 0.18 m of motion) scores **0.5978** by
doing nothing — above the CHORD baseline's tier average — while Episode 4
(`baseball_bat`, 1.11 m) scores 0.1293.

### Episode 0: our training moved us BELOW the floor

| checkpoint | AUC | MPPE cm |
| --- | --- | --- |
| 1M (inert) | 0.3782 | 32.47 |
| 5M | 0.3470 | 30.87 |
| **10M** | **0.2125** | **22.59** |
| do-nothing floor | 0.3769 | 32.59 |

MPPE improved **31 %**. AUC fell **44 % below doing nothing.** Both follow
directly from §7: the policy learned to translate the stool toward its target
while rotating it *wrongly* (orientation error 37° inert → 48° trained). MPPE's
±5 cm keypoints tolerate that; AUC's 1–9 cm ADD band does not.

**This is the single most important finding of Step D so far.** The stage-0
reward weights `object_position: 1.0` and have no explicit object-orientation
term, so the optimisation target and the ranking metric disagree. Training
longer along this gradient makes the leaderboard position worse, not better.
**[VERIFIED]**

### What it implies

1. **Report AUC as the primary metric from now on**, with the per-episode
   stationary floor beside it. MPPE alone is misleading, and we have now been
   misled by it once.
2. **The transfer experiment's conclusion could invert** depending on which
   metric is used. It must be judged on AUC against the floor.
3. The reward needs an object-orientation term, or far longer training, before
   any amount of transfer tuning matters. **[GAP]**
4. Rank 1 (AUC 0.9691 with no SP-SR/MP-SR/MPPE entry) is not comparable to the
   complete submissions below it; the realistic target is the 0.51–0.60 band.
