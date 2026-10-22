# Orientation-aware FlashSAC — implementation and readiness

A thirteenth, bounded object-orientation reward term for NVIDIA's ReconBody
FlashSAC objective, implemented entirely in our own code.

Companion documents: [the implementation plan](../plans/2026-10-21-orientation-aware-flashsac.md) ·
[the curriculum audit](../../outputs/step_D/curriculum_audit/REPORT.md) ·
[Step D feasibility](../findings/STEP_D_FLASHSAC_FEASIBILITY.md) ·
[tier references](../findings/TIER_REFERENCES.md).

- **Upstream code:** commit `193382a`, **unmodified** — verified by `git status`
- **Image:** `flash_chord:latest`
- **Entry point:** `scripts/train/train_orientation_flash_sac.py`
- **Verification:** `bash scripts/verify/verify_orientation_flash_sac.sh`

Conclusions are marked **[VERIFIED]** (observed directly on this machine),
**[DOC]** (stated by NVIDIA, not independently checked) or **[GAP]**.

---

## 1. Why a thirteenth term

The [curriculum audit](../../outputs/step_D/curriculum_audit/REPORT.md) established
that **all nine curriculum stages share the same twelve reward weights, and none
of them is a dense object-orientation term** [VERIFIED]. Orientation enters
training only as a coarse failure threshold (`object_orientation_threshold:
1.5` rad = 85.94°) and indirectly through palm/pelvis imitation.

Track 2 ranks by ADD-AUC over thresholds of 1–9 cm. On a half-metre object a few
degrees of rotation leaves that band, so AUC is dominated by orientation while
the optimised objective contains no orientation signal. The measured consequence:
Episode 0's 10M policy reaches MPPE 22.59 cm (31% better than the 32.59 cm
stationary floor) but AUC 0.2125 against a 0.3769 floor — **44% worse than not
touching the object**.

## 2. The reward

```text
q, q_ref  : object body quaternion and command target, both xyzw (Newton)
q_error   = q_ref * inverse(q)
theta     = 2 * atan2(|q_error.xyz|, |q_error.w|)        # [0, pi]
raw       = -(theta / pi) ** 2                           # [-1, 0]
weight    = 2.0, in every stage
```

| Orientation error | Raw term | Contribution at 50 Hz | [VERIFIED] |
|---|---:|---:|---|
| 0° | 0 | 0 | kernel and host reference |
| 30° | −1/36 | −0.00111111 | kernel and host reference |
| 90° | −0.25 | −0.01 | kernel and host reference |
| 180° | −1 | −0.04 | kernel and host reference |

For scale: object position rewards at most +0.02/step, force closure +0.10, body
joints +0.10, and a termination costs −6.0. The new term is a bounded penalty,
zero at the target, and is never allowed to dominate.

### Conventions, each one tested

| Hazard | Decision | Test |
|---|---|---|
| wxyz vs xyzw | `CommandBuffers` already converted the parquet's wxyz target to xyzw; we read `command.body_target_quat_w` and do **not** convert again | `test_wxyz_input_would_be_wrong_and_is_detectable` |
| MuJoCo Warp `qpos` is wxyz | never read; we read Newton `state.body_q` | kernel binds `state.body_q` only |
| double cover (`q` vs `−q`) | `abs(w)` of the error quaternion | `test_kernel_double_cover_invariance` |
| derivative blow-up near identity | small-angle branch `theta² = 4·s²` for `s² < 1e-12`, instead of `acos` | `test_small_angle_branch_is_accurate_and_continuous` |
| invalid quaternions | raise; never silently substitute identity | `test_invalid_quaternions_raise` |
| object symmetry | **identity only** — the stool is not assumed yaw-symmetric, because ADD compares corresponding points | `test_identity_only_symmetry_distinguishes_yaw` |

There is no unique geodesic derivative at exactly 180°; we do not claim global
smoothness. FlashSAC does not differentiate through the simulator, so these
numerical tests validate the shaping function, **not** that it produces useful
policy gradients [GAP — that is what the training run measures].

## 3. How it reaches the learner

A reward change that only altered a scalar would **not** be optimised: FlashSAC
stores *raw* objective terms, discounts them over the n-step window, and
recomposes the reward from the current curriculum weights at sample time
(`_relabel_rewards`). The thirteenth term therefore has to be a real term.

```
pipeline.flash_sac_orientation.OrientationReconBodyObjective
  └─ delegates to NVIDIA's ReconBodyObjective       (12 terms, untouched)
  └─ append_object_orientation kernel
        copies terms 0..11 verbatim, writes term 12, recomposes score
             │
env.objective_term_names  ->  13
             │
WarpRLEnv.step  ->  ReplaySpec.objective_term_count = 13
             │
replay: pending (13) -> n-step discount (13) -> ring (13)
             │
relabel_rewards(batch, curriculum_weights[13], frame_dt)  ->  learning target
```

Everything downstream of the objective sizes itself from
`len(env.objective_term_names)`, so **no upstream file needed editing**
[VERIFIED by reading `replay.py`, `learner.py`, `runner.py`, `environment.py`].

Proven by test, not by inspection alone:

- `test_replay_allocates_thirteen_term_storage` — the ring really is 13 wide.
- `test_n_step_window_discounts_the_thirteenth_term_like_the_others` — pushes a
  full 7-step window through NVIDIA's own `insert_replay` and reads back the
  stored discounted terms.
- `test_changing_only_the_thirteenth_weight_changes_the_learning_target` — the
  decisive one: with every other weight at zero, moving the orientation weight
  changes the relabelled reward by exactly `frame_dt · 2 · raw`.
- `test_stored_terms_are_raw_not_weighted` — replay holds −0.25, and only
  relabeling turns it into −0.01.

## 4. What is preserved

| Preserved | How it is guaranteed |
|---|---|
| NVIDIA upstream | never written; `git -C official/video_to_data status` is clean, checked by the verify script |
| The original twelve terms | copied byte-for-byte by the kernel (`test_first_twelve_terms_are_copied_bit_for_bit`); their weights come from a config **generated** from upstream's YAML, which refuses to run if upstream's weights or thresholds moved |
| SONIC, Newton physics, obs/action | untouched; actor stays 486/43, critic context 10 |
| Curriculum thresholds, VOC schedule, reset mixture | copied verbatim by the generator; 9 stages, same 9 thresholds |
| `object_orientation_threshold: 1.5` | that is upstream's **failure limit**, not our reward; the exporter explicitly preserves it |
| The official evaluator and packer | never modified; the portable actor is what adapts |

## 5. Checkpoint compatibility

The official evaluator rebuilds the whole experiment from the config in a
checkpoint's safetensors header *before* loading tensors. A native thirteen-term
checkpoint therefore names our module and carries thirteen curriculum weights.

Two artifacts per gate:

1. **Native** (`policy_*.safetensors`, `state_*.safetensors`) — the true
   thirteen-term training config, resumable by our launcher.
2. **Portable** (`pipeline/flash_sac_actor_export.py`) — byte-identical actor
   tensors, with `user.resolved_config_json` projected back to upstream's
   twelve-term schema, and the real training config preserved beside it in
   `user.training_resolved_config_json`, plus the exact reward definition,
   parent SHA-256 and exporter SHA-256.

**[VERIFIED] on this machine, from the 65,536-step integration smoke:**

| Check | Result |
|---|---|
| Portable actor in the unchanged `evaluate_policy.py`, no local module on `PYTHONPATH` | **ran**, MPPE 32.474 cm, AUC 0.37821 |
| Control: the native 13-term checkpoint, same conditions | **correctly rejected** — `Error locating target 'pipeline.flash_sac_orientation...'`, `full_key: env.objective` |
| Tensor index and payload after export | **identical** |
| `env.objective` as well as `objective` projected | both, because Hydra resolves `env.objective: ${objective}` |

Those smoke metrics are the **inert floor**, not a result: a 65k-step policy
barely moves the object. They are a format proof only.

The exporter never projects learner state or replay, and refuses if asked
(`test_export_refuses_a_learner_state_file`).

## 6. Measured resources

All on the RTX 5070 Laptop (8,151 MiB), driver 580.178.04, 1024 worlds, replay
capacity 500,000, `updates_per_collection=2.0`.

**[VERIFIED]** by `bash scripts/train/benchmark_orientation_arms.sh 2097152`, both arms
back to back, 2,097,152 environment steps each
(`outputs/step_D/bench_arms.txt`):

| Quantity | Original (12 terms) | Orientation (13 terms) | Cost |
|---|---:|---:|---:|
| Steady throughput, env steps/s | 16,504 | 16,330 | **−1.1 %** |
| Peak GPU memory, MiB | 7,165 | 7,276 | **+111** |
| Startup (Warp + JAX compile), s | 39 | 40 | +1 |
| Wall clock for 2.097M steps, s | 164 | 165 | +1 |
| Projected 350M run, h | 5.89 | **5.95** | +0.06 |

Peak GPU memory includes ~670 MiB of desktop (VS Code and Chrome). Against the
card's 8,151 MiB that leaves **875 MiB of headroom** for the orientation arm.
The progress bar's `env steps/s` is a **cumulative average including the ~40 s
compile**, so it climbs monotonically and must never be read as the steady
rate; the benchmark differences the log instead.

Replay cost of the extra column: **exactly 4 B/transition**, 1.91 MiB in a 500k
ring [VERIFIED by `test_extra_term_costs_exactly_four_bytes_per_transition`
against NVIDIA's own `replay_memory`].

Checkpoint storage for the 350M run: 34 gates × (4.6 MiB actor + 152 MiB learner
state) = **5.22 GiB**, `save_replay=false`. `scripts/train/monitor_training_disk.py`
prunes only that run's learner states, keeps every actor, and refuses protected
directories outright.

## 7. The 350M configuration

```
world_count                  1,024
total_environment_steps      349,999,104     (NVIDIA's terminal budget)
collections                  341,796         (exactly divisible, so every
                                              curriculum threshold lands on a
                                              collection boundary)
updates_per_collection       2.0
total_optimizer_updates      683,592         (= the full-recipe LR horizon,
                                              so no LR override is needed)
replay capacity / minimum    500,000 / 50,000
seed (training and reset)    42
object_orientation weight    2.0, every stage
```

Fresh actor, critic, targets, optimizer, normalizer, replay and counters. **No
warm start from the existing 10M policy.**

## 7b. The term is nearly inert for the first 35M steps — by design

**[VERIFIED]** from the 262,144-step metrics run
(`outputs/step_D/smoke_metrics/metrics.jsonl`), weighted contributions per
environment step at stage 0:

| Term | Contribution |
|---|---:|
| termination | −0.064453 |
| force_closure | +0.024542 |
| body_joint_position | +0.021903 |
| object_position | +0.018469 |
| … | … |
| **object_orientation** | **−0.000042** |

The measured object orientation error at stage 0 is **0.0697 rad (4.0°)**,
because the stage-0 VOC controller runs at scale 1.0 and supplies orientation
torque as well as translation and gravity compensation. A penalty that is zero
on target is therefore almost silent while the assist is holding the object.

This is correct behaviour, not a bug, but it has a schedule consequence:

- The term can only start to matter as VOC decays — **0.75 at 35,000,320 steps,
  and 0.0 playback VOC only at 209,999,872**.
- **A short run cannot validate that the term helps.** Any run stopped before
  ~35M measures an objective that is, in contribution terms, upstream's.
- **[GAP] Weight 2.0 was chosen against the 180° worst case (−0.04/step), not
  against the 4° assisted regime.** It may prove too small once VOC is gone.
  Do not tune it mid-run; that is a separate, matched experiment.

## 8. Honest limitations

- **[GAP] There is no matched original-reward control at 350M.** This run is a
  single arm. A quality difference against the existing 10M policy confounds the
  reward change with 35× more training. The implementation plan's paired design
  exists precisely to avoid that; running one arm is a deliberate schedule
  choice, and the result must be reported as such.
- **[GAP] Weight 2.0 is a first hypothesis, not an optimised value.** Do not tune
  it mid-run.
- **[VERIFIED] Single-rollout evaluation is not repeatable.** Five fresh-process
  evaluations of the same 10M checkpoint gave AUC 0.1127–0.1290 and MPPE
  47.6–189.6 cm. Every quality claim needs repeated evaluations with reported
  spread, never a best repeat.
- **Submission numbers must come from `evaluation.world_count=1`**, because the
  packer scores world 0 only.
- Assisted replay is not policy success: stage 0 trains at `voc_scale: 1.0`, and
  evaluation forces it to 0.0.

## 9. Reproduce

```bash
cd /home/ibrahim/v2d-track2-robotic-grounding

# everything: upstream clean, config reproducible, host + container tests,
# preflight, disk-monitor protections, evaluator round trip and its control
bash scripts/verify/verify_orientation_flash_sac.sh

# just the tests
.venvs/episode0/bin/python -m pytest pipeline/tests/test_orientation_reward.py \
    pipeline/tests/test_flash_sac_actor_export.py -q

# the exact training command, without launching anything
python3 scripts/train/train_orientation_flash_sac.py --dry-run

# throughput and VRAM, both arms
bash scripts/train/benchmark_orientation_arms.sh 2097152

# losses and per-term rewards from any run with a metrics.jsonl
.venvs/episode0/bin/python scripts/train/show_training_metrics.py outputs/step_D/smoke_metrics
.venvs/episode0/bin/python scripts/train/show_training_metrics.py outputs/step_D/smoke_metrics --terms
.venvs/episode0/bin/python scripts/train/show_training_metrics.py outputs/step_D/smoke_metrics --plot /tmp/metrics.png
```

## 10. Logging

`logging.mode=disabled` makes upstream's `train_flash_sac.py` attach **no
logger at all**, so a local run prints environment steps per second and nothing
else. `pipeline/flash_sac_train_entry.py` is upstream's `main()` with
`pipeline/flash_sac_metrics_logger.py` swapped in for that case (W&B mode is
untouched). It writes, per `log_interval` collections — 25 at 1024 worlds, so
every 25,600 steps, about 13,600 records over the 350M run:

- `metrics.jsonl` and `metrics.csv` beside the checkpoints — 47 metrics per
  record: actor/critic/temperature loss, entropy, episode return and length,
  replay fill, reward-normalizer scale, loss scale, curriculum stage and VOC,
  **all thirteen per-term reward contributions**, six tracking errors and the
  termination-cause breakdown;
- a one-line digest per boundary on stdout, so `tail -f` on the training log
  shows losses live.

It also sets `publish_diagnostics=True`, which upstream disables because W&B was
its only sink; that is what fills the tracking-error and termination-cause
series.
