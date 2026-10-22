# Orientation-aware FlashSAC implementation and experiment plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task once implementation is authorized. Steps use checkbox syntax. This document does not authorize training, committing, pushing, or submitting.

**Goal:** Compare the original G1 FlashSAC objective with one additional bounded object-orientation term on Episode 0, with reliable evaluation statistics and portable official-evaluator actor exports.

**Architecture:** A local `ObjectiveSpec` delegates the original twelve terms to NVIDIA and appends a thirteenth raw term. A local launcher uses the upstream learner unchanged. Native learner/replay checkpoints preserve the full training configuration; a separate actor export retains the exact actor tensors and supplies an upstream-only evaluation configuration plus explicit training provenance.

**Tech stack:** Existing pinned FlashCHORD image, Newton/MuJoCo Warp, Warp 1.14, JAX/Flax/Optax, safetensors, NumPy/SciPy, PyArrow, and the supplied unchanged submission kit. No dependency upgrades are proposed.

**Spec:** [Experiment specification](../../../outputs/step_D/orientation_plan_20261021/experiment_spec.json), [resolved seed-42 baseline](../../../outputs/step_D/orientation_plan_20261021/proposed_original_seed_42.json), and the [completed curriculum audit](../../../outputs/step_D/curriculum_audit/REPORT.md).

## Status and constraints

This is the requested design and implementation plan, not an implemented training feature.
The work completed for this plan comprises read-only inspection, eleven inference-only evaluations, a throwaway numerical reward probe, and local checkpoint-format/packer probes. **No training was started.** All pre-existing code, documents and the exact checkpoint used for evaluation are protected by a SHA-256 preservation manifest. NVIDIA upstream was mounted read-only for all evaluations. No commits, pushes, uploads or submissions were made.

- Never write under `official/`, the pristine input mirror, or an existing checkpoint directory.
- Preserve the original references, all twelve original weights, observation/action semantics, termination, collision physics and curriculum thresholds.
- Actor observation/action dimensions remain 486/43; critic context remains 10.
- Use one GPU job at a time; the measured training allocation leaves insufficient room for a separate evaluator process.
- Native checkpoints must describe actual training. Portable actors must explicitly distinguish their evaluation reconstruction config from preserved training provenance.
- No new substantial training until reward/kernel/replay/export tests and the evaluation protocol below pass review.

## Verified evaluation diagnosis

Checkpoint: `outputs/step_D/source_ep0_w1024/policy_10000384.safetensors`.

SHA-256: `81dc6d3aeda309bb89269bf3866f1a9fe26e644eef77cd91d0b12523a232a079`.

Image: `sha256:d51290087ec5c3d0028ff914c2dc6314bc98af2414f6b425cbf47c7f129d2661`.

Hardware: RTX 5070 Laptop, 8151 MiB VRAM, driver 580.178.04. All five baseline runs used the unchanged official `scripts/evaluate_policy.py`, one world, explicit frame-zero reset, deterministic actor inference, zero VOC/force/torque limits, zero settling, and logging disabled. Each run was a fresh container/process. Only output paths differed.

| Repeat | AUC | MPPE, cm | Mean orientation error |
|---|---:|---:|---:|
| 1 | 0.112709 | 48.844 | 46.52° |
| 2 | 0.119296 | 81.313 | 83.50° |
| 3 | 0.119055 | 102.180 | 84.97° |
| 4 | 0.129017 | 189.555 | 78.31° |
| 5 | 0.115119 | 47.574 | 94.95° |

Mean AUC is 0.119039, sample standard deviation 0.006226. Mean MPPE is 93.893 cm, sample standard deviation 58.209 cm. These values are a small execution sample, not an estimate of training-seed variability. The old archived 22.59 cm MPPE is a real archived rollout, but is not representative of a guaranteed repeatable policy outcome.

Across repeats, reference poses and evaluation provenance match. Object-position differences start at sub-micrometre scale; compared with run 1, every other run crosses 1 micrometre at frame 73 and 1 millimetre at frame 74. Centimetre differences appear at frames 85–104. This localizes the sharp divergence to approximately 1.5–2.1 seconds, around the interaction phase. The exact first differing solver/controller kernel has not been isolated.

Three further runs with `CUDA_LAUNCH_BLOCKING=1` still vary: AUC 0.120662–0.211279 and MPPE 39.03–153.85 cm. Launch blocking is not a repeatability fix and is not selected for the experiment. Two unchanged evaluations of the inert 1M policy produce bit-identical object positions and identical AUC 0.3782133676 / MPPE 32.47377337 cm.

All ten ordinary/control trajectories reproduce their saved AUC and mean ADD when rescored on the host with the exact evaluator mesh sample. The scoring computation is repeatable for fixed trajectories; the active policy's achieved trajectories are not.

**Leading explanation [INFERENCE]:** GPU numerical nondeterminism is amplified during contact by a fragile closed-loop policy. This is consistent with the official [MuJoCo Warp determinism documentation](https://mujoco.readthedocs.io/en/stable/mjwarp/index.html#features), which states that GPU executions may differ because of atomic ordering. The evidence does not isolate a particular atomic operation or exclude all controller effects. The ONNX provider warning alone does not establish CPU fallback: the source explicitly requires CUDA provider availability and binds CUDA buffers.

**Decision:** Do not attempt to obtain deterministic scores by modifying the official evaluator. Quantify execution variability and treat poor contact robustness as a policy outcome. A single favorable rollout is not an experiment result.

## Reward design

Let `q` and `q_ref` be normalized Newton body/command quaternions in **xyzw** order. Form

```text
q_error = q_ref * inverse(q)
theta = 2 * atan2(norm(q_error.xyz), abs(q_error.w))     # [0, pi] radians
raw_object_orientation = -(theta / pi)^2                # [-1, 0]
orientation_weight = 2.0
per_step_orientation_contribution = 0.02 * 2.0 * raw_object_orientation
```

This is a bounded reward contribution (a penalty), zero at correct orientation. It avoids adding an arbitrary positive constant to every surviving step. Compared with the earlier linear-angle proposal, squared angle has a finite zero gradient at identity; its trade-off is a weaker signal close to perfect tracking. Weight 2 is a fixed first hypothesis, not an optimized value. Do not tune it within this experiment.

| Orientation error | Raw term | Contribution at 50 Hz |
|---|---:|---:|
| 0° | 0 | 0 |
| 30° | -1/36 | -0.00111111 |
| 90° | -0.25 | -0.01 |
| 180° | -1 | -0.04 |

For context, original object-position reward is at most +0.02 per step, force closure +0.10, body joints +0.10, and termination -6.0. The new term is not multiplied by `frame_dt` or its weight until normal reward composition. Replay stores the **raw** negative squared-angle value in float32.

### Conventions, gradients and symmetry

- Retargeted reference Parquet uses wxyz. `CommandBuffers` already converts it to xyzw. Read `command.body_target_quat_w` and `state.body_q`; do not convert them a second time.
- MuJoCo Warp's internal qpos convention is wxyz, unlike Newton body transforms. The local reward must not read internal qpos as xyzw.
- Normalize finite nonzero inputs. Invalid reference quaternions fail preflight; nonfinite simulator states stop the run. Do not silently replace invalid orientations with identity.
- Compute squared angle directly with an identity branch: for `s² = dot(q_error.xyz, q_error.xyz) < 1e-12`, use `theta² = 4*s²`; otherwise use `(2*atan2(sqrt(s²), abs(w)))²`. Clamp rounding to the mathematical range only.
- Analytically `d(raw)/d(theta) = -2*theta/pi²`. There is no unique geodesic derivative at exactly 180°; do not claim global smoothness. The small-angle branch avoids the unstable raw `acos` derivative near identity.
- FlashSAC does not differentiate through the simulator reward. Its actor gradient comes through the learned critic. Numerical derivative tests validate the shaping function; they do not prove useful policy gradients or learning success.
- Apply quaternion double-cover invariance (`q` and `-q`). Use **identity-only physical object symmetry** for this experiment. The stool is not assumed continuously yaw-symmetric, and ADD compares corresponding object points; treating different headings as equivalent would change the target.

**Math verified now:** 10,000 random pairs agree with SciPy squared geodesic angle to 3.6e-15; float32 raw-reward error is below 2.0e-7; sign/scale/common-frame invariance passes; finite-difference derivative error is below 1.2e-9. Identity, 30°, 90°, 180° and invalid-input cases pass. These tests use a throwaway host implementation, not the future Warp kernel.

## Implementation approach and alternatives

**Recommended:** append a local thirteenth raw term through `ObjectiveSpec` and export a separate portable actor. The original terms stay independently observable and replay relabeling remains honest.

Rejected alternatives:

1. Add orientation only to `env.reward`/`objective.score`: insufficient. Upstream FlashSAC recomposes one-step and sampled n-step rewards from raw objective terms and weights, so the learner would not optimize that scalar-only change.
2. Hide a composite pose reward under the existing `object_position` raw term: easier to keep a twelve-term config, but changes that term's meaning, obscures diagnostics, and makes training provenance misleading.
3. Switch to generic contact-wrench tracking: Episode 0 has binary contact labels but empty positions, normals and part IDs; that is a different experiment requiring data preparation.

### Planned files and interfaces

Only **new** files are proposed; no existing source file needs modification.

| File | Responsibility |
|---|---|
| `pipeline/flash_sac_orientation.py` | Local objective config/wrapper and Warp reward kernel |
| `pipeline/tests/test_flash_sac_orientation.py` | Quaternion and raw-term/composition checks |
| `pipeline/flash_sac_actor_export.py` | Explicit evaluation-config projection and byte-preserving actor export |
| `pipeline/tests/test_flash_sac_actor_export.py` | Projection boundaries, metadata and tensor identity |
| `scripts/train_orientation_flash_sac.py` | Local launcher using upstream scene, learner and checkpoint APIs |
| `scripts/run_orientation_experiment.py` | Per-arm/seed segment scheduling, gates, manifests and storage preflight |
| `scripts/verify_orientation_flash_sac.sh` | Unit, zero-update export and official evaluator/packer verification |

Interfaces:

- `OrientationReconBodyObjectiveConfig.build(...) -> OrientationReconBodyObjective`: same binding signature as upstream `ObjectiveSpec.build`.
- `OrientationReconBodyObjective`: exposes `score`, `terms`, `weights`, `term_names`, `evaluate`, `reset`, and `set_weights` protocols.
- `project_evaluation_config(training_config: Mapping[str, object]) -> dict`: produces an upstream-only reconstruction config, allowing only reward/objective-schema projection.
- `export_official_actor(native_policy: Path, destination: Path, training_config: Mapping[str, object], provenance: Mapping[str, str]) -> Path`: creates a new file and refuses overwriting source or destination.
- Local launcher accepts one arm, one seed and one terminal gate; experiment orchestrator refuses implicit continuation beyond an unreviewed gate.

## Checkpoint and official-tool compatibility

The unchanged evaluator instantiates the objective/curriculum before loading actor tensors. A custom `_target_` or a thirteen-key curriculum therefore cannot be put directly into its runtime reconstruction config without the local package/schema being available.

Keep two artifacts at each gate:

1. **Native** learner state, replay and actor with the actual custom thirteen-term training configuration, suitable for the local launcher to resume.
2. **Portable actor** with identical actor tensor descriptors and bytes. `user.resolved_config_json` reconstructs standard upstream `ReconBodyObjectiveConfig` and twelve-term curricula, with all other observation/action/physics/reference/reset/termination settings preserved. Store the complete actual training config separately as `user.training_resolved_config_json`, together with the exact reward definition, source hashes, exporter hash, parent actor SHA and native state/checkpoint identity. Explicitly label the runtime projection as evaluation-only. Never claim the training reward was the original reward.

Projection changes only the objective specification and orientation key in the top-level and nested training curriculum copies. It preserves thresholds, stage VOC/reset settings and all nonreward configuration. Unexpected objective keys or physics differences cause export to fail. State/replay files are never projected or offered as policy artifacts.

**Verified format probe now:** a copy of the existing 10M actor with byte-identical tensors and additional proposal/provenance metadata was accepted by the unchanged official evaluator without importing any local training module. The unchanged supplied packer accepted its 778-row trajectory, verified frame conversion, and cross-checked **AUC, SP-SR, MP-SR and MPPE**. This is a deliberately scoped Episode 0 format probe; no upload occurred and it is not a complete submission. The probe does not establish that an unimplemented thirteen-term training wrapper exports correctly. That is a mandatory zero-update integration test before training.

The AUC dependency gap was closed for this probe by using an isolated copy of the image's existing pure-Python trimesh 4.12.2 under the diagnostic directory; existing virtual environments were not changed. The current repository HEAD URL is present only for local format validation: there are pre-existing uncommitted changes, so these probe CSVs are not reproducible submission artifacts.

## Exact experiment configuration

Pilot: matched seed **42**. Follow-up: add matched seeds **43 and 44**, giving three paired seeds total if the design remains unchanged. If the pilot leads to reward/configuration tuning, label it exploratory and use new seeds 43/44/45 for a fresh three-seed comparison instead.

| Setting | Both arms |
|---|---|
| Episode/reference | Episode 0, unchanged staged `tall_bar_stool_tip_up_04` G1 Parquet |
| Initialization | Fresh actor, critic, targets, optimizer, normalizer, replay and counters; no old policy warm-start |
| `reset.seed`, `training.seed` | Same seed within each pair: 42, then 43 and 44 |
| Worlds | 1024 |
| Total per arm | 40,000,512 aggregate environment transitions |
| Optimizer ratio | 2.0 updates/collection; 2/1024 = 0.001953125 updates/environment step after warmup |
| Replay | capacity 500,000; minimum 50,000; batch 2048; n-step 7; uniform; observation storage float16 |
| Replay on stage change | Retain (`clear_on_stage_change=false`) |
| Discount | 0.99 |
| Actor | 2 blocks, width 256, residual zero-dense mean head, tanh, scale 10, normalized initial std 0.05 |
| Critic | 2 blocks, width 512, two critics, 101 atoms |
| Mixed precision | Enabled, compute dtype float16 |
| Learning rate | Initial/peak 0.0003, end 0.00015; fixed horizon **683,592 optimizer updates** |
| Other optimizer settings | Upstream Adam 0.9/0.999, epsilon 1e-8; actor update period 2; tau 0.01; BC coefficient 0 |
| Reward normalization | Enabled, upstream epsilon/moment settings and bound 5; fresh in both arms |
| Exploration | Initial temperature 0.01; target sigma 0.07; policy warmup actions; repeat max 16 |
| Physics | 50 Hz control / 200 Hz physics, friction 1.0, root damping 0.01, support false; unchanged capacities/solver settings |
| Curriculum | Original absolute thresholds and reset mixture, never compressed |
| Original objective | Original twelve terms and weights |
| Orientation arm | Same twelve terms plus `object_orientation: 2.0`, raw `-(theta/pi)^2`, every stage |
| Logging | Local only; W&B disabled |

The complete baseline dictionaries are saved as `proposed_original_seed_42.json`, `_43.json` and `_44.json` in the diagnostic folder. The only intended behavioral difference between arms is the added reward term; the thirteen-term storage/logging schema is an implementation necessity.

### Stop/evaluate/resume protocol

Use four identical-budget segments in each arm. The fixed LR horizon prevents changing a segment endpoint from changing learning-rate decay. Set checkpoint interval 1,000,000,000 (larger than the experiment) and `save_replay=true`, so upstream saves full terminal artifacts at each requested gate without redundant intermediate replay dumps.

| Gate | Exact step count | Exposure | Required evaluation |
|---|---:|---|---|
| 10M | 10,000,384 | Stage 0 | Five fresh-process official, unassisted world-1 rollouts |
| 20M | 20,000,768 | Stage 0 | Same five-repeat protocol |
| 35M | 35,000,320 | Last collection was Stage 0 | Same five-repeat protocol; verify next segment selects Stage 1 |
| 40M | 40,000,512 | About 5M transitions at VOC 0.75 | Same protocol and final paired comparison |

At the 35M exact boundary, checkpoint metadata may correctly describe Stage 0 for the last completed collection. Stage 1 applies to the next collection, not retrospectively to the saved policy.

Both arms begin from scratch only at the first segment. Later segments restore the **same arm's** learner state, optimizer/normalizer and matched replay file (checkpoint-set and step identities must agree). Do not refill empty replay or warm-start from the other arm. The upstream checkpoint does not serialize the full live physics/reset/SONIC history; a new segment resets environments. This experiment deliberately includes matched environment restarts in both arms and must not be described as identical to uninterrupted 40M training.

Each segment gets a new output directory, preserving all native and portable checkpoints. Alternate arm execution order by gate, and reverse initial arm order in seed 43, to reduce systematic time/thermal ordering effects. GPU jobs remain sequential.

### Evaluation and decision gates

- Run the unchanged official evaluator with world count 1, explicit reset, start frame 0, deterministic actor and no assistance/settling. Do not alter reset mode to improve scores. The evaluator fixes its reset seed to 42; training seeds remain the paired seed.
- Five fresh-process evaluations per checkpoint; retain every trajectory/JSON/log. Report mean, median, IQR, standard deviation and min/max for all metrics, not the best repeat. Track orientation/time, early ADD disturbance and late tip-up as separate diagnostics.
- First gates enforce integrity: correct dimensions and provenance, exact counters, finite observations/rewards/losses, correct replay restore, no unexpected curriculum/configuration changes, and official export/packing success. Hard failures stop both arms at the same budget for inspection.
- A quality result at 10M/20M/35M is a progress report, not permission to change reward weight, reset schedule or seeds. No arm gets extra steps because it looks promising.
- Predeclare primary comparison as mean ADD-AUC at 40M; report mean MPPE, orientation error, all success rates and execution-variance/unstable-rollout rates alongside it. AUC gains alone do not establish successful tip-up.
- If the paired effect is unresolved relative to execution variability, extend **both checkpoints'** evaluations from five to twenty repeats before making a quality decision. Use a fixed analysis RNG seed 20261021 for any bootstrap intervals. Do not treat repeated executions as additional training seeds, and do not claim narrow asymptotic uncertainty from three seeds.
- For a useful manipulation claim, seek a repeat-distribution gain over the inert AUC (~0.3782), improved orientation and MPPE, no increased catastrophic excursions, and visible sustained tipping. The original arm can improve after assistance reduction; compare against it at the same gate.
- Single-rollout repeatability is currently inadequate. Numerical determinism is not promised. Before authorizing the pilot, accept the repeated-evaluation/variance protocol explicitly or finish a more precise local diagnosis; do not silently substitute an evaluator patch or a new simulator.

## Measured and projected resource budget

Historical steady training rate: **16,171 environment transitions/s**, checked against the saved source-training log. Current measured official evaluation: **38.30 seconds/run** mean across five fresh processes, with a copied warm Warp cache. These are laptop measurements, not A100 estimates. The orientation kernel and complete resumed-gate training overhead are unmeasured.

| Scope | Aggregate transitions | Steady training | Five repeats at four gates | Subtotal before training startup/checkpoint I/O |
|---|---:|---:|---:|---:|
| Pilot, seed 42, two arms | 80,001,024 | 1.374 h | 40 evals = 0.426 h | 1.800 h |
| Follow-up, seeds 43/44, four new arms | 160,002,048 | 2.748 h | 80 evals = 0.851 h | 3.599 h |
| Three seeds total, six arms | 240,003,072 | 4.123 h | 120 evals = 1.277 h | 5.399 h |

Planning allowance: approximately **2–2.5 GPU-hours for the paired pilot** and **6–7 GPU-hours for three seeds total**, provided throughput remains close to baseline and five repeats suffice. These are estimates. Extending all gates to twenty repeats adds approximately 1.28 h for the pilot, or 3.83 h for all three seeds, at the measured evaluation rate. Measure the new objective before replacing these estimates with claims.

Memory: the previously measured original configuration peaks around **6,825 MiB**. A thirteenth float32 raw term adds exactly 4 B/transition: **2,000,000 bytes (1.91 MiB)** in a 500k replay ring, plus small pending/sample/logging allocations. That calculation is not a measured peak for the new kernel. Preserve the 1024-world configuration only if a build/short approved benchmark confirms adequate headroom.

Checkpoint storage: original/orientation replay payloads including pending windows are approximately **1.059 / 1.061 GiB per gate**. Existing state and policy files measure 0.149 / 0.00453 GiB. Four gates, retaining native states, replays and both actor forms, require approximately **9.74 GiB per paired seed** and **29.22 GiB for three seeds** before logs/figures/cache overhead. Available filesystem space measured about **16.4 GiB**. The pilot fits with roughly 6.7 GiB remaining; the three-seed follow-up requires additional storage. Do not delete old checkpoints or silently omit replay to fit the follow-up.

## Review focus

1. A scalar reward-only patch would leave FlashSAC targets unchanged: prove thirteen raw terms reach n-step replay and relabeling.
2. xyzw/wxyz or reference-frame misalignment could reward the wrong rotation: use Newton command buffers and known-angle/two-frame tests.
3. The official evaluator could import a custom target or reject thirteen weights: test portable export without the local training package.
4. Gate resumes could reset schedules or lose replay: verify exact state/replay identities, fixed LR horizon and matched restart protocol.
5. A lucky rollout could be mistaken for learning: retain all repeats, predeclare statistics, and separate execution uncertainty from seed variability.

## Implementation tasks, after plan review

### Task 1: Local objective and raw-reward integration

**Create:** `pipeline/flash_sac_orientation.py`; `pipeline/tests/test_flash_sac_orientation.py`.

**Consumes:** upstream `ReconBodyObjectiveConfig`, objective protocols, command/state/reset buffers. **Produces:** the local config/wrapper interfaces above.

- [ ] Write tests for identity/sign flips, known angles, nonunit valid inputs, invalid inputs, common-frame invariance, and 180° finite values without claiming a unique gradient there.
- [ ] Verify the tests fail because the local module does not exist.
- [ ] Implement the wrapper: delegate original evaluation/reset; append world-major thirteen-term storage; keep the first twelve terms unchanged; expose mutable thirteen-entry weights; compose score with `frame_dt` exactly once.
- [ ] Implement the float32 Warp angle-squared kernel with the stated identity branch and command quaternion binding.
- [ ] Verify fixed-state first-twelve-term equality and `theta=90° -> raw=-0.25 -> weighted frame contribution=-0.01` at world counts 1 and 1024.
- [ ] Verify a two-frame reference switch uses the command target for the current transition; verify reset and termination transitions publish thirteen finite terms.
- [ ] Verify n-step stored objective terms and `_relabel_rewards` include the thirteenth term. A score-only test is insufficient.
- [ ] Run host math tests and container Warp tests, recording exact commands/results. No training yet.

### Task 2: Portable actor export

**Create:** `pipeline/flash_sac_actor_export.py`; `pipeline/tests/test_flash_sac_actor_export.py`.

**Consumes:** native policy, actual full training config, provenance. **Produces:** the projection/export interfaces above.

- [ ] Write failing tests for top-level/nested projection, preserved nonreward settings, extra-key rejection, overwrite refusal, source preservation and byte-identical actor tensors.
- [ ] Implement the explicit projection with all actual training details retained separately; never project learner/replay artifacts.
- [ ] Verify no custom module target remains in the runtime evaluation config, while the preserved training config retains the new reward and thirteen-term schema.
- [ ] Generate a **zero-update** native learner/actor through Task 1 and export it. Verify official evaluation and all four packer metrics without local training modules on `PYTHONPATH`.
- [ ] Use an explicitly scoped Episode 0 roster for this format test, never claim a full submission. No upload.

### Task 3: Exact local launcher and segment/gate orchestration

**Create:** `scripts/train_orientation_flash_sac.py`; `scripts/run_orientation_experiment.py`; corresponding launcher/config tests.

**Consumes:** Task 1 objective, Task 2 exporter and the saved experiment spec. **Produces:** resolved arm/seed/gate configurations and new output manifests.

- [ ] Write failing configuration tests: only the prescribed reward/schema differences, all four exact gates, matched initialization seeds, fixed LR horizon, strict replay restore, existing-output refusal and storage preflight.
- [ ] Implement composition using NVIDIA's public `setup_scene`, `RLEnv`, `WarpRLEnv`, learner and checkpoint APIs; baseline uses the original objective. Never edit or monkey-patch NVIDIA source.
- [ ] Verify paired initial actor/critic/target/optimizer values match before stepping, excluding only objective-width-dependent buffers.
- [ ] Implement terminal segment saving, resume identities, local metadata/export hooks and serialized gate evaluation. Restore state/replay within an arm; never cross arms.
- [ ] Test interrupted/incorrect replay paths, phase-end versus next-stage metadata at 35M, full actual-config provenance, and no automatic continuation past a gate.
- [ ] Produce dry-run commands/configs only. A later approved minimal benchmark measures new-kernel throughput/VRAM and resume/export behavior; record it as a benchmark, not experimental evidence.

### Task 4: Verification and pilot release gate

**Create:** `scripts/verify_orientation_flash_sac.sh`; new validation report.

- [ ] Run Tasks 1–3 tests, zero-update checkpoint export, unchanged official evaluator and packer checks.
- [ ] Recheck input hashes, upstream cleanliness, disk headroom and that no old artifact has been overwritten.
- [ ] Freeze the code/config hashes, weight, seeds, evaluation aggregation and output roots before the first experimental step.
- [ ] Present the implementation validation and measured resource update. Obtain the user's training authorization before launching either substantial arm.
- [ ] Execute the paired pilot and later three-seed follow-up only after that authorization; perform no commit/push/submit action under this plan.

## Reproduce completed diagnostics

```bash
cd /home/ibrahim/v2d-track2-robotic-grounding
MPLCONFIGDIR=/tmp/v2d-plan-mpl .venvs/episode0/bin/python \
  outputs/step_D/orientation_plan_20261021/analyze_diagnostics.py
git -C official/video_to_data status --short
```

This command only rescores saved trajectories and redraws diagnostic figures. It does not rerun evaluation or training. Per-run exact Docker commands, timings, checkpoint/image identities and control flags are recorded in the diagnostic manifests.
