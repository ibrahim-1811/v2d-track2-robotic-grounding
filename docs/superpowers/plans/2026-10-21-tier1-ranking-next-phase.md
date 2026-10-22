# Tier 1 Ranking Improvement — Next Phase Plan

**Status: planning only, 21 October 2026 (Europe/Berlin).** No new training, cluster jobs, commits, pushes or submissions are authorized or launched by this document. User currently has no larger GPU available; a friend's cluster is an optional execution venue after its hardware is checked. A local inference-only Viser viewer may run independently.

**Goal:** Improve the same all-30-episode Tier 1 submission across AUC, SP-SR, MP-SR and MPPE, aiming for first place without assuming that a particular reward change will succeed.

**Architecture:** Keep NVIDIA physics, embodiment, evaluator and packer unchanged. Establish task-specific original-reward controls, then compare a local orientation-aware objective under matched training/evaluation conditions. Expand a demonstrated improvement across tasks before using a large training budget.

**References/specification:** completed [curriculum audit](../../../outputs/step_D/curriculum_audit/REPORT.md), [orientation implementation plan](2026-10-21-orientation-aware-flashsac.md), [scored submission](../../../outputs/step_D/submission_trial_20261021/REPORT.md), and [cluster handoff](../../../outputs/step_D/cluster_handoff_20261021/README.md). The handoff is a draft for the friend, not a claim that cluster execution has passed.

## Current gap and useful targets

Live Tier 1 leaders checked at 03:12 Europe/Berlin on 21 October 2026:

| Metric | Our first trial | Current leader | Aspirational next objective |
|---|---:|---:|---:|
| AUC ↑ | 0.17520 | 0.96908 | Above 0.96908; aim >=0.97 |
| SP-SR ↑ | 0 | 0.86666 | At least 27/30 successful episodes (0.90) |
| MP-SR ↑ | 0 | 0.83333 | At least 26/30 successful episodes (0.8667) |
| MPPE ↓ | 64.78526 cm | 7.44319 cm | Below 7.44319 cm; aim <7.4 cm |

Sources: [AUC](https://www.kaggle.com/competitions/v2d-challenge-track2-tier1-auc/leaderboard), [SP-SR](https://www.kaggle.com/competitions/v2d-challenge-track2-tier1-sp-sr/leaderboard), [MP-SR](https://www.kaggle.com/competitions/v2d-challenge-track2-tier1-mp-sr/leaderboard), [MPPE](https://www.kaggle.com/competitions/v2d-challenge-track2-tier1-mppe/leaderboard). Different teams lead different metrics. These targets are goals, not forecasts, and standings change.

Our trial reused one incomplete 10M Episode 0 actor for all 30 tasks. It is not evidence about a fully trained task-specific FlashSAC solution. Current performance therefore cannot isolate reward weakness from insufficient training, assistance dependence, small replay, reference feasibility or poor transfer.

## The 40M limitation

The existing paired 40M plan remains a useful early diagnostic. At 40M, the original schedule is only in Stage 1, with target VOC **0.75**. Playback VOC first reaches zero at **209,999,872** steps. Immediate frame-zero reset exposure increases around **288.75M** and **306.25M**; the recipe ends at **349,999,104**. Ordinary sampled resets can still use assisted settling at zero playback VOC.

Do not call 40M complete training, treat an assisted replay as learned success, or reject the original curriculum solely from early unassisted failures. Any shortened assistance schedule is a separate experimental variable and must have its own matched control. Do not simultaneously change reward, replay and curriculum and attribute the result to orientation.

## Phase 0 — preserve the result and make the next run measurable

- [x] Preserve the submitted CSV, four COMPLETE receipts, canonical scores, exact actors, code commit and input hashes.
- [x] Provide an inference-only Viser launcher for the submitted checkpoints; explain that object-only exports cannot reconstruct the exact submitted full robot motion.
- [ ] Review Episodes **0, 21 and 7** at the dataset/reference/policy layers: object trajectory and orientation, wrist/palm contact timing, hand actuator authority, joint limits, support collision shapes, reachability and penetrations. Episode 7 is an iron that must slide on a table and stand upright; support is required there. These three are a diversity check, not proof of representativeness of all 30.
- [ ] Use the unchanged official evaluator for all quality decisions. Retain five independent executions per checkpoint and all trajectories; summarize means, medians, spread and catastrophic excursions. Do not confuse execution repeats with training seeds or choose a lucky repeat for submission.
- [ ] Obtain the friend's hardware probe before a cluster budget or training configuration is finalized.

## Phase 1 — implement the smallest controlled reward experiment

The local thirteen-term reward/launcher/exporter work in the existing orientation plan is **not implemented**. Keep that work separate from the cluster control commands. Its deliverables are:

- [ ] Bounded squared quaternion-geodesic penalty `-(theta/pi)^2`, initial weight 2, with wxyz/xyzw, sign invariance, normalization, finite endpoint values and symmetry tests.
- [ ] Original twelve terms unchanged; thirteenth **raw term** stored in n-step replay and relabeled by FlashSAC. A scalar reward-only patch would not implement the intended learner objective.
- [ ] Native state/replay checkpoints retain true training config. Portable actor export retains identical actor tensors and explicit custom-training provenance while loading in the unchanged official evaluator.
- [ ] Validate reward scale, memory and throughput before any substantive pilot. Unit/config/export tests come first; GPU kernel behavior remains to be measured.

Original long training, the added orientation term, and reference-directed contact/wrench matching are three distinct approaches. Start with the smallest reward intervention against a stronger control. Contact/wrench guidance is a later candidate if pose rewards still produce ineffective contact; our current reference lacks the geometry needed for a valid drop-in wrench objective. Preserve G1 body tracking when testing any alternative. See the audit for the missing-contact-geometry limitation and the official [CHORD method](https://nvidia-isaac.github.io/video_to_data/chord/).

## Phase 2 — bounded pilot, then meaningful later-stage comparison

1. **Hardware/config qualification:** friend returns probe, then a small approved smoke and a post-warmup benchmark. A 2,097,152-transition benchmark exceeds the full recipe's 409,600-transition replay minimum. Record actual optimization throughput, startup and checkpoint I/O separately.
2. **Early comparison:** original versus orientation-aware, from scratch, matched seed/worlds/replay/update ratio/reset/physics and 10M/20M/35M/40M gates. The laptop plan uses 1024 worlds/500k replay/2 updates per collection; the cluster draft uses 2048/10M/4. Both have the same optimizer-update/environment-step ratio. Do not compare different hardware/config arms as an isolated reward test.
3. **Repeatability gate:** compare repeat distributions and visible task completion. If a difference is unclear relative to execution variability, add evaluation repeats, not arbitrary extra training to the favored arm. No universal numerical gain threshold is invented before measuring the pilot's variability.
4. **Later-stage decision:** if mechanics are correct but early quality remains unresolved, propose matched extensions through a zero-playback-VOC stage and the final reset stages, up to the full 349,999,104 budget where needed. Extensions require a separately reviewed budget; no job silently continues beyond 40M.
5. **Diversity and seeds:** test Episodes 21 and 7, then fresh training-seed pairs. If reward/settings were tuned after exploratory seed 42, use fresh seeds for confirmatory comparisons. Transfer from Episode 0 must earn its place against scratch policies.

The cluster control command supports four exact absolute gates at 2048 worlds: 10,000,384; 20,000,768; 35,000,320; **40,001,536**. The prior 1024-world 40,000,512 budget is not divisible by 2048. LR horizon remains the full-recipe 683,592 updates. Restore matching learner state and replay within the same arm; segmented environments restart, so both arms must use the same restart schedule.

## Phase 3 — improve the competition submission

- [ ] Train/validate task-specific controllers across all 30 episodes using the selected approach. A generalist or warm start is optional only after transfer is demonstrated; reusing the failed source actor is not the production plan.
- [ ] Group failures into reaching/contact, grasp retention, lifting/rotation, body balance, and reference/support problems. Fix the appropriate layer rather than changing all reward weights indiscriminately.
- [ ] Use pose errors that cover both translation and rotation; consider canonical mesh-point tracking only as a separately tested extension. Account for object size and scorer orientation conventions. Symmetry handling must match the metric; visual symmetry alone does not erase a quaternion error in SP-SR/MP-SR.
- [ ] Aim first to exceed CHORD Baseline (AUC 0.51038, SP-SR 0.70, MP-SR 0.20, MPPE 17.73644 cm at the check), then the live leaders. Success rates move in increments of 1/30 for these single-world per-episode results; systematic task completion matters.
- [ ] Freeze one checkpoint per episode using predeclared multi-metric local criteria. Use the **same CSV** for all four competitions. Do not submit different robot trajectories separately to optimize individual metrics, fill missing episodes with references, or pick a lucky execution.
- [ ] Re-run the full official packer/dry-run and preserve provenance before the next authorized upload. Other Track 2 tiers are separate projects with their own inputs; Tier 1 ranking is not a complete three-tier Track 2 result.

## Resource budget and practical limits

Reference arithmetic at our measured 16,171 steps/s (8 GB configuration), before new-kernel overhead:

| Scope | Training time | Interpretation |
|---|---:|---|
| Two 40M arms, one episode/seed | 1.37 GPU-h | About 2–2.5 h including planned repeated eval/startup |
| Same paired pilot, three seeds | 4.12 GPU-h training | About 6–7 h including planned eval/overhead |
| One 350M task | 6.01 GPU-h | Laptop-size replay; not a measured full-replay cluster run |
| 30 full-budget tasks, one tier/seed | 180.36 GPU-h | Planning arithmetic only; validate actual convergence |
| 90 tasks, all three tiers/seed | 541.09 GPU-h | Separate input preparation and validation still needed |

Cluster time must be recomputed from its **measured** post-warmup throughput: `hours = episodes * arms * seeds * transitions / (steps_per_second * 3600)`, plus evaluation, compilation and checkpoint I/O. No rental, allocation or monetary budget has been authorized. The official challenge freeze is 4 November 2026 at 17:00 EST, which is 23:00 Europe/Berlin; large sweeps must leave time for diagnosis and submission.

Storage is now a real constraint: laptop free space was about **12 GiB** before making the transfer archive. A 500k paired seed retains about 9.74 GiB across the four gates; three seeds need about 29.22 GiB. Add storage or offload copies with verified hashes before training; preserve current artifacts.

At cluster replay capacity 10M, each replay is about 20.92 GiB. Four gates require about **85 GiB per episode/arm/seed**. Plan scratch and transfer bandwidth explicitly; the handoff retains all files and does not silently drop replay. Do not send a friend a multi-episode job before confirming the quota.

## Handoff and stop state

The recipient guide, manifests, probe, original-arm scripts and return checklist are in `outputs/step_D/cluster_handoff_20261021/`. The transfer archive contains inputs and no credentials or old replay buffers. The friend starts with hardware information, then the team reviews the first approved compute work. Orientation training waits for its implementation and tests.

Training remains stopped. No automatic continuations, paid cloud jobs, new commits/pushes or further submissions are part of this plan.
