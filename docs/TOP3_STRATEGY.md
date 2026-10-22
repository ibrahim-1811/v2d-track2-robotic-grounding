# Getting into the Tier 1 top 3

2026-10-22, **13 days to the 4 Nov 17:00 EST freeze.** Strategy only — nothing
trained, committed or submitted under this document.

## 1. Where we stand

Tier 1 AUC leaderboard, 17 ranked teams (downloaded 2026-10-21):

| # | team | AUC | SP-SR |
|---:|---|---:|---:|
| 1 | Vi Nguyễn Văn | 0.96908 | *no entry* |
| 2 | DAP | 0.73467 | 0.83333 |
| 3 | Xueyi Wang (FF) | 0.59530 | 0.86666 |
| 4 | maniphys.ai | 0.58258 | 0.83333 |
| 5 | **CHORD Baseline** | **0.51038** | 0.70000 |
| 6 | wte09fl | 0.42955 | 0.53333 |
| 16 | **us (Izus_systems)** | **0.17520** | 0.00000 |

Rank 1 posted no SP-SR/MP-SR, so it is a partial entry and not a like-for-like
target. **Top 3 means beating 0.59530.**

Two reference points that matter more than the ranking:
- **Doing nothing scores 0.26606** — our submitted entry is worse than inaction
  on 26 of 30 episodes.
- **CHORD Baseline is NVIDIA's own method on their own recipe: 0.51038.**
  It is the number to reproduce before trying to beat anyone.

## 2. The central finding: we have never run NVIDIA's recipe

From the CHORD paper (arXiv 2607.00033), the whole-body configuration is
exactly our `g1_recon_body_flash_sac`: FlashSAC, G1 with Dex3 hands, the
*reduced force-closure* objective (chosen because video-reconstructed finger
contacts are too noisy for full wrench matching), VOC annealing, reference-state
resets. **We are running the right method.** The paper reports **≈2 hours per
task on a single L40S at 2,048 environments.**

What we actually ran, forced by 8 GB of laptop VRAM:

| setting | NVIDIA shipped | ours | deviation |
|---|---:|---:|---|
| `scene.world_count` | 2,048 | 1,024 | 0.5× |
| `replay.capacity` | 10,000,000 | 500,000 | **0.05× (20× smaller)** |
| `updates_per_collection` | 4.0 | 2.0 | 0.5× |
| VRAM | ~48 GiB (README) | 7.2 GiB | — |

The replay ring alone is 10,000,000 × 2,246 B = **20.92 GiB**, which is 2.6× our
whole card. FlashSAC is off-policy; cutting its replay 20× changes the algorithm,
not just the budget.

**Our Episode 0, trained the full 350M steps on the crippled config, scores
0.3491 against a 0.3769 stationary floor. CHORD averages 0.510 across the tier.**
The most probable explanation for that gap is the configuration, not the method,
and it has never been tested because we have never had the memory to test it.

This is the single highest-information experiment available, and the cluster is
what unlocks it.

## 3. Plan

### Phase 1 — Cluster qualification (days 1–2). One run decides everything.

1. Hardware probe: `outputs/step_D/cluster_handoff_20261021/cluster_probe.sh`.
   **Requirement: ≥48 GiB VRAM per GPU** (L40S, A100-40/80, H100). Below that,
   the real recipe does not fit and the whole plan changes.
2. Image/asset validation and a 2-minute smoke.
3. **Episode 0, exact shipped recipe**: 2,048 worlds, 10,000,000 replay,
   4.0 updates/collection, 349,999,104 steps, stock 12-term objective, seed 42.
   Expect ≈2 h on an L40S.
4. Evaluate with 5 fresh unassisted world-1 repeats.

**Decision gate.** Episode 0's stationary floor is 0.3769 and our crippled run
reached 0.3491.
- **AUC ≥ 0.50** → the recipe was the problem. Go straight to Phase 2.
- **0.38–0.50** → recipe helps but is not sufficient; Phase 2 proceeds while we
  add the orientation edge.
- **< 0.38** → the configuration was *not* the explanation. Stop and re-diagnose;
  do not spend 60 GPU-hours on a recipe that cannot clear doing-nothing.

### Phase 2 — Tier 1 at scale (days 3–9)

30 episodes × ~2 h = **60 GPU-hours**; on 4 concurrent GPUs ≈ 15 wall-clock
hours, so there is room for 2–3 seeds on the hard episodes. All 30 G1
references are already staged and validated (Step C), so there is no data work
on the critical path.

Per-episode discipline, from the phase profile:
- **Group D (high stationary floor: ep 8 = 0.598, 3 = 0.539, 20 = 0.471,
  26 = 0.441, 6 = 0.404, 24 = 0.391)** — hardest to beat; give these the extra
  seeds. Episode 8 alone is worth more than several Group B episodes.
- **Group B (12 near-continuous-manipulation episodes)** — plain recipe, one seed.
- **16 of 30 episodes need `scene.support=false`**; the other 14 have real
  collision shapes. Get this wrong and FlashCHORD raises.

### Phase 3 — The edge beyond CHORD (days 9–12)

Reproducing CHORD lands around 0.51. Third place needs 0.60. Three sources of
the remaining ~0.09, in order of confidence:

**(a) Never ship a regression — worth up to +0.08 on its own.**
For every episode, compare the trained policy's local AUC against that episode's
stationary floor and keep the better trajectory. Our current entry loses to
inaction on 26 of 30 episodes; simply not shipping those regressions moves a
0.51 tier mean up wherever a policy underperforms. This costs no compute and
uses the local packer, so it is free to evaluate. *Caveat: a stationary
trajectory contributes nothing to SP-SR/MP-SR, and those are separate
leaderboards. It is a floor, not an ambition.*

**(b) An object-orientation reward term — the measured gap in NVIDIA's objective.**
The curriculum audit established that **none of the nine stages has a dense
object-orientation reward**. Track 2 ranks by AUC over 1–9 cm ADD thresholds,
and on a 0.55 m-radius object 9 cm is about **10° of rotation** — so AUC is
orientation-dominated while the optimised objective ignores orientation
entirely. That is a real, documented mismatch between NVIDIA's reward and the
ranking metric, and it is the most defensible place to beat the baseline.
Our 13-term implementation is built and verified (16/16), **but the first
attempt was mis-scaled**: `-(θ/π)²` with weight 2.0 contributed 1.51 % of
`object_position`. The fix is the exponential form upstream already uses for
pelvis and palm orientation, with σ chosen against *evaluation-time* error
(43–110° post-failure), not training-time error. Run it as a matched pair on
2–3 episodes before any tier-wide commitment.

**(c) Seeds and selection.** CHORD's own paper runs 5 seeds per sequence. Picking
the best seed per episode by *local* AUC is free score, and the local packer
means it costs no Kaggle submissions.

### What the laptop does meanwhile

It is not idle: the reset-distribution control/treatment pair (2.2 GPU-h,
already specified), evaluation sweeps, and instrumented failure diagnosis — all
while the cluster trains.

## 4. Submission discipline

- **Measure locally.** `tools/pack_submission.py` prints all four scores. Kaggle
  allows 5 per week; never burn one to learn something the packer would tell us.
- **Do not submit now.** With the new Episode 0 the tier mean is 0.18290 — still
  below the 0.26606 do-nothing floor.
- One CSV for all four leaderboards. Do not tune separate trajectories per metric.
- Submission numbers must come from `evaluation.world_count=1`.

## 5. Honest risks

| Risk | Impact | Mitigation |
|---|---|---|
| **Cluster has <48 GiB GPUs** | Plan's premise fails | Probe first, day 1. Fall back to the largest replay that fits and accept a weaker reproduction |
| Phase 1 does not clear 0.38 | The config was not the explanation | Hard stop and re-diagnose rather than spending 60 GPU-h |
| 13 days, 5 submissions/week | ~11 submissions total | Local packer for every decision; submit only on measured improvement |
| Episode 0 may not represent the tier | Group A is ~7 of 30 | Qualify on one Group A and one Group B episode before scaling |
| Execution nondeterminism | ±0.02 AUC per rollout | 5 repeats minimum, report spread, never a best repeat |

**The realistic ceiling.** Reproducing CHORD (0.51) plus not shipping regressions
plus a working orientation term plausibly reaches 0.58–0.65. That is third place
territory, and it depends almost entirely on Phase 1 clearing its gate.
Promising more than that from 13 days and an unverified cluster would be
dishonest.
