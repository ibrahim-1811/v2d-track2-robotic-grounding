# Onboarding: NVIDIA V2D Challenge 2026, Track 2

Read this first, then [CLAUDE.md](../CLAUDE.md) for the hard-won environment
facts. Every line of CLAUDE.md was learned by something breaking; it will save
you hours.

**Deadline: 4 November 2026, 17:00 EST.**

---

## 1. What the challenge is

Turn a human demonstration video into a robot policy. We get, per episode, a
retargeted reference trajectory (object pose + human motion); we must train a
Unitree G1 with Dex3 hands to reproduce the object's motion in simulation,
**unassisted**, and submit the achieved object trajectory.

**Track 2 has 3 tiers x 4 metrics = 12 separate Kaggle competitions.** Each tier
is a separate submission with its own 5-per-week budget. All three tiers are
scored against the *same clean multi-view reference*; only the **input** you
train from differs:

| Tier | Input | Status for us |
|---|---|---|
| 1 `tier_1_multiview_caption` | clean multi-view (= the scoring reference) | **30/30 G1 references staged and validated** |
| 2 `tier_2_synthetic_noise` | corrupted inputs | data mirrored, **0/30 references staged** |
| 3 `tier_3_cari4d` | your own reconstruction | **[UNRESOLVED]** — see §6 |

Tier 1 is the easiest: its input *is* the scoring reference, so there is no
input error to inherit. Tiers 2 and 3 are strictly harder.

The four metrics per tier: **AUC** (the headline), MPPE, SP-SR, MP-SR.
**One CSV goes to all four competitions within a tier.**

## 2. Where we stand

Tier 1 AUC leaderboard, 17 teams (2026-10-21):

| # | team | AUC |
|---:|---|---:|
| 2 | DAP | 0.73467 |
| 3 | Xueyi Wang (FF) | 0.59530 ← top-3 bar |
| 5 | CHORD Baseline (NVIDIA's own) | 0.51038 |
| 16 | **us** | **0.17520** |

(Rank 1 at 0.969 posted no SP-SR/MP-SR, so it is a partial entry.)

**Two numbers that matter more than the ranking:**
- **A frozen object scores 0.26606.** Our submitted entry is *worse than doing
  nothing* on 26 of 30 episodes.
- CHORD Baseline is NVIDIA's own method on their own recipe. Reproducing it is
  the first goal, not beating anyone.

## 3. The central hypothesis

From the [CHORD paper](https://arxiv.org/abs/2607.00033), NVIDIA's whole-body
configuration *is* our `g1_recon_body_flash_sac`: FlashSAC, G1+Dex3, the reduced
force-closure objective. **We are running the right method.** They report
**~2 hours per task on one L40S at 2,048 environments.**

But 8 GB of laptop VRAM forced us to cripple it:

| setting | NVIDIA shipped | our laptop | |
|---|---:|---:|---|
| `scene.world_count` | 2,048 | 1,024 | 0.5x |
| `replay.capacity` | 10,000,000 | 500,000 | **20x smaller** |
| `updates_per_collection` | 4.0 | 2.0 | 0.5x |
| VRAM | ~48 GiB | 7.2 GiB | |

The replay ring alone is 10M x 2,246 B = **20.92 GiB**, 2.6x our whole card.
FlashSAC is off-policy, so a 20x replay cut changes the algorithm.

**We have never run NVIDIA's actual recipe.** Episode 0 trained the full 350M
steps on the crippled config scores **0.3491** against a **0.3769** stationary
floor. CHORD averages 0.510. The cluster is what tests whether the gap is the
configuration.

**This is the first thing to do, and one run answers it.** See §7.

## 4. What is already built and proven

| Thing | Where | Status |
|---|---|---|
| Challenge mirror -> SOMA -> G1 reference pipeline | `pipeline/`, `scripts/run_tier_pipeline.py` | **30/30 Tier 1 validated** |
| Chunked retarget that beats the 8 GB OOM | `scripts/retarget_episode_g1.py` | works, 79 s/episode |
| 13-term orientation objective (no upstream edit) | `pipeline/flash_sac_orientation.py` | **16/16 verified** |
| Portable actor export for the official evaluator | `pipeline/flash_sac_actor_export.py` | verified both ways |
| Local metrics logging (losses, per-term rewards) | `pipeline/flash_sac_metrics_logger.py` | works |
| Repeated evaluation sweep + aggregation | `scripts/evaluate_orientation_sweep.sh` | works |
| Instrumented contact/failure diagnosis | `scripts/diagnose_contact_rollout.py` | works |
| Stationary-baseline scorer | `scripts/leaderboard_floor.py` | agrees with NVIDIA to 0.3% |

## 5. What we learned the hard way

Full detail in [TOP3_STRATEGY.md](TOP3_STRATEGY.md),
[ORIENTATION_AWARE_FLASHSAC.md](ORIENTATION_AWARE_FLASHSAC.md) and
`outputs/step_D/ep0_contact_diagnosis/REPORT.md`.

1. **Training metrics lie.** Our 350M run reported 7.85 deg object orientation
   error; the real unassisted rollout measures **53.7 deg**. Training episodes
   terminate at the failure point, so the metric never sees the failure. Always
   export the trajectory and score it.
2. **Always compare against the stationary floor**, per episode, not against
   zero and not against a previous bad policy.
3. **Episode 0 fails at one specific moment.** It tracks the static prefix to
   0.5 cm / 1.1 deg, makes contact exactly on the reference cue at frame 200
   with 122 N of force, then loses grip at frames ~292-323 and the stool is gone
   by 375. The contact is a **push (net/summed force 0.61-0.73), not an opposing
   grasp** - it can tip but cannot retain.
4. **The 13th reward term was mis-scaled.** `-(theta/pi)^2` at weight 2.0
   contributed **1.51 %** of `object_position`. The fix is the exponential form
   upstream already uses, with sigma chosen against *evaluation*-time error.
   `pipeline/flash_sac_orientation_exp.py` is built but untested on GPU.
5. **Single rollouts are not repeatable.** Five evaluations of one fixed
   checkpoint span AUC 0.113-0.129 and MPPE 47.6-189.6 cm. Always 5 repeats,
   always report spread, never a best repeat.
6. **The viewer restarts halfway because that is the termination firing.**
   `view_policy.py` keeps the training thresholds; the evaluator disables them.

## 6. Setting up from scratch

`git clone` gives you code and docs only. `data/`, `official/`, `outputs/` and
`.venvs/` are gitignored by design.

```bash
git clone git@github.com:ibrahim-1811/v2d-track2-robotic-grounding.git
cd v2d-track2-robotic-grounding
```

**a. NVIDIA upstream (1.5 GB, pinned - never modify it).**
```bash
mkdir -p official && cd official
git clone https://github.com/nvidia-isaac/video_to_data.git
cd video_to_data && git checkout 193382a && git lfs pull && cd ../..
```

**b. Docker images.** `flash_chord:latest` (21 GB) is all you need for training
and evaluation; build it from `official/video_to_data/.../flash_chord/docker/`.
`robotic-grounding:latest` (31 GB) is only needed to *re-stage* references.

**c. Python envs.**
```bash
python3 -m venv --system-site-packages .venvs/episode0
.venvs/episode0/bin/pip install pyarrow==15.0.2
python3 -m venv .venvs/kit                 # fully isolated, needs numpy>=2
.venvs/kit/bin/pip install -r <kit>/requirements.txt
```

**d. Data — THE BLOCKER.** The HF dataset
`nvidia/PhysicalAI-Robotics-Video2Data` now returns **HTTP 401 for every tier**,
including ones we downloaded successfully on 2026-10-20. It has been gated since.
`scripts/download_challenge_episodes.py` will fail without credentials.
**Ask Ibrahim for an HF token, or for the data directly.** What you need:

| Path | Size | Can you regenerate it? |
|---|---:|---|
| `data/v2d_challenge/` | 772 MB | only from HF |
| `data/adapter_out/` (the 30 G1 references) | 2.0 GB | yes, but needs the 31 GB retarget image + SOMA-X |
| `data/body_models/soma/` | 236 MB | HF `nvidia/SOMA-X`, pinned rev `466879a8…` |

**Shipping `data/adapter_out` directly is strongly recommended** - it is the
most valuable derived artifact and re-deriving it costs a 31 GB image plus hours.

**e. Verify.**
```bash
bash scripts/verify_step_C.sh                  # all 30 Tier 1 references valid
bash scripts/verify_orientation_flash_sac.sh   # 16/16, our own code
```

## 7. What to do, in order

### Task 1 — cluster probe (do this first, blocks everything)
```bash
bash outputs/step_D/cluster_handoff_20261021/cluster_probe.sh
```
Report: GPU model, **VRAM per GPU**, count, driver, CUDA, free scratch, whether
Docker or Singularity/Apptainer, and queue limits.
**The plan needs >=48 GiB per GPU.** Below that, say so immediately - the whole
strategy changes.

### Task 2 — one run that decides everything (~2 h)
Episode 0, **NVIDIA's exact shipped recipe**, nothing of ours:
`scene.world_count=2048`, `training.replay.capacity=10000000`,
`training.updates_per_collection=4.0`,
`training.total_environment_steps=349999104`, stock 12-term objective,
`scene.support=false`, seed 42.

Then 5 fresh unassisted **world-1** evaluations.

**Gate** (Episode 0 floor is 0.3769; our crippled run got 0.3491):
- **AUC >= 0.50** -> the config was the problem. Scale to 30 episodes.
- **0.38-0.50** -> helps but insufficient.
- **< 0.38** -> stop. Do not spend 60 GPU-hours. Re-diagnose.

### Task 3 — Tier 1 at scale (only after Task 2 passes)
30 episodes x ~2 h = **60 GPU-hours**. 16 of 30 episodes need
`scene.support=false`; the other 14 have real collision shapes - get this wrong
and FlashCHORD raises. Extra seeds for the high-floor episodes
(8 = 0.598, 3 = 0.539, 20 = 0.471, 26 = 0.441).

### Task 4 — Tier 2 (a second, less contested set of 4 leaderboards)
Stage and retarget Tier 2 (`pipeline/episodes.py` already supports it;
`data/adapter_out_tier_2` is its staged root), then train 30 more. Another
60 GPU-hours. **Submit Tier-2-trained policies only to Tier 2** - the rules say
to choose the tier whose input you used.

### Running in parallel on the laptop (Ibrahim)
Reset-distribution control/treatment pair, evaluation sweeps, failure diagnosis.

## 8. Rules we hold ourselves to

- **Never modify `official/`.** Our code imports upstream; it never patches it.
  `git -C official/video_to_data status` must stay clean.
- **Never modify the official evaluator or the scoring.**
- **Measure locally.** `tools/pack_submission.py` prints all four scores. Kaggle
  allows 5/week per competition; never burn one to learn what the packer knows.
- **Submission numbers come from `evaluation.world_count=1`** - the packer
  scores world 0 only.
- **Never present an assisted replay as policy success.** Stage 0 trains at
  `voc_scale 1.0`; evaluation forces 0.0.
- **Never invent a measurement.** Mark conclusions
  [VERIFIED] / [DOC] / [GAP] / [UNRESOLVED].
- Do not commit, push or submit without Ibrahim's go-ahead.

## 9. Map of the docs

| Doc | What it answers |
|---|---|
| [CLAUDE.md](../CLAUDE.md) | every environment trap, learned by breaking something |
| [TOP3_STRATEGY.md](TOP3_STRATEGY.md) | the plan to reach top 3, with gates and risks |
| `outputs/step_D/ep0_contact_diagnosis/REPORT.md` | why Episode 0 fails at frame 323 |
| `outputs/step_D/ep0_contact_diagnosis/PHASE_PROFILE_AND_EXPERIMENT.md` | all 30 episodes profiled; the next experiment spec |
| [ORIENTATION_AWARE_FLASHSAC.md](ORIENTATION_AWARE_FLASHSAC.md) | the 13-term reward: design, tests, measurements |
| [ORIENTATION_RUN_COMMANDS.md](ORIENTATION_RUN_COMMANDS.md) | how to launch and watch a training run |
| [TIER_REFERENCES.md](TIER_REFERENCES.md) | how the 30 references are built and validated |
| `outputs/step_D/curriculum_audit/REPORT.md` | all 9 curriculum stages, all 12 reward terms |
