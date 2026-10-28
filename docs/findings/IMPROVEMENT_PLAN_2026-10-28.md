# What stops us from winning Track 2, and what to do in the time left

2026-10-28, ~11:30-13:00 CET, cluster login node. Propose-only: no Slurm job was submitted for this.
Marks: **[VERIFIED]** measured here (command in §9) · **[DOC]** stated by organisers/upstream/paper ·
**[GAP]** not known yet · **[UNRESOLVED]** open question · *estimate* = a projection, not a measurement.

Rule discipline (CLAUDE.md): no Tier 2/3 decision below uses the Tier 1 caption. Every Tier 2/3 number
here is either measured against that tier's own input (Tier 2 noisy/recovered reference, our Tier 3
reconstruction) or comes from the public board. Picking between *whole* submissions by local pack or
Kaggle score stays allowed.

## 0. The ranked list

Rank-point deltas are *estimates* from the live board (§1) with our Tier 3 at a "mid" level and MBZUAI
assumed to enter Tier 3. Lower track total is better; we are at 51 (3rd) in that scenario.

| # | Lever | Tier / metric | Expected gain | GPU-h | When | Risk |
|---|---|---|---|---|---|---|
| 0 | **Select final submissions by hand in all 12 metric competitions** | all | protects ~6 rank points on Tier 2; avoids a rules problem | 0 | before freeze | none |
| 1 | **Tier 3 complete and uploaded before the maintenance** (in flight: 372217-372244) | T3 all | unranked -> ranked | ~50 (queued) | upload by Oct 30 12:00 | PC/rsync done; only queue + pack left |
| 2 | **Tier 2 second sample per episode, best-of-2 by own-reference eval** (`ARM=keypoints SEED=43` on the t2 input; t2c smooth wave already running) | T2 all four | -4 to -10 (if 2-3 lost episodes come back) | pilot 13, full 100 | pilot now, full wave by Oct 29 noon | low: same pipeline, Tier-2-only selection |
| 3 | Tier 1 smooth stragglers (already queued: 19, 21, 22, 23, 26, 27, 29) | T1 AUC | -1 (AUC >= 0.8237 = 3rd) | 0 extra | lands today | none |
| 4 | Tier 3 second sample per episode (`SEED=43`), best-of-2 by own-reference eval | T3 all | -2 to -4 | 100 | Oct 29 or after maintenance | low |
| 5 | Tier 1 drop episodes 1/4/25: ADEPT-style post-training (actor warm start + critic warm-up + conservative actor LR + BC-to-replay) | T1 MPPE | -3 if ep25 MPPE reaches its do-nothing level (~50 cm) | ~12 | after maintenance | high: ten arms already failed here; needs ~1 day of tooling |
| 6 | Tier 3 object-track repair ("t3f": persistence-gated slew limit, 30 deg / 0.15 m per frame) on the 14 affected episodes | T3 AUC/SR | *unknown*, possibly -2 to -4 | 50 | after maintenance | medium: whole-pack comparison only; causal link not shown (§4.3) |
| 7 | Training jobs ask for 128 GB but use ~7.5 GB -> 3 jobs per 4-GPU node | ops | +33% concurrent training (18 -> 24) | - | now | low |
| - | t2c release-snap blend | T2 | small (15 events, 12-29 deg; ep4 91 deg) | 100 | not before #2 | the large jumps are raw Tier 2 steps, §3.2 |
| - | Full ADEPT recipe (reposing pre-training) | - | - | weeks | no | code not released, §5 |

Honest headline: **if MBZUAI-RCL submits a reasonable Tier 3, they win Track 2 whatever we do** (rank
sums 14-18 vs our best case ~41). Without MBZUAI in Tier 3 we are 1st-3rd depending mostly on Tier 2
and on our Tier 3 quality. First place is in reach only through Tier 2 + Tier 3; Tier 1 is already
our best tier.

## 1. Where we stand (board 2026-10-28 09:01 UTC) [VERIFIED]

Source: `https://raw.githubusercontent.com/MVerghese/v2d-leaderboard-data/main/leaderboard.json`
(`generated_at 2026-10-28T09:01:15Z`, before the 11:15 CET Tier 1 upload). AUC / SP-SR / MP-SR / MPPE:

| Tier | Leader | 2nd | Us on the board | Us, rules-clean planning numbers |
|---|---|---|---|---|
| 1 | MBZUAI-RCL 0.8935 / 0.9667 / 0.9333 / 2.78 | Chivatam 0.8305 / 0.90 / 0.8333 / 6.72 | 0.8053 / 0.8667 / 0.8667 / 10.45 | upload #3: 0.8122 / 0.9000 / 0.8667 / 8.35 |
| 2 | MBZUAI-RCL 0.5677 / 0.80 / 0.80 / 11.52 | Xueyi Wang 0.5227 / 0.6333 / 0.60 / 9.28 | 0.4807 / 0.70 / 0.50 / 20.98 | t2 entry: 0.4419 / 0.600 / 0.500 / 23.33 |
| 3 | nima arki 0.3294 / 0.40 / 0.10 / 42.30 | maniphys.ai 0.2259 / 0.333 / 0.10 / 49.94 | - | none yet |

- **Each metric is its own Kaggle competition** (kit `LEADERBOARDS.md`; `submit/v2d_submit.py`
  "upload a packed file to every metric competition") and **the pooled board shows our best score per
  metric across uploads**: our Tier 1 row combines upload #2's AUC with upload #1's MPPE 10.45; our Tier 2
  row is t2r's AUC/SP/MPPE with t2's MP-SR. [VERIFIED]
- So **the board currently overstates our Tier 2 by 6 rank points** (board row ranks 4+3+4+9 = 20; the
  rules-clean t2 entry ranks 7+6+4+9 = 26), because t2r (grey, tuned on Tier 1) will not be a final pick.
- Teams with all three tiers today: nima arki, maniphys.ai, domoarigato, Kelly To the H, Xueyi Wang,
  Team OPS, Open4D. MBZUAI, Chivatam and Rice have no Tier 3 and are unranked (upstream #209). [VERIFIED]
- nima arki improved since the brief: Tier 1 now 0.7606 / 0.90 / 0.7667 / 5.97. [VERIFIED]

### 1.1 Rank-sum math

Ranking rule used (upstream `leaderboard.js`, FAQ, site): standard competition ranks per metric among
complete non-baseline rows; track total = sum of the 12 metric ranks over the three tiers. How the
organisers combine tiers, and whether per-tier ranks are recomputed among three-tier teams only, is
**[UNRESOLVED]**; both variants give the same lever ordering.

Our per-tier rank sums with the planning numbers (ranks on each tier board): **Tier 1 4+2+2+7 = 15,
Tier 2 7+6+4+9 = 26**. Tier 3 scenarios (AUC/SP/MP/MPPE): weak 0.20/0.25/0.05/45, mid 0.30/0.40/0.15/35,
good 0.40/0.50/0.25/25.

| Scenario | Track total, us | Place | Who beats us |
|---|---|---|---|
| T3 weak | 55 | 3rd | nima 45, Xueyi 45 |
| T3 mid | 47 | 2nd | Xueyi 46 |
| T3 good | 45 | **1st** | - |
| T3 mid, MBZUAI enters T3 at 0.35/0.45/0.20/30 | 51 | 3rd | MBZUAI 14, Xueyi 50 |
| T3 good, MBZUAI and Chivatam enter T3 | 45 | 2nd | MBZUAI 18 |
| Ranks recomputed among three-tier teams only, T3 mid | 30 | tied 1st | nima 30 |

What each metric is worth, per tier (thresholds to the next ranks on the live board):

| Tier | Our worst metric | To gain ranks | Lever that moves it |
|---|---|---|---|
| 1 | **MPPE, 7th** (8.35) | < 7.68 -> 5th, < 6.72 -> 4th, < 5.97 -> 3rd, < 5.20 -> 2nd | only eps 1/4/25: 214 of the 250 cm summed MPPE in upload #3; ep25 alone is 107 cm = 3.6 cm of tier MPPE |
| 1 | AUC, 4th (0.8122) | >= 0.8237 -> 3rd (+0.35 summed over 30 eps) | smooth stragglers (lever 3) |
| 2 | **MPPE, 9th** (23.33) | < 21.79 +1, < 19.36 +2, < 18.09 +3, < 15.32 +4, < 14.14 +5 | object losses (8 eps in own-ref rep1) |
| 2 | AUC 7th, SP 6th, MP 4th | AUC 0.4425/0.4482/0.4631 = +1/+2/+3; SP 0.6667 = +3; MP 0.5333/0.60 = +1/+2 | 2-3 recovered episodes move all four |
| 3 | - | rank 1 needs > 0.3294 AUC, > 0.40 SP (13/30), > 0.10 MP (4/30), < 31.34 MPPE | a complete entry; then quality |

Lever what-ifs (MBZUAI in T3, our T3 mid, baseline total 51): T2 best-of-2 recovering 3 episodes
(0.465/0.70/0.567/17.5) -> **41**; recovering 2 -> 45; T3 good -> 45; T1 MPPE 5.0 -> 46; T1 MPPE 6.4 -> 48;
T1 AUC 0.825 -> 50; T3 weak -> 59; switching Tier 2 to t2c as packed today -> 59.
**Tier 2 is the cheapest rank in the competition; Tier 3 quality is the biggest risk.**

## 2. Time and GPU budget

- Window A: now -> last 6 h job start ~07:00 Oct 30 (reservation `maint_2026-10-30` from 13:00, open
  end) [VERIFIED `scontrol show res`]. Window B: maintenance end (*expected* Oct 30 evening - Oct 31) ->
  freeze **Nov 4 17:00 EST = Nov 4 23:00 CET**.
- Capacity [VERIFIED]: gpu4 = wr20-25, 4 GPUs each, and we are the only GPU user. Training jobs ask for
  `--mem=128G` against 510 GB nodes, so **only 3 trainings fit per node (18 concurrent)**; the 4th GPU
  takes evals at best (wr23: 448 GB allocated). Completed training jobs peaked at **7.1-7.7 GB MaxRSS**
  (8 jobs, `sacct --units=G`). A 350M run takes 2.8-3.6 h (metrics.jsonl `wall_seconds`); an
  evaluation of 5 repeats ~5 min.
- Committed: t2c smooth (30 runs, ~105 GPU-h incl. evals), t1 smooth stragglers (7), Tier 3 batch 3
  (14 runs, nice 0, ~50 GPU-h). That leaves roughly 400-500 GPU-h in Window A at 18 slots.
  **GPU-hours are not the constraint; decisions and the pre-maintenance upload are.**
- Uploads (5 per week per tier, unlimited in the final 3 days [DOC, CLAUDE.md]): Tier 1 used 3 this week,
  Tier 2 used 2, Tier 3 none. Calendar definition of "week" is **[UNRESOLVED]**.

## 3. Tier 2

### 3.1 Lever 2: a second sample per episode, best-of-2 by own-reference evaluation

Evidence [VERIFIED, own-reference rep1, `outputs/submission/{t2,t2c}/selection.json`]:
- t2 (noisy input) loses the object (rep1 MPPE > 30 cm vs its own reference) on 1, 2, 4, 11, 14, 19, 25,
  28; t2c on 1, 2, 4, 7, 11, 14, 15, 16, 25, 28. Overlap 1, 2, 4, 11, 14, 25, 28; t2-only 19; t2c-only 7, 15, 16.
- Per-rep MPPE of the t2 losses: consistent in all 5 reps on 1, 4, 14, 19, 25, 28; partial on 2
  (30.6 / 4.1 / 88.7 / 3.6 / 4.2) and 11 (rep1 70.3, reps 2-5 2.1-2.7 cm).
- Rep-to-rep noise is not a rep1 artefact: over all 210 five-repeat evaluations, the largest outlier
  (|AUC - median| > 0.1) is spread evenly over reps 1-5 (9/8/9/10/10). Each rep has about a 5% chance
  of falling > 0.1 AUC below its policy's median, so expect ~1.5 unlucky episodes per 30-episode pack.
  Keep rep1 pre-declared. Mean-based selection over more candidates is the honest hedge.
- `assemble_submission.py` already picks the best mean-AUC candidate per episode across every
  `t2_epNN_*` run, so a second run per episode needs no new code. Both candidates are scored against
  the same Tier 2 reference, so the comparison is Tier-2-only. Do not mix t2 and t2c runs per episode
  (different references).

Recommendation: keep the running t2c smooth wave (that is t2c's second sample). Add a **reseed on the
t2 input with the same arm** (`ARM=keypoints SEED=43`), the rule-safest second sample: it picks no method
on Tier 1 evidence. Pack t2-best2 and t2c-best2, then choose between whole packs by local score
(allowed). *Estimate*: 1-3 recovered episodes -> T2 total -4 to -10.

Pilot (4 episodes, **NICE=100**; own-reference evidence only: 14 and 19 lost in all 5 reps, 2 and 11 partially):
```bash
PREFIX=t2 MANIFEST=training_manifest_tier2.json ARM=keypoints SEED=43 LANES=4 EVAL=1 NICE=100 \
    EPISODES="2 11 14 19" bash cluster/submit_tier_lanes.sh
```
Gate (Tier-2-only): if >= 2 of the 4 hold the object in their 5-repeat own-reference eval where seed 42
lost it, queue the remaining 26 at `LANES=8 NICE=200` (~90 GPU-h, ~11 h wall at 8 lanes). It must start
by ~Oct 29 18:00 to land before the maintenance.

### 3.2 The t2c "contact-onset jumps" are mostly raw Tier 2 steps [VERIFIED]

Every t2c step > 10 deg, classified against the raw Tier 2 `observation.object.pose` (§9, `t2c_jumps2.py`):
- The four big jumps in the brief (ep2 34 deg at f763, ep3 43 deg at f435, ep9 55 deg at f323, ep17
  87 deg at f142) happen **while following** (0 held frames before them), and the raw Tier 2 measurement
  has the same step. They are persistent (still 55-60 deg off 30 frames later for ep9 f323), so they look
  like real fast rotation or corruption, not single-frame outliers.
- t2c's own release snaps (hold -> follow) are **15 events**: 14 of 11.9-28.7 deg (eps 2 twice, 3, 6, 7,
  9, 13, 17, 18, 22, 24, 25, 26, 27) plus **ep4 at 90.9 deg** after a 102-frame hold.
- A rate-limited blend at release would fix only those 15 and costs a t2c retrain of the affected episodes.
  Below lever 2.

## 4. Tier 3

### 4.1 Lever 1: coverage, in flight

All 30 episodes are in `training_manifest_tier3.json`, with 4, 11, 19 and 27 coverage-only. Batch 3
(2, 3, 4, 6, 7, 9, 13, 16, 17, 19, 20, 25, 27, 28) is queued at nice 0 (372217-372244). [VERIFIED `squeue`]
At 18 slots, with t2c smooth at nice 300 releasing slots from ~13:00-14:00, all 14 should start by
mid-afternoon and land by ~18:00-19:00 (*estimate*).

- **Upload Tier 3 the moment it packs, before Oct 30 12:00.** It is the difference between ranked and
  unranked, and the reservation has no announced end.
- Fix vs accept-failed for 4/19/27: **accept-failed is right for the first upload.** Each costs at most
  1/30 SP-SR and MP-SR. The coverage-only ep11 policy scored own-reference AUC 0.109 (5 reps), i.e.
  near do-nothing. ep19's validation failure (one frame at 1.04x a URDF velocity limit) is cosmetic; its
  real problem is the object orientation (63 steps > 45 deg/frame, 24 > 90). ep4/ep27 lose the object
  (23 / 13 steps > 15 cm, max 1.16 / 1.29 m), and ep27 sinks 0.52 m under the floor, which no
  post-filter fixes. Re-running CARI4D cannot land before the maintenance.

### 4.2 What the 16 evaluated Tier 3 policies show [VERIFIED, own reference, 5 reps]

`decompose_auc_loss.py --run-fmt t3_ep%02d_keypoints --manifest training_manifest_tier3.json`:
own-reference mean AUC 0.575.
- Holding (AUC >= 0.70): 0, 8, 18, 22, 23, 24, 29 (0.70-0.90), and 21 (mean 0.745, but **rep1 0.183**).
- Partial: 5 (0.45, lost at f70), 10 (0.50), 14 (0.36), 26 (0.52).
- Lost: 1 (0.16, f130), 11 (0.11, f128), 12 (0.18, f160), 15 (0.32, f85).
- Rep1 variance is real here: ep21 rep1 0.183 vs mean 0.745, ep10 0.287 vs 0.502. With rep1
  pre-declared (correctly), a second seed per episode (lever 4) is the honest way to lower that risk.

Per-rep MPPE: ep21 59.1 / 1.3 / 1.4 / 1.5 / 1.4 and ep10 17.9 / 3.4 x4 (rep1 unlucky); ep5 two of five
lost; ep12 lost in all five.

Lever 4 pilot (4 episodes, **NICE=100**: rep1-unlucky 21 and 10, partial 5, consistent loss 12):
```bash
PREFIX=t3 MANIFEST=training_manifest_tier3.json ARM=keypoints SEED=43 LANES=4 EVAL=1 NICE=100 \
    EPISODES="21 10 5 12" bash cluster/submit_tier_lanes.sh
```
Selection is the existing best-mean-AUC rule against our own reconstruction (same reference, Tier-3-only).

### 4.3 Lever 6: physically impossible object steps in our reconstruction [VERIFIED input stats; effect UNRESOLVED]

Object-step statistics of `data/reconstructed/track_2/tier_3_cari4d_ours` (§9, `t3_steps.py`):
- 16 episodes have **zero** steps > 45 deg/frame: 0, 5, 6, 7, 8, 10, 12, 14, 18, 20, 21, 22, 23, 24, 26, 29.
- 14 have many: 1 (42), 2 (17), 3 (94), 4 (29), 9 (14), 11 (110), 13 (74), 15 (46), 16 (6), 17 (16),
  19 (63), 25 (46), 27 (13), 28 (27). Position teleports > 15 cm: 4 (23), 11 (75), 27 (13), 28 (24), 13 (5).
- They are not clean symmetry flips. Unwrapping with the four 180-degree principal-axis flips removes
  most > 90 deg steps but almost none of the 45-90 deg ones (ep3: 94 -> 93 steps > 45 deg). Per episode,
  13-50% of them come back within 5 frames (transient); the rest persist.
- **Causal link not shown:** of the trained affected episodes, ep11 fails at f128, next to its first
  > 20 deg step (f129), but ep1 fails at f130 (first > 45 deg step at f345) and ep15 at f85 (first at f501).
  So the repair will not by itself rescue 1 or 15. It removes untrackable targets later in the episode.

CPU prototype (scratch only, not wired into training): a **persistence-gated slew limiter**. A deviation
must persist 5 frames, then the output follows at <= 30 deg/frame (15.7 rad/s) and <= 0.15 m/frame
(4.5 m/s), both fixed from physics before any scoring. It changes **0 frames on all 16 clean episodes**
and 9-28% of frames on the 14 affected ones (§9, `t3_slew_proto.py`). That makes "t3f" a 14-episode
retrain, with the other 16 symlinked to t3 like `t2c_ep{00,28,29}`.

Path, after the maintenance: rewrite the object columns of the G1 references cluster-side, as
`recover_tier2_contact.py` does, so no PC retarget is needed. Write frame references through the per-episode
rigid transform the packer already checks. Validate, then train `PREFIX=t3f`. Choose t3 vs t3f only as
whole packs (Kaggle/local), never per episode. House-rule note: this synthesises object poses for our
*own reconstruction*. That is a reconstruction method, not a filled-in challenge measurement, but the
maintainer should sign off against the "never interpolate" convention.

## 5. ADEPT (arXiv 2608.19182, CoRL 2026): what transfers [DOC + VERIFIED]

- **Code: not released.** The project page's Code button is marked "Coming soon" (fetched 2026-10-28).
  The abstract on arXiv (v2, 16 Oct 2026) has no repo link. [VERIFIED]
- Recipe [DOC]: PPO + PBT pre-training on generic reposing of 16 primitives with ADR. Post-training per
  task: (A) BC distillation of the prior into the task's observation space, (B) a fresh critic warmed up
  with the actor frozen, (C) conservative PPO (actor lr 1e-5, clip 0.05, critic lr 5e-5). About 3B env
  steps per task vs about 9B from scratch. Arm-hand robots with a geometric fabric, no human data.

| Idea | Fit with us | Cost / verdict |
|---|---|---|
| Reposing pre-training prior | No G1 + Dex3 reposing env, reward or ADR exists. Their per-task *post*-training alone (~3B steps) is ~9x our whole 350M budget per policy. | weeks; **no** |
| Shared pre-training across our 30 episodes, then per-episode fine-tune | FlashCHORD trains on one `task.parquet`, so multi-reference sampling is new env code. Cross-tier warm starts (a Tier 1 policy into Tier 2/3) **break the rule discipline**. Within a tier, scratch already solves 26/30 Tier 1 episodes in 3 h. | days; **no** before freeze |
| (B) Critic warm-up with frozen actor | It answers our measured failure "actor-only transfer into a fresh critic destabilizes" (MPPE 55.98 -> 109.46). It matters now because **the best drop-episode runs have no `state_*` left** (0 files in `t1_ep25_keypoints_term25`, `t1_ep04_keypoints_closegrip`, `t1_ep01_keypoints_grip15`, `t1_ep28_keypoints_closegrip`), so continuing them means an actor-only warm start. | **yes, lever 5** |
| (C) Conservative updates | Upstream FlashSAC already has the knobs: `training.optimization.actor_learning_rate_multiplier` (must be > 0; 1e-3 ~ frozen), `behavior_cloning_coefficient` (BC toward replay actions, scaled by mean Q), `exploration.warmup_action: policy`, `replay.minimum_size: 100000`. [VERIFIED `update.py:160-186`, `learner.py:597-613`] | config only |
| ADR | It could cut rep-to-rep variance, but only with a full retrain wave and an unmeasured effect. | **no** |

Why lever 5 is not a re-run of the ten failed arms: it changes the **budget at the zero-assist stage and
the starting point**, not the reward. In the last stage (306M -> 350M), 3 of 5 drop runs were still
improving: training episode length ep25 term25 204 -> 257, ep04 closegrip 142 -> 180, ep01 grip15
168 -> 201. Two were flat: ep25 keypoints 127 -> 127, ep25 slowvoc 133 -> 134 at 508M. [VERIFIED `trend.py`]

Tooling needed (about one day, after the maintenance):
1. Restore `scripts/train/transfer_flash_sac_policy.py` from `a30d854^` (same-episode actor splice).
2. Write a single-stage curriculum at the final stage settings (voc 0), so a fresh learner does not restart
   the assist fade at stage 0.
3. Run two phases. Phase 1: ~10M steps with `actor_learning_rate_multiplier=0.001`, the critic warm-up
   with a near-frozen actor, ending in a `state_*`. Phase 2: `training.resume=true` with multiplier 0.1
   and `behavior_cloning_coefficient` 0.1-0.5. Pilot on eps 25, 4, 1, 28 at `NICE=100` (4 episodes,
   ~12 GPU-h with evals).

Gate: 5-repeat unassisted world-1 MPPE on ep25 below its do-nothing 49.7 cm (current best mean 73,
rep1 107). Only then does Tier 1 MPPE reach ~6.4 cm, 4th place (-3 rank points). [*estimate*]

## 6. Tier 1 notes

- Selection rule check [VERIFIED]: averaging rep1 of the best mean-AUC candidate per episode reproduces
  the pack (0.8118 vs packer 0.8122; MPPE 8.35). A pre-declared "lowest mean MPPE" pack would score MPPE
  7.87 / AUC 0.7968. It is still 7th on MPPE (needs < 7.68), so **a per-metric MPPE upload does not pay
  today**, even though per-metric final selection would allow it.
- Do-nothing floors of the weak episodes [VERIFIED `leaderboard_floor.py`]: ep1 0.170 / 45.4 cm, ep4
  0.129 / 111.9, ep25 0.138 / 49.7, ep28 0.116 / 48.5, ep9 0.182 / 49.3. Our ep25 and ep1 policies are
  **worse than doing nothing on MPPE** (they fling the object). That is what lever 5 targets. Shipping
  a frozen object stays forbidden.
- Do not re-propose reward shaping for eps 1, 4, 25, or tight keypoints (CLAUDE.md, measured dead ends).

## 7. Things not to do

- Warm-starting Tier 2/3 policies from Tier 1 policies of the same episode: that is the Tier 1 caption
  as input.
- Submitting the best of the 5 evaluation repeats: it would not reproduce. Keep rep1 pre-declared.
- Letting Kaggle auto-select final submissions (lever 0). Kaggle's usual default is to auto-select the
  best public scores **[DOC, Kaggle convention; confirm on each competition page]**, which would put the
  grey t2r on Tier 2 AUC/SP/MPPE.
- Mixing t2 and t2c, or t3 and t3f, per episode by eval score: different references.
- Choosing the Tier 2/3 second-sample arm because it won on Tier 1. A reseed (`SEED=43`) avoids the
  question. If smooth is used, as in the running t2c wave, selection must still be own-reference only.

## 8. Schedule

| When (CET) | What |
|---|---|
| Oct 28 now | Lever 2 pilot (t2, 4 eps, NICE=100); optional lever 4 pilot; consider `--mem=64G` on one training job (lever 7) |
| Oct 28 evening | Tier 3 pack -> upload (Tier 3 upload #1); t2c smooth evals land -> pack t2c-best2 |
| Oct 29 by 18:00 | Lever 2 gate -> full t2 reseed wave (26 eps); Tier 1 re-pack once stragglers land (upload #4 only if AUC >= 0.8237) |
| Oct 30 by 07:00 | Last training starts; by 12:00 upload the best Tier 2 whole pack and any Tier 3 improvement |
| After maintenance | Levers 4, 5, 6; final 3 days: unlimited uploads, pick 2 per metric competition in all 12 |

## 9. Reproduce every measurement

Scratch scripts (not in the repo) live in
`/tmp/claude-2711/-work-mnidgu2s-project-v2d-track2-robotic-grounding/6e6f78f8-8eb3-43ac-b1e9-93b74a536487/scratchpad/improvement/`.
Copy them into `scripts/` if they should outlive this session.

```bash
cd /work/mnidgu2s/project/v2d-track2-robotic-grounding
S=/tmp/claude-2711/-work-mnidgu2s-project-v2d-track2-robotic-grounding/6e6f78f8-8eb3-43ac-b1e9-93b74a536487/scratchpad/improvement
SIF="apptainer exec -B $PWD --pwd $PWD -B /tmp/claude-2711 containers/flash_chord.sif python"
# live board + rank-sum what-ifs (stdlib)
curl -sS -o $S/dl/leaderboard.json https://raw.githubusercontent.com/MVerghese/v2d-leaderboard-data/main/leaderboard.json
(cd $S && python3 -I ranksum.py dl/leaderboard.json && python3 -I whatif.py)
# Tier 1 pack under alternative pre-declared selection rules (stdlib)
python3 $S/candidates.py . t1
# is rep1 a biased draw? (stdlib; answer: no)
python3 $S/rep_outliers.py
# t2c jumps: release snaps vs raw Tier 2 steps; raw-step persistence
$SIF $S/t2c_jumps2.py . ; $SIF $S/t2_spikes.py .
# Tier 3 input step statistics, first bad frames, slew-limiter prototype
$SIF $S/t3_steps.py . ; $SIF $S/t3_first.py . 1,11,15 ; $SIF $S/t3_slew_proto.py .
# Tier 3 own-reference AUC decomposition
apptainer exec --cleanenv --no-mount home,cwd -B "$PWD":/r containers/flash_chord.sif \
  python /r/scripts/evaluate/decompose_auc_loss.py --root /r --run-fmt t3_ep%02d_keypoints \
  --episodes 0,1,5,8,10,11,12,14,15,18,21,22,23,24,26,29 --manifest training_manifest_tier3.json
# Tier 1 do-nothing floors of the weak episodes
$SIF scripts/evaluate/leaderboard_floor.py --episodes 1,4,9,13,17,19,25,26,27,28
# last-stage learning-curve tails of the drop runs (stdlib)
python3 $S/trend.py outputs/cluster/t1_ep25_keypoints_term25 outputs/cluster/t1_ep04_keypoints_closegrip outputs/cluster/t1_ep01_keypoints_grip15
# capacity: memory actually used by finished training jobs, node allocation
sacct -u $USER -S 2026-10-27 --format=JobID,Elapsed,MaxRSS,ReqMem,State -n --units=G | grep '\.batch' | grep COMPLETED | head
scontrol show node wr23 | grep -E 'AllocTRES|RealMemory'
# ADEPT code status
curl -sSL https://adept-dexterity.github.io/ | grep -o 'Coming soon">Code'
```

## 10. Gaps

- [UNRESOLVED] Exact tier aggregation and whether final ranks use selected submissions only, best per
  metric, or recomputed three-tier-only ranks.
- [GAP] Whether MBZUAI-RCL / Chivatam will enter Tier 3. This decides first place more than anything we do.
- [GAP] How our Tier 3 own-reference scores translate to the label. That can only be known from the
  whole-submission pack or Kaggle; we deliberately do not score Tier 3 rollouts against Tier 1.
- [UNRESOLVED] Whether `--mem=64G` is enough under JAX/Warp peaks. sacct MaxRSS says ~7.5 GB; pilot on
  one job before changing the default.

## 11. Why episodes 1, 4 and 25 lose the object, and the two pilots (2026-10-28 ~12:00-13:30 CET)

Tier 1 only, so Tier 1 data is used freely here. Evidence: the 180 evaluated rollouts on disk
(12 arms x 5 unassisted world-1 reps x 3 episodes, `outputs/cluster/t1_ep{01,04,25}_*/eval_349999104/`),
the instrumented contact rollouts `outputs/diag/t1_ep{01,04,25}_keypoints.json`
(`scripts/diagnose/diagnose_contact_rollout.py`: contacts, forces, palm/pelvis errors per step), the
finger logs in `outputs/diag/jitter/t1_ep04_keypoints{,_smooth}_policy.npz` and
`t1_ep01_keypoints_smooth_policy.npz`, and the training `metrics.jsonl`. Commands in §11.6.

### 11.1 What happens at the failure frame [VERIFIED]

| | ep4 baseball bat | ep25 water bottle | ep1 vacuum |
|---|---|---|---|
| Reference | lifted off the floor to 1.32 m by f174, 1.49 m at f228, held high to ~f700; both hands | lifted off a table (0.92 m) from f118 to 1.57 m (f440), right hand alone until the left joins at f219 | upright on its base, pushed to and fro with the left hand (f103-633); 6.5 m path, 0.10 m net travel |
| Grasp formed? | yes: right hand from f110, left from f155; 1-6 raw contacts per hand | yes: right hand from f111, 2-5 contacts, force falls to 3-24 N by f147 | yes: left hand on the handle f108-234, 3-12 contacts, 60-775 N |
| Tracking while held | 1-8 cm through a 1.4 m lift (keypoints instrumented run) | 1-5 cm through a 0.33 m lift | 2-9 cm for ~130 frames |
| In-hand rotation before loss | 50-80 deg | 60-74 deg | 5-24 deg, then 54 deg as it tips |
| Loss | contacts go to 0 at the top of the lift (f215-235); the bat free-falls (v_z -4.0 m/s at f250) | bottle slides down in the hand (v_z -0.3 to -0.65 m/s at f159-171 while the target still rises), contact lost f175, free fall (-3.8 m/s) | handle contact becomes intermittent from f240, pelvis drifts 20 cm, the vacuum tips over (z 0.38 -> 0.07 by f324) |
| End state, all 60 reps | on the floor, 60/60 | on the floor 58/60 (one never lifted, stayed on the table) | lying on the floor 58/60 |

- **It is a drop, not a throw.** Peak speeds after loss match free fall from 1-1.5 m; the reference's
  downward acceleration never exceeds 4.7 m/s^2 (5-frame smoothed, sim time), so the motion itself
  never asks the object to leave a supporting hand (`refacc.py`). On ep4 the hands do overshoot at the
  top (object 1.52 m vs 1.47 m target at f225 with no contact), then it falls.
- **The robot falls after the object** on ep4/ep25 (pelvis 0.5-1.1 m, 30-70 deg off). Irrelevant to MPPE.
- **MPPE arithmetic**: once dropped, the per-frame error is ~1.1-1.5 m for the rest of the episode
  (ep4/ep25 references stay 1.3-1.5 m up), so MPPE is roughly (fraction of frames after the drop) x 1.4 m.
  ep4 drops at f230 of 803 -> ~100 cm. Reaching 20 cm needs the bat held to ~f700.

### 11.2 Why: the policy's hold radius is its training termination radius [VERIFIED]

- **Frames between 10 cm and 25 cm object error** (`linger.py`): every 10-cm-terminated arm
  (stock, keypoints, seeds, grip, vel, slowvoc, closegrip, smooth; 55 reps per episode) crosses that band in
  a median **9 / 7 / 8 frames** (ep1 / ep4 / ep25; p90 27 / 19 / 17). term25, whose object termination
  is 25 cm, lingers there **55-165 / 41-65 / 9-384 frames** and drops the object once past 25 cm. On
  ep25 two term25 reps carried the bottle at 10-25 cm error for 336 and 384 frames (MPPE 45.8 / 52.4,
  below the 49.7 do-nothing floor); that is the whole of term25's ep25 gain (mean 73.0 vs 105-126).
  Training terminates outside this radius, so the policy has never acted there; evaluation does not
  terminate, and outside the radius it lets go.
- **On ep4 the hand opens after the slip** (smooth run, `fingers_ts.py`): object error 5.9 -> 10.1 cm at
  f225-228 with the left hand closed (0.7-0.9 of its closing range), then the left-hand targets swing
  from +0.8 to -0.5...-0.8 of range at f231-234 and the joints open to ~0 by f237. Slip first, release
  second; one rollout, consistent with the contact counts in the keypoints run (to 0 at f235).
- **The palm radius was binding when term25 widened the object radius**: palms sit 10-14 cm off their
  reference during carries (instrumented runs) against a 15 cm palm termination; term25's palm
  terminations rose from 17% to 41% on ep25 (final stage, `metrics.jsonl`). Rotation at loss (55-95 deg)
  also sits at the 86 deg orientation termination.
- **Training never practises holding late**: final-stage training episodes average 127-201 steps
  (including the 50-step reset freeze) vs 432-688 on solved episodes, 54-79% end on object termination.

### 11.3 Candidates rejected, with reasons

- **Larger finger residual scale**: range is not the limit. With `action_scale` 10 the residual already
  spans +-1.5 rad, and the logged finger targets swing between -0.9 and +0.9 of each joint's range
  (18-47% of finger actions beyond |a| = 9). Grip force is capped by the Dex3 effort limit (1.4 Nm,
  URDF), which a larger scale cannot raise.
- **closegrip + smooth**: smooth made ep25 worse (126.4 vs 114.6) and did not move eps 1/4; no mechanism.
- **ADEPT warm start**: the selected drop runs have no `state_*` (lever 5); about a day of tooling.
- Physics edits (friction 1.0 against the vacuum base, 1 kg isotropic objects) are out: the evaluator
  rebuilds the scene from the checkpoint's config, so they would change the scored physics.

### 11.4 The two pilots (queued 2026-10-28 ~12:10 CET, NICE=100, 3 episodes each, EVAL=1)

| Arm | Change (config only, `cluster/train_tier_episode.sbatch`) | Train jobs (ep 1 / 4 / 25) | Eval jobs |
|---|---|---|---|
| `keypoints_loose` | keypoints recipe, termination object 0.10 -> **0.50 m**, palm 0.15 -> **0.30 m**, object rotation 1.5 -> **2.5 rad** | 372335 / 372337 / 372339 | 372336 / 372338 / 372340 |
| `keypoints_loosegrip` | loose + scripted-grip references (`training_manifest_t1grip.json`, alpha 0.6) + `action.finger_residual_scale` 0.15 -> **0.05**: the hand cannot open below ~0.3 closure while the reference holds, nor pre-close above ~0.3 while it does not | 372341 / 372343 / 372345 | 372342 / 372344 / 372346 |

- Not a re-run of term25: term25 moved only the object radius, to 25 cm, and the palm radius then
  terminated 41% of ep25 episodes. These widen all three radii, to where a dropped object (>= 0.5 m
  away within ~0.3 s of free fall) still terminates.
- The evaluator never terminates and the packer scores trajectories, so the thresholds touch training
  only; `termination` and `action` are immutable roots of the portable export
  (`pipeline/flash_sac_actor_export.py`), so evaluation uses the 0.05 finger scale it trained with.
- Checks run: Hydra compose of both arms (`--cfg job --resolve`, every override reaches `env.*`), `bash -n`
  on the sbatch, `scripts/verify/verify_keypoint_flash_sac.sh` 7 passed / 0 failed (upstream unmodified).
- Risk: loosegrip caps thumb_0 opposition at +-0.5 rad, and every logged policy swings thumb_0 across its
  full +-1.05 rad; looser radii may make early tracking sloppier (term25 crossed 10 cm at f160 on ep1, vs
  ~290 for keypoints). ep1 is the least likely rescue: its fingers stayed closed while the vacuum tipped.
- Cost ~20 GPU-h. Expected landing (*estimate*): start as the queued Tier 3 / t2 s43 jobs free GPUs, results
  ~17:30-20:00 CET today; far inside the 2026-10-30 13:00 reservation.

### 11.5 Gate

Mean MPPE over the 5 reps against the current best per episode: **ep1 41.6** (s44; the packed run is
grip15 at 54.8), **ep4 94.2** (closegrip), **ep25 73.0** (term25); and the mean-AUC change against the
packed runs (ep1 0.250, ep4 0.167, ep25 0.162). Reps swing 45-107 cm on one policy (ep25 term25), so call a
win only for a mean at least ~15 cm below the best with 3 of 5 reps below it. Each 10 cm saved on one
episode is 0.33 cm of tier MPPE. If loose wins and loosegrip wins more, the lock adds; if loosegrip alone
fails, suspect the thumb_0 cap. An MPPE-better / AUC-worse run is a per-metric selection decision (§6).

### 11.6 Reproduce

```bash
cd /work/mnidgu2s/project/v2d-track2-robotic-grounding
SIF="apptainer exec -B $PWD --pwd $PWD containers/flash_chord.sif python"
$SIF scripts/diagnose/drop/rollouts.py . 1,4,25      # fail frame, lift, peak speed, end state, MPPE per arm/rep
$SIF scripts/diagnose/drop/thresholds.py . 1,4,25    # first frame past 10/25/50 cm, frame on the floor
$SIF scripts/diagnose/drop/linger.py .               # frames between 10 and 25 cm: 10-cm arms vs term25
$SIF scripts/diagnose/drop/refacc.py .               # reference speed/acceleration in sim time
python3 scripts/diagnose/drop/contact_fine.py . 25 95 215 4   # contacts/forces/palm/pelvis around the loss (also 4 100 290 5, 1 90 340 6)
python3 scripts/diagnose/drop/fingers_ts.py outputs/diag/jitter/t1_ep04_keypoints_smooth_policy.npz 150 250 3
# pilots: queue state, then results once the evals land
squeue -u $USER -o "%.10i %.30j %.8T %R" | grep loose
for r in outputs/cluster/t1_ep{01,04,25}_keypoints_loose{,grip}; do python3 - "$r" <<'PY'
import glob, json, sys
m = [json.load(open(j))["metrics"] for j in sorted(glob.glob(sys.argv[1] + "/eval_349999104/rep*.json"))]
if m: print(sys.argv[1].split("/")[-1], "AUC %.3f" % (sum(x["add_auc"] for x in m) / len(m)),
            "MPPE %.1f" % (sum(x["mppe_cm"] for x in m) / len(m)), [round(x["mppe_cm"], 1) for x in m])
PY
done
apptainer exec --cleanenv --no-mount home,cwd -B "$PWD":/r containers/flash_chord.sif python \
  /r/scripts/evaluate/decompose_auc_loss.py --root /r --run-fmt t1_ep%02d_keypoints_loose --episodes 1,4,25
$SIF scripts/diagnose/drop/linger.py .               # does the new arm hold between 10 and 50 cm?
```

### 11.7 Gaps

- [GAP] No finger log for ep25, and none for the packed runs (grip15, closegrip, term25): the finger
  evidence is the ep4 keypoints/smooth and ep1 smooth rollouts only. `cluster/diagnose_jitter.sbatch` can
  log them (`RUNS=t1_ep25_keypoints_term25 ...`) when an A100 is free.
- [UNRESOLVED] Whether a policy trained to tolerate 10-50 cm uses the room to hold on rather than to
  track loosely. term25 is the only evidence, and it says hold on.
