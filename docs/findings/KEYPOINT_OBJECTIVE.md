# Why stock FlashSAC stalls at AUC 0.38, and the keypoint objective (arm 3)

2026-10-25. Cluster (A100-80GB), NVIDIA's exact `g1_recon_body_flash_sac` recipe, seed 42,
27 of 30 Tier 1 episodes evaluated (5 unassisted world-1 repeats each).
Marks: **[VERIFIED]** measured here · **[DOC]** stated by NVIDIA · **[GAP]** not yet known.

## 1. How Track 2 is actually ranked

Earlier docs (and CLAUDE.md before today) said "Track 2 ranks by AUC". That is incomplete.

| Rule | Source |
|---|---|
| Final rank within a track = **lowest sum of the per-metric ranks** (AUC + SP-SR + MP-SR + MPPE) | FAQ PDF, "How is the final ranking calculated" **[DOC]** |
| The **three tiers are weighted equally** in the Track 2 score | challenge site, Track 2 "Evaluation" **[DOC]** |
| A team missing any metric is **unranked**, listed below every complete team | upstream `0593d80d` (#209), `leaderboard.js` **[DOC]** |
| Trajectories must come **from learned policies** for all 30 episodes | FAQ, "valid Track 2 submission" **[DOC]** |
| 5 submissions per week **per track and tier** | FAQ **[DOC]** |
| Tier 3 may use Tier 1's **first frame** for alignment, nothing else from Tier 1 | FAQ (updated `d9d6bf04`) **[DOC]** |

Consequences:
- Shipping a frozen-object trajectory where it beats the policy (`TOP3_STRATEGY.md` §3a) is
  **not allowed** — it is not a learned policy's output.
- Tier 1 alone can never win Track 2; Tiers 2 and 3 are two thirds of the score.
- The 0.969 Tier 1 "AUC leader" posted AUC only and is unranked.

Live standings are public JSON (the site's `data-src`), no Kaggle login needed:
`https://raw.githubusercontent.com/MVerghese/v2d-leaderboard-data/main/leaderboard.json`

## 2. Where we stand (2026-10-25) [VERIFIED]

| Board | Ranked teams | Best AUC | Us |
|---|---:|---:|---|
| Tier 1 | 17 | 0.8305 (Chivatam; SP-SR 0.90, MP-SR 0.83, MPPE 6.72) | AUC 0.2128, rank-sum 60 (15th) — old 10M submission |
| Tier 2 | 13 | 0.5180 (Xueyi Wang) | not entered |
| Tier 3 | 6 | 0.3189 (nima arki); one team ranks 6th with AUC 0.000 | not entered |

CHORD Baseline: Tier 1 AUC 0.5104 / Tier 2 0.3737 / Tier 3 (CARI4D) 0.0023.

Stock recipe on our 27 evaluated Tier 1 episodes: **mean AUC 0.384** (do-nothing floor on
the same episodes 0.269). Four episodes are still below their floor: 1, 10, 20, 25.

## 3. Diagnosis [VERIFIED]

`scripts/evaluate/decompose_auc_loss.py` scores each exported rollout three ways:

| | mean AUC (27 eps) |
|---|---:|
| as achieved | 0.384 |
| achieved position, **reference rotation** | 0.558 |
| reference position, **achieved rotation** | 0.521 |

- **No setup offset.** Frames 0-30 sit at 0.2-2.8 cm and ~1 deg in every episode.
- **Both error sources matter**; fixing either alone cannot reach the 0.83 leaders.
- **Small objects that turn are rotation-bound**: mug 2.6 cm / 36.5 deg, cone 3.6 cm /
  49.6 deg, rice cooker 1.8 cm / 40.7 deg, watering can 2.6 cm / 33.0 deg.
- **Most episodes fail mid-motion and never recover** (first ADD > 9 cm at frame 106-476).

The cause is the objective. Per-step weighted reward components from `metrics.csv`:

| run | stage | `object_position` | its maximum | object terminations |
|---|---|---:|---:|---:|
| ep05 | 3 / 6 / 8 | 0.0187 / 0.0188 / 0.0190 | 0.0200 | 21 / 21 / 16 % |
| ep03 | 3 / 6 / 8 | 0.0189 / 0.0188 / 0.0194 | 0.0200 | 6 / 51 / 26 % |
| ep25 (fails) | 3 / 6 / 8 | 0.0184 / 0.0181 / 0.0183 | 0.0200 | 45 / 73 / 74 % |

`object_position = exp(-|e|^2 / 0.2^2)` is **saturated at 92-97 % of its maximum in every
run and stage**. Training terminates at 10 cm (`object_position_threshold`), and inside
that window the term never drops below 0.78 and is flat across the AUC band (1 cm 0.998,
5 cm 0.94). The policy's only object signal is "stay inside 10 cm" — which is exactly the
3-4 cm it learned. There is **no object-orientation term** at any stage, and the
orientation termination is 1.5 rad (86 deg).

NVIDIA's own Sharpa objective already has the missing piece: `object_keypoints`, six
object-axis keypoints with a Laplacian (`objectives/keypoints.py`, `objective/tracking.yaml`
weight 1.0, var 0.1). G1's `recon_body` objective does not use it. **[VERIFIED]**

### Ruled out
- **Playback speed.** `source_frame_playback: true` plays one 30 fps frame per 50 Hz step,
  so every motion runs 1.67x faster than the human. The evaluator numbers rows
  `reference_frame = step + start_frame` and the packer needs one row per dataset frame,
  so this is fixed for every team. **[VERIFIED]**
- **Upstream drift.** `origin/main` is 7 commits past our `193382a`; none touch
  `flash_chord` or `robotic_grounding` (docs, FAQ and leaderboard renderer only). **[VERIFIED]**
- **Evaluation-time termination.** The evaluator disables termination and never resets
  (`evaluation/harness.py:222-250`), so training thresholds cannot change scoring. **[VERIFIED]**

## 4. Episode 7 diverges on its 1 cm tabletop

Seed 42 went NaN at 5.27M steps and trained 6.5 h on NaN; seed 43 went NaN at 81.8M. In the
first NaN row the episode return and length are still finite while reward and networks are
NaN — a physics blow-up poisoning the shared normalizer. Episode 7 is the **only** episode
whose support is a 1 cm slab (`tabletop`, z-scale 0.01); every other support is a 4 cm
cylinder or a 2-27 cm box, and none of them ever produced a NaN. **[VERIFIED correlation;
causation UNRESOLVED]**

Mitigations in place: `cluster/train_flash_sac.sbatch` now stops a job at the first NaN
metrics line, and Episode 7 resumes seed 43 from `state_80003072` (pre-NaN) with seed 44
(`outputs/cluster/t1_ep07_s43_resume80M`). Thickening the slab below its top surface would
remove the trigger but edits NVIDIA's scene geometry, which the evaluator also uses — a
rules question, not done.

## 5. Arm 3: the `object_keypoints` term

`pipeline/flash_sac_keypoints.py`, same delegation pattern as arms 1-2 (upstream's twelve
terms copied byte for byte, one raw thirteenth column FlashSAC replays and relabels):

    raw = mean_i exp(-|(p + R r e_i) - (p* + R* r e_i)| / 0.05),   r = 0.5 * object_radius

- `e_i`: upstream's `KEYPOINT_VECS_NP`; Laplacian: upstream's `shaped_objective`.
- `r`: FlashCHORD's per-world bounding-ball radius (`command.object_radius`) x 0.5, because
  ADD averages mesh vertices (stool: mean 0.30 m vs bounding 0.55 m). Rotation costs what
  it costs ADD, per object.
- scale 0.05 m: 1 cm -> 0.82, 3 cm -> 0.55, 5 cm -> 0.37, 9 cm -> 0.17 (the AUC band).
- weight 5.0: equal to `body_joint_position` and `force_closure`, the two dominant terms.

Tests: `pipeline/tests/test_keypoint_reward.py` (host math), `container_flash_sac_keypoints_kernel.py`
(Warp kernel vs NumPy, Warp CPU), exporter tests for the new target. The exporter
(`pipeline/flash_sac_actor_export.py`) is now a registry of all three thirteen-term targets
and `cluster/evaluate_policy.sbatch` projects such checkpoints automatically.

## 6. Pilot (submitted 2026-10-25)

Keypoint arm on episodes **0, 3, 5, 17, 21, 25**, everything else identical to the stock
runs (seed 42, 2048 worlds, 10M replay, 350M steps), gated on smoke job `358922`, each
evaluated automatically (5 repeats) when it finishes. Stock mean AUC on these six:
**0.3844** (`decompose_auc_loss.py --episodes 0,3,5,17,21,25`).

Decision gate: roll arm 3 out tier-wide only if its six-episode mean AUC beats 0.3844 by
more than the repeat spread, without lowering MPPE-based success. Compare with

    apptainer exec --cleanenv --no-mount home,cwd -B "$PWD":/r containers/flash_chord.sif \
      python /r/scripts/evaluate/decompose_auc_loss.py --root /r --run-fmt t1_ep%02d_keypoints

## 7. Open [GAP]
- Tier 2 references need the laptop (`robotic-grounding` image + Tier 2 mirror):
  `run_tier_pipeline.py all --tier tier_2_synthetic_noise`, then stage to the cluster.
- Tier 3 needs our own reconstruction from video; not started.
- SP-SR/MP-SR are only available from `tools/pack_submission.py`; the evaluator's
  `spider_sr_uncentered` is not the leaderboard SP-SR.
