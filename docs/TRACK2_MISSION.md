# Track 2 — What We Are Supposed To Do

The challenge requirements, extracted from NVIDIA's own committed sources.
Companion document: [NVIDIA_UPSTREAM.md](NVIDIA_UPSTREAM.md) — where the code is.

**Sources** (all inside the pinned upstream checkout, commit `193382a`):

- `official/video_to_data/docs/v2d_challenge/index.html` — the rendered challenge site
- `official/video_to_data/docs/v2d_challenge/assets/v2d_submission_kit.zip` — the submission kit (698 KB, 57 entries)
- `.../v2d_submission_kit/tracks/track_2/README.md` — the Track 2 guide
- `.../v2d_submission_kit/config/tracks.json` — machine-readable tier/metric/rule config
- `official/video_to_data/docs/v2d_challenge/leaderboard.json`

Recorded 2026-10-15. **Figures and dates below are quoted, not inferred** — but the
live site is authoritative and should be re-checked before acting on any deadline.

---

## 1. The task in one line

> **Third-Person-View 4D HOI → Policy.** *"How much reconstruction error can be tolerated?"*

Given 4D human-object-interaction reconstructions, turn human demonstrations into
robot policies, at varying levels of upstream noise and inaccuracy. The track
measures the **tolerance of human-to-robot transfer pipelines to errors from
upstream reconstruction**.

---

## 2. What is fixed, and what we compete on

NVIDIA's wording:

> *"Embodiment and simulator are fixed and provided: G1 + Dex3, so entries compete
> on retargeting and policy learning, not simulation engineering."*

| Fixed for us | We compete on |
|---|---|
| Robot: Unitree **G1** + **Dex3** hands | Retargeting quality |
| Simulator and physics | Policy learning |
| Evaluation metrics and reference | Robustness to corrupted input (tiers 2–3) |
| The scoring world frame | Object-trajectory recovery (tiers 2–3) |

This is why the G1/Dex3 and SONIC code in
[NVIDIA_UPSTREAM.md §4](NVIDIA_UPSTREAM.md#4-code-map--the-five-components-we-were-asked-to-find)
is a *baseline to build on*, not a starting suggestion — the embodiment is not ours
to change.

---

## 3. Three tiers — three separate submissions

Track 2 is split by **input quality**. All three are scored against the *same*
clean multi-view reference.

| Tier | Input | Released dataset folder | Meaning |
|---|---|---|---|
| **1** | Clean multi-view caption | `track_2/tier_1_multiview_caption/` | Upper bound for upstream reconstruction. SOMA-X human + object reconstruction |
| **2** | Synthetic corruption | `track_2/tier_2_synthetic_noise/` | Jitter, dropout and contact errors sampled from **Track 1's error distributions** |
| **3** | Off-the-shelf reconstruction | `track_2/tier_3_cari4d/` | Today's best practice. NVIDIA provides **CARI4D** as the off-the-shelf solution |

From `tracks.json`, every tier shares: `family: object_pose`,
`reference: multi_view_ground_truth`, `scored_episodes: all`,
`worlds_per_episode: 1`, `private_fraction: 0`, `align_initial_pose: false`.

**Rules that follow from this:**

- A valid Track 2 submission is **a separate submission to each tier**.
- Teams **may modify their approach between tiers**.
- Final Track 2 score is a **weighted combination of the three tiers** — the site states each tier is **weighted equally**.
- For tiers 2 and 3, teams are *encouraged* to recover better object-trajectory estimates using **anything except** the clean Tier 1 captions.

That last point is the actual research question of the track, and the main place
to win points.

---

## 4. How we are scored

Four metrics per tier, all computed by comparing **submitted object poses** against
the clean reference — per episode, then averaged across episodes.

| Metric | Key in `tracks.json` | Direction | What it measures |
|---|---|---|---|
| **AUC** | `add_auc` | ↑ higher better | Transform fixed object points by predicted and reference poses, take mean distance (ADD), integrate accuracy over thresholds **1–9 cm** |
| **SP-SR** | `spider_sr` | ↑ higher better | Fraction of episodes with mean object position error **≤ 10 cm** *and* mean orientation error **≤ 0.5 rad** |
| **MP-SR** | `maniptrans_sr` | ↑ higher better | Fraction of episodes where **every** object stays under **3 cm** mean position error *and* **30°** mean orientation error |
| **MPPE** | `mppe_cm` | ↓ **lower** better | Mean position error (cm) of six points attached **5 cm** from each object origin along its body axes — captures translation *and* rotation error |

Metrics are defined in the **CHORD manuscript** (`docs/chord/chord.pdf` in the
checkout). Code-only copies of all four scorers ship in the kit at
`metric_code/track_2/{AUC,SP-SR,MP-SR,MPPE}.py` (~21 KB each) — worth reading
before optimising against them.

**12 Kaggle leaderboards** for Track 2 (4 metrics × 3 tiers), named
`v2d-challenge-track2-tier{1,2,3}-{auc,sp-sr,mp-sr,mppe}`. One CSV is submitted to
all four competitions **within** a tier.

MP-SR is by far the strictest gate (3 cm / 30° on *every* object), so it is the
metric most likely to read zero early on.

---

## 5. Submission mechanics

### Pipeline

```
train a policy per scored episode
          │
          ▼
examples/run_policy_evaluation.py  --checkpoints checkpoints.json --results results/
   (wraps flash_chord/scripts/evaluate_policy.py; one episode_XXXXXX.parquet per episode)
          │
          ▼
tools/pack_submission.py --track track_2_tier1 \
    --results results/ --dataset-root <challenge_dataset> \
    --code-commit "$COMMIT_URL" -o track2_tier1.csv
          │
          ▼
submit/v2d_submit.py track2_tier1.csv --track track_2_tier1 -m "Tier 1 policy" --wait
```

Repeat per tier, substituting `track_2_tier2` / `track_2_tier3`.

### The thing to notice

**Track 2 evaluation requires FlashCHORD.** The kit README states tracks 2 and 3
*"use the included `flash_chord/scripts/evaluate_policy.py` and its Hydra configs"*
and must be run against *"the public FlashCHORD runtime at this revision"*:

```
repo    nvidia-isaac/video_to_data
branch  main
commit  8c778f905cb35f76f8e6cd6e2ea486da4fb9e0ef
path    robotic_grounding/flash_chord/scripts/evaluate_policy.py
```

So FlashCHORD is **on the critical path for submission**, not an optional
accelerated-training extra. See §7 for why that is a problem here.

Note also: the challenge site refers to an `eval_robotic_grounding.py` script.
**No such file exists** in the repo or the kit at this commit — the real path is
`examples/run_policy_evaluation.py` + `flash_chord/scripts/evaluate_policy.py`.
Don't go looking for the former.

### Format and provenance requirements

- Output: **one CSV per tier**, same CSV to all four of that tier's competitions.
- Rows are validated against the public template `data/track_2_sample_submission.csv` (11.6 MB) — exact rows are checked.
- Must include **all scored episodes**, **one checkpoint per episode**.
- The packer converts evaluator poses using **one rigid transform** between the evaluator's recorded reference and that tier's released input, deliberately *preserving* tracking and input errors.
- `COMMIT_URL` must be set to our GitHub `/commit/<sha>` URL.
- Provenance chain collected by the scorer: policy-checkpoint SHA-256 per episode, evaluator script + evaluation-code SHA-256, packer SHA-256, uploader SHA-256.
- For a custom Tier 3 reconstruction, add `--frame-references`, expressed in the released Track 2 world frame — either a LeRobot dataset (`data/` + `meta/`) or one `episode_XXXXXX.npz` per episode with `episode_index`, `coordinate_frame = "released_track2_world"`, `frame_index [T]`, `pose_xyzw [T,1,7]` (position in metres, then `xyzw` quaternion).

### Accounts and data

| Item | Value |
|---|---|
| Dataset | <https://huggingface.co/datasets/nvidia/video_to_data_challenge> (`nvidia/video_to_data_challenge`) |
| Challenge site | <https://nvidia-isaac.github.io/video_to_data/v2d_challenge/> |
| Registration | Google Form (linked from the site) |
| Scoring platform | Kaggle — account + API token required, **rules must be accepted per competition** |
| Contact | `v2d_challenge@nvidia.com` |
| FAQ | `docs/v2d_challenge/assets/v2d_challenge_2026_faqs.pdf` |

The kit ships an agent skill at `skills/v2d-submit/SKILL.md` that walks through
Kaggle setup, rule acceptance, evaluation, packing and upload.

---

## 6. Timeline and rules

From `tracks.json` → `timeline` and `rules`:

| Date | Event |
|---|---|
| September 21 | Registration opens; data release and launch |
| **November 4, 2026, 5:00 p.m. EST** | **Leaderboard freeze** |
| November 12 | Winners announced at the CoRL 2026 tutorial |
| After November 12 | Winner solution eval + tech report preparation |

> **As of 2026-10-15 that leaves roughly three weeks.**

| Rule | Value |
|---|---|
| Submission rate | **5 per week**, unlimited during the final **3 days** |
| Max team size | 10 |
| Max final submissions | 2 |
| Sponsor | NVIDIA Corporation |
| Data use / licence | commercial; CC-BY-4.0 |
| Winner code licence | **Apache-2.0**, publicly released (required by §2.9.e) |
| Governing law | California (Santa Clara County) |

`tracks.json` carries its own caveat that the website is authoritative where the
three sources disagree, and that prize totals were still *"To be announced"* in
that file even though the site states a **$54K** pool across all tracks.

### Eligibility for awards

A valid final submission **plus** agreed code release (research licence) **plus** a
short technical report. Prizes per track: winner gets a DGX Spark (~$6K),
runner-up $3K GPU credits. The separate **Industry Innovation Award** (DGX Spark +
$3K credits) is open to any valid entry regardless of rank, judged on economic
viability, data flywheel, innovation and complexity reduction.

Organizers also ask that participants **include a GitHub commit** in submissions
and share the repository with
[Nvidia-V2D-Challenge](https://github.com/Nvidia-V2D-Challenge) — invite them as a
collaborator if the repo is private.

---

## 7. Known gaps and risks

Ordered by how much they threaten a submission.

| # | Gap | Severity | Detail |
|---|---|---|---|
| 1 | **FlashCHORD is required for submission, and this machine cannot run its full training** | **Critical** | Evaluation is mandatory via `flash_chord/scripts/evaluate_policy.py`. [REQUIREMENTS.md §8.1](REQUIREMENTS.md) records 8151 MiB VRAM against FlashCHORD's stated **≥46,000 MiB free / 48 GiB** — about 5.9× short. *Evaluation* may be far lighter than training ("replay and one-world viewing use less memory", and tiers use `worlds_per_episode: 1`), but that is **unmeasured**. Measuring it is the highest-value next experiment |
| 2 | **Three weeks to the freeze** | **Critical** | Nov 4, 2026. Scope accordingly — Tier 1 alone is a valid partial result |
| 3 | Pinned eval commit is unreachable from our clone | High | Kit pins `8c778f905cb35f76f8e6cd6e2ea486da4fb9e0ef`; we cloned `--depth 1` at `193382a`. Needs `git fetch --depth 1 origin 8c778f9…` or a deeper clone |
| 4 | No NGC credentials | High | Blocks the `nvcr.io/nvidia/isaac-lab:2.3.2` pull, so no image build ([REQUIREMENTS.md](REQUIREMENTS.md) 3.1/3.2) |
| 5 | ~~LFS not pulled~~ | **Partly resolved** | The G1 ReconBody baseline set is materialised — 123 objects, 197.0 MB, including both SONIC ONNX graphs, verified as real assets ([NVIDIA_UPSTREAM.md §7](NVIDIA_UPSTREAM.md#7-baseline-assets-prepared--g1-reconbody-snack-box)). Still pointers: the 714 MB planner models, needed only if we move to the **ReconHand** curriculum ([§6](NVIDIA_UPSTREAM.md#6-lfs-state)) |
| 6 | MANO models not present | High | Licence-gated, register at mano.is.tue.mpg.de ([REQUIREMENTS.md](REQUIREMENTS.md) 7.3) |
| 7 | Challenge dataset not downloaded | High | ~unknown size from HuggingFace; 121 GB free on `/` |
| 8 | Not registered / no Kaggle setup | Medium | Registration form, Kaggle account + token, and rule acceptance on **12** competitions |
| 9 | Technical report + code release not started | Medium | Required for award eligibility, not for leaderboard placement |
| 10 | Disk headroom | Medium | 121 GB free, shared root fs, before images (`isaac-lab:2.3.2` is large) + dataset + checkpoints |

---

## 8. Suggested order of work

Nothing below has been done yet; this is a plan, not a status.

**Unblock (all external or cheap):**

1. Register for the challenge; create the Kaggle account and API token; accept rules on the 12 Track 2 competitions.
2. Obtain NGC credentials, `docker login nvcr.io`.
3. `git lfs pull` in the upstream checkout (SONIC ONNX + meshes).
4. Register for MANO and stage the two `.pkl` files.
5. Fetch the pinned eval commit `8c778f9…` so the required evaluator is available.

**Measure before committing (the decision point):**

6. Download the Tier 1 dataset subset.
7. **Measure FlashCHORD `evaluate_policy.py` VRAM at `worlds_per_episode: 1`.** This single number decides whether this machine can produce a submission at all, or whether we need a ≥48 GB GPU for the whole track rather than just for training.
8. Build the `robotic-grounding` image; measure retargeting VRAM at 8 GiB (REQUIREMENTS 2.5 is now PASS, 8.2 still open).

**Baseline:**

9. Reproduce `EXAMPLE_SEQUENCES.md` end to end on one sequence: retarget → plan → stage 1 → eval.
10. Run the full `SonicG1-ReconHand-Stage{1,2,3}-v0` curriculum on Tier 1 episodes.
11. Pack and submit **Tier 1** early, even if weak — it validates the entire submission chain while there is still time to iterate.

**Compete:**

12. Tier 2: robustness to synthetic corruption sampled from Track 1 error distributions.
13. Tier 3: object-trajectory recovery from CARI4D reconstructions — the most open-ended, highest-upside part.
14. Technical report; ensure the repo is shareable with `Nvidia-V2D-Challenge`.

Step 11 before step 12 is deliberate: a scored-but-poor Tier 1 submission is worth
more than an unsubmitted good one, and it de-risks the packer/uploader format
checks that are easy to get wrong.

---

## 9. Provenance of this document

Extracted read-only from the committed challenge site and submission kit at
upstream commit `193382a`. The site's HTML was converted to text; the kit zip was
**listed and read in memory**, never extracted to disk, and no file from it was
executed. `tracks.json` and the Track 2 guide were read directly.

Nothing was registered, downloaded, submitted, installed or run. Dates, metric
definitions, thresholds and rules are quoted from those sources; the live
challenge site and Kaggle pages remain authoritative, and §6's caveat about the
three sources disagreeing is NVIDIA's own.
