# V2D Challenge 2026, Track 2: Robotic Grounding (team Izus_Systems)

A simulated Unitree G1 humanoid (with Dex3 hands) has to reproduce each challenge episode's
**object** motion, unassisted, starting from a human video reconstruction. We train one
FlashSAC policy per episode, per tier, on top of NVIDIA's pinned GEAR-SONIC whole-body
controller, export what the object actually did, and pack it into a Kaggle CSV.

- **Ranking:** lowest sum of per-metric ranks (AUC, SP-SR, MP-SR, MPPE). The three tiers count
  equally; a team missing a tier is unranked.
- **Freeze:** 4 November 2026, 17:00 EST. Each tier allows 5 uploads per week (unlimited in the
  final 3 days) and 2 final selections.

| Tier | Input | Our pipeline |
|---|---|---|
| 1 | NVIDIA's clean multi-view caption | retarget, then train |
| 2 | the same data with a corrupted object track | Tier-2-only contact-gated recovery, then train |
| 3 | nothing released: our own reconstruction from the exo video | SAM3 masks, CARI4D reconstruction, conversion, retarget, then train |

**Read next:** [docs/architecture/ARCHITECTURE.md](docs/architecture/ARCHITECTURE.md) (how it
works) · [docs/architecture/CODE_MAP.md](docs/architecture/CODE_MAP.md) (what every file does) ·
[CLAUDE.md](CLAUDE.md) (environment traps and measured results: read before running anything).

---

## Contents

1. [Rules of the repo](#1-rules-of-the-repo)
2. [Layout](#2-layout)
3. [Machines and one-time setup](#3-machines-and-one-time-setup)
4. [Daily status](#4-daily-status)
5. [Training](#5-training)
6. [Evaluation](#6-evaluation)
7. [Tier 2: rules-clean object recovery](#7-tier-2-rules-clean-object-recovery)
8. [Tier 3: reconstruction pipeline](#8-tier-3-reconstruction-pipeline)
9. [Retargeting on Ibrahim's PC](#9-retargeting-on-ibrahims-pc)
10. [Watching episodes](#10-watching-episodes)
11. [Packing and submitting](#11-packing-and-submitting)
12. [Tests and health checks](#12-tests-and-health-checks)

---

## 1. Rules of the repo

- **Branch `submission_trials`.** Commits and pushes are Ibrahim's only. The cluster commits
  with `git ci` (ibrahim-1811), and Ibrahim pushes from his PC (section 11).
- **Never edit** `official/` (NVIDIA upstream, pinned at `193382a`) or `data/v2d_challenge/` (the
  pristine Hugging Face mirror). We extend upstream by wrapping it, never by patching it.
- **No Tier 1 caption in any Tier 2 or Tier 3 decision**, not as input, threshold, method choice
  or go/no-go report. Tier 3 may use Tier 1's **first frame** for alignment only (FAQ).
- **Never ship a frozen-object trajectory**: the FAQ requires trajectories from learned policies.
- **Every policy is judged unassisted** (`voc_scale = 0`, world 1, as the packer scores). Never
  present an object-assisted replay as success.

## 2. Layout

```
pipeline/            Python package: adapters, reward objectives, validators (+ tests/)
scripts/
  data/              download, stage, retarget, validate; Tier 2 recovery
  train/             manifests, curricula, grip references, metrics
  evaluate/          submission assembly, stationary floor, AUC decomposition
  diagnose/          rollout diagnostics, media recording, viewers
  tier3/             SAM3 masks -> CARI4D -> challenge-format episodes
  verify/            one-command health checks
cluster/             Slurm jobs and launchers (training, evaluation, Tier 3, media, packing)
configs/             generated Hydra overrides for our objectives and curricula
docker/              the pyarrow fix for robotic-grounding:latest (PC only)
docs/                architecture/ guides/ reference/ findings/
training_manifest_*.json   per-tier episode facts the trainer reads (support flag, floor, paths)

official/  data/  outputs/  containers/  logs/  .venvs/      gitignored (large or machine-specific)
```

| Manifest | Tier | Reference tree |
|---|---|---|
| `training_manifest_verified.json` | Tier 1 | `data/adapter_out` |
| `training_manifest_t1grip.json` | Tier 1, scripted grip | `data/adapter_out_grip` |
| `training_manifest_tier2.json` | Tier 2, raw input | `data/adapter_out_tier_2` |
| `training_manifest_tier2c.json` | Tier 2, contact-gated recovery | `data/adapter_out_tier_2c` |
| `training_manifest_tier3.json` | Tier 3, validated episodes only | `data/adapter_out_tier_3_cari4d_ours` |

## 3. Machines and one-time setup

| Machine | Used for | Has |
|---|---|---|
| **Cluster** `wr0.wr.inf.h-brs.de`, repo at `/work/mnidgu2s/project/v2d-track2-robotic-grounding` | training, evaluation, Tier 3 reconstruction, packing; the main workspace | `containers/flash_chord.sif`, `containers/cari4d.sif`, 24 A100s (`wr20-25`) |
| **Ibrahim's PC** `~/v2d-track2-robotic-grounding` | retargeting (the only place `robotic-grounding:latest` exists), pushing, Kaggle uploads | Docker, 8 GB GPU, `~/v2d_submission_kit` |

**Cluster, from scratch** (already done; kept for reference):
```bash
git clone -b submission_trials https://github.com/ibrahim-1811/v2d-track2-robotic-grounding.git
cd v2d-track2-robotic-grounding
mkdir -p official && cd official && git clone https://github.com/nvidia-isaac/video_to_data.git \
  && cd video_to_data && git checkout 193382a && git lfs pull && cd ../..
apptainer build containers/flash_chord.sif cluster/flash_chord.def     # training/evaluation image
apptainer build containers/cari4d.sif cluster/cari4d.def               # Tier 3 reconstruction image
python3 scripts/data/download_challenge_episodes.py                     # Tier 1 + 2 mirror -> data/v2d_challenge
```
V100 nodes (`wr14-19`) cannot run the SONIC controller or CARI4D's Warp build; every A100 job
already excludes them. Only SAM3 masks run there.

**Ibrahim's PC, from scratch:**
```bash
git clone -b submission_trials git@github.com:ibrahim-1811/v2d-track2-robotic-grounding.git
cd v2d-track2-robotic-grounding
docker build -f docker/Dockerfile.pyarrow-fix -t robotic-grounding:latest .   # on top of upstream's image
python3 -m venv --system-site-packages .venvs/episode0 && .venvs/episode0/bin/pip install pyarrow==15.0.2
```

## 4. Daily status

```bash
cd /work/mnidgu2s/project/v2d-track2-robotic-grounding
bash cluster/train_status.sh 't1_ep*' 't2c_ep*' 't3_ep*'    # progress % and evaluations per run
bash cluster/status.sh                                       # queue + rate/ETA of running jobs
squeue -u $USER -o '%.10i %.28j %.8T %.10M %R'               # raw queue
scontrol show res                                            # maintenance windows
```
Live standings without a Kaggle login:
`https://raw.githubusercontent.com/MVerghese/v2d-leaderboard-data/main/leaderboard.json`.

## 5. Training

One FlashSAC policy per episode: NVIDIA's `g1_recon_body_flash_sac` recipe, 349,999,104
environment steps at 2,048 worlds (about 3 h on an A100). `ARM` picks the objective:

| `ARM` | What changes | Status |
|---|---|---|
| `stock` | NVIDIA's 12-term objective | baseline |
| `keypoints` | + `object_keypoints` (six points, 5 cm Laplacian, weight 5): what AUC measures | **default** |
| `keypoints_smooth` | + 100x action-rate and action penalties | less shaking, equal or better AUC |
| `keypoints_gentle` | smooth + half the body residual (`action.residual_scale=0.075`) | pilot (eps 8, 18, 23) |
| `keypoints_closegrip` | keypoints on references with a scripted Dex3 grip | helped squeegee and bat |
| `keypoints_vel`, `keypoints_slowvoc`, `keypoints_vel_slowvoc` | + object velocity term / slower assist fade (507.5M steps) | no gain on the dropping episodes |
| `keypoints_tight` | keypoint scale 3 cm | worse on every episode tried |

```bash
# A whole tier, spread over LANES chains, each training followed by its 5-repeat evaluation:
ARM=keypoints LANES=6 EVAL=1 EPISODES="$(seq 0 29)" bash cluster/submit_tier_lanes.sh                      # Tier 1
PREFIX=t2c MANIFEST=training_manifest_tier2c.json ARM=keypoints LANES=6 EVAL=1 EPISODES="1 2 3" bash cluster/submit_tier_lanes.sh
PREFIX=t3  MANIFEST=training_manifest_tier3.json  ARM=keypoints LANES=6 EVAL=1 EPISODES="0 1 5" bash cluster/submit_tier_lanes.sh
# Variants: ARM=keypoints_smooth, SEED=43 (run gets a _s43 suffix), NICE=100 (queue behind other work), DRY=1 (print only)
MANIFEST=training_manifest_t1grip.json ARM=keypoints_closegrip LANES=4 EVAL=1 EPISODES="1 4 25 28" bash cluster/submit_tier_lanes.sh

# One episode by hand:
EP=8 ARM=keypoints sbatch --job-name=v2d-t1-ep08 cluster/train_tier_episode.sbatch
```
Runs land in `outputs/cluster/<prefix>_epNN_<arm>[_sSEED]/`: `policy_<step>.safetensors`
(actor, 4.6 MB), `metrics.jsonl`, and `eval_<step>/rep1..5.json` plus the exported trajectories.
A run whose final policy exists is skipped on re-queue. Training metrics overstate quality
(episodes terminate at 10 cm); only the unassisted evaluation counts.

**If a run dies with NaN** (episode 7's 1 cm support slab), `train_flash_sac.sbatch` stops at the
first NaN line. Resume from the last state before it (see CLAUDE.md):
`EXTRA="training.resume=true training.checkpoint_path=/out/<run>/state_<step>.safetensors"`.

```bash
.venvs/episode0/bin/python scripts/train/show_training_metrics.py outputs/cluster/t1_ep24_keypoints --terms
python3 scripts/train/monitor_training_disk.py outputs/cluster/t1_ep24_keypoints  # dry run; --apply prunes old state_*
```

## 6. Evaluation

Five fresh unassisted world-1 rollouts with NVIDIA's unchanged `evaluate_policy.py`, scored
against the episode's own stationary floor. `EVAL=1` above queues these automatically.

```bash
RUN=t1_ep24_keypoints STEP=349999104 REPEATS=5 EPISODE=24 FLOOR=0.6 sbatch cluster/evaluate_policy.sbatch
EPISODES="1 2 3" bash cluster/submit_tier_evals.sh                                # many at once
python3 scripts/evaluate/assemble_submission.py --prefix t1 -o /tmp/sel/results  # best run per episode (dry run)
apptainer exec -B $PWD:/r --pwd /r containers/flash_chord.sif python scripts/evaluate/decompose_auc_loss.py --root /r --run-fmt t1_ep%02d_keypoints --episodes 2,9   # position vs rotation loss
RUNS="t1_ep02_keypoints t1_ep02_keypoints_smooth" sbatch cluster/diagnose_jitter.sbatch    # joint-shake measurement
```
Selection rule: the highest **mean** AUC over the 5 repeats per episode, MPPE as the tie-break.
Rep 1 of the winner is what gets submitted (fixed in advance, not cherry-picked).

## 7. Tier 2: rules-clean object recovery

Tier 2's object noise is a random walk (up to 30 deg / 3 cm) that keeps moving while the real
object is still. The recovery holds the object unless a palm or foot (URDF forward kinematics)
is within its bounding radius + 0.10 m, with a 5-frame release. Both parameters come from physics
and Tier 2's own noise limit. There is no Tier 1 code path.

```bash
apptainer exec -B $PWD --pwd $PWD containers/flash_chord.sif python scripts/data/recover_tier2_contact.py --write
# -> data/adapter_out_tier_2c, outputs/tier2c_frame_references/, training_manifest_tier2c.json
```

## 8. Tier 3: reconstruction pipeline

Per episode, on the cluster:

```bash
EPS="7" PROMPT="clothes iron" sbatch --mem=150G cluster/tier3_masks.sbatch   # SAM3 masks on a V100 (optional: the A100 job does it too)
EP=7 sbatch cluster/tier3_cari4d_episode.sbatch                              # masks if missing, then CARI4D (~1-1.5 h)
EPS="4 11 15" sbatch cluster/tier3_convert.sbatch                            # CARI4D -> data/reconstructed/track_2/tier_3_cari4d_ours
```

What each step guards against (details in CLAUDE.md):
- **Masks** (`scripts/tier3/sam3_text_masks.py`) pick the object instance whose image size
  matches its released mesh, drop one-frame actor and object glitches (a stray blob, or the strip
  along the left image border), and refuse an implausible pick.
- **CARI4D** runs with two one-line patches, bind-mounted copies applied by the sbatch: undefined
  centroid distances are written as NaN, and off-image object poses are treated as invalid crops.
- **Conversion** (`scripts/tier3/cari4d_to_lerobot.py`) aligns to the released world with Tier 1's
  frame-0 object pose. If that disagrees with Tier 1's frame-0 human by more than 20 deg (a symmetric
  object), the human places the scene and the object's orientation is relabelled to frame 0.

Then retarget on Ibrahim's PC (section 9), rsync back, and on the cluster:

```bash
apptainer exec -B $PWD --pwd $PWD containers/flash_chord.sif python scripts/data/run_tier_pipeline.py validate --tier tier_3_cari4d_ours --episodes 11,12,14 --force
apptainer exec -B $PWD --pwd $PWD containers/flash_chord.sif python scripts/tier3/make_tier3_manifest.py      # [--accept-failed 11] coverage only
PREFIX=t3 MANIFEST=training_manifest_tier3.json ARM=keypoints LANES=3 EVAL=1 EPISODES="12 14 15" bash cluster/submit_tier_lanes.sh
```
The manifest admits an episode only when validation passed on the cluster, the frame-0
alignment check passed, and the reference is newer than its conversion.

## 9. Retargeting on Ibrahim's PC

```bash
cd ~/v2d-track2-robotic-grounding
git pull ssh://mnidgu2s@wr0.wr.inf.h-brs.de/work/mnidgu2s/project/v2d-track2-robotic-grounding submission_trials
rsync -avz mnidgu2s@wr0.wr.inf.h-brs.de:/work/mnidgu2s/project/v2d-track2-robotic-grounding/data/reconstructed/track_2/tier_3_cari4d_ours data/reconstructed/track_2/
E=1,5,8-10                                     # episode selector
.venvs/episode0/bin/python scripts/data/run_tier_pipeline.py stage    --tier tier_3_cari4d_ours --episodes $E --force
.venvs/episode0/bin/python scripts/data/run_tier_pipeline.py retarget --tier tier_3_cari4d_ours --episodes $E --force
.venvs/episode0/bin/python scripts/data/run_tier_pipeline.py validate --tier tier_3_cari4d_ours --episodes $E --force
rsync -avz data/adapter_out_tier_3_cari4d_ours mnidgu2s@wr0.wr.inf.h-brs.de:/work/mnidgu2s/project/v2d-track2-robotic-grounding/data/
```
`--force` is needed whenever an episode was retargeted before: the pipeline otherwise reuses
the old reference even if its input changed. About 1.5 minutes per episode. Tiers 1 and 2 use
the same commands with `--tier tier_1_multiview_caption` / `tier_2_synthetic_noise`.

## 10. Watching episodes

Three ways. All show the **unassisted** policy, the same rollout the evaluator scores.

### a) Recorded videos and 3D replays: any PC, no GPU

```bash
# cluster: record a selection (mp4 + .viser 3D replay per episode) into outputs/media/<set>/
sbatch cluster/record_policy_media.sbatch                                         # the current Tier 1 selection
SELECTION=outputs/submission/t2c/selection.json sbatch cluster/record_policy_media.sbatch
RUNS="t1_ep08_keypoints_gentle t1_ep08_keypoints_smooth" SET=gentle sbatch cluster/record_policy_media.sbatch
python3 scripts/diagnose/make_media_index.py                                      # rebuild outputs/media/index.html

# your PC: copy and open
rsync -avz --progress mnidgu2s@wr0.wr.inf.h-brs.de:/work/mnidgu2s/project/v2d-track2-robotic-grounding/outputs/media/ ~/v2d_media/
cd ~/v2d_media && python3 -m http.server 8000                                     # then open http://localhost:8000
```
Each row links the MP4 and the interactive 3D replay (orbit, zoom, timeline). `t1` = the
submitted Tier 1 policies, `t2r` = the submitted (grey) Tier 2 policies.

### b) Live viewer on a cluster GPU, opened from any PC

```bash
# cluster: start the viewer for an episode's selected policy (or any run)
bash cluster/view_episode.sh 24
RUN=t1_ep08_keypoints_gentle bash cluster/view_episode.sh
# it prints the node, e.g. wr22. On your PC:
ssh -N -L 8080:wr22:8080 mnidgu2s@wr0.wr.inf.h-brs.de                             # keep this open
# then open http://localhost:8080 (blank until "Starting viser server"). Stop: scancel the v2d-viser job.
```
In VS Code Remote-SSH you can instead forward `wr22:8080` in the PORTS tab.

### c) Live viewer on Ibrahim's PC (local GPU)

```bash
cd ~/v2d-track2-robotic-grounding
RUN=t1_ep24_keypoints
mkdir -p outputs/cluster/$RUN
rsync -avz mnidgu2s@wr0.wr.inf.h-brs.de:/work/mnidgu2s/project/v2d-track2-robotic-grounding/outputs/cluster/$RUN/portable_policy_349999104.safetensors outputs/cluster/$RUN/
bash scripts/diagnose/view_policy_viser.sh /out/$RUN/portable_policy_349999104.safetensors       # http://localhost:8080
```
The script mounts every local `data/adapter_out*` tree. A Tier 2 or Tier 3 policy needs its
tree on the PC: rsync `data/adapter_out_tier_2c` (plus `data/adapter_out_tier_2` for its meshes)
or `data/adapter_out_tier_3_cari4d_ours`. Use `policy_<step>` when no `portable_` file exists.

**A reference without a policy** (no training needed):
`scripts/diagnose/flashchord_headless_replay.py` replays it headless. Remember that replay is
object-assisted.

## 11. Packing and submitting

```bash
# cluster: pick the best evaluated run per episode and pack with the kit's unchanged packer
COMMIT_URL=https://github.com/ibrahim-1811/v2d-track2-robotic-grounding/commit/<sha> bash cluster/pack_tier.sh 1
PREFIX=t2c FRAME_REFS=outputs/tier2c_frame_references COMMIT_URL=... bash cluster/pack_tier.sh 2
PREFIX=t3 FRAME_REFS=data/reconstructed/track_2/tier_3_cari4d_ours COMMIT_URL=... bash cluster/pack_tier.sh 3
# -> outputs/submission/<prefix>/track2_tierN.csv; the packer prints the same four scores Kaggle will
```
`<sha>` must be a pushed commit that contains the training code. The local scores have matched
Kaggle exactly on every upload so far, so test locally first.

```bash
# Ibrahim's PC: push the cluster's commits, fetch the CSV, upload
cd ~/v2d-track2-robotic-grounding
git pull ssh://mnidgu2s@wr0.wr.inf.h-brs.de/work/mnidgu2s/project/v2d-track2-robotic-grounding submission_trials
git push origin submission_trials
mkdir -p ~/v2d_submit && rsync -avz mnidgu2s@wr0.wr.inf.h-brs.de:/work/mnidgu2s/project/v2d-track2-robotic-grounding/outputs/submission/t1/track2_tier1.csv ~/v2d_submit/
cd ~/v2d_submission_kit && source .venv/bin/activate
python3 submit/v2d_submit.py ~/v2d_submit/track2_tier1.csv --track track_2_tier1 -m "<what changed>, <sha>" --wait
```
Select the two final submissions per tier on Kaggle before the freeze. The t2r entry is not
to be a final selection (it was tuned against Tier 1).

## 12. Tests and health checks

```bash
apptainer exec -B $PWD --pwd $PWD containers/flash_chord.sif python -m pytest pipeline/tests -q   # host-side tests
bash scripts/verify/verify_keypoint_flash_sac.sh     # keypoint objective: configs reproducible, kernels match NumPy
bash scripts/verify/verify_velocity_flash_sac.sh     # velocity objective
bash scripts/verify/verify_step_C.sh                 # PC: all 30 Tier 1 references valid
bash scripts/verify/verify_flashchord_smoke.sh       # PC: NVIDIA's G1 example replays
```
