# V2D Challenge Track 2: Robotic Grounding

Our entry for **Track 2 (Robotic Grounding)** of the
[NVIDIA Video-to-Data Challenge 2026](https://nvidia-isaac.github.io/video_to_data/v2d_challenge/).

Track 1 (our WHAM + FoundationPose 4D reconstruction baseline) lives in a separate
repository: [`Swithinraj/v2d-track1-wham-foundationpose`](https://github.com/Swithinraj/v2d-track1-wham-foundationpose).
This repository is independent of it — no shared history, no shared working tree.

- **Status:** Step 1 complete — machine audit and documentation only. No installation yet.
- **Leaderboard freeze:** 4 November 2026, 5:00 p.m. EST

---

## 1. What Track 2 asks for

> "Third-Person-View 4D HOI → Policy. How much reconstruction error can be tolerated?"

Where Track 1 reconstructs human–object interaction *from* monocular video, Track 2
takes reconstructed hand–object motion and **grounds it on a robot in simulation**:
turning human demonstrations into robot-executable policies, and measuring how much
upstream reconstruction error that transfer can absorb.

Two things are **fixed and provided** by the organisers, so entries compete on
retargeting and policy learning rather than simulation engineering:

| Fixed | Value |
|---|---|
| Embodiment | **Unitree G1 + Dex3** hands |
| Simulator | Isaac Lab (`nvcr.io/nvidia/isaac-lab:2.3.2`) |

### The three tiers

Track 2 is split into three tiers by the quality of the input reference trajectory.
**A valid submission is a separate submission to each tier**, and teams may change
approach between them. All three are evaluated against the *clean* multi-view
captured object trajectory, and each tier is weighted equally in the final score.

| Tier | Input reference trajectory |
|---|---|
| **1** | Clean multi-view capture — an upper bound for upstream reconstruction |
| **2** | Synthetic corruption — jitter, dropout and contact errors sampled from Track 1's error distributions |
| **3** | Off-the-shelf reconstructions — today's best practice; CARI4D is provided as the off-the-shelf solution |

For Tiers 2 and 3 the organisers encourage methods that **recover better object
trajectory estimates** using anything except the clean Tier 1 captions.

### Metrics

Object-tracking accuracy by **AUC**, **SP-SR**, **MP-SR** and **MPPE**, as defined
in the CHORD manuscript.

### Submission mechanics

- Scored through **Kaggle**; each metric has its own leaderboard.
- Submissions are produced by the official `eval_robotic_grounding.py` script — you
  submit *recorded evaluations* of trained policies.
- Up to **five submissions per week**, unlimited in the final three days.
- Submissions must **include a GitHub commit**, and the code repository must be
  **shared with [`Nvidia-V2D-Challenge`](https://github.com/Nvidia-V2D-Challenge)**.
- Award eligibility also needs an agreed **code release under a research license**
  and a short technical report.

---

## 2. The official pipeline

The official workflow is a chain of containerised stages over a dataset:

```text
raw dataset + MANO models
          │
          ▼  stage: load          (image: v2d_task_library_loader)
   <ds>_loaded  —  per-frame MANO joints + object poses (Parquet)
          │
          ▼  stages: segment → urdf → processed → support → vis
          │          (image: robotic-grounding, on Isaac Lab 2.3.2)
          ▼
   <ds>_processed          retargeted robot motion
   object_assets/urdfs/    object URDFs
   reconstructed_stage/    support surfaces (.usda)
   <ds>_html/              viser recording + MP4
```

| Stage | What it produces |
|---|---|
| `load` | MANO forward kinematics over the raw dataset → per-frame hand joints and object poses |
| `segment` | Sequence segmentation (hot3d only) |
| `urdf` | Object URDFs from the dataset meshes |
| `processed` | **IK retargeting** of human hands onto the robot |
| `support` | Reconstructed support surfaces as USD stages |
| `vis` | viser recording and MP4 for review |

Two further tracks of work sit alongside that chain:

- **FlashCHORD** — GPU-parallel reference tracking with Newton, Warp, JAX, PPO and
  FlashSAC, in its own container. Needs a 48 GiB GPU (see §5).
- **GR00T post-training** — deliberately split out, because it needs Python 3.10
  against the pipeline's 3.11.

You drive the chain from the host with `scripts/run_pipeline_docker.py`; it selects
the right image per stage, so you never shell into a container.

### Official sources we track

| What | Where |
|---|---|
| Challenge, tiers, submission rules | [v2d_challenge](https://nvidia-isaac.github.io/video_to_data/v2d_challenge/) |
| Track 2 code and prerequisites | [`robotic_grounding`](https://github.com/nvidia-isaac/video_to_data/tree/main/robotic_grounding) |
| Full setup walkthrough | [`docs/SETUP.md`](https://github.com/nvidia-isaac/video_to_data/blob/main/robotic_grounding/docs/SETUP.md) |
| Architecture map | [`docs/ARCHITECTURE.md`](https://github.com/nvidia-isaac/video_to_data/blob/main/robotic_grounding/docs/ARCHITECTURE.md) |
| Pinned container base | [`workflow/Dockerfile`](https://github.com/nvidia-isaac/video_to_data/blob/main/robotic_grounding/workflow/Dockerfile) → `nvcr.io/nvidia/isaac-lab:2.3.2` |
| FlashCHORD | [`flash_chord/README.md`](https://github.com/nvidia-isaac/video_to_data/blob/main/robotic_grounding/flash_chord/README.md) |

Everything in our docs is quoted from these sources with the date it was read.
We do not record version numbers we have not verified against them.

---

## 3. How this repository is organised

```text
.
├── README.md              this file — objective, layout, setup status
└── docs/
    ├── SETUP_AUDIT.md     measured machine specs, raw command output, detected issues
    └── REQUIREMENTS.md    official requirements vs. our machine, classified
```

Planned, added as each step lands:

```text
├── config/                our pinned settings, tier and dataset choices
├── pipeline/
│   ├── setup/             one-time setup (Docker, images, assets)
│   └── stages/            per-stage entry points
└── scripts/               analysis, validation and review helpers
```

We mirror the Track 1 repository's conventions deliberately — `config/`, `docs/`,
`pipeline/`, `scripts/` — so both tracks read the same way.

### Rules for this repository

- **The official repo is a dependency, not a fork.** We clone
  `nvidia-isaac/video_to_data` separately (gitignored as `official/`) and keep our
  own code here. We do not vendor it.
- **No large or licensed files.** [`.gitignore`](.gitignore) excludes `data/`,
  `weights/`, `outputs/`, `containers/`, `third_party/` and `official/`. MANO and
  other license-gated assets are never committed.
- **This repository is public.** Nothing secret goes in — no tokens, no NGC or
  Kaggle credentials, no license-gated model weights.
- **Versions get verified, not guessed.** Quote the official source and the date.

---

## 4. Setup status

Step 1 is an audit. **Nothing has been installed, downloaded, built or launched.**

| Area | Status |
|---|---|
| Machine audited | Done — [docs/SETUP_AUDIT.md](docs/SETUP_AUDIT.md) |
| Requirements compared against official docs | Done — [docs/REQUIREMENTS.md](docs/REQUIREMENTS.md) |
| Docker installed | **No — blocker** |
| NVIDIA Container Toolkit | Yes, 1.20.1, verified against the GPU |
| NVIDIA driver / CUDA | Yes, 580.178.04 / 13.0 — exceeds the recommended 580.126.09 |
| NGC credentials | No |
| Kaggle account / API token | No |
| Official repo cloned | No |
| Container images built | No |
| MANO models staged | No |
| Datasets downloaded | No |

Of 46 catalogued requirements: **19 PASS, 18 MISSING, 2 INCOMPATIBLE, 7 NOT TESTED**.

The host is in good shape underneath — recent driver, working container toolkit,
the NVIDIA Docker runtime already registered in `/etc/docker/daemon.json`, the user
already in the `docker` group, Git LFS present, and both Python 3.11 and 3.10
available. Nothing on the system conflicts with Track 2.

### Development machine at a glance

| | |
|---|---|
| OS | Ubuntu 24.04.5 LTS, kernel 7.0.0-38-generic |
| CPU | Intel i7-13650HX — 14 cores / 20 threads |
| RAM | 31 GiB (~20 GiB available) + 8 GiB swap |
| GPU | NVIDIA RTX 5070 Laptop — **8151 MiB VRAM**, sm_120 |
| Driver / CUDA | 580.178.04 / 13.0 |
| Disk | 122 GB free of 270 GB on `/` |
| Local Isaac | Isaac Sim 5.1.0, Isaac Lab v2.3.0 (reference only — the pipeline uses the 2.3.2 container) |

---

## 5. Known constraints

**FlashCHORD full training will not run on this machine.** The official requirement
is "48 GiB GPU with ≥46,000 MiB free"; this GPU has 8151 MiB in total and is also
driving the desktop. That is roughly a 6× shortfall and no configuration change
closes it. FlashCHORD needs a 48 GB-class GPU (A6000 Ada, L40S, A100, H100) or a
cluster allocation.

The **retargeting pipeline itself states no VRAM figure** in the official docs, so
it is not ruled out — but Isaac Lab rendering plus IK on ~7.3 GiB usable VRAM is
tight and needs measuring once Docker exists.

Also worth knowing:

- **This is a development machine, not the run machine.** Track 1 required compute
  capability 8.0+ and ≥24 GB VRAM and ran on different hardware; Track 2's heavier
  stages probably need to as well. Worth settling where the real compute lives
  before committing to a tier strategy.
- **The GPU drives the desktop**, costing ~0.7–1.0 GiB of an already tight budget.
  Headless would recover it.
- **The host `nvcc` is CUDA 12.0**, which cannot target sm_120. Harmless for Track 2
  as specified — the container brings its own toolchain — but nothing CUDA can be
  built on the host.
- **Disk is shared and 53% full.** Three container images plus datasets will press
  on the 122 GB free.
- **Three weeks to the freeze.** With a separate submission required per tier, the
  schedule is the binding constraint as much as the hardware.

---

## 6. Next steps

Step 2 onward, in dependency order. None of this has been started.

1. **Install Docker Engine** and confirm the daemon starts with the existing
   `/etc/docker/daemon.json`. Unblocks everything else.
2. **Verify GPU passthrough** — `docker run --rm --gpus all … nvidia-smi`. The one
   check Step 1 could not perform.
3. **Get NGC access** and confirm `docker login nvcr.io` succeeds, so the
   `nvcr.io/nvidia/isaac-lab:2.3.2` pull works.
4. **Clone the official repo** and `git lfs pull` the robot URDFs and meshes.
5. **Run `workflow/setup_deps.sh`** for git-lfs, pipx and pre-commit.
6. **Create the host orchestrator venv** and install the three `reconstruction/`
   modules editable.
7. **Register for MANO** at [mano.is.tue.mpg.de](https://mano.is.tue.mpg.de) and
   stage the two `.pkl` files in the manotorch layout.
8. **Build both images**, watching disk headroom.
9. **Measure VRAM** on the retargeting stages at 8 GiB before committing this
   machine to the pipeline.
10. **Set up Kaggle** — account, API token, and accept the competition rules for
    each of the three tiers.
11. **Arrange a ≥48 GB GPU** for FlashCHORD.
12. **Run one sequence end to end** on Tier 1, then plan Tiers 2 and 3.

Steps 1–8 and 10 are setup. Step 11 is a resourcing decision for the team.

---

## 7. Licence

Not yet chosen. Award eligibility requires "agreed code release (research
license)", and this repository is public with no licence file, which currently
means all rights reserved. To be decided before submission.
