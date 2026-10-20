# Track 2 — Requirements vs. This Machine

Every requirement below is taken from NVIDIA's current official Track 2 sources,
read on **2026-10-15**:

- [`robotic_grounding/README.md`](https://github.com/nvidia-isaac/video_to_data/blob/main/robotic_grounding/README.md) — "Prerequisites"
- [`robotic_grounding/docs/SETUP.md`](https://github.com/nvidia-isaac/video_to_data/blob/main/robotic_grounding/docs/SETUP.md) — "§2 Prerequisites (host)"
- [`robotic_grounding/workflow/Dockerfile`](https://github.com/nvidia-isaac/video_to_data/blob/main/robotic_grounding/workflow/Dockerfile) — pinned base image
- [`robotic_grounding/workflow/setup_deps.sh`](https://github.com/nvidia-isaac/video_to_data/blob/main/robotic_grounding/workflow/setup_deps.sh)
- [`robotic_grounding/flash_chord/README.md`](https://github.com/nvidia-isaac/video_to_data/blob/main/robotic_grounding/flash_chord/README.md) — "Host prerequisites"

Versions are quoted from those files, not inferred. Where the official docs state
no figure, the requirement is marked **NOT TESTED** rather than guessed.

Measured values come from [SETUP_AUDIT.md](SETUP_AUDIT.md).

**Updated 2026-10-15** after Docker Engine 29.9.1 was installed, GPU passthrough
was validated, and NGC registry access was tested. Rows 2.1, 2.5, 2.6 and **3.1**
are now **PASS**; 3.2, 3.3 and 8.4 changed status or gained evidence; 2.2 and 2.4
were re-verified against a running daemon. Everything else is unchanged from the
original audit.

**Classification key**

| Status | Meaning |
|---|---|
| **PASS** | Requirement is met on this machine today |
| **MISSING** | Not present, but installable without a hardware change |
| **INCOMPATIBLE** | Present or fixed in a form that cannot satisfy the requirement on this hardware |
| **NOT TESTED** | Could not be verified under this step's read-only, no-download constraint, or the official docs state no target |

---

## 1. Host platform

| # | Requirement | Official wording / source | This machine | Status |
|---|---|---|---|---|
| 1.1 | Linux | "Linux and NVIDIA driver **580 or newer**" (flash_chord) | Ubuntu 24.04.5 LTS, kernel 7.0.0-38-generic | **PASS** |
| 1.2 | Ubuntu | Docker install link targets Ubuntu; SETUP.md refers to "a stock Ubuntu host" | Ubuntu 24.04.5 LTS (noble) | **PASS** |
| 1.3 | NVIDIA GPU | "a recent NVIDIA driver + an NVIDIA GPU" (SETUP.md §2) | RTX 5070 Laptop GPU (sm_120) | **PASS** |
| 1.4 | NVIDIA driver | "NVIDIA Driver Version 580.126.09 … Recommended"; "driver **580 or newer**" | 580.178.04 | **PASS** |
| 1.5 | CUDA | "CUDA Version: 13.0 Recommended" | 13.0 (driver-reported); `nvidia-container-cli info` confirms | **PASS** |
| 1.6 | CPU | No figure stated | i7-13650HX, 14C/20T | **NOT TESTED** |
| 1.7 | System RAM | No overall figure. `dummy_agent` is "RAM-bound (~10 GB/process)" | 31 GiB total, ~20 GiB available | **PASS** (single process) |
| 1.8 | Disk space | No figure stated anywhere in the official docs | 122 GB free on `/` | **NOT TESTED** |

On 1.5: the *driver* exposes CUDA 13.0, which is what the requirement is about.
The host `nvcc` is a separate matter — see §6.1.

---

## 2. Container stack

| # | Requirement | Official wording / source | This machine | Status |
|---|---|---|---|---|
| 2.1 | Docker | "Install [Docker] and [post-installation] steps." (README Prerequisites) | **Docker Engine 29.9.1** (`docker-ce 5:29.9.1-1~ubuntu.24.04~noble`), `containerd.io` 2.4.2, `runc` 1.5.2, buildx 0.38.0. `docker.service` active + enabled; `docker run --rm hello-world` exits 0 | **PASS** |
| 2.2 | Docker non-root access | "[non-root access]" (flash_chord); Docker post-install step | `ibrahim` in `docker` group (GID 984) *before* install, so no re-login needed. `/var/run/docker.sock` is `srw-rw---- root:docker`; `docker info` and `hello-world` both verified without `sudo` | **PASS** |
| 2.3 | NVIDIA Container Toolkit | "Install [NVIDIA Container Toolkit]." (README Prerequisites) | 1.20.1, all four packages, `nvidia-ctk`/`nvidia-container-cli`/`nvidia-container-runtime` on PATH | **PASS** |
| 2.4 | NVIDIA runtime registered with Docker | Implied by the toolkit install guide | Confirmed live: `docker info` reports `Runtimes: io.containerd.runc.v2 nvidia runc`. The pre-existing `/etc/docker/daemon.json` survived the install unmodified, and Docker 29 additionally auto-discovers the toolkit's CDI specs (`cdi: nvidia.com/gpu=all`) | **PASS** (verified) |
| 2.5 | GPU visible inside a container | Standard `docker run --gpus all … nvidia-smi` check | **Validated.** `docker run --rm --gpus all nvidia/cuda:13.0.0-base-ubuntu24.04 nvidia-smi` → **exit 0**, reporting `NVIDIA GeForce RTX 5070 Laptop GPU`, `8151 MiB`, driver `580.178.04`, CUDA `13.0`, `compute_cap 12.0`. Also passes via CDI (`--device nvidia.com/gpu=all`) and `--runtime=nvidia`. `/dev/nvidia-uvm` and `libcuda.so.580.178.04` confirmed injected | **PASS** |
| 2.6 | Docker Compose | **Not required** — official workflow uses `docker run` via `workflow/run.sh` | `docker-compose-plugin` v5.6.0, installed as part of Docker's standard install command | **PASS** (not required) |

**2.1 — previously the one requirement blocking everything else — is now met.**
The container stack is complete end to end: Docker Engine, containerd, the NVIDIA
Container Toolkit, and the `nvidia` runtime registered and visible to the daemon.

**§2 is now complete — every requirement in the container stack passes.** 2.5,
the last gap, was validated on 2026-10-15: see [SETUP_AUDIT.md §10](SETUP_AUDIT.md).
No fix was needed; passthrough worked on the first attempt.

One Docker 29 detail worth carrying into the image builds: the default storage
driver is now `overlayfs` on the containerd snapshotter
(`io.containerd.snapshotter.v1`), not the older `overlay2` graph driver. Nothing
in the official Track 2 docs depends on either, but layer accounting and
`docker system df` output differ from pre-28 Docker.

---

## 3. Registry and credentials

| # | Requirement | Official wording / source | This machine | Status |
|---|---|---|---|---|
| 3.1 | NGC access | "the retarget image builds `FROM nvcr.io/nvidia/isaac-lab:2.3.2`, which Docker pulls automatically during the build. Make sure `docker login nvcr.io` … succeeds before you build." (SETUP.md §2) | **Access verified without credentials.** `nvcr.io` grants an anonymous token (`sub: sf:guestclient`) with `pull` on `nvidia/isaac-lab`; a layer blob returns HTTP 200. The image is publicly pullable, so the build is not gated on a login | **PASS** |
| 3.2 | NGC API key | "password = your [NGC API key]" (SETUP.md §2) | Still not configured — no `~/.docker/config.json`, no `~/.ngc`, no `ngc` CLI, no NGC env vars. **Not required** for any image Track 2 uses (see 4.1 and the note below) | **MISSING** (not required) |
| 3.3 | Team registry access | "Make sure you have access to `nvcr.io/your-org/your-team`." (README) | No organiser registry path has been given to us. Evidence that private paths do exist and would need credentials: `nvcr.io/nvidian/v2p_video_agent` returns an anonymous token with **empty** actions (`[]`) versus `['pull']` for every `nvidia/*` repo | **NOT TESTED** |
| 3.4 | Hugging Face token | Needed only for the optional pretrained RL checkpoints ("run `huggingface-cli login` with an authorized token first", SETUP.md §10) | Not checked / optional. Note the **challenge dataset** also lives on Hugging Face (`nvidia/video_to_data_challenge`), so a token will likely be needed for that regardless — see [TRACK2_MISSION.md](TRACK2_MISSION.md) | **NOT TESTED** |

### Anonymous `nvcr.io` pull scope, measured 2026-10-15

Every repository the Track 2 workflow references was checked by requesting an
anonymous token and decoding its granted actions. No credentials were sent and no
image was downloaded.

| Repository | Anonymous actions | Used by |
|---|---|---|
| `nvidia/isaac-lab` | `['pull']` | **Track 2 base image** (`workflow/Dockerfile`) |
| `nvidia/isaac-sim` | `['pull']` | referenced upstream |
| `nvidia/cuda` | `['pull']` | FlashCHORD container |
| `nvidia/tensorrt` | `['pull']` | referenced upstream |
| `nvidia/base/ubuntu` | `['pull']` | referenced upstream |
| `nvcr.io/nvidian/v2p_video_agent` | `[]` — **no access** | `video_ingestion_agent/osmo_workflows/*.yaml` only — **not Track 2** |

So **no NGC credential is required for anything Track 2 builds or runs.** The one
inaccessible image belongs to NVIDIA's internal `nvidian` org and is referenced
only by the video-ingestion agent, which Track 2 does not use.

A `docker login nvcr.io` is still worth configuring if the organisers later hand
us a team registry path (3.3), but it is no longer on the critical path.

---

## 4. Container images

| # | Requirement | Official wording / source | This machine | Status |
|---|---|---|---|---|
| 4.1 | `nvcr.io/nvidia/isaac-lab:2.3.2` base | `FROM nvcr.io/nvidia/isaac-lab:2.3.2 AS base` (workflow/Dockerfile) | Not pulled. Access is confirmed (3.1) and needs no login; pulling is simply out of scope so far. Manifest inspected: 29 layers, **8.41 GB compressed** for `linux/amd64`, digest `sha256:f07c37e3…` | **MISSING** |
| 4.2 | `robotic-grounding` image | Built locally on the 2.3.2 base (SETUP.md §1, §3) | Not built | **MISSING** |
| 4.3 | `v2d_task_library_loader` image | "Built from a public PyTorch base" (SETUP.md §1) | Not built | **MISSING** |
| 4.4 | FlashCHORD Newton/Warp/JAX image | "Uses a separate Newton/Warp/JAX container" (README); needs access to `nvidia/cuda` and `ghcr.io/astral-sh/uv` | Not built | **MISSING** |
| 4.5 | Local Isaac Lab install | "**No local Isaac Lab install is needed.**" (SETUP.md §2) | Isaac Lab **v2.3.0** + Isaac Sim 5.1.0 present locally | **PASS** (not required; version-skewed from 2.3.2 — do not substitute) |

---

## 5. Python environments

| # | Requirement | Official wording / source | This machine | Status |
|---|---|---|---|---|
| 5.1 | Python 3.11 (pipeline) | "this repo's Python 3.11 + IsaacLab container" (README §GR00T); `py_version = 311` (pyproject.toml) | Supplied by the image; host also has 3.11.17 | **PASS** |
| 5.2 | Python 3.10 (GR00T post-training) | "GR00T needs Python 3.10 and its own dependencies, which conflict with this repo's Python 3.11 + IsaacLab container, so the split is deliberate" (README) | 3.10.20 available (uv-managed) | **PASS** |
| 5.3 | Host venv for the orchestrator | "`python3 -m venv ~/venvs/v2d`" then `pip install -e modules/v2d_common -e modules/v2d_docker -e modules/v2d_task_library_loader/docker` (README Prerequisites) | Not created | **MISSING** |
| 5.4 | Bare `python` resolvable on the host | "a stock Ubuntu host ships only `python3` and otherwise fails with `python: command not found`" (README) | Only `python3` exists; satisfied once 5.3's venv is activated | **MISSING** (resolved by 5.3) |
| 5.5 | uv (FlashCHORD host install only) | "`uv venv --python 3.11`" (flash_chord, "Host installation (without Docker)") | uv 0.11.7 | **PASS** |
| 5.6 | Host PyTorch | Not required on the host — PyTorch lives in the images | None (one unrelated venv has torch 2.5.1+**cpu**) | **PASS** (not required) |

---

## 6. Build and developer tooling

| # | Requirement | Official wording / source | This machine | Status |
|---|---|---|---|---|
| 6.1 | CUDA toolkit on host | Not required; the Dockerfile compiles nothing on the host | `nvcc` 12.0 (Ubuntu archive), no `/usr/local/cuda`, no cuDNN. CUDA 12.0 predates `sm_120` and cannot target this GPU | **INCOMPATIBLE** (not required) |
| 6.2 | Git | Implied throughout | git 2.43.0 | **PASS** |
| 6.3 | Git LFS | "Install Git LFS and `pre-commit` dependencies" (README); "Robot URDFs/meshes are **Git-LFS** — run `git lfs pull` or they'll be broken pointers" (SETUP.md §9) | git-lfs 3.4.1 | **PASS** |
| 6.4 | `git lfs pull` of robot assets | "fetches the robot URDFs + meshes (sharpa_wave, g1)" (SETUP.md §2) | Official repo not cloned yet | **MISSING** |
| 6.5 | pipx | `sudo apt install -y git-lfs pipx` (workflow/setup_deps.sh) | Not installed | **MISSING** |
| 6.6 | pre-commit | `pipx install pre-commit` (workflow/setup_deps.sh) | Not installed | **MISSING** |
| 6.7 | ffmpeg | Installed inside the image; useful on the host for video work | ffmpeg 6.1.1 | **PASS** |
| 6.8 | gcc / cmake | Not required on the host | gcc 13.3.0, cmake 3.28.3 | **PASS** (not required) |

---

## 7. Repository and licensed assets

| # | Requirement | Official wording / source | This machine | Status |
|---|---|---|---|---|
| 7.1 | Official `video_to_data` repo clone | Needed for `robotic_grounding/` **and** the sibling `reconstruction/` modules the orchestrator installs | Not cloned | **MISSING** |
| 7.2 | `reconstruction/modules/{v2d_common,v2d_docker,v2d_task_library_loader}` | `pip install -e` targets in the README's venv step | Not present (part of 7.1) | **MISSING** |
| 7.3 | MANO hand models | License-gated; "Extract just the two `.pkl` into `<HMD>/mano/models/` (the manotorch layout)" (SETUP.md); register at mano.is.tue.mpg.de | Not present | **MISSING** |
| 7.4 | `<HMD>` human-motion-data root | "a directory you choose" (README §Environment & Credentials) | Not chosen yet | **MISSING** |
| 7.5 | Per-dataset raw downloads | One setup doc per dataset (ARCTIC, DexYCB, GRAB, H2O, HOT3D, TACO) | None downloaded (out of scope for this step) | **NOT TESTED** |

---

## 8. FlashCHORD (accelerated training) — the hardware wall

| # | Requirement | Official wording / source | This machine | Status |
|---|---|---|---|---|
| 8.1 | VRAM for full training | "Full training: **48 GiB GPU with ≥46,000 MiB free**, as validated in EVT-07." (flash_chord) | **8151 MiB total**, 661 MiB already used by the desktop | **INCOMPATIBLE** |
| 8.2 | VRAM for replay / one-world viewing | "Replay and one-world viewing use less memory." — no figure given | **Measured** for Isaac Lab `SonicG1-ReconBody-v0` at `--num_envs 1`: peak **3949 MiB** total (≈2.9 GiB for the scene above a ~1.0 GiB desktop), flat across a 60-step rollout, no OOM. See [SETUP_AUDIT.md §20.11](SETUP_AUDIT.md). FlashCHORD's own replay path is still unmeasured | **PASS** (one env, Isaac Lab) |
| 8.3 | Driver 580+ | "Linux and NVIDIA driver **580 or newer**" | 580.178.04 | **PASS** |
| 8.4 | Access to `nvidia/cuda` and `ghcr.io/astral-sh/uv` | flash_chord host prerequisites | **`nvidia/cuda` confirmed** — `13.0.0-base-ubuntu24.04` pulled (562 MB) and run successfully. `ghcr.io/astral-sh/uv` not pulled | **PASS** (`nvidia/cuda`) / **NOT TESTED** (`uv` image) |

**8.1 cannot be fixed on this machine.** The requirement is 46,000 MiB free; the
GPU has 8151 MiB in total — about **5.9× short**, and it is also driving the
desktop. No configuration change, headless boot or memory-saving flag closes a
gap of that size. FlashCHORD full training needs different hardware: a 48 GB
class GPU (A6000 Ada, L40S, A100 40/80 GB, H100) or a cluster allocation.

Note also that the **retargeting / Isaac Lab pipeline states no VRAM figure** in
the official docs, so it is not classified as blocked — it is untested at 8 GiB
and will need an empirical check once Docker exists. Isaac Lab rendering plus IK
on 7.3 GiB usable VRAM is plausible for small scenes and likely to be tight.

---

## 9. Score card

| Status | Count | Requirements |
|---|---|---|
| **PASS** | 23 | 1.1–1.5, 1.7, 2.1–2.6, **3.1**, 4.5, 5.1, 5.2, 5.5, 5.6, 6.2, 6.3, 6.7, 6.8, 8.3 |
| **MISSING** | 14 | 3.2, 4.1–4.4, 5.3, 5.4, 6.4–6.6, 7.1–7.4 |
| **INCOMPATIBLE** | 2 | 6.1 (host `nvcc`, not required), **8.1 (FlashCHORD VRAM)** |
| **NOT TESTED** | 7 | 1.6, 1.8, 3.3, 3.4, 7.5, 8.2, 8.4 (`uv` image half only) |

46 requirements in total.

Change since the original audit: **PASS 19 → 23**, **MISSING 18 → 14**,
**NOT TESTED 7 → 7** (2.5 left, 8.4 arrived).

- 2.1 (Docker) and 2.6 (Compose) moved MISSING → PASS when Docker was installed.
- 2.5 (GPU in container) moved NOT TESTED → **PASS** on validation.
- **3.1 (NGC access) moved MISSING → PASS** — not by authenticating, but by
  measuring that the base image needs no authentication.
- 3.2 stays MISSING but is reclassified as **not required**.
- 8.4 moved MISSING → split: `nvidia/cuda` verified, the `uv` image untested.

**Section 2 is fully green.** All 15 remaining **MISSING** items are ordinary
installation or download steps, and none of them are in the container stack. The
next ones are gated on credentials (3.1, 3.2) and a license registration (7.3),
not on this machine. Only one **INCOMPATIBLE** item matters: **8.1**.

---

## 10. Verdict

**This machine can host the Track 2 retargeting and data-generation pipeline
once NGC credentials and the MANO assets are in place. It cannot run FlashCHORD
full training at any configuration.**

Docker was the one hard blocker and it is now gone. The container stack is
complete and verified **end to end**: Docker Engine 29.9.1 active and enabled,
non-root access working, NVIDIA Container Toolkit 1.20.1 intact, the `nvidia`
runtime registered and visible in `docker info`, and — as of 2026-10-15 — a
GPU-enabled container confirmed to see the RTX 5070, driver 580.178.04, CUDA
13.0 and all 8151 MiB of VRAM. The rest of the foundation was already
strong — the driver and CUDA version exceed what NVIDIA recommends, Git LFS is
present, and both required Python versions (3.11 and 3.10) are available.
Nothing on the system conflicts with Track 2.

The remaining blockers are **no longer about this machine**, and one of them has
evaporated: **NGC credentials turned out not to be needed** (3.1/3.2). What is
left externally is the MANO licence registration (7.3) and the challenge dataset
download. The only local unknowns are measurements that cannot be taken until the
Track 2 images exist — real VRAM use under the retargeting pipeline (8.2), the
FlashCHORD evaluator's VRAM, and disk headroom (1.8) against the 8.41 GB
compressed base image.

What stands between here and a first run, in order:

1. ~~Install Docker Engine (**2.1**)~~ — **done 2026-10-15**, Docker 29.9.1.
2. ~~Verify GPU-in-container (**2.5**)~~ — **done 2026-10-15, PASS**, exit 0, no
   fix required.
3. ~~Obtain NGC credentials and `docker login nvcr.io` (**3.1**, **3.2**)~~ —
   **not required.** Verified 2026-10-15 that `nvcr.io/nvidia/isaac-lab:2.3.2` is
   anonymously pullable. Only needed if the organisers hand us a private team
   registry path (**3.3**).
4. Clone the official repo and `git lfs pull` (**7.1**, **6.4**).
5. Run `workflow/setup_deps.sh` for pipx + pre-commit (**6.5**, **6.6**).
6. Create the host orchestrator venv (**5.3**).
7. Register for and stage the MANO models (**7.3**).
8. Build the two images (**4.1**–**4.3**) — budget disk headroom against the
   122 GB free (**1.8**).
9. Measure actual VRAM use of the retargeting stages at 8 GiB (**8.2**) before
   committing to this machine for the pipeline.
10. Secure a ≥48 GB GPU elsewhere for FlashCHORD (**8.1**).

Steps 1 and 2 are complete, and step 3 turned out to be unnecessary — the
container stack is finished and the registry is open. Steps 4–9 remain; the
MANO licence (step 7) is now the only external dependency. Step 10 is a
resourcing decision, not a setup task.
