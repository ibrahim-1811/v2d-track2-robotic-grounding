# Track 2 — System Setup Audit

Audit of the development machine against what Track 2 (Robotic Grounding) needs.
All figures below are command output captured on this host, not estimates.

- **Host:** `ibrahim5i`
- **Audit date:** 2026-10-15
- **Repository:** `ibrahim-1811/v2d-track2-robotic-grounding`
- **Scope:** read-only. Nothing was installed, upgraded, removed or downloaded.

---

## 1. Summary

| Area | Result |
|---|---|
| OS / kernel | Ubuntu 24.04.5 LTS, kernel 7.0.0-38-generic — OK |
| CPU | i7-13650HX, 14 cores / 20 threads — OK |
| RAM | 31 GiB total, ~20 GiB available — OK |
| GPU | RTX 5070 Laptop, **8151 MiB VRAM** — under-sized for FlashCHORD training |
| Driver | 580.178.04, CUDA 13.0 — OK, exceeds the recommended 580.126.09 |
| NVIDIA Container Toolkit | 1.20.1 installed, 3 CDI devices enumerate — OK |
| Docker | **not installed** — the single hard blocker |
| NGC registry login | **not configured** |
| Git LFS | 3.4.1 — OK |
| Disk | 122 GB free on `/` — adequate but not generous |

---

## 2. Operating system and kernel

```
$ cat /etc/os-release
PRETTY_NAME="Ubuntu 24.04.5 LTS"
VERSION="24.04.5 LTS (Noble Numbat)"
VERSION_CODENAME=noble

$ uname -a
Linux ibrahim5i 7.0.0-38-generic #38~24.04.4-Ubuntu SMP PREEMPT_DYNAMIC
Mon Sep 14 16:37:11 UTC 2 x86_64 x86_64 x86_64 GNU/Linux
```

Ubuntu 24.04 is the distribution the official Docker and NVIDIA Container Toolkit
install guides target, so no distro-level work is needed.

---

## 3. CPU

```
$ lscpu
Architecture:          x86_64
Vendor ID:             GenuineIntel
Model name:            13th Gen Intel(R) Core(TM) i7-13650HX
CPU(s):                20
Core(s) per socket:    14
Thread(s) per core:    2
Socket(s):             1
CPU max MHz:           4900.0000
```

20 logical CPUs. Enough for the retargeting and IK stages, which are
GPU-bound rather than CPU-bound.

---

## 4. Memory and swap

```
$ free -h
               total        used        free      shared  buff/cache   available
Mem:            31Gi        10Gi        14Gi       379Mi       6.6Gi        20Gi
Swap:          8.0Gi          0B       8.0Gi

MemTotal:       32575204 kB
SwapTotal:       8388604 kB
```

Note: ~10 GiB was already in use by the running desktop session at audit time.
The official setup notes that `dummy_agent` (optional RL playback) is
"RAM-bound (~10 GB/process) and must run headless" — one such process fits in
the ~20 GiB available; several would not.

---

## 5. GPU, VRAM, driver and CUDA

```
$ nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version,compute_cap,pcie.link.gen.max --format=csv
name, memory.total [MiB], memory.used [MiB], driver_version, compute_cap, pcie.link.gen.max
NVIDIA GeForce RTX 5070 Laptop GPU, 8151 MiB, 661 MiB, 580.178.04, 12.0, 5
```

```
$ nvidia-smi   (header)
NVIDIA-SMI 580.178.04   Driver Version: 580.178.04   CUDA Version: 13.0
GPU 0: NVIDIA GeForce RTX 5070 Laptop GPU — 661MiB / 8151MiB, 7W / 100W, 47C
```

```
$ lspci | grep -Ei "vga|display"
00:02.0 Display controller: Intel Corporation Raptor Lake-S UHD Graphics (rev 04)
01:00.0 VGA compatible controller: NVIDIA Corporation Device 2d18 (rev a1)
```

Key points:

- **Driver 580.178.04 / CUDA 13.0** meets and exceeds the officially recommended
  "NVIDIA Driver Version 580.126.09, CUDA Version: 13.0" and FlashCHORD's
  "driver 580 or newer".
- **Compute capability 12.0** (Blackwell, `sm_120`). Any CUDA code we build
  ourselves must be built with a toolkit that knows `sm_120` (CUDA 12.8+).
- **8151 MiB total VRAM, 661 MiB already consumed** by gnome-shell/Xwayland.
  Usable VRAM is therefore ~7.3 GiB. This is the machine's main limitation —
  see [REQUIREMENTS.md](REQUIREMENTS.md) §FlashCHORD. Desktop VRAM use drifts
  with the session: a re-check minutes later during the validation pass read
  979 MiB, so budget ~0.7–1.0 GiB lost to the desktop, not a fixed 661 MiB.
- The GPU is also driving the desktop (`Disp.A = On`), so it is never fully free.
  Running headless would recover ~660 MiB.

### Host CUDA toolkit

```
$ nvcc --version
Cuda compilation tools, release 12.0, V12.0.140

$ which nvcc
/usr/bin/nvcc

$ ls -d /usr/local/cuda*
ls: cannot access '/usr/local/cuda*': No such file or directory

$ dpkg -l | grep -Ei "cudnn|cuda-toolkit|nvidia-driver"
ii  nvidia-cuda-toolkit       12.0.140~12.0.1-4build4   NVIDIA CUDA development toolkit
ii  nvidia-cuda-toolkit-doc   12.0.1-4build4
ii  nvidia-driver-580-open    580.178.04-0ubuntu0.24.04.1
```

The host `nvcc` is the Ubuntu-archive **CUDA 12.0**, which predates `sm_120`
and cannot compile for this GPU. No cuDNN package is installed. This does **not**
block Track 2 as officially specified — all compilation happens inside the
Isaac Lab container, which ships its own toolchain — but it does rule out
building CUDA extensions on the host.

---

## 6. Disk space

```
$ df -hT /
Filesystem     Type  Size  Used Avail Use% Mounted on
/dev/nvme0n1p6 ext4  270G  134G  122G  53% /

$ df -h (physical only)
/dev/nvme0n1p6 ext4      270G  134G  122G  53% /
/dev/nvme0n1p1 vfat      256M   54M  203M  21% /boot/efi
```

Single 270 GB ext4 root, **122 GB free**. `/home` is on the same filesystem.
The official docs state no disk figure, but the stack is image-heavy: the
`nvcr.io/nvidia/isaac-lab:2.3.2` base plus the locally built `robotic-grounding`
layer and the `v2d_task_library_loader` image, before any dataset. 122 GB is
workable for one or two datasets; it is not enough for the full dataset set
alongside the existing 134 GB of other data on this disk.

---

## 7. Python, pip and uv

```
$ python3 --version
Python 3.12.3            (/usr/bin/python3)

$ pip3 --version
pip 24.0 from /usr/lib/python3/dist-packages/pip (python 3.12)

$ uv --version
uv 0.11.7                (/home/ibrahim/.local/bin/uv)

$ ls /usr/bin/python3.1*
/usr/bin/python3.11
/usr/bin/python3.12

$ uv python list   (installed only)
cpython-3.13.13   /home/ibrahim/.local/share/uv/python/cpython-3.13-.../bin/python3.13
cpython-3.12.3    /usr/bin/python3.12  (= python3)
cpython-3.11.17   /usr/bin/python3.11
cpython-3.10.20   /home/ibrahim/.local/bin/python3.10
```

Both interpreters Track 2 cares about are present on the host: **3.11** (the
version inside the Isaac Lab container) and **3.10** (what GR00T post-training
needs, which the official README says conflicts with the 3.11 container and is
therefore kept separate).

`conda` resolves to a pixi trampoline at `~/.pixi/bin/conda` but is broken:

```
$ conda --version
Failed to execute command: Os { code: 2, kind: NotFound, message: "No such file or directory" }
```

Not needed for Track 2; recorded so nobody relies on it.

---

## 8. Docker and Docker Compose

```
$ docker --version
bash: line 1: docker: command not found

$ docker compose version
bash: line 1: docker: command not found

$ docker-compose --version
bash: line 1: docker-compose: command not found

$ dpkg -l | grep -Ei "docker|containerd|podman"
ii  podman   4.9.3+ds1-1ubuntu0.2   amd64   tool to manage containers and pods

$ systemctl status docker
Unit docker.service could not be found.

$ ls -la /var/run/docker.sock
ls: cannot access '/var/run/docker.sock': No such file or directory

$ ls -d /var/lib/docker
ls: cannot access '/var/lib/docker': No such file or directory
```

**Docker is not installed.** This is the one hard blocker: the entire official
Track 2 workflow (`workflow/run.sh`, `scripts/run_pipeline_docker.py`) drives
`docker build` / `docker run`.

Two leftovers from an earlier setup are worth knowing about:

1. `/etc/docker/daemon.json` already exists and pre-registers the NVIDIA runtime:

   ```json
   {
       "runtimes": {
           "nvidia": {
               "args": [],
               "path": "nvidia-container-runtime"
           }
       }
   }
   ```

   A later Docker install will pick this up, which is what we want — but it also
   means the daemon will fail to start if `nvidia-container-runtime` is ever
   absent.

2. The user is **already in the `docker` group**:

   ```
   $ groups
   ibrahim adm cdrom sudo dip plugdev users lpadmin docker
   ```

   So the non-root post-install step is effectively already satisfied; no
   re-login will be needed after installing Docker.

`podman` 4.9.3 (crun runtime) is installed. The official tooling calls the
`docker` CLI directly, so podman is not a drop-in substitute without aliasing,
and we have not validated that path.

Docker Compose is not required by the official workflow — it uses plain
`docker run` through `workflow/run.sh`.

---

## 9. NVIDIA Container Toolkit

```
$ dpkg -l | grep -Ei "nvidia-container|nvidia-docker"
ii  libnvidia-container-tools       1.20.1-1
ii  libnvidia-container1:amd64      1.20.1-1
ii  nvidia-container-toolkit        1.20.1-1
ii  nvidia-container-toolkit-base   1.20.1-1

$ nvidia-ctk --version
NVIDIA Container Toolkit CLI version 1.20.1
commit: dffc40b4f820ce5c512633bac9e0418d0e05a2ee

$ which nvidia-ctk nvidia-container-cli nvidia-container-runtime
/usr/bin/nvidia-ctk
/usr/bin/nvidia-container-cli
/usr/bin/nvidia-container-runtime
```

Installed and complete. Verified read-only that it can actually see the GPU:

```
$ nvidia-container-cli info
NVRM version:   580.178.04
CUDA version:   13.0
Device Index:   0
Model:          NVIDIA GeForce RTX 5070 Laptop GPU
GPU UUID:       GPU-<redacted>
Bus Location:   00000000:01:00.0
Architecture:   12.0

$ nvidia-ctk cdi list
INFO Found 3 CDI devices
nvidia.com/gpu=0
nvidia.com/gpu=GPU-<redacted>
nvidia.com/gpu=all
```

---

## 10. GPU access through a container

**Not tested.** The usual check is
`docker run --rm --gpus all nvidia/cuda:... nvidia-smi`, which needs both Docker
and an image pull — neither is permitted at this step.

What we *can* say from the read-only checks in §9: the container stack below
Docker is healthy. `nvidia-container-cli info` talks to the driver successfully
and `nvidia-ctk cdi list` enumerates the device, so GPU passthrough is expected
to work as soon as Docker is installed. It remains unproven until tested.

---

## 11. Git and Git LFS

```
$ git --version
git version 2.43.0

$ git lfs version
git-lfs/3.4.1 (GitHub; linux amd64; go 1.22.2)
```

Both present. Git LFS matters for Track 2: the official repo stores the robot
URDFs and meshes (`sharpa_wave`, `g1`) in LFS and they are broken pointers
until `git lfs pull` runs.

---

## 12. PyTorch

```
$ python3 -c "import torch"
ModuleNotFoundError: No module named 'torch'

$ python3 -m pip list | grep -Ei "torch|numpy|opencv|isaac|omni|warp|trimesh|open3d|transformers"
aruco-opencv-msgs   6.1.2
numpy               1.26.4
```

No PyTorch in the system interpreter. One unrelated venv has a CPU-only build:

```
$ ~/real2libero/.venv/bin/python -c "import torch; ..."
torch 2.5.1+cpu None False        # version, built-CUDA, cuda.is_available()
```

That venv is Python 3.10.20 (uv-managed) and belongs to a different project.
No CUDA-enabled PyTorch exists anywhere on this host. Track 2 does not need one
on the host — PyTorch lives inside the Isaac Lab image.

---

## 13. Existing Isaac Sim and Isaac Lab installations

Both are installed locally, outside this repository:

```
$ cat ~/isaacsim/VERSION
5.1.0-rc.19+release.26219.9c81211b.glapps

$ ~/isaacsim/python.sh -c "import sys; print(sys.version)"
3.11.13 (tags/v3.11.13-dirty:498b971, Jun  9 2025) [GCC 7.3.1]

$ cd ~/IsaacLab && git describe --tags
v2.3.0

$ cd ~/IsaacLab && git log --oneline -1
3c6e67bb5c Adds matplotlib-inline to license exceptions (#3853)

$ ls -la ~/IsaacLab/_isaac_sim
~/IsaacLab/_isaac_sim -> /home/ibrahim/isaacsim
```

So: **Isaac Sim 5.1.0** (binary install, bundled Python 3.11.13) and
**Isaac Lab v2.3.0**, correctly symlinked to each other.

Two caveats:

1. Track 2's container is built `FROM nvcr.io/nvidia/isaac-lab:2.3.2`. The local
   checkout is **2.3.0** — a different, older patch release. The local install is
   *not* a substitute; the official SETUP.md says plainly
   "No local Isaac Lab install is needed."
2. `~/launch_isaac.sh` is stale — it sources
   `~/isaacsim/env_isaacsim/bin/activate`, and that venv no longer exists:

   ```
   $ ls -d ~/isaacsim/env_isaacsim
   ls: cannot access '/home/ibrahim/isaacsim/env_isaacsim': No such file or directory
   ```

   So the local Isaac Sim is not currently launchable through that script.

The local install is still useful for offline USD/Isaac Lab API inspection, and
it means Omniverse assets and caches (`~/.nvidia-omniverse`, `~/.local/share/ov`)
are already on disk.

---

## 14. Other host tooling

```
$ ffmpeg -version | head -1
ffmpeg version 6.1.1-3ubuntu5

$ cmake --version | head -1
cmake version 3.28.3

$ gcc --version | head -1
gcc (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0

$ which pipx pre-commit ngc
pipx: not found
pre-commit: not found
ngc: not found

$ ls -la ~/.docker ~/.ngc
ls: cannot access '/home/ibrahim/.docker': No such file or directory
ls: cannot access '/home/ibrahim/.ngc': No such file or directory

$ apptainer --version ; singularity --version
apptainer: command not found
singularity: command not found
```

`pipx` and `pre-commit` are what the official `workflow/setup_deps.sh` installs;
neither is present yet. No NGC or Docker registry credentials exist, so the
`nvcr.io` pull that the image build depends on would fail today.

Apptainer is absent, which is also why Track 1's `config/*.def` images cannot be
built on this machine (see §16).

---

## 15. Existing environments and dependency conflicts

Environments found on this host:

| Environment | Interpreter | Relevant contents | Relation to Track 2 |
|---|---|---|---|
| System Python | 3.12.3 (`/usr/bin/python3`) | numpy 1.26.4, no torch | Not suitable as the orchestrator env (official docs say use a venv) |
| `~/real2libero/.venv` | 3.10.20 (uv) | torch 2.5.1+cpu | Unrelated project; leave alone |
| `~/isaacsim` bundled | 3.11.13 | Isaac Sim 5.1.0 extensions | Local-only; not the Track 2 container |
| `~/IsaacLab` | uses `~/isaacsim` | Isaac Lab v2.3.0 | Version-skewed vs. the required 2.3.2 image |
| `~/.pixi` | — | broken `conda` shim | Unused |

No dependency conflicts affect Track 2 today, because Track 2 has no environment
on this machine yet and its Python dependencies are all resolved inside the
container. Two conflicts are *structural* and worth planning around:

1. **Isaac Lab 2.3.0 (local) vs 2.3.2 (required image).** Do not mix them. Track 2
   work must go through the container; the local checkout is reference material.
2. **Python 3.11 vs 3.10.** The container is 3.11; GR00T post-training needs 3.10
   and, per the official README, its dependencies "conflict with this repo's
   Python 3.11 + IsaacLab container, so the split is deliberate." Both
   interpreters are available here, so this is manageable — but it must stay two
   separate environments.
3. **Host `nvcc` 12.0 vs `sm_120` GPU.** Harmless as long as nothing is compiled
   on the host.

---

## 16. Track 1 context on this machine

Track 1 lives in a separate repository
([`Swithinraj/v2d-track1-wham-foundationpose`](https://github.com/Swithinraj/v2d-track1-wham-foundationpose)),
which this audit left untouched — no tracked file there was modified.

Worth recording: Track 1's own README requires "compute capability 8.0 or newer
… ≥ 24 GB VRAM" and Apptainer ≥ 1.2. Neither holds on this host (8 GiB VRAM, no
Apptainer), and none of Track 1's gitignored working directories
(`containers/`, `weights/`, `data/`, `outputs/`, `third_party/`) exist on this
machine — the whole checkout is under 1 MB. **Track 1 was developed and run on
different hardware.** This laptop is a development and authoring machine, and the
same is likely to be true for Track 2's heavier stages.

---

## 17. Detected issues

| # | Issue | Severity | Note |
|---|---|---|---|
| 1 | Docker not installed | **Blocker** | Whole official workflow depends on it |
| 2 | 8151 MiB VRAM vs FlashCHORD's 48 GiB requirement | **Blocker** for FlashCHORD training | ~6× short; see REQUIREMENTS.md |
| 3 | No NGC / `docker login nvcr.io` credentials | **Blocker** for the image build | `nvcr.io/nvidia/isaac-lab:2.3.2` pull will fail |
| 4 | MANO hand models not present (license-gated) | High | Needed by the loader stage; register at mano.is.tue.mpg.de |
| 5 | `pipx` and `pre-commit` missing | Low | `workflow/setup_deps.sh` installs them |
| 6 | No host orchestrator venv yet | Low | One `python3 -m venv` away |
| 7 | GPU also drives the desktop (661 MiB used) | Medium | Costs usable VRAM on an already-tight GPU |
| 8 | Host `nvcc` 12.0 cannot target `sm_120` | Low | Only matters if we build CUDA on the host |
| 9 | `~/launch_isaac.sh` references a deleted venv | Informational | Local Isaac Sim not launchable via that script |
| 10 | Broken `conda` shim in `~/.pixi/bin` | Informational | Unused by Track 2 |
| 11 | Only 122 GB free on a shared root filesystem | Medium | Images + datasets will press on this |

---

## 18. Commands run during this audit

Every command was read-only. For the record:

`pwd`, `git status`, `git branch --show-current`, `git remote -v`, `git branch -a`,
`git log --oneline`, `ls -la`, `find`, `du -sh`, `cat /etc/os-release`,
`lsb_release -a`, `uname -a`, `lscpu`, `free -h`, `cat /proc/meminfo`, `df -hT`,
`nvidia-smi`, `nvidia-smi --query-gpu=...`, `nvcc --version`, `dpkg -l`, `lspci`,
`python3 --version`, `pip3 --version`, `uv --version`, `uv python list`,
`conda --version`, `docker --version`, `docker compose version`,
`systemctl status docker`, `git --version`, `git lfs version`,
`apptainer --version`, `nvidia-ctk --version`, `nvidia-container-cli info`,
`nvidia-ctk cdi list`, `podman --version`, `podman info`, `ffmpeg -version`,
`cmake --version`, `gcc --version`, `which`, `ls`, `cat` on config files.

Nothing was installed, upgraded, removed, downloaded or launched during the audit,
and no global system setting was changed. The GPU UUID reported by
`nvidia-container-cli info` and `nvidia-ctk cdi list` is redacted above: this
repository is public and the identifier serves no documentation purpose.
