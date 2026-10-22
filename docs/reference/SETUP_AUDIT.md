# Track 2 — System Setup Audit

Audit of the development machine against what Track 2 (Robotic Grounding) needs.
All figures below are command output captured on this host, not estimates.

- **Host:** `ibrahim5i`
- **Audit date:** 2026-10-15
- **Last updated:** 2026-10-15 (pyarrow fixed; G1 ReconBody smoke test **PASSED** — see §20.11)
- **Repository:** `ibrahim-1811/v2d-track2-robotic-grounding`
- **Scope:** the original audit was read-only. A second pass on the same day
  **installed Docker Engine** and nothing else — see §8. A third pass
  **validated GPU passthrough** by pulling `nvidia/cuda:13.0.0-base-ubuntu24.04`
  and running `nvidia-smi` in a container — see §10. A fourth pass **tested NGC
  registry access** without authenticating or downloading any image — see §19.
  A fifth pass ran the **first G1 ReconBody simulation smoke test**, which
  **failed on an image dependency defect** — see §20. GPU drivers, CUDA and
  Podman were left untouched, and `/etc/docker/daemon.json` was preserved.

---

## 1. Summary

| Area | Result |
|---|---|
| OS / kernel | Ubuntu 24.04.5 LTS, kernel 7.0.0-38-generic — OK |
| CPU | i7-13650HX, 14 cores / 20 threads — OK |
| RAM | 31 GiB total, ~20 GiB available — OK |
| GPU | RTX 5070 Laptop, **8151 MiB VRAM** — under-sized for FlashCHORD training |
| Driver | 580.178.04, CUDA 13.0 — OK, exceeds the recommended 580.126.09 |
| NVIDIA Container Toolkit | 1.20.1 installed, 3 CDI devices enumerate, now registered with Docker — OK |
| Docker | **29.9.1 installed**, active + enabled, non-root access works — OK |
| NGC registry login | not configured — **and not needed**: `nvcr.io/nvidia/isaac-lab:2.3.2` is anonymously pullable (§19) |
| Git LFS | 3.4.1 — OK |
| Docker Compose | v5.6.0 (plugin) — OK, though the official workflow does not use it |
| GPU passthrough | **PASS** — `--gpus all` container sees the GPU, driver and 8151 MiB |
| `robotic-grounding` image | built, 30.4 GB (`fe0ca1346c8b`) |
| G1 ReconBody smoke test | **PASS** — 60/60 steps, peak **3949 MiB** VRAM, after the pyarrow fix (§20.11) |
| Image `pyarrow` conflict | **RESOLVED** — pinned to 25.0.1 in a derived layer (§20.11) |
| Disk | **121 GB free** on `/` (122 GB before the 562 MB image) — adequate but not generous |

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
  see [REQUIREMENTS.md](../reference/REQUIREMENTS.md) §FlashCHORD. Desktop VRAM use drifts
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

**Docker Engine was installed on 2026-10-15** using Docker's official
apt-repository procedure for Ubuntu. This section records the resulting state;
the pre-install state is kept at the end for the record.

### 8.1 Installed versions

```
$ docker --version
Docker version 29.9.1, build dfc520d

$ docker compose version
Docker Compose version v5.6.0

$ docker buildx version
github.com/docker/buildx v0.38.0 0dbe87f36ed472fe36b6796c3f70b4210a731fc5

$ containerd --version
containerd containerd v2.4.2 b7a25261e029d68978bbf290df0a62fc819b3c51

$ runc --version
runc version 1.5.2
commit: v1.5.2-0-g29dd3dc2
spec: 1.3.0
```

| Package | Version |
|---|---|
| `docker-ce` | `5:29.9.1-1~ubuntu.24.04~noble` |
| `docker-ce-cli` | `5:29.9.1-1~ubuntu.24.04~noble` |
| `docker-ce-rootless-extras` | `5:29.9.1-1~ubuntu.24.04~noble` |
| `containerd.io` | `2.4.2-2~ubuntu.24.04~noble` |
| `docker-buildx-plugin` | `0.38.0-1~ubuntu.24.04~noble` |
| `docker-compose-plugin` | `5.6.0-1~ubuntu.24.04~noble` |
| `pigz` | `2.8-1` (Ubuntu archive; a `docker-ce` Recommends) |

### 8.2 Install method and APT impact

Official procedure: key to `/etc/apt/keyrings/docker.asc`, repo
`deb [arch=amd64 signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu noble stable`
in `/etc/apt/sources.list.d/docker.list`, then
`apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin`.

The key fingerprint was checked against Docker's published key before use:

```
pub   rsa4096 2017-02-22 [SCEA]
      9DC8 5822 9FC7 DD38 854A  E2D8 8D81 803C 0EBF CD88
uid                      Docker Release (CE deb) <docker@docker.com>
```

A dry run was inspected before committing, and the install was gated on it
proposing no removals:

```
0 upgraded, 7 newly installed, 0 to remove and 13 not upgraded.
Need to get 102 MB of archives.
After this operation, 388 MB of additional disk space will be used.
```

**Nothing was removed.** This was verified in advance rather than assumed:
`containerd.io` declares `Conflicts: containerd, runc` and `docker-ce` declares
`Conflicts: docker.io`, but none of `docker.io`, `docker-cli`, `containerd` or
`runc` were installed on this host, so no conflict fired. Podman was never at
risk — it depends on `crun | runc` and `crun` is installed, satisfying the
alternative either way.

One unrelated change rode along in the prerequisite step: `curl`,
`libcurl4t64`, `libcurl3t64-gnutls` and `libcurl4-openssl-dev` were upgraded
`8.5.0-2ubuntu10.15` → `8.5.0-2ubuntu10.16`. That is a routine Ubuntu security
update from `noble-security`, not a Docker dependency.

### 8.3 Service state

```
$ systemctl status docker
● docker.service - Docker Application Container Engine
     Loaded: loaded (/usr/lib/systemd/system/docker.service; enabled; preset: enabled)
     Active: active (running) since Thu 2026-10-15 22:15:43 CEST
TriggeredBy: ● docker.socket
   Main PID: 31870 (dockerd)
     CGroup: /system.slice/docker.service
             └─31870 /usr/bin/dockerd -H fd:// --containerd=/run/containerd/containerd.sock

$ systemctl is-active docker ; systemctl is-enabled docker
active
enabled

$ systemctl is-active containerd ; systemctl is-enabled containerd
active
enabled
```

Two benign messages appear in the first-start log, where dockerd tries to delete
nftables tables that do not exist yet:

```
level=info msg="Deleting nftables IPv4 rules" error="... delete table ip docker-bridges ... exit status 1"
level=info msg="Deleting nftables IPv6 rules" error="... delete table ip6 docker-bridges ... exit status 1"
```

These are `level=info`, occur on a clean install, and the daemon reaches
`"Daemon has completed initialization"` immediately after. `docker info` reports
no warnings.

### 8.4 Non-root access

Already satisfied — the `docker` group existed (GID 984) with `ibrahim` in it
*before* Docker was installed, and that membership was already active in the
shell, so **no re-login was needed**:

```
$ ls -la /var/run/docker.sock
srw-rw---- 1 root docker 0 Oct 15 22:15 /var/run/docker.sock

$ whoami
ibrahim

$ docker info        # no sudo
Server Version: 29.9.1
```

Every verification command in this section was run as `ibrahim` without `sudo`.

### 8.5 `docker info` — key fields

```
Server Version: 29.9.1
Storage Driver: overlayfs
  driver-type: io.containerd.snapshotter.v1
Logging Driver: json-file
Cgroup Driver: systemd
Cgroup Version: 2
Runtimes: io.containerd.runc.v2 nvidia runc
Default Runtime: runc
CDI spec directories:
  /etc/cdi
  /run/cdi
Discovered Devices:
  cdi: nvidia.com/gpu=0
  cdi: nvidia.com/gpu=GPU-<redacted>
  cdi: nvidia.com/gpu=all
Kernel Version: 7.0.0-38-generic
Operating System: Ubuntu 24.04.5 LTS
CPUs: 20
Total Memory: 31.07GiB
Docker Root Dir: /var/lib/docker
Firewall Backend: iptables
Security Options: apparmor (default), seccomp (builtin), cgroupns
```

Three things worth carrying forward to Track 2:

1. **`Runtimes: … nvidia …`** — the pre-existing `/etc/docker/daemon.json` was
   picked up exactly as hoped. See §9.
2. **`Discovered Devices: cdi: nvidia.com/gpu=…`** — Docker 29 auto-discovers
   the CDI specs the NVIDIA Container Toolkit wrote, so GPUs are addressable as
   `--device nvidia.com/gpu=all` in addition to `--gpus all`.
3. **`Storage Driver: overlayfs` / `io.containerd.snapshotter.v1`** — Docker 29
   defaults to the containerd image store, not the older `overlay2` graph
   driver. Worth knowing before building the large Isaac Lab images: layer
   handling and `docker system df` accounting differ from pre-28 Docker.

### 8.6 Functional test

```
$ docker run --rm hello-world        # as ibrahim, no sudo
Unable to find image 'hello-world:latest' locally
latest: Pulling from library/hello-world
Digest: sha256:11a3eb1597e864c51b3f56733cb2e8a90bf0c284bfc05f9444d81c558bc848aa
Status: Downloaded newer image for hello-world:latest

Hello from Docker!
This message shows that your installation appears to be working correctly.
...
$ echo $?
0
```

Pull, create, run, output and `--rm` cleanup all work, and Docker Hub is
reachable from this host.

### 8.7 Disk usage

```
$ docker system df
TYPE            TOTAL     ACTIVE    SIZE      RECLAIMABLE
Images          1         0         36.94kB   25.87kB (70%)
Containers      0         0         0B        0B
Local Volumes   0         0         0B        0B
Build Cache     0         0         0B        0B

$ docker system df -v
REPOSITORY    TAG      IMAGE ID       CREATED        SIZE     SHARED SIZE   UNIQUE SIZE   CONTAINERS
hello-world   latest   11a3eb1597e8   6 months ago   25.9kB   0B            25.87kB       0
```

`Docker Root Dir` is `/var/lib/docker`, on the same single ext4 root filesystem
as everything else. Free space went from 122 GB to 122 GB (`df` rounding; the
packages cost ~388 MB and the image 26 kB), so the disk picture in §6 is
unchanged — and still the constraint to watch once the Isaac Lab images land.

These figures are from immediately after the install. The GPU validation pass
(§10) later added a 562 MB `nvidia/cuda` image; see §10.5 for current usage.

### 8.8 Pre-install state (for the record)

Before this pass, Docker was absent but partially pre-staged:

```
$ docker --version
bash: line 1: docker: command not found

$ systemctl status docker
Unit docker.service could not be found.

$ ls -la /var/run/docker.sock
ls: cannot access '/var/run/docker.sock': No such file or directory

$ ls -d /var/lib/docker
ls: cannot access '/var/lib/docker': No such file or directory

$ dpkg -l | grep -Ei "docker|containerd|podman"
ii  podman   4.9.3+ds1-1ubuntu0.2   amd64   tool to manage containers and pods
```

Three leftovers from an earlier attempt shaped how this install went:

1. `/etc/docker/daemon.json` already existed and pre-registered the NVIDIA
   runtime (§9). It was backed up and **not** rewritten — `docker-ce` does not
   ship that file, so the registration carried over intact.
2. `/etc/apt/keyrings/docker.gpg` already existed, holding the correct Docker
   Release (CE deb) key — but **no APT source referenced it**, so the repo had
   never actually been added. The official procedure writes `docker.asc`
   instead; `docker.gpg` is now an unreferenced orphan and can be deleted.
3. The user was already in the `docker` group, which is why §8.4 needed no work.

`podman` 4.9.3 (crun runtime, zero images, zero containers) is still installed
and untouched. The official tooling calls the `docker` CLI directly, so podman
was never a drop-in substitute; now that Docker is present the question is moot.

Docker Compose is **not required** by the official workflow — it uses plain
`docker run` through `workflow/run.sh` — but `docker-compose-plugin` v5.6.0
comes with Docker's standard install command and is harmless.

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

Installed and complete, and **unchanged by the Docker install** — all four
packages are still at 1.20.1-1 and `nvidia-ctk --version` still reports 1.20.1.

The pre-existing `/etc/docker/daemon.json` survived the install byte-for-byte
(same size, same mtime as the backup taken beforehand):

```
$ cat /etc/docker/daemon.json
{
    "runtimes": {
        "nvidia": {
            "args": [],
            "path": "nvidia-container-runtime"
        }
    }
}

$ ls -la /etc/docker/
-rw-r--r-- 1 root root 128 Mar 23  2026 daemon.json
-rw-r--r-- 1 root root 128 Mar 23  2026 daemon.json.bak.20261015-221526
```

And the daemon now actually honours it — `docker info` lists `nvidia` among the
runtimes and separately auto-discovers the toolkit's CDI specs:

```
Runtimes: io.containerd.runc.v2 nvidia runc
Default Runtime: runc
Discovered Devices:
  cdi: nvidia.com/gpu=0
  cdi: nvidia.com/gpu=GPU-<redacted>
  cdi: nvidia.com/gpu=all
```

That closes the loop on what was previously only "pre-staged": the NVIDIA
runtime is registered and visible to Docker. Note the consequence flagged in the
original audit still holds — because `daemon.json` names
`nvidia-container-runtime` by path, `docker.service` will fail to start if the
NVIDIA Container Toolkit is ever uninstalled.

The toolkit was verified read-only against the GPU before Docker existed:

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

**PASS — validated 2026-10-15.** A GPU-enabled container sees the GPU, the
driver and the full 8151 MiB.

### 10.1 The test and its exit code

```
$ docker run --rm --gpus all nvidia/cuda:13.0.0-base-ubuntu24.04 nvidia-smi
Thu Oct 15 20:25:03 2026
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.178.04             Driver Version: 580.178.04     CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 5070 ...    Off |   00000000:01:00.0  On |                  N/A |
| N/A   47C    P8              8W /  100W |     631MiB /   8151MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+

$ echo $?
0
```

The container ran as `ibrahim` with no `sudo`. The timestamp reads UTC inside
the container versus CEST on the host, which is expected — the image carries no
timezone data.

### 10.2 Required detections — all four confirmed

`nvidia-smi` truncates the GPU name in its table view, so the identity was
re-queried in a second container:

```
$ docker run --rm --gpus all nvidia/cuda:13.0.0-base-ubuntu24.04 \
    nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version,compute_cap,uuid --format=csv
name, memory.total [MiB], memory.used [MiB], driver_version, compute_cap, uuid
NVIDIA GeForce RTX 5070 Laptop GPU, 8151 MiB, 669 MiB, 580.178.04, 12.0, GPU-<redacted>
```

| # | Required | Observed in container | Result |
|---|---|---|---|
| 1 | RTX 5070 Laptop GPU | `NVIDIA GeForce RTX 5070 Laptop GPU` | **PASS** |
| 2 | 8151 MiB total VRAM | `8151 MiB` | **PASS** |
| 3 | Driver 580.178.04 | `580.178.04` | **PASS** |
| 4 | CUDA driver capability | `CUDA Version: 13.0`, `compute_cap 12.0` | **PASS** |

The reported total VRAM is the full 8151 MiB — passthrough does not reserve or
hide any of it. Compute capability `12.0` (`sm_120`, Blackwell) is reported
correctly inside the container, which matters for anything built against this
GPU later.

### 10.3 What the toolkit injects

```
$ docker run --rm --gpus all nvidia/cuda:13.0.0-base-ubuntu24.04 bash -lc '...'
CUDA_VERSION env   : 13.0.0
driver libs        :
/usr/lib/x86_64-linux-gnu/libcuda.so
/usr/lib/x86_64-linux-gnu/libcuda.so.1
/usr/lib/x86_64-linux-gnu/libcuda.so.580.178.04
nvidia devices     :
/dev/nvidia-modeset
/dev/nvidia-uvm
/dev/nvidia-uvm-tools
/dev/nvidia0
/dev/nvidiactl
nvidia-smi path    : /usr/bin/nvidia-smi
```

`libcuda.so.580.178.04` is the host driver's user-mode library, bind-mounted in
by the toolkit — the version matches the host exactly, which is the thing that
has to be true for CUDA to work. `/dev/nvidia-uvm` is present, which Isaac Sim
and PyTorch both need (unified memory); a container with only `/dev/nvidia0`
and `/dev/nvidiactl` would pass `nvidia-smi` and then fail under CUDA.

### 10.4 Three invocation styles, all working

```
$ docker run --rm --gpus all ... nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
NVIDIA GeForce RTX 5070 Laptop GPU, 8151 MiB, 580.178.04          # exit 0

$ docker run --rm --device nvidia.com/gpu=all ... (same query)
NVIDIA GeForce RTX 5070 Laptop GPU, 8151 MiB, 580.178.04          # exit 0

$ docker run --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all ... (same query)
NVIDIA GeForce RTX 5070 Laptop GPU, 580.178.04                    # exit 0
```

All three work, so Track 2 tooling will function whichever convention it uses:
`--gpus all` (what `workflow/run.sh` and most NVIDIA docs use), CDI
(`--device nvidia.com/gpu=all`, the modern path Docker 29 auto-discovers), or an
explicit `--runtime=nvidia` with `NVIDIA_VISIBLE_DEVICES`.

A negative control confirms the GPU plumbing is really what is doing the work:

```
$ docker run --rm nvidia/cuda:13.0.0-base-ubuntu24.04 nvidia-smi    # no GPU flags
... error during container init: exec: "nvidia-smi": executable file not found in $PATH
$ echo $?
127
```

Without a GPU flag `nvidia-smi` is not even present — the `base` image does not
ship it, the toolkit injects it. So the successful runs above are genuine
passthrough, not a binary that happened to be baked into the image.

### 10.5 Image size and disk

```
$ docker images
REPOSITORY    TAG                       IMAGE ID       SIZE
hello-world   latest                    11a3eb1597e8   25.9kB
nvidia/cuda   13.0.0-base-ubuntu24.04   6e43a6b02e5f   562MB

$ docker system df
TYPE            TOTAL     ACTIVE    SIZE      RECLAIMABLE
Images          2         0         561.9MB   561.9MB (99%)
Containers      0         0         0B        0B
Local Volumes   0         0         0B        0B
Build Cache     0         0         0B        0B

$ df -h /
/dev/nvme0n1p6  270G  135G  121G  53% /
```

The `nvidia/cuda` base image is **562 MB** — well short of the "multi-GB" the
previous revision of this section guessed, because `-base` carries only the CUDA
runtime stub, not the toolkit. Free space went 122 GB → **121 GB**.

Both images are 100% reclaimable (`docker image rm` or `docker system prune`)
since no container references them. They were deliberately left in place.

The pulled digest, for reproducibility:
`sha256:6e43a6b02e5f16e4a715953be8b29c40acbe84aed11a61357ef0fc899967fbd9`.

### 10.6 No fixes were needed

Passthrough worked on the first attempt. Nothing was diagnosed, changed or
worked around: no daemon restart, no `nvidia-ctk runtime configure`, no
`daemon.json` edit, no driver or toolkit change. `/etc/docker/daemon.json` is
still the file described in §9, and the host GPU state is unchanged.

One observation recorded because it could matter later, not because it caused a
problem: this image's `NVIDIA_REQUIRE_CUDA` constraint string enumerates driver
bands only up to `driver>=575,driver<576` and has no 580 band, yet the container
started cleanly under driver 580.178.04 with exit 0. If a future image or a
stricter toolkit configuration ever does enforce that constraint, the escape
hatch is `-e NVIDIA_DISABLE_REQUIRE=1`. Not needed today.

### 10.7 Observed VRAM drift

Across this pass the desktop's own VRAM use read 631 MiB, 669 MiB and 1071 MiB
at different moments. That widens the "budget 0.7–1.0 GiB lost to the desktop"
note in §5 to roughly **0.6–1.1 GiB**. Usable VRAM for Track 2 work is therefore
about **7.1–7.5 GiB**, and it moves while the desktop session runs.

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
| 1 | ~~Docker not installed~~ | **RESOLVED** 2026-10-15 | Docker Engine 29.9.1 installed, active, enabled, non-root verified — see §8 |
| 2 | 8151 MiB VRAM vs FlashCHORD's 48 GiB requirement | **Blocker** for FlashCHORD training | ~6× short; see REQUIREMENTS.md |
| 3 | ~~No NGC / `docker login nvcr.io` credentials~~ | **RESOLVED** 2026-10-15 — was never a blocker | Measured: `nvcr.io` grants anonymous `pull` on `nvidia/isaac-lab`; a layer blob returns HTTP 200 without credentials. See §19 |
| 4 | MANO hand models not present (license-gated) | High | Needed by the loader stage; register at mano.is.tue.mpg.de |
| 5 | `pipx` and `pre-commit` missing | Low | `workflow/setup_deps.sh` installs them |
| 6 | No host orchestrator venv yet | Low | One `python3 -m venv` away |
| 7 | GPU also drives the desktop (661 MiB used) | Medium | Costs usable VRAM on an already-tight GPU |
| 8 | Host `nvcc` 12.0 cannot target `sm_120` | Low | Only matters if we build CUDA on the host |
| 9 | `~/launch_isaac.sh` references a deleted venv | Informational | Local Isaac Sim not launchable via that script |
| 10 | Broken `conda` shim in `~/.pixi/bin` | Informational | Unused by Track 2 |
| 11 | Only 122 GB free on a shared root filesystem | Medium | Images + datasets will press on this; `/var/lib/docker` shares it |
| 12 | ~~GPU-in-container unproven~~ | **RESOLVED** 2026-10-15 | `docker run --gpus all nvidia/cuda:13.0.0-base-ubuntu24.04 nvidia-smi` → exit 0, GPU/driver/8151 MiB all correct. Also verified via CDI and `--runtime=nvidia` — see §10 |
| 13 | `daemon.json` hard-codes `nvidia-container-runtime` | Low | `docker.service` will fail to start if the toolkit is ever removed (§9) |
| 14 | 6 packages pending `apt autoremove` | Informational | **Pre-existing, not caused by Docker** — `libfwupd2`, `python3-pygraphviz` and 4 stale `nvidia-firmware-*` point releases. `nvidia-firmware-580-580.178.04`, which matches the running driver, is *not* a candidate, so autoremove would not break the GPU. Still, nothing requires running it |
| 15 | Orphaned `/etc/apt/keyrings/docker.gpg` | Informational | Pre-staged key no longer referenced by any APT source; the official install uses `docker.asc` (§8.8) |

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

Nothing was installed, upgraded, removed, downloaded or launched during **that
first audit pass**, and no global system setting was changed.

### Second pass — Docker Engine install (2026-10-15)

This pass did change the system. What it ran, with root:

`apt-get update`, `apt-get install -y ca-certificates curl`,
`install -m 0755 -d /etc/apt/keyrings`,
`curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc`,
`chmod a+r /etc/apt/keyrings/docker.asc`,
`gpg --show-keys --with-fingerprint`, writing
`/etc/apt/sources.list.d/docker.list`, `apt-get update`,
`apt-get install -y --dry-run docker-ce docker-ce-cli containerd.io
docker-buildx-plugin docker-compose-plugin` (inspected, gated on zero removals),
the same `apt-get install` for real, `cp -a` of `/etc/docker/daemon.json` to a
timestamped backup, and `systemctl enable --now docker`.

Verification, all as `ibrahim` without `sudo`: `docker --version`,
`docker compose version`, `docker buildx version`, `containerd --version`,
`runc --version`, `dpkg -l`, `systemctl status docker`,
`systemctl is-active/is-enabled docker containerd`, `ls -la /var/run/docker.sock`,
`docker info`, `docker run --rm hello-world`, `docker system df`,
`docker system df -v`, `df -h`, `podman --version`, `nvidia-ctk --version`,
`nvidia-smi`, `apt-get autoremove -s` (simulate only).

Scope held: no GPU driver or CUDA change, Podman untouched,
`/etc/docker/daemon.json` preserved, and no Isaac Lab install, NVIDIA repo clone
or dataset download. Nothing was committed or pushed.

### Third pass — GPU passthrough validation (2026-10-15)

Read-only with respect to the system; it pulled one image and ran five
throwaway `--rm` containers. Nothing was installed, configured or removed.

Pre-checks: `systemctl is-active/is-enabled docker`, `docker --version`,
`docker info --format '{{.Runtimes}}'`, `docker info --format '{{.DefaultRuntime}}'`,
`which nvidia-container-runtime nvidia-container-cli nvidia-ctk`,
`nvidia-ctk --version`, `nvidia-smi --query-gpu=...`, `df -h /`.

The test itself: `docker pull nvidia/cuda:13.0.0-base-ubuntu24.04`, then
`docker run --rm --gpus all nvidia/cuda:13.0.0-base-ubuntu24.04 nvidia-smi`
(the specified check), plus `nvidia-smi --query-gpu=...` for the untruncated GPU
name, a `bash -lc` probe listing injected driver libraries and `/dev/nvidia*`
nodes, the same query under `--device nvidia.com/gpu=all` and under
`--runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all`, and a no-GPU-flag negative
control.

After: `docker images`, `docker system df`, `docker system df -v`, `df -h /`,
`nvidia-smi --query-gpu=...`.

Scope held: no driver, CUDA, toolkit or `daemon.json` change; no image removed;
no Isaac Lab install, NVIDIA repo clone, dataset download or training. Both
images (`hello-world`, `nvidia/cuda`) were left in place. Nothing was committed
or pushed.

### Fourth pass — NGC registry access (2026-10-15)

Read-only. One container run (the already-local `nvidia/cuda` image, to re-confirm
GPU passthrough) plus registry metadata queries. No image was downloaded, no
credential was created, entered or stored.

Pre-checks: `systemctl is-active/is-enabled docker`, `docker --version`,
`docker info --format`, `docker run --rm --gpus all … nvidia-smi --query-gpu=…`.

Credential state: `ls -la ~/.docker ~/.ngc`, `which ngc`, `env | grep -oiE` for
NGC-ish **variable names only**, and a Python inspection of
`~/.docker/config.json` written to print registry names and field names but never
values (the file does not exist, so it printed nothing).

Access test: `docker manifest inspect nvcr.io/nvidia/isaac-lab:2.3.2`,
`docker manifest inspect nvidia/cuda:13.0.0-base-ubuntu24.04` (method control),
`curl` to `nvcr.io/proxy_auth` for an **anonymous** token per repository with the
granted actions decoded from the token payload, a `curl` GET of the amd64 manifest
to count layers and sum compressed sizes, and `curl -I` (HEAD) on a layer blob
following redirects to confirm HTTP 200 with zero bytes transferred.

Also: `grep` of the upstream checkout for `nvcr.io/…`, `ghcr.io/…` and `quay.io/…`
references, and a probe for installed Docker credential helpers plus
`busctl --user list` for a running secret service.

`docker login` was **not** run. Scope held: no repository cloned, no image
downloaded, no LFS asset fetched, no package installed, no training. Nothing was
committed or pushed.

The GPU UUID reported by `nvidia-container-cli info`, `nvidia-ctk cdi list`,
`docker info` and in-container `nvidia-smi --query-gpu=uuid` is redacted
throughout: this repository is public and the identifier serves no documentation
purpose. No NGC API key, token or credential value appears anywhere in this
repository.

---

## 19. NGC registry access (`nvcr.io`)

**PASS — but not by authenticating.** The requirement turned out not to apply:
every image Track 2 needs is publicly pullable, so no NGC API key is involved.

### 19.1 Credential state — none present

Checked without printing or capturing any secret value:

```
$ ls -la ~/.docker
ls: cannot access '/home/ibrahim/.docker': No such file or directory

$ ls -la ~/.ngc
ls: cannot access '/home/ibrahim/.ngc': No such file or directory

$ which ngc
(not found)

$ env | grep -oiE '^(NGC[A-Z_]*|NVIDIA[A-Z_]*KEY|DOCKER[A-Z_]*(PASS|TOKEN)[A-Z_]*)='
(no matches)
```

No `~/.docker/config.json` exists, so there is no `auths` map, no `credsStore`
and no `credHelpers` — nothing is authenticated to any registry, `nvcr.io`
included. Had a config existed, the inspection was written to print only
**registry names and field names**, never credential values.

### 19.2 Access test — anonymous `pull` is granted

The registry was asked directly what an unauthenticated client may do. No
credentials were sent.

```
$ curl -fsS "https://nvcr.io/proxy_auth?scope=repository:nvidia/isaac-lab:pull"
   → token issued

   issuer : authn.nvidia.com
   subject: sf:guestclient            # i.e. anonymous
   access : [ { "type": "repository",
                "name": "nvidia/isaac-lab",
                "actions": [ "pull" ] } ]
```

`docker manifest inspect` confirms the same at the CLI level:

```
$ docker manifest inspect nvcr.io/nvidia/isaac-lab:2.3.2
{ "schemaVersion": 2,
  "mediaType": "application/vnd.docker.distribution.manifest.list.v2+json",
  "manifests": [ { ... "platform": { "architecture": "amd64", "os": "linux" } },
                 { ... "platform": { "architecture": "arm64", "os": "linux" } } ] }
$ echo $?
0
```

Manifest read alone only proves *metadata* access, so the check was pushed one
layer deeper — a `HEAD` on a real layer blob, which transfers no content:

```
$ curl -sS -L -I -H "Authorization: Bearer $ANON_TOKEN" \
    https://nvcr.io/v2/nvidia/isaac-lab/blobs/sha256:b08e2ff4...
HTTP/1.1 307 Temporary Redirect
HTTP/2 200
content-type: application/octet-stream
content-length: 29718366

bytes downloaded: 0        # HEAD, no body
```

HTTP 200 on a layer blob is the real proof: anonymous clients can pull, not just
look. **0 bytes of image data were transferred.**

### 19.3 Every repository the workflow references

| Repository | Anonymous actions | Used by |
|---|---|---|
| `nvidia/isaac-lab` | `['pull']` | **Track 2 base image** (`workflow/Dockerfile`) |
| `nvidia/isaac-sim` | `['pull']` | referenced upstream |
| `nvidia/cuda` | `['pull']` | FlashCHORD container |
| `nvidia/tensorrt` | `['pull']` | referenced upstream |
| `nvidia/base/ubuntu` | `['pull']` | referenced upstream |
| `nvidian/v2p_video_agent` | `[]` — **no access** | `video_ingestion_agent/osmo_workflows/*.yaml` only |

The single inaccessible image sits in NVIDIA's internal `nvidian` org and is
referenced only by the video-ingestion agent, which Track 2 does not use. Its
empty action list is also the useful control: it shows the probe can distinguish
"public" from "private", so the `['pull']` results above are meaningful rather
than an artefact of the registry handing out permissive tokens.

### 19.4 Base image size, for disk planning

From the inspected `linux/amd64` manifest — metadata only, nothing downloaded:

```
digest            sha256:f07c37e3f0c9f58f7febd0aa9a425523d282be623c0db81ac61006d0e24be07f
layers            29
total compressed  8.41 GB
```

Uncompressed on disk will be substantially larger. Against the **121 GB free**
recorded in §8.7, one base image is comfortable; base + the locally built
`robotic-grounding` layer + `v2d_task_library_loader` + the FlashCHORD
Newton/CUDA image + a dataset is the combination to watch.

### 19.5 Authentication was not performed

`docker login nvcr.io` was **not** run. Two reasons:

1. It is not required for anything Track 2 builds or runs (§19.3).
2. It cannot be done from this session — the API key must be typed interactively
   by the operator, and no credential was ever to be entered, printed, captured
   or stored by the agent.

If a login is wanted later (the one scenario being requirement 3.3, a private
organiser registry), the official procedure from `robotic_grounding/docs/SETUP.md`
§2 is `docker login nvcr.io` with user `$oauthtoken` and the NGC API key as the
password, entered at the interactive prompt.

One security note for whenever that happens: **no Docker credential helper is
installed** on this host (`docker-credential-{pass,secretservice,gnome-keyring}`
are all absent), so `docker login` would write the key **base64-encoded — not
encrypted — into `~/.docker/config.json`**. A GNOME keyring *is* running
(`org.freedesktop.secrets`, PID 2903), so installing
`docker-credential-secretservice` would let Docker store it encrypted instead.
Installing packages was out of scope for this pass, so it was not done.

No API key was created, requested, entered, printed or stored during this pass,
and nothing in this repository contains one.

---

## 20. First G1 ReconBody simulation smoke test

> **Superseded by §20.11.** The failure described in §20.1–§20.10 was diagnosed
> and fixed on the same day; the smoke test now **passes**. This section is kept
> because the diagnosis and the exit-code trap remain worth knowing.

**FAIL (at the time of this run).** 0 of 60 simulation steps executed. The cause
is a dependency conflict inside the `robotic-grounding` image — **not** VRAM,
**not** missing assets, and **not** the Track 2 code.

Full artefacts: `outputs/step7_g1_reconbody_smoke/` (gitignored) — `SUMMARY.md`,
`smoke_run.log`, `vram_before.txt`, `vram_during.csv`, `vram_after.txt`.

### 20.1 Preconditions — all verified

```
$ docker images
robotic-grounding   latest   fe0ca1346c8b   30.4GB

$ nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
NVIDIA GeForce RTX 5070 Laptop GPU, 8151 MiB, 6683 MiB

$ df -h /
/dev/nvme0n1p6  270G  185G   72G  73% /
```

Assets re-verified real (not LFS pointers) on the host before the run:
`data.parquet` 3,688,593 B · `textured_mesh.obj` 22,108,986 B ·
`textured_mesh.urdf` 572 B · `ground_plane.json` 431 B · `…_support.usda` 569 B ·
`encoder_batched.onnx` 50,112,213 B · `decoder_batched.onnx` 40,903,373 B ·
`urdfs/g1/main.urdf` 57,957 B. All **67** mesh references in `main.urdf`
resolved: **0 missing, 0 still pointers**.

### 20.2 How it was run

NVIDIA's supported non-interactive path. `run.sh start` ends in
`docker exec -it`, which needs a TTY; `run.sh e2e-run` is the mode built for
automation (`DOCKER_TTY_ARGS=()`, no `-it`). The image was **not** rebuilt.

```
./workflow/run.sh e2e-run latest 0 --run-data <outputs> -- \
  bash -lc 'cd /workspace/video_to_data/robotic_grounding && \
    python scripts/rsl_rl/dummy_agent.py \
      --headless --task SonicG1-ReconBody-v0 \
      --motion_file whole_body/soma/2026-03-06_10-24-18_snack_box_pick_and_place_01/g1 \
      --num_envs 1 --max_steps 60'
```

`--max_steps` is supported by the pinned script (`dummy_agent.py:107`).
`--action_mode` defaults to `zero` and is its only choice, so actions are zero by
construction. No `--record_video` and no `--output_dir` were passed, so cameras
stayed off.

### 20.3 The failure

```
ImportError: pyarrow requires NumPy 2.0 or newer, found 1.26.4
```

Installed in the image: `numpy 1.26.4`, `pyarrow 26.0.0`, `pandas 3.0.0`,
`torch 2.7.0+cu128`, `onnxruntime-gpu 1.24.4`, `gymnasium 1.2.1`,
`rsl-rl-lib 3.1.2`. Checked directly: `import pyarrow` fails, `import pandas`
succeeds.

Import chain, from the log:

```
dummy_agent.py:160            from robotic_grounding.tasks import *
tasks/__init__.py:10          import_packages(...)
tasks/v2d/config/sharpa_wave/__init__.py:5
sharpa_v2d_dr_env_cfg.py:28   from ...tasks.scene_utils.visual_dr import ...
tasks/scene_utils/__init__.py:7
apply_scene_config.py:29      from robotic_grounding.motion_schema import ...
motion_schema/__init__.py:9   from .reader import load_motion_data_parquet
motion_schema/reader.py:24    import pyarrow as pa      <-- ImportError
```

The trigger is the **Sharpa** DR config, reached by the blanket `import_packages`
walk in `tasks/__init__.py` — not the G1 path. Every task import therefore fails,
so this cannot be dodged by selecting a different task.

### 20.4 Stage-by-stage

| Stage | Result |
|---|---|
| Container start (`run.sh e2e-run`) | **PASS** |
| Isaac Sim 5.1 / Kit headless launch | **PASS** — `Using device: cuda:0`, `isaaclab.python.headless.kit` loaded, PhysX plugin loaded |
| Task package import | **FAIL** — pyarrow / numpy |
| `SonicG1-ReconBody-v0` gym registration | not reached |
| Asset + motion Parquet loading | not reached |
| SONIC ONNX initialisation | not reached |
| Environment stepping (60 steps) | **0 executed** |
| Shutdown | clean — no crash, no CUDA error |

Note what *did* work: the simulator itself initialises correctly in this image,
headless, on this GPU. That is a real (if partial) result — Isaac Sim 5.1 + Kit +
PhysX start and bind to `cuda:0`.

Registration could therefore only be confirmed **statically** (the `gym.register`
calls in
`tasks/v2d_whole_body/config/sonic/g1/__init__.py`, 7 `SonicG1-*` IDs), not at
runtime. A dedicated runtime check via `AppLauncher` hit the same ImportError.

### 20.5 VRAM

| Point | Used | Free |
|---|---|---|
| Before | 1059 MiB | 6622 MiB |
| **Peak during** | **1195 MiB** | 6456 MiB |
| After | 1027 MiB | 6655 MiB |

GPU total 8151 MiB. Delta attributable to the test: **136 MiB** — Kit startup
only; no scene was ever built. Zero matches for OOM patterns in the log, so
**there was no CUDA out-of-memory**.

This figure must not be read as "the env fits comfortably". The scene never
loaded. **Actual G1 ReconBody VRAM remains unmeasured** and is still the open
question for this machine ([REQUIREMENTS.md](../reference/REQUIREMENTS.md) 8.2).

### 20.6 Root cause — upstream version drift

`robotic_grounding/workflow/Dockerfile`:

- **lines 102-104:** `pip install "numpy<2.0.0"`, commented *"to avoid
  compatibility issues in Isaac Lab"* — deliberate.
- **lines 122-123:** `pip install -e ./source/robotic_grounding`, which pulls
  `pyarrow` **unpinned**. Today that resolves to 26.0.0, whose compiled
  `lib.pyx` hard-requires numpy ≥ 2.0.

`pyproject.toml` has no `[project].dependencies` pins for these. So the image is
internally inconsistent by construction, and anyone building it today reproduces
the same break. This is an upstream packaging bug, not a local misconfiguration.

### 20.7 Exit-code trap worth remembering

**The failing run exits 0.** Verified twice — `run.sh` returned 0, and the inner
Isaac Sim `python.sh` wrapper returned 0, despite printing the traceback. A
control (`python -c "import sys; sys.exit(42)"`) returned 1 through the same
wrapper, so the masking is specific to an uncaught exception inside a launched
Kit app.

Any automation around these scripts must grep logs for `Traceback`/`Error`
rather than trust exit status. This would otherwise read as a silent pass.

### 20.8 What was deliberately not done

No fix was applied. Both candidate fixes change the image, which is Step 6
territory:

1. **Correct fix:** pin a numpy-1.x-compatible `pyarrow` in `workflow/Dockerfile`
   and rebuild. PyPI metadata declares no numpy requirement for any recent
   pyarrow release — the guard lives in the compiled extension — so the exact
   last compatible version needs an empirical check, not a lookup.
2. **Temporary unblock:** install a compatible `pyarrow` into a running
   container; not reproducible, lost on container recreation.

No upstream file was modified and no model configuration was changed. After the
run, the upstream checkout is still at
`193382a184308f6800331f3db829f76dc4294cfe` with `git status --porcelain`
returning 0 lines, despite being bind-mounted read-write. The container was
stopped with `run.sh stop`.

### 20.9 Smoke test vs. successful manipulation

Worth stating plainly, because the distinction is easy to lose: even on success
this test proves only that the environment constructs, loads assets, initialises
SONIC, steps physics and shuts down — **under zero actions**. Zero actions means
the robot is commanded nothing, so the snack box is never picked or placed. It is
an integration check. It is **not** evidence of successful manipulation, and no
task-success claim can be derived from it.

### 20.10 Readiness

**Not ready for the next test.** One dependency pin stands between here and a
first real G1 ReconBody rollout. Hardware, assets, GPU passthrough, the image
build and the simulator launch are all verified working; the Python environment
inside the image is not. Once `pyarrow` imports, the command in §20.2 is the
correct next thing to run, and it will for the first time yield a genuine VRAM
measurement for this scene.

### 20.11 Resolution — pyarrow pinned, smoke test PASSES

**Fixed 2026-10-15.** The compatibility boundary was measured, not guessed.

#### The measurement

In a throwaway Python 3.11 venv with `numpy==1.26.4` pinned (matching the image's
interpreter and pin), pyarrow releases were installed newest-first:

| pyarrow | `import pyarrow` under numpy 1.26.4 |
|---|---|
| 26.0.0 | **ImportError: pyarrow requires NumPy 2.0 or newer** |
| 25.0.1 | OK |
| 24.0.0 / 23.0.0 / 22.0.0 / 21.0.0 / 20.0.0 / 19.0.1 / 18.1.0 / 17.0.0 | OK |

So **26.0.0 is exactly where numpy 1.x support was dropped**, and 25.0.1 is the
newest usable release. PyPI metadata declares no numpy requirement for any of
them — the guard is a runtime check inside the compiled extension — which is why
this had to be tested rather than looked up.

#### The fix

A **derived image layer**, not a 30 GB rebuild and not an edit to the upstream
checkout. Build file lives in our repo at `docker/Dockerfile.pyarrow-fix`:

```dockerfile
ARG BASE=robotic-grounding:pre-pyarrow-fix
FROM ${BASE}
RUN ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install --no-cache-dir \
      "numpy<2.0.0" \
      "pyarrow==25.0.1" \
 && ${ISAACLAB_PATH}/isaaclab.sh -p -c "import numpy, pyarrow, pyarrow.parquet; ..."
```

`numpy` is re-pinned in the same transaction so the resolver cannot drag it to
2.x. The build asserts the result, so a bad resolve fails the build rather than
shipping a broken image.

`${ISAACLAB_PATH}/isaaclab.sh -p -m pip` is the same invocation upstream's
Dockerfile uses, and it was confirmed to target the site-packages that held the
broken copy: `/workspace/isaaclab/_isaac_sim/kit/python/lib/python3.11/site-packages/`.

Tags:

```
robotic-grounding:pre-pyarrow-fix   fe0ca1346c8b   the original 30.4 GB build, preserved
robotic-grounding:latest            (rebuilt)      original + the pyarrow layer
```

`:latest` was retagged so every existing command, script and doc keeps working
unchanged. Build took **4.4 s**; the extra layer is the pyarrow wheel.

**The permanent upstream fix** is one line — pin `"pyarrow<26"` next to the
existing `"numpy<2.0.0"` at `robotic_grounding/workflow/Dockerfile:103-104`. That
is the change to send upstream or to apply before any rebuild from scratch.

#### Verification

```
$ docker exec … isaaclab.sh -p -c "import numpy, pyarrow, pyarrow.parquet; …"
pyarrow 25.0.1 | numpy 1.26.4
motion_schema import: OK          # the exact module that was failing
```

Full import chain and registration, probed step by step inside the container:

```
PROBE: app up
PROBE: gymnasium ok
PROBE: isaaclab_tasks ok
PROBE: robotic_grounding.tasks ok
PROBE: SonicG1 registered = 7 -> ['SonicG1-ReconBody-v0', 'SonicG1-ReconHand-EpisodeTimeout-v0',
  'SonicG1-ReconHand-Stage1-v0', 'SonicG1-ReconHand-Stage2-v0', 'SonicG1-ReconHand-Stage3-v0',
  'SonicG1-ReconHand-v0', 'SonicG1-v0']
```

**All 7 `SonicG1-*` tasks register at runtime**, which §20.4 could previously only
confirm statically from source.

#### The smoke test now passes

Same command as §20.2, `--num_envs 1 --max_steps 60`, headless, zero actions, no
cameras:

```
[INFO]: Parsing configuration from: …config.sonic.g1.g1_sonic_env_cfg:G1SonicReconBodyEnvCfg
| Active Observation Terms in Group: 'sonic_tokenizer' (shape: (1762,)) |
| Active Observation Terms in Group: 'sonic_policy'    (shape: (930,))  |
[INFO]: Completed setting up the environment...
[INFO]: Gym observation space: Dict('hand_policy': Box(…(1, 14)), 'policy': Box(…(1, 486)),
        'sonic_policy': Box(…(1, 930)), 'sonic_tokenizer': Box(…(1, 1762)))
[INFO]: Gym action space: Box(-inf, inf, (1, 43), float32)
[INFO] Running 60 simulation steps
```

521 log lines, **zero** tracebacks, OOM or CUDA errors. The G1 URDF was converted
to USD, the articulation and its Dex3 hands loaded, PhysX initialised the command
and visual markers, and the `sonic_tokenizer` group reports **1762** floats —
matching the "immutable 1762-float tokenizer input" in FlashCHORD's
`runtime/actions/sonic.py`, i.e. **SONIC initialised correctly**.

| Stage | Result |
|---|---|
| Container start | PASS |
| Isaac Sim 5.1 / Kit headless | PASS |
| Task package import | **PASS** (was the failure) |
| `SonicG1-ReconBody-v0` registration | **PASS** — runtime-verified |
| Asset + motion Parquet loading | **PASS** |
| SONIC ONNX initialisation | **PASS** — 1762-float tokenizer group active |
| Environment stepping | **PASS — 60/60 steps** |
| Shutdown | PASS — VRAM released, exit 0 |

#### VRAM — the first real measurement

| Point | Used | Free |
|---|---|---|
| Before | 1011 MiB | 6671 MiB |
| **Peak during stepping** | **3949 MiB** | 4202 MiB |
| After | 1034 MiB | 6648 MiB |

Trace shape: ~1.2–1.4 GiB through Kit startup and USD conversion, then a step to
**3.6 GiB** when the scene instantiates, then a flat plateau of **3919–3949 MiB**
for the whole 60-step loop. Return to 1034 MiB confirms clean teardown.

**So the G1 ReconBody scene costs ≈2.9 GiB of VRAM at `--num_envs 1`** on top of
the ~1.0 GiB desktop. That closes the open question in
[REQUIREMENTS.md](../reference/REQUIREMENTS.md) 8.2 for this configuration — and it is a
*single* environment. The documented smoke setting is 256 envs and full training
is 4096, so this measurement does not imply either is feasible here; it only
proves the scene runs and bounds the one-env cost.

#### Still true

This remains a **zero-action integration test**, not manipulation — see §20.9.
The robot is commanded nothing and the snack box is neither picked nor placed.
And the exit-code trap in §20.7 still holds: this run also exited 0, as did the
failing one, so exit status remains useless as a pass/fail signal here.

### 20.12 Persistence verification of the pyarrow fix

**PASS.** The fix lives in the **image**, not in a container. Re-verified
2026-10-15 after stopping *and removing* `robotic-grounding-latest-gpu0`, so no
prior container state could be reused. Every check below ran in a brand-new
`docker run --rm` container created from `robotic-grounding:latest`.

#### 1 · Versions

```
$ docker run --rm robotic-grounding:latest …
RESULT numpy   version: 1.26.4
RESULT pyarrow version: 25.0.1
```

#### 2 · Imports through Isaac Lab's interpreter

```
RESULT import numpy, pyarrow: OK
RESULT numpy.__version__  = 1.26.4
RESULT pyarrow.__version__= 25.0.1
RESULT import pyarrow.parquet: OK
RESULT interpreter: /workspace/isaaclab/_isaac_sim/kit/python/bin/python3
```

That is Isaac Sim's bundled Python 3.11 — the same interpreter that failed
before, not a side environment.

#### 3 · `robotic_grounding.motion_schema`

```
RESULT robotic_grounding.motion_schema import: OK
RESULT module file: /workspace/video_to_data/robotic_grounding/source/robotic_grounding/robotic_grounding/motion_schema/__init__.py
RESULT reader.load_motion_data_parquet: OK
```

This is the exact module whose `import pyarrow as pa` at `reader.py:24` broke the
whole task tree (§20.3).

#### 4 · Fresh container, and proof it is a layer

`docker history` shows the fix as the top layer of the image:

```
RUN /bin/bash -c ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install --no-cache-dir \
      "numpy<2.0.0" "pyarrow==25.0.1" && …
```

**Control — the preserved original image still fails**, which rules out
environment drift as the explanation:

```
$ docker run --rm robotic-grounding:pre-pyarrow-fix …
CONTROL pyarrow version: 26.0.0
CONTROL import FAILED as expected: pyarrow requires NumPy 2.0 or newer, found 1.26.4
```

**Functional check — a real Parquet is actually read**, not merely imported:

```
RESULT parquet rows: 1 cols: 42        # the G1 snack-box motion_v1 file
RESULT read_table OK, bytes: 4810493
```

(Worth noting in passing: `motion_v1` stores a whole trajectory as **one row of
42 list-valued columns**, not one row per frame — unlike the challenge dataset,
which is row-per-frame. Relevant when writing the adapter in
[CHALLENGE_BASELINE.md §5.4](../findings/CHALLENGE_BASELINE.md).)

#### Summary

| Check | Result |
|---|---|
| numpy / pyarrow versions in the image | **PASS** — 1.26.4 / 25.0.1 |
| `import numpy, pyarrow` via Isaac Lab python | **PASS** |
| `import pyarrow.parquet` | **PASS** |
| `robotic_grounding.motion_schema` | **PASS** |
| Fresh container (old container removed first) | **PASS** |
| Baked as an image layer | **PASS** — visible in `docker history` |
| Original image still broken (control) | **PASS** — isolates the fix as the cause |
| Reads a real Parquet end to end | **PASS** |

No rebuild was performed for this verification, nothing was trained, and no asset
was downloaded.

---

## 21. Where the data lives

Recorded because it is easy to get wrong and expensive to discover late.

| What | Path | Persistence |
|---|---|---|
| Challenge dataset sample (~1.1 MB) | `data/v2d_challenge/track_2/tier_1_multiview_caption/` | **in the repo**, gitignored (`.gitignore:15`) |
| Run logs, VRAM traces, probe scripts | `outputs/step7_g1_reconbody_smoke/` | in the repo, gitignored (`.gitignore:18`) |
| Upstream checkout + its LFS assets (1.3 GB) | `official/video_to_data/` | in the repo, gitignored (`.gitignore:28`) |
| Demo human-motion assets (50 MB) | `official/.../assets/human_motion_data/` | part of the upstream checkout |

The challenge sample deliberately **mirrors the upstream Hugging Face layout**
(`track_2/<tier>/{meta,data/chunk-000,…}`) so that paths in NVIDIA's docs and in
[CHALLENGE_BASELINE.md](../findings/CHALLENGE_BASELINE.md) resolve without translation, and so
the remaining 29 episodes, videos, meshes, ground planes and support surfaces can
be fetched into the same tree later.

It holds only `meta/info.json`, `meta/episodes_metadata.jsonl` and
`episode_000000.parquet`. Videos, meshes and the other 29 episodes were
deliberately not downloaded. See `data/v2d_challenge/README.md`.

**Nothing project-related should live under `/tmp`.** Scratch space is
session-scoped and is lost on cleanup or reboot; anything we want to keep belongs
in the repo's gitignored `data/` or `outputs/`.
