# FlashCHORD G1 Baseline Smoke Test — Official Snack-Box Example

End-to-end run of NVIDIA's official G1 + Dex3 FlashCHORD example: kinematic
reference playback, controller-driven physics replay, and an attempt at a
trained-policy evaluation.

Companion documents: [CHALLENGE_BASELINE.md](CHALLENGE_BASELINE.md) (dataset and
submission gap) · [NVIDIA_UPSTREAM.md](NVIDIA_UPSTREAM.md) (code map) ·
[SETUP_AUDIT.md](SETUP_AUDIT.md) (machine state).

- **Run:** 2026-10-19 → 2026-10-20
- **Upstream code:** commit `193382a184308f6800331f3db829f76dc4294cfe` (unmodified)
- **Image:** `flash_chord:latest` (21.3 GB), container `flash_chord-latest-gpu0`
- **GPU:** NVIDIA GeForce RTX 5070 Laptop, 8151 MiB total, driver 580.178.04
- **Reference:** `sequence_id=2026-03-06_10-24-18_snack_box_pick_and_place_01`, `robot_name=g1`

Everything below is marked **[VERIFIED]** (observed directly), **[DOC]** (stated
by NVIDIA, not independently checked) or **[GAP]/[BLOCKED]**.

---

## 1. Headline findings

1. **Both replay modes completed all 800 frames.** Kinematic reference 800/800,
   controller-driven physics replay 800/800. No crash, no CUDA OOM, no
   non-finite joint or body state in any frame of either mode. **[VERIFIED]**
2. **Peak GPU memory was 1841 MiB of 8151 MiB**, including a ~1100–1200 MiB
   desktop baseline. FlashCHORD's own footprint at `world_count=1` is roughly
   **500–650 MiB**. Replay is nowhere near the 8 GB limit. **[VERIFIED]**
3. **The physics replay is object-assisted.** `voc_scale: 1.0` — Virtual Rigid
   Object Control applies an external PD wrench plus gravity compensation to the
   snack box every physics substep. The box follows its recorded trajectory to
   **≈1 mm for the first 450 frames whether or not the robot grasps it**. This
   is *not* unassisted policy success. **[VERIFIED]**
4. **The robot does not track the reference and falls.** Wrist position error
   crosses its 0.15 m termination threshold at **frame 138**, reaches **1.66 m**,
   and the G1 pelvis collapses from ≈0.79 m to **0.14 m at frame 663**. 662 of
   800 frames exceed at least one tracking threshold. This is the documented
   expectation, not a defect — the README states task success is not guaranteed
   for the configured replay. **[VERIFIED]**
5. **BLOCKED: NO COMPATIBLE CHECKPOINT.** There are **zero `.safetensors` files
   in the checkout and zero in the Git LFS manifest**. No FlashCHORD G1
   PPO/FlashSAC checkpoint ships with this commit, and no documentation
   references a downloadable one. A genuine trained-policy evaluation cannot be
   run until we train one. **[VERIFIED]**
6. **The SONIC and G1 integrity checks all pass.** Both ONNX models and the G1
   URDF match their pinned SHA-256 digests exactly. The previous session's hash
   failure is resolved. **[VERIFIED]**

---

## 2. Asset and integrity verification **[VERIFIED]**

All three digests are enforced in NVIDIA's own code and all three match. No
integrity check was weakened or bypassed.

| Asset | Pinned digest (source) | On-disk | Match |
| --- | --- | --- | --- |
| `assets/policies/sonic/encoder_batched.onnx` | `6071c943…8d02a` (`runtime/actions/sonic.py:26`) | `6071c943…8d02a` | ✅ |
| `assets/policies/sonic/decoder_batched.onnx` | `de629511…1d2a18` (`runtime/actions/sonic.py:27`) | `de629511…1d2a18` | ✅ |
| `assets/urdfs/g1/main_with_hand.urdf` | `8c7f768d…9d19e9` (`embodiments/g1_dex3.py:45`) | `8c7f768d…9d19e9` | ✅ |

Also confirmed:

- Both ONNX models load on `CUDAExecutionProvider` (not a CPU fallback).
- The G1 asset's **26 cylinder colliders** were found and replaced with capsules;
  a mismatch would have raised (`g1_dex3.py:213`).
- 33 MB of G1 meshes are real content, not unpulled LFS pointers.
- The snack-box parquet is 3,688,593 bytes and resolves to **800 frames**.

---

## 3. Exact commands and configuration

### 3.1 Container

The existing container was reused; nothing was rebuilt or re-downloaded.

```bash
docker ps --format '{{.Names}}\t{{.Image}}\t{{.Status}}'
# flash_chord-latest-gpu0  flash_chord:latest  Up
```

Mounts: `/workspace` → `robotic_grounding/flash_chord` (rw),
`/v2d/assets` → `robotic_grounding/source/.../assets` (ro).

> **`docker/run.sh exec` cannot be used from a non-interactive shell.** It
> hardcodes `docker exec -it` (`docker/run.sh:214`), and `-i` without a terminal
> fails with `cannot attach stdin to a TTY-enabled container`. Use plain
> `docker exec -w /workspace` instead; it is otherwise equivalent. **[VERIFIED]**

### 3.2 The Hydra quoting is mandatory, not cosmetic

The parquet path contains Hive-style `sequence_id=…` and `robot_name=…`
partitions. Those `=` characters are parsed by Hydra's override grammar, so the
**inner single quotes the README shows are required**. Omitting them fails
immediately:

```
mismatched input '=' expecting <EOF>
```

Correct form — note `'…'` *inside* `"…"`:

```bash
G1_PARQUET=/v2d/assets/human_motion_data/whole_body/soma/sequence_id=2026-03-06_10-24-18_snack_box_pick_and_place_01/robot_name=g1/data.parquet

python scripts/debug/replay.py --config-name=sonic_replay \
  "task.parquet='${G1_PARQUET}'" viewer.backend=viser start_paused=true markers.enabled=false
```

### 3.3 Composed configuration (`--cfg job --resolve`) **[VERIFIED]**

```
scene.world_count: 1            sim.fps: 50            sim.physics_fps: 200
sim.substeps: 4                 sim.solver: newton     sim.integrator: implicitfast
sim.cone: pyramidal             sim.impratio: 1.0      sim.use_mujoco_contacts: true
sim.njmax: 16384                sim.nconmax: 4096      scene.support: true
collision: physical (robot_scope=all, self+object+support+ground)
embodiment: g1_dex3             action: sonic_joint_residual (action_dim=43)
replay.voc_scale: 1.0           task.source_frame_playback: true
replay.voc_params: k_lin=50 d_lin=10 k_ang=10 d_ang=0.1
                   max_force=60 max_torque=60 mass_normalized_gains=false
```

`sonic_replay.yaml` overrides the generic `replay/physical.yaml` VOC gains to
match `robotic_grounding`'s ReconBody `VirtualRigidObjectControl` exactly, in
direct N/m and N·m/rad rather than mass-normalized units.

---

## 4. Initialization reached `Replay kernels are ready` **[VERIFIED]**

```
server listening on 0.0.0.0:8080
╭────── viser (listening *:8080) ───────╮
Compiling replay kernels before enabling Viser playback...
Replay kernels are ready; playback remains paused at frame 0.
Playback is paused at frame 0; choose a replay mode in Viser.
replay: 800 frames, robot_collision=all, control=50 Hz, physics=200 Hz,
        motion_speed=1, cone=pyramidal, impratio=1, VOC=1,
        action=SonicJointResidualAction, action_dim=43,
        encoder=CUDAExecutionProvider, decoder=CUDAExecutionProvider
```

First-time Warp/MuJoCo kernel compilation took **≈111 s** (the dominant single
kernel, `update_gradient_cholesky_blocked`, took 28 s). Subsequent runs reuse the
cache and reach readiness in **≈1 s**.

---

## 5. Frames completed

Both Viser buttons drive `ReplayRunner` through the exact same two calls, so both
were reproduced headlessly and every frame was checked. See §9 for why a headless
harness was used in addition to the buttons.

| Mode | Viser button | Runner call | Frames requested | **Frames completed** | Non-finite frames | Wall time |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Kinematic reference | **Show kinematic reference** | `runner.reset(f)` | 800 | **800** | **0** | 0.49 s (1624 fps) |
| Controller physics replay | **Run configured replay** | `runner.step(f)` | 800 | **800** | **0** | 20.9 s (38 fps) |

800 frames at 50 Hz control = **16.0 s of motion**; the physics replay integrates
800 × 4 = **3200 substeps** at 200 Hz.

The official script was additionally run unmodified through Newton's null viewer
for the physics pass (`viewer.backend=none viewer.num_frames=800`), which exited
0 in 31 s — an independent confirmation using no custom code at all.

### 5.1 Kinematic reference — exact, as expected **[VERIFIED]**

| Metric | Peak over 800 frames |
| --- | --- |
| Object position error | **0.000000 m** |
| Object orientation error | 0.000691 rad |
| Wrist position error | 0.000001 m |
| Wrist orientation error | 0.000977 rad |
| Frames exceeding any threshold | **0** |

`reset(f)` snaps robot *and* object onto reference frame `f`, so agreement to
within float32 noise is the correct result. This confirms the parquet, the
retargeted G1 reference and the scene build are mutually consistent.

### 5.2 Controller physics replay — completes, but does not track **[VERIFIED]**

| Metric | Value |
| --- | --- |
| Frames completed | **800 / 800** |
| Wrist position error — first threshold breach (0.15 m) | **frame 138** |
| Wrist position error — peak | **1.664 m** |
| Object position error — first threshold breach (0.10 m) | **frame 464** |
| Object position error — peak / mean / final | 1.033 m / 0.208 m / 0.907 m |
| Object orientation error — peak | 1.130 rad |
| Frames exceeding ≥1 threshold | **662 / 800** |
| Non-finite body/joint state | **0 frames** |

Per-frame trace (pelvis height is the fall indicator):

| Frame | Pelvis z (m) | Object z (m) | Wrist err (m) | Object err (m) |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.793 | 0.875 | 0.006 | 0.000 |
| 100 | 0.786 | 0.876 | 0.062 | 0.001 |
| 200 | 0.787 | 0.876 | 0.502 | 0.001 |
| 400 | 0.794 | 0.876 | 1.361 | 0.001 |
| 450 | 0.799 | 0.881 | 1.584 | 0.027 |
| 600 | 0.799 | 1.012 | 0.395 | 0.322 |
| **663** | **0.589** | 0.804 | 0.440 | 0.793 |
| 680 | **0.142** | 0.293 | 0.738 | 1.152 |
| 799 | **0.125** | 0.151 | 0.600 | 1.084 |

Reading of this trace:

- **Frames 0–450.** The G1 stays upright (pelvis ≈0.79 m) while its wrists drift
  steadily away from the reference, to 1.58 m. The object error stays at ≈1 mm
  **only because VOC is holding the box on its recorded trajectory** — the box is
  not being carried by the robot.
- **Frames 450–663.** Object error grows past VOC's 60 N / 60 N·m limits as the
  robot's divergence becomes gross.
- **Frame 663 onward.** The pelvis collapses to 0.14 m: the G1 **falls over** and
  the box falls with it (0.95 m → 0.15 m). The simulation remains numerically
  healthy throughout — this is a physical fall, not instability.

**This is the documented behaviour, not a bug to fix.** The README's own button
table says *"Run configured replay — Controller-driven physical replay; task
success is not guaranteed."* The replay drives a **zero policy residual** through
the pinned GEAR-SONIC prior; the residual that would close the tracking gap is
exactly what training produces. A fall here is the honest baseline: it is the
"before" picture against which a trained policy is measured.

---

## 6. GPU memory **[VERIFIED]**

Sampled with `nvidia-smi` at 1 Hz for the duration of each run. Figures are
whole-device `memory.used`, so they include the desktop's ~1100–1200 MiB.

| Run | Peak used (MiB) | Idle floor (MiB) | FlashCHORD share (MiB) | Samples |
| --- | ---: | ---: | ---: | ---: |
| 5-frame smoke (first compile) | 1823 | 1222 | ≈600 | 109 |
| Kinematic reference, 800 frames | 1632 | 1181 | ≈450 | 11 |
| Physics replay, 800 frames (harness) | 1666 | 1191 | ≈475 | 31 |
| Physics replay, 800 frames (official null viewer) | 1656 | 1208 | ≈450 | 31 |
| **Viser interactive, paused at frame 0** | **1841** | 1068 | **≈775** | 294 |

**Peak across every run: 1841 MiB of 8151 MiB (22.6 %).** Headroom is ample at
`world_count=1`. **No CUDA out-of-memory condition occurred at any point.**

For contrast, the README requires **48 GiB with ≥46,000 MiB free** for full
training, and the default `evaluation.world_count=4096` is sized for that class
of GPU. Neither was attempted. **[DOC]**

---

## 7. Errors encountered and fixes

| # | Symptom | Cause | Fix | Upstream touched? |
| --- | --- | --- | --- | --- |
| 1 | `cannot attach stdin to a TTY-enabled container` | `docker/run.sh exec` hardcodes `-it`; a non-interactive shell has no TTY | Call `docker exec -w /workspace <container>` directly | No |
| 2 | `mismatched input '=' expecting <EOF>` | Parquet path's Hive `key=value` partitions parsed by Hydra's override grammar | Use the README's documented inner single quotes: `"task.parquet='…'"` | No |
| 3 | Harness reported object error up to 1.97 m in *reference* mode | **Our harness bug.** `ReplayRunner.reset()` sets `timestep` but does not gather command targets — only `step()` does. Frame *f*'s state was being compared against frame *0*'s targets, measuring the box's travel distance | Call `runner.command.gather(runner.timestep)` after `reset()` — the same official call `step()` makes. Object error then reads 0.000 m | No |
| 4 | Previous session: SONIC SHA-256 mismatch | LFS objects not pulled | Resolved before this session; both digests now verified | No |

Benign warnings, left alone:

- `UserWarning: Inertia validation corrected 2 bodies` — Newton's own inertia
  sanitizer, emitted twice during scene build.
- `onnxruntime … This model has shape massaging nodes that will execute on CPU`
  — informational note about CUDA graph capture; both sessions still report
  `CUDAExecutionProvider`.
- At Viser shutdown, `RuntimeError: cannot schedule new futures after shutdown`
  from `websockets` — a teardown race in the viewer, after all frames complete.

**No NVIDIA file under `official/` was modified, and no integrity check was
disabled.** Fix #3 was a defect in our own harness, found *because* the official
termination API disagreed with the official reset semantics.

---

## 8. Replay assistance — do not read this as policy success **[VERIFIED]**

This is the single most important distinction in this report.

### 8.1 Replay *is* assisted

`sonic_replay.yaml` sets `replay.voc_scale: 1.0`, and `replay/physical.yaml`
documents `voc_scale` as *"1.0 = full virtual assist (kinematic-faithful object
motion)"*. At every one of the 3200 physics substeps, `ReplayRunner.step()` calls
`apply_object_control(...)` (`runtime/replay.py:133`), which writes an external
PD wrench **plus gravity compensation** into `state.body_f` for the snack box:

```
k_lin=50  d_lin=10  k_ang=10  d_ang=0.1  max_force=60 N  max_torque=60 N·m
gravity=9.81 (adds +m·g along +z)   mass_normalized_gains=false
```

Gravity compensation alone means **the box cannot fall while VOC is authoritative,
even with no robot contact at all**. The ≈1 mm object error over frames 0–450 is
therefore a measurement of VOC's stiffness, *not* evidence that the G1 grasped
or carried anything. The robot's wrists were already 1.58 m off reference at
frame 450.

### 8.2 Evaluation is *not* assisted

`evaluate_policy.py`'s docstring reads *"Score one supported RL checkpoint over a
batch of **unassisted** full-sequence attempts,"* and
`apply_evaluation_composition` (`evaluation/harness.py:228`) enforces it:

```python
config["env"]["voc_scale"] = 0.0
config["env"]["voc"]["max_force"]  = 0.0   # reset_mode == "explicit"
config["env"]["voc"]["max_torque"] = 0.0
config["env"]["auto_reset"] = False
config["env"]["reset"]["always_reset_to_first_frame"] = True
```

Tracking-failure terminations are also disabled so a failed world keeps
advancing to the end of the reference instead of freezing at its failure frame;
the thresholds are then applied to the recorded errors afterwards.

### 8.3 The resulting ladder

| Rung | What it proves | Status here |
| --- | --- | --- |
| **Simulation** | Scene, assets, contacts, solver are consistent; reference is reproducible | ✅ **PASS** — 800/800 kinematic frames, 0.000 m object error |
| **Controller replay (assisted)** | The runtime executes end-to-end: SONIC prior, VOC, contacts, 3200 substeps, no NaN/OOM | ✅ **PASS as a smoke test** — 800/800 frames, robot falls at frame 663, task not achieved |
| **Learned-policy evaluation (unassisted)** | A trained residual actually performs the task with `voc_scale=0` | ❌ **BLOCKED — no checkpoint exists** |

Rung 2 passing says nothing about rung 3. Our replay completed every frame *and*
the robot fell over; with VOC removed the box would simply have dropped at the
first frame the robot failed to hold it.

---

## 9. Why a headless harness, and what it is

The two replay modes are reachable only through Viser button callbacks
(`scripts/debug/replay.py:112-118`), and `start_paused` *requires* the Viser GUI
(`replay.py:99`). A button click cannot be scripted from this session, and the
official script prints no completion count under `viewer.backend=none`. To report
frame counts and state validity rather than assert them, both modes were also run
through `scripts/flashchord_headless_replay.py` (ours, outside `official/`).

The harness is a faithful transcription of `Replay.step()`:

| Viser button | `Replay.step()` branch | Harness |
| --- | --- | --- |
| Show kinematic reference | `runner.reset(f)` (`replay.py:163`) | `runner.reset(f)` + `command.gather` |
| Run configured replay | `runner.step(f)` (`replay.py:160`) | `runner.step(f)` |

It uses only official APIs — `setup_scene`, `ReplayRunner`,
`ReplayRunner.check_termination`, `CommandBuffers.gather` — composes the real
`sonic_replay` config through Hydra, and adds per-frame instrumentation a viewer
cannot give: finiteness of `body_q`/`body_qd`/`joint_q`/`joint_qd`, the official
tracking-termination peaks, and a pelvis/object height trace. It is staged into
the container at `/tmp` via `docker cp`, so **nothing is written under
`official/`**.

Cross-check: the official script under Newton's null viewer and the harness agree
on frame count (800), clean exit, and peak memory to within 16 MiB.

---

## 10. Trained-policy evaluation — **BLOCKED: NO COMPATIBLE CHECKPOINT**

### 10.1 Evidence **[VERIFIED]**

| Check | Result |
| --- | --- |
| `.safetensors` files on disk in `official/` | **0** |
| `.safetensors` entries in the Git LFS manifest (1057 files) | **0** |
| `outputs/` tree | only `outputs/sonic_replay/<timestamp>/replay.log` — no checkpoints |
| README/docs mentioning a pretrained or downloadable checkpoint | **none** |
| `evaluation.checkpoint` default | `???` — mandatory, no default |

`scripts/evaluate_policy.py` with `evaluation.world_count=1` and no checkpoint:

```
omegaconf.errors.MissingMandatoryValue: Missing mandatory value: evaluation.checkpoint
    full_key: evaluation.checkpoint
```

The only policy-shaped artifacts in the whole checkout are ONNX files, and none
is a FlashCHORD checkpoint:

| Artifact | Why it cannot be used |
| --- | --- |
| `assets/policies/sonic/{encoder,decoder}_batched.onnx` | The frozen SONIC *prior*, already consumed by `sonic_joint_residual`. Not a trained task residual. |
| `assets/policies/e2e_example/tissue_box/vega_sharpa_policy.onnx` | Wrong embodiment (Sharpa, not G1); wrong format; and an unpulled LFS pointer. |
| `planner/motionbricks/assets/models/{decoder,pose,root}.onnx` | Motion-planner components, unrelated to FlashCHORD RL. |

Pointing the evaluator at the Sharpa ONNX fails in NVIDIA's loader, as it should:

```
File "/workspace/src/flash_chord/training/checkpoint.py", line 81, in read_checkpoint_metadata
    with safe_open(str(path), framework="numpy") as checkpoint:
safetensors._safetensors_rust.SafetensorError: Error while deserializing header: header too large
```

`evaluate_checkpoint` calls `read_checkpoint_metadata` first
(`evaluation/harness.py:377`) and rebuilds the entire experiment — embodiment,
task parquet, observation and reward composition — from configuration **embedded
in the safetensors header**. Nothing but a real FlashCHORD training checkpoint
can satisfy it. **No checkpoint was invented or substituted.**

### 10.2 What would unblock it **[DOC]**

Only training. Per the README:

| Recipe | Terminal environment steps | Produces |
| --- | ---: | --- |
| `g1_recon_body_flash_sac` | 349,999,104 | `outputs/g1_full/policy_349999104.safetensors` |

Requiring **48 GiB with ≥46,000 MiB free**, against this machine's 8151 MiB.
**[GAP]** Training was not attempted, as instructed. Whether the G1 recipe can be
made to fit 8 GB at reduced `scene.world_count` is open. **[UNRESOLVED]**

---

## 11. Result

| Task | Verdict |
| --- | --- |
| SONIC + G1 asset hashes | ✅ **PASS** |
| Initialization reaches `Replay kernels are ready` | ✅ **PASS** |
| Kinematic reference playback, all frames | ✅ **PASS** — 800/800 |
| Controller physics replay, all frames | ✅ **PASS** — 800/800, no crash/OOM/NaN |
| Reference tracking by the zero-residual controller | ❌ **not achieved** — falls at frame 663 (expected; README says success not guaranteed) |
| Peak GPU memory captured | ✅ **PASS** — 1841 / 8151 MiB |
| Learned-policy evaluation | ⛔ **BLOCKED: NO COMPATIBLE CHECKPOINT** |

**Overall: PASS for the official example replay. BLOCKED for trained-policy
evaluation.** The FlashCHORD runtime is fully functional on this machine for
`world_count=1` replay. The baseline number we ultimately need — unassisted MPPE
from a trained G1 policy — cannot be produced here yet.

### Remaining blockers

1. **No G1 checkpoint exists.** Must be trained. **[BLOCKED]**
2. **Training needs ~46 GB VRAM; we have 8 GB.** Needs either a smaller
   `scene.world_count` configuration or different hardware. **[GAP]**
3. **Our 30 challenge episodes are not yet wired into FlashCHORD.** This smoke
   test used NVIDIA's own snack-box parquet, not our adapter output. Connecting
   `data/adapter_out/whole_body/soma/` is a separate step. **[GAP]**

---

## 12. Commands you can run yourself

### 12.0 Everything at once

```bash
cd /home/ibrahim/v2d-track2-robotic-grounding
bash scripts/verify_flashchord_smoke.sh          # ~1 min: includes both 800-frame replays
bash scripts/verify_flashchord_smoke.sh --fast   # ~15 s: skips the replays
```

Exit code 0 and `ALL CHECKS PASSED` means every claim in this document still
holds. The sections below are the same checks, run by hand.

### 12.1 Verify the asset hashes (host, no container)

```bash
cd /home/ibrahim/v2d-track2-robotic-grounding/official/video_to_data/robotic_grounding/flash_chord

# Expect the three "OK" lines.
sha256sum -c <<'EOF'
6071c943492cb6c91e99b392200a288e829f5fea9491b530b34451fcf9e8d02a  src/flash_chord/assets/policies/sonic/encoder_batched.onnx
de629511bc3a110a26827a54389e82d35a722ab9b7cc4232aab865b8451d2a18  src/flash_chord/assets/policies/sonic/decoder_batched.onnx
8c7f768dc8da6c969d8a4b4efb3dcbe3d9d2b05fca5104df12205a69879d19e9  src/flash_chord/assets/urdfs/g1/main_with_hand.urdf
EOF
```

### 12.2 Confirm no checkpoint exists

```bash
cd /home/ibrahim/v2d-track2-robotic-grounding/official/video_to_data

find . -name '*.safetensors' | wc -l        # expect 0
git lfs ls-files | grep -c safetensors      # expect 0
```

### 12.3 Replay in the browser — the two buttons

```bash
G1_PARQUET=/v2d/assets/human_motion_data/whole_body/soma/sequence_id=2026-03-06_10-24-18_snack_box_pick_and_place_01/robot_name=g1/data.parquet

docker exec -w /workspace flash_chord-latest-gpu0 \
  python scripts/debug/replay.py --config-name=sonic_replay \
  "task.parquet='${G1_PARQUET}'" viewer.backend=viser start_paused=true markers.enabled=false
```

Wait for `Replay kernels are ready; playback remains paused at frame 0.`, then
open **http://localhost:8080** and use the GUI panel on the right:

1. Click the **blue "Show kinematic reference"** button → plays the recorded
   robot and box poses with physics off. The terminal then prints
   `Full reference reached frame 799 and is paused at the final pose.`
2. Both buttons re-enable. Click the **green "Run configured replay"** button →
   controller-driven physics. Expect the G1 to drift, then topple around frame
   663 (~13 s in). The terminal prints
   `Full replay reached frame 799 and is paused at the final pose.`

Each button restarts at frame 0 and pauses at the final pose. Stop the viewer
with **Ctrl+C** — closing the browser tab leaves the process running and holding
port 8080.

### 12.4 Reproduce the headless frame counts

```bash
cd /home/ibrahim/v2d-track2-robotic-grounding
docker cp scripts/flashchord_headless_replay.py flash_chord-latest-gpu0:/tmp/

G1_PARQUET=/v2d/assets/human_motion_data/whole_body/soma/sequence_id=2026-03-06_10-24-18_snack_box_pick_and_place_01/robot_name=g1/data.parquet

# Kinematic reference: expect frames_completed 800, object error 0.0, PASS.
docker exec -w /workspace flash_chord-latest-gpu0 \
  python /tmp/flashchord_headless_replay.py --parquet "$G1_PARQUET" --mode reference

# Controller physics replay: expect frames_completed 800, PASS, wrist peak ~1.66 m.
docker exec -w /workspace flash_chord-latest-gpu0 \
  python /tmp/flashchord_headless_replay.py --parquet "$G1_PARQUET" --mode replay \
  --trace-csv /tmp/replay800_trace.csv

# The fall: pelvis height drops below 0.60 m at frame 663.
docker cp flash_chord-latest-gpu0:/tmp/replay800_trace.csv /tmp/replay800_trace.csv
awk -F, 'NR>1 && $2<0.60 {print "fall onset at frame",$1,"pelvis_z",$2; exit}' /tmp/replay800_trace.csv
```

### 12.5 Physics replay through the official script only (no custom code)

```bash
G1_PARQUET=/v2d/assets/human_motion_data/whole_body/soma/sequence_id=2026-03-06_10-24-18_snack_box_pick_and_place_01/robot_name=g1/data.parquet

docker exec -w /workspace flash_chord-latest-gpu0 \
  python scripts/debug/replay.py --config-name=sonic_replay \
  "task.parquet='${G1_PARQUET}'" viewer.backend=none viewer.num_frames=800 \
  start_paused=false markers.enabled=false
echo "exit=$?"   # expect 0
```

### 12.6 Watch GPU memory during any of the above

```bash
# In a second terminal, while a replay runs:
nvidia-smi --query-gpu=memory.used,memory.total --format=csv -l 1
# Expect a peak near 1650-1850 MiB of 8151 MiB.
```

### 12.7 Confirm the replay is object-assisted

```bash
cd /home/ibrahim/v2d-track2-robotic-grounding/official/video_to_data/robotic_grounding/flash_chord

grep -n "voc_scale" src/flash_chord/configs/replay/physical.yaml   # 1.0 = full virtual assist
grep -n "voc_scale\|max_force\|max_torque" src/flash_chord/evaluation/harness.py | head -5
# Evaluation forces voc_scale = 0.0 and zero force/torque limits: unassisted.
```
