# Episode 0 — SOMA → G1 Retargeting (Step B)

Producing a complete, validated G1 `motion_v1` reference for the scored
competition episode `tall_bar_stool_tip_up_04`, and viewing it in FlashCHORD.

Companion documents: [CHALLENGE_ADAPTER.md](CHALLENGE_ADAPTER.md) (Step A, the
LeRobot → SOMA adapter) · [EPISODE_0_FRAME_AUDIT.md](EPISODE_0_FRAME_AUDIT.md)
(coordinate frames) · [FLASHCHORD_BASELINE_SMOKE.md](FLASHCHORD_BASELINE_SMOKE.md)
(replay verification) · [FLASHCHORD_TRAINING_FEASIBILITY.md](FLASHCHORD_TRAINING_FEASIBILITY.md)
(PPO feasibility).

- **Run:** 2026-10-20
- **Upstream code:** commit `193382a184308f6800331f3db829f76dc4294cfe` (unmodified)
- **Episode:** `2026-06-04_14-16-23_tall_bar_stool_tip_up_04`, **778 frames @ 30 fps**
- **Task text:** *"Walk over to the tall bar stool and using both hands, tip the stool from a face down lying down position back to upright."*
- **Images:** `robotic-grounding:latest` (retarget), `flash_chord:latest` (load check + viewer)

Everything below is marked **[VERIFIED]** (observed directly), **[DOC]** (stated
by NVIDIA, not independently checked) or **[GAP]/[UNRESOLVED]**.

---

## 1. Result: PASS

| Deliverable | Verdict |
| --- | --- |
| All 778 frames retargeted (not a 64-frame sample) | ✅ **778 / 778** |
| 8 GB OOM resolved without touching NVIDIA source | ✅ peak **3,411 MiB** (was a hard OOM) |
| Frame-0 normalization and frame continuity preserved | ✅ by construction + verified |
| Schema identical to NVIDIA's shipped G1 reference | ✅ all **42** columns, same types/order |
| All poses finite, quaternions normalized | ✅ 0 non-finite of 59,906 values |
| IK residuals comparable to NVIDIA's own demo | ✅ **1.07×** their median |
| Loads in FlashCHORD and builds a physics scene | ✅ 55 bodies, 118 shapes, exact frame-0 snap |
| Visualized in Viser | ✅ live at **http://localhost:8080** |

**Output:**

```
data/adapter_out/whole_body/soma/
  sequence_id=2026-06-04_14-16-23_tall_bar_stool_tip_up_04/
    robot_name=g1/data.parquet        3,586,335 bytes   <-- the deliverable
    object/textured_mesh.obj          6,673,620 bytes
    object/textured_mesh.urdf               572 bytes
    object/material_0.png            12,094,183 bytes
```

For comparison, NVIDIA's own demo G1 reference is 3,688,593 bytes. **[VERIFIED]**

---

## 2. Input audit — everything was reused, nothing re-derived **[VERIFIED]**

| Input | Path | Size / shape |
| --- | --- | --- |
| SOMA body params | `.../soma/<seq>/soma_params.npz` | 1,106,972 B; `poses (778,77,3)`, `transl (778,3)`, `identity_coeffs (778,45)`, `scale_params (778,68)`, `keep_root=False`, `absolute_pose=False` |
| Object poses | `.../soma/<seq>/poses.npy` | 99,712 B; 778 frames, OpenCV world (passed through verbatim) |
| Object mesh | `.../reconstructed_mesh/output_aligned.glb` | 14,269,716 B |
| Ground plane | `.../ground_plane.json` | fitted plane, 489,305 inliers of 3,100,964 points |
| SOMA-X bundle | `data/body_models/soma/` | `SOMA_neutral.npz` 27.5 MB, `correctives_model.pt` 186.4 MB, `MHR/` 32.7 MB |
| Support stage | `.../reconstructed_stage/<seq>_support.usda` | 125 B — an **empty stub** (see §6) |

Mounts used (`robotic-grounding:latest`):

| Host | Container | Mode |
| --- | --- | --- |
| `data/adapter_out` | `/work/adapter_out` | rw — `soma_to_g1.py` caches `object/textured_mesh.obj` into the data folder *and* writes the parquet here |
| `data/body_models/soma` | `/soma` | ro |
| `scripts/retarget_episode_g1.py` | `/driver/retarget_episode_g1.py` | ro |

Per the repo rules the container runs as **root** and `chown -R 1000:1000 /work/adapter_out`
is the last command inside the same `docker run`, so all outputs land owned by the
host user. **[VERIFIED]**

---

## 3. The 8 GB OOM: cause, and the fix **[VERIFIED]**

### 3.1 What actually fails

A direct full-sequence run dies, and **`--start-frame`/`--end-frame` do not help** —
`soma_to_g1.py` slices at line 516, long *after* the full-sequence load at line 360,
so even a 64-frame window pays the entire cost:

```
SOMALayer.forward -> prepare_identity -> skeleton_transfer.fit
  -> fit_rotations_warp -> align_vectors_warp -> wp.from_torch
RuntimeError: Failed to allocate 283254240 bytes on device 'cuda:0'
Warp CUDA error 2: out of memory (in function wp_alloc_device_default)
```

The cause is **not** the mesh vertex buffer (778 × 18,439 × 3 floats ≈ 172 MB). It is
that `read_soma._reconstruct` broadcasts the *frame-invariant* identity across frames:

```python
identity_t = torch.from_numpy(identity).to(device).unsqueeze(0).expand(T_frames, -1)
scale_t    = torch.from_numpy(scale).to(device).unsqueeze(0).expand(T_frames, -1)
```

so `SOMALayer.forward` runs `prepare_identity` over **778 identical identities**, and
its skeleton-transfer fit allocates Warp buffers proportional to the batch.
`prepare_identity`'s own docstring says the opposite is intended:

> *"Call once per identity, then call `pose()` for each new frame. This avoids
> recomputing the identity model and skeleton transfer per frame."*

This reads as an upstream inefficiency, not a deliberate choice. It is invisible on a
large GPU and fatal on 8 GB. **[UNRESOLVED]** whether NVIDIA intends it.

### 3.2 The fix: `scripts/retarget_episode_g1.py`

A driver **in our repo**, bind-mounted read-only beside NVIDIA's source, which
replaces `SOMA.load_motion` with a frame-chunked equivalent that calls the
*unmodified* `SOMA._reconstruct` over blocks and concatenates, then delegates to
`soma_to_g1.main()` untouched. NVIDIA's files are never edited.

**Why chunking is numerically equivalent, not an approximation:**

| Property | Why it holds |
| --- | --- |
| No cross-frame coupling | `SOMALayer.forward` and `_compute_global_rotations` are elementwise over the batch (`batch_rodrigues`, `apply_joint_orient_local`, `joint_local_to_world_levelorder`) — no frame reads another frame |
| Same identity per chunk | `identity` and `scale` are taken from index `[0]` upstream and are constant across frames |
| **Frame-0 anchoring preserved** | `_first_frame_transform(poses[0,0], transl[0])` is built from the **full** arrays *before* chunking, and the same transform is applied to every chunk — matching upstream's whole-array `_normalize_to_first_frame`. `first_frame_transl` / `first_frame_R_inv` are therefore identical, which is what `soma_to_g1.py` uses to anchor the object trajectory *and* the ground plane |
| Frame continuity preserved | chunks are concatenated in source order; the driver asserts the reassembled length equals the source length |

### 3.3 Measured

| Run | Frames | Chunk | Wall | Peak VRAM | Peak host RAM | Result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Unmodified script, 64-frame window | — | — | 11 s | 3,563 MiB | 13,516 MiB | ❌ **OOM** |
| Chunked driver, 64-frame IK window | 778 loaded | 64 | 28 s | **3,411 MiB** | 7,106 MiB | ✅ |
| **Chunked driver, full sequence** | **778** | **64** | **79 s** | **3,411 MiB** | **7,025 MiB** | ✅ |

Peak VRAM is flat in sequence length once chunked, so longer episodes will not
regress. IK ran at ~13 frames/s (59 s of the 79 s). **[VERIFIED]**

---

## 4. The retargeting run **[VERIFIED]**

NVIDIA's own transforms, grounding, IK and output schema are all preserved — the
driver changes only *how many frames at a time* reach `SOMALayer`.

```
[retarget_episode_g1] SOMA.load_motion patched: chunk_frames=64
[retarget_episode_g1] SOMA reconstruct frames [0, 64) of 778
... 13 chunks ...
[retarget_episode_g1] SOMA reconstruct frames [768, 778) of 778
[soma_to_g1] grounding plane: reconstructed
             (normal=(-0.1370975, 0.0829239, 0.9870805), offset=+0.9035)
[INFO] Same-frame robot-only anti-penetration lift: [+0.0000, +0.0000] m; 0/778 frames
[INFO] Object trajectory: constant ground lift applied to plane-leveled raw poses;
       no contact-mask rewrite
Retargeting complete. Processed 778 frames.
Saved to /work/adapter_out/whole_body/soma
```

Notes:

- The **reconstructed** ground plane was used (not the frame-0 body-mesh fallback),
  so our `ground_plane.json` from Step A was consumed as intended.
- **Zero anti-penetration lift was needed on any of 778 frames** — the retargeted
  robot never penetrated the ground plane, which is independent evidence that the
  Step A ground fit and the coordinate handling are consistent.
- The object trajectory got a *constant* ground lift with **no contact-mask rewrite**,
  i.e. no per-frame data was invented.

### Coordinate rule — not broken

`soma_to_g1.py:308` applies `_R_CV_TO_SOMA = diag(1,-1,-1)` to the object poses and
the ground plane itself. Our adapter passes object poses, ground plane and human data
through **verbatim**, so this ran exactly once. The retargeted human stands with the
stool within arm's reach (§5.6), which is the symptom that would be absent if the
matrix had been double-applied. **[VERIFIED]**

---

## 5. Validation — `scripts/validate_g1_reference.py`

Run against our parquet **with NVIDIA's own shipped G1 demo reference as the control**.
`ALL CHECKS PASSED`.

### 5.1 Schema **[VERIFIED]**

`schema_version=motion_v1`, `motion_kind=single_robot`, `coord_frame=robot_base_z_up`,
`source_dataset=soma`, `fps=30.0`. The column set is **identical** to NVIDIA's demo —
all 42 columns, same names, same Arrow types, same order. `robot_joint_names` (43),
`ee_link_names`, `hand_sides` all match the demo exactly.

### 5.2 Frame counts agree across every per-frame column **[VERIFIED]**

778 in `robot_root_position`, `robot_root_wxyz`, `robot_joint_positions`,
`object_root_position`, `object_root_axis_angle`, `ik_error_per_frame`,
`ik_num_iterations`, `frame_task_errors`, and `[778, 778]` for `hand_contact_active`.

### 5.3 Joint dimensions **[VERIFIED]**

Every frame's `robot_joint_positions` is width **43**, matching `robot_joint_names`.

### 5.4 Finiteness and quaternions **[VERIFIED]**

0 non-finite values across 59,906 numbers spanning root position/orientation, joint
positions, object pose, and end-effector poses. Max `|‖q‖−1|` is 3.8e-08
(`robot_root_wxyz`) and 4.3e-08 (`object_body_wxyz`).

### 5.5 Grounding and continuity **[VERIFIED]**

| Quantity | Value |
| --- | --- |
| Robot pelvis height | min +0.704, max +0.807, **mean +0.779 m** (humanoid-plausible) |
| Object root height | min +0.262, max +0.593 m (never sinks below ground) |
| Robot root per-frame step | max **0.0139 m**, mean 0.0045 m (no teleports) |
| Object per-frame step | max **0.0056 m** (no teleports) |

Smooth sub-centimetre motion at 30 fps is what frame continuity looks like
numerically; a chunk-boundary discontinuity would appear as a step spike at a
multiple of 64 and does not.

### 5.6 Hand/object relationship **[VERIFIED]**

End effectors are `left_hand_palm_link`, `right_hand_palm_link`. Distance from the
nearest palm to the object **origin**: min 0.308, median 0.402, max 0.655 m. The
stool's `object_mesh_radius` is **0.599 m**, so a 0.31 m origin distance means the
palms are well inside the object's extent — contact, not a 5–7 m frame error.
`hand_contact_active` flags at least one hand in contact on **454 of 778 frames**,
consistent with "walk over, then tip with both hands".

### 5.7 IK residuals — the important control **[VERIFIED]**

`--diagnose-ik` flagged **778/778 frames**, which looks alarming until compared with
NVIDIA's own reference. The default threshold is 0.05 m on the **sum** of all
whole-body task errors, which essentially every frame of a G1 whole-body retarget
exceeds:

| | min | median | mean | max |
| --- | ---: | ---: | ---: | ---: |
| **Ours** (bar stool, 778 frames) | 0.5548 | **0.6934** | 0.6950 | 0.8075 |
| **NVIDIA demo** (snack box, 800 frames) | 0.5817 | **0.6453** | 0.6794 | 1.0620 |

**Our median IK residual is 1.07× NVIDIA's own shipped reference**, and our maximum is
*lower* than theirs. The dominant task is `torso_link` in all 778 frames (avg 0.1924 m)
— a stable G1-vs-human morphology offset, not drift: consecutive frames read 0.1689,
0.1688, 0.1693, 0.1700. So the flagging is a property of the retargeter's multi-task
IK on a humanoid, not a defect in our data.

One genuine difference: **iteration counts are 8.6× the demo's median** (ours median 69,
max 104; demo median 8, max 34). The solve never saturated `max_iter=200` and reached a
comparable residual, so it converged — but this sequence (tipping a floor-lying stool
with both hands, with a 0.599 m-radius object) poses a harder QP than a standing
pick-and-place. Reported, not gated. **[UNRESOLVED]** whether the gap is purely task
difficulty.

### 5.8 Mesh paths **[VERIFIED]**

`object_mesh_paths` and `object_urdf_paths` both resolve on the host. See §6 for the
portability caveat.

---

## 6. Two findings that matter for training later

### 6.1 The support surface must be disabled for this episode **[VERIFIED]**

Our `<seq>_support.usda` is a 125-byte stub declaring an empty `Xform` and no
collision shapes, and FlashCHORD rejects that:

```
ValueError: support surface contains no Cylinder or Cube collision shapes
```

**This is NVIDIA's own convention, not a gap in our pipeline.** Two of the three
`whole_body` support stages NVIDIA ships are the same empty stub:

| Support USDA | Size | Cylinder/Cube shapes |
| --- | ---: | ---: |
| `corn_can_right_left_handover_01_support.usda` | 128 B | **0** |
| `dark_blue_book_pick_handover_place_02_support.usda` | 129 B | **0** |
| `snack_box_pick_and_place_01_support.usda` | 569 B | 1 (a cylinder disk) |
| **ours** — `tall_bar_stool_tip_up_04_support.usda` | 125 B | **0** |

A mid-air handover has no support surface; neither does a stool lying on the floor —
the ground plane is the support, and FlashCHORD adds that separately via
`collision.ground=true`. `include_support=True` would fail on NVIDIA's own `corn_can`
sequence too.

**Consequence:** pass `scene.support=false` for this episode. Per-episode, the rule is
"enable the support surface only if its USDA declares collision shapes" — which
`scripts/check_reference_in_flashchord.py --include-support auto` implements. This will
need handling across all 30 episodes. **[GAP]**

### 6.2 Mesh paths are absolute container paths **[VERIFIED]**

`soma_to_g1.py` records whatever absolute path it saw, so our parquet contains
`/work/adapter_out/...`. NVIDIA's demo instead stores a *repo-relative* path
(`source/robotic_grounding/.../object/textured_mesh.obj`). That matters because
`registry.resolve_asset_path` tries, in order: the path as-is if it is a file → re-root
on an `assets/` boundary → re-root on `/data/object_assets/` → **return unchanged**.
Our path contains none of those markers, so it only resolves via the first branch.

**Consequence:** when feeding this reference to FlashCHORD, mount the asset tree at the
*same* path recorded in the parquet:

```
-v "$PWD/data/adapter_out:/work/adapter_out:ro"
```

That keeps the parquet byte-identical and needs no rewriting. Verified working in §7.
Alternatively a future run could place the output under an `assets/` boundary to get
relocatable paths. **[VERIFIED]** that the mount approach works.

---

## 7. FlashCHORD compatibility — the real gate **[VERIFIED]**

`scripts/check_reference_in_flashchord.py` loads the parquet through FlashCHORD's own
`load_reference` and `setup_scene`, builds the Newton model with G1+Dex3, and snaps to
reference frame 0:

```
num_frames      : 778
fps             : 50.0
object name     : 2026-06-04_14-16-23_tall_bar_stool_tip_up_04_object
object bodies   : ['object']
mesh radius     : [0.5988196730613708]
support usda    : .../tall_bar_stool_tip_up_04_support.usda (0 collision shapes)
include_support : False
world_count     : 1
model bodies    : 55
model shapes    : 118
joint dof       : 55
objects         : ['object']
support bound   : False
wrist pos err   : 0.000000 m
object pos err  : 0.000000 m
object ori err  : 0.000000 rad
non-finite body_q entries after reset: 0
FLASHCHORD_LOAD=PASS
```

`fps: 50.0` is the *control* rate; with `source_frame_playback=True` the frame count
stays 778, one reference frame per policy step. The exact frame-0 snap (0.000 m on both
wrist and object) is the same result the verified snack-box replay produced, and
confirms robot and object agree with the reference in FlashCHORD's own frame.

Two notes:

- `ReplayConfig()`'s defaults target Sharpa's floating-wrist action and reject G1
  (`right policy action requires a 3-DOF position and 3-DOF orientation wrist`). G1
  needs `SonicJointResidualActionConfig`, as `configs/sonic_replay.yaml` uses.
- The `No registered plugin EP device found for 'CUDAExecutionProvider'` warning
  appears here as it does in the training path. **[UNRESOLVED]** (tracked in
  [FLASHCHORD_TRAINING_FEASIBILITY.md](FLASHCHORD_TRAINING_FEASIBILITY.md)).

---

## 8. Watch it — Viser is running now

A viewer is **live** on the container `v2d-ep0-viser`, paused at frame 0 with kernels
compiled (`Replay kernels are ready`, HTTP 200 on port 8080).

### Open it

**http://localhost:8080**

In the GUI panel on the right:

1. Click the **blue "Show kinematic reference"** button → plays all 778 retargeted
   frames of the G1 and the bar stool with physics off. **This is the button that shows
   whether Step B is correct**: you should see the G1 walk to the stool and tip it
   upright, with the stool tracking the recorded trajectory exactly.
2. Both buttons re-enable at the end. The **green "Run configured replay"** button runs
   controller-driven physics with VOC assistance — expect the robot to diverge, as it
   does on NVIDIA's own demo. It is *not* a correctness test of this reference.

Stop it with:

```bash
docker rm -f v2d-ep0-viser
```

Closing the browser tab leaves the container running and holding port 8080.

> The kinematic-reference button is the meaningful visual check. The physics button
> exercises an untrained zero-residual controller and is expected to fail the task —
> see [FLASHCHORD_BASELINE_SMOKE.md](FLASHCHORD_BASELINE_SMOKE.md) §8.

---

## 9. Commands you can run yourself

### 9.0 Everything at once

```bash
cd /home/ibrahim/v2d-track2-robotic-grounding
bash scripts/verify_step_B.sh          # ~3 min: re-retargets all 778 frames and validates
bash scripts/verify_step_B.sh --fast   # ~15 s: validates the existing parquet only
```

### 9.1 Retarget all 778 frames (the memory-bounded driver)

```bash
cd /home/ibrahim/v2d-track2-robotic-grounding
SEQ=2026-06-04_14-16-23_tall_bar_stool_tip_up_04

docker run --rm --gpus all \
  -v "$PWD/data/adapter_out:/work/adapter_out" \
  -v "$PWD/data/body_models/soma:/soma:ro" \
  -v "$PWD/scripts/retarget_episode_g1.py:/driver/retarget_episode_g1.py:ro" \
  --entrypoint bash robotic-grounding:latest -lc "
python /driver/retarget_episode_g1.py --chunk-frames 64 -- \
  /work/adapter_out/whole_body/soma/$SEQ \
  --save --soma-data-root /soma --robot-name g1 \
  --motion-root /work/adapter_out --soma-subdir soma --fps 30 ;
rc=\$? ; chown -R 1000:1000 /work/adapter_out ; exit \$rc
"
# expect: "Retargeting complete. Processed 778 frames." then "Saved to ..."
```

Add `--diagnose-ik` for per-frame IK diagnostics. Lower `--chunk-frames` if a longer
episode still runs out of memory; 64 peaked at 3,411 MiB here.

### 9.2 Reproduce the OOM (optional — shows why the driver exists)

```bash
SEQ=2026-06-04_14-16-23_tall_bar_stool_tip_up_04
docker run --rm --gpus all \
  -v "$PWD/data/adapter_out:/work/adapter_out" \
  -v "$PWD/data/body_models/soma:/soma:ro" \
  --entrypoint bash robotic-grounding:latest -lc "
python /workspace/video_to_data/robotic_grounding/scripts/retarget/soma_to_g1.py \
  /work/adapter_out/whole_body/soma/$SEQ \
  --save --soma-data-root /soma --robot-name g1 \
  --motion-root /work/adapter_out --soma-subdir soma --fps 30 --end-frame 64 ;
chown -R 1000:1000 /work/adapter_out
" 2>&1 | grep -E "out of memory|Failed to allocate"
# expect: Failed to allocate 283254240 bytes on device 'cuda:0'
```

### 9.3 Validate the parquet against NVIDIA's own reference

```bash
SEQ=2026-06-04_14-16-23_tall_bar_stool_tip_up_04
DEMO=official/video_to_data/robotic_grounding/source/robotic_grounding/robotic_grounding/assets/human_motion_data/whole_body/soma/sequence_id=2026-03-06_10-24-18_snack_box_pick_and_place_01/robot_name=g1/data.parquet

.venvs/episode0/bin/python scripts/validate_g1_reference.py \
  --parquet "data/adapter_out/whole_body/soma/sequence_id=$SEQ/robot_name=g1/data.parquet" \
  --reference "$DEMO" --frames 778 \
  --mesh-root "data/adapter_out/whole_body/soma/sequence_id=$SEQ/object"
# expect: ALL CHECKS PASSED
```

### 9.4 Prove FlashCHORD can load it and build the physics scene

```bash
SEQ=2026-06-04_14-16-23_tall_bar_stool_tip_up_04
FC="$PWD/official/video_to_data/robotic_grounding/flash_chord"

docker run --rm --gpus all \
  -v "$FC:/workspace" \
  -v "$PWD/official/video_to_data/robotic_grounding/source/robotic_grounding/robotic_grounding/assets:/v2d/assets:ro" \
  -v "$PWD/data/adapter_out:/work/adapter_out:ro" \
  -v "$PWD/scripts/check_reference_in_flashchord.py:/driver/check_reference_in_flashchord.py:ro" \
  -e XLA_PYTHON_CLIENT_PREALLOCATE=false \
  -w /workspace flash_chord:latest \
  python /driver/check_reference_in_flashchord.py \
    --parquet "/work/adapter_out/whole_body/soma/sequence_id=$SEQ/robot_name=g1/data.parquet" \
    --expect-frames 778 2>&1 | grep -vE "^Module |load on device"
# expect: FLASHCHORD_LOAD=PASS
```

### 9.5 Launch the viewer yourself

```bash
SEQ=2026-06-04_14-16-23_tall_bar_stool_tip_up_04
FC="$PWD/official/video_to_data/robotic_grounding/flash_chord"
PQ="/work/adapter_out/whole_body/soma/sequence_id=$SEQ/robot_name=g1/data.parquet"

docker rm -f v2d-ep0-viser 2>/dev/null
docker run -d --name v2d-ep0-viser --gpus all -p 8080:8080 \
  -v "$FC:/workspace" \
  -v "$PWD/official/video_to_data/robotic_grounding/source/robotic_grounding/robotic_grounding/assets:/v2d/assets:ro" \
  -v "$PWD/data/adapter_out:/work/adapter_out:ro" \
  -e XLA_PYTHON_CLIENT_PREALLOCATE=false \
  -w /workspace flash_chord:latest \
  python scripts/debug/replay.py --config-name=sonic_replay \
    "task.parquet='$PQ'" scene.support=false \
    viewer.backend=viser start_paused=true markers.enabled=false

# Wait for readiness (first run on a new scene recompiles kernels, ~80 s):
until docker logs v2d-ep0-viser 2>&1 | grep -q "Replay kernels are ready"; do sleep 5; done
echo "open http://localhost:8080"
```

Note the inner single quotes around the parquet path (its `sequence_id=` and
`robot_name=` partitions would otherwise break Hydra's override grammar), and
`scene.support=false` (§6.1).

### 9.6 Confirm upstream was never modified

```bash
git -C official/video_to_data status --short    # expect empty
git -C official/video_to_data log --oneline -1  # expect 193382a
```

---

## 10. What this unblocks, and what is still missing

**Unblocked:** this is the first time our own challenge data reaches a trainable G1
reference. The episode can now be fed to `train_rl.py` / `train_flash_sac.py` exactly
like NVIDIA's demo parquet, using `scene.support=false` and the `/work/adapter_out`
mount.

**Remaining gaps:**

1. **29 of 30 episodes are not retargeted.** Only `tall_bar_stool_tip_up_04` has been
   through Step A and now Step B. **[GAP]**
2. **Support-surface handling must be decided per episode** — enable only when the
   USDA declares collision shapes (§6.1). **[GAP]**
3. **Mesh paths are environment-specific** — the `/work/adapter_out` mount contract must
   be honoured, or the output relocated under an `assets/` boundary (§6.2). **[GAP]**
4. **No trained policy for this episode.** Training feasibility is proven but a
   competitive policy is compute-bound; see
   [FLASHCHORD_TRAINING_FEASIBILITY.md](FLASHCHORD_TRAINING_FEASIBILITY.md). **[GAP]**
5. **The submission packer is still untested** — `run_policy_evaluation.py` and
   `pack_submission.py` live in the submission kit zip, not in this checkout. **[GAP]**
