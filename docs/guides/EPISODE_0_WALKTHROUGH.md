# Episode 0 Walkthrough — reading one official Track 2 demonstration

A beginner-oriented tour of a single official competition episode: what the human
did, what the dataset actually stores, how to prove the conventions rather than
trust them, and what is still missing before a G1 robot could attempt the task.

**Step 8.2 has since corrected three claims in this document** — see
[EPISODE_0_FRAME_AUDIT.md](../findings/EPISODE_0_FRAME_AUDIT.md) and the marked passages in
§6.1, §7C, §7D and §7F.

Companion documents: [EPISODE_0_FRAME_AUDIT.md](../findings/EPISODE_0_FRAME_AUDIT.md) (frame
verification) · [CHALLENGE_BASELINE.md](../findings/CHALLENGE_BASELINE.md) (dataset and
submission contract) · [TRACK2_MISSION.md](../reference/TRACK2_MISSION.md) (rules, metrics) ·
[NVIDIA_UPSTREAM.md](../reference/NVIDIA_UPSTREAM.md) (code map) ·
[SETUP_AUDIT.md](../reference/SETUP_AUDIT.md) (machine state).

- **Analysed:** 2026-10-19
- **Episode:** 0 — `2026-06-04_14-16-23_tall_bar_stool_tip_up_04`
- **Tier:** `track_2/tier_1_multiview_caption`
- **Scope:** Step 8.1 only. No adapter, no retargeting, no training, no Isaac Lab.

Claims are marked **[VERIFIED]** (computed here from the data), **[DOC]** (stated
by NVIDIA) or **[GAP]** (missing or unresolved).

---

## 1. The episode in one paragraph

A person walks up to a tall wooden bar stool lying face-down on the floor of an
office, crouches, takes it in both hands, and rotates it back onto its feet, then
steps away leaving it standing. It takes **25.9 seconds**, recorded as **778
frames at exactly 30 fps** from one fixed third-person camera. The official task
string is:

> "Walk over to the tall bar stool and using both hands, tip the stool from a
> face down lying down position back to upright."

That sentence is the *goal*; the Parquet file is the *measurement* of someone
achieving it. Everything in Track 2 hinges on turning the second into something a
robot can reproduce.

---

## 2. What we have on disk

| File | Size | What it is |
|---|---|---|
| `data/.../data/chunk-000/episode_000000.parquet` | 1,057,902 B | the trajectory — 778 rows × 13 columns |
| `data/.../videos/chunk-000/observation.images.exo_camera/episode_000000.mp4` | 24,069,401 B | the exocentric video, 1536×1152, h264, 778 frames |
| `data/.../meta/info.json` | 22,981 B | schema, joint names, fps, all 30 task strings |
| `data/.../meta/episodes_metadata.jsonl` | 20,551 B | per-episode object, mesh, ground plane, support surface |

**The video on disk was initially corrupt.** It was 18,087,936 B — exactly
17,664 KiB, a cut-off download — and `ffprobe` rejected it with `moov atom not
found`: the `mdat` box declared ~24 MB of video but the trailing index was never
written. It was re-downloaded (23 MB, below the 300 MB ask-first threshold) and
verified byte-exact against the Hugging Face LFS hash. **[VERIFIED]**

```
sha256  8e0c8217247dd9c420d59b5695ab11a0eebcde6b4c6f1df170adbba5279fe1f0
        == the dataset's published LFS oid
```

Frame count agrees three ways: Parquet rows 778, `ffprobe nb_frames` 778,
`meta/episodes_metadata.jsonl` `frames` 778. **[VERIFIED]**

---

## 3. The fields, in plain language

Each of the 778 rows is one video frame. The columns split into four groups.

### 3.1 Human joint rotations — `observation.human.pose`

**Shape `(778, 231)`, float32. 231 = 77 joints × 3.** Reshape to `(778, 77, 3)`.

For each of 77 skeleton joints this stores **how that joint is rotated relative to
its parent bone**, written as an *axis-angle vector* in **radians**: the vector's
direction is the axis to spin about, and its length is how far to spin. Three
numbers describe any 3-D rotation this way.

The thing to internalise: **these are rotations only — there are no joint
positions anywhere in the dataset.** A shoulder angle does not tell you where the
hand is until you walk down the bone chain from the root, applying each rotation
in turn. That operation (forward kinematics) needs the SOMA-X body model, which we
do not yet have — see §7.

Observed range: **−1.2759 … +2.2890 rad** (−73.1° … +131.1°). Example values from
frame 0:

```
Hips      (joint  0) = [-0.07309 -0.17277 -0.08103] rad ->  11.71 deg about its axis
RightArm  (joint 40) = [-0.25512 +0.11872 +1.21672] rad ->  71.55 deg
RightHand (joint 42) = [+0.28053 +0.10826 -0.05254] rad ->  17.49 deg
LeftArm   (joint 12) = [-0.29596 -0.09823 -1.18414] rad ->  70.16 deg
RightLeg  (joint 72) = [+0.01095 -0.09714 +0.02876] rad ->   5.84 deg
```

The two `Arm` values are large and near-opposite in their third component — the
arms are already out from the body in the very first frame.

The 77 joints, in order **[VERIFIED]** — index 0 is the root and parents always
precede children:

```
 0- 6: Hips, Spine1, Spine2, Chest, Neck1, Neck2, Head
 7-13: HeadEnd, Jaw, LeftEye, RightEye, LeftShoulder, LeftArm, LeftForeArm
14-38: LeftHand + 24 left-hand finger joints (thumb, index, middle, ring, pinky)
39-41: RightShoulder, RightArm, RightForeArm
42-66: RightHand + 24 right-hand finger joints
67-71: LeftLeg, LeftShin, LeftFoot, LeftToeBase, LeftToeEnd
72-76: RightLeg, RightShin, RightFoot, RightToeBase, RightToeEnd
```

Key indices: `Hips=0, LeftArm=12, LeftHand=14, RightArm=40, RightHand=42,
LeftLeg=67, LeftFoot=69, RightLeg=72, RightFoot=74`.

Note how much of the skeleton is **fingers** — 48 of the 77 joints, 24 per hand. This is a
hand-aware format, which is why the baseline targets a G1 fitted with Dex3 hands.

In `meta/info.json` these appear as 231 names like `Hips.rx, Hips.ry, Hips.rz`;
collapsing the `.rx/.ry/.rz` suffix recovers the 77 joint names, and that list is
**identical and in identical order** to `SOMA_JOINTS_ORDER` in the upstream
retargeting code. No joint remapping will be needed. **[VERIFIED — see
[CHALLENGE_BASELINE.md §5.3](../findings/CHALLENGE_BASELINE.md)]**

### 3.2 Human root translation — `observation.human.translation`

**Shape `(778, 3)`, float32, names `['x','y','z']`, units metres.**

Joint rotations describe the body's *shape through time*; this single vector says
**where that whole body is standing**. Together they place the person in the
world. It tracks the root joint, the Hips.

```
frame   0 (t= 0.00s) = [-0.5878 -0.3123 -3.2727] m
frame 389 (t=12.97s) = [-0.5147 -0.3372 -2.6639] m
frame 777 (t=25.90s) = [+0.5243 -0.2639 -3.7347] m

x: range 1.3224 m     y: range 0.1514 m     z: range 1.2981 m
horizontal path length walked = 3.54 m
```

**The y range is roughly ten times smaller than x and z.** That is the signature of
the vertical axis: a person walking on a flat floor sweeps metres horizontally
while their pelvis height barely changes. **Y is vertical.** **[VERIFIED]**

### 3.3 Object position and quaternion — `observation.object.pose`

**Shape `(778, 7)`, float32, names `['x','y','z','qw','qx','qy','qz']`.**

The full 6-DOF pose of the one tracked rigid object — here the `tall_bar_stool` —
as **3 numbers of position in metres** plus a **4-number quaternion** for
orientation. A quaternion is a compact, gimbal-lock-free encoding of "rotate by
this angle about this axis"; a unit quaternion always represents a valid rotation,
which is why it is preferred over Euler angles for recorded motion.

The convention is **`world_T_object`**: the transform takes points in the *mesh's
own local frame* and expresses them in the *world frame*. So the position is where
the stool's origin sits in the room, and the quaternion is how the stool is turned.

```
frame   0 xyz=[-0.1333 +0.9722 +2.7959] m  quat(w,x,y,z)=[+0.62281 +0.33432 +0.29434 +0.64319]
frame 389 xyz=[-0.0375 +0.7455 +2.8699] m  quat(w,x,y,z)=[+0.53570 +0.15420 +0.43250 +0.70866]
frame 777 xyz=[+0.4401 +0.6690 +3.2370] m  quat(w,x,y,z)=[+0.23827 -0.23980 +0.65013 +0.68048]
```

### 3.4 Timestamps — `timestamp`, `frame_index`

**Shape `(778,)`.** Seconds since the episode started, one row per frame.

```
0.000 s .. 25.900 s
dt: mean 0.033333 s, min 0.033333, max 0.033335  -> 30.0000 fps
frame_index runs 0..777 and indexes the MP4 one-to-one
```

The sampling is **uniform to float32 rounding** (the 2 µs spread is storage
precision, not jitter) — no dropped frames. That matters
because retargeting and the physics simulator both assume a fixed timestep.
**[VERIFIED]**

### 3.5 Object visibility — `observation.object.visible`

**Shape `(778,)`, bool.**

A per-frame flag saying whether the object tracker produced a valid pose. When it
is `False`, **the pose row is zero-filled** — and a zero quaternion is not a
rotation at all, it has norm 0. Feeding one into a rotation library yields
garbage or a crash, so **always filter on this flag before using a pose**.

For episode 0: **visible on 778/778 frames (100 %)**, and the count of zero-filled
rows is 0, exactly matching the count of invisible rows. Nothing needs masking
here — but other episodes will differ. **[VERIFIED]**

### 3.6 The body-shape fields (who the person is, not what they did)

| Column | Shape | Constant over time? | What it is |
|---|---|---|---|
| `observation.human.identity_coeffs` | `(778, 45)` | **yes** | a 45-number shape code for this person's build |
| `observation.human.scale_params` | `(778, 68)` | **yes** | 68 per-bone scale factors sizing the skeleton |
| `observation.human.bone_length_flexibles` | `(778, 6)` | **no** | 6 bone-length slack terms the fit was allowed to bend |

The first two are **constant across all 778 frames** — they describe the performer,
not the performance, and are repeated every row only because LeRobot's format is
one flat row per frame. **[VERIFIED]**

---

## 4. Visual outputs

All under `outputs/episode_0/` (gitignored — regenerate with §8).

| File | What it shows |
|---|---|
| `storyboard.png` | **the headline figure** — six video keyframes above the same motion as recorded numbers |
| `object_xyz_vs_time.png` | object XYZ position vs time, phases shaded |
| `human_root_xyz_vs_time.png` | human root XYZ position vs time, phases shaded |
| `episode_0_exo_preview.mp4` | 768-px-wide, 0.26 MB re-encode of the demonstration, for quick viewing |
| `keyframes/frame_*.jpg` | the six full-resolution stills used in the storyboard |
| `episode_0_summary.json` | every number in this document, machine-readable |

The full-resolution demonstration video stays at
`data/v2d_challenge/track_2/tier_1_multiview_caption/videos/chunk-000/observation.images.exo_camera/episode_000000.mp4`.
Its canonical source is the
[Hugging Face dataset](https://huggingface.co/datasets/nvidia/video_to_data_challenge/blob/main/track_2/tier_1_multiview_caption/videos/chunk-000/observation.images.exo_camera/episode_000000.mp4).

### 4.1 Connecting the demonstration to the trajectory

This is the part worth dwelling on, because it is the whole premise of the track:
**the video and the numbers are two recordings of one event.** `storyboard.png`
puts them on a shared time axis — each video still sits above a dashed line
dropping into the trajectory plots at exactly that timestamp.

Reading them together, the episode divides into three measured phases
**[VERIFIED]**:

| Phase | Frames | Time | What the video shows | What the numbers show |
|---|---|---|---|---|
| **1 approach** | 0–281 | 0.00–9.37 s | person walks over and crouches | object moves **< 1 cm and < 1°**; human root walks 1.14 m (0.84 m net) |
| **2 tip-up** | 281–618 | 9.37–20.60 s | both hands on the stool, lifting it | object rotates **0° → 91°**, travels **0.78 m**, centre rises **35 cm** |
| **3 release** | 618–777 | 20.60–25.90 s | person lets go and steps away | object **holds ~92°**, flat line; human root walks 0.85 m away |

The phase boundaries were not chosen by eye — they were detected from the data
(first frame where object displacement exceeds 1 cm; first frame reaching 98 % of
peak rotation) and then *confirmed* against the video stills. The flat line in
phase 1 is the proof that the tracker is stable: when nothing moves, the recorded
pose does not drift.

Peak object speed is **0.167 m/s at t = 18.73 s** — slow, deliberate manipulation.

---

## 5. Quaternion ordering — verified, not assumed

**Result: the ordering is `[qw, qx, qy, qz]` — w-FIRST.** **[VERIFIED]**

Getting this wrong silently produces a plausible-looking but completely wrong
rotation, so it is worth proving rather than trusting. NVIDIA declares it three
independent ways:

| Source | Statement |
|---|---|
| Parquet schema metadata `pose_note` | *"observation.object.pose is [x,y,z,qw,qx,qy,qz] for the scanned mesh in the source OpenCV world frame; invalid poses are zero-filled and marked by observation.object.visible."* |
| Parquet schema `pose_convention` | `world_T_object` |
| `meta/info.json` feature names | `['x','y','z','qw','qx','qy','qz']` |

Declarations are not verification. Two numerical checks:

**Check 1 — is it even a quaternion?** Norms across all 778 frames: min
`0.9999999`, max `1.0000000`. It is a unit quaternion. **But this does not
discriminate the ordering** — relabelling four numbers does not change their norm,
so both readings pass. A surprising number of bug reports stop here.

**Check 2 — does it match the physics?** The task tips the stool from face-down to
upright. That is a **~90° rotation about a horizontal axis**. Vertical is Y (§3.2).
Compute the net rotation from frame 0 to frame 777 under both readings:

| Reading | Net angle | Rotation axis | Tilt of axis from vertical |
|---|---|---|---|
| **A — `[qw,qx,qy,qz]` (w-first)** | 91.58° | `[−0.623, −0.065, +0.779]` | **86.24° → horizontal** ✅ |
| B — `[qx,qy,qz,qw]` (w-last) | 91.58° | `[+0.024, −1.000, −0.016]` | **1.65° → vertical** ❌ |

Both give the same *angle*; the **axis** is what separates them. Reading A says the
stool rotated about a horizontal axis — a tip-up. Reading B says it spun about its
own vertical axis, i.e. pirouetted on the spot while staying face-down.

**Independent corroboration:** the stool's centre **rises 35 cm** (peak; 30 cm
after it settles). Only a tip-up about a horizontal axis lifts the centre of mass;
a spin about the vertical axis would leave it flat. And the video confirms it
directly — the stool is unambiguously upright by t = 20.8 s.

Reading A is correct on all three counts. **Ordering confirmed: `[qw,qx,qy,qz]`.**

> **Practical note.** SciPy, ROS and many libraries use **w-LAST** (`[x,y,z,w]`).
> Converting from this dataset therefore always needs an explicit reorder:
> `scipy_quat = pose[[4,5,6,3]]`. This is the single easiest place to introduce a
> silent bug in the adapter.

---

## 6. The source coordinate frame

The object pose lives in what NVIDIA calls the **source OpenCV world frame**:
right-handed, **X right, Y DOWN, Z forward** (away from the camera). It is
**not Z-up**, which is the convention Isaac Lab and the G1 retargeted world use —
so a frame conversion is unavoidable downstream.

Confirmed from the data itself, not just the label **[VERIFIED]**:

- The human root's **y** range is 0.151 m while x and z range 1.322 m and 1.298 m.
  Y is the axis the body does not travel along → **Y is vertical**.
- The stool's **y decreases by 35 cm** as it is stood up. It rises while y falls →
  **y is down-positive**, so gravity points along **+y**.

The ground plane is published as `ax+by+cz+d=0` **in this same frame**
(`ground_plane_convention` in `info.json`), so it can be used directly with the
object pose. NVIDIA warns against substituting a horizontal world-plane
estimate. **[DOC]**

**`camera_calibration` is `null`** in this release, with the note *"Not present in
the supplied Track 2 source export."* There are **no intrinsics or extrinsics**, so
the stool mesh cannot be projected into the video from this dataset alone. The
storyboard aligns video and trajectory **in time**, not in pixels — overlaying the
mesh on the frames is not possible here. **[VERIFIED]**

### 6.1 [RESOLVED in Step 8.2] The human and the object are not in the same frame

**This is a real finding, and Step 8.2 has since explained it.** The two fields
differ by a documented, deliberate convention — see
[EPISODE_0_FRAME_AUDIT.md](../findings/EPISODE_0_FRAME_AUDIT.md). The paragraph below about
`joint_orient` being the "prime suspect" was **wrong**; the answer is NVIDIA's
`_R_CV_TO_SOMA = diag(1,-1,-1)` in `soma_to_g1.py:308`, applied to the object.

`observation.object.pose` has a documented frame. `observation.human.translation`
has **none** — `info.json` gives its names as `['x','y','z']` and says nothing more.
They do not appear to agree.

The test: this task is performed **with both hands**, so at some point the human
root must be within roughly an arm's length of the stool. Measuring the horizontal
distance between the two across all 778 frames:

| Axis mapping applied to `human.translation` | min distance | max distance |
|---|---|---|
| identity (assume the same frame) | **5.26 m** | **6.97 m** |
| negate z | 0.39 m | 0.70 m |
| negate y and z (180° about x) | 0.39 m | 0.70 m |

Assuming a shared frame puts the person **5.3–7.0 m from the stool for the entire
episode** — impossible for someone picking it up with both hands, and flatly
contradicted by the video. A sign flip on z gives **0.39–0.70 m throughout**, which
is exactly arm's reach for a crouched person.

**Conclusion: `observation.human.translation` is not in the same frame as
`observation.object.pose`.** **[VERIFIED]**

**Do not hard-code a sign flip.** Not because the relationship is unknown — it is
now known — but because **the official retargeter already applies it for us**.
`soma_to_g1.py` converts the object from the OpenCV world to the SOMA world with
`diag(1,-1,-1)` and never touches the human, which is already SOMA-frame. Our
adapter must pass both through **verbatim**; converting them ourselves would
double-apply the matrix and, since it is its own inverse, restore the broken
frame. Full analysis and the conversion-ownership table:
[EPISODE_0_FRAME_AUDIT.md §7](../findings/EPISODE_0_FRAME_AUDIT.md). **[VERIFIED]**

---

## 7. What else we need to reach a G1 reference trajectory

The chain is: *challenge Parquet → SOMA layout → `soma_to_g1.py` → G1 motion
Parquet → FlashCHORD training*. Episode 0 gets us the first box only. Missing:

**A. Episode-0 assets not yet downloaded** (deliberately out of scope for Step 8.1):

| Asset | Size on HF | Why it is needed |
|---|---|---|
| `mesh/tall_bar_stool/tall_bar_stool.glb` | 14,269,716 B | the object's geometry — required for contact and collision |
| `ground_plane/<seq>.json` | 396 B | where the floor is, in the OpenCV frame |
| `support_surface/<seq>_support.usda` | 125 B | the static collision stage, in the **G1 retargeted world** |

The support surface being only **125 B** is informative: this is one of the
ground-only episodes, so the stage is a valid *empty* one. The stool rests on the
floor and no table or desk needs to be reconstructed. **[VERIFIED]**

**B. The SOMA-X body model.** Joint rotations alone are not a motion — without the
body model there is no way to turn 77 angles into joint positions. Staged via
`scripts/setup_soma_assets.py` / `py-soma-x==0.2.1`; license-gated and not yet
present. **[GAP]**

**C. ~~`joint_orient (78,3,3)`~~ — NOT NEEDED.** Corrected in Step 8.2: it is
**optional**. `read_soma.py:626-629` supplies it from the active SOMA model when
the npz omits it, and the docstring calls it *"optional legacy rest-pose
orientations; the active SOMA model is authoritative"*. It is absent from the
mandatory key set. My earlier billing of this as "the single biggest technical
unknown" was wrong. **[VERIFIED — not a gap]**

**D. The npz scalar flags** — narrowed in Step 8.2. `rotation_repr` is never read;
`absolute_pose` is optional and defaults to the correct value; `unit` is
`"meters"`. Only `keep_root` remains an inference (`False`, strongly indicated),
unproven until a retarget runs. **[mostly resolved]**

**E. The adapter itself** — roughly 100 lines, not yet written, deliberately out of
scope here. **[GAP]**

**F. ~~Working `pyarrow` inside the container.~~ FIXED.** The container
`pyarrow`/`numpy` conflict has since been resolved. The host still uses
`.venvs/episode0` for the analysis in §8. **[resolved]**

**G. A ≥48 GB GPU** for FlashCHORD training. This machine has 8151 MiB.
**[hardware]**

Note what is **not** needed: no joint remapping (the 77 names already match
upstream exactly), and no camera calibration for the retargeting path.

---

## 8. Exact commands to reproduce

All commands run from the repository root, `/home/ibrahim/v2d-track2-robotic-grounding`.

### 8.1 One-time environment

The host Python has `numpy`, `scipy`, `matplotlib` and `ffmpeg`, but **no
`pyarrow`**, so the Parquet cannot be read without one extra package:

```bash
python3 -m venv --system-site-packages .venvs/episode0
.venvs/episode0/bin/pip install pyarrow==15.0.2
```

### 8.2 Fetch episode 0's exocentric video (only if missing or corrupt)

Check the published size first:

```bash
curl -s "https://huggingface.co/api/datasets/nvidia/video_to_data_challenge/tree/main/track_2/tier_1_multiview_caption/videos/chunk-000/observation.images.exo_camera" \
  | python3 -c "import json,sys; [print(e['size'], e['path']) for e in json.load(sys.stdin)[:1]]"
# -> 24069401   (23 MB, well under the 300 MB ask-first threshold)
```

Download and verify:

```bash
DEST=data/v2d_challenge/track_2/tier_1_multiview_caption/videos/chunk-000/observation.images.exo_camera
mkdir -p "$DEST"
curl -L --fail -o "$DEST/episode_000000.mp4" \
  "https://huggingface.co/datasets/nvidia/video_to_data_challenge/resolve/main/track_2/tier_1_multiview_caption/videos/chunk-000/observation.images.exo_camera/episode_000000.mp4"

# integrity: both must pass
sha256sum "$DEST/episode_000000.mp4"   # 8e0c8217247dd9c420d59b5695ab11a0eebcde6b4c6f1df170adbba5279fe1f0
ffprobe -v error -show_entries stream=nb_frames,width,height,r_frame_rate \
        -of default=noprint_wrappers=1 "$DEST/episode_000000.mp4"   # 778, 1536x1152, 30/1
```

> A truncated download still *looks* like an MP4 to `file` and still has a
> plausible size. `ffprobe` failing with `moov atom not found` is the tell.
> Always check the hash.

### 8.3 Run the analysis

```bash
.venvs/episode0/bin/python -I scripts/diagnose/analyze_episode_0.py
```

This prints every finding in this document and writes all of `outputs/episode_0/`.
It is read-only with respect to the dataset and touches nothing in `official/`.

Runtime is a few seconds plus roughly 20 s for the preview re-encode. Re-running is
safe; keyframes and the preview are skipped if already present, so delete
`outputs/episode_0/` to force a clean rebuild.

### 8.4 View the results

```bash
xdg-open outputs/episode_0/storyboard.png
xdg-open outputs/episode_0/episode_0_exo_preview.mp4
cat outputs/episode_0/episode_0_summary.json
```

---

## 9. Ledger

**[VERIFIED] — computed directly from the data in this step**

- 778 rows, 13 columns; frame count agrees across Parquet, MP4 and metadata
- video re-downloaded and hash-matched after the on-disk copy was found truncated
- 231 = 77 × 3 axis-angle radians; 77 joint names recovered and ordered
- timestamps exactly uniform at 30.0000 fps
- object visible on 778/778 frames; zero-fill count matches invisible count
- `identity_coeffs` and `scale_params` constant over time; `bone_length_flexibles` is not
- **quaternion ordering `[qw,qx,qy,qz]` proven by rotation-axis geometry**, not just declaration
- Y is the vertical axis and is down-positive, established from the data
- three motion phases detected numerically and confirmed against video stills
- object: 91.58° net rotation, 0.784 m displacement, 35 cm peak centre rise
- **human root translation is NOT in the object's frame** (5.3–7.0 m vs 0.39–0.70 m)
  — explained in Step 8.2 as NVIDIA's deliberate OpenCV/SOMA split
- episode 0's support surface is 125 B — a valid empty, ground-only stage

**[DOC] — NVIDIA's statement, not independently checked here**

- `world_T_object` semantics; ground plane usable directly with the object pose
- support surfaces are metres, Z-up, in `g1_retarget_world`
- the warning against substituting a horizontal ground-plane estimate

**[GAP] — unresolved**

- ~~the human/object frame relationship (§6.1)~~ — **RESOLVED in Step 8.2**
- ~~`joint_orient (78,3,3)`~~ — **optional; not a gap** (Step 8.2)
- SOMA-X body model not staged; `keep_root=False` inferred but unproven
- the LeRobot→SOMA adapter does not exist
- no camera calibration, so no mesh-to-pixel overlay is possible
- ~~container `pyarrow`/`numpy` conflict~~ fixed; GPU VRAM far below FlashCHORD's need

---

## 10. Provenance

One Parquet, one MP4 and two metadata files were read. The MP4 was re-downloaded
from Hugging Face because the existing copy was truncated; **no other dataset file
was downloaded**, and the mesh, ground plane and support surface were only *sized*
via the HF API. Nothing in `official/` was read or modified. No adapter was
written, no retargeting run, no training, no Isaac Lab, no commit, no push.
