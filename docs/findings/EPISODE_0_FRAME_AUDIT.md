# Episode 0 Frame Audit — do the human and the stool live in a common frame?

Step 8.2. Resolves the open **[GAP]** raised in
[EPISODE_0_WALKTHROUGH.md §6.1](../guides/EPISODE_0_WALKTHROUGH.md), using NVIDIA's own
documented transform rather than a guessed sign flip, and records which
conversions the official retargeter already performs so our future adapter does
not apply them twice.

- **Audited:** 2026-10-19
- **Episode:** 0 — `2026-06-04_14-16-23_tall_bar_stool_tip_up_04`
- **Upstream read (not modified, not imported):** commit `193382a`
- **Scope:** audit only. No adapter, no downloads, no training, no simulation.

---

## 1. Verdict

**VERIFIED — the two fields are in different frames as stored, and NVIDIA's
`diag(1, -1, -1)` applied to the *object alone* brings them into one coherent
frame.** The separation that results is physically plausible across all three
phases, and the vertical axis — not the horizontal distance — is what proves it.

My earlier framing in the walkthrough was wrong in one important respect: I
treated this as an unresolved gap requiring the missing `joint_orient`. It is
neither unresolved nor related to `joint_orient`. The transform is sitting in the
pinned source with a comment explaining it.

| Measure | As stored | After `diag(1,-1,-1)` on the object |
|---|---|---|
| 3-D human-root ↔ stool distance | **5.42 – 7.03 m** | **0.534 – 0.957 m** |
| horizontal (x, z) distance | 5.26 – 6.97 m | 0.387 – 0.703 m |
| vertical offset (human − stool) | — | **+0.345 – +0.660 m, always positive** |

---

## 2. The transform, quoted from the pinned source

`robotic_grounding/scripts/retarget/soma_to_g1.py:307-313` **[VERIFIED]**

```python
# OpenCV world (X right, Y down, Z forward) to SOMA world (X right, Y up).
_R_CV_TO_SOMA = np.diag([1.0, -1.0, -1.0, 1.0])


def _convert_object_poses_cv_to_soma(object_poses_cv: np.ndarray) -> np.ndarray:
    """Convert object poses from OpenCV world axes to SOMA world axes."""
    return np.einsum("ij,tjk->tik", _R_CV_TO_SOMA, object_poses_cv)
```

Applied at `soma_to_g1.py:303`, inside `load_data()`, to `poses.npy` — **the
object trajectory only**:

```python
object_poses_world = np.load(poses_path)
object_poses_world = _convert_object_poses_cv_to_soma(object_poses_world)
```

It is a left-multiply on the full 4×4, so it converts **both** the rotation block
and the translation. Note what is *absent*: nothing anywhere converts
`soma_params.npz`'s `transl`. **The human is already in the SOMA frame and is
never touched.** That asymmetry is the whole explanation for the discrepancy.

---

## 3. Task 1 — reproducing the discrepancy

Taking both fields exactly as stored, the distance between the human root and the
stool is **5.42 – 7.03 m for all 778 frames**. The episode is a two-handed
manipulation, so this is impossible; the video shows the performer with the stool
at their feet from frame 0. **[VERIFIED]**

The discrepancy is not noise or drift — it is a constant several-metre offset,
exactly what a wrong axis convention produces.

---

## 4. Tasks 2 & 3 — comparison in a common frame

`diag(1,-1,-1)` is its own inverse, so it can be applied to either side; applying
it to the **object** is what upstream does, and leaves the human untouched. The
source dataset is not modified — the transform is applied in memory at read time.

In the resulting SOMA frame (**Y is up**):

```
3D distance        : 0.534 .. 0.957 m
horizontal (x,z)   : 0.387 .. 0.703 m
vertical human-obj : +0.345 .. +0.660 m   (positive = human above)
object Y rises     : +30.3 cm             (sign correct: it goes UP in SOMA +Y)
```

### 4.1 The vertical axis is the decisive test

The walkthrough could not separate "negate z" from "negate y and z" because both
give **identical horizontal distances** — a y-flip cannot change an (x, z)
distance. The vertical axis separates them immediately:

| Candidate | Human relative to stool | Verdict |
|---|---|---|
| `diag(1,-1,-1)` — NVIDIA's | **+0.345 … +0.660 m (above)** | physical ✅ |
| `diag(1,1,-1)` — z only | −1.384 … −0.897 m (**below**) | impossible ❌ |

A pelvis cannot sit a metre *below* a stool resting on the floor. Only NVIDIA's
mapping survives. **[VERIFIED]**

### 4.2 Independent confirmation: the two bodies move together

A distance threshold is a weak test on its own — so check coupling instead. During
the tip-up (frames 281–618, while the stool is held):

```
correlation of human x vs object x = +0.918
correlation of human z vs object z = +0.903
angle between net horizontal displacements = 18.8 deg
```

The human and the stool travel together, in nearly the same direction. That is the
signature of a real grasp and it does not depend on any assumed reach. **[VERIFIED]**

---

## 5. Task 4 — is the separation plausible in each phase?

| Phase | Frames | Horizontal | 3-D | Vertical (human − stool) | Plausible? |
|---|---|---|---|---|---|
| **approach** | 0–281 | 0.518 – 0.703 m | 0.804 – 0.957 m | +0.560 – +0.660 m | **yes** |
| **grasp / tip-up** | 281–618 | 0.387 – 0.669 m | 0.534 – 0.872 m | +0.345 – +0.559 m | **yes** |
| **release** | 618–777 | 0.421 – 0.512 m | 0.579 – 0.647 m | +0.374 – +0.411 m | **yes** |

Reading these physically:

- **Approach.** The performer stands ~0.5–0.7 m from the stool with the pelvis
  ~0.6 m above it — standing upright beside an object lying flat on the floor.
- **Grasp.** Both distances reach their minimum, and the vertical gap narrows to
  0.345 m: the person crouches, lowering the pelvis toward the stool while the
  stool rises toward the pelvis. Closest approach coincides with manipulation,
  which is what should happen.
- **Release.** Distances settle into a narrow band (0.42–0.51 m horizontal) and
  stop varying, consistent with letting go and standing back up.

The vertical gap closing by ~0.25 m while the stool independently rises ~30 cm
is a consistency check the frame conversion was never fitted to — it follows from
the data. **[VERIFIED]**

### 5.1 One apparent anomaly, resolved

The approach phase shows **no net closing**: distance to the stool's resting spot
changes by only **+0.020 m** over 9.4 seconds, which contradicts "walk over to the
tall bar stool".

This is **not** a frame error. Inspecting `outputs/episode_0/keyframes/frame_0000.jpg`,
the performer begins standing **directly over the stool, with it at their feet**.
The task string is generic boilerplate shared across the 30 episodes, not a
description of this particular take. The recorded 0.52–0.70 m matches the video.
**[VERIFIED]**

### 5.2 Bounding the claim honestly

`observation.object.pose` gives the pose of the **mesh origin**, which need not
coincide with the stool's centroid or with the part that is gripped. The mesh was
deliberately not downloaded, so the origin's placement inside the stool is
unknown — an offset of some tens of centimetres is possible.

This does not threaten the conclusion. A constant offset cannot convert 5–7 m into
0.4–0.9 m, cannot flip the sign of the vertical relationship, and cannot create the
0.9 correlation during the grasp. But it does mean the absolute distances above
should be read as **"consistent with arm's reach"**, not as a calibrated
measurement of hand-to-stool distance. **[UNRESOLVED — bounded, non-blocking]**

---

## 6. Task 5 — the comparison plot

`outputs/episode_0/frame_audit.png`, with `frame_audit_summary.json` alongside it.

- **Panel A** — human-root-to-stool distance before and after the conversion, on
  one axis in metres. The "as stored" line sits at 5–7 m; the converted line drops
  into the shaded sub-metre band.
- **Panel B** — the decisive vertical check: human root and stool origin plotted as
  heights in the SOMA frame. The pelvis stays above the stool for every frame, and
  the shaded gap narrows as the stool comes upright.

---

## 7. Task 6 — which conversions the retargeter already performs

**This is the section that matters for the adapter.** Every transform below is
applied by `soma_to_g1.py` itself. Our adapter must hand over data in the **input**
frame of each row and apply **none** of them.

| # | Transform | Applied by | Applied to | Source |
|---|---|---|---|---|
| 1 | `diag(1,-1,-1)` — OpenCV world → SOMA world (Y up) | `load_data()` | **object poses only** | `soma_to_g1.py:303, 308, 313` |
| 2 | the same matrix, as `cv_to_source` | `load_ground_plane_robot_frame()` | **ground plane** | `soma_to_g1.py:373` |
| 3 | `norm_transform` — anchor to the body's frame-0 root pose | `soma.load_motion(normalize=True)` and an explicit `einsum` | **human, object and plane** | `soma_to_g1.py:361-373` |
| 4 | `r_world` = `world_axis_swap_matrix` = `[[0,0,-1],[-1,0,0],[0,1,0]]` — SOMA (Y-up) → robot (Z-up) | `kin.transform_*`, `source_to_robot` | **everything** | `configs/g1/frame_alignment.json` |

**Consequences for the adapter — do NOT pre-convert:**

- **Object poses → `poses.npy` must stay in the raw OpenCV world frame.** The
  challenge Parquet already stores them that way, so build the 4×4 from
  `[x,y,z,qw,qx,qy,qz]` and write it **verbatim**. Applying `diag(1,-1,-1)`
  ourselves would double-apply it; since the matrix is its own inverse, the result
  would be the *original* broken frame — and the failure would look like a data
  problem, not an adapter bug.
- **Ground plane → copy verbatim.** It is published as `ax+by+cz+d=0` in the same
  OpenCV frame, which is exactly `cv_to_source`'s input.
- **Human `transl` and `poses` → copy verbatim.** They are already SOMA-frame.
  There is no human-side camera conversion anywhere in the pipeline.
- **Never apply the Z-up swap.** Row 4 is the retargeter's job, downstream of
  everything we produce.

The rule of thumb: **our adapter is a format translator, not a frame translator.**

---

## 8. Corrections to earlier documents

Three claims in [EPISODE_0_WALKTHROUGH.md](../guides/EPISODE_0_WALKTHROUGH.md) are now known
to be wrong and have been corrected there:

1. **§6.1 was marked `[GAP]`** — it is resolved. The frames differ by a documented,
   deliberate convention, not an unknown.
2. **`joint_orient (78,3,3)` was called "the single biggest technical unknown".**
   It is **optional**. `read_soma.py:626-629` supplies it from the active SOMA
   model when the npz omits it, and the docstring calls it *"optional legacy
   rest-pose orientations; the active SOMA model is authoritative"*. It is also
   absent from the mandatory key set at `read_soma.py:610-618`. **[VERIFIED]**
3. **The container `pyarrow`/`numpy` conflict** was listed as an open blocker. It
   has been fixed.

### 8.1 A related gap also narrows

The walkthrough listed the npz scalar flags as an open assumption. Reading
`read_soma.py` **[VERIFIED]**:

| Flag | Status |
|---|---|
| `keep_root` | **required** (`read_soma.py:617`). Challenge data has Hips at joint 0 with translation stored separately ⇒ `False`. Strongly indicated, unproven until a retarget runs. |
| `unit` | **required**. Challenge data is metres ⇒ `"meters"`. |
| `absolute_pose` | **optional**, defaults `False` (line 389); `True` raises `NotImplementedError`. Challenge stores local rotations ⇒ `False` is correct. |
| `rotation_repr` | **never read** — documentation only. |

So the mandatory npz keys are exactly: `poses`, `transl`, `joint_names`,
`identity_model_type`, `identity_coeffs`, `scale_params`, `unit`, `keep_root` —
**all eight available or derivable from the challenge release.**

---

## 9. Ledger

**[VERIFIED]**

- as stored, human↔stool distance is 5.42–7.03 m for every frame — not a common frame
- `_R_CV_TO_SOMA = diag(1,-1,-1)` exists at `soma_to_g1.py:308` and is applied to the object only
- after conversion: 3-D 0.534–0.957 m, horizontal 0.387–0.703 m
- the human root is **above** the stool on every frame (+0.345 … +0.660 m)
- the rival `diag(1,1,-1)` puts the pelvis 0.90–1.38 m **below** the stool — excluded
- object rises +30.3 cm along SOMA **+Y** — sign correct
- grasp-phase coupling: corr(x) +0.918, corr(z) +0.903, net directions 18.8° apart
- the flat approach distance matches the video — the performer starts at the stool
- `joint_orient` is optional and defaulted from `SOMALayer.t_pose_world`
- the retargeter owns four frame conversions; the adapter must apply none of them

**[UNRESOLVED]**

- the mesh origin's placement inside the stool is unknown (mesh not downloaded), so
  absolute distances are "consistent with arm's reach", not calibrated. Bounded and
  non-blocking — §5.2.
- `keep_root=False` is strongly indicated but unproven until a retarget actually runs
- whether `support_surface`'s `g1_retarget_world` coincides with the output of
  conversion 4 is untested — it only matters once a motion file exists

---

## 10. Reproduce

```bash
.venvs/episode0/bin/python -I scripts/diagnose/audit_frames_episode_0.py
```

Reads the episode Parquet read-only and writes `outputs/episode_0/frame_audit.png`
and `frame_audit_summary.json`. The transform is **copied** into the script, not
imported, so nothing in `official/` is loaded or modified. No network access.
