# The LeRobot → SOMA Adapter

Step 8.3. The piece of code NVIDIA does not supply: a bridge from the challenge's
LeRobot Parquet layout to the folder layout `soma_to_g1.py` actually reads.

Companion documents: [EPISODE_0_WALKTHROUGH.md](EPISODE_0_WALKTHROUGH.md) (what
the data means) · [EPISODE_0_FRAME_AUDIT.md](EPISODE_0_FRAME_AUDIT.md) (the
coordinate contract) · [CHALLENGE_BASELINE.md](CHALLENGE_BASELINE.md) (dataset
and submission contract).

- **Written:** 2026-10-19
- **Code:** [pipeline/lerobot_to_soma.py](../pipeline/lerobot_to_soma.py)
- **Tests:** [pipeline/tests/test_lerobot_to_soma.py](../pipeline/tests/test_lerobot_to_soma.py) — **37 passed**
- **Upstream read at:** `193382a`, unmodified
- **Scope:** adapter + tests only. No retargeting, no SOMA assets, no simulation, no training.

---

## 1. Why this exists

The chain to a leaderboard score is:

```
challenge Parquet  →  [ THIS ADAPTER ]  →  soma_to_g1.py  →  G1 motion parquet  →  FlashCHORD
```

`soma_to_g1.py` takes a positional `data_folder` and expects to find
`soma_params.npz`, `poses.npy` and `reconstructed_mesh/output_aligned.glb` inside
it. The challenge ships a LeRobot v2.1 Parquet instead. Nothing in the upstream
checkout references `observation.human.pose` or any other challenge column — the
translation is ours to write.

**It is a format translator, not a frame translator.** That distinction is the
single most important thing in this document.

---

## 2. The coordinate contract — what we must NOT do

Verified in [EPISODE_0_FRAME_AUDIT.md](EPISODE_0_FRAME_AUDIT.md):

| Field | Frame as stored | Who converts it |
|---|---|---|
| `observation.human.pose` / `.translation` | already the SOMA source frame | **nobody — it is already right** |
| `observation.object.pose` | source OpenCV world (X right, Y **down**, Z forward) | **`soma_to_g1.py` itself** |

```python
# soma_to_g1.py:307-313  — upstream does this, so we must not
_R_CV_TO_SOMA = np.diag([1.0, -1.0, -1.0, 1.0])

def _convert_object_poses_cv_to_soma(object_poses_cv):
    return np.einsum("ij,tjk->tik", _R_CV_TO_SOMA, object_poses_cv)
```

**The trap.** `diag(1,-1,-1)` is its own inverse. If the adapter also applied it,
the two applications would cancel and `soma_to_g1.py` would receive the *original
OpenCV* data while believing it had converted it. The symptom would be the
human sitting 5–7 m from the object it is holding — and it would look like a
dataset problem, not an adapter bug.

A dedicated test, `test_cv_to_soma_was_NOT_applied`, asserts the sign is
untouched so this can never regress silently.

---

## 3. What the code does, step by step

### 3.1 Recovering the joint names — `collapse_joint_names()`

`meta/info.json` names the pose column one **scalar** at a time:
`['Hips.rx', 'Hips.ry', 'Hips.rz', 'Spine1.rx', …]` — 231 entries. SOMA wants one
name per **joint** — 77 entries.

The function does more than strip suffixes: it **verifies the layout**. For every
joint it checks that the three consecutive names share one base name and carry
exactly `.rx, .ry, .rz` in that order. This matters because the reshape in the
next step is only correct if the 231 values really are consecutive per-joint
triplets. If the dataset were ever laid out axis-major instead
(`all .rx, then all .ry`), the reshape would silently scramble every rotation —
so the adapter refuses rather than assume.

### 3.2 The reshape — `reshape_human_pose()`

```
(778, 231)  →  (778, 77, 3)
```

Because the triplet layout is verified, a plain C-order reshape is exactly the
right regrouping: flat value `3*j + k` becomes `[j, k]`. Each `[j]` is an
**axis-angle vector in radians** — direction is the axis, length is the angle —
giving that joint's rotation **relative to its parent**.

`test_reshape_regroups_values_correctly` pins this with a counting array so the
index arithmetic is proven, not assumed.

### 3.3 Quaternion → rotation matrix — `quaternion_wxyz_to_matrix()`

The dataset stores **`[qw, qx, qy, qz]` — w first**. SciPy, ROS and most libraries
use **w last**. Mixing them up produces a perfectly valid unit quaternion that
encodes the *wrong rotation*, so nothing downstream would complain.

Two deliberate choices:

- **Components are reordered by hand**, not via SciPy's `scalar_first=True`. That
  argument only exists in SciPy ≥ 1.14 and this repo pins 1.11.4, so using it
  would break silently on the pinned environment.
- **The matrix expansion is written out longhand**, so the adapter has no hidden
  dependency on a rotation library. `test_quaternion_roundtrips_via_scipy`
  cross-checks it against SciPy anyway, to 1e-9.

Quaternions are normalised only to absorb float32 storage error (norms come in at
1.0 ± 1e-7). A **zero** quaternion raises instead — see §5.

### 3.4 Object poses → homogeneous transforms — `object_poses_to_matrices()`

```
(778, 7) [x,y,z,qw,qx,qy,qz]  →  (778, 4, 4)
```

Each matrix is `world_T_object`: rotation block from the quaternion, translation
block copied **bit for bit** from the stored xyz, bottom row `[0,0,0,1]`. This
array is `poses.npy`.

---

## 4. Field mapping

| `soma_params.npz` key | Source | Shape | Treatment |
|---|---|---|---|
| `poses` | `observation.human.pose` | `(778, 77, 3)` f32 | reshaped only |
| `transl` | `observation.human.translation` | `(778, 3)` f32 | **verbatim** |
| `identity_coeffs` | `observation.human.identity_coeffs` | `(778, 45)` f32 | **verbatim** |
| `scale_params` | `observation.human.scale_params` | `(778, 68)` f32 | **verbatim** |
| `joint_names` | `meta/info.json`, collapsed | `(77,)` `<U18` | derived + verified |
| `identity_model_type` | `info.json` `robot_type` | scalar `<U3` = `'mhr'` | `soma_x_mhr` → `mhr` |
| `unit` | — | scalar `<U6` = `'meters'` | dataset is metric |
| `keep_root` | — | scalar bool `False` | see §6 |
| `absolute_pose` | — | scalar bool `False` | poses are local rotations |
| `rotation_repr` | — | scalar `<U6` = `'rotvec'` | never read upstream |
| `bone_length_flexibles` | `observation.human.bone_length_flexibles` | `(778, 6)` f32 | carried through |
| `joint_orient` | — | **absent** | optional; see §6.2 |

| `poses.npy` | Source | Shape |
|---|---|---|
| object trajectory | `observation.object.pose` | `(778, 4, 4)` f64, **OpenCV frame unchanged** |

Scalars are wrapped in `np.array(...)` so they survive the npz round-trip in the
exact form upstream parses them: `unit` and `identity_model_type` are read with
`.item()` when `dtype.kind == 'U'` (`read_soma.py:631-639`), and `keep_root` with
`bool(np.asarray(...).item())` (`:388`). A plain Python `str` would not decode the
same way. `test_scalar_fields_decode_the_way_upstream_parses_them` mirrors those
exact lines.

### 4.1 Why `bone_length_flexibles` is included

It has no counterpart in the documented npz schema, and `read_soma.py` never reads
it. It is included anyway because **the official SOMA exporter writes it the same
way** — as an extra array:

```python
# reconstruction/modules/v2d_sam3d_body/lib/export_soma.py:603-606
keep_root=False,
extra_arrays={
    "bone_length_flexibles": bone_length_flexibles.numpy(),
},
```

So carrying it is not an invention; it is matching upstream's own convention, and
it keeps the adapter lossless. **[VERIFIED]**

---

## 5. What the adapter refuses to do

> Task constraint: *do not silently interpolate invalid poses or invent missing
> measurements.*

Where `observation.object.visible` is `False`, NVIDIA **zero-fills** the pose row,
and a zero quaternion is not a rotation — it has norm 0. Three things the adapter
deliberately does **not** do: interpolate between neighbouring frames, substitute
the identity transform, or normalise the zero away.

Instead, `--on-invalid` forces an explicit choice:

| Mode | Behaviour |
|---|---|
| `fail` *(default)* | raises `AdapterError`, naming the count and first bad index |
| `passthrough` | writes an **all-zero 4×4** through verbatim |

The all-zero matrix is chosen precisely because it is *obviously* not a transform:
it stays detectable downstream instead of masquerading as a measurement.

**Episode 0 is visible on 778/778 frames**, so neither path is exercised by the
real data — but both are unit-tested on synthetic input, because other episodes
will differ.

---

## 6. Task 6 — determining `keep_root`

**Answer: `False`. [VERIFIED]** — and from the authoritative source, not inference.

### 6.1 Three independent lines of evidence

1. **The official exporter hard-codes it.**
   `reconstruction/modules/v2d_sam3d_body/lib/export_soma.py:603` calls
   `save_soma_npz(..., keep_root=False, ...)`. Every SOMA-X export is written with
   `keep_root=False`, so that is simply the convention. This is the decisive one.

2. **The semantics match our data.** `read_soma.py:20` documents *"`keep_root`
   bool — when False, `poses[:, 0]` is the root"*. In the challenge data joint 0
   is `Hips` and the root **translation** is a separate column, which is exactly
   the `keep_root=False` arrangement.

3. **The value is inert in the retargeting path anyway.** It is read at
   `read_soma.py:388` and passed to `_reconstruct`, which immediately discards it:

   ```python
   # read_soma.py:461-464
   del (
       joint_orient,
       keep_root,
   )  # info captured by ``self.layer.t_pose_world`` already
   ```

   `_first_frame_transform` uses `poses_rotvec[0, 0]` unconditionally. A
   repository-wide grep finds no other consumer: only the exporter (which writes
   `False`), `read_soma.py` (which discards it) and `soma_loader_probe.py` (which
   only prints it). So **presence is mandatory, but the value changes nothing**
   in this pipeline.

Points 1 and 2 agree, and point 3 means a mistake here would be harmless anyway.
This was listed as an open assumption in
[EPISODE_0_FRAME_AUDIT.md §8.1](EPISODE_0_FRAME_AUDIT.md); it is now closed.

### 6.2 `joint_orient` is deliberately omitted

Optional. When absent, `read_soma.py:626-629` fills it from the live SOMA model,
and the docstring calls that model *"authoritative"*. Fabricating a `(78,3,3)`
array would be inventing data, so the adapter writes none.
`test_joint_orient_is_absent` keeps it that way.

### 6.3 Why 77 names and not 78

`read_soma.py:312-317` builds `rig_joint_names` as `rig_data["joint_names"][1:]`,
dropping a synthetic `"Root"` parent at index 0, and `:314` states this is *"the
`save_soma_npz` convention which drops the root"*. So the npz carries **77** names
beginning with `Hips`, while `joint_orient` — had we written it — would be `(78,3,3)`.

Our 77 names were compared directly against upstream's canonical
`SOMA_JOINTS_ORDER` (`retarget/params.py`): **identical, and in identical order.**
This is now a permanent test, `test_joint_names_match_upstream_SOMA_JOINTS_ORDER`,
which parses `params.py` **as text** — importing it would pull in torch and the
licence-gated SOMA assets. **[VERIFIED]**

---

## 7. Outputs

Written under gitignored `data/`; the source dataset is never touched (its SHA-256
hashes were re-checked after the run and are unchanged).

```
data/adapter_out/whole_body/soma/2026-06-04_14-16-23_tall_bar_stool_tip_up_04/
├── soma_params.npz        1,106,972 B
├── poses.npy                 99,712 B   (778, 4, 4) float64
└── adapter_manifest.json        645 B   provenance + what is still missing
```

```
soma_params.npz contents
  poses                 float32 (778, 77, 3)
  transl                float32 (778, 3)
  identity_coeffs       float32 (778, 45)
  scale_params          float32 (778, 68)
  bone_length_flexibles float32 (778, 6)
  joint_names           <U18    (77,)
  identity_model_type   <U3     ()        'mhr'
  unit                  <U6     ()        'meters'
  rotation_repr         <U6     ()        'rotvec'
  keep_root             bool    ()        False
  absolute_pose         bool    ()        False
```

### 7.1 End-to-end check

Pushing our `poses.npy` through NVIDIA's own conversion reproduces both earlier
audits **exactly** — a check the adapter was never fitted to:

| Quantity | Via the adapter | Independently measured earlier |
|---|---|---|
| frame-0 object in SOMA frame | `[-0.133, -0.972, -2.796]` | `[-0.133, -0.972, -2.796]` |
| 3-D human ↔ object | 0.534 – 0.957 m | 0.534 – 0.957 m |
| vertical (human − object) | +0.345 – +0.660 m | +0.345 – +0.660 m |
| net object rotation | 91.58° | 91.58° |
| object rise in SOMA +Y | +30.3 cm | +30.3 cm |

---

## 8. Test results — 37 passed

```
$ .venvs/episode0/bin/python -m pytest pipeline/tests/ -q
.....................................                                    [100%]
37 passed in 0.17s
```

| Area required | Tests |
|---|---|
| **shapes** | frame count vs metadata, all array shapes, float32 dtypes, `(778,4,4)` output |
| **joint ordering** | count + `Hips` first, nine known indices, uniqueness, **match to upstream `SOMA_JOINTS_ORDER`**, rejects axis-major layout, rejects wrong length, reshape index arithmetic |
| **quaternion conversion** | identity, known 90°-about-Z case, w-first ≠ w-last on real data, SciPy cross-check, zero quaternion rejected |
| **finite values** | all inputs finite, all outputs finite, unit norms |
| **rotation matrices** | `R Rᵀ = I`, `det R = +1`, homogeneous bottom row |
| **frame preservation** | translation copied bit-for-bit, **negative test that `diag(1,-1,-1)` was not applied**, human pose and translation unchanged |
| **npz contract** | mandatory keys, `joint_orient` absent, scalar decoding mirrors upstream, `joint_names` round-trip, `[0]`-indexability, identity/scale constant over time |
| **no invention** | invalid frames raise by default, passthrough is not interpolated, row count preserved, episode 0 fully visible |

---

## 9. Reproduce

```bash
# generate
.venvs/episode0/bin/python pipeline/lerobot_to_soma.py

# test
.venvs/episode0/bin/python -m pytest pipeline/tests/ -v
```

Useful flags: `--episode N`, `--tier-root PATH`, `--out-root PATH`,
`--on-invalid {fail,passthrough}`.

Note `python -I` is **not** usable for these, unlike the Step 8.1/8.2 scripts:
isolated mode strips the repository root from `sys.path`, so `import pipeline`
fails.

---

## 10. Ledger

**[VERIFIED]**

- 231 → `(778,77,3)` regrouping is correct, and the triplet layout is checked, not assumed
- our 77 joint names are identical and in order to upstream's `SOMA_JOINTS_ORDER`
- quaternion handled w-first; cross-checked against SciPy to 1e-9
- all 778 object transforms are proper rotations (`R Rᵀ = I`, `det = +1`)
- object translations are bit-identical to source — **no frame conversion applied**
- all eight mandatory npz keys present and decoding exactly as `read_soma.py` parses them
- `keep_root = False` — hard-coded by the official exporter, and inert in this path
- `joint_orient` correctly omitted
- `bone_length_flexibles` carried as an extra array, matching the official exporter
- the source dataset is unmodified (SHA-256 re-verified after the run)
- end-to-end output reproduces Steps 8.1 and 8.2 exactly

**[UNRESOLVED / open assumptions]**

1. **The npz has never been fed to the real `read_soma.py`.** Doing so needs the
   licence-gated SOMA-X assets, which are deliberately not staged yet. Our tests
   mirror upstream's validation logic faithfully but cannot substitute for the
   live `SOMALayer`. The one check that could still fail there is `joint_names`
   vs the *live* rig — mitigated, not eliminated, by §6.3.
2. **The output folder is intentionally incomplete.** `soma_to_g1.load_data()`
   also requires `reconstructed_mesh/output_aligned.glb`, and will raise
   `FileNotFoundError` until the 14 MB mesh is fetched. `ground_plane.json` is
   needed too, or the retargeter silently falls back to a horizontal plane at the
   frame-0 body-mesh minimum — which NVIDIA warns against. Both are listed in
   `adapter_manifest.json` under `still_required_by_soma_to_g1`. Downloading them
   was out of scope for this step.
3. **Only episode 0 has been exercised.** The adapter is written to be
   episode-agnostic, but the invalid-frame path has only been tested on synthetic
   input, since episode 0 is 100 % visible.
4. **`identity_coeffs` / `scale_params` are written as `(T, K)`** to match the
   demo sample, even though upstream reads only row `[0]`. A test asserts they are
   genuinely constant over time, so this is lossless for episode 0.

---

# Step 8.4 — Asset preparation and validation

Added 2026-10-19. Everything below is **PASS**. The single biggest open item from
§10 — *"the npz has never been fed to the real `read_soma.py`"* — is now **closed**.

## 11. What was staged, and what each asset is for

### 11.1 The three Episode 0 assets

Downloaded from `nvidia/video_to_data_challenge`, hash-verified, copied verbatim.

| Asset | Size | Role |
|---|---|---|
| `tall_bar_stool.glb` | 14,269,716 B | the scanned stool geometry. `soma_to_g1.load_data()` converts it to OBJ for contact and collision reasoning — without it the retargeter cannot know what the hands are touching. |
| `<seq>.json` (ground plane) | 396 B | where the floor is, as `ax+by+cz+d=0` in the **source OpenCV world frame**. Used to ground the retargeted body. NVIDIA warns against substituting a horizontal estimate. |
| `<seq>_support.usda` | 125 B | the static collision stage in the **`g1_retarget_world` (Z-up, metres)** — a *different* frame. Consumed downstream by FlashCHORD, **not** by `soma_to_g1.py`. |

**Verification.** The GLB's SHA-256 matches the published LFS oid exactly:
`cc214961423d674d55239077043e670c0696bf9113e225e11c1c2623c7ed26ad`. The JSON and
USDA are stored as plain (non-LFS) files, so Hugging Face publishes no hash for
them; their sizes match the API exactly and their hashes are now pinned in
`scripts/stage_episode_0_assets.py` so later runs detect drift. Format checks:
GLB reports *glTF binary, version 2*, with its internal length field equal to the
file size; the USDA parses as `#usda 1.0`; the JSON parses and its plane normal is
unit-length. **[VERIFIED]**

**The support surface is an empty stage** — `def Xform "support_surfaces" {}` with
no children. This is the predicted ground-only case: the stool rests on the floor,
so there is no table or desk to reconstruct. It is a *valid* empty stage, not a
missing file. **[VERIFIED]**

### 11.2 The ground plane independently confirms earlier work

The plane normal is `[-0.0189, -0.9983, -0.0555]` — **99.8 % aligned with Y**,
confirming from a completely independent source that Y is the vertical axis in the
OpenCV frame (Steps 8.1/8.2 deduced this from motion statistics alone).

Measuring heights above the fitted floor:

| | Height above floor |
|---|---|
| stool origin, lying flat (frame 0) | **+0.263 m** |
| stool origin, upright (frame 777) | **+0.530 m** |
| human root (pelvis) over the episode | **+0.855 … +0.930 m** |

A bar stool's centre sits ~0.25 m up lying flat and ~0.55 m up standing; an adult
pelvis sits ~0.9 m up. Step 8.1 estimated the floor at `y ≈ +1.25`; the fitted
plane gives `y = +1.2352` at the stool's (x, z) — **agreement within 1.5 cm**.
This also tightens the Step 8.2 caveat about the unknown mesh-origin offset: the
origin evidently sits near the stool's centroid, not at some far corner. **[VERIFIED]**

### 11.3 Staged layout

```
data/
├── v2d_challenge/.../              PRISTINE - exact HF mirror, never modified
│   ├── mesh/tall_bar_stool/tall_bar_stool.glb
│   ├── ground_plane/<seq>.json
│   └── support_surface/<seq>_support.usda
│
├── adapter_out/whole_body/
│   ├── soma/<seq>/                 <- soma_to_g1.py's positional data_folder
│   │   ├── soma_params.npz                 (adapter, Step 8.3)
│   │   ├── poses.npy                       (adapter, Step 8.3)
│   │   ├── ground_plane.json               <- soma_to_g1.py:371
│   │   ├── reconstructed_mesh/output_aligned.glb   <- load_data()
│   │   └── adapter_manifest.json
│   └── reconstructed_stage/
│       └── <seq>_support.usda      <- registry.py:127, for FlashCHORD
│
└── body_models/soma/               SOMA-X pinned bundle (see §12)
```

Two layers on purpose: the pristine mirror is downloaded once and never touched,
and the staged tree is derived and disposable — delete and re-run to rebuild
without re-downloading. The support USDA path was read off
`flash_chord/assets/registry.py:127`, which resolves
`sequence_dir.parent.parent / "reconstructed_stage"`, i.e.
`<motion-root>/whole_body/reconstructed_stage/`.

## 12. SOMA-X body model — access, licensing, staging

### 12.1 Access conditions — no action needed from you

| Question | Answer |
|---|---|
| Repository | `nvidia/SOMA-X` on Hugging Face |
| Gated? | **No** — the API reports `"gated": false` |
| Licence | **Apache-2.0** |
| Token / login required? | **No** — unauthenticated range requests on all three large files return HTTP 206 |
| Pinned revision | `466879a83d57eabf3d875ded2d869f2075f90348` |
| Python package | `py-soma-x` on PyPI, Apache-2.0; the container ships **0.2.1**, the pinned version |

**This corrects an assumption carried in earlier documents**, which described the
SOMA assets as "licence-gated". They are not: the model is public and Apache-2.0,
and no licence acceptance, authentication or further authorisation is required.
**[VERIFIED]**

### 12.2 Why the revision is pinned

`setup_soma_assets.py` hard-codes the revision above and explains why: it is the
last asset commit whose `SOMA_neutral.npz` embeds the public 78-joint rig. From
SOMA-X v0.2.2 onward, NVIDIA publishes a **slim** npz under the same filename with
the rig keys removed (moved into `SOMA_template_rig.usda` +
`SOMA_procedural_transforms.json`, which `read_soma` deliberately does not
consume). A slim npz would pass a file-presence check but fail inside `SOMALayer`,
so `_missing_assets` inspects npz *content*, not just presence. Our staged npz
reports **11/11 rig keys present**. **[VERIFIED]**

### 12.3 What was staged, and where

Staged with NVIDIA's own `setup_soma_assets.py` into **our** gitignored
`data/body_models/soma/` via its supported `--data-root` flag, rather than its
default inside the upstream checkout. `read_soma._resolve_data_root` treats an
explicit root as first-class, and `soma_to_g1.py --soma-data-root` passes it
through — so this keeps `official/` pristine at the cost of one extra flag at
retarget time.

| File | Size | SHA-256 vs published |
|---|---|---|
| `SOMA_neutral.npz` | 27,488,638 B | **MATCH** `515f7d5bb74be4e3…` |
| `correctives_model.pt` | 186,415,442 B | **MATCH** `067237a235f626ee…` |
| `MHR/mhr_model_lod1.pt` | 29,380,612 B | **MATCH** `93f1b6d7fbcfb10e…` |
| `MHR/base_body_lod1.obj` | 2,277,452 B | size match (plain file, no published hash) |
| `MHR/SOMA_wrap_lod1.obj` | 1,816,801 B | size match (plain file, no published hash) |

**Roles:** `SOMA_neutral.npz` is the neutral body + the 78-joint rig that turns 77
joint rotations into joint positions; `correctives_model.pt` is the learned
corrective-deformation network; the `MHR/*` files are the MHR identity model and
its base/wrap meshes, selected by `identity_model_type='mhr'` — which is exactly
what our adapter writes.

**Disk:** 236 MB staged. The container additionally cached the full 19-file bundle
(~280 MB) in its own ephemeral HF cache, which is discarded with the container.

### 12.4 Upstream's own validator passes

```
required core    : ['SOMA_neutral.npz', 'correctives_model.pt']
required for mhr : ['MHR/mhr_model_lod1.pt', 'MHR/base_body_lod1.obj', 'MHR/SOMA_wrap_lod1.obj']
MISSING/UNUSABLE : none  <-- PASS
SOMA_neutral.npz : usable (full rig, not a slim npz)
rig keys present : 11/11
_resolve_data_root accepts it: /staging/soma
```

## 13. The adapter's npz is accepted by the real `read_soma`

With the body model staged, our Step 8.3 output was fed to the genuine
`SOMA._load_npz()` — the validation path, stopping deliberately short of
`load_motion()` so that no reconstruction or retargeting was performed:

```
SOMALayer loaded. rig_joint_names: 77
  required-key check      : PASS
  identity_model_type     : mhr -> PASS
  unit                    : meters -> PASS
  joint_names vs live rig : PASS (would have raised otherwise)
  joint_orient auto-filled: (78, 3, 3)
  poses (778, 77, 3)  transl (778, 3)
```

This closes **unresolved item #1** from §10. Three predictions are now confirmed
against the live model rather than inferred from source reading:

1. **`joint_names` matches the live 77-joint rig.** `read_soma.py:648` raises if
   not; it did not. The §6.3 reasoning about 77-vs-78 was correct.
2. **`joint_orient` is genuinely optional** and was auto-filled to `(78, 3, 3)`
   from `SOMALayer.t_pose_world`, exactly the shape predicted.
3. **The scalar encodings decode correctly** — `unit='meters'`,
   `identity_model_type='mhr'` — confirming the `np.array(...)` wrapping.

## 14. Reproduce

```bash
# 1. the three Episode 0 assets -> pristine mirror
SEQ=2026-06-04_14-16-23_tall_bar_stool_tip_up_04
BASE=https://huggingface.co/datasets/nvidia/video_to_data_challenge/resolve/main/track_2/tier_1_multiview_caption
T=data/v2d_challenge/track_2/tier_1_multiview_caption
mkdir -p $T/mesh/tall_bar_stool $T/ground_plane $T/support_surface
curl -L --fail -o $T/mesh/tall_bar_stool/tall_bar_stool.glb "$BASE/mesh/tall_bar_stool/tall_bar_stool.glb"
curl -L --fail -o $T/ground_plane/$SEQ.json                 "$BASE/ground_plane/$SEQ.json"
curl -L --fail -o $T/support_surface/${SEQ}_support.usda    "$BASE/support_surface/${SEQ}_support.usda"

# 2. verify hashes + stage into the retargeter's expected paths
.venvs/episode0/bin/python scripts/stage_episode_0_assets.py

# 3. SOMA-X pinned bundle, via NVIDIA's own script, into our gitignored data/
mkdir -p data/body_models/soma
docker run --rm -v "$PWD/data/body_models/soma:/staging/soma" \
  --entrypoint bash robotic-grounding:latest -lc '
    python /workspace/video_to_data/robotic_grounding/scripts/setup_soma_assets.py \
      --data-root /staging/soma --identity-model-type mhr
    chown -R 1000:1000 /staging/soma'
```

Note the container must run as **root** (its default). Passing `docker --user`
breaks the image's `python` wrapper (`Unable to find any Python executable at
.../python.sh`) and calling the real interpreter directly fails with `Permission
denied`, because the Isaac Sim tree is owned by `isaac-sim`. The trailing
`chown` inside the same run restores host ownership.

```bash
# 4. validate (upstream's own checks + our npz against the real read_soma)
docker run --rm -v "$PWD/data/body_models/soma:/staging/soma:ro" \
  -v "$PWD/data/adapter_out/whole_body/soma/$SEQ:/staging/episode0:ro" \
  --entrypoint bash robotic-grounding:latest -lc '
    python -c "
import torch, pathlib
from robotic_grounding.retarget.read_soma import SOMA, _missing_assets
print(_missing_assets(pathlib.Path(\"/staging/soma\"), \"mhr\") or \"assets OK\")
s = SOMA(data_root=\"/staging/soma\", identity_model_type=\"mhr\", device=torch.device(\"cpu\"))
p = s._load_npz(\"/staging/episode0/soma_params.npz\")
print(\"npz accepted:\", p[\"poses\"].shape, p[\"transl\"].shape)
"'
```

## 15. Status after Step 8.4

**Everything `soma_to_g1.py` needs is now present.** Disk: 38 MB dataset mirror +
15 MB staged episode + 236 MB SOMA-X = **289 MB**.

**[VERIFIED]**

- all three Episode 0 assets downloaded, hash/size-verified and staged verbatim
- GLB SHA-256 matches the published LFS oid
- ground plane independently confirms Y-vertical and agrees with the Step 8.1
  floor estimate to 1.5 cm
- support surface is a valid *empty* (ground-only) stage, as predicted
- SOMA-X is public, Apache-2.0 and **ungated** — no user action required
- all five SOMA-X assets staged at the pinned revision; three LFS hashes match
- `SOMA_neutral.npz` is the full rig (11/11 keys), not the slim variant
- upstream's `_missing_assets` and `_resolve_data_root` both accept the bundle
- **our adapter's npz is accepted by the real `read_soma._load_npz()`**
- `official/` is unmodified; the pristine dataset mirror's hashes are unchanged

**[UNRESOLVED]**

1. **`load_motion()` has still not been run.** Validation stopped at `_load_npz()`
   by design, since reconstruction was out of scope. The remaining untested step
   is whether SOMALayer reconstructs plausible joint positions from our poses —
   the natural first action of Step 8.5.
2. **`keep_root=False` is still only inferred-and-inert.** `_load_npz` does not
   exercise it; `_reconstruct` discards it. Unchanged from §6.1, and harmless.
3. **The two `.obj` files have no published SHA-256** (plain files on HF). Sizes
   match exactly and they came through NVIDIA's own downloader, but there is no
   published digest to check them against.
4. **GPU VRAM.** The validation above ran on CPU. FlashCHORD training still needs
   ≥48 GB; this machine has ~8 GB. Unchanged, and still the hard blocker for
   submission rather than for retargeting.
