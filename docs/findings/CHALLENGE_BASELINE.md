# Track 2 Challenge Baseline — Dataset, Submission Kit and the Gap

What the official 30-trajectory benchmark actually contains, which baseline code
NVIDIA supplies, and precisely what is missing between that data and our
`robotic_grounding` checkout.

Companion documents: [TRACK2_MISSION.md](../reference/TRACK2_MISSION.md) (rules, metrics,
deadline) · [NVIDIA_UPSTREAM.md](../reference/NVIDIA_UPSTREAM.md) (code map) ·
[SETUP_AUDIT.md](../reference/SETUP_AUDIT.md) (machine state).

- **Inspected:** 2026-10-15
- **Dataset:** `nvidia/video_to_data_challenge`, path `track_2/`
- **Upstream code:** commit `193382a184308f6800331f3db829f76dc4294cfe`
- **Submission kit:** `docs/v2d_challenge/assets/v2d_submission_kit.zip` (committed in the checkout)

Everything below is marked **[VERIFIED]** (observed directly), **[DOC]** (stated
by NVIDIA, not independently checked) or **[GAP]/[ASSUMPTION]**.

---

## 1. Headline findings

1. **Only two tiers are published.** `tier_1_multiview_caption` and
   `tier_2_synthetic_noise` exist on Hugging Face. **`tier_3_cari4d` is absent**
   from the dataset even though the submission kit and the challenge site both
   describe a Tier 3. **[VERIFIED]**
2. **30 episodes, 22,990 frames**, 30 tasks, 29 distinct objects, 30 fps, one
   static exocentric camera. **[VERIFIED]**
3. **The dataset does not contain a G1 motion file.** It ships SOMA-X *human*
   parameters plus a single tracked *object* pose per frame. The G1 retargeted
   reference that both the Isaac Lab and FlashCHORD trainers consume must be
   produced by us. **[VERIFIED]**
4. **The 77-joint SOMA order in the dataset matches `robotic_grounding`'s
   `SOMA_JOINTS_ORDER` exactly** — same names, same order. The data is
   semantically compatible with the existing retargeting code. **[VERIFIED]**
5. **No official adapter exists** from the challenge LeRobot layout to the
   `soma_to_g1.py` input layout. Nothing in upstream references
   `observation.human.pose`, `episodes_metadata` or the challenge dataset at all.
   This is the one missing piece of code. **[VERIFIED]**
6. **The official evaluator only accepts FlashCHORD checkpoints.** Isaac Lab
   RSL-RL PPO checkpoints — the `SonicG1-ReconHand-Stage*` path — are rejected by
   construction. **[VERIFIED]**
7. **The demo snack-box sequence is not in the benchmark.** Episode 13 is a
   snack box, but `snack_box_desk_through_06`, not the demo's
   `snack_box_pick_and_place_01`. **[VERIFIED]**

---

## 2. Tiers and trajectories

### 2.1 Published tiers

| Tier | Directory | Human trajectory | Object trajectory / mesh | On HF |
|---|---|---|---|---|
| 1 — multiview caption | `track_2/tier_1_multiview_caption/` | Original SOMA-X reconstruction | Original pose + aligned mesh | **yes** |
| 2 — synthetic noise | `track_2/tier_2_synthetic_noise/` | Synthetically perturbed | Synthetically perturbed pose; original mesh | **yes** |
| 3 — CARI4D | `track_2/tier_3_cari4d/` | — | — | **ABSENT** |

**[VERIFIED]** via the HF tree API. The submission kit's `config/tracks.json`
defines `track_2_tier3` with `input_path: track_2/tier_3_cari4d`, and the kit's
Track 2 guide tabulates it — so the competition expects it, but the data is not
released at this path today. **[GAP]** Either it appears later, or Tier 3 is
meant to be produced by running CARI4D ourselves on the exo video (the dataset
README says *"Reconstruction results are not published; participants generate
them with their own method"*), which the `v2d_cari4d` module in
`reconstruction/modules/` would support. Not resolvable from the data alone.

Both tiers are structurally identical: same 13-column schema, same episode order,
same 22,990-frame timeline, same tasks, **byte-identical `exo_camera` video**, same
ground planes and support surfaces. NVIDIA explicitly warns: *"Do not concatenate
tiers: episode and global frame indices intentionally overlap."* **[DOC]**

### 2.2 The 30 trajectories

| ep | frames | object | sequence_id |
|---|---|---|---|
| 0 | 778 | `tall_bar_stool` | `2026-06-04_14-16-23_tall_bar_stool_tip_up_04` |
| 1 | 747 | `vacuum` | `2026-05-18_17-06-21_vacuum_left_lean_02` |
| 2 | 980 | `blue_easter_basket` | `2026-06-02_10-04-42_blue_easter_basket_place_10` |
| 3 | 781 | `contigo_coffee_mug` | `2026-04-27_11-50-10_contigo_coffee_mug_handle_table_05` |
| 4 | 803 | `baseball_bat` | `2026-04-24_11-40-52_baseball_bat_rotate_05` |
| 5 | 1003 | `big_black_box` | `2026-03-18_19-36-56_g1_box_rotate_01` |
| 6 | 546 | `monitor` | `2026-06-01_11-26-44_monitor_rotate_05` |
| 7 | 896 | `iron` | `2026-05-29_15-20-43_iron_round_table_02` |
| 8 | 708 | `skinny_wood_chair` | `2026-06-05_15-08-08_skinny_wood_chair_push_10` |
| 9 | 870 | `big_red_bowl` | `2026-06-04_16-45-04_big_red_bowl_under_skinny_wood_chair_03` |
| 10 | 794 | `white_desk` | `2026-05-01_11-36-40_white_desk_lower_to_ground_10` |
| 11 | 901 | `light_blue_book` | `2026-04-24_13-56-58_light_blue_book_pick_place_09` |
| 12 | 700 | `black_platform` | `2026-04-25_17-20-04_black_platform_turn_05` |
| 13 | 922 | `snack_box` | `2026-06-09_10-19-09_snack_box_desk_through_06` |
| 14 | 993 | `round_table` | `2026-05-28_14-44-09_round_table_flip_08` |
| 15 | 715 | `wet_floor_sign` | `2026-05-06_11-06-15_wet_floor_sign_pick_place_ground_08` |
| 16 | 477 | `spray_bottle` | `2026-03-12_15-06-36_spray_bottle_left_desk_ground_02` |
| 17 | 440 | `traffic_cone` | `2026-05-01_17-04-55_traffic_cone_ground_table_02` |
| 18 | 858 | `blue_trash_can` | `2026-05-06_10-00-19_blue_trash_can_drag_sideways_08` |
| 19 | 886 | `cane` | `2026-05-19_10-19-48_cane_swing_05` |
| 20 | 553 | `rice_cooker` | `2026-06-08_16-57-34_rice_cooker_clockwise_04` |
| 21 | 678 | `tall_bar_stool` | `2026-04-29_11-53-57_tall_bar_stool_pull_08` |
| 22 | 719 | `step_ladder` | `2026-05-22_16-52-36_step_ladder_drag_03` |
| 23 | 946 | `toy_stroller` | `2026-06-03_15-08-03_toy_stroller_back_forth_ground_04` |
| 24 | 748 | `watering_can` | `2026-05-29_15-43-20_watering_can_pour_right_02` |
| 25 | 859 | `cyan_water_bottle` | `2026-03-20_11-59-57_cyan_water_bottle_rotate_01` |
| 26 | 669 | `skateboard` | `2026-06-05_16-05-08_skateboard_rotate_ground_10` |
| 27 | 565 | `woven_basket` | `2026-05-01_11-15-05_woven_basket_ground_desk_10` |
| 28 | 988 | `squeegee` | `2026-05-19_11-27-07_squeegee_sweep_away_from_you_03_` |
| 29 | 467 | `rolling_luggage` | `2026-06-05_18-27-40_rolling_luggage_left_foot_push_06` |

**[VERIFIED]** from `meta/episodes_metadata.jsonl`. Frames sum to **22,990**
(440–1,003, mean 766). 29 unique objects — `tall_bar_stool` appears twice
(episodes 0 and 21). All 30 episodes declare a support surface, all with
`support_surface_frame: g1_retarget_world`.

---

## 3. Formats, conventions and fields

### 3.1 Container format

**LeRobot v2.1** (`codebase_version: v2.1`), `robot_type: soma_x_mhr`. **[VERIFIED]**

```
track_2/<tier>/
├── data/chunk-000/episode_0000NN.parquet          human + object pose, one row per frame
├── videos/chunk-000/observation.images.exo_camera/episode_0000NN.mp4
├── mesh/<object>/<object>.glb                     textured mesh
├── ground_plane/<sequence_id>.json                fitted ground plane
├── support_surface/<sequence_id>_support.usda      static G1 collision stage
└── meta/{info.json,episodes_metadata.jsonl,episodes.jsonl,episodes_stats.jsonl,tasks.jsonl}
```

Sizes: 30 parquets totalling **31.2 MB**; `meta/` ~2.05 MB (of which
`episodes_stats.jsonl` is 1.998 MB). Videos and meshes were not sized — not
downloaded. **[VERIFIED]**

### 3.2 Parquet schema — 13 columns

Verified by reading `episode_000000.parquet` (1,057,902 B, **778 rows**):

| Column | Type |
|---|---|
| `observation.human.pose` | `fixed_size_list<float32, 231>` |
| `observation.human.translation` | `fixed_size_list<float32, 3>` |
| `observation.human.identity_coeffs` | `fixed_size_list<float32, 45>` |
| `observation.human.scale_params` | `fixed_size_list<float32, 68>` |
| `observation.human.bone_length_flexibles` | `fixed_size_list<float32, 6>` |
| `observation.object.pose` | `fixed_size_list<float32, 7>` |
| `observation.object.visible` | `bool` |
| `timestamp`, `frame_index`, `episode_index`, `index`, `task_index`, `next.done` | scalars |

### 3.3 Coordinate conventions

Declared in the parquet's own schema metadata and confirmed numerically:

```
pose_convention        = world_T_object
pose_note              = observation.object.pose is [x,y,z,qw,qx,qy,qz] for the scanned
                         mesh in the source OpenCV world frame; invalid poses are
                         zero-filled and marked by observation.object.visible.
human_pose_convention  = 77-joint local rotation vectors in radians; joint order is in
                         meta/info.json
objects                = ["tall_bar_stool"]
schema_layout          = flat
source_sequence_id     = 2026-06-04_14-16-23_tall_bar_stool_tip_up_04
```

**[VERIFIED]** numerically on episode 0:

- `observation.human.pose` is 231 floats = **77 × 3** axis-angle rotation vectors, radians.
- `observation.object.pose` = `[x, y, z, qw, qx, qy, qz]`, translation in metres, **quaternion w-first**. Quaternion norms on visible frames: min 1.00000, max 1.00000.
- `observation.object.visible`: 778/778 on episode 0; all-zero pose count equals the invisible count exactly (0 == 0). **Filter on visibility before using a pose** — a zero quaternion is not a rotation.
- Timestamps 0.000 → 25.900 s over 778 frames → **30.00 fps** exactly.
- Human translation in metres, e.g. frame 0 `[-0.5878, -0.3123, -3.2727]` (note negative Z: OpenCV world frame, not Z-up).

**Ground plane:** `ground_plane/<sequence_id>.json`, normalized `[a,b,c,d]` for
`ax+by+cz+d=0` in the **same source OpenCV world frame** as the object pose.
NVIDIA warns: *"Do not replace it with a horizontal world-plane estimate when
reproducing a retargeted training reference."* **[DOC]**

**Support surface:** `support_surface/<sequence_id>_support.usda`, **metres, Z-up,
in the released G1 retargeted world** — a *different* frame from the pose data.
Must be placed at
`<motion-root>/whole_body/reconstructed_stage/<sequence_id>_support.usda`.
Ground-only episodes have a valid empty stage. **[DOC]**

**Camera:** one `exo_camera` stream, 1536×1152, static third-person. 
`camera_calibration: null`, with the note *"Not present in the supplied Track 2
source export."* **There are no intrinsics or extrinsics**, so meshes cannot be
projected into the images from this dataset alone. The calibration *sequence id*
is preserved per episode but is not a matrix. **[VERIFIED]** — this materially
constrains any Tier 3 reconstruction approach that needs calibration.

---

## 4. Baseline code NVIDIA explicitly supplies

### 4.1 The official G1 ReconBody experiment recipes — the real baseline

`robotic_grounding/flash_chord/src/flash_chord/configs/experiment/`:

| Config | Learner |
|---|---|
| **`g1_recon_body_ppo.yaml`** | FlashCHORD PPO — *"Canonical G1+Dex3/SONIC ReconBody PPO recipe"* |
| **`g1_recon_body_flash_sac.yaml`** | FlashCHORD flash_sac |

`g1_recon_body_ppo.yaml` composes: `sim=recon_body`, `collision=physical`,
**`embodiment=g1_dex3`**, **`action=sonic_joint_residual`**,
`observation=recon_body`, `objective=recon_body`, `termination=recon_body`,
`reset=recon_body`, `curriculum=recon_body_ppo`. `scene.world_count: 4096`,
`training.max_iterations: 20000`, and a comment that the PPO hyperparameters
*"Match G1SonicRslRlPpoCfg"*. **[VERIFIED]**

Its only required input is `task.parquet` (`task/default.yaml` sets
`parquet: ???`), documented as *"Reference trajectories are data inputs, not
package assets."* That is the **G1 retargeted motion parquet** — see §5.

### 4.2 Submission kit contents

| Path | Role |
|---|---|
| `flash_chord/scripts/evaluate_policy.py` | **the authoritative Track 2 evaluator** |
| `flash_chord/src/flash_chord/configs/evaluate_policy.yaml` + `evaluation/flash_sac.yaml` | its Hydra configs |
| `examples/run_policy_evaluation.py` | batch runner: episode→checkpoint JSON map → per-episode outputs |
| `metric_code/track_2/{AUC,SP-SR,MP-SR,MPPE}.py` | code-only copies of the four scorers (~21 KB each) |
| `tools/pack_submission.py` | builds the submission CSV |
| `tools/preflight.py` | pre-upload validation |
| `submit/v2d_submit.py` | Kaggle uploader |
| `data/track_2_sample_submission.csv` | 11.6 MB row template; exact rows are validated |
| `skills/v2d-submit/SKILL.md` | guided-submission agent skill |
| `tracks/track_2/README.md`, `config/tracks.json` | the Track 2 contract |

**[VERIFIED]** by listing and reading the zip in memory (never extracted).

### 4.3 Also supplied, but NOT the submission path

The Isaac Lab tasks (`SonicG1-v0`, `SonicG1-ReconBody-v0`,
`SonicG1-ReconHand-Stage{1,2,3}-v0`) with `scripts/rsl_rl/train.py` are genuine
NVIDIA baseline code and are useful for development and visualisation — but see
§4.4: their checkpoints are not accepted by the official evaluator.

### 4.4 What the evaluator accepts — a hard constraint

`flash_chord/evaluation/harness.py` identifies the learner *"from its exact typed-config target"*:

```python
_FLASH_SAC_TRAINING_TARGET = "flash_chord.training.flash_sac.config.TrainingConfig"
_PPO_TRAINING_TARGET       = "flash_chord.training.ppo.config.TrainingConfig"

def checkpoint_training_algorithm(config):
    ...
    if target == _FLASH_SAC_TRAINING_TARGET: return "flash_sac"
    if target == _PPO_TRAINING_TARGET:       return "ppo"
    raise ValueError(f"unsupported checkpoint training target {target!r}")
```

**[VERIFIED].** So exactly two checkpoint families are valid, both FlashCHORD:

- `flash_chord/scripts/train_flash_sac.py` → `flash_sac`
- `flash_chord/scripts/train_rl.py` → `ppo` (it imports `flash_chord.training.ppo.config.TrainingConfig`)

An Isaac Lab RSL-RL checkpoint has neither target and raises
`unsupported checkpoint training target`. **FlashCHORD is mandatory for
submission** — this confirms and sharpens
[TRACK2_MISSION.md §7](../reference/TRACK2_MISSION.md#7-known-gaps-and-risks) risk 1.

### 4.5 Evaluator mechanics and rollout format

- Pinned runtime: *"Run it with the public FlashCHORD runtime at this revision"* — commit **`8c778f905cb35f76f8e6cd6e2ea486da4fb9e0ef`**, which our `--depth 1` clone cannot check out. **[DOC]** / **[GAP]**
- The evaluator *"Rebuild[s] the saved experiment from its checkpoint's own resolved config"* — environment, task, embodiment and strategy all come from the checkpoint, not from CLI flags. `validate_policy_schema: true`, `deterministic: true`.
- Invocation per episode (from `run_policy_evaluation.py`):
  `evaluation.checkpoint=… evaluation.world_count=1 evaluation.reset_mode=explicit evaluation.metrics_output=…/episode_NNNNNN.json evaluation.object_trajectories_output=…/episode_NNNNNN.parquet logging.mode=disabled`
- **Rollout output:** one **`episode_XXXXXX.parquet`** of object trajectories per episode; the metrics JSON is optional. `tools/pack_submission.py` turns the directory into one CSV per tier, applying *"one rigid transform between the evaluator's recorded reference and that tier's released input"*.
- **One trained checkpoint per episode** — 30 checkpoints for 30 episodes, per tier. **[DOC]**
- `--world-count` defaults to 1 in the kit runner (the config's own default is 4096).

---

## 5. How our `robotic_grounding` checkout connects to the challenge data

### 5.1 The required chain

```
challenge LeRobot parquet  (SOMA-X human params + object world_T_object)
          │
          │  ❌ NO OFFICIAL ADAPTER  ← the one missing piece
          ▼
soma_to_g1.py  data_folder/{soma_params.npz, poses.npy, reconstructed_mesh/output_aligned.glb}
          │    (+ ground_plane.json, object/ mesh)
          ▼
G1 retargeted motion_v1 parquet          ← this is FlashCHORD's task.parquet
          │
          ▼
flash_chord/scripts/train_rl.py +experiment=g1_recon_body_ppo task.parquet=…
          │
          ▼
checkpoint (target = flash_chord.training.ppo.config.TrainingConfig)
          │
          ▼
flash_chord/scripts/evaluate_policy.py → episode_XXXXXX.parquet
          │
          ▼
tools/pack_submission.py → CSV → submit/v2d_submit.py → 4 Kaggle competitions per tier
```

### 5.2 What `soma_to_g1.py` expects

Positional `data_folder`, documented as *"Path to folder containing
soma_params.npz, poses.npy, and reconstructed_mesh/output_aligned.glb"*, plus
`--motion-root`, `--soma-data-root`, `--soma-subdir`, `--robot-name`, `--fps`,
`--start-frame`, `--end-frame`, `--scale`, `--contact-threshold`,
`--identity-model-type` (default `mhr`), `--save`, `--visualize`. **[VERIFIED]**

It uses `G1_URDF_DIR / "main_with_hand.urdf"` — present, 58,075 B. **[VERIFIED]**

### 5.3 Field mapping — challenge → `soma_to_g1.py`

From the demo sequence's actual `soma_params.npz` (12 arrays) against the
challenge schema:

| `soma_params.npz` key | shape (demo, T=479) | Challenge source | Status |
|---|---|---|---|
| `poses` | `(T, 77, 3)` float32 | `observation.human.pose` (231) → `reshape(T,77,3)` | **exact** |
| `transl` | `(T, 3)` float32 | `observation.human.translation` (3) | **exact** |
| `identity_coeffs` | `(T, 45)` float32 | `observation.human.identity_coeffs` (45) | **exact** |
| `scale_params` | `(T, 68)` float32 | `observation.human.scale_params` (68) | **exact** |
| `joint_names` | `(77,)` `<U18` | `meta/info.json` → `features["observation.human.pose"]["names"]`, collapse `.rx/.ry/.rz` | **derivable** |
| `identity_model_type` | `<U3` = `mhr` | `robot_type: soma_x_mhr` | **derivable** |
| `joint_orient` | `(78, 3, 3)` float32 | — | **[GAP] not in the challenge release** |
| `keep_root`, `absolute_pose`, `rotation_repr`, `unit`, `allow_pickle` | scalars | — | **[ASSUMPTION]** must be set to match the demo's values |
| — | — | `observation.human.bone_length_flexibles` (6) | challenge-only; no npz counterpart |

| Other input | Demo form | Challenge source | Status |
|---|---|---|---|
| `poses.npy` | `(T, 4, 4)` float64 homogeneous | `observation.object.pose` `[x,y,z,qw,qx,qy,qz]` → 4×4 | **constructible** (w-first quaternion → R, translation → t) |
| `reconstructed_mesh/output_aligned.glb` | GLB | `mesh/<object>/<object>.glb` | **direct copy** |
| `ground_plane.json` | `{"plane":[a,b,c,d], …}` | `ground_plane/<sequence_id>.json`, same `ax+by+cz+d=0` OpenCV convention | **direct** (demo file carries extra diagnostic keys) |
| support `.usda` | `whole_body/reconstructed_stage/<seq>_support.usda` | `support_surface/<sequence_id>_support.usda` | **direct copy to the declared path** |

**The decisive compatibility result [VERIFIED]:** the 77 joint names from
`meta/info.json`, with `.rx/.ry/.rz` collapsed, are **identical and in identical
order** to `SOMA_JOINTS_ORDER` in
`source/robotic_grounding/robotic_grounding/retarget/params.py` —
`['Hips','Spine1','Spine2','Chest', …, 'RightFoot','RightToeBase','RightToeEnd']`.
No joint remapping is required.

### 5.4 The gap, stated precisely

**[GAP]** A LeRobot→SOMA adapter must be written. It is roughly 100 lines:
read the episode parquet, reshape the human pose, build 4×4 object transforms
from the w-first quaternions, synthesise the npz scalar metadata, copy the mesh,
ground plane and support surface into the layout `soma_to_g1.py` expects.

Two residual unknowns inside that work:

1. `joint_orient (78,3,3)` is in the demo npz but **not** in the challenge release. Whether `read_soma.SOMA` requires it, can default it, or derives it from the SOMA-X body model (via `scripts/setup_soma_assets.py` / `py-soma-x==0.2.1`) is **not yet determined**. This is the single biggest technical unknown. **[GAP]**
2. The npz scalar flags (`keep_root`, `absolute_pose`, `rotation_repr`, `unit`) must match what the challenge export assumed. The demo values are readable, but that they apply to the challenge data is an **[ASSUMPTION]** until a retarget runs and is visually/numerically checked.

---

## 6. Minimum steps to run one challenge trajectory in Isaac Lab

Taking **episode 0** — `2026-06-04_14-16-23_tall_bar_stool_tip_up_04`, 778
frames, object `tall_bar_stool`, support surface present — as the worked case.

**Prerequisites not yet met** (each blocks the step after it):

| # | Blocker | Status |
|---|---|---|
| A | Container `pyarrow` is unimportable (`pyarrow 26.0.0` vs pinned `numpy 1.26.4`) | **open** — [SETUP_AUDIT.md §20](../reference/SETUP_AUDIT.md) |
| B | The LeRobot→SOMA adapter does not exist | **open** — §5.4 |
| C | SOMA-X body assets not staged (`scripts/setup_soma_assets.py`) | **open** |
| D | FlashCHORD training needs ≥46 GB VRAM; this GPU has 8151 MiB | **hardware** |

**Steps, once A–C are resolved:**

1. Download **one** episode from `tier_1_multiview_caption`: `data/chunk-000/episode_000000.parquet`, `mesh/tall_bar_stool/tall_bar_stool.glb`, `ground_plane/2026-06-04_14-16-23_tall_bar_stool_tip_up_04.json`, `support_surface/2026-06-04_14-16-23_tall_bar_stool_tip_up_04_support.usda`, plus `meta/`.
2. Run the adapter (§5.4) to produce
   `<HMD>/whole_body/soma/sequence_id=2026-06-04_14-16-23_tall_bar_stool_tip_up_04/` containing `soma_params.npz`, `poses.npy`, `reconstructed_mesh/output_aligned.glb`, `object/`, `ground_plane.json`, and place the support `.usda` at `<HMD>/whole_body/reconstructed_stage/<seq>_support.usda`.
3. Retarget inside the container:
   `python scripts/retarget/soma_to_g1.py <that folder> --save --fps 30` → a G1 `motion_v1` parquet at `robot_name=g1/data.parquet`.
4. **Isaac Lab visual/kinematic check** (this is the Isaac Lab part, and the cheapest first real milestone):
   `python scripts/rsl_rl/dummy_agent.py --headless --task SonicG1-ReconBody-v0 --motion_file whole_body/soma/2026-06-04_14-16-23_tall_bar_stool_tip_up_04/g1 --num_envs 1 --max_steps 60`
   — zero actions, proves the challenge trajectory loads and steps. **Not** manipulation.
5. For an actual submission, training and evaluation must run under **FlashCHORD**, not Isaac Lab (§4.4), on a ≥48 GB GPU.

---

## 7. The exact first runnable challenge example

**Honest answer: nothing is runnable today.** Blocker A alone stops any container
run, and blocker B means there is no challenge-derived motion file to run.

The **first runnable challenge example, once A and B are cleared**, is step 4
above:

```bash
# inside the robotic-grounding container, after the adapter has produced the motion file
python scripts/rsl_rl/dummy_agent.py \
  --headless \
  --task SonicG1-ReconBody-v0 \
  --motion_file whole_body/soma/2026-06-04_14-16-23_tall_bar_stool_tip_up_04/g1 \
  --num_envs 1 \
  --max_steps 60
```

Chosen because episode 0 is mid-length (778 frames), has a valid support surface,
100 % object visibility on every frame (verified), and a single rigid object —
so nothing about the episode itself complicates a first load.

That command is an **integration check under zero actions**, exactly as in
[SETUP_AUDIT.md §20.9](../reference/SETUP_AUDIT.md): it would prove the challenge trajectory
parses, the scene builds and physics steps. It is **not** manipulation, **not** a
score, and **not** on the official submission path.

The shortest route to an actual leaderboard number remains: adapter → retarget all
30 → FlashCHORD `g1_recon_body_ppo` training per episode on a ≥48 GB GPU →
`evaluate_policy.py` → `pack_submission.py` → `v2d_submit.py`.

---

## 8. Verified / unverified ledger

**[VERIFIED] by direct observation**

- 2 published tiers; `tier_3_cari4d` absent from HF
- 30 episodes, 22,990 frames, per-episode frame counts, objects, sequence ids, support surfaces
- LeRobot v2.1, `robot_type: soma_x_mhr`, 13-column schema, 30 parquets = 31.2 MB
- `world_T_object`, `[x,y,z,qw,qx,qy,qz]` w-first, quaternion norms = 1.00000
- 231 = 77×3 radian rotation vectors; **77 joint names match `SOMA_JOINTS_ORDER` exactly, in order**
- visibility ↔ zero-fill correspondence; 30.00 fps from timestamps
- `camera_calibration: null`
- Evaluator accepts only the two FlashCHORD typed-config targets
- `g1_recon_body_{ppo,flash_sac}.yaml` exist and compose `embodiment=g1_dex3`, `action=sonic_joint_residual`
- `task.parquet: ???` is required and is a data input
- No upstream file references the challenge dataset or its columns
- `soma_to_g1.py` input contract; `main_with_hand.urdf` present (58,075 B)
- `soma_params.npz` key/shape inventory from the demo sequence

**[DOC] — NVIDIA's statement, not independently checked**

- Tier 2 perturbation is sampled from Track 1 error distributions
- `exo_camera` byte-identical across tiers; frame-exact with parquets across all 30
- Support surfaces are metres/Z-up in `g1_retarget_world`
- The ground-plane warning about not substituting a horizontal estimate
- One checkpoint per episode; the pinned evaluator revision `8c778f9…`
- Metric definitions and thresholds (see [TRACK2_MISSION.md §4](../reference/TRACK2_MISSION.md#4-how-we-are-scored))

**[GAP] / [ASSUMPTION] — missing functionality or unresolved**

- **No LeRobot→SOMA adapter** (the blocking gap)
- `joint_orient (78,3,3)` absent from the challenge release; requirement unknown
- npz scalar flags assumed transferable from the demo
- Tier 3 data not published; whether we must generate it with CARI4D is unconfirmed
- `observation.human.bone_length_flexibles` (6) has no npz counterpart — its role in retargeting is unexamined
- Pinned evaluator commit unreachable from our shallow clone
- Container `pyarrow`/`numpy` conflict unresolved
- FlashCHORD VRAM on this GPU still unmeasured

---

## 9. Provenance

Dataset facts come from the HF API tree listings, `track_2/README.md`, both tier
READMEs, `meta/info.json`, `meta/episodes_metadata.jsonl`, and **one** episode
parquet (`episode_000000.parquet`, 1.06 MB) read with pyarrow in a throwaway
scratchpad venv — the host and container both lack a working pyarrow. No video,
mesh or second episode was downloaded; the dataset was not cloned.

Kit facts come from reading `v2d_submission_kit.zip` **in memory**; it was never
extracted and nothing in it was executed. Code facts come from reading the
upstream checkout at `193382a`, which was not modified. No training, no model
change, no Docker rebuild, no commit, no push. The generic snack-box example was
not run.
