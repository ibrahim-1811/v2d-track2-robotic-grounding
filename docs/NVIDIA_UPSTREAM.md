# NVIDIA Upstream — `video_to_data`

Repo map, code map, architecture and documentation index for the official Track 2
implementation. Companion document: [TRACK2_MISSION.md](TRACK2_MISSION.md) — what
the challenge actually asks us to deliver.

- **Upstream:** <https://github.com/nvidia-isaac/video_to_data>
- **Local checkout:** `official/video_to_data` (gitignored)
- **Pinned commit:** `193382a184308f6800331f3db829f76dc4294cfe`
- **Commit date / subject:** 2026-10-08 — *Add CoRL 2026 conference label to CHORD page (#202)* (Yan Chang)
- **Clone depth:** 1 (shallow)
- **Git LFS:** 123 of 1057 objects materialised (197.0 MB of 1.63 GB) — the G1 ReconBody baseline set, see §7
- **Recorded:** 2026-10-15 · assets pass 2026-10-15

All paths are relative to `official/video_to_data/` unless stated otherwise.

---

## Contents

1. [Checkout status](#1-checkout-status)
2. [Repo map](#2-repo-map)
3. [Architecture — how the pipeline fits together](#3-architecture--how-the-pipeline-fits-together)
4. [Code map — the five components we were asked to find](#4-code-map--the-five-components-we-were-asked-to-find)
5. [Documentation NVIDIA ships](#5-documentation-nvidia-ships)
6. [LFS state](#6-lfs-state)
7. [Baseline assets prepared — G1 ReconBody snack-box](#7-baseline-assets-prepared--g1-reconbody-snack-box)
8. [Provenance of this document](#8-provenance-of-this-document)

---

## 1. Checkout status

| Property | Value |
|---|---|
| Remote | `https://github.com/nvidia-isaac/video_to_data.git` |
| Branch | `main` |
| HEAD | `193382a184308f6800331f3db829f76dc4294cfe` |
| `git rev-parse --is-shallow-repository` | `true` |
| `git rev-list --count HEAD` | `1` |
| Working tree | **clean** — `git status --porcelain` returns 0 lines |
| LFS objects downloaded | **123** (197.0 MB) — selective, see §7 |
| LFS pointers still unresolved | **934** of 1057 |

Upstream is **unmodified**. Treat it as read-only reference; our own work belongs
outside `official/`.

`official/` is excluded from version control by `.gitignore:28`, verified with
`git check-ignore -v official/video_to_data/README.md`. The 912 MB checkout never
enters our repository, so anyone cloning our repo must re-clone upstream.

### Reproducing this checkout

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 \
  https://github.com/nvidia-isaac/video_to_data.git \
  official/video_to_data
```

`GIT_LFS_SKIP_SMUDGE=1` is required, not optional: `git lfs install` is active
globally on this host (`filter.lfs.smudge` is set), so a plain clone would pull
all 1057 LFS objects.

> **Caveat — `--depth 1` has a cost.** The submission kit pins policy evaluation
> to commit `8c778f905cb35f76f8e6cd6e2ea486da4fb9e0ef`, which this shallow clone
> **cannot check out**. See [TRACK2_MISSION.md §7](TRACK2_MISSION.md#7-known-gaps-and-risks).
> Fixing it needs `git fetch --depth 1 origin <sha>` or a deeper clone.

---

## 2. Repo map

### Disk

```
912M  official/video_to_data
├── 432M  .git
├── 361M  docs
├── 100M  reconstruction
├──  14M  robotic_grounding        ← the Track 2 code
└── 4.6M  video_ingestion_agent
```

The bulk is **not** LFS. It is project-website media committed directly to git:
180 MB `.mp4`, 79 MB `.gif`, 62 MB `.png`, 41 MB `.pdf`. Largest single files are
`docs/chord/chord.pdf` (40 MB), `docs/hydra-0/media/overview.mp4` (35 MB),
`docs/figures/real.gif` (28 MB). A shallow LFS-skipped clone still costs ~900 MB;
only `--filter=blob:none` would shrink it further. Actual code is small — 13.7 MB
of `.py` across the whole tree.

### Top level

| Path | Size | What it is |
|---|---|---|
| `robotic_grounding/` | 14M | **Track 2.** Retargeting, RL, SONIC, FlashCHORD, GR00T |
| `reconstruction/` | 100M | Track 1-side perception modules; Track 2 installs 3 of 37 |
| `docs/` | 361M | Project websites (CHORD, Hydra-0, **v2d_challenge**), figures, dataset guides |
| `video_ingestion_agent/` | 4.6M | Video ingestion agent — not used by Track 2 |
| `tools/`, `tests/`, `run_e2e.sh` | 228K | Cross-repo end-to-end harness |

### `robotic_grounding/`

```
robotic_grounding/
├── source/robotic_grounding/robotic_grounding/   # installable package (Isaac Lab extension)
│   ├── motion_schema/   versioned Parquet interchange format (motion_v1)
│   ├── retarget/        MANO/SOMA → robot IK, dataset registry, support recon
│   ├── planner/         G1 whole-body planning ("motionbricks")
│   ├── assets/          robot/object registries, policies, URDFs, meshes
│   └── tasks/           Isaac Lab envs: v2d, v2d_whole_body, scene_utils
├── scripts/
│   ├── retarget/        22 per-dataset retargeting entry points
│   ├── rsl_rl/          RL train / eval / record / replay
│   └── data_quality_checks/
├── flash_chord/         vendored Newton/Warp/JAX trainer (separate container)
├── groot_finetune/      GR00T VLA post-training (Python 3.10, separate env)
├── workflow/            Dockerfile + run.sh + YAML job specs
├── visualizer/          viser-based dataset viewer
├── tests/               schema, retarget e2e, replay, train e2e
└── docs/                SETUP.md, ARCHITECTURE.md, per-dataset guides
```

### `reconstruction/`

37 `v2d_*` modules (95 MB). Track 2's README asks us to `pip install -e` **three**,
and all three are present:

| Module | Contents |
|---|---|
| `modules/v2d_common` | `datatypes.py`, `broadcast.py`, `hdf5_transcode.py`, `ffv1_sidecar.py` |
| `modules/v2d_docker` | `container.py`, `pyproject.toml` |
| `modules/v2d_task_library_loader` | `docker/`, `lib/`, `tests/` |

The other 34 are Track 1-style perception components (`v2d_foundation_pose`,
`v2d_hamer`, `v2d_sam2`, `v2d_droid_slam`, `v2d_wilor`, `v2d_moge`,
`v2d_depth_anything`, `v2d_grounding_dino`, `v2d_bundlesdf`, **`v2d_cari4d`**,
`v2d_hawor`, `v2d_gsplat_refinement`, …). Not required for Track 2's retargeting
+ RL path — but `v2d_cari4d` is the off-the-shelf reconstruction the challenge
offers for **Tier 3**, so it is not irrelevant to us.

---

## 3. Architecture — how the pipeline fits together

### 3.1 NVIDIA's data flow

Quoted from `robotic_grounding/docs/ARCHITECTURE.md` §2. Two Docker images, one
host orchestrator:

```
raw dataset  +  MANO models
          │
          ▼
╔═══════════════════════════════════════════════╗
║  IMAGE 1 · v2d_task_library_loader            ║
║  stage: load  —  MANO forward-kinematics      ║
╚═══════════════════════════════════════════════╝
          │
          ▼
     <ds>_loaded
  Parquet — per-frame MANO joints + object poses
          │
          ▼
╔═══════════════════════════════════════════════╗
║  IMAGE 2 · robotic-grounding                  ║
║  segment → urdf → processed → support → vis   ║
║  (IK retarget · support surfaces · viz)       ║
╚═══════════════════════════════════════════════╝
          │
          ▼
       outputs
  <ds>_processed              retargeted robot motion
  object_assets/urdfs/<ds>    object URDFs
  reconstructed_stage/*.usda  support surfaces
  <ds>_html/                  viser recording + MP4
          │
          ▼
    Isaac Lab RL training
```

Stage meanings, per NVIDIA: `segment` splits atomic clips (hot3d only), `urdf`
generates object URDFs, `processed` runs IK retargeting to the robot, `support`
reconstructs support surfaces for sim collision geometry, `vis` records viser +
MP4. Both images are driven from the host by **one** orchestrator —
`scripts/run_pipeline_docker.py` — with no need to shell into a container.

### 3.2 The whole-body chain for G1

For the G1/Dex3 target specifically, the chain has an extra planning hop that the
hand-only path does not:

```
MANO hands  ──retarget──▶  Dex3 end-effector poses   (Dex3HandKinematics, IK)
                                    │
SOMA body   ──retarget──▶           │                 (soma_to_g1.py)
                                    ▼
                         planner/motionbricks        (learned motion model)
                                    │
                                    ▼
                        full-body G1 joint trajectory  (motion_v1 Parquet)
                                    │
                                    ▼
                    Isaac Lab  SonicG1-ReconHand-Stage{1,2,3}-v0
```

`motion_schema` (`motion_v1`) is the contract between every stage: the retargeter
and planner write it; training loaders, replay, support reconstruction and the
visualiser read it through one stable API.

### 3.3 The SONIC residual design

This is the central architectural idea of the baseline and worth stating plainly.

SONIC is a **pretrained whole-body controller** shipped as two ONNX graphs
(encoder + decoder) with a quantized token bottleneck between them. RL does
**not** learn joint targets directly. It learns *residuals* layered onto SONIC's
output, which is why the action space is small and training is tractable.

Where the residual enters is the main design knob — upstream implements six
variants:

| Variant | Residual enters |
|---|---|
| `SONICJointResidualAction` | **after** SONIC output, in joint space |
| `SONICHierarchicalResidualAction` | **before** SONIC, perturbing its commanded positions |
| `SONICHierachicalAction` | hierarchical, non-residual |
| `SONICLatentAction` | latent space |
| `SONICLatentResidualAction` | latent space, residual |
| `SONICLatentHandPolicyAction` | latent + separate hand policy |

Observations are split to match: `sonic_tokenizer_observations.py` feeds the
encoder, `sonic_policy_observations.py` feeds the decoder, and
`policy_observations.py` feeds the RL policy itself.

Training is a **three-stage curriculum**, baked into env configs so a run needs
only per-run flags:

1. **Stage 1** — no-collision warm-up
2. **Stage 2** — contact grounding (adds contact rewards + object terminations)
3. **Stage 3** — full-sequence finetune (reverts to trajectory-end timeout)

### 3.4 Two simulators, one contract

The G1/Dex3 + SONIC stack is implemented **twice**, against different physics
engines. They are not interchangeable.

| | Isaac Lab path | FlashCHORD path |
|---|---|---|
| Engine | Isaac Sim / PhysX | Newton / Warp (+ JAX) |
| Robot def | `assets/g1.py` (`ArticulationCfg`) | `embodiments/g1_dex3.py` (`G1Dex3`) |
| SONIC | `mdp/actions/sonic_*.py` (PyTorch + ORT) | `runtime/actions/sonic.py` (Warp kernels + ORT bound-IO) |
| Trainer | `scripts/rsl_rl/train.py` (RSL-RL PPO) | `scripts/train_rl.py`, `train_flash_sac.py` |
| Config | Isaac Lab `@configclass` | Hydra YAML |
| Our GPU | plausible, untested at 8 GiB | **full training impossible** (needs ≥46 GiB) |

The FlashCHORD tests (`tests/embodiments/test_g1_dex3.py`) are the most explicit
written form of the G1/Dex3 contract — joint layout, fixed-link topology,
cylinder→capsule collider conversion, control params vs upstream. Useful as a
spec even where we cannot run the trainer.

> **This matters more than it first appears:** Track 2 *submission* requires
> FlashCHORD's `evaluate_policy.py`, so FlashCHORD is on the critical path even
> though we cannot train with it. See [TRACK2_MISSION.md §5](TRACK2_MISSION.md#5-submission-mechanics).

---

## 4. Code map — the five components we were asked to find

### 4.1 G1 / Dex3 baseline

**Isaac Lab path:**

| File | Role |
|---|---|
| `source/.../assets/g1.py` | G1 `ArticulationCfg` — the Isaac Lab robot definition |
| `source/.../assets/robot_registry.py` | Robot name → config lookup |
| `source/.../assets/joint_order_registry.py` | Canonical joint ordering |
| `source/.../assets/urdfs/g1/main.urdf` | Full G1 URDF (58 KB, plain text) |
| `source/.../assets/urdfs/g1/main_nodex.urdf` | G1 without Dex3 hands (40 KB) |
| `source/.../assets/urdfs/g1/dex3_left.urdf`, `dex3_right.urdf` | Dex3 hands (~11 KB each) |
| `source/.../assets/meshes/g1/collision/hands/` | Collision meshes — **LFS pointers** |
| `source/.../assets/policies/grasp/grasp.py` | `G1GraspPolicy` — distance-based hand closing |

**FlashCHORD path:**

| File | Role |
|---|---|
| `flash_chord/src/flash_chord/embodiments/g1_dex3.py` | `G1Dex3` (483 lines), `@register_embodiment("g1_dex3")`, plus `G1Dex3Config` |
| `flash_chord/tests/embodiments/test_g1_dex3.py` | The contract tests described in §3.4 |

`G1Dex3` exposes `build(builder) -> EmbodimentLayout` and `bind_reference(...)`;
`G1_BODY_JOINT_NAMES` / `G1_DEX3_JOINT_NAMES` are the joint-name contracts SONIC
also imports.

### 4.2 SONIC integration

**Isaac Lab** — `source/.../tasks/v2d_whole_body/mdp/actions/`:

| File | Role |
|---|---|
| `sonic_actions.py` | `SonicPolicy` (ONNX load, `encode`/`decode`/`quantize`) and `SONICActionBase` |
| `sonic_action_cfg.py` | `SONICActionType` enum + `SONICActionCfg`, dispatches `class_type` |
| `sonic_joint_residual_action.py` | residuals **after** SONIC output |
| `sonic_hierarchical_residual_action.py` | residuals **before** SONIC |
| `sonic_hierarchical_action.py` | hierarchical, non-residual |
| `sonic_latent_action.py`, `sonic_latent_residual_action.py`, `sonic_latent_hand_policy_action.py` | latent-space variants |
| `../observations/sonic_tokenizer_observations.py` | encoder inputs |
| `../observations/sonic_policy_observations.py` | decoder inputs |

**FlashCHORD** — `flash_chord/src/flash_chord/runtime/actions/sonic.py` (975
lines), a Warp-kernel reimplementation: `build_sonic_encoder_reference` (described
in-source as "the immutable 1762-float tokenizer input"),
`gather_sonic_encoder_input`, `compute_sonic_current_observation`,
`append_sonic_history`, `pack_sonic_decoder_input`, `process_sonic_joint_targets`,
`reset_sonic_action`, plus `SonicJointResidualAction` and `_OrtBoundSession` for
bound-IO ONNX Runtime inference.

**Checkpoints — all four are LFS pointers (133 bytes each):**

```
source/.../assets/policies/sonic/encoder_batched.onnx
source/.../assets/policies/sonic/decoder_batched.onnx
flash_chord/src/flash_chord/assets/policies/sonic/encoder_batched.onnx
flash_chord/src/flash_chord/assets/policies/sonic/decoder_batched.onnx
```

**Nothing SONIC-based runs until `git lfs pull`.** Licence and model card ship as
plain text alongside (`MODEL_CARD.md`, `LICENSE`, `NOTICE`, `model-card-subcards/`).

### 4.3 Retargeting scripts

`scripts/retarget/` — 22 files. `run_retarget.py` is the common driver.

| Target | Scripts |
|---|---|
| **→ Dex3** | `arctic_to_dex3.py`, `taco_to_dex3.py` |
| **→ G1 whole-body** | `soma_to_g1.py`, `create_soma_task.py`, `process_soma_sequence.sh`, `soma_loader_probe.py` |
| → Sharpa (hand-only) | `arctic_to_sharpa.py`, `dexycb_to_sharpa.py`, `grab_to_sharpa.py`, `h2o_to_sharpa.py`, `hot3d_to_sharpa.py`, `oakink2_to_sharpa.py`, `taco_to_sharpa.py`, `synthbox_to_sharpa.py` |
| → Sharpa from ego video | `ego_recon_to_sharpa.py`, `ego_recon_to_dexmate_sharpa.py`, `mano_to_dexmate_sharpa.py` |
| Support | `run_retarget.py`, `contact_labels.py`, `verify_tips_distance.py`, `vis_retargeted.py`, `_offline_video.py` |

The IK core is in the package, not the scripts:

| File | Role |
|---|---|
| `source/.../retarget/hand_kinematics.py` | `HandKinematics` base, `SharpaHandKinematics`, **`Dex3HandKinematics`** (MANO → Dex3). Pinocchio + `pink` IK with `FrameTask`/`RelativeFrameTask` |
| `source/.../retarget/whole_body_kinematics.py` | whole-body IK |
| `source/.../retarget/retarget_utils.py` | `setup_dex3_kinematics()`, `setup_sharpa_kinematics()`, `run_frame_ik()`, `wrist_pose_from_mano_joint0()` |
| `source/.../retarget/dataset_registry.py` | dataset registry |
| `source/.../retarget/robot_config.py` | loads per-robot config from `retarget/configs/<robot_name>/` |
| `source/.../retarget/support_recon.py`, `ground_alignment.py` | support surfaces, ground alignment |
| `source/.../retarget/__init__.py` | Path constants: `G1_URDF_DIR`, `MESHES_DIR`, `SHARPA_WAVE_XMLS_DIR`, `BODY_MODELS_DIR`, and the `HUMAN_MOTION_DATA_DIR` / `ROBOTIC_GROUNDING_INTERMEDIATE_DIR` env overrides |

`HUMAN_MOTION_DATA_DIR` is the `<HMD>` root the official README asks us to choose;
it defaults to the in-repo `assets/human_motion_data/`.

Whole-body planning: `source/.../planner/` — `g1_planner.py`, `trajectory.py`,
`cli.py`, and `motionbricks/` (including `smoothing.py` with
`_smooth_g1_qpos_wxyz`, `smooth_qpos_global`, `smooth_qpos_at_boundaries`). Its
ONNX models are at `planner/motionbricks/assets/models/{root,pose,decoder}.onnx`.

### 4.4 RL training scripts

`scripts/rsl_rl/` — RSL-RL PPO:

| File | Role |
|---|---|
| `train.py` | **main training entry point** |
| `eval.py` | policy evaluation |
| `cli_args.py` | shared CLI plumbing |
| `dummy_agent.py` | playback/no-learning agent (~10 GB/process, headless only) |
| `record_dataset.py`, `replay_record.py` | dataset recording / replay |
| `export_parallel_rollouts.py` | parallel rollout export |
| `domain_randomization.py` | visual domain randomization |
| `rerender_demo_visuals.py` | re-render recorded demos |
| `gr00t_infer.py` | GR00T inference in-loop |
| `download_from_wandb.py` | fetch checkpoints from W&B |
| `viewer_utils.py` | viewer helpers |
| `misc/run_sonic_policy.py` | run SONIC standalone (573 lines) |

FlashCHORD trainers: `flash_chord/scripts/train_rl.py`, `train_flash_sac.py`,
`evaluate_policy.py`, `view_policy.py`.

Shell wrappers at the `robotic_grounding/` root: `run_retarget_local.sh`,
`run_pipeline_local.sh`, `run_load_local.sh`, `run_example_sequences.sh`, plus
`scripts/train_tpv_wholebody_workflow.sh` and `scripts/launch_sharded_datagen.sh`.

### 4.5 Environment configurations

**19 registered Gym IDs.** The G1/SONIC ones are the Track 2 baseline:

| Gym ID | Env config class |
|---|---|
| `SonicG1-v0` | `G1SonicEnvCfg` |
| `SonicG1-ReconBody-v0` | `G1SonicReconBodyEnvCfg` |
| `SonicG1-ReconHand-v0` | `G1SonicReconHandEnvCfg` |
| `SonicG1-ReconHand-EpisodeTimeout-v0` | `G1SonicReconHandEpisodeTimeoutEnvCfg` |
| `SonicG1-ReconHand-Stage1-v0` | `G1SonicReconHandStage1EnvCfg` |
| `SonicG1-ReconHand-Stage2-v0` | `G1SonicReconHandStage2EnvCfg` |
| `SonicG1-ReconHand-Stage3-v0` | `G1SonicReconHandStage3EnvCfg` |

Registered in `source/.../tasks/v2d_whole_body/config/sonic/g1/__init__.py`, all
with `entry_point="isaaclab.envs:ManagerBasedRLEnv"` plus `env_cfg_entry_point`
and `rsl_rl_cfg_entry_point` kwargs.

Everything is defined in one 881-line file,
`source/.../tasks/v2d_whole_body/config/sonic/g1/g1_sonic_env_cfg.py`:
observation groups (`G1SONICEncoderCfg`, `G1SONICDecoderCfg`, `G1PolicyCfg`,
`G1HandPolicyCfg`, `G1ReconHandPolicyCfg`), actions (`G1SonicActionsCfg`), rewards
(`G1SonicRewardsCfg`, `G1SonicReconBodyRewardsCfg`, `G1SonicReconHandRewardsCfg`),
terminations, events, curricula (`G1SonicReconBodyVOCCurriculumCfg`,
`G1SonicReconHandCurriculumCfg`), and the env classes. The private helper
`_apply_contact_rewards` is what stages 2 and 3 share.

PPO hyperparameters are separate:
`.../config/sonic/g1/agents/rsl_rl_ppo_cfg.py` — `G1SonicRslRlPpoCfg` →
`G1SonicReconHandRslRlPpoCfg` → per-stage subclasses.

Other task families, for orientation: `tasks/v2d/` (Sharpa hand-only,
`Sharpa-V2D-*`), `tasks/v2d_whole_body/config/vega_sharpa/`
(`VegaSharpa-WholeBody-*`), `tasks/scene_utils/`. MDP terms live under
`mdp/{actions,commands,curriculum,events,observations,rewards,terminations}/`,
several with their own READMEs.

### 4.6 FlashCHORD provenance

`flash_chord/` is a **vendored native shallow port**, not a submodule.
`flash_chord/SOURCE.json` records: upstream release commit
`677457016257eb3d5188538627549160bd40bc80`, prepared commit
`25c8573a2ab9978b788cba6e938e12d3c265f4b6`, integration base
`19ae526a2c90841739abc76543a6329f5eb91ecb`, 847 tracked source files, 559 LFS
files, and three named pre-copy compatibility patches.

Layout: `src/flash_chord/{embodiments,envs,runtime,scene,objectives,training,evaluation,lifecycle,data,visualization,utils}`,
Hydra configs under `src/flash_chord/configs/`, and
`workflow/{train_rl,train_flash_sac,evaluate_policy}.yaml`.

### 4.7 GR00T post-training

`groot_finetune/` — deliberately a **separate Python 3.10 environment**; its
dependencies conflict with the 3.11 Isaac Lab container. Key files:
`convert_to_gr00t.py`, `source_policy.py`, `source_rollout.py`,
`replay_export.py`, `recording_contract.py`, `contracts.py`, `task_profile.py` +
`task_profiles/`, `sharpa_dual_hand_three_camera_config.py`,
`vega_sharpa_joint_config.py`, `closed_loop/`, and 10 `test_*.py`.

### 4.8 Container workflow

| File | Role |
|---|---|
| `workflow/Dockerfile` | `FROM nvcr.io/nvidia/isaac-lab:2.3.2` |
| `workflow/run.sh` | the `docker run` driver the official docs point at |
| `workflow/setup_deps.sh` | installs `git-lfs`, `pipx`, then `pipx install pre-commit` |
| `workflow/{retarget,train,train_vega_manip,dev_env}.yaml` | job specs |
| `scripts/run_pipeline_docker.py` | Python pipeline orchestrator (both images) |
| `scripts/run_osmo.py` | OSMO cluster submission |

---

## 5. Documentation NVIDIA ships

Read in roughly this order.

### Start here

| Doc | Size | Why |
|---|---|---|
| `robotic_grounding/README.md` | 37 KB | The main entry point. Prerequisites, credentials, Docker usage, retargeting recipes, RL training, FlashCHORD, data generation, GR00T, visualizer |
| `robotic_grounding/docs/ARCHITECTURE.md` | 18 KB | 11 sections: what the package is, data flow, directory map, source package, scripts, workflows, datasets, RL tasks, tests/CI, "where do I go for X", conventions & gotchas |
| `robotic_grounding/docs/SETUP.md` | 14 KB | Host prerequisites, NGC login, MANO layout, `git lfs pull`, pretrained checkpoints |

### The worked example

| Doc | Why |
|---|---|
| `source/.../tasks/v2d_whole_body/EXAMPLE_SEQUENCES.md` | **The authoritative end-to-end recipe**: retarget (MANO→Dex3) → plan (Dex3 EE→G1 whole-body) → train stages 1/2/3 → local smoke → eval |
| `robotic_grounding/docs/EXAMPLE_SEQUENCES.md` | Shorter variant at docs level |

### Per-dataset setup

`robotic_grounding/docs/{ARCTIC,DEXYCB,GRAB,H2O,HOT3D,TACO}_SETUP.md` — one guide
per dataset, 7–11 KB each, covering download, licence and on-disk layout.

### Component references

| Doc | Covers |
|---|---|
| `source/.../motion_schema/README.md` | the `motion_v1` Parquet contract |
| `source/.../retarget/configs/README.md` | per-robot retarget configuration |
| `source/.../planner/README.md` | G1 planner / motionbricks |
| `source/.../tasks/v2d_whole_body/README.md` | whole-body task family |
| `source/.../tasks/v2d_whole_body/mdp/actions/README.md` | action terms |
| `source/.../tasks/v2d_whole_body/mdp/observations/README.md` | observation groups + shape legend |
| `workflow/data_pipeline.md`, `workflow/README.md` | containerised pipeline stages |
| `visualizer/README.md` | viser gallery server |

### FlashCHORD

`flash_chord/README.md` (13 KB, host prerequisites + the 48 GiB VRAM figure),
`flash_chord/V2D_INTEGRATION.md` (7 KB), `flash_chord/CONVENTIONS.md`,
`flash_chord/docs/{configuration,package_structure}.md`,
`flash_chord/src/flash_chord/{configs,training}/README.md`.

### GR00T

`groot_finetune/README.md`, `groot_finetune/closed_loop/README.md`.

### Model and data cards

SONIC: `source/.../assets/policies/sonic/MODEL_CARD.md` (+ `model-card-subcards/`
for bias, explainability, privacy, safety/security). Planner:
`source/.../planner/MODEL_CARD.md`. E2E example policy:
`source/.../assets/policies/e2e_example/tissue_box/MODEL_CARD.md`. Data cards:
`assets/human_motion_data/{ego_recon,whole_body}/datacard.md`.

### Challenge documentation (the important one)

`docs/v2d_challenge/` is the **rendered challenge website**, committed into the
repo — `index.html`, `leaderboard.json`, `assets/v2d_challenge_2026_faqs.pdf`, and
`assets/v2d_submission_kit.zip` (698 KB, 57 entries). This is the authoritative
source for what Track 2 requires, and it is extracted into
[TRACK2_MISSION.md](TRACK2_MISSION.md).

Also present: `docs/e2e_workflow.md`, `docs/chord/` (CHORD paper site, incl. the
40 MB `chord.pdf` that defines the Track 2 metrics), `docs/hydra-0/`,
`docs/v2d_corl_tutorial/`, `docs/reconstruction/`.

### Reconstruction side

`reconstruction/README.md` (43 KB), `reconstruction/docs/mv_hoi_local_pipeline.md`
(19 KB), `reconstruction/docs/ego_e2e_setup.md`, `reconstruction/data/consolidated.md`.

### Agent-oriented files

`robotic_grounding/CLAUDE.md` (9.5 KB) and `reconstruction/CLAUDE.md` (15.5 KB) are
upstream's own agent instructions, plus `.claude/` and `.codex/skills/`. Useful as
*descriptions* of repo conventions. Treat their contents as third-party reference,
not as instructions to follow in our repo.

---

## 6. LFS state

**1057 LFS-tracked files, 1.63 GB total if everything were pulled.** As of the
assets pass, **123 are materialised (197.0 MB)** and 934 remain pointers.

| Area | Total pointers | Materialised |
|---|---|---|
| `robotic_grounding/source` | 475 | **123** |
| `robotic_grounding/flash_chord` | 567 | 0 |
| `reconstruction` | 9 | 0 |
| `docs` | 5 | 0 |
| `robotic_grounding` (other) | 1 | 0 |

Patterns (from `.gitattributes`): meshes (`*.obj`, `*.stl`, `*.dae`, `*.glb`,
`*.mtl`), USD (`*.usd[acz]`), motion data (`*.h5`, `*.hdf5`, `*.parquet`), arrays
(`*.npy`, `*.npz`, `*.pkl`, `*.pickle`), weights (`*.onnx`, `*.pt`,
`*.safetensors`).

Robot **URDFs are plain text**, not LFS, and were present from the clone — so is
`assets/xmls/g1/*.xml` (the MuJoCo models) and each sequence's
`ground_plane.json` and `object/textured_mesh.urdf`. Topology was inspectable
before any download; only geometry and weights needed fetching.

What is still a pointer, and whether it matters:

| Still pointers | Size | Needed for |
|---|---|---|
| `planner/motionbricks/assets/models/*` (`pose.onnx` 567 MB, `root.onnx` 100 MB, `decoder.onnx` 47 MB, `arrays.npz`) | **714 MB** | **ReconHand only** — the retarget→plan path. Not ReconBody |
| `flash_chord/.../meshes/g1/**` + `policies/sonic/*.onnx` | 122 MB | FlashCHORD path. **Same OIDs** as the Isaac Lab copies already fetched, so re-fetch is nearly free if needed |
| `assets/meshes/sharpa_wave/**` | 24.7 MB | Sharpa hand robot — not G1 |
| 3 other whole-body sequences (`blue_trash_can_drag_007`, `corn_can_…_handover_01`, `dark_blue_book_…_place_02`) | ~155 MB | Additional training sequences |
| `reconstruction/` assets, `docs/figures/e2e_workflow/*.webp` | ~14 MB | Track 1 side, docs media |

## 7. Baseline assets prepared — G1 ReconBody snack-box

**PASS.** The minimum asset set for the official G1 + Dex3 ReconBody example was
identified, sized, selectively downloaded and content-verified on 2026-10-15.

The target is the command NVIDIA documents in `robotic_grounding/README.md` (L441)
and `tasks/v2d_whole_body/README.md`:

```bash
python scripts/rsl_rl/train.py --headless \
  --task SonicG1-ReconBody-v0 \
  --motion_file whole_body/soma/2026-03-06_10-24-18_snack_box_pick_and_place_01/g1 \
  --num_envs 256 --max_iterations 8 --zero-actor \
  --logger tensorboard --run_name recon_body_snack_box_smoke
```

### 7.1 What was downloaded

All paths below are under
`robotic_grounding/source/robotic_grounding/robotic_grounding/assets/`.

| Group | Files | Size | Contents |
|---|---|---|---|
| `meshes/g1/**` | 113 | **54.5 MB** | 67 body `*.STL` referenced by `urdfs/g1/main.urdf`, plus `collision/hands/*.{obj,mtl}` for the Dex3 hands (`package://g1` in `dex3_{left,right}.urdf`) |
| `policies/sonic/*.onnx` | 2 | **91.0 MB** | `encoder_batched.onnx` (50.1 MB), `decoder_batched.onnx` (40.9 MB) |
| `human_motion_data/whole_body/soma/sequence_id=2026-03-06_10-24-18_snack_box_pick_and_place_01/**` | 7 | **51.4 MB** | motion Parquet, object geometry, reconstructed mesh, SOMA params, poses |
| `human_motion_data/whole_body/reconstructed_stage/…_support.usda` | 1 | 569 B | support surface |
| **Total** | **123** | **197.0 MB** | |

Per-file detail for the sequence:

| File | Size | Kind |
|---|---|---|
| `…/robot_name=g1/data.parquet` | 3.69 MB | **the motion file** — `motion_v1` reference trajectory |
| `…/object/textured_mesh.obj` | 22.1 MB | object geometry — 293,289 verts / 97,862 faces |
| `…/object/material_0.png` | 9.04 MB | texture, 2996 × 3064 |
| `…/object/material.mtl` | 198 B | material |
| `…/object/textured_mesh.urdf` | 572 B | *(plain text, already present)* |
| `…/reconstructed_mesh/output_aligned.glb` | 16.1 MB | aligned reconstruction, glTF v2 |
| `…/soma_params.npz` | 676 KB | SOMA body params — 12 arrays |
| `…/poses.npy` | 61.4 KB | `float64`, shape `(479, …)` → **479 frames** |
| `…/ground_plane.json` | 431 B | *(plain text, already present)* |
| `reconstructed_stage/…_support.usda` | 569 B | `#usda 1.0` support surface |

### 7.2 Download method

A bare `git lfs pull` was never run. One selective pull with a **comma-separated**
include list:

```bash
A=robotic_grounding/source/robotic_grounding/robotic_grounding/assets
SEQ=sequence_id=2026-03-06_10-24-18_snack_box_pick_and_place_01
git lfs pull --include="\
$A/meshes/g1/**,\
$A/policies/sonic/*.onnx,\
$A/human_motion_data/whole_body/soma/$SEQ/**,\
$A/human_motion_data/whole_body/reconstructed_stage/2026-03-06_10-24-18_snack_box_pick_and_place_01_support.usda"
```

> **Gotcha worth recording:** git-lfs takes **one** `--include` with
> comma-separated patterns. Repeating the flag does **not** OR the patterns — the
> last occurrence wins. A first attempt using four separate `--include` flags
> silently fetched only 1 object instead of 123, and exited 0.

### 7.3 Verification — real assets, not pointers

`git lfs ls-files` now marks all 8 snack-box files and all 113 G1 meshes `*`
(content) rather than `-` (pointer). Materialised totals: **123 files, 197.0 MB**,
exactly the estimate. Beyond the marker, each asset class was structurally
validated:

| Asset | Check | Result |
|---|---|---|
| `data.parquet` | `PAR1` magic at head **and** footer | PASS |
| `encoder/decoder_batched.onnx` | protobuf field 0, graph op names present | PASS (50.1 / 40.9 MB) |
| `poses.npy` | `\x93NUMPY` magic + header parse | PASS — `<f8`, shape `(479, …)` |
| `soma_params.npz` | zip integrity (`testzip`) + member list | PASS — 12 arrays |
| `output_aligned.glb` | glTF magic + **declared length == actual file size** | PASS — v2, 16,079,288 B |
| `material_0.png` | PNG signature + IHDR dimensions | PASS — 2996 × 3064 |
| `textured_mesh.obj` | vertex / face line counts | PASS — 293,289 / 97,862 |
| `support.usda` | `#usda 1.0` header | PASS |
| `pelvis.STL` | binary STL arithmetic: `84 + n*50 == filesize` | PASS — 21,216 tris, 1,060,884 B |

The GLB declared-length and STL triangle-count identities are the strongest of
these: both would fail on a truncated or pointer-substituted file.

### 7.4 Deliberately not downloaded

| Excluded | Size | Why |
|---|---|---|
| `planner/motionbricks/assets/models/**` | **714 MB** | The planner is the **ReconHand** retarget→plan stage. `g1_sonic_env_cfg.py` references it only in comments for `G1SonicReconHand*`; ReconBody trains directly from the shipped `data.parquet`. Excluding it is what keeps this pass at 197 MB instead of ~911 MB |
| `flash_chord/.../meshes/g1/**`, `policies/sonic/*.onnx` | 122 MB | FlashCHORD duplicates of assets already fetched on the Isaac Lab side — **identical OIDs** |
| `meshes/sharpa_wave/**` | 24.7 MB | Sharpa hand, not G1 |
| 3 other whole-body sequences | ~155 MB | Not the documented snack-box example |

### 7.5 Missing dependencies — what still blocks a run

The **assets** are ready. These are not:

| # | Missing | Blocks |
|---|---|---|
| 1 | The `robotic-grounding` Docker image (`FROM nvcr.io/nvidia/isaac-lab:2.3.2`, 8.41 GB compressed) | Everything — nothing runs on the host. Registry access is confirmed and needs no login ([REQUIREMENTS.md](REQUIREMENTS.md) 3.1) |
| 2 | Isaac Lab / Isaac Sim runtime | `train.py` imports `isaaclab`; supplied by the image |
| 3 | Host orchestrator venv + `pip install -e` of `v2d_common`, `v2d_docker`, `v2d_task_library_loader` | [REQUIREMENTS.md](REQUIREMENTS.md) 5.3 |
| 4 | `pipx` + `pre-commit` (`workflow/setup_deps.sh`) | 6.5, 6.6 |
| 5 | MANO hand models (licence-gated) | Retargeting *new* sequences. **Not** needed to train on the shipped `data.parquet` |
| 6 | VRAM headroom unmeasured at `--num_envs 256` | 8151 MiB total, ~7.1–7.5 GiB usable. The README's own smoke command uses 256 envs; full training uses 4096 |
| 7 | The planner models (714 MB) | Only if we move from ReconBody to the **ReconHand** three-stage curriculum |

So the honest status is: **assets PASS; the example still cannot be executed**,
because the container image is not built. Nothing about the asset set is the
obstacle.

### 7.6 Disk

```
before this pass   121 GB free   (checkout 912 MB)
after              120 GB free   (checkout 1.3 GB, .git 617 MB, worktree assets 202 MB)
```

Against item 1 above — an 8.41 GB compressed base image, larger uncompressed —
120 GB is comfortable for the image plus this asset set, and remains the figure to
watch once the challenge dataset lands.

---

## 8. Provenance of this document

Sections 1–6 were written from the pinned commit by read-only inspection: `git`
metadata queries, `find`/`du`/`ls`, `grep` for Gym IDs and config entry points,
tree-sitter structural search (AST symbol extraction, no full-file reads), reading
small text files (`__init__.py` registrations, `SOURCE.json`, `.gitattributes`,
`ARCHITECTURE.md`), and HTML-to-text extraction of the committed challenge site.
The submission kit zip was **listed and read in memory**, not extracted to disk.

Section 7 records the one pass that *did* write to disk: a selective
`git lfs pull` fetching 123 LFS objects (197.0 MB) into the upstream checkout.
Sizes come from `git lfs ls-files --size`; content checks are magic-byte and
structural parses, not estimates. **No upstream source file was modified** —
`git status --porcelain` in the checkout returns 0 lines and HEAD is still
`193382a184308f6800331f3db829f76dc4294cfe`. LFS smudge replaces pointer *contents*
with asset data under the LFS filter, which is why the tree stays clean.

Across all passes: no Docker image was built, no Isaac Lab installed, no challenge
dataset downloaded, no simulation or training started, and nothing was committed
or pushed. Line counts and symbol names come from the parser. Quoted figures are
NVIDIA's own wording where marked.
