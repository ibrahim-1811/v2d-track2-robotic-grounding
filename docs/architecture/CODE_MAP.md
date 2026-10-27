# Code map: what every file does

A tour of the repo in plain language. For *why* it is shaped this way, read
[ARCHITECTURE.md](ARCHITECTURE.md); for the commands, the [README](../../README.md).

---

## The shape of the repo

```
v2d-track2-robotic-grounding/
├── README.md                 every command, by task
├── CLAUDE.md                 environment traps and measured results (read before running)
├── training_manifest_*.json  per-tier episode facts the trainer reads
├── pipeline/                 the importable library, and its tests
├── scripts/                  things you run, grouped by purpose
│   ├── data/                 download, stage, retarget, validate; Tier 2 recovery
│   ├── train/                manifests, curricula, grip references, metrics
│   ├── evaluate/             submission assembly, floors, AUC decomposition
│   ├── diagnose/             rollout diagnostics, media, viewers
│   ├── tier3/                video -> challenge-format episode
│   └── verify/               one-command health checks
├── cluster/                  Slurm jobs and launchers, container definitions
├── configs/                  generated Hydra overrides (NVIDIA's own stay in official/)
├── docker/                   the pyarrow fix for robotic-grounding:latest (PC)
├── docs/                     architecture/ guides/ reference/ findings/
├── official/                 NVIDIA upstream @ 193382a, READ-ONLY        (gitignored)
├── data/                     inputs, references, weights                  (gitignored)
├── outputs/                  runs, evaluations, media, submissions        (gitignored)
├── containers/               flash_chord.sif, cari4d.sif                  (gitignored)
└── logs/                     Slurm job logs                               (gitignored)
```

---

## Training manifests (repo root)

One row per episode: sequence id, the container path of its G1 reference, `scene_support`
(true only when the support USDA declares collision shapes), frame count, and the stationary-
floor AUC printed next to every evaluation. `train_tier_episode.sbatch` and
`submit_tier_lanes.sh` read them at job start.

| File | Tier / variant | Made by |
|---|---|---|
| `training_manifest_verified.json` | Tier 1 | `scripts/train/make_training_manifest.py` |
| `training_manifest_t1grip.json` | Tier 1, scripted grip (eps 1, 4, 25, 28) | `scripts/train/make_grip_references.py` |
| `training_manifest_tier2.json` | Tier 2, raw input | `scripts/train/make_tier_manifest.py` |
| `training_manifest_tier2c.json` | Tier 2, contact-gated recovery | `scripts/data/recover_tier2_contact.py` |
| `training_manifest_tier3.json` | Tier 3, validated + aligned episodes only | `scripts/tier3/make_tier3_manifest.py` |

---

## `pipeline/`: the library

Imported, not run. Scripts stay thin; the logic lives here so it can be tested.

| File | What it does |
|---|---|
| `episodes.py` | The tier registry: which episodes exist, where their inputs are (published mirror, or our `data/reconstructed/` for Tier 3), and where each tier stages its outputs. |
| `lerobot_to_soma.py` | Converts the released LeRobot format to the SOMA layout the retargeter expects. Copies object poses through unchanged. |
| `validate_staged.py` | Checks that conversion: frame counts, timestamps and object poses bit-identical, coordinate convention untouched. |
| `validate_reference.py` | The 45 reference checks (joint speeds vs URDF, residual vs NVIDIA's demo, teleports, grounding, hands reach the object). |
| `checks.py` | Shared pass/fail helpers so every validator reads the same way. |
| `orientation_reward.py` | Quaternion conventions and the orientation maths in NumPy, the reference every GPU kernel is tested against. |
| `flash_sac_orientation.py` | The wrapping pattern: delegate to NVIDIA's 12-term objective, copy the terms, append one. The keypoint and velocity objectives build on it. |
| `flash_sac_orientation_exp.py` | The exponential-shaped orientation variant (kept for the exporter's registry). |
| `keypoint_reward.py` | NumPy reference for the six-keypoint ADD-style term. |
| `flash_sac_keypoints.py` | **The default objective**: 12 upstream terms + `object_keypoints` (Warp kernel). |
| `flash_sac_velocity.py` | Keypoints + `object_velocity` (linear and angular, vs central differences of the reference). |
| `flash_sac_actor_export.py` | Writes the portable policy: stock config, byte-identical weights, so NVIDIA's evaluator can load a 13/14-term checkpoint. |
| `flash_sac_metrics_logger.py` | Writes losses and per-term rewards to `metrics.jsonl`/`.csv` (upstream logs nothing locally with W&B off). |
| `flash_sac_train_entry.py` | Upstream's training entry point plus the logger above. |

### `pipeline/tests/`

`test_*.py` run anywhere with pyarrow (on the cluster: inside `flash_chord.sif`).
`container_*.py` need Warp/JAX and run inside the container via the verify scripts. They are
deliberately not named `test_*`; `pipeline/tests/README.md` explains why.

---

## `scripts/data/`

| Script | What it does |
|---|---|
| `download_challenge_episodes.py` | Mirrors Tier 1 and 2 from Hugging Face with the standard library (videos optional). |
| `run_tier_pipeline.py` | **The data entry point**: `stage`, `retarget`, `validate`, `report` for any tier, one episode per container run, resumable (`--force` to redo). |
| `retarget_episode_g1.py` | Wraps NVIDIA's `soma_to_g1.py` with chunked body-model evaluation, so it fits in 8 GB. Numerically identical. |
| `validate_g1_reference.py` | One-reference validation with a readable report. |
| `validate_soma_in_container.py` | Confirms SOMA-X loads inside the retarget container. |
| `stage_episode_0_assets.py` | The original single-episode staging (`verify_step_8_4.sh` and `episodes.py` still use it). |
| `recover_tier2_contact.py` | **Rules-clean Tier 2 input**: hold the object unless a palm or foot is near. Writes `data/adapter_out_tier_2c`, the packer's frame references and the manifest. |

## `scripts/train/`

| Script | What it does |
|---|---|
| `make_orientation_curriculum.py` | Generates every file in `configs/` from NVIDIA's YAML, and refuses if upstream's weights or thresholds moved. Re-running must be a no-op. |
| `make_training_manifest.py`, `make_tier_manifest.py` | Build the Tier 1 / per-tier training manifests. |
| `make_grip_references.py` | Scripted Dex3 grip: closes the 12 finger joints during the reference's hand contact. Writes `data/adapter_out_grip`. |
| `show_training_metrics.py` | Tables, per-term breakdowns (`--terms`) and plots from a run's `metrics.jsonl`. |
| `monitor_training_disk.py` | Prunes old `state_*` from one named run; dry run unless `--apply`. |

## `scripts/evaluate/`

| Script | What it does |
|---|---|
| `assemble_submission.py` | Best evaluated run per episode (mean AUC over 5 repeats), copies rep 1 into the packer's input, writes `selection.json`. |
| `leaderboard_floor.py` | The stationary floor: a frozen object scored with the official formulas. |
| `decompose_auc_loss.py` | Splits an episode's AUC loss into position vs rotation, with the first failure frame. |
| `check_unassisted.py` | Proves the evaluator really strips the object assist. |

## `scripts/tier3/`

| Script | What it does |
|---|---|
| `sam3_text_masks.py` | Person and object masks from text prompts (SAM3 video). Picks the object by its expected image size from the mesh; drops one-frame glitches; refuses implausible picks. |
| `cari4d_to_lerobot.py` | CARI4D output to a Tier-1-shaped episode: frame-0 alignment (object, or human for symmetric objects), MHR to SOMA body fit, self-checks. |
| `frame0_alignment.py` | Compares our frame-0 body with Tier 1's frame-0 body (first frame only, as the FAQ allows); `--record` stores the check for the manifest. |
| `make_tier3_manifest.py` | Tier 3 manifest: validated on the cluster, alignment check passed, reference newer than its conversion; `--accept-failed` for coverage-only episodes. |
| `score_cari4d_object.py` | One-off Episode 0 feasibility check. **Never** use it to select or tune anything for Tier 3. |

## `scripts/diagnose/`

| Script | What it does |
|---|---|
| `diagnose_jitter_rollout.py` | Step-to-step joint vibration of reference, SONIC output, PD target and simulated joints, on the evaluator's rollout. |
| `diagnose_contact_rollout.py` | The evaluator's rollout plus palm poses, contacts and forces per frame. |
| `profile_episode_phases.py` | Per episode: how long the object is still, when manipulation starts, how far it travels. |
| `record_policy_viser.py` | Records a rollout to a `.viser` 3D replay. |
| `make_media_index.py` | Builds `outputs/media/index.html` linking every MP4 and 3D replay. |
| `view_policy_viser.sh` | Live viewer on a Docker PC (`http://localhost:8080`); mounts every `data/adapter_out*` tree. |
| `flashchord_headless_replay.py` | Replays a reference with no policy (object-assisted). |
| `read_checkpoint_metadata.py`, `inspect_checkpoint.py` | What a checkpoint was trained on: stdlib header read / NVIDIA's loader. |
| `check_reference_in_flashchord.py` | Loads a reference through FlashCHORD's own reader. |
| `analyze_episode_0.py`, `audit_frames_episode_0.py`, `analyze_manipulation.py` | The original per-episode and per-frame audits. |

## `scripts/verify/`

Each prints its checks and exits non-zero on any failure.

| Script | Confirms | Runs on |
|---|---|---|
| `verify_keypoint_flash_sac.sh` | keypoint objective: configs reproducible, kernels match NumPy | cluster |
| `verify_velocity_flash_sac.sh` | velocity objective, same checks | cluster |
| `verify_step_B.sh` / `verify_step_C.sh` | Episode 0 / all 30 Tier 1 references valid | PC |
| `verify_step_8_4.sh` | environment and early pipeline intact | PC |
| `verify_flashchord_smoke.sh` | NVIDIA's G1 example replays | PC |
| `verify_flashchord_training.sh` | train, checkpoint, evaluate end to end | PC |
| `check_cluster.sh` | what hardware a node has | any |

---

## `cluster/`

| File | What it does |
|---|---|
| `train_tier_episode.sbatch` | One episode: reads the manifest, maps `ARM` to Hydra overrides, runs `train_flash_sac.sbatch`. 6 h limit. |
| `train_flash_sac.sbatch` | The training job itself: binds data trees, Warp cache, our configs; stops at the first NaN metrics line. |
| `submit_tier_lanes.sh` | Queues a tier as `LANES` chains (`afterany`), optional `EVAL=1`, `NICE`, `SEED`, `DRY`. |
| `evaluate_policy.sbatch` | 5 unassisted world-1 evaluations; exports the portable policy first. |
| `submit_tier_evals.sh` | Evaluations for many finished runs. |
| `train_status.sh`, `status.sh` | Progress per run / queue with rates and ETAs. |
| `pack_tier.sh` | Assemble + the kit's unchanged packer to a CSV with local scores. |
| `tier3_masks.sbatch` | SAM3 masks on V100s. |
| `tier3_cari4d_episode.sbatch` | Masks if missing, then CARI4D with the two bind-mounted patches. |
| `tier3_convert.sbatch` | `cari4d_to_lerobot.py` for a list of episodes. |
| `view_policy_viser.sbatch`, `view_episode.sh` | Live viewer on a GPU node, opened through an SSH tunnel. |
| `record_policy_media.sbatch` | MP4 + `.viser` replay per policy into `outputs/media/<set>/`. |
| `diagnose_jitter.sbatch` | Runs the jitter diagnosis for a list of runs. |
| `flash_chord.def`, `cari4d.def` | Apptainer definitions of the two images. |
| `probe_gpu.sbatch`, `plot_losses.py` | GPU probe; loss-curve comparison plot. |

## `configs/`

Generated by `make_orientation_curriculum.py`; never hand-edited. `objective/` points Hydra at
our objective class; `curriculum/` adds the new term's weight to all nine curriculum stages
(`slowvoc` stretches the assist fade; `smooth` is the keypoint curriculum used with the 100x
penalties). `orientation/` holds the first 13-term variant, which the generator still emits.

---

## `outputs/`: where results land

| Path | What it is |
|---|---|
| `cluster/<prefix>_epNN_<arm>/` | One training run: `policy_<step>`, `portable_policy_<step>`, `metrics.jsonl`, `eval_<step>/` (5 repeats + trajectories). Resume states only while training. |
| `submission/<prefix>/` | `selection.json`, the packer's input, `track2_tierN.csv`. `t1` = current Tier 1 pack. |
| `media/` | `index.html`, `t1/`, `t2r/`, ... MP4 + `.viser` replays, `viser_client/`. |
| `tier3/` | SAM3 masks, CARI4D runs, inputs. |
| `tier2c_frame_references/` | Released-frame object tracks for packing Tier 2. |
| `diag/jitter/` | Jitter measurements per run. |
| `kit/v2d_submission_kit/` | NVIDIA's submission kit (packer, metric code). |
| `warp_cache/` (`cluster/warp_cache/`) | Compiled Warp kernels; mounted so jobs skip a ~2 min compile. |
