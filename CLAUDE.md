# Working notes for this repository

NVIDIA V2D Challenge 2026 — Track 2 Robotic Grounding. These are hard-won
environment facts; each line below was learned by something breaking. Read before
running anything.

## Which Python

Host Python 3.12 has **no `pyarrow`, `pandas` or `huggingface_hub`**. Use the
project venv for anything that touches the challenge Parquet:

```bash
.venvs/episode0/bin/python          # pyarrow 15.0.2 + system numpy/scipy/matplotlib/pytest
# recreate:
python3 -m venv --system-site-packages .venvs/episode0
.venvs/episode0/bin/pip install pyarrow==15.0.2
```

- **`python -I` strips the repo root from `sys.path`.** `import pipeline` fails
  under it. Use `-I` only for standalone `scripts/` that import nothing local;
  never for `pipeline/` code or pytest.
- **scipy is 1.11.4 — `Rotation.from_quat(..., scalar_first=True)` does not exist**
  (added in 1.14). Reorder by hand: `q_xyzw = q_wxyz[[1, 2, 3, 0]]`.

## The retarget container

`robotic-grounding:latest` is the only place `soma`, `robotic_grounding`, `torch`
and Isaac Sim exist.

- **Run as root (the default). Never pass `docker --user`** — it breaks the image's
  `python` wrapper (`Unable to find any Python executable at .../python.sh`), and
  the real interpreter then fails with `Permission denied` or a missing
  `libpython3.11.so.1.0`. To keep host ownership, make
  `chown -R 1000:1000 <mount>` the last command inside the same `docker run`.
- **Container Python is 3.11** — nested same-quote f-strings (`f"{d["k"]}"`) are a
  SyntaxError. Use `.format()`.
- **Never inline non-trivial Python in `bash -lc "python -c ..."`.** Quoting passes
  through heredoc → bash → bash -lc → python and mangles. Write a script,
  bind-mount it read-only, run it by path — see
  `scripts/data/validate_soma_in_container.py`.
- `HOME=/tmp` is tmpfs inside the image, so its HF cache is ephemeral. Bind-mount a
  host directory for anything that must survive.

Paths inside the image: checkout `/workspace/video_to_data/`; setup script
`/workspace/video_to_data/robotic_grounding/scripts/setup_soma_assets.py`.

## The coordinate rule — never break this

`soma_to_g1.py:308` applies `_R_CV_TO_SOMA = diag(1,-1,-1)` to the object poses
**and** the ground plane itself. Our adapter passes object poses, ground plane and
human data through **verbatim**.

The matrix is its own inverse, so double-applying it silently restores the broken
frame — the human ends up 5–7 m from the object they are holding — and it presents
as a dataset bug, not an adapter bug. `test_cv_to_soma_was_NOT_applied` guards it.

Human data is already in the SOMA frame and is never converted by anyone.

## Layout

```
data/v2d_challenge/...      PRISTINE Hugging Face mirror — never edit
data/adapter_out/whole_body/
  soma/<seq>/               soma_to_g1.py's positional data_folder
  reconstructed_stage/      support USDA (registry.py:127, for FlashCHORD)
data/body_models/soma/      SOMA-X bundle — pass as --soma-data-root
official/                   NVIDIA upstream @ 193382a — READ ONLY, never modify
pipeline/                   our adapter + tests
scripts/                    analysis, staging and verification scripts
```

`data/`, `outputs/`, `.venvs/` and `official/` are gitignored, so generated
artifacts never appear in `git status`. That is by design.

**SOMA-X is not licence-gated** — `nvidia/SOMA-X` is public, Apache-2.0,
`"gated": false`, no token. Stage it at pinned revision
`466879a83d57eabf3d875ded2d869f2075f90348`; later revisions ship a *slim*
`SOMA_neutral.npz` with the rig stripped out, which passes a presence check and
then fails inside `SOMALayer`.

## Conventions

- Finish every development step with copy-pasteable verification commands — the
  maintainer is learning the project and wants to confirm results independently.
  Prefer a `scripts/verify_step_N.sh` plus the commands inline.
- Mark conclusions **[VERIFIED]** / **[DOC]** / **[GAP]** / **[UNRESOLVED]** in
  `docs/`, matching the existing house style.
- Never interpolate or invent a measurement. Invalid frames
  (`observation.object.visible == False`) are zero-filled by NVIDIA — fail loudly
  or pass the zeros through, never fill them in.

## The FlashCHORD container

`flash_chord:latest` / `flash_chord-latest-gpu0` is a *different* image from
`robotic-grounding:latest` and has none of its quirks. Mounts: `/workspace` →
`official/.../flash_chord` (rw, so **never write there**), `/v2d/assets` →
`.../source/robotic_grounding/robotic_grounding/assets` (ro).

- **`docker/run.sh exec` is unusable non-interactively.** It hardcodes
  `docker exec -it`, and `-i` without a TTY fails with `cannot attach stdin to a
  TTY-enabled container`. Use `docker exec -w /workspace flash_chord-latest-gpu0 …`.
- **The parquet path's `=` characters need the README's inner single quotes.**
  `"task.parquet='/v2d/.../sequence_id=…/robot_name=g1/data.parquet'"`. Without
  them Hydra's override grammar dies on `mismatched input '=' expecting <EOF>`.
- **Headless replay needs no browser:** `viewer.backend=none viewer.num_frames=N
  start_paused=false` runs N physics frames through the official script.
  `start_paused=true` *requires* Viser — it raises otherwise, because the two
  playback modes are Viser button callbacks.
- First run compiles Warp/MuJoCo kernels for ~111 s; after that the cache makes
  it ~1 s. Replay at `world_count=1` peaks at **~1.8 GB VRAM**, so it fits 8 GB
  easily. Training does not: the README wants 48 GiB.
- **`ReplayRunner.reset(f)` does not gather command targets — only `step(f)`
  does.** Calling `check_termination()` after a bare `reset()` compares frame `f`
  against the *last gathered* frame's object target and reports the object's
  travel distance as a tracking error. Follow `reset()` with
  `runner.command.gather(runner.timestep)`.

**No FlashCHORD checkpoint exists at `193382a`** — zero `.safetensors` on disk
and zero in the 1057-file LFS manifest. `evaluate_policy.py` rebuilds the whole
experiment from config embedded in a safetensors header, so nothing can stand in
for one. Any baseline MPPE number requires training first. See
`docs/findings/FLASHCHORD_BASELINE_SMOKE.md`.

Replay is **object-assisted** (`voc_scale: 1.0` — external PD wrench *plus
gravity compensation* on the object every substep). Evaluation forces
`voc_scale=0.0` and zero force/torque limits. Never present an assisted replay as
policy success.

## FlashCHORD PPO training on 8 GB

- **`scripts/train_rl.py` (PPO) is the only entrypoint that does not set
  `XLA_PYTHON_CLIENT_PREALLOCATE=false`** — `train_flash_sac.py`,
  `evaluate_policy.py` and `view_policy.py` all do. JAX then reserves ~75 % of
  the device and Warp's CUDA graph capture OOMs. Pass it yourself:
  `docker exec -e XLA_PYTHON_CLIENT_PREALLOCATE=false …`. Needed above ~512
  worlds; 32 worlds works either way.
- **Measured world-count ceiling on 8 GB:** 1024 → 5,437 MiB and peak throughput
  (899 env steps/s); 1536 → 7,517 MiB but throughput *collapses* to 311; 2048 →
  OOM in `wp_cuda_graph_create_exec`. **Use 1024.**
- A 2-iteration smoke test takes 45 s and 1,966 MiB. `model_<iter>.safetensors`
  is written when `(iter+1) % save_interval == 0` **or** `iter+1 ==
  max_iterations`, so a short run always saves a terminal checkpoint.
- Curriculum stage 0 (iterations < 2000) trains with `voc_scale: 1.0`.
  Evaluation always forces it to 0.0 — see the replay note above.
- **VRAM is not the binding constraint; wall-clock is.** The PPO recipe's
  terminal budget is 20000 × 4096 × 24 = 1.97e9 env steps ≈ **25 days** at our
  best measured rate. Host RAM offloading does not help: the OOM is in CUDA
  graph creation, Warp exposes no offload knob, and this GPU reports
  `is_gpu_memory_access_from_cpu_supported: False`.
- Write checkpoints to `outputs/…` — NVIDIA's own `.gitignore` excludes
  `outputs/` and `logs/`, so upstream `git status` stays clean. Never write
  anywhere else under `official/`.
- The evaluator's JSON report nests metrics: `report["metrics"]["mppe_cm"]`, not
  the `evaluation/`-prefixed stdout keys. Re-runs drift ~0.1 %.

## SOMA -> G1 retargeting (Step B)

- **`soma_to_g1.py` OOMs on 8 GB for any sequence length, and
  `--start-frame`/`--end-frame` do NOT help** — it slices at line 516, long after
  the full-sequence load at line 360. The cause is `read_soma._reconstruct`
  broadcasting the frame-invariant identity to `(T, K)`, so `SOMALayer.forward`
  runs `prepare_identity` over T identical identities (its own docstring says to
  call it once per identity). Fails in `align_vectors_warp`:
  `Failed to allocate 283254240 bytes`.
- **Fix: `scripts/data/retarget_episode_g1.py`** — chunks `SOMA._reconstruct` over
  frame blocks and delegates to `soma_to_g1.main()` unmodified. Numerically
  equivalent (no cross-frame coupling; frame-0 transform built from the full
  arrays before chunking). 778 frames at `--chunk-frames 64`: **79 s, 3,411 MiB
  VRAM** (flat in sequence length).
- **Mount `data/adapter_out` at `/work/adapter_out` in *both* containers.**
  `soma_to_g1.py` bakes absolute paths into `object_mesh_paths`, and
  `registry.resolve_asset_path` only resolves them via its "path is a file"
  branch (our paths have no `assets/` marker, unlike NVIDIA's relative ones).
- **`scene.support=false` is required for this episode.** Its
  reconstructed_stage USDA is a 125-byte stub with no Cylinder/Cube shapes, and
  FlashCHORD raises on that. This is NVIDIA's own convention — 2 of their 3
  whole_body support USDAs are the same empty stub. Rule: enable support only
  when the USDA declares collision shapes.
- **`--diagnose-ik` flags 778/778 frames, and that is normal.** The threshold is
  0.05 m on the *sum* of all whole-body task errors. Our median residual is
  **1.07x NVIDIA's own shipped demo** (0.693 vs 0.645) and our max is lower.
  `torso_link` dominates every frame as a stable G1-vs-human morphology offset.
  Compare against the demo's `ik_error_per_frame` before chasing it.
- `ReplayConfig()`'s defaults are Sharpa's floating-wrist action and reject G1
  (`requires a 3-DOF position and 3-DOF orientation wrist`). G1 needs
  `SonicJointResidualActionConfig`.

## Challenge dataset and submission packing

- **Full mirror is local**: both available tiers, 30 episodes each
  (`tier_1_multiview_caption`, `tier_2_synthetic_noise`; 30 parquets + 30 ground
  planes + 30 support USDAs + 29 meshes per tier, ~390 MB each). Re-fetch or
  extend with `scripts/data/download_challenge_episodes.py` (stdlib urllib; there is
  no `huggingface_hub` or HF CLI anywhere on this box). Videos are skipped by
  default — they are only needed to build a Tier 3 reconstruction.
- **Tier 3 does not exist as a download.** Only tiers 1 and 2 are on Hugging
  Face (re-confirmed 2026-10-20). Tier 3 is "your own reconstruction" — you
  generate it and pass `--frame-references` to the packer.
- **16 of 30 episodes have a support USDA with ZERO collision shapes** and need
  `scene.support=false`; the other 14 have real shapes. Decide this per episode
  by counting `def Cylinder`/`def Cube` in the tier's
  `support_surface/<seq>_support.usda`, or FlashCHORD raises.
- **A valid Track 2 submission is one per tier, with policies trained on that
  tier's own inputs** — so the real scope is 30 episodes x 3 tiers = 90 policies,
  tiers weighted equally.
- **The kit's `evaluate_policy.py` is byte-identical to our `193382a` copy**, so
  the kit's pinned `8c778f9` is not a conflict. The kit also ships a `v2d-submit`
  agent skill and `--source-root` to remap the parquet path baked into a
  checkpoint.
- **The packer consumes the trajectory parquet, not the metrics JSON.** Pass
  `evaluation.object_trajectories_output=.../episode_XXXXXX.parquet`. Our
  exported trajectory (778 rows, achieved + reference pose per frame) is
  **accepted by `tools/pack_submission.py`**; the only remaining blocker is
  episode coverage. The packer also prints the four local scores, so iterate
  locally instead of spending the 5-per-week Kaggle budget.
- The kit needs `numpy>=2`, `pandas`, `pyarrow` — incompatible with this box's
  system numpy 1.26, so use the **fully isolated** `.venvs/kit` (no
  `--system-site-packages`), not `.venvs/episode0`.

## Tier-wide G1 references (Step C)

- **`ik_num_iterations` hitting 200 is NOT a failure for G1, and must never be
  gated.** `configs/g1/retargeter.json` sets `posture_task.q_prev_cost = 0.05`,
  so `has_extra_tasks` is True on every frame, which makes the plateau exit at
  `whole_body_kinematics.py:564` unreachable and leaves `vel_norm < 1e-6`
  (line 566) as the solve's only early exit. The cap therefore means "the QP did
  not reach a 1e-6 velocity norm" — asymptotic, not divergent. Gating it failed
  **17 of 30** Tier 1 episodes on that one check alone. Measured: residual on
  capped frames is 1.07x the same episode's uncapped median (Episode 11: 0.91x,
  i.e. better), the iteration/residual relation is a smooth continuum
  (199 iters → 1.08, 200 → 1.13), and NVIDIA's demo is not comparable because
  its median 8 / max 34 iterations came from the plateau exit. See
  `docs/findings/TIER_REFERENCES.md` §6.4.
- **A solve that genuinely gives up says so on stdout**:
  `[whole_body_kinematics] giving up on frame after N Pink/eigenpy rebuilds`.
  Grep the retarget logs for it; zero Tier 1 episodes have it. That, not the
  iteration count, is the real early-exit detector.
- **Gate the reference, not the solver's effort.** What is gated:
  residual median vs NVIDIA's demo (< 3x; worst episode 1.81x), a per-episode
  residual **spike** test (worst frame vs that episode's own median, < 3x; worst
  1.63x, NVIDIA's own demo 1.65x), and **per-joint velocity against
  `assets/urdfs/g1/main_with_hand.urdf`** (worst 0.57x of the limit, so every
  reference is trackable).
- **G1's 14 hand joints are undriven** by `soma_to_g1` — constant across the
  sequence here *and* in NVIDIA's demo. Exclude them from any joint-space
  measurement or you will report the rig's rest pose (they sit on a position
  limit for 100% of frames).
- `waist_pitch_joint` resting on its position limit (median 33.7% of frames) is
  the known G1-vs-human torso offset, the same one `torso_link` shows in the IK
  residuals. NVIDIA's demo does it too (11.4%). Report it; do not gate it.
- **`Checker.summary()` prints `ALL CHECKS PASSED` per episode.** In a 30-episode
  batch that banner appears once per passing episode, so the log can end with it
  immediately above `=== validate: 13/30 ===`. Pass `summary=False` from batch
  callers; the single-file `scripts/data/validate_g1_reference.py` keeps it because
  `verify_step_B.sh` runs that path.
- **`load_tier()` does not probe object visibility** — `visible_object_frames`
  is `-1` and `dropped_frames` is empty until `annotate_object_validity` runs.
  Use `tier_specs()` (which probes via pyarrow), or Episode 11 will look like
  901 frames instead of 900.
- Episode 11 is the only episode with a dropped frame: index 900 of 901 has
  `observation.object.visible == False`, so the retarget window stops at 900.
  It is recorded in the manifest, the `validate` record's metrics and notes, and
  the report.

## FlashSAC on 8 GB (Step D)

- **The replay ring is the binding constraint, not the world count.**
  `replay.create_replay` allocates it with `jnp.zeros`, so it lives in VRAM.
  Measured for `g1_recon_body_flash_sac` on our G1 references: obs 486, action
  43, critic_context 10, objective terms 12 -> **2,246 B/transition** at
  float16. The shipped `capacity: 10000000` is therefore **20.92 GiB — 2.6x the
  whole card**. Workable ceiling is **1,000,000** (2.09 GiB). Use
  `scripts/train/probe_flash_sac_memory.py` to recompute for any episode.
- **Measured world-count ceiling (replay 500k): 1024 worlds, 6,826 MiB.** 2048
  OOMs in `wp_cuda_graph_create_exec` — the same CUDA-graph failure as PPO, so
  host RAM cannot buy it back. 256 -> 1024 worlds costs only 2,580 MiB, which is
  why the replay, not the worlds, is what does not fit.
- **The progress bar's `env steps/s` is a cumulative average** (total/elapsed,
  including startup), so it climbs monotonically and never plateaus. Difference
  the log for the real rate: **~3,000 env steps/s at 256 worlds**, vs PPO's best
  of 899 at 1024. Full recipe = 350M steps = **32 h/episode** at that rate.
- **Bind-mount the Warp kernel cache.** `/root/.cache/warp` is inside the image,
  so every `docker run` recompiles (~130 s; one module alone took 25.8 s). Mount
  a host dir there and modules load `(cached)` in ~200 ms. The CPU saturation
  during that phase is nvcc, not training — the GPU is idle until
  `compiled FlashSAC JAX executables` appears.
- **`updates_per_collection` is a replay ratio, not a constant.**
  `total_optimizer_updates = (total_environment_steps // world_count) *
  updates_per_collection`. Keeping 4.0 while cutting 2048 -> 256 worlds gives
  **8x** the gradient updates per env step. At 1024 worlds, `2.0` matches
  NVIDIA's reference ratio.
- **`resume` is not transfer, and the guard does not guard.**
  `restore_learner_checkpoint` carries `environment_steps` (hence the curriculum
  stage and the consumed budget), Adam state, the reward normalizer's moments
  and exploration state. Only the replay is safe — it loads solely via an
  explicit `checkpoint.load_replay_path`. `validate_resume_reference_config`
  compares **only** the object-articulation contract, which is `[]` for every
  rigid-object episode, so it will happily resume Episode 0's state onto
  Episode 21. The baked `task.parquet` is recorded but never compared.
- **There is no weight-only warm start upstream.** `training.checkpoint_path`
  goes through `_validate_metadata(path, "training_state")` and rejects the
  4.6 MiB `policy_*.safetensors` actor file outright. A clean transfer needs
  checkpoint surgery in our own `scripts/` (fresh learner template + ep0 actor
  weights + everything else reset).
- **`evaluation.source_root` cannot repoint a checkpoint at another episode.**
  `_resolve_parquet` only rebases paths that are relative or under
  `/workspace`; ours are absolute `/work/adapter_out/...`. Zero-shot
  cross-episode eval needs the baked `task.parquet` rewritten in the
  checkpoint's `user.resolved_config_json` metadata (header-only edit — read it
  with `scripts/diagnose/read_checkpoint_metadata.py`, which is stdlib and runs on the
  host).
- Checkpoints come in three files per boundary: `policy_<step>` (4.6 MiB, actor
  only), `state_<step>` (152 MiB, full learner), `replay_<step>` (only when
  `save_replay=true`, default false). A short run always saves a terminal
  checkpoint, as with PPO.
- Episodes **0 and 21 are the only two episodes in Tier 1 sharing an object**
  (both `tall_bar_stool`), so ep0->ep21 is same-object/different-task — the most
  favourable transfer case available, and not evidence about the other 28.

## FlashSAC weight-only transfer (Step D, Stage 2)

- **`scripts/train/transfer_flash_sac_policy.py` is the only honest way to warm-start
  one episode from another.** It builds the TARGET's fresh learner, splices in
  only the source actor's `params`/`batch_stats`, proves by leaf-by-leaf array
  comparison that nothing else moved (31 leaves changed, 157 identical), and
  writes a valid `training_state`. Upstream is imported, never modified.
- **A hand-built checkpoint needs `curriculum_stage` in its metadata.**
  `save_learner_checkpoint` adds it at save time; without it
  `evaluate_checkpoint` raises `FlashSAC checkpoint is missing its curriculum
  stage`. Derive it from the target's own curriculum (`stage_index(0)`), and
  carry `voc_scale` and the two reset probabilities too.
- **Proof a transfer is not a resume:** the first progress line reads
  `0/65,536`, not `65,536/65,536`.
- **Matching tensor dims is not enough.** The tool compares the resolved config
  of `action, observation, embodiment, objective, sim, termination, reset, env`
  and refuses on any difference; only `task` may differ.
- **MPPE ~55 cm on Episode 21 is the "never touched the object" floor**, not a
  score. The reference object travels 0.547 m from frame 0; a policy that moves
  it 1.5 cm scores 54.97 cm, and freezing it at frame 0 scores 54.74 cm. Always
  check achieved object travel from the exported trajectory before reading
  anything into an MPPE near that floor.
- **Actor-only transfer into a FRESH critic destabilizes early fine-tuning.**
  Measured: scratch and zero-shot both move the object 0.0146 m (inert), while
  65k steps of fine-tuning from a transferred actor moves it **1.14 m** - 2.1x
  the reference's own 0.55 m - and MPPE degrades 55.98 -> 109.46 cm. Transfer
  actor+critic, or freeze/warm the critic, before trusting a fine-tune.
- **Submission numbers must come from `evaluation.world_count=1`.** The kit's
  `track_2_tier1` sets `worlds_per_episode: 1` and `pack_submission.py` scores
  **world 0 only**, so a 64-world evaluation disagrees with the packer (55.977
  vs 55.008 cm) and the kit warns about it. Fine for A/B at matched world count;
  never for a submitted number.
- **The packer accepts our FlashSAC trajectory exports as-is**
  (`flash_chord_object_trajectories_v1`), and its frame check independently
  validates the Step A->C pipeline: *"one rigid frame transform per episode;
  maximum residual 1.23e-07 m, 1.26e-07 rad"*. `--dataset-root data/v2d_challenge`
  resolves because `participant_reference_root` falls back to
  `track_2/tier_1_multiview_caption`.
- `.venvs/kit` lacks `trimesh`, so the packer skips its AUC cross-check
  (MPPE/SP-SR/MP-SR still cross-check).

## The leaderboard metric is AUC, and it is not MPPE (Step D)

- **Track 2 is NOT ranked by AUC alone** — the final rank is the lowest *sum of ranks*
  over AUC, SP-SR, MP-SR and MPPE, tiers weighted equally (see "Scoring rules" below).
  AUC is the area under "fraction of frames with ADD < t" for
  **t in 1..9 cm** (`_ADD_THRESHOLDS_M = np.arange(0.01, 0.10, 0.01)` in the
  kit's `metric_code/track_2/AUC.py`). At that tolerance on a half-metre object,
  **a few degrees of rotation leaves the band**, so AUC is dominated by
  orientation. MPPE is six keypoints at +/-5 cm per axis - far more forgiving.
- **Doing nothing is not zero.** `scripts/evaluate/leaderboard_floor.py` scores a frozen
  object with the official formulas: **tier mean AUC 0.2661, MPPE 49.12 cm**.
  It agrees with NVIDIA's evaluator to 0.3% (AUC) / 0.4% (MPPE). Episodes whose
  object barely moves score very high by doing nothing (ep 8 `skinny_wood_chair`
  **0.5978**, above the CHORD baseline's tier average); ep 4 `baseball_bat`
  scores 0.1293.
- **Always quote a policy's AUC against that episode's stationary floor.**
  Episode 0 trained to 10M has MPPE 22.59 (31% better than the 32.59 floor) but
  AUC 0.2125 against a 0.3769 floor - **44% worse than not touching the
  object**. The stage-0 reward has `object_position: 1.0` and **no object
  orientation term**, so the optimisation target and the ranking metric
  disagree.
- Leaderboard reference points (Tier 1, 2026-10-20): CHORD Baseline AUC 0.5104 /
  MPPE 17.74; best complete entry AUC 0.6003 / MPPE 12.66; best MPPE 7.44
  (AUC 0.5953). Rank 1 shows AUC 0.9691 with no SP-SR/MP-SR/MPPE entry and is
  not comparable.

## Verify the current state

```bash
bash scripts/verify/verify_step_8_4.sh              # exit 0 = everything through Step 8.4 is good
bash scripts/verify/verify_flashchord_smoke.sh      # exit 0 = FlashCHORD G1 example replays cleanly
bash scripts/verify/verify_flashchord_training.sh   # exit 0 = PPO train + checkpoint + evaluate works
bash scripts/verify/verify_step_B.sh                # exit 0 = Episode 0 G1 reference is valid (--fast to skip the retarget)
bash scripts/verify/verify_step_C.sh                # exit 0 = all 30 Tier 1 references valid (--tier/--rerun)
bash scripts/verify/verify_keypoint_flash_sac.sh    # cluster: exit 0 = arm-3 keypoint objective is sound
```

## Orientation-aware FlashSAC (thirteenth reward term)

- **FlashSAC is generic in `len(env.objective_term_names)`** — replay,
  n-step discounting, `_relabel_rewards`, the learner and the runner all size
  themselves from it. A thirteenth objective term therefore needs **no upstream
  edit**: `pipeline/flash_sac_orientation.py` delegates to `ReconBodyObjective`,
  copies its twelve terms verbatim and appends `object_orientation`.
- **A scalar reward patch would not be learned.** FlashSAC stores *raw* terms
  and recomposes the reward from curriculum weights at sample time, so the term
  must be a real column. `pipeline/tests/test_flash_sac_orientation_replay.py`
  drives NVIDIA's own `insert_replay`/`relabel_rewards` to prove it.
- **`object_orientation_threshold: 1.5` in `termination` is NOT our reward** —
  it is upstream's 85.94° failure limit. The exporter preserves it; only
  `objective` and the curriculum weights are projected.
- **Hydra resolves `env.objective: ${objective}`, so the objective appears
  TWICE** in the resolved config. Project both or the official evaluator still
  tries to import our module and dies with `full_key: env.objective`.
- **Curriculum and objective configs are generated**, not hand-written:
  `scripts/train/make_orientation_curriculum.py` reads upstream's YAML and refuses to
  run if the twelve weights, nine thresholds or stage settings moved. Re-running
  it must be a no-op; `verify_orientation_flash_sac.sh` checks that.
- **At 1024 worlds the 350M budget is exact**: 349,999,104 / 1024 = 341,796
  collections, × 2.0 updates = **683,592 optimizer updates**, which is the
  full-recipe LR horizon, so no LR override is needed and every curriculum
  threshold lands on a collection boundary.
- Measured, 1024 worlds / 500k replay: **~16,500 env steps/s, ~7,170 MiB peak**,
  so a 350M run is **~6 h**. The extra column costs exactly 4 B/transition
  (1.91 MiB in a 500k ring). Startup (Warp + JAX compile) is ~39 s.
- `scripts/train/monitor_training_disk.py` prunes only one named run's
  `state_*.safetensors`, never actors, and hard-refuses the protected dirs
  (`source_ep0_w1024`, `submission_trial_20261021`, …). Dry run unless
  `--apply`.

## Episode 0 at 350M: what the first orientation arm actually showed

- **Training tracking errors massively overstate evaluation quality.** The run's
  own metrics ended at `obj_rot` 7.85 deg / `obj_pos` 3.4 cm, but the unassisted
  778-frame evaluation rollout measures **53.7 deg / 25.2 cm**. Training
  episodes terminate on the 10 cm / 1.5 rad thresholds and average 275 steps, so
  the metric only ever sees the part of the episode before failure. Never quote
  a training tracking error as a result; export the trajectory.
- **Scores, 5 fresh unassisted world-1 repeats each**
  (`scripts/evaluate/evaluate_orientation_sweep.sh`):
  | checkpoint | AUC | MPPE cm |
  |---|---|---|
  | previous 10M | 0.1190 +/- 0.0062 | 93.89 |
  | 230M | 0.1579 +/- 0.0360 | 36.13 |
  | **350M** | **0.3491 +/- 0.0185** | **26.84 +/- 12.47** |
  | stationary floor | 0.3769 | 32.59 |
  So +193% AUC over the old policy and MPPE finally beats the floor, but AUC is
  still **7.4% BELOW doing nothing**, and all three success rates are 0.
- **Stages 7-8 were worth the wall clock**: AUC 0.1579 -> 0.3491 over the last
  34% of the budget. Do not judge a run before its reset-distribution stages.
- **The failure is one specific event, not a gradual drift.** Per-phase error of
  the 350M rollout: frames 0-100 **0.5 cm / 1.1 deg**, 100-300 2.5 cm / 5.1 deg,
  300-500 28.7 cm / 50.4 deg, 500-778 47.9 cm / 110 deg. Position passes 10 cm
  at **frame 360**, rotation passes 20 deg at **frame 323**.
- **Why there**: the reference stool is *stationary* for the first ~4.7 s
  (frames 0-234, tilt 0.5 deg, travel 3 mm) and only then tips up - z 0.266 ->
  0.59 m, tilt 0 -> 91.6 deg, travel 0.78 m. Our policy tracks the do-nothing
  prefix almost perfectly and loses the object the moment real manipulation
  starts. It has not learned the tip-up; it has learned to stand still.
  Achieved travel is **0.88 m, 271% of the reference's 0.325 m** - it throws the
  stool rather than tipping it.
- **That also explains the competitive stationary floor**: ~40% of the episode's
  correct answer is "do not move", which a frozen object gets for free.
- **Both shaping terms are dead in the failure regime.** `object_position` is
  `exp(-sse/0.2^2)`, which is 0.018 at 40 cm and 0.000 at 80 cm; an exp-shaped
  orientation term at std 0.2 is ~6e-7 at 43 deg. Past the failure point there
  is no gradient at all, and training terminates there anyway, so recovery is
  never learned. Pick std against the error the *evaluation* shows, not the
  error training reports.
- **The thirteenth term was not the bottleneck.** It contributed 1.51% of
  `object_position`'s magnitude (11th of 13, below `action_rate_l2`). Dominant
  termination cause is **palm 56.9%**, not object - the hands leave the
  reference pose exactly when force is required.

## Scoring rules, the live board, and why stock stalls (2026-10-25)

Full write-up: `docs/findings/KEYPOINT_OBJECTIVE.md`.

- **Final rank = lowest sum of per-metric ranks** (AUC + SP-SR + MP-SR + MPPE; FAQ), and
  the **three tiers count equally** (site). A team missing any metric is **unranked**
  (upstream #209). Tier 1 alone cannot win Track 2.
- **Never ship a frozen-object trajectory**, even where it beats the policy: the FAQ
  requires trajectories *from learned policies*. `TOP3_STRATEGY.md` §3a is withdrawn.
- **Live standings need no Kaggle login**:
  `https://raw.githubusercontent.com/MVerghese/v2d-leaderboard-data/main/leaderboard.json`
  (the site's `data-src`). `official/.../leaderboard.json` at `193382a` is an empty stub.
- `origin/main` is ahead of our pinned `193382a`, but only in docs/FAQ/leaderboard code —
  nothing under `flash_chord` or `robotic_grounding`. The FAQ added: Tier 3 may use Tier 1's
  **first frame** for alignment and nothing else.
- **`object_position` is saturated**: 0.92-0.97 of its maximum in every stock run and stage,
  failing episodes included, because training terminates at 10 cm and
  `exp(-e^2/0.2^2)` is flat inside that window. The policy learns "stay inside 10 cm" and
  parks at 3-4 cm / 10-50 deg. There is no object-orientation term. NVIDIA's Sharpa
  objective has `object_keypoints`; G1's does not.
- **Arm 3 = `pipeline/flash_sac_keypoints.py`**: six axis keypoints at 0.5 x
  `command.object_radius`, upstream's Laplacian at 0.05 m, weight 5.0. Run it with
  `ARM=keypoints` (`train_tier_episode.sbatch` / `submit_tier_lanes.sh`) or
  `CONFIGS=keypoints EXTRA="objective=keypoints_recon_body curriculum=keypoints_recon_body_flash_sac"`.
  `submit_tier_lanes.sh EVAL=1` queues each evaluation `afterok` behind its training.
- **The exporter is a registry of all three thirteen-term targets** and
  `evaluate_policy.sbatch` projects them automatically (`portable_policy_<step>`).
- **The 1.67x playback speed is fixed for everyone**: `source_frame_playback: true` plays
  one 30 fps frame per 50 Hz step, and the packer needs one row per dataset frame.
- **Episode 7 diverges to NaN** (seed 42 at 5.3M, seed 43 at 81.8M). It is the only episode
  with a 1 cm support slab. `train_flash_sac.sbatch` now kills a job at the first NaN
  metrics line; resume from the last `state_*` before it with
  `training.resume=true training.checkpoint_path=/out/<run>/state_<step>.safetensors`
  (replay restarts empty; remaining steps must divide by 2048).
- **Cluster Python**: login/compute `python3` is 3.9 with no pytest/pyarrow — run tests and
  parquet analysis inside `containers/flash_chord.sif`, which has numpy 2.4 (no
  `np.trapz`: use `np.trapezoid`), scipy, pytest, pyarrow, but **no pandas**. Warp kernels
  test fine on Warp's CPU backend from the login node.
- `scripts/evaluate/decompose_auc_loss.py` splits a run's AUC into position vs rotation loss
  and first-failure frame; use it to compare arms on identical terms.
- **`docker run -v <missing host file>:<path>` does not fail — Docker creates an empty
  directory there**, and Python then dies with `can't find '__main__' module`. The repo
  restructure (`365b4b3`) moved scripts into `scripts/{data,diagnose,evaluate}/` but left
  the old paths in `run_tier_pipeline.py` and four `verify_*.sh` mounts; Tier 2's retarget
  failed 30/30 this way (2026-10-25, fixed). `run_tier_pipeline.py` now refuses to start
  if a driver is missing. Delete the root-owned empty dirs Docker left behind with `sudo rmdir`.
- **First keypoint submission, Tier 1 (2026-10-26, Kaggle 57664273-80)**: AUC 0.77919,
  SP-SR 0.86666, MP-SR 0.80000, MPPE 10.45054 — **identical to `pack_tier.sh`'s local
  scores**, so pack locally to test anything before uploading. **The limit is 5 uploads per
  WEEK per tier** (unlimited in the final 3 days): the uploader's "5 left today" is only Kaggle's
  technical cap, and the kit's rules say the website's 5-per-week policy wins where they differ.
  Each competition also allows **2 final submissions** — select them on Kaggle before the freeze.

## Tier 2: what the corruption is, and the recovered trajectory (2026-10-26)

- **Tier 2's object noise is a random walk, not jitter**: ~0 at frame 0, ~1 deg/frame,
  clipped at **30 deg / 3 cm** (median 23 deg, 1.9 cm; jitter only 0.13 cm, so smoothing
  does nothing). It keeps walking while the real object is static. Human motion is nearly
  clean (~1 cm, <0.001 rad). Policies track the corrupted orientation faithfully, which is
  what AUC and MP-SR (30 deg) punish.
- **Static hold is the only recovery that helped**: hold the pose while the (smoothed)
  object speed is below 3 cm/s, follow the measurement while it moves
  (`scripts/data/recover_tier2_object.py`). Input AUC vs clean **0.422 -> 0.480**, 27/30
  episodes better (worse: rice cooker, which turns in place). Carrying the orientation with
  the nearest palm (`follow_palm`) was **much worse** (21 -> 75 deg): pushes, drags and
  flips do not turn the object with the hand. Palm-spin motion detection lowered the mean.
- `--write` produces `data/adapter_out_tier_2r` (G1 references; meshes stay in
  `adapter_out_tier_2`, so training mounts every `data/adapter_out*` tree),
  `outputs/tier2r_frame_references/episode_XXXXXX.npz` (released frame, for the packer's
  `--frame-references`) and `training_manifest_tier2r.json`. Both come from one hold-index
  map, so they agree to 1.4e-07 m / 9e-06 deg.
- Tier 2's `hand_contact_active` is degraded (42% of frames vs Tier 1's 61%) — it was
  computed against the noisy object. Do not trust it as a contact signal.
- Pack recovered Tier 2 with
  `PREFIX=t2r FRAME_REFS=outputs/tier2r_frame_references COMMIT_URL=... bash cluster/pack_tier.sh 2`.
  Packing now writes to `outputs/submission/<prefix>/`.
- **First Tier 2 submission (2026-10-26, Kaggle 57669386-94)**: keypoint policies on the
  noisy input, AUC 0.44190, SP-SR 0.600, MP-SR 0.500, MPPE 23.33331 — again identical to the
  local pack (5th of 16 at the time). MPPE is dominated by 7 episodes that lose the object
  (the same bat / bottle / squeegee tasks that fail in Tier 1).

## Arm 4: object_velocity and the slow VOC fade (2026-10-27)

- **The four object-dropping Tier 1 episodes (vacuum 1, bat 4, bottle 25, squeegee 28) are not
  missing a contact reward.** `force_closure` is earned at the same rate as in solved episodes
  (0.03-0.07 weighted per step vs 0.05-0.07). What differs: object terminations climb exactly as
  the VOC assist fades (bottle: 11% at voc 1.0, 47% at 0.1, 72% at 0), and they are the **fast**
  episodes - median reference object speed 0.36-0.48 m/s (bat peaks 2.47) vs 0.02-0.04 for eps 0/3.
- Failure points in the submitted rollouts: bottle never lifts (fails at frame 126, two frames after
  the reference lift starts; 0.17 m vs 0.65 m), bat is lost during a 1.26 m lift in under 2 s,
  vacuum (frame 298) and squeegee (frame 527) are dropped mid-task. Five arms (s43, s44, term25,
  grip10, grip15) left all four at AUC 0.12-0.25 (ep28 0.26-0.43).
- **Arm 4 = `pipeline/flash_sac_velocity.py`**: arm 3's thirteen terms plus `object_velocity`
  (origin linear + world angular velocity vs central differences of the reference, Laplacian,
  0.5 m/s / 4 rad/s, weight 2.0). The reference's angular velocity has single-frame spikes up to
  41 rad/s, so keep the angular scale loose. The exporter now projects multi-term objectives.
- Arms: `ARM=keypoints_vel` (350M), `keypoints_slowvoc`, `keypoints_vel_slowvoc` (both
  **507,500,544 steps** - the fade from 0.5 to 0 takes 262.5M instead of 105M, the zero-assist
  phases keep upstream's 140M; evaluations land in `eval_507500544`).
  `bash scripts/verify/verify_velocity_flash_sac.sh` checks it all.
- **Result (2026-10-27, 5 repeats each) [VERIFIED]: neither change rescues vacuum, bat or bottle.**
  Mean AUC vel / slowvoc / vel_slowvoc: ep01 0.199 / 0.186 / 0.217 (best earlier 0.250),
  ep04 0.154 / 0.165 / 0.162 (0.167), ep25 0.108 / 0.127 / 0.126 (0.162). Only the squeegee moved:
  ep28 0.453 / **0.527** / 0.495 (best earlier 0.434). Ten arms have now failed on eps 1, 4, 25 -
  stop spending reward-shaping runs on them.
- Smooth arm (100x action penalties) helped ep02 0.576 -> **0.750** and ep13 0.688 -> 0.702; no
  gain elsewhere. Best-per-episode Tier 1 pack (local, official packer): AUC **0.8053**, SP-SR
  0.8667, MP-SR 0.8667, MPPE 10.64. Recovered Tier 2 (t2r) pack: AUC **0.4807**, SP-SR 0.700,
  MP-SR 0.467, MPPE 20.98 (Kaggle t2 entry: 0.4419 / 0.600 / 0.500 / 23.33).

- **Second uploads (2026-10-27, commit 870a798), again identical to the local packs**: Tier 1
  Kaggle 57699735-46 AUC 0.80534 / SP-SR 0.86666 / MP-SR 0.86666 / MPPE 10.64053; Tier 2 (t2r)
  57699751-56 AUC 0.48070 / 0.70000 / 0.46666 / 20.97937. The working kit on Ibrahim's PC is
  `~/v2d_submission_kit` (own `.venv`, Kaggle login `ibrahimmemon1811`).

## Tier 3 via CARI4D: first measurement (2026-10-27)

- **Still no Tier 3 download** (HF tree re-checked 2026-10-27: only tiers 1-2), although the kit's
  packer already names `track_2/tier_3_cari4d/`. We build our own and pack with `--frame-references`.
- **CARI4D's Episode 0 object track is good once aligned with Tier 1's first frame** (the only
  alignment the FAQ allows): AUC **0.7405**, MPPE 2.97 cm, 1.66 deg mean rotation error, vs the
  stationary floor 0.3769 (reproduced exactly) and the Tier 3 leader's 0.329 *policy* score.
  FoundationPose alone 0.6915; CoCoNet and post-opt add the rest. One easy episode only.
  `scripts/tier3/score_cari4d_object.py` (dump-reference in flash_chord.sif, score in cari4d.sif).
  The GLB CARI4D uses and the adapter's OBJ share vertices exactly (rigid-fit residual 0.000 mm).
- CARI4D peaks at ~17 GB RAM / 1 h 14 min for 778 frames on an A100. SAM3 masks with
  `--state_on_cpu` exceed 64 GB on long episodes; run them one episode per job at 150 GB.
- **cari4d.sif's Warp is a CUDA 13 build and refuses sm_70**: anything that runs SOMA-X (Warp
  kernels) or CARI4D must exclude the V100 nodes `wr[14-19]`. SAM3 masks (PyTorch only) run there fine.
- **The fingers have no prior at all.** The body runs on the pinned GEAR-SONIC controller plus a 0.15
  residual; each Dex3 finger target is the reference joint (constant - soma_to_g1 does not drive the
  hands) plus the policy residual. The Tier 1 human fingers barely move either (median range 5-8 deg,
  max ~28 deg per episode), so retargeting human fingers would not produce grasps.
- Converter still to build: CARI4D `refined.pth` (`pr.mhr_*`, `postopt.optimized_parameters`) ->
  vertices via CARI4D's own `lib_mhr.MHRLayer` -> SOMA via upstream `v2d_sam3d_body/lib/export_soma.py`
  (Path A, vertices; `soma` is in cari4d.sif, SOMA-X assets are a public download) -> challenge-format
  episode in Tier 1's raw frame -> retarget on the PC (`robotic-grounding:latest` is not on the cluster).

## Rule discipline for Tiers 2 and 3 (decided 2026-10-27)

- **No design decision for Tier 2 or Tier 3 may use the Tier 1 caption** - not as input, not to tune a
  threshold, not to pick a method, not as a report used for go/no-go. Tier 3 may read Tier 1's frame-0
  object pose for alignment only (FAQ); Tier 2 recovery must stand on Tier 2 data plus physics.
  Choosing between whole submissions by their Kaggle (or identical local-pack) score is ordinary
  leaderboard use and stays allowed.
- Consequences: `recover_tier2_object.py`'s static hold and its 3 cm/s threshold were chosen by scoring
  against Tier 1 (`--evaluate`), so the t2r entry (Kaggle 0.48070) is treated as grey and will not be
  a final selection; it is being replaced by a Tier-2-only recovery. `cari4d_to_lerobot.py` has no
  Tier 1 comparison. `score_cari4d_object.py` was a one-off Episode 0 feasibility check (AUC 0.7405)
  and must not be used to select or tune anything for Tier 3.

## The "robust system" wave (started 2026-10-27 ~14:30 CET)

- **Tier 3 end to end works on Episode 0**: CARI4D -> `cari4d_to_lerobot.py` -> retarget on Ibrahim's
  PC (`run_tier_pipeline.py stage/retarget/validate --tier tier_3_cari4d_ours`, rsync back) -> all 45
  reference checks pass on the cluster (IK 0.92x NVIDIA's demo) -> `make_tier3_manifest.py`
  (`training_manifest_tier3.json`, Tier-1-free floor) -> `PREFIX=t3 ARM=keypoints` training.
- **CARI4D "fp_centroid_distance contains infinity"** (small hand-held objects fully covered by the
  hand: mug, bat, book, sign, cone, cane, bottle) is fixed by a one-line patch applied in
  `tier3_cari4d_episode.sbatch` to a copy of upstream's export, bind-mounted over the image's file.
  SAM3 found no "iron": `PROMPT="metallic iron with blue accents"` (masks sbatch override).
- **Scripted grip** (`make_grip_references.py`, `ARM=keypoints_closegrip`,
  `MANIFEST=training_manifest_t1grip.json`): Dex3 fingers close to 0.6 x their far limit while the
  reference's `hand_contact_active` is set; only the 12 closing finger columns change.
- **Rules-clean Tier 2** (`recover_tier2_contact.py`, prefix `t2c`): hold the object unless a palm or
  foot (URDF FK, checked against the stored palms: 0.00 cm) is within bounding radius + 0.10 m, with
  a 5-frame release. Parameters fixed from physics before scoring; no Tier 1 code path. Episodes
  0, 28, 29 are never held, so `t2c_ep{00,28,29}_keypoints` are symlinks to the identical-input t2 runs.
- **Tighter keypoints** (`ARM=keypoints_tight`, scale 0.03) made every finished episode WORSE [VERIFIED]:
  ep02 0.679 vs 0.750, ep09 0.389 vs 0.577, ep11 0.509 vs 0.621 (5 repeats); 19/26/27 cancelled.
- **Scripted grip** helped the squeegee (ep28 AUC 0.527 -> 0.581, MPPE 25.0 -> 6.2 cm) and the bat's MPPE
  (rep1 101.9 -> 53.2 cm); vacuum and bottle unchanged. Best-per-episode Tier 1 now packs to about AUC
  0.808 / SP-SR 0.900 / MP-SR 0.867 / MPPE 8.39 (evaluator rep1, before the packer).
- `ARM=keypoints_gentle` = smooth + `action.residual_scale=0.075` (fingers keep 0.15); pilot on eps 8, 18, 23.
- **Tier 3 batch 1 (2026-10-27): 15 of 20 references validate.** Eps 6 and 9: SAM3 handed frame 4 of the
  actor track to a 2.4k-px blob elsewhere (vs 34k px on the actor), so CARI4D put the human 2.5 m away
  for one frame; `sam3_text_masks.py` now drops such frames (area < 25% of median AND centroid jump >
  half the actor's height) and lets the tracker refill them -- a scan of every mask set flags only
  those two frames. Eps 20, 27, 28: the frame-0 alignment rotation is wrong (pelvis 113 deg / 33 deg /
  60 deg from upright, object up to 1.9 m under the floor), i.e. CARI4D's frame-0 object orientation is
  off for these objects. Pelvis tilt alone is NOT a failure signal: real Tier 2 humans bend to 55 deg.
- CARI4D's CoCoNet crop raises "mesh geometry projects entirely outside the image" when a "valid"
  object pose lands off-screen (ep 2, basket); `tier3_cari4d_episode.sbatch` patches that crop to treat
  such a pose as invalid (upstream's own path), alongside the fp_centroid_distance patch.
- `make_tier3_manifest.py` admits only episodes whose cluster-side `run_tier_pipeline.py validate`
  record has zero failed checks; validate on the cluster after every rsync from the PC.
- **Validation cannot catch a turned scene.** `scripts/tier3/frame0_alignment.py` compares our frame-0
  body with Tier 1's frame-0 body (same SOMA rig and MHR identity; first frame only, as the FAQ allows):
  14 episodes agree within 0.5-4.8 deg / 8-24 cm, while 3, 6, 9, 12, 14, 20, 27, 28 are off by 53-178 deg --
  all objects with a symmetry (mug, monitor, bowl, platform, round table, rice cooker, basket, squeegee),
  whose frame-0 orientation CARI4D resolves differently from the label. 12 and 14 had passed all 45
  checks. The converter now places the scene by the human when the gap exceeds 20 deg and relabels the
  object's body frame so frame 0 equals Tier 1's; the manifest also requires a passed alignment check.
- **SAM3 picked the wrong object in 9 episodes** -- a strip along the left image border (2, 3, 4, 9, 16,
  17, 20, 25, 27; in 2/16/25 only for 27-38% of frames). Area x proximity rewarded the big wrong region.
  `sam3_text_masks.py --object_mesh` now scores instances by the object's expected image size (mesh
  extent / 1.7 m x actor pixel height; no reward above it, log-normal fall-off past 1.5x), refuses an
  implausible pick (>3x size and >1 actor height away), and drops single object frames that are both
  (refilled by the tracker). Both mask sbatches pass the released Tier 2 mesh. Check with an overlay
  before trusting any Tier 3 mask set.
- The Tier 3 manifest skips a reference older than its conversion (re-aligned 12/14 after batch 1).
- **The human-placed scene relabels orientation only** (about the object origin) plus one constant world
  offset to frame 0. A body-frame translation swung with CARI4D's symmetric orientation flips: episode
  28's 0.43 m lever arm turned raw 0.42 m steps into a 0.65 m teleport. 12/14/15 were converted with the
  earlier relabel (0.05-0.11 m) and validated; they were not redone.
- **Episode 11 (book) is trained for coverage only** (`make_tier3_manifest.py --accept-failed 11`): CARI4D
  loses the hand-hidden book from frame ~209 (75 object steps > 15 cm, up to 1.73 m; human steps up to
  0.72 m). The accepted failures are recorded in the manifest. Tier 3 needs a policy for all 30.
  Stale conversions live in `data/reconstructed/superseded/` until their rerun lands.
- `cluster/diagnose_jitter.sbatch` measures per-policy joint shake (mean step-to-step joint change
  of the simulated robot, unassisted world-1 rollout). Reference 0.1-0.2 deg, SONIC alone 0.24-0.42,
  keypoints 0.74-1.07. **The smooth arm cuts it on every paired episode**: ep02 0.74 -> 0.60, ep04
  1.01 -> 0.68, ep13 1.07 -> 0.47, ep24 0.90 -> **0.28** (SONIC alone: 0.24), with AUC equal or better
  where the object is held. `ARM=keypoints_smooth` is now a named arm (100x `action_rate_l2` /
  `action_l2`).
- **Slurm reservation `maint_2026-10-30`**: every GPU node from 2026-10-30 13:00 CET, no end date
  (365-day placeholder), no MOTD notice. The maintainer expects it to last hours, a day at most, so
  training stays on the cluster; jobs simply cannot run across the window. `train_tier_episode.sbatch`
  now asks for 6 h (runs take ~3 h) so jobs keep starting until ~07:00 that day.
