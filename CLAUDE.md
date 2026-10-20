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
  `scripts/validate_soma_in_container.py`.
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
`docs/FLASHCHORD_BASELINE_SMOKE.md`.

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
- **Fix: `scripts/retarget_episode_g1.py`** — chunks `SOMA._reconstruct` over
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
  extend with `scripts/download_challenge_episodes.py` (stdlib urllib; there is
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

## Verify the current state

```bash
bash scripts/verify_step_8_4.sh              # exit 0 = everything through Step 8.4 is good
bash scripts/verify_flashchord_smoke.sh      # exit 0 = FlashCHORD G1 example replays cleanly
bash scripts/verify_flashchord_training.sh   # exit 0 = PPO train + checkpoint + evaluate works
bash scripts/verify_step_B.sh                # exit 0 = Episode 0 G1 reference is valid (--fast to skip the retarget)
```
