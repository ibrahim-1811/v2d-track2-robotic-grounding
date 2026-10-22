# Code map: what every file does

A tour of the repo in plain language. For *why* it is shaped this way, read
[ARCHITECTURE.md](ARCHITECTURE.md) first.

---

## The shape of the repo

```
v2d-track2-robotic-grounding/
├── README.md            start here
├── CLAUDE.md            environment traps — every line learned by breaking something
├── pipeline/            our Python package: the logic, and its tests
├── scripts/             things you run, grouped by what they are for
│   ├── data/            get the data, convert it, retarget it, validate it
│   ├── train/           launch and watch training
│   ├── evaluate/        score a policy honestly
│   ├── diagnose/        work out why something failed
│   └── verify/          one-command health checks
├── configs/             our Hydra config overrides (NVIDIA's are in official/)
├── docs/
│   ├── architecture/    this doc and ARCHITECTURE.md
│   ├── guides/          how to do a thing
│   ├── reference/       facts about the challenge and upstream
│   ├── findings/        what we measured, step by step
│   └── plans/           what we intend to do next
├── official/            NVIDIA's code, pinned, READ-ONLY  (gitignored)
├── data/                inputs and staged references        (gitignored)
└── outputs/             runs, checkpoints, logs, reports    (gitignored)
```

Four directories are gitignored because they are big (`official/` 1.5 GB,
`data/` 3 GB, `outputs/` 16 GB) or machine-specific (`.venvs/`). A fresh clone
gives you code and docs; [ONBOARDING.md](../guides/ONBOARDING.md) tells you how
to recreate the rest.

---

## `pipeline/` — the library

This is the part that is *imported*, not run. Scripts are thin; the logic lives
here so it can be tested.

| File | What it does |
|---|---|
| `episodes.py` | The registry. Answers "which episodes exist in this tier, and where are their four input files?" Every other stage asks this rather than guessing paths. Supports Tier 1 and Tier 2, each with its own staged output tree. |
| `lerobot_to_soma.py` | **Stage A.** Converts NVIDIA's released format into the layout their retargeter expects. Copies object poses through unchanged — it never smooths or fills anything. |
| `validate_staged.py` | Checks Stage A's output: are frame counts, timestamps and object poses bit-for-bit identical to the input, and was the coordinate convention left alone? |
| `validate_reference.py` | **Stage C.** Checks a retargeted G1 reference is usable: joint velocities inside the robot's real limits, solver residuals comparable to NVIDIA's own demo, no exploded frames. |
| `checks.py` | The small assertion helpers the two validators share, so failures read the same way. |
| `flash_sac_orientation.py` | **Our 13th reward term.** Wraps NVIDIA's 12-term objective, copies those 12 through untouched, and appends an object-orientation penalty `-(θ/π)²`. Contains the GPU kernel. Never edits upstream. |
| `flash_sac_orientation_exp.py` | The same idea with different shaping: `exp(-θ²/σ²)`, the form NVIDIA already uses for every other orientation term. Built because the first version turned out ~60x too weak. Not yet tested on GPU. |
| `orientation_reward.py` | The same maths in plain NumPy, with no GPU. This is the reference the GPU kernel is tested against, and it is where all the quaternion conventions are written down. |
| `flash_sac_actor_export.py` | Rewrites a trained policy's embedded config so NVIDIA's unmodified evaluator can load it, while keeping the network weights byte-identical and preserving the real training config alongside. |
| `flash_sac_metrics_logger.py` | Writes losses and per-reward-term values to `metrics.jsonl`/`.csv` and prints a summary line. Exists because with logging disabled, upstream records nothing at all locally. |
| `flash_sac_train_entry.py` | A copy of upstream's training entrypoint with one change: it attaches the logger above. Imports upstream, never modifies it. |

### `pipeline/tests/`

`test_*.py` run on your laptop in about a second. `container_*.py` need GPU
libraries and only run inside Docker — they are deliberately *not* named
`test_*` so a plain `pytest pipeline/tests/` does not choke on them. See
`pipeline/tests/README.md` for why that naming, and not a skip marker, is the
only thing that works here.

```bash
.venvs/episode0/bin/python -m pytest pipeline/tests/ -q     # 142 tests, ~1 s
```

---

## `scripts/data/` — getting from the challenge files to a usable reference

| Script | What it does |
|---|---|
| `download_challenge_episodes.py` | Mirrors the challenge episodes from Hugging Face using only the standard library (there is no `huggingface_hub` on this machine). Safe to re-run. **Note: the dataset is now gated and returns 401.** |
| `stage_episode_0_assets.py` | The original one-episode staging, kept because the walkthrough doc refers to it. |
| `run_tier_pipeline.py` | **The main entry point for data.** Runs stage → retarget → validate for a whole tier, 30 episodes, and writes a manifest. |
| `retarget_episode_g1.py` | **Stage B.** Wraps NVIDIA's retargeter so it fits in 8 GB by feeding the sequence in chunks. Numerically identical to running it whole; 79 s per episode. |
| `validate_g1_reference.py` | Validates one retargeted reference and prints a readable report. |
| `validate_soma_in_container.py` | Confirms the SOMA body model actually loads inside the retarget container. |

---

## `scripts/train/` — producing a policy

| Script | What it does |
|---|---|
| `train_orientation_flash_sac.py` | **The launcher.** Builds the 30-line Docker command, runs preflight (disk, budget divisibility, output directory free, valid tag), streams training output to your terminal *and* a log, and handles Ctrl-C cleanly. Use `--dry-run` to see the command without running it. |
| `make_orientation_curriculum.py` | Generates our config files *from* NVIDIA's, and refuses to run if their weights or thresholds have changed. That is what makes "we did not alter the original 12 terms" checkable rather than just claimed. |
| `show_training_metrics.py` | Reads `metrics.jsonl` and prints a table of losses and tracking errors, a per-reward-term breakdown (`--terms`), or PNG curves (`--plot`). `--watch 60` refreshes. |
| `monitor_training_disk.py` | Prunes old resumable checkpoints from **one named run**. Never deletes a policy file, hard-refuses protected directories, and is a dry run unless you pass `--apply`. |
| `benchmark_orientation_arms.sh` | Runs the 12-term and 13-term objectives back to back and reports steady throughput and peak GPU memory. The progress bar's own rate is a cumulative average and must not be trusted; this differences the log instead. |
| `probe_flash_sac_memory.py` | Works out how much GPU memory a given world count and replay size will need, before you waste a run discovering it. |
| `transfer_flash_sac_policy.py` | Warm-starts one episode from another's policy, proving by array comparison that only the actor weights moved. |

---

## `scripts/evaluate/` — finding out if it is actually good

| Script | What it does |
|---|---|
| `evaluate_step_d.sh` | Runs NVIDIA's **unmodified** evaluator on one checkpoint and exports the object trajectory the packer needs. |
| `evaluate_orientation_sweep.sh` | Runs the above N times per checkpoint in fresh processes, because a single rollout is not repeatable here. **Refuses to start if training is using the GPU.** |
| `summarize_orientation_sweep.py` | Aggregates a sweep into mean/median/spread and compares it against the three things that matter: the stationary floor, the previous policy, and how far the object actually moved. Warns when results overlap and the difference is unresolved. |
| `leaderboard_floor.py` | Scores a frozen object with the official formulas — the "doing nothing" baseline every result must beat. Agrees with NVIDIA's evaluator to 0.3 %. |
| `check_unassisted.py` | Proves the evaluator really does strip object assistance, rather than taking it on trust. |

---

## `scripts/diagnose/` — working out why it failed

| Script | What it does |
|---|---|
| `diagnose_contact_rollout.py` | Reproduces the evaluator's rollout exactly, but also records palm poses, contact counts, **real contact forces** and per-frame tracking errors. This is what identified that Episode 0 pushes the stool rather than gripping it. |
| `profile_episode_phases.py` | For all 30 episodes: how long is the object stationary, when does manipulation start, how far does it travel. Tells you which episodes a given fix can even help. |
| `analyze_episode_0.py`, `audit_frames_episode_0.py` | The original per-episode and per-frame audits. |
| `analyze_manipulation.py` | Summarises what an episode's manipulation actually consists of. |
| `read_checkpoint_metadata.py` | Prints what a checkpoint was trained on, without a GPU or any special libraries — safetensors headers are just JSON. |
| `inspect_checkpoint.py` | The same question answered through NVIDIA's own loader. |
| `check_reference_in_flashchord.py` | Loads a reference through FlashCHORD's own reader to prove it is acceptable to the real thing. |
| `flashchord_headless_replay.py` | Replays a reference with no policy and no browser. How you look at an episode you have not trained yet. |
| `view_policy_viser.sh` | Watch a trained policy in a browser at `localhost:8080`. Note the viewer keeps training's failure thresholds, so it **restarts when the policy fails** — that restart is information, not a bug. |

---

## `scripts/verify/` — one command, pass or fail

Each prints a list of checks and exits non-zero if any fail.

| Script | Confirms |
|---|---|
| `verify_step_8_4.sh` | the environment and early pipeline are intact |
| `verify_step_B.sh` | Episode 0's G1 reference is valid |
| `verify_step_C.sh` | all 30 Tier 1 references are valid |
| `verify_flashchord_smoke.sh` | NVIDIA's example replays cleanly |
| `verify_flashchord_training.sh` | train → checkpoint → evaluate works end to end |
| `verify_orientation_flash_sac.sh` | **our 13th reward term**: upstream untouched, configs reproducible, 142 host + 26 container tests, preflight sane, and a policy round-trips through the official evaluator (with the control proving the round-trip is necessary) |
| `check_cluster.sh` | what hardware a machine has |

---

## `configs/`

Two Hydra config files per reward variant, **generated** by
`make_orientation_curriculum.py`, never hand-edited. One points Hydra at our
objective class; the other adds the 13th weight to all nine curriculum stages.
NVIDIA's own configs stay in `official/` and are never modified.

---

## `outputs/` — where runs land

Gitignored and large. The important ones:

| Path | What it is |
|---|---|
| `step_D/ep0_orient_350M/` | **The completed 350M Episode 0 run.** 35 checkpoints, `metrics.jsonl`. 5.5 GB. |
| `step_D/source_ep0_w1024/` | The earlier 10M run. **Protected** — the disk monitor refuses to touch it. |
| `step_D/submission_trial_20261021/` | Our submitted entry and its per-episode scores. **Protected.** |
| `step_D/sweep_ep0_orient_350M/` | The 5-repeat evaluation behind the headline numbers. |
| `step_D/ep0_contact_diagnosis/` | The failure diagnosis: report, timeline figure, instrumented rollouts, the 30-episode phase profile. |
| `step_D/curriculum_audit/` | All 9 curriculum stages and all 12 reward weights, read from the source. |
| `step_D/leaderboard_floor.json` | The "doing nothing" score for every episode. Small, and you will use it constantly. |
| `step_D/warp_cache/` | Compiled GPU kernels. Mount it or every run recompiles for ~130 s. |
| `step_D/trajectories/` | Exported object trajectories — what the packer consumes. |

**Why `outputs/` was not reorganised:** checkpoints have absolute paths baked
into their metadata, and the containers mount this tree at a fixed location.
Moving it mid-competition would break reruns for no real gain. It is documented
instead.

---

## Reading order for someone new

1. [ARCHITECTURE.md](ARCHITECTURE.md) — the shape of the problem
2. [CLAUDE.md](../../CLAUDE.md) — the traps, before you run anything
3. [ONBOARDING.md](../guides/ONBOARDING.md) — set up and run
4. This file — when you need to know what a particular thing does
