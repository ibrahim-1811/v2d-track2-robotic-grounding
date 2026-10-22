# Step C — every episode of a tier, from Parquet to a validated G1 reference

Generalizing [Step A](../findings/CHALLENGE_ADAPTER.md) (LeRobot → SOMA) and
[Step B](../findings/EPISODE_0_G1_RETARGET.md) (SOMA → G1) from the one hand-built Episode 0
reference to **all 30 episodes of a tier**, with the architecture reusable for
Tier 2 and a future Tier 3 without change.

Companion documents: [CHALLENGE_ADAPTER.md](../findings/CHALLENGE_ADAPTER.md) ·
[EPISODE_0_G1_RETARGET.md](../findings/EPISODE_0_G1_RETARGET.md) ·
[EPISODE_0_FRAME_AUDIT.md](../findings/EPISODE_0_FRAME_AUDIT.md) (coordinate frames) ·
[CHALLENGE_BASELINE.md](../findings/CHALLENGE_BASELINE.md) (submission shape).

- **Upstream code:** commit `193382a184308f6800331f3db829f76dc4294cfe`, unmodified
- **Images:** `robotic-grounding:latest` (retarget), `flash_chord:latest` (load check)
- **Entry point:** `scripts/data/run_tier_pipeline.py`
- **Verification:** `bash scripts/verify/verify_step_C.sh`

Conclusions are marked **[VERIFIED]** (observed directly), **[DOC]** (stated by
NVIDIA, not independently checked) or **[GAP]/[UNRESOLVED]**.

---

## 1. What runs, in order

```
meta/episodes_metadata.jsonl ──► pipeline/episodes.py          (the registry)
                                      │
      ┌───────────────────────────────┴──────────────────────────────┐
      ▼                                                              ▼
 STAGE  stage        pipeline/lerobot_to_soma.py + verbatim asset copies
                     └─► data/<out_root>/whole_body/soma/<seq>/
                            soma_params.npz, poses.npy, ground_plane.json,
                            reconstructed_mesh/output_aligned.glb,
                            episode_manifest.json
                     then pipeline/validate_staged.py  (identity, frames,
                     timestamps, object poses bit-for-bit, coordinate rule)
      ▼
 STAGE  retarget     scripts/data/retarget_episode_g1.py inside robotic-grounding
                     (chunked SOMA reconstruction → NVIDIA's soma_to_g1.main())
                     └─► .../sequence_id=<seq>/robot_name=g1/data.parquet
      ▼
 STAGE  validate     pipeline/validate_reference.py vs NVIDIA's demo reference
      ▼
 STAGE  flashchord   scripts/diagnose/check_reference_in_flashchord.py inside flash_chord
                     (load_reference → setup_scene → snap to frame 0)
      ▼
 STAGE  regress      re-run Episode 0, diff against the preserved baseline
      ▼
 STAGE  report       outputs/<tier>/report.{json,md} + references_index.json
```

Every stage is independently runnable and resumable:

```bash
PY=.venvs/episode0/bin/python

$PY scripts/data/run_tier_pipeline.py inventory                  # read-only: what's there
$PY scripts/data/run_tier_pipeline.py all                        # stage -> report
$PY scripts/data/run_tier_pipeline.py retarget --episodes 11,25   # one or a few
$PY scripts/data/run_tier_pipeline.py retarget --episodes 3 --force
$PY scripts/data/run_tier_pipeline.py regress                     # Episode 0 regression (re-retargets)
$PY scripts/data/run_tier_pipeline.py regress --no-rerun          # same diff, nothing overwritten
$PY scripts/data/run_tier_pipeline.py all --tier tier_2_synthetic_noise
```

**Resumability rule.** Each stage writes one record per episode to
`outputs/<tier>/records/<idx>_<stage>.json` (write-then-rename, so an
interrupted run leaves no half-written record). A stage skips an episode only
when that episode's record says it succeeded *and* its output still exists. A
**failed** episode is never skipped and never dropped from the report — the next
run retries it. `--force` redoes work regardless. **[VERIFIED]**

---

## 2. The registry is the only place that knows the layout

`pipeline/episodes.py` turns one line of `episodes_metadata.jsonl` into an
`EpisodeSpec` that resolves every input and output path, counts the support
surface's collision shapes, decides `scene.support`, and plans the reference's
frame count. It is **standard-library only**, because the orchestrator drives
Docker from host Python 3.12, where there is no `pyarrow` (see the top of
`CLAUDE.md`).

It also cross-checks itself against the dataset: the metadata names its own
`mesh` / `ground_plane` / `support_surface` paths, and the registry raises if its
convention resolves anywhere else. A missing input is reported once, up front,
with every missing path listed — not at the point of use, 20 minutes into a run.
**[VERIFIED]**

---

## 3. Support surfaces: `scene.support` decided per episode, twice

`flash_chord/scene/support.py:69` turns only `Cylinder` and `Cube` prims into
collision shapes, and `add_support_surfaces` **raises** when a stage declares
none (`support surface contains no Cylinder or Cube collision shapes`). NVIDIA
ships such empty stubs deliberately — a sequence with no distinct support
surface gets a 125-byte `Xform` and nothing else. So the flag has to be decided
per episode, and getting it wrong is a hard failure either way: `true` on an
empty stub raises, `false` on a real stage drops the table the object rests on.

Two independent counts must agree before an episode passes:

| | where | how |
| --- | --- | --- |
| host | `pipeline/episodes.count_support_collision_shapes` | regex over the flat ASCII USDA |
| container | `check_reference_in_flashchord.py` | `pxr.Usd.Stage.Traverse()` prim types |

The orchestrator passes its own count in as `--expect-support-shapes`, and the
container-side check **fails** on a disagreement rather than reconciling it. The
host scan also refuses to answer at all if a stage ever gains a USD composition
arc (`subLayers`, `references`, `payload`, …), because a text scan cannot see
prims pulled in from another layer and would silently under-count.
**[VERIFIED]**

---

## 4. Unmeasured object frames are dropped, never invented

`observation.object.visible == False` means NVIDIA zero-filled that row: the
pose was not measured. A zero quaternion is not a rotation, so there are exactly
three honest options — stop, pass the zeros through, or emit no reference frame
for it. Interpolating is not one of them (`CLAUDE.md`, and
`test_invalid_frames_passthrough_is_not_interpolated`).

This pipeline:

1. stages the zero row through **verbatim** (`--on-invalid passthrough`), so the
   staged inputs remain an exact image of the Parquet;
2. retargets the window `[0, first_unmeasured_frame)` via `soma_to_g1
   --end-frame`, so no reference frame is ever built from a zero pose;
3. records the dropped indices and the reason in `episode_manifest.json`, in the
   episode's `validate` record, and in the report's "Recorded notes" section.

`--end-frame` is safe for this: `soma_to_g1.py` slices at line 516, long after
the frame-0 anchoring, so the retained frames keep exactly the coordinates a
full-sequence run would give them (this is also why `--end-frame` does **not**
reduce memory — see `scripts/data/retarget_episode_g1.py`).

An unmeasured frame that is **not** part of a trailing run cannot be handled by
truncation, and the registry refuses the episode rather than guess
(`EpisodeSpec.validate_frame_policy`, `test_mid_sequence_invalid_frame_is_refused`).
**[VERIFIED]**

---

## 5. Tier isolation

Tier 1 and Tier 2 publish **the same 30 sequence ids** — same objects, same
tasks, same frame counts; only the measurements differ. One shared staged tree
would let Tier 2's noisy inputs overwrite Tier 1's clean ones, or let a Tier 2
policy train against a Tier 1 reference and look excellent for the wrong reason.
A valid Track 2 submission is one per tier with policies trained on that tier's
own inputs, so this is a correctness property, not housekeeping.

Three mechanisms, all checked:

| | mechanism |
| --- | --- |
| host paths | `out_root_for_tier`: `data/adapter_out` (tier 1, unchanged), `data/adapter_out_tier_2`, `data/adapter_out_<tier>` for anything else — including a future Tier 3 |
| container paths | each tier mounts at `/work/<out_root name>`, so the absolute `object_mesh_paths` that `soma_to_g1.py` bakes into the parquet **names the tier it came from** |
| provenance | every `episode_manifest.json` records its tier plus the sha256 of the source Parquet, and `validate_staged` asserts both |

Tier 1 deliberately keeps the original `data/adapter_out` root and the original
`/work/adapter_out` mount point, so Episode 0's existing artifacts, its baked
mesh paths, `verify_step_B.sh` and the Step B docs all keep working unchanged.
**[VERIFIED]**

---

## 6. What "validated" means

### 6.1 Staged inputs — `pipeline/validate_staged.py`

Per episode, against the pristine Parquet:

| Check | Why it exists |
| --- | --- |
| manifest tier / index / sequence id, source Parquet sha256 | an episode must state which tier's measurements it holds |
| `poses`/`transl` rows == metadata `frames` == Parquet rows | a silent off-by-one desynchronises everything downstream |
| timestamps strictly increase, spacing is 1/fps within 1e-4 s | the published timestamps are float32, so exactness is impossible; drift is not |
| `poses.npy` rebuilt from the Parquet, compared **bit-for-bit** | "close enough" hides a frame-ordering or dtype mistake |
| staged translations == raw Parquet xyz | the one statement equivalent to "we did not apply `diag(1,-1,-1)`" |
| human `poses`/`transl` bit-for-bit | the human data is already in the SOMA frame and must be copied, not converted |
| invalid rows are still all-zero | proves the zero-fill was passed through, not repaired |
| staged mesh / ground plane / support sha256 == mirror | a truncated copy is not a copy |

### 6.2 The reference — `pipeline/validate_reference.py`

Eleven sections; the ones beyond Step B's original set:

- **Root alignment (new).** `soma_to_g1.py` stores the plane-leveled SOMA joints
  in the parquet's `source_payload`. The G1 free-flyer is solved to track the
  SOMA hips, so `r_world @ soma_joints[:, Hips]` and `robot_root_position` must
  stay within centimetres. On Episode 0 the horizontal offset is **0.026 m mean
  / 0.088 m max** with a height offset whose standard deviation is **0.014 m**.
  The gates are an order of magnitude looser (0.50 m, 0.20 m) because the
  failure they exist to catch — a double-applied or missing frame conversion —
  shows up as **metres**. The same section checks that frame 0's root sits on the
  anchored origin, which it does by construction. **[VERIFIED]**
- **IK iteration count — measured, reported, deliberately not gated.** This
  started life as a gate ("saturation after frame 0 means the solver gave up")
  and **failed 17 of 30 episodes**. The gate was unsound; §6.4 records the
  measurements that retired it. What is gated instead: residual quality against
  NVIDIA's own demo (median ratio < 3×, worst episode **1.81×**) plus a new
  per-episode **spike test** — the worst single frame against that episode's own
  median, so one diverged frame cannot hide inside a healthy median. Measured
  worst across Tier 1 is **1.63×** and NVIDIA's own demo sits at **1.65×**, so
  the 3.0× gate keeps about a factor of two of headroom over both. Frame 0 is
  still called out in a note: it is the only frame solved from the robot's
  neutral `q0` with no warm start, so using the whole budget there says nothing
  about the reference. **[VERIFIED]**
- **Kinematic feasibility (new).** The question that actually matters before
  training is whether the G1 can execute the reference at all. Every joint step
  is divided by that joint's own `<limit velocity=…>` from
  `assets/urdfs/g1/main_with_hand.urdf` — the robot's published number, not one
  we chose. The worst case across all 30 episodes is **0.57×** the limit
  (NVIDIA's demo: 0.12×), so **no episode demands a motion the robot cannot
  track**. G1's 14 hand joints are excluded because `soma_to_g1` leaves them
  undriven — they are constant here *and* in NVIDIA's demo, so measuring them
  would report the rig's rest pose rather than this reference. The fraction of
  frames with a driven joint resting on a position limit is **reported, not
  gated** (median 33.7%, dominated by `waist_pitch_joint` on 20 of 30
  episodes): that is the same stable G1-vs-human torso offset already
  documented for `torso_link`, and NVIDIA's own demo does it on 11.4% of its
  frames. **[VERIFIED]**

### 6.3 FlashCHORD compatibility

`check_reference_in_flashchord.py` loads each reference through
`flash_chord.data.load_reference`, builds the Newton scene with `setup_scene`
(G1+Dex3 embodiment, real collision policy, the per-episode `scene.support`),
snaps to reference frame 0 and asserts the resulting `body_q` is finite. This is
a **load and scene-build gate, not a policy result** — replay assistance and
evaluation semantics are covered in
[FLASHCHORD_BASELINE_SMOKE.md](../findings/FLASHCHORD_BASELINE_SMOKE.md).

---

### 6.4 Why the IK iteration-cap gate was retired

The first full Tier 1 run reported `13/30 episode(s) OK`. All 17 failures were
**the same single check** — `the IK solve never hit the 200-iteration cap after
frame 0` — and in every case it was the *only* failing check out of 44. No other
gate failed on any episode. Six independent measurements say that gate was
measuring the solver's effort, not the reference's quality.

**1. For G1 the cap cannot mean "gave up".** `configs/g1/retargeter.json` sets
`posture_task.q_prev_cost = 0.05`. Because that is `> 0`,
`WholeBodyKinematics.__init__` builds `_posture_qprev_task`, so `_extra_tasks()`
is non-empty and `has_extra_tasks` is **True on every frame**. That makes the
plateau exit at `whole_body_kinematics.py:564`
(`if not has_extra_tasks and all(frame_tasks_converged...)`) **unreachable**,
leaving `vel_norm < 1e-6` at line 566 as the solve's only early exit. Reaching
`max_iter` therefore means "did not drive a 43-DOF whole-body QP to a 1e-6
velocity norm" — an asymptotic condition — not that the solve diverged.

**2. Residual quality does not break at the cap.** Per episode, comparing capped
frames against that same episode's uncapped frames:

| | median residual on capped frames | on uncapped frames | ratio |
| --- | --- | --- | --- |
| median over the 17 episodes | 0.99 | — | **1.07×** |
| best (Episode 11) | 0.706 | 0.775 | **0.91×** (capped frames are *better*) |
| worst (Episode 16) | 0.901 | 0.681 | 1.32× |

The root moves **millimetres** across capped frames (median 0.006 m/frame), so
there is no teleport or discontinuity either.

**3. The cap is a pile-up on a continuum, not a cliff.** Binned over all 22,959
non-frame-0 frames of the tier:

| iterations | frames | median residual |
| --- | --- | --- |
| 150–154 | 479 | 0.959 |
| 170–174 | 439 | 0.992 |
| 190–194 | 341 | 1.022 |
| 195–199 | 307 | 1.042 |
| **= 200** | **3,382** | **1.128** |

Residual rises smoothly and monotonically with iteration count, because hard
frames need both. Frames at exactly 199 (median 1.076) **pass** and frames at
200 (median 1.128) **failed** — there is no physical discontinuity for a gate to
sit on. Residual and iterations are correlated through a common cause (fast
motion: joint velocity on capped frames is 2–4× that of uncapped frames), which
is the opposite of the "solver gave up" story.

**4. NVIDIA's demo reference is not admissible evidence here.** Its iterations
are **median 8, max 34** — unreachable under a `vel_norm < 1e-6` exit, so it was
produced under the plateau exit and its iteration counts are not comparable to
ours. Even Episode 0, which *passed* the old gate, has median 69. So "NVIDIA's
reference never saturates" was never evidence that ours should not.

**5. The failure the gate was reaching for never happened.** A solve genuinely
giving up is the early `break` on the Pink/eigenpy binding flake
(`whole_body_kinematics.py:522`), and it prints
`[whole_body_kinematics] giving up on frame after N Pink/eigenpy rebuilds`.
**Zero** of the 30 Tier 1 retarget logs contain that message, or the
`forwardKinematics post-solve flake` message. Every solve ran its loop to
completion.

**6. NVIDIA does not treat it as an error.** `soma_to_g1 --diagnose-ik` flags
iteration saturation at `0.9 × max_iter` as a *printed one-line diagnostic*
(`--diagnose-ik-iter-fraction`, default 0.9), alongside its 0.05 m residual
threshold that every frame of every episode exceeds. `CLAUDE.md` already records
that `--diagnose-ik` flags 778/778 frames on Episode 0 and that this is normal.

The validator was also internally inconsistent: §9 already said *"Iteration
count is reported, not gated: a harder QP legitimately needs more iterations
than the demo's standing pick-and-place"* for the demo comparison, while
hard-gating absolute saturation three lines earlier.

**Not weakened, replaced.** The saturation count and fraction are still computed
and still recorded — per episode, in the `validate` record's metrics
(`ik_saturated_frames`, `ik_saturated_after_frame0`, `ik_saturated_fraction`)
and as a printed `NOTE`, so nothing is hidden. In its place the validator gained
two gates that measure the reference instead of the solve: the **residual spike
test** (§6.2) and the **URDF joint-velocity feasibility check** (§6.2), the
latter being strictly stronger than anything that existed before. Check count
per episode went from 44 to **45**.
`pipeline/tests/test_validate_reference.py::test_iteration_saturation_is_reported_not_failed`
keeps the old gate from coming back. **[VERIFIED]**

### 6.5 The misleading `ALL CHECKS PASSED` line

The batch log ended with `ALL CHECKS PASSED` immediately above
`=== validate: 13/30 episode(s) OK ===`, which reads like a contradiction. It
was not a reporting bug in the gate logic: `Checker.summary()` emits that banner
**per episode**, `run_checks` called it unconditionally, and Episode 29 — the
last episode in the batch — passed. So the log genuinely contained 13 such
banners, the last of which landed just before the tier verdict.

Nothing was being hidden: `verify_step_C.sh` reads `report.json` structurally
and never greps stdout for that string, and `stage_validate`'s exit code was
already driven by the failure list. But a per-episode banner printed 30 times is
a footgun. `run_checks` now takes `summary=False`, which the orchestrator
passes, and prints an episode-scoped line instead:

```
  episode 11: all 45 checks passed
```

`scripts/data/validate_g1_reference.py` (the single-file Step B entry point) keeps
`summary=True`, so its output and `verify_step_B.sh` are unchanged.
**[VERIFIED]**

## 7. Episode 0 regression

The brief was to preserve previous outputs, so the first time the orchestrator
touches an episode that already has a reference *and no record of its own*, it
copies that reference to
`outputs/<tier>/regression/episode_<idx>_baseline.parquet` and never overwrites
it. For Episode 0 that file is the hand-built Step B artifact.

`run_tier_pipeline.py regress` then re-runs Episode 0 through the generalized
pipeline and diffs every identity field and every per-frame numeric column
against that baseline. Bitwise equality is **not** asserted — the IK solve runs
on the GPU — but the worst absolute delta over all numeric columns is reported
and gated at 1e-3 (a real regression moves things by centimetres).

**`regress --no-rerun`** does the same diff against the reference *already on
disk* instead of solving the IK again. Once the batch `retarget` has run, that
reference **is** the generalized pipeline's output and the baseline **is** the
pre-generalization artifact, so the comparison the stage exists to make is
already available — and re-solving would overwrite a reference we were asked to
preserve. Measured this way on 2026-10-20: worst delta **7.63e-06** over every
numeric column, with the object columns bit-identical (they pass through
verbatim). Use the plain `regress` when the point is to prove the retarget is
*reproducible*; use `--no-rerun` when the point is to prove the current
references match the baseline. **[VERIFIED]**

---

## 8. For the next step (FlashSAC / BC / cross-episode transfer)

`outputs/<tier>/references_index.json` is the contract a training sweep should
read. Per episode it carries the reference parquet path, the frame count, the
Hydra `scene.support` value, the container mount point, the dropped frames, the
object and task text, and a `usable` flag that is true only when **every** gate
passed. Deriving those from the raw dataset a second time — or hard-coding the
per-episode support flag — is how a 30-run sweep ends up with silent failures.

No BC dataset extraction is implemented yet. What a future extractor needs is
already in the reference parquet (`robot_root_position`, `robot_root_wxyz`,
`robot_joint_positions`, `ee_pose_w`, plus `source_payload`'s SOMA joints) and
in this index.

**Are these references fit to train on?** On the kinematic side, yes, and that
is now measured rather than assumed: all 30 are schema-identical to NVIDIA's
demo, finite, continuous, framed correctly (root tracks the SOMA hips to
centimetres), within 0.57× of G1's own joint velocity limits, and they build a
Newton scene at reference frame 0. All 30 carry `usable: true`.

Two measured properties a training run should expect rather than be surprised
by:

- **`waist_pitch_joint` rests on its position limit on a third of frames**
  (median 33.7%, up to 92.8%). The human torso lean exceeds G1's waist range —
  the same morphology offset `torso_link` shows in the IK residuals. A tracking
  reward on waist pitch will see a reference the robot can reach but not pass,
  and NVIDIA's own demo has the same property on 11.4% of its frames.
- **IK residual varies ~2.3× across episodes** (median 0.51 to 1.17 m summed
  over all whole-body tasks). Episodes 24, 8, 3 are the cleanest; 25, 23, 28,
  19 the hardest. Cross-episode transfer experiments should stratify on this
  rather than treat the 30 episodes as interchangeable.

What is **not** established is anything about policy performance: no policy has
been trained and no MPPE measured. See §9.4. **[GAP]**

---

## 9. Results — Tier 1, all 30 episodes

Measured 2026-10-20 on an RTX 5070 Laptop (8 GiB). The per-episode table lives in
`outputs/tier_1_multiview_caption/report.md`; regenerate both with:

```bash
.venvs/episode0/bin/python scripts/data/run_tier_pipeline.py report
cat outputs/tier_1_multiview_caption/report.md
bash scripts/verify/verify_step_C.sh          # exit 0
```

### 9.1 Headline

| | result |
| --- | --- |
| episodes staged / retargeted / **validated** / FlashCHORD-loaded | 30 / 30 / **30** / **30** |
| reference checks | **1,350 of 1,350 passed** (45 per episode, 0 failed) |
| reference frames | **22,989** of 22,990 source frames (1 deliberately excluded, Episode 11) |
| `scene.support` | `true` for 14, `false` for 16 — host text scan and FlashCHORD's USD parse **agree on all 30** |
| Episode 0 regression | **PASS**, worst numeric delta **7.63e-06** (tolerance 1e-3) |
| `pipeline/tests` | **81 passed** |
| upstream | unmodified, still pinned at `193382a` |

### 9.2 Reference quality, aggregated over the 30 episodes

| metric | median | worst | gate | source of the gate |
| --- | --- | --- | --- | --- |
| IK residual median vs NVIDIA's demo | 1.25× | **1.81×** | < 3× | NVIDIA's own shipped reference |
| IK residual spike (worst frame vs episode median) | 1.22× | **1.63×** | < 3× | demo's own spread is 1.65× |
| joint step vs that joint's URDF velocity limit | 0.17× | **0.57×** | < 1.0× | G1's own URDF |
| root-vs-SOMA-hips horizontal offset, max | 0.088 m | **0.158 m** | < 0.50 m | Episode 0 measurement ×~5 |
| root-vs-hips height offset, std | 0.013 m | **0.040 m** | < 0.20 m | Episode 0 measurement ×~5 |
| frames at the 200-iteration cap | 0.5 % | 100 % | **not gated** (§6.4) | — |
| frames with a driven joint on a position limit | 33.7 % | 92.8 % | **not gated** (§6.2) | demo does it on 11.4 % |

Every episode clears every gate with at least a 1.7× margin on the tightest one
(IK residual ratio, Episode 25 at 1.81× against 3×). **[VERIFIED]**

### 9.3 Cost, as measured

| stage | wall clock | peak GPU | peak container RAM |
| --- | --- | --- | --- |
| stage | 2 s | — | — |
| retarget | 3,725 s total, **122 s median/episode** | 2,510 MiB | 3,009 MiB |
| validate | < 1 s (host, no GPU) | — | — |
| flashchord | 1,910 s total, **61 s median/episode** | 581 MiB | 3,967 MiB |

The retarget's 2,510 MiB is flat in sequence length, as Step B predicted from
Episode 0's 3,411 MiB at `--chunk-frames 64`. FlashCHORD's 61 s per episode is
dominated by Warp/Newton kernel compilation: the cache lives in the container's
`/root/.cache/warp`, which is ephemeral, so every `docker run` recompiles.
**[VERIFIED]**

### 9.4 What this does and does not establish

**Does:** all 30 Tier 1 references are schema-identical to NVIDIA's own demo,
finite, continuous, correctly framed (root tracks the SOMA hips to
centimetres), kinematically executable by a real G1, and they load and build a
Newton scene through `flash_chord.data.load_reference` + `setup_scene` with the
correct per-episode `scene.support`.

**Does not:** say anything about policy performance. No policy has been trained
and no MPPE has been measured — `docs/findings/FLASHCHORD_BASELINE_SMOKE.md` records why
(no checkpoint exists at `193382a`, and the PPO recipe's terminal budget is
~25 days of wall clock on this GPU). A FlashCHORD **load** gate is not a replay
and is certainly not an evaluation. **[GAP]**

<a name="results-table"></a>
