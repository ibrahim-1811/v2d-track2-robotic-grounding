# Documentation index

**New here? Read in this order:**

1. [architecture/ARCHITECTURE.md](architecture/ARCHITECTURE.md) — what this
   project is and how it works, in plain language with diagrams
2. [../CLAUDE.md](../CLAUDE.md) — the environment traps. Every line was learned
   by something breaking. Read before running anything.
3. [guides/ONBOARDING.md](guides/ONBOARDING.md) — set the machine up and run
   your first thing
4. [architecture/CODE_MAP.md](architecture/CODE_MAP.md) — what every file does

## architecture/ — how it fits together
| | |
|---|---|
| [ARCHITECTURE.md](architecture/ARCHITECTURE.md) | the four stages, the training/evaluation distinction, why the 8 GB laptop shaped everything |
| [CODE_MAP.md](architecture/CODE_MAP.md) | every module, script and output directory, explained |

## guides/ — how to do a thing
| | |
|---|---|
| [ONBOARDING.md](guides/ONBOARDING.md) | set up from scratch; what to work on, in order |
| [ORIENTATION_RUN_COMMANDS.md](guides/ORIENTATION_RUN_COMMANDS.md) | launch a training run and watch it |
| [EPISODE_0_WALKTHROUGH.md](guides/EPISODE_0_WALKTHROUGH.md) | one episode end to end |

## reference/ — facts that do not change often
| | |
|---|---|
| [TRACK2_MISSION.md](reference/TRACK2_MISSION.md) | the rules, the metrics, how we are scored |
| [NVIDIA_UPSTREAM.md](reference/NVIDIA_UPSTREAM.md) | what is in `official/` and how it is organised |
| [REQUIREMENTS.md](reference/REQUIREMENTS.md) | what the challenge officially requires |
| [SETUP_AUDIT.md](reference/SETUP_AUDIT.md) | this machine, measured against those requirements |

## findings/ — what we measured
| | |
|---|---|
| [TIER_REFERENCES.md](findings/TIER_REFERENCES.md) | building and validating all 30 references |
| [CHALLENGE_ADAPTER.md](findings/CHALLENGE_ADAPTER.md) | Stage A, the format conversion |
| [EPISODE_0_G1_RETARGET.md](findings/EPISODE_0_G1_RETARGET.md) | Stage B, human → robot |
| [EPISODE_0_FRAME_AUDIT.md](findings/EPISODE_0_FRAME_AUDIT.md) | coordinate frames, and the rule never to break |
| [CHALLENGE_BASELINE.md](findings/CHALLENGE_BASELINE.md) | what a submission has to look like |
| [FLASHCHORD_BASELINE_SMOKE.md](findings/FLASHCHORD_BASELINE_SMOKE.md) | getting NVIDIA's example to run |
| [FLASHCHORD_TRAINING_FEASIBILITY.md](findings/FLASHCHORD_TRAINING_FEASIBILITY.md) | can this laptop train at all |
| [STEP_D_FLASHSAC_FEASIBILITY.md](findings/STEP_D_FLASHSAC_FEASIBILITY.md) | memory and throughput limits, measured |
| [STEP_D_TRANSFER_SMOKE.md](findings/STEP_D_TRANSFER_SMOKE.md) | warm-starting one episode from another |
| [ORIENTATION_AWARE_FLASHSAC.md](findings/ORIENTATION_AWARE_FLASHSAC.md) | the 13th reward term: design, tests, measurements |

Two findings live with their data in `outputs/` because they are inseparable
from it:
- `outputs/step_D/curriculum_audit/REPORT.md` — all 9 stages, all 12 weights
- `outputs/step_D/ep0_contact_diagnosis/REPORT.md` — why Episode 0 fails at
  frame 323, and `PHASE_PROFILE_AND_EXPERIMENT.md` beside it

## plans/ — what we intend to do
| | |
|---|---|
| [TOP3_STRATEGY.md](plans/TOP3_STRATEGY.md) | **current plan.** Where we stand, why the cluster matters, the gates |
| [STEP_D_A100_PLAN.md](plans/STEP_D_A100_PLAN.md) | running on bigger hardware |
| [2026-10-21-orientation-aware-flashsac.md](plans/2026-10-21-orientation-aware-flashsac.md) | the original 13th-term design |
| [2026-10-21-tier1-ranking-next-phase.md](plans/2026-10-21-tier1-ranking-next-phase.md) | earlier ranking plan |
