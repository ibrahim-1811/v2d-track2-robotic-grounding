# Documentation index

**New here? Read in this order:**

1. [../README.md](../README.md): what the project is, and every command by task
2. [architecture/ARCHITECTURE.md](architecture/ARCHITECTURE.md): how it works, with diagrams
3. [../CLAUDE.md](../CLAUDE.md): environment traps and measured results. Every line was learned
   by something breaking. Read before running anything.
4. [architecture/CODE_MAP.md](architecture/CODE_MAP.md): what every file does

## architecture/: how it fits together
| | |
|---|---|
| [ARCHITECTURE.md](architecture/ARCHITECTURE.md) | the three tiers, the shared retarget/train/evaluate/pack chain, the control stack, how we extend upstream, Tier 3's guards |
| [CODE_MAP.md](architecture/CODE_MAP.md) | every module, script, Slurm job and output folder |

## guides/: how to do a thing
| | |
|---|---|
| [ONBOARDING.md](guides/ONBOARDING.md) | first-time setup on the PC (laptop-era; the README covers the cluster) |

## reference/: facts that change rarely
| | |
|---|---|
| [TRACK2_MISSION.md](reference/TRACK2_MISSION.md) | the rules, the metrics, how we are scored |
| [REQUIREMENTS.md](reference/REQUIREMENTS.md) | what the challenge officially requires |
| [NVIDIA_UPSTREAM.md](reference/NVIDIA_UPSTREAM.md) | what is in `official/` and how it is organised |

## findings/: what we measured
| | |
|---|---|
| [TIER_REFERENCES.md](findings/TIER_REFERENCES.md) | building and validating all 30 references per tier |
| [KEYPOINT_OBJECTIVE.md](findings/KEYPOINT_OBJECTIVE.md) | why the stock reward stalls, the keypoint term, the scoring rules |
| [ORIENTATION_AWARE_FLASHSAC.md](findings/ORIENTATION_AWARE_FLASHSAC.md) | the first 13th term: design, tests, Episode 0 at 350M |
| [CHALLENGE_ADAPTER.md](findings/CHALLENGE_ADAPTER.md) | the format conversion |
| [EPISODE_0_G1_RETARGET.md](findings/EPISODE_0_G1_RETARGET.md) | human to robot |
| [EPISODE_0_FRAME_AUDIT.md](findings/EPISODE_0_FRAME_AUDIT.md) | coordinate frames, and the rule never to break |
| [CHALLENGE_BASELINE.md](findings/CHALLENGE_BASELINE.md) | what a submission has to look like |
| [FLASHCHORD_BASELINE_SMOKE.md](findings/FLASHCHORD_BASELINE_SMOKE.md) | getting NVIDIA's example to run |
| [FLASHCHORD_TRAINING_FEASIBILITY.md](findings/FLASHCHORD_TRAINING_FEASIBILITY.md) | PPO on 8 GB |
| [STEP_D_FLASHSAC_FEASIBILITY.md](findings/STEP_D_FLASHSAC_FEASIBILITY.md) | FlashSAC memory and throughput limits, measured |
| [STEP_D_TRANSFER_SMOKE.md](findings/STEP_D_TRANSFER_SMOKE.md) | warm-starting one episode from another (abandoned; tools removed 2026-10-27) |

Results from the cluster phase (Tier 1 arms, Tier 2 recovery, Tier 3 reconstruction, jitter)
are recorded in [../CLAUDE.md](../CLAUDE.md), dated per section.
