#!/usr/bin/env python3
"""Report FlashSAC's exact tensor dimensions and replay VRAM for a G1 reference.

Step D feasibility probe. Answers two questions without starting a training run:

1. What are ``observation_dim`` / ``action_dim`` / ``critic_context_dim`` /
   ``objective_term_count`` for the ``g1_recon_body_flash_sac`` recipe on one of
   our own Step C references? Those four numbers set the replay buffer's size,
   and nothing in the config files states them - they come from the instantiated
   embodiment and observation strategy.
2. How much VRAM does the replay ring need at a given capacity? The ring is
   allocated with ``jnp.zeros`` (``replay.create_replay``), so it lives on the
   GPU and competes with Warp's scene and the JAX executables. On an 8 GiB card
   it, not the world count, is the binding constraint.

``flash_chord.training.flash_sac.replay.replay_memory`` is NVIDIA's own exact
byte accounting, so this script reports their arithmetic rather than inventing
its own.

Runs inside ``flash_chord:latest``. Mount our references and invoke by path -
never inline this in ``bash -lc "python -c ..."`` (see CLAUDE.md):

    docker run --rm --gpus all \
      -v .../flash_chord:/workspace -v .../assets:/v2d/assets:ro \
      -v .../data/adapter_out:/work/adapter_out:ro -v .../scripts:/driver:ro \
      -e XLA_PYTHON_CLIENT_PREALLOCATE=false -w /workspace flash_chord:latest \
      python /driver/probe_flash_sac_memory.py --parquet /work/adapter_out/...
"""

from __future__ import annotations

import argparse
import json
import sys

MIB = 1024.0 * 1024.0
GIB = 1024.0 * 1024.0 * 1024.0

# Capacities to cost out. 10,000,000 is the shipped g1_recon_body_flash_sac
# value; the rest bracket what an 8 GiB card can hold.
CANDIDATE_CAPACITIES = (10_000_000, 4_000_000, 2_000_000, 1_000_000, 500_000, 250_000)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", required=True, help="container-side reference parquet")
    parser.add_argument(
        "--experiment",
        default="g1_recon_body_flash_sac",
        help="Hydra experiment group (default: the canonical G1 FlashSAC recipe)",
    )
    parser.add_argument(
        "--world-count",
        type=int,
        default=32,
        help="worlds to build the probe scene with; dims are world-count "
        "independent, so keep this small (default 32)",
    )
    parser.add_argument(
        "--support",
        default=None,
        help="'true'/'false' to override scene.support; default leaves the recipe's value",
    )
    parser.add_argument("--json-out", default=None, help="also write the findings here")
    return parser.parse_args()


def build_env(args):
    """Compose the real experiment and instantiate the env, as training would."""
    import warp as wp
    from hydra import compose, initialize_config_module
    from flash_chord.configuration import instantiate_typed
    from flash_chord.embodiments.base import Embodiment
    from flash_chord.envs.rl import RLEnv, RLEnvConfig
    from flash_chord.scene.collision import CollisionPolicy
    from flash_chord.scene.setup import setup_scene
    from flash_chord.training.environment import WarpRLEnv
    from flash_chord.training.flash_sac.config import TrainingConfig

    overrides = [
        "experiment={}".format(args.experiment),
        # The sequence_id/robot_name path contains '=', which Hydra's override
        # grammar rejects unless the VALUE is single-quoted - the same quoting the
        # README needs on the command line (see CLAUDE.md).
        "task.parquet='{}'".format(args.parquet),
        "scene.world_count={}".format(args.world_count),
        "logging.mode=disabled",
    ]
    if args.support is not None:
        overrides.append("scene.support={}".format(args.support))

    with initialize_config_module(config_module="flash_chord.configs", version_base="1.3"):
        cfg = compose(config_name="train_flash_sac", overrides=overrides)

    env_config = instantiate_typed(cfg.env, RLEnvConfig)
    embodiment = instantiate_typed(cfg.embodiment, Embodiment)
    collision = instantiate_typed(cfg.collision, CollisionPolicy)
    training = instantiate_typed(cfg.training, TrainingConfig)

    with wp.ScopedDevice(training.device):
        setup = setup_scene(
            parquet=cfg.task.parquet,
            control_fps=env_config.sim.fps,
            motion_speed=cfg.task.motion_speed,
            embodiment=embodiment,
            collision=collision,
            world_count=cfg.scene.world_count,
            include_support=cfg.scene.support,
            decompose_objects=cfg.scene.decompose_objects,
            object_scale_min=cfg.scene.object_scale_min,
            object_scale_max=cfg.scene.object_scale_max,
            object_scale_seed=cfg.scene.object_scale_seed,
            contact_friction=cfg.scene.get("contact_friction", 1.0),
            object_free_joint_damping=cfg.scene.get("object_free_joint_damping", 0.0),
            source_frame_playback=bool(cfg.task.source_frame_playback),
            motion_start_frame=int(cfg.task.motion_start_frame),
            motion_end_frame=int(cfg.task.motion_end_frame),
        )
        env = RLEnv(setup.scene, setup.reference, config=env_config)
        env.reset()
        jax_env = WarpRLEnv(env, publish_diagnostics=False, publish_step_metrics=False)
    return cfg, training, setup, env, jax_env


def main() -> int:
    args = parse_args()
    cfg, training, setup, env, jax_env = build_env(args)

    from flash_chord.training.flash_sac.replay import ReplaySpec, replay_memory

    reference = setup.reference
    # The dims live on WarpRLEnv, which is what ``run_training`` is handed as
    # ``env``: ``action_dim`` is ``env.action.action_dim`` on the RLEnv, and
    # ``critic_context_dim`` is derived from the critic-context names.
    dims = {
        "observation_dim": int(jax_env.observation_dim),
        "action_dim": int(jax_env.action_dim),
        "critic_context_dim": int(jax_env.critic_context_dim),
        "objective_term_count": len(jax_env.objective_term_names),
    }

    print()
    print("=== reference under test ===")
    print("  parquet          : {}".format(setup.parquet_path))
    print("  frames           : {}".format(reference.num_frames))
    print("  playback fps     : {}".format(reference.fps))
    print("  scene.support    : {}".format(cfg.scene.support))
    print("  object assets    : {}".format([a.name for a in reference.object_assets()]))
    articulations = [
        (asset.name, art.simulation_joint_name)
        for asset in reference.object_assets()
        for art in asset.articulations
    ]
    print("  object articulations: {}  <- the ONLY thing a resume validates".format(articulations))

    print()
    print("=== policy/critic dimensions (world-count independent) ===")
    for name, value in dims.items():
        print("  {:22s} {}".format(name, value))
    print("  objective terms      : {}".format(list(jax_env.objective_term_names)))

    print()
    print("=== replay ring VRAM, by capacity (NVIDIA's own replay_memory()) ===")
    print("  shipped recipe: capacity={} minimum_size={} n_step={} dtype={}".format(
        training.replay.capacity,
        training.replay.minimum_size,
        training.replay.n_step,
        training.replay.observation_storage_dtype,
    ))
    print()
    header = "  {:>12} {:>8} {:>12} {:>11} {:>9}".format(
        "capacity", "dtype", "ring", "per-trans B", "fits 8GiB?"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    rows = []
    for dtype in ("float16", "float32"):
        for capacity in CANDIDATE_CAPACITIES:
            spec = ReplaySpec(
                capacity=capacity,
                minimum_size=min(training.replay.minimum_size, capacity),
                batch_size=min(training.replay.batch_size, capacity),
                n_step=training.replay.n_step,
                discount=training.discount,
                world_count=args.world_count,
                observation_dim=dims["observation_dim"],
                action_dim=dims["action_dim"],
                objective_term_count=dims["objective_term_count"],
                critic_context_dim=dims["critic_context_dim"],
                observation_storage_dtype=dtype,
            )
            memory = replay_memory(spec)
            ring_gib = memory.ring_bytes / GIB
            # Leave room for the Warp scene + JAX executables, measured
            # separately; 8 GiB total with ~2 GiB already committed.
            verdict = "yes" if ring_gib < 4.0 else "NO"
            print("  {:>12,} {:>8} {:>9.2f} GiB {:>11,} {:>9}".format(
                capacity, dtype, ring_gib, memory.bytes_per_transition, verdict
            ))
            rows.append({
                "capacity": capacity,
                "dtype": dtype,
                "ring_bytes": int(memory.ring_bytes),
                "ring_gib": round(ring_gib, 4),
                "bytes_per_transition": int(memory.bytes_per_transition),
                "pending_bytes": int(memory.pending_bytes),
                "sampled_batch_bytes": int(memory.sampled_batch_bytes),
            })

    findings = {
        "parquet": str(setup.parquet_path),
        "experiment": args.experiment,
        "frames": int(reference.num_frames),
        "scene_support": bool(cfg.scene.support),
        "dims": dims,
        "objective_terms": list(jax_env.objective_term_names),
        "critic_context_names": list(jax_env.critic_context_names),
        "object_articulations": articulations,
        "shipped_replay": {
            "capacity": int(training.replay.capacity),
            "minimum_size": int(training.replay.minimum_size),
            "batch_size": int(training.replay.batch_size),
            "n_step": int(training.replay.n_step),
            "observation_storage_dtype": training.replay.observation_storage_dtype,
        },
        "shipped_world_count": int(training.world_count),
        "shipped_total_environment_steps": int(training.total_environment_steps),
        "replay_rows": rows,
    }
    if args.json_out:
        with open(args.json_out, "w") as handle:
            json.dump(findings, handle, indent=2)
        print()
        print("  wrote {}".format(args.json_out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
