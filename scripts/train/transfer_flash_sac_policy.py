#!/usr/bin/env python3
"""Build a weight-only FlashSAC warm start for a DIFFERENT episode.

Step D, Stage 2. The question is "does Episode 0's policy help Episode 21?", and
answering it honestly requires transferring *weights* and nothing else. Upstream
offers no way to do that:

* ``training.checkpoint_path`` is fed to ``restore_learner_checkpoint``, which
  calls ``_validate_metadata(path, "training_state")`` and rejects the
  actor-only ``policy_*.safetensors`` outright;
* a plain ``resume`` from a ``state_*.safetensors`` carries the source's
  ``environment_steps`` (hence its curriculum stage and spent budget), Adam
  state, reward-normalizer moments and exploration state;
* ``validate_resume_reference_config`` does not catch the episode swap, because
  it compares only the object-articulation contract and every rigid-object
  episode in this tier has an empty one.

So this script constructs the file upstream would have accepted, using
upstream's own functions and without modifying a line of it:

1. compose the TARGET episode's experiment and build its env, exactly as
   ``scripts/train_flash_sac.py`` does;
2. ``create_learner`` a FRESH target learner - fresh optimizer, fresh critic,
   fresh reward normalizer, fresh exploration, ``environment_steps = 0``;
3. load the SOURCE actor and splice ONLY its weights into that fresh learner;
4. prove every other leaf is still bit-identical to the fresh learner;
5. save a valid ``training_state`` checkpoint carrying the TARGET's metadata.

The result is loadable with ``training.resume=true``, which then starts the
target at step 0, in curriculum stage 0, with a cold replay - the only
difference from a scratch run being the actor's initial weights.

Why the learner must be built at the training world count: ``RewardNormalizerState
.discounted_return`` and ``ExplorationState.noise`` are per-world arrays, so the
saved state's shapes depend on it. ``--world-count`` must match the world count
of the run that will consume this file; the script refuses otherwise at load
time rather than silently reshaping.

Runs inside ``flash_chord:latest``. Mount our references and invoke by path.
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

TRANSFER_CHOICES = ("actor", "actor_critic")

# Config groups whose semantics the policy depends on. A difference in any of
# these means the source actor is being asked to drive a different problem, so
# they are compared and a mismatch is fatal. task/scene/logging/training are
# expected to differ (that IS the transfer) and are reported, not enforced.
SEMANTIC_GROUPS = ("action", "observation", "embodiment", "objective", "sim", "termination", "reset", "env")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source-policy", help="actor-only policy_<step>.safetensors to transfer FROM")
    source.add_argument("--source-state", help="learner state_<step>.safetensors to take the actor FROM")
    parser.add_argument("--target-parquet", required=True, help="container-side reference parquet to transfer TO")
    parser.add_argument("--target-support", default=None, help="'true'/'false' override for scene.support")
    parser.add_argument("--world-count", type=int, required=True,
                        help="MUST equal the world count of the run that will consume this file")
    parser.add_argument("--experiment", default="g1_recon_body_flash_sac")
    parser.add_argument("--transfer", choices=TRANSFER_CHOICES, default="actor",
                        help="'actor' transfers the policy only (conservative); "
                             "'actor_critic' also transfers critic + target critic")
    parser.add_argument("--replay-capacity", type=int, default=None,
                        help="match the consuming run, so the learner template is built identically")
    parser.add_argument("--replay-minimum-size", type=int, default=None)
    parser.add_argument("--updates-per-collection", type=float, default=None)
    parser.add_argument("--out", required=True, help="training_state safetensors to write")
    parser.add_argument("--emit-policy", default=None,
                        help="also write an actor-only policy checkpoint carrying the TARGET's "
                             "metadata. That is what makes a ZERO-SHOT evaluation possible: "
                             "evaluate_policy.py rebuilds the environment from the checkpoint's "
                             "own baked config, and evaluation.source_root cannot repoint an "
                             "absolute /work/... path at another episode.")
    parser.add_argument("--report-json", default=None)
    return parser.parse_args()


def build_target(args):
    """Compose the target experiment and build its env + fresh learner."""
    import jax
    import jax.numpy as jnp
    import warp as wp
    from hydra import compose, initialize_config_module

    from flash_chord.configuration import instantiate_typed, resolved_dict
    from flash_chord.embodiments.base import Embodiment
    from flash_chord.envs.rl import RLEnv, RLEnvConfig
    from flash_chord.scene.collision import CollisionPolicy
    from flash_chord.scene.setup import setup_scene
    from flash_chord.training.checkpoint_metadata import (
        build_checkpoint_metadata,
        critic_schema,
        policy_schema,
        resolved_reference_config,
    )
    from flash_chord.training.environment import WarpRLEnv
    from flash_chord.training.flash_sac.config import TrainingConfig
    from flash_chord.training.flash_sac.learner import create_learner, reset_context_indices
    from flash_chord.training.flash_sac.runner import resolve_jax_device

    overrides = [
        "experiment={}".format(args.experiment),
        # The sequence_id/robot_name path contains '=', which Hydra's override
        # grammar rejects unless the VALUE is single-quoted (see CLAUDE.md).
        "task.parquet='{}'".format(args.target_parquet),
        "scene.world_count={}".format(args.world_count),
        "logging.mode=disabled",
    ]
    if args.target_support is not None:
        overrides.append("scene.support={}".format(args.target_support))
    if args.replay_capacity is not None:
        overrides.append("training.replay.capacity={}".format(args.replay_capacity))
    if args.replay_minimum_size is not None:
        overrides.append("training.replay.minimum_size={}".format(args.replay_minimum_size))
    if args.updates_per_collection is not None:
        overrides.append("training.updates_per_collection={}".format(args.updates_per_collection))

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
        env.capture_step()
        env.reset()
        jax_env = WarpRLEnv(env, publish_diagnostics=False, publish_step_metrics=False)

        environment_config = resolved_dict(cfg)
        environment_config["task"]["parquet"] = str(setup.parquet_path)
        environment_config["reference"] = resolved_reference_config(setup.reference, str(setup.parquet_path))
        target_metadata = build_checkpoint_metadata(
            environment_config,
            policy_schema(env.action, env.observation_strategy),
            critic_schema(env.action, env.observation_strategy, jax_env.critic_context_names),
        )

        context_names = tuple(jax_env.critic_context_names)
        reset_context_indices(context_names)
        jax_device = resolve_jax_device(training.device)
        # Stage 0 is what a scratch run starts in, and a weight-only warm start
        # must start there too.
        if training.curriculum is not None:
            jax_env.set_curriculum(training.curriculum.stages[0])
        initial_observation = jax.device_put(jnp.array(jax_env.reset(), copy=True), jax_device)
        initial_context = jax.device_put(jnp.array(jax_env.initial_critic_context, copy=True), jax_device)
        learner, _ = create_learner(
            training,
            initial_observation,
            initial_context,
            action_dim=jax_env.action_dim,
            objective_term_count=len(jax_env.objective_term_names),
            termination_cause_count=len(jax_env.termination_cause_names),
            tracking_error_count=len(jax_env.tracking_error_names),
            critic_context_names=context_names,
            device=jax_device,
        )
    return cfg, training, setup, jax_env, target_metadata, learner


def compare_semantics(source_metadata, target_metadata, emit) -> tuple[list[str], list[str]]:
    """Compare config semantics beyond tensor shapes. Returns (fatal, informational)."""
    from flash_chord.training.checkpoint_metadata import checkpoint_config

    source_config = checkpoint_config(dict(source_metadata))
    target_strings = {k: v for k, v in target_metadata.items() if isinstance(v, str)}
    target_config = checkpoint_config(target_strings)

    fatal: list[str] = []
    informational: list[str] = []
    for group in SEMANTIC_GROUPS:
        a = source_config.get(group)
        b = target_config.get(group)
        if a == b:
            emit("  PASS  {:12s} identical between source and target".format(group))
        else:
            fatal.append(group)
            emit("  FAIL  {:12s} DIFFERS between source and target".format(group))
            emit("          source: {}".format(json.dumps(a, sort_keys=True)[:200]))
            emit("          target: {}".format(json.dumps(b, sort_keys=True)[:200]))
    # These are expected to differ; that difference is the experiment.
    for group in ("task", "scene"):
        a = (source_config.get(group) or {})
        b = (target_config.get(group) or {})
        if a != b:
            informational.append(group)
    return fatal, informational


def check_critic_compatibility(source_metadata, target_metadata, emit) -> list[str]:
    """Verify a transferred critic will mean the same thing on the target task.

    Matching tensor shapes say nothing about whether a critic's OUTPUT is
    interpretable on a new task. FlashSAC's critic is distributional: it emits a
    categorical distribution over a FIXED value support, and predicts returns in
    REWARD-NORMALIZED units. So a transferred critic is only meaningful if

      * the value support is the same  (``+/- reward_normalization.normalized_return_bound``,
        config-derived in ``learner.py:732`` - never learned, never task-derived),
      * the reward function is the same: identical objective term names AND
        identical stage-0 weights, since the critic predicts THAT reward's return,
      * the critic's architecture and its context inputs match.

    Returns a list of fatal mismatches.
    """
    from flash_chord.training.checkpoint_metadata import checkpoint_config

    source = checkpoint_config(dict(source_metadata))
    target = checkpoint_config({k: v for k, v in target_metadata.items() if isinstance(v, str)})
    fatal: list[str] = []

    def dig(config, *path):
        node = config
        for key in path:
            if not isinstance(node, dict) or key not in node:
                return None
            node = node[key]
        return node

    bound_a = dig(source, "training", "reward_normalization", "normalized_return_bound")
    bound_b = dig(target, "training", "reward_normalization", "normalized_return_bound")
    if bound_a == bound_b and bound_a is not None:
        emit("  PASS  critic value support identical: +/-{} (normalized return units)".format(bound_a))
    else:
        fatal.append("value_support")
        emit("  FAIL  critic value support differs: {} vs {}".format(bound_a, bound_b))

    for key in ("atom_count", "critic_count", "critic_blocks", "critic_hidden_dim", "expansion"):
        a, b = dig(source, "training", "network", key), dig(target, "training", "network", key)
        if a != b:
            fatal.append("network." + key)
            emit("  FAIL  network.{} differs: {} vs {}".format(key, a, b))
    if not any(f.startswith("network.") for f in fatal):
        emit("  PASS  critic architecture identical (atoms, count, blocks, width, expansion)")

    # The critic predicts the return OF A REWARD FUNCTION. Same terms, same
    # stage-0 weights => the same quantity is being predicted on both episodes.
    stages_a = dig(source, "training", "curriculum", "stages")
    stages_b = dig(target, "training", "curriculum", "stages")
    if isinstance(stages_a, list) and isinstance(stages_b, list) and stages_a and stages_b:
        wa = (stages_a[0] or {}).get("objective_weights")
        wb = (stages_b[0] or {}).get("objective_weights")
        if wa == wb:
            emit("  PASS  stage-0 objective weights identical ({} terms) - the critic "
                 "predicts the same reward".format(len(wa or {})))
        else:
            fatal.append("objective_weights")
            emit("  FAIL  stage-0 objective weights differ")
        va = (stages_a[0] or {}).get("voc_scale")
        vb = (stages_b[0] or {}).get("voc_scale")
        if va != vb:
            fatal.append("voc_scale")
            emit("  FAIL  stage-0 voc_scale differs: {} vs {}".format(va, vb))
    else:
        emit("  ..    curriculum stages not present in metadata; weight check skipped")

    emit("  NOTE  the reward normalizer is RESET (mean 0, variance 1, count 0), so the")
    emit("        transferred critic is briefly miscalibrated until the target's running")
    emit("        moments adapt. Bounded, not a blow-up, but it is the known risk of")
    emit("        actor+critic transfer here.")
    return fatal


def leaf_paths(tree):
    """Map a pytree to {dotted path: numpy array} for exact leaf comparison."""
    import jax
    import numpy as np

    flat, _ = jax.tree_util.tree_flatten_with_path(tree)
    out = {}
    for path, value in flat:
        key = jax.tree_util.keystr(path)
        out[key] = np.asarray(value)
    return out


def main() -> int:
    args = parse_args()
    out_path = Path(args.out)
    emit = print

    import jax
    import numpy as np

    from flash_chord.training.checkpoint import load_checkpoint, read_checkpoint_metadata, save_checkpoint
    from flash_chord.training.flash_sac.checkpoint import (
        ALGORITHM_METADATA_KEY,
        ALGORITHM_NAME,
        CHECKPOINT_KIND_METADATA_KEY,
        CHECKPOINT_SET_METADATA_KEY,
        ENVIRONMENT_STEPS_METADATA_KEY,
        ActorInferenceState,
        load_actor_checkpoint,
        training_checkpoint,
    )

    source_path = Path(args.source_policy or args.source_state)
    source_kind = "policy" if args.source_policy else "training_state"

    emit("=== 1. build the TARGET episode's fresh learner ===")
    emit("  target parquet : {}".format(args.target_parquet))
    emit("  world count    : {}".format(args.world_count))
    cfg, training, setup, jax_env, target_metadata, fresh = build_target(args)
    emit("  target frames  : {}".format(setup.reference.num_frames))
    emit("  fresh learner  : environment_steps={} global_update_step={}".format(
        int(jax.device_get(fresh.environment_steps)), int(jax.device_get(fresh.global_update_step))))

    emit("")
    emit("=== 2. source checkpoint ===")
    source_metadata = read_checkpoint_metadata(source_path)
    emit("  file           : {}".format(source_path))
    emit("  kind           : {}".format(source_metadata.get(CHECKPOINT_KIND_METADATA_KEY)))
    emit("  algorithm      : {}".format(source_metadata.get(ALGORITHM_METADATA_KEY)))
    emit("  trained steps  : {}".format(source_metadata.get(ENVIRONMENT_STEPS_METADATA_KEY)))
    if source_metadata.get(CHECKPOINT_KIND_METADATA_KEY) != source_kind:
        emit("  ERROR: expected a {} checkpoint".format(source_kind))
        return 1

    emit("")
    emit("=== 3. schema compatibility (upstream's own resume validation) ===")
    from flash_chord.training.flash_sac.runner import _required_checkpoint_schemas, _validate_resume_schemas

    try:
        _validate_resume_schemas(source_metadata, *_required_checkpoint_schemas(target_metadata))
        emit("  PASS  policy and critic I/O schemas are compatible")
    except Exception as error:  # noqa: BLE001
        emit("  FAIL  {}: {}".format(type(error).__name__, error))
        return 1

    emit("")
    emit("=== 4. semantic compatibility beyond tensor shapes ===")
    fatal, informational = compare_semantics(source_metadata, target_metadata, emit)
    if fatal:
        emit("  REFUSING: the source actor was trained against different {}".format(", ".join(fatal)))
        return 1
    emit("  ..    expected-to-differ groups (this IS the transfer): {}".format(informational or "none"))

    if args.transfer == "actor_critic":
        emit("")
        emit("=== 4b. critic/reward compatibility (distributional critic) ===")
        critic_fatal = check_critic_compatibility(source_metadata, target_metadata, emit)
        if critic_fatal:
            emit("  REFUSING: a transferred critic would not be interpretable on the target "
                 "({})".format(", ".join(critic_fatal)))
            return 1

    emit("")
    emit("=== 5. splice the source weights into the fresh target learner ===")
    if source_kind == "policy":
        template = ActorInferenceState(params=fresh.actor.params, batch_stats=fresh.actor.batch_stats)
        restored_actor, _ = load_actor_checkpoint(source_path, template)
        source_params, source_batch_stats = restored_actor.params, restored_actor.batch_stats
        source_critic = None
    else:
        restored, _ = load_checkpoint(source_path, training_checkpoint(fresh))
        source_params, source_batch_stats = restored.actor.params, restored.actor.batch_stats
        source_critic = (restored.critic, restored.target_critic)

    merged = fresh.replace(
        actor=fresh.actor.replace(params=source_params, batch_stats=source_batch_stats)
    )
    transferred = ["actor.params", "actor.batch_stats"]
    if args.transfer == "actor_critic":
        if source_critic is None:
            emit("  ERROR: --transfer actor_critic needs --source-state (a policy file has no critic)")
            return 1
        critic, target_critic = source_critic
        merged = merged.replace(
            critic=merged.critic.replace(params=critic.params, batch_stats=critic.batch_stats),
            target_critic=target_critic,
        )
        transferred += ["critic.params", "critic.batch_stats", "target_critic"]
    emit("  transferring: {}".format(", ".join(transferred)))

    emit("")
    emit("=== 6. prove everything else is still FRESH ===")
    fresh_leaves = leaf_paths(training_checkpoint(fresh))
    merged_leaves = leaf_paths(training_checkpoint(merged))
    changed, identical = [], []
    for key in sorted(fresh_leaves):
        if fresh_leaves[key].shape != merged_leaves[key].shape:
            changed.append(key)
        elif np.array_equal(fresh_leaves[key], merged_leaves[key]):
            identical.append(key)
        else:
            changed.append(key)

    def group_of(key: str) -> str:
        parts = [p for p in key.replace("'", "").replace("[", ".").replace("]", "").split(".") if p]
        return parts[0] if parts else key

    changed_groups = sorted({group_of(k) for k in changed})
    emit("  leaves changed from fresh : {} (groups: {})".format(len(changed), changed_groups))
    emit("  leaves identical to fresh : {}".format(len(identical)))

    expected_groups = {"actor"} if args.transfer == "actor" else {"actor", "critic", "target_critic"}
    unexpected = set(changed_groups) - expected_groups
    if unexpected:
        emit("  FAIL  unexpected state changed: {}".format(sorted(unexpected)))
        return 1
    emit("  PASS  only {} differs from a fresh target learner".format(sorted(expected_groups)))

    # The counters and per-task state that make a resume not-a-transfer.
    checks = {
        "environment_steps": int(jax.device_get(merged.environment_steps)),
        "global_update_step": int(jax.device_get(merged.global_update_step)),
        "update_credit": float(jax.device_get(merged.update_credit)),
    }
    ok = True
    for name, value in checks.items():
        good = value == 0
        ok = ok and good
        emit("  {}  {:22s} = {}".format("PASS" if good else "FAIL", name, value))
    for name, group in (("reward_normalizer", "reward_normalizer"), ("exploration", "exploration"),
                        ("loss_scale", "loss_scale"), ("temperature", "temperature")):
        good = group not in changed_groups
        ok = ok and good
        emit("  {}  {:22s} is the target's own fresh state".format("PASS" if good else "FAIL", name))
    emit("  PASS  replay buffer is not part of a training_state checkpoint at all")
    if not ok:
        emit("  REFUSING to write a checkpoint that is not a clean weight-only transfer")
        return 1

    emit("")
    emit("=== 7. write the TARGET's training_state checkpoint ===")
    metadata = dict(target_metadata)
    metadata[ALGORITHM_METADATA_KEY] = ALGORITHM_NAME
    metadata[CHECKPOINT_KIND_METADATA_KEY] = "training_state"
    metadata[CHECKPOINT_SET_METADATA_KEY] = uuid.uuid4().hex
    metadata[ENVIRONMENT_STEPS_METADATA_KEY] = 0
    # Provenance: this file is NOT a checkpoint of a run that happened.
    metadata["transfer_source_checkpoint"] = str(source_path)
    metadata["transfer_source_kind"] = source_kind
    metadata["transfer_source_environment_steps"] = str(
        source_metadata.get(ENVIRONMENT_STEPS_METADATA_KEY)
    )
    metadata["transfer_mode"] = args.transfer
    metadata["transfer_world_count"] = str(args.world_count)
    # ``save_learner_checkpoint`` adds these at save time from the live stage and
    # env; ``evaluate_checkpoint`` REQUIRES curriculum_stage and raises
    # "FlashSAC checkpoint is missing its curriculum stage" without it. This file
    # represents step 0, so it is stage 0 - taken from the target's own
    # curriculum rather than hard-coded.
    if training.curriculum is not None:
        stage_index = training.curriculum.stage_index(0)
        metadata["curriculum_stage"] = stage_index
        metadata["voc_scale"] = training.curriculum.stages[stage_index].voc_scale
    reset_probability = getattr(jax_env, "reset_to_first_frame_probability", None)
    if reset_probability is not None:
        metadata["reset_to_first_frame_probability"] = reset_probability
    immediate_probability = getattr(jax_env, "immediate_first_frame_probability", None)
    if immediate_probability is not None:
        metadata["immediate_first_frame_probability"] = immediate_probability
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(out_path, training_checkpoint(merged), metadata=metadata)
    emit("  wrote {} ({:.1f} MiB)".format(out_path, out_path.stat().st_size / (1024 * 1024)))

    if args.emit_policy:
        # Same actor weights, but the TARGET's resolved config - so
        # evaluate_policy.py builds the target episode's scene and reference
        # while running the source's policy. That is the zero-shot measurement.
        from flash_chord.training.flash_sac.checkpoint import save_actor_checkpoint

        policy_metadata = dict(metadata)
        policy_metadata[CHECKPOINT_KIND_METADATA_KEY] = "policy"
        policy_path = Path(args.emit_policy)
        policy_path.parent.mkdir(parents=True, exist_ok=True)
        save_actor_checkpoint(
            policy_path,
            ActorInferenceState(params=merged.actor.params, batch_stats=merged.actor.batch_stats),
            metadata=policy_metadata,
        )
        emit("  wrote {} ({:.1f} MiB)  <- zero-shot evaluation input".format(
            policy_path, policy_path.stat().st_size / (1024 * 1024)))

    report = {
        "source_checkpoint": str(source_path),
        "emitted_policy": args.emit_policy,
        "source_kind": source_kind,
        "source_environment_steps": source_metadata.get(ENVIRONMENT_STEPS_METADATA_KEY),
        "target_parquet": str(setup.parquet_path),
        "target_frames": int(setup.reference.num_frames),
        "world_count": args.world_count,
        "transfer_mode": args.transfer,
        "transferred_leaves": transferred,
        "leaves_changed_from_fresh": len(changed),
        "leaves_identical_to_fresh": len(identical),
        "changed_groups": changed_groups,
        "counters": checks,
        "semantic_groups_compared": list(SEMANTIC_GROUPS),
        "semantic_groups_differing": fatal,
        "out": str(out_path),
    }
    if args.report_json:
        with open(args.report_json, "w") as handle:
            json.dump(report, handle, indent=2)
        emit("  wrote {}".format(args.report_json))
    emit("")
    emit("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
