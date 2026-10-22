#!/usr/bin/env python3
"""Instrumented, unassisted world-1 rollout: why Episode 0 fails at frame ~323.

The official evaluator records only the object pose, so nothing in
``outputs/step_D/trajectories`` can say whether the robot was in contact, where
its palms were, or what it commanded. This reproduces the evaluator's rollout
**exactly** -- same ``evaluate_checkpoint`` setup path, frame-zero explicit
reset, zero VOC, no settling, deterministic actor -- and additionally reads
back, every control step:

* object pose and its reference target,
* both palm poses and the reference palm targets,
* the packed termination diagnostics (palm/pelvis/object position and
  orientation error), which upstream already computes but does not export,
* live contact from ``ContactTracker``: per-hand raw contact count, summed
  normal force magnitude, and contact position,
* the action vector's magnitude.

Every read is read-only. The script re-scores the recorded object trajectory
with the evaluator's own AUC/MPPE code and prints it next to the uninstrumented
evaluation, so any behavioural effect of instrumentation is visible rather than
assumed.

Runs inside ``flash_chord:latest`` with ``pipeline`` on ``PYTHONPATH``.
Upstream is imported, never modified.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import warp as wp


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--start-frame", type=int, default=0)
    args = parser.parse_args()

    import jax
    import jax.numpy as jnp
    from omegaconf import OmegaConf

    from flash_chord.configuration import instantiate_typed
    from flash_chord.embodiments.base import Embodiment
    from flash_chord.envs.rl import RLEnv, RLEnvConfig
    from flash_chord.evaluation.harness import (
        apply_evaluation_composition,
        _evaluation_curriculum_stage,
        _resolve_parquet,
    )
    from flash_chord.scene.collision import CollisionPolicy
    from flash_chord.scene.setup import setup_scene
    from flash_chord.training.checkpoint_metadata import checkpoint_config
    from flash_chord.training.environment import WarpRLEnv
    from flash_chord.training.flash_sac.evaluation import load_policy_for_inference as load_flash_sac_policy
    from flash_chord.training.flash_sac.config import TrainingConfig
    from flash_chord.training.checkpoint import read_checkpoint_metadata
    from flash_chord.training.flash_sac.runner import resolve_jax_device

    checkpoint_path = Path(args.checkpoint).resolve()
    metadata = read_checkpoint_metadata(checkpoint_path)
    config = checkpoint_config(metadata)

    # Exactly the evaluator's composition: world 1, explicit reset, zero VOC.
    apply_evaluation_composition(config, world_count=1, reset_mode="explicit")
    config["task"]["parquet"] = _resolve_parquet(config, None)
    cfg = OmegaConf.create(config)

    env_config = instantiate_typed(cfg.env, RLEnvConfig)
    embodiment = instantiate_typed(cfg.embodiment, Embodiment)
    collision = instantiate_typed(cfg.collision, CollisionPolicy)
    training = instantiate_typed(cfg.training, TrainingConfig)
    device = resolve_jax_device(training.device)

    setup = setup_scene(
        parquet=cfg.task.parquet,
        control_fps=env_config.sim.fps,
        motion_speed=cfg.task.motion_speed,
        embodiment=embodiment,
        collision=collision,
        world_count=cfg.scene.world_count,
        include_support=cfg.scene.support,
        decompose_objects=cfg.scene.decompose_objects,
        object_scale_min=cfg.scene.get("object_scale_min", 1.0),
        object_scale_max=cfg.scene.get("object_scale_max", 1.0),
        object_scale_seed=cfg.scene.get("object_scale_seed", 0),
        contact_friction=cfg.scene.get("contact_friction", 1.0),
        object_free_joint_damping=cfg.scene.get("object_free_joint_damping", 0.0),
        source_frame_playback=bool(cfg.task.get("source_frame_playback", False)),
        motion_start_frame=int(cfg.task.get("motion_start_frame", 0)),
        motion_end_frame=int(cfg.task.get("motion_end_frame", -1)),
    )
    reference = setup.reference
    step_count = reference.num_frames - args.start_frame

    env = RLEnv(setup.scene, reference, config=env_config)
    env.reset()
    env.capture_step()
    jax_env = WarpRLEnv(env, publish_diagnostics=True, publish_step_metrics=True)
    if training.curriculum is not None:
        stage_index = int(metadata["curriculum_stage"])
        jax_env.set_curriculum(_evaluation_curriculum_stage(training.curriculum.stages[stage_index], "explicit"))

    env.reset(frame_id=args.start_frame)
    observation = jax.device_put(
        jnp.asarray(env.observation.numpy().reshape(env.world_count, env.observation_dim)), device
    )
    actor_state, policy_action, key, _ = load_flash_sac_policy(
        str(checkpoint_path), training, observation, env.action.action_dim, deterministic=True, device=device
    )

    termination = env.termination
    error_names = tuple(termination.error_names)
    cause_names = tuple(termination.cause_names)
    contact = env.contact
    layout = contact.layout
    num_hands = layout.num_hands
    slots = layout.slots_per_world
    object_body_id = env.command.layout.body_ids[0]
    bodies_per_world = env.model.body_count // env.world_count
    palm_body_ids = [int(i) for i in contact.wrist_frames.body_ids.numpy()[:num_hands]]

    print(f"world_count={env.world_count} steps={step_count} hands={num_hands} slots/world={slots}", flush=True)
    print(f"error_names={error_names}", flush=True)
    print(f"cause_names={cause_names}", flush=True)
    print(f"palm_body_ids={palm_body_ids} object_body_id={object_body_id}", flush=True)

    records: list[dict] = []
    for step in range(step_count):
        action, key = policy_action(actor_state, observation, key)
        observation = jax_env.step(action).observation

        body_q = env.state_0.body_q.numpy()
        diagnostics = termination.packed_diagnostics.numpy().reshape(env.world_count, -1)[0]
        counts = contact.raw_contact_count.numpy().reshape(env.world_count, slots)[0]
        forces = contact.contact_force_w.numpy().reshape(env.world_count, slots, 3)[0]
        positions = contact.contact_pos_w.numpy().reshape(env.world_count, slots, 3)[0]
        active = contact.contact_active.numpy().reshape(env.world_count, slots)[0]

        row: dict = {"step": step, "frame": args.start_frame + step}
        row["object_pos"] = body_q[object_body_id][:3].tolist()
        row["object_quat"] = body_q[object_body_id][3:7].tolist()
        row["object_target_pos"] = env.command.body_target_pos_w.numpy()[0].tolist()
        row["object_target_quat"] = env.command.body_target_quat_w.numpy()[0].tolist()
        for hand in range(num_hands):
            row[f"palm{hand}_pos"] = body_q[palm_body_ids[hand]][:3].tolist()
            lo = layout.hand_slot_starts[hand]
            hi = lo + layout.link_counts[hand] * layout.num_objects
            row[f"hand{hand}_raw_contacts"] = int(counts[lo:hi].sum())
            row[f"hand{hand}_active_slots"] = int(active[lo:hi].sum())
            row[f"hand{hand}_force_n"] = float(np.linalg.norm(forces[lo:hi].sum(axis=0)))
            row[f"hand{hand}_force_sum_abs_n"] = float(np.linalg.norm(forces[lo:hi], axis=1).sum())
            touched = np.flatnonzero(counts[lo:hi] > 0)
            row[f"hand{hand}_contact_pos"] = (
                positions[lo:hi][touched].mean(axis=0).tolist() if touched.size else [0.0, 0.0, 0.0]
            )
        for index, name in enumerate(cause_names + error_names):
            row[name] = float(diagnostics[index])
        act = np.asarray(jax.device_get(action)).reshape(-1)
        row["action_absmax"] = float(np.abs(act).max())
        row["action_l2"] = float(np.linalg.norm(act))
        row["done"] = int(env.done.numpy()[0])
        records.append(row)
        if step % 100 == 0:
            print(f"  step {step}/{step_count}", flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"checkpoint": str(checkpoint_path), "records": records}) + "\n")
    print(f"wrote {output} ({len(records)} records)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
