#!/usr/bin/env python3
"""Where does the G1's visible vibration come from: the reference, SONIC, or our policy?

Same evaluator composition as ``diagnose_contact_rollout.py`` (world 1, explicit frame-zero
reset, zero VOC, deterministic actor). Every control step it reads back, for the 29 SONIC
body joints:

* ``reference``  -- the retargeted G1 reference joint target at that frame (SOMA -> G1 output),
* ``decoded``    -- SONIC's decoder output before the policy residual,
* ``target``     -- the PD target actually sent (SONIC + 0.15 x policy residual),
* ``q``          -- the simulated joint position,
* ``action``     -- the policy's action as the env receives it (body residuals + 14 finger
  residuals; FlashSAC's G1 recipe emits 10 x tanh, so up to +/-10),
* ``finger_q`` / ``finger_target`` -- the 14 Dex3 finger joints and their PD targets.

``--zero-action`` replaces the policy with zeros, i.e. SONIC tracking the reference alone, so
the two runs separate SONIC's own jitter from the residual's. Vibration is measured as the
RMS of each signal minus its 5-step (100 ms) centred moving average: slow, intended motion
cancels, the step-to-step shake remains. Upstream is imported, never modified.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def vibration_deg(signal: np.ndarray, window: int = 5) -> np.ndarray:
    """Per-joint RMS (deg) of ``signal`` [T, J] minus its centred moving average."""
    kernel = np.ones(window) / window
    pad = window // 2
    padded = np.pad(signal, ((pad, pad), (0, 0)), mode="edge")
    smooth = np.stack([np.convolve(padded[:, j], kernel, mode="valid") for j in range(signal.shape[1])], axis=1)
    return np.degrees(np.sqrt(np.mean((signal - smooth) ** 2, axis=0)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True, help=".npz with the per-step arrays; a .json summary is written beside it")
    parser.add_argument("--zero-action", action="store_true", help="drive SONIC with a zero residual instead of the policy")
    args = parser.parse_args()

    import jax
    import jax.numpy as jnp
    from omegaconf import OmegaConf

    from flash_chord.configuration import instantiate_typed
    from flash_chord.embodiments.base import Embodiment
    from flash_chord.envs.rl import RLEnv, RLEnvConfig
    from flash_chord.evaluation.harness import (
        _evaluation_curriculum_stage,
        _resolve_parquet,
        apply_evaluation_composition,
    )
    from flash_chord.scene.collision import CollisionPolicy
    from flash_chord.scene.setup import setup_scene
    from flash_chord.training.checkpoint import read_checkpoint_metadata
    from flash_chord.training.checkpoint_metadata import checkpoint_config
    from flash_chord.training.environment import WarpRLEnv
    from flash_chord.training.flash_sac.config import TrainingConfig
    from flash_chord.training.flash_sac.evaluation import load_policy_for_inference as load_flash_sac_policy
    from flash_chord.training.flash_sac.runner import resolve_jax_device

    checkpoint_path = Path(args.checkpoint).resolve()
    metadata = read_checkpoint_metadata(checkpoint_path)
    config = checkpoint_config(metadata)
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
    env = RLEnv(setup.scene, reference, config=env_config)
    env.reset()
    env.capture_step()
    jax_env = WarpRLEnv(env)
    if training.curriculum is not None:
        stage_index = int(metadata["curriculum_stage"])
        jax_env.set_curriculum(_evaluation_curriculum_stage(training.curriculum.stages[stage_index], "explicit"))

    env.reset(frame_id=0)
    observation = jax.device_put(
        jnp.asarray(env.observation.numpy().reshape(env.world_count, env.observation_dim)), device
    )
    actor_state, policy_action, key, _ = load_flash_sac_policy(
        str(checkpoint_path), training, observation, env.action.action_dim, deterministic=True, device=device
    )

    sonic = env.action
    q_ids = sonic.sonic_q_ids.numpy()
    dof_ids = sonic.sonic_dof_ids.numpy()
    finger_q_ids = sonic.finger_q_ids.numpy()
    finger_dof_ids = sonic.finger_dof_ids.numpy()
    reference_targets = sonic.reference_joint_target.numpy()
    object_body_id = env.command.layout.body_ids[0]
    steps = reference.num_frames
    names = tuple(sonic.residual_joint_names)
    rec = {
        k: []
        for k in ("reference", "decoded", "target", "q", "action", "finger_q", "finger_target", "object_pos", "object_target_pos")
    }
    for step in range(steps):
        frame = int(env.timestep.numpy()[0])
        action, key = policy_action(actor_state, observation, key)
        if args.zero_action:
            action = jnp.zeros_like(action)
        observation = jax_env.step(action).observation
        rec["reference"].append(reference_targets[min(frame, len(reference_targets) - 1), dof_ids])
        rec["decoded"].append(sonic.decoder_action.numpy()[0])
        rec["target"].append(sonic.joint_target.numpy()[dof_ids])
        joint_q = env.state_0.joint_q.numpy()
        rec["q"].append(joint_q[q_ids])
        rec["finger_q"].append(joint_q[finger_q_ids])
        rec["finger_target"].append(sonic.joint_target.numpy()[finger_dof_ids])
        rec["action"].append(np.asarray(jax.device_get(action)).reshape(-1))
        rec["object_pos"].append(env.state_0.body_q.numpy()[object_body_id][:3])
        rec["object_target_pos"].append(env.command.body_target_pos_w.numpy()[0])
    arrays = {k: np.asarray(v, dtype=np.float64) for k, v in rec.items()}

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, joint_names=np.asarray(names), finger_names=np.asarray(sonic.finger_joint_names), **arrays)

    action = arrays["action"]
    summary = {
        "checkpoint": str(checkpoint_path),
        "zero_action": args.zero_action,
        "steps": steps,
        "vibration_deg_mean": {k: float(vibration_deg(arrays[k]).mean()) for k in ("reference", "decoded", "target", "q")},
        "vibration_deg_per_joint_q": dict(zip(names, np.round(vibration_deg(arrays["q"]), 3).tolist())),
        "vibration_deg_per_joint_reference": dict(zip(names, np.round(vibration_deg(arrays["reference"]), 3).tolist())),
        "action_abs_mean": float(np.abs(action).mean()),
        "action_saturated_fraction": float((np.abs(action) > 9.5).mean()),
        "finger_q_range_rad": [float(arrays["finger_q"].min()), float(arrays["finger_q"].max())],
        "finger_q_abs_mean_rad": float(np.abs(arrays["finger_q"]).mean()),
        "action_step_change_mean": float(np.abs(np.diff(action, axis=0)).mean()),
        "action_sign_flip_fraction": float((np.sign(action[1:]) != np.sign(action[:-1])).mean()),
        "object_error_cm_mean": float(100 * np.linalg.norm(arrays["object_pos"] - arrays["object_target_pos"], axis=1).mean()),
    }
    output.with_suffix(".json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if not k.startswith("vibration_deg_per")}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
