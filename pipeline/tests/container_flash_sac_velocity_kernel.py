"""Container test: the object-velocity Warp kernel against a NumPy/SciPy host reference.

Runs inside the flash_chord image only (needs Warp, SciPy and ``flash_chord``).
Warp's CPU backend is enough, so no GPU allocation is needed:

    apptainer exec --cleanenv -B "$R":/r -B "$FC":/workspace --pwd /r \
        --env PYTHONPATH=/r:/workspace/src containers/flash_chord.sif \
        python -m pytest -q -p no:cacheprovider pipeline/tests/container_flash_sac_velocity_kernel.py
"""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp
from scipy.spatial.transform import Rotation

from pipeline.flash_sac_keypoints import KEYPOINT_RECON_BODY_OBJECTIVE_TERM_NAMES
from pipeline.flash_sac_velocity import (
    VELOCITY_RECON_BODY_OBJECTIVE_TERM_NAMES,
    VelocityReconBodyObjectiveConfig,
    append_object_velocity,
)

BASE_TERMS = len(KEYPOINT_RECON_BODY_OBJECTIVE_TERM_NAMES)
ALL_TERMS = len(VELOCITY_RECON_BODY_OBJECTIVE_TERM_NAMES)
FRAME_DT = 0.02
DEVICE = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"


def random_quats(count: int, rng: np.random.Generator) -> np.ndarray:
    q = rng.normal(size=(count, 4))
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def host_velocity_term(world, linear_scale, angular_scale):
    """Independent reference: SciPy rotations, NumPy differences, no Warp."""
    rotation = Rotation.from_quat(world["quat"])  # xyzw, as Warp
    linear = world["linear_com"] - np.cross(world["angular"], rotation.apply(world["com"]))
    frames = len(world["reference_position"])
    frame = np.clip(world["timestep"], 0, frames - 1)
    previous, following = np.maximum(frame - 1, 0), np.minimum(frame + 1, frames - 1)
    span = ((following - previous) * FRAME_DT)[:, None]
    reference_linear = (world["reference_position"][following] - world["reference_position"][previous]) / span
    relative = Rotation.from_quat(world["reference_quat"][following]) * Rotation.from_quat(
        world["reference_quat"][previous]
    ).inv()
    reference_angular = relative.as_rotvec() / span
    return 0.5 * (
        np.exp(-np.linalg.norm(linear - reference_linear, axis=1) / linear_scale)
        + np.exp(-np.linalg.norm(world["angular"] - reference_angular, axis=1) / angular_scale)
    )


def run_kernel(world, weights, linear_scale, angular_scale):
    count = len(world["timestep"])
    transforms = np.concatenate([world["position"], world["quat"]], axis=1).astype(np.float32)
    velocities = np.concatenate([world["linear_com"], world["angular"]], axis=1).astype(np.float32)
    with wp.ScopedDevice(DEVICE):
        terms = wp.zeros(count * ALL_TERMS, dtype=wp.float32)
        score = wp.zeros(count, dtype=wp.float32)
        wp.launch(
            append_object_velocity,
            dim=count,
            inputs=[
                wp.array(world["base_terms"].reshape(-1).astype(np.float32), dtype=wp.float32),
                wp.array(transforms, dtype=wp.transform),
                wp.array(velocities, dtype=wp.spatial_vector),
                wp.array(world["com"].astype(np.float32), dtype=wp.vec3),
                wp.array(world["timestep"].astype(np.int32), dtype=wp.int32),
                wp.array(world["reference_position"].astype(np.float32), dtype=wp.vec3),
                wp.array(world["reference_quat"].astype(np.float32), dtype=wp.quat),
                0,  # object body id
                1,  # bodies per world
                len(world["reference_position"]),
                linear_scale,
                angular_scale,
                wp.array(weights.astype(np.float32), dtype=wp.float32),
                FRAME_DT,
            ],
            outputs=[terms, score],
        )
        return terms.numpy().reshape(count, ALL_TERMS), score.numpy()


def smooth_reference(frames: int, rng: np.random.Generator):
    """A plausible object path: 1 m/s-scale motion, small rotation steps per frame."""
    position = np.cumsum(rng.normal(scale=0.01, size=(frames, 3)), axis=0)
    steps = Rotation.from_rotvec(rng.normal(scale=0.03, size=(frames, 3)))
    rotation = [Rotation.identity()]
    for step in steps[1:]:
        rotation.append(step * rotation[-1])
    return position, Rotation.concatenate(rotation).as_quat()


@pytest.fixture
def world():
    rng = np.random.default_rng(0)
    n, frames = 256, 40
    reference_position, reference_quat = smooth_reference(frames, rng)
    # Flip the sign of half the stored quaternions: the kernel must be double-cover aware.
    reference_quat[::2] *= -1.0
    return {
        "base_terms": rng.uniform(-1.0, 1.0, size=(n, BASE_TERMS)),
        "position": rng.normal(scale=0.5, size=(n, 3)),
        "quat": random_quats(n, rng),
        "linear_com": rng.normal(scale=0.6, size=(n, 3)),
        "angular": rng.normal(scale=2.0, size=(n, 3)),
        "com": rng.normal(scale=0.1, size=(n, 3)),
        # include frames before the start and past the end to exercise clamping
        "timestep": rng.integers(-3, frames + 3, size=n),
        "reference_position": reference_position,
        "reference_quat": reference_quat,
    }


@pytest.mark.parametrize(("linear_scale", "angular_scale"), [(0.5, 4.0), (0.2, 1.0)])
def test_kernel_matches_host_reference(world, linear_scale: float, angular_scale: float) -> None:
    weights = np.random.default_rng(1).uniform(-2.0, 5.0, size=ALL_TERMS)
    terms, _ = run_kernel(world, weights, linear_scale, angular_scale)
    expected = host_velocity_term(world, linear_scale, angular_scale)
    np.testing.assert_allclose(terms[:, BASE_TERMS], expected, rtol=1e-4, atol=2e-5)


def test_base_terms_pass_through_and_score_recomposes(world) -> None:
    weights = np.random.default_rng(2).uniform(-2.0, 5.0, size=ALL_TERMS)
    terms, score = run_kernel(world, weights, 0.5, 4.0)
    np.testing.assert_array_equal(terms[:, :BASE_TERMS], world["base_terms"].astype(np.float32))
    np.testing.assert_allclose(score, FRAME_DT * terms @ weights.astype(np.float32), rtol=1e-5, atol=1e-6)


def test_matching_the_reference_velocity_scores_one(world) -> None:
    """An object moving exactly like the reference, measured at its COM, scores 1.0."""
    frames = len(world["reference_position"])
    frame = np.clip(world["timestep"], 0, frames - 1)
    previous, following = np.maximum(frame - 1, 0), np.minimum(frame + 1, frames - 1)
    span = ((following - previous) * FRAME_DT)[:, None]
    reference_linear = (world["reference_position"][following] - world["reference_position"][previous]) / span
    reference_angular = (
        Rotation.from_quat(world["reference_quat"][following])
        * Rotation.from_quat(world["reference_quat"][previous]).inv()
    ).as_rotvec() / span
    lever = Rotation.from_quat(world["quat"]).apply(world["com"])
    matched = {
        **world,
        "angular": reference_angular,
        "linear_com": reference_linear + np.cross(reference_angular, lever),
    }
    terms, _ = run_kernel(matched, np.ones(ALL_TERMS), 0.5, 4.0)
    np.testing.assert_allclose(terms[:, BASE_TERMS], 1.0, atol=1e-4)


def test_a_falling_object_under_a_static_reference_loses_reward(world) -> None:
    static = {
        **world,
        "reference_position": np.zeros_like(world["reference_position"]),
        "reference_quat": np.tile([0.0, 0.0, 0.0, 1.0], (len(world["reference_quat"]), 1)),
        "angular": np.zeros_like(world["angular"]),
        "com": np.zeros_like(world["com"]),
    }
    still, _ = run_kernel({**static, "linear_com": np.zeros_like(world["linear_com"])}, np.ones(ALL_TERMS), 0.5, 4.0)
    falling = np.tile([0.0, 0.0, -1.0], (len(world["timestep"]), 1))  # 1 m/s down, ~0.1 s into a drop
    dropped, _ = run_kernel({**static, "linear_com": falling}, np.ones(ALL_TERMS), 0.5, 4.0)
    np.testing.assert_allclose(still[:, BASE_TERMS], 1.0, atol=1e-6)
    np.testing.assert_allclose(dropped[:, BASE_TERMS], 0.5 * (np.exp(-2.0) + 1.0), rtol=1e-5)


def test_config_defaults_and_weight_vector() -> None:
    config = VelocityReconBodyObjectiveConfig()
    assert (config.object_velocity_weight, config.object_velocity_linear_scale) == (2.0, 0.5)
    assert config.object_velocity_angular_scale == 4.0
    assert config.object_keypoints_weight == 5.0
    assert len(config.all_weights) == ALL_TERMS and config.all_weights[-2:] == (5.0, 2.0)
    assert VELOCITY_RECON_BODY_OBJECTIVE_TERM_NAMES[:BASE_TERMS] == KEYPOINT_RECON_BODY_OBJECTIVE_TERM_NAMES


@pytest.mark.parametrize(
    "bad",
    [
        {"object_velocity_weight": -1.0},
        {"object_velocity_linear_scale": 0.0},
        {"object_velocity_angular_scale": float("nan")},
    ],
)
def test_config_rejects_bad_values(bad: dict) -> None:
    with pytest.raises(ValueError):
        VelocityReconBodyObjectiveConfig(**bad)
