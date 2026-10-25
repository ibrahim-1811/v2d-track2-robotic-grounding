"""Container test: the keypoint Warp kernel against its NumPy host reference.

Runs inside the flash_chord image only (needs Warp and ``flash_chord``). Warp's
CPU backend is enough, so no GPU allocation is needed:

    apptainer exec --cleanenv -B "$R":/r -B "$FC":/workspace --pwd /r \
        --env PYTHONPATH=/r:/workspace/src containers/flash_chord.sif \
        python -m pytest -q -p no:cacheprovider pipeline/tests/container_flash_sac_keypoints_kernel.py
"""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from flash_chord.objectives.keypoints import KEYPOINT_VECS_NP
from flash_chord.objectives.recon_body import RECON_BODY_OBJECTIVE_TERM_NAMES
from pipeline.flash_sac_keypoints import (
    KEYPOINT_RECON_BODY_OBJECTIVE_TERM_NAMES,
    KeypointReconBodyObjectiveConfig,
    append_object_keypoints,
)
from pipeline.keypoint_reward import KEYPOINT_AXES, object_keypoints_raw_term

BASE_TERMS = len(RECON_BODY_OBJECTIVE_TERM_NAMES)
ALL_TERMS = len(KEYPOINT_RECON_BODY_OBJECTIVE_TERM_NAMES)
FRAME_DT = 0.02
DEVICE = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"


def random_quats(count: int, rng: np.random.Generator) -> np.ndarray:
    q = rng.normal(size=(count, 4))
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def run_kernel(base_terms, position, quat, target_position, target_quat, radius, weights, fraction, scale):
    world_count = position.shape[0]
    transforms = np.concatenate([position, quat], axis=1).astype(np.float32)
    with wp.ScopedDevice(DEVICE):
        terms = wp.zeros(world_count * ALL_TERMS, dtype=wp.float32)
        score = wp.zeros(world_count, dtype=wp.float32)
        wp.launch(
            append_object_keypoints,
            dim=world_count,
            inputs=[
                wp.array(base_terms.reshape(-1).astype(np.float32), dtype=wp.float32),
                wp.array(transforms, dtype=wp.transform),
                wp.array(target_position.astype(np.float32), dtype=wp.vec3),
                wp.array(target_quat.astype(np.float32), dtype=wp.quat),
                wp.array(radius.astype(np.float32), dtype=wp.float32),
                wp.array(KEYPOINT_VECS_NP, dtype=wp.vec3),
                0,  # object body id
                1,  # bodies per world
                1,  # object bodies
                fraction,
                scale,
                wp.array(weights.astype(np.float32), dtype=wp.float32),
                FRAME_DT,
            ],
            outputs=[terms, score],
        )
        return terms.numpy().reshape(world_count, ALL_TERMS), score.numpy()


@pytest.fixture
def batch():
    rng = np.random.default_rng(0)
    n = 256
    return {
        "base_terms": rng.uniform(-1.0, 1.0, size=(n, BASE_TERMS)),
        "position": rng.normal(scale=0.05, size=(n, 3)),
        "quat": random_quats(n, rng),
        "target_position": rng.normal(scale=0.05, size=(n, 3)),
        "target_quat": random_quats(n, rng),
        "radius": rng.uniform(0.05, 0.9, size=n),
        "weights": rng.uniform(-2.0, 5.0, size=ALL_TERMS),
    }


def test_upstream_keypoint_axes_are_the_host_reference_axes() -> None:
    np.testing.assert_array_equal(KEYPOINT_VECS_NP, KEYPOINT_AXES)


@pytest.mark.parametrize(("fraction", "scale"), [(0.5, 0.05), (1.0, 0.1)])
def test_kernel_matches_host_reference(batch, fraction: float, scale: float) -> None:
    terms, _ = run_kernel(**batch, fraction=fraction, scale=scale)
    expected = object_keypoints_raw_term(
        batch["position"], batch["quat"], batch["target_position"], batch["target_quat"],
        batch["radius"], radius_fraction=fraction, scale=scale,
    )
    np.testing.assert_allclose(terms[:, BASE_TERMS], expected, rtol=2e-5, atol=2e-6)


def test_upstream_terms_pass_through_and_score_recomposes(batch) -> None:
    terms, score = run_kernel(**batch, fraction=0.5, scale=0.05)
    np.testing.assert_array_equal(terms[:, :BASE_TERMS], batch["base_terms"].astype(np.float32))
    np.testing.assert_allclose(score, FRAME_DT * terms @ batch["weights"].astype(np.float32), rtol=1e-5, atol=1e-6)


def test_on_target_scores_one(batch) -> None:
    batch = {**batch, "target_position": batch["position"], "target_quat": batch["quat"]}
    terms, _ = run_kernel(**batch, fraction=0.5, scale=0.05)
    np.testing.assert_allclose(terms[:, BASE_TERMS], 1.0, atol=1e-6)


def test_config_defaults_and_weight_vector() -> None:
    config = KeypointReconBodyObjectiveConfig()
    assert (config.object_keypoints_weight, config.object_keypoints_scale) == (5.0, 0.05)
    assert config.object_keypoints_radius_fraction == 0.5
    assert len(config.all_weights) == ALL_TERMS and config.all_weights[-1] == 5.0
    assert KEYPOINT_RECON_BODY_OBJECTIVE_TERM_NAMES[:BASE_TERMS] == RECON_BODY_OBJECTIVE_TERM_NAMES


@pytest.mark.parametrize(
    "bad", [{"object_keypoints_scale": 0.0}, {"object_keypoints_radius_fraction": -0.1}, {"object_keypoints_weight": -1}]
)
def test_config_rejects_bad_values(bad: dict) -> None:
    with pytest.raises(ValueError):
        KeypointReconBodyObjectiveConfig(**bad)
