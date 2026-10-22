"""Container tests for the orientation Warp kernel and its replay integration.

Runs inside ``flash_chord:latest`` only (needs Warp, JAX and ``flash_chord``).
The host reference is :mod:`pipeline.orientation_reward`, which has its own
container-free test suite.

    docker run ... python -m pytest /v2d/pipeline/pipeline/tests/test_flash_sac_orientation_kernel.py
"""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from pipeline.flash_sac_orientation import (
    ORIENTATION_RECON_BODY_OBJECTIVE_TERM_NAMES,
    OrientationReconBodyObjectiveConfig,
    append_object_orientation,
)
from pipeline.orientation_reward import object_orientation_raw_term

from flash_chord.objectives.recon_body import RECON_BODY_OBJECTIVE_TERM_NAMES

BASE_TERMS = len(RECON_BODY_OBJECTIVE_TERM_NAMES)
ALL_TERMS = len(ORIENTATION_RECON_BODY_OBJECTIVE_TERM_NAMES)
FRAME_DT = 0.02


def random_quats(count: int, seed: int) -> np.ndarray:
    """Uniform random unit quaternions, xyzw, without SciPy."""
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(count, 4))
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def run_kernel(
    base_terms: np.ndarray,
    object_quats: np.ndarray,
    target_quats: np.ndarray,
    weights: np.ndarray,
    device: str = "cuda:0",
):
    """Launch the kernel with synthetic inputs; returns (terms, score) as numpy."""
    world_count = object_quats.shape[0]
    # One body per world, object at local body index 0, so body_q is world-major.
    transforms = np.zeros((world_count, 7), dtype=np.float32)
    transforms[:, 3:7] = object_quats
    with wp.ScopedDevice(device):
        terms = wp.zeros(world_count * ALL_TERMS, dtype=wp.float32)
        score = wp.zeros(world_count, dtype=wp.float32)
        wp.launch(
            append_object_orientation,
            dim=world_count,
            inputs=[
                wp.array(base_terms.reshape(-1).astype(np.float32), dtype=wp.float32),
                wp.array(transforms, dtype=wp.transform),
                wp.array(target_quats.astype(np.float32), dtype=wp.quat),
                0,
                1,
                1,
                wp.array(weights.astype(np.float32), dtype=wp.float32),
                FRAME_DT,
            ],
            outputs=[terms, score],
        )
        wp.synchronize()
        return terms.numpy().reshape(world_count, ALL_TERMS), score.numpy()


# --- naming and schema ------------------------------------------------------


def test_term_names_extend_upstream_without_reordering() -> None:
    assert ORIENTATION_RECON_BODY_OBJECTIVE_TERM_NAMES[:BASE_TERMS] == RECON_BODY_OBJECTIVE_TERM_NAMES
    assert ORIENTATION_RECON_BODY_OBJECTIVE_TERM_NAMES[-1] == "object_orientation"
    assert ALL_TERMS == 13


def test_config_keeps_twelve_weights_and_exposes_thirteen() -> None:
    config = OrientationReconBodyObjectiveConfig()
    assert len(config.weights) == BASE_TERMS
    assert config.weights == OrientationReconBodyObjectiveConfig.__mro__[1]().weights
    assert len(config.all_weights) == ALL_TERMS
    assert config.all_weights[-1] == 2.0


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0])
def test_config_rejects_invalid_orientation_weight(bad: float) -> None:
    with pytest.raises(ValueError):
        OrientationReconBodyObjectiveConfig(object_orientation_weight=bad)


# --- kernel math vs the host reference --------------------------------------


@pytest.mark.parametrize("world_count", [1, 1024])
def test_kernel_matches_host_reference(world_count: int) -> None:
    achieved = random_quats(world_count, 1234)
    target = random_quats(world_count, 5678)
    base = np.zeros((world_count, BASE_TERMS))
    weights = np.zeros(ALL_TERMS)
    weights[-1] = 1.0
    terms, _ = run_kernel(base, achieved, target, weights)
    expected = object_orientation_raw_term(achieved, target)
    # float32 kernel against float64 reference.
    assert np.max(np.abs(terms[:, -1] - expected)) < 2e-6


@pytest.mark.parametrize(
    ("degrees", "expected_raw"),
    [(0.0, 0.0), (30.0, -1.0 / 36.0), (90.0, -0.25), (180.0, -1.0)],
)
def test_kernel_known_angles(degrees: float, expected_raw: float) -> None:
    half = np.deg2rad(degrees) / 2.0
    achieved = np.array([[0.0, 0.0, 0.0, 1.0]])
    target = np.array([[0.0, 0.0, np.sin(half), np.cos(half)]])
    weights = np.zeros(ALL_TERMS)
    weights[-1] = 2.0
    terms, score = run_kernel(np.zeros((1, BASE_TERMS)), achieved, target, weights)
    assert terms[0, -1] == pytest.approx(expected_raw, abs=2e-6)
    # The plan's scale table: 90 deg -> raw -0.25 -> weighted frame contribution -0.01.
    assert score[0] == pytest.approx(FRAME_DT * 2.0 * expected_raw, abs=2e-8)


def test_ninety_degrees_costs_exactly_one_hundredth() -> None:
    achieved = np.array([[0.0, 0.0, 0.0, 1.0]])
    target = np.array([[0.0, 0.0, np.sin(np.pi / 4.0), np.cos(np.pi / 4.0)]])
    weights = np.zeros(ALL_TERMS)
    weights[-1] = 2.0
    _, score = run_kernel(np.zeros((1, BASE_TERMS)), achieved, target, weights)
    assert score[0] == pytest.approx(-0.01, abs=1e-8)


def test_kernel_is_bounded_and_non_positive() -> None:
    achieved, target = random_quats(4096, 91), random_quats(4096, 92)
    weights = np.zeros(ALL_TERMS)
    terms, _ = run_kernel(np.zeros((4096, BASE_TERMS)), achieved, target, weights)
    assert np.all(terms[:, -1] <= 1e-7)
    assert np.all(terms[:, -1] >= -1.0 - 1e-7)


def test_kernel_double_cover_invariance() -> None:
    achieved, target = random_quats(1024, 101), random_quats(1024, 102)
    weights = np.zeros(ALL_TERMS)
    base = np.zeros((1024, BASE_TERMS))
    reference, _ = run_kernel(base, achieved, target, weights)
    for flip_a, flip_b in ((-1, 1), (1, -1), (-1, -1)):
        flipped, _ = run_kernel(base, flip_a * achieved, flip_b * target, weights)
        assert np.max(np.abs(flipped[:, -1] - reference[:, -1])) < 1e-6


def test_kernel_identity_branch_is_exact_at_zero() -> None:
    identity = np.tile([0.0, 0.0, 0.0, 1.0], (8, 1))
    weights = np.zeros(ALL_TERMS)
    terms, _ = run_kernel(np.zeros((8, BASE_TERMS)), identity, identity, weights)
    assert np.all(terms[:, -1] == 0.0)


# --- the first twelve terms must pass through untouched ---------------------


def test_first_twelve_terms_are_copied_bit_for_bit() -> None:
    world_count = 256
    rng = np.random.default_rng(7)
    base = rng.normal(size=(world_count, BASE_TERMS)).astype(np.float32)
    terms, _ = run_kernel(
        base,
        random_quats(world_count, 111),
        random_quats(world_count, 112),
        np.zeros(ALL_TERMS),
    )
    assert np.array_equal(terms[:, :BASE_TERMS], base)


def test_score_is_frame_dt_times_weighted_sum() -> None:
    world_count = 64
    rng = np.random.default_rng(13)
    base = rng.normal(size=(world_count, BASE_TERMS)).astype(np.float32)
    weights = rng.normal(size=ALL_TERMS).astype(np.float32)
    terms, score = run_kernel(base, random_quats(world_count, 121), random_quats(world_count, 122), weights)
    expected = FRAME_DT * (terms * weights).sum(axis=1)
    assert np.max(np.abs(score - expected)) < 1e-6


def test_zero_orientation_weight_reproduces_the_original_score() -> None:
    """With weight 0 the composed score equals upstream's twelve-term score."""
    world_count = 128
    rng = np.random.default_rng(17)
    base = rng.normal(size=(world_count, BASE_TERMS)).astype(np.float32)
    original_weights = np.asarray(OrientationReconBodyObjectiveConfig().weights, dtype=np.float32)
    weights = np.concatenate([original_weights, [0.0]]).astype(np.float32)
    _, score = run_kernel(base, random_quats(world_count, 131), random_quats(world_count, 132), weights)
    expected = FRAME_DT * (base * original_weights).sum(axis=1)
    assert np.max(np.abs(score - expected)) < 1e-4
