"""Host tests for the thirteenth object-keypoint reward term.

Pure NumPy/SciPy, no container. The Warp kernel is checked against this
reference inside the container by
``pipeline/tests/container_flash_sac_keypoints_kernel.py``.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from pipeline.keypoint_reward import (
    DEFAULT_OBJECT_KEYPOINTS_SCALE,
    object_keypoints_raw_term,
    rotate_xyzw,
)

IDENTITY = np.array([0.0, 0.0, 0.0, 1.0])
ORIGIN = np.zeros(3)


def yaw(angle_rad: float) -> np.ndarray:
    return np.array([0.0, 0.0, np.sin(angle_rad / 2.0), np.cos(angle_rad / 2.0)])


def test_on_target_is_exactly_one() -> None:
    q = Rotation.random(random_state=3).as_quat()  # scipy is xyzw
    assert object_keypoints_raw_term(ORIGIN + 0.3, q, ORIGIN + 0.3, q, 0.4)[0] == pytest.approx(1.0)


def test_rotation_matches_scipy() -> None:
    q = Rotation.random(5, random_state=7)
    v = np.random.default_rng(0).normal(size=(5, 3))
    np.testing.assert_allclose(rotate_xyzw(q.as_quat(), v), q.apply(v), atol=1e-12)


@pytest.mark.parametrize("error_m", [0.01, 0.03, 0.05, 0.09])
def test_pure_translation_is_upstream_laplacian(error_m: float) -> None:
    """Every keypoint moves by the translation, so the mean is exp(-e/scale)."""
    value = object_keypoints_raw_term([error_m, 0.0, 0.0], IDENTITY, ORIGIN, IDENTITY, 0.3)[0]
    assert value == pytest.approx(np.exp(-error_m / DEFAULT_OBJECT_KEYPOINTS_SCALE))


def test_yaw_moves_only_the_four_horizontal_keypoints() -> None:
    radius, theta = 0.4, np.radians(10.0)
    r = 0.5 * radius
    chord = 2.0 * r * np.sin(theta / 2.0)
    expected = (4.0 * np.exp(-chord / DEFAULT_OBJECT_KEYPOINTS_SCALE) + 2.0) / 6.0
    assert object_keypoints_raw_term(ORIGIN, yaw(theta), ORIGIN, IDENTITY, radius)[0] == pytest.approx(expected)


def test_double_cover_gives_the_same_value() -> None:
    q = Rotation.random(random_state=11).as_quat()
    a = object_keypoints_raw_term(ORIGIN, q, [0.02, 0.0, 0.0], IDENTITY, 0.3)
    b = object_keypoints_raw_term(ORIGIN, -q, [0.02, 0.0, 0.0], IDENTITY, 0.3)
    np.testing.assert_allclose(a, b, atol=1e-12)


def test_larger_objects_are_penalised_more_for_the_same_rotation() -> None:
    small = object_keypoints_raw_term(ORIGIN, yaw(0.2), ORIGIN, IDENTITY, 0.1)[0]
    large = object_keypoints_raw_term(ORIGIN, yaw(0.2), ORIGIN, IDENTITY, 0.9)[0]
    assert large < small < 1.0


def test_discriminates_inside_the_auc_band_where_object_position_is_flat() -> None:
    """The reason this term exists: upstream's exp(-e^2/0.2^2) barely moves 1 -> 5 cm."""
    keypoints = [object_keypoints_raw_term([e, 0, 0], IDENTITY, ORIGIN, IDENTITY, 0.3)[0] for e in (0.01, 0.05)]
    upstream = [np.exp(-(e**2) / 0.2**2) for e in (0.01, 0.05)]
    assert keypoints[0] - keypoints[1] > 0.4
    assert upstream[0] - upstream[1] < 0.07


def test_batched_values_match_one_at_a_time() -> None:
    rng = np.random.default_rng(5)
    p, p_ref = rng.normal(scale=0.05, size=(8, 3)), rng.normal(scale=0.05, size=(8, 3))
    q, q_ref = Rotation.random(8, random_state=1).as_quat(), Rotation.random(8, random_state=2).as_quat()
    radius = rng.uniform(0.1, 0.9, size=8)
    batch = object_keypoints_raw_term(p, q, p_ref, q_ref, radius)
    single = [object_keypoints_raw_term(p[i], q[i], p_ref[i], q_ref[i], radius[i])[0] for i in range(8)]
    np.testing.assert_allclose(batch, single, atol=1e-12)
    assert np.all((batch > 0.0) & (batch <= 1.0))


@pytest.mark.parametrize("bad", [{"scale": 0.0}, {"radius_fraction": -1.0}])
def test_rejects_non_positive_scales(bad: dict) -> None:
    with pytest.raises(ValueError):
        object_keypoints_raw_term(ORIGIN, IDENTITY, ORIGIN, IDENTITY, 0.3, **bad)
