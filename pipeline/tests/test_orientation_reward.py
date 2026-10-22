"""Host tests for the thirteenth object-orientation reward term.

Pure NumPy/SciPy; runs under ``.venvs/episode0`` with no container. The Warp
kernel is checked against this same reference inside the container by
``pipeline/tests/test_flash_sac_orientation.py``.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from pipeline.orientation_reward import (
    DEFAULT_OBJECT_ORIENTATION_WEIGHT,
    geodesic_angle_squared,
    object_orientation_raw_term,
    quat_inv_xyzw,
    quat_mul_xyzw,
    weighted_frame_contribution,
)

IDENTITY = np.array([0.0, 0.0, 0.0, 1.0])


def yaw(angle_rad: float) -> np.ndarray:
    """A rotation of ``angle_rad`` about +z, xyzw."""
    return np.array([0.0, 0.0, np.sin(angle_rad / 2.0), np.cos(angle_rad / 2.0)])


def random_quats(count: int, seed: int) -> np.ndarray:
    """Uniform random unit quaternions in xyzw (SciPy's own layout)."""
    return Rotation.random(count, random_state=seed).as_quat()


# --- known values -----------------------------------------------------------


@pytest.mark.parametrize(
    ("degrees", "expected_raw"),
    [(0.0, 0.0), (30.0, -1.0 / 36.0), (90.0, -0.25), (180.0, -1.0)],
)
def test_known_angles_match_the_plan(degrees: float, expected_raw: float) -> None:
    raw = object_orientation_raw_term(IDENTITY, yaw(np.deg2rad(degrees)))
    assert raw == pytest.approx(expected_raw, abs=1e-12)


def test_term_is_bounded_and_non_positive() -> None:
    a, b = random_quats(4096, 11), random_quats(4096, 12)
    raw = object_orientation_raw_term(a, b)
    assert np.all(raw <= 0.0)
    assert np.all(raw >= -1.0)


def test_weighted_contribution_scale_against_other_terms() -> None:
    """90 deg must cost 0.01/step -- half of object_position's 0.02 maximum."""
    contribution = weighted_frame_contribution(IDENTITY, yaw(np.pi / 2.0))
    assert contribution == pytest.approx(-0.01, abs=1e-12)
    worst = weighted_frame_contribution(IDENTITY, yaw(np.pi))
    assert worst == pytest.approx(-0.04, abs=1e-12)
    # Far smaller than the -6.0 that a termination costs in one step.
    assert abs(worst) < 0.1


def test_180_degrees_is_finite_and_exactly_minus_one() -> None:
    raw = object_orientation_raw_term(IDENTITY, yaw(np.pi))
    assert np.isfinite(raw)
    assert raw == pytest.approx(-1.0, abs=1e-12)


# --- agreement with SciPy ---------------------------------------------------


def test_matches_scipy_geodesic_angle() -> None:
    a, b = random_quats(10_000, 42), random_quats(10_000, 43)
    ours = geodesic_angle_squared(a, b)
    relative = Rotation.from_quat(b) * Rotation.from_quat(a).inv()
    scipy_angle = np.linalg.norm(relative.as_rotvec(), axis=-1)
    assert np.max(np.abs(ours - scipy_angle**2)) < 1e-12


# --- invariances ------------------------------------------------------------


def test_double_cover_sign_invariance() -> None:
    a, b = random_quats(2048, 7), random_quats(2048, 8)
    base = object_orientation_raw_term(a, b)
    assert np.allclose(object_orientation_raw_term(-a, b), base, atol=1e-14)
    assert np.allclose(object_orientation_raw_term(a, -b), base, atol=1e-14)
    assert np.allclose(object_orientation_raw_term(-a, -b), base, atol=1e-14)


def test_common_frame_invariance() -> None:
    """Rotating achieved and reference by the same world rotation changes nothing."""
    a, b = random_quats(2048, 21), random_quats(2048, 22)
    common = random_quats(2048, 23)
    base = object_orientation_raw_term(a, b)
    rotated = object_orientation_raw_term(quat_mul_xyzw(common, a), quat_mul_xyzw(common, b))
    assert np.max(np.abs(rotated - base)) < 1e-12


def test_symmetry_in_its_arguments() -> None:
    a, b = random_quats(2048, 31), random_quats(2048, 32)
    assert np.allclose(object_orientation_raw_term(a, b), object_orientation_raw_term(b, a), atol=1e-12)


def test_non_unit_inputs_are_normalized() -> None:
    a, b = random_quats(512, 51), random_quats(512, 52)
    scaled = object_orientation_raw_term(a * 3.7, b * 0.21)
    assert np.allclose(scaled, object_orientation_raw_term(a, b), atol=1e-12)


def test_identity_only_symmetry_distinguishes_yaw() -> None:
    """A 90 deg yaw must NOT be free: the metric compares corresponding points."""
    assert object_orientation_raw_term(IDENTITY, yaw(np.pi / 2.0)) == pytest.approx(-0.25)
    assert object_orientation_raw_term(IDENTITY, yaw(np.pi)) == pytest.approx(-1.0)


# --- numerical behaviour ----------------------------------------------------


def test_small_angle_branch_is_accurate_and_continuous() -> None:
    angles = np.array([0.0, 1e-9, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3])
    raw = np.array([object_orientation_raw_term(IDENTITY, yaw(a)) for a in angles])
    expected = -((angles / np.pi) ** 2)
    assert np.max(np.abs(raw - expected)) < 1e-18
    # Monotone and finite across the branch boundary.
    assert np.all(np.diff(raw) <= 0.0)
    assert np.all(np.isfinite(raw))


def test_float32_round_trip_error_is_small() -> None:
    """Replay stores the raw term in float32; check the quantization cost."""
    a, b = random_quats(4096, 61), random_quats(4096, 62)
    exact = object_orientation_raw_term(a, b)
    rounded = np.float32(exact).astype(np.float64)
    assert np.max(np.abs(rounded - exact)) < 2e-7


def test_finite_difference_derivative_matches_analytic() -> None:
    """d(raw)/d(theta) = -2 * theta / pi^2, away from the 180 deg endpoint."""
    thetas = np.linspace(0.05, 3.0, 40)
    step = 1e-6
    for theta in thetas:
        plus = object_orientation_raw_term(IDENTITY, yaw(theta + step))
        minus = object_orientation_raw_term(IDENTITY, yaw(theta - step))
        numeric = (plus - minus) / (2.0 * step)
        analytic = -2.0 * theta / (np.pi * np.pi)
        assert numeric == pytest.approx(analytic, abs=1e-8)


# --- invalid input must fail loudly ----------------------------------------


@pytest.mark.parametrize("bad", [[np.nan, 0, 0, 1], [np.inf, 0, 0, 1], [0, 0, 0, 0]])
def test_invalid_quaternions_raise(bad: list[float]) -> None:
    with pytest.raises(ValueError):
        object_orientation_raw_term(np.array(bad, dtype=float), IDENTITY)
    with pytest.raises(ValueError):
        object_orientation_raw_term(IDENTITY, np.array(bad, dtype=float))


def test_wrong_trailing_dimension_raises() -> None:
    with pytest.raises(ValueError):
        object_orientation_raw_term(np.zeros(3), IDENTITY)


# --- convention guards ------------------------------------------------------


def test_wxyz_input_would_be_wrong_and_is_detectable() -> None:
    """Guard the xyzw/wxyz hazard: feeding wxyz-ordered data gives a different angle.

    A pure yaw is a poor probe -- reinterpreting ``[0, 0, z, w]`` as wxyz is a
    rotation of the same angle about another axis, so the scalar term is
    unchanged. A 120 deg yaw does discriminate: its scalar part 0.5 moves into a
    vector slot and 0.866 becomes the scalar, turning 120 deg into 60 deg.
    """
    q_xyzw = yaw(np.deg2rad(120.0))
    q_misread = q_xyzw[[3, 0, 1, 2]]
    assert object_orientation_raw_term(IDENTITY, q_xyzw) == pytest.approx(-((120.0 / 180.0) ** 2))
    assert object_orientation_raw_term(IDENTITY, q_misread) == pytest.approx(-((60.0 / 180.0) ** 2))

    # And over random rotations the two readings disagree almost everywhere.
    sample = random_quats(512, 81)
    correct = object_orientation_raw_term(IDENTITY, sample)
    misread = object_orientation_raw_term(IDENTITY, sample[:, [3, 0, 1, 2]])
    assert np.mean(np.abs(correct - misread) > 1e-6) > 0.99


def test_quat_helpers_match_scipy() -> None:
    a, b = random_quats(1024, 71), random_quats(1024, 72)
    product = quat_mul_xyzw(a, b)
    expected = (Rotation.from_quat(a) * Rotation.from_quat(b)).as_quat()
    # Compare as rotations -- as_quat may return the opposite sign branch.
    assert np.allclose(np.abs(np.sum(product * expected, axis=-1)), 1.0, atol=1e-12)
    inverse = quat_inv_xyzw(a)
    assert np.allclose(np.abs(np.sum(inverse * Rotation.from_quat(a).inv().as_quat(), axis=-1)), 1.0, atol=1e-12)


def test_default_weight_is_two() -> None:
    assert DEFAULT_OBJECT_ORIENTATION_WEIGHT == 2.0


# --- arm 2: exponential shaping -------------------------------------------
#
# Arm 1's squared penalty measured 1.51% of object_position's magnitude in the
# 350M run. These tests pin the properties that make arm 2 a usable signal in
# the regime the policy actually occupies.

from pipeline.orientation_reward import (  # noqa: E402
    DEFAULT_OBJECT_ORIENTATION_STD,
    object_orientation_exponential_term,
)


def test_exponential_is_one_on_target_and_bounded() -> None:
    assert object_orientation_exponential_term(IDENTITY, IDENTITY) == pytest.approx(1.0)
    a, b = random_quats(4096, 301), random_quats(4096, 302)
    values = object_orientation_exponential_term(a, b)
    assert np.all(values > 0.0)
    assert np.all(values <= 1.0)


def test_exponential_matches_upstreams_own_shaping_form() -> None:
    """Identical to how ReconBody rewards pelvis and palm orientation."""
    for degrees in (0.0, 5.0, 11.459, 20.0):
        theta = np.deg2rad(degrees)
        for std in (0.2, 0.4):
            expected = np.exp(-(theta**2) / std**2)
            assert object_orientation_exponential_term(IDENTITY, yaw(theta), std) == pytest.approx(expected, abs=1e-12)


def test_exponential_is_monotone_decreasing_in_angle() -> None:
    angles = np.linspace(0.0, np.pi, 60)
    values = np.array([object_orientation_exponential_term(IDENTITY, yaw(a)) for a in angles])
    assert np.all(np.diff(values) <= 1e-15)


def test_exponential_discriminates_across_the_auc_band() -> None:
    """The stool's max radius is 0.552 m, so 1-9 cm of ADD is about 1-9.3 deg.

    The term has to separate those angles to be able to move ADD-AUC at all.
    """
    band = [object_orientation_exponential_term(IDENTITY, yaw(np.deg2rad(d))) for d in (1, 3, 6, 9.3)]
    # Strictly decreasing, and spanning a wide fraction of the full range.
    assert all(later < earlier for earlier, later in zip(band, band[1:]))
    assert band[0] - band[-1] > 0.4


def test_exponential_beats_the_squared_penalty_where_it_matters() -> None:
    """At the measured 10.5 deg operating point, arm 2 must dominate arm 1."""
    theta = np.deg2rad(10.5)
    arm1 = 0.02 * DEFAULT_OBJECT_ORIENTATION_WEIGHT * object_orientation_raw_term(IDENTITY, yaw(theta))
    arm2 = 0.02 * 1.0 * object_orientation_exponential_term(IDENTITY, yaw(theta))
    assert abs(arm2) > 50 * abs(arm1)
    # And it is a real fraction of object_position's measured +0.018714/step.
    assert abs(arm2) > 0.3 * 0.018714


def test_exponential_default_std_is_tight_enough_for_auc() -> None:
    assert DEFAULT_OBJECT_ORIENTATION_STD == 0.2
    # 0.2 rad is 11.5 deg: the half-value point sits inside the AUC band.
    half = object_orientation_exponential_term(IDENTITY, yaw(DEFAULT_OBJECT_ORIENTATION_STD))
    assert half == pytest.approx(np.exp(-1.0), abs=1e-12)


@pytest.mark.parametrize("bad", [0.0, -0.1, np.nan, np.inf])
def test_exponential_rejects_invalid_std(bad: float) -> None:
    with pytest.raises(ValueError):
        object_orientation_exponential_term(IDENTITY, yaw(0.1), bad)


def test_exponential_keeps_arm_ones_invariances() -> None:
    a, b = random_quats(1024, 311), random_quats(1024, 312)
    base = object_orientation_exponential_term(a, b)
    assert np.allclose(object_orientation_exponential_term(-a, b), base, atol=1e-14)
    assert np.allclose(object_orientation_exponential_term(a, -b), base, atol=1e-14)
    common = random_quats(1024, 313)
    rotated = object_orientation_exponential_term(quat_mul_xyzw(common, a), quat_mul_xyzw(common, b))
    assert np.max(np.abs(rotated - base)) < 1e-12
