"""Reference implementation of the thirteenth object-orientation reward term.

This module is pure NumPy and imports nothing from NVIDIA upstream or Warp, so
it runs under ``.venvs/episode0`` on the host. :mod:`pipeline.flash_sac_orientation`
implements the same function as a Warp kernel and is tested against this one
inside the container.

Conventions, in full, because getting any of them wrong rewards the wrong
rotation (see ``docs/plans/2026-10-21-orientation-aware-flashsac.md``):

* Quaternions here are **xyzw**, the Newton/Warp layout. The retargeted parquet
  and :class:`flash_chord.data.reference.Reference` are wxyz, but
  ``CommandBuffers`` has already converted the target to xyzw by the time the
  objective reads ``command.body_target_quat_w``. Do not convert it again.
* MuJoCo Warp's internal ``qpos`` is wxyz. Never feed that to this function.
* The term is ``-(theta / pi) ** 2`` where ``theta`` is the geodesic angle in
  ``[0, pi]``. It is a bounded penalty in ``[-1, 0]``, zero at the target.
* Double cover: ``q`` and ``-q`` are the same rotation, so the angle uses
  ``abs(w)`` of the error quaternion. Physical object symmetry is **identity
  only** -- the stool is not assumed yaw-symmetric, because ADD compares
  corresponding object points and the metric would disagree.
"""

from __future__ import annotations

import numpy as np

#: Name of the appended term. The first twelve keep upstream's names and order.
OBJECT_ORIENTATION_TERM_NAME = "object_orientation"

#: Default weight for the squared-penalty shaping. A first hypothesis, and
#: measured in the 350M run to be far too weak: at the 4-11 degree errors the
#: policy actually produces it contributes 1.5% of object_position's magnitude.
DEFAULT_OBJECT_ORIENTATION_WEIGHT = 2.0

#: Default weight and width for the exponential shaping, which mirrors the form
#: upstream already uses for every other orientation term. ``std`` is in
#: radians; 0.2 rad is 11.5 degrees, matching ``object_position_std``'s
#: tightness rather than the looser 0.4 rad of the pelvis and palm terms.
DEFAULT_OBJECT_ORIENTATION_EXP_WEIGHT = 1.0
DEFAULT_OBJECT_ORIENTATION_STD = 0.2

#: Below this squared vector norm the error quaternion is treated as identity
#: and ``theta ** 2`` is taken as its exact small-angle limit ``4 * s2``. This
#: avoids the unbounded derivative of ``acos`` (and of ``atan2`` composed with
#: ``sqrt``) at zero rotation.
_IDENTITY_S2 = 1.0e-12


def _as_quat_array(q: np.ndarray, name: str) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    if q.shape[-1] != 4:
        raise ValueError(f"{name} must have trailing dimension 4 (xyzw), got shape {q.shape}")
    if not np.all(np.isfinite(q)):
        raise ValueError(f"{name} contains non-finite values; refusing to substitute identity")
    norm = np.linalg.norm(q, axis=-1)
    if np.any(norm < 1.0e-8):
        raise ValueError(f"{name} contains a (near-)zero quaternion; refusing to substitute identity")
    return q / norm[..., None]


def quat_mul_xyzw(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of xyzw quaternions, matching ``flash_chord.utils.quat``."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ax, ay, az, aw = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack(
        (
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ),
        axis=-1,
    )


def quat_inv_xyzw(q: np.ndarray) -> np.ndarray:
    """Conjugate of a unit xyzw quaternion."""
    q = np.asarray(q, dtype=np.float64)
    return np.stack((-q[..., 0], -q[..., 1], -q[..., 2], q[..., 3]), axis=-1)


def geodesic_angle_squared(q_xyzw: np.ndarray, q_ref_xyzw: np.ndarray) -> np.ndarray:
    """Squared geodesic angle in rad^2 between achieved and reference rotations.

    Inputs are normalized first and must be finite and non-degenerate. The
    result is in ``[0, pi ** 2]`` and is invariant to the sign of either input.
    """
    q = _as_quat_array(q_xyzw, "q_xyzw")
    q_ref = _as_quat_array(q_ref_xyzw, "q_ref_xyzw")
    q_error = quat_mul_xyzw(q_ref, quat_inv_xyzw(q))
    s2 = np.sum(q_error[..., :3] ** 2, axis=-1)
    w = np.abs(q_error[..., 3])
    # Small-angle branch: theta = 2 * atan2(s, w) -> 2 * s as s -> 0, w -> 1.
    small = s2 < _IDENTITY_S2
    safe_s2 = np.where(small, 1.0, s2)
    theta = 2.0 * np.arctan2(np.sqrt(safe_s2), w)
    theta_squared = np.where(small, 4.0 * s2, theta * theta)
    return np.clip(theta_squared, 0.0, np.pi * np.pi)


def object_orientation_raw_term(q_xyzw: np.ndarray, q_ref_xyzw: np.ndarray) -> np.ndarray:
    """The raw thirteenth term: ``-(theta / pi) ** 2`` in ``[-1, 0]``.

    This is the value stored in the replay buffer. It is **not** multiplied by
    its weight or by ``frame_dt`` here; FlashSAC applies both during ordinary
    reward composition and relabeling.
    """
    return -geodesic_angle_squared(q_xyzw, q_ref_xyzw) / (np.pi * np.pi)


def object_orientation_exponential_term(
    q_xyzw: np.ndarray,
    q_ref_xyzw: np.ndarray,
    std: float = DEFAULT_OBJECT_ORIENTATION_STD,
) -> np.ndarray:
    """``exp(-theta**2 / std**2)`` in ``(0, 1]``, the shaping upstream itself uses.

    ``ReconBodyObjective`` rewards pelvis and palm orientation as
    ``exp(-angle**2 / 0.4**2)`` and object *position* as
    ``exp(-sse / 0.2**2)``. The squared penalty in
    :func:`object_orientation_raw_term` is on a completely different scale in
    the regime the policy occupies: at 10.5 degrees it returns -0.0034, while
    this returns +0.43. Use this form when the term has to compete with the
    other twelve rather than merely exist.
    """
    if not np.isfinite(std) or std <= 0.0:
        raise ValueError(f"std must be finite and positive, got {std}")
    return np.exp(-geodesic_angle_squared(q_xyzw, q_ref_xyzw) / (std * std))


def weighted_frame_contribution(
    q_xyzw: np.ndarray,
    q_ref_xyzw: np.ndarray,
    weight: float = DEFAULT_OBJECT_ORIENTATION_WEIGHT,
    frame_dt: float = 0.02,
) -> np.ndarray:
    """Per-step reward contribution, for scale checks against the other terms."""
    return frame_dt * weight * object_orientation_raw_term(q_xyzw, q_ref_xyzw)
