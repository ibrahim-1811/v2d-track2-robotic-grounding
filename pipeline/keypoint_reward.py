"""Reference implementation of the thirteenth object-keypoint reward term.

Pure NumPy, no Warp and no upstream import, so it runs on the host.
:mod:`pipeline.flash_sac_keypoints` implements the same function as a Warp
kernel and is checked against this one inside the container.

Conventions, matching :mod:`pipeline.orientation_reward`:

* Quaternions are **xyzw** (Newton/Warp), the layout ``CommandBuffers`` already
  holds in ``body_target_quat_w``. Do not convert again.
* Keypoints are upstream's six object-frame axes ``+-x, +-y, +-z``
  (``flash_chord.objectives.keypoints.KEYPOINT_VECS_NP``) scaled by
  ``radius_fraction * object_radius``, where ``object_radius`` is FlashCHORD's
  per-world bounding-ball radius.
* Shaping is upstream's Laplacian ``exp(-|d| / scale)`` per keypoint, averaged
  over the six, so the term lies in ``(0, 1]`` and is 1.0 on target. Position
  error moves every keypoint; rotation error moves the four off-axis ones by
  roughly ``theta * r`` -- the same lever ADD applies to mesh vertices.
"""

from __future__ import annotations

import numpy as np

OBJECT_KEYPOINTS_TERM_NAME = "object_keypoints"

#: Same weight as the two dominant upstream terms (body_joint_position,
#: force_closure), so object precision competes with them instead of being the
#: saturated 1.0-weight afterthought object_position is.
DEFAULT_OBJECT_KEYPOINTS_WEIGHT = 5.0
#: Laplacian width in metres: the middle of AUC's 1..9 cm thresholds.
DEFAULT_OBJECT_KEYPOINTS_SCALE = 0.05
#: ADD averages mesh vertices, which sit near half the bounding radius.
DEFAULT_OBJECT_KEYPOINTS_RADIUS_FRACTION = 0.5

KEYPOINT_AXES = np.array(
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]
)


def rotate_xyzw(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vectors ``v [..., 3]`` by unit quaternions ``q [..., 4]`` (xyzw)."""
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    xyz, w = q[..., :3], q[..., 3:4]
    t = 2.0 * np.cross(xyz, v)
    return v + w * t + np.cross(xyz, t)


def object_keypoints_raw_term(
    position: np.ndarray,
    q_xyzw: np.ndarray,
    reference_position: np.ndarray,
    reference_q_xyzw: np.ndarray,
    object_radius: np.ndarray | float,
    radius_fraction: float = DEFAULT_OBJECT_KEYPOINTS_RADIUS_FRACTION,
    scale: float = DEFAULT_OBJECT_KEYPOINTS_SCALE,
) -> np.ndarray:
    """Mean over six keypoints of ``exp(-|k - k*| / scale)``; one value per pose."""
    if scale <= 0.0 or radius_fraction <= 0.0:
        raise ValueError("scale and radius_fraction must be positive")
    position = np.atleast_2d(np.asarray(position, dtype=np.float64))
    reference_position = np.atleast_2d(np.asarray(reference_position, dtype=np.float64))
    q = np.atleast_2d(q_xyzw)
    q_ref = np.atleast_2d(reference_q_xyzw)
    radius = radius_fraction * np.broadcast_to(np.asarray(object_radius, dtype=np.float64), (position.shape[0],))
    offsets = KEYPOINT_AXES[None, :, :] * radius[:, None, None]  # [N, 6, 3]
    achieved = position[:, None, :] + rotate_xyzw(q[:, None, :], offsets)
    target = reference_position[:, None, :] + rotate_xyzw(q_ref[:, None, :], offsets)
    distance = np.linalg.norm(achieved - target, axis=-1)
    return np.exp(-distance / scale).mean(axis=-1)
