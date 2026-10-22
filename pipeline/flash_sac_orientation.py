"""A thirteenth, bounded object-orientation term for the ReconBody objective.

NVIDIA's ``ReconBodyObjective`` has no dense object-orientation reward at any
curriculum stage (``outputs/step_D/curriculum_audit/REPORT.md``), while Track 2
ranks by ADD-AUC over 1..9 cm thresholds, which a few degrees of rotation on a
half-metre object is enough to miss. This module appends one raw term,
``object_orientation = -(theta / pi) ** 2``, without touching upstream.

How it stays honest with FlashSAC's learner:

* The wrapper **delegates** to a real ``ReconBodyObjective`` and copies its
  twelve terms through byte for byte; only a thirteenth slot is added, at the
  end, so indices 0..11 keep upstream's meaning and order.
* The term is stored **raw** (unweighted, not scaled by ``frame_dt``).
  ``flash_chord.training.flash_sac.replay`` discounts raw terms over the n-step
  window and ``_relabel_rewards`` re-applies the current curriculum weights, so
  a scalar-only reward patch would silently not be optimized. Everything
  downstream sizes itself from ``len(env.objective_term_names)``, so thirteen
  terms propagate with no upstream edit.
* ``set_weights`` accepts exactly the thirteen names, which is what
  ``CurriculumStage.weights_for`` validates against.

Conventions are documented and separately tested in
:mod:`pipeline.orientation_reward`, which is this kernel's host reference.
This module is importable only where Warp and ``flash_chord`` exist, i.e.
inside ``flash_chord:latest``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import warp as wp

from flash_chord.objectives.recon_body import (
    RECON_BODY_OBJECTIVE_TERM_NAMES,
    ReconBodyObjective,
    ReconBodyObjectiveConfig,
)
from flash_chord.utils.quat import quat_inv_xyzw, quat_mul_xyzw

OBJECT_ORIENTATION_TERM_NAME = "object_orientation"

#: Upstream's twelve names, in order, plus ours appended last.
ORIENTATION_RECON_BODY_OBJECTIVE_TERM_NAMES = (
    *RECON_BODY_OBJECTIVE_TERM_NAMES,
    OBJECT_ORIENTATION_TERM_NAME,
)

_BASE_TERM_COUNT = wp.constant(len(RECON_BODY_OBJECTIVE_TERM_NAMES))
_TERM_COUNT = wp.constant(len(ORIENTATION_RECON_BODY_OBJECTIVE_TERM_NAMES))
_INV_PI_SQUARED = wp.constant(1.0 / (math.pi * math.pi))
_IDENTITY_S2 = wp.constant(1.0e-12)


@wp.func
def object_orientation_raw(q_xyzw: wp.quat, q_ref_xyzw: wp.quat) -> wp.float32:
    """``-(theta / pi) ** 2`` for two xyzw quaternions; bounded in ``[-1, 0]``.

    ``abs(w)`` gives double-cover invariance. The ``s2`` branch is the exact
    small-angle limit ``theta ** 2 -> 4 * s2``, taken because ``acos`` (and
    ``atan2 . sqrt``) have an unbounded derivative at zero rotation.
    """
    q = wp.normalize(q_xyzw)
    q_ref = wp.normalize(q_ref_xyzw)
    q_error = quat_mul_xyzw(q_ref, quat_inv_xyzw(q))
    s2 = q_error[0] * q_error[0] + q_error[1] * q_error[1] + q_error[2] * q_error[2]
    if s2 < _IDENTITY_S2:
        theta_squared = 4.0 * s2
    else:
        theta = 2.0 * wp.atan2(wp.sqrt(s2), wp.abs(q_error[3]))
        theta_squared = theta * theta
    # theta is mathematically in [0, pi]; clamp only rounding.
    return -wp.clamp(theta_squared * _INV_PI_SQUARED, 0.0, 1.0)


@wp.kernel
def append_object_orientation(
    base_terms: wp.array(dtype=wp.float32),
    body_q: wp.array(dtype=wp.transform),
    object_target_quat_w: wp.array(dtype=wp.quat),
    object_body_id: int,
    bodies_per_world: int,
    num_bodies: int,
    weights: wp.array(dtype=wp.float32),
    frame_dt: float,
    terms: wp.array(dtype=wp.float32),
    score: wp.array(dtype=wp.float32),
) -> None:
    """Copy the twelve upstream terms, append the thirteenth, recompose score."""
    world = wp.tid()
    source = world * _BASE_TERM_COUNT
    target = world * _TERM_COUNT
    for term in range(_BASE_TERM_COUNT):
        terms[target + term] = base_terms[source + term]

    object_quat = wp.transform_get_rotation(body_q[world * bodies_per_world + object_body_id])
    reference_quat = object_target_quat_w[world * num_bodies]
    terms[target + _BASE_TERM_COUNT] = object_orientation_raw(object_quat, reference_quat)

    total = float(0.0)  # noqa: UP018 - Warp requires explicitly typed mutable locals.
    for term in range(_TERM_COUNT):
        total += weights[term] * terms[target + term]
    score[world] = frame_dt * total


@dataclass(frozen=True)
class OrientationReconBodyObjectiveConfig(ReconBodyObjectiveConfig):
    """Upstream's exact ReconBody configuration plus one orientation weight.

    ``weights`` deliberately keeps returning upstream's **twelve** values so the
    delegated :class:`ReconBodyObjective` builds unchanged; the thirteen-wide
    vector is :attr:`all_weights`.
    """

    object_orientation_weight: float = 2.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if not math.isfinite(self.object_orientation_weight):
            raise ValueError("object_orientation_weight must be finite")
        if self.object_orientation_weight < 0.0:
            raise ValueError(
                "object_orientation_weight must be non-negative: the raw term is already a "
                f"penalty in [-1, 0], got {self.object_orientation_weight}"
            )

    @property
    def all_weights(self) -> tuple[float, ...]:
        """The thirteen weights, in :data:`ORIENTATION_RECON_BODY_OBJECTIVE_TERM_NAMES` order."""
        return (*self.weights, self.object_orientation_weight)

    def build(
        self,
        model,
        embodiment,
        robot_reference,
        reference,
        command,
        contact,
        action,
        world_count: int,
        frame_dt: float,
        device=None,
        reference_joint_q_offset: wp.array | None = None,
        episode_start_frame: wp.array | None = None,
    ) -> "OrientationReconBodyObjective":
        base = super().build(
            model,
            embodiment,
            robot_reference,
            reference,
            command,
            contact,
            action,
            world_count,
            frame_dt,
            device=device,
            reference_joint_q_offset=reference_joint_q_offset,
            episode_start_frame=episode_start_frame,
        )
        return OrientationReconBodyObjective(
            base=base,
            world_count=world_count,
            frame_dt=frame_dt,
            weights=wp.array(self.all_weights, dtype=wp.float32, device=device),
            terms=wp.zeros(
                world_count * len(ORIENTATION_RECON_BODY_OBJECTIVE_TERM_NAMES),
                dtype=wp.float32,
                device=device,
            ),
            score=wp.zeros(world_count, dtype=wp.float32, device=device),
        )


@dataclass
class OrientationReconBodyObjective:
    """Thirteen-term objective delegating its first twelve terms to upstream."""

    base: ReconBodyObjective
    world_count: int
    frame_dt: float
    weights: wp.array
    terms: wp.array
    score: wp.array

    @property
    def term_names(self) -> tuple[str, ...]:
        return ORIENTATION_RECON_BODY_OBJECTIVE_TERM_NAMES

    @property
    def config(self) -> ReconBodyObjectiveConfig:
        return self.base.config

    def set_weights(self, weights: Mapping[str, float]) -> None:
        """Upload one complete thirteen-name weight mapping from the curriculum."""
        if set(weights) != set(self.term_names):
            raise ValueError(
                f"orientation ReconBody objective weights must contain exactly {self.term_names}, "
                f"got {tuple(weights)}"
            )
        values = tuple(float(weights[name]) for name in self.term_names)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("orientation ReconBody objective weights must be finite")
        self.weights.assign(values)
        # Keep the delegate's own (unused) score consistent for diagnostics.
        self.base.set_weights({name: weights[name] for name in self.base.term_names})

    def reset(self, reset_mask: wp.array) -> None:
        self.base.reset(reset_mask)

    def evaluate(self, state, timestep: wp.array, terminated: wp.array) -> None:
        self.base.evaluate(state, timestep, terminated)
        wp.launch(
            append_object_orientation,
            dim=self.world_count,
            inputs=[
                self.base.terms,
                state.body_q,
                self.base.command.body_target_quat_w,
                self.base.object_body_id,
                self.base.bodies_per_world,
                self.base.num_bodies,
                self.weights,
                self.frame_dt,
            ],
            outputs=[self.terms, self.score],
        )
