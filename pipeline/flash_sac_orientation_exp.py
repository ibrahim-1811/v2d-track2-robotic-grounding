"""Arm 2: an object-orientation term shaped like upstream's own orientation terms.

Arm 1 (:mod:`pipeline.flash_sac_orientation`) appends
``-(theta / pi) ** 2`` with weight 2.0. The 350M Episode 0 run measured that
term at **1.51% of object_position's magnitude** -- eleventh of thirteen, below
the ``action_rate_l2`` regularizer -- because a penalty scaled to the 180 deg
worst case returns only -0.0034 at the 4-11 deg errors the policy actually
produces. It is too weak to shape behaviour.

This module keeps everything else identical and changes only the shaping to the
form ``ReconBodyObjective`` already uses for pelvis and palm orientation:

    raw = exp(-theta ** 2 / std ** 2)        in (0, 1], 1.0 on target

At std 0.2 rad and weight 1.0 that contributes 0.0086/step at 10.5 deg -- 46% of
object_position, and 63x arm 1's signal -- with its steepest gradient across the
4-15 deg band that ADD-AUC is decided in. The stool's maximum radius is 0.552 m,
so 9 cm of ADD is about 9.3 deg: the term must discriminate there or it cannot
help the ranking metric.

Deliberately a **separate module** from arm 1, not an edit to it, so a running
arm-1 job keeps a byte-identical source file and arm 1's provenance stays
reproducible.

The twelve upstream terms, the curriculum thresholds, the VOC schedule and the
reset mixture are untouched here exactly as in arm 1.
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

ORIENTATION_RECON_BODY_OBJECTIVE_TERM_NAMES = (
    *RECON_BODY_OBJECTIVE_TERM_NAMES,
    OBJECT_ORIENTATION_TERM_NAME,
)

_BASE_TERM_COUNT = wp.constant(len(RECON_BODY_OBJECTIVE_TERM_NAMES))
_TERM_COUNT = wp.constant(len(ORIENTATION_RECON_BODY_OBJECTIVE_TERM_NAMES))
_IDENTITY_S2 = wp.constant(1.0e-12)


@wp.func
def object_orientation_exponential(q_xyzw: wp.quat, q_ref_xyzw: wp.quat, inv_std_squared: float) -> wp.float32:
    """``exp(-theta ** 2 / std ** 2)`` for two xyzw quaternions; in ``(0, 1]``.

    Conventions are arm 1's, unchanged: ``abs(w)`` for double-cover invariance,
    and an exact small-angle branch instead of ``acos``, whose derivative is
    unbounded at zero rotation.
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
    return wp.exp(-theta_squared * inv_std_squared)


@wp.kernel
def append_object_orientation_exponential(
    base_terms: wp.array(dtype=wp.float32),
    body_q: wp.array(dtype=wp.transform),
    object_target_quat_w: wp.array(dtype=wp.quat),
    object_body_id: int,
    bodies_per_world: int,
    num_bodies: int,
    inv_std_squared: float,
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
    terms[target + _BASE_TERM_COUNT] = object_orientation_exponential(object_quat, reference_quat, inv_std_squared)

    total = float(0.0)  # noqa: UP018 - Warp requires explicitly typed mutable locals.
    for term in range(_TERM_COUNT):
        total += weights[term] * terms[target + term]
    score[world] = frame_dt * total


@dataclass(frozen=True)
class ExponentialOrientationReconBodyObjectiveConfig(ReconBodyObjectiveConfig):
    """Upstream's exact ReconBody configuration plus an exp-shaped orientation term."""

    object_orientation_weight: float = 1.0
    object_orientation_std: float = 0.2

    def __post_init__(self) -> None:
        super().__post_init__()
        if not math.isfinite(self.object_orientation_weight) or self.object_orientation_weight < 0.0:
            raise ValueError(
                f"object_orientation_weight must be finite and non-negative, got {self.object_orientation_weight}"
            )
        if not math.isfinite(self.object_orientation_std) or self.object_orientation_std <= 0.0:
            raise ValueError(
                f"object_orientation_std must be finite and positive, got {self.object_orientation_std}"
            )

    @property
    def all_weights(self) -> tuple[float, ...]:
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
    ) -> "ExponentialOrientationReconBodyObjective":
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
        return ExponentialOrientationReconBodyObjective(
            base=base,
            world_count=world_count,
            frame_dt=frame_dt,
            inv_std_squared=1.0 / (self.object_orientation_std * self.object_orientation_std),
            weights=wp.array(self.all_weights, dtype=wp.float32, device=device),
            terms=wp.zeros(world_count * len(ORIENTATION_RECON_BODY_OBJECTIVE_TERM_NAMES), dtype=wp.float32, device=device),
            score=wp.zeros(world_count, dtype=wp.float32, device=device),
        )


@dataclass
class ExponentialOrientationReconBodyObjective:
    """Thirteen-term objective delegating its first twelve terms to upstream."""

    base: ReconBodyObjective
    world_count: int
    frame_dt: float
    inv_std_squared: float
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
        if set(weights) != set(self.term_names):
            raise ValueError(
                f"exponential orientation objective weights must contain exactly {self.term_names}, "
                f"got {tuple(weights)}"
            )
        values = tuple(float(weights[name]) for name in self.term_names)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("exponential orientation objective weights must be finite")
        self.weights.assign(values)
        self.base.set_weights({name: weights[name] for name in self.base.term_names})

    def reset(self, reset_mask: wp.array) -> None:
        self.base.reset(reset_mask)

    def evaluate(self, state, timestep: wp.array, terminated: wp.array) -> None:
        self.base.evaluate(state, timestep, terminated)
        wp.launch(
            append_object_orientation_exponential,
            dim=self.world_count,
            inputs=[
                self.base.terms,
                state.body_q,
                self.base.command.body_target_quat_w,
                self.base.object_body_id,
                self.base.bodies_per_world,
                self.base.num_bodies,
                self.inv_std_squared,
                self.weights,
                self.frame_dt,
            ],
            outputs=[self.terms, self.score],
        )
