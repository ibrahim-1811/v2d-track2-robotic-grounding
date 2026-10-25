"""Arm 3: an ADD-shaped object-keypoint term for the ReconBody objective.

Measured on the 26 Tier 1 stock policies (NVIDIA's exact recipe, 350M steps):
upstream's ``object_position`` term, ``exp(-|err|^2 / 0.2^2)``, averages
**0.92-0.97 of its maximum in every run and every curriculum stage**, failing
episodes included. Training terminates once the object is 10 cm off, and inside
that window the term never drops below 0.78 and is nearly flat between 1 cm
(0.998) and 5 cm (0.94). The policy is therefore only ever told "stay inside
10 cm", and its evaluation errors park at 3-4 cm and 10-50 deg -- while Track 2
ranks by ADD-AUC over 1..9 cm, which needs both position and orientation.

NVIDIA already solves this for the Sharpa hand: its tracking objective scores
six object keypoints (``flash_chord.objectives.keypoints``). This module appends
that formulation to G1's objective, reusing upstream's keypoint axes and
upstream's ``shaped_objective`` Laplacian verbatim, with two scale choices made
for the ranking metric instead of a hand:

    k_i  = p + R (r e_i),  k_i* = p* + R* (r e_i),  e_i in {+-x, +-y, +-z}
    raw  = mean_i exp(-|k_i - k_i*| / scale)          in (0, 1], 1.0 on target

* ``r = radius_fraction * command.object_radius``. ``object_radius`` is the
  per-world bounding-ball radius FlashCHORD already computes for force closure;
  ADD averages over mesh vertices, which sit at about half of it (stool: mean
  0.30 m vs bounding 0.55 m), hence the 0.5 default. A rotation error therefore
  costs what it costs ADD, scaled to each object.
* ``scale = 0.05`` m puts the Laplacian's width in the middle of the 1..9 cm
  band: 1 cm -> 0.82, 3 cm -> 0.55, 5 cm -> 0.37, 9 cm -> 0.17.

Everything else follows arm 1 (:mod:`pipeline.flash_sac_orientation`): the
wrapper delegates to a real ``ReconBodyObjective``, copies its twelve terms
byte for byte, appends one raw term last and recomposes the score, so
FlashSAC's replay and ``_relabel_rewards`` see a genuine thirteenth column.
Deliberately a separate module, so arms 1 and 2 keep byte-identical sources.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import warp as wp

from flash_chord.objectives.keypoints import KEYPOINT_VECS_NP
from flash_chord.objectives.recon_body import (
    RECON_BODY_OBJECTIVE_TERM_NAMES,
    ReconBodyObjective,
    ReconBodyObjectiveConfig,
)
from flash_chord.objectives.shaping import ObjectiveShape, shaped_objective

OBJECT_KEYPOINTS_TERM_NAME = "object_keypoints"

KEYPOINT_RECON_BODY_OBJECTIVE_TERM_NAMES = (
    *RECON_BODY_OBJECTIVE_TERM_NAMES,
    OBJECT_KEYPOINTS_TERM_NAME,
)

_BASE_TERM_COUNT = wp.constant(len(RECON_BODY_OBJECTIVE_TERM_NAMES))
_TERM_COUNT = wp.constant(len(KEYPOINT_RECON_BODY_OBJECTIVE_TERM_NAMES))
_KEYPOINT_COUNT = wp.constant(len(KEYPOINT_VECS_NP))
_LAPLACIAN = wp.constant(int(ObjectiveShape.LAPLACIAN))


@wp.kernel
def append_object_keypoints(
    base_terms: wp.array(dtype=wp.float32),
    body_q: wp.array(dtype=wp.transform),
    object_target_pos_w: wp.array(dtype=wp.vec3),
    object_target_quat_w: wp.array(dtype=wp.quat),
    object_radius: wp.array(dtype=wp.float32),
    keypoint_vecs: wp.array(dtype=wp.vec3),
    object_body_id: int,
    bodies_per_world: int,
    num_bodies: int,
    radius_fraction: float,
    scale: float,
    weights: wp.array(dtype=wp.float32),
    frame_dt: float,
    terms: wp.array(dtype=wp.float32),
    score: wp.array(dtype=wp.float32),
) -> None:
    """Copy the twelve upstream terms, append the keypoint term, recompose score."""
    world = wp.tid()
    source = world * _BASE_TERM_COUNT
    target = world * _TERM_COUNT
    for term in range(_BASE_TERM_COUNT):
        terms[target + term] = base_terms[source + term]

    body = body_q[world * bodies_per_world + object_body_id]
    position = wp.transform_get_translation(body)
    rotation = wp.transform_get_rotation(body)
    command = world * num_bodies
    reference_position = object_target_pos_w[command]
    reference_rotation = object_target_quat_w[command]
    radius = radius_fraction * object_radius[command]

    keypoints = float(0.0)  # noqa: UP018 - Warp requires explicitly typed mutable locals.
    for index in range(_KEYPOINT_COUNT):
        offset = keypoint_vecs[index] * radius
        delta = (position + wp.quat_rotate(rotation, offset)) - (
            reference_position + wp.quat_rotate(reference_rotation, offset)
        )
        keypoints += shaped_objective(wp.dot(delta, delta), scale, _LAPLACIAN, 0.0)
    terms[target + _BASE_TERM_COUNT] = keypoints / float(_KEYPOINT_COUNT)

    total = float(0.0)  # noqa: UP018
    for term in range(_TERM_COUNT):
        total += weights[term] * terms[target + term]
    score[world] = frame_dt * total


@dataclass(frozen=True)
class KeypointReconBodyObjectiveConfig(ReconBodyObjectiveConfig):
    """Upstream's exact ReconBody configuration plus an ADD-shaped keypoint term."""

    object_keypoints_weight: float = 5.0
    object_keypoints_scale: float = 0.05
    object_keypoints_radius_fraction: float = 0.5

    def __post_init__(self) -> None:
        super().__post_init__()
        if not math.isfinite(self.object_keypoints_weight) or self.object_keypoints_weight < 0.0:
            raise ValueError(
                f"object_keypoints_weight must be finite and non-negative, got {self.object_keypoints_weight}"
            )
        if not math.isfinite(self.object_keypoints_scale) or self.object_keypoints_scale <= 0.0:
            raise ValueError(f"object_keypoints_scale must be finite and positive, got {self.object_keypoints_scale}")
        if not math.isfinite(self.object_keypoints_radius_fraction) or self.object_keypoints_radius_fraction <= 0.0:
            raise ValueError(
                "object_keypoints_radius_fraction must be finite and positive, "
                f"got {self.object_keypoints_radius_fraction}"
            )

    @property
    def all_weights(self) -> tuple[float, ...]:
        return (*self.weights, self.object_keypoints_weight)

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
    ) -> "KeypointReconBodyObjective":
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
        return KeypointReconBodyObjective(
            base=base,
            world_count=world_count,
            frame_dt=frame_dt,
            radius_fraction=self.object_keypoints_radius_fraction,
            scale=self.object_keypoints_scale,
            keypoint_vecs=wp.array(KEYPOINT_VECS_NP, dtype=wp.vec3, device=device),
            weights=wp.array(self.all_weights, dtype=wp.float32, device=device),
            terms=wp.zeros(world_count * len(KEYPOINT_RECON_BODY_OBJECTIVE_TERM_NAMES), dtype=wp.float32, device=device),
            score=wp.zeros(world_count, dtype=wp.float32, device=device),
        )


@dataclass
class KeypointReconBodyObjective:
    """Thirteen-term objective delegating its first twelve terms to upstream."""

    base: ReconBodyObjective
    world_count: int
    frame_dt: float
    radius_fraction: float
    scale: float
    keypoint_vecs: wp.array
    weights: wp.array
    terms: wp.array
    score: wp.array

    @property
    def term_names(self) -> tuple[str, ...]:
        return KEYPOINT_RECON_BODY_OBJECTIVE_TERM_NAMES

    @property
    def config(self) -> ReconBodyObjectiveConfig:
        return self.base.config

    def set_weights(self, weights: Mapping[str, float]) -> None:
        if set(weights) != set(self.term_names):
            raise ValueError(
                f"keypoint ReconBody objective weights must contain exactly {self.term_names}, got {tuple(weights)}"
            )
        values = tuple(float(weights[name]) for name in self.term_names)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("keypoint ReconBody objective weights must be finite")
        self.weights.assign(values)
        self.base.set_weights({name: weights[name] for name in self.base.term_names})

    def reset(self, reset_mask: wp.array) -> None:
        self.base.reset(reset_mask)

    def evaluate(self, state, timestep: wp.array, terminated: wp.array) -> None:
        self.base.evaluate(state, timestep, terminated)
        wp.launch(
            append_object_keypoints,
            dim=self.world_count,
            inputs=[
                self.base.terms,
                state.body_q,
                self.base.command.body_target_pos_w,
                self.base.command.body_target_quat_w,
                self.base.command.object_radius,
                self.keypoint_vecs,
                self.base.object_body_id,
                self.base.bodies_per_world,
                self.base.num_bodies,
                self.radius_fraction,
                self.scale,
                self.weights,
                self.frame_dt,
            ],
            outputs=[self.terms, self.score],
        )
