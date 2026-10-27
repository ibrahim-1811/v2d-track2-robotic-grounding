"""Arm 4: an object-velocity term on top of arm 3's keypoint objective.

The four Tier 1 episodes that keep losing the object (vacuum ep01, bat ep04,
bottle ep25, squeegee ep28) survived five training variants (seeds 43/44,
``term25``, ``grip10``, ``grip15``). Measured on their training logs and
evaluation rollouts:

* ``force_closure`` is earned at the same rate as in solved episodes
  (0.03-0.07 weighted per step vs 0.05-0.07), so a contact reward is not missing;
* object terminations climb exactly as the VOC assist fades: bottle 11% of
  episodes at voc 1.0, 47% at voc 0.1, 72% at voc 0;
* they are the fast episodes -- median reference object speed 0.36-0.48 m/s
  (bat peaks at 2.47 m/s) against 0.02-0.04 m/s for episodes 0 and 3.

Every position-shaped term only notices a slipping object once it is off by
centimetres, and training terminates at 10 cm, so a slip is learned about
roughly when the episode ends. Velocity disagrees the moment the object starts
to slide or fall, which is the signal this term adds:

    v_ref = (p*[f+1] - p*[f-1]) / (2 dt)          central difference, one-sided at the ends
    w_ref = log(q*[f+1] q*[f-1]^-1) / (2 dt)      world-frame, double-cover aware
    raw   = 0.5 exp(-|v - v_ref| / linear_scale) + 0.5 exp(-|w - w_ref| / angular_scale)

``v`` is the velocity of the object's body origin (the point the reference
tracks), recovered from Newton's COM velocity exactly as upstream's VOC does in
``flash_chord.runtime.object_control``. ``dt`` is the control step: with
``source_frame_playback`` every control step advances one reference frame.

Scales: 0.5 m/s is about the median speed of the failing episodes; 4 rad/s sits
above their median angular speed (0.6-3.7 rad/s) because the references carry
single-frame angular spikes of 20-41 rad/s (symmetric objects, capture), which
a tight scale would turn into reward noise.

Structure follows arm 3 (:mod:`pipeline.flash_sac_keypoints`): this wrapper
delegates to a real keypoint objective, copies its thirteen terms byte for
byte, appends one raw term and recomposes the score, so FlashSAC's replay and
``_relabel_rewards`` see a genuine fourteenth column. Rigid objects only --
build refuses an articulated reference rather than guessing its root.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import warp as wp

from flash_chord.objectives.shaping import ObjectiveShape, shaped_objective
from pipeline.flash_sac_keypoints import (
    KEYPOINT_RECON_BODY_OBJECTIVE_TERM_NAMES,
    KeypointReconBodyObjective,
    KeypointReconBodyObjectiveConfig,
)

OBJECT_VELOCITY_TERM_NAME = "object_velocity"

VELOCITY_RECON_BODY_OBJECTIVE_TERM_NAMES = (
    *KEYPOINT_RECON_BODY_OBJECTIVE_TERM_NAMES,
    OBJECT_VELOCITY_TERM_NAME,
)

_BASE_TERM_COUNT = wp.constant(len(KEYPOINT_RECON_BODY_OBJECTIVE_TERM_NAMES))
_TERM_COUNT = wp.constant(len(VELOCITY_RECON_BODY_OBJECTIVE_TERM_NAMES))
_LAPLACIAN = wp.constant(int(ObjectiveShape.LAPLACIAN))


@wp.kernel
def append_object_velocity(
    base_terms: wp.array(dtype=wp.float32),
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    timestep: wp.array(dtype=wp.int32),
    reference_position: wp.array(dtype=wp.vec3),
    reference_quat: wp.array(dtype=wp.quat),
    object_body_id: int,
    bodies_per_world: int,
    num_frames: int,
    linear_scale: float,
    angular_scale: float,
    weights: wp.array(dtype=wp.float32),
    frame_dt: float,
    terms: wp.array(dtype=wp.float32),
    score: wp.array(dtype=wp.float32),
) -> None:
    """Copy the thirteen keypoint-objective terms, append the velocity term, recompose score."""
    world = wp.tid()
    source = world * _BASE_TERM_COUNT
    target = world * _TERM_COUNT
    for term in range(_BASE_TERM_COUNT):
        terms[target + term] = base_terms[source + term]

    body = world * bodies_per_world + object_body_id
    rotation = wp.transform_get_rotation(body_q[body])
    angular = wp.spatial_bottom(body_qd[body])
    # body_qd's linear part is measured at the COM; the reference tracks the body origin.
    linear = wp.spatial_top(body_qd[body]) - wp.cross(angular, wp.quat_rotate(rotation, body_com[body]))

    frame = wp.clamp(timestep[world], 0, num_frames - 1)
    previous = wp.max(frame - 1, 0)
    following = wp.min(frame + 1, num_frames - 1)
    reference_linear = wp.vec3(0.0)
    reference_angular = wp.vec3(0.0)
    if following > previous:
        span = float(following - previous) * frame_dt
        reference_linear = (reference_position[following] - reference_position[previous]) / span
        relative = reference_quat[following] * wp.quat_inverse(reference_quat[previous])
        if relative[3] < 0.0:
            relative = wp.quat(-relative[0], -relative[1], -relative[2], -relative[3])
        axis = wp.vec3(relative[0], relative[1], relative[2])
        sine = wp.length(axis)
        if sine > 1.0e-9:
            reference_angular = axis * (2.0 * wp.atan2(sine, relative[3]) / (sine * span))

    linear_error = linear - reference_linear
    angular_error = angular - reference_angular
    terms[target + _BASE_TERM_COUNT] = 0.5 * (
        shaped_objective(wp.dot(linear_error, linear_error), linear_scale, _LAPLACIAN, 0.0)
        + shaped_objective(wp.dot(angular_error, angular_error), angular_scale, _LAPLACIAN, 0.0)
    )

    total = float(0.0)  # noqa: UP018 - Warp requires explicitly typed mutable locals.
    for term in range(_TERM_COUNT):
        total += weights[term] * terms[target + term]
    score[world] = frame_dt * total


@dataclass(frozen=True)
class VelocityReconBodyObjectiveConfig(KeypointReconBodyObjectiveConfig):
    """Arm 3's keypoint objective plus an object-velocity term."""

    object_velocity_weight: float = 2.0
    object_velocity_linear_scale: float = 0.5
    object_velocity_angular_scale: float = 4.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if not math.isfinite(self.object_velocity_weight) or self.object_velocity_weight < 0.0:
            raise ValueError(
                f"object_velocity_weight must be finite and non-negative, got {self.object_velocity_weight}"
            )
        for name in ("object_velocity_linear_scale", "object_velocity_angular_scale"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive, got {value}")

    @property
    def all_weights(self) -> tuple[float, ...]:
        return (*super().all_weights, self.object_velocity_weight)

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
    ) -> "VelocityReconBodyObjective":
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
        object_reference = base.base.command.reference
        if object_reference.num_bodies != 1:
            raise ValueError(
                f"object_velocity supports rigid objects only; the reference has {object_reference.num_bodies} bodies"
            )
        return VelocityReconBodyObjective(
            base=base,
            world_count=world_count,
            frame_dt=frame_dt,
            linear_scale=self.object_velocity_linear_scale,
            angular_scale=self.object_velocity_angular_scale,
            body_com=model.body_com,
            weights=wp.array(self.all_weights, dtype=wp.float32, device=device),
            terms=wp.zeros(world_count * len(VELOCITY_RECON_BODY_OBJECTIVE_TERM_NAMES), dtype=wp.float32, device=device),
            score=wp.zeros(world_count, dtype=wp.float32, device=device),
        )


@dataclass
class VelocityReconBodyObjective:
    """Fourteen-term objective delegating its first thirteen terms to arm 3."""

    base: KeypointReconBodyObjective
    world_count: int
    frame_dt: float
    linear_scale: float
    angular_scale: float
    body_com: wp.array
    weights: wp.array
    terms: wp.array
    score: wp.array

    @property
    def term_names(self) -> tuple[str, ...]:
        return VELOCITY_RECON_BODY_OBJECTIVE_TERM_NAMES

    @property
    def config(self):
        return self.base.config

    def set_weights(self, weights: Mapping[str, float]) -> None:
        if set(weights) != set(self.term_names):
            raise ValueError(
                f"velocity ReconBody objective weights must contain exactly {self.term_names}, got {tuple(weights)}"
            )
        values = tuple(float(weights[name]) for name in self.term_names)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("velocity ReconBody objective weights must be finite")
        self.weights.assign(values)
        self.base.set_weights({name: weights[name] for name in self.base.term_names})

    def reset(self, reset_mask: wp.array) -> None:
        self.base.reset(reset_mask)

    def evaluate(self, state, timestep: wp.array, terminated: wp.array) -> None:
        self.base.evaluate(state, timestep, terminated)
        recon = self.base.base
        wp.launch(
            append_object_velocity,
            dim=self.world_count,
            inputs=[
                self.base.terms,
                state.body_q,
                state.body_qd,
                self.body_com,
                timestep,
                recon.command.reference.body_pos_w,
                recon.command.reference.body_quat_w,
                recon.object_body_id,
                recon.bodies_per_world,
                recon.command.reference.num_frames,
                self.linear_scale,
                self.angular_scale,
                self.weights,
                self.frame_dt,
            ],
            outputs=[self.terms, self.score],
        )
