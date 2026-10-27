"""Project a thirteen-term actor checkpoint into one the official evaluator loads.

``evaluate_policy.py`` rebuilds the whole experiment from the config embedded in
a checkpoint's safetensors header *before* it loads any tensor, so a header that
names ``pipeline.flash_sac_orientation...`` would make the unchanged official
evaluator import our training package. It also instantiates a curriculum whose
stages must match the objective's term names exactly, so thirteen weights would
be rejected by a twelve-term objective.

This module writes a second file whose **actor tensors are byte-identical** and
whose runtime config is upstream-only, while the real training configuration is
preserved verbatim alongside it:

* ``user.resolved_config_json``          -- evaluation reconstruction, upstream only
* ``user.training_resolved_config_json`` -- what actually trained this actor
* ``user.training_reward_definition``    -- the exact thirteenth term
* ``user.export_*``                      -- parent hash, exporter hash, provenance

Never claim the training reward was the original reward: the projection exists
so the evaluator can *run*, not to describe how the policy was produced.

Stdlib only -- safetensors is an 8-byte little-endian header length, a JSON
header and an opaque payload, so this runs on the host with no torch, no
safetensors package and no container (see "Which Python" in CLAUDE.md).
"""

from __future__ import annotations

import hashlib
import json
import struct
from collections.abc import Mapping
from pathlib import Path
from typing import Any

CONFIG_KEY = "user.resolved_config_json"
TRAINING_CONFIG_KEY = "user.training_resolved_config_json"
REWARD_DEFINITION_KEY = "user.training_reward_definition"

ORIENTATION_TARGET = "pipeline.flash_sac_orientation.OrientationReconBodyObjectiveConfig"
UPSTREAM_TARGET = "flash_chord.objectives.recon_body.ReconBodyObjectiveConfig"
ORIENTATION_TERM = "object_orientation"
ORIENTATION_WEIGHT_FIELD = "object_orientation_weight"

UPSTREAM_TERM_NAMES = (
    "termination",
    "action_rate_l2",
    "action_l2",
    "soft_joint_limit",
    "pelvis_position",
    "pelvis_orientation",
    "body_joint_position",
    "object_position",
    "progress",
    "palm_position",
    "palm_orientation",
    "force_closure",
)

REWARD_DEFINITION = (
    "object_orientation = -(theta / pi) ** 2, theta = geodesic angle in [0, pi] "
    "between the object's Newton body quaternion and command.body_target_quat_w "
    "(both xyzw), double-cover aware via abs(w), identity physical symmetry. "
    "Stored raw in replay; weighted and frame_dt-scaled only during reward "
    "composition and relabeling."
)
EXPONENTIAL_ORIENTATION_TARGET = (
    "pipeline.flash_sac_orientation_exp.ExponentialOrientationReconBodyObjectiveConfig"
)
EXPONENTIAL_REWARD_DEFINITION = (
    "object_orientation = exp(-theta ** 2 / object_orientation_std ** 2), theta as in the "
    "squared arm. Stored raw in replay; weighted and frame_dt-scaled only during reward "
    "composition and relabeling."
)
KEYPOINT_TARGET = "pipeline.flash_sac_keypoints.KeypointReconBodyObjectiveConfig"
KEYPOINT_TERM = "object_keypoints"
KEYPOINT_REWARD_DEFINITION = (
    "object_keypoints = mean over upstream's six object axes e_i of "
    "exp(-|(p + R r e_i) - (p* + R* r e_i)| / object_keypoints_scale), r = "
    "object_keypoints_radius_fraction * command.object_radius (bounding-ball radius); "
    "(p, R) the object's Newton body pose, (p*, R*) command.body_target_pos_w/quat_w (xyzw). "
    "Stored raw in replay; weighted and frame_dt-scaled only during reward "
    "composition and relabeling."
)

VELOCITY_TARGET = "pipeline.flash_sac_velocity.VelocityReconBodyObjectiveConfig"
VELOCITY_TERMS = (KEYPOINT_TERM, "object_velocity")
VELOCITY_REWARD_DEFINITION = (
    KEYPOINT_REWARD_DEFINITION
    + " object_velocity = 0.5 exp(-|v - v*| / object_velocity_linear_scale) + 0.5 exp(-|w - w*| / "
    "object_velocity_angular_scale); (v, w) the object body origin's world linear and angular velocity "
    "from Newton body_qd, (v*, w*) central differences of the object reference over frames f-1..f+1 "
    "divided by the control step. Stored raw in replay; weighted and frame_dt-scaled only during "
    "reward composition and relabeling."
)

#: Every objective we train beyond upstream's twelve terms, keyed by its ``_target_``:
#: the appended term (or terms, in column order), the config fields only it defines
#: (the first appended term's weight first), and its definition.
THIRTEENTH_TERM_OBJECTIVES: dict[str, tuple[str | tuple[str, ...], tuple[str, ...], str]] = {
    ORIENTATION_TARGET: (ORIENTATION_TERM, (ORIENTATION_WEIGHT_FIELD,), REWARD_DEFINITION),
    EXPONENTIAL_ORIENTATION_TARGET: (
        ORIENTATION_TERM,
        (ORIENTATION_WEIGHT_FIELD, "object_orientation_std"),
        EXPONENTIAL_REWARD_DEFINITION,
    ),
    KEYPOINT_TARGET: (
        KEYPOINT_TERM,
        ("object_keypoints_weight", "object_keypoints_scale", "object_keypoints_radius_fraction"),
        KEYPOINT_REWARD_DEFINITION,
    ),
    VELOCITY_TARGET: (
        VELOCITY_TERMS,
        (
            "object_keypoints_weight",
            "object_keypoints_scale",
            "object_keypoints_radius_fraction",
            "object_velocity_weight",
            "object_velocity_linear_scale",
            "object_velocity_angular_scale",
        ),
        VELOCITY_REWARD_DEFINITION,
    ),
}


def _appended_terms(target: str) -> tuple[str, ...]:
    term = THIRTEENTH_TERM_OBJECTIVES[target][0]
    return (term,) if isinstance(term, str) else term

#: Config subtrees that must be identical between the native and the projected
#: runtime config. Only the objective schema and the orientation weight may move.
_IMMUTABLE_ROOTS = ("action", "observation", "embodiment", "sim", "termination", "reset", "scene", "task")

#: Hydra resolves ``env.objective: ${objective}``, so the objective appears twice
#: and both copies must be projected. Everything else under ``env`` -- including
#: ``termination.object_orientation_threshold``, which is upstream's 1.5 rad
#: failure limit and NOT our reward -- must survive untouched.
_OBJECTIVE_PATHS = ("objective", "env.objective")
_CURRICULUM_PATHS = ("curriculum", "training.curriculum")


class ExportError(RuntimeError):
    """Raised when a checkpoint cannot be projected safely."""


# --- safetensors container --------------------------------------------------


def read_checkpoint(path: Path) -> tuple[dict[str, Any], bytes]:
    """Return the parsed JSON header and the untouched tensor payload."""
    blob = path.read_bytes()
    if len(blob) < 8:
        raise ExportError(f"{path} is too short to be a safetensors file")
    (length,) = struct.unpack("<Q", blob[:8])
    if 8 + length > len(blob):
        raise ExportError(f"{path} header is truncated")
    header = json.loads(blob[8 : 8 + length])
    return header, blob[8 + length :]


def write_checkpoint(path: Path, header: Mapping[str, Any], payload: bytes) -> None:
    """Write a safetensors file; tensor offsets are payload-relative, so this is safe."""
    encoded = json.dumps(header, separators=(",", ":"), sort_keys=True).encode()
    # safetensors requires the payload to start 8-byte aligned.
    padding = (-len(encoded)) % 8
    encoded += b" " * padding
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(encoded)))
        handle.write(encoded)
        handle.write(payload)


def tensor_index(header: Mapping[str, Any]) -> dict[str, Any]:
    return {name: value for name, value in header.items() if name != "__metadata__"}


# --- the projection ---------------------------------------------------------


def _project_objective(objective: dict[str, Any]) -> tuple[dict[str, Any], float, str]:
    """Return the upstream-only objective, the removed weight and the local target."""
    target = objective.get("_target_")
    if target not in THIRTEENTH_TERM_OBJECTIVES:
        raise ExportError(
            f"objective._target_ is {target!r}; expected one of {sorted(THIRTEENTH_TERM_OBJECTIVES)}. "
            "This checkpoint was not produced by a thirteen-term objective."
        )
    _, fields, _ = THIRTEENTH_TERM_OBJECTIVES[target]
    projected = dict(objective)
    removed = {field: projected.pop(field, None) for field in fields}
    missing = [field for field, value in removed.items() if value is None]
    if missing:
        raise ExportError(f"objective is missing {', '.join(missing)}")
    projected["_target_"] = UPSTREAM_TARGET
    unexpected = set(projected) - set(_upstream_objective_fields())
    if unexpected:
        raise ExportError(f"objective carries keys upstream does not define: {sorted(unexpected)}")
    return projected, float(removed[fields[0]]), target


def _upstream_objective_fields() -> frozenset[str]:
    """The exact field set of ``ReconBodyObjectiveConfig``, plus ``_target_``."""
    return frozenset(
        {
            "_target_",
            *(f"{name}_weight" for name in UPSTREAM_TERM_NAMES),
            "pelvis_position_std",
            "pelvis_orientation_std",
            "body_joint_position_std",
            "object_position_std",
            "palm_position_std",
            "palm_orientation_std",
            "soft_joint_limit_factor",
            "force_closure_min_support",
            "num_wrench_basis",
            "num_friction_cone_edges",
            "friction_coefficient",
            "wrench_basis_seed",
            "sides",
        }
    )


def _project_curriculum(
    curriculum: dict[str, Any], where: str, term: str | tuple[str, ...] = ORIENTATION_TERM
) -> dict[str, Any]:
    terms = (term,) if isinstance(term, str) else term
    stages = curriculum.get("stages")
    if not isinstance(stages, list) or not stages:
        raise ExportError(f"{where} has no stages")
    projected_stages = []
    for index, stage in enumerate(stages):
        if not isinstance(stage, dict):
            raise ExportError(f"{where} stage {index} is not a mapping")
        weights = stage.get("objective_weights")
        if not isinstance(weights, dict):
            raise ExportError(f"{where} stage {index} has no objective_weights")
        for name in terms:
            if name not in weights:
                raise ExportError(f"{where} stage {index} has no {name} weight to project away")
        remaining = {name: value for name, value in weights.items() if name not in terms}
        if set(remaining) != set(UPSTREAM_TERM_NAMES):
            raise ExportError(
                f"{where} stage {index} would leave {sorted(remaining)}; expected upstream's twelve terms"
            )
        projected = dict(stage)
        projected["objective_weights"] = remaining
        projected_stages.append(projected)
    result = dict(curriculum)
    result["stages"] = projected_stages
    return result


def project_evaluation_config(training_config: Mapping[str, Any]) -> dict[str, Any]:
    """Return an upstream-only reconstruction config for the official evaluator.

    Only the objective schema and the orientation weight are removed. Thresholds,
    VOC scales, reset probabilities, physics, observation/action semantics,
    reference and task are preserved exactly.
    """
    config = json.loads(json.dumps(training_config))  # deep copy, JSON-safe
    if "objective" not in config:
        raise ExportError("training config has no objective section")

    def parent_of(path: str) -> tuple[Any, str]:
        node: Any = config
        parts = path.split(".")
        for part in parts[:-1]:
            node = node.get(part) if isinstance(node, dict) else None
        return node, parts[-1]

    weights: set[float] = set()
    targets: set[str] = set()
    projected_objectives = 0
    for path in _OBJECTIVE_PATHS:
        node, leaf = parent_of(path)
        if not isinstance(node, dict) or leaf not in node:
            continue
        node[leaf], weight, target = _project_objective(node[leaf])
        weights.add(weight)
        targets.add(target)
        projected_objectives += 1
    if projected_objectives == 0:
        raise ExportError("training config has no objective to project")
    if len(targets) != 1:
        raise ExportError(f"objective copies disagree on the objective target: {sorted(targets)}")
    if len(weights) != 1:
        raise ExportError(f"objective copies disagree on the thirteenth-term weight: {sorted(weights)}")
    weight = weights.pop()
    target = targets.pop()
    terms = _appended_terms(target)

    projected_curricula = 0
    for path in _CURRICULUM_PATHS:
        node, leaf = parent_of(path)
        if not isinstance(node, dict) or leaf not in node:
            continue
        node[leaf] = _project_curriculum(node[leaf], path, terms)
        projected_curricula += 1
    if projected_curricula == 0:
        raise ExportError("training config has no curriculum to project")

    # Nothing outside the objective/curriculum may differ.
    for root in _IMMUTABLE_ROOTS:
        if root in config and config[root] != training_config.get(root):
            raise ExportError(f"projection changed {root!r}, which must be preserved exactly")
    # ``env`` minus its objective copy must also be untouched.
    original_env = training_config.get("env")
    if isinstance(config.get("env"), dict) and isinstance(original_env, dict):
        changed = {
            key
            for key in set(config["env"]) | set(original_env)
            if key != "objective" and config["env"].get(key) != original_env.get(key)
        }
        if changed:
            raise ExportError(f"projection changed env keys that must be preserved: {sorted(changed)}")

    remaining = json.dumps(config)
    if any(local in remaining for local in THIRTEENTH_TERM_OBJECTIVES):
        raise ExportError("projected config still names the local training module")
    for term in terms:
        if f'"{term}"' in remaining:
            raise ExportError(f"projected config still carries an {term} weight")
    config["_orientation_projection"] = {
        "note": "EVALUATION RECONSTRUCTION ONLY. This is not the reward that trained the actor.",
        "removed_objective_target": target,
        "removed_term": ",".join(terms),
        "removed_weight": weight,
        "training_config_key": TRAINING_CONFIG_KEY,
    }
    return config


# --- export -----------------------------------------------------------------


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_official_actor(
    native_policy: Path,
    destination: Path,
    provenance: Mapping[str, str] | None = None,
) -> Path:
    """Write a portable actor whose tensors are byte-identical to ``native_policy``."""
    native_policy = Path(native_policy)
    destination = Path(destination)
    if destination.exists():
        raise ExportError(f"{destination} already exists; refusing to overwrite an exported actor")
    if destination.resolve() == native_policy.resolve():
        raise ExportError("destination must differ from the source checkpoint")

    header, payload = read_checkpoint(native_policy)
    metadata = dict(header.get("__metadata__") or {})
    if CONFIG_KEY not in metadata:
        raise ExportError(f"{native_policy} has no {CONFIG_KEY}; is it a FlashCHORD checkpoint?")
    if metadata.get("user.checkpoint_kind") not in (None, "policy"):
        raise ExportError(
            f"{native_policy} is a {metadata.get('user.checkpoint_kind')!r} checkpoint; "
            "only actor-only policy files may be projected. Learner state and replay never are."
        )

    training_config = json.loads(metadata[CONFIG_KEY])
    evaluation_config = project_evaluation_config(training_config)
    target = evaluation_config["_orientation_projection"]["removed_objective_target"]
    _, _, definition = THIRTEENTH_TERM_OBJECTIVES[target]
    terms = _appended_terms(target)

    exported = dict(metadata)
    exported[CONFIG_KEY] = json.dumps(evaluation_config, separators=(",", ":"), sort_keys=True)
    exported[TRAINING_CONFIG_KEY] = metadata[CONFIG_KEY]
    exported[REWARD_DEFINITION_KEY] = definition
    exported["user.export_kind"] = "orientation_projected_actor"
    exported["user.export_parent_sha256"] = _sha256(native_policy)
    exported["user.export_parent_name"] = native_policy.name
    exported["user.export_exporter_sha256"] = _sha256(Path(__file__))
    exported["user.export_objective_terms_trained"] = str(len(UPSTREAM_TERM_NAMES) + len(terms))
    exported["user.export_objective_terms_evaluated"] = str(len(UPSTREAM_TERM_NAMES))
    trained_with = f"a thirteenth {terms[0]} reward term" if len(terms) == 1 else "extra reward terms " + ", ".join(terms)
    exported["user.export_warning"] = (
        "resolved_config_json is an evaluation-only reconstruction. The actor was trained with "
        f"{trained_with}; see " + TRAINING_CONFIG_KEY
    )
    for name, value in (provenance or {}).items():
        exported[f"user.export_{name}"] = str(value)

    new_header = {**tensor_index(header), "__metadata__": exported}
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_checkpoint(destination, new_header, payload)

    # Prove the tensors survived unchanged.
    written_header, written_payload = read_checkpoint(destination)
    if tensor_index(written_header) != tensor_index(header):
        raise ExportError("exported tensor index differs from the source")
    if written_payload != payload:
        raise ExportError("exported tensor payload differs from the source")
    return destination


#: ``main`` exit code for a stock twelve-term checkpoint: nothing to project, use it as-is.
EXIT_STOCK = 10


def main(argv: list[str] | None = None) -> int:
    """``python3 -m pipeline.flash_sac_actor_export SOURCE DEST`` -- project only if needed.

    Exit 0 when DEST was written, :data:`EXIT_STOCK` when SOURCE was trained by
    upstream's own objective and the evaluator can load it unchanged.
    """
    import argparse
    import sys

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args(argv)
    metadata = read_checkpoint(args.source)[0].get("__metadata__") or {}
    target = json.loads(metadata.get(CONFIG_KEY, "{}")).get("objective", {}).get("_target_")
    if target == UPSTREAM_TARGET:
        print(f"{args.source.name}: stock twelve-term objective, no projection needed")
        return EXIT_STOCK
    try:
        export_official_actor(args.source, args.destination)
    except ExportError as error:
        print(f"export failed: {error}", file=sys.stderr)
        return 1
    print(f"{args.source.name} ({target}) -> {args.destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
