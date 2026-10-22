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


def _project_objective(objective: dict[str, Any]) -> tuple[dict[str, Any], float]:
    if objective.get("_target_") != ORIENTATION_TARGET:
        raise ExportError(
            f"objective._target_ is {objective.get('_target_')!r}; expected {ORIENTATION_TARGET!r}. "
            "This checkpoint was not produced by the orientation objective."
        )
    projected = dict(objective)
    weight = projected.pop(ORIENTATION_WEIGHT_FIELD, None)
    if weight is None:
        raise ExportError(f"objective is missing {ORIENTATION_WEIGHT_FIELD}")
    projected["_target_"] = UPSTREAM_TARGET
    unexpected = set(projected) - set(_upstream_objective_fields())
    if unexpected:
        raise ExportError(f"objective carries keys upstream does not define: {sorted(unexpected)}")
    return projected, float(weight)


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


def _project_curriculum(curriculum: dict[str, Any], where: str) -> dict[str, Any]:
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
        if ORIENTATION_TERM not in weights:
            raise ExportError(f"{where} stage {index} has no {ORIENTATION_TERM} weight to project away")
        remaining = {name: value for name, value in weights.items() if name != ORIENTATION_TERM}
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
    projected_objectives = 0
    for path in _OBJECTIVE_PATHS:
        node, leaf = parent_of(path)
        if not isinstance(node, dict) or leaf not in node:
            continue
        node[leaf], weight = _project_objective(node[leaf])
        weights.add(weight)
        projected_objectives += 1
    if projected_objectives == 0:
        raise ExportError("training config has no objective to project")
    if len(weights) != 1:
        raise ExportError(f"objective copies disagree on the orientation weight: {sorted(weights)}")
    weight = weights.pop()

    projected_curricula = 0
    for path in _CURRICULUM_PATHS:
        node, leaf = parent_of(path)
        if not isinstance(node, dict) or leaf not in node:
            continue
        node[leaf] = _project_curriculum(node[leaf], path)
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
    if ORIENTATION_TARGET in remaining:
        raise ExportError("projected config still names the local training module")
    if f'"{ORIENTATION_TERM}"' in remaining:
        raise ExportError("projected config still carries an object_orientation weight")
    config["_orientation_projection"] = {
        "note": "EVALUATION RECONSTRUCTION ONLY. This is not the reward that trained the actor.",
        "removed_objective_target": ORIENTATION_TARGET,
        "removed_term": ORIENTATION_TERM,
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

    exported = dict(metadata)
    exported[CONFIG_KEY] = json.dumps(evaluation_config, separators=(",", ":"), sort_keys=True)
    exported[TRAINING_CONFIG_KEY] = metadata[CONFIG_KEY]
    exported[REWARD_DEFINITION_KEY] = REWARD_DEFINITION
    exported["user.export_kind"] = "orientation_projected_actor"
    exported["user.export_parent_sha256"] = _sha256(native_policy)
    exported["user.export_parent_name"] = native_policy.name
    exported["user.export_exporter_sha256"] = _sha256(Path(__file__))
    exported["user.export_objective_terms_trained"] = str(len(UPSTREAM_TERM_NAMES) + 1)
    exported["user.export_objective_terms_evaluated"] = str(len(UPSTREAM_TERM_NAMES))
    exported["user.export_warning"] = (
        "resolved_config_json is an evaluation-only reconstruction. The actor was trained with "
        "a thirteenth object_orientation reward term; see " + TRAINING_CONFIG_KEY
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
