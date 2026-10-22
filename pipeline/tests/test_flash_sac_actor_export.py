"""Host tests for the portable actor export.

The exporter is stdlib-only, so these run under ``.venvs/episode0`` with no
container. They cover the projection boundary (what may change and what must
not), the refusals, and byte-identical tensors.
"""

from __future__ import annotations

import copy
import json
import struct

import pytest

from pipeline.flash_sac_actor_export import (
    CONFIG_KEY,
    ORIENTATION_TARGET,
    REWARD_DEFINITION_KEY,
    TRAINING_CONFIG_KEY,
    UPSTREAM_TARGET,
    UPSTREAM_TERM_NAMES,
    ExportError,
    export_official_actor,
    project_evaluation_config,
    read_checkpoint,
    tensor_index,
    write_checkpoint,
)

OBJECTIVE = {
    "_target_": ORIENTATION_TARGET,
    **{f"{name}_weight": 1.0 for name in UPSTREAM_TERM_NAMES},
    "pelvis_position_std": 0.3,
    "pelvis_orientation_std": 0.4,
    "body_joint_position_std": 1.0,
    "object_position_std": 0.2,
    "palm_position_std": 0.2,
    "palm_orientation_std": 0.4,
    "soft_joint_limit_factor": 0.9,
    "force_closure_min_support": 0.01,
    "num_wrench_basis": 512,
    "num_friction_cone_edges": 8,
    "friction_coefficient": 0.1,
    "wrench_basis_seed": 0,
    "sides": ["left", "right"],
    "object_orientation_weight": 2.0,
}

STAGE = {
    "_target_": "flash_chord.configuration.make_curriculum_stage",
    "voc_scale": 1.0,
    "objective_weights": {**{name: 1.0 for name in UPSTREAM_TERM_NAMES}, "object_orientation": 2.0},
}
CURRICULUM = {"_target_": "flash_chord.configuration.make_fixed_curriculum", "thresholds": [1, 2], "stages": [STAGE, STAGE]}


def training_config() -> dict:
    return copy.deepcopy(
        {
            "objective": OBJECTIVE,
            "curriculum": CURRICULUM,
            "training": {"curriculum": CURRICULUM, "seed": 42},
            "env": {
                "objective": OBJECTIVE,
                "voc_scale": 1.0,
                # Upstream's 1.5 rad FAILURE limit -- not our reward term.
                "termination": {"object_orientation_threshold": 1.5},
            },
            "termination": {"object_orientation_threshold": 1.5},
            "action": {"input_mode": "raw"},
            "observation": {"dim": 486},
            "embodiment": {"name": "g1_dex3"},
            "sim": {"fps": 50},
            "reset": {"seed": 42},
            "scene": {"world_count": 1024, "support": False},
            "task": {"parquet": "/work/adapter_out/.../data.parquet"},
        }
    )


# --- what the projection changes -------------------------------------------


def test_both_objective_copies_are_repointed_at_upstream() -> None:
    projected = project_evaluation_config(training_config())
    assert projected["objective"]["_target_"] == UPSTREAM_TARGET
    assert projected["env"]["objective"]["_target_"] == UPSTREAM_TARGET
    assert "object_orientation_weight" not in projected["objective"]
    assert "object_orientation_weight" not in projected["env"]["objective"]


def test_both_curricula_drop_the_thirteenth_weight() -> None:
    projected = project_evaluation_config(training_config())
    for curriculum in (projected["curriculum"], projected["training"]["curriculum"]):
        for stage in curriculum["stages"]:
            assert set(stage["objective_weights"]) == set(UPSTREAM_TERM_NAMES)
            assert stage["voc_scale"] == 1.0


def test_projection_records_what_it_removed() -> None:
    projected = project_evaluation_config(training_config())
    note = projected["_orientation_projection"]
    assert note["removed_weight"] == 2.0
    assert note["removed_term"] == "object_orientation"
    assert "not the reward that trained" in note["note"]


# --- what it must not change ------------------------------------------------


def test_termination_threshold_is_preserved() -> None:
    """``object_orientation_threshold`` is upstream's failure limit, not our term."""
    projected = project_evaluation_config(training_config())
    assert projected["termination"]["object_orientation_threshold"] == 1.5
    assert projected["env"]["termination"]["object_orientation_threshold"] == 1.5


@pytest.mark.parametrize("root", ["action", "observation", "embodiment", "sim", "reset", "scene", "task", "termination"])
def test_non_objective_roots_are_untouched(root: str) -> None:
    config = training_config()
    projected = project_evaluation_config(config)
    assert projected[root] == config[root]


def test_env_is_untouched_apart_from_its_objective() -> None:
    config = training_config()
    projected = project_evaluation_config(config)
    for key in config["env"]:
        if key != "objective":
            assert projected["env"][key] == config["env"][key]


def test_thresholds_and_stage_count_survive() -> None:
    config = training_config()
    projected = project_evaluation_config(config)
    assert projected["curriculum"]["thresholds"] == config["curriculum"]["thresholds"]
    assert len(projected["curriculum"]["stages"]) == len(config["curriculum"]["stages"])


# --- refusals ---------------------------------------------------------------


def test_refuses_a_config_that_was_not_orientation_trained() -> None:
    config = training_config()
    config["objective"]["_target_"] = UPSTREAM_TARGET
    config["env"]["objective"]["_target_"] = UPSTREAM_TARGET
    with pytest.raises(ExportError, match="expected"):
        project_evaluation_config(config)


def test_refuses_an_unknown_objective_key() -> None:
    config = training_config()
    config["objective"]["mystery_weight"] = 1.0
    with pytest.raises(ExportError, match="upstream does not define"):
        project_evaluation_config(config)


def test_refuses_a_stage_without_the_orientation_weight() -> None:
    config = training_config()
    config["curriculum"]["stages"][0] = copy.deepcopy(STAGE)
    del config["curriculum"]["stages"][0]["objective_weights"]["object_orientation"]
    with pytest.raises(ExportError, match="no object_orientation weight"):
        project_evaluation_config(config)


def test_refuses_a_stage_missing_an_upstream_term() -> None:
    config = training_config()
    config["curriculum"]["stages"][0] = copy.deepcopy(STAGE)
    del config["curriculum"]["stages"][0]["objective_weights"]["force_closure"]
    with pytest.raises(ExportError, match="expected upstream's twelve terms"):
        project_evaluation_config(config)


def test_refuses_when_the_two_objective_copies_disagree() -> None:
    config = training_config()
    config["env"]["objective"] = copy.deepcopy(OBJECTIVE)
    config["env"]["objective"]["object_orientation_weight"] = 5.0
    with pytest.raises(ExportError, match="disagree"):
        project_evaluation_config(config)


# --- the safetensors container ---------------------------------------------


def make_checkpoint(path, metadata: dict, payload: bytes = b"\x00" * 64) -> None:
    header = {
        "weights": {"dtype": "F32", "shape": [4, 4], "data_offsets": [0, 64]},
        "__metadata__": metadata,
    }
    write_checkpoint(path, header, payload)


def test_export_preserves_tensors_byte_for_byte(tmp_path) -> None:
    payload = bytes(range(64))
    source = tmp_path / "policy.safetensors"
    make_checkpoint(
        source,
        {CONFIG_KEY: json.dumps(training_config()), "user.checkpoint_kind": "policy"},
        payload,
    )
    destination = tmp_path / "portable.safetensors"
    export_official_actor(source, destination, provenance={"arm": "orientation"})

    source_header, source_payload = read_checkpoint(source)
    exported_header, exported_payload = read_checkpoint(destination)
    assert exported_payload == source_payload == payload
    assert tensor_index(exported_header) == tensor_index(source_header)


def test_export_keeps_the_true_training_config_and_labels_the_projection(tmp_path) -> None:
    source = tmp_path / "policy.safetensors"
    make_checkpoint(source, {CONFIG_KEY: json.dumps(training_config()), "user.checkpoint_kind": "policy"})
    destination = tmp_path / "portable.safetensors"
    export_official_actor(source, destination)

    metadata = read_checkpoint(destination)[0]["__metadata__"]
    evaluation = json.loads(metadata[CONFIG_KEY])
    training = json.loads(metadata[TRAINING_CONFIG_KEY])
    assert evaluation["objective"]["_target_"] == UPSTREAM_TARGET
    assert training["objective"]["_target_"] == ORIENTATION_TARGET
    assert len(training["curriculum"]["stages"][0]["objective_weights"]) == 13
    assert "-(theta / pi) ** 2" in metadata[REWARD_DEFINITION_KEY]
    assert metadata["user.export_objective_terms_trained"] == "13"
    assert metadata["user.export_objective_terms_evaluated"] == "12"
    assert "thirteenth" in metadata["user.export_warning"]


def test_provenance_is_namespaced_into_the_metadata(tmp_path) -> None:
    source = tmp_path / "policy.safetensors"
    make_checkpoint(source, {CONFIG_KEY: json.dumps(training_config()), "user.checkpoint_kind": "policy"})
    destination = tmp_path / "portable.safetensors"
    export_official_actor(source, destination, provenance={"arm": "orientation", "episode": "0"})
    metadata = read_checkpoint(destination)[0]["__metadata__"]
    assert metadata["user.export_arm"] == "orientation"
    assert metadata["user.export_episode"] == "0"
    assert len(metadata["user.export_parent_sha256"]) == 64
    assert len(metadata["user.export_exporter_sha256"]) == 64


def test_export_refuses_to_overwrite(tmp_path) -> None:
    source = tmp_path / "policy.safetensors"
    make_checkpoint(source, {CONFIG_KEY: json.dumps(training_config()), "user.checkpoint_kind": "policy"})
    destination = tmp_path / "portable.safetensors"
    export_official_actor(source, destination)
    with pytest.raises(ExportError, match="refusing to overwrite"):
        export_official_actor(source, destination)


def test_export_refuses_to_write_over_its_source(tmp_path) -> None:
    source = tmp_path / "policy.safetensors"
    make_checkpoint(source, {CONFIG_KEY: json.dumps(training_config()), "user.checkpoint_kind": "policy"})
    with pytest.raises(ExportError):
        export_official_actor(source, source)


def test_export_refuses_a_learner_state_file(tmp_path) -> None:
    """Learner state and replay are never projected or offered as policy artifacts."""
    source = tmp_path / "state.safetensors"
    make_checkpoint(
        source,
        {CONFIG_KEY: json.dumps(training_config()), "user.checkpoint_kind": "training_state"},
        b"\x00" * 64,
    )
    with pytest.raises(ExportError, match="only actor-only policy files"):
        export_official_actor(source, tmp_path / "out.safetensors")


def test_export_refuses_a_file_without_a_config(tmp_path) -> None:
    source = tmp_path / "policy.safetensors"
    make_checkpoint(source, {"user.checkpoint_kind": "policy"})
    with pytest.raises(ExportError, match="resolved_config_json"):
        export_official_actor(source, tmp_path / "out.safetensors")


def test_written_payload_starts_eight_byte_aligned(tmp_path) -> None:
    path = tmp_path / "aligned.safetensors"
    make_checkpoint(path, {CONFIG_KEY: json.dumps(training_config()), "user.checkpoint_kind": "policy"})
    blob = path.read_bytes()
    (length,) = struct.unpack("<Q", blob[:8])
    assert (8 + length) % 8 == 0
