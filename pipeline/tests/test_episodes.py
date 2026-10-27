"""Unit tests for the episode registry (Step C - generalizing A+B to a tier).

These cover the decisions that would otherwise only be checked by running the
whole 30-episode pipeline:

* the support-surface collision-shape scan, against both published shapes and a
  crafted stage that uses USD composition;
* ``scene.support``, which FlashCHORD raises on if it is wrong;
* the trailing-invalid-frame policy, including the refusal to handle an invalid
  frame in the middle of a sequence;
* tier isolation - Tier 1 and Tier 2 publish the same sequence ids, so their
  staged trees and container mount points must not collide;
* the episode selector, which must reject a typo rather than silently run a
  subset.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline.episodes import (
    TIER_1,
    TIER_2,
    EpisodeSpec,
    RegistryError,
    annotate_object_validity,
    count_support_collision_shapes,
    load_tier,
    out_root_for_tier,
    parse_episode_selector,
    select,
    tier_root_for,
)

REPO = Path(__file__).resolve().parents[2]
TIER_1_ROOT = REPO / "data/v2d_challenge/track_2" / TIER_1

pytestmark = pytest.mark.skipif(
    not (TIER_1_ROOT / "meta/episodes_metadata.jsonl").is_file(),
    reason="tier 1 mirror not present",
)

EXPECTED_EPISODES = 30
#: Measured over the published mirror: 16 of 30 support stages are empty stubs.
EXPECTED_WITH_SUPPORT = 14
#: Episode 11 is the only episode with an unmeasured object frame, and it is the
#: last frame of the sequence.
EPISODE_WITH_INVALID_FRAME = 11


@pytest.fixture(scope="module")
def tier_1() -> list[EpisodeSpec]:
    return load_tier(TIER_1)


# --------------------------------------------------------------------------
# registry shape
# --------------------------------------------------------------------------
def test_tier_has_every_declared_episode(tier_1):
    assert len(tier_1) == EXPECTED_EPISODES
    assert [s.episode_index for s in tier_1] == list(range(EXPECTED_EPISODES))


def test_every_input_file_resolves(tier_1):
    for spec in tier_1:
        assert spec.parquet.is_file()
        assert spec.mesh.is_file()
        assert spec.ground_plane.is_file()
        assert spec.support_surface.is_file()


def test_staged_paths_are_what_upstream_resolves(tier_1):
    spec = tier_1[0]
    # soma_to_g1.py's positional data_folder, and the Hive partition it writes.
    assert spec.data_folder.name == spec.sequence_id
    assert spec.retarget_dir.name == f"sequence_id={spec.sequence_id}"
    assert spec.reference_parquet == spec.retarget_dir / "robot_name=g1/data.parquet"
    # registry.py:127 looks for <motion-root>/whole_body/reconstructed_stage/
    assert spec.staged_support.parent.name == "reconstructed_stage"
    assert spec.staged_support.parent.parent == spec.data_folder.parent.parent


def test_episode_0_keeps_its_original_staged_location(tier_1):
    """Step B's artifacts and verify_step_B.sh must not move."""
    spec = tier_1[0]
    assert spec.out_root == REPO / "data/adapter_out"
    assert spec.data_folder == (
        REPO / "data/adapter_out/whole_body/soma"
        / "2026-06-04_14-16-23_tall_bar_stool_tip_up_04"
    )


# --------------------------------------------------------------------------
# support surfaces
# --------------------------------------------------------------------------
def test_support_shape_counts_match_the_published_mirror(tier_1):
    with_support = [s.episode_index for s in tier_1 if s.scene_support]
    assert len(with_support) == EXPECTED_WITH_SUPPORT
    for spec in tier_1:
        assert spec.support_shape_count >= 0
        assert spec.scene_support == (spec.support_shape_count > 0)


def test_empty_stub_is_counted_as_zero(tier_1):
    spec = tier_1[0]  # tall_bar_stool_tip_up_04 ships a 125-byte stub
    assert count_support_collision_shapes(spec.support_surface) == 0
    assert spec.scene_support is False


def test_real_shapes_are_counted(tier_1):
    spec = next(s for s in tier_1 if s.episode_index == 25)
    assert count_support_collision_shapes(spec.support_surface) == 2
    assert spec.scene_support is True


def test_shape_scan_counts_cubes_too(tmp_path: Path):
    stage = tmp_path / "x_support.usda"
    stage.write_text(
        '#usda 1.0\n'
        'def Xform "support_surfaces"\n{\n'
        '    def Cube "a"\n    {\n        double size = 1\n    }\n'
        '    def Cylinder "b"\n    {\n        double radius = 1\n    }\n'
        '    def Sphere "ignored"\n    {\n    }\n'
        "}\n"
    )
    assert count_support_collision_shapes(stage) == 2


def test_shape_scan_refuses_a_composed_stage(tmp_path: Path):
    """A text scan cannot see prims pulled in from another layer, so it must
    refuse rather than under-count them."""
    stage = tmp_path / "composed_support.usda"
    stage.write_text(
        '#usda 1.0\n(\n    subLayers = [\n        @./shapes.usda@\n    ]\n)\n'
        'def Xform "support_surfaces"\n{\n}\n'
    )
    with pytest.raises(RegistryError, match="composition"):
        count_support_collision_shapes(stage)


def test_shape_scan_refuses_a_missing_stage(tmp_path: Path):
    with pytest.raises(RegistryError, match="missing"):
        count_support_collision_shapes(tmp_path / "nope.usda")


# --------------------------------------------------------------------------
# invalid object frames
# --------------------------------------------------------------------------
def test_clean_episode_retargets_every_frame(tier_1):
    spec = annotate_object_validity(tier_1[0], tier_1[0].frames, [])
    spec.validate_frame_policy()
    assert spec.retarget_end_frame is None
    assert spec.retarget_frames == spec.frames
    assert spec.dropped_frames == ()


def test_trailing_invalid_frame_shortens_the_window(tier_1):
    source = next(s for s in tier_1 if s.episode_index == EPISODE_WITH_INVALID_FRAME)
    spec = annotate_object_validity(source, source.frames - 1, [source.frames - 1])
    spec.validate_frame_policy()
    assert spec.retarget_end_frame == source.frames - 1
    assert spec.retarget_frames == source.frames - 1
    assert spec.dropped_frames == (source.frames - 1,)
    assert "not interpolated" in spec.provenance()["dropped_reason"]


def test_trailing_run_of_several_invalid_frames(tier_1):
    source = tier_1[0]
    invalid = [source.frames - 3, source.frames - 2, source.frames - 1]
    spec = annotate_object_validity(source, source.frames - 3, invalid)
    spec.validate_frame_policy()
    assert spec.retarget_frames == source.frames - 3
    assert list(spec.dropped_frames) == invalid


def test_mid_sequence_invalid_frame_is_refused(tier_1):
    """Truncating cannot honour a gap in the middle, and interpolating is banned."""
    source = tier_1[0]
    spec = annotate_object_validity(source, source.frames - 1, [100])
    with pytest.raises(RegistryError, match="MIDDLE"):
        spec.validate_frame_policy()


def test_unprobed_spec_cannot_pass_the_frame_policy(tier_1):
    with pytest.raises(RegistryError, match="never probed"):
        tier_1[0].validate_frame_policy()


# --------------------------------------------------------------------------
# tier isolation
# --------------------------------------------------------------------------
def test_tiers_share_sequence_ids(tier_1):
    """The premise that makes isolation necessary rather than theoretical."""
    tier_2 = load_tier(TIER_2)
    assert [s.sequence_id for s in tier_2] == [s.sequence_id for s in tier_1]


def test_tiers_stage_into_separate_trees():
    one = out_root_for_tier(TIER_1)
    two = out_root_for_tier(TIER_2)
    assert one != two
    assert not str(two).startswith(str(one) + "/")
    assert not str(one).startswith(str(two) + "/")


def test_tier_2_staged_paths_never_collide_with_tier_1(tier_1):
    tier_2 = load_tier(TIER_2)
    for a, b in zip(tier_1, tier_2):
        assert a.sequence_id == b.sequence_id
        assert a.data_folder != b.data_folder
        assert a.reference_parquet != b.reference_parquet
        assert a.staged_support != b.staged_support


def test_an_unknown_tier_gets_its_own_root():
    """A future Tier 3 reconstruction must not default into Tier 1's tree."""
    root = out_root_for_tier("tier_3_reconstruction")
    assert root != out_root_for_tier(TIER_1)
    assert "tier_3_reconstruction" in root.name


def test_a_reconstructed_tier_resolves_outside_the_mirror(tmp_path: Path):
    """Our Tier 3 lives under data/reconstructed/, never inside the pristine HF mirror."""
    (tmp_path / "data/reconstructed/track_2/tier_3_x/meta").mkdir(parents=True)
    assert tier_root_for("tier_3_x", tmp_path) == tmp_path / "data/reconstructed/track_2/tier_3_x"
    assert tier_root_for(TIER_1, tmp_path) == tmp_path / "data/v2d_challenge/track_2" / TIER_1


def test_a_published_tier_wins_over_a_reconstructed_namesake(tmp_path: Path):
    for root in ("data/v2d_challenge/track_2/tier_3_x", "data/reconstructed/track_2/tier_3_x"):
        (tmp_path / root).mkdir(parents=True)
    assert tier_root_for("tier_3_x", tmp_path) == tmp_path / "data/v2d_challenge/track_2/tier_3_x"


def test_provenance_records_the_tier_and_the_source_hash(tier_1):
    spec = annotate_object_validity(tier_1[0], tier_1[0].frames, [])
    record = spec.provenance()
    assert record["tier"] == TIER_1
    assert len(record["source_sha256"]["parquet"]) == 64
    assert record["scene_support"] is False
    # json-serialisable, because it is written into the staged tree verbatim.
    json.dumps(record)


# --------------------------------------------------------------------------
# selector
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "selector,expected",
    [
        ("all", list(range(5))),
        ("", list(range(5))),
        ("0", [0]),
        ("0,2,4", [0, 2, 4]),
        ("1-3", [1, 2, 3]),
        ("4,0-1", [0, 1, 4]),
        (" 2 , 3 ", [2, 3]),
    ],
)
def test_selector_parsing(selector, expected):
    assert parse_episode_selector(selector, list(range(5))) == expected


@pytest.mark.parametrize("selector", ["7", "0-9", "x", "3-1", "0,99"])
def test_selector_rejects_out_of_range_and_garbage(selector):
    with pytest.raises(RegistryError):
        parse_episode_selector(selector, list(range(5)))


def test_select_preserves_dataset_order(tier_1):
    chosen = select(tier_1, "5,1,3")
    assert [s.episode_index for s in chosen] == [1, 3, 5]
