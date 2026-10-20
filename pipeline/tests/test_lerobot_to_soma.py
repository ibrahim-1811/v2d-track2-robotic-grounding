"""Unit tests for the LeRobot -> SOMA adapter (Step 8.3).

Covers the six areas required by the task: shapes, joint ordering, quaternion
conversion, finite values, rotation-matrix validity, and frame preservation -
plus the npz contract and the no-invention guarantee.

These run against the real Episode 0 Parquet. They import nothing from
official/; the upstream contract is mirrored as data in the adapter module.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pipeline.lerobot_to_soma import (
    NUM_SOMA_JOINTS,
    POSE_VALUES_PER_FRAME,
    REQUIRED_NPZ_KEYS,
    AdapterError,
    Episode,
    build_soma_params,
    collapse_joint_names,
    load_episode,
    object_poses_to_matrices,
    quaternion_wxyz_to_matrix,
    reshape_human_pose,
    write_outputs,
)

REPO = Path(__file__).resolve().parents[2]
TIER = REPO / "data/v2d_challenge/track_2/tier_1_multiview_caption"
EXPECTED_FRAMES = 778

pytestmark = pytest.mark.skipif(
    not (TIER / "data/chunk-000/episode_000000.parquet").is_file(),
    reason="Episode 0 Parquet not present",
)


@pytest.fixture(scope="module")
def ep() -> Episode:
    return load_episode(TIER, 0)


@pytest.fixture(scope="module")
def written(ep: Episode, tmp_path_factory) -> dict[str, Path]:
    out = tmp_path_factory.mktemp("adapter_out") / ep.sequence_id
    return write_outputs(ep, out)


# ---------------------------------------------------------------- shapes ----
def test_frame_count_matches_metadata(ep: Episode):
    assert ep.poses.shape[0] == EXPECTED_FRAMES


def test_array_shapes(ep: Episode):
    assert ep.poses.shape == (EXPECTED_FRAMES, NUM_SOMA_JOINTS, 3)
    assert ep.transl.shape == (EXPECTED_FRAMES, 3)
    assert ep.identity_coeffs.shape == (EXPECTED_FRAMES, 45)
    assert ep.scale_params.shape == (EXPECTED_FRAMES, 68)
    assert ep.bone_length_flexibles.shape == (EXPECTED_FRAMES, 6)
    assert ep.object_pose.shape == (EXPECTED_FRAMES, 7)
    assert ep.object_visible.shape == (EXPECTED_FRAMES,)


def test_dtypes_are_float32(ep: Episode):
    for a in (ep.poses, ep.transl, ep.identity_coeffs, ep.scale_params):
        assert a.dtype == np.float32


def test_object_matrices_shape(written):
    assert np.load(written["poses"]).shape == (EXPECTED_FRAMES, 4, 4)


# --------------------------------------------------------- joint ordering ----
def test_joint_count_and_anchors(ep: Episode):
    assert len(ep.joint_names) == NUM_SOMA_JOINTS
    assert ep.joint_names[0] == "Hips"            # the root must be index 0
    assert ep.joint_names[-1] == "RightToeEnd"


def test_known_joint_indices(ep: Episode):
    # Verified against meta/info.json in docs/EPISODE_0_WALKTHROUGH.md section 3.1
    expected = {"Hips": 0, "LeftArm": 12, "LeftHand": 14, "RightArm": 40,
                "RightHand": 42, "LeftLeg": 67, "LeftFoot": 69,
                "RightLeg": 72, "RightFoot": 74}
    for name, idx in expected.items():
        assert ep.joint_names[idx] == name, f"{name} expected at {idx}"


def test_joint_names_unique(ep: Episode):
    assert len(set(ep.joint_names)) == NUM_SOMA_JOINTS


def test_collapse_rejects_non_triplet_layout():
    """If the 231 values were NOT per-joint triplets, the reshape would be wrong."""
    # interleaved by axis (all .rx first) instead of grouped by joint
    bad = [f"J{j}.r{ax}" for ax in "xyz" for j in range(NUM_SOMA_JOINTS)]
    with pytest.raises(AdapterError, match="consecutive per-joint triplets"):
        collapse_joint_names(bad)


def test_collapse_rejects_wrong_length():
    with pytest.raises(AdapterError, match="expected 231"):
        collapse_joint_names(["Hips.rx", "Hips.ry"])


def test_reshape_regroups_values_correctly():
    """Value 3*j+k must land at [j, k] - the heart of the 231 -> (77,3) step."""
    flat = np.arange(POSE_VALUES_PER_FRAME, dtype=np.float32)[None, :]
    out = reshape_human_pose(flat)
    assert out.shape == (1, NUM_SOMA_JOINTS, 3)
    for j in (0, 1, 40, 76):
        for k in range(3):
            assert out[0, j, k] == 3 * j + k


def test_reshape_rejects_wrong_width():
    with pytest.raises(AdapterError, match=r"\(T, 231\)"):
        reshape_human_pose(np.zeros((5, 230), dtype=np.float32))


# --------------------------------------------- quaternion / rotation maths ----
def test_quaternion_identity_is_identity():
    m = quaternion_wxyz_to_matrix(np.array([[1.0, 0.0, 0.0, 0.0]]))
    assert np.allclose(m[0], np.eye(3), atol=1e-12)


def test_quaternion_90deg_about_z_is_correct():
    """A known case pinning the w-FIRST convention and the sign of rotation."""
    s = np.sqrt(0.5)
    m = quaternion_wxyz_to_matrix(np.array([[s, 0.0, 0.0, s]]))[0]  # w,x,y,z
    expected = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert np.allclose(m, expected, atol=1e-12)


def test_w_first_differs_from_w_last_on_real_data(ep: Episode):
    """Guards the most dangerous silent bug: reading the quaternion w-last."""
    q = ep.object_pose[:, 3:7].astype(np.float64)
    correct = quaternion_wxyz_to_matrix(q)
    swapped = quaternion_wxyz_to_matrix(q[:, [1, 2, 3, 0]])   # pretend w-last
    assert not np.allclose(correct, swapped, atol=1e-3)


def test_quaternion_roundtrips_via_scipy(ep: Episode):
    """Cross-check our hand-written expansion against an independent impl."""
    sp = pytest.importorskip("scipy.spatial.transform")
    q = ep.object_pose[:, 3:7].astype(np.float64)
    ours = quaternion_wxyz_to_matrix(q)
    theirs = sp.Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()  # scipy = w-last
    assert np.allclose(ours, theirs, atol=1e-9)


def test_zero_quaternion_is_rejected():
    with pytest.raises(AdapterError, match="zero quaternion"):
        quaternion_wxyz_to_matrix(np.zeros((1, 4)))


def test_rotation_blocks_are_valid_rotations(written):
    """R R^T = I and det(R) = +1 for every frame - a proper rotation, no scale/flip."""
    m = np.load(written["poses"])
    R = m[:, :3, :3]
    eye = np.einsum("tij,tkj->tik", R, R)
    assert np.allclose(eye, np.eye(3)[None], atol=1e-6)
    assert np.allclose(np.linalg.det(R), 1.0, atol=1e-6)


def test_homogeneous_bottom_row(written):
    m = np.load(written["poses"])
    assert np.allclose(m[:, 3, :], np.array([0.0, 0.0, 0.0, 1.0])[None], atol=0)


# ------------------------------------------------------------- finiteness ----
def test_all_inputs_finite(ep: Episode):
    for name, a in (("poses", ep.poses), ("transl", ep.transl),
                    ("identity_coeffs", ep.identity_coeffs),
                    ("scale_params", ep.scale_params),
                    ("object_pose", ep.object_pose)):
        assert np.all(np.isfinite(a)), f"{name} contains NaN or Inf"


def test_outputs_finite(written):
    assert np.all(np.isfinite(np.load(written["poses"])))
    with np.load(written["soma_params"], allow_pickle=True) as z:
        for k in ("poses", "transl", "identity_coeffs", "scale_params"):
            assert np.all(np.isfinite(z[k])), f"{k} not finite"


def test_quaternions_are_unit_norm(ep: Episode):
    n = np.linalg.norm(ep.object_pose[:, 3:7].astype(np.float64), axis=1)
    assert np.allclose(n, 1.0, atol=1e-6)


# ------------------------------------------------------ FRAME PRESERVATION ----
def test_translation_is_copied_bit_for_bit(ep: Episode, written):
    """poses.npy translations must equal the stored xyz EXACTLY - no conversion."""
    m = np.load(written["poses"])
    assert np.array_equal(m[:, :3, 3], ep.object_pose[:, :3].astype(np.float64))


def test_cv_to_soma_was_NOT_applied(ep: Episode, written):
    """Negative test: prove we did not pre-apply NVIDIA's diag(1,-1,-1).

    soma_to_g1.py:308 applies that matrix itself. If we also applied it, the
    stored Y and Z would be negated relative to the source - and because the
    matrix is its own inverse, downstream would silently get the ORIGINAL
    broken frame back. This asserts the sign is untouched.
    """
    m = np.load(written["poses"])
    src = ep.object_pose[:, :3].astype(np.float64)
    assert np.array_equal(m[:, :3, 3], src)
    flipped = src @ np.diag([1.0, -1.0, -1.0])
    # Y and Z genuinely differ in sign on this episode, so this is a real check.
    assert not np.allclose(m[:, :3, 3], flipped)


def test_human_translation_passed_through_unchanged(ep: Episode, written):
    with np.load(written["soma_params"], allow_pickle=True) as z:
        assert np.array_equal(z["transl"], ep.transl)


def test_human_poses_passed_through_unchanged(ep: Episode, written):
    with np.load(written["soma_params"], allow_pickle=True) as z:
        assert np.array_equal(z["poses"], ep.poses)


# ------------------------------------------------------------ npz contract ----
def test_all_mandatory_keys_present(written):
    with np.load(written["soma_params"], allow_pickle=True) as z:
        assert REQUIRED_NPZ_KEYS <= set(z.files)


def test_joint_orient_is_absent(written):
    """Optional upstream; read_soma.py derives it. Fabricating one is inventing data."""
    with np.load(written["soma_params"], allow_pickle=True) as z:
        assert "joint_orient" not in z.files


def test_scalar_fields_decode_the_way_upstream_parses_them(written):
    """Mirror read_soma.py:631-646 and :388 exactly, without importing it."""
    with np.load(written["soma_params"], allow_pickle=True) as z:
        unit, imt = z["unit"], z["identity_model_type"]
        assert unit.dtype.kind == "U" and str(unit.item()) == "meters"
        assert imt.dtype.kind == "U" and str(imt.item()) == "mhr"
        assert bool(np.asarray(z["keep_root"]).item()) is False
        assert bool(np.asarray(z["absolute_pose"]).item()) is False


def test_joint_names_survive_npz_roundtrip(ep: Episode, written):
    """read_soma.py does [str(n) for n in params['joint_names']] then compares."""
    with np.load(written["soma_params"], allow_pickle=True) as z:
        assert [str(n) for n in z["joint_names"]] == ep.joint_names


def test_identity_and_scale_indexable_at_zero(written):
    """Upstream reads identity_coeffs[0] / scale_params[0]."""
    with np.load(written["soma_params"], allow_pickle=True) as z:
        assert z["identity_coeffs"][0].shape == (45,)
        assert z["scale_params"][0].shape == (68,)


def test_identity_and_scale_are_constant_over_time(ep: Episode):
    """Upstream takes only frame 0; that is lossless only if they never change."""
    assert np.array_equal(ep.identity_coeffs, np.repeat(
        ep.identity_coeffs[:1], ep.identity_coeffs.shape[0], axis=0))
    assert np.array_equal(ep.scale_params, np.repeat(
        ep.scale_params[:1], ep.scale_params.shape[0], axis=0))


# ----------------------------------------------------------- no invention ----
def test_invalid_frames_raise_by_default():
    """The adapter must refuse to guess a pose for an invalid frame."""
    pose = np.zeros((3, 7), dtype=np.float32)
    pose[:, 3] = 1.0
    vis = np.array([True, False, True])
    with pytest.raises(AdapterError, match="visible == False"):
        object_poses_to_matrices(pose, vis)


def test_invalid_frames_passthrough_is_not_interpolated():
    """With passthrough, the invalid row stays obviously invalid - never filled in."""
    pose = np.zeros((3, 7), dtype=np.float32)
    pose[:, 3] = 1.0
    pose[0, :3] = [1.0, 2.0, 3.0]
    pose[2, :3] = [5.0, 6.0, 7.0]
    pose[1] = 0.0
    vis = np.array([True, False, True])
    m = object_poses_to_matrices(pose, vis, on_invalid="passthrough")
    assert np.array_equal(m[1], np.zeros((4, 4)))        # not an average of its neighbours
    assert np.array_equal(m[0, :3, 3], [1.0, 2.0, 3.0])
    assert np.array_equal(m[2, :3, 3], [5.0, 6.0, 7.0])


def test_row_count_is_preserved(ep: Episode, written):
    """No frames dropped, none added."""
    assert np.load(written["poses"]).shape[0] == ep.object_pose.shape[0]


def test_episode_0_has_no_invalid_frames(ep: Episode):
    assert ep.object_visible.all()


# ----------------------------------------------------------------- manifest ----
def test_manifest_records_frame_policy(written):
    man = json.loads(written["manifest"].read_text())
    assert man["frames"] == EXPECTED_FRAMES
    assert "UNCHANGED" in man["object_frame"]
    assert "reconstructed_mesh/output_aligned.glb" in man["still_required_by_soma_to_g1"]


# ------------------------------------------- cross-check against upstream ----
UPSTREAM_PARAMS = (
    REPO / "official/video_to_data/robotic_grounding/source/robotic_grounding"
    / "robotic_grounding/retarget/params.py"
)


@pytest.mark.skipif(not UPSTREAM_PARAMS.is_file(), reason="upstream checkout absent")
def test_joint_names_match_upstream_SOMA_JOINTS_ORDER(ep: Episode):
    """Our 77 names must equal upstream's canonical order, exactly and in order.

    read_soma.py:648 rejects the npz outright if joint_names disagree with the
    live SOMA rig, so this is the check that would otherwise only fail much
    later, after the licensed SOMA assets are staged.

    params.py is PARSED AS TEXT, never imported - importing upstream would drag
    in torch and the licence-gated SOMA assets.
    """
    import ast
    import re

    src = UPSTREAM_PARAMS.read_text()
    m = re.search(r"SOMA_JOINTS_ORDER\s*(?::[^=]+)?=\s*(\[.*?\])", src, re.S)
    assert m, "could not locate SOMA_JOINTS_ORDER in upstream params.py"
    upstream = ast.literal_eval(m.group(1))
    assert len(upstream) == NUM_SOMA_JOINTS
    assert ep.joint_names == upstream
