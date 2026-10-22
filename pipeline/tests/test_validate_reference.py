"""Unit tests for the G1 reference validator's gates (Step C).

These exist because the gates themselves are the thing most likely to be wrong
in a way that looks right: a gate that fails healthy data costs 17 episodes, and
a gate that passes broken data costs a training run. Each test below mutates one
field of a real reference and asserts the validator reacts the way the measured
evidence says it should.

The headline case is ``test_iteration_saturation_is_reported_not_failed``. An
earlier version of this validator failed any episode whose IK solve reached the
200-iteration cap after frame 0, which failed 17 of 30 Tier 1 episodes. That
gate was retired because for G1 the cap is not a failure signal at all:
``posture_task.q_prev_cost`` is 0.05, so ``has_extra_tasks`` is always True and
the plateau exit in ``whole_body_kinematics.py`` is unreachable, leaving
``vel_norm < 1e-6`` as the only early exit. See section 9 of
``pipeline/validate_reference.py`` for the full measurement. This test keeps the
gate from coming back.
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("pyarrow")

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from pipeline.validate_reference import (
    G1_URDF,
    IK_MAX_ITER,
    JOINT_VELOCITY_LIMIT_FRACTION,
    RESIDUAL_SPIKE_TOLERANCE,
    load_urdf_joint_limits,
    run_checks,
)

REPO = Path(__file__).resolve().parents[2]
#: Episode 0 of Tier 1 - the hand-built, documented Step B reference, and the
#: one file whose numbers are quoted throughout the docs.
REFERENCE = (
    REPO / "data/adapter_out/whole_body/soma"
    / "sequence_id=2026-06-04_14-16-23_tall_bar_stool_tip_up_04"
    / "robot_name=g1/data.parquet"
)
REFERENCE_FRAMES = 778

pytestmark = pytest.mark.skipif(
    not REFERENCE.is_file(),
    reason=(
        "needs Episode 0's retargeted reference; run "
        "scripts/data/run_tier_pipeline.py retarget --episodes 0"
    ),
)


@pytest.fixture(scope="module")
def table() -> pa.Table:
    return pq.read_table(REFERENCE)


def write_variant(table: pa.Table, out: Path, **columns) -> Path:
    """Rewrite the single motion_v1 row with some columns replaced."""
    data = table.to_pydict()
    for name, value in columns.items():
        assert name in data, f"{name} is not a column of this reference"
        data[name] = [value]
    pq.write_table(pa.Table.from_pydict(data, schema=table.schema), out)
    return out


def check(path: Path, **kwargs):
    """Run the validator silently; frames default to the file's own length."""
    frames = kwargs.pop("frames", None)
    if frames is None:
        frames = len(pq.read_table(path).column("robot_root_position")[0].as_py())
    return run_checks(path, frames=frames, emit=lambda _line: None, **kwargs)


# --------------------------------------------------------------- URDF limits
def test_urdf_limits_are_readable_as_data():
    """The validator must not need torch or Isaac Sim to know G1's limits."""
    limits = load_urdf_joint_limits(G1_URDF)
    assert limits, f"no joint limits parsed from {G1_URDF}"
    assert len(limits) == 43
    elbow = limits["left_elbow_joint"]
    assert elbow["velocity"] == pytest.approx(37.0)
    assert elbow["lower"] < elbow["upper"]


def test_urdf_limits_missing_file_is_empty_not_an_error(tmp_path: Path):
    """A missing URDF must skip the check, never crash and never pass by default."""
    assert load_urdf_joint_limits(tmp_path / "nope.urdf") == {}


def test_missing_urdf_skips_the_check_with_a_note(tmp_path: Path):
    report = check(REFERENCE, urdf=tmp_path / "nope.urdf")
    assert report.ok
    assert any("URDF not found" in note for note in report.checker.notes)
    assert report.checker.metrics["joint_velocity_worst_fraction"] is None


# ------------------------------------------------- the gate that was retired
def test_the_unmodified_reference_passes():
    report = check(REFERENCE)
    assert report.ok, report.checker.failures


def test_iteration_saturation_is_reported_not_failed(table, tmp_path: Path):
    """Every frame at the iteration cap must NOT fail the episode.

    Regression guard for the 17/30 Tier 1 failure. The cap means the whole-body
    QP did not reach a 1e-6 velocity norm, which is an asymptotic condition, not
    a divergence; residual quality is gated separately.
    """
    frames = len(table.column("ik_num_iterations")[0].as_py())
    path = write_variant(
        table, tmp_path / "saturated.parquet",
        ik_num_iterations=[IK_MAX_ITER] * frames,
    )
    report = check(path)
    assert report.ok, report.checker.failures
    assert not any("iteration cap" in f for f in report.checker.failures)
    assert report.checker.metrics["ik_saturated_after_frame0"] == frames - 1
    assert report.checker.metrics["ik_saturated_fraction"] == 1.0
    # Reported, never hidden.
    assert any("reached the 200-iteration cap" in n for n in report.checker.notes)


def test_saturation_is_still_counted_when_it_does_not_fail(table, tmp_path: Path):
    """A reported metric is only useful if it still tracks reality."""
    iterations = list(table.column("ik_num_iterations")[0].as_py())
    before = sum(1 for i, n in enumerate(iterations) if i != 0 and n >= IK_MAX_ITER)
    iterations[5] = IK_MAX_ITER
    iterations[9] = IK_MAX_ITER
    path = write_variant(
        table, tmp_path / "two_saturated.parquet", ik_num_iterations=iterations
    )
    report = check(path)
    assert report.ok
    assert report.checker.metrics["ik_saturated_after_frame0"] == before + 2


# ------------------------------------------------------ the gates that replaced it
def test_a_single_diverged_frame_fails_the_residual_spike_gate(table, tmp_path: Path):
    """One broken frame must not hide inside a healthy median."""
    residuals = list(table.column("ik_error_per_frame")[0].as_py())
    median = float(np.median(residuals))
    residuals[400] = median * (RESIDUAL_SPIKE_TOLERANCE + 1.0)
    path = write_variant(
        table, tmp_path / "spike.parquet", ik_error_per_frame=residuals
    )
    report = check(path)
    assert not report.ok
    assert any("spikes away from the episode median" in f
               for f in report.checker.failures)
    assert report.checker.metrics["ik_worst_frame"] == 400


def test_an_untrackable_joint_step_fails_the_velocity_gate(table, tmp_path: Path):
    """A reference the robot physically cannot execute is not a usable target."""
    names = table.column("robot_joint_names")[0].as_py()
    positions = np.asarray(
        table.column("robot_joint_positions")[0].as_py(), dtype=np.float64
    )
    elbow = names.index("left_elbow_joint")
    limit = load_urdf_joint_limits(G1_URDF)["left_elbow_joint"]["velocity"]
    fps = float(table.column("fps")[0].as_py())
    # Two frames apart so the induced step is one-sided and unambiguous.
    positions[400, elbow] += 2.0 * limit / fps
    path = write_variant(
        table, tmp_path / "fast.parquet",
        robot_joint_positions=positions.tolist(),
    )
    report = check(path)
    assert not report.ok
    assert any("URDF velocity limit" in f for f in report.checker.failures)
    assert report.checker.metrics["joint_velocity_worst_fraction"] > (
        JOINT_VELOCITY_LIMIT_FRACTION
    )


def test_undriven_hand_joints_are_excluded_from_the_feasibility_check(table):
    """G1's 14 hand joints are constant here and in NVIDIA's demo.

    Measuring them would report the rig's rest pose - they sit on a position
    limit for the whole sequence - rather than anything about this reference.
    """
    report = check(REFERENCE)
    assert report.ok
    names = table.column("robot_joint_names")[0].as_py()
    hand = [n for n in names if "hand" in n]
    assert len(hand) == 14
    positions = np.asarray(
        table.column("robot_joint_positions")[0].as_py(), dtype=np.float64
    )
    indices = [names.index(n) for n in hand]
    assert np.allclose(positions[:, indices], positions[0, indices]), (
        "hand joints are expected to be undriven; if that changes, the "
        "feasibility check's 'driven' filter needs revisiting"
    )


# --------------------------------------------------------- reporting contract
def test_dropped_frames_are_recorded_as_a_note():
    """Episode 11 drops its final unmeasured frame; the gate output must say so."""
    report = check(REFERENCE, dropped_frames=(900,))
    assert report.ok
    assert report.checker.metrics["dropped_frames"] == 1
    note = next(n for n in report.checker.notes if "deliberately absent" in n)
    assert "[900]" in note
    assert "never measured" in note


def test_summary_banner_can_be_suppressed_for_batch_runs():
    """A per-episode 'ALL CHECKS PASSED' printed 30 times reads as a tier verdict."""
    lines: list[str] = []
    run_checks(REFERENCE, frames=REFERENCE_FRAMES, summary=False, emit=lines.append)
    assert not any(line.startswith("ALL CHECKS PASSED") for line in lines)

    lines.clear()
    run_checks(REFERENCE, frames=REFERENCE_FRAMES, summary=True, emit=lines.append)
    assert any(line.startswith("ALL CHECKS PASSED") for line in lines)
