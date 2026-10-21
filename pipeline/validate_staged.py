#!/usr/bin/env python3
"""Validate one episode's STAGED Step A inputs against the pristine Parquet.

Step B's validator looks at the retargeted reference. This one looks at what
goes *into* the retargeter, and it is the check that proves the adapter did not
quietly alter the episode on the way through:

* **identity**   - the staged folder says which tier, episode index and sequence
                   it came from, and the source Parquet's sha256 still matches.
* **frame count**- ``poses``/``transl`` rows == metadata ``frames`` == Parquet rows.
* **timestamps** - monotonic, evenly spaced at 1/fps, duration as published.
                   The npz has no timestamp field (upstream's contract has no
                   slot for one), so the values are re-derived from the Parquet
                   and recorded in the manifest rather than invented.
* **object poses**- the staged ``poses.npy`` is rebuilt from the Parquet and
                   compared **bit-for-bit**. Not "close enough": exactly equal.
* **coordinates**- the translation block of ``poses.npy`` equals the Parquet's
                   raw object xyz, which is only true if the CV->SOMA matrix was
                   NOT applied here. That is the one rule in this repo that
                   breaks silently (see CLAUDE.md), so it is asserted on real
                   staged bytes for every episode, not just unit-tested.
* **human data** - ``poses``/``transl`` equal the Parquet's human columns exactly.
* **assets**     - staged mesh/ground-plane/support bytes hash-match the mirror.

Nothing here interpolates, repairs or tolerates. A mismatch is a failure.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
import pyarrow.parquet as pq

from pipeline.checks import Checker
from pipeline.episodes import (
    EpisodeSpec,
    count_support_collision_shapes,
    sha256,
)
from pipeline.lerobot_to_soma import (
    NUM_SOMA_JOINTS,
    object_poses_to_matrices,
    reshape_human_pose,
)

#: Published timestamps are float32 seconds, so a 30 fps step cannot be exact.
#: 1/30 is 0.0333333..., float32 rounds it, and the error accumulates over ~1000
#: frames - the measured worst case across all 60 published episodes is 4e-6 s.
TIMESTAMP_STEP_TOLERANCE_S = 1e-4


def validate_staged_episode(
    spec: EpisodeSpec, *, emit: Callable[[str], None] = print
) -> Checker:
    """Run every staged-input check for one episode. Returns the checker."""
    checker = Checker(emit=emit)
    checker.emit(f"=== staged inputs: episode {spec.episode_index} {spec.sequence_id} ===")
    checker.info(f"tier       : {spec.tier}")
    checker.info(f"data_folder: {spec.data_folder}")

    present = {
        "soma_params.npz": spec.data_folder / "soma_params.npz",
        "poses.npy": spec.data_folder / "poses.npy",
        "ground_plane.json": spec.staged_ground_plane,
        "reconstructed_mesh/output_aligned.glb": spec.staged_mesh,
        "episode_manifest.json": spec.manifest,
        "reconstructed_stage/<seq>_support.usda": spec.staged_support,
    }
    all_present = True
    for name, path in present.items():
        ok = path.is_file()
        all_present &= ok
        checker.check(ok, f"{name:40s} present")
    if not all_present:
        checker.note("staged inputs incomplete; the remaining checks need them")
        checker.summary()
        return checker

    # ---------------------------------------------------------------- identity
    manifest = json.loads(spec.manifest.read_text())
    checker.check(
        manifest.get("tier") == spec.tier,
        f"manifest tier is {spec.tier} (got {manifest.get('tier')!r})",
    )
    checker.check(
        manifest.get("sequence_id") == spec.sequence_id,
        f"manifest sequence_id is {spec.sequence_id}",
    )
    checker.check(
        manifest.get("episode_index") == spec.episode_index,
        f"manifest episode_index is {spec.episode_index}",
    )
    parquet_hash = sha256(spec.parquet)
    checker.check(
        manifest.get("source_sha256", {}).get("parquet") == parquet_hash,
        "manifest records the source Parquet's current sha256 "
        f"({parquet_hash[:12]}...)",
    )

    # ---------------------------------------------------------------- contents
    table = pq.read_table(spec.parquet)
    column = lambda name: np.stack(table[name].to_numpy(zero_copy_only=False))  # noqa: E731
    params = np.load(spec.data_folder / "soma_params.npz", allow_pickle=True)
    staged_object = np.load(spec.data_folder / "poses.npy")

    rows = table.num_rows
    checker.check(
        rows == spec.frames,
        f"Parquet rows ({rows}) == metadata frames ({spec.frames})",
    )
    checker.check(
        params["poses"].shape == (spec.frames, NUM_SOMA_JOINTS, 3),
        "staged poses shape is ({}, {}, 3) (got {})".format(
            spec.frames, NUM_SOMA_JOINTS, tuple(params["poses"].shape)
        ),
    )
    checker.check(
        params["transl"].shape == (spec.frames, 3),
        f"staged transl shape is ({spec.frames}, 3) (got {tuple(params['transl'].shape)})",
    )
    checker.check(
        staged_object.shape == (spec.frames, 4, 4),
        f"staged poses.npy shape is ({spec.frames}, 4, 4) (got {tuple(staged_object.shape)})",
    )
    checker.metric("frames", int(rows))

    # -------------------------------------------------------------- timestamps
    timestamps = table["timestamp"].to_numpy().astype(np.float64)
    deltas = np.diff(timestamps)
    expected_step = 1.0 / float(spec.fps)
    step_error = float(np.max(np.abs(deltas - expected_step))) if deltas.size else 0.0
    duration = float(timestamps[-1] - timestamps[0])
    checker.info(
        "timestamps: t0={:.6f} tN={:.6f} duration={:.3f} s "
        "(expected {:.3f} s at {} fps)".format(
            timestamps[0], timestamps[-1], duration, (spec.frames - 1) * expected_step,
            spec.fps,
        )
    )
    checker.info(f"max |dt - 1/{spec.fps}| = {step_error:.2e} s")
    checker.metric("duration_s", round(duration, 4))
    checker.metric("timestamp_step_error_s", step_error)
    checker.check(bool(np.all(deltas > 0.0)), "timestamps strictly increase")
    checker.check(
        step_error < TIMESTAMP_STEP_TOLERANCE_S,
        f"timestamp spacing is 1/{spec.fps} s within {TIMESTAMP_STEP_TOLERANCE_S:.0e} s "
        f"(max error {step_error:.2e} s)",
    )
    checker.check(
        manifest.get("timestamps", {}).get("duration_s") is not None
        and abs(float(manifest["timestamps"]["duration_s"]) - duration) < 1e-6,
        "manifest records this episode's measured duration",
    )

    # ------------------------------------------------------------- human data
    source_poses = reshape_human_pose(column("observation.human.pose"))
    checker.check(
        np.array_equal(params["poses"], source_poses),
        "staged human poses are bit-for-bit the Parquet's (no transform applied)",
    )
    checker.check(
        np.array_equal(
            params["transl"], column("observation.human.translation").astype(np.float32)
        ),
        "staged human translation is bit-for-bit the Parquet's",
    )

    # ------------------------------------------------------------ object poses
    visible = table["observation.object.visible"].to_numpy().astype(bool)
    object_pose = column("observation.object.pose").astype(np.float32)
    rebuilt = object_poses_to_matrices(object_pose, visible, on_invalid="passthrough")
    checker.check(
        np.array_equal(staged_object, rebuilt),
        "staged poses.npy is bit-for-bit reproducible from the Parquet",
    )
    invalid = np.flatnonzero(~visible)
    checker.check(
        int(visible.sum()) == manifest.get("visible_object_frames"),
        "manifest's visible-object-frame count matches the Parquet "
        f"({int(visible.sum())}/{rows})",
    )
    checker.check(
        list(invalid) == list(manifest.get("invalid_object_frames", [])),
        f"manifest lists every invalid object frame ({list(invalid)})",
    )
    if invalid.size:
        checker.check(
            bool(np.all(staged_object[invalid] == 0.0)),
            f"the {invalid.size} invalid frame(s) carry NVIDIA's zero-fill through "
            "untouched (no interpolation, no identity substitute)",
        )

    # ----------------------------------------------- THE coordinate rule
    # soma_to_g1.py:308 applies diag(1,-1,-1) to the object poses itself. The
    # matrix is its own inverse, so applying it here too restores the broken
    # frame silently. Asserting the staged translation equals the raw Parquet
    # xyz is exactly the statement "we did not apply it".
    if invalid.size < rows:
        valid = np.flatnonzero(visible)
        checker.check(
            np.array_equal(
                staged_object[valid, :3, 3], object_pose[valid, :3].astype(np.float64)
            ),
            "object translations are the raw OpenCV-frame values: the CV->SOMA "
            "matrix was NOT applied here (soma_to_g1.py:308 applies it)",
        )

    # ---------------------------------------------------------------- assets
    for role, source, staged in (
        ("mesh", spec.mesh, spec.staged_mesh),
        ("ground_plane", spec.ground_plane, spec.staged_ground_plane),
        ("support_surface", spec.support_surface, spec.staged_support),
    ):
        checker.check(
            sha256(source) == sha256(staged),
            f"staged {role} is byte-identical to the pristine mirror",
        )

    # ------------------------------------------------- the support decision
    shapes = count_support_collision_shapes(spec.support_surface)
    checker.check(
        shapes == spec.support_shape_count == manifest.get("support_shape_count"),
        f"support collision shapes agree: {shapes} "
        f"(registry {spec.support_shape_count}, manifest "
        f"{manifest.get('support_shape_count')})",
    )
    checker.check(
        manifest.get("scene_support") == spec.scene_support,
        f"manifest records scene.support={str(spec.scene_support).lower()} "
        f"for {shapes} collision shape(s)",
    )
    checker.metric("support_shape_count", shapes)
    checker.metric("scene_support", spec.scene_support)

    # ------------------------------------------------- the retarget frame plan
    checker.check(
        manifest.get("retarget_frames") == spec.retarget_frames,
        f"manifest's planned reference length is {spec.retarget_frames} frames",
    )
    checker.metric("retarget_frames", spec.retarget_frames)
    if spec.dropped_frames:
        checker.note(
            "frames {} are excluded from the reference: "
            "observation.object.visible == False, so no object pose was measured "
            "there. They are neither interpolated nor emitted.".format(
                list(spec.dropped_frames)
            )
        )

    checker.summary()
    return checker
