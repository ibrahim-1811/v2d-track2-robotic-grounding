#!/usr/bin/env python3
"""Derive the thirteen-term curriculum from NVIDIA's twelve-term one.

Generating it (rather than hand-copying) is what makes "the original twelve
weights are unchanged" checkable: this script reads upstream's YAML, asserts the
twelve weights and the nine thresholds are exactly what we expect, and adds one
key, ``object_orientation``, to every stage.

``CurriculumStage.weights_for`` validates the stage's key set against the
objective's ``term_names`` exactly, so a thirteen-term objective *requires* this
file and a twelve-term objective would reject it. That is the intended guard.

Upstream is read only; output goes to ``configs/orientation/curriculum/``.

    .venvs/episode0/bin/python scripts/train/make_orientation_curriculum.py [--weight 2.0]
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
UPSTREAM = (
    REPO
    / "official/video_to_data/robotic_grounding/flash_chord/src/flash_chord"
    / "configs/curriculum/recon_body_flash_sac.yaml"
)
OUTPUT = REPO / "configs/orientation/curriculum/orientation_recon_body_flash_sac.yaml"
UPSTREAM_OBJECTIVE = (
    REPO
    / "official/video_to_data/robotic_grounding/flash_chord/src/flash_chord"
    / "configs/objective/recon_body.yaml"
)
OBJECTIVE_OUTPUT = REPO / "configs/orientation/objective/orientation_recon_body.yaml"
KEYPOINT_OUTPUT = REPO / "configs/keypoints/curriculum/keypoints_recon_body_flash_sac.yaml"
KEYPOINT_OBJECTIVE_OUTPUT = REPO / "configs/keypoints/objective/keypoints_recon_body.yaml"

# The twelve weights every upstream stage shares, in the objective's term order.
EXPECTED_WEIGHTS: tuple[tuple[str, str], ...] = (
    ("termination", "-300.0"),
    ("action_rate_l2", "-0.0001"),
    ("action_l2", "-0.000001"),
    ("soft_joint_limit", "-0.001"),
    ("pelvis_position", "1.0"),
    ("pelvis_orientation", "1.0"),
    ("body_joint_position", "5.0"),
    ("object_position", "1.0"),
    ("progress", "1.0"),
    ("palm_position", "1.0"),
    ("palm_orientation", "1.0"),
    ("force_closure", "5.0"),
)
EXPECTED_THRESHOLDS = (
    35000320,
    70000640,
    104998912,
    139999232,
    174999552,
    209999872,
    288749568,
    306249728,
    349999104,
)
# voc_scale and immediate_first_frame_probability per stage, from the audit.
EXPECTED_STAGES: tuple[tuple[str, str | None], ...] = (
    ("1.0", None),
    ("0.75", None),
    ("0.5", None),
    ("0.25", None),
    ("0.1", None),
    ("0.05", None),
    ("0.0", None),
    ("0.0", "0.25"),
    ("0.0", "0.50"),
)


def parse_upstream(text: str) -> tuple[tuple[int, ...], list[tuple[str, str | None]], dict[str, str]]:
    """Pull thresholds, per-stage settings and the shared weight block out of the YAML."""
    threshold_match = re.search(r"^thresholds:\s*\[([^\]]+)\]", text, re.MULTILINE)
    if threshold_match is None:
        raise SystemExit("upstream curriculum has no 'thresholds:' list")
    thresholds = tuple(int(value.strip()) for value in threshold_match.group(1).split(","))

    weights: dict[str, str] = {}
    in_block = False
    for line in text.splitlines():
        if "objective_weights: &recon_body_weights" in line:
            in_block = True
            continue
        if in_block:
            entry = re.match(r"^\s{6}([a-z0-9_]+):\s*(\S+)\s*$", line)
            if entry is None:
                in_block = False
            else:
                weights[entry.group(1)] = entry.group(2)

    stages: list[tuple[str, str | None]] = []
    for block in text.split("- _target_: flash_chord.configuration.make_curriculum_stage")[1:]:
        voc = re.search(r"voc_scale:\s*(\S+)", block)
        immediate = re.search(r"immediate_first_frame_probability:\s*(\S+)", block)
        if voc is None:
            raise SystemExit("a curriculum stage has no voc_scale")
        stages.append((voc.group(1), immediate.group(1) if immediate else None))
    return thresholds, stages, weights


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weight", type=float, default=None, help="object_orientation weight (arm default)")
    parser.add_argument(
        "--arm",
        choices=("squared", "exponential", "keypoints"),
        default="squared",
        help="squared: arm 1, -(theta/pi)**2, weight 2.0. exponential: arm 2, exp(-theta**2/std**2), "
        "weight 1.0. keypoints: arm 3, ADD-shaped object_keypoints, weight 5.0 (configs/keypoints/)",
    )
    parser.add_argument("--std", type=float, default=0.2, help="arm 2 only: shaping width in radians")
    parser.add_argument("--scale", type=float, default=0.05, help="arm 3 only: Laplacian width in metres")
    parser.add_argument(
        "--radius-fraction", type=float, default=0.5, help="arm 3 only: keypoint radius / bounding radius"
    )
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.weight is None:
        args.weight = {"squared": 2.0, "exponential": 1.0, "keypoints": 5.0}[args.arm]
    term = "object_keypoints" if args.arm == "keypoints" else "object_orientation"
    if args.arm == "keypoints":
        if args.output == OUTPUT:
            args.output = KEYPOINT_OUTPUT
        if not (0.0 < args.scale < 1e3) or not (0.0 < args.radius_fraction < 1e3):
            raise SystemExit("--scale and --radius-fraction must be finite and positive")
    if args.arm == "exponential":
        if args.output == OUTPUT:
            args.output = OUTPUT.with_name("orientation_exp_recon_body_flash_sac.yaml")
        if not (0.0 < args.std < 1e3):
            raise SystemExit(f"--std must be finite and positive, got {args.std}")

    if not (0.0 <= args.weight < 1e6):
        raise SystemExit(f"--weight must be non-negative and finite, got {args.weight}")

    text = UPSTREAM.read_text()
    upstream_sha = hashlib.sha256(text.encode()).hexdigest()
    thresholds, stages, weights = parse_upstream(text)

    # Refuse to generate anything if upstream is not exactly what we audited.
    if thresholds != EXPECTED_THRESHOLDS:
        raise SystemExit(f"upstream thresholds changed: {thresholds}")
    if tuple(weights.items()) != EXPECTED_WEIGHTS:
        raise SystemExit(f"upstream objective weights changed: {tuple(weights.items())}")
    if tuple(stages) != EXPECTED_STAGES:
        raise SystemExit(f"upstream stage settings changed: {tuple(stages)}")

    lines = [
        "# GENERATED by scripts/train/make_orientation_curriculum.py -- do not edit by hand.",
        "#",
        "# NVIDIA's recon_body_flash_sac curriculum with one key added to every stage.",
        "# Thresholds, VOC schedule, reset probabilities and the original twelve",
        f"# weights are copied verbatim; only '{term}' is new.",
        f"# source: configs/curriculum/recon_body_flash_sac.yaml @ sha256:{upstream_sha[:16]}",
        f"# {term} weight: {args.weight}  ({args.arm} shaping)",
        "",
        "_target_: flash_chord.configuration.make_fixed_curriculum",
        "",
        "thresholds: [" + ", ".join(str(value) for value in thresholds) + "]",
        "",
        "stages:",
    ]
    for index, (voc, immediate) in enumerate(stages):
        lines.append("  - _target_: flash_chord.configuration.make_curriculum_stage")
        lines.append(f"    voc_scale: {voc}")
        if index == 0:
            lines.append("    objective_weights: &orientation_recon_body_weights")
            for name, value in EXPECTED_WEIGHTS:
                lines.append(f"      {name}: {value}")
            lines.append(f"      {term}: {args.weight}")
        else:
            lines.append("    objective_weights: *orientation_recon_body_weights")
        if immediate is not None:
            lines.append(f"    immediate_first_frame_probability: {immediate}")

    # The objective group file: upstream's recon_body.yaml with only the
    # _target_ repointed at our wrapper and one weight appended.
    objective_text = UPSTREAM_OBJECTIVE.read_text()
    objective_sha = hashlib.sha256(objective_text.encode()).hexdigest()
    upstream_target = "_target_: flash_chord.objectives.recon_body.ReconBodyObjectiveConfig"
    if upstream_target not in objective_text:
        raise SystemExit("upstream objective config no longer declares the expected _target_")
    local_target = {
        "squared": "pipeline.flash_sac_orientation.OrientationReconBodyObjectiveConfig",
        "exponential": "pipeline.flash_sac_orientation_exp.ExponentialOrientationReconBodyObjectiveConfig",
        "keypoints": "pipeline.flash_sac_keypoints.KeypointReconBodyObjectiveConfig",
    }[args.arm]
    objective_body = objective_text.replace(upstream_target, f"_target_: {local_target}")
    shaping_note = {
        "squared": "-(theta / pi) ** 2, a bounded penalty in [-1, 0]",
        "exponential": f"exp(-theta ** 2 / {args.std} ** 2), upstream's own orientation shaping, in (0, 1]",
        "keypoints": f"mean_i exp(-|k_i - k_i*| / {args.scale}), six axis keypoints at "
        f"{args.radius_fraction} x the bounding radius, upstream's Laplacian, in (0, 1]",
    }[args.arm]
    objective_lines = [
        "# GENERATED by scripts/train/make_orientation_curriculum.py -- do not edit by hand.",
        "#",
        "# NVIDIA's recon_body objective config with the _target_ repointed at our",
        "# thirteen-term wrapper and one weight appended. Every shaping scale,",
        "# wrench setting and original weight is byte-identical to upstream.",
        f"# source: configs/objective/recon_body.yaml @ sha256:{objective_sha[:16]}",
        f"# shaping: {shaping_note}",
        "",
        objective_body.rstrip("\n"),
        f"{term}_weight: {args.weight}",
    ]
    if args.arm == "exponential":
        objective_lines.append(f"object_orientation_std: {args.std}")
    if args.arm == "keypoints":
        objective_lines.append(f"object_keypoints_scale: {args.scale}")
        objective_lines.append(f"object_keypoints_radius_fraction: {args.radius_fraction}")
    objective_output = {
        "squared": OBJECTIVE_OUTPUT,
        "exponential": OBJECTIVE_OUTPUT.with_name("orientation_exp_recon_body.yaml"),
        "keypoints": KEYPOINT_OBJECTIVE_OUTPUT,
    }[args.arm]
    objective_output.parent.mkdir(parents=True, exist_ok=True)
    objective_output.write_text("\n".join(objective_lines) + "\n")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")
    print(f"wrote {args.output.relative_to(REPO)}")
    print(f"  upstream sha256 : {upstream_sha}")
    print(f"  stages          : {len(stages)}  thresholds: {len(thresholds)}")
    print(f"  weights/stage   : {len(EXPECTED_WEIGHTS) + 1} (12 upstream + {term}={args.weight})")
    print(f"wrote {objective_output.relative_to(REPO)}")
    print(f"  upstream sha256 : {objective_sha}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
