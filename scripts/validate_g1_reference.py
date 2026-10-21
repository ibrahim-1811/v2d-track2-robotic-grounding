"""Validate a retargeted G1 motion_v1 parquet against NVIDIA's own demo reference.

Checks the things that would silently break FlashCHORD later: frame counts that
disagree across columns, non-finite poses, joint-name/dimension mismatches,
unresolvable mesh paths, a root that stops tracking the SOMA hips it was
retargeted from, an object that drifts away from the hands, and IK residuals far
outside what NVIDIA's own shipped reference exhibits.

The checks themselves live in ``pipeline/validate_reference.py`` so the
30-episode orchestrator can run them in-process and collect structured results.
This file stays the human/shell entry point - same arguments, same output.

Run with the project venv (needs pyarrow):

    .venvs/episode0/bin/python scripts/validate_g1_reference.py \
        --parquet <ours.parquet> --reference <nvidia_demo.parquet> --frames 778
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.validate_reference import run_checks  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", required=True, type=Path)
    parser.add_argument("--reference", type=Path, default=None)
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument(
        "--mesh-root",
        type=Path,
        default=None,
        help="host directory the parquet's container-side mesh paths map to",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run_checks(
        args.parquet,
        frames=args.frames,
        reference=args.reference,
        mesh_root=args.mesh_root,
    )
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
