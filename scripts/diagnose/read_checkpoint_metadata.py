#!/usr/bin/env python3
"""Print a FlashCHORD safetensors checkpoint's metadata without loading tensors.

Step D needs to know exactly what a checkpoint carries before we try to reuse
one episode's policy on another. ``evaluate_policy.py`` rebuilds a whole
experiment from the config embedded in the file's header, so that header is the
contract - and it is worth reading directly rather than inferring.

safetensors' container format is simple enough to parse with the standard
library: 8 bytes of little-endian header length, then a JSON header whose
``__metadata__`` object holds the string key/values ``save_checkpoint`` wrote.
That means this runs on the HOST (python3.12, no torch, no safetensors, no
container) - see "Which Python" in CLAUDE.md.

    python3 scripts/diagnose/read_checkpoint_metadata.py outputs/step_D/smoke_ep0/policy_65536.safetensors
    python3 scripts/diagnose/read_checkpoint_metadata.py --tensors .../state_65536.safetensors
    python3 scripts/diagnose/read_checkpoint_metadata.py --key user.resolved_config_json .../policy_65536.safetensors
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

# Keys worth surfacing first: they decide whether a checkpoint can be resumed,
# evaluated, or transferred to another episode.
HIGHLIGHT = (
    "user.algorithm",
    "user.checkpoint_kind",
    "user.checkpoint_set_id",
    "user.environment_steps",
    "user.global_update_step",
    "user.curriculum_stage",
    "user.voc_scale",
    "user.replay_included",
)


def read_header(path: Path) -> dict:
    """Return the safetensors JSON header (tensor index + __metadata__)."""
    with path.open("rb") as handle:
        raw_length = handle.read(8)
        if len(raw_length) != 8:
            raise ValueError(f"{path} is too short to be a safetensors file")
        (length,) = struct.unpack("<Q", raw_length)
        header = handle.read(length)
    if len(header) != length:
        raise ValueError(f"{path} header is truncated")
    return json.loads(header)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--key", default=None, help="print just this metadata key, pretty-printed if JSON")
    parser.add_argument("--tensors", action="store_true", help="also list tensor keys, shapes and dtypes")
    parser.add_argument("--json", action="store_true", help="dump the whole metadata object as JSON")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    header = read_header(args.checkpoint)
    metadata = header.get("__metadata__", {})

    if args.key is not None:
        value = metadata.get(args.key)
        if value is None:
            print(f"no such metadata key: {args.key}", file=sys.stderr)
            print(f"available: {sorted(metadata)}", file=sys.stderr)
            return 1
        try:
            print(json.dumps(json.loads(value), indent=2))
        except (json.JSONDecodeError, TypeError):
            print(value)
        return 0

    if args.json:
        print(json.dumps(metadata, indent=2))
        return 0

    tensors = {k: v for k, v in header.items() if k != "__metadata__"}
    total_bytes = sum(
        entry["data_offsets"][1] - entry["data_offsets"][0]
        for entry in tensors.values()
        if isinstance(entry, dict) and "data_offsets" in entry
    )

    print(f"=== {args.checkpoint} ===")
    print(f"  file size      : {args.checkpoint.stat().st_size / (1024 * 1024):.2f} MiB")
    print(f"  tensor count   : {len(tensors)}")
    print(f"  tensor bytes   : {total_bytes / (1024 * 1024):.2f} MiB")
    print()
    print("  -- identity --")
    for key in HIGHLIGHT:
        if key in metadata:
            print(f"    {key:28s} {metadata[key]}")

    # The reference block is the part that says WHICH EPISODE this came from,
    # and its object_articulations entry is the only thing a resume validates.
    config_raw = metadata.get("user.resolved_config_json")
    if config_raw:
        try:
            config = json.loads(config_raw)
        except json.JSONDecodeError:
            config = None
        if isinstance(config, dict):
            reference = config.get("reference") or {}
            task = config.get("task") or {}
            scene = config.get("scene") or {}
            print()
            print("  -- episode this checkpoint was trained on --")
            print(f"    task.parquet               {task.get('parquet')}")
            print(f"    reference.num_frames       {reference.get('num_frames')}")
            print(f"    reference.playback_fps     {reference.get('playback_fps')}")
            print(f"    scene.support              {scene.get('support')}")
            print(f"    scene.world_count          {scene.get('world_count')}")
            articulations = (reference.get("object_articulations") or {}).get("entries")
            print(f"    object_articulations       {articulations}")
            if articulations == []:
                print("      ^ EMPTY. validate_resume_reference_config() compares ONLY this,")
                print("        so it cannot tell two rigid-object episodes apart.")

    other = sorted(k for k in metadata if k not in HIGHLIGHT and k != "user.resolved_config_json")
    if other:
        print()
        print("  -- other metadata keys --")
        for key in other:
            value = metadata[key]
            shown = value if len(value) <= 70 else value[:67] + "..."
            print(f"    {key:28s} {shown}")

    if args.tensors:
        print()
        print("  -- tensors --")
        for name, entry in sorted(tensors.items()):
            if isinstance(entry, dict):
                start, end = entry.get("data_offsets", (0, 0))
                print("    {:14s} {:10s} {:24s} {:>12,} B".format(
                    name, entry.get("dtype", "?"), str(entry.get("shape")), end - start
                ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
