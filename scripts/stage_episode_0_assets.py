#!/usr/bin/env python3
"""Step 8.4 - stage the three official Episode 0 assets where the retargeter looks.

Two-layer design, on purpose:

  1. PRISTINE  data/v2d_challenge/track_2/tier_1_multiview_caption/...
     An exact mirror of the Hugging Face dataset layout. Downloaded once,
     hash-verified, then never touched again.

  2. STAGED    data/adapter_out/...
     Copies placed at the paths NVIDIA's scripts actually resolve. Derived and
     disposable - delete and re-run to rebuild.

Keeping them separate means the staged tree can be rebuilt at any time without
re-downloading, and a staging mistake can never corrupt the source dataset.

NO coordinate conversion happens here. Every byte is copied verbatim:
  * the GLB is the scanned mesh in its own local frame
  * ground_plane.json is ax+by+cz+d=0 in the source OpenCV world frame
  * the support USDA is metres / Z-up in the g1_retarget_world - a DIFFERENT
    frame, consumed downstream by FlashCHORD, not by soma_to_g1.py
"""
from __future__ import annotations

import hashlib
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TIER = REPO / "data/v2d_challenge/track_2/tier_1_multiview_caption"
SEQ = "2026-06-04_14-16-23_tall_bar_stool_tip_up_04"
OBJECT = "tall_bar_stool"

# Where soma_to_g1.py reads its inputs (the positional ``data_folder``)
DATA_FOLDER = REPO / "data/adapter_out/whole_body/soma" / SEQ
# Root we will pass as --motion-root. support_usda_for_parquet resolves the
# stage dir as ``sequence_dir.parent.parent / "reconstructed_stage"``, i.e.
# <motion-root>/whole_body/reconstructed_stage/   (registry.py:127)
MOTION_ROOT = REPO / "data/adapter_out"

# Published sha256 from the HF LFS oid. The JSON and USDA are stored as plain
# files, so HF publishes no hash for them - recorded here from first download
# so later runs still detect corruption or a silent upstream change.
EXPECTED_SHA256 = {
    "mesh": "cc214961423d674d55239077043e670c0696bf9113e225e11c1c2623c7ed26ad",
    "ground_plane": "b0d527d5e2a57a98f845f20ac639b0bb918f448edf3186d89d147b6a38655dd4",
    "support_surface": "4d1d1a74fcede154531e8eda6212c213c4dba83c6e3270deffd2fcf60418768b",
}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_verbatim(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    if sha256(src) != sha256(dst):
        raise RuntimeError(f"copy changed bytes: {src} -> {dst}")


def main() -> int:
    sources = {
        # role          -> (pristine source path, role in the pipeline)
        "mesh": (
            TIER / "mesh" / OBJECT / f"{OBJECT}.glb",
            "the scanned stool geometry; soma_to_g1 converts it to OBJ for contact/collision",
        ),
        "ground_plane": (
            TIER / "ground_plane" / f"{SEQ}.json",
            "where the floor is, as ax+by+cz+d=0 in the OpenCV world frame",
        ),
        "support_surface": (
            TIER / "support_surface" / f"{SEQ}_support.usda",
            "static collision stage in the G1 retarget world (Z-up, metres)",
        ),
    }

    missing = [str(p) for p, _ in sources.values() if not p.is_file()]
    if missing:
        print("FAIL - pristine sources missing (see docs/CHALLENGE_ADAPTER.md for the "
              "download commands):", file=sys.stderr)
        for m in missing:
            print(f"  {m}", file=sys.stderr)
        return 1

    print("=== 1. pristine sources (exact HF dataset mirror) ===")
    ok = True
    for role, (src, purpose) in sources.items():
        got = sha256(src)
        match = got == EXPECTED_SHA256[role]
        ok &= match
        print(f"  [{'OK ' if match else 'BAD'}] {role:16s} {src.stat().st_size:>12,} B")
        print(f"        sha256 {got}")
        print(f"        role:  {purpose}")
    if not ok:
        print("FAIL - a source file does not match its expected sha256", file=sys.stderr)
        return 1

    # Destinations, exactly as upstream resolves them.
    targets = {
        # soma_to_g1.load_data(): folder / "reconstructed_mesh" / "output_aligned.glb"
        "mesh": DATA_FOLDER / "reconstructed_mesh" / "output_aligned.glb",
        # soma_to_g1.py:371      : Path(args.data_folder) / "ground_plane.json"
        "ground_plane": DATA_FOLDER / "ground_plane.json",
        # registry.py:127        : <motion-root>/whole_body/reconstructed_stage/<seq>_support.usda
        "support_surface": MOTION_ROOT / "whole_body/reconstructed_stage" / f"{SEQ}_support.usda",
    }

    print("\n=== 2. staged copies (byte-identical, no conversion) ===")
    for role, dst in targets.items():
        copy_verbatim(sources[role][0], dst)
        print(f"  {role:16s} -> {dst.relative_to(REPO)}")

    print("\n=== 3. what soma_to_g1.py will find in data_folder ===")
    required = {
        "soma_params.npz": DATA_FOLDER / "soma_params.npz",
        "poses.npy": DATA_FOLDER / "poses.npy",
        "reconstructed_mesh/output_aligned.glb": targets["mesh"],
        "ground_plane.json": targets["ground_plane"],
    }
    all_ok = True
    for name, p in required.items():
        present = p.is_file()
        all_ok &= present
        print(f"  [{'OK ' if present else 'MISSING'}] {name:40s} "
              f"{p.stat().st_size if present else 0:>12,} B")
    print(f"\n  data_folder complete: {all_ok}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
