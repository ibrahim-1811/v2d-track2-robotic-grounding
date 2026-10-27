"""Frame-0 human alignment for Tier 3: how far our aligned human sits from Tier 1's first frame.

``cari4d_to_lerobot.py`` fixes the camera->world transform from the object alone (Tier 1's frame-0
object pose vs CARI4D's). When CARI4D's frame-0 object ORIENTATION is wrong, that one transform
rotates the whole scene: episode 20's human ends up upside down under the floor. The FAQ allows
Tier 3 to use the first frame of the Tier 1 data for alignment, and the human is in that frame too,
so this module compares the two frame-0 bodies -- same SOMA rig, same MHR identity model -- and
returns the rigid correction (Kabsch over corresponding vertices) that moves ours onto Tier 1's.
Only row 0 of the Tier 1 parquet is ever used; nothing after the first frame.

Run inside containers/cari4d.sif (``soma``)::

    python scripts/tier3/frame0_alignment.py --episodes 0,1,5      # report only, writes nothing
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
TIER_1 = REPO / "data/v2d_challenge/track_2/tier_1_multiview_caption"
SOMA_ASSETS = REPO / "data/body_models/soma"
HUMAN_COLUMNS = ("pose", "translation", "identity_coeffs", "scale_params", "bone_length_flexibles")


def first_frame_human(parquet: Path) -> dict:
    """Row 0 of a challenge-format parquet's human columns."""
    import pyarrow.parquet as pq

    table = pq.ParquetFile(parquet).read_row_group(0, columns=[f"observation.human.{c}" for c in HUMAN_COLUMNS])
    return {c: np.asarray(table.column(f"observation.human.{c}")[0].as_py(), dtype=np.float32) for c in HUMAN_COLUMNS}


def tier1_first_frame_human(episode: int) -> dict:
    return first_frame_human(TIER_1 / "data/chunk-000" / f"episode_{episode:06d}.parquet")


class Bodies:
    """SOMA vertices (metres, SOMA frame) from challenge-format human parameters."""

    def __init__(self, device: str):
        from soma import SOMALayer
        from soma.units import Unit

        self.device = device
        self.soma = SOMALayer(data_root=str(SOMA_ASSETS), identity_model_type="mhr", device=device, mode="warp",
                              output_unit=Unit.CENTIMETERS, enable_procedural_transforms=False)

    def vertices(self, human: dict) -> np.ndarray:
        import torch

        t = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float32)[None], device=self.device)  # noqa: E731
        with torch.no_grad():
            out = self.soma.forward(t(human["pose"].reshape(-1, 3)), t(human["identity_coeffs"]), t(human["scale_params"]),
                                    transl=t(human["translation"] * 100.0), apply_correctives=False,
                                    kwargs={"bone_length_flexibles": t(human["bone_length_flexibles"])}).vertices
        return out[0].cpu().numpy().astype(np.float64) / 100.0


def kabsch(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """(4, 4) rigid transform minimising |T @ source - target| over corresponding points (no scale)."""
    mu_s, mu_t = source.mean(0), target.mean(0)
    u, _, vt = np.linalg.svd((source - mu_s).T @ (target - mu_t))
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    out = np.eye(4)
    out[:3, :3], out[:3, 3] = rotation, mu_t - rotation @ mu_s
    return out


def correction(ours: np.ndarray, tier1: np.ndarray) -> tuple[np.ndarray, dict]:
    """SOMA-frame rigid correction moving our frame-0 body onto Tier 1's, plus how big it is."""
    fix = kabsch(ours, tier1)
    angle = float(np.degrees(np.arccos(np.clip((np.trace(fix[:3, :3]) - 1.0) / 2.0, -1.0, 1.0))))
    stats = {
        "rotation_deg": round(angle, 2),
        "translation_m": round(float(np.linalg.norm(fix[:3, 3])), 3),
        "vertex_error_before_m": round(float(np.linalg.norm(ours - tier1, axis=1).mean()), 3),
        "vertex_error_after_m": round(float(np.linalg.norm(ours @ fix[:3, :3].T + fix[:3, 3] - tier1, axis=1).mean()), 3),
    }
    return fix, stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", required=True, help="comma-separated episode indices")
    parser.add_argument("--tier", default="tier_3_cari4d_ours")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--record", action="store_true",
                        help="store the result as alignment_check in the episode's conversion report "
                             "(make_tier3_manifest.py requires one of alignment / alignment_check)")
    args = parser.parse_args()
    bodies = Bodies(args.device)
    tier_root = REPO / "data/reconstructed/track_2" / args.tier
    for episode in (int(e) for e in args.episodes.split(",")):
        ours_path = tier_root / "data/chunk-000" / f"episode_{episode:06d}.parquet"
        if not ours_path.is_file():
            print(f"ep{episode:02d}: not converted")
            continue
        _, stats = correction(bodies.vertices(first_frame_human(ours_path)), bodies.vertices(tier1_first_frame_human(episode)))
        print(f"ep{episode:02d} " + json.dumps(stats), flush=True)
        report = tier_root / "conversion" / f"episode_{episode:06d}.json"
        if args.record and report.is_file():
            payload = json.loads(report.read_text())
            payload["alignment_check"] = {**stats, "method": "object"}
            report.write_text(json.dumps(payload, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
