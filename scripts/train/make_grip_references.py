"""Scripted grip: close the Dex3 fingers in the G1 reference while the human hand is in contact.

Why. soma_to_g1 leaves G1's 14 Dex3 finger joints at 0 (fully open, on a joint limit) for every
frame, and FlashCHORD's finger PD target is ``reference joint + finger_residual_scale * residual``.
So every policy starts each grasp from an open hand and must discover closing through its residual
alone. The human data cannot supply a grasp either: Tier 1's finger joints move only 5-8 deg
(median range per episode). Ten reward/curriculum variants left the three object-dropping episodes
(vacuum 1, bat 4, bottle 25) at AUC 0.11-0.25.

What. For each hand, a blend weight w(t) in [0, 1] from that hand's ``hand_contact_active``
label -- 1 within ``lead`` frames of contact, ramping to 0 over the next ``ramp`` frames -- and

    q_finger(t) = w(t) * alpha * closing_limit

where closing_limit is the URDF limit away from 0 for index/middle/thumb_1/thumb_2 (thumb_0, the
opposition rotation, stays at 0). alpha < 1 leaves the residual room both ways, and a closed target
inside the object is what makes the PD squeeze. Only the 14 finger columns of
``robot_joint_positions`` change; body, object, contacts and every other column are copied, so
the object reference -- what the packer scores -- is untouched.

The contact labels are the tier's own reference columns (Tier 1: computed from the clean capture,
its own input). Not for Tier 2 as-is: its labels were computed against the noisy object.

Run inside containers/flash_chord.sif (pyarrow)::

    python scripts/train/make_grip_references.py --episodes 1,4,25,28
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[2]
URDF = REPO / "official/video_to_data/robotic_grounding/source/robotic_grounding/robotic_grounding/assets/urdfs/g1/main_with_hand.urdf"
CLOSING_JOINTS = ("index_0", "index_1", "middle_0", "middle_1", "thumb_1", "thumb_2")


def urdf_limits() -> dict[str, tuple[float, float]]:
    text = URDF.read_text()
    limits = {}
    for joint in re.finditer(r'<joint name="([^"]+)" type="revolute">(.*?)</joint>', text, re.S):
        found = re.search(r'<limit[^>]*lower="([-\d.e]+)"[^>]*upper="([-\d.e]+)"', joint.group(2))
        if found:
            limits[joint.group(1)] = (float(found.group(1)), float(found.group(2)))
    return limits


def grip_targets(names: list[str], alpha: float) -> dict[str, float]:
    """Closed-grip value per finger joint: alpha times the limit on the far side of 0."""
    limits = urdf_limits()
    targets = {}
    for name in names:
        match = re.fullmatch(r"(left|right)_hand_(\w+)_joint", name)
        if not match or match.group(2) == "thumb_0":
            continue
        lower, upper = limits[name]
        far = upper if abs(upper) > abs(lower) else lower
        if match.group(2) in CLOSING_JOINTS and far != 0.0:
            targets[name] = alpha * far
    return targets


def blend_weight(contact: np.ndarray, lead: int, ramp: int) -> np.ndarray:
    """1 within `lead` frames of a contact frame, linear down to 0 over the next `ramp` frames."""
    frames = np.arange(len(contact))
    hits = np.flatnonzero(contact)
    if hits.size == 0:
        return np.zeros(len(contact))
    distance = np.min(np.abs(frames[:, None] - hits[None, :]), axis=1)
    return np.clip(1.0 - (distance - lead) / float(ramp), 0.0, 1.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", required=True, help="comma-separated episode indices")
    parser.add_argument("--manifest", default=str(REPO / "training_manifest_verified.json"))
    parser.add_argument("--dst", default="adapter_out_grip", help="new tree under data/")
    parser.add_argument("--out-manifest", default=str(REPO / "training_manifest_t1grip.json"))
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--lead", type=int, default=3)
    parser.add_argument("--ramp", type=int, default=6)
    args = parser.parse_args()

    source = json.loads(Path(args.manifest).read_text())
    src_root = REPO / "data" / source.get("adapter_root", "adapter_out")
    dst_root = REPO / "data" / args.dst
    wanted = {int(e) for e in args.episodes.split(",")}
    episodes = []
    for entry in source["episodes"]:
        if entry["episode_index"] not in wanted:
            continue
        sequence = entry["sequence_id"]
        rel = Path("whole_body/soma") / f"sequence_id={sequence}" / "robot_name=g1/data.parquet"
        table = pq.read_table(src_root / rel)
        row = {name: table.column(name)[0].as_py() for name in ("robot_joint_names", "robot_joint_positions", "hand_contact_active", "hand_sides")}
        names, q = row["robot_joint_names"], np.asarray(row["robot_joint_positions"], dtype=np.float64)
        contact = np.asarray(row["hand_contact_active"], dtype=bool)
        targets = grip_targets(names, args.alpha)
        weights = {side: blend_weight(contact[i], args.lead, args.ramp) for i, side in enumerate(row["hand_sides"])}
        for name, value in targets.items():
            column = names.index(name)
            if np.abs(q[:, column]).max() > 1e-6:
                raise SystemExit(f"{sequence}: {name} is not the constant open pose; refusing to overwrite a driven joint")
            q[:, column] = weights[name.split("_")[0]] * value

        field = table.schema.field("robot_joint_positions")
        new_column = pa.array([q.astype(np.float32 if pa.types.is_float32(field.type.value_type.value_type) else np.float64).tolist()], type=field.type)
        out_table = table.set_column(table.schema.get_field_index("robot_joint_positions"), field, new_column)
        (dst_root / rel).parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(out_table, dst_root / rel)
        stage = Path("whole_body/reconstructed_stage") / f"{sequence}_support.usda"
        (dst_root / stage).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_root / stage, dst_root / stage)

        closed = {side: float((w > 0.99).mean()) for side, w in weights.items()}
        episodes.append({**entry, "task_parquet_container": f"/work/{args.dst}/{rel}",
                         "grip": {"alpha": args.alpha, "lead": args.lead, "ramp": args.ramp, "targets": targets,
                                  "closed_fraction": closed}})
        print(f"ep{entry['episode_index']:02d} {sequence}: closed {closed}  ({len(targets)} finger joints driven)")
    manifest = {**{k: v for k, v in source.items() if k != "episodes"}, "adapter_root": args.dst,
                "note": "Tier 1 references with a scripted Dex3 grip during human hand contact "
                        "(scripts/train/make_grip_references.py); body/object columns unchanged.",
                "episodes": episodes}
    Path(args.out_manifest).write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"{len(episodes)} episode(s) -> {args.out_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
