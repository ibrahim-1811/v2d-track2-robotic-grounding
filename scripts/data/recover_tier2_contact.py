"""Tier 2 object recovery from Tier 2 data and physics only: an object moves only when touched.

Replaces ``recover_tier2_object.py`` (removed 2026-10-27, in git history) for submissions. That script's static hold and 3 cm/s
threshold were chosen by scoring against the clean Tier 1 caption (its ``--evaluate``), which we
no longer allow for any Tier 2 decision (CLAUDE.md, "Rule discipline for Tiers 2 and 3"). This one
has no Tier 1 code path at all, and its two parameters are fixed from physics before anything is
scored:

* the object can only move or turn while a hand or a foot is near it. "Near" = within the object's
  bounding radius plus ``REACH_M`` = 0.10 m of its centre: Tier 2's object noise is clipped at
  3 cm, plus ~7 cm for the size of a hand or foot. The bounding sphere is used, not the surface,
  because Tier 2's rotation noise (up to 30 deg) would move far surface points by tens of cm;
* contact must be absent for ``RELEASE_FRAMES`` = 5 consecutive frames (1/6 s at 30 fps) before
  the object is held, so a hand brushing in and out does not flicker the decision.

While touched, the measurement is followed; while not, the last touched pose is held. Hands are the
G1 reference's palm poses (``ee_pose_w``); feet come from forward kinematics of
``main_with_hand.urdf`` on the reference's root pose and joint angles. Tier 2's human motion is
nearly clean (~1 cm), and so are both. FK is checked by reproducing the stored palm poses.

Outputs (same layout as the earlier recovery, new names): ``data/adapter_out_tier_2c`` (G1
references), ``outputs/tier2c_frame_references`` (released frame, for the packer) and
``training_manifest_tier2c.json``. Run inside containers/flash_chord.sif::

    python scripts/data/recover_tier2_contact.py --write
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[2]
import sys  # noqa: E402

sys.path.insert(0, str(REPO))
from scripts.tier3.make_tier3_manifest import stationary_auc  # noqa: E402

URDF =REPO / "official/video_to_data/robotic_grounding/source/robotic_grounding/robotic_grounding/assets/urdfs/g1/main_with_hand.urdf"
REACH_M = 0.10
RELEASE_FRAMES = 5
FEET = ("left_ankle_roll_link", "right_ankle_roll_link")
PALMS = ("left_hand_palm_link", "right_hand_palm_link")


class Kinematics:
    """Minimal URDF forward kinematics (revolute + fixed joints) from the floating base."""

    def __init__(self, urdf: Path):
        root = ET.parse(urdf).getroot()
        self.by_child = {}
        for joint in root.findall("joint"):
            origin = joint.find("origin")
            xyz = np.array([float(v) for v in (origin.get("xyz", "0 0 0") if origin is not None else "0 0 0").split()])
            rpy = np.array([float(v) for v in (origin.get("rpy", "0 0 0") if origin is not None else "0 0 0").split()])
            axis = joint.find("axis")
            self.by_child[joint.find("child").get("link")] = {
                "name": joint.get("name"), "type": joint.get("type"), "parent": joint.find("parent").get("link"),
                "T": self._transform(Rotation.from_euler("xyz", rpy).as_matrix(), xyz),
                "axis": np.array([float(v) for v in axis.get("xyz").split()]) if axis is not None else None,
            }
        children = set(self.by_child)
        parents = {j["parent"] for j in self.by_child.values()}
        self.base = (parents - children).pop()

    @staticmethod
    def _transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
        out = np.eye(4)
        out[:3, :3], out[:3, 3] = rotation, translation
        return out

    def chain(self, link: str) -> list[dict]:
        joints = []
        while link != self.base:
            joint = self.by_child[link]
            joints.append(joint)
            link = joint["parent"]
        return joints[::-1]

    def positions(self, link: str, root_pos: np.ndarray, root_wxyz: np.ndarray, joint_names: list[str], q: np.ndarray) -> np.ndarray:
        """World positions (T, 3) of ``link`` given the base pose and joint angles per frame."""
        frames = len(root_pos)
        world = np.tile(np.eye(4), (frames, 1, 1))
        world[:, :3, :3] = Rotation.from_quat(root_wxyz[:, [1, 2, 3, 0]]).as_matrix()
        world[:, :3, 3] = root_pos
        index = {name: i for i, name in enumerate(joint_names)}
        for joint in self.chain(link):
            world = world @ joint["T"]
            if joint["type"] in ("revolute", "continuous"):
                rotations = Rotation.from_rotvec(q[:, index[joint["name"]], None] * joint["axis"][None]).as_matrix()
                local = np.tile(np.eye(4), (frames, 1, 1))
                local[:, :3, :3] = rotations
                world = world @ local
        return world[:, :3, 3]


def contact_hold_index(touched: np.ndarray) -> np.ndarray:
    """Frame whose pose each frame uses: itself while touched (with release hysteresis), else the last touched."""
    frames = len(touched)
    engaged = np.zeros(frames, dtype=bool)
    since = RELEASE_FRAMES
    for t in range(frames):
        since = 0 if touched[t] else since + 1
        engaged[t] = since < RELEASE_FRAMES
    source = np.zeros(frames, dtype=np.int64)
    for t in range(1, frames):
        source[t] = t if engaged[t] else source[t - 1]
    return source


def touched_mask(table: pa.Table, kin: Kinematics) -> tuple[np.ndarray, float]:
    row = {name: table.column(name)[0].as_py() for name in
           ("robot_root_position", "robot_root_wxyz", "robot_joint_names", "robot_joint_positions",
            "ee_link_names", "ee_pose_w", "object_body_position", "object_mesh_radius")}
    root_pos, root_wxyz = np.asarray(row["robot_root_position"], float), np.asarray(row["robot_root_wxyz"], float)
    q = np.asarray(row["robot_joint_positions"], float)
    ee = np.asarray(row["ee_pose_w"], float)
    palms = np.stack([ee[:, row["ee_link_names"].index(name), :3] for name in PALMS], axis=1)
    fk_palms = np.stack([kin.positions(name, root_pos, root_wxyz, row["robot_joint_names"], q) for name in PALMS], axis=1)
    fk_error = float(np.linalg.norm(fk_palms - palms, axis=-1).max())
    feet = np.stack([kin.positions(name, root_pos, root_wxyz, row["robot_joint_names"], q) for name in FEET], axis=1)
    centre = np.asarray(row["object_body_position"], float)[:, 0]
    radius = float(np.asarray(row["object_mesh_radius"]).reshape(-1)[0])
    parts = np.concatenate([palms, feet], axis=1)
    distance = np.linalg.norm(parts - centre[:, None, :], axis=-1)
    return (distance < radius + REACH_M).any(axis=1), fk_error


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write", action="store_true", help="write references, frame references and the manifest")
    parser.add_argument("--src", default="adapter_out_tier_2")
    parser.add_argument("--dst", default="adapter_out_tier_2c")
    parser.add_argument("--frame-refs", default="outputs/tier2c_frame_references")
    parser.add_argument("--manifest-in", default="training_manifest_tier2.json")
    parser.add_argument("--manifest-out", default="training_manifest_tier2c.json")
    args = parser.parse_args()

    kin = Kinematics(URDF)
    manifest = json.loads((REPO / args.manifest_in).read_text())
    tree = REPO / "data" / args.dst / "whole_body"
    refs = REPO / args.frame_refs
    if args.write:
        shutil.copytree(REPO / "data" / args.src / "whole_body/reconstructed_stage", tree / "reconstructed_stage", dirs_exist_ok=True)
        refs.mkdir(parents=True, exist_ok=True)
    worst_fk = 0.0
    for episode in manifest["episodes"]:
        ep, seq = episode["episode_index"], episode["sequence_id"]
        table = pq.read_table(REPO / f"data/{args.src}/whole_body/soma/sequence_id={seq}/robot_name=g1/data.parquet")
        touched, fk_error = touched_mask(table, kin)
        worst_fk = max(worst_fk, fk_error)
        frames = len(touched)
        source = contact_hold_index(touched)
        held = int((source != np.arange(frames)).sum())
        print(f"ep{ep:02d} {seq[20:60]:40s} touched {touched.mean() * 100:5.1f}%  held {held:4d}/{frames}  FK palm error {fk_error * 100:.2f} cm")
        episode["recovery"] = {"method": "contact_hold", "reach_m": REACH_M, "release_frames": RELEASE_FRAMES,
                               "touched_fraction": round(float(touched.mean()), 4), "held_frames": held, "frames": frames}
        if not args.write:
            continue
        for name in ("object_body_position", "object_body_wxyz", "object_root_position", "object_root_axis_angle"):
            values = table.column(name)[0].as_py()
            table = table.set_column(table.schema.get_field_index(name), name,
                                     pa.array([[values[i] for i in source]], type=table.schema.field(name).type))
        out = tree / "soma" / f"sequence_id={seq}" / "robot_name=g1" / "data.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, out)
        # The evaluation prints results next to a do-nothing floor; compute it against THIS input
        # (frame-0 pose frozen vs the recovered trajectory), not against the Tier 1 caption.
        mesh = REPO / "data" / table.column("object_mesh_paths")[0].as_py()[0].split("/work/", 1)[1]
        episode["stationary_auc"] = stationary_auc(out, mesh)
        episode["stationary_auc_source"] = f"{args.dst} input, frame-0 pose frozen (no Tier 1 data)"
        released = pq.read_table(REPO / f"data/v2d_challenge/track_2/tier_2_synthetic_noise/data/chunk-000/episode_{ep:06d}.parquet",
                                 columns=["observation.object.pose", "frame_index"])
        pose = np.stack(released.column("observation.object.pose").to_numpy(zero_copy_only=False)).astype(float)[:frames]
        frame_index = np.asarray(released.column("frame_index").to_pylist())[:frames]
        recovered = pose[source]  # released world frame, [x y z qw qx qy qz]
        np.savez(refs / f"episode_{ep:06d}.npz", episode_index=np.int64(ep), coordinate_frame="released_track2_world",
                 frame_index=frame_index.astype(np.int64), pose_xyzw=recovered[:, [0, 1, 2, 4, 5, 6, 3]][:, None, :])
        episode["task_parquet_container"] = f"/work/{args.dst}/whole_body/soma/sequence_id={seq}/robot_name=g1/data.parquet"
    if worst_fk > 0.01:
        raise SystemExit(f"FK does not reproduce the stored palm poses (worst {worst_fk * 100:.2f} cm); foot positions untrusted")
    if args.write:
        manifest["adapter_root"] = args.dst
        manifest["note"] = ("Tier 2 object recovered by scripts/data/recover_tier2_contact.py: hold the pose unless a hand "
                            "or foot is within the bounding radius + 0.10 m (Tier 2 data and physics only, no Tier 1)")
        (REPO / args.manifest_out).write_text(json.dumps(manifest, indent=1) + "\n")
        print(f"wrote data/{args.dst}, {args.frame_refs}, {args.manifest_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
