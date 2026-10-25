#!/usr/bin/env python3
"""Where does each episode's AUC go: position, orientation, or a mid-episode failure?

Reads the exported evaluation trajectories (``eval_<step>/rep*/episode_*.parquet``)
and scores them with the kit's ADD-AUC (1..9 cm, mesh vertices from the staged
``textured_mesh.obj``), then twice more with one error source removed:

* ``AUC|rot=ref``  achieved position, reference rotation -> what perfect orientation buys
* ``AUC|pos=ref``  reference position, achieved rotation -> what perfect position buys

plus median position/rotation error, the first-30-frame error (a setup offset
shows up there), and the first frame whose ADD leaves the 9 cm band. Means over
repeats. Needs pyarrow + scipy, so run it in the flash_chord image:

    apptainer exec --cleanenv --no-mount home,cwd -B "$PWD":/r containers/flash_chord.sif \
        python /r/scripts/evaluate/decompose_auc_loss.py --root /r
    ... --run-fmt t1_ep%02d_keypoints --episodes 0,3,5,17,21,25
"""

from __future__ import annotations

import argparse
import glob
import json

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

ADD_THRESHOLDS_M = np.arange(0.01, 0.10, 0.01)  # verbatim from the kit's AUC.py


def add_auc(add: np.ndarray) -> float:
    curve = np.array([np.mean(add < t) for t in ADD_THRESHOLDS_M])
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    return float(trapezoid(curve, x=np.linspace(0.0, 1.0, ADD_THRESHOLDS_M.size)))


def obj_vertices(path: str, limit: int = 2000) -> np.ndarray:
    vertices = np.asarray([tuple(map(float, line.split()[1:4])) for line in open(path) if line.startswith("v ")])
    return vertices[:: max(1, len(vertices) // limit)][:limit]


def read_rollout(path: str) -> tuple[np.ndarray, ...]:
    table = pq.read_table(path)
    column = {name: table.column(name).to_numpy() for name in table.schema.names if name != "object_body_name"}
    keep = column["environment_id"] == 0
    order = np.argsort(column["step"][keep])

    def stack(names):
        return np.stack([column[name][keep][order] for name in names], axis=1).astype(np.float64)

    return (
        stack([f"achieved_position_{a}" for a in "xyz"]),
        stack([f"achieved_quaternion_{a}" for a in "xyzw"]),
        stack([f"reference_position_{a}" for a in "xyz"]),
        stack([f"reference_quaternion_{a}" for a in "xyzw"]),
    )


def score_rollout(path: str, vertices: np.ndarray) -> list[float]:
    ap, aq, rp, rq = read_rollout(path)
    ra, rr = Rotation.from_quat(aq), Rotation.from_quat(rq)
    ma, mr = ra.as_matrix(), rr.as_matrix()
    reference = rp[:, None] + np.einsum("tij,kj->tki", mr, vertices)
    add = np.linalg.norm(ap[:, None] + np.einsum("tij,kj->tki", ma, vertices) - reference, axis=-1).mean(1)
    add_rot_ref = np.linalg.norm(ap - rp, axis=1)  # every vertex moves by the translation
    add_pos_ref = np.linalg.norm(rp[:, None] + np.einsum("tij,kj->tki", ma, vertices) - reference, axis=-1).mean(1)
    position_cm = np.linalg.norm(ap - rp, axis=1) * 100.0
    rotation_deg = np.degrees((ra.inv() * rr).magnitude())
    failed = np.nonzero(add > ADD_THRESHOLDS_M[-1])[0]
    return [
        add_auc(add), add_auc(add_rot_ref), add_auc(add_pos_ref),
        float(np.median(position_cm)), float(np.median(rotation_deg)),
        float(position_cm[:30].mean()), float(rotation_deg[:30].mean()),
        float(failed[0] if failed.size else len(add)), float(len(add)),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=".", help="repo root (/r inside the container)")
    parser.add_argument("--run-fmt", default="t1_ep%02d_stock", help="run tag; ep 0 stock is ep0_stock_nvidia_recipe")
    parser.add_argument("--step", default="349999104")
    parser.add_argument("--episodes", default="all")
    parser.add_argument("--manifest", default="training_manifest_verified.json")
    args = parser.parse_args()

    manifest = {e["episode_index"]: e for e in json.load(open(f"{args.root}/{args.manifest}"))["episodes"]}
    wanted = sorted(manifest) if args.episodes == "all" else [int(x) for x in args.episodes.split(",")]
    print(f"{'ep':>2} {'sequence':26s} {'reps':>4} {'AUC':>6} {'AUC|rot=ref':>11} {'AUC|pos=ref':>11} "
          f"{'pos_med':>7} {'rot_med':>7} {'pos0-30':>7} {'rot0-30':>7} {'1stFail':>7} {'frames':>6}")
    aucs = []
    for episode in wanted:
        run = "ep0_stock_nvidia_recipe" if (episode == 0 and args.run_fmt == "t1_ep%02d_stock") else args.run_fmt % episode
        reps = sorted(glob.glob(f"{args.root}/outputs/cluster/{run}/eval_{args.step}/rep*/episode_*.parquet"))
        if not reps:
            continue
        sequence = manifest[episode]["sequence_id"]
        vertices = obj_vertices(f"{args.root}/data/adapter_out/whole_body/soma/{sequence}/object/textured_mesh.obj")
        m = np.mean([score_rollout(path, vertices) for path in reps], axis=0)
        aucs.append(m[0])
        print(f"{episode:2d} {sequence[20:46]:26s} {len(reps):4d} {m[0]:6.3f} {m[1]:11.3f} {m[2]:11.3f} "
              f"{m[3]:6.1f}c {m[4]:6.1f}d {m[5]:6.2f}c {m[6]:6.2f}d {int(m[7]):7d} {int(m[8]):6d}")
    if aucs:
        print(f"mean AUC over {len(aucs)} episodes: {np.mean(aucs):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
