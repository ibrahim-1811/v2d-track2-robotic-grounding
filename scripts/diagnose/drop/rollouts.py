"""Drop episodes 1/4/25: per-arm, per-rep failure frame, lift height, peak speed after failure, end state, MPPE.

flash_chord.sif (pyarrow, scipy): python scripts/diagnose/drop/rollouts.py . 1,4,25
"""
"""Failure mechanics of the drop episodes from the exported evaluation trajectories."""
import glob, json, sys, os
import numpy as np, pyarrow.parquet as pq
from scipy.spatial.transform import Rotation as R

root = sys.argv[1]
eps = [int(e) for e in sys.argv[2].split(",")]
man = {e["episode_index"]: e for e in json.load(open(f"{root}/training_manifest_verified.json"))["episodes"]}

def ref(ep):
    p = man[ep]["task_parquet_container"].replace("/work/adapter_out", f"{root}/data/adapter_out")
    t = pq.read_table(p).to_pylist()[0]
    pos = np.array(t["object_root_position"], float)
    hca = np.array(t["hand_contact_active"], float)  # (2, T)
    ee = np.array(t["ee_pose_w"], float)  # (T, 2, 7)
    return pos, hca, ee, t["ee_link_names"], t["hand_sides"], t["object_mesh_radius"][0]

def traj(path):
    t = pq.read_table(path)
    c = {n: t.column(n).to_numpy() for n in t.schema.names if n != "object_body_name"}
    k = c["environment_id"] == 0; o = np.argsort(c["step"][k])
    s = lambda ns: np.stack([c[n][k][o] for n in ns], 1).astype(float)
    return s([f"achieved_position_{a}" for a in "xyz"]), s([f"achieved_quaternion_{a}" for a in "xyzw"]), s([f"reference_position_{a}" for a in "xyz"]), s([f"reference_quaternion_{a}" for a in "xyzw"])

for ep in eps:
    pos, hca, ee, eenames, sides, rad = ref(ep)
    T = len(pos)
    z0 = pos[0, 2]
    sp = np.linalg.norm(np.diff(pos, axis=0), axis=1) * 30
    lift = np.flatnonzero(pos[:, 2] > z0 + 0.05)
    contact_any = (hca > 0.5).any(0)
    print(f"\n=== ep{ep:02d} frames {T} radius {rad:.3f} ref z0 {z0:.3f} zmax {pos[:,2].max():.3f} at f{pos[:,2].argmax()} "
          f"first lift(+5cm) f{lift[0] if lift.size else -1} travel {np.linalg.norm(pos[-1]-pos[0]):.2f} m path {sp.sum()/30:.2f} m "
          f"speed med {np.median(sp):.2f} p95 {np.percentile(sp,95):.2f} max {sp.max():.2f} m/s")
    for h in range(2):
        on = np.flatnonzero(hca[h] > 0.5)
        print(f"  ref contact {sides[h]} ({eenames[h]}): {on.size} frames, first f{on[0] if on.size else -1} last f{on[-1] if on.size else -1}")
    first_c = np.flatnonzero(contact_any)
    print(f"  ref first any contact f{first_c[0] if first_c.size else -1}")
    for run in sorted(glob.glob(f"{root}/outputs/cluster/t1_ep{ep:02d}_*")):
        tag = os.path.basename(run)[len(f"t1_ep{ep:02d}_"):]
        reps = sorted(glob.glob(f"{run}/eval_*/rep*/episode_*.parquet"))
        if not reps: continue
        rows = []
        for rp in reps:
            ap, aq, rpp, rq = traj(rp)
            err = np.linalg.norm(ap - rpp, axis=1)
            rot = np.degrees((R.from_quat(aq) * R.from_quat(rq).inv()).magnitude())
            f10 = np.flatnonzero(err > 0.10); f10 = f10[0] if f10.size else -1
            aspd = np.linalg.norm(np.diff(ap, axis=0), axis=1) * 30
            amax_z = ap[:, 2].max()
            # speed of achieved object in 10 frames after failure, peak
            pk = aspd[f10:f10 + 30].max() if f10 >= 0 else 0
            jpath = os.path.dirname(rp) + ".json"
            mppe = json.load(open(jpath))["metrics"]["mppe_cm"] if os.path.exists(jpath) else float("nan")
            rows.append((f10, ap[:, 2].max() - ap[0, 2], rpp[f10, 2] - ap[f10, 2] if f10 >= 0 else 0, pk,
                         np.linalg.norm(ap[-1] - ap[0]), ap[-1, 2], err[-1], mppe, err.mean() * 100))
        a = np.array(rows)
        print(f"  {tag:22s} n={len(a)} fail10cm f{a[:,0].astype(int).tolist()} dz_max {np.round(a[:,1],2).tolist()} "
              f"ref-ach z@fail {np.round(a[:,2],2).tolist()} peakspd30 {np.round(a[:,3],1).tolist()} "
              f"travel {np.round(a[:,4],2).tolist()} zend {np.round(a[:,5],2).tolist()} MPPE {np.round(a[:,7],1).tolist()} mean {a[:,7].mean():.1f}")
