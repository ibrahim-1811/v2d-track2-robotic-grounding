"""Frames where the object error first exceeds 10/25/50 cm and where it reaches the floor, per arm and rep.

flash_chord.sif: python scripts/diagnose/drop/thresholds.py . 1,4,25
"""
import glob, os, sys, json, numpy as np, pyarrow.parquet as pq
root = sys.argv[1]
def traj(path):
    t = pq.read_table(path); c = {n: t.column(n).to_numpy() for n in t.schema.names if n != "object_body_name"}
    k = c["environment_id"] == 0; o = np.argsort(c["step"][k])
    s = lambda ns: np.stack([c[n][k][o] for n in ns], 1).astype(float)
    return s([f"achieved_position_{a}" for a in "xyz"]), s([f"reference_position_{a}" for a in "xyz"])
def first(m): i = np.flatnonzero(m); return int(i[0]) if i.size else -1
for ep in map(int, sys.argv[2].split(",")):
    print(f"== ep{ep:02d}")
    for run in sorted(glob.glob(f"{root}/outputs/cluster/t1_ep{ep:02d}_*")):
        reps = sorted(glob.glob(f"{run}/eval_*/rep*/episode_*.parquet"))
        if not reps: continue
        out = []
        for rp in reps:
            ap, rpp = traj(rp); e = np.linalg.norm(ap - rpp, axis=1); T = len(e)
            floor_z = 0.12 if ep != 1 else 0.15
            out.append((first(e > .10), first(e > .25), first(e > .5), first(ap[:, 2] < floor_z), (e < .25).mean()))
        a = np.array(out)
        print(f"  {os.path.basename(run)[8:]:22s} >10cm {a[:,0].astype(int).tolist()}  >25cm {a[:,1].astype(int).tolist()}  >50cm {a[:,2].astype(int).tolist()}  on-floor {a[:,3].astype(int).tolist()}  frac<25cm {np.round(a[:,4],2).tolist()}")
