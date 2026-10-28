"""Frames the object spends between 10 cm and 25 cm error before the drop, term25 vs 10-cm arms; end state."""
import glob, os, sys, numpy as np, pyarrow.parquet as pq
root = sys.argv[1]
def traj(path):
    t = pq.read_table(path); c = {n: t.column(n).to_numpy() for n in t.schema.names if n != "object_body_name"}
    k = c["environment_id"] == 0; o = np.argsort(c["step"][k])
    s = lambda ns: np.stack([c[n][k][o] for n in ns], 1).astype(float)
    return s([f"achieved_position_{a}" for a in "xyz"]), s([f"reference_position_{a}" for a in "xyz"])
def first(m): i = np.flatnonzero(m); return int(i[0]) if i.size else len(m)
for ep in (1, 4, 25):
    lin = {"term25": [], "other": []}; zend = []; n = 0
    for run in sorted(glob.glob(f"{root}/outputs/cluster/t1_ep{ep:02d}_*")):
        for rp in sorted(glob.glob(f"{run}/eval_*/rep*/episode_*.parquet")):
            ap, rpp = traj(rp); e = np.linalg.norm(ap - rpp, axis=1)
            key = "term25" if run.endswith("term25") else "other"
            lin[key].append(first(e > .25) - first(e > .10)); zend.append(ap[-1, 2]); n += 1
    zend = np.array(zend)
    print(f"ep{ep:02d}: rollouts {n}; frames from 10 cm to 25 cm error: 10-cm arms median {np.median(lin['other']):.0f} "
          f"(p90 {np.percentile(lin['other'],90):.0f}, n={len(lin['other'])}), term25 {sorted(lin['term25'])}; "
          f"final object z: <=0.10 m in {(zend<=0.10).sum()}/{n}, min {zend.min():.2f} max {zend.max():.2f}")
