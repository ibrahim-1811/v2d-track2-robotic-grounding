"""Dex3 finger targets and joint positions as a fraction of each joint closing range (outputs/diag/jitter/*.npz).

host python3: python3 scripts/diagnose/drop/fingers_ts.py <npz> <first> <last> <stride>
"""
import numpy as np, sys
far = {'left_hand_index_0_joint': -1.5708, 'left_hand_index_1_joint': -1.7453, 'left_hand_middle_0_joint': -1.5708,
 'left_hand_middle_1_joint': -1.7453, 'left_hand_thumb_2_joint': 1.7453,
 'right_hand_index_0_joint': 1.5708, 'right_hand_index_1_joint': 1.7453, 'right_hand_middle_0_joint': 1.5708,
 'right_hand_middle_1_joint': 1.7453, 'right_hand_thumb_2_joint': -1.7453}
path, a, b, st = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
d = np.load(path); names = list(d['finger_names']); fq = d['finger_q']; ft = d['finger_target']
op = d['object_pos']; tp = d['object_target_pos']; err = np.linalg.norm(op - tp, axis=1)
cols = [i for i, n in enumerate(names) if n in far]
# flip statistics over whole rollout: steps where target closure changes by > 0.5 of range
tc = np.stack([ft[:, i] / far[names[i]] for i in cols], 1); qc = np.stack([fq[:, i] / far[names[i]] for i in cols], 1)
flips = (np.abs(np.diff(tc, axis=0)) > 0.5).mean()
print(f"{path.split('/')[-1]}: fraction of (step, finger) with target closure jump > 0.5 range: {flips:.3f}; "
      f"mean |dq| per step {np.degrees(np.abs(np.diff(fq,axis=0))).mean():.2f} deg")
print("cols:", " ".join(names[i].replace('_hand_','').replace('_joint','') for i in cols))
for s in range(a, b, st):
    print(f"f{s:4d} err {err[s]*100:5.1f} z {op[s,2]:.2f} | tgt " + " ".join(f"{v:+.1f}" for v in tc[s]) + " | q " + " ".join(f"{v:+.2f}" for v in qc[s]))
