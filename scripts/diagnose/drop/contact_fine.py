"""Timeline of an instrumented rollout (outputs/diag/t1_epNN_keypoints.json, diagnose_contact_rollout.py).

host python3: python3 scripts/diagnose/drop/contact_fine.py . <ep> <first> <last> <stride>
"""
import json, sys, numpy as np
ep, a, b, st = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
d = json.load(open(f"{sys.argv[1]}/outputs/diag/t1_ep{ep:02d}_keypoints.json"))["records"]
op = np.array([r["object_pos"] for r in d]); tp = np.array([r["object_target_pos"] for r in d])
err = np.linalg.norm(op - tp, axis=1)
v = np.r_[0, np.linalg.norm(np.diff(op, axis=0), axis=1) * 50]
vz = np.r_[0, np.diff(op[:, 2]) * 50]
for s in range(a, b, st):
    r = d[s]
    p0 = np.array(r["palm0_pos"]); p1 = np.array(r["palm1_pos"]); o = op[s]
    print(f"f{s:4d} err {err[s]*100:6.1f} objz {o[2]:.2f} tz {tp[s,2]:.2f} v {v[s]:4.2f} vz {vz[s]:+5.2f} | L c{r['hand0_raw_contacts']:3d} F{r['hand0_force_sum_abs_n']:6.1f} d{np.linalg.norm(p0-o):.2f} | "
          f"R c{r['hand1_raw_contacts']:3d} F{r['hand1_force_sum_abs_n']:6.1f} d{np.linalg.norm(p1-o):.2f} | palm {r['palm_position_m']*100:5.1f}cm pelvis {r['pelvis_position_m']*100:5.1f}cm {np.degrees(r['pelvis_orientation_rad']):5.1f}d "
          f"objrot {np.degrees(r['object_orientation_rad']):5.1f} act {r['action_absmax']:.1f}")
