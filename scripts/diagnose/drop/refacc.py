"""Reference object kinematics in simulation time (one dataset frame per 50 Hz step).

flash_chord.sif: python scripts/diagnose/drop/refacc.py .
"""
import json, sys, numpy as np, pyarrow.parquet as pq
from scipy.spatial.transform import Rotation as R
root = sys.argv[1]
man = {e["episode_index"]: e for e in json.load(open(f"{root}/training_manifest_verified.json"))["episodes"]}
for ep, a, b in ((4, 120, 260), (25, 100, 200), (1, 200, 330)):
    t = pq.read_table(man[ep]["task_parquet_container"].replace("/work/adapter_out", f"{root}/data/adapter_out")).to_pylist()[0]
    p = np.array(t["object_root_position"], float)
    q = np.array(t["object_body_wxyz"], float)[:, 0]  # (T,4) wxyz
    rot = R.from_quat(q[:, [1, 2, 3, 0]])
    dt = 1 / 50.0  # sim time per frame under source_frame_playback
    v = np.gradient(p, dt, axis=0); acc = np.gradient(v, dt, axis=0)
    w = np.linalg.norm((rot[1:] * rot[:-1].inv()).as_rotvec(), axis=1) / dt
    # smooth acc over 5 frames to suppress label noise
    k = np.ones(5) / 5; az = np.convolve(acc[:, 2], k, mode="same"); an = np.convolve(np.linalg.norm(acc, axis=1), k, mode="same")
    print(f"ep{ep:02d} sim-time (50 Hz/frame): max |v| {np.linalg.norm(v,axis=1).max():.2f} m/s, max smoothed |a| {an.max():.1f} m/s^2, frames with a_z < -g: {np.flatnonzero(az < -9.81).tolist()[:30]}")
    for f in range(a, b, 6):
        print(f"   f{f} z {p[f,2]:.2f} vz {v[f,2]:+.2f} az(5f) {az[f]:+6.1f} |a| {an[f]:5.1f} w {w[min(f,len(w)-1)]:5.1f} rad/s")
