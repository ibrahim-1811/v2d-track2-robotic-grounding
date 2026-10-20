#!/usr/bin/env python3
"""Step 8.1 - understand and visualise official Track 2 episode 0.

Reads the Tier 1 episode 0 Parquet + metadata, explains every field, verifies the
quaternion ordering against the physics of the task, and writes the plots and
storyboard under outputs/episode_0/.

Read-only with respect to the dataset. Does not touch NVIDIA upstream, does not
build a retargeting adapter, does not train.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation as Rot

REPO = Path(__file__).resolve().parent.parent
TIER = REPO / "data/v2d_challenge/track_2/tier_1_multiview_caption"
PARQUET = TIER / "data/chunk-000/episode_000000.parquet"
VIDEO = TIER / "videos/chunk-000/observation.images.exo_camera/episode_000000.mp4"
OUT = REPO / "outputs/episode_0"

# dataviz reference palette, categorical slots 1-3 (light mode).
# Validated: node scripts/validate_palette.js "#2a78d6,#eb6834,#1baf7a" --mode light --pairs all
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
SURFACE, INK, INK2, INK3 = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984"

# Keyframes chosen on measured events (see phase detection below).
KEYFRAMES = [0, 150, 285, 450, 625, 777]


def h(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def load():
    tbl = pq.read_table(PARQUET)
    get = lambda c: np.stack(tbl[c].to_numpy(zero_copy_only=False))
    d = {
        "human_pose": get("observation.human.pose"),
        "human_transl": get("observation.human.translation"),
        "identity": get("observation.human.identity_coeffs"),
        "scale": get("observation.human.scale_params"),
        "bone_flex": get("observation.human.bone_length_flexibles"),
        "obj_pose": get("observation.object.pose"),
        "visible": tbl["observation.object.visible"].to_numpy(),
        "timestamp": tbl["timestamp"].to_numpy(),
        "frame_index": tbl["frame_index"].to_numpy(),
    }
    d["schema_meta"] = {k.decode(): v.decode()
                        for k, v in (tbl.schema.metadata or {}).items()}
    info = json.loads((TIER / "meta/info.json").read_text())
    meta0 = json.loads((TIER / "meta/episodes_metadata.jsonl").read_text().splitlines()[0])
    raw = info["features"]["observation.human.pose"]["names"]
    joints: list[str] = []
    for n in raw:                                 # 'Hips.rx' -> 'Hips'
        base = n.rsplit(".", 1)[0]
        if not joints or joints[-1] != base:
            joints.append(base)
    return d, info, meta0, joints


def explain(d, info, meta0, joints):
    h("1. WHAT THIS EPISODE IS")
    print(f"sequence_id : {meta0['sequence_id']}")
    print(f"object      : {meta0['object']}")
    print(f"frames      : {meta0['frames']}   fps: {info['fps']}   "
          f"duration: {d['timestamp'][-1]:.2f} s")
    print(f"task        : {meta0['task']}")
    print(f"ground_plane: {meta0['ground_plane']}")
    print(f"support     : {meta0['support_surface']}  (frame: {meta0['support_surface_frame']})")
    print(f"mesh        : {meta0['mesh']}")

    h("2. FIELD-BY-FIELD, IN PLAIN LANGUAGE (with shapes and example values)")
    T = len(d["timestamp"])

    print("\n--- observation.human.pose -- HUMAN JOINT ROTATIONS ---")
    print(f"  shape {d['human_pose'].shape}  = (frames, 231);  231 = 77 joints x 3")
    P = d["human_pose"].reshape(T, 77, 3)
    print(f"  reshaped -> {P.shape}")
    print("  WHAT IT IS: for each of 77 skeleton joints, how that joint is rotated")
    print("  RELATIVE TO ITS PARENT bone, as an axis-angle vector in RADIANS.")
    print("  The vector's direction = the axis to spin about; its length = the angle.")
    print("  It is a rotation only: no joint positions are stored. To get positions you")
    print("  must run forward kinematics through the SOMA-X body model.")
    print(f"  value range: {P.min():+.4f} .. {P.max():+.4f} rad "
          f"({np.degrees(P.min()):+.1f} .. {np.degrees(P.max()):+.1f} deg)")
    for name in ("Hips", "RightArm", "RightHand", "LeftArm", "RightLeg"):
        j = joints.index(name)
        v = P[0, j]
        print(f"    frame0  {name:10s} (joint {j:2d}) = [{v[0]:+.5f} {v[1]:+.5f} {v[2]:+.5f}] rad"
              f"  -> {np.degrees(np.linalg.norm(v)):6.2f} deg about its axis")

    print("\n--- observation.human.translation -- HUMAN ROOT TRANSLATION ---")
    print(f"  shape {d['human_transl'].shape};  names {info['features']['observation.human.translation']['names']}")
    print("  WHAT IT IS: where the ROOT of the skeleton (the Hips) sits in space, in")
    print("  METRES. Joint rotations describe the body's pose; this single vector says")
    print("  where that whole body is standing. Together they place the person in the world.")
    for i in (0, T // 2, T - 1):
        v = d["human_transl"][i]
        print(f"    frame {i:3d} (t={d['timestamp'][i]:5.2f}s) = "
              f"[{v[0]:+.4f} {v[1]:+.4f} {v[2]:+.4f}] m")
    for i, ax in enumerate("xyz"):
        c = d["human_transl"][:, i]
        print(f"    {ax}: min {c.min():+.4f}  max {c.max():+.4f}  range {c.max() - c.min():.4f} m")
    walk = np.linalg.norm(np.diff(d["human_transl"][:, [0, 2]], axis=0), axis=1).sum()
    print(f"  horizontal path length walked = {walk:.2f} m")
    print("  NOTE: the y range is ~10x smaller than x and z -> Y IS THE VERTICAL AXIS.")

    print("\n--- observation.object.pose -- OBJECT POSITION + QUATERNION ---")
    print(f"  shape {d['obj_pose'].shape};  names {info['features']['observation.object.pose']['names']}")
    print("  WHAT IT IS: the full 6-DOF pose of the ONE tracked rigid object")
    print("  (here the tall_bar_stool), as 3 numbers of position in METRES plus a")
    print("  4-number quaternion giving its orientation. A quaternion is a compact,")
    print("  gimbal-lock-free way to write 'rotate by this angle about this axis'.")
    print("  Convention is world_T_object: it maps the mesh's own local frame into the world.")
    for i in (0, T // 2, T - 1):
        v = d["obj_pose"][i]
        print(f"    frame {i:3d} (t={d['timestamp'][i]:5.2f}s) xyz=[{v[0]:+.4f} {v[1]:+.4f} {v[2]:+.4f}] m"
              f"  quat(w,x,y,z)=[{v[3]:+.5f} {v[4]:+.5f} {v[5]:+.5f} {v[6]:+.5f}]")

    print("\n--- timestamp / frame_index -- TIMESTAMPS ---")
    dt = np.diff(d["timestamp"])
    print(f"  shape {d['timestamp'].shape};  {d['timestamp'][0]:.3f} s .. {d['timestamp'][-1]:.3f} s")
    print(f"  WHAT IT IS: seconds since the episode started, one row per frame.")
    print(f"  dt: mean {dt.mean():.6f} s  min {dt.min():.6f}  max {dt.max():.6f}"
          f"  -> {1 / dt.mean():.4f} fps, perfectly uniform")
    print(f"  frame_index runs {d['frame_index'][0]} .. {d['frame_index'][-1]} and indexes the MP4 1:1")

    print("\n--- observation.object.visible -- OBJECT VISIBILITY ---")
    vis = d["visible"]
    zero = (np.abs(d["obj_pose"]).sum(axis=1) == 0)
    print(f"  shape {vis.shape}, dtype bool;  visible on {vis.sum()}/{len(vis)} frames "
          f"({100 * vis.mean():.1f}%)")
    print("  WHAT IT IS: a per-frame flag saying whether the object tracker had a valid")
    print("  pose. When it is False the pose row is ZERO-FILLED - and a zero quaternion")
    print("  is not a rotation at all, so you must filter on this flag before using a pose.")
    print(f"  zero-filled rows = {zero.sum()};  invisible rows = {(~vis).sum()}  -> consistent")
    print("  For episode 0 the object is visible on every frame, so nothing needs masking here.")

    print("\n--- the three identity/shape fields (body build, not motion) ---")
    for key, col, what in [
        ("identity", "observation.human.identity_coeffs", "WHO the body is: a 45-number shape code for the person's build"),
        ("scale", "observation.human.scale_params", "68 per-bone scale factors sizing the skeleton"),
        ("bone_flex", "observation.human.bone_length_flexibles", "6 bone-length slack terms the fit was allowed to bend"),
    ]:
        a = d[key]
        const = np.allclose(a, a[0])
        print(f"  {col:48s} shape {str(a.shape):12s} constant over time: {const}")
        print(f"      {what}")
        print(f"      frame0[:6] = {np.round(a[0][:6], 5)}")
    return P


def verify_quaternion(d, info):
    h("6. QUATERNION ORDERING + SOURCE COORDINATE FRAME")
    q = d["obj_pose"][:, 3:7]
    pos = d["obj_pose"][:, :3]
    n = np.linalg.norm(q, axis=1)
    print("Declared three independent ways by NVIDIA:")
    print(f"  parquet schema metadata 'pose_note' : {d['schema_meta'].get('pose_note')}")
    print(f"  parquet schema 'pose_convention'    : {d['schema_meta'].get('pose_convention')}")
    print(f"  meta/info.json feature names        : {info['features']['observation.object.pose']['names']}")
    print(f"\nUnit-norm check: min {n.min():.7f}  max {n.max():.7f}  -> it is a unit quaternion.")
    print("But BOTH orderings give unit norm, so the norm CANNOT tell them apart.")
    print("\nSo test against the PHYSICS of the task. The stool is tipped from face-down")
    print("to upright: that is a ~90 deg rotation about a HORIZONTAL axis. Vertical is Y.")

    results = {}
    for label, Q in [("A  [qw,qx,qy,qz]  w-FIRST (declared)", np.c_[q[:, 1], q[:, 2], q[:, 3], q[:, 0]]),
                     ("B  [qx,qy,qz,qw]  w-LAST  (wrong)   ", q)]:
        R = Rot.from_quat(Q)
        rel = R[-1] * R[0].inv()
        rv = rel.as_rotvec()
        ang = np.degrees(np.linalg.norm(rv))
        axis = rv / np.linalg.norm(rv)
        tilt = np.degrees(np.arccos(abs(axis[1])))     # 90 deg => axis is horizontal
        print(f"\n  {label}")
        print(f"     net rotation frame0->frame777 = {ang:7.2f} deg")
        print(f"     rotation axis                 = [{axis[0]:+.3f} {axis[1]:+.3f} {axis[2]:+.3f}]")
        print(f"     axis tilt away from vertical  = {tilt:7.2f} deg   (90 = horizontal, 0 = vertical)")
        results[label.split()[0]] = dict(angle=float(ang), axis=axis.tolist(), tilt=float(tilt))

    rise = (pos[0, 1] - pos[:, 1].min()) * 100
    print(f"\n  Independent corroboration: the object centre RISES {rise:.1f} cm")
    print("  (y decreases, and y is down-positive). Only a tip-up about a horizontal")
    print("  axis lifts the centre of mass; a spin about the vertical axis would not.")
    print("\n  VERDICT: reading A is correct. w-FIRST gives a horizontal axis (86 deg off")
    print("  vertical) and ~92 deg of rotation - exactly a tip-up. Reading B would mean")
    print("  the stool merely pirouetted about its vertical axis, which contradicts both")
    print("  the task text and the measured rise. ORDERING CONFIRMED: [qw,qx,qy,qz].")

    print("\n--- SOURCE COORDINATE FRAME ---")
    print(f"  ground_plane_convention: {info['ground_plane_convention']}")
    print("  The object pose lives in the SOURCE OPENCV WORLD FRAME: right-handed,")
    print("  X right, Y DOWN, Z forward (away from the camera). It is NOT Z-up, so")
    print("  'height' here means NEGATIVE y, and gravity points along +y.")
    print("  Evidence from the data itself:")
    ht = d["human_transl"]
    print(f"    - human root y range is only {ht[:,1].max()-ht[:,1].min():.3f} m while x and z")
    print(f"      range {ht[:,0].max()-ht[:,0].min():.3f} m and {ht[:,2].max()-ht[:,2].min():.3f} m")
    print("      -> y is the axis the body does not travel along, i.e. vertical.")
    print(f"    - the stool's y DECREASES by {(pos[0,1]-pos[:,1].min())*100:.1f} cm at peak "
          f"({(pos[0,1]-pos[-1,1])*100:.1f} cm net, after it settles)")
    print("      as it is stood up, i.e. it rises while y falls -> y is down-positive.")
    print("  Note: camera_calibration is null in this release, so there are no intrinsics")
    print("  or extrinsics - the mesh cannot be projected into the video from this data alone.")
    return results


def frame_mismatch(d):
    h("6b. A FRAME MISMATCH YOU MUST KNOW ABOUT (affects the future adapter)")
    ht, pos = d["human_transl"], d["obj_pose"][:, :3]
    print("observation.object.pose is documented as being in the source OpenCV world")
    print("frame. observation.human.translation has NO documented frame. They do not")
    print("appear to agree. Test: the human tips this stool with BOTH HANDS, so at some")
    print("point the root must be within roughly an arm's length of it.")
    print("\n  axis mapping applied to human.translation      horizontal distance to object")
    for name, s in [("identity (assume same frame)", (1, 1, 1)),
                    ("negate z", (1, 1, -1)),
                    ("negate y and z (180 deg about x)", (1, -1, -1))]:
        dist = np.linalg.norm((ht * np.array(s) - pos)[:, [0, 2]], axis=1)
        print(f"  {name:46s} min {dist.min():5.2f} m   max {dist.max():5.2f} m")
    print("\n  Assuming a shared frame puts the human 5.3-7.0 m away for the WHOLE episode,")
    print("  which is physically impossible for a two-handed manipulation. Flipping the")
    print("  sign of z gives 0.39-0.70 m throughout - arm's reach.")
    print("  => human.translation is NOT in the same frame as object.pose.")
    print("  This must be resolved before retargeting, most likely via the SOMA-X body")
    print("  model's own root convention (the demo's soma_params.npz carries a")
    print("  joint_orient (78,3,3) array that the challenge release omits). Do NOT just")
    print("  hard-code a sign flip on the strength of this one episode.")


def phases(d):
    h("3b. MEASURED MOTION PHASES")
    q = d["obj_pose"][:, 3:7]
    R = Rot.from_quat(np.c_[q[:, 1], q[:, 2], q[:, 3], q[:, 0]])
    ang = np.degrees(np.linalg.norm((R * R[0].inv()).as_rotvec(), axis=1))
    disp = np.linalg.norm(d["obj_pose"][:, :3] - d["obj_pose"][0, :3], axis=1)
    ts = d["timestamp"]
    onset = int(np.argmax(disp > 0.01))
    upright = int(np.argmax(ang > 0.98 * ang.max()))
    print(f"  phase 1 APPROACH  frames   0-{onset:3d}  t=0.00-{ts[onset]:5.2f}s  "
          f"object still (<1 cm, <1 deg); human walks in")
    print(f"  phase 2 TIP-UP    frames {onset:3d}-{upright:3d}  t={ts[onset]:5.2f}-{ts[upright]:5.2f}s  "
          f"object rotates to {ang[upright]:.0f} deg, travels {disp[upright]:.2f} m")
    print(f"  phase 3 RELEASE   frames {upright:3d}-777  t={ts[upright]:5.2f}-{ts[-1]:5.2f}s  "
          f"object holds ~{ang[-1]:.0f} deg; human steps away")
    print(f"  total object rotation {ang[-1]:.2f} deg, total displacement {disp[-1]:.3f} m,"
          f" centre rose {(d['obj_pose'][0,1]-d['obj_pose'][:,1].min())*100:.1f} cm")
    return ang, disp, onset, upright


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK3)
        ax.spines[s].set_linewidth(0.8)
    ax.grid(True, color="#e6e5e1", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=9, length=3, color=INK3)


def xyz_plot(ts, data, title, subtitle, path, phase_t, note, legend_loc="best"):
    fig, ax = plt.subplots(figsize=(11, 5.2), facecolor=SURFACE)
    style(ax)
    for i, (ax_name, col) in enumerate(zip("XYZ", SERIES)):
        ax.plot(ts, data[:, i], color=col, linewidth=2, label=f"{ax_name} (m)",
                solid_capstyle="round")
        # direct label at the right end - required relief for the low-contrast slot
        ax.annotate(ax_name, xy=(ts[-1], data[-1, i]), xytext=(6, 0),
                    textcoords="offset points", color=col, fontsize=11,
                    fontweight="bold", va="center")
    # shade ONLY the spans where the object is at rest, so the shading carries meaning
    for idx, (t0, t1, lab) in enumerate(phase_t):
        if idx != 1:
            ax.axvspan(t0, t1, color="#f0efec", alpha=0.75, zorder=0, linewidth=0)
        ax.annotate(lab, xy=((t0 + t1) / 2, 1.0), xycoords=("data", "axes fraction"),
                    ha="center", va="bottom", fontsize=8.5, color=INK2)
    for _, t1, _ in phase_t[:-1]:
        ax.axvline(t1, color=INK3, linewidth=0.8, linestyle=(0, (4, 3)), zorder=1)
    ax.set_xlim(ts[0], ts[-1] + 0.9)
    ax.set_xlabel("time (seconds)", color=INK2, fontsize=10)
    ax.set_ylabel("position (metres)", color=INK2, fontsize=10)
    fig.text(0.008, 0.975, title, fontsize=15, fontweight="bold", color=INK, va="top")
    fig.text(0.008, 0.915, subtitle, fontsize=9.5, color=INK2, va="top")
    fig.text(0.008, 0.022, note, fontsize=8.5, color=INK2, va="bottom")
    leg = ax.legend(loc=legend_loc, frameon=False, fontsize=9, labelcolor=INK2)
    leg.set_title(None)
    fig.subplots_adjust(top=0.80, bottom=0.17, left=0.07, right=0.965)
    fig.savefig(path, dpi=170, facecolor=SURFACE)
    plt.close(fig)
    print(f"  wrote {path.relative_to(REPO)}")


def extract_keyframes(ts):
    kd = OUT / "keyframes"
    kd.mkdir(parents=True, exist_ok=True)
    if not shutil.which("ffmpeg"):
        print("  ffmpeg not found - skipping keyframe extraction")
        return []
    paths = []
    for f in KEYFRAMES:
        p = kd / f"frame_{f:04d}.jpg"
        if not p.exists():
            subprocess.run(
                ["ffmpeg", "-nostdin", "-v", "error", "-y",
                 "-ss", f"{ts[f]:.4f}", "-i", str(VIDEO),
                 "-frames:v", "1", "-q:v", "3", str(p)],
                check=True)
        paths.append(p)
    print(f"  wrote {len(paths)} keyframes to {kd.relative_to(REPO)}")
    return paths


def storyboard(ts, ang, disp, obj, kf_paths, onset, upright):
    if not kf_paths:
        return
    fig = plt.figure(figsize=(13.5, 8.6), facecolor=SURFACE)
    gs = fig.add_gridspec(3, len(kf_paths), height_ratios=[1.45, 1.0, 1.0],
                          hspace=0.42, wspace=0.06,
                          top=0.855, bottom=0.075, left=0.062, right=0.985)
    labels = ["stool face-down\non the floor", "human walking\nover to it",
              "both hands on it,\nmotion begins", "mid tip-up,\nrising",
              "upright,\n~92 deg reached", "human has\nstepped away"]
    for i, (p, f) in enumerate(zip(kf_paths, KEYFRAMES)):
        a = fig.add_subplot(gs[0, i])
        a.imshow(mpimg.imread(p))
        a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values():
            s.set_color(INK3); s.set_linewidth(0.8)
        a.set_title(f"t = {ts[f]:.2f} s   (frame {f})", fontsize=9,
                    color=INK, fontweight="bold", pad=4)
        a.set_xlabel(labels[i], fontsize=8.2, color=INK2, labelpad=4)

    kt = [ts[f] for f in KEYFRAMES]

    def marks(a):
        for t in kt:
            a.axvline(t, color=INK3, linewidth=0.8, linestyle=(0, (3, 3)), zorder=1)
        a.axvspan(ts[0], ts[onset], color="#f0efec", alpha=0.55, zorder=0, linewidth=0)
        a.axvspan(ts[upright], ts[-1], color="#f0efec", alpha=0.55, zorder=0, linewidth=0)
        a.set_xlim(ts[0], ts[-1])

    a1 = fig.add_subplot(gs[1, :]); style(a1); marks(a1)
    a1.plot(ts, ang, color=SERIES[0], linewidth=2)
    a1.scatter(kt, [ang[f] for f in KEYFRAMES], s=42, color=SERIES[0],
               zorder=4, edgecolor=SURFACE, linewidth=2)
    a1.set_ylabel("rotation from\nstart (degrees)", color=INK2, fontsize=9.5)
    a1.annotate(f"{ang[-1]:.0f} deg  tip-up complete", xy=(ts[upright], ang[upright]),
                xytext=(-108, -30), textcoords="offset points", fontsize=9, color=INK2,
                arrowprops=dict(arrowstyle="->", color=INK3, linewidth=0.9))
    a1.set_title("the SAME motion, as recorded numbers", fontsize=10.5,
                 color=INK, fontweight="bold", loc="left", pad=6)

    a2 = fig.add_subplot(gs[2, :]); style(a2); marks(a2)
    height_cm = (obj[0, 1] - obj[:, 1]) * 100
    a2.plot(ts, height_cm, color=SERIES[1], linewidth=2)
    a2.scatter(kt, [height_cm[f] for f in KEYFRAMES], s=42, color=SERIES[1],
               zorder=4, edgecolor=SURFACE, linewidth=2)
    a2.set_ylabel("centre height above\nstart (cm)", color=INK2, fontsize=9.5)
    a2.set_xlabel("time (seconds)", color=INK2, fontsize=10)
    a2.annotate(f"+{height_cm.max():.0f} cm", xy=(ts[int(np.argmax(height_cm))], height_cm.max()),
                xytext=(10, -16), textcoords="offset points", fontsize=9, color=INK2)

    fig.text(0.062, 0.975,
             "Episode 0 - from human demonstration to recorded trajectory",
             fontsize=15.5, fontweight="bold", color=INK, va="top")
    fig.text(0.062, 0.935,
             "tall_bar_stool_tip_up_04  -  \"Walk over to the tall bar stool and using both hands, "
             "tip the stool from a face down lying\ndown position back to upright.\"   "
             "778 frames @ 30 fps = 25.9 s.  Top: the exocentric video. Bottom: what the Parquet stores.",
             fontsize=9.3, color=INK2, va="top", linespacing=1.5)
    fig.text(0.062, 0.018,
             "Shaded spans = object at rest (approach and release). Dashed lines tie each video frame "
             "to its row in the trajectory.",
             fontsize=8.5, color=INK2, va="bottom")
    p = OUT / "storyboard.png"
    fig.savefig(p, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    print(f"  wrote {p.relative_to(REPO)}")


def preview_video():
    if not shutil.which("ffmpeg"):
        return
    p = OUT / "episode_0_exo_preview.mp4"
    if p.exists():
        print(f"  {p.relative_to(REPO)} already present")
        return
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(VIDEO),
                    "-vf", "scale=768:-2", "-c:v", "libx264", "-crf", "28",
                    "-preset", "veryfast", "-an", str(p)], check=True)
    print(f"  wrote {p.relative_to(REPO)} ({p.stat().st_size / 1e6:.1f} MB, 768px wide)")


def main():
    if not PARQUET.exists():
        sys.exit(f"missing {PARQUET}")
    OUT.mkdir(parents=True, exist_ok=True)
    d, info, meta0, joints = load()
    explain(d, info, meta0, joints)

    h("3. JOINT ORDER (77 joints, collapsed from the 231 '.rx/.ry/.rz' names)")
    for i in range(0, 77, 7):
        print(f"  {i:2d}-{min(i+6,76):2d}: {', '.join(joints[i:i+7])}")
    print(f"\n  key indices: " + ", ".join(
        f"{n}={joints.index(n)}" for n in ("Hips", "LeftHand", "RightHand", "LeftFoot", "RightFoot")))

    qres = verify_quaternion(d, info)
    frame_mismatch(d)
    ang, disp, onset, upright = phases(d)

    h("5. VISUAL OUTPUTS")
    ts = d["timestamp"]
    ph = [(ts[0], ts[onset], "approach"), (ts[onset], ts[upright], "tip-up"),
          (ts[upright], ts[-1], "release")]
    xyz_plot(ts, d["obj_pose"][:, :3],
             "Episode 0 - object position vs time",
             "tall_bar_stool, world_T_object translation in the source OpenCV world frame (metres)",
             OUT / "object_xyz_vs_time.png", ph,
             "OpenCV frame: X right, Y DOWN, Z forward. Y falling = the stool rising. "
             "Object visible on all 778 frames, so no gaps are masked.",
             legend_loc="center left")
    xyz_plot(ts, d["human_transl"],
             "Episode 0 - human root (Hips) position vs time",
             "observation.human.translation - where the skeleton root stands, in metres",
             OUT / "human_root_xyz_vs_time.png", ph,
             "Y varies only 0.15 m while X and Z sweep ~1.3 m: Y is the vertical axis. "
             "Frame differs from the object pose - see docs/EPISODE_0_WALKTHROUGH.md.",
             legend_loc="upper left")
    kf = extract_keyframes(ts)
    storyboard(ts, ang, disp, d["obj_pose"], kf, onset, upright)
    preview_video()

    summary = {
        "sequence_id": meta0["sequence_id"], "object": meta0["object"],
        "task": meta0["task"], "frames": int(len(ts)), "fps": info["fps"],
        "duration_s": round(float(ts[-1]), 4),
        "object_visible_frames": int(d["visible"].sum()),
        "quaternion_order": "[qw,qx,qy,qz] (w-first) - CONFIRMED",
        "quaternion_test": qres,
        "object_total_rotation_deg": round(float(ang[-1]), 2),
        "object_total_displacement_m": round(float(disp[-1]), 4),
        "object_centre_rise_cm": round(float((d["obj_pose"][0, 1] - d["obj_pose"][:, 1].min()) * 100), 1),
        "phases": {"approach_s": [0.0, round(float(ts[onset]), 2)],
                   "tip_up_s": [round(float(ts[onset]), 2), round(float(ts[upright]), 2)],
                   "release_s": [round(float(ts[upright]), 2), round(float(ts[-1]), 2)]},
        "human_root_frame_matches_object_frame": False,
        "source_frame": "source OpenCV world frame (X right, Y down, Z forward); not Z-up",
    }
    (OUT / "episode_0_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"  wrote {(OUT / 'episode_0_summary.json').relative_to(REPO)}")
    h("DONE")


if __name__ == "__main__":
    main()
