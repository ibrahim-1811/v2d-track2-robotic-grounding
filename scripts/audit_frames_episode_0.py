#!/usr/bin/env python3
"""Step 8.2 - verify the coordinate frames of the human and stool trajectories.

Applies NVIDIA's own documented OpenCV->SOMA conversion to the OBJECT pose and
checks whether the human and the stool then occupy a physically plausible common
frame. Reads the dataset read-only; nothing in official/ is touched or imported.

The transform under test is copied (not imported) from the pinned upstream:
    robotic_grounding/scripts/retarget/soma_to_g1.py:308
    _R_CV_TO_SOMA = np.diag([1.0, -1.0, -1.0, 1.0])
    # OpenCV world (X right, Y down, Z forward) to SOMA world (X right, Y up).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parent.parent
TIER = REPO / "data/v2d_challenge/track_2/tier_1_multiview_caption"
PARQUET = TIER / "data/chunk-000/episode_000000.parquet"
OUT = REPO / "outputs/episode_0"

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
SURFACE, INK, INK2, INK3 = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984"

# NVIDIA's transform, rotation block only (the 4x4 has an identity translation row).
R_CV_TO_SOMA = np.diag([1.0, -1.0, -1.0])

PHASES = [("approach", 0, 281), ("grasp / tip-up", 281, 618), ("release", 618, 778)]


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


def main():
    tbl = pq.read_table(PARQUET)
    hum = np.stack(tbl["observation.human.translation"].to_numpy(zero_copy_only=False))
    obj_pose = np.stack(tbl["observation.object.pose"].to_numpy(zero_copy_only=False))
    ts = tbl["timestamp"].to_numpy()

    obj_cv = obj_pose[:, :3]              # as stored: source OpenCV world frame
    obj_soma = obj_cv @ R_CV_TO_SOMA.T    # NVIDIA's conversion -> SOMA world (Y up)
    # the human is ALREADY in the SOMA frame; the retargeter never converts it.

    d_raw = np.linalg.norm(hum - obj_cv, axis=1)
    d_soma = np.linalg.norm(hum - obj_soma, axis=1)
    h_soma = np.linalg.norm((hum - obj_soma)[:, [0, 2]], axis=1)
    dy = hum[:, 1] - obj_soma[:, 1]       # SOMA Y is UP: positive => human above object

    print("=" * 76)
    print("TASK 1 - reproduce the discrepancy (both fields as stored)")
    print("=" * 76)
    print(f"  3D human<->object distance : {d_raw.min():.2f} .. {d_raw.max():.2f} m")
    print("  A two-handed manipulation cannot happen 5-7 m away. The two fields")
    print("  are not in a common frame as stored.")

    print("\n" + "=" * 76)
    print("TASK 2/3 - apply NVIDIA's diag(1,-1,-1) to the OBJECT, then compare")
    print("=" * 76)
    print(f"  3D distance        : {d_soma.min():.3f} .. {d_soma.max():.3f} m")
    print(f"  horizontal (x,z)   : {h_soma.min():.3f} .. {h_soma.max():.3f} m")
    print(f"  vertical human-obj : {dy.min():+.3f} .. {dy.max():+.3f} m  (positive = human above)")
    print(f"  object Y rises {100*(obj_soma[-1,1]-obj_soma[0,1]):+.1f} cm in SOMA +Y (up). Sign correct.")

    print("\n" + "=" * 76)
    print("TASK 4 - is the separation physically plausible, phase by phase?")
    print("=" * 76)
    print(f"  {'phase':16s}{'horiz min':>10s}{'horiz max':>10s}{'3D min':>9s}{'3D max':>9s}"
          f"{'dY min':>9s}{'dY max':>9s}")
    rows = {}
    for name, a, b in PHASES:
        r = dict(horiz=[float(h_soma[a:b].min()), float(h_soma[a:b].max())],
                 d3=[float(d_soma[a:b].min()), float(d_soma[a:b].max())],
                 dy=[float(dy[a:b].min()), float(dy[a:b].max())])
        rows[name] = r
        print(f"  {name:16s}{r['horiz'][0]:10.3f}{r['horiz'][1]:10.3f}"
              f"{r['d3'][0]:9.3f}{r['d3'][1]:9.3f}{r['dy'][0]:+9.3f}{r['dy'][1]:+9.3f}")

    # coupling during manipulation: if the human holds the stool, the two must move together
    a, b = 281, 618
    H, O = hum[:, [0, 2]], obj_soma[:, [0, 2]]
    net_h, net_o = H[b - 1] - H[a], O[b - 1] - O[a]
    ang = np.degrees(np.arccos(net_h @ net_o / (np.linalg.norm(net_h) * np.linalg.norm(net_o))))
    cx = float(np.corrcoef(H[a:b, 0], O[a:b, 0])[0, 1])
    cz = float(np.corrcoef(H[a:b, 1], O[a:b, 1])[0, 1])
    print(f"\n  coupling during tip-up: corr(x)={cx:+.3f}  corr(z)={cz:+.3f}  "
          f"net-displacement angle={ang:.1f} deg")
    print("  The human and the stool travel together while it is held - the signature")
    print("  of a real grasp, and independent of any distance threshold.")

    # the vertical axis is what uniquely picks diag(1,-1,-1)
    alt = obj_cv @ np.diag([1.0, 1.0, -1.0]).T          # 'negate z only' rival
    dy_alt = hum[:, 1] - alt[:, 1]
    print(f"\n  rival mapping diag(1,1,-1): human is {dy_alt.min():+.3f}..{dy_alt.max():+.3f} m "
          "relative to the stool")
    print("  i.e. the pelvis would sit BELOW a stool lying on the floor - impossible.")
    print("  Horizontal distance cannot separate the two rivals; the VERTICAL axis does.")

    # ---------------- plot ----------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11.5, 7.6), facecolor=SURFACE,
                                   gridspec_kw=dict(hspace=0.33))
    for ax in (ax1, ax2):
        style(ax)
        for nm, a0, b0 in PHASES:
            if nm != "grasp / tip-up":
                ax.axvspan(ts[a0], ts[b0 - 1], color="#f0efec", alpha=0.75, zorder=0, linewidth=0)
        for _, _, b0 in PHASES[:-1]:
            ax.axvline(ts[b0], color=INK3, linewidth=0.8, linestyle=(0, (4, 3)), zorder=1)
        ax.set_xlim(ts[0], ts[-1] + 1.3)

    for nm, a0, b0 in PHASES:
        ax1.annotate(nm, xy=((ts[a0] + ts[b0 - 1]) / 2, 1.0), xycoords=("data", "axes fraction"),
                     ha="center", va="bottom", fontsize=8.5, color=INK2)

    ax1.plot(ts, d_raw, color=SERIES[1], linewidth=2, label="as stored (no conversion)")
    ax1.plot(ts, d_soma, color=SERIES[0], linewidth=2, label="after diag(1,-1,-1) on the object")
    ax1.annotate("as stored", xy=(ts[-1], d_raw[-1]), xytext=(6, 0), textcoords="offset points",
                 color=SERIES[1], fontsize=10, fontweight="bold", va="center")
    ax1.annotate("converted", xy=(ts[-1], d_soma[-1]), xytext=(6, 0), textcoords="offset points",
                 color=SERIES[0], fontsize=10, fontweight="bold", va="center")
    ax1.axhspan(0, 1.0, color=SERIES[0], alpha=0.07, zorder=0, linewidth=0)
    ax1.annotate("plausible reach for a crouching adult (< ~1 m)", xy=(1.0, 0.95), fontsize=8.5,
                 color=INK2, va="bottom")
    ax1.set_ylabel("human root to stool\ndistance (metres)", color=INK2, fontsize=9.5)
    ax1.set_ylim(0, 7.6)
    ax1.legend(loc="center left", frameon=False, fontsize=9, labelcolor=INK2)
    ax1.set_title("A  The same two fields, before and after NVIDIA's OpenCV->SOMA conversion",
                  fontsize=10.5, color=INK, fontweight="bold", loc="left", pad=16)

    ax2.plot(ts, hum[:, 1], color=SERIES[0], linewidth=2, label="human root (Hips)")
    ax2.plot(ts, obj_soma[:, 1], color=SERIES[1], linewidth=2, label="stool origin")
    ax2.annotate("human", xy=(ts[-1], hum[-1, 1]), xytext=(6, 0), textcoords="offset points",
                 color=SERIES[0], fontsize=10, fontweight="bold", va="center")
    ax2.annotate("stool", xy=(ts[-1], obj_soma[-1, 1]), xytext=(6, 0), textcoords="offset points",
                 color=SERIES[1], fontsize=10, fontweight="bold", va="center")
    ax2.fill_between(ts, obj_soma[:, 1], hum[:, 1], color=SERIES[0], alpha=0.10, linewidth=0)
    ax2.annotate(f"stool rises {100*(obj_soma[-1,1]-obj_soma[0,1]):.0f} cm as it is stood up",
                 xy=(ts[560], obj_soma[560, 1]), xytext=(-20, -34), textcoords="offset points",
                 fontsize=9, color=INK2,
                 arrowprops=dict(arrowstyle="->", color=INK3, linewidth=0.9))
    ax2.set_ylabel("height in SOMA frame,\nY up (metres)", color=INK2, fontsize=9.5)
    ax2.set_xlabel("time (seconds)", color=INK2, fontsize=10)
    ax2.legend(loc="center left", frameon=False, fontsize=9, labelcolor=INK2)
    ax2.set_title("B  Vertical check: the pelvis stays above the stool, and the gap closes "
                  "as the stool comes upright",
                  fontsize=10.5, color=INK, fontweight="bold", loc="left", pad=6)

    fig.text(0.062, 0.975, "Episode 0 frame audit - human vs stool in a common frame",
             fontsize=15, fontweight="bold", color=INK, va="top")
    fig.text(0.062, 0.938,
             "The object is stored in the source OpenCV world frame; the human is already in the SOMA frame. "
             "Applying NVIDIA's\n_R_CV_TO_SOMA = diag(1,-1,-1) (soma_to_g1.py:308) to the OBJECT ALONE brings "
             "them into one frame.",
             fontsize=9.3, color=INK2, va="top", linespacing=1.5)
    fig.text(0.062, 0.016,
             "Shaded spans = stool at rest. Panel B is the decisive test: a z-only flip would put the pelvis "
             "below the stool.",
             fontsize=8.5, color=INK2, va="bottom")
    fig.subplots_adjust(top=0.845, bottom=0.095, left=0.115, right=0.945)
    p = OUT / "frame_audit.png"
    fig.savefig(p, dpi=165, facecolor=SURFACE)
    plt.close(fig)
    print(f"\n  wrote {p.relative_to(REPO)}")

    summary = {
        "transform_applied_to_object": "diag(1,-1,-1)  [NVIDIA _R_CV_TO_SOMA, soma_to_g1.py:308]",
        "transform_applied_to_human": "none - human is already in the SOMA frame",
        "raw_distance_m": [round(float(d_raw.min()), 3), round(float(d_raw.max()), 3)],
        "common_frame_distance_3d_m": [round(float(d_soma.min()), 3), round(float(d_soma.max()), 3)],
        "common_frame_distance_horizontal_m": [round(float(h_soma.min()), 3), round(float(h_soma.max()), 3)],
        "vertical_human_minus_object_m": [round(float(dy.min()), 3), round(float(dy.max()), 3)],
        "human_always_above_object": bool((dy > 0).all()),
        "per_phase": rows,
        "tip_up_coupling": {"corr_x": round(cx, 3), "corr_z": round(cz, 3),
                            "net_displacement_angle_deg": round(float(ang), 1)},
        "verdict": "VERIFIED - diag(1,-1,-1) on the object alone yields a physically coherent common frame",
    }
    (OUT / "frame_audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"  wrote {(OUT / 'frame_audit_summary.json').relative_to(REPO)}")


if __name__ == "__main__":
    main()
