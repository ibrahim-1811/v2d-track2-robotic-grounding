#!/usr/bin/env python3
"""Write outputs/media/index.html: one row per recorded policy, linking its 3D replay and MP4.

Serve the media folder and open the page (works offline, any PC with Python):
    cd <media folder> && python3 -m http.server 8000      # then http://localhost:8000
The 3D links open viser_client/ (a single self-contained index.html) with ?playbackPath=.
Stdlib only.
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path

R = Path(__file__).resolve().parents[2]
MEDIA = R / "outputs/media"
META = R / "data/v2d_challenge/track_2/tier_1_multiview_caption/meta/episodes_metadata.jsonl"


def main() -> None:
    episodes = {e["episode_index"]: e for e in map(json.loads, META.open())}
    scores = {}
    for selection in (R / "outputs/submission").glob("*/selection.json"):
        for s in json.loads(selection.read_text()):
            scores[(selection.parent.name, s["episode_index"])] = s.get("mean_auc")
    rows = []
    for recording in sorted((MEDIA / "viser_client/recordings").glob("*/*.viser")):
        group, name = recording.parent.name, recording.stem
        match = re.search(r"ep(\d+)", name)
        ep = int(match.group(1)) if match else None
        meta = episodes.get(ep, {})
        mp4 = MEDIA / group / f"{name}.mp4"
        auc = scores.get((group, ep))
        rows.append(
            "<tr><td>{group}</td><td>{ep}</td><td>{obj}</td><td>{run}</td><td>{auc}</td>"
            '<td><a href="viser_client/?playbackPath=recordings/{group}/{file}">3D</a></td><td>{video}</td>'
            '<td class="task">{task}</td></tr>'.format(
                group=html.escape(group),
                ep="" if ep is None else ep,
                obj=html.escape(meta.get("object", "")),
                run=html.escape(name),
                auc="" if auc is None else f"{auc:.3f}",
                file=html.escape(recording.name),
                video=f'<a href="{html.escape(group)}/{html.escape(mp4.name)}">MP4</a>' if mp4.exists() else "",
                task=html.escape(meta.get("task", "")),
            )
        )
    (MEDIA / "index.html").write_text(
        """<!doctype html><meta charset="utf-8"><title>G1 policy replays</title>
<style>body{font:14px system-ui;margin:16px}table{border-collapse:collapse}td,th{padding:4px 8px;border-bottom:1px solid #ddd;text-align:left}
.task{color:#666;max-width:520px}</style>
<h2>G1 policy replays</h2><p>3D = interactive replay (drag to orbit, scroll to zoom, timeline at the bottom). AUC = mean of the 5 evaluation repeats.</p>
<table><tr><th>set</th><th>ep</th><th>object</th><th>policy</th><th>AUC</th><th></th><th></th><th>task</th></tr>
""" + "\n".join(rows) + "\n</table>\n"
    )
    print(f"wrote {MEDIA / 'index.html'} ({len(rows)} recordings)")


if __name__ == "__main__":
    main()
