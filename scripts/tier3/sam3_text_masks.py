#!/usr/bin/env python3
"""Human and object masks for CARI4D from text prompts, with SAM3 video (no clicks).

CARI4D's README makes masks with SAM2 from hand-placed prompts (object ID 0 = human,
1 = object). This writes the same layout automatically: SAM3's promptable concept
segmentation tracks every instance of "person" and of the object's name through the video,
then one instance of each is kept:

* the actor  = the person instance with the largest total mask area (bystanders sit far away),
* the object = the object instance with the most mask area near the actor (a matching chair
  in the background is far from them).

Text detection misses an instance while it looks unlike its name (a stool lying face down is
not recognised as a "bar stool" until it stands up). So each kept instance is then followed
through the whole video by SAM3's tracker, seeded with its most confident detected mask and
propagated forward and backward; detected masks are kept where they exist and the tracker
fills only the frames detection missed.

Every written mask keeps only its connected components of at least 5% of the largest one.
SAM3's own hole filling and sprinkle removal need the optional ``kernels`` package, which the
CARI4D image lacks; without this, a few stray pixels in a corner stretch the actor's bounding
box over the whole frame and SAM 3D Body places the human metres away (episode 0, frame 671).

Output: <masks_dir>/0/000000.png ... (actor) and <masks_dir>/1/000000.png ... (object), one
PNG per video frame, empty where the instance is not visible -- the input of CARI4D's
``pack_masks``. Runs inside containers/cari4d.sif (transformers 5.3 ships Sam3VideoModel).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor, Sam3VideoModel, Sam3VideoProcessor


def read_frames(video_path: Path) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not frames:
        raise SystemExit(f"no frames decoded from {video_path}")
    return frames


def track(model, processor, frames, text: str, device: str, dtype: torch.dtype, state_device: str | None) -> dict[int, dict[int, tuple[np.ndarray, float]]]:
    """{object_id: {frame: (packed mask bits, score)}} for every instance SAM3 tracks."""
    session = processor.init_video_session(
        video=frames,
        inference_device=device,
        inference_state_device=state_device,
        processing_device="cpu",
        video_storage_device="cpu",
        dtype=dtype,
    )
    session = processor.add_text_prompt(inference_session=session, text=text)
    tracks: dict[int, dict[int, tuple[np.ndarray, float]]] = {}
    with torch.inference_mode():
        for output in model.propagate_in_video_iterator(inference_session=session, max_frame_num_to_track=len(frames)):
            result = processor.postprocess_outputs(session, output)
            masks = result["masks"].cpu().numpy().astype(bool)
            for object_id, score, mask in zip(result["object_ids"].tolist(), result["scores"].tolist(), masks):
                tracks.setdefault(int(object_id), {})[int(output.frame_idx)] = (np.packbits(mask), float(score))
    return tracks


def fill_with_tracker(tracker, processor, frames, seq, shape, device, dtype, state_device) -> tuple[dict, int]:
    """Detected masks plus tracker masks for every frame detection missed; returns (seq, seed)."""
    seed = max(seq, key=lambda f: int(np.unpackbits(seq[f][0]).sum()) * seq[f][1])
    session = processor.init_video_session(
        video=frames,
        inference_device=device,
        inference_state_device=state_device,
        processing_device="cpu",
        video_storage_device="cpu",
        dtype=dtype,
    )
    processor.add_inputs_to_inference_session(
        inference_session=session, frame_idx=seed, obj_ids=1, input_masks=unpack(seq[seed][0], shape)
    )
    filled = dict(seq)
    with torch.inference_mode():
        for reverse in (False, True):
            for output in tracker.propagate_in_video_iterator(inference_session=session, start_frame_idx=seed, reverse=reverse):
                frame = int(output.frame_idx)
                if frame in filled:
                    continue
                mask = processor.post_process_masks([output.pred_masks], original_sizes=[list(shape)], binarize=True)[0]
                mask = mask.reshape(-1, *shape)[0].cpu().numpy().astype(bool)
                if mask.any():
                    filled[frame] = (np.packbits(mask), 0.0)
    return filled, seed


def unpack(bits: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    return np.unpackbits(bits, count=shape[0] * shape[1]).reshape(shape).astype(bool)


def clean(mask: np.ndarray, keep_fraction: float = 0.05) -> np.ndarray:
    """Keep connected components of at least ``keep_fraction`` of the largest one."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if count <= 2:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = 1 + np.flatnonzero(areas >= keep_fraction * areas.max())
    return np.isin(labels, keep)


HUMAN_HEIGHT_M = 1.7


def size_plausibility(ratio: float) -> float:
    """1 up to 1.5x the object's expected image size (occlusion and distance only shrink it), then a
    log-normal fall-off with sigma ln 2: 3x -> 0.61, 6x -> 0.14, 10x -> 0.02."""
    if ratio <= 1.5:
        return 1.0
    return float(np.exp(-np.log(ratio / 1.5) ** 2 / (2 * np.log(2) ** 2)))


def object_glitch(mask: np.ndarray, anchor, expected_px: float, actor_height: float) -> bool:
    """A frame where an object track sits on something else: more than 3x the object's expected image
    size AND either more than one actor height from the actor or touching the image border. SAM3
    repeatedly returns a full-height strip along the left image border for small objects (episodes
    2, 3, 4, 9, 16, 17, 20, 25, 27); for the cone in episode 17 it is within one actor height."""
    ys, xs = np.nonzero(mask)
    if not len(ys) or anchor is None or max(np.ptp(ys), np.ptp(xs)) / expected_px <= 3.0:
        return False
    height, width = mask.shape
    border = xs.min() <= 1 or ys.min() <= 1 or xs.max() >= width - 2 or ys.max() >= height - 2
    return border or np.linalg.norm(np.array([xs.mean(), ys.mean()]) - anchor) / actor_height > 1.0


def drop_glitch_components(mask: np.ndarray, anchor, expected_px: float, actor_height: float) -> np.ndarray:
    """``mask`` without its connected components that are object_glitch on their own.

    SAM3 often returns the held object and a border strip as ONE mask (episodes 3, 4, 17, 25: mug,
    bat, cone, bottle). Judged whole, such a frame is a glitch and the object in it is lost -- 55 to
    184 clean frames out of 545 to 858 -- and ``clean`` would keep the strip, the larger component.
    Returns ``mask`` itself when nothing is dropped."""
    count, labels, stats, centres = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if count <= 1 or anchor is None:
        return mask
    height, width = mask.shape
    drop = []
    for i in range(1, count):
        x, y, w, h = stats[i, :4]
        if max(w, h) - 1 <= 3.0 * expected_px:
            continue
        border = x <= 1 or y <= 1 or x + w - 1 >= width - 2 or y + h - 1 >= height - 2
        if border or np.linalg.norm(centres[i] - anchor) / actor_height > 1.0:
            drop.append(i)
    return mask & ~np.isin(labels, drop) if drop else mask


def contact_sheet(frames: list[np.ndarray], layers: list[tuple[dict, tuple[int, int, int]]], shape: tuple[int, int],
                  path: Path, picks: list[int], width: int = 480) -> None:
    """A 4-column grid of ``picks`` with each (seq, RGB colour) layer tinted on top -- for checking a mask
    set by eye before CARI4D spends an hour on it."""
    tiles = []
    for f in picks:
        image = frames[f].astype(np.float32)
        for seq, colour in layers:
            if f in seq:
                mask = unpack(seq[f][0], shape)
                image[mask] = 0.45 * image[mask] + 0.55 * np.array(colour, np.float32)
        tile = cv2.resize(image.astype(np.uint8), (width, round(width * shape[0] / shape[1])))
        cv2.putText(tile, str(f), (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        tiles.append(tile)
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])
    cv2.imwrite(str(path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])


def implausible_object_frames(seq: dict, shape: tuple[int, int], expected_px: float, actor_centres: dict,
                              actor_height: float) -> list[int]:
    """The kept object's glitch frames (object_glitch); dropped and refilled by the tracker."""
    return sorted(int(f) for f, (bits, _) in seq.items()
                  if object_glitch(unpack(bits, shape), actor_centres.get(f), expected_px, actor_height))


def implausible_frames(seq: dict, shape: tuple[int, int], window: int = 3) -> list[int]:
    """Frames where the tracked actor's mask collapses AND jumps -- a stray blob, not a person.

    SAM3 occasionally hands one frame of the actor's track to a small blob elsewhere in the image
    (episodes 6 and 9, frame 4: 2.4k px at the far side vs 34k px on the actor either side), and
    CARI4D then puts the human metres away for that frame. A frame is dropped -- and refilled by
    the tracker like any missed frame -- only when both hold: area < 25% of the track's median, and
    centroid more than half the actor's median height from the median of its +-3 neighbours (no
    one moves half their height in 1/30 s). Occlusion alone shrinks the mask but does not move it.
    """
    stats = {}
    for f, (bits, _) in seq.items():
        mask = unpack(bits, shape)
        ys, xs = np.nonzero(mask)
        if len(ys):
            stats[f] = (len(ys), np.array([xs.mean(), ys.mean()]), np.ptp(ys))
    if len(stats) < 2 * window + 1:
        return []
    median_area = np.median([a for a, _, _ in stats.values()])
    median_height = np.median([h for _, _, h in stats.values()])
    bad = []
    for f, (a, c, _) in stats.items():
        neighbours = [stats[g][1] for g in range(f - window, f + window + 1) if g != f and g in stats]
        if a < 0.25 * median_area and neighbours and np.linalg.norm(c - np.median(neighbours, axis=0)) > 0.5 * median_height:
            bad.append(int(f))
    return sorted(bad)


def centroid(mask: np.ndarray) -> np.ndarray | None:
    ys, xs = np.nonzero(mask)
    return None if xs.size == 0 else np.array([xs.mean(), ys.mean()])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video_path", required=True, type=Path)
    parser.add_argument("--object_name", required=True, help="e.g. tall_bar_stool (underscores become spaces)")
    parser.add_argument("--masks_dir", required=True, type=Path)
    parser.add_argument("--sam3_path", required=True, type=Path, help="Hugging Face format SAM3 weights")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"),
                        help="float32 on V100s, which have no bfloat16")
    parser.add_argument("--state_on_cpu", action="store_true", help="keep SAM3's per-frame memory in host RAM (16 GB V100s)")
    parser.add_argument("--object_mesh", type=Path, default=None,
                        help="released object mesh: its size, relative to the actor, picks the right instance")
    args = parser.parse_args()

    frames = read_frames(args.video_path)
    shape = frames[0].shape[:2]
    dtype = getattr(torch, args.dtype)
    state_device = "cpu" if args.state_on_cpu else None
    model = Sam3VideoModel.from_pretrained(args.sam3_path).to(args.device, dtype=dtype)
    processor = Sam3VideoProcessor.from_pretrained(args.sam3_path)

    people = track(model, processor, frames, "person", args.device, dtype, state_device)
    if not people:
        raise SystemExit("SAM3 found no person")
    area = {oid: sum(int(np.unpackbits(bits).sum()) for bits, _ in seq.values()) for oid, seq in people.items()}
    actor_id = max(area, key=area.get)
    actor = people[actor_id]
    glitches = implausible_frames(actor, shape)
    for f in glitches:
        del actor[f]
    actor_centres = {f: centroid(unpack(bits, shape)) for f, (bits, _) in actor.items()}
    actor_height = np.median([np.ptp(np.nonzero(unpack(bits, shape))[0]) for bits, _ in actor.values()])

    object_text = args.object_name.replace("_", " ")
    objects = track(model, processor, frames, object_text, args.device, dtype, state_device)
    if not objects:
        raise SystemExit(f"SAM3 found no {object_text!r}")
    # Expected image size of the object, from its released mesh and the actor's height in pixels.
    # Without it a large wrong region won on area alone: a strip along the image border in
    # episodes 3, 4, 9, 17, 20, 27 (1-2.8x the actor's size, ~1 actor height away).
    expected_px = None
    if args.object_mesh is not None:
        import trimesh

        extent_m = float(np.max(trimesh.load(str(args.object_mesh), force="mesh").bounding_box.extents))
        expected_px = extent_m / HUMAN_HEIGHT_M * actor_height
    raw = {oid: dict(seq) for oid, seq in objects.items()}
    stripped = {oid: 0 for oid in objects}
    emptied = {oid: 0 for oid in objects}
    if expected_px is not None:
        for oid, seq in objects.items():
            for f, (bits, score) in list(seq.items()):
                mask = unpack(bits, shape)
                kept_mask = drop_glitch_components(mask, actor_centres.get(f), expected_px, actor_height)
                if kept_mask is mask:
                    continue
                if kept_mask.any():
                    seq[f] = (np.packbits(kept_mask), score)
                    stripped[oid] += 1
                else:
                    del seq[f]  # all glitch: missing, so the tracker refills it
                    emptied[oid] += 1
    near, fit = {}, {}
    for oid, seq in objects.items():
        total, ratios, distances, glitch_frames = 0.0, [], [], emptied[oid]
        for f, (bits, score) in seq.items():
            mask = unpack(bits, shape)
            centre, anchor = centroid(mask), actor_centres.get(f)
            if centre is None or anchor is None:
                continue
            distance = np.linalg.norm(centre - anchor) / actor_height
            mask_area, plausible = mask.sum(), 1.0  # not `area`: that names the person instances
            if expected_px is not None and object_glitch(mask, anchor, expected_px, actor_height):
                glitch_frames += 1
                continue
            if expected_px is not None:
                ys, xs = np.nonzero(mask)
                ratio = max(np.ptp(ys), np.ptp(xs)) / expected_px
                plausible = size_plausibility(ratio)
                mask_area = min(mask_area, expected_px ** 2)  # no reward for being bigger than the object
                ratios.append(ratio)
            distances.append(distance)
            total += mask_area * score * np.exp(-distance) * plausible
        near[oid] = total
        fit[oid] = {"size_ratio_median": round(float(np.median(ratios)), 2) if ratios else None,
                    "actor_distance_median": round(float(np.median(distances)), 2) if distances else None,
                    "glitch_frames": glitch_frames, "clean_frames": len(distances),
                    "strip_removed_frames": stripped[oid]}
    object_id = max(near, key=near.get)
    chosen = fit[object_id]
    wrong = (chosen["size_ratio_median"] is not None and chosen["size_ratio_median"] > 3.0
             and chosen["actor_distance_median"] > 1.0)
    # A hand covers a small held object for most of the video, so few clean frames is normal for it.
    # Below a quarter of the video, accept the instance only on at least a second of evidence that is
    # object-sized AND within reach on the median; the tracker fills the rest from its best frame.
    low_evidence = chosen["clean_frames"] < 0.25 * len(frames)
    if low_evidence and (expected_px is None or chosen["clean_frames"] < 30 or chosen["size_ratio_median"] > 3.0
                         or chosen["actor_distance_median"] > 1.0):
        wrong = True
    if wrong:
        seq = raw[object_id]
        picks = sorted(seq)[:: max(1, len(seq) // 12)][:12]
        contact_sheet(frames, [(seq, (255, 220, 0)), (objects[object_id], (255, 0, 0))], shape,
                      args.masks_dir.with_name(args.masks_dir.name + ".refused.jpg"), picks)
        raise SystemExit(f"best {object_text!r} instance is implausible ({chosen}; all instances {fit}); "
                         "refusing to hand CARI4D a wrong object")
    object_glitches = []
    if expected_px is not None:
        object_glitches = implausible_object_frames(objects[object_id], shape, expected_px, actor_centres, actor_height)
        for f in object_glitches:
            del objects[object_id][f]

    del model
    torch.cuda.empty_cache()
    tracker = Sam3TrackerVideoModel.from_pretrained(args.sam3_path).to(args.device, dtype=dtype)
    tracker_processor = Sam3TrackerVideoProcessor.from_pretrained(args.sam3_path)
    kept = {"0": actor, "1": objects[object_id]}
    detected = {role: len(seq) for role, seq in kept.items()}
    seeds = {}
    for role, seq in kept.items():
        if len(seq) < len(frames):
            kept[role], seeds[role] = fill_with_tracker(
                tracker, tracker_processor, frames, seq, shape, args.device, dtype, state_device
            )

    for role, seq in kept.items():
        out = args.masks_dir / role
        out.mkdir(parents=True, exist_ok=True)
        for f in range(len(frames)):
            mask = clean(unpack(seq[f][0], shape)) if f in seq else np.zeros(shape, bool)
            cv2.imwrite(str(out / f"{f:06d}.png"), mask.astype(np.uint8) * 255)
    contact_sheet(frames, [(kept["0"], (0, 90, 255)), (kept["1"], (255, 0, 0))], shape, args.masks_dir / "overlay.jpg",
                  [round(i * (len(frames) - 1) / 11) for i in range(12)])
    args.masks_dir.with_name(args.masks_dir.name + ".refused.jpg").unlink(missing_ok=True)

    report = {
        "video": str(args.video_path),
        "frames": len(frames),
        "person_instances": {str(k): v for k, v in area.items()},
        "actor_id": actor_id,
        "actor_frames": len(actor),
        "actor_glitch_frames_dropped": glitches,
        "object_glitch_frames_dropped": object_glitches,
        "object_text": object_text,
        "object_instances_near_score": {str(k): round(v, 1) for k, v in near.items()},
        "object_instances_fit": {str(k): v for k, v in fit.items()},
        "object_expected_px": None if expected_px is None else round(float(expected_px), 1),
        "object_id": object_id,
        "object_low_evidence": low_evidence,
        "detected_frames": detected,
        "tracker_seed_frame": seeds,
        "final_frames": {role: len(seq) for role, seq in kept.items()},
    }
    (args.masks_dir / "selection.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
