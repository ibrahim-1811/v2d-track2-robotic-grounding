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
    actor_centres = {f: centroid(unpack(bits, shape)) for f, (bits, _) in actor.items()}
    actor_height = np.median([np.ptp(np.nonzero(unpack(bits, shape))[0]) for bits, _ in actor.values()])

    object_text = args.object_name.replace("_", " ")
    objects = track(model, processor, frames, object_text, args.device, dtype, state_device)
    if not objects:
        raise SystemExit(f"SAM3 found no {object_text!r}")
    near = {}
    for oid, seq in objects.items():
        total = 0.0
        for f, (bits, score) in seq.items():
            mask = unpack(bits, shape)
            centre, anchor = centroid(mask), actor_centres.get(f)
            if centre is None or anchor is None:
                continue
            total += mask.sum() * score * np.exp(-np.linalg.norm(centre - anchor) / actor_height)
        near[oid] = total
    object_id = max(near, key=near.get)

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

    report = {
        "video": str(args.video_path),
        "frames": len(frames),
        "person_instances": {str(k): v for k, v in area.items()},
        "actor_id": actor_id,
        "actor_frames": len(actor),
        "object_text": object_text,
        "object_instances_near_score": {str(k): round(v, 1) for k, v in near.items()},
        "object_id": object_id,
        "detected_frames": detected,
        "tracker_seed_frame": seeds,
        "final_frames": {role: len(seq) for role, seq in kept.items()},
    }
    (args.masks_dir / "selection.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
