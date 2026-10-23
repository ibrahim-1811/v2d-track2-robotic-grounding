#!/usr/bin/env python3
"""Zero-shot one FlashSAC actor over every episode of a tier. No optimizer updates.

Step D. One episode's actor is the only trained policy we have, so the honest way
to produce a whole-tier submission from it is to evaluate that same actor on all
30 references and say so. ``evaluation.source_root`` cannot do it - it only
rebases relative or ``/workspace`` paths and ours are absolute ``/work/...`` (see
CLAUDE.md) - so each episode needs a policy file carrying the TARGET's baked
config, which is what ``transfer_flash_sac_policy.py --emit-policy`` writes.

This is a parameterised rewrite of the one-off
``outputs/step_D/submission_trial_20261021/run_zero_shot_trial.py`` that produced
Trial 1 (Episode 0 at 10M steps, Kaggle AUC 0.17520). Differences:

* ``--source-policy`` and ``--out-dir``, so a second arm is a flag, not a copy;
* **resumable** - an episode whose results JSON already exists is skipped, so a
  crash 25 episodes in does not throw away the GPU time;
* refuses to start while a training container holds the card.

Every rollout uses the unmodified official evaluator at ``world_count=1``, which
is what ``pack_submission.py`` scores (it reads world 0 only), with assistance
off - the evaluator forces ``voc_scale=0.0`` itself.

    .venvs/episode0/bin/python scripts/evaluate/run_zero_shot_tier_trial.py \
        --source-policy outputs/step_D/sweep_ep0_orient_350M/portable_349999104.safetensors \
        --out-dir outputs/step_D/submission_trial2_orient350M \
        --label "Episode 0 orientation-aware 350M actor, zero-shot on all 30 Tier 1 references"

Runs on the host with the project venv; the work happens in ``flash_chord:latest``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
IMAGE = "flash_chord:latest"
FC = ROOT / "official/video_to_data/robotic_grounding/flash_chord"
ASSETS = ROOT / "official/video_to_data/robotic_grounding/source/robotic_grounding/robotic_grounding/assets"
TRANSFER_TOOL = ROOT / "scripts/train/transfer_flash_sac_policy.py"
WARP_CACHE = ROOT / "outputs/step_D/warp_cache"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-policy", required=True,
                        help="actor-only policy checkpoint, HOST path under outputs/step_D")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--tier", default="tier_1_multiview_caption")
    parser.add_argument("--label", required=True, help="one line describing the method, stored in the manifest")
    parser.add_argument("--episodes", default="all", help="'all' or '0,1,2'")
    parser.add_argument("--world-count", type=int, default=1)
    return parser.parse_args()


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def actor_payload(path: Path) -> tuple[dict, str]:
    """Tensor schema plus a hash of the raw payload, read without safetensors."""
    with path.open("rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(size))
        payload = handle.read()
    tensors = {key: value for key, value in header.items() if key != "__metadata__"}
    return tensors, hashlib.sha256(payload).hexdigest()


def run(command: list[str], log: Path) -> float:
    started = time.monotonic()
    with log.open("w") as handle:
        result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode:
        print(log.read_text()[-7000:], flush=True)
        raise RuntimeError("Command failed ({}): {}".format(result.returncode, log))
    return time.monotonic() - started


def guard_gpu() -> None:
    names = subprocess.run(["docker", "ps", "--format", "{{.Names}}"],
                           capture_output=True, text=True).stdout.split()
    busy = [name for name in names if name.startswith("v2d-train-")]
    if busy:
        raise SystemExit("REFUSING: training container(s) hold the card: {}".format(busy))


def image_id() -> str:
    out = subprocess.run(["docker", "images", "--no-trunc", "--format", "{{.ID}}", IMAGE],
                         capture_output=True, text=True, check=True).stdout.split()
    if not out:
        raise SystemExit("image {} not found".format(IMAGE))
    return out[0]


def main() -> int:
    args = parse_args()
    guard_gpu()
    source = Path(args.source_policy).resolve()
    out = Path(args.out_dir).resolve()
    for directory in ("policies", "prepared_states", "preparation", "results", "logs", "hydra"):
        (out / directory).mkdir(parents=True, exist_ok=True)

    source_digest = digest(source)
    source_payload = actor_payload(source)
    index = json.loads((ROOT / "outputs" / args.tier / "references_index.json").read_text())
    references = sorted(index["episodes"], key=lambda item: item["episode_index"])
    if args.episodes != "all":
        wanted = {int(token) for token in args.episodes.split(",")}
        references = [item for item in references if item["episode_index"] in wanted]

    manifest_path = out / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest.update({
        "method": args.label,
        "tier": args.tier,
        "source_checkpoint": str(source.relative_to(ROOT)),
        "source_sha256": source_digest,
        "source_actor_payload_sha256": source_payload[1],
        "image": IMAGE,
        "image_id": image_id(),
        "new_training_steps": 0,
        "world_count": args.world_count,
        "assistance": "disabled by official evaluator (voc_scale=0, zero force/torque limits)",
        "evaluation_selection": "first completed unmodified rollout per task; no best-of selection",
        "status": "running",
    })
    manifest.setdefault("runs", [])
    done = {record["episode_index"] for record in manifest["runs"]}

    # Mounts are identical for preparation and evaluation; only the command differs.
    base = ["docker", "run", "--rm", "--gpus", "all",
            "-v", "{}:/workspace:ro".format(FC),
            "-v", "{}:/v2d/assets:ro".format(ASSETS),
            "-v", "{}:/work/adapter_out:ro".format(ROOT / "data/adapter_out"),
            "-v", "{}:/out:ro".format(ROOT / "outputs/step_D"),
            "-v", "{}:/audit".format(out),
            "-v", "{}:/tools/transfer_flash_sac_policy.py:ro".format(TRANSFER_TOOL),
            "-v", "{}:/root/.cache/warp".format(WARP_CACHE),
            "-e", "XLA_PYTHON_CLIENT_PREALLOCATE=false",
            "-e", "PYTHONDONTWRITEBYTECODE=1",
            "-w", "/workspace", IMAGE]
    container_source = "/out/" + str(source.relative_to(ROOT / "outputs/step_D"))

    mapping_path = out / "checkpoints_container.json"
    mapping = json.loads(mapping_path.read_text()) if mapping_path.exists() else {}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    for position, episode in enumerate(references, start=1):
        idx = episode["episode_index"]
        stem = "episode_{:06d}".format(idx)
        policy = out / "policies" / (stem + ".safetensors")
        report_path = out / "results" / (stem + ".json")
        if idx in done and report_path.exists():
            print("[{}/{}] {} already evaluated; skipping".format(position, len(references), stem), flush=True)
            continue

        record = {"episode_index": idx, "object": episode["object"],
                  "scene_support": episode["scene_support"]}
        print("[{}/{}] Preparing {} ({})".format(position, len(references), stem, episode["object"]), flush=True)
        if not policy.exists():
            if idx == 0:
                # The source actor is already baked on Episode 0; nothing to repoint.
                shutil.copyfile(source, policy)
                record["preparation_seconds"] = 0.0
            else:
                target = str(episode["reference_parquet"]).replace(
                    str(ROOT / "data/adapter_out"), "/work/adapter_out", 1)
                prep = base + ["python", "/tools/transfer_flash_sac_policy.py",
                               "--source-policy", container_source,
                               "--target-parquet", target,
                               "--target-support", str(episode["scene_support"]).lower(),
                               "--world-count", "1",
                               "--replay-capacity", "500000", "--replay-minimum-size", "50000",
                               "--updates-per-collection", "2", "--transfer", "actor",
                               "--out", "/audit/prepared_states/{}.safetensors".format(stem),
                               "--emit-policy", "/audit/policies/{}.safetensors".format(stem),
                               "--report-json", "/audit/preparation/{}.json".format(stem)]
                record["preparation_command"] = prep
                record["preparation_seconds"] = run(prep, out / "logs" / (stem + "_prepare.log"))
        else:
            record["preparation_seconds"] = 0.0
            record["preparation_reused"] = True

        # The whole claim of this trial is that one unchanged actor drives all 30
        # tasks, so prove the weights really are the source's on every episode.
        if actor_payload(policy) != source_payload:
            raise RuntimeError("Actor tensors or payload bytes differ for {}".format(stem))

        mapping[str(idx)] = "/audit/policies/{}.safetensors".format(stem)
        mapping_path.write_text(json.dumps(mapping, indent=2) + "\n")

        evaluation = base + ["python", "scripts/evaluate_policy.py",
                             "evaluation.checkpoint=/audit/policies/{}.safetensors".format(stem),
                             "evaluation.world_count={}".format(args.world_count),
                             "evaluation.reset_mode=explicit", "evaluation.start_frame=0",
                             "evaluation.metrics_output=/audit/results/{}.json".format(stem),
                             "evaluation.object_trajectories_output=/audit/results/{}.parquet".format(stem),
                             "logging.mode=disabled", "hydra.run.dir=/audit/hydra/{}".format(stem)]
        print("[{}/{}] Evaluating {} unassisted".format(position, len(references), stem), flush=True)
        record["evaluation_command"] = evaluation
        record["evaluation_seconds"] = run(evaluation, out / "logs" / (stem + "_evaluate.log"))

        report = json.loads(report_path.read_text())
        assert report["checkpoint_sha256"] == digest(policy), stem
        assert report["world_count"] == args.world_count, stem
        assert report["reset_mode"] == "explicit" and report["settling_steps"] == 0, stem
        record.update(checkpoint_sha256=digest(policy), actor_payload_unchanged=True,
                      metrics=report["metrics"])
        manifest["runs"] = [r for r in manifest["runs"] if r["episode_index"] != idx] + [record]
        manifest["runs"].sort(key=lambda r: r["episode_index"])
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps({"episode_index": idx, "object": episode["object"],
                          "add_auc": record["metrics"]["add_auc"],
                          "mppe_cm": record["metrics"]["mppe_cm"],
                          "evaluation_seconds": round(record["evaluation_seconds"], 1)}), flush=True)

    if digest(source) != source_digest:
        raise RuntimeError("source checkpoint changed under us")
    manifest["status"] = "all_evaluated_not_submitted"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print("{} task(s) evaluated. Source unchanged. No training or upload performed.".format(len(references)),
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
