"""Execute the authorized 30-task zero-shot trial; no optimizer updates or training."""
import hashlib
import json
import shutil
import struct
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
SOURCE = ROOT / "outputs/step_D/source_ep0_w1024/policy_10000384.safetensors"
IMAGE = "sha256:d51290087ec5c3d0028ff914c2dc6314bc98af2414f6b425cbf47c7f129d2661"
FC = ROOT / "official/video_to_data/robotic_grounding/flash_chord"
ASSETS = ROOT / "official/video_to_data/robotic_grounding/source/robotic_grounding/robotic_grounding/assets"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def actor_payload(path):
    with path.open("rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(size))
        payload = handle.read()
    tensors = {key: value for key, value in header.items() if key != "__metadata__"}
    return tensors, hashlib.sha256(payload).hexdigest()


def run(command, log):
    started = time.monotonic()
    with log.open("w") as handle:
        result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode:
        print(log.read_text()[-7000:], flush=True)
        raise RuntimeError(f"Command failed ({result.returncode}): {log}")
    return time.monotonic() - started


def main():
    for directory in ("policies", "prepared_states", "preparation", "results", "logs", "hydra"):
        (OUT / directory).mkdir(exist_ok=True)
    cache = OUT / "warp_cache"
    if not cache.exists():
        shutil.copytree(ROOT / "outputs/step_D/orientation_plan_20261021/warp_cache", cache)
    source_digest = digest(SOURCE)
    source_payload = actor_payload(SOURCE)
    references = json.loads((ROOT / "outputs/tier_1_multiview_caption/references_index.json").read_text())["episodes"]
    manifest_path = OUT / "run_manifest.json"
    if manifest_path.exists():
        raise RuntimeError("Refusing to overwrite an existing trial manifest")
    manifest = {"method": "Episode 0 10M actor, zero-shot on all 30 Tier 1 references",
                "source_checkpoint": str(SOURCE), "source_sha256": source_digest,
                "source_actor_payload_sha256": source_payload[1], "image_id": IMAGE,
                "new_training_steps": 0, "world_count": 1, "assistance": "disabled by official evaluator",
                "evaluation_selection": "first completed unmodified rollout per task; no best-of selection",
                "runs": [], "status": "running"}
    base = ["docker", "run", "--rm", "--gpus", "all",
            "-v", f"{FC}:/workspace:ro", "-v", f"{ASSETS}:/v2d/assets:ro",
            "-v", f"{ROOT}/data/adapter_out:/work/adapter_out:ro",
            "-v", f"{ROOT}/outputs/step_D:/out:ro", "-v", f"{OUT}:/audit",
            "-v", f"{ROOT}/scripts/transfer_flash_sac_policy.py:/tools/transfer_flash_sac_policy.py:ro",
            "-v", f"{cache}:/root/.cache/warp", "-e", "XLA_PYTHON_CLIENT_PREALLOCATE=false",
            "-e", "PYTHONDONTWRITEBYTECODE=1", "-w", "/workspace", IMAGE]
    mapping = {}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    for episode in sorted(references, key=lambda item: item["episode_index"]):
        idx = episode["episode_index"]
        stem = f"episode_{idx:06d}"
        policy = OUT / "policies" / (stem + ".safetensors")
        if policy.exists():
            raise RuntimeError(f"Refusing to overwrite {policy}")
        record = {"episode_index": idx, "object": episode["object"], "scene_support": episode["scene_support"]}
        print(f"[{idx + 1}/30] Preparing {episode['object']}", flush=True)
        if idx == 0:
            shutil.copyfile(SOURCE, policy)
            record["preparation_seconds"] = 0
        else:
            target = str(Path(episode["reference_parquet"])).replace(str(ROOT / "data/adapter_out"), "/work/adapter_out", 1)
            prep = base + ["python", "/tools/transfer_flash_sac_policy.py",
                           "--source-policy", "/out/source_ep0_w1024/policy_10000384.safetensors",
                           "--target-parquet", target, "--target-support", str(episode["scene_support"]).lower(),
                           "--world-count", "1", "--replay-capacity", "500000", "--replay-minimum-size", "50000",
                           "--updates-per-collection", "2", "--transfer", "actor",
                           "--out", f"/audit/prepared_states/{stem}.safetensors",
                           "--emit-policy", f"/audit/policies/{stem}.safetensors",
                           "--report-json", f"/audit/preparation/{stem}.json"]
            record["preparation_command"] = prep
            record["preparation_seconds"] = run(prep, OUT / "logs" / (stem + "_prepare.log"))
        if actor_payload(policy) != source_payload:
            raise RuntimeError(f"Actor tensors or bytes differ for {stem}")
        mapping[str(idx)] = f"/audit/policies/{stem}.safetensors"
        (OUT / "checkpoints_container.json").write_text(json.dumps(mapping, indent=2) + "\n")
        evaluation = base + ["python", "scripts/evaluate_policy.py",
                             f"evaluation.checkpoint=/audit/policies/{stem}.safetensors",
                             "evaluation.world_count=1", "evaluation.reset_mode=explicit", "evaluation.start_frame=0",
                             f"evaluation.metrics_output=/audit/results/{stem}.json",
                             f"evaluation.object_trajectories_output=/audit/results/{stem}.parquet",
                             "logging.mode=disabled", f"hydra.run.dir=/audit/hydra/{stem}"]
        print(f"[{idx + 1}/30] Evaluating {episode['object']} unassisted", flush=True)
        record["evaluation_command"] = evaluation
        record["evaluation_seconds"] = run(evaluation, OUT / "logs" / (stem + "_evaluate.log"))
        report = json.loads((OUT / "results" / (stem + ".json")).read_text())
        assert report["checkpoint_sha256"] == digest(policy)
        assert report["world_count"] == 1 and report["reset_mode"] == "explicit" and report["settling_steps"] == 0
        record.update(checkpoint_sha256=digest(policy), actor_payload_unchanged=True, metrics=report["metrics"])
        manifest["runs"].append(record)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps({key: record[key] for key in ("episode_index", "object", "preparation_seconds", "evaluation_seconds", "metrics")}), flush=True)
    assert digest(SOURCE) == source_digest
    manifest["status"] = "all_30_evaluated_not_submitted"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print("All 30 tasks evaluated. Source unchanged. No training or upload performed.", flush=True)


if __name__ == "__main__":
    main()
