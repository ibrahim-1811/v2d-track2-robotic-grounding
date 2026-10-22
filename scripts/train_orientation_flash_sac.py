#!/usr/bin/env python3
"""Launch an orientation-aware FlashSAC run against NVIDIA's unmodified trainer.

This composes a Docker invocation of upstream ``scripts/train_flash_sac.py``
with two generated config-group files (see
``scripts/make_orientation_curriculum.py``) layered on through Hydra's
``--config-dir``. Nothing under ``official/`` is written or patched; the
thirteenth reward term arrives purely as ``objective._target_`` pointing at
``pipeline.flash_sac_orientation``.

Stdlib only -- runs on host ``python3`` as well as ``.venvs/episode0``.

    # see the exact command without running it
    python3 scripts/train_orientation_flash_sac.py --dry-run

    # two-collection smoke
    python3 scripts/train_orientation_flash_sac.py --tag smoke --total-steps 2048

    # the real run
    python3 scripts/train_orientation_flash_sac.py --tag ep0_orient_350M
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

FLASH_CHORD = REPO / "official/video_to_data/robotic_grounding/flash_chord"
ASSETS = REPO / "official/video_to_data/robotic_grounding/source/robotic_grounding/robotic_grounding/assets"
ORIENTATION_CONFIGS = REPO / "configs/orientation"
STEP_D = REPO / "outputs/step_D"
WARP_CACHE = STEP_D / "warp_cache"

EPISODE_0_PARQUET = (
    "/work/adapter_out/whole_body/soma/"
    "sequence_id=2026-06-04_14-16-23_tall_bar_stool_tip_up_04/robot_name=g1/data.parquet"
)

#: NVIDIA's terminal budget. Divisible by 1024, so every curriculum threshold
#: still lands on a collection boundary and the LR horizon is the full
#: 683,592 optimizer updates at 2.0 updates/collection.
FULL_BUDGET = 349_999_104

#: Seconds between echoed progress lines. Every line still reaches the log.
PROGRESS_ECHO_SECONDS = 30.0

#: Per-checkpoint cost on disk: policy_*.safetensors + state_*.safetensors.
POLICY_BYTES = 4_859_852
STATE_BYTES = 159_940_692


def gib(value: int) -> float:
    return value / 1024**3


def preflight(args: argparse.Namespace, output_dir: Path) -> dict:
    """Refuse to launch on a bad budget, a reused output directory or thin disk."""
    problems: list[str] = []

    if args.total_steps % args.world_count != 0:
        problems.append(
            f"total_environment_steps {args.total_steps} is not divisible by world_count "
            f"{args.world_count}; curriculum thresholds would land mid-collection"
        )
    if args.save_interval % args.world_count != 0:
        problems.append(
            f"save_interval {args.save_interval} is not divisible by world_count {args.world_count}"
        )
    # The tag becomes a docker --name, which allows only [a-zA-Z0-9][a-zA-Z0-9_.-]*.
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.tag):
        problems.append(
            f"--tag {args.tag!r} is not a valid docker container name component; "
            "use letters, digits, '_', '.' or '-' only"
        )
    if output_dir.exists() and any(output_dir.iterdir()):
        problems.append(f"output directory {output_dir} already exists and is not empty; pick another --tag")
    for required in (FLASH_CHORD, ASSETS, ORIENTATION_CONFIGS / "curriculum", ORIENTATION_CONFIGS / "objective"):
        if not required.exists():
            problems.append(f"missing required path {required}")

    collections = args.total_steps // args.world_count
    updates = int(collections * args.updates_per_collection)
    checkpoints = args.total_steps // args.save_interval
    needed = checkpoints * (POLICY_BYTES + STATE_BYTES)
    free = shutil.disk_usage(REPO).free
    # Keep a 5 GiB floor so the machine stays usable and nothing else is at risk.
    if needed + 5 * 1024**3 > free:
        problems.append(
            f"checkpoints need {gib(needed):.1f} GiB plus a 5 GiB floor, but only {gib(free):.1f} GiB is free"
        )

    plan = {
        "world_count": args.world_count,
        "total_environment_steps": args.total_steps,
        "collections": collections,
        "updates_per_collection": args.updates_per_collection,
        "total_optimizer_updates": updates,
        "learning_rate_schedule_updates": updates,
        "replay_capacity": args.replay_capacity,
        "replay_minimum_size": args.replay_minimum,
        "save_interval_environment_steps": args.save_interval,
        "expected_checkpoints": checkpoints,
        "checkpoint_bytes_expected": needed,
        "disk_free_bytes": free,
        "seed": args.seed,
        "object_orientation_weight": args.orientation_weight,
        "problems": problems,
    }
    return plan


def docker_command(args: argparse.Namespace, output_dir: Path) -> list[str]:
    """The exact container invocation, as a list, so it is auditable and logged."""
    objective = "orientation_recon_body" if args.orientation else "recon_body"
    curriculum = "orientation_recon_body_flash_sac" if args.orientation else "recon_body_flash_sac"
    container_output = f"/out/{output_dir.name}"

    command = [
        "docker", "run", "--rm", "--gpus", "all",
        "--name", f"v2d-train-{args.tag}",
        "-v", f"{FLASH_CHORD}:/workspace",
        "-v", f"{ASSETS}:/v2d/assets:ro",
        "-v", f"{REPO / 'data/adapter_out'}:/work/adapter_out:ro",
        "-v", f"{REPO / 'pipeline'}:/v2d/repo/pipeline:ro",
        "-v", f"{ORIENTATION_CONFIGS}:/v2d/configs:ro",
        "-v", f"{STEP_D}:/out",
        "-v", f"{WARP_CACHE}:/root/.cache/warp",
        "-e", "PYTHONPATH=/v2d/repo",
        # train_flash_sac.py only setdefault()s this; make it explicit anyway.
        "-e", "XLA_PYTHON_CLIENT_PREALLOCATE=false",
        "-w", "/workspace",
        args.image,
        # Our entrypoint is upstream's main() with a local metric sink swapped
        # in for the W&B logger it otherwise skips when logging is disabled.
        "python", "-m", "pipeline.flash_sac_train_entry",
        "--config-dir", "/v2d/configs",
        "experiment=g1_recon_body_flash_sac",
        f"objective={objective}",
        f"curriculum={curriculum}",
        # Hydra's override grammar needs the inner quotes: the path itself
        # contains "=" in sequence_id=... and robot_name=... (see CLAUDE.md).
        f"task.parquet='{args.parquet}'",
        f"scene.world_count={args.world_count}",
        "scene.support=false",
        f"training.total_environment_steps={args.total_steps}",
        f"training.updates_per_collection={args.updates_per_collection}",
        f"training.replay.capacity={args.replay_capacity}",
        f"training.replay.minimum_size={args.replay_minimum}",
        f"training.checkpoint.save_interval_environment_steps={args.save_interval}",
        f"training.checkpoint.save_replay={str(args.save_replay).lower()}",
        f"training.seed={args.seed}",
        f"reset.seed={args.seed}",
        "logging.mode=disabled",
        f"output_dir={container_output}",
    ]
    if args.orientation:
        command.append(f"objective.object_orientation_weight={args.orientation_weight}")
    command.extend(args.extra)
    return command


def finalize(tag: str, output_dir: Path, image: str) -> int:
    """Close out a detached run: chown artefacts back and record the outcome."""
    manifest_path = output_dir / "launch_manifest.json"
    if not manifest_path.exists():
        print(f"no launch manifest at {manifest_path}")
        return 1
    still_running = subprocess.run(
        ["docker", "ps", "--quiet", "--filter", f"name=v2d-train-{tag}"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    if still_running:
        print(f"v2d-train-{tag} is still running; not finalizing")
        return 1
    subprocess.run(
        ["docker", "run", "--rm", "-v", f"{STEP_D}:/out", image, "chown", "-R", "1000:1000", "/out"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    plan = json.loads(manifest_path.read_text())
    plan["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    plan["elapsed_seconds"] = time.time() - plan.get("started_unix", time.time())
    manifest_path.write_text(json.dumps(plan, indent=2) + "\n")
    print(f"finalized outputs/step_D/{tag} after {plan['elapsed_seconds'] / 3600:.2f} h")
    for checkpoint in sorted(output_dir.glob("policy_*.safetensors")):
        print(f"  {checkpoint.name}")
    return 0


def render_command(command: list[str]) -> str:
    """One copy-pasteable shell command, wrapped at argument boundaries."""
    rendered: list[str] = []
    pending = ""
    for token in command:
        quoted = shlex.quote(token)
        if pending:
            rendered.append(f"{pending} {quoted}")
            pending = ""
        elif token in {"-v", "-e", "-w", "-m", "--name", "--config-dir", "--gpus"}:
            pending = quoted
        else:
            rendered.append(quoted)
    if pending:
        rendered.append(pending)
    return " \\\n  ".join(rendered)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", default="ep0_orient_350M", help="output subdirectory under outputs/step_D")
    parser.add_argument("--total-steps", type=int, default=FULL_BUDGET)
    parser.add_argument("--world-count", type=int, default=1024)
    parser.add_argument("--updates-per-collection", type=float, default=2.0)
    parser.add_argument("--replay-capacity", type=int, default=500_000)
    parser.add_argument("--replay-minimum", type=int, default=50_000)
    parser.add_argument("--save-interval", type=int, default=10_000_384)
    parser.add_argument("--save-replay", action="store_true", help="also dump the ~1 GiB replay at each gate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--orientation-weight", type=float, default=2.0)
    parser.add_argument(
        "--no-orientation",
        dest="orientation",
        action="store_false",
        help="control arm: upstream's unmodified twelve-term objective and curriculum",
    )
    parser.add_argument("--parquet", default=EPISODE_0_PARQUET)
    parser.add_argument("--image", default="flash_chord:latest")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and command, launch nothing")
    parser.add_argument(
        "--detach",
        action="store_true",
        help="run the container detached, so closing the shell or tmux cannot kill a six-hour job",
    )
    parser.add_argument("--force", action="store_true", help="proceed despite preflight problems")
    parser.add_argument(
        "--finalize",
        action="store_true",
        help="after a --detach run ends: hand artefacts back to the host user and close the manifest",
    )
    parser.add_argument("extra", nargs="*", help="extra Hydra overrides passed through verbatim")
    args = parser.parse_args()

    output_dir = STEP_D / args.tag
    if args.finalize:
        return finalize(args.tag, output_dir, args.image)
    plan = preflight(args, output_dir)
    command = docker_command(args, output_dir)

    print("=== orientation FlashSAC launch plan ===")
    print(f"  arm                  : {'orientation (13 terms)' if args.orientation else 'ORIGINAL (12 terms)'}")
    print(f"  output               : outputs/step_D/{args.tag}")
    print(f"  world_count          : {plan['world_count']:,}")
    print(f"  environment steps    : {plan['total_environment_steps']:,}")
    print(f"  collections          : {plan['collections']:,}")
    print(f"  optimizer updates    : {plan['total_optimizer_updates']:,}  (= LR horizon)")
    print(f"  replay capacity      : {plan['replay_capacity']:,}  (minimum {plan['replay_minimum_size']:,})")
    print(f"  checkpoints          : {plan['expected_checkpoints']} every {args.save_interval:,} steps")
    print(f"  checkpoint disk      : {gib(plan['checkpoint_bytes_expected']):.2f} GiB of {gib(plan['disk_free_bytes']):.1f} GiB free")
    print(f"  seed                 : {plan['seed']} (training and reset)")
    if args.orientation:
        print(f"  orientation weight   : {plan['object_orientation_weight']}")
    print()
    print("=== command ===")
    print(render_command(command))
    print()

    if plan["problems"]:
        print("=== PREFLIGHT PROBLEMS ===")
        for problem in plan["problems"]:
            print(f"  - {problem}")
        if not args.force:
            print("\nrefusing to launch; fix the above or pass --force")
            return 2
        print("\n--force given; continuing anyway")

    if args.dry_run:
        print("--dry-run: nothing launched")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    WARP_CACHE.mkdir(parents=True, exist_ok=True)
    manifest = output_dir / "launch_manifest.json"
    plan["command"] = command
    plan["started_unix"] = time.time()
    plan["started_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    plan["detached"] = bool(args.detach)
    manifest.write_text(json.dumps(plan, indent=2) + "\n")

    log_path = STEP_D / f"{args.tag}.log"

    if args.detach:
        # `docker run -d` outlives this process, the shell and tmux. The only
        # thing that stops it is an explicit `docker stop`.
        detached = [*command[:2], "-d", *command[2:]]
        started = subprocess.run(detached, capture_output=True, text=True)
        if started.returncode != 0:
            print("FAIL  could not start the container:")
            print(started.stderr.strip())
            return started.returncode
        container_id = started.stdout.strip()
        plan["container_id"] = container_id
        manifest.write_text(json.dumps(plan, indent=2) + "\n")
        print(f"started detached: {container_id[:12]}  (name v2d-train-{args.tag})")
        print()
        print("  follow it          : docker logs -f v2d-train-" + args.tag)
        print(f"  or mirror to a file: docker logs -f v2d-train-{args.tag} > outputs/step_D/{args.tag}.log 2>&1 &")
        print(f"  losses             : .venvs/episode0/bin/python scripts/show_training_metrics.py outputs/step_D/{args.tag} --watch 60")
        print("  still running?     : docker ps --filter name=v2d-train-" + args.tag)
        print()
        print(f"  when it finishes   : python3 {Path(__file__).name} --tag {args.tag} --finalize")
        return 0

    print(f"launching; log -> outputs/step_D/{args.tag}.log")
    print("  Ctrl-C here stops the run. To leave it running, detach tmux with Ctrl-b then d.")
    print()
    # Stream to the terminal AND the log file, so a tmux session shows progress
    # live without a second shell, and nothing is lost if the terminal dies.
    returncode = 0
    with log_path.open("w") as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            assert process.stdout is not None
            # The log keeps every line. The terminal keeps every metrics line
            # (one per log_interval collections) but throttles the per-collection
            # progress bar, which would otherwise print 341,796 lines over the
            # 350M run and bury the losses.
            last_progress = 0.0
            for line in process.stdout:
                log.write(line)
                now = time.time()
                is_progress = line.startswith("training progress:")
                if is_progress and now - last_progress < PROGRESS_ECHO_SECONDS:
                    continue
                if is_progress:
                    last_progress = now
                sys.stdout.write(line)
                sys.stdout.flush()
            returncode = process.wait()
        except KeyboardInterrupt:
            print("\ninterrupted; stopping the container (checkpoints already written are kept)")
            subprocess.run(["docker", "stop", f"v2d-train-{args.tag}"], stdout=subprocess.DEVNULL)
            returncode = process.wait()
    result = subprocess.CompletedProcess(command, returncode)

    # The image runs as root; hand artefacts back to the host user.
    subprocess.run(
        ["docker", "run", "--rm", "-v", f"{STEP_D}:/out", args.image, "chown", "-R", "1000:1000", "/out"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    plan["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    plan["elapsed_seconds"] = time.time() - plan["started_unix"]
    plan["returncode"] = result.returncode
    manifest.write_text(json.dumps(plan, indent=2) + "\n")

    if result.returncode != 0:
        print(f"FAIL  training exited {result.returncode}; tail of the log:")
        print("\n".join(log_path.read_text().splitlines()[-25:]))
        return result.returncode

    print(f"PASS  training completed in {plan['elapsed_seconds'] / 60:.1f} min")
    for checkpoint in sorted(output_dir.glob("policy_*.safetensors")):
        print(f"  {checkpoint.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
