#!/usr/bin/env python3
"""Run Step A + Step B over every episode of one challenge tier.

    .venvs/episode0/bin/python scripts/run_tier_pipeline.py <stage> [options]

Stages (each one is independently re-runnable and resumable):

    inventory    list the tier's episodes, assets and support decisions
    stage        Step A: adapt the Parquet + stage the three assets, then
                 validate the staged inputs against the pristine Parquet
    retarget     Step B: SOMA -> G1 in the robotic-grounding container, using
                 the memory-bounded chunked driver
    validate     check every produced reference against NVIDIA's demo reference
    flashchord   load every reference through FlashCHORD's own scene builder
    regress      re-run Episode 0 and diff it against the preserved baseline
    report       aggregate every record into report.json + report.md
    all          stage -> retarget -> validate -> flashchord -> report

Design notes
------------
**Resumable, not skip-happy.** Every stage writes one JSON record per episode
under ``outputs/<tier>/records/``. A stage skips an episode only when that
episode's record says it already succeeded *and* its output still exists;
``--force`` re-does it anyway. A failed episode is never skipped, never hidden,
and never silently dropped from the report - the next run retries it.

**One episode per container run.** It costs ~10 s of start-up per episode but
buys process isolation: a CUDA OOM or a Warp crash on one episode cannot take
the other 29 with it, and the peak-memory sample belongs to exactly one episode.

**Tier isolation.** Each tier stages into its own host tree and mounts it at its
own *container* path, so the absolute object-mesh path that ``soma_to_g1.py``
bakes into the parquet names the tier it came from. A Tier 2 reference can
therefore never resolve Tier 1's clean assets, and Tier 1 keeps the exact
``/work/adapter_out`` path Episode 0's existing artifacts already carry.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from pipeline.episodes import (  # noqa: E402
    TIER_1,
    EpisodeSpec,
    RegistryError,
    annotate_object_validity,
    load_tier,
    out_root_for_tier,
    select,
    sha256,
)

RETARGET_IMAGE = "robotic-grounding:latest"
FLASHCHORD_IMAGE = "flash_chord:latest"
SOMA_ASSETS = REPO / "data/body_models/soma"
DRIVER = REPO / "scripts/retarget_episode_g1.py"
FLASHCHORD_DRIVER = REPO / "scripts/check_reference_in_flashchord.py"
FLASHCHORD_ROOT = REPO / "official/video_to_data/robotic_grounding/flash_chord"
UPSTREAM_ASSETS = (
    REPO
    / "official/video_to_data/robotic_grounding/source/robotic_grounding"
    / "robotic_grounding/assets"
)
DEMO_REFERENCE = (
    UPSTREAM_ASSETS
    / "human_motion_data/whole_body/soma"
    / "sequence_id=2026-03-06_10-24-18_snack_box_pick_and_place_01/robot_name=g1/data.parquet"
)
#: Frames per SOMALayer forward pass. 64 is the measured setting from Step B
#: (3,411 MiB VRAM on a 778-frame sequence, flat in sequence length).
DEFAULT_CHUNK_FRAMES = 64

#: Episode 0 is the regression anchor: it is the one episode whose reference was
#: produced, inspected and documented by hand before this generalization.
REGRESSION_EPISODE = 0
#: Tolerance for the regression diff. The IK solve runs on the GPU, so bitwise
#: equality across runs is not guaranteed; a real regression moves things by
#: centimetres, not by 1e-5 m.
REGRESSION_TOLERANCE_M = 1e-3


# ---------------------------------------------------------------------------
# records
# ---------------------------------------------------------------------------
def records_dir(tier: str) -> Path:
    return REPO / "outputs" / tier / "records"


def logs_dir(tier: str) -> Path:
    return REPO / "outputs" / tier / "logs"


def record_path(tier: str, stage: str, episode_index: int) -> Path:
    return records_dir(tier) / f"{episode_index:06d}_{stage}.json"


def write_record(tier: str, stage: str, episode_index: int, payload: dict) -> Path:
    path = record_path(tier, stage, episode_index)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    payload.setdefault("stage", stage)
    payload.setdefault("episode_index", episode_index)
    payload.setdefault("tier", tier)
    payload["recorded_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    # Write-then-rename so an interrupted run never leaves a half-written record
    # that a later resume would read as authoritative.
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temp.replace(path)
    return path


def read_record(tier: str, stage: str, episode_index: int) -> dict | None:
    path = record_path(tier, stage, episode_index)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# resource sampling
# ---------------------------------------------------------------------------
@dataclass
class ResourceSample:
    """Peak memory observed while one container ran."""

    gpu_baseline_mib: int
    gpu_peak_mib: int
    container_peak_mib: float
    samples: int

    @property
    def gpu_delta_mib(self) -> int:
        """Peak minus the pre-run baseline: this run's own GPU footprint.

        ``nvidia-smi`` reports device-wide usage and the desktop session holds
        ~1.5 GiB of it, so the delta is the honest attribution. It is an upper
        bound on our use and a lower bound on nothing - stated, not inferred.
        """
        return max(0, self.gpu_peak_mib - self.gpu_baseline_mib)

    def as_dict(self) -> dict:
        return {
            "gpu_baseline_mib": self.gpu_baseline_mib,
            "gpu_peak_mib": self.gpu_peak_mib,
            "gpu_delta_mib": self.gpu_delta_mib,
            "container_peak_mib": round(self.container_peak_mib, 1),
            "samples": self.samples,
        }


def _gpu_used_mib() -> int | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip().splitlines()
        return int(out[0].strip())
    except Exception:  # noqa: BLE001 - sampling must never break a run
        return None


_MEM_UNITS = {"B": 1 / 2**20, "KIB": 1 / 1024, "MIB": 1.0, "GIB": 1024.0,
              "KB": 1 / 1024, "MB": 1.0, "GB": 1024.0}


def _container_mem_mib(name: str) -> float | None:
    try:
        out = subprocess.run(
            ["docker", "stats", "--no-stream", "--format", "{{.MemUsage}}", name],
            capture_output=True, text=True, timeout=15,
        )
        if out.returncode != 0:
            return None
        text = out.stdout.strip().split("/")[0].strip()
        match = re.match(r"([0-9.]+)\s*([A-Za-z]+)", text)
        if not match:
            return None
        return float(match.group(1)) * _MEM_UNITS.get(match.group(2).upper(), 1.0)
    except Exception:  # noqa: BLE001
        return None


class ResourceMonitor:
    """Polls GPU and container memory in the background while a run proceeds."""

    def __init__(self, container_name: str, interval: float = 2.0) -> None:
        self.container_name = container_name
        self.interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.gpu_baseline = _gpu_used_mib() or 0
        self.gpu_peak = self.gpu_baseline
        self.container_peak = 0.0
        self.samples = 0

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            gpu = _gpu_used_mib()
            if gpu is not None:
                self.gpu_peak = max(self.gpu_peak, gpu)
            mem = _container_mem_mib(self.container_name)
            if mem is not None:
                self.container_peak = max(self.container_peak, mem)
            self.samples += 1

    def __enter__(self) -> "ResourceMonitor":
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval * 2)

    def result(self) -> ResourceSample:
        return ResourceSample(
            gpu_baseline_mib=self.gpu_baseline,
            gpu_peak_mib=self.gpu_peak,
            container_peak_mib=self.container_peak,
            samples=self.samples,
        )


# ---------------------------------------------------------------------------
# tier loading
# ---------------------------------------------------------------------------
def tier_specs(tier: str, selector: str, *, probe: bool = True) -> list[EpisodeSpec]:
    """Load the tier registry and attach each episode's object validity.

    ``probe=False`` skips reading the Parquets, for stages that only need paths.
    """
    specs = select(load_tier(tier), selector)
    if not probe:
        return specs
    import numpy as np
    import pyarrow.parquet as pq

    probed = []
    for spec in specs:
        table = pq.read_table(spec.parquet, columns=["observation.object.visible"])
        visible = table["observation.object.visible"].to_numpy().astype(bool)
        invalid = [int(i) for i in np.flatnonzero(~visible)]
        annotated = annotate_object_validity(spec, int(visible.sum()), invalid)
        annotated.validate_frame_policy()
        probed.append(annotated)
    return probed


def container_root(spec: EpisodeSpec) -> str:
    """Mount point for this tier's staged tree inside both containers.

    Tier-specific on purpose: ``soma_to_g1.py`` bakes this absolute path into
    the parquet's ``object_mesh_paths``, so the path itself has to identify the
    tier. Tier 1 resolves to ``/work/adapter_out``, which is exactly what
    Episode 0's existing reference already records.
    """
    return f"/work/{spec.out_root.name}"


# ---------------------------------------------------------------------------
# stage: inventory
# ---------------------------------------------------------------------------
def stage_inventory(args) -> int:
    specs = tier_specs(args.tier, args.episodes)
    rows = []
    print(f"=== {args.tier}: {len(specs)} episode(s) ===")
    print(f"    staged tree : {out_root_for_tier(args.tier)}")
    print(f"    mounted at  : {container_root(specs[0])} in both containers")
    print()
    header = (
        f"{'idx':>3} {'sequence_id':58s} {'object':22s} {'src':>5} {'ref':>5} "
        f"{'shapes':>6} {'support':>7} notes"
    )
    print(header)
    print("-" * len(header))
    for spec in specs:
        notes = []
        if spec.dropped_frames:
            notes.append(
                f"drops frame(s) {list(spec.dropped_frames)}: object not visible"
            )
        print(
            f"{spec.episode_index:3d} {spec.sequence_id:58s} {spec.object_name:22s} "
            f"{spec.frames:5d} {spec.retarget_frames:5d} {spec.support_shape_count:6d} "
            f"{str(spec.scene_support).lower():7s} {'; '.join(notes)}"
        )
        rows.append(spec.provenance())
    total_src = sum(s.frames for s in specs)
    total_ref = sum(s.retarget_frames for s in specs)
    with_support = sum(1 for s in specs if s.scene_support)
    print()
    print(f"  total source frames    : {total_src}")
    print(f"  total reference frames : {total_ref}")
    print(f"  scene.support=true     : {with_support} / {len(specs)}")
    print(f"  scene.support=false    : {len(specs) - with_support} / {len(specs)}")

    out = REPO / "outputs" / args.tier / "inventory.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"tier": args.tier, "episodes": rows}, indent=2) + "\n")
    print(f"\n  wrote {out.relative_to(REPO)}")
    return 0


# ---------------------------------------------------------------------------
# stage: stage (Step A)
# ---------------------------------------------------------------------------
def _staged_is_current(spec: EpisodeSpec) -> bool:
    """True when the staged inputs exist and match the pristine sources.

    Hash-compared rather than existence-checked, so a changed upstream file or a
    truncated copy re-stages instead of being trusted.
    """
    if not spec.manifest.is_file():
        return False
    for path in (
        spec.data_folder / "soma_params.npz",
        spec.data_folder / "poses.npy",
        spec.staged_mesh,
        spec.staged_ground_plane,
        spec.staged_support,
    ):
        if not path.is_file():
            return False
    try:
        manifest = json.loads(spec.manifest.read_text())
    except json.JSONDecodeError:
        return False
    if manifest.get("tier") != spec.tier:
        return False
    if manifest.get("retarget_frames") != spec.retarget_frames:
        return False
    recorded = manifest.get("source_sha256", {})
    for role, path in (
        ("parquet", spec.parquet),
        ("mesh", spec.mesh),
        ("ground_plane", spec.ground_plane),
        ("support_surface", spec.support_surface),
    ):
        if recorded.get(role) != sha256(path):
            return False
    # The staged copies must still be the bytes we copied.
    for source, staged in (
        (spec.mesh, spec.staged_mesh),
        (spec.ground_plane, spec.staged_ground_plane),
        (spec.support_surface, spec.staged_support),
    ):
        if sha256(source) != sha256(staged):
            return False
    return True


def _copy_verbatim(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    if sha256(src) != sha256(dst):
        raise RuntimeError(f"copy changed bytes: {src} -> {dst}")


def stage_stage(args) -> int:
    import numpy as np
    import pyarrow.parquet as pq

    from pipeline.lerobot_to_soma import load_episode, write_outputs
    from pipeline.validate_staged import validate_staged_episode

    specs = tier_specs(args.tier, args.episodes)
    failures = 0
    for spec in specs:
        print()
        print("#" * 78)
        print(f"# stage  episode {spec.episode_index:3d}  {spec.sequence_id}")
        print("#" * 78)
        started = time.time()
        reused = False
        try:
            if _staged_is_current(spec) and not args.force:
                print("  staged inputs already match the pristine sources; reusing")
                reused = True
            else:
                # Step A: Parquet -> soma_params.npz + poses.npy, verbatim.
                episode = load_episode(spec.tier_root, spec.episode_index)
                if episode.sequence_id != spec.sequence_id:
                    raise RuntimeError(
                        f"Parquet sequence_id {episode.sequence_id!r} != metadata "
                        f"{spec.sequence_id!r}"
                    )
                on_invalid = "passthrough" if spec.invalid_object_frames else "fail"
                written = write_outputs(episode, spec.data_folder, on_invalid=on_invalid)
                print(f"  adapter wrote {len(written)} file(s), on_invalid={on_invalid}")
                # The three official assets, byte-for-byte.
                _copy_verbatim(spec.mesh, spec.staged_mesh)
                _copy_verbatim(spec.ground_plane, spec.staged_ground_plane)
                _copy_verbatim(spec.support_surface, spec.staged_support)
                print("  staged mesh, ground_plane and support surface verbatim")

                table = pq.read_table(spec.parquet, columns=["timestamp"])
                timestamps = table["timestamp"].to_numpy().astype(np.float64)
                manifest = spec.provenance()
                manifest["timestamps"] = {
                    "first_s": float(timestamps[0]),
                    "last_s": float(timestamps[-1]),
                    "duration_s": float(timestamps[-1] - timestamps[0]),
                    "max_step_error_s": float(
                        np.max(np.abs(np.diff(timestamps) - 1.0 / spec.fps))
                    ),
                }
                manifest["staged_paths"] = {
                    "data_folder": str(spec.data_folder),
                    "support_usda": str(spec.staged_support),
                    "container_root": container_root(spec),
                }
                spec.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
                print(f"  wrote {spec.manifest.relative_to(REPO)}")

            print()
            checker = validate_staged_episode(spec)
            ok = checker.failed == 0
            payload = {
                "ok": ok,
                "reused": reused,
                "seconds": round(time.time() - started, 2),
                "validation": checker.as_dict(),
                "provenance": spec.provenance(),
            }
        except Exception as error:  # noqa: BLE001 - one episode must not stop the tier
            print(f"  ERROR: {type(error).__name__}: {error}")
            ok = False
            payload = {
                "ok": False,
                "reused": reused,
                "seconds": round(time.time() - started, 2),
                "error": f"{type(error).__name__}: {error}",
            }
        write_record(args.tier, "stage", spec.episode_index, payload)
        if not ok:
            failures += 1

    print()
    print(f"=== stage: {len(specs) - failures}/{len(specs)} episode(s) OK ===")
    return 1 if failures else 0


# ---------------------------------------------------------------------------
# stage: retarget (Step B)
# ---------------------------------------------------------------------------
def _retarget_command(spec: EpisodeSpec, chunk_frames: int, container_name: str) -> list[str]:
    root = container_root(spec)
    inner = [
        "python /driver/retarget_episode_g1.py",
        f"--chunk-frames {chunk_frames}",
        "--",
        f"{root}/whole_body/soma/{spec.sequence_id}",
        "--save",
        "--soma-data-root /soma",
        "--robot-name g1",
        f"--motion-root {root}",
        "--soma-subdir soma",
        f"--fps {spec.fps}",
    ]
    if spec.retarget_end_frame is not None:
        # Declining to emit a reference frame for an unmeasured object pose.
        # Frame-0 anchoring is unaffected: soma_to_g1 slices after anchoring.
        inner.append(f"--end-frame {spec.retarget_end_frame}")
    # chown runs in the SAME container invocation so the host keeps ownership;
    # the image's python wrapper breaks under `docker --user` (see CLAUDE.md).
    script = (
        " ".join(inner)
        + f" ; rc=$? ; chown -R {os.getuid()}:{os.getgid()} {root} ; exit $rc"
    )
    return [
        "docker", "run", "--rm", "--gpus", "all", "--name", container_name,
        "-v", f"{spec.out_root}:{root}",
        "-v", f"{SOMA_ASSETS}:/soma:ro",
        "-v", f"{DRIVER}:/driver/retarget_episode_g1.py:ro",
        "--entrypoint", "bash", RETARGET_IMAGE, "-lc", script,
    ]


def _run_logged(command: list[str], log: Path, container_name: str, timeout: int) -> tuple[int, str, ResourceSample]:
    """Run a container, tee its output to ``log``, and sample peak memory."""
    log.parent.mkdir(parents=True, exist_ok=True)
    with ResourceMonitor(container_name) as monitor:
        with log.open("w") as handle:
            handle.write("$ " + " ".join(command) + "\n\n")
            handle.flush()
            try:
                proc = subprocess.run(
                    command, stdout=handle, stderr=subprocess.STDOUT, timeout=timeout
                )
                code = proc.returncode
            except subprocess.TimeoutExpired:
                handle.write(f"\n[orchestrator] TIMEOUT after {timeout} s\n")
                code = 124
                subprocess.run(["docker", "kill", container_name],
                               capture_output=True, text=True)
    return code, log.read_text(), monitor.result()


def stage_retarget(args) -> int:
    specs = tier_specs(args.tier, args.episodes)
    failures: list[int] = []
    for spec in specs:
        print()
        print("#" * 78)
        print(f"# retarget  episode {spec.episode_index:3d}  {spec.sequence_id} "
              f"({spec.retarget_frames} frames)")
        print("#" * 78)

        previous = read_record(args.tier, "retarget", spec.episode_index)
        if previous is None:
            # No record means this reference, if one exists, was produced before
            # the orchestrator existed - Episode 0's hand-built Step B output.
            # Keep a copy before anything can overwrite it. Once an episode has
            # a record its references are ours, so no further copies are made.
            kept = preserve_baseline(spec)
            if kept is not None:
                print(f"  pre-existing reference preserved at {kept.relative_to(REPO)}")

        if (
            not args.force
            and previous is not None
            and previous.get("ok")
            and spec.reference_parquet.is_file()
        ):
            print(f"  already retargeted in {previous.get('seconds')} s; reusing "
                  f"({spec.reference_parquet.relative_to(REPO)})")
            continue

        staged = read_record(args.tier, "stage", spec.episode_index)
        if staged is None or not staged.get("ok"):
            print("  SKIPPED: staged inputs did not pass validation. Run the "
                  "'stage' stage for this episode first.")
            write_record(args.tier, "retarget", spec.episode_index, {
                "ok": False,
                "error": "staged inputs missing or invalid; retarget not attempted",
            })
            failures.append(spec.episode_index)
            continue

        container_name = f"v2d-retarget-{args.tier[:6]}-{spec.episode_index:03d}-{os.getpid()}"
        log = logs_dir(args.tier) / f"{spec.episode_index:06d}_retarget.log"
        command = _retarget_command(spec, args.chunk_frames, container_name)
        started = time.time()
        code, text, resources = _run_logged(command, log, container_name, args.timeout)
        elapsed = time.time() - started

        processed = re.search(r"Processed (\d+) frames", text)
        oom = bool(re.search(r"out of memory|Failed to allocate", text, re.IGNORECASE))
        ground = "grounding plane: reconstructed" in text
        frames_ok = processed is not None and int(processed.group(1)) == spec.retarget_frames
        parquet_ok = spec.reference_parquet.is_file()
        ok = code == 0 and frames_ok and parquet_ok and not oom

        print(f"  exit {code} in {elapsed:.1f} s")
        print(f"  frames processed       : "
              f"{processed.group(1) if processed else 'n/a'} (expected {spec.retarget_frames})")
        print(f"  reconstructed plane    : {ground}")
        print(f"  CUDA OOM in the log    : {oom}")
        print(f"  peak GPU (device total): {resources.gpu_peak_mib} MiB "
              f"(baseline {resources.gpu_baseline_mib}, delta {resources.gpu_delta_mib})")
        print(f"  peak container RAM     : {resources.container_peak_mib:.0f} MiB")
        print(f"  reference parquet      : {parquet_ok}")
        print(f"  log                    : {log.relative_to(REPO)}")
        if not ok:
            tail = [line for line in text.splitlines() if line.strip()][-12:]
            print("  --- last log lines ---")
            for line in tail:
                print(f"  | {line}")

        write_record(args.tier, "retarget", spec.episode_index, {
            "ok": ok,
            "exit_code": code,
            "seconds": round(elapsed, 2),
            "frames_expected": spec.retarget_frames,
            "frames_processed": int(processed.group(1)) if processed else None,
            "ground_plane_reconstructed": ground,
            "cuda_oom": oom,
            "chunk_frames": args.chunk_frames,
            "end_frame": spec.retarget_end_frame,
            "resources": resources.as_dict(),
            "log": str(log.relative_to(REPO)),
            "reference_parquet": str(spec.reference_parquet),
            "reference_parquet_exists": parquet_ok,
        })
        if not ok:
            failures.append(spec.episode_index)

    print()
    print(f"=== retarget: {len(specs) - len(failures)}/{len(specs)} episode(s) OK ===")
    if failures:
        print(f"    failed episodes: {failures}")
    return 1 if failures else 0


# ---------------------------------------------------------------------------
# stage: validate
# ---------------------------------------------------------------------------
def stage_validate(args) -> int:
    from pipeline.validate_reference import run_checks

    specs = tier_specs(args.tier, args.episodes)
    reference = DEMO_REFERENCE if DEMO_REFERENCE.is_file() else None
    if reference is None:
        print(f"  NOTE  NVIDIA demo reference not found at {DEMO_REFERENCE};"
              " the IK-residual and schema comparisons will be skipped")
    failures: list[int] = []
    for spec in specs:
        print()
        print("#" * 78)
        print(f"# validate  episode {spec.episode_index:3d}  {spec.sequence_id}")
        print("#" * 78)
        if not spec.reference_parquet.is_file():
            print(f"  FAIL  no reference parquet at {spec.reference_parquet}")
            write_record(args.tier, "validate", spec.episode_index, {
                "ok": False, "error": "reference parquet missing",
            })
            failures.append(spec.episode_index)
            continue
        try:
            report = run_checks(
                spec.reference_parquet,
                frames=spec.retarget_frames,
                reference=reference,
                mesh_root=spec.object_dir,
                dropped_frames=spec.dropped_frames,
                # One per-episode "ALL CHECKS PASSED" banner per episode reads
                # like a verdict on the tier when 30 of them scroll past. The
                # episode-scoped line below replaces it; the tier verdict is
                # the "=== validate: N/30 ===" line at the end.
                summary=False,
            )
            payload = report.as_dict()
            ok = report.ok
            print()
            if ok:
                print("  episode {:d}: all {:d} checks passed".format(
                    spec.episode_index, report.checker.passed))
            else:
                print("  episode {:d}: {:d} of {:d} CHECK(S) FAILED".format(
                    spec.episode_index, report.checker.failed,
                    len(report.checker.results)))
        except Exception as error:  # noqa: BLE001
            print(f"  ERROR: {type(error).__name__}: {error}")
            ok = False
            payload = {"ok": False, "error": f"{type(error).__name__}: {error}"}
        write_record(args.tier, "validate", spec.episode_index, payload)
        if not ok:
            failures.append(spec.episode_index)

    print()
    print(f"=== validate: {len(specs) - len(failures)}/{len(specs)} episode(s) OK ===")
    if failures:
        print(f"    failed episodes: {failures}")
    return 1 if failures else 0


# ---------------------------------------------------------------------------
# stage: flashchord
# ---------------------------------------------------------------------------
def _flashchord_command(spec: EpisodeSpec, container_name: str) -> list[str]:
    root = container_root(spec)
    parquet = f"{root}/whole_body/soma/sequence_id={spec.sequence_id}/robot_name=g1/data.parquet"
    return [
        "docker", "run", "--rm", "--gpus", "all", "--name", container_name,
        "-v", f"{FLASHCHORD_ROOT}:/workspace",
        "-v", f"{UPSTREAM_ASSETS}:/v2d/assets:ro",
        "-v", f"{spec.out_root}:{root}:ro",
        "-v", f"{FLASHCHORD_DRIVER}:/driver/check_reference_in_flashchord.py:ro",
        "-e", "XLA_PYTHON_CLIENT_PREALLOCATE=false",
        "-w", "/workspace", FLASHCHORD_IMAGE,
        "python", "/driver/check_reference_in_flashchord.py",
        "--parquet", parquet,
        "--expect-frames", str(spec.retarget_frames),
        # Cross-check the host's text scan against a real USD parse. A
        # disagreement means our scene.support decision is unsound, so it fails
        # rather than being quietly reconciled.
        "--expect-support-shapes", str(spec.support_shape_count),
    ]


def stage_flashchord(args) -> int:
    specs = tier_specs(args.tier, args.episodes)
    failures: list[int] = []
    for spec in specs:
        print()
        print("#" * 78)
        print(f"# flashchord  episode {spec.episode_index:3d}  {spec.sequence_id}")
        print("#" * 78)
        previous = read_record(args.tier, "flashchord", spec.episode_index)
        if not args.force and previous is not None and previous.get("ok"):
            print(f"  already passed in {previous.get('seconds')} s; reusing")
            continue
        if not spec.reference_parquet.is_file():
            print("  SKIPPED: no reference parquet to load")
            write_record(args.tier, "flashchord", spec.episode_index, {
                "ok": False, "error": "reference parquet missing",
            })
            failures.append(spec.episode_index)
            continue

        container_name = f"v2d-fc-{args.tier[:6]}-{spec.episode_index:03d}-{os.getpid()}"
        log = logs_dir(args.tier) / f"{spec.episode_index:06d}_flashchord.log"
        started = time.time()
        code, text, resources = _run_logged(
            _flashchord_command(spec, container_name), log, container_name, args.timeout
        )
        elapsed = time.time() - started

        passed = "FLASHCHORD_LOAD=PASS" in text
        support = re.search(r"INCLUDE_SUPPORT=(true|false)", text)
        shapes = re.search(r"SUPPORT_SHAPES=(-?\d+)", text)
        bodies = re.search(r"model bodies\s*:\s*(\d+)", text)
        object_err = re.search(r"object pos err\s*:\s*([0-9.eE+-]+)", text)
        wrist_err = re.search(r"wrist pos err\s*:\s*([0-9.eE+-]+)", text)
        usd_shapes = int(shapes.group(1)) if shapes else None
        include_support = support.group(1) == "true" if support else None
        support_agrees = (
            usd_shapes is not None and usd_shapes == spec.support_shape_count
            and include_support == spec.scene_support
        )
        ok = code == 0 and passed and support_agrees

        print(f"  exit {code} in {elapsed:.1f} s")
        print(f"  FLASHCHORD_LOAD=PASS   : {passed}")
        print(f"  collision shapes (USD) : {usd_shapes} "
              f"(host text scan: {spec.support_shape_count})")
        print(f"  scene.support          : {include_support} "
              f"(registry: {str(spec.scene_support).lower()})")
        print(f"  model bodies           : {bodies.group(1) if bodies else 'n/a'}")
        print(f"  frame-0 object/wrist err: "
              f"{object_err.group(1) if object_err else 'n/a'} / "
              f"{wrist_err.group(1) if wrist_err else 'n/a'} m")
        print(f"  peak GPU delta         : {resources.gpu_delta_mib} MiB")
        print(f"  log                    : {log.relative_to(REPO)}")
        if not ok:
            tail = [line for line in text.splitlines() if line.strip()][-12:]
            print("  --- last log lines ---")
            for line in tail:
                print(f"  | {line}")

        write_record(args.tier, "flashchord", spec.episode_index, {
            "ok": ok,
            "exit_code": code,
            "seconds": round(elapsed, 2),
            "load_pass": passed,
            "usd_collision_shapes": usd_shapes,
            "host_collision_shapes": spec.support_shape_count,
            "include_support": include_support,
            "registry_scene_support": spec.scene_support,
            "support_decision_agrees": support_agrees,
            "model_bodies": int(bodies.group(1)) if bodies else None,
            "frame0_object_pos_err_m": float(object_err.group(1)) if object_err else None,
            "frame0_wrist_pos_err_m": float(wrist_err.group(1)) if wrist_err else None,
            "resources": resources.as_dict(),
            "log": str(log.relative_to(REPO)),
        })
        if not ok:
            failures.append(spec.episode_index)

    print()
    print(f"=== flashchord: {len(specs) - len(failures)}/{len(specs)} episode(s) OK ===")
    if failures:
        print(f"    failed episodes: {failures}")
    return 1 if failures else 0


# ---------------------------------------------------------------------------
# stage: regress (Episode 0)
# ---------------------------------------------------------------------------
def _baseline_path(tier: str, episode_index: int = REGRESSION_EPISODE) -> Path:
    return (
        REPO / "outputs" / tier / "regression"
        / f"episode_{episode_index:06d}_baseline.parquet"
    )


def preserve_baseline(spec: EpisodeSpec) -> Path | None:
    """Copy an existing reference aside before anything can overwrite it.

    Episode 0's reference was produced and documented by hand before this
    orchestrator existed, and the brief is to preserve previous outputs. Rather
    than special-casing Episode 0, any reference that already exists when the
    orchestrator first touches it is copied to ``outputs/<tier>/regression/``
    once, and never overwritten after that - so the ``regress`` stage always has
    a genuine pre-generalization artifact to diff against.
    """
    if not spec.reference_parquet.is_file():
        return None
    baseline = _baseline_path(spec.tier, spec.episode_index)
    if baseline.is_file():
        return baseline
    baseline.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(spec.reference_parquet, baseline)
    return baseline


def stage_regress(args) -> int:
    """Re-run Episode 0 and diff the new reference against the preserved one.

    The point is not that the file is identical - the IK solve runs on the GPU -
    but that generalizing the pipeline did not move Episode 0's reference. The
    pre-existing parquet is copied to a baseline on first run and never
    overwritten, so the comparison always has something to compare against.
    """
    import numpy as np

    from pipeline.validate_reference import load_row

    specs = [s for s in tier_specs(args.tier, str(REGRESSION_EPISODE))]
    if not specs:
        print(f"  episode {REGRESSION_EPISODE} is not in {args.tier}")
        return 1
    spec = specs[0]
    baseline = _baseline_path(args.tier)

    if not baseline.is_file():
        if preserve_baseline(spec) is None:
            print(f"  FAIL  no baseline and no current reference at "
                  f"{spec.reference_parquet}")
            return 1
        print("  preserved the pre-existing Episode 0 reference as the baseline:")
    print(f"  baseline: {baseline.relative_to(REPO)}  sha256 {sha256(baseline)[:16]}...")

    print()
    if getattr(args, "no_rerun", False):
        # The batch retarget already produced Episode 0's reference through the
        # generalized pipeline, and ``preserve_baseline`` copied the hand-built
        # Step B artifact aside before that happened. So the comparison this
        # stage exists to make is already on disk, and re-solving the IK would
        # only overwrite a reference we were asked to preserve.
        if not spec.reference_parquet.is_file():
            print(f"  FAIL  --no-rerun needs an existing reference at "
                  f"{spec.reference_parquet}")
            return 1
        retarget_record = read_record(args.tier, "retarget", REGRESSION_EPISODE)
        print("  --no-rerun: diffing the reference already on disk; the retarget "
              "is NOT re-run and nothing is overwritten")
        print("  that reference was produced by this pipeline at {}".format(
            (retarget_record or {}).get("recorded_at", "an unrecorded time")))
    else:
        print("  re-running the retarget for Episode 0 through the generalized "
              "pipeline")
        forced = argparse.Namespace(**vars(args))
        forced.force = True
        forced.episodes = str(REGRESSION_EPISODE)
        code = stage_retarget(forced)
        if code != 0:
            print("  FAIL  the regression retarget did not succeed")
            return 1

    print()
    print("=== Episode 0 regression diff: re-run vs preserved baseline ===")
    old = load_row(baseline)
    new = load_row(spec.reference_parquet)
    failures = 0
    for column in (
        "schema_version", "motion_kind", "coord_frame", "fps",
        "robot_joint_names", "ee_link_names", "hand_sides", "object_body_names",
    ):
        same = old[column] == new[column]
        print(("  PASS  " if same else "  FAIL  ") + f"{column} unchanged")
        failures += 0 if same else 1

    numeric = (
        "robot_root_position", "robot_root_wxyz", "robot_joint_positions",
        "object_root_position", "object_root_axis_angle",
        "object_body_position", "object_body_wxyz", "ee_pose_w",
        "ik_error_per_frame", "frame_task_errors",
    )
    worst = 0.0
    for column in numeric:
        a = np.asarray(old[column], dtype=np.float64)
        b = np.asarray(new[column], dtype=np.float64)
        if a.shape != b.shape:
            print(f"  FAIL  {column:26s} shape {a.shape} -> {b.shape}")
            failures += 1
            continue
        delta = float(np.max(np.abs(a - b))) if a.size else 0.0
        worst = max(worst, delta)
        ok = delta <= REGRESSION_TOLERANCE_M
        print(("  PASS  " if ok else "  FAIL  ")
              + f"{column:26s} max |delta| = {delta:.3e}")
        failures += 0 if ok else 1
    print(f"  ..    worst delta over every numeric column: {worst:.3e} "
          f"(tolerance {REGRESSION_TOLERANCE_M:.0e})")
    bitwise = sha256(baseline) == sha256(spec.reference_parquet)
    print(f"  ..    byte-identical parquet: {bitwise}")

    write_record(args.tier, "regress", REGRESSION_EPISODE, {
        "ok": failures == 0,
        "baseline": str(baseline.relative_to(REPO)),
        "baseline_sha256": sha256(baseline),
        "rerun_sha256": sha256(spec.reference_parquet),
        "byte_identical": bitwise,
        "worst_abs_delta": worst,
        "tolerance": REGRESSION_TOLERANCE_M,
        "failed_columns": failures,
        "rerun": not getattr(args, "no_rerun", False),
    })
    print()
    print("ALL CHECKS PASSED" if failures == 0 else f"{failures} CHECK(S) FAILED")
    return 1 if failures else 0


# ---------------------------------------------------------------------------
# stage: report
# ---------------------------------------------------------------------------
def stage_report(args) -> int:
    specs = tier_specs(args.tier, args.episodes)
    stages = ("stage", "retarget", "validate", "flashchord")
    episodes: list[dict] = []
    for spec in specs:
        entry = {
            "episode_index": spec.episode_index,
            "sequence_id": spec.sequence_id,
            "object": spec.object_name,
            "task": spec.task,
            "source_frames": spec.frames,
            "reference_frames": spec.retarget_frames,
            "dropped_frames": list(spec.dropped_frames),
            "support_shape_count": spec.support_shape_count,
            "scene_support": spec.scene_support,
            "reference_parquet": str(spec.reference_parquet),
            "reference_exists": spec.reference_parquet.is_file(),
        }
        for name in stages:
            entry[name] = read_record(args.tier, name, spec.episode_index)
        entry["ok"] = all(
            (entry[name] or {}).get("ok") is True for name in stages
        )
        episodes.append(entry)

    passing = [e for e in episodes if e["ok"]]
    failing = [e for e in episodes if not e["ok"]]

    def total_seconds(name: str) -> float:
        return sum(float((e[name] or {}).get("seconds") or 0.0) for e in episodes)

    def peak(name: str, key: str) -> int:
        values = [
            int(((e[name] or {}).get("resources") or {}).get(key) or 0)
            for e in episodes
        ]
        return max(values) if values else 0

    summary = {
        "tier": args.tier,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "episodes_total": len(episodes),
        "episodes_passing": len(passing),
        "episodes_failing": len(failing),
        "failing_indices": [e["episode_index"] for e in failing],
        "source_frames_total": sum(e["source_frames"] for e in episodes),
        "reference_frames_total": sum(
            e["reference_frames"] for e in episodes if e["reference_exists"]
        ),
        "scene_support_true": sum(1 for e in episodes if e["scene_support"]),
        "scene_support_false": sum(1 for e in episodes if not e["scene_support"]),
        "seconds": {name: round(total_seconds(name), 1) for name in stages},
        "retarget_peak_gpu_delta_mib": peak("retarget", "gpu_delta_mib"),
        "retarget_peak_container_mib": peak("retarget", "container_peak_mib"),
        "flashchord_peak_gpu_delta_mib": peak("flashchord", "gpu_delta_mib"),
        "regression": read_record(args.tier, "regress", REGRESSION_EPISODE),
        "episodes": episodes,
    }

    out_dir = REPO / "outputs" / args.tier
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(summary, indent=2) + "\n")
    _write_reference_index(out_dir / "references_index.json", args.tier, specs, episodes)
    markdown = _render_markdown(summary)
    (out_dir / "report.md").write_text(markdown)
    print(markdown)
    for name in ("report.json", "report.md", "references_index.json"):
        print(f"  wrote {(out_dir / name).relative_to(REPO)}")
    return 1 if failing else 0


def _write_reference_index(
    path: Path, tier: str, specs: list[EpisodeSpec], episodes: list[dict]
) -> None:
    """Emit the per-episode facts a training sweep needs, and nothing else.

    The next research step (FlashSAC, BC initialization, cross-episode transfer)
    has to launch one run per episode and needs, per episode: the reference
    parquet, its frame count, the Hydra ``scene.support`` value, and enough
    identity to label the run. Deriving those from the raw dataset again - or
    worse, hard-coding them - is how the per-episode support flag gets wrong.
    This file is the contract, with only episodes whose reference actually
    passed every gate marked usable.
    """
    by_index = {entry["episode_index"]: entry for entry in episodes}
    payload = {
        "tier": tier,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "note": (
            "Produced by scripts/run_tier_pipeline.py report. 'usable' means "
            "stage+retarget+validate+flashchord all passed for that episode. "
            "Train on this tier's own references only: tiers are scored "
            "separately and a Tier N policy must not see Tier M motion."
        ),
        "episodes": [
            {
                "episode_index": spec.episode_index,
                "sequence_id": spec.sequence_id,
                "object": spec.object_name,
                "task": spec.task,
                "reference_parquet": str(spec.reference_parquet),
                "object_dir": str(spec.object_dir),
                "container_root": container_root(spec),
                "frames": spec.retarget_frames,
                "source_frames": spec.frames,
                "fps": spec.fps,
                "scene_support": spec.scene_support,
                "support_shape_count": spec.support_shape_count,
                "dropped_frames": list(spec.dropped_frames),
                "usable": bool(by_index.get(spec.episode_index, {}).get("ok")),
            }
            for spec in specs
        ],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _render_markdown(summary: dict) -> str:
    lines: list[str] = []
    add = lines.append
    add(f"# Track 2 G1 references - {summary['tier']}")
    add("")
    add(f"Generated {summary['generated_at']} by `scripts/run_tier_pipeline.py report`.")
    add("")
    add(f"- episodes: **{summary['episodes_passing']} / {summary['episodes_total']} "
        f"fully validated**")
    if summary["failing_indices"]:
        add(f"- failing: `{summary['failing_indices']}`")
    add(f"- reference frames produced: {summary['reference_frames_total']} "
        f"of {summary['source_frames_total']} source frames")
    add(f"- `scene.support=true`: {summary['scene_support_true']}, "
        f"`false`: {summary['scene_support_false']}")
    seconds = summary["seconds"]
    add(f"- wall clock: stage {seconds['stage']:.0f} s, retarget "
        f"{seconds['retarget']:.0f} s, validate {seconds['validate']:.0f} s, "
        f"flashchord {seconds['flashchord']:.0f} s")
    add(f"- peak memory: retarget {summary['retarget_peak_gpu_delta_mib']} MiB GPU / "
        f"{summary['retarget_peak_container_mib']:.0f} MiB container RAM; "
        f"flashchord {summary['flashchord_peak_gpu_delta_mib']} MiB GPU")
    regression = summary.get("regression")
    if regression:
        add(f"- Episode 0 regression: "
            f"{'PASS' if regression.get('ok') else 'FAIL'}, worst numeric delta "
            f"{regression.get('worst_abs_delta', float('nan')):.2e} "
            f"(tolerance {regression.get('tolerance')})")
    add("")
    add("| idx | sequence | obj frames | ref frames | support | stage | retarget | "
        "validate | flashchord | retarget s | GPU MiB | IK med | IK ratio | root XY max |")
    add("|----:|---|----:|----:|---|---|---|---|---|----:|----:|----:|----:|----:|")
    for entry in summary["episodes"]:
        def mark(name: str) -> str:
            record = entry.get(name)
            if record is None:
                return "-"
            return "ok" if record.get("ok") else "**FAIL**"

        metrics = ((entry.get("validate") or {}).get("metrics") or {})
        resources = ((entry.get("retarget") or {}).get("resources") or {})
        add(
            "| {idx} | `{seq}` | {src} | {ref} | {sup} | {st} | {rt} | {va} | {fc} | "
            "{secs} | {gpu} | {ikm} | {ikr} | {rxy} |".format(
                idx=entry["episode_index"],
                seq=entry["sequence_id"],
                src=entry["source_frames"],
                ref=entry["reference_frames"],
                sup=str(entry["scene_support"]).lower()
                + f" ({entry['support_shape_count']})",
                st=mark("stage"), rt=mark("retarget"),
                va=mark("validate"), fc=mark("flashchord"),
                secs=round(float((entry.get("retarget") or {}).get("seconds") or 0), 1),
                gpu=resources.get("gpu_delta_mib", "-"),
                ikm=metrics.get("ik_median", "-"),
                ikr=metrics.get("ik_ratio_vs_demo", "-"),
                rxy=metrics.get("root_align_xy_max_m", "-"),
            )
        )
    add("")
    failures_seen = False
    for entry in summary["episodes"]:
        for name in ("stage", "retarget", "validate", "flashchord"):
            record = entry.get(name)
            if record is None or record.get("ok"):
                continue
            if not failures_seen:
                add("## Failures")
                add("")
                failures_seen = True
            detail = record.get("error") or ", ".join(
                (record.get("validation") or {}).get("failures", [])
                or record.get("failures", [])
            ) or f"exit {record.get('exit_code')}"
            add(f"- episode {entry['episode_index']} (`{entry['sequence_id']}`) "
                f"**{name}**: {detail}")
    if not failures_seen:
        add("No failures.")
    add("")
    notes = []
    for entry in summary["episodes"]:
        for name in ("stage", "validate"):
            record = entry.get(name) or {}
            for note in (record.get("validation") or record).get("notes", []) or []:
                notes.append(f"- episode {entry['episode_index']} ({name}): {note}")
    if notes:
        add("## Recorded notes")
        add("")
        lines.extend(notes)
        add("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
STAGES = {
    "inventory": stage_inventory,
    "stage": stage_stage,
    "retarget": stage_retarget,
    "validate": stage_validate,
    "flashchord": stage_flashchord,
    "regress": stage_regress,
    "report": stage_report,
}


def stage_all(args) -> int:
    code = 0
    for name in ("stage", "retarget", "validate", "flashchord", "report"):
        print()
        print("=" * 78)
        print(f"== {name}")
        print("=" * 78)
        code |= STAGES[name](args)
    return code


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("stage", choices=[*STAGES, "all"])
    parser.add_argument("--tier", default=TIER_1,
                        help=f"challenge tier directory name (default {TIER_1})")
    parser.add_argument("--episodes", default="all",
                        help="'all' or a selector like '0,3,5-9'")
    parser.add_argument("--chunk-frames", type=int, default=DEFAULT_CHUNK_FRAMES,
                        help="frames per SOMALayer forward pass (default "
                             f"{DEFAULT_CHUNK_FRAMES})")
    parser.add_argument("--timeout", type=int, default=3600,
                        help="per-episode container timeout in seconds")
    parser.add_argument("--force", action="store_true",
                        help="redo work that a previous record says succeeded")
    parser.add_argument("--no-rerun", action="store_true",
                        help="regress only: diff the reference that is already "
                             "on disk against the preserved baseline instead of "
                             "re-running the retarget. Non-destructive - the "
                             "existing reference is read, never overwritten.")
    args = parser.parse_args()

    try:
        if args.stage == "all":
            return stage_all(args)
        return STAGES[args.stage](args)
    except RegistryError as error:
        print(f"registry error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
