#!/usr/bin/env python3
"""The episode registry: one tier's Parquet mirror -> fully resolved episodes.

WHY THIS MODULE EXISTS
----------------------
Step A/B were written for Episode 0 with its sequence id hard-coded in three
places (``stage_episode_0_assets.py``, ``verify_step_B.sh``, the docs). Scaling
to 30 episodes x N tiers needs exactly one place that answers:

  * which episodes exist in this tier, and where are their four inputs;
  * where does the staged ``data_folder`` for this episode go;
  * where will ``soma_to_g1.py`` write its output parquet;
  * does this episode's support surface declare collision geometry, i.e. must
    FlashCHORD run with ``scene.support=false``;
  * how many frames should the retargeted reference have, and why.

Everything downstream (staging, retargeting, validation, reporting) reads this
module instead of re-deriving paths, so a layout change happens once.

STDLIB ONLY - ON PURPOSE
------------------------
The retarget orchestrator drives Docker from the host, where there is no
``pyarrow``/``pandas`` (see CLAUDE.md). This module therefore imports nothing
outside the standard library, so host Python 3.12 and the project venv can both
use it. Anything needing the Parquet itself lives in ``lerobot_to_soma``.

TIER ISOLATION
--------------
Tier 1 and Tier 2 publish *the same 30 sequence ids*. Staging them under one
root would make Tier 2's noisy inputs silently overwrite Tier 1's clean ones -
or worse, let a Tier 2 run train against a Tier 1 reference and look great.
``out_root_for_tier`` therefore gives every tier its own staged tree, and
``EpisodeSpec.provenance`` records the tier plus the source Parquet's hash into
every staged episode so a mix-up is detectable after the fact rather than
assumed impossible.

Tier 1 keeps the original ``data/adapter_out`` root so Episode 0's existing
artifacts, ``verify_step_B.sh`` and the docs all keep working unchanged.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

# Dataset-relative layout, as published on Hugging Face.
TRACK_2 = "data/v2d_challenge/track_2"
TIER_1 = "tier_1_multiview_caption"
TIER_2 = "tier_2_synthetic_noise"
#: Tiers we build ourselves (Tier 3 reconstructions), laid out exactly like a published tier
#: but kept out of the pristine Hugging Face mirror under TRACK_2.
TRACK_2_RECONSTRUCTED = "data/reconstructed/track_2"

#: Staged output root per tier, relative to the repo root. Tier 1 is pinned to
#: the pre-existing path so nothing produced before this change moves.
_OUT_ROOT = {
    TIER_1: "data/adapter_out",
    TIER_2: "data/adapter_out_tier_2",
}

#: Only these two USD prim types are collision geometry for FlashCHORD's
#: ``add_support_surfaces`` (flash_chord/scene/support.py:69). A stage with none
#: of them makes it raise, so the sequence must run with ``scene.support=false``.
SUPPORT_SHAPE_TYPES = ("Cylinder", "Cube")

_SHAPE_DEF = re.compile(
    r"^\s*def\s+(" + "|".join(SUPPORT_SHAPE_TYPES) + r")\s+\"", re.MULTILINE
)
#: Composition arcs that would pull prims in from another layer. A flat scan of
#: the file text cannot see those, so their presence invalidates the count
#: instead of under-reporting it.
_COMPOSITION_ARC = re.compile(
    r"\b(subLayers|references|payload|inherits|specializes|variantSet)\b"
)


class RegistryError(RuntimeError):
    """Raised when the dataset mirror does not look the way the registry needs."""


def repo_root() -> Path:
    """The repository root, derived from this file's location."""
    return Path(__file__).resolve().parents[1]


def tier_root_for(tier: str, repo: Path | None = None) -> Path:
    """Input root of ``tier``: the published mirror, else one of our reconstructed tiers.

    A name in both places resolves to the published mirror, so a later official release of a
    same-named tier can never be shadowed by our own data.
    """
    repo = repo or repo_root()
    published = repo / TRACK_2 / tier
    reconstructed = repo / TRACK_2_RECONSTRUCTED / tier
    if not published.is_dir() and reconstructed.is_dir():
        return reconstructed
    return published


def out_root_for_tier(tier: str, repo: Path | None = None) -> Path:
    """Staged output root for ``tier`` - a different tree for every tier."""
    repo = repo or repo_root()
    if tier not in _OUT_ROOT:
        # A new tier (e.g. a Tier 3 reconstruction) gets its own root by name
        # rather than defaulting into Tier 1's and overwriting it.
        return repo / f"data/adapter_out_{tier}"
    return repo / _OUT_ROOT[tier]


def sha256(path: Path) -> str:
    """Streaming sha256 of a file, used for provenance and copy verification."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_support_collision_shapes(usda: Path) -> int:
    """Count the Cylinder/Cube prims FlashCHORD would turn into collision shapes.

    This is a text scan rather than a USD parse so it runs on the host, with no
    ``pxr`` available. The challenge's support stages are flat, self-contained
    ASCII layers (verified over all 60 published files), so the scan is exact -
    but if a file ever gains a composition arc the scan would silently
    under-count, so that case raises instead.

    ``check_reference_in_flashchord.py`` re-counts the same prims through the
    real ``Usd.Stage`` inside the container; the two numbers are compared in
    the report, so this fast path is never the only evidence.
    """
    if not usda.is_file():
        raise RegistryError(f"support surface missing: {usda}")
    text = usda.read_text(encoding="utf-8")
    arc = _COMPOSITION_ARC.search(text)
    if arc is not None:
        raise RegistryError(
            f"{usda} uses USD composition ({arc.group(1)}); a flat text scan "
            "cannot count its collision shapes. Count them with pxr.Usd instead."
        )
    return len(_SHAPE_DEF.findall(text))


@dataclass(frozen=True)
class EpisodeSpec:
    """One episode of one tier, with every path the pipeline needs resolved.

    ``frames`` is the source frame count from ``episodes_metadata.jsonl``.
    ``retarget_frames`` is what the reference is expected to contain, which
    differs only when trailing frames carry no object measurement - see
    ``invalid_object_frames`` and ``retarget_end_frame``.
    """

    tier: str
    tier_root: Path
    out_root: Path
    episode_index: int
    sequence_id: str
    object_name: str
    task: str
    frames: int
    fps: int
    support_shape_count: int
    # Set by ``annotate_object_validity`` once the Parquet has been read; -1
    # means "not probed yet" so an unprobed spec can never masquerade as clean.
    visible_object_frames: int = -1
    invalid_object_frames: tuple[int, ...] = ()

    # ---------------------------------------------------------------- inputs
    @property
    def parquet(self) -> Path:
        """The pristine LeRobot episode Parquet."""
        return self.tier_root / "data/chunk-000" / f"episode_{self.episode_index:06d}.parquet"

    @property
    def mesh(self) -> Path:
        """The scanned object GLB, shared by every episode using that object."""
        return self.tier_root / "mesh" / self.object_name / f"{self.object_name}.glb"

    @property
    def ground_plane(self) -> Path:
        return self.tier_root / "ground_plane" / f"{self.sequence_id}.json"

    @property
    def support_surface(self) -> Path:
        return self.tier_root / "support_surface" / f"{self.sequence_id}_support.usda"

    # ---------------------------------------------------------------- staged
    @property
    def data_folder(self) -> Path:
        """``soma_to_g1.py``'s positional ``data_folder``."""
        return self.out_root / "whole_body/soma" / self.sequence_id

    @property
    def staged_mesh(self) -> Path:
        return self.data_folder / "reconstructed_mesh/output_aligned.glb"

    @property
    def staged_ground_plane(self) -> Path:
        return self.data_folder / "ground_plane.json"

    @property
    def staged_support(self) -> Path:
        """Where ``registry.support_usda_for_reference`` looks (registry.py:127)."""
        return self.out_root / "whole_body/reconstructed_stage" / f"{self.sequence_id}_support.usda"

    @property
    def manifest(self) -> Path:
        return self.data_folder / "episode_manifest.json"

    # --------------------------------------------------------------- outputs
    @property
    def retarget_dir(self) -> Path:
        """The Hive-partitioned output directory soma_to_g1.py writes into."""
        return self.out_root / "whole_body/soma" / f"sequence_id={self.sequence_id}"

    @property
    def reference_parquet(self) -> Path:
        return self.retarget_dir / "robot_name=g1/data.parquet"

    @property
    def object_dir(self) -> Path:
        return self.retarget_dir / "object"

    # ------------------------------------------------------------- decisions
    @property
    def scene_support(self) -> bool:
        """FlashCHORD's ``scene.support``: only true when real geometry exists.

        16 of the 30 published episodes ship a support stage with zero collision
        shapes - NVIDIA's own convention for "no distinct support surface" - and
        ``add_support_surfaces`` raises on those.
        """
        return self.support_shape_count > 0

    @property
    def retarget_end_frame(self) -> int | None:
        """``--end-frame`` for soma_to_g1, or None for the whole sequence.

        A trailing run of frames with ``observation.object.visible == False`` is
        zero-filled by NVIDIA, not measured. We neither interpolate them nor
        emit a reference frame built from a zero pose: the window simply stops
        at the last measured frame. Any invalid frame that is NOT part of a
        trailing run cannot be handled this way and raises during planning.
        """
        if not self.invalid_object_frames:
            return None
        trailing = self._trailing_invalid_run()
        if trailing is None:
            return None
        return trailing

    @property
    def retarget_frames(self) -> int:
        """Frames the retargeted reference must contain."""
        end = self.retarget_end_frame
        return self.frames if end is None else end

    @property
    def dropped_frames(self) -> tuple[int, ...]:
        """Source frames deliberately excluded from the reference, with reason."""
        end = self.retarget_end_frame
        if end is None:
            return ()
        return tuple(range(end, self.frames))

    def _trailing_invalid_run(self) -> int | None:
        """First index of the trailing all-invalid run, if that is all there is."""
        invalid = set(self.invalid_object_frames)
        if not invalid:
            return None
        start = self.frames - len(invalid)
        if invalid == set(range(start, self.frames)):
            return start
        return None

    def validate_frame_policy(self) -> None:
        """Fail loudly when invalid frames are not a trailing run we can honour."""
        if self.visible_object_frames < 0:
            raise RegistryError(
                f"{self.sequence_id}: object validity was never probed; call "
                "annotate_object_validity() before planning a retarget."
            )
        if self.invalid_object_frames and self._trailing_invalid_run() is None:
            raise RegistryError(
                f"{self.sequence_id}: frames {list(self.invalid_object_frames)} have "
                "observation.object.visible == False in the MIDDLE of the sequence. "
                "They are zero-filled, not measured, and this pipeline will not "
                "interpolate across them or emit a reference frame from a zero "
                "pose. Handle this episode explicitly."
            )

    # ------------------------------------------------------------ provenance
    def provenance(self) -> dict:
        """The identity record written beside the staged inputs.

        Recording the tier and the source Parquet hash is what makes the
        tier-isolation claim checkable: a staged episode states which tier's
        measurements it came from, so a Tier 2 run cannot quietly inherit Tier
        1's clean motion.
        """
        return {
            "tier": self.tier,
            "episode_index": self.episode_index,
            "sequence_id": self.sequence_id,
            "object": self.object_name,
            "task": self.task,
            "fps": self.fps,
            "source_frames": self.frames,
            "retarget_frames": self.retarget_frames,
            "retarget_end_frame": self.retarget_end_frame,
            "dropped_frames": list(self.dropped_frames),
            "dropped_reason": (
                "observation.object.visible == False (NVIDIA zero-fill); not "
                "interpolated, not emitted"
                if self.dropped_frames
                else None
            ),
            "visible_object_frames": self.visible_object_frames,
            "invalid_object_frames": list(self.invalid_object_frames),
            "support_shape_count": self.support_shape_count,
            "scene_support": self.scene_support,
            "source_sha256": {
                "parquet": sha256(self.parquet),
                "mesh": sha256(self.mesh),
                "ground_plane": sha256(self.ground_plane),
                "support_surface": sha256(self.support_surface),
            },
        }


def load_tier(
    tier: str = TIER_1,
    *,
    repo: Path | None = None,
    out_root: Path | None = None,
) -> list[EpisodeSpec]:
    """Build an ``EpisodeSpec`` for every episode declared by a tier's metadata.

    Every input file is checked for existence here rather than at use time, so a
    half-downloaded mirror is reported once, up front, with the missing paths.
    """
    repo = repo or repo_root()
    tier_root = tier_root_for(tier, repo)
    meta = tier_root / "meta/episodes_metadata.jsonl"
    info_path = tier_root / "meta/info.json"
    if not meta.is_file():
        raise RegistryError(
            f"tier metadata missing: {meta}\n"
            "Fetch it with scripts/data/download_challenge_episodes.py."
        )
    if not info_path.is_file():
        raise RegistryError(f"tier info missing: {info_path}")

    info = json.loads(info_path.read_text())
    fps = int(info["fps"])
    root = out_root or out_root_for_tier(tier, repo)

    specs: list[EpisodeSpec] = []
    missing: list[str] = []
    for line in meta.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        spec = EpisodeSpec(
            tier=tier,
            tier_root=tier_root,
            out_root=root,
            episode_index=int(record["episode_index"]),
            sequence_id=str(record["sequence_id"]),
            object_name=str(record["object"]),
            task=str(record["task"]),
            frames=int(record["frames"]),
            fps=fps,
            support_shape_count=-1,
        )
        for role, path in (
            ("parquet", spec.parquet),
            ("mesh", spec.mesh),
            ("ground_plane", spec.ground_plane),
            ("support_surface", spec.support_surface),
        ):
            if not path.is_file():
                missing.append(f"episode {spec.episode_index} {role}: {path}")
        # The metadata also names the asset paths; cross-check rather than
        # trust our own convention silently.
        for role, declared in (
            ("mesh", record.get("mesh")),
            ("ground_plane", record.get("ground_plane")),
            ("support_surface", record.get("support_surface")),
        ):
            if declared is None:
                continue
            expected = (tier_root / declared).resolve()
            ours = getattr(spec, role).resolve()
            if expected != ours:
                raise RegistryError(
                    f"episode {spec.episode_index}: metadata declares {role} at "
                    f"{expected}, registry resolves {ours}"
                )
        specs.append(spec)

    if missing:
        raise RegistryError(
            "tier mirror is incomplete:\n  " + "\n  ".join(missing)
        )
    if not specs:
        raise RegistryError(f"no episodes declared in {meta}")

    # Fill in the support decision now that the files are known to exist.
    specs = [
        _replace(spec, support_shape_count=count_support_collision_shapes(spec.support_surface))
        for spec in specs
    ]
    indices = [spec.episode_index for spec in specs]
    if len(set(indices)) != len(indices):
        raise RegistryError(f"duplicate episode_index values in {meta}")
    return specs


def annotate_object_validity(
    spec: EpisodeSpec, visible_frames: int, invalid_frames: list[int] | tuple[int, ...]
) -> EpisodeSpec:
    """Attach the Parquet-derived object visibility to a spec."""
    return _replace(
        spec,
        visible_object_frames=int(visible_frames),
        invalid_object_frames=tuple(int(i) for i in invalid_frames),
    )


def _replace(spec: EpisodeSpec, **changes) -> EpisodeSpec:
    """``dataclasses.replace`` without importing it just for one call."""
    fields = {
        "tier": spec.tier,
        "tier_root": spec.tier_root,
        "out_root": spec.out_root,
        "episode_index": spec.episode_index,
        "sequence_id": spec.sequence_id,
        "object_name": spec.object_name,
        "task": spec.task,
        "frames": spec.frames,
        "fps": spec.fps,
        "support_shape_count": spec.support_shape_count,
        "visible_object_frames": spec.visible_object_frames,
        "invalid_object_frames": spec.invalid_object_frames,
    }
    fields.update(changes)
    return EpisodeSpec(**fields)


def parse_episode_selector(selector: str, available: list[int]) -> list[int]:
    """Turn ``"all"`` / ``"0,3,5-9"`` into a sorted list of episode indices.

    Raises on anything not present in ``available`` so a typo selects nothing
    silently instead of quietly running 29 of the 30 episodes.
    """
    text = selector.strip()
    if not text or text == "all":
        return sorted(available)
    chosen: set[int] = set()
    for part in text.split(","):
        piece = part.strip()
        if not piece:
            continue
        if "-" in piece.lstrip("-"):
            lo_text, _, hi_text = piece.partition("-")
            try:
                lo, hi = int(lo_text), int(hi_text)
            except ValueError as error:
                raise RegistryError(f"bad episode range {piece!r}") from error
            if hi < lo:
                raise RegistryError(f"episode range {piece!r} is reversed")
            chosen.update(range(lo, hi + 1))
        else:
            try:
                chosen.add(int(piece))
            except ValueError as error:
                raise RegistryError(f"bad episode index {piece!r}") from error
    unknown = sorted(chosen - set(available))
    if unknown:
        raise RegistryError(
            f"episode(s) {unknown} are not in this tier (have "
            f"{min(available)}..{max(available)})"
        )
    return sorted(chosen)


def select(specs: list[EpisodeSpec], selector: str) -> list[EpisodeSpec]:
    """Filter ``specs`` by a selector string, preserving dataset order."""
    wanted = set(parse_episode_selector(selector, [s.episode_index for s in specs]))
    return [spec for spec in specs if spec.episode_index in wanted]
