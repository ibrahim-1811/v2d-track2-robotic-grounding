"""Mirror the Track 2 challenge episodes from Hugging Face with stdlib only.

The host and venv Pythons have no ``huggingface_hub`` and there is no HF CLI, so
this fetches the exact per-episode files listed in each tier's
``meta/episodes_metadata.jsonl`` over plain HTTPS. The dataset is public
(``nvidia/video_to_data_challenge``), so no token is needed.

Per episode it pulls:
  data/chunk-000/episode_XXXXXX.parquet   the LeRobot episode (human + object poses)
  ground_plane/<sequence_id>.json         fitted ground plane
  support_surface/<sequence_id>_support.usda
  mesh/<object>/<object>.glb              object mesh (deduplicated across episodes)

Videos are skipped by default: Track 2 needs them only to build a Tier 3
reconstruction, and they dominate the download size (~23 MB/episode). Pass
--videos to include them.

Existing files of nonzero size are left alone, so the script is resumable and
safe to re-run. It only ever writes under the mirror's own tier directory.

    .venvs/episode0/bin/python scripts/download_challenge_episodes.py --list
    .venvs/episode0/bin/python scripts/download_challenge_episodes.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = "nvidia/video_to_data_challenge"
BASE = "https://huggingface.co/datasets/{}/resolve/main".format(REPO)
TIERS = ("tier_1_multiview_caption", "tier_2_synthetic_noise")
MIRROR = Path("data/v2d_challenge/track_2")
META_FILES = ("meta/info.json", "meta/episodes_metadata.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tiers", nargs="*", default=list(TIERS))
    parser.add_argument("--episodes", nargs="*", type=int, default=None,
                        help="episode indices to fetch (default: all)")
    parser.add_argument("--videos", action="store_true",
                        help="also fetch the exo-camera MP4s (large; Tier 3 only)")
    parser.add_argument("--list", action="store_true",
                        help="print what would be fetched and exit")
    parser.add_argument("--retries", type=int, default=3)
    return parser.parse_args()


def fetch(remote: str, local: Path, retries: int) -> tuple[bool, str]:
    """Download one file unless it already exists nonempty. Returns (changed, note)."""
    if local.is_file() and local.stat().st_size > 0:
        return False, "have"
    local.parent.mkdir(parents=True, exist_ok=True)
    temporary = local.with_suffix(local.suffix + ".part")
    last_error = ""
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(remote, headers={"User-Agent": "v2d-mirror/1.0"})
            with urllib.request.urlopen(request, timeout=120) as response, \
                    temporary.open("wb") as handle:
                while True:
                    block = response.read(1 << 20)
                    if not block:
                        break
                    handle.write(block)
            if temporary.stat().st_size == 0:
                raise OSError("empty response body")
            temporary.replace(local)
            return True, "{:,} B".format(local.stat().st_size)
        except urllib.error.HTTPError as error:
            temporary.unlink(missing_ok=True)
            # A 404 will not become a 200 on retry.
            return False, "HTTP {}".format(error.code)
        except Exception as error:  # noqa: BLE001 - network flake, retry
            temporary.unlink(missing_ok=True)
            last_error = str(error)
            if attempt < retries:
                time.sleep(2 * attempt)
    return False, "failed: {}".format(last_error)


def tier_metadata(tier: str, retries: int) -> list[dict]:
    """Ensure the tier's meta/ files exist locally, then parse the episode list."""
    for name in META_FILES:
        changed, note = fetch("{}/track_2/{}/{}".format(BASE, tier, name),
                              MIRROR / tier / name, retries)
        print("    {:46s} {}".format(name, note))
    path = MIRROR / tier / "meta" / "episodes_metadata.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.open() if line.strip()]


def episode_files(row: dict, want_videos: bool) -> list[str]:
    """Relative paths this episode needs, inside the tier directory."""
    index = int(row["episode_index"])
    files = ["data/chunk-000/episode_{:06d}.parquet".format(index)]
    for key in ("ground_plane", "support_surface", "mesh"):
        value = row.get(key)
        if value:
            files.append(str(value))
    if want_videos:
        files.append(
            "videos/chunk-000/observation.images.exo_camera/episode_{:06d}.mp4".format(index)
        )
    return files


def main() -> int:
    if not MIRROR.parent.parent.is_dir():
        print("error: run this from the repository root", file=sys.stderr)
        return 2

    args = parse_args()
    grand_total = 0
    failures: list[str] = []

    for tier in args.tiers:
        print("=== {} ===".format(tier))
        rows = tier_metadata(tier, args.retries)
        if not rows:
            print("    no episode metadata available; skipping this tier")
            failures.append("{}: no metadata".format(tier))
            continue
        if args.episodes is not None:
            wanted = set(args.episodes)
            rows = [r for r in rows if int(r["episode_index"]) in wanted]
        print("    {} episodes declared".format(len(rows)))

        # Meshes are shared between episodes of the same object; fetch each once.
        seen: set[str] = set()
        plan: list[tuple[int, str]] = []
        for row in rows:
            for relative in episode_files(row, args.videos):
                if relative in seen:
                    continue
                seen.add(relative)
                plan.append((int(row["episode_index"]), relative))

        missing = [(i, r) for i, r in plan if not (MIRROR / tier / r).is_file()]
        print("    {} files planned, {} already present, {} to fetch".format(
            len(plan), len(plan) - len(missing), len(missing)))
        if args.list:
            for index, relative in missing[:40]:
                print("      ep{:03d}  {}".format(index, relative))
            if len(missing) > 40:
                print("      ... and {} more".format(len(missing) - 40))
            continue

        for index, relative in missing:
            changed, note = fetch("{}/track_2/{}/{}".format(BASE, tier, relative),
                                  MIRROR / tier / relative, args.retries)
            status = note if changed else note
            print("    ep{:03d}  {:58s} {}".format(index, relative, status))
            if changed:
                grand_total += 1
            elif note != "have":
                failures.append("{}/{}: {}".format(tier, relative, note))
        print()

    if args.list:
        return 0
    print("fetched {} new files".format(grand_total))
    if failures:
        print("{} problem(s):".format(len(failures)))
        for line in failures[:20]:
            print("  {}".format(line))
        return 1
    print("all requested files present")
    return 0


if __name__ == "__main__":
    sys.exit(main())
