"""Record a policy rollout to a .viser file for offline playback in a browser.

Runs NVIDIA's scripts/view_policy.py unchanged (same Hydra overrides), but builds its Viser
viewer with Newton's own ``record_to_viser`` option and stops after ``viewer.num_frames``
frames, so the loop ends and Newton's ``close()`` writes the recording. The output path comes
from ``V2D_RECORD_VISER``. Play it back with the Viser client build shipped in the image
(see cluster/record_policy_media.sbatch).
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

import newton.viewer
import warp as wp
from flash_chord.visualization import viewer as fc_viewer

VIEW_POLICY = Path("/workspace/scripts/view_policy.py")


class _BoundedRecorder:
    """Stop a recording Viser viewer after a fixed number of frames."""

    def __init__(self, viewer, num_frames: int):
        self._viewer = viewer
        self.num_frames = num_frames
        self.frame_count = 0

    def __getattr__(self, name: str):
        return getattr(self._viewer, name)

    def is_running(self) -> bool:
        return self.frame_count < self.num_frames and self._viewer.is_running()

    def end_frame(self) -> None:
        self._viewer.end_frame()
        self.frame_count += 1

    def close(self) -> None:
        self._viewer.close()


def _create_viewer(config, device=None):
    if config.backend != "viser":
        return _upstream_create_viewer(config, device)
    wp.config.quiet = config.quiet
    if device is not None:
        wp.set_device(device)
    output = Path(os.environ["V2D_RECORD_VISER"])
    output.parent.mkdir(parents=True, exist_ok=True)
    return _BoundedRecorder(newton.viewer.ViewerViser(record_to_viser=str(output)), config.num_frames)


_upstream_create_viewer = fc_viewer.create_viewer
fc_viewer.create_viewer = _create_viewer
sys.argv[0] = str(VIEW_POLICY)
runpy.run_path(str(VIEW_POLICY), run_name="__main__")
