#!/usr/bin/env python3
"""Validation that must run INSIDE the robotic-grounding container.

Checks two things against NVIDIA's real code:
  1. the staged SOMA-X bundle satisfies read_soma._missing_assets (content, not
     just presence - a 'slim' SOMA_neutral.npz is rejected)
  2. our adapter's soma_params.npz is accepted by the genuine _load_npz()

Stops deliberately before load_motion(): no reconstruction, no retargeting.

Usage (paths are the container-side mounts):
    python validate_soma_in_container.py /staging/soma /staging/episode0
"""
import pathlib
import sys

import torch

from robotic_grounding.retarget.read_soma import SOMA, _missing_assets

soma_root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/staging/soma")
episode = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else "/staging/episode0")
failed = False

missing = _missing_assets(soma_root, "mhr")
if missing:
    print("  FAIL  SOMA-X bundle incomplete/unusable: {}".format(missing))
    failed = True
else:
    print("  PASS  SOMA-X bundle complete and usable (full rig, not a slim npz)")

try:
    layer = SOMA(data_root=str(soma_root), identity_model_type="mhr",
                 device=torch.device("cpu"))
    print("  PASS  SOMALayer loaded, rig_joint_names={}".format(len(layer.rig_joint_names)))
    p = layer._load_npz(str(episode / "soma_params.npz"))
    print("  PASS  real read_soma accepts our npz: poses{} transl{}".format(
        p["poses"].shape, p["transl"].shape))
    print("  PASS  joint_names match the live rig (would have raised otherwise)")
    print("  PASS  joint_orient auto-filled {}".format(p["joint_orient"].shape))
    print("  PASS  unit={!r} identity_model_type={!r}".format(
        p["unit"], p["identity_model_type"]))
except Exception as exc:  # noqa: BLE001 - we want the reason printed, not a traceback
    print("  FAIL  {}: {}".format(type(exc).__name__, exc))
    failed = True

sys.exit(1 if failed else 0)
