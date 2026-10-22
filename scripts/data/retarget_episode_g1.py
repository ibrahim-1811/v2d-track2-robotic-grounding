"""Memory-bounded driver for NVIDIA's soma_to_g1.py retargeter.

Why this exists
---------------
``SOMA.load_motion`` reconstructs the whole sequence in one ``SOMALayer``
forward pass (``read_soma.py:397``). That call expands the *frame-invariant*
identity and scale coefficients to ``(T, K)``::

    identity_t = torch.from_numpy(identity).to(device).unsqueeze(0).expand(T_frames, -1)
    scale_t    = torch.from_numpy(scale).to(device).unsqueeze(0).expand(T_frames, -1)

so ``SOMALayer.forward`` runs ``prepare_identity`` over ``T`` identical
identities. ``prepare_identity``'s own docstring says to "call once per
identity, then call pose() for each new frame ... avoids recomputing the
identity model and skeleton transfer per frame", and its skeleton transfer
allocates Warp buffers proportional to the batch. On an 8 GB GPU a 778-frame
sequence dies inside that fit::

    SOMALayer.forward -> prepare_identity -> skeleton_transfer.fit
      -> fit_rotations_warp -> align_vectors_warp
    RuntimeError: Failed to allocate 283254240 bytes on device 'cuda:0'

``--start-frame``/``--end-frame`` do not help: ``soma_to_g1.py`` slices at
line 516, long after the full-sequence load, so even a 64-frame window pays
the full cost.

What this driver does
---------------------
It replaces ``SOMA.load_motion`` with a chunked implementation that calls the
*unmodified* ``SOMA._reconstruct`` over fixed-size frame blocks and
concatenates the results, then delegates to ``soma_to_g1.main()`` untouched.
NVIDIA's source is never edited; this file is mounted read-only beside it.

Correctness
-----------
Chunking is numerically equivalent to the single-shot call:

* ``SOMALayer.forward`` and ``_compute_global_rotations`` are per-frame
  (``batch_rodrigues`` / ``apply_joint_orient_local`` /
  ``joint_local_to_world_levelorder`` are elementwise over the batch), so no
  frame reads another frame's state.
* ``identity`` and ``scale`` come from index ``[0]`` upstream and are constant,
  so each chunk fits the same identity.
* **Frame-0 anchoring is preserved exactly.** The transform is built from the
  full-sequence ``poses[0, 0]`` and ``transl[0]`` *before* chunking, and the
  same transform is applied to every chunk, matching upstream's
  ``_normalize_to_first_frame`` over the whole array. This also keeps
  ``first_frame_transl`` / ``first_frame_R_inv`` identical, which is what
  ``soma_to_g1.py`` uses to anchor the object trajectory and ground plane.
* Chunks are concatenated in source order, so frame continuity is preserved.

Usage (inside the robotic-grounding container)::

    python retarget_episode_g1.py --chunk-frames 64 -- \
        <data_folder> --save --soma-data-root /soma ... (soma_to_g1 args)
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

SOMA_TO_G1 = Path(
    "/workspace/video_to_data/robotic_grounding/scripts/retarget/soma_to_g1.py"
)


def parse_driver_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    """Split our own flags from the arguments forwarded to soma_to_g1."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--chunk-frames",
        type=int,
        default=64,
        help="frames per SOMALayer forward pass (default 64)",
    )
    parser.add_argument(
        "--soma-to-g1",
        type=Path,
        default=SOMA_TO_G1,
        help="path to NVIDIA's unmodified soma_to_g1.py",
    )
    if "--" in argv:
        split = argv.index("--")
        own, forwarded = argv[:split], argv[split + 1 :]
    else:
        own, forwarded = argv, []
    return parser.parse_args(own), forwarded


def install_chunked_load_motion(chunk_frames: int) -> None:
    """Monkeypatch SOMA.load_motion with a frame-chunked equivalent."""
    if chunk_frames < 1:
        raise ValueError("--chunk-frames must be >= 1, got {}".format(chunk_frames))

    from robotic_grounding.retarget import read_soma as rs

    def chunked_load_motion(self, params_path, *, normalize: bool = False):
        params = self._load_npz(params_path)
        meters = rs._unit_to_meters_factor(params["unit"])

        poses_rotvec = params["poses"].astype(np.float32)  # (T, J, 3)
        transl = params["transl"].astype(np.float32) * meters  # (T, 3)
        identity = params["identity_coeffs"][0].astype(np.float32)
        scale = params["scale_params"][0].astype(np.float32)
        joint_orient = params["joint_orient"].astype(np.float32)
        keep_root = bool(np.asarray(params["keep_root"]).item())
        absolute_pose = bool(np.asarray(params.get("absolute_pose", False)).item())
        if absolute_pose:
            raise ValueError(
                "SOMA export with absolute_pose=True is not yet supported by read_soma."
            )

        total_frames = int(poses_rotvec.shape[0])

        # Built from the FULL sequence before chunking, so anchoring matches a
        # single-shot run exactly.
        first_frame_transform = rs._first_frame_transform(
            poses_rotvec[0, 0], transl[0]
        )

        joint_blocks: list[np.ndarray] = []
        rotation_blocks: list[np.ndarray] = []
        vertex_blocks: list[np.ndarray] = []

        for begin in range(0, total_frames, chunk_frames):
            end = min(begin + chunk_frames, total_frames)
            print(
                "[retarget_episode_g1] SOMA reconstruct frames "
                "[{}, {}) of {}".format(begin, end, total_frames),
                flush=True,
            )
            block_joints, block_rotations, block_vertices = self._reconstruct(
                poses_rotvec=poses_rotvec[begin:end],
                transl=transl[begin:end],
                identity=identity,
                scale=scale,
                joint_orient=joint_orient,
                keep_root=keep_root,
            )
            joint_blocks.append(block_joints)
            rotation_blocks.append(block_rotations)
            vertex_blocks.append(block_vertices)
            _release_cuda_cache()

        joints = np.concatenate(joint_blocks, axis=0)
        joints_wxyz = np.concatenate(rotation_blocks, axis=0)
        vertices = np.concatenate(vertex_blocks, axis=0)
        del joint_blocks, rotation_blocks, vertex_blocks

        if joints.shape[0] != total_frames:
            raise AssertionError(
                "chunked reconstruction produced {} frames, expected {}".format(
                    joints.shape[0], total_frames
                )
            )

        if normalize:
            joints, joints_wxyz, vertices = rs._normalize_to_first_frame(
                joints=joints,
                joints_wxyz=joints_wxyz,
                vertices=vertices,
                transform=first_frame_transform,
            )

        return {
            "joints": joints.astype(np.float64),
            "joints_wxyz": joints_wxyz.astype(np.float64),
            "vertices": vertices.astype(np.float64),
            "num_frames": total_frames,
            "joint_names": list(self.rig_joint_names),
            "identity_model_type": self.identity_model_type,
            "unit": "meters",
            "first_frame_transl": first_frame_transform["transl_first"],
            "first_frame_R_inv": first_frame_transform["R_first_inv"],
            "normalized": bool(normalize),
        }

    rs.SOMA.load_motion = chunked_load_motion
    print(
        "[retarget_episode_g1] SOMA.load_motion patched: "
        "chunk_frames={}".format(chunk_frames),
        flush=True,
    )


def _release_cuda_cache() -> None:
    """Return each chunk's cached blocks so peak memory stays per-chunk."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 - freeing cache is best-effort
        pass


def load_soma_to_g1(path: Path):
    """Import NVIDIA's script from its path without copying or editing it."""
    if not path.is_file():
        raise FileNotFoundError("soma_to_g1.py not found at {}".format(path))
    spec = importlib.util.spec_from_file_location("nvidia_soma_to_g1", path)
    if spec is None or spec.loader is None:
        raise ImportError("could not load a module spec from {}".format(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules["nvidia_soma_to_g1"] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    driver_args, forwarded = parse_driver_args(sys.argv[1:])
    if not forwarded:
        print(
            "error: no soma_to_g1 arguments given; separate them with ' -- '",
            file=sys.stderr,
        )
        return 2

    install_chunked_load_motion(driver_args.chunk_frames)
    module = load_soma_to_g1(driver_args.soma_to_g1)

    # soma_to_g1.main() reads sys.argv through its own parse_args().
    sys.argv = [str(driver_args.soma_to_g1), *forwarded]
    module.main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
