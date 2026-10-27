"""Tier 3 policy input from a CARI4D reconstruction, written as a Tier-1-shaped LeRobot episode.

Everything downstream (pipeline/lerobot_to_soma.py, the soma_to_g1 retarget, validation,
FlashCHORD training, and the kit's packer via ``--frame-references``) consumes the published
Tier 1 layout. This writes exactly that layout from CARI4D's output, under
``data/reconstructed/track_2/<tier>/`` so the pristine Hugging Face mirror is never touched.

  object  ``pr.pose_abs_postopt`` (camera <- CARI4D's aligned mesh) -> the released mesh frame via
          the export's ``source_object_mesh_to_aligned_transform`` -> world via ONE rigid transform
          fixed by Tier 1's frame-0 object pose, the only Tier 1 information Tier 3 may use (FAQ,
          upstream d9d6bf04). Only row 0 of the Tier 1 parquet is ever read for it. When that
          transform disagrees with Tier 1's frame-0 HUMAN by more than 20 deg (a symmetric object
          whose frame-0 orientation CARI4D resolved differently), the human places the scene and
          the object's orientation is relabelled and its track offset to match frame 0
          (scripts/tier3/frame0_alignment.py).
  human   CARI4D's refined MHR parameters (``pr.mhr_*``, identical to the post-opt checkpoint) ->
          vertices through CARI4D's own MHR head -> the same world transform -> SOMA frame ->
          SOMA-X PoseInversion with the MHR identity model, as upstream's
          ``v2d_sam3d_body/lib/export_soma.py`` does for SAM 3D Body output.
  scene   meta, ground plane, support surfaces, meshes and the bookkeeping columns (timestamp,
          index, task) come from the released Tier 2 inputs, not Tier 1.

Frames. CARI4D's camera frame is OpenCV: its human is ``MHR-native * diag(1,-1,-1) + trans``. The
challenge stores objects in a source OpenCV world frame and humans in the SOMA frame, related by
``_R_CV_TO_SOMA = diag(1,-1,-1)`` (soma_to_g1.py:308)::

    world_cv = T_align @ camera_cv,   T_align = P_T1(0) @ inv(P_cari(0))
    soma     = diag(1,-1,-1) @ world_cv

The human vertices are moved into the SOMA frame BEFORE the inversion, so the fit returns
SOMA-frame parameters directly. The identity (shape, 68 scales, 6 bone lengths) is held at its
per-episode median so the SOMA fit and the vertices it fits agree; upstream's retarget reads
only frame 0 of the identity anyway.

Run inside containers/cari4d.sif on a GPU (see cluster/tier3_convert.sbatch)::

    python scripts/tier3/cari4d_to_lerobot.py --episode 0

Deliberately no comparison with the Tier 1 caption here, not even as a report: the FAQ allows
Tier 3 to use Tier 1's first frame for alignment "provided the Tier 1 caption isn't used in
Tier 3 otherwise". The checks below (SOMA fit error, re-posing the saved parameters, the MHR
decoder self-check) need no Tier 1 data.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
TRACK_2 = REPO / "data/v2d_challenge/track_2"
TIER_1 = TRACK_2 / "tier_1_multiview_caption"
TIER_2 = TRACK_2 / "tier_2_synthetic_noise"
DEFAULT_TIER = "tier_3_cari4d_ours"
OUT_TRACK_2 = REPO / "data/reconstructed/track_2"
SOMA_ASSETS = REPO / "data/body_models/soma"  # nvidia/SOMA-X @ 466879a8, as the retarget uses
MHR_ASSETS = REPO / "data/weights/cari4d/sam3d_body"
CV_TO_SOMA = np.diag([1.0, -1.0, -1.0])
HUMAN_COLUMNS = ("pose", "translation", "identity_coeffs", "scale_params", "bone_length_flexibles")


def pose7_to_matrix(pose7: np.ndarray) -> np.ndarray:
    """[x, y, z, qw, qx, qy, qz] -> (4, 4); the challenge's observation.object.pose layout."""
    from scipy.spatial.transform import Rotation

    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat(np.asarray(pose7[3:7])[[1, 2, 3, 0]]).as_matrix()
    matrix[:3, 3] = pose7[:3]
    return matrix


def matrices_to_pose7(matrices: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    quat_xyzw = Rotation.from_matrix(matrices[:, :3, :3]).as_quat()
    return np.concatenate([matrices[:, :3, 3], quat_xyzw[:, [3, 0, 1, 2]]], axis=1)


def episode_row(episode: int) -> dict:
    rows = [json.loads(line) for line in (TIER_2 / "meta/episodes_metadata.jsonl").open()]
    return [r for r in rows if r["episode_index"] == episode][0]


def tier1_first_frame_object_pose(episode: int) -> np.ndarray:
    """Row 0 of Tier 1's observation.object.pose -- the one Tier 1 value Tier 3 may use."""
    import pyarrow.parquet as pq

    parquet = TIER_1 / "data/chunk-000" / f"episode_{episode:06d}.parquet"
    table = pq.ParquetFile(parquet).read_row_group(0, columns=["observation.object.pose", "observation.object.visible"])
    if not table.column("observation.object.visible")[0].as_py():
        raise SystemExit(f"Tier 1 frame 0 of episode {episode} is not visible; refusing to align on a zero-filled pose")
    return np.asarray(table.column("observation.object.pose")[0].as_py(), dtype=np.float64)


def cari4d_object(root: Path, sequence: str, refined: dict) -> np.ndarray:
    """(T, 4, 4) camera <- released-mesh poses."""
    export = json.loads((root / "export" / sequence / "wild_export.json").read_text())
    aligned_from_source = np.asarray(export["source_object_mesh_to_aligned_transform"], dtype=np.float64)
    return np.asarray(refined["pr"]["pose_abs_postopt"], dtype=np.float64) @ aligned_from_source


def mhr_vertices(refined: dict, device: str, batch: int = 128):
    """CARI4D's refined human -> camera-frame vertices (m), with the identity held at its median.

    Mirrors ``lib_mhr.mhr_layer.MHRParametricBackend.vertices_only`` exactly, except that it also
    asks the head for its 204-value model parameters, whose [130:136] are the bone-length
    flexibles and [136:204] the resolved body-part scales that SOMA's MHR identity model takes.
    """
    import roma
    import torch
    from lib_mhr.mhr_layer import MHRLayer
    from lib_mhr.rotations import rot6d_to_rotmat
    from sam_3d_body.models.modules.mhr_utils import compact_cont_to_model_params_body

    layer = MHRLayer.from_mhr_assets(mhr_assets_root=str(MHR_ASSETS), device=device)
    head = layer.backend._ensure_head(torch.device(device))
    pr = refined["pr"]

    def tensor(key: str) -> torch.Tensor:
        return torch.as_tensor(np.asarray(pr[key]), dtype=torch.float32, device=device)

    rot6d, trans, cont = tensor("mhr_global_rot6d"), tensor("mhr_trans"), tensor("mhr_body_pose_cont")
    hand, shape, scale, face = tensor("mhr_hand"), tensor("mhr_shape"), tensor("mhr_scale"), tensor("mhr_face")
    frames = rot6d.shape[0]
    body = compact_cont_to_model_params_body(cont)
    identity_spread = {
        "shape_max_abs_dev": float((shape - shape.median(0).values).abs().max()),
        "scale_max_abs_dev": float((scale - scale.median(0).values).abs().max()),
        "bone_flex_max_abs_dev": float((body[:, 124:130] - body[:, 124:130].median(0).values).abs().max()),
    }
    shape_c = shape.median(0).values.expand(frames, -1)
    scale_c = scale.median(0).values.expand(frames, -1)
    body = body.clone()
    body[:, 124:130] = body[:, 124:130].median(0).values
    global_rot = roma.rotmat_to_euler("ZYX", rot6d_to_rotmat(rot6d))
    flip = torch.tensor([1.0, -1.0, -1.0], device=device)

    vertices, params = [], []
    with torch.no_grad():
        for start in range(0, frames, batch):
            sl = slice(start, start + batch)
            verts, model_params = head.mhr_forward(
                global_trans=trans[sl].new_zeros(trans[sl].shape), global_rot=global_rot[sl],
                body_pose_params=body[sl], hand_pose_params=hand[sl], scale_params=scale_c[sl],
                shape_params=shape_c[sl], expr_params=face[sl], return_model_params=True,
            )
            vertices.append((verts * flip + trans[sl][:, None, :]).cpu())
            params.append(model_params.cpu())
        # Self-check against CARI4D's own decoder on frame 0 with the real per-frame identity.
        reference = layer.backend.vertices_only({k: tensor(k)[:1] for k in pr if k.startswith("mhr_")})
        rebuilt = head.mhr_forward(
            global_trans=trans[:1] * 0, global_rot=global_rot[:1], body_pose_params=compact_cont_to_model_params_body(cont[:1]),
            hand_pose_params=hand[:1], scale_params=scale[:1], shape_params=shape[:1], expr_params=face[:1],
        ) * flip + trans[:1][:, None, :]
        identity_spread["decoder_selfcheck_max_m"] = float((reference - rebuilt).abs().max())
    model_params = torch.cat(params).numpy().astype(np.float64)
    identity = {
        "identity_coeffs": shape_c[0].cpu().numpy(),
        "scale_params": model_params[0, 136:],
        "bone_length_flexibles": model_params[0, 130:136],
    }
    return torch.cat(vertices).numpy().astype(np.float64), identity, identity_spread


def soma_inversion(vertices_soma_m: np.ndarray, identity: dict, device: str, batch: int = 64) -> dict:
    """SOMA-X PoseInversion with the MHR identity model, as export_soma.py does (its defaults)."""
    import torch
    from soma import SOMALayer
    from soma.geometry.rig_utils import remove_joint_orient_local
    from soma.geometry.transforms import matrix_to_rotvec
    from soma.pose_inversion import PoseInversion
    from soma.units import Unit

    soma = SOMALayer(data_root=str(SOMA_ASSETS), identity_model_type="mhr", device=device, mode="warp",
                     output_unit=Unit.CENTIMETERS, enable_procedural_transforms=False)
    inversion = PoseInversion(soma, low_lod=True)
    frames = vertices_soma_m.shape[0]
    expand = lambda a: torch.as_tensor(np.repeat(a[None], frames, 0), dtype=torch.float32)  # noqa: E731
    ic, sp, bl = expand(identity["identity_coeffs"]), expand(identity["scale_params"]), expand(identity["bone_length_flexibles"])
    verts_cm = torch.as_tensor(vertices_soma_m * 100.0, dtype=torch.float32)
    rotations, translations, errors = [], [], []
    for start in range(0, frames, batch):
        sl = slice(start, start + batch)
        inversion.prepare_identity(ic[sl].to(device), sp[sl].to(device), kwargs={"bone_length_flexibles": bl[sl].to(device)})
        result = inversion.fit(verts_cm[sl].to(device), body_iters=2, finger_iters=0, full_iters=1, lie_iters=3,
                               lie_lambda=1e-1, autograd_iters=0, leaf_weight=1.0, batch_size=None)
        rotations.append(result["rotations"].cpu())
        translations.append(result["root_translation"].cpu())
        errors.append(result["per_vertex_error"].cpu())  # fit error against the MHR vertices
    rotations = torch.cat(rotations)
    root_translation_cm = torch.cat(translations)
    local = remove_joint_orient_local(rotations.to(soma._t_pose_orient.device), soma._t_pose_orient, soma._t_pose_orient_parent_T)
    poses = matrix_to_rotvec(local.reshape(-1, 3, 3)).reshape(rotations.shape[0], rotations.shape[1], 3).cpu().numpy()
    error = torch.cat(errors)
    # The fit returns the 78-joint rig including the virtual Root (index 0). The dataset and
    # SOMA's public pose API use the 77 joints from Hips on; soma.io.save_soma_npz(keep_root=False)
    # -- upstream export_soma's own path -- drops index 0 and keeps the translation unchanged.
    out = {
        "joint_names": list(soma.rig_data["joint_names"])[1:],
        "poses": poses[:, 1:].astype(np.float32),
        "transl": (root_translation_cm.numpy() / 100.0).astype(np.float32),  # cm -> m
        "fit_vertex_error_cm": {"mean": float(error.mean()), "median": float(error.median()), "max": float(error.max())},
    }
    # Re-pose the SAVED parameters (77 local rotvecs, Hips translation in metres) through SOMA's
    # public forward and compare with the fitted absolute pose skinned directly, as upstream's
    # export_soma debug path does (_reconstruct_soma_vertices). Same SOMA mesh on both sides, so
    # any error in the stored representation shows up here.
    check = []
    with torch.no_grad():
        for start in range(0, frames, batch):
            sl = slice(start, start + batch)
            saved = soma.forward(torch.as_tensor(out["poses"][sl], device=device), ic[sl].to(device), sp[sl].to(device),
                                 transl=torch.as_tensor(out["transl"][sl] * 100.0, device=device),
                                 apply_correctives=False,  # the rig is built without them, as export_soma's is
                                 kwargs={"bone_length_flexibles": bl[sl].to(device)}).vertices
            skinning = soma.batched_skinning
            skinning.rebind(soma._cached_bind_transforms_world, soma._cached_rest_shape)
            fitted, _ = skinning.pose(rotations[sl].to(device), root_translation_cm[sl].to(device),
                                      absolute_pose=True, return_transforms=True)
            check.append((saved - fitted).norm(dim=-1).mean(dim=1).cpu())
    out["reposed_vertex_error_cm_mean"] = float(torch.cat(check).mean())
    return out


def init_tier(tier: str) -> Path:
    """Create the tier folder from the released Tier 2 meta and scene files (once)."""
    root = OUT_TRACK_2 / tier
    if (root / "meta/info.json").is_file():
        return root
    (root / "meta").mkdir(parents=True, exist_ok=True)
    for name in ("ground_plane", "support_surface", "mesh"):
        if not (root / name).exists():
            shutil.copytree(TIER_2 / name, root / name)
    shutil.copy2(TIER_2 / "meta/episodes_metadata.jsonl", root / "meta/episodes_metadata.jsonl")
    info = json.loads((TIER_2 / "meta/info.json").read_text())
    info["trajectory_variant"] = tier
    info["trajectory_variant_note"] = (
        "Our Tier 3 input: CARI4D monocular reconstruction of the exo video, aligned to the released "
        "world frame with Tier 1's first-frame object pose only (scripts/tier3/cari4d_to_lerobot.py). "
        "Meta, meshes, ground planes and support surfaces are the released Tier 2 files."
    )
    (root / "meta/info.json").write_text(json.dumps(info, indent=2))
    return root


def write_episode(root: Path, episode: int, human: dict, object_pose7: np.ndarray) -> Path:
    import pyarrow as pa
    import pyarrow.parquet as pq

    template = pq.read_table(TIER_2 / "data/chunk-000" / f"episode_{episode:06d}.parquet")
    frames = template.num_rows
    if len(object_pose7) != frames or len(human["poses"]) != frames:
        raise SystemExit(f"frame mismatch: Tier 2 has {frames}, CARI4D object {len(object_pose7)}, human {len(human['poses'])}")

    def fixed(values: np.ndarray, field: pa.Field) -> pa.Array:
        values = np.ascontiguousarray(values, dtype=np.float32).reshape(frames, -1)
        return pa.FixedSizeListArray.from_arrays(pa.array(values.reshape(-1), type=pa.float32()), values.shape[1])

    replace = {
        "observation.human.pose": human["poses"].reshape(frames, -1),
        "observation.human.translation": human["transl"],
        "observation.human.identity_coeffs": np.repeat(human["identity_coeffs"][None], frames, 0),
        "observation.human.scale_params": np.repeat(human["scale_params"][None], frames, 0),
        "observation.human.bone_length_flexibles": np.repeat(human["bone_length_flexibles"][None], frames, 0),
        "observation.object.pose": object_pose7,
    }
    columns = []
    for field in template.schema:
        if field.name in replace:
            array = fixed(replace[field.name], field)
            if array.type != field.type:
                raise SystemExit(f"{field.name}: built {array.type}, template has {field.type}")
            columns.append(array)
        elif field.name == "observation.object.visible":
            columns.append(pa.array(np.ones(frames, dtype=bool)))  # CARI4D tracks every frame
        else:
            columns.append(template.column(field.name))
    table = pa.Table.from_arrays(columns, schema=template.schema)
    out = root / "data/chunk-000" / f"episode_{episode:06d}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out)
    # pipeline.episodes.load_tier refuses a tier whose declared episodes are missing, so declare
    # only the episodes converted so far; the full 30 appear as CARI4D finishes them.
    rows = [line for line in (TIER_2 / "meta/episodes_metadata.jsonl").read_text().splitlines() if line.strip()]
    present = [r for r in rows if (root / "data/chunk-000" / f"episode_{json.loads(r)['episode_index']:06d}.parquet").is_file()]
    (root / "meta/episodes_metadata.jsonl").write_text("\n".join(present) + "\n")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--tier", default=DEFAULT_TIER)
    parser.add_argument("--cari4d-root", default=str(REPO / "outputs/tier3/cari4d"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-object-align-deg", type=float, default=20.0,
                        help="frame-0 human disagreement above which the human, not the object, places the scene")
    args = parser.parse_args()

    row = episode_row(args.episode)
    sequence = row["sequence_id"]
    root = Path(args.cari4d_root) / sequence
    if not (root / "pipeline_report.json").is_file() or json.loads((root / "pipeline_report.json").read_text()).get("verdict") != "PASS":
        print(f"episode {args.episode}: no passing CARI4D run at {root}")
        return 2
    import torch

    refined = torch.load(root / "inference/refined.pth", map_location="cpu", weights_only=False)

    camera_from_object = cari4d_object(root, sequence, refined)
    tier1_object0 = pose7_to_matrix(tier1_first_frame_object_pose(args.episode))
    world_from_camera = tier1_object0 @ np.linalg.inv(camera_from_object[0])

    vertices_cam, identity, spread = mhr_vertices(refined, args.device)
    to_soma = np.eye(4)
    to_soma[:3, :3] = CV_TO_SOMA
    soma_from_camera = to_soma @ world_from_camera
    to_frame = lambda T, v: v @ T[:3, :3].T + T[:3, 3]  # noqa: E731

    # Frame-0 check of the object-only alignment against Tier 1's frame-0 human (first frame only).
    # For an object with a symmetry, CARI4D may pick a different but equally valid orientation from
    # the labelled one, and the object-only transform then turns the whole scene with it (episodes
    # 3, 6, 9, 12, 14, 20, 27, 28: 53-178 deg; all others 0.5-4.8 deg). Past --max-object-align-deg
    # the scene is placed by the human instead; the object's orientation is relabelled (for a
    # symmetric object, by that symmetry) and its track offset so frame 0 equals Tier 1's pose.
    # A sibling import: inside cari4d.sif, CARI4D's own ``scripts`` package shadows ours.
    from frame0_alignment import Bodies, correction, tier1_first_frame_human

    bodies = Bodies(args.device)
    frame0 = soma_inversion(to_frame(soma_from_camera, vertices_cam[:1]), identity, args.device)
    ours0 = bodies.vertices({"pose": frame0["poses"][0].reshape(-1), "translation": frame0["transl"][0], **identity})
    fix, alignment = correction(ours0, bodies.vertices(tier1_first_frame_human(args.episode)))
    alignment["method"] = "object"
    world_from_object = world_from_camera[None] @ camera_from_object
    if alignment["rotation_deg"] > args.max_object_align_deg:
        soma_from_camera = fix @ soma_from_camera
        world_from_camera = to_soma @ soma_from_camera  # CV_TO_SOMA is its own inverse
        # Relabel the ORIENTATION only, about the object's own origin, then move the whole track by
        # one constant world offset so frame 0 equals Tier 1's frame-0 pose. A body-frame translation
        # here would swing with every orientation flip in CARI4D's track (episode 28: a 0.43 m lever
        # arm turned 0.42 m steps into a 0.65 m teleport).
        relabel = np.eye(4)
        relabel[:3, :3] = (np.linalg.inv(world_from_camera @ camera_from_object[0]) @ tier1_object0)[:3, :3]
        world_from_object = world_from_camera[None] @ camera_from_object @ relabel
        offset = tier1_object0[:3, 3] - world_from_object[0, :3, 3]
        world_from_object[:, :3, 3] += offset
        alignment.update(method="human", object_relabel_deg=round(float(np.degrees(np.arccos(
            np.clip((np.trace(relabel[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)))), 2),
            object_offset_m=round(float(np.linalg.norm(offset)), 3))
    print(f"episode {args.episode}: alignment {alignment}")
    object_pose7 = matrices_to_pose7(world_from_object)
    vertices_soma = to_frame(soma_from_camera, vertices_cam)
    human = soma_inversion(vertices_soma, identity, args.device)
    human.update(identity)

    names = json.loads((TIER_2 / "meta/info.json").read_text())["features"]["observation.human.pose"]["names"]
    expected = [n[:-3] for n in names[::3]]
    if human["joint_names"] != expected:
        print(f"SOMA rig joints {human['joint_names'][:3]}... do not match the dataset's {expected[:3]}...; refusing to write")
        return 2

    tier_root = init_tier(args.tier)
    written = write_episode(tier_root, args.episode, human, object_pose7)
    report = {
        "episode_index": args.episode, "sequence": sequence, "frames": len(object_pose7), "written": str(written.relative_to(REPO)),
        "identity_spread": spread, "alignment": alignment, "soma_fit_vertex_error_cm": human["fit_vertex_error_cm"],
        "reposed_vertex_error_cm_mean": human["reposed_vertex_error_cm_mean"],
    }
    report_path = tier_root / "conversion" / f"episode_{args.episode:06d}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
