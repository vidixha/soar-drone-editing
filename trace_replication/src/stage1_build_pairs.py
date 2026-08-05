"""
Assembles Stage 1 training pairs (.pt files matching stage1_train.py's
Stage1PairDataset schema) by chaining:
  mavrec_source_filter.filter_mavrec()  -> qualifying hovering-drone scenes
  recam_wrapper.render_all_trajectories() -> 10 ReCamMaster-rendered variants per scene
  cotracker_wrapper.extract_point_track_grid() -> 25x25 point tracks per rendered clip
  a first-frame feature extractor (placeholder; paper doesn't specify the encoder)

This is the glue script stage1_source_filter.py and stage1_train.py's docstrings
both flagged as not yet built. It did not exist before this POC.

render_fn / track_fn / feature_fn are injectable so this can be smoke-tested on
CPU without ReCamMaster's or CoTracker's real (GPU-only) models: pass in stub
callables with the same signature as the real ones. The __main__ block below
does exactly that, on synthetic data, to prove the assembly and .pt schema are
correct before spending GPU time. Swapping to the real functions is passing
recam_wrapper.render_all_trajectories and cotracker_wrapper.extract_point_track_grid
directly -- no other change needed.
"""
import pathlib
from typing import Callable

import numpy as np
import torch

TARGET_FRAMES = 81
POSE_SAMPLE_STRIDE = 4   # paper samples cam poses every 4th of 81 frames -> 21, matching stage1_dit.py's num_frames default
GRID_SIZE = 25


def interpolate_ref_boxes(boxes_normalized: np.ndarray, num_frames: int) -> torch.Tensor:
    """B_ref: dense per-frame reference boxes, built by temporally interpolating the
    sparse first-frame box across the clip (paper: 'sparse user-placed key boxes'
    interpolated to B_ref; here we only have the single source-frame box available
    pre-rendering, so we hold it constant, which is the degenerate single-keybox
    case of that interpolation)."""
    first_box = boxes_normalized[0]
    return torch.from_numpy(np.tile(first_box, (num_frames, 1))).float()


def sample_pose_frames(tracks: torch.Tensor, stride: int = POSE_SAMPLE_STRIDE) -> torch.Tensor:
    return tracks[::stride]


def build_one_pair(
    seq_name: str,
    boxes_normalized: np.ndarray,
    rendered_clip_path: pathlib.Path,
    cam_type: str,
    render_fn: Callable,
    track_fn: Callable,
    feature_fn: Callable,
    device: str,
) -> dict:
    """Assembles a single (source scene, cam_type) training pair into the
    stage1_train.py .pt schema. render_fn is assumed to have already produced
    rendered_clip_path (rendering all 10 cam_types per scene at once is cheaper
    than per-pair, see build_all_pairs); this function handles tracking, feature
    extraction, and target-box placeholder assembly for one rendered clip."""
    tracks = track_fn(rendered_clip_path, grid_size=GRID_SIZE, device=device)  # (T, grid*grid, 2)
    pose_tracks = sample_pose_frames(tracks)  # (num_frames, grid*grid, 2)

    first_frame_feat = feature_fn(rendered_clip_path)  # (768,)
    ref_boxes = interpolate_ref_boxes(boxes_normalized, pose_tracks.shape[0])

    # Target boxes (ground truth in the rendered/moved-camera view) require
    # re-localizing the object in the rendered clip -- notes/stage1_spec.md flags
    # this as unspecified by the paper, DEVA being our planned candidate. DEVA has
    # not been run yet (see w2/trace_replication README), so target_boxes here are
    # a placeholder equal to ref_boxes, valid only for shape/wiring checks, not for
    # actual training signal.
    target_boxes = ref_boxes.clone()

    return {
        "first_frame_feat": first_frame_feat,
        "point_tracks": pose_tracks,
        "ref_boxes": ref_boxes,
        "target_boxes": target_boxes,
        "seq_name": seq_name,
        "cam_type": cam_type,
    }


def build_all_pairs(
    qualifying: list[dict],
    recammaster_repo: pathlib.Path,
    dataset_path: pathlib.Path,
    render_output_dir: pathlib.Path,
    ckpt_path: pathlib.Path,
    out_dir: pathlib.Path,
    render_fn: Callable,
    track_fn: Callable,
    feature_fn: Callable,
    device: str = "cuda",
) -> list[pathlib.Path]:
    """qualifying: output of mavrec_source_filter.filter_mavrec(). Assumes each
    qualifying['seq_name'] has already been encoded to '<seq_name>.mp4' under
    dataset_path (frame->video encoding is a separate ffmpeg step) and
    dataset_path/metadata.csv exists per recam_wrapper.build_metadata_csv."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cam_dirs = render_fn(recammaster_repo, dataset_path, render_output_dir, ckpt_path)

    written = []
    by_seq = {item["seq_name"]: item for item in qualifying}
    for cam_dir in cam_dirs:
        cam_type = cam_dir.name.replace("cam_type", "")
        for item in by_seq.values():
            rendered_clip_path = cam_dir / f"{item['seq_name']}.mp4"
            if not rendered_clip_path.exists():
                continue
            pair = build_one_pair(
                item["seq_name"], item["boxes_normalized"], rendered_clip_path,
                cam_type, render_fn, track_fn, feature_fn, device,
            )
            out_path = out_dir / f"{item['seq_name']}_cam{cam_type}.pt"
            torch.save(pair, out_path)
            written.append(out_path)
    return written


def _stub_render_fn(recammaster_repo, dataset_path, output_dir, ckpt_path):
    """Stands in for recam_wrapper.render_all_trajectories: writes empty placeholder
    files instead of actually running ReCamMaster, for the CPU smoke test."""
    produced = []
    for cam_type in [str(i) for i in range(1, 11)]:
        cam_dir = output_dir / f"cam_type{cam_type}"
        cam_dir.mkdir(parents=True, exist_ok=True)
        for mp4_path in dataset_path.glob("*.mp4"):
            (cam_dir / mp4_path.name).touch()
        produced.append(cam_dir)
    return produced


def _stub_track_fn(video_path, grid_size=25, device="cuda"):
    """Stands in for cotracker_wrapper.extract_point_track_grid."""
    return torch.rand(TARGET_FRAMES, grid_size * grid_size, 2)


def _stub_feature_fn(video_path):
    """Stands in for a first-frame vision-encoder feature extractor (unspecified
    by the paper; stage1_dit.py's first_frame_dim=768 placeholder)."""
    return torch.randn(768)


if __name__ == "__main__":
    import shutil
    import tempfile

    tmp = pathlib.Path(tempfile.mkdtemp())
    try:
        dataset_path = tmp / "dataset"
        dataset_path.mkdir()
        (dataset_path / "scene_0001.mp4").touch()
        (dataset_path / "scene_0002.mp4").touch()

        qualifying = [
            {"seq_name": "scene_0001", "boxes_normalized": np.tile([0.5, 0.5, 0.1, 0.1], (TARGET_FRAMES, 1))},
            {"seq_name": "scene_0002", "boxes_normalized": np.tile([0.4, 0.6, 0.15, 0.2], (TARGET_FRAMES, 1))},
        ]

        written = build_all_pairs(
            qualifying,
            recammaster_repo=tmp, dataset_path=dataset_path,
            render_output_dir=tmp / "rendered", ckpt_path=tmp / "fake.ckpt",
            out_dir=tmp / "pairs",
            render_fn=_stub_render_fn, track_fn=_stub_track_fn, feature_fn=_stub_feature_fn,
            device="cpu",
        )
        print(f"smoke test wrote {len(written)} pairs (expect {2 * 10})")

        sample = torch.load(written[0])
        for key in ("first_frame_feat", "point_tracks", "ref_boxes", "target_boxes"):
            print(f"  {key}: {tuple(sample[key].shape)}")

        from stage1_dit import CrossViewMotionDiT, flow_matching_loss
        model = CrossViewMotionDiT()
        batch = [torch.load(p) for p in written[:4]]
        loss = flow_matching_loss(
            model,
            x1_boxes=torch.stack([b["target_boxes"] for b in batch]),
            first_frame_feat=torch.stack([b["first_frame_feat"] for b in batch]),
            point_tracks=torch.stack([b["point_tracks"] for b in batch]),
            ref_boxes=torch.stack([b["ref_boxes"] for b in batch]),
        )
        print(f"end-to-end CPU loss on assembled pairs: {loss.item():.4f}")
    finally:
        shutil.rmtree(tmp)
