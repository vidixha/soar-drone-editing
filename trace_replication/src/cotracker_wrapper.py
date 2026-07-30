"""
Thin wrapper around CoTracker (github.com/facebookresearch/co-tracker) to extract
the 25x25 grid of sparse point tracks the paper feeds into Stage 1 as the camera
motion signal. Step 3 of the Stage 1 data pipeline (see notes/stage1_spec.md).

Requires cotracker installed (pip install git+https://github.com/facebookresearch/co-tracker.git)
and its pretrained weights (loaded automatically via torch.hub in the common case).
"""
import torch
from cotracker.predictor import CoTrackerPredictor


def extract_point_track_grid(
    video: torch.Tensor,       # (1, T, 3, H, W), values in [0, 255]
    grid_size: int = 25,
    device: str = "cuda",
    checkpoint: str | None = None,
) -> torch.Tensor:
    """
    Returns (T, grid_size*grid_size, 2) xy pixel-coordinate tracks, matching the
    "grid of 25x25 point trajectories" described for Stage 1 training data.
    """
    predictor = CoTrackerPredictor(checkpoint=checkpoint) if checkpoint else CoTrackerPredictor()
    predictor = predictor.to(device)
    video = video.to(device)

    tracks, visibility = predictor(video, grid_size=grid_size)  # tracks: (1, T, N, 2)
    return tracks[0].detach().cpu()


def normalize_tracks(tracks: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """Normalize pixel xy tracks to [0, 1] so they're resolution-independent, matching
    how boxes are parameterized in stage1_dit.py."""
    out = tracks.clone()
    out[..., 0] /= width
    out[..., 1] /= height
    return out
