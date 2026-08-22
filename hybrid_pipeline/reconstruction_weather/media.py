"""Small media layer owned by reconstruction_weather."""

from __future__ import annotations

from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np


def load_video(path: Path) -> tuple[np.ndarray, float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"could not open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 24.0)
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    if not frames:
        raise ValueError(f"video contains no decodable frames: {path}")
    return np.stack(frames), fps


def center_frames_and_resize(
    video: np.ndarray,
    *,
    num_frames: int,
    width: int,
    height: int,
    sampling: str = "span",
) -> np.ndarray:
    if len(video) < num_frames:
        raise ValueError(f"video has {len(video)} frames; {num_frames} required")
    if sampling == "span":
        indices = np.linspace(0, len(video) - 1, num_frames)
        selected = video[np.rint(indices).astype(int)]
    elif sampling == "center":
        start = (len(video) - num_frames) // 2
        selected = video[start : start + num_frames]
    else:
        raise ValueError("frame sampling must be 'span' or 'center'")
    source_height, source_width = selected.shape[1:3]
    source_ratio = source_width / source_height
    target_ratio = width / height
    if source_ratio > target_ratio:
        crop_width = round(source_height * target_ratio)
        left = (source_width - crop_width) // 2
        selected = selected[:, :, left : left + crop_width]
    elif source_ratio < target_ratio:
        crop_height = round(source_width / target_ratio)
        top = (source_height - crop_height) // 2
        selected = selected[:, top : top + crop_height]
    return np.stack(
        [cv2.resize(frame, (width, height), interpolation=cv2.INTER_LANCZOS4) for frame in selected]
    )


def save_video(path: Path, video: np.ndarray, fps: float, *, quality: int = 9) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(path), fps=fps, quality=quality, macro_block_size=1)
    try:
        for frame in video:
            writer.append_data(frame)
    finally:
        writer.close()


def save_masks(folder: Path, masks: np.ndarray) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for index, mask in enumerate(masks):
        cv2.imwrite(str(folder / f"{index:05d}.png"), mask.astype(np.uint8) * 255)


def save_depth_preview(path: Path, video: np.ndarray, depths: np.ndarray, fps: float) -> None:
    finite = depths[np.isfinite(depths) & (depths > 0)]
    low, high = np.percentile(finite, (2, 98))
    normalized = np.clip((depths - low) / max(high - low, 1e-6), 0, 1)
    previews = []
    for frame, depth in zip(video, normalized):
        colour = cv2.applyColorMap((depth * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        previews.append(np.concatenate((frame, cv2.cvtColor(colour, cv2.COLOR_BGR2RGB)), axis=1))
    save_video(path, np.stack(previews), fps, quality=7)
