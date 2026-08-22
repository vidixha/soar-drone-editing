"""Direct Pi3X 4D reconstruction without Vista4D."""

from __future__ import annotations

from math import sqrt

import cv2
import numpy as np


def _processing_size(height: int, width: int, pixel_limit: int) -> tuple[int, int]:
    scale = sqrt(pixel_limit / (height * width)) if height * width > pixel_limit else 1.0
    grid_height = max(1, round(height * scale / 14))
    grid_width = max(1, round(width * scale / 14))
    while grid_height * 14 * grid_width * 14 > pixel_limit:
        if grid_height / grid_width > height / width:
            grid_height -= 1
        else:
            grid_width -= 1
    return grid_height * 14, grid_width * 14


def reconstruct_pi3x(
    video: np.ndarray,
    *,
    model_id: str = "yyfz233/Pi3X",
    pixel_limit: int = 255_000,
    device: str = "cuda",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return Pi3X depth, camera-to-world matrices, intrinsics, and confidence mask."""
    import torch
    import torch.nn.functional as functional
    from pi3.models.pi3x import Pi3X
    from pi3.utils.geometry import recover_intrinsic_from_rays_d

    frame_count, output_height, output_width, _ = video.shape
    process_height, process_width = _processing_size(
        output_height, output_width, pixel_limit
    )
    resized = np.stack(
        [
            cv2.resize(frame, (process_width, process_height), interpolation=cv2.INTER_LANCZOS4)
            for frame in video
        ]
    )
    frames = (
        torch.from_numpy(resized)
        .permute(0, 3, 1, 2)
        .float()
        .div(255)
        .unsqueeze(0)
        .to(device)
    )
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    print(f"[Pi3X] loading {model_id}", flush=True)
    model = Pi3X.from_pretrained(model_id).to(device).eval()
    model.disable_multimodal()
    print(
        f"[Pi3X] reconstructing {frame_count} frames at {process_width}x{process_height}",
        flush=True,
    )
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=dtype):
        result = model(imgs=frames)

    local_points = result["local_points"][0].float()
    rays = functional.normalize(local_points, dim=-1)
    matrices = recover_intrinsic_from_rays_d(
        rays, force_center_principal_point=True
    )
    depths = local_points[..., 2].cpu().numpy().astype(np.float32)
    cameras = result["camera_poses"][0].float().cpu().numpy().astype(np.float32)
    cameras = np.linalg.inv(cameras[0])[None] @ cameras
    confidence = torch.sigmoid(result["conf"][0][..., 0]).cpu().numpy() > 0.1
    matrices = matrices.cpu().numpy().astype(np.float32)

    if (process_height, process_width) != (output_height, output_width):
        depths = np.stack(
            [
                cv2.resize(frame, (output_width, output_height), interpolation=cv2.INTER_LINEAR)
                for frame in depths
            ]
        )
        confidence = np.stack(
            [
                cv2.resize(
                    frame.astype(np.uint8),
                    (output_width, output_height),
                    interpolation=cv2.INTER_NEAREST,
                )
                > 0
                for frame in confidence
            ]
        )
        matrices[:, 0, :] *= output_width / process_width
        matrices[:, 1, :] *= output_height / process_height
    intrinsics = np.stack(
        (
            matrices[:, 0, 0],
            matrices[:, 1, 1],
            matrices[:, 0, 2],
            matrices[:, 1, 2],
        ),
        axis=-1,
    )
    del model, result, frames, local_points
    torch.cuda.empty_cache()
    return depths, cameras.astype(np.float32), intrinsics.astype(np.float32), confidence
