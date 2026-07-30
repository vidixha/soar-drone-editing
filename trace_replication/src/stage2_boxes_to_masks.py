"""
Renders dense per-frame box sequences into binary spatial mask videos, for the
two box-shaped conditioning signals Stage 2 needs (see notes/stage2_spec.md):
  - synthesis boxes M_obj: where the object should appear in the edited output
  - inpainting boxes M_inpaint: where the object was in the source, to be erased

Boxes are (cx, cy, w, h) normalized to [0, 1], same convention as stage1_dit.py,
so a Stage 1 prediction can feed directly into this without reparameterizing.
"""
import torch


def boxes_to_mask_video(
    boxes: torch.Tensor,   # (T, 4) normalized (cx, cy, w, h)
    height: int,
    width: int,
) -> torch.Tensor:
    """Returns (T, 1, H, W) float mask video, 1 inside each frame's box, 0 outside."""
    T = boxes.shape[0]
    ys = torch.linspace(0, 1, height).view(1, height, 1)
    xs = torch.linspace(0, 1, width).view(1, 1, width)

    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x0 = (cx - w / 2).view(T, 1, 1)
    x1 = (cx + w / 2).view(T, 1, 1)
    y0 = (cy - h / 2).view(T, 1, 1)
    y1 = (cy + h / 2).view(T, 1, 1)

    inside = (xs >= x0) & (xs <= x1) & (ys >= y0) & (ys <= y1)
    return inside.float().unsqueeze(1)  # (T, 1, H, W)


def apply_mask_to_video(video: torch.Tensor, inpaint_mask: torch.Tensor, fill_value: float = 0.0) -> torch.Tensor:
    """
    Builds V_mask by erasing the inpainting-box region from the source video.
    video: (T, 3, H, W) in [-1, 1] (Wan's usual normalization).
    inpaint_mask: (T, 1, H, W) from boxes_to_mask_video, using M_inpaint boxes.
    """
    return video * (1 - inpaint_mask) + fill_value * inpaint_mask


if __name__ == "__main__":
    boxes = torch.tensor([[0.5, 0.5, 0.2, 0.3], [0.6, 0.4, 0.2, 0.3]])
    m = boxes_to_mask_video(boxes, height=64, width=64)
    print("mask shape:", m.shape, "coverage frac:", m.mean().item())
