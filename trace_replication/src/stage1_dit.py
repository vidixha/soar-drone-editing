"""
Stage 1 model: Cross-View Motion Transformation.

Reimplementation from the TRACE paper's method description (arxiv 2603.25707).
No official code or checkpoint exists for this model; nothing here is ported from
a reference implementation.

Given: first frame (visual context), a 25x25 grid of point tracks across T frames
(camera motion signal), and a dense per-frame reference box sequence B_ref.
Predicted: a dense per-frame target box sequence, via flow matching.

Boxes are parameterized as (cx, cy, w, h) normalized to [0, 1] by frame size.
"""
import math

import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t[:, None].float() * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2:
            emb = torch.nn.functional.pad(emb, (0, 1))
        return emb


class DiTBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )
        # AdaLN-zero style conditioning on the timestep + context embedding
        self.ada_ln = nn.Linear(dim, 6 * dim)
        nn.init.zeros_(self.ada_ln.weight)
        nn.init.zeros_(self.ada_ln.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        shift1, scale1, gate1, shift2, scale2, gate2 = self.ada_ln(cond).chunk(6, dim=-1)
        h = self.norm1(x) * (1 + scale1.unsqueeze(1)) + shift1.unsqueeze(1)
        h, _ = self.attn(h, ctx, ctx, need_weights=False)
        x = x + gate1.unsqueeze(1) * h
        h = self.norm2(x) * (1 + scale2.unsqueeze(1)) + shift2.unsqueeze(1)
        h = self.mlp(h)
        x = x + gate2.unsqueeze(1) * h
        return x


class CrossViewMotionDiT(nn.Module):
    """
    8-layer DiT matching the paper's stated depth. Depth/width otherwise unstated
    in the paper; defaults below are a reasonable, trainable-on-modest-GPU choice,
    not a claim about the paper's actual hyperparameters.
    """

    def __init__(
        self,
        dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 8,
        num_frames: int = 21,   # paper samples cam poses every 4th of 81 frames -> 21
        grid_size: int = 25,    # CoTracker point grid is 25x25
        box_dim: int = 4,       # (cx, cy, w, h)
        first_frame_dim: int = 768,  # placeholder 3D-VAE / vision-encoder feature dim
    ):
        super().__init__()
        self.dim = dim
        self.num_frames = num_frames
        self.grid_size = grid_size

        self.time_embed = SinusoidalTimeEmbedding(dim)
        self.time_mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

        # context tokens: first-frame visual features + flattened point-track grid
        self.first_frame_proj = nn.Linear(first_frame_dim, dim)
        # point tracks: (T, grid*grid, 2) xy coords -> per-track-point token, summed over T via linear
        self.point_track_proj = nn.Linear(num_frames * 2, dim)

        # noisy box sequence tokens, one token per frame
        self.box_in_proj = nn.Linear(box_dim, dim)
        self.frame_pos_embed = nn.Parameter(torch.randn(1, num_frames, dim) * 0.02)

        # reference box sequence is concatenated as extra conditioning per frame token
        self.ref_box_proj = nn.Linear(box_dim, dim)

        self.blocks = nn.ModuleList([DiTBlock(dim, num_heads) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(dim)
        self.box_out_proj = nn.Linear(dim, box_dim)

    def forward(
        self,
        noisy_boxes: torch.Tensor,      # (B, T, 4) = X_t
        t: torch.Tensor,                # (B,) flow-matching timestep in [0, 1]
        first_frame_feat: torch.Tensor, # (B, first_frame_dim)
        point_tracks: torch.Tensor,     # (B, T, grid*grid, 2)
        ref_boxes: torch.Tensor,        # (B, T, 4) = B_ref
    ) -> torch.Tensor:
        B, T, _ = noisy_boxes.shape

        time_cond = self.time_mlp(self.time_embed(t))  # (B, dim)

        first_tok = self.first_frame_proj(first_frame_feat).unsqueeze(1)  # (B, 1, dim)
        tracks_flat = point_tracks.permute(0, 2, 1, 3).reshape(B, self.grid_size * self.grid_size, T * 2)
        track_toks = self.point_track_proj(tracks_flat)  # (B, grid*grid, dim)
        ctx = torch.cat([first_tok, track_toks], dim=1)  # (B, 1 + grid*grid, dim)

        x = self.box_in_proj(noisy_boxes) + self.frame_pos_embed[:, :T] + self.ref_box_proj(ref_boxes)

        for block in self.blocks:
            x = block(x, time_cond, ctx)

        x = self.final_norm(x)
        return self.box_out_proj(x)  # predicted velocity, (B, T, 4)


def flow_matching_loss(
    model: CrossViewMotionDiT,
    x1_boxes: torch.Tensor,         # (B, T, 4) ground-truth target boxes
    first_frame_feat: torch.Tensor,
    point_tracks: torch.Tensor,
    ref_boxes: torch.Tensor,
) -> torch.Tensor:
    B = x1_boxes.shape[0]
    x0 = torch.randn_like(x1_boxes)
    t = torch.rand(B, device=x1_boxes.device)
    t_ = t.view(B, 1, 1)
    xt = (1 - t_) * x0 + t_ * x1_boxes
    target_v = x1_boxes - x0
    pred_v = model(xt, t, first_frame_feat, point_tracks, ref_boxes)
    return torch.nn.functional.mse_loss(pred_v, target_v)


@torch.no_grad()
def sample(
    model: CrossViewMotionDiT,
    first_frame_feat: torch.Tensor,
    point_tracks: torch.Tensor,
    ref_boxes: torch.Tensor,
    num_steps: int = 50,
) -> torch.Tensor:
    """Euler-integrate the learned velocity field from noise to the predicted box sequence."""
    B, T, _ = ref_boxes.shape
    x = torch.randn(B, T, 4, device=ref_boxes.device)
    dt = 1.0 / num_steps
    for step in range(num_steps):
        t = torch.full((B,), step * dt, device=ref_boxes.device)
        v = model(x, t, first_frame_feat, point_tracks, ref_boxes)
        x = x + v * dt
    return x


if __name__ == "__main__":
    m = CrossViewMotionDiT()
    B, T, G = 2, 21, 25
    loss = flow_matching_loss(
        m,
        x1_boxes=torch.rand(B, T, 4),
        first_frame_feat=torch.randn(B, 768),
        point_tracks=torch.rand(B, T, G * G, 2),
        ref_boxes=torch.rand(B, T, 4),
    )
    print("smoke test loss:", loss.item())
