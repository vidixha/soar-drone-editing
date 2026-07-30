"""
Stage 2: Wan2.1 1.4B LoRA fine-tune for motion-conditioned video resynthesis.

Reimplementation from the paper's method paragraph (arxiv 2603.25707); no official
code or checkpoint exists. Conditioning-injection mechanism (channel-concat of
V_mask / M_obj / M_inpaint latents with the noised video latent) is OUR design
choice to fill a gap the paper doesn't specify, modeled on VACE's own conditioning
pattern -- see notes/stage2_spec.md for the reasoning, don't mistake this for a
reported architectural detail.

Requires, on the machine actually running this (not this dev machine):
  the `wan` package (already used in aerial_box_propagation/modal/vace_modal.py:
  pip install --no-deps 'wan@git+https://github.com/Wan-Video/Wan2.1.git')
  peft (for LoRA injection)
  Wan2.1-T2V-1.3B pretrained weights + VAE + T5 text encoder
"""
import torch
import torch.nn as nn

from stage2_boxes_to_masks import apply_mask_to_video, boxes_to_mask_video

LORA_TARGET_MODULES = ["self_attn.q", "self_attn.k", "self_attn.v", "self_attn.o"]


def build_lora_wan_dit(wan_dit: nn.Module, rank: int = 32, alpha: int = 32):
    """Wraps a pretrained Wan2.1 DiT's attention projections with LoRA adapters,
    freezing the base weights. Requires `peft`."""
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=0.0,
        bias="none",
    )
    return get_peft_model(wan_dit, config)


class ConditionInjector(nn.Module):
    """
    Encodes V_mask, M_obj mask video, M_inpaint mask video (all already VAE-latent
    shaped) and projects their concatenation down to the DiT's expected input
    channel count, so the noised latent + conditioning can be summed before the
    first transformer block. Mirrors VACE's own "extra context channels" pattern.
    """

    def __init__(self, latent_channels: int, cond_channels: int):
        super().__init__()
        # cond_channels = latent_channels (V_mask) + 1 (M_obj mask, single channel,
        # broadcast/VAE-encoded upstream) + 1 (M_inpaint mask)
        self.proj = nn.Conv3d(cond_channels, latent_channels, kernel_size=1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, vmask_latent: torch.Tensor, mobj_latent: torch.Tensor, minpaint_latent: torch.Tensor) -> torch.Tensor:
        cond = torch.cat([vmask_latent, mobj_latent, minpaint_latent], dim=1)  # channel dim
        return self.proj(cond)


def build_conditioning(
    source_video: torch.Tensor,   # (T, 3, H, W) in [-1, 1]
    synth_boxes: torch.Tensor,    # (T, 4) normalized, M_obj -- the NEW/edited path
    inpaint_boxes: torch.Tensor,  # (T, 4) normalized, M_inpaint -- the ORIGINAL location
) -> dict:
    """Builds the pixel/mask-space conditioning tensors; VAE-encoding into latents
    happens with the actual Wan VAE at train/inference time, not here."""
    T, _, H, W = source_video.shape
    inpaint_mask = boxes_to_mask_video(inpaint_boxes, H, W)
    synth_mask = boxes_to_mask_video(synth_boxes, H, W)
    masked_video = apply_mask_to_video(source_video, inpaint_mask)
    return {
        "masked_video": masked_video,      # V_mask, feed through Wan's VAE
        "synth_mask": synth_mask,          # M_obj, feed through Wan's VAE (or a lighter encoder)
        "inpaint_mask": inpaint_mask,      # M_inpaint, feed through Wan's VAE (or a lighter encoder)
    }


def train_step(
    wan_pipe,               # loaded Wan2.1 pipeline (VAE + T5 + LoRA-wrapped DiT)
    injector: ConditionInjector,
    batch: dict,
    optimizer: torch.optim.Optimizer,
) -> float:
    """
    One flow-matching training step, matching the paper's stated recipe:
    AdamW lr=1.2e-5, wd=0.01, batch 32, 81-frame clips @ 480x832 @ 24fps, 8k steps.
    This function assumes `batch` already contains VAE-encoded latents (encoding
    the actual video is the expensive step and belongs in the dataloader, not
    here) -- see notes/stage2_spec.md for what still needs deciding around the
    training data source.
    """
    x1 = batch["target_latent"]         # (B, C, T, H, W) ground-truth clean latent
    vmask_lat = batch["vmask_latent"]
    mobj_lat = batch["mobj_latent"]
    minpaint_lat = batch["minpaint_latent"]
    text_emb = batch["text_emb"]
    first_frame_lat = batch["first_frame_latent"]

    B = x1.shape[0]
    x0 = torch.randn_like(x1)
    t = torch.rand(B, device=x1.device)
    t_ = t.view(B, 1, 1, 1, 1)
    xt = (1 - t_) * x0 + t_ * x1
    target_v = x1 - x0

    cond_inject = injector(vmask_lat, mobj_lat, minpaint_lat)
    dit_input = xt + cond_inject
    pred_v = wan_pipe.dit(dit_input, t, text_emb=text_emb, first_frame_latent=first_frame_lat)

    loss = torch.nn.functional.mse_loss(pred_v, target_v)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


if __name__ == "__main__":
    T, H, W = 4, 32, 32
    src = torch.rand(T, 3, H, W) * 2 - 1
    synth = torch.tensor([[0.5, 0.5, 0.2, 0.3]] * T)
    inpaint = torch.tensor([[0.3, 0.3, 0.2, 0.3]] * T)
    cond = build_conditioning(src, synth, inpaint)
    for k, v in cond.items():
        print(k, v.shape)

    injector = ConditionInjector(latent_channels=16, cond_channels=16 + 1 + 1)
    fake_vmask_lat = torch.randn(2, 16, T, H, W)
    fake_mobj_lat = torch.randn(2, 1, T, H, W)
    fake_minpaint_lat = torch.randn(2, 1, T, H, W)
    out = injector(fake_vmask_lat, fake_mobj_lat, fake_minpaint_lat)
    print("injector output:", out.shape)
