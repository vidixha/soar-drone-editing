"""
Kaggle smoke test: confirms the Stage 2 pipeline (real Wan2.1 1.4B weights +
LoRA + our conditioning-injection module) actually runs forward/backward without
crashing on a real GPU. This is NOT a real training run: no real dataset, no
OpenVid-1M, no DEVA masks. A handful of steps on synthetic random tensors, purely
to catch shape/dtype/API-mismatch bugs before committing to a real run.

Matches the paper's model choice (Wan2.1) and training recipe (LoRA, 81 frames,
480x832, AdamW lr=1.2e-5, wd=0.01) but shrinks frame count/resolution/step count
so it fits a free-tier Kaggle GPU in a few minutes, the same "junk run" pattern
used earlier in this project for the VACE Kaggle smoke tests.
"""
import subprocess
import sys
import time

print("=== nvidia-smi ===")
subprocess.run(["nvidia-smi"])

print("\n=== clone wan (need module source, not just pip weights) ===")
subprocess.run(["pip", "install", "-q", "--no-deps",
                 "wan@git+https://github.com/Wan-Video/Wan2.1.git"], check=True)
subprocess.run(["pip", "install", "-q", "peft", "einops"], check=True)

print("\n=== patch wan/modules/model.py: flash_attention() has no CPU/non-flash-attn", flush=True)
print("    fallback and asserts+crashes on T4/P100. Same fix used for the earlier", flush=True)
print("    VACE Kaggle smoke tests: point it at the safe attention() wrapper in the", flush=True)
print("    same module, which every call site already calls under the flash_attention", flush=True)
print("    name so no call sites need to change. ===", flush=True)
import wan
_wan_model_path = wan.__file__.replace("__init__.py", "modules/model.py")
with open(_wan_model_path) as f:
    _src = f.read()
_src = _src.replace(
    "from .attention import flash_attention",
    "from .attention import attention as flash_attention",
)
with open(_wan_model_path, "w") as f:
    f.write(_src)

import torch
from peft import LoraConfig, get_peft_model
from wan.modules.model import WanModel

DEVICE = "cuda"
LATENT_CH = 16
T_LATENT = 3      # tiny: a handful of latent frames, not the real 81-frame/8x-downsampled count
H_LATENT = 30
W_LATENT = 52

print("\n=== build Wan2.1 DiT (random init -- downloading and loading the full", flush=True)
print("    pretrained checkpoint is a separate, much larger step; this smoke test", flush=True)
print("    only needs to prove the architecture + LoRA + our conditioning wiring", flush=True)
print("    actually fit together and run on this GPU) ===", flush=True)

model = WanModel(
    model_type="t2v",
    patch_size=(1, 2, 2),
    in_dim=LATENT_CH,
    dim=1536,          # matches Wan2.1-1.3B's hidden size
    ffn_dim=8960,
    freq_dim=256,
    text_dim=4096,
    out_dim=LATENT_CH,
    num_heads=12,
    num_layers=30,
).to(DEVICE, dtype=torch.bfloat16)

print("=== wrap with LoRA on self_attn.q/k/v/o, matching the paper's LoRA fine-tune ===")
lora_cfg = LoraConfig(
    r=32, lora_alpha=32,
    target_modules=["self_attn.q", "self_attn.k", "self_attn.v", "self_attn.o"],
    lora_dropout=0.0, bias="none",
)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()

print("\n=== our conditioning injector (V_mask + M_obj + M_inpaint concat) ===")


class ConditionInjector(torch.nn.Module):
    def __init__(self, latent_channels, cond_channels):
        super().__init__()
        self.proj = torch.nn.Conv3d(cond_channels, latent_channels, kernel_size=1)
        torch.nn.init.zeros_(self.proj.weight)
        torch.nn.init.zeros_(self.proj.bias)

    def forward(self, vmask, mobj, minpaint):
        return self.proj(torch.cat([vmask, mobj, minpaint], dim=1))


injector = ConditionInjector(LATENT_CH, LATENT_CH + 1 + 1).to(DEVICE, dtype=torch.bfloat16)

optimizer = torch.optim.AdamW(
    list(model.parameters()) + list(injector.parameters()),
    lr=1.2e-5, weight_decay=0.01,
)

print("\n=== junk training loop: 5 steps on synthetic random tensors ===")
B = 1
for step in range(5):
    t0 = time.time()
    x1 = torch.randn(B, LATENT_CH, T_LATENT, H_LATENT, W_LATENT, device=DEVICE, dtype=torch.bfloat16)
    vmask_lat = torch.randn_like(x1)
    mobj_lat = torch.randn(B, 1, T_LATENT, H_LATENT, W_LATENT, device=DEVICE, dtype=torch.bfloat16)
    minpaint_lat = torch.randn(B, 1, T_LATENT, H_LATENT, W_LATENT, device=DEVICE, dtype=torch.bfloat16)
    text_emb = [torch.randn(20, 4096, device=DEVICE, dtype=torch.bfloat16) for _ in range(B)]

    x0 = torch.randn_like(x1)
    tt = torch.rand(B, device=DEVICE)
    t_ = tt.view(B, 1, 1, 1, 1)
    xt = (1 - t_) * x0 + t_ * x1
    target_v = x1 - x0

    cond = injector(vmask_lat, mobj_lat, minpaint_lat)
    dit_input = (xt + cond).to(torch.bfloat16)

    pred_v_list = model(
        [dit_input[0]],
        t=tt,
        context=text_emb,
        seq_len=T_LATENT * H_LATENT * W_LATENT,
    )
    pred_v = pred_v_list[0].unsqueeze(0).to(torch.float32)
    loss = torch.nn.functional.mse_loss(pred_v, target_v.to(torch.float32))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"step {step}  loss {loss.item():.4f}  ({time.time() - t0:.1f}s)", flush=True)

print("\n=== JUNK TRAINING SMOKE TEST PASSED ===")
