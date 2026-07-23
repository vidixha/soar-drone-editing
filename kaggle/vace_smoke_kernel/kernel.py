"""Smoke test: does VACE (video insertion/editing diffusion model) install and run on
Kaggle's single T4 GPU. Not trying for a polished result yet, just proving the install +
model download + minimal inference path works, same spirit as the DROID-SLAM smoke test.

Uses the smallest available checkpoint (Wan2.1-based 1.3B, ~19GB total across VAE, T5
text encoder, and diffusion weights) with CPU-offload flags, since the underlying Wan2.1
model documents 8.19GB VRAM usage with --offload_model True --t5_cpu, comfortably under
a T4's 16GB.
"""

import os
import subprocess
import sys


def run(cmd, cwd=None):
    print(f"\n+ {cmd}", flush=True)
    result = subprocess.run(cmd, shell=True, cwd=cwd)
    print(f"  [exit code {result.returncode}]", flush=True)
    return result.returncode


print("=== GPU check ===", flush=True)
run("nvidia-smi")

print("\n=== clone VACE ===", flush=True)
WORK = "/kaggle/working/VACE"
rc = run(f"git clone https://github.com/ali-vilab/VACE.git {WORK}")
if rc != 0:
    print("FATAL: clone failed", flush=True)
    sys.exit(1)

print("\n=== checkpointing: all of this kernel's installs are prebuilt pip wheels, no "
      "local CUDA compilation (unlike the DROID-SLAM kernel), so they're safe to cache "
      "and reuse regardless of which GPU Kaggle hands out this session. Snapshotting "
      "site-packages before installing so only what these steps actually add gets "
      "cached, not the whole environment ===", flush=True)
import site
SITE_PACKAGES = site.getsitepackages()[0]
CHECKPOINT_IN = "/kaggle/input/vace-smoke-test/checkpoint"
CHECKPOINT_OUT = "/kaggle/working/checkpoint"
os.makedirs(CHECKPOINT_OUT, exist_ok=True)
before_install = set(os.listdir(SITE_PACKAGES))

install_cache_path = f"{CHECKPOINT_IN}/install_cache.tar.gz"
reuse_installs = os.path.exists(install_cache_path)
print(f"install cache present: {reuse_installs}", flush=True)

if reuse_installs:
    print("\n=== extracting cached installs, skipping torch/requirements/wan/flash_attn "
          "steps below ===", flush=True)
    run(f"tar -xzf {install_cache_path} -C {SITE_PACKAGES}")
else:
    print("\n=== install torch 2.5.1 + cu124 ===", flush=True)
    run("pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124")

    print("\n=== install VACE requirements ===", flush=True)
    rc = run("pip install -r requirements.txt", cwd=WORK)

    print("\n=== fix onnxruntime (requirements.txt pulled a GPU build expecting CUDA 13, "
          "we have CUDA 12.4; not actually needed since bbox mode never exercises the pose "
          "annotator this is imported for, so plain CPU onnxruntime is enough to satisfy "
          "the import) ===", flush=True)
    run("pip uninstall -y onnxruntime-gpu onnxruntime")
    run("pip install onnxruntime")
    print(f"requirements.txt install rc: {rc}", flush=True)

    print("\n=== install wan package ===", flush=True)
    rc = run("pip install 'wan@git+https://github.com/Wan-Video/Wan2.1'")
    print(f"wan install rc: {rc}", flush=True)

    print("\n=== flash_attn situation on this GPU (no FlashAttention support on some of "
          "Kaggle's free-tier GPUs, e.g. T4/Turing) ===", flush=True)
    # Uninstalling flash_attn (tried previously) does not help on its own: wan/modules/
    # model.py imports and calls flash_attention() directly with no fallback switch at
    # all, and that function asserts FLASH_ATTN_2_AVAILABLE and crashes immediately if
    # the package is missing, rather than falling back. The actual safe fallback lives
    # in a *different*, signature-compatible function, attention(), in the same module,
    # which model.py never calls; the source-file patch below handles that.
    run("pip uninstall -y flash-attn flash_attn")

    print("\n=== saving install checkpoint (only the new site-packages entries these "
          "steps added, not the whole environment) ===", flush=True)
    after_install = set(os.listdir(SITE_PACKAGES))
    new_entries = sorted(after_install - before_install)
    print(f"new site-packages entries to cache: {len(new_entries)}", flush=True)
    if new_entries:
        run(f"tar -czf {CHECKPOINT_OUT}/install_cache.tar.gz -C {SITE_PACKAGES} "
            f"{' '.join(new_entries)}")
        print(f"checkpoint saved: {os.listdir(CHECKPOINT_OUT)}", flush=True)

import torch
print(f"\ntorch {torch.__version__}, cuda {torch.version.cuda}, "
      f"available {torch.cuda.is_available()}", flush=True)

print("\n=== download VACE-Wan2.1-1.3B-Preview model (~19GB) ===", flush=True)
from huggingface_hub import snapshot_download, hf_hub_download
try:
    model_dir = snapshot_download(repo_id="ali-vilab/VACE-Wan2.1-1.3B-Preview",
                                   local_dir="/kaggle/working/model")
    print(f"model downloaded to {model_dir}", flush=True)
except Exception as e:
    print(f"FATAL: model download failed: {e}", flush=True)
    sys.exit(1)

print("\n=== download only the one annotator file bbox mode actually needs "
      "(salient/u2net.pt, 176MB), not the full 18.8GB VACE-Annotators repo ===", flush=True)
try:
    annot_path = hf_hub_download(repo_id="ali-vilab/VACE-Annotators", filename="salient/u2net.pt",
                                  local_dir=f"{WORK}/models/VACE-Annotators")
    print(f"downloaded {annot_path}", flush=True)
except Exception as e:
    print(f"FATAL: annotator download failed: {e}", flush=True)
    sys.exit(1)

print("\n=== build a short test clip (pulled directly from HF, not the Kaggle dataset "
      "mount, which failed to attach last run for unclear reasons) ===", flush=True)
import cv2
from huggingface_hub import hf_hub_download
frame_files = []
for i in range(1, 11):
    p = hf_hub_download(repo_id="Voxel51/visdrone-mot", repo_type="dataset",
                         filename=f"data/{i:07d}-3.jpg")
    frame_files.append(p)
print("input frames:", frame_files, flush=True)
first = cv2.imread(frame_files[0])
h, w = first.shape[:2]
test_video_path = "/kaggle/working/test_clip.mp4"
writer = cv2.VideoWriter(test_video_path, cv2.VideoWriter_fourcc(*"mp4v"), 4, (w, h))
for fn in frame_files:
    img = cv2.imread(fn)
    writer.write(img)
writer.release()
print(f"wrote {test_video_path} ({w}x{h}, {len(frame_files)} frames)", flush=True)

print("\n=== build a mask that follows a REAL per-frame trajectory instead of a fixed "
      "box in one place. Previous two attempts used a static box/region for all 10 "
      "frames, which is not how the object actually moves and is not how this will be "
      "used for real insertion later. This is the real ground-truth path of an actual "
      "moving car (track 97) in this exact clip, one box per frame, not a placeholder. "
      "Tests directly whether accurate per-frame placement changes VACE's output "
      "compared to the previous two arbitrary-box attempts ===", flush=True)
import numpy as np
# Real per-frame absolute boxes (x, y, w, h) for track 97 (a car) across frames 1-10 of
# this exact scene, pulled from VisDrone ground truth via src/data_loader.py. This is a
# genuine trajectory, not a guess: the car visibly drives through the intersection.
real_boxes = [
    [1808.0, 760.0, 231.0, 129.0], [1601.0, 661.0, 230.0, 128.0], [1573.0, 655.0, 213.0, 115.0],
    [1548.0, 644.0, 219.0, 119.0], [1523.0, 633.0, 225.0, 124.0], [1499.0, 623.0, 230.0, 128.0],
    [1479.0, 614.0, 230.0, 128.0], [1459.0, 606.0, 230.0, 128.0], [1439.0, 597.0, 230.0, 128.0],
    [1419.0, 589.0, 230.0, 128.0],
]
mask_video_path = "/kaggle/working/mask_clip.mp4"
mask_writer = cv2.VideoWriter(mask_video_path, cv2.VideoWriter_fourcc(*"mp4v"), 4, (w, h), isColor=False)
for bx, by, bw, bh in real_boxes:
    mask_frame = np.zeros((h, w), dtype=np.uint8)
    mask_frame[int(by):int(by + bh), int(bx):int(bx + bw)] = 255
    mask_writer.write(mask_frame)
mask_writer.release()
print(f"wrote {mask_video_path}, {len(real_boxes)} frames, real per-frame boxes "
      f"(track 97, moves ~426px total across the clip)", flush=True)

print("\n=== patch the installed wan package directly instead of wrapping it: "
      "wan/modules/model.py imports flash_attention() with no fallback switch at all, "
      "and that function asserts FLASH_ATTN_2_AVAILABLE and crashes if the package is "
      "absent (which we need it to be, T4 can't run it). The safe fallback is a "
      "different, signature-compatible function, attention(), in the same module, that "
      "model.py never calls. A runpy-based wrapper script was tried first but broke the "
      "package's internal relative-import path resolution (ModuleNotFoundError:  "
      "models.wan); patching the installed file directly avoids that entirely, since "
      "the normal, unmodified invocation of vace_wan_inference.py still runs exactly as "
      "before. ===", flush=True)
import wan.modules.model as _wan_model_check
wan_model_path = _wan_model_check.__file__
print(f"patching {wan_model_path}", flush=True)
with open(wan_model_path) as f:
    content = f.read()
content = content.replace(
    "from .attention import flash_attention",
    "from .attention import attention as flash_attention",
)
with open(wan_model_path, "w") as f:
    f.write(content)
print("patched: model.py now imports the safe attention() wrapper under the "
      "flash_attention name, so every call site (which never changes) gets the "
      "hardware-appropriate fallback automatically.", flush=True)

print("\n=== run VACE's inference script directly (skips vace_pipeline.py's "
      "auto-preprocessing and Grounding DINO too) ===", flush=True)
# Attempt 3: the mask now follows track 97's real path (a car actually driving through
# this intersection), so the prompt describes the same kind of object that is really
# there and really moving along exactly this path, isolating the one variable this test
# is about (does accurate per-frame placement change the outcome) rather than also
# changing what kind of object is being asked for.
prompt = (
    "a dark sedan car driving through a night city intersection, seen from directly "
    "above from an aerial drone camera, headlights on, moving smoothly along the road, "
    "lit by warm orange streetlights and reflections on the road surface, "
    "casting a short shadow, photorealistic, sharp focus, "
    "consistent with the surrounding night traffic scene"
)
rc = run(
    f"python vace/vace_wan_inference.py --model_name vace-1.3B "
    f"--ckpt_dir /kaggle/working/model "
    f"--src_video {test_video_path} --src_mask {mask_video_path} "
    f"--prompt \"{prompt}\" "
    f"--save_dir /kaggle/working/output "
    f"--offload_model True --t5_cpu",
    cwd=WORK,
)
print(f"\n=== vace_wan_inference.py result: {'SUCCESS' if rc == 0 else 'FAILED, rc=' + str(rc)} ===", flush=True)

print("\n=== output search ===", flush=True)
run("find /kaggle/working/output -type f")
run("find /kaggle/working -maxdepth 2 -type d")

print("\n=== DONE ===", flush=True)
