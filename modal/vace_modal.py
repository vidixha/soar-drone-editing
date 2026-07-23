"""VACE on Modal, ported from Kaggle after the GPU lottery bit us the same way it did
DROID-SLAM: Kaggle handed one run a P100 that our pinned torch build doesn't have
compatible CUDA kernels for ("no kernel image is available for execution on the
device"), identical failure class to DROID-SLAM's. Pinning gpu="T4" here removes that
uncertainty entirely, same fix as the DROID-SLAM port.

This is attempt 3 of the actual experiment: does giving VACE a mask that follows a REAL
per-frame object trajectory (not an arbitrary fixed box) change its output, compared to
the first two attempts which both used a static box. The mask below is track 97's real
ground-truth path (a car driving through this exact intersection clip), pulled from
VisDrone via src/data_loader.py.
"""

import os

import modal

app = modal.App("vace-trajectory-test")

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0", "g++")
    .env({"CC": "gcc", "CXX": "g++"})
    .run_commands("git clone https://github.com/ali-vilab/VACE.git /root/VACE")
    .pip_install(
        "torch==2.5.1", "torchvision==0.20.1",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    # requirements.txt tries to build flash_attn from source, which we don't want at
    # all (T4 doesn't support FlashAttention, and we handle the fallback ourselves by
    # patching wan's import later) and which fails to build here regardless (its own
    # setup.py needs 'packaging' present without build isolation, a separate problem
    # from ours). Strip it out before installing rather than fighting its build.
    .run_commands(
        "cd /root/VACE && grep -Ev 'flash[-_]attn' requirements/framework.txt > /tmp/fw.txt "
        "&& cp /tmp/fw.txt requirements/framework.txt "
        # --no-build-isolation: some other package's build backend imports torch to
        # build its wheel; pip's isolated build env doesn't see the torch already
        # installed above without this flag.
        "&& pip install --no-build-isolation -r requirements.txt"
    )
    # requirements.txt pulls an onnxruntime-gpu build expecting CUDA 13, we have 12.4;
    # not actually needed since we never exercise the pose/GDINO annotators (we build
    # our own mask directly), plain CPU onnxruntime is enough to satisfy the import.
    .run_commands("pip uninstall -y onnxruntime-gpu onnxruntime")
    .pip_install("onnxruntime")
    # wan's own setup.py also lists flash_attn as a dependency (same problem as VACE's
    # requirements.txt, and installing it is pointless anyway on a T4). --no-deps skips
    # wan's declared dependencies entirely; everything it actually needs (torch,
    # transformers, diffusers, etc) is already installed from requirements.txt above,
    # confirmed by "Requirement already satisfied" for all of them except flash_attn.
    .run_commands("pip install --no-deps 'wan@git+https://github.com/Wan-Video/Wan2.1'")
    .pip_install("huggingface_hub", "opencv-python-headless", "numpy", "matplotlib")
)


# modal.is_local() guards this: Modal re-imports this module inside the remote
# container too, just to locate the function object, and that container has no local
# HF token file at all. Reading it unconditionally crashed the remote import itself
# before the function could even run. Only build the Secret in the local process.
if modal.is_local():
    with open(os.path.expanduser("~/.cache/huggingface/token")) as _f:
        _hf_token = _f.read().strip()
    hf_secret = modal.Secret.from_dict({"HF_TOKEN": _hf_token})
else:
    hf_secret = modal.Secret.from_dict({})


@app.function(image=image, gpu="T4", timeout=1800, secrets=[hf_secret])
def run_vace_trajectory():
    import os
    import shutil
    import subprocess

    import torch
    print(f"torch {torch.__version__}, cuda {torch.version.cuda}, "
          f"available {torch.cuda.is_available()}", flush=True)
    print(f"GPU: {torch.cuda.get_device_name(0)}, compute capability: "
          f"{torch.cuda.get_device_capability(0)}", flush=True)

    from huggingface_hub import hf_hub_download, snapshot_download

    print("=== download VACE-Wan2.1-1.3B-Preview model (~19GB) ===", flush=True)
    model_dir = snapshot_download(repo_id="ali-vilab/VACE-Wan2.1-1.3B-Preview",
                                   local_dir="/root/model")
    print(f"model downloaded to {model_dir}", flush=True)

    print("=== build test clip (frames 1-9, uav0000117_02622_v, same scene as before; "
          "9 not 10 to match --frame_num 9 below, needed to fit the T4's VRAM) ===",
          flush=True)
    import cv2
    frame_files = []
    for i in range(1, 10):
        p = hf_hub_download(repo_id="Voxel51/visdrone-mot", repo_type="dataset",
                             filename=f"data/{i:07d}-3.jpg")
        frame_files.append(p)
    first = cv2.imread(frame_files[0])
    h, w = first.shape[:2]
    test_video_path = "/root/test_clip.mp4"
    writer = cv2.VideoWriter(test_video_path, cv2.VideoWriter_fourcc(*"mp4v"), 4, (w, h))
    for fn in frame_files:
        writer.write(cv2.imread(fn))
    writer.release()
    print(f"wrote {test_video_path} ({w}x{h}, {len(frame_files)} frames)", flush=True)

    print("=== build mask following track 97's REAL per-frame trajectory (a car "
          "actually driving through this intersection), not a fixed box ===", flush=True)
    import numpy as np
    real_boxes = [
        [1808.0, 760.0, 231.0, 129.0], [1601.0, 661.0, 230.0, 128.0], [1573.0, 655.0, 213.0, 115.0],
        [1548.0, 644.0, 219.0, 119.0], [1523.0, 633.0, 225.0, 124.0], [1499.0, 623.0, 230.0, 128.0],
        [1479.0, 614.0, 230.0, 128.0], [1459.0, 606.0, 230.0, 128.0], [1439.0, 597.0, 230.0, 128.0],
    ]
    mask_video_path = "/root/mask_clip.mp4"
    mask_writer = cv2.VideoWriter(mask_video_path, cv2.VideoWriter_fourcc(*"mp4v"), 4, (w, h), isColor=False)
    for bx, by, bw, bh in real_boxes:
        mask_frame = np.zeros((h, w), dtype=np.uint8)
        mask_frame[int(by):int(by + bh), int(bx):int(bx + bw)] = 255
        mask_writer.write(mask_frame)
    mask_writer.release()
    print(f"wrote {mask_video_path}, {len(real_boxes)} frames of real per-frame boxes", flush=True)

    print("=== patch wan's flash_attention import to the safe attention() fallback "
          "(T4 doesn't support FlashAttention; that function has no fallback switch "
          "of its own and crashes if flash_attn isn't installed, which it correctly "
          "is not here) ===", flush=True)
    import wan.modules.model as _wan_model_check
    wan_model_path = _wan_model_check.__file__
    with open(wan_model_path) as f:
        content = f.read()
    content = content.replace(
        "from .attention import flash_attention",
        "from .attention import attention as flash_attention",
    )
    with open(wan_model_path, "w") as f:
        f.write(content)
    print(f"patched {wan_model_path}", flush=True)

    print("=== run VACE inference with the real trajectory mask ===", flush=True)
    prompt = (
        "a dark sedan car driving through a night city intersection, seen from directly "
        "above from an aerial drone camera, headlights on, moving smoothly along the road, "
        "lit by warm orange streetlights and reflections on the road surface, "
        "casting a short shadow, photorealistic, sharp focus, "
        "consistent with the surrounding night traffic scene"
    )
    OUT_DIR = "/root/output"
    os.makedirs(OUT_DIR, exist_ok=True)
    # Ran out of VRAM (14.55/14.56 GiB used) with the default --frame_num 81 despite
    # only having a 10-frame source clip: it was still encoding as if for an 81-frame
    # clip. Setting frame_num to match our actual clip (must satisfy (n-1)%4==0 for this
    # video VAE, 9 is the closest value to 10) cuts that encoding work substantially.
    # expandable_segments reduces fragmentation on top of that, per the OOM error's own
    # suggestion.
    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    result = subprocess.run(
        f"python vace/vace_wan_inference.py --model_name vace-1.3B "
        f"--ckpt_dir {model_dir} "
        f"--src_video {test_video_path} --src_mask {mask_video_path} "
        f"--prompt \"{prompt}\" "
        f"--save_dir {OUT_DIR} "
        f"--frame_num 9 "
        f"--offload_model True --t5_cpu",
        shell=True, cwd="/root/VACE", env=env,
    )
    print(f"vace_wan_inference.py exit code: {result.returncode}", flush=True)
    subprocess.run(f"find {OUT_DIR} -type f", shell=True)

    out_video_bytes = None
    src_video_bytes = None
    out_video_path = f"{OUT_DIR}/out_video.mp4"
    if os.path.exists(out_video_path):
        with open(out_video_path, "rb") as f:
            out_video_bytes = f.read()
        print(f"out_video.mp4: {len(out_video_bytes)/1e6:.2f} MB", flush=True)
    with open(test_video_path, "rb") as f:
        src_video_bytes = f.read()

    return result.returncode, out_video_bytes, src_video_bytes


@app.local_entrypoint()
def main():
    rc, out_bytes, src_bytes = run_vace_trajectory.remote()
    print(f"\nFinal result code: {rc}")
    out_dir = "/home/akshata/projects/soar_drone_editing/aerial_box_propagation/results/vace_trajectory_result"
    import os
    os.makedirs(out_dir, exist_ok=True)
    if out_bytes is not None:
        with open(f"{out_dir}/out_video.mp4", "wb") as f:
            f.write(out_bytes)
        print(f"saved {out_dir}/out_video.mp4")
    if src_bytes is not None:
        with open(f"{out_dir}/src_video.mp4", "wb") as f:
            f.write(src_bytes)
        print(f"saved {out_dir}/src_video.mp4")
