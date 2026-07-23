"""DROID-SLAM smoke test on Modal, ported from the Kaggle version after 10 attempts
there were blocked by an unfixable-from-our-side issue: Kaggle's free GPU pool
sometimes hands out a P100 (Pascal, compute capability 6.0), and Kaggle's preinstalled
PyTorch build has dropped support for that GPU generation entirely. Modal lets us pin
an exact GPU type instead, so that whole class of problem goes away.

Cost control, per instruction to use Modal carefully:
- T4 explicitly requested: Modal's cheapest capable GPU tier, and one PyTorch still
  fully supports (T4 = compute capability 7.5, well within modern PyTorch's supported
  range, unlike the P100 that broke every Kaggle attempt).
- Every install step baked into the Image definition (not the function body), so Modal
  caches each layer. Iterating on the function logic alone (fixing a flag, changing an
  argument) does not re-trigger any of the slow steps (CUDA extension builds, package
  installs), unlike Kaggle where every single attempt re-ran the entire pipeline from
  scratch regardless of which step actually needed fixing.
- Hard timeout on the function call so a hang cannot run away with credits.
- TORCH_CUDA_ARCH_LIST is hardcoded to 7.5 here, safely, because we are the ones
  choosing gpu="T4" below, not depending on whatever Kaggle's pool happens to hand out.
"""

import modal

app = modal.App("droid-slam-smoke-test")

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0", "g++")
    # modal's add_python (python-build-standalone) defaults CC to a clang that isn't
    # actually present in this image, causing every C++ extension build below to fail
    # with "command 'clang' failed: No such file or directory". Force gcc/g++ explicitly.
    .env({"CC": "gcc", "CXX": "g++"})
    .pip_install(
        "torch==2.5.1", "torchvision==0.20.1",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .run_commands(
        "git clone --recursive https://github.com/princeton-vl/DROID-SLAM.git /root/DROID-SLAM"
    )
    .pip_install("huggingface_hub", "opencv-python-headless", "numpy")
    .run_commands("cd /root/DROID-SLAM && pip install -r requirements.txt")
    .env({"TORCH_CUDA_ARCH_LIST": "7.5"})  # T4, pinned below via gpu="T4", safe to hardcode
    .run_commands("cd /root/DROID-SLAM && pip install thirdparty/lietorch")
    .run_commands(
        "pip install torch_scatter -f https://data.pyg.org/whl/torch-2.5.1+cu124.html"
    )
    .run_commands("cd /root/DROID-SLAM && python setup.py install")
)


@app.function(image=image, gpu="T4", timeout=600)
def run_smoke_test():
    import math
    import os
    import shutil
    import subprocess

    import torch
    print(f"torch {torch.__version__}, cuda {torch.version.cuda}, "
          f"available {torch.cuda.is_available()}", flush=True)
    print(f"GPU: {torch.cuda.get_device_name(0)}, compute capability: "
          f"{torch.cuda.get_device_capability(0)}", flush=True)

    from huggingface_hub import hf_hub_download

    print("=== fetch 10 smoke frames from HF ===", flush=True)
    INPUT_DIR = "/root/smoke_frames"
    os.makedirs(INPUT_DIR, exist_ok=True)
    for i in range(1, 11):
        p = hf_hub_download(repo_id="Voxel51/visdrone-mot", repo_type="dataset",
                             filename=f"data/{i:07d}-3.jpg")
        shutil.copy(p, os.path.join(INPUT_DIR, f"frame_{i:03d}.jpg"))
    print(f"frames in {INPUT_DIR}: {sorted(os.listdir(INPUT_DIR))}", flush=True)

    print("=== fetch pretrained weights (HF mirror, the original gdown link is stale) ===",
          flush=True)
    WORK = "/root/DROID-SLAM"
    os.makedirs(f"{WORK}/weights", exist_ok=True)
    weight_path = hf_hub_download(repo_id="vslamlab/droidslam", filename="droid.pth")
    shutil.copy(weight_path, f"{WORK}/weights/droid.pth")
    print(f"weights: {os.path.getsize(f'{WORK}/weights/droid.pth')/1e6:.1f} MB", flush=True)

    print("=== write a rough calibration file (no true intrinsics for this footage) ===",
          flush=True)
    width, height = 2720, 1530
    fx = width / (2 * math.tan(math.radians(60) / 2))
    fy = fx
    cx, cy = width / 2, height / 2
    calib_path = f"{WORK}/calib_smoke.txt"
    with open(calib_path, "w") as f:
        f.write(f"{fx} {fy} {cx} {cy}\n")
    print(f"calib: fx={fx:.1f} fy={fy:.1f} cx={cx} cy={cy}", flush=True)

    print("=== run demo.py ===", flush=True)
    OUT_DIR = "/root/output"
    os.makedirs(OUT_DIR, exist_ok=True)
    result = subprocess.run(
        f"python demo.py --imagedir={INPUT_DIR} --calib={calib_path} "
        f"--weights={WORK}/weights/droid.pth --disable_vis "
        f"--reconstruction_path={OUT_DIR}/recon",
        shell=True, cwd=WORK,
    )
    print(f"demo.py exit code: {result.returncode}", flush=True)

    subprocess.run(f"find {OUT_DIR} -type f", shell=True)

    # The first successful run wrote this file inside the container and then the
    # container was torn down with nothing persisted outside it, so there was nothing
    # to inspect afterward. Fix: read it back as bytes and return it directly, no Modal
    # Volume needed for a single-file, one-off result like this.
    recon_path = f"{OUT_DIR}/recon"
    recon_bytes = None
    if os.path.exists(recon_path):
        with open(recon_path, "rb") as f:
            recon_bytes = f.read()
        print(f"reconstruction file: {len(recon_bytes)/1e6:.2f} MB, returning to caller",
              flush=True)
    else:
        print("WARNING: no reconstruction file found to return", flush=True)

    return result.returncode, recon_bytes


@app.local_entrypoint()
def main():
    rc, recon_bytes = run_smoke_test.remote()
    print(f"\nFinal result code: {rc}")
    if recon_bytes is not None:
        out_path = "/home/akshata/projects/soar_drone_editing/aerial_box_propagation/results/droid_slam_first_result/recon.pth"
        import os
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "wb") as f:
            f.write(recon_bytes)
        print(f"saved reconstruction to {out_path} ({len(recon_bytes)/1e6:.2f} MB)")
    else:
        print("no reconstruction bytes returned")
