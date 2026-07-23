"""Smoke test: does DROID-SLAM install and run at all on Kaggle's GPU environment.

Not trying to get accurate poses yet, camera intrinsics below are a rough guess (no
true calibration available for this drone footage). Goal is only to prove the install
(custom CUDA extension build) and inference path work end to end on 10 frames before
committing more GPU quota to a real run.
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

import torch
print(f"torch {torch.__version__}, cuda {torch.version.cuda}, "
      f"available {torch.cuda.is_available()}", flush=True)
print(f"GPU: {torch.cuda.get_device_name(0)}, compute capability: "
      f"{torch.cuda.get_device_capability(0)}", flush=True)

print("\n=== set TORCH_CUDA_ARCH_LIST for the CUDA extension builds below, detected at "
      "runtime rather than hardcoded: Kaggle's free GPU pool is not homogeneous, this "
      "kernel got a Tesla P100 (Pascal, compute capability 6.0) on one run and a T4 "
      "(Turing, 7.5) on another. Hardcoding 7.5 previously fixed the T4 case but broke "
      "this P100 run the exact same way in reverse (built for 7.5, can't run on 6.0 "
      "hardware: 'no kernel image is available for execution on the device'). Detecting "
      "and matching whatever GPU actually got allocated this session avoids that. ===",
      flush=True)
cap = torch.cuda.get_device_capability(0)
current_arch = f"{cap[0]}.{cap[1]}"
os.environ["TORCH_CUDA_ARCH_LIST"] = current_arch
print(f"TORCH_CUDA_ARCH_LIST set to {current_arch} "
      f"(detected from {torch.cuda.get_device_name(0)})", flush=True)

print("\n=== checkpointing: reuse a previous successful build if this run gets the same "
      "GPU architecture, otherwise rebuild from scratch (safe either way; each retry "
      "so far has re-run the full ~5min build after failing on something unrelated "
      "later on, this skips that once we have one matching-arch success saved) ===",
      flush=True)
CHECKPOINT_IN = "/kaggle/input/droid-slam-smoke-test/checkpoint"
CHECKPOINT_OUT = "/kaggle/working/checkpoint"
os.makedirs(CHECKPOINT_OUT, exist_ok=True)
import site
SITE_PACKAGES = site.getsitepackages()[0]

cached_arch = None
if os.path.exists(f"{CHECKPOINT_IN}/gpu_arch.txt"):
    with open(f"{CHECKPOINT_IN}/gpu_arch.txt") as f:
        cached_arch = f.read().strip()
build_cache_path = f"{CHECKPOINT_IN}/build_cache.tar.gz"
reuse_build = (cached_arch == current_arch) and os.path.exists(build_cache_path)
print(f"cached arch: {cached_arch!r}, current arch: {current_arch!r}, "
      f"build cache present: {os.path.exists(build_cache_path)}, "
      f"reuse_build: {reuse_build}", flush=True)
if reuse_build:
    run(f"tar -xzf {build_cache_path} -C {SITE_PACKAGES}")
    print("extracted cached lietorch + droid_backends build into site-packages, "
          "skipping their build steps below", flush=True)

cached_weights = f"{CHECKPOINT_IN}/droid.pth"
reuse_weights = os.path.exists(cached_weights)
print(f"cached weights present: {reuse_weights}", flush=True)

print("\n=== fetch input frames directly from HF (the Kaggle dataset mount failed to "
      "attach on this run for the same unclear reason it did for the VACE kernel; "
      "downloading directly is more robust than depending on it) ===", flush=True)
import shutil as _shutil
from huggingface_hub import hf_hub_download
INPUT_DIR = "/kaggle/working/smoke_frames"
os.makedirs(INPUT_DIR, exist_ok=True)
for i in range(1, 11):
    p = hf_hub_download(repo_id="Voxel51/visdrone-mot", repo_type="dataset",
                         filename=f"data/{i:07d}-3.jpg")
    _shutil.copy(p, os.path.join(INPUT_DIR, f"frame_{i:03d}.jpg"))
run(f"ls -la {INPUT_DIR}")

print("\n=== clone DROID-SLAM ===", flush=True)
WORK = "/kaggle/working/DROID-SLAM"
rc = run(f"git clone --recursive https://github.com/princeton-vl/DROID-SLAM.git {WORK}")
if rc != 0:
    print("FATAL: clone failed", flush=True)
    sys.exit(1)

print("\n=== install requirements.txt ===", flush=True)
rc = run("pip install -r requirements.txt", cwd=WORK)
print(f"requirements.txt install rc: {rc}", flush=True)

if reuse_build:
    print("\n=== skipping lietorch build (reused from checkpoint) ===", flush=True)
else:
    print("\n=== install lietorch (thirdparty submodule, its own CUDA extension) ===", flush=True)
    rc = run("pip install thirdparty/lietorch", cwd=WORK)
    if rc != 0:
        print("FATAL: lietorch install failed. This was the missing step last run "
              "(demo.py needs it, ModuleNotFoundError: No module named 'lietorch').", flush=True)
        sys.exit(1)

print("\n=== install torch_scatter from a prebuilt wheel (source build kept failing on "
      "a CUDA-version-parity check baked into PyTorch's own extension-build utilities; "
      "FORCE_CUDA=0 did not bypass it, so avoid compiling entirely) ===", flush=True)
torch_tag = torch.__version__.replace("+", "%2B")  # e.g. "2.10.0%2Bcu128"
wheel_index = f"https://data.pyg.org/whl/torch-{torch_tag}.html"
print(f"using wheel index: {wheel_index}", flush=True)
rc = run(f"pip install torch_scatter -f {wheel_index}")
print(f"torch_scatter prebuilt-wheel install rc: {rc}", flush=True)
if rc != 0:
    print("FATAL: torch_scatter install failed even from prebuilt wheel.", flush=True)
    sys.exit(1)

if reuse_build:
    print("\n=== skipping droid_backends build (reused from checkpoint) ===", flush=True)
else:
    print("\n=== build droid_backends CUDA extension (python setup.py install, this worked before) ===", flush=True)
    # Reverting to setup.py install: pip install -e . failed this run, setup.py install
    # succeeded in the previous run. Same underlying build, different pip wrapper behavior.
    rc = run("python setup.py install", cwd=WORK)
    print(f"\n=== droid_backends install result: {'SUCCESS' if rc == 0 else 'FAILED, rc=' + str(rc)} ===", flush=True)
    if rc != 0:
        print("Stopping here, this is the step we need to see fail/succeed for the smoke test.", flush=True)
        sys.exit(1)

os.makedirs(f"{WORK}/weights", exist_ok=True)
if reuse_weights:
    print("\n=== reusing cached weights (skipping download) ===", flush=True)
    import shutil
    shutil.copy(cached_weights, f"{WORK}/weights/droid.pth")
else:
    print("\n=== download pretrained weights (HuggingFace mirror, gdown link was stale) ===", flush=True)
    try:
        weight_path = hf_hub_download(repo_id="vslamlab/droidslam", filename="droid.pth")
        import shutil
        shutil.copy(weight_path, f"{WORK}/weights/droid.pth")
        print(f"copied weights from {weight_path}", flush=True)
    except Exception as e:
        print(f"FATAL: weight download failed: {e}", flush=True)
        sys.exit(1)
if not os.path.exists(f"{WORK}/weights/droid.pth"):
    print("FATAL: weight file missing after download", flush=True)
    sys.exit(1)
print(f"weight file size: {os.path.getsize(f'{WORK}/weights/droid.pth')/1e6:.1f} MB", flush=True)

print("\n=== save checkpoint for next run (build succeeded up to here, regardless of "
      "what happens below in demo.py; this is the expensive ~5min part worth saving, "
      "next run only reuses it if it gets the same GPU architecture) ===", flush=True)
if not reuse_build:
    installed = [d for d in os.listdir(SITE_PACKAGES) if d.startswith("lietorch") or d.startswith("droid_backends")]
    print(f"packaging for checkpoint: {installed}", flush=True)
    if installed:
        run(f"tar -czf {CHECKPOINT_OUT}/build_cache.tar.gz -C {SITE_PACKAGES} {' '.join(installed)}")
    with open(f"{CHECKPOINT_OUT}/gpu_arch.txt", "w") as f:
        f.write(current_arch)
import shutil as _shutil2
_shutil2.copy(f"{WORK}/weights/droid.pth", f"{CHECKPOINT_OUT}/droid.pth")
print(f"checkpoint saved to {CHECKPOINT_OUT}: {os.listdir(CHECKPOINT_OUT)}", flush=True)

print("\n=== write a rough calibration file (no true intrinsics available) ===", flush=True)
# uav0000117_02622_v is 2720x1530. fx/fy guessed assuming ~60deg horizontal FOV, a
# placeholder, not a real calibration. Fine for a smoke test of the run path, not for
# any pose accuracy claim.
width, height = 2720, 1530
import math
fx = width / (2 * math.tan(math.radians(60) / 2))
fy = fx
cx, cy = width / 2, height / 2
calib_path = f"{WORK}/calib_smoke.txt"
with open(calib_path, "w") as f:
    f.write(f"{fx} {fy} {cx} {cy}\n")
print(f"wrote {calib_path}: fx={fx:.1f} fy={fy:.1f} cx={cx} cy={cy}", flush=True)

print("\n=== run demo.py on the 10 smoke frames ===", flush=True)
OUT_DIR = "/kaggle/working/output"
os.makedirs(OUT_DIR, exist_ok=True)
rc = run(
    f"python demo.py --imagedir={INPUT_DIR} --calib={calib_path} "
    f"--weights={WORK}/weights/droid.pth "
    f"--disable_vis --reconstruction_path={OUT_DIR}/recon",
    cwd=WORK,
)
print(f"\n=== demo.py result: {'SUCCESS' if rc == 0 else 'FAILED, rc=' + str(rc)} ===", flush=True)

print("\n=== output directory contents ===", flush=True)
run(f"find {OUT_DIR} -type f")
run(f"find /kaggle/working -maxdepth 3 -iname '*recon*'")

print("\n=== DONE ===", flush=True)
