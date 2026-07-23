"""DROID-SLAM on a real 100-frame stretch (not the 10-frame smoke test), to get enough
keyframes for an actual trajectory, not just a before/after pair. Same scene used in the
project's drift-overlay videos (uav0000137_00458_v), so this connects back to work
already shown. Reuses the identical image definition as droid_slam_modal.py so Modal's
layer cache is hit and no rebuild is needed, just the new frame range and the
visualization step at the end.
"""

import modal

app = modal.App("droid-slam-trajectory")

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0", "g++")
    .env({"CC": "gcc", "CXX": "g++"})
    .pip_install(
        "torch==2.5.1", "torchvision==0.20.1",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .run_commands(
        "git clone --recursive https://github.com/princeton-vl/DROID-SLAM.git /root/DROID-SLAM"
    )
    .pip_install("huggingface_hub", "opencv-python-headless", "numpy", "matplotlib")
    .run_commands("cd /root/DROID-SLAM && pip install -r requirements.txt")
    .env({"TORCH_CUDA_ARCH_LIST": "7.5"})
    .run_commands("cd /root/DROID-SLAM && pip install thirdparty/lietorch")
    .run_commands(
        "pip install torch_scatter -f https://data.pyg.org/whl/torch-2.5.1+cu124.html"
    )
    .run_commands("cd /root/DROID-SLAM && python setup.py install")
)


@app.function(image=image, gpu="T4", timeout=900)
def run_trajectory():
    import math
    import os
    import shutil
    import subprocess

    import torch
    print(f"torch {torch.__version__}, GPU: {torch.cuda.get_device_name(0)}", flush=True)

    from huggingface_hub import hf_hub_download

    print("=== fetch 100 real frames (uav0000137_00458_v, same scene as the project's "
          "drift-overlay videos) ===", flush=True)
    INPUT_DIR = "/root/frames"
    os.makedirs(INPUT_DIR, exist_ok=True)
    N_FRAMES = 100
    for i in range(1, N_FRAMES + 1):
        p = hf_hub_download(repo_id="Voxel51/visdrone-mot", repo_type="dataset",
                             filename=f"data/{i:07d}-5.jpg")
        shutil.copy(p, os.path.join(INPUT_DIR, f"frame_{i:03d}.jpg"))
    print(f"downloaded {len(os.listdir(INPUT_DIR))} frames", flush=True)

    WORK = "/root/DROID-SLAM"
    os.makedirs(f"{WORK}/weights", exist_ok=True)
    weight_path = hf_hub_download(repo_id="vslamlab/droidslam", filename="droid.pth")
    shutil.copy(weight_path, f"{WORK}/weights/droid.pth")

    print("=== calibration (rough guess, no true intrinsics for this footage) ===", flush=True)
    width, height = 2688, 1512
    fx = width / (2 * math.tan(math.radians(60) / 2))
    fy = fx
    cx, cy = width / 2, height / 2
    calib_path = f"{WORK}/calib.txt"
    with open(calib_path, "w") as f:
        f.write(f"{fx} {fy} {cx} {cy}\n")

    print("=== run demo.py on 100 frames ===", flush=True)
    OUT_DIR = "/root/output"
    os.makedirs(OUT_DIR, exist_ok=True)
    result = subprocess.run(
        f"python demo.py --imagedir={INPUT_DIR} --calib={calib_path} "
        f"--weights={WORK}/weights/droid.pth --disable_vis "
        f"--reconstruction_path={OUT_DIR}/recon",
        shell=True, cwd=WORK,
    )
    print(f"demo.py exit code: {result.returncode}", flush=True)
    if result.returncode != 0:
        return result.returncode, None, None, None

    print("=== load reconstruction, build trajectory plot + keyframe video ===", flush=True)
    import cv2
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = torch.load(f"{OUT_DIR}/recon", weights_only=False)
    poses = data["poses"].numpy()       # [K, 7]: tx ty tz qx qy qz qw
    images = data["images"].numpy()     # [K, 3, H, W], BGR
    disps = data["disps"].numpy()       # [K, H, W]
    tstamps = data["tstamps"].numpy()
    K = poses.shape[0]
    print(f"{K} keyframes kept out of {N_FRAMES} input frames", flush=True)
    print(f"source frame indices used as keyframes: {tstamps.tolist()}", flush=True)

    # Trajectory plot: top-down view (x vs z, typical camera-forward convention)
    xs, zs = poses[:, 0], poses[:, 2]
    fig, ax = plt.subplots(figsize=(6, 6), dpi=130)
    ax.plot(xs, zs, "-o", color="#3a7bd5", markersize=4, linewidth=1.5)
    ax.scatter([xs[0]], [zs[0]], color="#4caf6d", s=80, zorder=5, label="start")
    ax.scatter([xs[-1]], [zs[-1]], color="#d9615d", s=80, zorder=5, label="end")
    ax.set_xlabel("x (arbitrary scale)")
    ax.set_ylabel("z (arbitrary scale)")
    ax.set_title(f"DROID-SLAM estimated camera path, {K} keyframes / {N_FRAMES} frames")
    ax.legend()
    ax.set_aspect("equal")
    plot_path = "/root/trajectory.png"
    fig.savefig(plot_path)
    with open(plot_path, "rb") as f:
        plot_bytes = f.read()

    # Keyframe video: each keyframe's real image, upscaled, with cumulative path-length
    # so far as a simple, honest motion indicator (no fabricated smoothing between them).
    video_path = "/root/keyframes.mp4"
    h, w = images.shape[2], images.shape[3]
    scale = 2
    writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), 2, (w * scale, h * scale))
    cum_dist = 0.0
    for i in range(K):
        img = images[i].transpose(1, 2, 0)  # BGR already
        img = cv2.resize(img, (w * scale, h * scale))
        if i > 0:
            cum_dist += float(np.linalg.norm(poses[i, :3] - poses[i - 1, :3]))
        cv2.putText(img, f"keyframe {i+1}/{K}  source frame {int(tstamps[i])+1}",
                    (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, f"cumulative path length: {cum_dist:.3f} (arbitrary scale)",
                    (16, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2, cv2.LINE_AA)
        writer.write(img)
    writer.release()
    with open(video_path, "rb") as f:
        video_bytes = f.read()
    print(f"trajectory plot {len(plot_bytes)/1e3:.1f} KB, keyframe video {len(video_bytes)/1e6:.2f} MB",
          flush=True)

    with open(f"{OUT_DIR}/recon", "rb") as f:
        recon_bytes = f.read()

    return result.returncode, recon_bytes, plot_bytes, video_bytes


@app.local_entrypoint()
def main():
    rc, recon_bytes, plot_bytes, video_bytes = run_trajectory.remote()
    print(f"\nFinal result code: {rc}")
    out_dir = "/home/akshata/projects/soar_drone_editing/aerial_box_propagation/results/droid_slam_trajectory"
    import os
    os.makedirs(out_dir, exist_ok=True)
    if recon_bytes is not None:
        with open(f"{out_dir}/recon.pth", "wb") as f:
            f.write(recon_bytes)
        print(f"saved {out_dir}/recon.pth")
    if plot_bytes is not None:
        with open(f"{out_dir}/trajectory.png", "wb") as f:
            f.write(plot_bytes)
        print(f"saved {out_dir}/trajectory.png")
    if video_bytes is not None:
        with open(f"{out_dir}/keyframes.mp4", "wb") as f:
            f.write(video_bytes)
        print(f"saved {out_dir}/keyframes.mp4")
