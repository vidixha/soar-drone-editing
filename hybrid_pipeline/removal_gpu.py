"""GPU-accelerated version of modules.remove_objects's hot loop, run on
Modal. Per output frame, remove_objects warps ~20-40 neighbor frames and
takes a per-pixel median -- that's the actual cost (confirmed: ~8-27 min on
CPU for a 75-125 frame clip). ORB feature detection and RANSAC homography
estimation stay on CPU (cv2, not the bottleneck, no good GPU path anyway);
only the warp+median step moves to GPU via kornia + torch. Reuses
modules.py's homography estimation and mask/compositing directly (bundled
into the image) so this can't silently drift from the CPU path's algorithm.
"""
import modal

app = modal.App("removal-gpu")

image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.10")
    .apt_install("ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install("torch==2.4.0", "torchvision==0.19.0", index_url="https://download.pytorch.org/whl/cu121")
    .pip_install("opencv-python-headless", "numpy", "kornia")
    .add_local_dir(
        "/home/akshata/projects/soar_drone_editing/aerial_box_propagation/hybrid_pipeline",
        remote_path="/opt/hybrid_pipeline",
    )
)


@app.function(image=image, gpu="L4", timeout=1800)
def remove_objects_of_bytes(video_bytes: bytes, target: str = None, window: int = 60, stride: int = 3) -> bytes:
    import sys, tempfile, subprocess
    sys.path.insert(0, "/opt/hybrid_pipeline")
    import modules as M
    import cv2, numpy as np, torch
    import kornia.geometry.transform as KT

    d = tempfile.mkdtemp()
    inpath = f"{d}/in.mp4"
    with open(inpath, "wb") as f:
        f.write(video_bytes)

    frames = M.load_clip(inpath)
    H, W = frames[0].shape[:2]; T = len(frames)
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    fwd = M._estimate_step_homographies(grays)  # CPU, list of T-1 3x3 arrays

    verify = None
    if target and target != "objects":
        from vlm_match import matches_target
        verify = lambda crop: matches_target(crop, target)

    device = "cuda"
    frames_t = torch.from_numpy(np.stack(frames)).to(device).permute(0, 3, 1, 2).float()  # (T,3,H,W)
    ones_t = torch.ones((1, H, W), device=device)

    def step_span_matrix(t, i):
        """Homography mapping frame i's pixels into frame t's coordinate
        system, composed over the short i..t chain -- same construction as
        modules._local_plate's step_span, just returning the matrix instead
        of warping on the spot."""
        Hm = np.eye(3, dtype=np.float32)
        if i > t:
            for k in range(t, i): Hm = fwd[k] @ Hm
        elif i < t:
            for k in range(t - 1, i - 1, -1): Hm = np.linalg.inv(fwd[k]) @ Hm
        return Hm

    out = []
    for t in range(T):
        idx = sorted(set([t] + list(range(t, max(-1, t - window - 1), -stride))
                          + list(range(t, min(T, t + window + 1), stride))))
        idx = [i for i in idx if 0 <= i < T]
        Hs = np.stack([step_span_matrix(t, i) for i in idx]).astype(np.float32)
        Hs_t = torch.from_numpy(Hs).to(device)
        batch = frames_t[idx]                                    # (N,3,H,W)

        warped = KT.warp_perspective(batch, Hs_t, dsize=(H, W), mode="bilinear")
        valid = KT.warp_perspective(ones_t.unsqueeze(0).expand(len(idx), 1, H, W),
                                    Hs_t, dsize=(H, W), mode="nearest") > 0.5   # (N,1,H,W)

        warped_nan = warped.clone()
        warped_nan[~valid.expand(-1, 3, -1, -1)] = float("nan")
        plate = torch.nanmedian(warped_nan, dim=0).values                     # (3,H,W)
        unseen = torch.isnan(plate)
        plate = torch.where(unseen, frames_t[t], plate)

        # Same confidence signal as the CPU path: fraction of the window's
        # samples that agree with the median -- gates the blend so a
        # partially-cleared object fades toward the original pixel instead
        # of a smeared multi-copy guess (see modules.py for why this exists).
        devmax = torch.amax(torch.abs(warped - plate.unsqueeze(0)), dim=1)    # (N,H,W)
        valid2d = valid[:, 0]
        valid_count = valid2d.sum(0).float()
        agree = (valid2d & (devmax < 20)).sum(0).float()
        confidence = torch.where(valid_count > 0, agree / valid_count.clamp(min=1),
                                 torch.zeros_like(valid_count))

        plate_np = plate.clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
        conf_np = confidence.cpu().numpy().astype(np.float32)

        mask = M._mask_from_diff(frames[t], plate_np, verify=verify)
        mf = cv2.GaussianBlur(mask, (0, 0), 1.5).astype(np.float32)[..., None] / 255.0
        mf = mf * cv2.GaussianBlur(conf_np, (0, 0), 1.5)[..., None]
        out.append((frames[t] * (1 - mf) + plate_np * mf).astype(np.uint8))

    outpath = f"{d}/out.mp4"
    M.save_video(out, outpath, 25)
    with open(outpath, "rb") as f:
        result = f.read()
    subprocess.run(["rm", "-rf", d])
    return result
