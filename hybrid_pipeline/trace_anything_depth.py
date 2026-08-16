"""Production Trace Anything depth-sequence backbone, replacing the two ad-hoc
Depth Anything V2 mechanisms in router.py (one-shot initial load +
post-trajectory single-frame refresh) with ONE correct primitive: per-frame,
temporally-consistent depth for a whole clip in a single ~5s feed-forward
pass.

Why this replaces both prior mechanisms:
  - The initial depth_extract.py call only ever produced ONE static depth
    map, silently reused for every frame in the clip. Fine only because our
    test clip is a locked-off hover shot; wrong in general.
  - The post-trajectory depth_of_bytes() refresh was a patch for the bug
    that stale/proxy depth broke weather's collision physics after a
    viewpoint change -- itself just another single-frame snapshot.
  - Trace Anything's ctrl_pts3d field is queryable per-frame across the
    whole clip, so one call here gives genuinely correct depth for every
    frame, in both cases, via the same code path.

Validated (CPU-only, reusing an already-paid GPU run, see trace_anything_diag.py):
  mean frame-to-frame depth change 0.004, max 0.008 (normalized 0-1) on our
  hover clip -- stable, not noisy; visually tracks real scene structure
  including small moving-object signatures.

License note: Trace Anything's model weights are CC-BY-NC-4.0 (non-commercial).
Consistent with several other components already in this project.
"""
import pathlib
import modal

app = modal.App("trace-anything-depth")

weights_volume = modal.Volume.from_name("trace-anything-weights", create_if_missing=True)
WEIGHTS_DIR = "/weights"

image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install("setuptools", "wheel")
    .pip_install("torch", "torchvision", index_url="https://download.pytorch.org/whl/cu121")
    .run_commands("git clone --depth 1 https://github.com/ByteDance-Seed/TraceAnything.git /opt/TraceAnything")
    .pip_install("einops", "omegaconf", "pillow", "opencv-python-headless", "viser",
                 "imageio", "imageio-ffmpeg", "matplotlib", "numpy",
                 "huggingface_hub<1.0", "hf_transfer")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)


@app.function(image=image, volumes={WEIGHTS_DIR: weights_volume}, gpu="A100-80GB", timeout=600)
def depth_sequence_of_bytes(video_bytes: bytes, max_frames: int = 36) -> bytes:
    """Run Trace Anything on an arbitrary clip and return a compact per-frame
    depth stack (T,H,W) float32, normalized [0,1] with a GLOBAL min/max across
    all frames (not per-frame independent normalization -- that would let the
    scale drift frame to frame and break collision thresholds tuned against a
    consistent scale). Returned as .npz, ~20MB for 36 frames at 272x512 --
    small enough to transfer directly, unlike the raw ctrl_pts3d tensor
    (~4.5GB) which is what caused the earlier stuck-transfer investigation.
    """
    import io, os, subprocess, sys, time
    import cv2, numpy as np, torch

    tmp_vid = "/tmp/in.mp4"
    open(tmp_vid, "wb").write(video_bytes)
    scene_dir = "/tmp/scene_input/clip"
    os.makedirs(scene_dir, exist_ok=True)

    cap = cv2.VideoCapture(tmp_vid)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok: break
        frames.append(f)
    cap.release()
    step = max(1, len(frames) // max_frames)
    picked = frames[::step][:max_frames]
    for i, f in enumerate(picked):
        cv2.imwrite(f"{scene_dir}/{i:04d}.jpg", f)
    print(f"wrote {len(picked)} frames")

    out_dir = "/tmp/trace_out"
    os.makedirs(out_dir, exist_ok=True)
    cmd = ["python", "scripts/infer.py", "--ckpt", f"{WEIGHTS_DIR}/trace_anything.pt",
           "--input_dir", "/tmp/scene_input", "--output_dir", out_dir]
    t0 = time.time()
    result = subprocess.run(cmd, cwd="/opt/TraceAnything", capture_output=True, text=True)
    print(f"inference done in {time.time()-t0:.1f}s, returncode={result.returncode}")
    if result.returncode != 0:
        print(result.stdout[-3000:]); print(result.stderr[-3000:])
    result.check_returncode()

    # Load the (large) output.pt IN THIS SAME CONTAINER, extract only the
    # small derived depth stack, and never touch the raw tensor again.
    data = torch.load(f"{out_dir}/clip/output.pt", map_location="cpu")
    preds = data["preds"]
    T = len(preds)
    H, W = None, None
    zs = []
    for p in preds:
        cp = p["ctrl_pts3d"].float().numpy()  # (K, H, W, 3)
        z = cp[-1, ..., -1]
        H, W = z.shape
        zs.append(z)
    zs = np.stack(zs, axis=0)  # (T, H, W)

    # GLOBAL normalization across all frames -- keeps the scale consistent
    # frame to frame, unlike the earlier per-frame-independent diagnostic.
    zmin, zmax = float(zs.min()), float(zs.max())
    depth = (zs - zmin) / (zmax - zmin + 1e-6)
    print(f"depth stack: {depth.shape}, global range [{zmin:.4f},{zmax:.4f}]")

    buf = io.BytesIO()
    np.savez_compressed(buf, depth=depth.astype(np.float32))
    return buf.getvalue()


@app.local_entrypoint()
def main(clip_path: str = "/tmp/triplet_clips/clip_s4_car.mp4"):
    video_bytes = open(clip_path, "rb").read()
    result = depth_sequence_of_bytes.remote(video_bytes)
    open("/tmp/depth_sequence.npz", "wb").write(result)
    print(f"saved /tmp/depth_sequence.npz, {len(result)} bytes")
