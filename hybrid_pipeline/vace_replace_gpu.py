"""VACE (github.com/ali-vilab/VACE, ICCV 2025) object replacement, on Modal.

Reworked from the project's earlier VACE experiment (modal/vace_pipeline.py,
a hardcoded CLI script with hand-staged /tmp files, single fixed scene) into
a real callable pipeline function: takes an already-clean background (our
own GPU-inpaint removal, not VACE's/LaMa's), a reference object image, and a
target-clip byte stream, and returns the composited result.

Same reasoning as removal_inpaint_gpu.py's fix carries over here: don't let
the model guess *where* the object goes. The mask video is built from our
own YOLO detector (validated on this exact footage -- accurate box location
even where it gets the class label wrong, see removal_inpaint_gpu.py's
docstring) run on the ORIGINAL clip, so placement/scale come from real
per-frame positions, and VACE is only responsible for the pixels: harmonizing
the reference object into that position, frame by frame, with native
temporal consistency (Wan2.1 generates the whole clip jointly, unlike
per-frame diffusion models -- confirmed in this project's earlier AnyDoor
attempts to flicker for exactly that reason).

Known, documented limitation carried over from the earlier experiment,
not fixed here: VACE does not reliably follow a specific pose/orientation
for the inserted object -- it tends toward a generic three-quarter
"showroom" angle rather than matching a flat top-down aerial view, and
pushing the prompt harder to correct this can make it drop the object
entirely. PISCO (arXiv 2602.08277) is the closest published fix, and it
requires fine-tuning + depth conditioning not available in vanilla VACE.
"""
import pathlib
import modal

app = modal.App("vace-replace-gpu")

weights_volume = modal.Volume.from_name("vace-weights", create_if_missing=True)
WEIGHTS_DIR = "/weights"

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0", "wget")
    .pip_install("setuptools", "wheel")
    .pip_install("torch==2.5.1", "torchvision==0.20.1", index_url="https://download.pytorch.org/whl/cu124")
    .run_commands("git clone --depth 1 https://github.com/ali-vilab/VACE.git /opt/VACE")
    .run_commands("cd /opt/VACE && pip install -r requirements.txt")
    .pip_install("wan@git+https://github.com/Wan-Video/Wan2.1")
    .pip_install("matplotlib", "hf_transfer", "ultralytics")
    # Latest opencv-python-headless (5.x) requires numpy>=2, but wan pins
    # numpy<2 -- no version of opencv-python-headless satisfies both, so
    # pin an older 4.x release that predates the numpy>=2 requirement
    # instead of fighting the two constraints against each other.
    .pip_install("opencv-python-headless<4.10", "numpy<2")
    .run_commands(
        # wan's code does `from huggingface_hub import is_offline_mode`, a
        # symbol removed from huggingface_hub around 0.26 -- VACE's own
        # requirements need >=0.26 for other reasons, so no single pinned
        # version satisfies both. Shim the missing symbol back in instead,
        # via sitecustomize.py so it's live before vace_wan_inference.py
        # runs as a subprocess (an in-process monkeypatch wouldn't reach it).
        "echo 'import huggingface_hub\\n"
        "if not hasattr(huggingface_hub, \"is_offline_mode\"):\\n"
        "    huggingface_hub.is_offline_mode = lambda: False' "
        "> /usr/local/lib/python3.10/site-packages/sitecustomize.py"
    )
    .add_local_dir(
        "/home/akshata/projects/soar_drone_editing/aerial_box_propagation/hybrid_pipeline",
        remote_path="/opt/hybrid_pipeline",
    )
)


@app.function(image=image, volumes={WEIGHTS_DIR: weights_volume}, timeout=1800, cpu=2)
def download_weights():
    """CPU-only: downloads the 1.3B-Preview checkpoint (smallest available)
    into the persistent volume, once."""
    import os
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    from huggingface_hub import snapshot_download

    ckpt_dir = pathlib.Path(WEIGHTS_DIR) / "VACE-Wan2.1-1.3B-Preview"
    if not ckpt_dir.exists() or not any(ckpt_dir.iterdir()):
        print("downloading VACE-Wan2.1-1.3B-Preview...")
        snapshot_download(repo_id="ali-vilab/VACE-Wan2.1-1.3B-Preview", local_dir=str(ckpt_dir))
        weights_volume.commit()
        print("done")
    else:
        print("already cached")


@app.function(image=image, volumes={WEIGHTS_DIR: weights_volume}, gpu="A100-40GB", timeout=2400)
def replace_object(background_video_bytes: bytes, ref_image_bytes: bytes, orig_video_bytes: bytes,
                   prompt: str, frame_num: int = 49, size: str = "832*480", debug: bool = False,
                   guide_scale: float = 5.0) -> bytes:
    """background_video_bytes: clean background with the target already
    removed (e.g. removal_inpaint_gpu's output) -- VACE only has to
    harmonize the reference object in, not also erase the original one.
    orig_video_bytes: the ORIGINAL (unedited) clip, used only to run the
    YOLO detector for real per-frame box placement -- never touched
    otherwise. ref_image_bytes: a crop of the object to insert.
    frame_num must be 4n+1 (VACE's requirement); 49 matches this project's
    other GPU pipelines' clip length.

    guide_scale: classifier-free guidance strength (VACE/wan default 5.0,
    left untouched in the two runs that produced zero trace of the
    reference object anywhere in the output, despite the reference image,
    mask, and background all confirmed correctly received -- see
    vace_wan_inference.py's own generate(): at low guidance the model can
    fall back to reconstructing near-background content in the masked hole
    instead of committing to the prompt/reference, a known failure mode in
    diffusion editing models generally, not specific to this checkpoint.
    Testing a substantially higher value here to see if that's the cause."""
    import sys, os, subprocess, tempfile
    sys.path.insert(0, "/opt/hybrid_pipeline")
    import cv2, numpy as np
    import modules as M

    d = tempfile.mkdtemp()
    bg_path = f"{d}/bg.mp4"; open(bg_path, "wb").write(background_video_bytes)
    orig_path = f"{d}/orig.mp4"; open(orig_path, "wb").write(orig_video_bytes)
    ref_path = f"{d}/ref.png"; open(ref_path, "wb").write(ref_image_bytes)

    bg_frames = M.load_clip(bg_path)[:frame_num]
    orig_frames = M.load_clip(orig_path)[:frame_num]
    while len(bg_frames) < frame_num: bg_frames.append(bg_frames[-1])
    while len(orig_frames) < frame_num: orig_frames.append(orig_frames[-1])
    H, W = bg_frames[0].shape[:2]

    # Real per-frame placement from our own detector (see module docstring
    # for why: class-agnostic YOLO box location is reliable on this footage
    # even where the label isn't), not VACE's own guess.
    from ultralytics import YOLO
    yolo = YOLO("yolov8n.pt")
    mask_frames = []
    for f in orig_frames:
        res = yolo(f, verbose=False, conf=0.1)[0]
        m = np.zeros((H, W, 3), np.uint8)
        boxes = res.boxes.xyxy.cpu().numpy()
        if len(boxes):
            x1, y1, x2, y2 = boxes[0].astype(int)  # single target object
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
            pad = max(4, int(0.1 * max(x2 - x1, y2 - y1)))
            m[max(0, y1 - pad):min(H, y2 + pad), max(0, x1 - pad):min(W, x2 + pad)] = 255
        mask_frames.append(m)
    print(f"[vace] detector placed a box in {sum(1 for m in mask_frames if m.any())}/{frame_num} frames")

    mask_path = f"{d}/mask.mp4"
    M.save_video(mask_frames, mask_path, 25)

    os.chdir("/opt/VACE")
    ckpt_dir = f"{WEIGHTS_DIR}/VACE-Wan2.1-1.3B-Preview"
    save_dir = f"{d}/vace_out"
    cmd = [
        "python", "vace/vace_wan_inference.py",
        "--ckpt_dir", ckpt_dir,
        "--src_video", bg_path,
        "--src_mask", mask_path,
        "--src_ref_images", ref_path,
        "--prompt", prompt,
        "--frame_num", str(frame_num),
        "--size", size,
        "--save_dir", save_dir,
        "--offload_model", "True",
        "--sample_guide_scale", str(guide_scale),
    ]
    print("running:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    print("STDOUT:\n", result.stdout[-8000:])
    print("STDERR:\n", result.stderr[-8000:])
    result.check_returncode()

    all_files = list(pathlib.Path(save_dir).rglob("*"))
    print(f"[vace] save_dir contents: {all_files}")
    mp4s = [p for p in all_files if p.name == "out_video.mp4"]
    if not mp4s:
        raise RuntimeError(f"no out_video.mp4 in {save_dir}, contents: {all_files}")

    if debug:
        import io, zipfile
        buf = io.BytesIO(); zf = zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED)
        for p in all_files:
            if p.is_file():
                zf.write(p, arcname=p.name)
        zf.write(mask_path, arcname="my_mask_input.mp4")
        zf.write(ref_path, arcname="my_ref_input.png")
        zf.close()
        return buf.getvalue()

    with open(mp4s[0], "rb") as f:
        return f.read()
