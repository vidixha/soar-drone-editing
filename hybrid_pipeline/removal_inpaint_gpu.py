"""GPU removal with a generative video-inpainting fallback, on Modal.

The classical homography-median approach (removal_gpu.py) can't recover
background that's never exposed in any sampled frame -- confirmed on a
precision-tracked clip where even a near-full-clip window left the target
fully visible throughout. That's a hard ceiling for any median-based method,
not a tuning problem. This module adds a real fallback for exactly that
residual: wherever the classical pass is genuinely low-confidence (not just
"declined to remove," but "detected the object and had no clean signal to
fill it with"), a video inpainting model fills that specific gap instead.

Model: ProPainter (github.com/sczhou/ProPainter), flow-guided video
inpainting. Chosen specifically over a per-frame image inpainter (e.g. LaMa,
already tried in this project and found insufficient -- flickering between
frames and visibly wrong surrounding content) because it reasons across the
whole clip via optical flow rather than each frame independently, which
fixes temporal flicker by construction rather than needing a temporal-
consistency patch bolted onto a single-image model.

Only the residual gap is handed to ProPainter, not the whole object region:
the classical pass already correctly recovers real background wherever it
has genuine signal (validated separately), so restricting the generative
model to the smaller, genuinely-unrecoverable area both reduces GPU cost
and gives ProPainter an easier problem than inpainting the entire object
on every frame.

Finding the residual region: NOT the classical motion-diff mask. Confirmed
by direct inspection (dumped intermediate frames rather than trusting
summary stats) that on this footage the motion-diff mask frequently misses
the target entirely -- the reconstructed background already resembles the
object there, so there's no diff to threshold -- which meant the "residual"
handed to ProPainter in an earlier version was scattered field-texture
noise nowhere near the actual object, not the object itself; ProPainter's
fill looked like a no-op because it was solving the wrong problem, not
because it can't inpaint. Fixed by localizing the object with a generic
object detector (YOLOv8, COCO-pretrained) instead: appearance-based, so it
doesn't care whether the object moved relative to the frame. It reliably
finds the right bounding box on this near-nadir aerial footage even though
COCO has ~no top-down car examples to name it correctly (confirmed: labeled
"parking meter"/"toaster" at ~30-45% confidence, but positioned within a
few px of the true box across every sampled frame) -- so detections are
taken class-agnostically for *where*, and vlm_match's CLIP matcher (already
validated separately) still decides *what*/whether it matches the target.
"""
import modal

app = modal.App("removal-inpaint-gpu")

image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0", "wget")
    .pip_install("torch==2.4.0", "torchvision==0.19.0", index_url="https://download.pytorch.org/whl/cu121")
    .pip_install("opencv-python-headless", "numpy", "kornia", "av", "addict", "einops", "future",
                 "scipy", "matplotlib", "scikit-image", "imageio", "imageio-ffmpeg", "pyyaml",
                 "requests", "timm", "yapf", "tqdm", "ultralytics",
                 "transformers>=4.45", "huggingface_hub<1.0")
    .run_commands(
        "git clone --depth 1 https://github.com/sczhou/ProPainter.git /opt/ProPainter",
        "mkdir -p /opt/ProPainter/weights",
        "wget -q -O /opt/ProPainter/weights/raft-things.pth "
        "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/raft-things.pth",
        "wget -q -O /opt/ProPainter/weights/recurrent_flow_completion.pth "
        "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/recurrent_flow_completion.pth",
        "wget -q -O /opt/ProPainter/weights/ProPainter.pth "
        "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/ProPainter.pth",
    )
    .add_local_dir(
        "/home/akshata/projects/soar_drone_editing/aerial_box_propagation/hybrid_pipeline",
        remote_path="/opt/hybrid_pipeline",
    )
)


@app.function(image=image, gpu="A100-40GB", timeout=1800)
def remove_with_inpaint_fallback(video_bytes: bytes, target: str = None, window: int = 60,
                                 stride: int = 3, confidence_thresh: float = 0.6,
                                 debug_frame: int = None) -> bytes:
    """debug_frame: if set, returns a small zip of PNGs for that frame index
    (original, residual mask, ProPainter's own masked_in overlay, ProPainter's
    fill, classical result) instead of the final video -- for directly
    inspecting what each stage actually produced rather than inferring it
    from summary statistics."""
    import sys, os, tempfile, subprocess, shutil
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
    fwd = M._estimate_step_homographies(grays)

    verify = None
    if target and target != "objects":
        from vlm_match import matches_target
        verify = lambda crop: matches_target(crop, target)

    device = "cuda"
    frames_t = torch.from_numpy(np.stack(frames)).to(device).permute(0, 3, 1, 2).float()
    ones_t = torch.ones((1, H, W), device=device)

    def step_span_matrix(t, i):
        Hm = np.eye(3, dtype=np.float32)
        if i > t:
            for k in range(t, i): Hm = fwd[k] @ Hm
        elif i < t:
            for k in range(t - 1, i - 1, -1): Hm = np.linalg.inv(fwd[k]) @ Hm
        return Hm

    # Class-agnostic object localization -- see module docstring for why this
    # replaces the motion-diff mask as the signal for "where does the
    # residual/inpaint region need to go." Low conf threshold since the
    # class-agnostic *location* is reliable here even though the *label*
    # usually isn't (COCO has ~no top-down car examples).
    from ultralytics import YOLO
    yolo = YOLO("yolov8n.pt")
    detector_masks = []
    for t in range(T):
        res = yolo(frames[t], verbose=False, conf=0.1)[0]
        dm = np.zeros((H, W), np.uint8)
        for box in res.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = [int(v) for v in box]
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
            if x2 <= x1 or y2 <= y1: continue
            if verify is not None:
                pad = max(4, int(0.15 * max(x2 - x1, y2 - y1)))
                xx1, yy1 = max(0, x1 - pad), max(0, y1 - pad)
                xx2, yy2 = min(W, x2 + pad), min(H, y2 + pad)
                if not verify(frames[t][yy1:yy2, xx1:xx2]): continue
            dm[y1:y2, x1:x2] = 255
        detector_masks.append(cv2.dilate(dm, np.ones((9, 9), np.uint8), 1))
    print(f"[inpaint] detector found objects in {sum(1 for m in detector_masks if m.any())}/{T} frames")

    classical_out = []
    residual_masks = []  # detector says object is here, but classical didn't confidently fill it
    for t in range(T):
        idx = sorted(set([t] + list(range(t, max(-1, t - window - 1), -stride))
                          + list(range(t, min(T, t + window + 1), stride))))
        idx = [i for i in idx if 0 <= i < T]
        Hs = np.stack([step_span_matrix(t, i) for i in idx]).astype(np.float32)
        Hs_t = torch.from_numpy(Hs).to(device)
        batch = frames_t[idx]

        warped = KT.warp_perspective(batch, Hs_t, dsize=(H, W), mode="bilinear")
        valid = KT.warp_perspective(ones_t.unsqueeze(0).expand(len(idx), 1, H, W),
                                    Hs_t, dsize=(H, W), mode="nearest") > 0.5

        warped_nan = warped.clone()
        warped_nan[~valid.expand(-1, 3, -1, -1)] = float("nan")
        plate = torch.nanmedian(warped_nan, dim=0).values
        unseen = torch.isnan(plate)
        plate = torch.where(unseen, frames_t[t], plate)

        devmax = torch.amax(torch.abs(warped - plate.unsqueeze(0)), dim=1)
        valid2d = valid[:, 0]
        valid_count = valid2d.sum(0).float()
        agree = (valid2d & (devmax < 20)).sum(0).float()
        confidence = torch.where(valid_count > 0, agree / valid_count.clamp(min=1),
                                 torch.zeros_like(valid_count))

        plate_np = plate.clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
        conf_np = confidence.cpu().numpy().astype(np.float32)

        mask = M._mask_from_diff(frames[t], plate_np, verify=verify)
        conf_blur = cv2.GaussianBlur(conf_np, (0, 0), 1.5)
        mf = cv2.GaussianBlur(mask, (0, 0), 1.5).astype(np.float32)[..., None] / 255.0
        gated_mf = mf * conf_blur[..., None]
        classical_out.append((frames[t] * (1 - gated_mf) + plate_np * gated_mf).astype(np.uint8))

        # Residual = detector says the object is here, but the classical
        # blend didn't confidently paint over it (low mask/confidence
        # weight actually applied) -- not "classical mask fired weakly,"
        # since that mask can miss the object entirely on this footage.
        residual = ((detector_masks[t] > 0) & (gated_mf[:, :, 0] < confidence_thresh)).astype(np.uint8) * 255
        residual_masks.append(residual)

    total_residual = sum(int((m > 0).sum()) for m in residual_masks)
    if total_residual == 0:
        out = classical_out
    else:
        # Free the classical stage's GPU tensors before handing off to
        # ProPainter's own model + RAFT flow network -- both stages are
        # GPU-heavy and don't need to coexist; confirmed this OOM'd on an
        # L4 (24GB) when they did.
        del frames_t, ones_t, warped, warped_nan, plate, valid
        torch.cuda.empty_cache()

        frame_dir = f"{d}/frames"; mask_dir = f"{d}/masks"; out_dir = f"{d}/pp_out"
        os.makedirs(frame_dir); os.makedirs(mask_dir)
        for i, f in enumerate(frames):
            cv2.imwrite(f"{frame_dir}/{i:05d}.png", f)
        for i, m in enumerate(residual_masks):
            cv2.imwrite(f"{mask_dir}/{i:05d}.png", m)

        subprocess.run(
            ["python", "/opt/ProPainter/inference_propainter.py",
             "--video", frame_dir, "--mask", mask_dir, "--output", out_dir,
             "--width", str(W), "--height", str(H), "--fp16",
             "--subvideo_length", "40", "--neighbor_length", "10", "--ref_stride", "10"],
            check=True, cwd="/opt/ProPainter",
        )

        result_video = None
        found = []
        for root, _, files in os.walk(out_dir):
            for fn in files:
                found.append(os.path.join(root, fn))
                if fn == "inpaint_out.mp4":
                    result_video = os.path.join(root, fn)
        print(f"[inpaint] out_dir contents: {found}")
        if result_video is None:
            raise RuntimeError(f"ProPainter output not found under {out_dir}; contents: {found}")
        pp_frames = M.load_clip(result_video)
        print(f"[inpaint] loaded {len(pp_frames)} propainter frames, shape {pp_frames[0].shape if pp_frames else None}")

        masked_in_path = os.path.join(os.path.dirname(result_video), "masked_in.mp4")

        if debug_frame is not None:
            import io, zipfile
            buf = io.BytesIO(); zf = zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED)
            t = debug_frame
            def add(name, img):
                ok, enc = cv2.imencode(".png", img)
                zf.writestr(name, enc.tobytes())
            add("original.png", frames[t])
            add("residual_mask.png", residual_masks[t])
            add("classical_out.png", classical_out[t])
            add("propainter_fill.png", pp_frames[min(t, len(pp_frames) - 1)])
            if os.path.exists(masked_in_path):
                preview_frames = M.load_clip(masked_in_path)
                add("propainter_masked_in.png", preview_frames[min(t, len(preview_frames) - 1)])
            zf.close()
            shutil.rmtree(d, ignore_errors=True)
            return buf.getvalue()

        out = []
        for t in range(T):
            pf = pp_frames[min(t, len(pp_frames) - 1)]
            if pf.shape[:2] != (H, W):
                pf = cv2.resize(pf, (W, H))
            m = cv2.GaussianBlur(residual_masks[t], (0, 0), 2.0).astype(np.float32)[..., None] / 255.0
            out.append((classical_out[t] * (1 - m) + pf * m).astype(np.uint8))

    outpath = f"{d}/out.mp4"
    M.save_video(out, outpath, 25)
    with open(outpath, "rb") as f:
        result = f.read()
    shutil.rmtree(d, ignore_errors=True)
    return result
