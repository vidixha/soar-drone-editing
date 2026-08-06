"""
Modal app: runs the GPU-only pieces of Stage 1's data pipeline that have been
validated on CPU everywhere else (see trace_replication/README section
"MAVREC drone-view POC") -- ReCamMaster rendering and CoTracker tracking --
plus a real (GPU) DEVA re-localization pass on the rendered output, for a
single scene and a single camera trajectory at a time, to keep cost minimal
and let each piece be checked before scaling up.

Cost notes (2026-08-05 Modal pricing):
  T4:  $0.000164/s (~$0.59/hr) -- NOT used here: ReCamMaster runs everything in
       bfloat16, which needs Ampere+ for proper hardware support; T4 is Turing
       and risks slow/broken bf16 ops, i.e. wasted debugging time = wasted money.
  L4:  $0.000222/s (~$0.80/hr) -- used here, full bf16 support.
Weight downloads (~20.5GB: Wan2.1 DiT 5.7GB, T5-XXL text encoder 11.4GB,
Wan2.1 VAE 0.5GB, ReCamMaster ckpt 3.0GB) are done in a CPU-only function into
a persistent Volume, so they are billed at CPU rates once, not repeated on
every GPU run.

Usage:
  modal run modal/stage1_gpu_pipeline.py::download_weights
  modal run modal/stage1_gpu_pipeline.py::run_one_trajectory --cam-type 1
"""
import pathlib

import modal

app = modal.App("trace-stage1-gpu")

weights_volume = modal.Volume.from_name("trace-stage1-weights", create_if_missing=True)
WEIGHTS_DIR = "/weights"

image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "ffmpeg", "wget")
    .pip_install("setuptools", "wheel")
    .pip_install(
        "torch==2.3.1", "torchvision==0.18.1",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    .run_commands(
        "git clone --depth 1 https://github.com/KwaiVGI/ReCamMaster.git /opt/ReCamMaster",
        "pip install -e /opt/ReCamMaster",
    )
    .pip_install(
        "cupy-cuda12x", "transformers==4.46.2", "controlnet-aux==0.0.7",
        "imageio", "imageio[ffmpeg]", "safetensors", "einops", "sentencepiece",
        "protobuf", "modelscope", "ftfy", "pandas",
        "huggingface_hub<1.0", "hf_transfer", "opencv-python-headless", "timm",
    )
    .pip_install("git+https://github.com/facebookresearch/co-tracker.git")
    .pip_install("git+https://github.com/facebookresearch/segment-anything.git")
    .run_commands(
        "git clone --depth 1 https://github.com/hkchengrex/Tracking-Anything-with-DEVA.git /opt/DEVA",
        "pip install -e /opt/DEVA",
    )
    # DEVA's gradio dependency pulls a newer huggingface_hub that breaks
    # transformers==4.46.2 (needs <1.0); re-pin last so it wins.
    .pip_install("huggingface_hub<1.0")
    .add_local_dir(
        str(pathlib.Path(__file__).parent.parent.parent.parent / "trace_replication" / "src"),
        remote_path="/opt/trace_src",
    )
    .add_local_file(
        str(pathlib.Path(__file__).parent / "inputs" / "scene_1.mp4"),
        remote_path="/opt/scene_1.mp4",
    )
    .add_local_file(
        str(pathlib.Path(__file__).parent / "inputs" / "orig_boxes_normalized.npy"),
        remote_path="/opt/orig_boxes_normalized.npy",
    )
    .add_local_file(
        str(pathlib.Path(__file__).parent / "inputs" / "car_boxes_normalized.npy"),
        remote_path="/opt/car_boxes_normalized.npy",
    )
    .add_local_file(
        str(pathlib.Path(__file__).parent.parent.parent / "viewer" / "trajcrafter_gen.mp4"),
        remote_path="/opt/traj_theta10.mp4",
    )
    .add_local_file(
        str(pathlib.Path(__file__).parent.parent.parent / "viewer" / "trajcrafter_gen2.mp4"),
        remote_path="/opt/traj_theta15phi20.mp4",
    )
    .add_local_file(
        str(pathlib.Path(__file__).parent.parent.parent / "viewer" / "trajcrafter_gen3.mp4"),
        remote_path="/opt/traj_forward.mp4",
    )
)


@app.function(image=image, volumes={WEIGHTS_DIR: weights_volume}, timeout=1800, cpu=2)
def download_weights():
    """CPU-only: downloads Wan2.1, ReCamMaster, and DEVA/MobileSAM checkpoints
    into the persistent volume. Run once; later GPU runs reuse this."""
    import os
    import pathlib as pl
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    from huggingface_hub import hf_hub_download

    wan_dir = pl.Path(WEIGHTS_DIR) / "Wan-AI" / "Wan2.1-T2V-1.3B"
    wan_dir.mkdir(parents=True, exist_ok=True)
    if not (wan_dir / "diffusion_pytorch_model.safetensors").exists():
        # modelscope's own endpoint measured 360-830 KB/s from Modal's network
        # (3-4 hour ETAs for these files); HF's CDN is far faster for the same
        # public, ungated files, so fetch from there instead. Only pull what
        # inference_recammaster.py's ModelManager.load_models() actually needs.
        from huggingface_hub import snapshot_download
        snapshot_download(
            "Wan-AI/Wan2.1-T2V-1.3B",
            local_dir=str(wan_dir),
            allow_patterns=[
                "diffusion_pytorch_model.safetensors",
                "models_t5_umt5-xxl-enc-bf16.pth",
                "Wan2.1_VAE.pth",
                "config.json",
                "google/umt5-xxl/*",
            ],
        )
        print("downloaded Wan2.1-T2V-1.3B")
    else:
        print("Wan2.1-T2V-1.3B already cached")

    recam_dir = pl.Path(WEIGHTS_DIR) / "ReCamMaster" / "checkpoints"
    recam_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = recam_dir / "step20000.ckpt"
    if not ckpt_path.exists():
        downloaded = hf_hub_download(
            repo_id="KlingTeam/ReCamMaster-Wan2.1", filename="step20000.ckpt",
            local_dir=str(recam_dir),
        )
        print(f"downloaded ReCamMaster checkpoint to {downloaded}")
    else:
        print("ReCamMaster checkpoint already cached")

    deva_dir = pl.Path(WEIGHTS_DIR) / "DEVA"
    deva_dir.mkdir(parents=True, exist_ok=True)
    import urllib.request
    for fname, url in [
        ("DEVA-propagation.pth",
         "https://github.com/hkchengrex/Tracking-Anything-with-DEVA/releases/download/v1.0/DEVA-propagation.pth"),
        ("mobile_sam.pt",
         "https://github.com/hkchengrex/Tracking-Anything-with-DEVA/releases/download/v1.0/mobile_sam.pt"),
    ]:
        out = deva_dir / fname
        if not out.exists():
            urllib.request.urlretrieve(url, str(out))
            print(f"downloaded {fname}")
        else:
            print(f"{fname} already cached")

    weights_volume.commit()
    print("done, weights persisted to volume")


@app.function(
    image=image,
    volumes={WEIGHTS_DIR: weights_volume},
    gpu="L4",
    # Measured: 50-step cfg_scale=5.0 sampling runs ~37s/step on L4 (cfg means
    # 2 forward passes/step), i.e. ~31 min for sampling alone, plus ~7 min for
    # container boot + model load + VAE encode. 1800s was too tight and got a
    # real run killed at step 15/50; 3600s leaves real margin.
    timeout=3600,
)
def run_one_trajectory(cam_type: str = "1", cfg_scale: float = 5.0) -> dict:
    """Runs ReCamMaster for a single camera trajectory on scene 1's real clip,
    then CoTracker on the rendered output, then real (GPU) DEVA re-localization
    of the tracked object in the rendered clip, and assembles one .pt pair.
    Deliberately scoped to ONE trajectory to validate before scaling to all 10.
    """
    import subprocess
    import sys
    import time

    sys.path.insert(0, "/opt/trace_src")

    t_start = time.time()

    dataset_path = pathlib.Path("/tmp/stage1_source")
    videos_dir = dataset_path / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy("/opt/scene_1.mp4", videos_dir / "scene_1.mp4")

    import csv
    with open(dataset_path / "metadata.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file_name", "text"])
        writer.writerow(["scene_1.mp4", "a tracked object viewed from a hovering drone camera"])

    output_dir = pathlib.Path("/tmp/stage1_rendered")
    output_dir.mkdir(parents=True, exist_ok=True)

    # inference_recammaster.py hardcodes weight paths relative to cwd as
    # "models/Wan-AI/..." and args.ckpt_path defaults under "models/ReCamMaster/...";
    # our downloaded weights live in the volume at WEIGHTS_DIR, so symlink it in.
    # The repo ships its own "models/ReCamMaster/checkpoints/Put ReCamMaster
    # ckpt file here.txt" placeholder, so "models/" already exists as a real
    # directory after clone -- a plain existence check silently skips the
    # symlink and leaves the real weights unreachable. Remove the placeholder
    # dir first (only if it's a real dir, not already our symlink from a
    # previous invocation of this same container).
    import shutil
    models_link = pathlib.Path("/opt/ReCamMaster/models")
    if models_link.exists() and not models_link.is_symlink():
        shutil.rmtree(models_link)
    if not models_link.exists():
        models_link.symlink_to(WEIGHTS_DIR)

    cmd = [
        "python", "/opt/ReCamMaster/inference_recammaster.py",
        "--dataset_path", str(dataset_path),
        "--ckpt_path", f"{WEIGHTS_DIR}/ReCamMaster/checkpoints/step20000.ckpt",
        "--output_dir", str(output_dir),
        "--cam_type", cam_type,
        "--cfg_scale", str(cfg_scale),
    ]
    print("running ReCamMaster:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd="/opt/ReCamMaster")
    t_render = time.time()
    print(f"ReCamMaster render done in {t_render - t_start:.1f}s")

    rendered_clips = list(output_dir.rglob("*.mp4"))
    print("rendered clips:", rendered_clips)

    # BUG (fixed): previously these files were left in the container's
    # ephemeral /tmp and never persisted anywhere -- the container tore down
    # and the successful render was lost. Copy into the persistent volume
    # (under outputs/) and commit, so it survives past this function call.
    # Commit after each file individually (not once at the end) so that if a
    # later clip's copy/commit fails, earlier clips are already durable on the
    # volume and don't need the whole (expensive) GPU render to be rerun.
    persisted_dir = pathlib.Path(WEIGHTS_DIR) / "outputs" / f"cam_type_{cam_type}"
    persisted_dir.mkdir(parents=True, exist_ok=True)
    persisted_paths = []
    for clip in rendered_clips:
        dest = persisted_dir / clip.name
        try:
            shutil.copy(clip, dest)
            weights_volume.commit()
            persisted_paths.append(str(dest))
            print(f"persisted to volume: {dest}")
        except Exception as e:
            print(f"FAILED to persist {clip} -> {dest}: {e}")

    return {
        "cam_type": cam_type,
        "render_time_s": t_render - t_start,
        "rendered_clips": persisted_paths,
    }


@app.function(
    image=image,
    volumes={WEIGHTS_DIR: weights_volume},
    # CoTracker is a lightweight point-tracking model (no diffusion sampling),
    # so a T4 is enough here -- no bf16 requirement like ReCamMaster/Wan2.1 had.
    gpu="T4",
    timeout=600,
)
def run_cotracker(cam_type: str = "1", grid_size: int = 25) -> dict:
    """Step 3: run CoTracker's 25x25 point-track grid on the rendered clip that
    run_one_trajectory already persisted to the volume, and save the tracks
    back to the volume next to it."""
    import sys
    import time

    sys.path.insert(0, "/opt/trace_src")
    import cv2
    import numpy as np
    import torch

    from cotracker_wrapper import extract_point_track_grid, normalize_tracks

    # CoTrackerPredictor() with no checkpoint arg looks for a hardcoded local
    # path ('./checkpoints/scaled_offline.pth') and does NOT auto-download via
    # torch.hub -- fetch the public HF-hosted checkpoint into the volume once,
    # reused on every future call.
    ckpt_dir = pathlib.Path(WEIGHTS_DIR) / "CoTracker"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "scaled_offline.pth"
    if not ckpt_path.exists():
        import urllib.request
        urllib.request.urlretrieve(
            "https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth",
            str(ckpt_path),
        )
        weights_volume.commit()
        print(f"downloaded CoTracker checkpoint to {ckpt_path}")

    t0 = time.time()
    video_path = pathlib.Path(WEIGHTS_DIR) / "outputs" / f"cam_type_{cam_type}" / "video0.mp4"
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames read from {video_path}")

    video_np = np.stack(frames)  # (T, H, W, 3)
    h, w = video_np.shape[1], video_np.shape[2]
    video = torch.from_numpy(video_np).permute(0, 3, 1, 2)[None].float()  # (1, T, 3, H, W)

    tracks_px = extract_point_track_grid(video, grid_size=grid_size, device="cuda", checkpoint=str(ckpt_path))
    tracks_norm = normalize_tracks(tracks_px, width=w, height=h)
    t_done = time.time()
    print(f"CoTracker done in {t_done - t0:.1f}s, tracks shape {tuple(tracks_px.shape)}")

    out_dir = pathlib.Path(WEIGHTS_DIR) / "outputs" / f"cam_type_{cam_type}"
    out_path = out_dir / "cotracker_tracks.pt"
    torch.save(
        {"tracks_px": tracks_px, "tracks_norm": tracks_norm, "grid_size": grid_size,
         "video_h": h, "video_w": w},
        out_path,
    )
    weights_volume.commit()
    print(f"persisted tracks to volume: {out_path}")

    return {
        "cam_type": cam_type,
        "cotracker_time_s": t_done - t0,
        "tracks_shape": list(tracks_px.shape),
        "out_path": str(out_path),
    }


@app.function(
    image=image,
    volumes={WEIGHTS_DIR: weights_volume},
    # DEVA + MobileSAM are lightweight compared to diffusion sampling; a T4 is
    # plenty (this mirrors the already-validated CPU dry run, just on GPU).
    gpu="T4",
    timeout=900,
)
def run_deva(cam_type: str = "1", seed_boxes: str = "orig") -> dict:
    """Step 4 (the real one, not the CPU dry run): re-localize the tracked
    object's box in the ReCamMaster-rendered clip using DEVA, seeded by a SAM
    box-prompt on the rendered clip's frame 0.

    Seeding note: ReCamMaster changes camera viewpoint via diffusion synthesis,
    not a known geometric transform, so there's no principled way to reproject
    the original clip's frame-0 box into the rendered clip's pixel space from
    CoTracker's tracks alone (those only track within the rendered clip, frame
    to frame -- not across the original/rendered pair). As an approximation
    for this small-camera-angle-change trajectory, we reuse the *same*
    normalized box coordinates from the original clip's frame 0 as the SAM
    prompt on the rendered clip's frame 0, since the object's rough screen
    position shouldn't move drastically for a modest re-camera trajectory.
    This is a known simplification -- worth sanity-checking visually.
    """
    import sys
    import time

    sys.path.insert(0, "/opt/trace_src")
    sys.path.insert(0, "/opt/DEVA")
    import cv2
    import numpy as np
    import torch
    import torch.nn.functional as F
    from argparse import ArgumentParser

    torch.autograd.set_grad_enabled(False)  # see external/deva_dry_run.py: the CPU OOM leak fix

    from deva.model.network import DEVA
    from deva.inference.inference_core import DEVAInferenceCore
    from deva.inference.object_info import ObjectInfo
    from deva.inference.eval_args import add_common_eval_args
    from deva.ext.ext_eval_args import add_ext_eval_args, add_text_default_args
    from deva.dataset.utils import im_normalization
    from deva.ext.MobileSAM.setup_mobile_sam import setup_model as setup_mobile_sam
    from segment_anything import SamPredictor

    def get_input_frame_for_deva(image_np, min_side):
        image = torch.from_numpy(image_np).permute(2, 0, 1).float().cuda() / 255
        image = im_normalization(image)
        if min_side > 0:
            h, w = image_np.shape[:2]
            scale = min_side / min(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            image = F.interpolate(image.unsqueeze(0), (new_h, new_w),
                                   mode="bilinear", align_corners=False)[0]
        return image

    def mask_to_xyxy(mask):
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return None
        return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())

    t0 = time.time()
    deva_dir = pathlib.Path(WEIGHTS_DIR) / "DEVA"
    parser = ArgumentParser()
    add_common_eval_args(parser)
    add_ext_eval_args(parser)
    add_text_default_args(parser)
    args = parser.parse_args([
        "--model", str(deva_dir / "DEVA-propagation.pth"),
        "--MOBILE_SAM_CHECKPOINT_PATH", str(deva_dir / "mobile_sam.pt"),
        "--size", "480",
    ])
    config = vars(args)
    config["enable_long_term"] = not config["disable_long_term"]
    config["enable_long_term_count_usage"] = (
        config["enable_long_term"]
        and (81 / (config["max_mid_term_frames"] - config["min_mid_term_frames"])
             * config["num_prototypes"]) >= config["max_long_term_elements"]
    )

    print("loading DEVA network on GPU...")
    network = DEVA(config).cuda().eval()
    weights = torch.load(config["model"], map_location="cuda")
    network.load_weights(weights)

    print("loading MobileSAM on GPU...")
    sam_checkpoint = torch.load(config["MOBILE_SAM_CHECKPOINT_PATH"], map_location="cuda")
    mobile_sam = setup_mobile_sam()
    mobile_sam.load_state_dict(sam_checkpoint, strict=True)
    mobile_sam.to(device="cuda").eval()
    predictor = SamPredictor(mobile_sam)

    deva = DEVAInferenceCore(network, config=config)
    deva.enabled_long_id()

    video_path = pathlib.Path(WEIGHTS_DIR) / "outputs" / f"cam_type_{cam_type}" / "video0.mp4"
    cap = cv2.VideoCapture(str(video_path))
    frames_bgr = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames_bgr.append(frame)
    cap.release()
    h, w = frames_bgr[0].shape[:2]

    boxes_file = "/opt/orig_boxes_normalized.npy" if seed_boxes == "orig" else "/opt/car_boxes_normalized.npy"
    orig_boxes_norm = np.load(boxes_file)  # (81, 4) cx,cy,w,h, original clip
    n = min(len(frames_bgr), len(orig_boxes_norm))
    frames_bgr = frames_bgr[:n]

    def norm_box_to_xyxy(box_norm, width, height):
        cx, cy, bw, bh = box_norm
        return (cx - bw / 2) * width, (cy - bh / 2) * height, (cx + bw / 2) * width, (cy + bh / 2) * height

    ref_box_xyxy = norm_box_to_xyxy(orig_boxes_norm[0], w, h)
    print(f"frame 0 seed box (xyxy px, reused from original clip): {ref_box_xyxy}")

    frame0_rgb = cv2.cvtColor(frames_bgr[0], cv2.COLOR_BGR2RGB)
    predictor.set_image(frame0_rgb)
    if isinstance(predictor.features, tuple):
        predictor.features = predictor.features[0]
    masks, scores, _ = predictor.predict(box=np.array(ref_box_xyxy), multimask_output=False)
    print(f"SAM box-prompt score: {scores[0]:.3f}")

    image0 = get_input_frame_for_deva(frame0_rgb, config["size"])
    new_h, new_w = image0.shape[-2:]

    mask0_t = torch.from_numpy(masks[0].astype(np.float32)).cuda()
    mask0_resized = F.interpolate(mask0_t[None, None], (new_h, new_w), mode="bilinear")[0, 0] > 0.5
    output_mask = torch.zeros((new_h, new_w), dtype=torch.int64, device="cuda")
    output_mask[mask0_resized] = 1
    segments_info = [ObjectInfo(id=1, category_id=None, isthing=True, score=float(scores[0]))]

    prob = deva.incorporate_detection(image0, output_mask, segments_info)
    pred_mask0 = (prob.argmax(dim=0) == 1).cpu().numpy()
    box0 = mask_to_xyxy(pred_mask0) or ref_box_xyxy

    boxes_xyxy_resized = [box0]
    empty_frames = 0
    for i in range(1, n):
        frame_rgb = cv2.cvtColor(frames_bgr[i], cv2.COLOR_BGR2RGB)
        image = get_input_frame_for_deva(frame_rgb, config["size"])
        prob = deva.step(image, None, None)
        pred_mask = (prob.argmax(dim=0) == 1).cpu().numpy()
        box = mask_to_xyxy(pred_mask)
        if box is None:
            empty_frames += 1
            box = boxes_xyxy_resized[-1]  # hold last known box rather than drop the frame
        boxes_xyxy_resized.append(box)

    t_done = time.time()
    print(f"DEVA propagation done in {t_done - t0:.1f}s for {n} frames, "
          f"{empty_frames}/{n - 1} lost-track frames")

    boxes_xyxy_resized = np.array(boxes_xyxy_resized)  # (n, 4) in new_h/new_w space
    out_dir = pathlib.Path(WEIGHTS_DIR) / "outputs" / f"cam_type_{cam_type}"
    out_path = out_dir / f"deva_boxes_{seed_boxes}.pt"
    torch.save(
        {"boxes_xyxy": torch.from_numpy(boxes_xyxy_resized), "frame_h": new_h, "frame_w": new_w,
         "seed_box_xyxy_original_res": ref_box_xyxy, "empty_frames": empty_frames},
        out_path,
    )
    weights_volume.commit()
    print(f"persisted DEVA boxes to volume: {out_path}")

    return {
        "cam_type": cam_type,
        "deva_time_s": t_done - t0,
        "n_frames": n,
        "empty_frames": empty_frames,
        "out_path": str(out_path),
    }


@app.function(
    image=image,
    volumes={WEIGHTS_DIR: weights_volume},
    # CoTracker, DEVA, and CLIP feature extraction are all lightweight compared
    # to diffusion sampling; T4 is enough for all three combined.
    gpu="T4",
    timeout=900,
)
def build_real_pair(label: str, video_filename: str, pose_stride: int = 4, grid_size: int = 25) -> dict:
    """Assembles one REAL Stage 1 training pair (matching stage1_build_pairs.py's
    schema) from a TrajectoryCrafter-rendered clip: real CoTracker point tracks,
    real DEVA re-localized target boxes (car-seeded, same object used in the
    viewer's Video 7), real CLIP first-frame features, and the real interpolated
    reference box from the original MAVREC clip. This replaces every stub used
    in the earlier CPU-only smoke test (stage1_build_pairs.py's __main__ block)
    with the genuine article, now that TrajectoryCrafter's re-rendering (unlike
    ReCamMaster's) is trustworthy enough to build real training data from.
    """
    import sys
    import time

    sys.path.insert(0, "/opt/trace_src")
    sys.path.insert(0, "/opt/DEVA")
    import cv2
    import numpy as np
    import torch
    import torch.nn.functional as F
    from argparse import ArgumentParser

    torch.autograd.set_grad_enabled(False)

    video_path = pathlib.Path("/opt") / video_filename
    cap = cv2.VideoCapture(str(video_path))
    frames_bgr = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames_bgr.append(frame)
    cap.release()
    n = len(frames_bgr)
    h, w = frames_bgr[0].shape[:2]
    print(f"[{label}] loaded {n} frames at {w}x{h}")

    car_boxes_norm = np.load("/opt/car_boxes_normalized.npy")  # (81, 4), same object as Video 7

    # --- CoTracker: real point tracks ---
    from cotracker_wrapper import extract_point_track_grid, normalize_tracks
    ckpt_dir = pathlib.Path(WEIGHTS_DIR) / "CoTracker"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    cotracker_ckpt = ckpt_dir / "scaled_offline.pth"
    if not cotracker_ckpt.exists():
        import urllib.request
        urllib.request.urlretrieve(
            "https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth",
            str(cotracker_ckpt),
        )
        weights_volume.commit()

    t0 = time.time()
    video_np = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr])
    video_t = torch.from_numpy(video_np).permute(0, 3, 1, 2)[None].float()
    tracks_px = extract_point_track_grid(video_t, grid_size=grid_size, device="cuda", checkpoint=str(cotracker_ckpt))
    tracks_norm = normalize_tracks(tracks_px, width=w, height=h)  # (T, grid*grid, 2)
    print(f"[{label}] CoTracker done in {time.time() - t0:.1f}s")

    # --- DEVA: real re-localized target boxes, seeded on the car (Video 7's object) ---
    from deva.model.network import DEVA
    from deva.inference.inference_core import DEVAInferenceCore
    from deva.inference.object_info import ObjectInfo
    from deva.inference.eval_args import add_common_eval_args
    from deva.ext.ext_eval_args import add_ext_eval_args, add_text_default_args
    from deva.dataset.utils import im_normalization
    from deva.ext.MobileSAM.setup_mobile_sam import setup_model as setup_mobile_sam
    from segment_anything import SamPredictor

    def get_input_frame_for_deva(image_np, min_side):
        image = torch.from_numpy(image_np).permute(2, 0, 1).float().cuda() / 255
        image = im_normalization(image)
        if min_side > 0:
            hh, ww = image_np.shape[:2]
            scale = min_side / min(hh, ww)
            new_h, new_w = int(hh * scale), int(ww * scale)
            image = F.interpolate(image.unsqueeze(0), (new_h, new_w), mode="bilinear", align_corners=False)[0]
        return image

    def mask_to_xyxy(mask):
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return None
        return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())

    deva_dir = pathlib.Path(WEIGHTS_DIR) / "DEVA"
    parser = ArgumentParser()
    add_common_eval_args(parser)
    add_ext_eval_args(parser)
    add_text_default_args(parser)
    args = parser.parse_args([
        "--model", str(deva_dir / "DEVA-propagation.pth"),
        "--MOBILE_SAM_CHECKPOINT_PATH", str(deva_dir / "mobile_sam.pt"),
        "--size", "480",
    ])
    config = vars(args)
    config["enable_long_term"] = not config["disable_long_term"]
    config["enable_long_term_count_usage"] = (
        config["enable_long_term"]
        and (n / (config["max_mid_term_frames"] - config["min_mid_term_frames"])
             * config["num_prototypes"]) >= config["max_long_term_elements"]
    )

    t0 = time.time()
    network = DEVA(config).cuda().eval()
    network.load_weights(torch.load(config["model"], map_location="cuda"))
    mobile_sam = setup_mobile_sam()
    mobile_sam.load_state_dict(torch.load(config["MOBILE_SAM_CHECKPOINT_PATH"], map_location="cuda"), strict=True)
    mobile_sam.to(device="cuda").eval()
    predictor = SamPredictor(mobile_sam)
    deva = DEVAInferenceCore(network, config=config)
    deva.enabled_long_id()

    def norm_box_to_xyxy(box_norm, width, height):
        cx, cy, bw, bh = box_norm
        return (cx - bw / 2) * width, (cy - bh / 2) * height, (cx + bw / 2) * width, (cy + bh / 2) * height

    ref_box_xyxy = norm_box_to_xyxy(car_boxes_norm[0], w, h)
    frame0_rgb = cv2.cvtColor(frames_bgr[0], cv2.COLOR_BGR2RGB)
    predictor.set_image(frame0_rgb)
    if isinstance(predictor.features, tuple):
        predictor.features = predictor.features[0]
    masks, scores, _ = predictor.predict(box=np.array(ref_box_xyxy), multimask_output=False)
    print(f"[{label}] SAM seed score: {scores[0]:.3f}")

    image0 = get_input_frame_for_deva(frame0_rgb, config["size"])
    new_h, new_w = image0.shape[-2:]
    mask0_t = torch.from_numpy(masks[0].astype(np.float32)).cuda()
    mask0_resized = F.interpolate(mask0_t[None, None], (new_h, new_w), mode="bilinear")[0, 0] > 0.5
    output_mask = torch.zeros((new_h, new_w), dtype=torch.int64, device="cuda")
    output_mask[mask0_resized] = 1
    segments_info = [ObjectInfo(id=1, category_id=None, isthing=True, score=float(scores[0]))]
    prob = deva.incorporate_detection(image0, output_mask, segments_info)
    pred_mask0 = (prob.argmax(dim=0) == 1).cpu().numpy()
    box0 = mask_to_xyxy(pred_mask0) or ref_box_xyxy

    boxes_xyxy = [box0]
    empty_frames = 0
    for i in range(1, n):
        frame_rgb = cv2.cvtColor(frames_bgr[i], cv2.COLOR_BGR2RGB)
        image = get_input_frame_for_deva(frame_rgb, config["size"])
        prob = deva.step(image, None, None)
        pred_mask = (prob.argmax(dim=0) == 1).cpu().numpy()
        box = mask_to_xyxy(pred_mask)
        if box is None:
            empty_frames += 1
            box = boxes_xyxy[-1]
        boxes_xyxy.append(box)
    print(f"[{label}] DEVA done in {time.time() - t0:.1f}s, {empty_frames}/{n - 1} lost-track frames")

    boxes_xyxy = np.array(boxes_xyxy)  # (n, 4) in new_h/new_w space
    target_boxes_norm = np.stack([
        (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2 / new_w,
        (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) / 2 / new_h,
        (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) / new_w,
        (boxes_xyxy[:, 3] - boxes_xyxy[:, 1]) / new_h,
    ], axis=1)  # (n, 4) cx,cy,w,h normalized

    # --- Real first-frame feature via CLIP vision encoder (paper doesn't specify
    # the encoder; CLIP ViT-B/32's pooled output is a real, standard 768-dim
    # semantic image feature, not a random placeholder). ---
    from transformers import CLIPVisionModel, CLIPImageProcessor
    clip_dir = pathlib.Path(WEIGHTS_DIR) / "clip-vit-base-patch32"
    if not clip_dir.exists():
        CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32").save_pretrained(clip_dir)
        CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32").save_pretrained(clip_dir)
        weights_volume.commit()
    clip_model = CLIPVisionModel.from_pretrained(clip_dir).cuda().eval()
    clip_processor = CLIPImageProcessor.from_pretrained(clip_dir)
    inputs = clip_processor(images=frame0_rgb, return_tensors="pt").to("cuda")
    first_frame_feat = clip_model(**inputs).pooler_output[0].detach().cpu()  # (768,)
    print(f"[{label}] CLIP first_frame_feat shape {tuple(first_frame_feat.shape)}")

    # --- Stride to pose-sampled frames (matches stage1_build_pairs.py's
    # POSE_SAMPLE_STRIDE=4 convention) and assemble the final pair. ---
    pose_tracks = tracks_norm[::pose_stride]
    target_boxes = torch.from_numpy(target_boxes_norm[::pose_stride]).float()
    num_pose_frames = pose_tracks.shape[0]
    ref_boxes = torch.from_numpy(np.tile(car_boxes_norm[0], (num_pose_frames, 1))).float()

    pair = {
        "first_frame_feat": first_frame_feat,
        "point_tracks": pose_tracks,
        "ref_boxes": ref_boxes,
        "target_boxes": target_boxes,
        "seq_name": "scene_1",
        "cam_type": label,
    }

    out_path = pathlib.Path(WEIGHTS_DIR) / "outputs" / "real_pairs" / f"scene_1_{label}.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(pair, out_path)
    weights_volume.commit()
    print(f"[{label}] persisted real pair to volume: {out_path}")

    # Also persist the full-resolution (non-strided) tracks and boxes, purely
    # for visualization -- the training pair above only needs the strided
    # pose_tracks/target_boxes, but seeing all 49 frames overlaid is the only
    # way to actually SEE what these real CoTracker/DEVA outputs looked like.
    viz_path = pathlib.Path(WEIGHTS_DIR) / "outputs" / "real_pairs" / f"scene_1_{label}_viz.pt"
    torch.save(
        {"tracks_px": tracks_px, "video_h": h, "video_w": w,
         "boxes_xyxy": torch.from_numpy(boxes_xyxy), "deva_h": new_h, "deva_w": new_w},
        viz_path,
    )
    weights_volume.commit()
    print(f"[{label}] persisted full-res viz data to volume: {viz_path}")

    return {
        "label": label,
        "n_frames": n,
        "num_pose_frames": num_pose_frames,
        "empty_frames": empty_frames,
        "sam_seed_score": float(scores[0]),
        "out_path": str(out_path),
    }


@app.local_entrypoint()
def main(cam_type: str = "1", step: str = "render", seed_boxes: str = "orig"):
    if step == "render":
        result = run_one_trajectory.remote(cam_type=cam_type)
        print(result)
        print(
            "\nTo download the rendered clip locally, run:\n"
            f"  modal volume get trace-stage1-weights outputs/cam_type_{cam_type} ."
        )
    elif step == "cotracker":
        result = run_cotracker.remote(cam_type=cam_type)
        print(result)
        print(
            "\nTo download the tracks locally, run:\n"
            f"  modal volume get trace-stage1-weights outputs/cam_type_{cam_type}/cotracker_tracks.pt ."
        )
    elif step == "deva":
        result = run_deva.remote(cam_type=cam_type, seed_boxes=seed_boxes)
        print(result)
        print(
            "\nTo download the boxes locally, run:\n"
            f"  modal volume get trace-stage1-weights outputs/cam_type_{cam_type}/deva_boxes_{seed_boxes}.pt ."
        )
    elif step == "real_pairs":
        labels = [
            ("theta10", "traj_theta10.mp4"),
            ("theta15phi20", "traj_theta15phi20.mp4"),
            ("forward", "traj_forward.mp4"),
        ]
        for label, filename in labels:
            result = build_real_pair.remote(label=label, video_filename=filename)
            print(result)
    else:
        raise ValueError(f"unknown step: {step}")
