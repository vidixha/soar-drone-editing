"""AnyDoor (github.com/ali-vilab/AnyDoor) for inserting a reference object
into a target scene, in place of a plain pixel copy-paste. Zero-shot, no
fine-tuning needed for a new object or scene.

Produces a more recognizable object shape than a flat pixel paste when
viewed closely, but the result is still small and easy to miss at normal
video scale. Needs a Poisson blend pass afterward (already included) to
remove a border artifact from the raw model output.

Usage:
  modal run pipeline.py::main --step download
  modal run pipeline.py::main --step check
  modal run pipeline.py::main --step test
  modal run pipeline.py::main --step video
"""
import pathlib

import modal

app = modal.App("anydoor-insert-test")

weights_volume = modal.Volume.from_name("anydoor-weights", create_if_missing=True)
WEIGHTS_DIR = "/weights"

image = (
    modal.Image.from_registry("nvidia/cuda:11.8.0-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install("setuptools==66.0.0", "wheel")
    .pip_install(
        "torch==2.0.0", "torchvision==0.15.1",
        index_url="https://download.pytorch.org/whl/cu118",
    )
    .run_commands(
        "git clone --depth 1 https://github.com/ali-vilab/AnyDoor.git /opt/AnyDoor",
    )
    .pip_install(
        "albumentations==1.3.0", "einops==0.3.0", "fvcore==0.1.5.post20221221",
        "omegaconf==2.1.1", "open_clip_torch==2.17.1",
        # opencv_contrib_python==4.3.0.36 (AnyDoor's pin) has no wheel for
        # Python 3.10 on this index; dropped since nothing in the repo uses
        # contrib-only modules (ximgproc/xphoto/etc) -- base opencv covers it.
        "opencv_python==4.7.0.72",
        "opencv_python_headless==4.7.0.72", "Pillow==9.4.0",
        "pytorch_lightning==1.5.0", "safetensors==0.2.7", "scipy==1.9.1",
        # share==1.0.4 (AnyDoor's pin) doesn't exist on PyPI at all -- only
        # used by tool_add_control_sd21.py (a weight-conversion script we
        # don't need, since we're using the pretrained checkpoint directly).
        "timm==0.6.12", "torchmetrics==0.6.0", "tqdm==4.65.0",
        "transformers==4.19.2",
    )
    # xformers==0.0.18 (AnyDoor's pin) is no longer downloadable for cu118 at
    # all -- dropped since every xformers import in the codebase is wrapped
    # in a try/except with a vanilla-attention fallback, and the config's
    # attn_type line that would request it is commented out anyway.
    .pip_install("huggingface_hub<1.0", "hf_transfer")
    .add_local_file(
        str(pathlib.Path(__file__).parent / "inputs" / "trajcrafter_gen.mp4"),
        remote_path="/opt/trajcrafter_gen.mp4",
    )
    .add_local_file(
        str(pathlib.Path(__file__).parent / "inputs" / "ref_crop.png"),
        remote_path="/opt/ref_crop.png",
    )
    .add_local_file(
        str(pathlib.Path(__file__).parent / "inputs" / "ref_mask.png"),
        remote_path="/opt/ref_mask.png",
    )
    .add_local_file(
        str(pathlib.Path(__file__).parent / "inputs" / "target_frame.png"),
        remote_path="/opt/target_frame.png",
    )
    .add_local_file(
        str(pathlib.Path(__file__).parent / "inputs" / "target_mask.png"),
        remote_path="/opt/target_mask.png",
    )
)


@app.function(image=image, volumes={WEIGHTS_DIR: weights_volume}, timeout=3600, cpu=2)
def download_weights():
    """CPU-only: downloads AnyDoor's checkpoint (~16.8GB, from an HF Space
    repo) and DINOv2 ViT-g/14 backbone (~4.5GB, from fbaipublicfiles) into
    the persistent volume."""
    import os
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    from huggingface_hub import hf_hub_download

    ckpt_dir = pathlib.Path(WEIGHTS_DIR)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    anydoor_ckpt = ckpt_dir / "epoch=1-step=8687.ckpt"
    if not anydoor_ckpt.exists():
        downloaded = hf_hub_download(
            repo_id="xichenhku/AnyDoor", repo_type="space",
            filename="epoch=1-step=8687.ckpt", local_dir=str(ckpt_dir),
        )
        print(f"downloaded AnyDoor checkpoint to {downloaded}")
        weights_volume.commit()
    else:
        print("AnyDoor checkpoint already cached")

    dinov2_ckpt = ckpt_dir / "dinov2_vitg14_pretrain.pth"
    if not dinov2_ckpt.exists():
        import urllib.request
        print("downloading DINOv2 ViT-g/14 backbone...")
        urllib.request.urlretrieve(
            "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_pretrain.pth",
            str(dinov2_ckpt),
        )
        print(f"downloaded DINOv2 to {dinov2_ckpt}")
        weights_volume.commit()
    else:
        print("DINOv2 checkpoint already cached")

    print("done")


@app.function(image=image, cpu=1, timeout=120)
def check_imports():
    """CPU-only: verify AnyDoor's import chain works before spending on GPU."""
    import sys
    sys.path.insert(0, "/opt/AnyDoor")
    errors = []
    try:
        from cldm.model import create_model, load_state_dict  # noqa: F401
        print("OK: cldm.model imports")
    except Exception as e:
        errors.append(f"cldm.model: {e}")
    try:
        from cldm.ddim_hacked import DDIMSampler  # noqa: F401
        print("OK: cldm.ddim_hacked imports")
    except Exception as e:
        errors.append(f"cldm.ddim_hacked: {e}")
    try:
        from datasets.data_utils import get_bbox_from_mask  # noqa: F401
        print("OK: datasets.data_utils imports")
    except Exception as e:
        errors.append(f"datasets.data_utils: {e}")
    if errors:
        print("FAILED:", errors)
        raise RuntimeError(str(errors))
    print("all imports OK")


@app.function(image=image, volumes={WEIGHTS_DIR: weights_volume}, gpu="L4", timeout=600)
def run_single_test() -> bytes:
    """Single-frame test: insert our real reference person into one frame of
    the TrajectoryCrafter render, using AnyDoor's zero-shot diffusion
    insertion instead of raw pixel copy-paste. Cheapest possible signal on
    whether this approach can produce something recognizable at the tiny
    scale we need, before committing to a full 48-frame video loop.
    """
    import sys
    import os

    os.chdir("/opt/AnyDoor")
    sys.path.insert(0, "/opt/AnyDoor")

    # Point AnyDoor's config at our downloaded checkpoints before its
    # top-level model-loading code runs on import.
    from omegaconf import OmegaConf
    cfg = OmegaConf.load("configs/inference.yaml")
    cfg.pretrained_model = f"{WEIGHTS_DIR}/epoch=1-step=8687.ckpt"
    OmegaConf.save(cfg, "configs/inference.yaml")

    anydoor_cfg = OmegaConf.load("configs/anydoor.yaml")
    anydoor_cfg.model.params.cond_stage_config.weight = f"{WEIGHTS_DIR}/dinov2_vitg14_pretrain.pth"
    OmegaConf.save(anydoor_cfg, "configs/anydoor.yaml")

    import cv2
    import numpy as np

    import run_inference as ai  # noqa: E402  (executes AnyDoor's model load at import time)

    ref_image = cv2.cvtColor(cv2.imread("/opt/ref_crop.png"), cv2.COLOR_BGR2RGB)
    ref_mask = (cv2.imread("/opt/ref_mask.png", cv2.IMREAD_GRAYSCALE) > 128).astype(np.uint8)
    tar_image = cv2.cvtColor(cv2.imread("/opt/target_frame.png"), cv2.COLOR_BGR2RGB)
    tar_mask = (cv2.imread("/opt/target_mask.png", cv2.IMREAD_GRAYSCALE) > 128).astype(np.uint8)

    print("running AnyDoor inference...")
    gen_image = ai.inference_single_image(ref_image, ref_mask, tar_image.copy(), tar_mask)
    print("inference done, shape:", gen_image.shape)

    gen_bgr = cv2.cvtColor(gen_image.astype(np.uint8), cv2.COLOR_RGB2BGR)
    out_path = "/tmp/anydoor_result.png"
    cv2.imwrite(out_path, gen_bgr)
    with open(out_path, "rb") as f:
        return f.read()


@app.function(image=image, volumes={WEIGHTS_DIR: weights_volume}, gpu="L4", timeout=3600)
def run_full_video() -> bytes:
    """Full 49-frame version: insert the reference person into every frame of
    the TrajectoryCrafter render, walking along the same park path, using
    AnyDoor for each frame (model loaded once, reused across all frames to
    avoid reloading the 16.8GB checkpoint 49 times), then a Poisson-blend
    cleanup pass per frame to remove the black letterbox border seen in the
    single-frame test.
    """
    import sys
    import os
    import time

    os.chdir("/opt/AnyDoor")
    sys.path.insert(0, "/opt/AnyDoor")

    from omegaconf import OmegaConf
    cfg = OmegaConf.load("configs/inference.yaml")
    cfg.pretrained_model = f"{WEIGHTS_DIR}/epoch=1-step=8687.ckpt"
    OmegaConf.save(cfg, "configs/inference.yaml")

    anydoor_cfg = OmegaConf.load("configs/anydoor.yaml")
    anydoor_cfg.model.params.cond_stage_config.weight = f"{WEIGHTS_DIR}/dinov2_vitg14_pretrain.pth"
    OmegaConf.save(anydoor_cfg, "configs/anydoor.yaml")

    import cv2
    import numpy as np

    import run_inference as ai  # noqa: E402  (loads the model once at import time)

    ref_image = cv2.cvtColor(cv2.imread("/opt/ref_crop.png"), cv2.COLOR_BGR2RGB)
    ref_mask = (cv2.imread("/opt/ref_mask.png", cv2.IMREAD_GRAYSCALE) > 128).astype(np.uint8)

    cap = cv2.VideoCapture("/opt/trajcrafter_gen.mp4")
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    frames = []
    while True:
        ret, f = cap.read()
        if not ret:
            break
        frames.append(f)
    cap.release()
    T = len(frames)
    h, w = frames[0].shape[:2]

    # Same park-path walking line used in the classical attempt, but this
    # time each frame's generation is independent (no real gait cycle), so
    # motion realism relies purely on scale/position changing smoothly frame
    # to frame -- not on real walking frames like the earlier attempt.
    PATH_X0, PATH_X1 = 255, 375
    PATH_Y0, PATH_Y1 = 72, 45
    MASK_W, MASK_H = 8, 16

    out_frames = []
    t0 = time.time()
    for t in range(T):
        frac = t / (T - 1)
        cx = int(PATH_X0 + (PATH_X1 - PATH_X0) * frac)
        cy = int(PATH_Y0 + (PATH_Y1 - PATH_Y0) * frac)

        tar_image = cv2.cvtColor(frames[t], cv2.COLOR_BGR2RGB)
        tar_mask = np.zeros((h, w), dtype=np.uint8)
        y1m, y2m = max(0, cy - MASK_H // 2), min(h, cy + MASK_H // 2)
        x1m, x2m = max(0, cx - MASK_W // 2), min(w, cx + MASK_W // 2)
        tar_mask[y1m:y2m, x1m:x2m] = 1

        gen_image = ai.inference_single_image(ref_image, ref_mask, tar_image.copy(), tar_mask)
        gen_bgr = cv2.cvtColor(gen_image.astype(np.uint8), cv2.COLOR_RGB2BGR)

        # Poisson-blend cleanup: extract just the generated region (padded a
        # bit beyond the mask, matching the single-frame test), drop
        # near-black letterbox pixels, blend the rest onto the clean frame.
        py1, py2 = max(0, y1m - 5), min(h, y2m + 5)
        px1, px2 = max(0, x1m - 5), min(w, x2m + 5)
        patch = gen_bgr[py1:py2, px1:px2]
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        blend_mask = (gray > 15).astype(np.uint8) * 255
        result = frames[t].copy()
        center = (px1 + (px2 - px1) // 2, py1 + (py2 - py1) // 2)
        try:
            result = cv2.seamlessClone(patch, result, blend_mask, center, cv2.NORMAL_CLONE)
        except cv2.error:
            mask_bool = blend_mask > 0
            region = result[py1:py2, px1:px2]
            region[mask_bool] = patch[mask_bool]
            result[py1:py2, px1:px2] = region
        out_frames.append(result)
        print(f"frame {t + 1}/{T} done, {time.time() - t0:.1f}s elapsed")

    out_path = "/tmp/anydoor_video_out.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    for fr in out_frames:
        writer.write(fr)
    writer.release()
    print(f"wrote {out_path}, {T} frames, total {time.time() - t0:.1f}s")

    with open(out_path, "rb") as f:
        return f.read()


@app.local_entrypoint()
def main(step: str = "download"):
    if step == "download":
        download_weights.remote()
    elif step == "check":
        check_imports.remote()
    elif step == "test":
        result = run_single_test.remote()
        with open("/tmp/anydoor_result.png", "wb") as f:
            f.write(result)
        print("saved result to /tmp/anydoor_result.png")
    elif step == "video":
        result = run_full_video.remote()
        with open("/tmp/anydoor_video_out.mp4", "wb") as f:
            f.write(result)
        print("saved video to /tmp/anydoor_video_out.mp4")
    else:
        raise ValueError(f"unknown step: {step}")
