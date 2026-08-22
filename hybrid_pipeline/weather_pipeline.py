"""Modal app: metric reconstruct + weather simulate.

Same reconstruct (Pi3X cameras + AerialMetric depth) and particle simulate
as the local reconstruction_weather path. Router calls this through
gpu_backend.GPUBackend.apply_weather / modal_backend.ModalBackend -- it
never imports Modal itself.

SAM3 dynamic masks are skipped here (Vista4D is not on this image).
Collision uses the static cloud from Pi3X + AerialMetric only.

Usage:
  modal deploy hybrid_pipeline/weather_pipeline.py
  modal run hybrid_pipeline/weather_pipeline.py --step download
"""
from __future__ import annotations

import pathlib
import sys

import modal

app = modal.App("aerie-weather")

weights_volume = modal.Volume.from_name("aerie-weather-weights", create_if_missing=True)
WEIGHTS_DIR = "/weights"
CHECKPOINT = f"{WEIGHTS_DIR}/Moge2-Aerial.pt"
HF_HOME = f"{WEIGHTS_DIR}/hf"

PIPE = pathlib.Path(__file__).resolve().parent
ROOT = PIPE.parent
PI3 = ROOT / "third_party" / "Pi3"
MOGE = ROOT / "third_party" / "AerialMetric" / "MoGe"

_IGNORE = ["**/.git/**", "**/.venv/**", "**/__pycache__/**", "**/*.pyc", "**/checkpoints/**"]

image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0", "wget")
    .pip_install("torch==2.4.0", "torchvision==0.19.0", index_url="https://download.pytorch.org/whl/cu121")
    .pip_install(
        "opencv-python-headless",
        "numpy",
        "scipy",
        "pillow",
        "safetensors",
        "einops",
        "timm",
        "peft>=0.10",
        "transformers>=4.40",
        "accelerate",
        "huggingface_hub<1.0",
        "hf_transfer",
        "imageio[ffmpeg]",
        "warp-lang",
        "utils3d @ git+https://github.com/EasternJournalist/utils3d.git@3fab839f0be9931dac7c8488eb0e1600c236e183",
    )
    .env(
        {
            "PYTHONPATH": "/opt/hybrid_pipeline:/opt/Pi3:/opt/AerialMetric/MoGe",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HOME": HF_HOME,
            "HUGGINGFACE_HUB_CACHE": f"{HF_HOME}/hub",
        }
    )
    .add_local_dir(str(PIPE), remote_path="/opt/hybrid_pipeline", ignore=_IGNORE)
    .add_local_dir(str(PI3), remote_path="/opt/Pi3", ignore=_IGNORE)
    .add_local_dir(str(MOGE), remote_path="/opt/AerialMetric/MoGe", ignore=_IGNORE)
)


def _require_sources() -> None:
    if not (PI3 / "pi3").is_dir():
        raise FileNotFoundError(
            f"Pi3 checkout missing at {PI3}. Run: git submodule update --init third_party/Pi3"
        )
    if not (MOGE / "moge").is_dir():
        raise FileNotFoundError(
            f"AerialMetric MoGe missing at {MOGE}. "
            "Run: git submodule update --init third_party/AerialMetric"
        )


@app.function(image=image, volumes={WEIGHTS_DIR: weights_volume}, timeout=1800, cpu=2)
def download_weights() -> None:
    """CPU-billed: AerialMetric checkpoint + Pi3X weights onto the volume."""
    import os
    from pathlib import Path

    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    os.environ["HF_HOME"] = HF_HOME
    from huggingface_hub import snapshot_download

    ckpt = Path(CHECKPOINT)
    if not ckpt.is_file():
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        import urllib.request

        url = (
            "https://huggingface.co/datasets/Kuiee/AerialMetric-ECCV2026/"
            "resolve/main/weights/Moge2-Aerial.pt"
        )
        print(f"downloading AerialMetric checkpoint -> {ckpt}", flush=True)
        urllib.request.urlretrieve(url, ckpt)
        print(f"AerialMetric checkpoint {ckpt.stat().st_size} bytes", flush=True)
    else:
        print("AerialMetric checkpoint already cached", flush=True)

    pi3x = Path(HF_HOME) / "hub"
    print("prefetching yyfz233/Pi3X ...", flush=True)
    snapshot_download("yyfz233/Pi3X", cache_dir=str(pi3x))
    weights_volume.commit()
    print("weather weights persisted", flush=True)


@app.function(image=image, gpu="L4", timeout=3600, volumes={WEIGHTS_DIR: weights_volume})
def weather_of_bytes(
    video_bytes: bytes,
    kind: str = "snow",
    intensity: str = "medium",
    fps: float = 30.0,
) -> bytes:
    """Router-callable: reconstruct the clip, simulate weather, return mp4."""
    import os
    import tempfile
    from pathlib import Path

    os.environ.setdefault("HF_HOME", HF_HOME)
    sys.path[:0] = [
        "/opt/hybrid_pipeline",
        "/opt/Pi3",
        "/opt/AerialMetric/MoGe",
    ]

    ckpt = Path(CHECKPOINT)
    if not ckpt.is_file():
        raise FileNotFoundError(
            f"AerialMetric checkpoint missing at {ckpt}. "
            "Run: modal run hybrid_pipeline/weather_pipeline.py --step download"
        )

    work = Path(tempfile.mkdtemp(prefix="aerie_weather_"))
    clip = work / "clip.mp4"
    recon = work / "recon"
    clip.write_bytes(video_bytes)

    import cv2

    cap = cv2.VideoCapture(str(clip))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if n_frames <= 0:
        raise ValueError("could not read any frames from the uploaded clip")
    num_frames = min(49, n_frames)

    from reconstruction_weather.config import ReconstructionConfig
    from reconstruction_weather.pipeline import reconstruct
    from weather import apply_weather
    import modules as M

    print(
        f"[aerie-weather] reconstruct {num_frames} frames, then {kind}/{intensity}",
        flush=True,
    )
    reconstruct(
        clip,
        recon,
        config=ReconstructionConfig(
            num_frames=num_frames,
            dynamic_segmentation="backend",
            save_visualization=False,
        ),
        aerialmetric_dir=Path("/opt/AerialMetric"),
        aerialmetric_python=Path(sys.executable),
        aerialmetric_checkpoint=ckpt,
        megasam_dir=Path("/opt/AerialMetric"),
        megasam_python=Path(sys.executable),
        megasam_checkpoint=ckpt,
        sam3_dir=Path("/opt/AerialMetric"),
        sam3_python=Path(sys.executable),
        sam3_checkpoint=None,
    )
    frames = M.load_clip(str(recon / "video.mp4"))
    weathered, detail = apply_weather(
        frames,
        None,
        kind,
        intensity,
        reconstruction_dir=recon,
        fps=fps,
    )
    print(f"[aerie-weather] {detail}", flush=True)
    out = work / "weathered.mp4"
    M.save_video(weathered, str(out), fps)
    return out.read_bytes()


@app.local_entrypoint()
def main(step: str = "download"):
    _require_sources()
    if step == "download":
        download_weights.remote()
        return
    raise ValueError(f"unknown step {step!r} (only 'download' is local)")
