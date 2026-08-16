"""Feed-forward zero-shot depth for the harbor clip (Depth Anything V2) -> the
geometry the physically-based weather is built on. No per-scene fitting, one
forward pass. Static hover means one good depth map covers the clip, but we
export several frames for temporal use. Depth used for: Beer-Lambert fog
(correct, not a proxy), ground-plane fit -> wet-road reflections + puddles.
"""
import pathlib
import modal

app = modal.App("depth-extract")

image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.10")
    .apt_install("ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install("torch==2.4.0", "torchvision==0.19.0", index_url="https://download.pytorch.org/whl/cu121")
    .pip_install("transformers>=4.45", "accelerate", "opencv-python-headless", "Pillow",
                 "numpy", "huggingface_hub<1.0", "hf_transfer")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_file("/tmp/triplet_clips/clip_s4_car.mp4", remote_path="/opt/harbor.mp4")
)


@app.function(image=image, gpu="L4", timeout=1200)
def run() -> bytes:
    import io, zipfile
    import torch, cv2, numpy as np
    from PIL import Image
    from transformers import pipeline

    pipe = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Large-hf",
                    device=0, cache_dir="/tmp/hf")

    cap = cv2.VideoCapture("/opt/harbor.mp4"); frames = []
    while True:
        ok, f = cap.read()
        if not ok: break
        h, w = f.shape[:2]; s = 1280 / max(h, w)
        frames.append(cv2.resize(f, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA))
    cap.release()

    buf = io.BytesIO(); zf = zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED)
    # export depth for a sparse set of frames (hover -> static bg); 0,30,60,89
    for idx in [0, 30, 45, 60, 89]:
        rgb = cv2.cvtColor(frames[idx], cv2.COLOR_BGR2RGB)
        d = pipe(Image.fromarray(rgb))["predicted_depth"]  # tensor HxW (relative, larger=closer)
        d = torch.nn.functional.interpolate(d[None, None], size=rgb.shape[:2],
                                            mode="bilinear")[0, 0].cpu().numpy().astype(np.float32)
        # store as 16-bit PNG (normalized) + raw min/max for reconstruction
        dn = (d - d.min()) / (np.ptp(d) + 1e-6)
        ok, enc = cv2.imencode(".png", (dn * 65535).astype(np.uint16))
        zf.writestr(f"depth_{idx:05d}.png", enc.tobytes())
        zf.writestr(f"depth_{idx:05d}.txt", f"min {float(d.min())} max {float(d.max())}\n")
        print(f"depth frame {idx} done, range [{d.min():.2f},{d.max():.2f}]")
    zf.close()
    return buf.getvalue()


@app.function(image=image, gpu="L4", timeout=600)
def depth_of_bytes(video_bytes: bytes, frame_idx: int = 0) -> bytes:
    """Router-callable: real zero-shot depth for frame `frame_idx` of an
    arbitrary clip (returned as a 16-bit normalized PNG). Used to refresh
    depth after a trajectory edit changes the viewpoint, instead of falling
    back to a proxy that doesn't match the new geometry."""
    import io, torch, cv2, numpy as np
    from PIL import Image
    from transformers import pipeline

    pipe = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Large-hf",
                    device=0, cache_dir="/tmp/hf")
    tmp = "/tmp/router_in.mp4"; open(tmp, "wb").write(video_bytes)
    cap = cv2.VideoCapture(tmp); frames = []
    for _ in range(frame_idx + 1):
        ok, f = cap.read()
        if not ok: break
        frames.append(f)
    cap.release()
    f = frames[-1]
    rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
    d = pipe(Image.fromarray(rgb))["predicted_depth"]
    d = torch.nn.functional.interpolate(d[None, None], size=rgb.shape[:2],
                                        mode="bilinear")[0, 0].cpu().numpy().astype(np.float32)
    dn = (d - d.min()) / (np.ptp(d) + 1e-6)
    ok, enc = cv2.imencode(".png", (dn * 65535).astype(np.uint16))
    return enc.tobytes()


@app.local_entrypoint()
def main():
    open("/tmp/harbor_depth.zip", "wb").write(run.remote())
    print("saved /tmp/harbor_depth.zip")
