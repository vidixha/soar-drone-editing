"""Modal app: try LaMa (a proven, lightweight image-inpainting model) instead
of classical cv2.inpaint for the "erase the object" half of the training-free
Stage 2 editing demo. cv2.inpaint left a visible gray smudge where the car
was (couldn't cleanly reconstruct the road/grass boundary) -- LaMa is
specifically built for exactly this kind of structural inpainting and should
do meaningfully better, at much lower cost than a full video diffusion model.

Usage:
  modal run modal/lama_pipeline.py::main
"""
import pathlib

import modal

app = modal.App("lama-erase-test")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg")
    .pip_install("torch", "torchvision", "opencv-python-headless", "numpy", "simple-lama-inpainting")
    .add_local_file(
        str(pathlib.Path(__file__).parent.parent.parent / "viewer" / "trajcrafter_gen.mp4"),
        remote_path="/opt/trajcrafter_gen.mp4",
    )
    .add_local_file(
        str(pathlib.Path(__file__).parent / "inputs" / "real_viz_theta10.pt"),
        remote_path="/opt/real_viz_theta10.pt",
    )
)


@app.function(image=image, gpu="T4", timeout=600)
def erase_with_lama() -> dict:
    import time

    import cv2
    import numpy as np
    import torch
    from PIL import Image
    from simple_lama_inpainting import SimpleLama

    d = torch.load("/opt/real_viz_theta10.pt", map_location="cpu", weights_only=False)
    boxes_xyxy = d["boxes_xyxy"].numpy()
    deva_h, deva_w = d["deva_h"], d["deva_w"]

    cap = cv2.VideoCapture("/opt/trajcrafter_gen.mp4")
    frames = []
    while True:
        ret, f = cap.read()
        if not ret:
            break
        frames.append(f)
    cap.release()
    h, w = frames[0].shape[:2]
    sx, sy = w / deva_w, h / deva_h
    T = min(len(frames), len(boxes_xyxy))

    print("loading LaMa...")
    simple_lama = SimpleLama(device="cuda")

    PAD = 6
    erased_frames = []
    crops = []
    boxes_px = []
    t0 = time.time()
    for t in range(T):
        x1, y1, x2, y2 = boxes_xyxy[t]
        x1, y1, x2, y2 = int(x1 * sx) - PAD, int(y1 * sy) - PAD, int(x2 * sx) + PAD, int(y2 * sy) + PAD
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        boxes_px.append((x1, y1, x2, y2))
        crops.append(frames[t][y1:y2, x1:x2].copy())

        mask = np.zeros((h, w), dtype=np.uint8)
        mask[y1:y2, x1:x2] = 255
        frame_rgb = cv2.cvtColor(frames[t], cv2.COLOR_BGR2RGB)
        result = simple_lama(Image.fromarray(frame_rgb), Image.fromarray(mask))
        result_bgr = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
        erased_frames.append(result_bgr)
        if t % 10 == 0:
            print(f"  frame {t}/{T}")
    print(f"LaMa erase done in {time.time() - t0:.1f}s for {T} frames")

    out_dir = pathlib.Path("/tmp/lama_out")
    out_dir.mkdir(exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_dir / "erased_lama.mp4"), fourcc, 10.0, (w, h))
    for fr in erased_frames:
        writer.write(fr)
    writer.release()

    # Also save crops + boxes for the compositing step to reuse locally.
    np.savez(
        out_dir / "crops_and_boxes.npz",
        boxes_px=np.array(boxes_px),
    )
    for i, c in enumerate(crops):
        cv2.imwrite(str(out_dir / f"crop_{i:03d}.png"), c)

    import shutil
    shutil.make_archive("/tmp/lama_result", "zip", out_dir)
    with open("/tmp/lama_result.zip", "rb") as f:
        data = f.read()
    return {"n_frames": T, "time_s": time.time() - t0, "zip_bytes": data}


@app.local_entrypoint()
def main():
    result = erase_with_lama.remote()
    with open("/tmp/lama_result.zip", "wb") as f:
        f.write(result["zip_bytes"])
    print(f"done: {result['n_frames']} frames in {result['time_s']:.1f}s, saved to /tmp/lama_result.zip")
