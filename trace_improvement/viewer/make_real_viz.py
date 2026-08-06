"""Render CoTracker point-track and DEVA box overlays for the real Stage 1
pairs built on TrajectoryCrafter's three renders, matching the style of the
earlier Video 6/7 (ReCamMaster) visualizations."""
import pathlib

import cv2
import numpy as np
import torch

HERE = pathlib.Path(__file__).parent

SOURCES = {
    "theta10": "trajcrafter_gen.mp4",
    "theta15phi20": "trajcrafter_gen2.mp4",
    "forward": "trajcrafter_gen3.mp4",
}

for label, video_name in SOURCES.items():
    d = torch.load(HERE / f"real_viz_{label}.pt", map_location="cpu", weights_only=False)
    tracks_px = d["tracks_px"].numpy()  # (T, N, 2)
    boxes_xyxy = d["boxes_xyxy"].numpy()  # (T, 4) in deva_h/deva_w space
    deva_h, deva_w = d["deva_h"], d["deva_w"]

    cap = cv2.VideoCapture(str(HERE / video_name))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frames = []
    while True:
        ret, f = cap.read()
        if not ret:
            break
        frames.append(f)
    cap.release()
    h, w = frames[0].shape[:2]
    sx, sy = w / deva_w, h / deva_h

    T, N, _ = tracks_px.shape
    rng = np.random.default_rng(0)
    colors = rng.integers(60, 255, size=(N, 3)).tolist()
    TRAIL = 8

    track_frames = []
    box_frames = []
    for t in range(min(T, len(frames))):
        tf = frames[t].copy()
        for nidx in range(N):
            for k in range(max(0, t - TRAIL), t + 1):
                x, y = tracks_px[k, nidx]
                alpha = (k - max(0, t - TRAIL) + 1) / (TRAIL + 1)
                radius = 2 if k < t else 3
                color = tuple(int(c * alpha) for c in colors[nidx])
                cv2.circle(tf, (int(x), int(y)), radius, color, -1, lineType=cv2.LINE_AA)
        track_frames.append(tf)

        bf = frames[t].copy()
        x1, y1, x2, y2 = boxes_xyxy[t]
        x1, y1, x2, y2 = int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)
        cv2.rectangle(bf, (x1, y1), (x2, y2), (0, 255, 0), 2, lineType=cv2.LINE_AA)
        box_frames.append(bf)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    for name, out_frames in [("tracks", track_frames), ("box", box_frames)]:
        tmp = HERE / f"real_{label}_{name}_tmp.mp4"
        writer = cv2.VideoWriter(str(tmp), fourcc, fps, (w, h))
        for fr in out_frames:
            writer.write(fr)
        writer.release()
    print(f"wrote real_{label}_tracks_tmp.mp4 and real_{label}_box_tmp.mp4, {len(track_frames)} frames")
