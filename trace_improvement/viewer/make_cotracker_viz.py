"""Overlay CoTracker's 25x25 point-track grid onto the rendered clip, so the
tracks can be viewed alongside the original/rendered videos in index.html."""
import pathlib

import cv2
import numpy as np
import torch

HERE = pathlib.Path(__file__).parent
tracks_data = torch.load(HERE / "cotracker_tracks.pt", map_location="cpu")
tracks_px = tracks_data["tracks_px"].numpy()  # (T, N, 2)

cap = cv2.VideoCapture(str(HERE / "rendered.mp4"))
fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)
cap.release()

T, N, _ = tracks_px.shape
assert T == len(frames), f"track length {T} != frame count {len(frames)}"

rng = np.random.default_rng(0)
colors = rng.integers(60, 255, size=(N, 3)).tolist()

TRAIL = 8  # frames of trailing history per point, for motion visibility
out_frames = []
for t in range(T):
    frame = frames[t].copy()
    for n in range(N):
        for k in range(max(0, t - TRAIL), t + 1):
            x, y = tracks_px[k, n]
            alpha = (k - max(0, t - TRAIL) + 1) / (TRAIL + 1)
            radius = 2 if k < t else 3
            color = tuple(int(c * alpha) for c in colors[n])
            cv2.circle(frame, (int(x), int(y)), radius, color, -1, lineType=cv2.LINE_AA)
    out_frames.append(frame)

h, w = out_frames[0].shape[:2]
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
tmp_path = HERE / "cotracker_viz_tmp.mp4"
writer = cv2.VideoWriter(str(tmp_path), fourcc, fps, (w, h))
for f in out_frames:
    writer.write(f)
writer.release()
print(f"wrote {tmp_path}, {T} frames at {fps}fps, {w}x{h}")
