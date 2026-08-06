"""Overlay DEVA's re-localized per-frame box onto the rendered clip, so the
tracking result can be viewed alongside the other panels in index.html."""
import pathlib

import cv2
import torch

HERE = pathlib.Path(__file__).parent
d = torch.load(HERE / "deva_boxes.pt", map_location="cpu", weights_only=False)
boxes = d["boxes_xyxy"].numpy()  # (T, 4) xyxy in (frame_h, frame_w) space
deva_h, deva_w = d["frame_h"], d["frame_w"]

cap = cv2.VideoCapture(str(HERE / "rendered.mp4"))
fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)
cap.release()

h, w = frames[0].shape[:2]
sx, sy = w / deva_w, h / deva_h  # DEVA worked at its own resized resolution; scale boxes back

out_frames = []
for t, frame in enumerate(frames):
    frame = frame.copy()
    x1, y1, x2, y2 = boxes[t]
    x1, y1, x2, y2 = int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2, lineType=cv2.LINE_AA)
    out_frames.append(frame)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
tmp_path = HERE / "deva_viz_tmp.mp4"
writer = cv2.VideoWriter(str(tmp_path), fourcc, fps, (w, h))
for f in out_frames:
    writer.write(f)
writer.release()
print(f"wrote {tmp_path}, {len(out_frames)} frames at {fps}fps, {w}x{h}")
