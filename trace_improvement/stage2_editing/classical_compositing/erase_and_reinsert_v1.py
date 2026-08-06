"""Training-free object motion editing demo: no diffusion, no fine-tuning, pure
classical compositing. Uses the real DEVA box (already computed) on Video 4
(TrajectoryCrafter, theta=10) to:
  1. erase the object from its real, tracked location (classical inpainting)
  2. crop the object out of each frame at its real location
  3. paste it back along a DIFFERENT, hand-specified path, i.e. an edit

This is a sanity-check baseline for the "training-free Stage 2" brainstorm --
crude, box-based (not a pixel-accurate mask), but zero GPU cost and a direct
test of whether the erase+reinsert mechanic looks plausible before spending on
any of the diffusion-based training-free methods (ObjCtrl-2.5D, DiTraj, etc).
"""
import pathlib

import cv2
import numpy as np
import torch

HERE = pathlib.Path(__file__).parent
VIEWER = HERE.parent.parent / "viewer"

d = torch.load(HERE / "inputs" / "real_viz_theta10.pt", map_location="cpu", weights_only=False)
boxes_xyxy = d["boxes_xyxy"].numpy()  # (T, 4) in deva_h/deva_w space
deva_h, deva_w = d["deva_h"], d["deva_w"]

cap = cv2.VideoCapture(str(VIEWER / "trajcrafter_gen.mp4"))
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

T = min(len(frames), len(boxes_xyxy))
PAD = 4  # px padding around the box when cropping/erasing, to catch antialiased edges

boxes_px = []
for t in range(T):
    x1, y1, x2, y2 = boxes_xyxy[t]
    x1, y1, x2, y2 = int(x1 * sx) - PAD, int(y1 * sy) - PAD, int(x2 * sx) + PAD, int(y2 * sy) + PAD
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    boxes_px.append((x1, y1, x2, y2))

# --- Step 1+2: erase the object from its real location, save the crop ---
erased_frames = []
crops = []
for t in range(T):
    x1, y1, x2, y2 = boxes_px[t]
    frame = frames[t]
    crop = frame[y1:y2, x1:x2].copy()
    crops.append(crop)

    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    erased = cv2.inpaint(frame, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
    erased_frames.append(erased)

# --- Step 3: define a new, edited path -- shift the real path further left and
# down, exaggerating the same general direction of motion the car already had,
# to simulate "make the object go further/faster along a related but different
# path" rather than reproducing the exact original trajectory. ---
edited_boxes_px = []
for t in range(T):
    x1, y1, x2, y2 = boxes_px[t]
    bw, bh = x2 - x1, y2 - y1
    shift_x = -int(40 * (t / (T - 1)))   # drift further left over time than reality did
    shift_y = int(25 * (t / (T - 1)))    # drift down over time (toward camera, larger)
    scale = 1.0 + 0.4 * (t / (T - 1))    # grow slightly, consistent with the down/forward drift
    new_w, new_h = int(bw * scale), int(bh * scale)
    cx, cy = (x1 + x2) // 2 + shift_x, (y1 + y2) // 2 + shift_y
    nx1, ny1 = max(0, cx - new_w // 2), max(0, cy - new_h // 2)
    nx2, ny2 = min(w, nx1 + new_w), min(h, ny1 + new_h)
    edited_boxes_px.append((nx1, ny1, nx2, ny2))

# --- Step 4: composite the cropped object onto the erased background along
# the edited path, resizing each crop to its new box size. ---
edited_frames = []
for t in range(T):
    bg = erased_frames[t].copy()
    nx1, ny1, nx2, ny2 = edited_boxes_px[t]
    nw, nh = nx2 - nx1, ny2 - ny1
    if nw > 0 and nh > 0 and crops[t].size > 0:
        patch = cv2.resize(crops[t], (nw, nh))
        bg[ny1:ny2, nx1:nx2] = patch
        cv2.rectangle(bg, (nx1, ny1), (nx2, ny2), (0, 200, 255), 1, lineType=cv2.LINE_AA)
    edited_frames.append(bg)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
for name, out_frames in [("erased", erased_frames), ("edited", edited_frames)]:
    tmp = HERE / f"edit_demo_{name}_tmp.mp4"
    writer = cv2.VideoWriter(str(tmp), fourcc, fps, (w, h))
    for fr in out_frames:
        writer.write(fr)
    writer.release()
print(f"wrote edit_demo_erased_tmp.mp4 and edit_demo_edited_tmp.mp4, {T} frames")
