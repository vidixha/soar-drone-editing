"""Improved training-free editing demo: LaMa for erasing (replaces cv2.inpaint,
which left a visible ghost/smudge), Lanczos resize + Poisson (seamlessClone)
blending for pasting (replaces the naive resize+overwrite, which looked
blurry and had hard seams), and a capped scale factor to avoid over-upscaling
a small crop into visible softness.

Run lama_erase.py first. It returns a zip with erased_lama.mp4, crop_NNN.png
per frame, and crops_and_boxes.npz. Unzip that into a "lama_output" folder
next to this script before running.
"""
import pathlib

import cv2
import numpy as np

HERE = pathlib.Path(__file__).parent
SRC = HERE / "lama_output"
OUT = HERE.parent.parent / "viewer"

boxes_px = np.load(SRC / "crops_and_boxes.npz")["boxes_px"]
T = len(boxes_px)

cap = cv2.VideoCapture(str(SRC / "erased_lama.mp4"))
erased_frames = []
while True:
    ret, f = cap.read()
    if not ret:
        break
    erased_frames.append(f)
cap.release()
h, w = erased_frames[0].shape[:2]

crops = [cv2.imread(str(SRC / f"crop_{i:03d}.png")) for i in range(T)]

# Same edited path as before, but cap scale growth at 1.15x instead of 1.4x --
# the earlier version's larger scale factor was the main source of visible
# blur when upscaling a small (~30x20px) crop.
edited_boxes_px = []
for t in range(T):
    x1, y1, x2, y2 = boxes_px[t]
    bw, bh = x2 - x1, y2 - y1
    shift_x = -int(40 * (t / (T - 1)))
    shift_y = int(25 * (t / (T - 1)))
    scale = 1.0 + 0.15 * (t / (T - 1))
    new_w, new_h = int(bw * scale), int(bh * scale)
    cx, cy = (x1 + x2) // 2 + shift_x, (y1 + y2) // 2 + shift_y
    nx1, ny1 = max(0, cx - new_w // 2), max(0, cy - new_h // 2)
    nx2, ny2 = min(w, nx1 + new_w), min(h, ny1 + new_h)
    edited_boxes_px.append((nx1, ny1, nx2, ny2))

edited_frames = []
for t in range(T):
    bg = erased_frames[t].copy()
    nx1, ny1, nx2, ny2 = edited_boxes_px[t]
    nw, nh = nx2 - nx1, ny2 - ny1
    crop = crops[t]
    if nw > 2 and nh > 2 and crop.size > 0:
        patch = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
        # Poisson blending (seamlessClone) instead of a hard overwrite --
        # blends the patch's edges into the background instead of leaving a
        # visible rectangular seam.
        mask = np.full((nh, nw), 255, dtype=np.uint8)
        center = (nx1 + nw // 2, ny1 + nh // 2)
        try:
            bg = cv2.seamlessClone(patch, bg, mask, center, cv2.NORMAL_CLONE)
        except cv2.error:
            bg[ny1:ny2, nx1:nx2] = patch  # fallback if seamlessClone rejects tiny/edge regions
    edited_frames.append(bg)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
tmp = OUT / "edit_demo_v2_tmp.mp4"
writer = cv2.VideoWriter(str(tmp), fourcc, 10.0, (w, h))
for fr in edited_frames:
    writer.write(fr)
writer.release()
print(f"wrote {tmp}, {T} frames")

# Also re-save the erased-only video at the right location/name for the viewer.
tmp_erased = OUT / "edit_demo_v2_erased_tmp.mp4"
writer = cv2.VideoWriter(str(tmp_erased), fourcc, 10.0, (w, h))
for fr in erased_frames:
    writer.write(fr)
writer.release()
print(f"wrote {tmp_erased}, {T} frames")
