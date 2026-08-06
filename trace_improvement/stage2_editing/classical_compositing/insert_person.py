"""Insert a real walking pedestrian (cropped from the original 4K MAVREC
footage, where two people are genuinely visible walking on a path) into the
TrajectoryCrafter-rendered scene. Uses real per-frame crops (so the legs
actually move like real walking, not a static sprite slid across the frame),
composited with Poisson blending onto a new, chosen path across the scene.
No synthetic/generated imagery -- the person is real, only their placement
in this specific scene is fabricated.

Note: this reads too small and too fast once resized to scene scale (see
Video 19 in the viewer). Kept for reference; anydoor_insertion is the better
approach for this task.

Needs the first 20 frames of the original source video extracted as PNGs
first:
  ffmpeg -i <source_video> -vf "select=lt(n\,20)" -vsync 0 person_frames/f%03d.png
"""
import pathlib

import cv2
import numpy as np

HERE = pathlib.Path(__file__).parent
FRAMES_DIR = HERE / "person_frames"
OUT = HERE.parent.parent / "viewer"
TARGET_VIDEO = OUT / "trajcrafter_gen.mp4"  # Video 4, theta=10 render

N_PERSON_FRAMES = 20
# Linearly-interpolated crop window per source frame (x shifts slightly left
# as the person walks along the path; y roughly fixed over this short span).
X0, X1 = 1838, 1826
Y0, Y1 = 360, 400
CROP_W, CROP_H = 24, 40

person_crops = []
for i in range(N_PERSON_FRAMES):
    frame = cv2.imread(str(FRAMES_DIR / f"f{i + 1:03d}.png"))
    t = i / (N_PERSON_FRAMES - 1)
    x = int(X0 + (X1 - X0) * t)
    y = int(Y0 + (Y1 - Y0) * t)
    crop = frame[y:y + CROP_H, x:x + CROP_W].copy()
    person_crops.append(crop)
print(f"extracted {len(person_crops)} real walking-person crops at native 4K res")

cap = cv2.VideoCapture(str(TARGET_VIDEO))
fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
frames = []
while True:
    ret, f = cap.read()
    if not ret:
        break
    frames.append(f)
cap.release()
h, w = frames[0].shape[:2]
T = len(frames)

# Deliberately visible scale for this demo -- true aerial scale would be only
# ~4-7px tall on this 672x384 render and essentially invisible. Chosen path:
# the real winding park path visible at the top of frame, the same path real
# pedestrians walk on in this scene (see the two tiny real figures there).
DISPLAY_H = 14  # px tall -- smaller than the first attempt, matches this path's real scale better
PATH_X0, PATH_X1 = 255, 375
PATH_Y0, PATH_Y1 = 72, 45

out_frames = []
for t in range(T):
    bg = frames[t].copy()
    person_idx = t % N_PERSON_FRAMES  # loop the real 20-frame gait cycle
    crop = person_crops[person_idx]
    ch, cw = crop.shape[:2]
    scale = DISPLAY_H / ch
    new_w, new_h = max(1, int(cw * scale)), DISPLAY_H
    patch = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    frac = t / (T - 1)
    cx = int(PATH_X0 + (PATH_X1 - PATH_X0) * frac)
    cy = int(PATH_Y0 + (PATH_Y1 - PATH_Y0) * frac)
    x1, y1 = cx - new_w // 2, cy - new_h // 2
    x2, y2 = x1 + new_w, y1 + new_h
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    pw, ph = x2 - x1, y2 - y1
    if pw > 2 and ph > 2:
        patch_fit = cv2.resize(patch, (pw, ph), interpolation=cv2.INTER_LANCZOS4)
        mask = np.full((ph, pw), 255, dtype=np.uint8)
        center = (x1 + pw // 2, y1 + ph // 2)
        try:
            bg = cv2.seamlessClone(patch_fit, bg, mask, center, cv2.NORMAL_CLONE)
        except cv2.error:
            bg[y1:y2, x1:x2] = patch_fit
    out_frames.append(bg)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
tmp = OUT / "person_insert_tmp.mp4"
writer = cv2.VideoWriter(str(tmp), fourcc, fps, (w, h))
for fr in out_frames:
    writer.write(fr)
writer.release()
print(f"wrote {tmp}, {T} frames")
