"""Tier 1: naive compositing baseline. This is Module B from the original TASK.md
(before the task was rewritten into the pure box-propagation POC), rebuilt here because
it makes the drift concrete rather than abstract: pasting a visible object at the
propagated box position turns a number into something that visibly looks wrong.

Deliberately crude on purpose, per the original Module B spec: a procedurally drawn
sprite (no real asset library available, one was explicitly out of scope), pasted flat
with no relighting, no perspective warp beyond the box's own scale, and no depth-gated
occlusion (Depth-Anything-V3 is not installed in this environment, see NOTES.md 1.5, so
occlusion gating that TASK.md originally specified is skipped, not faked). The ugliness
is the point: it is the floor any learned insertion model has to beat, and the sim-to-real
gap is meant to be visible, not hidden.
"""

import json
import os
import sys

import cv2
import numpy as np

import geometry as geom
from data_loader import load_scenes, scene_size, tracks_by_index, download_frame_image
from methods import homography_chained, static_box
from scene_transforms import build_scene_transforms

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "composite_videos")


def make_person_sprite(h=200, w=90):
    """Procedural placeholder sprite (BGRA), deliberately not photorealistic, no asset
    library was in scope. A bright, obviously-synthetic color so it is unmistakable in
    the composite, not trying to pass as real."""
    sprite = np.zeros((h, w, 4), dtype=np.uint8)
    color = (0, 140, 255, 255)  # orange, BGRA
    head_r = w // 4
    head_cx, head_cy = w // 2, head_r + 4
    cv2.circle(sprite, (head_cx, head_cy), head_r, color, -1)
    body_top = head_cy + head_r
    cv2.ellipse(sprite, (w // 2, (body_top + h) // 2), (w // 3, (h - body_top) // 2),
                0, 0, 360, color, -1)
    # soften edges slightly so the paste isn't a hard rectangle of alpha
    sprite[:, :, 3] = cv2.GaussianBlur(sprite[:, :, 3], (5, 5), 0)
    return sprite


def paste_sprite(frame_bgr, sprite_bgra, box_xywh):
    x, y, w, h = [int(round(v)) for v in box_xywh]
    w, h = max(1, w), max(1, h)
    fh, fw = frame_bgr.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(fw, x + w), min(fh, y + h)
    if x1 <= x0 or y1 <= y0:
        return frame_bgr
    resized = cv2.resize(sprite_bgra, (w, h), interpolation=cv2.INTER_LINEAR)
    sub = resized[y0 - y:y1 - y, x0 - x:x1 - x]
    alpha = sub[:, :, 3:4].astype(np.float32) / 255.0
    region = frame_bgr[y0:y1, x0:x1].astype(np.float32)
    blended = alpha * sub[:, :, :3].astype(np.float32) + (1 - alpha) * region
    frame_bgr[y0:y1, x0:x1] = blended.astype(np.uint8)
    return frame_bgr


def render_composite(scene_id, track_id, fps=3):
    scenes = load_scenes()
    scene_frames = scenes[scene_id]
    width, height = scene_size(scene_frames)

    with open(os.path.join(RESULTS_DIR, "protocol_a_records.json")) as f:
        all_records = json.load(f)
    recs = [r for r in all_records if r["scene_id"] == scene_id and r["track_id"] == track_id]
    if not recs:
        raise ValueError(f"no records for {scene_id} track {track_id}")

    ref_frame = recs[0]["ref_frame"]
    later_frames = sorted(set(r["frame"] for r in recs))
    frames_sorted = [ref_frame] + later_frames

    key_frames, T_from_first, step_info, images = build_scene_transforms(scene_frames)

    ref_det = tracks_by_index(scene_frames[ref_frame])[track_id]
    ref_box = geom.rel_box_to_abs(ref_det["bounding_box"], width, height)
    sprite = make_person_sprite()

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"{scene_id}_track{track_id}_composite.mp4")
    writer = None

    for fn in frames_sorted:
        path = download_frame_image(scene_frames[fn])
        img = cv2.imread(path)

        if fn == ref_frame:
            homog_box = ref_box
        else:
            homog_box = homography_chained(ref_box, T_from_first=T_from_first,
                                            ref_frame=ref_frame, target_frame=fn)

        # No depth-gated occlusion: not implemented, this is naive compositing only, the
        # sprite will incorrectly stay fully visible even if the real object would be
        # occluded behind structure. Documented, not hidden.
        img = paste_sprite(img, sprite, homog_box)

        offset = 0 if fn == ref_frame else fn - ref_frame
        cv2.putText(img, f"{scene_id} track {track_id} frame {fn} offset +{offset}  "
                          f"(naive composite, no occlusion gating, no relighting)",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        if writer is None:
            h, w = img.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        writer.write(img)

    writer.release()
    print(f"wrote {out_path} ({len(frames_sorted)} frames)")
    return out_path


if __name__ == "__main__":
    scene_id = sys.argv[1] if len(sys.argv) > 1 else "uav0000137_00458_v"
    track_id = int(sys.argv[2]) if len(sys.argv) > 2 else 24
    render_composite(scene_id, track_id)
