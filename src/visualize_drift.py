"""Tier 0 qualitative figure: overlay predicted vs ground-truth boxes on real footage.

Re-renders what run_protocol_a.py already scored, no new modeling. Produces one mp4 per
chosen track showing ground truth (green), the static-box floor (red, frozen at the
reference-frame position), and chained-homography propagation (blue), frame by frame at
key-frame stride. This is the same overlay tracking papers use as their qualitative
figure: it turns "IoU 0.358" into a box visibly sliding off the object as the drone
moves, which is far more legible in a group sync than a metrics table.
"""

import json
import os
import sys

import cv2
import numpy as np

import geometry as geom
from data_loader import load_scenes, scene_size, tracks_by_index, download_frame_image
from methods import homography_chained
from scene_transforms import build_scene_transforms

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "drift_videos")

COLOR_GT = (0, 200, 0)        # green, BGR
COLOR_STATIC = (0, 0, 220)    # red
COLOR_HOMOG = (220, 130, 0)   # blue


def draw_box(img, box_xywh, color, label, label_y_offset=0):
    x, y, w, h = [int(round(v)) for v in box_xywh]
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
    ty = max(20, y - 8 - label_y_offset)
    cv2.putText(img, label, (x, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)


def render_track(scene_id, track_id, fps=3):
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

    by_frame_metrics = {}
    for r in recs:
        by_frame_metrics.setdefault(r["frame"], {})[r["method"]] = r

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"{scene_id}_track{track_id}_drift.mp4")
    writer = None

    for fn in frames_sorted:
        path = download_frame_image(scene_frames[fn])
        img = cv2.imread(path)

        gt_det = tracks_by_index(scene_frames[fn])[track_id]
        gt_box = geom.rel_box_to_abs(gt_det["bounding_box"], width, height)

        if fn == ref_frame:
            offset = 0
            static_iou = homog_iou = 1.0
            homog_box = ref_box
        else:
            offset = fn - ref_frame
            static_iou = by_frame_metrics[fn]["static_box"]["iou"]
            homog_iou = by_frame_metrics[fn]["homography_chained"]["iou"]
            homog_box = homography_chained(ref_box, T_from_first=T_from_first,
                                            ref_frame=ref_frame, target_frame=fn)

        draw_box(img, gt_box, COLOR_GT, "ground truth", label_y_offset=0)
        draw_box(img, ref_box, COLOR_STATIC, f"static IoU={static_iou:.2f}", label_y_offset=30)
        draw_box(img, homog_box, COLOR_HOMOG, f"homography IoU={homog_iou:.2f}", label_y_offset=60)

        cv2.putText(img, f"{scene_id}  track {track_id}  frame {fn}  offset +{offset}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

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
    render_track(scene_id, track_id)
