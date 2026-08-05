"""
Filters MAVREC's drone-view footage down to the near-static (hovering) subset
needed as Stage 1's source corpus (see notes/stage1_spec.md, open question 1),
and converts it into the metadata.csv + per-frame box format recam_wrapper.py /
ReCamMaster expect.

MAVREC (arxiv 2312.04548) is gated on Hugging Face (huggingface.co/datasets/
rjccv/MAVREC): it requires requesting access (name/email/affiliation/country)
and acknowledging the CC-BY license before ACCESS_INSTRUCTIONS.md reveals the
actual download link. We have not been through that gate yet, so the directory
layout below is our best-guess reconstruction from the paper's description
("extended MSCOCO" annotations, synchronized ground+drone views, 10 object
classes), not a confirmed layout. It will need adjusting once real files are in
hand. Assumed layout:

  mavrec_root/
    images/
      drone/
        <scene_id>/
          000001.jpg ...
      ground/
        <scene_id>/
          000001.jpg ...
    annotations/
      drone_annotations.json   # COCO format: images[], annotations[], categories[]
      ground_annotations.json

Only the drone view is used here. The drone is semi-static (hovering ~25-45m,
per the paper), which is why it's a plausible Stage 1 source unlike continuous-
flight aerial footage (VisDrone/UAVDT): it satisfies Stage 1's "static camera"
requirement while still being real drone footage, unlike GOT-10k.

MAVREC is a multi-object detection dataset, not a single-object tracking
dataset like GOT-10k -- there is no persistent track id given per box. Stage 1
needs one tracked object's box per frame (B_ref), so this module picks a single
object per scene via a simple greedy nearest-center linker across frames. This
linking approach is ours, not paper-stated or dataset-provided.
"""
import json
import pathlib
from collections import defaultdict

import cv2
import numpy as np

from motion_filter import is_static_camera, sequence_motion_score

TARGET_FRAMES = 81   # Stage 1's ReCamMaster step operates on 81-frame clips
MAX_CENTER_DIST_NORM = 0.08  # frame-to-frame box-center jump allowed to still count as the same tracked object, normalized by image diagonal


def load_coco_annotations(json_path: pathlib.Path) -> dict:
    """Returns {scene_id: [(frame_idx, file_name, [ (bbox_xywh_px, category_id), ... ]), ...]},
    sorted by frame_idx within each scene. Assumes file_name is '<scene_id>/<frame_idx>.jpg'
    and categories/annotations follow standard COCO fields (bbox, category_id, image_id)."""
    with open(json_path) as f:
        coco = json.load(f)

    images_by_id = {img["id"]: img for img in coco["images"]}
    boxes_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        boxes_by_image[ann["image_id"]].append((ann["bbox"], ann["category_id"]))

    scenes = defaultdict(list)
    for image_id, img in images_by_id.items():
        scene_id, frame_name = pathlib.PurePosixPath(img["file_name"]).parts[-2:]
        frame_idx = int(pathlib.PurePosixPath(frame_name).stem)
        scenes[scene_id].append((frame_idx, img["file_name"], boxes_by_image.get(image_id, [])))

    for scene_id in scenes:
        scenes[scene_id].sort(key=lambda row: row[0])
    return scenes


def link_single_object(
    frames: list[tuple[int, str, list]],
    img_h: int,
    img_w: int,
) -> np.ndarray | None:
    """Greedy nearest-center linker: starts from the largest box in the first frame
    with any detections, then follows the closest-center box in each subsequent
    frame within MAX_CENTER_DIST_NORM. Returns a (n_frames, 4) array of pixel-space
    (x, y, w, h) boxes, or None if no frame has a detection to start from."""
    diag = float(np.hypot(img_h, img_w))
    start_idx = next((i for i, (_, _, boxes) in enumerate(frames) if boxes), None)
    if start_idx is None:
        return None

    def center(box_xywh):
        x, y, w, h = box_xywh
        return np.array([x + w / 2, y + h / 2])

    track = [None] * len(frames)
    start_boxes = frames[start_idx][2]
    cur_box = max(start_boxes, key=lambda b: b[0][2] * b[0][3])[0]
    track[start_idx] = cur_box
    cur_center = center(cur_box)

    for i in range(start_idx + 1, len(frames)):
        boxes = [b[0] for b in frames[i][2]]
        if not boxes:
            break
        dists = [np.hypot(*(center(b) - cur_center)) / diag for b in boxes]
        best = int(np.argmin(dists))
        if dists[best] > MAX_CENTER_DIST_NORM:
            break
        cur_box = boxes[best]
        cur_center = center(cur_box)
        track[i] = cur_box

    valid = [b for b in track if b is not None]
    if len(valid) < TARGET_FRAMES:
        return None
    return np.array(valid[:TARGET_FRAMES])


def boxes_to_normalized(boxes_xywh_px: np.ndarray, img_h: int, img_w: int) -> np.ndarray:
    """Pixel top-left (x, y, w, h) -> our (cx, cy, w, h) normalized-to-[0,1] convention."""
    x, y, w, h = boxes_xywh_px[:, 0], boxes_xywh_px[:, 1], boxes_xywh_px[:, 2], boxes_xywh_px[:, 3]
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    return np.stack([cx, cy, w / img_w, h / img_h], axis=1)


def load_frames_gray(image_dir: pathlib.Path, file_names: list[str]) -> list[np.ndarray]:
    return [cv2.imread(str(image_dir / name), cv2.IMREAD_GRAYSCALE) for name in file_names]


def filter_mavrec(mavrec_root: pathlib.Path, limit: int | None = None) -> list[dict]:
    """
    Returns a list of {seq_name, caption, boxes_normalized, n_frames, motion_score}
    for every MAVREC drone-view scene that has >= TARGET_FRAMES of a single linkable
    tracked object AND passes the hovering/near-static camera filter.
    """
    ann_path = mavrec_root / "annotations" / "drone_annotations.json"
    image_root = mavrec_root / "images" / "drone"
    scenes = load_coco_annotations(ann_path)

    scene_ids = sorted(scenes)
    if limit:
        scene_ids = scene_ids[:limit]

    qualifying = []
    for scene_id in scene_ids:
        frames = scenes[scene_id]
        if len(frames) < TARGET_FRAMES:
            continue

        file_names = [f[1] for f in frames]
        frames_gray = load_frames_gray(image_root, [pathlib.PurePosixPath(n).name for n in file_names])
        if any(f is None for f in frames_gray):
            continue
        img_h, img_w = frames_gray[0].shape

        boxes_px = link_single_object(frames, img_h, img_w)
        if boxes_px is None:
            continue

        score = sequence_motion_score(frames_gray[:TARGET_FRAMES], stride=5)
        if not is_static_camera(score):
            continue  # not hovering steadily enough for this POC's threshold

        boxes_norm = boxes_to_normalized(boxes_px, img_h, img_w)
        qualifying.append({
            "seq_name": scene_id,
            "caption": "a tracked object viewed from a hovering drone camera",
            "boxes_normalized": boxes_norm,
            "n_frames": TARGET_FRAMES,
            "motion_score": score,
        })

    return qualifying


if __name__ == "__main__":
    print("This module needs real MAVREC data to run end to end (gated on "
          "Hugging Face, see module docstring). No smoke test runs here without "
          "real data -- motion_filter.py and stage1_build_pairs.py's synthetic "
          "smoke test cover the reusable logic in isolation.")
