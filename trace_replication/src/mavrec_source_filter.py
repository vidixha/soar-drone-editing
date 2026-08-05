"""
Filters MAVREC's drone-view footage down to the near-static (hovering) subset
needed as Stage 1's source corpus (see notes/stage1_spec.md, open question 1),
and converts it into the metadata.csv + per-frame box format recam_wrapper.py /
ReCamMaster expect.

MAVREC (arxiv 2312.04548) is gated on Hugging Face (huggingface.co/datasets/
rjccv/MAVREC): requesting access (name/email/affiliation/country) and
acknowledging its CC-BY license is required before `ACCESS_INSTRUCTIONS.md`
reveals the actual download link, a Google Drive folder. We have access.

Confirmed real layout of that Drive folder (from the test-split JSON and the
dataset's own readme.md / videos_to_frames.py, not a guess):

  mavrec_root/
    Annotations/
      aerial_test_aligned_ids.json   # standard COCO: images[], annotations[], categories[]
      ground_test_aligned_ids.json
    unlabelled/
      video_scene_<N>.zip            # full-length, full-resolution raw video per scene,
                                      # ~6.8GB each, both drone and ground cameras

Only the drone view is used here. The drone is semi-static (hovering ~25-45m,
per the paper), which is why it's a plausible Stage 1 source unlike continuous-
flight aerial footage (VisDrone/UAVDT): it satisfies Stage 1's "static camera"
requirement while still being real drone footage, unlike GOT-10k.

Only the first ~900 frames (30 seconds) of each scene were annotated, and only
sparsely within that window (COCO `images[].frameID` values have irregular
gaps, e.g. scene 1 has 146 annotated frames spread across frameID 2..897).
This is actually a good match for Stage 1's own B_ref construction: the paper
builds its dense per-frame reference boxes by "temporally interpolating sparse
user-placed key boxes." MAVREC's sparse COCO boxes serve directly as those key
boxes; this module interpolates them across the dense frame range extracted
from the scene's raw video.

MAVREC is a multi-object detection dataset (10 categories per scene, no
persistent track id), so a single object per scene is picked via a greedy
nearest-center linker across the sparse annotated frames, same idea as
GOT-10k's single tracked object but adapted for irregular frame gaps. This
linking approach is ours, not paper-stated or dataset-provided.

The exact video filename inside each unzipped video_scene_<N>.zip (drone vs.
ground, naming convention) has not been confirmed yet -- the zips are large
(~6.8GB each) and are still being fetched as of this writing. `find_drone_video`
below documents the current best guess and will need adjusting once a zip is
actually unzipped and inspected.
"""
import json
import pathlib
from collections import defaultdict

import cv2
import numpy as np

from motion_filter import is_static_camera, sequence_motion_score

TARGET_FRAMES = 81   # Stage 1's ReCamMaster step operates on 81-frame clips
ANNOTATED_WINDOW_FRAMES = 900  # only the first ~30s (900 frames) of each scene is annotated at all
MAX_CENTER_DIST_NORM = 0.08  # per-annotated-frame box-center jump allowed to still count as the same tracked object, normalized by image diagonal; not scaled by frame-gap size, a known simplification given irregular annotation gaps


def load_coco_annotations(json_path: pathlib.Path) -> tuple[dict, dict]:
    """Returns (scenes, dims):
    scenes: {scene_id: [(frameID, file_name, [ (bbox_xywh_px, category_id), ... ]), ...]},
    sorted by frameID within each scene.
    dims: {scene_id: (img_h, img_w)}, read directly from the COCO images[] entries
    (assumed constant per scene, true in the confirmed real data).
    Matches the confirmed real schema: images[] has file_name/frameID/scene/
    width/height, annotations[] has image_id/bbox/category_id (standard COCO)."""
    with open(json_path) as f:
        coco = json.load(f)

    boxes_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        boxes_by_image[ann["image_id"]].append((ann["bbox"], ann["category_id"]))

    scenes = defaultdict(list)
    dims = {}
    for img in coco["images"]:
        image_id = img.get("id")
        scenes[img["scene"]].append((img["frameID"], img["file_name"], boxes_by_image.get(image_id, [])))
        dims[img["scene"]] = (int(img["height"]), int(img["width"]))

    for scene_id in scenes:
        scenes[scene_id].sort(key=lambda row: row[0])
    return scenes, dims


def link_single_object(
    frames: list[tuple[int, str, list]],
    img_h: int,
    img_w: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Greedy nearest-center linker over sparse annotated frames: starts from the
    largest box in the first annotated frame with any detections, then follows
    the closest-center box in each subsequent annotated frame within
    MAX_CENTER_DIST_NORM. Returns (frame_ids, boxes_px) for the linked frames,
    or None if no frame has a detection to start from."""
    diag = float(np.hypot(img_h, img_w))
    start_idx = next((i for i, (_, _, boxes) in enumerate(frames) if boxes), None)
    if start_idx is None:
        return None

    def center(box_xywh):
        x, y, w, h = box_xywh
        return np.array([x + w / 2, y + h / 2])

    linked_ids = []
    linked_boxes = []
    start_boxes = frames[start_idx][2]
    cur_box = max(start_boxes, key=lambda b: b[0][2] * b[0][3])[0]
    linked_ids.append(frames[start_idx][0])
    linked_boxes.append(cur_box)
    cur_center = center(cur_box)

    for i in range(start_idx + 1, len(frames)):
        frame_id, _, boxes_with_cat = frames[i]
        boxes = [b[0] for b in boxes_with_cat]
        if not boxes:
            continue
        dists = [np.hypot(*(center(b) - cur_center)) / diag for b in boxes]
        best = int(np.argmin(dists))
        if dists[best] > MAX_CENTER_DIST_NORM:
            continue
        cur_box = boxes[best]
        cur_center = center(cur_box)
        linked_ids.append(frame_id)
        linked_boxes.append(cur_box)

    if len(linked_ids) < 2:
        return None  # need at least 2 key boxes to interpolate a dense window
    return np.array(linked_ids), np.array(linked_boxes)


def interpolate_dense_boxes(
    key_frame_ids: np.ndarray,
    key_boxes_px: np.ndarray,
    target_frame_ids: np.ndarray,
) -> np.ndarray:
    """Builds B_ref: dense per-frame boxes for target_frame_ids, by linearly
    interpolating the sparse key boxes -- matching the paper's own description
    of B_ref as sparse key boxes interpolated to a dense sequence. Target frames
    outside [key_frame_ids.min(), key_frame_ids.max()] are clamped (np.interp's
    default edge behavior), not extrapolated."""
    dense = np.stack([
        np.interp(target_frame_ids, key_frame_ids, key_boxes_px[:, dim])
        for dim in range(4)
    ], axis=1)
    return dense


def boxes_to_normalized(boxes_xywh_px: np.ndarray, img_h: int, img_w: int) -> np.ndarray:
    """Pixel top-left (x, y, w, h) -> our (cx, cy, w, h) normalized-to-[0,1] convention."""
    x, y, w, h = boxes_xywh_px[:, 0], boxes_xywh_px[:, 1], boxes_xywh_px[:, 2], boxes_xywh_px[:, 3]
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    return np.stack([cx, cy, w / img_w, h / img_h], axis=1)


def find_drone_video(video_dir: pathlib.Path, scene_id: int) -> pathlib.Path | None:
    """Locates the drone-view video for a scene after video_scene_<N>.zip has been
    unzipped into video_dir. Naming convention not yet confirmed against a real
    unzipped file (see module docstring); this glob is a best guess based on the
    'droneView' token used in the labelled-frame filenames."""
    matches = list(video_dir.glob(f"*scene_{scene_id}_*droneView*")) or \
        list(video_dir.glob(f"*scene_{scene_id}_*drone*"))
    return matches[0] if matches else None


def extract_dense_frames(video_path: pathlib.Path, start_frame: int, num_frames: int) -> list[np.ndarray]:
    """Reads num_frames consecutive BGR frames from video_path starting at start_frame."""
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames = []
    for _ in range(num_frames):
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


def write_clip_mp4(frames_bgr: list[np.ndarray], out_path: pathlib.Path, fps: float = 24.0) -> None:
    """Encodes the extracted window's frames to an mp4, matching the file recam_wrapper's
    metadata.csv / ReCamMaster inference script expects (see recam_wrapper.py docstring)."""
    h, w = frames_bgr[0].shape[:2]
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for frame in frames_bgr:
        writer.write(frame)
    writer.release()


def filter_mavrec(mavrec_root: pathlib.Path, video_dir: pathlib.Path, limit: int | None = None) -> list[dict]:
    """
    Returns a list of {seq_name, video_path, window_start, frames_bgr,
    boxes_normalized, n_frames, motion_score} for every MAVREC drone-view scene
    that has a linkable single tracked object AND passes the hovering/near-static
    camera filter over a TARGET_FRAMES-length window within the annotated
    900-frame range. frames_bgr holds the already-extracted window (avoids a
    second video read); pass it to write_clip_mp4 to produce ReCamMaster's
    expected input file.
    """
    ann_path = mavrec_root / "Annotations" / "aerial_test_aligned_ids.json"
    scenes, dims = load_coco_annotations(ann_path)

    scene_ids = sorted(scenes)
    if limit:
        scene_ids = scene_ids[:limit]

    qualifying = []
    for scene_id in scene_ids:
        frames = scenes[scene_id]
        img_h, img_w = dims[scene_id]
        linked = link_single_object(frames, img_h, img_w)
        if linked is None:
            continue
        key_frame_ids, key_boxes_px = linked

        video_path = find_drone_video(video_dir, scene_id)
        if video_path is None:
            continue

        window_start = max(0, int(key_frame_ids.min()))
        window_end = min(ANNOTATED_WINDOW_FRAMES, window_start + TARGET_FRAMES)
        if window_end - window_start < TARGET_FRAMES:
            window_start = max(0, window_end - TARGET_FRAMES)

        color_frames = extract_dense_frames(video_path, window_start, TARGET_FRAMES)
        if len(color_frames) < TARGET_FRAMES:
            continue
        frames_gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in color_frames]

        score = sequence_motion_score(frames_gray, stride=5)
        if not is_static_camera(score):
            continue  # not hovering steadily enough for this POC's threshold

        target_frame_ids = np.arange(window_start, window_start + TARGET_FRAMES)
        dense_boxes_px = interpolate_dense_boxes(key_frame_ids, key_boxes_px, target_frame_ids)
        boxes_norm = boxes_to_normalized(dense_boxes_px, img_h, img_w)

        qualifying.append({
            "seq_name": f"scene_{scene_id}",
            "video_path": video_path,
            "window_start": window_start,
            "frames_bgr": color_frames,
            "caption": "a tracked object viewed from a hovering drone camera",
            "boxes_normalized": boxes_norm,
            "n_frames": TARGET_FRAMES,
            "motion_score": score,
        })

    return qualifying


if __name__ == "__main__":
    print("This module needs a real unzipped MAVREC video_scene_<N>.zip to run "
          "end to end. No smoke test runs here without real video -- "
          "motion_filter.py, the linker, and the interpolation logic are "
          "smoke-tested in isolation in stage1_build_pairs.py / tests below.")
