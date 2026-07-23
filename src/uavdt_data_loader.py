"""Loads UAVDT-2024-MOT (local files, already extracted, no download needed) into the
same schema data_loader.py uses for VisDrone, so the rest of the pipeline
(static_track_selector, methods, metrics, run_protocol_a) works unchanged on either
dataset. Only this file and scene_transforms' image-loading hook are dataset-specific.

Known limitation, not hidden: UAVDT's gt.txt documents a <visibility> column but the
dataset's own README says it is "set to 1" (constant), so occlusion analysis is not
meaningful on this dataset the way it was on VisDrone's real occlusion flag.
"""

import configparser
import os

UAVDT_ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "uavdt_raw", "UAVDT-2024-MOT")
CLASS_NAMES = {0: "car", 1: "truck", 2: "bus", 3: "van"}


def list_sequences():
    seqs = []
    for split in ["train", "val"]:
        split_dir = os.path.join(UAVDT_ROOT, split)
        for seq_id in sorted(os.listdir(split_dir)):
            if os.path.isdir(os.path.join(split_dir, seq_id)):
                seqs.append((split, seq_id))
    return seqs


def load_scenes():
    """Returns dict: scene_id -> {frame_number: frame_dict}, matching data_loader's
    VisDrone schema (relative [x,y,w,h] boxes, 'index'=track id, 'label', 'occlusion')."""
    scenes = {}
    for split, seq_id in list_sequences():
        seq_dir = os.path.join(UAVDT_ROOT, split, seq_id)
        cfg = configparser.ConfigParser()
        cfg.read(os.path.join(seq_dir, "seqinfo.ini"))
        s = cfg["Sequence"]
        n_frames, width, height = int(s["seqlength"]), int(s["imwidth"]), int(s["imheight"])

        frames = {i: dict(filepath=os.path.join(seq_dir, "img1", f"{i:06d}.jpg"),
                           metadata=dict(width=width, height=height), detections=[])
                  for i in range(1, n_frames + 1)}

        gt_path = os.path.join(seq_dir, "gt", "gt.txt")
        with open(gt_path) as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 8:
                    continue
                frame_idx, target_id, left, top, w, h, conf, cls_id = parts[:8]
                frame_idx, target_id, cls_id = int(frame_idx), int(target_id), int(cls_id)
                left, top, w, h = float(left), float(top), float(w), float(h)
                if frame_idx not in frames or w <= 0 or h <= 0:
                    continue
                frames[frame_idx]["detections"].append(dict(
                    index=target_id,
                    label=CLASS_NAMES.get(cls_id, f"class_{cls_id}"),
                    bounding_box=[left / width, top / height, w / width, h / height],
                    occlusion=0,  # UAVDT's visibility column is constant, see module docstring
                ))

        scene_id = f"{split}_{seq_id}"
        scenes[scene_id] = frames
    return scenes


def scene_size(scene_frames):
    first = scene_frames[min(scene_frames)]
    m = first["metadata"]
    return m["width"], m["height"]


def download_frame_image(frame):
    """Same name/signature as data_loader's, for drop-in use by scene_transforms. No
    actual download, files are already local."""
    return frame["filepath"]


def tracks_by_index(frame):
    return {d["index"]: d for d in frame["detections"]}
