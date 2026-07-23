"""Loads VisDrone-MOT ground truth directly from the FiftyOne export's samples.json.

Bypasses FiftyOne and MongoDB entirely, see NOTES.md section 1.1 for why: this machine
has no working mongod, and the HF repo has no parquet/loading-script for plain
`datasets.load_dataset` either. samples.json is the underlying export and has everything
needed: filepath, scene_id, frame_number, and per-frame detections with relative
bounding boxes, track index, label, occlusion, visibility.
"""

import json
import os
from collections import defaultdict

from huggingface_hub import hf_hub_download

REPO_ID = "Voxel51/visdrone-mot"
SAMPLES_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "samples.json")
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "frames_cache")


def load_scenes():
    """Returns dict: scene_id -> {frame_number: frame_dict}, sorted by frame_number."""
    with open(SAMPLES_PATH) as f:
        data = json.load(f)
    by_scene = defaultdict(dict)
    for s in data["samples"]:
        by_scene[s["scene_id"]][s["frame_number"]] = s
    return dict(by_scene)


def scene_size(scene_frames):
    """Width, height from the first frame's metadata. Constant within a scene."""
    first = scene_frames[min(scene_frames)]
    m = first["metadata"]
    return m["width"], m["height"]


def download_frame_image(frame):
    return hf_hub_download(
        repo_id=REPO_ID,
        filename=frame["filepath"],
        repo_type="dataset",
        local_dir=CACHE_DIR,
    )


def tracks_by_index(frame):
    return {d["index"]: d for d in frame["detections"] if d["label"] != "ignored_region"}
