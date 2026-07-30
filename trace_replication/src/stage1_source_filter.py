"""
Filters GOT-10k sequences down to the near-static-camera subset needed as Stage 1's
source corpus (see notes/stage1_spec.md, open question 1), and converts them into
the metadata.csv + per-frame box format recam_wrapper.py / ReCamMaster expect.

GOT-10k is not hosted on Hugging Face; it requires manual download from
http://got-10k.aitestunion.com/ (free registration). This script assumes the
official layout after download+extract:

  got10k_root/
    train/
      GOT-10k_Train_000001/
        00000001.jpg ...
        groundtruth.txt      # one "x,y,w,h" pixel box per frame
        meta_info.ini        # contains the object class name, used as caption text

We do not attempt to download GOT-10k ourselves here (it's behind a registration
wall, not a public direct-download URL) -- that step is manual, on whoever runs
this for real.
"""
import configparser
import csv
import pathlib

import cv2
import numpy as np

from motion_filter import is_static_camera, sequence_motion_score

TARGET_FRAMES = 81   # Stage 1's ReCamMaster step operates on 81-frame clips
FRAME_SIZE_HW = (480, 832)


def read_got10k_boxes(seq_dir: pathlib.Path) -> np.ndarray:
    """GOT-10k groundtruth.txt: one 'x,y,w,h' pixel-space row per frame."""
    gt_path = seq_dir / "groundtruth.txt"
    return np.loadtxt(gt_path, delimiter=",")


def read_got10k_caption(seq_dir: pathlib.Path) -> str:
    """meta_info.ini has an 'object_class' field; used as a minimal text prompt
    since GOT-10k has no natural-language captions."""
    ini_path = seq_dir / "meta_info.ini"
    config = configparser.ConfigParser()
    with open(ini_path) as f:
        config.read_string("[meta]\n" + f.read())
    obj_class = config.get("meta", "object_class", fallback="object").strip('"')
    return f"a {obj_class} in a static outdoor scene"


def load_frames_gray(seq_dir: pathlib.Path, max_frames: int = 60) -> list[np.ndarray]:
    frame_paths = sorted(seq_dir.glob("*.jpg"))[:max_frames]
    return [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in frame_paths]


def boxes_to_normalized(boxes_xywh_px: np.ndarray, img_h: int, img_w: int) -> np.ndarray:
    """GOT-10k boxes are top-left (x, y, w, h) in pixels; convert to our
    (cx, cy, w, h) normalized-to-[0,1] convention used across stage1/stage2."""
    x, y, w, h = boxes_xywh_px[:, 0], boxes_xywh_px[:, 1], boxes_xywh_px[:, 2], boxes_xywh_px[:, 3]
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    return np.stack([cx, cy, w / img_w, h / img_h], axis=1)


def filter_got10k(got10k_root: pathlib.Path, split: str = "train", limit: int | None = None) -> list[dict]:
    """
    Returns a list of {seq_name, caption, boxes_normalized, n_frames} for every
    sequence that passes the static-camera filter, capped at 7,500 to match the
    paper's Stage 1 source-video count (see notes/stage1_spec.md).
    """
    split_dir = got10k_root / split
    seq_dirs = sorted(p for p in split_dir.iterdir() if p.is_dir())
    if limit:
        seq_dirs = seq_dirs[:limit]

    qualifying = []
    for seq_dir in seq_dirs:
        frames = load_frames_gray(seq_dir)
        if len(frames) < TARGET_FRAMES:
            continue  # need at least 81 frames for the downstream ReCamMaster step
        score = sequence_motion_score(frames, stride=5)
        if not is_static_camera(score):
            continue

        img_h, img_w = frames[0].shape
        boxes_px = read_got10k_boxes(seq_dir)
        boxes_norm = boxes_to_normalized(boxes_px, img_h, img_w)
        caption = read_got10k_caption(seq_dir)

        qualifying.append({
            "seq_name": seq_dir.name,
            "caption": caption,
            "boxes_normalized": boxes_norm[:TARGET_FRAMES],
            "n_frames": len(frames),
            "motion_score": score,
        })
        if len(qualifying) >= 7500:
            break

    return qualifying


def write_metadata_csv(qualifying: list[dict], video_dir: pathlib.Path, out_csv: pathlib.Path) -> None:
    """Matches the file_name,text schema recam_wrapper.py's TextVideoCameraDataset expects.
    Assumes each qualifying sequence has already been encoded to an mp4 named
    '<seq_name>.mp4' under video_dir (frame->video encoding is a separate, ordinary
    ffmpeg step, not included here)."""
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file_name", "text"])
        for item in qualifying:
            file_name = f"{item['seq_name']}.mp4"
            assert (video_dir / file_name).exists(), f"missing {file_name}, encode frames to mp4 first"
            writer.writerow([file_name, item["caption"]])


if __name__ == "__main__":
    print("This module needs a real GOT-10k download to run end to end; see the "
          "module docstring for the expected directory layout. No smoke test runs "
          "here without real data -- motion_filter.py already covers the "
          "motion-scoring logic in isolation.")
