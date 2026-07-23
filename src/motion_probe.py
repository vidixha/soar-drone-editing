"""Section 1 verification: measure real inter-frame camera motion per VisDrone-MOT scene.

Estimates a frame-to-frame partial-affine transform (rotation, uniform scale, translation)
from ORB feature matches on background regions. This is a coarse proxy, not the final
homography baseline, used only to confirm which scenes have real camera motion versus
near-static hovering, as required by TASK.md Section 1 point 2.
"""

import json
import os
from collections import defaultdict

import cv2
import numpy as np
from huggingface_hub import hf_hub_download

REPO_ID = "Voxel51/visdrone-mot"
SAMPLES_PATH = "data/raw/samples.json"
CACHE_DIR = "data/frames_cache"
SAMPLE_STRIDE = 8
MAX_FRAMES_PER_SCENE = 40


def load_index():
    with open(SAMPLES_PATH) as f:
        data = json.load(f)
    by_scene = defaultdict(list)
    for s in data["samples"]:
        by_scene[s["scene_id"]].append(s)
    for scene_id in by_scene:
        by_scene[scene_id].sort(key=lambda s: s["frame_number"])
    return by_scene


def download_frame(filepath):
    return hf_hub_download(
        repo_id=REPO_ID,
        filename=filepath,
        repo_type="dataset",
        local_dir=CACHE_DIR,
    )


def estimate_transform(img1_gray, img2_gray):
    orb = cv2.ORB_create(nfeatures=800)
    k1, d1 = orb.detectAndCompute(img1_gray, None)
    k2, d2 = orb.detectAndCompute(img2_gray, None)
    if d1 is None or d2 is None or len(k1) < 8 or len(k2) < 8:
        return None
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(d1, d2)
    if len(matches) < 8:
        return None
    matches = sorted(matches, key=lambda m: m.distance)[:200]
    pts1 = np.float32([k1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([k2[m.trainIdx].pt for m in matches])
    M, inliers = cv2.estimateAffinePartial2D(pts1, pts2, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if M is None:
        return None
    dx, dy = M[0, 2], M[1, 2]
    angle = np.degrees(np.arctan2(M[1, 0], M[0, 0]))
    scale = np.sqrt(M[0, 0] ** 2 + M[1, 0] ** 2)
    n_inliers = int(inliers.sum()) if inliers is not None else 0
    return dict(dx=float(dx), dy=float(dy), angle_deg=float(angle), scale=float(scale),
                n_matches=len(matches), n_inliers=n_inliers)


def main():
    by_scene = load_index()
    results = {}
    for scene_id, frames in by_scene.items():
        n = len(frames)
        idxs = list(range(0, n, SAMPLE_STRIDE))[:MAX_FRAMES_PER_SCENE]
        sampled = [frames[i] for i in idxs]
        print(f"[{scene_id}] {n} frames total, sampling {len(sampled)} at stride {SAMPLE_STRIDE}")

        imgs = []
        for s in sampled:
            path = download_frame(s["filepath"])
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            imgs.append((s["frame_number"], img))

        transforms = []
        for (fn1, img1), (fn2, img2) in zip(imgs, imgs[1:]):
            t = estimate_transform(img1, img2)
            if t is not None:
                t["frame_gap"] = fn2 - fn1
                transforms.append(t)

        if not transforms:
            results[scene_id] = {"error": "no valid transforms estimated"}
            continue

        h, w = imgs[0][1].shape
        diag = float(np.hypot(h, w))
        trans_mag_px = [np.hypot(t["dx"], t["dy"]) / t["frame_gap"] for t in transforms]
        trans_mag_norm = [m / diag for m in trans_mag_px]
        rot_deg = [abs(t["angle_deg"]) / t["frame_gap"] for t in transforms]
        scale_dev = [abs(t["scale"] - 1.0) / t["frame_gap"] for t in transforms]

        results[scene_id] = {
            "n_frames_total": n,
            "n_sampled": len(sampled),
            "n_transforms": len(transforms),
            "image_size_hw": [h, w],
            "translation_px_per_frame_median": float(np.median(trans_mag_px)),
            "translation_norm_per_frame_median": float(np.median(trans_mag_norm)),
            "rotation_deg_per_frame_median": float(np.median(rot_deg)),
            "scale_change_per_frame_median": float(np.median(scale_dev)),
        }
        print(f"  translation median {results[scene_id]['translation_px_per_frame_median']:.2f} px/frame, "
              f"rotation median {results[scene_id]['rotation_deg_per_frame_median']:.3f} deg/frame, "
              f"scale-change median {results[scene_id]['scale_change_per_frame_median']:.4f} /frame")

    os.makedirs("results", exist_ok=True)
    with open("results/motion_probe.json", "w") as f:
        json.dump(results, f, indent=2)
    print("wrote results/motion_probe.json")


if __name__ == "__main__":
    main()
