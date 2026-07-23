"""Verifies real camera motion in UAVDT the same way motion_probe.py did for VisDrone,
self-measured rather than trusting the dataset's attribute labels. See REPORT.md /
NOTES.md: VisDrone turned out to be near-hovering footage despite looking like a
plausible drone dataset, so this check is not skippable just because UAVDT's paper
claims altitude/camera-view attributes.
"""

import configparser
import json
import os

import cv2
import numpy as np

import sys
sys.path.insert(0, os.path.dirname(__file__))
import geometry as geom

UAVDT_ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "uavdt_raw", "UAVDT-2024-MOT")
SAMPLE_STRIDE = 8
MAX_FRAMES_PER_SEQ = 40


def list_sequences(split):
    split_dir = os.path.join(UAVDT_ROOT, split)
    return sorted(d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d)))


def seq_info(split, seq_id):
    cfg = configparser.ConfigParser()
    cfg.read(os.path.join(UAVDT_ROOT, split, seq_id, "seqinfo.ini"))
    s = cfg["Sequence"]
    return int(s["seqlength"]), int(s["imwidth"]), int(s["imheight"])


def frame_path(split, seq_id, frame_num):
    return os.path.join(UAVDT_ROOT, split, seq_id, "img1", f"{frame_num:06d}.jpg")


def measure_sequence(split, seq_id):
    n_frames, w, h = seq_info(split, seq_id)
    idxs = list(range(1, n_frames + 1, SAMPLE_STRIDE))[:MAX_FRAMES_PER_SEQ]
    imgs = []
    for i in idxs:
        p = frame_path(split, seq_id, i)
        if not os.path.exists(p):
            continue
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        imgs.append((i, img))

    transforms = []
    for (fn1, img1), (fn2, img2) in zip(imgs, imgs[1:]):
        M = geom.estimate_transform(img1, img2)
        if M is None:
            continue
        gap = fn2 - fn1
        transforms.append(dict(
            dx=float(M[0, 2]), dy=float(M[1, 2]), frame_gap=gap,
            angle_deg=geom.transform_rotation_deg(geom.affine_to_3x3(M)),
            scale=geom.transform_scale(geom.affine_to_3x3(M)),
        ))

    if not transforms:
        return dict(error="no valid transforms", n_frames_total=n_frames)

    trans_px = [np.hypot(t["dx"], t["dy"]) / t["frame_gap"] for t in transforms]
    rot_deg = [abs(t["angle_deg"]) / t["frame_gap"] for t in transforms]
    diag = float(np.hypot(w, h))
    return dict(
        n_frames_total=n_frames, n_sampled=len(imgs), n_transforms=len(transforms),
        width=w, height=h,
        translation_px_per_frame_median=float(np.median(trans_px)),
        translation_norm_per_frame_median=float(np.median(trans_px) / diag),
        rotation_deg_per_frame_median=float(np.median(rot_deg)),
    )


def main():
    results = {}
    for split in ["train", "val"]:
        for seq_id in list_sequences(split):
            print(f"[{split}/{seq_id}] measuring...", flush=True)
            r = measure_sequence(split, seq_id)
            results[f"{split}/{seq_id}"] = r
            if "error" not in r:
                print(f"  translation={r['translation_px_per_frame_median']:.2f}px/frame "
                      f"rotation={r['rotation_deg_per_frame_median']:.3f}deg/frame "
                      f"({r['n_frames_total']} frames, {r['width']}x{r['height']})", flush=True)
            else:
                print(f"  {r['error']}", flush=True)

    out_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "uavdt_motion_probe.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out_dir}/uavdt_motion_probe.json")

    valid = {k: v for k, v in results.items() if "error" not in v}
    trans_all = sorted(v["translation_px_per_frame_median"] for v in valid.values())
    print(f"\n{len(valid)}/{len(results)} sequences measured successfully")
    print(f"translation px/frame across sequences: min={trans_all[0]:.2f} "
          f"median={trans_all[len(trans_all)//2]:.2f} max={trans_all[-1]:.2f}")
    n_above_visdrone_max = sum(1 for t in trans_all if t > 2.76)  # VisDrone's max was 2.76
    print(f"sequences exceeding VisDrone's max observed motion (2.76 px/frame): "
          f"{n_above_visdrone_max}/{len(valid)}")


if __name__ == "__main__":
    main()
