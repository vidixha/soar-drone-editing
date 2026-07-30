"""
Builds Stage 2 training pairs from OpenVid-1M clips + DEVA object masks.
See notes/stage2_spec.md ("How training pairs are actually built from DEVA masks")
for the reconstruction-as-training / editing-as-inference reasoning this
implements.

Requires, on the machine actually running this:
  - OpenVid-1M downloaded locally (huggingface.co/datasets/nkp37/OpenVid-1M)
  - DEVA installed (github.com/hkchengrex/Tracking-Anything-with-DEVA), used in
    its "automatic" mode (SAM-based open-world segmentation + temporal
    propagation) to get per-object mask sequences without manual prompts, since
    OpenVid-1M has no object annotations of its own.

This module does not call DEVA directly (that's demo_automatic.py, a full CLI
script with its own model loading) -- it consumes DEVA's output JSON/mask format
and turns it into the (video, M_obj, M_inpaint, text) training tuples
stage2_lora.py's train_step expects.
"""
import json
import pathlib

import numpy as np
import torch


def mask_to_box(mask: np.ndarray) -> np.ndarray | None:
    """Binary (H, W) mask -> (cx, cy, w, h) normalized box, or None if empty."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    h_img, w_img = mask.shape
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    cx = (x0 + x1) / 2 / w_img
    cy = (y0 + y1) / 2 / h_img
    w = (x1 - x0) / w_img
    h = (y1 - y0) / h_img
    return np.array([cx, cy, w, h], dtype=np.float32)


def select_most_persistent_object(deva_masks: list[np.ndarray]) -> int:
    """
    deva_masks: list of (H, W) integer label maps, one per frame, DEVA's
    automatic-mode output format (each pixel value is an object id, 0 = background).
    Returns the object id present in the most frames, weighted by average area,
    used as the training object for this clip. Larger, more persistent objects
    make for cleaner erase/reinsert training pairs than small transient ones.
    """
    presence = {}
    for mask in deva_masks:
        ids, counts = np.unique(mask, return_counts=True)
        for obj_id, count in zip(ids, counts):
            if obj_id == 0:
                continue
            presence.setdefault(int(obj_id), []).append(int(count))
    if not presence:
        raise ValueError("no non-background objects found in this clip")
    scored = {oid: len(counts) * np.mean(counts) for oid, counts in presence.items()}
    return max(scored, key=scored.get)


def extract_box_trajectory(deva_masks: list[np.ndarray], object_id: int) -> np.ndarray:
    """Returns (T, 4) normalized box trajectory for one object across all frames.
    Frames where the object is absent (e.g. briefly occluded) hold the last known
    box, matching the "temporally interpolate sparse boxes into a dense sequence"
    treatment used elsewhere in this project for reference boxes."""
    boxes = []
    last_valid = None
    for mask in deva_masks:
        box = mask_to_box(mask == object_id)
        if box is None:
            box = last_valid
        else:
            last_valid = box
        boxes.append(box)
    if boxes[0] is None:
        raise ValueError("object not present in first frame; unsuitable training clip")
    for i in range(1, len(boxes)):
        if boxes[i] is None:
            boxes[i] = boxes[i - 1]
    return np.stack(boxes)


def augment_conditioning_boxes(
    boxes: np.ndarray,           # (T, 4) real trajectory
    smoothing_window: int = 3,
    noise_std: float = 0.01,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Paper states Stage 2 training augments boxes via "smoothing bounding boxes and
    adding noise" without giving parameters (see notes/stage2_spec.md). Window and
    noise_std here are our chosen defaults, not reported values -- kept small
    relative to typical box sizes so the conditioning stays close to the real
    trajectory while preventing the model from treating it as an exact copy task.
    """
    rng = rng or np.random.default_rng()
    T = boxes.shape[0]
    smoothed = np.copy(boxes)
    half = smoothing_window // 2
    for i in range(T):
        lo, hi = max(0, i - half), min(T, i + half + 1)
        smoothed[i] = boxes[lo:hi].mean(axis=0)
    noise = rng.normal(0, noise_std, size=smoothed.shape).astype(np.float32)
    return np.clip(smoothed + noise, 0.0, 1.0)


def build_training_pair(
    video: torch.Tensor,          # (T, 3, H, W) in [-1, 1], the ground-truth clip
    deva_masks: list[np.ndarray],
    caption: str,
    condition_drop_prob: float = 0.1,
    rng: np.random.Generator | None = None,
) -> dict:
    """
    Assembles one Stage 2 training example: real object erased and put back at
    its own (augmented) trajectory, per the reconstruction-as-training pattern.
    "random condition dropping" (paper-stated, rate unspecified) is applied here
    per-signal so the model doesn't become fully dependent on every conditioning
    channel being present -- standard classifier-free-guidance-style training.
    """
    rng = rng or np.random.default_rng()
    object_id = select_most_persistent_object(deva_masks)
    real_boxes = extract_box_trajectory(deva_masks, object_id)
    cond_boxes = augment_conditioning_boxes(real_boxes, rng=rng)

    drop_text = rng.random() < condition_drop_prob
    drop_first_frame = rng.random() < condition_drop_prob

    return {
        "video": video,
        "target_boxes_real": torch.from_numpy(real_boxes),
        "inpaint_boxes": torch.from_numpy(cond_boxes),  # M_inpaint = where it was
        "synth_boxes": torch.from_numpy(cond_boxes),    # M_obj = same, during training
        "caption": "" if drop_text else caption,
        "use_first_frame": not drop_first_frame,
    }


if __name__ == "__main__":
    T, H, W = 6, 64, 64
    rng = np.random.default_rng(0)
    masks = []
    for i in range(T):
        m = np.zeros((H, W), dtype=np.int32)
        m[10 + i:30 + i, 10 + i:30 + i] = 1  # a moving square, object id 1
        masks.append(m)

    obj_id = select_most_persistent_object(masks)
    traj = extract_box_trajectory(masks, obj_id)
    aug = augment_conditioning_boxes(traj, rng=rng)
    print("object id:", obj_id)
    print("real trajectory:\n", traj)
    print("augmented trajectory:\n", aug)

    pair = build_training_pair(torch.zeros(T, 3, H, W), masks, "a red square moving", rng=rng)
    print("pair keys:", list(pair.keys()), "caption kept:", pair["caption"] != "")
