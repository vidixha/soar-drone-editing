"""Per-frame box comparison metrics. All boxes are absolute pixel [x, y, w, h]."""

import numpy as np

import geometry as geom


def iou(box_a, box_b):
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


def center_displacement(box_pred, box_gt):
    px, py = geom.box_center(box_pred)
    gx, gy = geom.box_center(box_gt)
    px_error = float(np.hypot(px - gx, py - gy))
    diag = float(np.hypot(box_gt[2], box_gt[3]))
    norm_error = px_error / diag if diag > 0 else float("nan")
    return px_error, norm_error


def scale_error(box_pred, box_gt):
    area_pred = box_pred[2] * box_pred[3]
    area_gt = box_gt[2] * box_gt[3]
    if area_gt <= 0:
        return float("nan")
    return float(area_pred / area_gt)
