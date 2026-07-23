"""Shared geometry helpers: frame-to-frame affine estimation and box math.

Uses a similarity/partial-affine model (rotation, uniform scale, translation), which is
the correct near-planar approximation for top-down aerial video and is what the
homography-propagation baseline is built on.
"""

import cv2
import numpy as np


def estimate_transform(img1_gray, img2_gray, n_features=800, max_matches=200):
    """Estimate the 2x3 affine transform mapping points in img1 to img2.

    Returns None if not enough matches are found.
    """
    orb = cv2.ORB_create(nfeatures=n_features)
    k1, d1 = orb.detectAndCompute(img1_gray, None)
    k2, d2 = orb.detectAndCompute(img2_gray, None)
    if d1 is None or d2 is None or len(k1) < 8 or len(k2) < 8:
        return None
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(d1, d2)
    if len(matches) < 8:
        return None
    matches = sorted(matches, key=lambda m: m.distance)[:max_matches]
    pts1 = np.float32([k1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([k2[m.trainIdx].pt for m in matches])
    M, inliers = cv2.estimateAffinePartial2D(pts1, pts2, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    return M


def affine_to_3x3(M):
    T = np.eye(3)
    T[:2, :] = M
    return T


def compose(T_b_from_a, T_c_from_b):
    """Compose transforms: result maps a-space points to c-space."""
    return T_c_from_b @ T_b_from_a


def apply_transform(T, point_xy):
    p = np.array([point_xy[0], point_xy[1], 1.0])
    q = T @ p
    return q[0], q[1]


def transform_scale(T):
    """Uniform scale factor of a similarity transform."""
    a, b = T[0, 0], T[1, 0]
    return float(np.sqrt(a * a + b * b))


def transform_rotation_deg(T):
    a, b = T[0, 0], T[1, 0]
    return float(np.degrees(np.arctan2(b, a)))


def rel_box_to_abs(box_rel, width, height):
    """FiftyOne relative [x, y, w, h] (top-left, normalized) to absolute pixel [x, y, w, h]."""
    x, y, w, h = box_rel
    return x * width, y * height, w * width, h * height


def box_center(box_xywh):
    x, y, w, h = box_xywh
    return x + w / 2.0, y + h / 2.0


def box_from_center(cx, cy, w, h):
    return cx - w / 2.0, cy - h / 2.0, w, h
