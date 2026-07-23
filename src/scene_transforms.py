"""Builds a chain of frame-to-frame background transforms for one scene.

Downloading and running ORB matching on every single frame of a 978-frame 4K sequence is
not necessary to get a real drift curve: we sample key frames at a fixed stride (always
including the first and last frame of the scene) and chain the transform through them.
This is a deliberate resource tradeoff, recorded in NOTES.md and FINDINGS.md, not a hidden
shortcut. It bounds download/compute cost while still producing a genuine per-offset
drift curve, just at coarser temporal resolution than per-frame.
"""

import cv2
import numpy as np

from data_loader import download_frame_image
import geometry as geom

KEY_FRAME_STRIDE = 5


def select_key_frames(frame_numbers):
    frame_numbers = sorted(frame_numbers)
    keys = frame_numbers[::KEY_FRAME_STRIDE]
    if keys[-1] != frame_numbers[-1]:
        keys.append(frame_numbers[-1])
    return keys


def build_scene_transforms(scene_frames, get_image_path=download_frame_image):
    """Returns:
    key_frames: sorted list of frame numbers used
    T_from_first: dict frame_number -> 3x3 transform mapping first-key-frame coords to
        this frame's coords (identity at the first key frame)
    step_info: list of dicts, one per consecutive key-frame pair, with dx, dy, angle_deg,
        scale, frame_gap (for stratification: motion magnitude/type, altitude change)
    images: dict frame_number -> grayscale image array, for reuse by direct-homography
        method (avoids re-downloading/re-reading)

    get_image_path defaults to VisDrone's HF-download loader; pass
    uavdt_data_loader.download_frame_image (same signature, local files, no download) to
    run on UAVDT instead. Everything downstream of this function is dataset-agnostic.
    """
    key_frames = select_key_frames(list(scene_frames.keys()))
    images = {}
    for fn in key_frames:
        path = get_image_path(scene_frames[fn])
        images[fn] = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    T_from_first = {key_frames[0]: np.eye(3)}
    step_info = []
    cumulative = np.eye(3)
    for fn_a, fn_b in zip(key_frames, key_frames[1:]):
        M = geom.estimate_transform(images[fn_a], images[fn_b])
        if M is None:
            # Fall back to identity (no motion estimate); flagged via matched=False.
            M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
            matched = False
        else:
            matched = True
        T_step = geom.affine_to_3x3(M)
        cumulative = geom.compose(cumulative, T_step)
        T_from_first[fn_b] = cumulative.copy()
        step_info.append(dict(
            frame_a=fn_a, frame_b=fn_b, frame_gap=fn_b - fn_a,
            dx=float(M[0, 2]), dy=float(M[1, 2]),
            angle_deg=geom.transform_rotation_deg(T_step),
            scale=geom.transform_scale(T_step),
            matched=matched,
        ))
    return key_frames, T_from_first, step_info, images


def direct_transform(images, fn_ref, fn_target):
    """Direct (non-chained) transform from fn_ref to fn_target, re-matching images."""
    if fn_ref == fn_target:
        return np.eye(3)
    M = geom.estimate_transform(images[fn_ref], images[fn_target])
    if M is None:
        return None
    return geom.affine_to_3x3(M)
