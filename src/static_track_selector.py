"""Selects static tracks per TASK.md's practical proxy (see NOTES.md section 1.4).

A track is treated as static if its box motion is well explained by the *local*
frame-to-frame background transform, checked between each pair of consecutive key-frame
appearances of that track, not by chaining from a single reference frame across the whole
scene. This distinction matters: chaining the background transform across dozens or
hundreds of key-frame steps accumulates the estimator's own drift (small per-step
rotation/scale bias compounds geometrically with distance from the origin), and that
drift is indistinguishable from real object motion if measured end to end. An early
version of this selector did exactly that and it produced median residuals in the
hundreds of pixels for nearly every track, static or not, which is the chain's own error,
not the objects'. Measuring residual only between consecutive appearances bounds the
compounding to the (usually small) gap between them and isolates real motion.

Threshold and rationale documented in NOTES.md: median residual under 3px, matching the
spot-check figure, now computed the same way (local step) that the spot check used.
"""

import numpy as np

import geometry as geom
from data_loader import tracks_by_index

STATIC_RESIDUAL_THRESHOLD_PX = 3.0
MIN_KEY_FRAME_APPEARANCES = 3


def select_static_tracks(scene_frames, key_frames, T_from_first, width, height):
    """Returns list of dicts, one per qualifying static track:
    track_id, label, ref_frame, appearances (sorted key frame numbers), residual_median
    """
    track_appearances = {}  # track_id -> {frame_number: detection}
    for fn in key_frames:
        for tid, det in tracks_by_index(scene_frames[fn]).items():
            track_appearances.setdefault(tid, {})[fn] = det

    results = []
    for tid, appearances in track_appearances.items():
        frames = sorted(appearances.keys())
        if len(frames) < MIN_KEY_FRAME_APPEARANCES:
            continue

        residuals = []
        for fn_i, fn_j in zip(frames, frames[1:]):
            box_i = geom.rel_box_to_abs(appearances[fn_i]["bounding_box"], width, height)
            box_j = geom.rel_box_to_abs(appearances[fn_j]["bounding_box"], width, height)
            cx_i, cy_i = geom.box_center(box_i)
            cx_j, cy_j = geom.box_center(box_j)
            T_i_to_j = T_from_first[fn_j] @ np.linalg.inv(T_from_first[fn_i])
            pred_cx, pred_cy = geom.apply_transform(T_i_to_j, (cx_i, cy_i))
            residuals.append(float(np.hypot(pred_cx - cx_j, pred_cy - cy_j)))

        median_residual = float(np.median(residuals))
        if median_residual < STATIC_RESIDUAL_THRESHOLD_PX:
            ref_frame = frames[0]
            results.append(dict(
                track_id=tid,
                label=appearances[ref_frame]["label"],
                ref_frame=ref_frame,
                appearances=frames,
                residual_median=median_residual,
            ))
    return results
