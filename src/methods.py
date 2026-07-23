"""The four (three implemented here; depth-based is deferred, see NOTES.md 1.5) methods.

All take a reference box (absolute pixel xywh at the track's reference frame) and predict
the box at a later target frame, given only geometry, no learned model.

Protocol A note: TASK.md's method 2 ("linear interpolation between endpoint
specifications") needs two known points. Protocol A gives only one (the reference-frame
box), so linear interpolation has nothing to interpolate toward and is mathematically
identical to the static-box control in this protocol. That is not a bug, it is
implemented as such below (`linear_interp_protocol_a` literally calls `static_box`), and
is documented again in FINDINGS.md so it isn't mistaken for a missing method. The real
distinguishing linear-interpolation behavior only shows up in Protocol B, where a second
endpoint exists.
"""

import geometry as geom
import numpy as np

from scene_transforms import direct_transform


def static_box(ref_box, **kwargs):
    return ref_box


def linear_interp_protocol_a(ref_box, **kwargs):
    return static_box(ref_box)


def homography_chained(ref_box, T_from_first, ref_frame, target_frame, **kwargs):
    T_ref_inv = np.linalg.inv(T_from_first[ref_frame])
    T_ref_to_target = T_from_first[target_frame] @ T_ref_inv
    cx, cy = geom.box_center(ref_box)
    pred_cx, pred_cy = geom.apply_transform(T_ref_to_target, (cx, cy))
    scale = geom.transform_scale(T_ref_to_target)
    return geom.box_from_center(pred_cx, pred_cy, ref_box[2] * scale, ref_box[3] * scale)


def homography_direct(ref_box, images, ref_frame, target_frame, **kwargs):
    T = direct_transform(images, ref_frame, target_frame)
    if T is None:
        return None
    cx, cy = geom.box_center(ref_box)
    pred_cx, pred_cy = geom.apply_transform(T, (cx, cy))
    scale = geom.transform_scale(T)
    return geom.box_from_center(pred_cx, pred_cy, ref_box[2] * scale, ref_box[3] * scale)


METHODS = {
    "static_box": static_box,
    "linear_interp": linear_interp_protocol_a,
    "homography_chained": homography_chained,
    "homography_direct": homography_direct,
}
