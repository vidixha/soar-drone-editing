"""Protocol A: static-object camera-motion compensation. Entry point.

For every qualifying static track (see static_track_selector.py) in every scene, gives
each method the ground-truth box at the track's reference (first key-frame) appearance,
asks it to predict the box at every later key frame the track appears in, and scores
against ground truth. Produces results/protocol_a_records.json (one record per
track x method x frame) which downstream stratification/reporting reads.

Run from the aerial_box_propagation/ directory: python3 src/run_protocol_a.py
"""

import json
import os

import numpy as np

import geometry as geom
import metrics
from data_loader import load_scenes, scene_size, tracks_by_index
from methods import METHODS
from scene_transforms import build_scene_transforms
from static_track_selector import select_static_tracks

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def scene_motion_summary(step_info):
    trans = [np.hypot(s["dx"], s["dy"]) / s["frame_gap"] for s in step_info]
    rot = [abs(s["angle_deg"]) / s["frame_gap"] for s in step_info]
    scales = [s["scale"] for s in step_info]
    # Cumulative deviation compounds the chain's own per-step estimation noise over
    # dozens to hundreds of steps and is not a reliable altitude-change signal on its
    # own (values up to 176% were observed on scenes with modest per-frame motion,
    # implausible as real altitude change). Kept for the record but per-step median
    # scale deviation is what stratification actually uses, since it is not sensitive
    # to chain length.
    cumulative_scale_dev = max(abs(np.prod(scales[:i + 1]) - 1.0) for i in range(len(scales))) if scales else 0.0
    scale_dev_per_step = [abs(s - 1.0) for s in scales]
    return dict(
        translation_px_per_frame_median=float(np.median(trans)) if trans else 0.0,
        rotation_deg_per_frame_median=float(np.median(rot)) if rot else 0.0,
        max_cumulative_scale_deviation=float(cumulative_scale_dev),
        scale_deviation_per_step_median=float(np.median(scale_dev_per_step)) if scale_dev_per_step else 0.0,
    )


def main():
    scenes = load_scenes()
    print(f"loaded {len(scenes)} scenes")

    all_records = []
    scene_summaries = {}

    for scene_id, scene_frames in sorted(scenes.items()):
        width, height = scene_size(scene_frames)
        print(f"[{scene_id}] {len(scene_frames)} frames, {width}x{height}")

        key_frames, T_from_first, step_info, images = build_scene_transforms(scene_frames)
        n_matched = sum(1 for s in step_info if s["matched"])
        print(f"  {len(key_frames)} key frames, {n_matched}/{len(step_info)} steps matched")

        motion_summary = scene_motion_summary(step_info)
        scene_summaries[scene_id] = dict(
            n_frames_total=len(scene_frames),
            n_key_frames=len(key_frames),
            n_steps_matched=n_matched,
            n_steps_total=len(step_info),
            **motion_summary,
        )

        static_tracks = select_static_tracks(scene_frames, key_frames, T_from_first, width, height)
        print(f"  {len(static_tracks)} qualifying static tracks")

        for track in static_tracks:
            ref_frame = track["ref_frame"]
            ref_det = tracks_by_index(scene_frames[ref_frame])[track["track_id"]]
            ref_box = geom.rel_box_to_abs(ref_det["bounding_box"], width, height)
            horizon = track["appearances"][-1] - track["appearances"][0]

            for fn in track["appearances"][1:]:
                gt_det = tracks_by_index(scene_frames[fn])[track["track_id"]]
                gt_box = geom.rel_box_to_abs(gt_det["bounding_box"], width, height)
                frame_offset = fn - ref_frame
                occluded = bool(gt_det.get("occlusion", 0) and gt_det["occlusion"] > 0)

                for method_name, method_fn in METHODS.items():
                    pred_box = method_fn(
                        ref_box, T_from_first=T_from_first, ref_frame=ref_frame,
                        target_frame=fn, images=images,
                    )
                    if pred_box is None:
                        continue
                    iou_val = metrics.iou(pred_box, gt_box)
                    center_px, center_norm = metrics.center_displacement(pred_box, gt_box)
                    scale_err = metrics.scale_error(pred_box, gt_box)

                    all_records.append(dict(
                        scene_id=scene_id, track_id=track["track_id"], label=track["label"],
                        method=method_name, ref_frame=ref_frame, frame=fn,
                        frame_offset=frame_offset, horizon=horizon, occluded=occluded,
                        iou=iou_val, center_error_px=center_px, center_error_norm=center_norm,
                        scale_error=scale_err,
                    ))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "protocol_a_records.json"), "w") as f:
        json.dump(all_records, f)
    with open(os.path.join(RESULTS_DIR, "scene_summaries.json"), "w") as f:
        json.dump(scene_summaries, f, indent=2)

    n_tracks = len(set((r["scene_id"], r["track_id"]) for r in all_records))
    print(f"\nwrote {len(all_records)} records, {n_tracks} unique static tracks, "
          f"{len(scenes)} scenes to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
