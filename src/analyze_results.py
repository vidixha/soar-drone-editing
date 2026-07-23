"""Stratified reporting over results/protocol_a_records.json.

Per TASK.md Section 5/6: no single aggregate number, bucket by motion magnitude, motion
type, altitude change, occlusion, and horizon, report variance not just mean, and check
the static-box control is clearly beaten by real methods before trusting anything else.
"""

import json
import os
from collections import defaultdict

import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
HORIZON_SHORT_MAX = 20


def load():
    with open(os.path.join(RESULTS_DIR, "protocol_a_records.json")) as f:
        records = json.load(f)
    with open(os.path.join(RESULTS_DIR, "scene_summaries.json")) as f:
        scene_summaries = json.load(f)
    return records, scene_summaries


def classify_scenes(scene_summaries):
    """Per-scene strata: motion magnitude tercile, motion type, altitude change."""
    trans = {sid: s["translation_px_per_frame_median"] for sid, s in scene_summaries.items()}
    order = sorted(trans, key=lambda k: trans[k])
    n = len(order)
    magnitude = {}
    for i, sid in enumerate(order):
        if i < n / 3:
            magnitude[sid] = "low"
        elif i < 2 * n / 3:
            magnitude[sid] = "medium"
        else:
            magnitude[sid] = "high"

    motion_type = {}
    for sid, s in scene_summaries.items():
        # rotation-induced pixel displacement at a representative radius (half the frame
        # diagonal is unavailable here without width/height; approximate with 1000px,
        # documented as an approximation for relative type classification only)
        rot_px_equiv = np.radians(s["rotation_deg_per_frame_median"]) * 1000.0
        motion_type[sid] = "rotation_dominant" if rot_px_equiv > s["translation_px_per_frame_median"] else "translation_dominant"

    # Altitude change: per-step scale deviation, terciled across scenes, not a fixed
    # absolute threshold on the cumulative deviation. The cumulative figure compounds
    # chain length (up to 197 steps) with per-step noise and produced deviations up to
    # 176% that are not credible as real altitude change (see run_protocol_a.py note).
    # Per-step median is chain-length-independent and is what is actually compared here.
    scale_dev = {sid: s["scale_deviation_per_step_median"] for sid, s in scene_summaries.items()}
    order_scale = sorted(scale_dev, key=lambda k: scale_dev[k])
    altitude_change = {}
    for i, sid in enumerate(order_scale):
        altitude_change[sid] = "high" if i >= 2 * n / 3 else ("medium" if i >= n / 3 else "low")

    return dict(magnitude=magnitude, motion_type=motion_type, altitude_change=altitude_change)


def summarize(records, key_fn):
    """Group records by key_fn(record), then by method, report mean/median/std/n for
    iou, center_error_px, center_error_norm, and mean |scale_error - 1|."""
    groups = defaultdict(lambda: defaultdict(list))
    for r in records:
        groups[key_fn(r)][r["method"]].append(r)

    out = {}
    for stratum, by_method in groups.items():
        out[stratum] = {}
        for method, recs in by_method.items():
            iou = np.array([r["iou"] for r in recs])
            cerr_px = np.array([r["center_error_px"] for r in recs])
            cerr_norm = np.array([r["center_error_norm"] for r in recs])
            scale_dev = np.array([abs(r["scale_error"] - 1.0) for r in recs])
            out[stratum][method] = dict(
                n=len(recs),
                iou_mean=float(iou.mean()), iou_median=float(np.median(iou)), iou_std=float(iou.std()),
                center_px_mean=float(cerr_px.mean()), center_px_median=float(np.median(cerr_px)),
                center_norm_mean=float(cerr_norm.mean()),
                scale_dev_mean=float(scale_dev.mean()),
            )
    return out


def drift_curve(records, offset_bin_size=10):
    groups = defaultdict(lambda: defaultdict(list))
    for r in records:
        bin_idx = r["frame_offset"] // offset_bin_size
        groups[bin_idx][r["method"]].append(r)

    curve = defaultdict(dict)
    for bin_idx in sorted(groups):
        for method, recs in groups[bin_idx].items():
            iou = np.array([r["iou"] for r in recs])
            cerr_norm = np.array([r["center_error_norm"] for r in recs])
            curve[bin_idx * offset_bin_size][method] = dict(
                n=len(recs), iou_mean=float(iou.mean()), center_norm_mean=float(cerr_norm.mean()),
            )
    return dict(curve)


def main():
    records, scene_summaries = load()
    strata = classify_scenes(scene_summaries)

    print(f"total records: {len(records)}")
    n_tracks = len(set((r["scene_id"], r["track_id"]) for r in records))
    n_scenes = len(scene_summaries)
    print(f"unique static tracks: {n_tracks}, scenes: {n_scenes}")
    print()
    print("scene strata:")
    for sid in scene_summaries:
        print(f"  {sid}: motion={strata['magnitude'][sid]}, type={strata['motion_type'][sid]}, "
              f"altitude_change={strata['altitude_change'][sid]}")

    overall = summarize(records, key_fn=lambda r: "overall")
    print("\noverall (all strata pooled):")
    for method, stats in overall["overall"].items():
        print(f"  {method}: n={stats['n']} iou_mean={stats['iou_mean']:.3f} "
              f"iou_median={stats['iou_median']:.3f} iou_std={stats['iou_std']:.3f} "
              f"center_px_mean={stats['center_px_mean']:.1f} scale_dev_mean={stats['scale_dev_mean']:.3f}")

    by_magnitude = summarize(records, key_fn=lambda r: strata["magnitude"][r["scene_id"]])
    by_type = summarize(records, key_fn=lambda r: strata["motion_type"][r["scene_id"]])
    by_altitude = summarize(records, key_fn=lambda r: strata["altitude_change"][r["scene_id"]])
    by_occlusion = summarize(records, key_fn=lambda r: "occluded" if r["occluded"] else "visible")
    by_horizon = summarize(records, key_fn=lambda r: "short" if r["horizon"] < HORIZON_SHORT_MAX else "long")
    curve = drift_curve(records)

    print("\nby motion magnitude:")
    for stratum, by_method in by_magnitude.items():
        for method, stats in by_method.items():
            print(f"  {stratum:8s} {method:20s} n={stats['n']:5d} iou_mean={stats['iou_mean']:.3f}")

    print("\nby motion type:")
    for stratum, by_method in by_type.items():
        for method, stats in by_method.items():
            print(f"  {stratum:20s} {method:20s} n={stats['n']:5d} iou_mean={stats['iou_mean']:.3f}")

    print("\nby altitude change (per-step scale deviation tercile across scenes):")
    for stratum, by_method in by_altitude.items():
        for method, stats in by_method.items():
            print(f"  {str(stratum):8s} {method:20s} n={stats['n']:5d} iou_mean={stats['iou_mean']:.3f} "
                  f"scale_dev_mean={stats['scale_dev_mean']:.3f}")

    print("\nby occlusion:")
    for stratum, by_method in by_occlusion.items():
        for method, stats in by_method.items():
            print(f"  {stratum:10s} {method:20s} n={stats['n']:5d} iou_mean={stats['iou_mean']:.3f}")

    print("\nby horizon:")
    for stratum, by_method in by_horizon.items():
        for method, stats in by_method.items():
            print(f"  {stratum:6s} {method:20s} n={stats['n']:5d} iou_mean={stats['iou_mean']:.3f}")

    print("\ndrift curve (iou_mean by frame_offset bin):")
    for offset, by_method in curve.items():
        line = f"  offset={offset:4d} "
        for method, stats in by_method.items():
            line += f"{method}={stats['iou_mean']:.3f}(n={stats['n']}) "
        print(line)

    out = dict(
        overall=overall["overall"], by_magnitude=by_magnitude, by_motion_type=by_type,
        by_altitude_change=by_altitude, by_occlusion=by_occlusion, by_horizon=by_horizon,
        drift_curve=curve, scene_strata=strata,
    )
    with open(os.path.join(RESULTS_DIR, "protocol_a_summary.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {RESULTS_DIR}/protocol_a_summary.json")


if __name__ == "__main__":
    main()
