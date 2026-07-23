"""Stratified reporting over results/uavdt_protocol_a_records.json. Mirrors
analyze_results.py's approach (per-scene reporting, not fake 3-bucket strata) but this
time we have 46 scenes, enough for the tercile buckets to mean something more than they
did on VisDrone's 7.
"""

import json
import os
from collections import defaultdict

import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
HORIZON_SHORT_MAX = 20


def load():
    with open(os.path.join(RESULTS_DIR, "uavdt_protocol_a_records.json")) as f:
        records = json.load(f)
    with open(os.path.join(RESULTS_DIR, "uavdt_scene_summaries.json")) as f:
        scene_summaries = json.load(f)
    return records, scene_summaries


def summarize(records, key_fn):
    groups = defaultdict(lambda: defaultdict(list))
    for r in records:
        groups[key_fn(r)][r["method"]].append(r)
    out = {}
    for stratum, by_method in groups.items():
        out[stratum] = {}
        for method, recs in by_method.items():
            iou = np.array([r["iou"] for r in recs])
            cerr_px = np.array([r["center_error_px"] for r in recs])
            scale_dev = np.array([abs(r["scale_error"] - 1.0) for r in recs])
            out[stratum][method] = dict(
                n=len(recs),
                iou_mean=float(iou.mean()), iou_median=float(np.median(iou)), iou_std=float(iou.std()),
                center_px_mean=float(cerr_px.mean()),
                scale_dev_mean=float(scale_dev.mean()),
            )
    return out


def main():
    records, scene_summaries = load()
    print(f"total records: {len(records)}")
    n_tracks = len(set((r["scene_id"], r["track_id"]) for r in records))
    print(f"unique static tracks: {n_tracks}, scenes: {len(scene_summaries)}")

    overall = summarize(records, key_fn=lambda r: "overall")
    print("\noverall (all 46 scenes pooled):")
    for method, stats in overall["overall"].items():
        print(f"  {method:20s} n={stats['n']:6d} iou_mean={stats['iou_mean']:.3f} "
              f"iou_median={stats['iou_median']:.3f} iou_std={stats['iou_std']:.3f} "
              f"center_px_mean={stats['center_px_mean']:.1f} scale_dev_mean={stats['scale_dev_mean']:.3f}")

    # Per-scene table sorted by measured translation, same honesty rule as VisDrone:
    # report per-scene, don't pretend 46 scenes collapse cleanly into 3 buckets without
    # checking first.
    trans = {sid: s["translation_px_per_frame_median"] for sid, s in scene_summaries.items()}
    by_scene = summarize(records, key_fn=lambda r: r["scene_id"])
    print("\nper-scene (sorted by measured translation px/frame):")
    print(f"{'scene':20s} {'trans':>7s} {'n_static':>9s} {'n_homog':>9s} "
          f"{'static_iou':>10s} {'chained_iou':>11s} {'direct_iou':>10s}")
    for sid in sorted(scene_summaries, key=lambda s: trans[s]):
        if sid not in by_scene or "static_box" not in by_scene[sid]:
            continue
        s = by_scene[sid]
        static_iou = s.get("static_box", {}).get("iou_mean", float("nan"))
        chained_iou = s.get("homography_chained", {}).get("iou_mean", float("nan"))
        direct_iou = s.get("homography_direct", {}).get("iou_mean", float("nan"))
        n_static = s.get("static_box", {}).get("n", 0)
        n_homog = s.get("homography_chained", {}).get("n", 0)
        print(f"{sid:20s} {trans[sid]:7.2f} {n_static:9d} {n_homog:9d} "
              f"{static_iou:10.3f} {chained_iou:11.3f} {direct_iou:10.3f}")

    # Highlight the handful of genuinely high-motion scenes identified earlier
    high_motion = ["train_M0703", "train_M0901", "val_M0205"]
    print("\nhigh-motion scenes specifically (the real stress test):")
    for sid in high_motion:
        if sid in by_scene:
            s = by_scene[sid]
            static_iou = s.get("static_box", {}).get("iou_mean", float("nan"))
            chained_iou = s.get("homography_chained", {}).get("iou_mean", float("nan"))
            direct_iou = s.get("homography_direct", {}).get("iou_mean", float("nan"))
            print(f"  {sid}: trans={trans.get(sid, float('nan')):.2f}px/f "
                  f"static={static_iou:.3f} chained={chained_iou:.3f} direct={direct_iou:.3f}")

    by_occlusion = summarize(records, key_fn=lambda r: "occluded" if r["occluded"] else "visible")
    print("\nby occlusion (note: UAVDT's visibility field is constant per NOTES.md, "
          "so this is expected to show ~no effect, unlike a real occlusion signal):")
    for stratum, by_method in by_occlusion.items():
        for method, stats in by_method.items():
            print(f"  {stratum:10s} {method:20s} n={stats['n']:7d} iou_mean={stats['iou_mean']:.3f}")

    out = dict(overall=overall["overall"], by_scene=by_scene, translation_by_scene=trans)
    with open(os.path.join(RESULTS_DIR, "uavdt_protocol_a_summary.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {RESULTS_DIR}/uavdt_protocol_a_summary.json")


if __name__ == "__main__":
    main()
