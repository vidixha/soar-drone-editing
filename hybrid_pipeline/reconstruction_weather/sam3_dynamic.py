"""Run text-prompted SAM3 video segmentation and save per-instance tracks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def _load_video(path: Path) -> list[Image.Image]:
    capture = cv2.VideoCapture(str(path))
    frames: list[Image.Image] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    capture.release()
    if not frames:
        raise ValueError(f"video contains no readable frames: {path}")
    return frames


def _mask_to_xywh(mask: np.ndarray, width: int, height: int) -> list[float] | None:
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    x0, x1 = float(xs.min()), float(xs.max()) + 1.0
    y0, y1 = float(ys.min()), float(ys.max()) + 1.0
    return [x0 / width, y0 / height, (x1 - x0) / width, (y1 - y0) / height]


def _iter_detections(outputs: dict) -> list[tuple[int, np.ndarray, float]]:
    if outputs is None:
        return []
    ids = np.asarray(outputs.get("out_obj_ids", []))
    masks = outputs.get("out_binary_masks")
    if masks is None or len(ids) == 0:
        return []
    masks = np.asarray(masks)
    if masks.ndim == 2:
        masks = masks[None]
    probs = np.asarray(outputs.get("out_probs", np.ones(len(ids))))
    detections = []
    for obj_id, mask, prob in zip(ids.tolist(), masks, probs.tolist(), strict=False):
        detections.append((int(obj_id), np.asarray(mask).astype(bool), float(prob)))
    return detections


def _ingest(
    result: dict,
    *,
    keyword: str,
    tracks: dict[tuple[str, int], np.ndarray],
    scores: dict[tuple[str, int], float],
    dynamic: np.ndarray,
    height: int,
    width: int,
    frame_count: int,
    obj_id_override: int | None = None,
) -> None:
    frame_index = int(result["frame_index"])
    for obj_id, mask, prob in _iter_detections(result.get("outputs") or {}):
        if mask.shape != (height, width):
            mask = cv2.resize(
                mask.astype(np.uint8),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        key = (keyword, obj_id if obj_id_override is None else obj_id_override)
        if key not in tracks:
            tracks[key] = np.zeros((frame_count, height, width), dtype=bool)
            scores[key] = prob
        tracks[key][frame_index] |= mask
        scores[key] = max(scores[key], prob)
        dynamic[frame_index] |= mask


def _propagate(predictor, session_id: str, max_frames: int):
    yield from predictor.handle_stream_request(
        {
            "type": "propagate_in_video",
            "session_id": session_id,
            "propagation_direction": "forward",
            "start_frame_index": 0,
            "max_frame_num_to_track": max_frames,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--instances-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--keywords", nargs="+", required=True)
    parser.add_argument("--max-objects", type=int, default=24)
    parser.add_argument(
        "--max-per-keyword",
        type=int,
        default=8,
        help="Keep this many detections per class before propagating (avoids OOM on crowded frames).",
    )
    parser.add_argument("--max-frames", type=int, default=0, help="0 = all frames")
    args = parser.parse_args()

    from sam3.model_builder import build_sam3_video_predictor
    import torch

    predictor = build_sam3_video_predictor(
        checkpoint_path=str(args.checkpoint) if args.checkpoint else None,
    )
    frames = _load_video(args.video)
    width, height = frames[0].size
    frame_count = len(frames)
    dynamic = np.zeros((frame_count, height, width), dtype=bool)
    tracks: dict[tuple[str, int], np.ndarray] = {}
    scores: dict[tuple[str, int], float] = {}
    max_objects = int(getattr(args, "max_objects", 24) or 24)
    max_per_keyword = int(getattr(args, "max_per_keyword", 8) or 8)
    max_frames = int(getattr(args, "max_frames", 0) or 0) or frame_count
    ingest_kw = dict(
        tracks=tracks,
        scores=scores,
        dynamic=dynamic,
        height=height,
        width=width,
        frame_count=frame_count,
    )

    for keyword in (item.strip() for item in args.keywords):
        if not keyword:
            continue
        torch.cuda.empty_cache()
        response = predictor.handle_request({"type": "start_session", "resource_path": frames})
        session_id = response["session_id"]
        crowded = False
        boxes: list[list[float]] = []
        try:
            prompted = predictor.handle_request(
                {
                    "type": "add_prompt",
                    "session_id": session_id,
                    "frame_index": 0,
                    "text": keyword,
                }
            )
            detections = _iter_detections(prompted.get("outputs") or {})
            crowded = len(detections) > max_per_keyword
            if crowded:
                ranked = sorted(
                    detections,
                    key=lambda item: float(item[1].sum()) * max(item[2], 1e-3),
                    reverse=True,
                )
                for _, mask, _ in ranked[: max(1, max_per_keyword)]:
                    box = _mask_to_xywh(mask, width, height)
                    if box is not None:
                        boxes.append(box)
                print(
                    f"[sam3] {keyword}: {len(detections)} detections, "
                    f"track {len(boxes)} objects one-by-one over {max_frames} frames",
                    flush=True,
                )
            elif detections:
                print(
                    f"[sam3] {keyword}: {len(detections)} detections, track {max_frames} frames",
                    flush=True,
                )
            if not crowded:
                for result in _propagate(predictor, session_id, max_frames):
                    _ingest(result, keyword=keyword, **ingest_kw)
        finally:
            predictor.handle_request({"type": "close_session", "session_id": session_id})
            torch.cuda.empty_cache()

        if not crowded:
            continue
        for index, box in enumerate(boxes):
            torch.cuda.empty_cache()
            response = predictor.handle_request(
                {"type": "start_session", "resource_path": frames}
            )
            session_id = response["session_id"]
            try:
                predictor.handle_request(
                    {
                        "type": "add_prompt",
                        "session_id": session_id,
                        "frame_index": 0,
                        "bounding_boxes": [box],
                        "bounding_box_labels": [1],
                    }
                )
                for result in _propagate(predictor, session_id, max_frames):
                    _ingest(result, keyword=keyword, obj_id_override=index, **ingest_kw)
            finally:
                predictor.handle_request({"type": "close_session", "session_id": session_id})
                torch.cuda.empty_cache()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, dynamic)
    instances_dir = args.instances_dir or args.output.with_name("instances")
    instances_dir.mkdir(parents=True, exist_ok=True)
    objects = []
    ranked = sorted(
        tracks.items(),
        key=lambda item: float(item[1].mean()),
        reverse=True,
    )
    kept = ranked[: max(1, max_objects)]
    dynamic[:] = False
    for (keyword, obj_id), masks in kept:
        if not masks.any():
            continue
        dynamic |= masks
        track_id = f"{keyword}_{obj_id}"
        filename = f"{track_id}.npz"
        np.savez_compressed(instances_dir / filename, masks=masks.astype(np.uint8))
        objects.append(
            {
                "id": track_id,
                "category": keyword,
                "sam3_obj_id": obj_id,
                "file": filename,
                "frames": int(masks.any(axis=(1, 2)).sum()),
                "coverage": float(masks.mean()),
                "score": float(scores.get((keyword, obj_id), 0.0)),
            }
        )
    (instances_dir / "index.json").write_text(
        json.dumps({"objects": objects, "frame_count": frame_count}, indent=2)
    )
    print(
        f"SAM3 dynamic coverage: {100 * float(dynamic.mean()):.3f}%; "
        f"{len(objects)} instance tracks",
        flush=True,
    )


if __name__ == "__main__":
    main()
