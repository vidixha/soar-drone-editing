"""AerialMetric inference entry point used by the standalone pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


def _load_model(checkpoint: Path, rank: int) -> Any:
    import torch
    from moge.model import import_model_class_by_version
    from peft import LoraConfig, get_peft_model

    if not checkpoint.exists():
        raise FileNotFoundError(
            f"AerialMetric checkpoint missing: {checkpoint}. "
            "Run scripts/setup_reconstruction_weather.sh --download-checkpoint."
        )
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    model_config = saved.get("model_config")
    if not model_config:
        raise ValueError("AerialMetric checkpoint contains no model_config")
    model = import_model_class_by_version("v2")(**model_config)
    model = get_peft_model(
        model,
        LoraConfig(
            r=rank,
            lora_alpha=2 * rank,
            bias="none",
            target_modules=["qkv", "proj", "fc1", "fc2"],
            modules_to_save=["scale_head"],
        ),
    )
    state = saved.get("model", saved)
    model_keys = set(model.state_dict())
    remapped = {}
    for key, value in state.items():
        candidates = [key, f"base_model.model.{key}", f"model.{key}"]
        for candidate in candidates:
            if candidate in model_keys:
                remapped[candidate] = value
                break
            parts = candidate.split(".")
            if parts[-1] in {"weight", "bias"}:
                for wrapper in ("base_layer", "original_module"):
                    wrapped = ".".join(parts[:-1] + [wrapper, parts[-1]])
                    if wrapped in model_keys:
                        remapped[wrapped] = value
                        break
                else:
                    continue
                break
        else:
            if key.startswith("scale_head."):
                target = (
                    "base_model.model.scale_head.modules_to_save.default."
                    + key.removeprefix("scale_head.")
                )
                if target in model_keys:
                    remapped[target] = value
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    print(
        f"[AerialMetric] loaded {len(remapped)}/{len(state)} tensors "
        f"({len(missing)} missing, {len(unexpected)} unexpected)"
    )
    if unexpected:
        raise ValueError(f"unexpected AerialMetric checkpoint keys: {unexpected[:3]}")
    return model.half().to("cuda").eval()


def _read_video(path: Path) -> list[Any]:
    import cv2

    capture = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if not frames:
        raise ValueError(f"could not decode video: {path}")
    return frames


def _infer(model: Any, frame: Any, resize: int, resolution_level: int) -> tuple[Any, Any, Any]:
    import cv2
    import numpy as np
    import torch

    height, width = frame.shape[:2]
    scale = resize / max(height, width)
    process_width = max(14, int(width * scale) // 14 * 14)
    process_height = max(14, int(height * scale) // 14 * 14)
    process = cv2.resize(frame, (process_width, process_height), interpolation=cv2.INTER_AREA)
    image = (
        torch.from_numpy(cv2.cvtColor(process, cv2.COLOR_BGR2RGB))
        .float()
        .div(255)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .half()
        .to("cuda")
    )
    with torch.inference_mode():
        output = model.infer(
            image,
            resolution_level=resolution_level,
            force_projection=True,
            apply_mask=True,
            use_fp16=True,
        )
    depth = output["depth"].float().squeeze().cpu().numpy()
    intrinsics = output["intrinsics"].float().squeeze().cpu().numpy()
    mask = output.get("mask")
    valid = (
        mask.bool().squeeze().cpu().numpy()
        if mask is not None
        else np.isfinite(depth) & (depth > 0)
    )
    if depth.shape != (height, width):
        depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_LINEAR)
        valid = cv2.resize(valid.astype("uint8"), (width, height), interpolation=cv2.INTER_NEAREST) > 0
    return depth.astype(np.float32), intrinsics.astype(np.float32), valid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--resize", type=int, default=1024)
    parser.add_argument("--resolution-level", type=int, default=9)
    parser.add_argument("--lora-rank", type=int, default=96)
    args = parser.parse_args()

    import numpy as np

    frames = _read_video(args.video)
    model = _load_model(args.checkpoint, args.lora_rank)
    depths, intrinsics, valid_masks = [], [], []
    for index, frame in enumerate(frames):
        print(f"[AerialMetric] frame {index + 1}/{len(frames)}", flush=True)
        depth, camera_intrinsics, valid = _infer(
            model, frame, args.resize, args.resolution_level
        )
        depths.append(depth)
        intrinsics.append(camera_intrinsics)
        valid_masks.append(valid)
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "aerialmetric_depths.npy", np.stack(depths).astype(np.float32))
    np.save(
        args.output / "aerialmetric_intrinsics.npy",
        np.stack(intrinsics).astype(np.float32),
    )
    np.save(args.output / "aerialmetric_valid.npy", np.stack(valid_masks).astype(bool))


if __name__ == "__main__":
    main()
