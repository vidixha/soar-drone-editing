"""Standalone 4D reconstruction and weather pipeline."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from .config import ReconstructionConfig, SnowConfig
from .media import (
    center_frames_and_resize,
    load_video,
    save_depth_preview,
    save_masks,
    save_video,
)
from .pi3x import reconstruct_pi3x
from .weather import simulate_fog, simulate_rain, simulate_sandstorm, simulate_snow


def trajectory_metrics(cam_c2w: np.ndarray, depths: np.ndarray) -> dict:
    """Measure whether cameras actually move in metric 4D, vs a 2.5D lift."""
    translations = cam_c2w[:, :3, 3]
    span = float(np.linalg.norm(translations.max(axis=0) - translations.min(axis=0)))
    path = float(np.sum(np.linalg.norm(np.diff(translations, axis=0), axis=1))) if len(translations) > 1 else 0.0
    valid = np.isfinite(depths) & (depths > 0) & (depths < 999)
    median_depth = float(np.median(depths[valid])) if valid.any() else 0.0
    rotation = cam_c2w[0, :3, :3].T @ cam_c2w[-1, :3, :3]
    angle = float(
        np.degrees(np.arccos(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0)))
    )
    baseline = span / max(median_depth, 1e-3)
    return {
        "translation_span_metres": span,
        "path_metres": path,
        "median_depth_metres": median_depth,
        "baseline_over_depth": baseline,
        "end_rotation_degrees": angle,
        "is_multiview": bool(baseline >= 0.05 or angle >= 3.0),
    }


def _run(command: list[str | Path], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    printable = " ".join(str(part) for part in command)
    print(f"[reconstruction-weather] {printable}", flush=True)
    subprocess.run([str(part) for part in command], cwd=cwd, env=env, check=True)


def _segment_dynamic_sam3(
    video_path: Path,
    output_dir: Path,
    *,
    config: ReconstructionConfig,
    sam3_dir: Path,
    sam3_python: Path,
    sam3_checkpoint: Path | None,
) -> np.ndarray:
    if not sam3_dir.is_dir():
        raise FileNotFoundError(f"SAM3 checkout does not exist: {sam3_dir}")
    if not sam3_python.is_file():
        raise FileNotFoundError(f"SAM3 Python does not exist: {sam3_python}")
    if sam3_checkpoint is not None and not sam3_checkpoint.is_file():
        raise FileNotFoundError(f"SAM3 checkpoint does not exist: {sam3_checkpoint}")
    output = output_dir / "sam3_dynamic_mask.npy"
    instances_dir = output_dir / "instances"
    command: list[str | Path] = [
        sam3_python,
        Path(__file__).with_name("sam3_dynamic.py"),
        "--video",
        video_path,
        "--output",
        output,
        "--instances-dir",
        instances_dir,
        "--keywords",
        *config.dynamic_keywords,
        "--max-objects",
        "24",
        "--max-per-keyword",
        "8",
    ]
    if sam3_checkpoint is not None:
        command.extend(("--checkpoint", sam3_checkpoint))
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(sam3_dir), str(Path(__file__).resolve().parents[1]), env.get("PYTHONPATH", "")]
    )
    _run(command, cwd=sam3_dir, env=env)
    dynamic = np.load(output).astype(bool)
    expected = (config.num_frames, config.height, config.width)
    if dynamic.shape != expected:
        raise ValueError(f"SAM3 mask shape {dynamic.shape} does not match {expected}")
    if config.dynamic_mask_dilation:
        size = 2 * config.dynamic_mask_dilation + 1
        kernel = np.ones((size, size), dtype=np.uint8)
        dynamic = np.stack(
            [cv2.dilate(frame.astype(np.uint8), kernel).astype(bool) for frame in dynamic]
        )
    return dynamic


def reconstruct(
    input_video: Path,
    output_dir: Path,
    *,
    config: ReconstructionConfig,
    aerialmetric_dir: Path,
    aerialmetric_python: Path,
    aerialmetric_checkpoint: Path,
    megasam_dir: Path,
    megasam_python: Path,
    megasam_checkpoint: Path,
    sam3_dir: Path,
    sam3_python: Path,
    sam3_checkpoint: Path | None,
) -> Path:
    """Run the selected reconstruction backend with AerialMetric depth."""
    config.validate()
    input_video = input_video.resolve()
    output_dir = output_dir.resolve()
    aerialmetric_dir = aerialmetric_dir.resolve()
    sam3_dir = sam3_dir.resolve()
    sam3_checkpoint = sam3_checkpoint.resolve() if sam3_checkpoint else None
    if not input_video.is_file():
        raise FileNotFoundError(f"input video does not exist: {input_video}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"reconstruction output is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    source_video, fps = load_video(input_video)
    video = center_frames_and_resize(
        source_video,
        num_frames=config.num_frames,
        width=config.width,
        height=config.height,
        sampling=config.frame_sampling,
    )
    save_video(output_dir / "video.mp4", video, fps)
    if config.backend == "pi3x":
        depths, cam_c2w, intrinsics, static, sky = _reconstruct_pi3x_aerial(
            video,
            output_dir,
            config=config,
            aerialmetric_dir=aerialmetric_dir,
            aerialmetric_python=aerialmetric_python,
            aerialmetric_checkpoint=aerialmetric_checkpoint,
        )
        pipeline_name = "direct Pi3X 4D reconstruction + MoGe2-Aerial metric depth"
    else:
        depths, cam_c2w, intrinsics, static, sky = _reconstruct_megasam_aerial(
            video,
            output_dir,
            config=config,
            aerialmetric_dir=aerialmetric_dir,
            aerialmetric_python=aerialmetric_python,
            aerialmetric_checkpoint=aerialmetric_checkpoint,
            megasam_dir=megasam_dir.resolve(),
            megasam_python=megasam_python,
            megasam_checkpoint=megasam_checkpoint.resolve(),
        )
        pipeline_name = "MegaSaM structure and motion + MoGe2-Aerial metric depth"
    # Pi3X confidence is reconstruction certainty, not motion. Using it as
    # "dynamic" punches holes in distant buildings. MegaSaM static_confidence
    # is the actual motion/outlier signal.
    backend_dynamic = (
        np.zeros_like(static)
        if config.backend == "pi3x"
        else (~static & ~sky)
    )
    if config.dynamic_segmentation == "backend":
        dynamic = backend_dynamic
    else:
        sam3_dynamic = _segment_dynamic_sam3(
            output_dir / "video.mp4",
            output_dir,
            config=config,
            sam3_dir=sam3_dir,
            sam3_python=sam3_python,
            sam3_checkpoint=sam3_checkpoint,
        )
        dynamic = (
            sam3_dynamic | backend_dynamic
            if config.dynamic_segmentation == "hybrid"
            else sam3_dynamic
        )
    static = ~sky & ~dynamic
    np.save(output_dir / "depths.npy", depths)
    np.savez(output_dir / "cameras.npz", cam_c2w=cam_c2w, intrinsics=intrinsics)
    np.save(output_dir / "static_mask.npy", static)
    np.save(output_dir / "dynamic_mask.npy", dynamic)
    np.save(output_dir / "sky_mask.npy", sky)
    save_masks(output_dir / "static_mask", static)
    save_masks(output_dir / "dynamic_mask", dynamic)
    save_masks(output_dir / "sky_mask", sky)
    pose_stats = trajectory_metrics(cam_c2w, depths)
    print(
        f"[reconstruction-weather] camera path {pose_stats['path_metres']:.2f} m, "
        f"span {pose_stats['translation_span_metres']:.2f} m, "
        f"median depth {pose_stats['median_depth_metres']:.2f} m, "
        f"baseline/depth {pose_stats['baseline_over_depth']:.3f}, "
        f"rotation {pose_stats['end_rotation_degrees']:.1f} deg",
        flush=True,
    )
    if not pose_stats["is_multiview"]:
        print(
            "[reconstruction-weather] WARNING: camera baseline is tiny versus depth. "
            "Re-run with Pi3X and --frame-sampling span on a longer clip.",
            flush=True,
        )
    if config.save_visualization:
        save_depth_preview(output_dir / "reconstruction_preview.mp4", video, depths, fps)
    metadata = {
        "pipeline": pipeline_name,
        "input_video": str(input_video),
        "fps": fps,
        "config": config.to_dict(),
        "trajectory": pose_stats,
        "outputs": {
            "video": "video.mp4",
            "depths": "depths.npy",
            "cameras": "cameras.npz",
            "static_mask": "static_mask/*.png",
            "dynamic_mask": "dynamic_mask/*.png",
            "sky_mask": "sky_mask/*.png",
        },
    }
    (output_dir / "reconstruction.json").write_text(json.dumps(metadata, indent=2))
    return output_dir


def _reconstruct_pi3x_aerial(
    video: np.ndarray,
    output_dir: Path,
    *,
    config: ReconstructionConfig,
    aerialmetric_dir: Path,
    aerialmetric_python: Path,
    aerialmetric_checkpoint: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pose_depths, cam_c2w, pi3x_intrinsics, confidence = reconstruct_pi3x(
        video,
        model_id=config.pi3x_model_id,
        pixel_limit=config.pi3_pixel_limit,
    )
    np.save(output_dir / "pi3x_depths.npy", pose_depths)
    np.save(output_dir / "pi3x_confidence.npy", confidence)
    np.savez(
        output_dir / "pi3x_cameras.npz",
        cam_c2w=cam_c2w,
        intrinsics=pi3x_intrinsics,
    )
    env = os.environ.copy()
    root = Path(__file__).resolve().parents[1]
    env["PYTHONPATH"] = os.pathsep.join(
        [str(root), str(aerialmetric_dir / "MoGe"), env.get("PYTHONPATH", "")]
    )
    _run(
        [
            aerialmetric_python,
            "-m",
            "reconstruction_weather.aerialmetric",
            "--video",
            output_dir / "video.mp4",
            "--output",
            output_dir,
            "--checkpoint",
            aerialmetric_checkpoint,
            "--resize",
            str(config.aerialmetric_resize),
            "--resolution-level",
            str(config.aerialmetric_resolution_level),
            "--lora-rank",
            str(config.aerialmetric_lora_rank),
        ],
        cwd=aerialmetric_dir,
        env=env,
    )
    return _align_metric_depth(
        output_dir, pose_depths, cam_c2w, confidence, video=video
    )


def _reconstruct_megasam_aerial(
    video: np.ndarray,
    output_dir: Path,
    *,
    config: ReconstructionConfig,
    aerialmetric_dir: Path,
    aerialmetric_python: Path,
    aerialmetric_checkpoint: Path,
    megasam_dir: Path,
    megasam_python: Path,
    megasam_checkpoint: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    runner = megasam_dir / "aerialmetric" / "run_aerialmetric.py"
    tracker = megasam_dir / "camera_tracking_scripts" / "test_demo.py"
    for required in (aerialmetric_python, aerialmetric_checkpoint, megasam_python, megasam_checkpoint, runner, tracker):
        if not required.exists():
            raise FileNotFoundError(f"MegaSaM backend requirement missing: {required}")

    scene_name = "reconstruction_weather"
    frame_dir = output_dir / "megasam_frames"
    frame_dir.mkdir(parents=True)
    for index, frame in enumerate(video):
        cv2.imwrite(
            str(frame_dir / f"{index:05d}.png"),
            cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
        )
    aerial_output = output_dir / "megasam_aerialmetric"
    aerial_env = os.environ.copy()
    aerial_env["PYTHONPATH"] = os.pathsep.join(
        [
            str(megasam_dir),
            str(aerialmetric_dir / "MoGe"),
            aerial_env.get("PYTHONPATH", ""),
        ]
    )
    _run(
        [
            aerialmetric_python,
            runner,
            "--img-path",
            frame_dir,
            "--scene-name",
            scene_name,
            "--outdir",
            aerial_output,
            "--checkpoint",
            aerialmetric_checkpoint,
            "--resize",
            str(config.aerialmetric_resize),
            "--resolution-level",
            str(config.aerialmetric_resolution_level),
            "--lora-rank",
            str(config.aerialmetric_lora_rank),
        ],
        cwd=megasam_dir,
        env=aerial_env,
    )
    megasam_output = output_dir / "megasam_reconstruction.npz"
    megasam_env = os.environ.copy()
    megasam_env["PYTHONPATH"] = os.pathsep.join(
        [
            str(megasam_dir / "base"),
            str(megasam_dir / "base" / "thirdparty" / "lietorch"),
            megasam_env.get("PYTHONPATH", ""),
        ]
    )
    _run(
        [
            megasam_python,
            tracker,
            "--datapath",
            frame_dir,
            "--weights",
            megasam_checkpoint,
            "--scene_name",
            scene_name,
            "--mono_depth_path",
            aerial_output / "disparity",
            "--metric_depth_path",
            aerial_output / "metric",
            "--output_path",
            megasam_output,
            "--disable_vis",
        ],
        cwd=megasam_dir,
        env=megasam_env,
    )
    result = np.load(megasam_output)
    depths = result["depths"].astype(np.float32)
    cam_c2w = result["cam_c2w"].astype(np.float32)
    matrix = result["intrinsic"].astype(np.float32)
    output_height, output_width = video.shape[1:3]
    source_height, source_width = depths.shape[1:]
    if (source_height, source_width) != (output_height, output_width):
        depths = np.stack(
            [
                cv2.resize(
                    frame,
                    (output_width, output_height),
                    interpolation=cv2.INTER_LINEAR,
                )
                for frame in depths
            ]
        )
        matrix[0, :] *= output_width / source_width
        matrix[1, :] *= output_height / source_height
    intrinsics = np.repeat(
        np.array(
            [[matrix[0, 0], matrix[1, 1], matrix[0, 2], matrix[1, 2]]],
            dtype=np.float32,
        ),
        len(depths),
        axis=0,
    )
    metric_files = sorted(
        (aerial_output / "metric" / scene_name).glob("*.npz")
    )
    valid = np.stack([np.load(path)["valid"].astype(bool) for path in metric_files])
    valid = valid[: len(depths)]
    finite = np.isfinite(depths) & (depths > 0)
    if "static_confidence" not in result.files:
        raise KeyError(
            "MegaSaM output lacks static_confidence; update its reconstruction exporter"
        )
    static_confidence = np.asarray(result["static_confidence"]).squeeze()
    if static_confidence.ndim != 3 or len(static_confidence) < len(depths):
        raise ValueError(
            "MegaSaM static_confidence must have shape (frames, height, width)"
        )
    static_confidence = np.stack(
        [
            cv2.GaussianBlur(
                cv2.resize(
                    frame,
                    (output_width, output_height),
                    interpolation=cv2.INTER_LINEAR,
                ),
                (0, 0),
                0.8,
            )
            for frame in static_confidence[: len(depths)]
        ]
    ).astype(np.float32)
    np.save(output_dir / "megasam_static_confidence.npy", static_confidence)
    static = (
        valid
        & finite
        & (static_confidence >= config.megasam_static_threshold)
    )
    sky = ~valid
    safe_depths = np.nan_to_num(depths, nan=1000, posinf=1000, neginf=1000)
    safe_depths[~static] = 1000
    return safe_depths, cam_c2w[: len(depths)], intrinsics, static, sky


def _align_metric_depth(
    output_dir: Path,
    pose_depths: np.ndarray,
    cam_c2w: np.ndarray,
    confidence: np.ndarray,
    video: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    depths = np.load(output_dir / "aerialmetric_depths.npy").astype(np.float32)
    normalized_intrinsics = np.load(output_dir / "aerialmetric_intrinsics.npy").astype(np.float32)
    valid = np.load(output_dir / "aerialmetric_valid.npy").astype(bool)
    if depths.shape != pose_depths.shape:
        raise ValueError(
            f"AerialMetric depth shape {depths.shape} does not match Pi3X {pose_depths.shape}"
        )
    overlap = (
        valid
        & confidence
        & np.isfinite(depths)
        & (depths > 0)
        & np.isfinite(pose_depths)
        & (pose_depths > 0)
    )
    if not overlap.any():
        raise ValueError("AerialMetric and Pi3X have no valid depth overlap")
    pose_scale = float(np.median(depths[overlap] / pose_depths[overlap]))
    cam_c2w[:, :3, 3] *= pose_scale
    height, width = depths.shape[1:]
    intrinsics = np.stack(
        [
            normalized_intrinsics[:, 0, 0] * width,
            normalized_intrinsics[:, 1, 1] * height,
            normalized_intrinsics[:, 0, 2] * width,
            normalized_intrinsics[:, 1, 2] * height,
        ],
        axis=-1,
    ).astype(np.float32)
    metric_pose = pose_depths.astype(np.float32) * pose_scale
    finite_metric = np.isfinite(depths) & (depths > 0) & (depths < 900)
    finite_pose = np.isfinite(metric_pose) & (metric_pose > 0) & (metric_pose < 900)
    sky = ~valid
    safe_depths = np.where(valid & finite_metric, depths, np.nan).astype(np.float32)
    fill = valid & ~np.isfinite(safe_depths) & finite_pose
    safe_depths[fill] = metric_pose[fill]
    safe_depths = np.nan_to_num(safe_depths, nan=1000, posinf=1000, neginf=1000)
    safe_depths[sky] = 1000
    static = valid
    finite_report = safe_depths[~sky]
    print(
        f"[reconstruction-weather] metric depth {float(finite_report.min()):.2f}.."
        f"{float(finite_report.max()):.2f} m; filled {int(fill.sum())} AerialMetric "
        f"holes from Pi3X; sky {sky.mean():.1%}; Pi3X translation scale {pose_scale:.4f}",
        flush=True,
    )
    return safe_depths.astype(np.float32), cam_c2w, intrinsics, static, sky


def simulate_weather(
    reconstruction_dir: Path,
    output_dir: Path,
    *,
    config: SnowConfig,
) -> Path:
    """Apply weather directly to reconstructed source frames; no diffusion generation."""
    config.validate()
    reconstruction_dir = reconstruction_dir.resolve()
    output_dir = output_dir.resolve()
    if not (reconstruction_dir / "reconstruction.json").is_file():
        raise FileNotFoundError(
            f"not a standalone reconstruction artifact: {reconstruction_dir}"
        )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"weather output is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    video, fps = load_video(reconstruction_dir / "video.mp4")
    depths = np.load(reconstruction_dir / "depths.npy").astype(np.float32)
    cameras = np.load(reconstruction_dir / "cameras.npz")
    cam_c2w, intrinsics = cameras["cam_c2w"], cameras["intrinsics"]
    static_masks = np.load(reconstruction_dir / "static_mask.npy").astype(bool)
    accumulation_mask_path = reconstruction_dir / "accumulation_mask.npy"
    accumulation_masks = (
        np.load(accumulation_mask_path).astype(bool)
        if accumulation_mask_path.is_file()
        else static_masks
    )
    am_valid_path = reconstruction_dir / "aerialmetric_valid.npy"
    sky_mask_path = reconstruction_dir / "sky_mask.npy"
    if am_valid_path.is_file():
        sky_masks = ~np.load(am_valid_path).astype(bool)
    elif sky_mask_path.is_file():
        sky_masks = np.load(sky_mask_path).astype(bool)
    else:
        sky_masks = depths >= 999
    simulate = {
        "rain": simulate_rain,
        "fog": simulate_fog,
        "sandstorm": simulate_sandstorm,
    }.get(config.effect, simulate_snow)
    result = simulate(
        video,
        depths,
        cam_c2w,
        intrinsics,
        static_masks,
        accumulation_masks,
        sky_masks,
        fps=fps,
        config=config,
    )
    save_video(output_dir / "weather_video.mp4", result.video, fps, quality=9)
    np.savez_compressed(
        output_dir / "weather_particles.npz",
        positions=result.final_positions.astype(np.float32),
        landed=result.landed,
    )
    metadata = {
        "effect": config.effect,
        "method": {
            "rain": "3D rain streaks back-projected onto source frames",
            "fog": "metric-depth haze plus sparse 3D motes",
            "sandstorm": "3D sand flakes on depth mesh (snow engine, sand colour/wind)",
        }.get(config.effect, "3D snow on depth mesh, back-projected onto source frames"),
        "source_reconstruction": str(reconstruction_dir),
        "config": asdict(config),
        "metrics": {
            "emitted": result.emitted,
            "collisions": result.collision_count,
            "surface_points": result.surface_points,
            "collision_radius_metres": result.collision_radius,
            "frustum_volume_m3": result.frustum_volume,
            "target_particles": result.target_particles,
            "warmup_frames": result.warmup_frames,
            "emission_rate": result.emission_rate,
            "landed_particles": int(result.landed.sum()),
        },
        "video_generation": False,
    }
    (output_dir / "weather.json").write_text(json.dumps(metadata, indent=2))
    return output_dir
