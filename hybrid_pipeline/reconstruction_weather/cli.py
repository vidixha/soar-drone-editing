"""Command-line interface for standalone reconstruction and weather."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from .config import ReconstructionConfig, SnowConfig
from .pipeline import reconstruct, simulate_weather

PIPELINE = Path(__file__).resolve().parents[1]
ROOT = PIPELINE.parent


def _load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key:
            os.environ.setdefault(key, value)


_load_dotenv(PIPELINE / ".env")
_load_dotenv(ROOT / ".env")


def _env_path(name: str, fallback: Path) -> Path:
    value = os.environ.get(name, "").strip()
    if not value:
        return fallback
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    return path


def _env_optional_path(name: str) -> Path | None:
    value = os.environ.get(name, "").strip()
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    return path


def _paths(parser: argparse.ArgumentParser, *, aerialmetric: bool) -> None:
    if not aerialmetric:
        return
    parser.add_argument(
        "--aerialmetric-dir",
        type=Path,
        default=_env_path("AERIALMETRIC_DIR", ROOT / "third_party" / "AerialMetric"),
    )
    parser.add_argument(
        "--aerialmetric-python",
        type=Path,
        default=_env_path(
            "AERIALMETRIC_PYTHON",
            ROOT / "third_party" / "AerialMetric" / ".venv" / "bin" / "python",
        ),
    )
    parser.add_argument(
        "--aerialmetric-checkpoint",
        type=Path,
        default=_env_path("AERIALMETRIC_CHECKPOINT", ROOT / "checkpoints" / "Moge2-Aerial.pt"),
    )
    parser.add_argument(
        "--megasam-dir",
        type=Path,
        default=_env_path("MEGASAM_DIR", ROOT.parent / "mega-sam"),
    )
    parser.add_argument(
        "--megasam-python",
        type=Path,
        default=_env_path("MEGASAM_PYTHON", ROOT.parent / "mega-sam" / ".venv" / "bin" / "python"),
    )
    parser.add_argument(
        "--megasam-checkpoint",
        type=Path,
        default=_env_path(
            "MEGASAM_CHECKPOINT",
            ROOT.parent / "mega-sam" / "checkpoints" / "megasam_final.pth",
        ),
    )
    parser.add_argument(
        "--sam3-dir",
        type=Path,
        default=_env_path("SAM3_DIR", ROOT.parent / "Vista4D"),
    )
    parser.add_argument(
        "--sam3-python",
        type=Path,
        default=_env_path("SAM3_PYTHON", ROOT.parent / "Vista4D" / ".venv" / "bin" / "python"),
    )
    parser.add_argument(
        "--sam3-checkpoint",
        type=Path,
        default=_env_optional_path("SAM3_CHECKPOINT"),
    )


def _reconstruction_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backend", choices=("pi3x", "megasam"), default="pi3x")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--pi3x-model-id", default="yyfz233/Pi3X")
    parser.add_argument("--pi3-pixel-limit", type=int, default=255_000)
    parser.add_argument("--aerialmetric-resize", type=int, default=1024)
    parser.add_argument("--aerialmetric-resolution-level", type=int, default=9)
    parser.add_argument("--aerialmetric-lora-rank", type=int, default=96)
    parser.add_argument("--megasam-static-threshold", type=float, default=0.5)
    parser.add_argument(
        "--dynamic-segmentation",
        choices=("backend", "sam3", "hybrid"),
        default="hybrid",
    )
    parser.add_argument(
        "--dynamic-keywords",
        nargs="+",
        default=("person", "car", "truck", "bus", "bicycle", "motorcycle", "animal"),
    )
    parser.add_argument("--dynamic-mask-dilation", type=int, default=2)
    parser.add_argument(
        "--frame-sampling",
        choices=("span", "center"),
        default="span",
        help="span=uniform across the whole clip; center=middle N frames.",
    )
    parser.add_argument("--no-reconstruction-visualization", action="store_true")


def _weather_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--effect", choices=("snow", "rain", "fog", "sandstorm"), default="snow")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--particles-per-frame", type=int, default=0)
    parser.add_argument("--particles-per-cubic-metre", type=float, default=0.025)
    parser.add_argument("--warmup-seconds", type=float, default=2.0)
    parser.add_argument("--intensity", type=float, default=0.65)
    parser.add_argument("--wind-metres-per-second", type=float, default=0.35)
    parser.add_argument("--turbulence-metres-per-second", type=float, default=0.18)
    parser.add_argument("--flake-diameter-mm", type=float, default=3.0)
    parser.add_argument("--no-collision", action="store_true")
    parser.add_argument("--no-accumulation", action="store_true")
    parser.add_argument("--accumulation-minutes", type=float, default=0.0)
    parser.add_argument("--surface-stride", type=int, default=10)
    parser.add_argument("--surface-frame-stride", type=int, default=3)
    parser.add_argument("--no-sky", action="store_true")
    parser.add_argument("--fog-visibility-metres", type=float, default=0.0)


def _reconstruction_config(args: argparse.Namespace) -> ReconstructionConfig:
    return ReconstructionConfig(
        backend=args.backend,
        width=args.width,
        height=args.height,
        num_frames=args.num_frames,
        pi3x_model_id=args.pi3x_model_id,
        pi3_pixel_limit=args.pi3_pixel_limit,
        aerialmetric_resize=args.aerialmetric_resize,
        aerialmetric_resolution_level=args.aerialmetric_resolution_level,
        aerialmetric_lora_rank=args.aerialmetric_lora_rank,
        megasam_static_threshold=args.megasam_static_threshold,
        dynamic_segmentation=args.dynamic_segmentation,
        dynamic_keywords=tuple(args.dynamic_keywords),
        dynamic_mask_dilation=args.dynamic_mask_dilation,
        frame_sampling=getattr(args, "frame_sampling", "span"),
        save_visualization=not args.no_reconstruction_visualization,
    )


def _reconstruct_kwargs(args: argparse.Namespace) -> dict:
    return {
        "config": _reconstruction_config(args),
        "aerialmetric_dir": args.aerialmetric_dir,
        "aerialmetric_python": args.aerialmetric_python,
        "aerialmetric_checkpoint": args.aerialmetric_checkpoint,
        "megasam_dir": args.megasam_dir,
        "megasam_python": args.megasam_python,
        "megasam_checkpoint": args.megasam_checkpoint,
        "sam3_dir": args.sam3_dir,
        "sam3_python": args.sam3_python,
        "sam3_checkpoint": args.sam3_checkpoint,
    }


def _snow_config(args: argparse.Namespace) -> SnowConfig:
    effect = getattr(args, "effect", "snow")
    intensity = args.intensity
    density = args.particles_per_cubic_metre
    diameter = args.flake_diameter_mm
    wind = args.wind_metres_per_second
    turbulence = args.turbulence_metres_per_second
    minutes = args.accumulation_minutes
    warmup = args.warmup_seconds
    visibility = args.fog_visibility_metres
    collision = not args.no_collision
    accumulation = not args.no_accumulation
    if effect == "rain":
        if intensity == 0.65:
            intensity = 0.95
        if density == 0.025:
            density = 0.07
        if diameter == 3.0:
            diameter = 1.6
        if wind == 0.35:
            wind = 1.2
        if turbulence == 0.18:
            turbulence = 0.28
        if minutes == 0.0:
            minutes = 10.0
        if warmup == 2.0:
            warmup = 1.5
    elif effect == "fog":
        if density == 0.025:
            density = 0.012
        if diameter == 3.0:
            diameter = 1.2
        if wind == 0.35:
            wind = 0.15
        if turbulence == 0.18:
            turbulence = 0.08
        if warmup == 2.0:
            warmup = 0.5
        if visibility == 0.0:
            visibility = 32.0
        collision = False
        accumulation = False
    elif effect == "sandstorm":
        if density == 0.025:
            density = 0.045
        if diameter == 3.0:
            diameter = 1.8
        if wind == 0.35:
            wind = 2.8
        if turbulence == 0.18:
            turbulence = 0.55
        if warmup == 2.0:
            warmup = 1.0
        if visibility == 0.0:
            visibility = 70.0
        accumulation = False
    return SnowConfig(
        effect=effect,
        seed=args.seed,
        particles_per_frame=args.particles_per_frame,
        particles_per_cubic_metre=density,
        warmup_seconds=warmup,
        intensity=intensity,
        wind_metres_per_second=wind,
        turbulence_metres_per_second=turbulence,
        flake_diameter_mm=diameter,
        collision=collision,
        accumulation=accumulation,
        accumulation_minutes=minutes,
        surface_stride=args.surface_stride,
        surface_frame_stride=args.surface_frame_stride,
        replace_sky=not args.no_sky,
        fog_visibility_metres=visibility,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    reconstruct_parser = subparsers.add_parser(
        "reconstruct",
        help="Build metric cameras, depth, and static/dynamic/sky masks.",
    )
    reconstruct_parser.add_argument("input_video", type=Path)
    reconstruct_parser.add_argument("output_dir", type=Path)
    _paths(reconstruct_parser, aerialmetric=True)
    _reconstruction_arguments(reconstruct_parser)

    weather_parser = subparsers.add_parser(
        "weather",
        help="Apply snow or rain to an existing reconstruction.",
    )
    weather_parser.add_argument("reconstruction_dir", type=Path)
    weather_parser.add_argument("output_dir", type=Path)
    _paths(weather_parser, aerialmetric=False)
    _weather_arguments(weather_parser)

    run_parser = subparsers.add_parser(
        "run",
        help="Reconstruct, then apply snow or rain.",
    )
    run_parser.add_argument("input_video", type=Path)
    run_parser.add_argument("output_dir", type=Path)
    _paths(run_parser, aerialmetric=True)
    _reconstruction_arguments(run_parser)
    _weather_arguments(run_parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "reconstruct":
        reconstruct(args.input_video, args.output_dir, **_reconstruct_kwargs(args))
    elif args.command == "weather":
        simulate_weather(
            args.reconstruction_dir,
            args.output_dir,
            config=_snow_config(args),
        )
    else:
        reconstruction_dir = args.output_dir / "reconstruction"
        weather_dir = args.output_dir / "weather"
        reconstruct(args.input_video, reconstruction_dir, **_reconstruct_kwargs(args))
        simulate_weather(
            reconstruction_dir,
            weather_dir,
            config=_snow_config(args),
        )


if __name__ == "__main__":
    main()
