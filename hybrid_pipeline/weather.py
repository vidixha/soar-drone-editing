"""AERIE weather module. Router calls apply_weather().

All four kinds go through reconstruction_weather: metric cameras + AerialMetric
depth. Snow and sandstorm share the flake engine (sand is colour + wind).
Rain is streaks. Fog is metric Beer-Lambert haze plus sparse motes.

Same instruction surface as the rest of the router:
  --instruction "add snow" / "make it rainy" / "add fog" / "add a sandstorm"
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from reconstruction_weather.config import SnowConfig
from reconstruction_weather.weather import (
    simulate_fog,
    simulate_rain,
    simulate_sandstorm,
    simulate_snow,
)

INTENSITY = {"light": 0.35, "medium": 0.65, "heavy": 1.8}
METRIC_KINDS = {"snow", "rain", "fog", "sandstorm"}
_SIMULATE = {
    "snow": simulate_snow,
    "rain": simulate_rain,
    "fog": simulate_fog,
    "sandstorm": simulate_sandstorm,
}


def snow_config(kind: str, intensity: str | float, **overrides) -> SnowConfig:
    word = intensity if isinstance(intensity, str) and intensity in INTENSITY else "medium"
    numeric_maps = {
        "rain": {"light": 0.55, "medium": 0.95, "heavy": 1.8},
        "fog": {"light": 0.45, "medium": 0.85, "heavy": 1.5},
        "sandstorm": {"light": 0.45, "medium": 0.85, "heavy": 1.6},
    }
    if isinstance(intensity, str):
        numeric = numeric_maps.get(kind, INTENSITY)[word]
    else:
        numeric = float(intensity)
    density, diameter, wind, turbulence, minutes = 0.025, 3.0, 0.35, 0.18, 0.0
    warmup = 0.5 if word == "light" else 1.5 if word == "heavy" else 1.0
    visibility = 0.0
    collision, accumulation = True, True
    if kind == "rain":
        density, diameter, wind, turbulence, minutes, warmup = 0.07, 1.6, 1.2, 0.28, 10.0, 1.5
    elif kind == "fog":
        density, diameter, wind, turbulence, minutes, warmup = 0.012, 1.2, 0.15, 0.08, 0.0, 0.5
        visibility = {"light": 70.0, "medium": 32.0, "heavy": 16.0}[word]
        collision, accumulation = False, False
    elif kind == "sandstorm":
        density, diameter, wind, turbulence, minutes, warmup = 0.045, 1.8, 2.8, 0.55, 0.0, 1.0
        visibility = {"light": 110.0, "medium": 70.0, "heavy": 38.0}[word]
        accumulation = False
    payload = dict(
        effect=kind,
        seed=1,
        particles_per_frame=0,
        particles_per_cubic_metre=density,
        warmup_seconds=warmup,
        intensity=numeric,
        wind_metres_per_second=wind,
        turbulence_metres_per_second=turbulence,
        flake_diameter_mm=diameter,
        collision=collision,
        accumulation=accumulation,
        accumulation_minutes=minutes,
        surface_stride=10,
        surface_frame_stride=3,
        replace_sky=True,
        fog_visibility_metres=visibility,
    )
    payload.update(overrides)
    return SnowConfig(**payload)


def load_reconstruction(reconstruction_dir: Path) -> dict:
    reconstruction_dir = Path(reconstruction_dir).resolve()
    if not (reconstruction_dir / "reconstruction.json").is_file():
        raise FileNotFoundError(
            f"not a reconstruction artifact (missing reconstruction.json): {reconstruction_dir}"
        )
    cameras = np.load(reconstruction_dir / "cameras.npz")
    depths = np.load(reconstruction_dir / "depths.npy").astype(np.float32)
    static = np.load(reconstruction_dir / "static_mask.npy").astype(bool)
    accumulation_path = reconstruction_dir / "accumulation_mask.npy"
    accumulation = (
        np.load(accumulation_path).astype(bool) if accumulation_path.is_file() else static
    )
    valid_path = reconstruction_dir / "aerialmetric_valid.npy"
    sky_path = reconstruction_dir / "sky_mask.npy"
    if valid_path.is_file():
        sky = ~np.load(valid_path).astype(bool)
    elif sky_path.is_file():
        sky = np.load(sky_path).astype(bool)
    else:
        sky = depths >= 999
    return {
        "dir": reconstruction_dir,
        "depths": depths,
        "cam_c2w": cameras["cam_c2w"],
        "intrinsics": cameras["intrinsics"],
        "static": static,
        "accumulation": accumulation,
        "sky": sky,
    }


def _align_rgb(frames_bgr: list, height: int, width: int, count: int) -> np.ndarray:
    if not frames_bgr:
        raise ValueError("no frames to weather")
    if len(frames_bgr) == count:
        picked = frames_bgr
    else:
        indices = np.linspace(0, len(frames_bgr) - 1, count)
        picked = [frames_bgr[int(round(i))] for i in indices]
    aligned = []
    for frame in picked:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if rgb.shape[0] != height or rgb.shape[1] != width:
            rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
        aligned.append(rgb)
    return np.stack(aligned)


def _apply_metric(frames_bgr, kind, intensity, *, reconstruction_dir, fps, config_overrides):
    recon = load_reconstruction(reconstruction_dir)
    count, height, width = recon["depths"].shape
    video = _align_rgb(frames_bgr, height, width, count)
    config = snow_config(kind, intensity, **(config_overrides or {}))
    simulate = _SIMULATE[kind]
    result = simulate(
        video,
        recon["depths"],
        recon["cam_c2w"],
        recon["intrinsics"],
        recon["static"],
        recon["accumulation"],
        recon["sky"],
        fps=float(fps),
        config=config,
    )
    out_h, out_w = frames_bgr[0].shape[:2]
    out = []
    for frame in result.video:
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if bgr.shape[0] != out_h or bgr.shape[1] != out_w:
            bgr = cv2.resize(bgr, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        out.append(bgr)
    detail = (
        f"metric {kind} intensity={config.intensity:g} "
        f"particles={result.target_particles} collisions={result.collision_count} "
        f"recon={recon['dir'].name}"
    )
    return out, detail


def apply_weather(frames, depth, kind, intensity="medium", *, reconstruction_dir=None,
                  fps=30, config_overrides=None, **_ignored):
    """Router entry. All kinds need a reconstruction dir."""
    if kind not in _SIMULATE:
        raise ValueError(f"unknown weather kind {kind!r}")
    if not reconstruction_dir:
        raise FileNotFoundError(
            "weather needs a reconstruction (router builds one from --clip)"
        )
    return _apply_metric(
        frames, kind, intensity,
        reconstruction_dir=reconstruction_dir,
        fps=fps,
        config_overrides=config_overrides,
    )
