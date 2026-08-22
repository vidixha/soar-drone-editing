"""Configuration models for reconstruction and weather simulation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ReconstructionConfig:
    """Selectable 4D reconstruction with AerialMetric metric depth."""

    backend: str = "pi3x"
    width: int = 1280
    height: int = 720
    num_frames: int = 49
    pi3x_model_id: str = "yyfz233/Pi3X"
    pi3_pixel_limit: int = 255_000
    aerialmetric_resize: int = 1024
    aerialmetric_resolution_level: int = 9
    aerialmetric_lora_rank: int = 96
    megasam_static_threshold: float = 0.5
    dynamic_segmentation: str = "hybrid"
    dynamic_keywords: tuple[str, ...] = (
        "person",
        "car",
        "truck",
        "bus",
        "bicycle",
        "motorcycle",
        "animal",
    )
    dynamic_mask_dilation: int = 2
    frame_sampling: str = "span"
    save_visualization: bool = True

    def validate(self) -> None:
        if self.backend not in {"pi3x", "megasam"}:
            raise ValueError("backend must be 'pi3x' or 'megasam'")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("reconstruction width and height must be positive")
        if self.num_frames <= 0:
            raise ValueError("num_frames must be positive")
        if not self.pi3x_model_id:
            raise ValueError("pi3x_model_id cannot be empty")
        if self.pi3_pixel_limit <= 0 or self.aerialmetric_resize <= 0:
            raise ValueError("model inference resolutions must be positive")
        if not 0 <= self.aerialmetric_resolution_level <= 9:
            raise ValueError("aerialmetric_resolution_level must be between 0 and 9")
        if self.aerialmetric_lora_rank <= 0:
            raise ValueError("aerialmetric_lora_rank must be positive")
        if not 0 <= self.megasam_static_threshold <= 1:
            raise ValueError("megasam_static_threshold must be between 0 and 1")
        if self.dynamic_segmentation not in {"backend", "sam3", "hybrid"}:
            raise ValueError(
                "dynamic_segmentation must be 'backend', 'sam3', or 'hybrid'"
            )
        if self.dynamic_segmentation != "backend" and not self.dynamic_keywords:
            raise ValueError("dynamic_keywords cannot be empty when using SAM3")
        if self.dynamic_mask_dilation < 0:
            raise ValueError("dynamic_mask_dilation cannot be negative")
        if self.frame_sampling not in {"span", "center"}:
            raise ValueError("frame_sampling must be 'span' or 'center'")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SnowConfig:
    """Geometry-aware weather controls, expressed in metric units."""

    effect: str = "snow"
    seed: int = 1
    particles_per_frame: int = 0
    particles_per_cubic_metre: float = 0.025
    warmup_seconds: float = 2.0
    intensity: float = 0.65
    wind_metres_per_second: float = 0.35
    turbulence_metres_per_second: float = 0.18
    flake_diameter_mm: float = 3.0
    collision: bool = True
    accumulation: bool = True
    accumulation_minutes: float = 0.0
    surface_stride: int = 10
    surface_frame_stride: int = 3
    replace_sky: bool = True
    fog_visibility_metres: float = 0.0

    def validate(self) -> None:
        if self.effect not in {"snow", "rain", "fog", "sandstorm"}:
            raise ValueError("effect must be 'snow', 'rain', 'fog', or 'sandstorm'")
        if self.particles_per_frame < 0:
            raise ValueError("particles_per_frame cannot be negative")
        if self.particles_per_cubic_metre < 0:
            raise ValueError("particles_per_cubic_metre cannot be negative")
        if self.warmup_seconds < 0 or self.intensity <= 0:
            raise ValueError("warmup_seconds cannot be negative and intensity must be positive")
        if self.turbulence_metres_per_second < 0 or self.flake_diameter_mm <= 0:
            raise ValueError("turbulence cannot be negative and flake diameter must be positive")
        if self.accumulation_minutes < 0:
            raise ValueError("accumulation_minutes cannot be negative")
        if self.surface_stride < 2 or self.surface_frame_stride < 1:
            raise ValueError("surface_stride must be >= 2 and surface_frame_stride must be >= 1")
        if self.fog_visibility_metres < 0:
            raise ValueError("fog_visibility_metres cannot be negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
