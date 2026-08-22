"""Standalone 4D reconstruction and geometry-aware weather simulation."""

from typing import TYPE_CHECKING, Any

from .config import ReconstructionConfig, SnowConfig

if TYPE_CHECKING:
    from .pipeline import reconstruct, simulate_weather
    from .weather import SnowResult, simulate_fog, simulate_rain, simulate_sandstorm, simulate_snow

__all__ = [
    "ReconstructionConfig",
    "SnowConfig",
    "SnowResult",
    "reconstruct",
    "simulate_fog",
    "simulate_rain",
    "simulate_sandstorm",
    "simulate_snow",
    "simulate_weather",
]


def __getattr__(name: str) -> Any:
    """Keep AerialMetric's subprocess free from reconstruction runtime imports."""
    if name in {"reconstruct", "simulate_weather"}:
        from . import pipeline

        return getattr(pipeline, name)
    if name in {"SnowResult", "simulate_fog", "simulate_rain", "simulate_sandstorm", "simulate_snow"}:
        from . import weather

        return getattr(weather, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
