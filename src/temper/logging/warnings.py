"""Warning categories emitted by TEMPER."""

from __future__ import annotations


class TemperWarning(UserWarning):
    """Base class for actionable, non-fatal TEMPER conditions."""


class DataQualityWarning(TemperWarning):
    """Warn that input data may be incomplete or unsuitable for some uses."""


class BackendFallbackWarning(TemperWarning):
    """Warn that a requested automatic backend route used a slower fallback."""


class PerformanceWarning(TemperWarning):
    """Warn that a configuration may perform poorly or produce sparse output."""


__all__ = [
    "TemperWarning",
    "DataQualityWarning",
    "BackendFallbackWarning",
    "PerformanceWarning",
]
