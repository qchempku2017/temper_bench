"""Small shared validation for native MLFF training lengths."""

from __future__ import annotations

from typing import Any


def validate_epoch(parameters: dict[str, Any], key: str, backend: str) -> int:
    """Return one positive integer epoch count or raise a useful error."""
    value = parameters[key]
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(
            f"{backend} training parameter {key!r} must be a positive integer."
        )
    return value


__all__ = ["validate_epoch"]
