"""Small shared validation for native MLFF training lengths."""

from __future__ import annotations

from typing import Any


def validate_epoch(parameters: dict[str, Any], key: str, backend: str) -> int:
    """Return one positive integer epoch count or raise a useful error."""
    # Comment: If this function is only used for validating epochs, remove it and inline the logic where it is used.
    #  Otherwise, rename it to something more generic like `validate_positive_integer`, and move to general utilities.
    value = parameters[key]
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(
            f"{backend} training parameter {key!r} must be a positive integer."
        )
    return value


__all__ = ["validate_epoch"]
