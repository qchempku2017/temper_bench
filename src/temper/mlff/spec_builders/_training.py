"""Small shared validation for native MLFF training lengths."""

from __future__ import annotations

from typing import Any


def validate_epoch(parameters: dict[str, Any], key: str, backend: str) -> int:
    """Return one positive integer epoch count or raise a useful error."""
    # Comment: I prefer moving this function to a more general utility module such as src/temper/utils/validation.py, since it might be used for other purposes as well.
    #    Also, consider renaming it to something more general like validate_positive_integer, since it could be used for other parameters as well.
    #    Find out where this function might be used in the current codebase and replace the original implementation to use it in those places as well.
    value = parameters[key]
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(
            f"{backend} training parameter {key!r} must be a positive integer."
        )
    return value


__all__ = ["validate_epoch"]
