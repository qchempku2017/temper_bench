"""Public API for splitting grouped domain data and exporting the resulting datasets."""

from src.temper.splitting.io import (
    FrameReferenceResolver,
    write_all_sets_in_split_group_to_extxyz,
)
from src.temper.splitting.split import split_grouped_domain

__all__ = [
    "FrameReferenceResolver",
    "split_grouped_domain",
    "write_all_sets_in_split_group_to_extxyz",
]
