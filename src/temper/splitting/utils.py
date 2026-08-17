"""Shared deterministic reference and splitting logic for MLFF dataset splitting.

This module provides the common building blocks shared by all splitting
methods ("random" now, "quests" later):

- :func:`get_references_from_frames`: deterministic construction of frame
  references from per-file frame indices.
- :func:`partition_trainval_test`: initial seeded train+validation vs test
  partition (always random), with explicit size/ratio semantics and documented
  rounding.
- :func:`normalize_requested_train_sizes`: normalization of requested training
  sizes, including ratio-to-count conversion.

All randomness here is local to :func:`partition_trainval_test` (a
``numpy.random.default_rng(seed)`` generator); global NumPy state is never
touched.
"""
from __future__ import annotations

from typing import List

import numpy as np

from src.temper.utils.defaults import (
    DEFAULT_MAX_N_TRAIN,
    DEFAULT_TRAIN_RATIOS,
)


def get_requested_train_sizes_from_ratios(
    trainval_pool_size: int,
    requested_train_ratios: List[float] | None = None,
    max_train_size: int = DEFAULT_MAX_N_TRAIN,
) -> List[int]:
    """Get training set sizes from training set ratios.

    If requested training set size exceeds max_train_size, all requested training
    set sizes are scaled down proportionally so that the largest requested
    training set size equals max_train_size.

    Parameters
    ----------
    trainval_pool_size : int
        Size of the trainval pool.
    requested_train_ratios : list[float], optional
        Requested training set ratios. Defaults to
        "DEFAULT_REQUESTED_TRAIN_RATIOS".
    max_train_size : int, optional
        Maximum training set size. Defaults to "DEFAULT_MAX_TRAIN_SIZE".

    Returns
    -------
    list[int]
        Training set sizes in integers sizes.
    """
    if requested_train_ratios is None:
        requested_train_ratios = DEFAULT_TRAIN_RATIOS

    ratios = np.sort(requested_train_ratios)

    if np.any(ratios <= 0.0) or np.any(ratios >= 1.0):
        raise ValueError(
            "requested_train_ratios must be in (0, 1), "
            f"got {requested_train_ratios}."
        )

    sizes = trainval_pool_size * ratios
    round_sizes = np.round(sizes).astype(int)

    if np.max(round_sizes) > max_train_size:
        round_sizes = np.round(sizes * max_train_size / np.max(sizes)).astype(int)

    if np.any(round_sizes <= 0):
        raise ValueError(
            f"requested_train_ratios too small for trainval_pool_size={trainval_pool_size}, "
            f"produced training set sizes {round_sizes.tolist()}."
            f" Choose more appropriate requested_train_ratios."
        )

    return round_sizes.tolist()
