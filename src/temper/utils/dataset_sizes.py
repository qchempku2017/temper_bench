"""Provides shared helpers for converting requested training ratios into
valid frame counts.

The counts are sorted and scaled when needed to respect the maximum
training size.
"""

from __future__ import annotations

import numpy as np

from temper.utils.defaults import (
    DEFAULT_MAX_N_TRAIN,
    DEFAULT_TRAIN_RATIOS,
)


def get_requested_train_sizes_from_ratios(
    trainval_pool_size: int,
    requested_train_ratios: list[float] | None = None,
    max_train_size: int = DEFAULT_MAX_N_TRAIN,
) -> list[int]:
    """Convert sorted training ratios into positive integer frame counts.

    If requested training set size exceeds max_train_size, all requested
    training set sizes are scaled down proportionally so that the
    largest requested training set size equals max_train_size.

    Parameters
    ----------
    trainval_pool_size : int
        Number of available training-plus-validation frames. Must be positive
        and large enough for every requested count to round to at least one.
    requested_train_ratios : list[float] or None, optional
        Nonempty list of fractions strictly between zero and one. None uses
        DEFAULT_TRAIN_RATIOS from temper.utils.defaults. The ratios are sorted
        without modifying the supplied list.
    max_train_size : int, optional
        Positive upper limit on the number of training frames. Defaults to
        DEFAULT_MAX_N_TRAIN, read from the environment at module import time.
        Counts are scaled proportionally when their rounded maximum exceeds
        this limit.

    Returns
    -------
    list[int]
        Frame counts in nondecreasing order, rounded to the nearest integer
        using NumPy's round-to-even rule for ties. Duplicate counts are kept.

    Raises
    ------
    ValueError
        If the ratio list is empty, a ratio is outside (0, 1), or a resulting
        frame count is nonpositive after rounding and any rescaling.
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
        round_sizes = np.round(sizes * max_train_size / np.max(sizes)).astype(
            int
        )

    if np.any(round_sizes <= 0):
        raise ValueError(
            "requested_train_ratios too small for"
            f" trainval_pool_size={trainval_pool_size}, produced training set"
            f" sizes {round_sizes.tolist()}. Choose more appropriate"
            " requested_train_ratios."
        )

    return round_sizes.tolist()
