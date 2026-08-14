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

from typing import Dict, List, Tuple

import numpy as np

from src.temper.schemas.split import FrameReference
from src.temper.utils.env import (
    DEFAULT_MAX_N_TRAIN,
    DEFAULT_TEST_RATIO,
    DEFAULT_TRAIN_RATIOS,
)


def get_references_from_frames(
    frames_by_filename: Dict[str, List[int]],
    domain: str,
) -> List[FrameReference]:
    """Deterministically flatten grouped frame indices into frame references.

    Given a mapping ``{filename: [frame_index, ...]}`` (the per-file frame
    indices of a data group), produce a single canonical list of
    :class:`FrameReference`. Filenames are sorted, and the frame indices of
    each file are sorted, so the result is deterministic and independent of the
    order in which the mapping was provided.

    Parameters
    ----------
    frames_by_filename : Dict[str, List[int]]
        Mapping from relative extxyz filenames to lists of frame indices.
    domain : str
        Domain name shared by all produced references.

    Returns
    -------
    list[FrameReference]
        Canonical, deterministic list of frame references.

    Raises
    ------
    ValueError
        If a filename is not a valid relative extxyz path (see
        :func:`src.temper.schemas.utils.validate_relative_extxyz_path`), if a
        frame index is negative, or if a file lists a frame index more than
        once.
"""
    references: List[FrameReference] = []
    for filename in sorted(frames_by_filename):
        frame_indices = frames_by_filename[filename]
        if not isinstance(frame_indices, (list, tuple)):
            raise TypeError(
                f"Frame indices for {filename} must be a list of ints, "
                f"got {type(frame_indices).__name__}."
            )
        unique_indices = set(frame_indices)
        if len(unique_indices) != len(frame_indices):
            raise ValueError(
                f"File {filename} lists duplicate frame indices; "
                "each frame may appear at most once in a pool."
            )
        for frame_index in sorted(frame_indices):
            references.append(
                FrameReference(
                    domain=domain,
                    filename=filename,
                    frame_index=frame_index,
                )
            )
    return references


def partition_trainval_test(
    pool: List[FrameReference],
    *,
    seed: int,
    test_ratio: float = DEFAULT_TEST_RATIO,
) -> Tuple[List[FrameReference], List[FrameReference]]:
    """Perform the initial seeded train+validation vs test partition.

    The partition into (train+validation, test) is always performed at random
    for every splitting method. Exactly one of ``test_ratio`` or ``test_size``
    must be provided.

    Rounding: when ``test_ratio`` is used, the test size is computed as
    ``round(len(pool) * test_ratio)`` using Python's built-in ``round``
    (banker's rounding, half-to-even).

    Parameters
    ----------
    pool : List[FrameReference]
        The complete frame pool of the group (typically the deterministic
        output of :func:`flatten_frames`).
    seed : int
        Random seed for the partition. The partition is reproducible for a
        given ``pool`` order and ``seed``.
    test_ratio : float, optional
        Requested ratio of the test set to the total pool.
        Defaults to "DEFAULT_TEST_RATIO".

    Returns
    -------
    tuple[list[FrameReference], list[FrameReference]]
        ``(trainval_pool, test_set)``. ``trainval_pool`` preserves the order of
        ``pool``; ``test_set`` preserves the order of the selected positions
        within ``pool``.

    Raises
    ------
    ValueError
        If neither nor both of ``test_ratio``/``test_size`` are provided, if the
        resulting test size is outside ``[1, len(pool) - 1]``, or if the pool
        contains duplicate frames.
    """

    n_total = len(pool)

    # test_ratio is not None here.
    if not 0.0 < test_ratio < 1.0:
        raise ValueError(f"test_ratio must be in (0, 1), got {test_ratio}.")
    test_size = round(n_total * test_ratio)
    if test_size <= 0 or test_size >= n_total:
        raise ValueError(
            f"test_ratio {test_ratio} on a pool of {n_total} frames rounds "
            f"to test_size {test_size}, which is outside "
            f"[1, {n_total - 1}]."
        )

    identities = [ref.identity for ref in pool]
    if len(set(identities)) != len(identities):
        raise ValueError("pool must not contain duplicate frames.")

    rng = np.random.default_rng(seed)
    test_positions: list[int] = [  # Annotate to prevent typing mismatch at return.
        int(i)
        for i in np.sort(
            rng.choice(n_total, size=test_size, replace=False)
        )
    ]

    train_positions: list[int] = [
        int(i)
        for i in np.setdiff1d(
            np.arange(n_total),
            test_positions,
        )
    ]

    # Index both outputs against the original pool. The test positions refer
    # to the original pool and must not be applied after it is shortened to
    # the train/validation subset.
    trainval_pool = [pool[i] for i in train_positions]
    test_set = [pool[i] for i in test_positions]
    return trainval_pool, test_set


def get_requested_train_sizes_from_ratios(
    trainval_pool_size: int,
    requested_train_ratios: List[float] | None = None,
    *,
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
    sizes = trainval_pool_size * ratios
    round_sizes = np.round(sizes).astype(int)

    if np.max(round_sizes) > max_train_size:
        round_sizes = np.round(sizes * max_train_size / np.max(sizes)).astype(int)

    return round_sizes.tolist()
