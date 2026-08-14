"""Random (reproducible) dataset splitting.

Provides generation of a nested random train/validation trajectory that
follows the shared :class:`TrainValSplitTrajectory` output convention. Only local
``numpy.random.Generator`` instances are used; global NumPy state is never
touched.
"""
# TODO: need further human review.
from __future__ import annotations

from typing import List, Sequence

import numpy as np

from src.temper.schemas.split import FrameReference, TrainValSplitTrajectory


def _validate_requested_train_sizes(
    requested_train_sizes: Sequence[int], pool_size: int
) -> List[int]:
    """Validate trajectory sizes without depending on the removed common helper."""
    sizes = list(requested_train_sizes)
    if not sizes:
        raise ValueError("requested_train_sizes must not be empty.")
    for index, size in enumerate(sizes):
        if not isinstance(size, (int, np.integer)) or isinstance(size, (bool, np.bool_)):
            raise TypeError(f"requested_train_sizes[{index}] must be an integer.")
        if size <= 0 or size > pool_size:
            raise ValueError(
                f"requested_train_sizes[{index}] must be in [1, {pool_size}], got {size}."
            )
        if index and size <= sizes[index - 1]:
            raise ValueError("requested_train_sizes must be strictly increasing.")
    return [int(size) for size in sizes]


def generate_random_trajectory(
    *,
    seed: int,
    pool: Sequence[FrameReference],
    requested_train_sizes: Sequence[int],
) -> TrainValSplitTrajectory:
    """Generate a nested random train trajectory.

    The frames of ``pool`` are shuffled with a local
    ``numpy.random.default_rng(seed)`` generator and the resulting order is
    stored as the trajectory's ``selected_frames`` (truncated to the largest
    requested training size). Prefixes of this list are the nested training
    sets; the validation set at any requested size is the remaining selected
    suffix plus additional train/validation frames (see
    :meth:`TrainValSplitTrajectory.get_train_set` and
    :meth:`TrainValSplitTrajectory.get_val_set`).

    The result is fully reproducible for a given ``seed`` and ``pool`` order.
    The returned trajectory carries ``entropy_profile=None``: the entropy
    values are reserved to be populated later by the QUESTS objective, and no
    unrelated entropy proxy is invented.

    Parameters
    ----------
    seed : int
        Random seed. The trajectory stores this seed.
    pool : Sequence[FrameReference]
        The train+validation pool (typically the output of
        :func:`src.temper.splitting.common.partition_trainval_test`).
    requested_train_sizes : Sequence[int]
        Strictly increasing requested training sizes, each at most the pool
        size.

    Returns
    -------
    TrainValSplitTrajectory
        A ``method="random"`` trajectory.

    Raises
    ------
    ValueError
        If the requested sizes are invalid.
    """
    sizes = _validate_requested_train_sizes(requested_train_sizes, len(pool))

    rng = np.random.default_rng(seed)
    positions = rng.permutation(len(pool))
    selected_count = sizes[-1]
    selected_positions = positions.tolist()[:selected_count]
    selected_frames: List[FrameReference] = [pool[i] for i in selected_positions]
    selected_identities = {ref.identity for ref in selected_frames}
    additional_trainval_frames = [
        ref for ref in pool if ref.identity not in selected_identities
    ]

    return TrainValSplitTrajectory(
        method="random",
        seed=seed,
        requested_train_sizes=sizes,
        selected_frames=selected_frames,
        additional_trainval_frames=additional_trainval_frames,
        entropy_profile=None,
    )
