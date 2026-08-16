"""QUESTS maximum-information-entropy splitting for MLFF datasets.

Selects training set from trainval set, but does not split trainval and test.

At each step, from a pool of candidate structures, the top-`K` structures
that maximizes information entropy gain are selected,
where `K` is the number of structures to select in a step.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Tuple
import logging

import numpy as np

from src.temper.schemas.split import (
    EntropyProfilePoint,
    EntropyProfile,
)
from src.temper.splitting import QuestsDescriptorsStorage, QuestsAdapter
from src.temper.splitting.quests_adapter import compute_information_gain_per_candidate_frame
from src.temper.splitting.utils import get_requested_train_sizes_from_ratios

from src.temper.utils.env import DEFAULT_TRAIN_RATIOS, DEFAULT_MAX_N_TRAIN


class BaseIndicesSelector(ABC):
    """Base selector that defines shared training frame selection mechanism from the trainval set.

    Currently shared by Quests and Random.

    The ``run`` method is the main entry point, which returns a list of training indices in the pool,
    a list of extra validation indices in the pool, and an entropy profile of the training set.
    """
    def __init__(
            self,
            pool_descriptors: QuestsDescriptorsStorage,
            trainval_indices_in_pool: List[int],
            requested_train_ratios: List[float] | None = None,
            max_train_size: int = DEFAULT_MAX_N_TRAIN,
            seed: int | None = None,
            num_selected_per_step: int | None = None,
    ):
        """Selector class for selecting training frames from the trainval set.

        Call the ``run`` method to get the training indices in the full pool,
        as well as the entropy profile of the training set.

        Parameters
        ----------
        pool_descriptors : QuestsDescriptorsStorage
            Descriptors storage for the full pool.
        trainval_indices_in_pool : List[int]
            Indices of the trainval set in the full pool.
        requested_train_ratios : List[float] | None, optional
            List of requested training ratios, by default None.
        max_train_size : int, optional
            Maximum number of training frames, by default DEFAULT_MAX_N_TRAIN.
            See src.temper.utils.env.DEFAULT_MAX_N_TRAIN.
        seed : int | None, optional
            Random seed, by default None.
        num_selected_per_step : int | None, optional
            Number of frames to select per step, by default None.
        """
        self.pool_descriptors = pool_descriptors
        self.trainval_indices_in_pool = trainval_indices_in_pool
        self.pool_indices_in_trainval = {j: i for i, j in enumerate(trainval_indices_in_pool)}
        self.requested_train_ratios = requested_train_ratios or DEFAULT_TRAIN_RATIOS
        self.seed = seed if (seed is not None and seed >= 0) else int(np.random.randint(0, 2**32))
        self.rng = np.random.default_rng(self.seed)
        self.adapter = QuestsAdapter(pool_descriptors.quests_adapter_config)
        self.logger = logging.getLogger(self.__name__)

        self.n_trainval = len(self.trainval_indices_in_pool)
        self.requested_train_sizes = get_requested_train_sizes_from_ratios(
            self.n_trainval, self.requested_train_ratios, max_train_size=max_train_size
        )
        self.num_selected_per_step = num_selected_per_step or max(1, self.n_trainval // 20)
        if self.num_selected_per_step > max(self.requested_train_sizes):
            raise ValueError(
                f"Number of selected frames per step ({self.num_selected_per_step}) "
                f"must be less than or equal to the smallest requested train size "
                f"({min(self.requested_train_sizes)})."
            )
        if self.num_selected_per_step > max(self.requested_train_sizes) // 10:
            self.logger.warning(
                "Number of selected frames per step is more than 10% of the smallest requested train size. "
                "This may lead to overly sparse entropy profile. Consider using finer steps."
            )

    def _initialize_selection(self) -> List[int]:
        # Randomly select num_selected_per_step frames from the trainval pool.
        # Return indices in the full pool, not in the trainval pool.
        return np.sort(
            self.rng.choice(self.trainval_indices_in_pool, self.num_selected_per_step, replace=False)
        ).tolist()

    @abstractmethod
    def _select_func(
            self,
            selected_frame_indices: List[int],
            step_size: int,
    ):
        raise NotImplementedError

    def run(self) -> Tuple[List[int], List[int], EntropyProfile]:
        """Perform the selection process.

        Returns
        -------
        Tuple[List[int], List[int], EntropyProfile]
            Tuple of (selected_frame_indices, remaining_frame_indices, entropy_profile).
        """
        # Initialize the selection with a random subset of the trainval pool.
        self.logger.info("Initializing selection with random subset of the trainval pool.")
        selected_frame_indices = self._initialize_selection()  # Indices in full pool.
        entropy = self.adapter.get_entropy(
            self.pool_descriptors.get_multiple_frames(
                selected_frame_indices
            )
        )
        entropy_trace = [
            EntropyProfilePoint(
                training_size=self.num_selected_per_step,
                cumulative_entropy=entropy,
                information_gain=entropy,
            )
        ]

        # Determine steps of training set size in entropy profile.
        max_size = max(self.requested_train_sizes)
        steps = list(range(
            self.num_selected_per_step,
            max_size,
            self.num_selected_per_step,
        ))
        steps.append(max_size)
        for step_id in range(1, len(steps)):
            step_size = steps[step_id] - steps[step_id - 1]
            new_frame_indices = self._select_func(
                selected_frame_indices,
                step_size,
            )  # Returns indices in full pool.
            selected_frame_indices.extend(new_frame_indices)
            entropy = self.adapter.get_entropy(
                self.pool_descriptors.get_multiple_frames(
                    selected_frame_indices,
                )
            )
            entropy_trace.append(
                EntropyProfilePoint(
                    training_size=len(selected_frame_indices),
                    cumulative_entropy=entropy,
                    information_gain=entropy - entropy_trace[-1].cumulative_entropy,
                )
            )

        # Convert to indices in pool.
        remaining_indices = np.sort(
            np.setdiff1d(self.trainval_indices_in_pool, selected_frame_indices)
        ).astype(int).tolist()
        entropy_profile = EntropyProfile(points=entropy_trace)

        # Returns indices in the full pool.
        return selected_frame_indices, remaining_indices, entropy_profile


#####
# Greedy selection by entropy gain
#####
def greedy_select_frame_indices_by_entropy_gain(
    descriptors: QuestsDescriptorsStorage,
    adapter: QuestsAdapter,
    selected_frame_indices: List[int],
    num_select: int,
    total_frame_indices: List[int] | None = None,
) -> List[int]:
    """Greedily select `K` frame indices to maximize information gain.

    The candidates with the top-`K` summed per-atom differential
    entropy given the currently selected descriptors is chosen.

    Parameters
    ----------
    descriptors : QuestsDescriptorsStorage
        Descriptor slices for the full pool.
    adapter : QuestsAdapter
        Entropy adapter.
    selected_frame_indices : List[int]
        Indices of selected frames in structure pool (i.e, descriptors),
        respecting selection order.
    num_select : int
        Number of frames to select (at most the pool size), i.e., `K`.
    total_frame_indices: List[int]
        Indices of all selectable frames in structure pool (i.e, descriptors).
        If not provided, will use all frames in the pool.

    Returns
    -------
    list[int]
        Selected candidate indices in pool, following order of decreasing information gain.
    """
    total_frame_indices = total_frame_indices or np.arange(descriptors.n_frames).astype(int).tolist()
    # Selected frames must all be within total frames.
    if not np.all(np.isin(selected_frame_indices, total_frame_indices)):
        raise ValueError(
            f"Selected frame indices {selected_frame_indices} are not all within "
            f"total frame indices {total_frame_indices}."
        )

    remaining = np.setdiff1d(
        total_frame_indices, selected_frame_indices
    ).astype(int).tolist()
    if num_select > len(remaining):
        raise ValueError(
            f"Cannot select {num_select} frames from a pool of "
            f"{len(remaining)} remaining frames."
        )

    deltas = compute_information_gain_per_candidate_frame(
        descriptors,
        adapter,
        selected_frame_indices,
        remaining,
    )

    # Select by decreasing delta entropy. Select top-K. Map to indices in total pool.
    selected = np.array(remaining)[np.argsort(deltas)[-num_select:][::-1]].tolist()
    return selected


class QuestsIndicesSelector(BaseIndicesSelector):
    """Quests selector that provides selection of training frames by maximization of entropy."""
    def _select_func(
            self,
            selected_frame_indices: List[int],
            step_size: int,
    ):
        return greedy_select_frame_indices_by_entropy_gain(
            self.pool_descriptors,
            self.adapter,
            selected_frame_indices,
            step_size,
            total_frame_indices=self.trainval_indices_in_pool,
        )

#####
# Select at fully random
#####
def select_frame_indices_at_random(
    selected_frame_indices: List[int],
    num_select: int,
    total_frame_indices: List[int],
    rng: np.random.Generator,
) -> List[int]:
    """Select `K` frame indices at random.

    Parameters
    ----------
    selected_frame_indices : List[int]
        Indices of selected frames in structure pool (i.e, descriptors),
        respecting selection order.
    num_select : int
        Number of frames to select (at most the pool size), i.e., `K`.
    total_frame_indices: List[int]
        Indices of all selectable frames in structure pool (i.e, descriptors).
        Must be provided.
    rng: np.random.Generator
        Random number generator.

    Returns
    -------
    list[int]
        Selected candidate indices in pool, following order of decreasing information gain.
    """
    # Selected frames must all be within total frames.
    if not np.all(np.isin(selected_frame_indices, total_frame_indices)):
        raise ValueError(
            f"Selected frame indices {selected_frame_indices} are not all within "
            f"total frame indices {total_frame_indices}."
        )

    remaining = np.setdiff1d(
        total_frame_indices, selected_frame_indices
    ).astype(int).tolist()
    if num_select > len(remaining):
        raise ValueError(
            f"Cannot select {num_select} frames from a pool of "
            f"{len(remaining)} remaining frames."
        )

    # Select at random.
    return np.sort(rng.choice(remaining, num_select, replace=False)).tolist()



class RandomIndicesSelector(BaseIndicesSelector):
    """Random selector that selects training frames at random."""
    def _select_func(
            self,
            selected_frame_indices: List[int],
            step_size: int,
    ):
        return select_frame_indices_at_random(
            selected_frame_indices,
            step_size,
            total_frame_indices=self.trainval_indices_in_pool,
            rng=self.rng,
        )


def selector_class_factory(
    selection_method: str,
) -> type[BaseIndicesSelector]:
    if selection_method == "random":
        return RandomIndicesSelector
    if selection_method == "quests":
        return QuestsIndicesSelector
    raise ValueError(f"Unknown trainval_test_split_method: {selection_method}")