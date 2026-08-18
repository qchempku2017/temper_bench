"""Splits each group of a grouped domain into repeatable train, validation, and test records. It configures the initial test partition and subsequent training-frame selection."""
from __future__ import annotations

from typing import List, Tuple, Dict

import numpy as np

from ase import Atoms

from src.temper.schemas.split import SplitGroup, TrainValSplitTrajectory
from src.temper.schemas.group import GroupedDomain
from src.temper.schemas.frame_refrence import FrameReference
from src.temper.utils.defaults import DEFAULT_TEST_RATIO

from src.temper.splitting.io import FrameReferenceResolver, load_frames_from_references
from src.temper.splitting.selectors import selector_class_factory
from src.temper.splitting.quests_adapter import QuestsDescriptorsStorage, QuestsAdapter
from src.temper.schemas.split import SplitConfig


def partition_trainval_test(
    pool: List[FrameReference],
    seed: int,
    test_ratio: float = DEFAULT_TEST_RATIO,
) -> Tuple[List[FrameReference], List[FrameReference], List[int], List[int]]:
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
    tuple[list[FrameReference], list[FrameReference], list[int], list[int]]
        ``(trainval_pool, test_set, train_positions, test_positions)``. The
        positions are the indices of the references in the original ``pool``,
        i.e. ``trainval_pool = [pool[i] for i in train_positions]`` and
        ``test_set = [pool[i] for i in test_positions]``. The positions are
        always sorted in ascending order, i.e. ``train_positions ==
        sorted(train_positions)`` and ``test_positions == sorted(test_positions)``.
        ``trainval_pool`` preserves the order of
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
    return trainval_pool, test_set, train_positions, test_positions


def split_grouped_domain(
    grouped_domain: GroupedDomain,
    config: SplitConfig | None = None,
) -> List[SplitGroup]:
    """Split groups in a grouped domain into train/validation/test sets.

    Results are reported as `SplitGroup` objects.

    Parameters
    ----------
    grouped_domain : GroupedDomain
        Grouped domain to split.
    config : SplitConfig | None
        Configuration for the split. If not provided, will use all default settings.
        See src.temper.splitting.split_config ``SplitConfig`` for details.

    Returns
    -------
    List[SplitGroup]
        List of split groups hosting train-val-test split records, each corresponds to
        one group in the input grouped domain, and one independent repeated split.

    Raises
    ------
    ValueError
        If ``unified_seed`` is provided and either ``trainval_test_split_seed``
        or ``train_val_split_seed`` is provided, if ``unified_seed`` is negative,
        if ``trainval_test_split_seed`` is negative,
    """
    config = config or SplitConfig()

    # Get pools of references in groups.
    group_pools = grouped_domain.load_frame_references_in_groups()

    # Load pools of frames in groups from references. Resolver reused.
    # For resolver to be reusable, root_path must have been expanded and resolved
    # to match the stored root_path in resolver.
    group_frames: Dict[str, List[Atoms]] = {}
    resolver = FrameReferenceResolver(config.root_path)
    for group_name, pool in group_pools.items():
        structures, resolver = load_frames_from_references(
            pool,
            root_path=config.root_path,
            resolver=resolver,
        )
        group_frames[group_name] = structures

    # Compute descriptor for each pool.
    group_descriptors: Dict[str, QuestsDescriptorsStorage] = {}
    adapter = QuestsAdapter(config.quests_adapter_config)
    for group_name, structures in group_frames.items():
        group_descriptors[group_name] = adapter.compute_descriptors(structures)

    # Determine extra_tests. Pre-specified tests over-write automatic determination.
    group_extra_tests: Dict[str, List[str]] = {}
    if grouped_domain.specify_cross_tests is not None:
        group_extra_tests = grouped_domain.specify_cross_tests
        # Deduplicate.
        for group_name in group_extra_tests:
            group_extra_tests[group_name] = list(
                set(group_extra_tests[group_name]) - {group_name}
            )
    else:
        # Add extra cross tests only if requested.
        if grouped_domain.add_extra_cross_tests:
            all_group_names = set(grouped_domain.groups.keys())
            for group_name in all_group_names:
                other_group_names = all_group_names - {group_name}
                group_extra_tests[group_name] = list(other_group_names)

    # Determine selector class.
    selector_cls = selector_class_factory(selection_method=config.train_val_split_method)

    split_groups = []
    for repeat_id in range(config.split_repeats):
        for group_name, pool in group_pools.items():
            trainval_pool, test_set, trainval_positions, test_positions = partition_trainval_test(
                pool,
                seed=config.trainval_test_split_seeds[repeat_id],
                test_ratio=config.test_ratio,
            )
            selector = selector_cls(
                pool_descriptors=group_descriptors[group_name],
                trainval_indices_in_pool=trainval_positions,
                requested_train_ratios=config.requested_train_ratios,
                max_train_size=config.max_train_size,
                seed=config.train_val_split_seeds[repeat_id],
            )
            selected_frame_indices, remaining_frame_indices, entropy_profile = selector.run()
            selected_frames = [pool[i] for i in selected_frame_indices]
            additional_trainval_frames = [pool[i] for i in remaining_frame_indices]
            trajectory = TrainValSplitTrajectory(
                method=config.train_val_split_method,
                seed=config.train_val_split_seeds[repeat_id],
                requested_train_sizes=selector.requested_train_sizes,
                selected_frames=selected_frames,
                additional_trainval_frames=additional_trainval_frames,
                entropy_profile=entropy_profile,
            )
            split_group = SplitGroup(
                domain=grouped_domain.domain,
                grouping_strategy=grouped_domain.grouping_strategy,
                group_name=group_name,
                test_set=test_set,
                extra_tested_groups=group_extra_tests.get(group_name, []),
                test_ratio=config.test_ratio,
                trainval_test_split_seed=config.trainval_test_split_seeds[repeat_id],
                train_val_split_trajectory=trajectory,
                repeat_id=repeat_id,
                quests_adapter_config=config.quests_adapter_config,
            )
            split_groups.append(split_group)

    return split_groups
