"""High-level split orchestration for the temper benchmark.

This module composes the completed dataset-splitting components into a single
explicit, typed entry point. Given the grouped source inventory of a data
group (per-file frame indices plus the aligned ASE structures), it:

1. builds the canonical, deterministic :class:`FrameReference` list and the
   positionally aligned structure list (see
   :func:`flatten_frames_and_structures`),
2. performs the initial seeded train+validation vs test partition (always
   random; see :func:`src.temper.splitting.common.partition_trainval_test`),
3. normalizes the requested training sizes (counts or pool ratios, with the
   documented ``DEFAULT_MAX_N_TRAIN`` cap; see
   :func:`src.temper.splitting.common.normalize_requested_train_sizes`),
4. generates one trajectory per explicitly selected method
   (``"random"`` and/or ``"quests"``) over the same train+validation pool, so
   both methods can be produced consistently in a single schema,
5. evaluates the QUESTS maximum-entropy profile of every trajectory — the
   ``"random"`` trajectory gets its entropy profile computed with the exact
   same :class:`QuestsObjective` used for ``"quests"`` selection,
6. persists everything as :class:`SplitDataSchema` results, one per selected
   method, each with a singular trajectory, storing the
   :class:`QuestsSplitConfig` provenance in ``quests_config``.

Method selection is an explicit sequence (a single method or a list of
methods); no registry or opaque dispatch is used. The schema never stores
structures or descriptors — only domain, relative extxyz source filename, and
nonnegative frame index.

Semantics
---------
- **train+validation vs test** is always partitioned at random with
  ``split_seed`` (see ``partition_trainval_test``).
- **train vs validation** follows each trajectory's ``selected_frames``: the
  prefix of length ``s`` is the training set of size ``s`` and the validation
  set is the remaining selected suffix plus additional train/validation frames
   (see :meth:`TrainValSplitTrajectory.get_train_set` and
   :meth:`TrainValSplitTrajectory.get_val_set`).
- **requested training sizes** are normalized to strictly increasing positive
  integers at most the train+validation pool size.

Threading / QUESTS oversubscription
-----------------------------------
The orchestration never mutates global environment variables at import time.
The only process-wide thread control is
:attr:`QuestsSplitConfig.numba_threads`: it is applied by
:meth:`QuestsObjective._configure_cpu_threads` immediately before the numba
CPU backend is imported, so no CPU backend import happens before the thread
count is set. Because ``numba.set_num_threads`` is process-wide, the last
objective created with a non-``None`` ``numba_threads`` wins for the process;
document this when sharing a process across splits.
"""
# TODO: need further human review.
from __future__ import annotations

from typing import Dict, List, Literal, Mapping, Sequence, Tuple

from ase import Atoms

from src.temper.schemas.split import (
    FrameReference,
    QuestsSplitConfig,
    SplitDataSchema,
)
from src.temper.splitting import (
    get_references_from_frames,
    get_requested_train_sizes_from_ratios,
    partition_trainval_test,
)


def _normalize_requested_train_sizes(
    requested_train_sizes: Sequence[float | int],
    pool_size: int,
    *,
    as_ratio: bool,
    max_train_size: int,
) -> List[int]:
    """Normalize exact counts locally or delegate ratio conversion to common."""
    if as_ratio:
        return get_requested_train_sizes_from_ratios(
            pool_size,
            [float(ratio) for ratio in requested_train_sizes],
            max_train_size=max_train_size,
        )

    sizes = list(requested_train_sizes)
    if not sizes:
        raise ValueError("requested_train_sizes must not be empty.")
    for index, size in enumerate(sizes):
        if not isinstance(size, int) or isinstance(size, bool):
            raise TypeError(f"requested_train_sizes[{index}] must be an integer.")
        if size <= 0 or size > pool_size or size > max_train_size:
            raise ValueError(
                f"requested_train_sizes[{index}] must be in [1, "
                f"{min(pool_size, max_train_size)}], got {size}."
            )
        if index and size <= sizes[index - 1]:
            raise ValueError("requested_train_sizes must be strictly increasing.")
    return sizes
from src.temper.splitting.quests import (
    QuestsObjective,
    _validate_objective_config,
    generate_quests_trajectory,
    populate_entropy_profile,
)
from src.temper.splitting.random import generate_random_trajectory
from src.temper.utils.env import (
    DEFAULT_MAX_N_TRAIN,
    DEFAULT_TEST_RATIO,
    DEFAULT_TRAIN_RATIOS,
)

#: The splitting methods supported by the high-level orchestration.
SplitMethod = Literal["random", "quests"]


def flatten_frames_and_structures(
    frames_by_filename: Mapping[str, Sequence[int]],
    structures_by_filename: Mapping[str, Sequence[Atoms]],
    domain: str,
) -> Tuple[List[FrameReference], List[Atoms]]:
    """Build canonical frame references and their positionally aligned structures.

    The references are produced by :func:`get_references_from_frames` (filenames sorted,
    frame indices sorted, duplicates rejected). The returned structures list is
    aligned positionally with those references: ``structures[i]`` is the
    structure that reference ``references[i]`` points to, where
    ``structures_by_filename[filename][frame_index]`` is the structure of the
    frame identified by ``(domain, filename, frame_index)``.

    This is the single place that verifies the structure inventory ordering
    exactly matches the generated references: the two mappings must have
    exactly the same filenames, and every referenced ``frame_index`` must lie
    within ``structures_by_filename[filename]``.

    Parameters
    ----------
    frames_by_filename : Mapping[str, Sequence[int]]
        Per-file frame indices of the group (see :func:`get_references_from_frames`).
    structures_by_filename : Mapping[str, Sequence[Atoms]]
        Per-file ordered lists of ASE structures, aligned with the filenames of
        ``frames_by_filename``. Each file must contain at least one structure
        per referenced frame index.
    domain : str
        Domain name shared by all produced references.

    Returns
    -------
    tuple[list[FrameReference], list[Atoms]]
        ``(references, structures)`` with ``structures[i]`` aligned to
        ``references[i]``.

    Raises
    ------
    ValueError
        If the two mappings do not have exactly the same filenames, if a
        referenced frame index is out of range for its file's structure list,
        or if the frame-index mapping is invalid (see
        :func:`get_references_from_frames`).
    TypeError
        If a filename's structure value is not a sequence, or if a frame-index
        value is not a list/tuple of ints.
    """
    # Validate the container types BEFORE coercion: list(Atoms) would silently
    # iterate an Atoms object into a list of Atom objects, so non-sequence
    # values must be rejected explicitly.
    frames: Dict[str, List[int]] = {}
    for name, indices in frames_by_filename.items():
        if not isinstance(indices, (list, tuple)):
            raise TypeError(
                f"frames_by_filename[{name!r}] must be a list/tuple of ints, "
                f"got {type(indices).__name__}."
            )
        frames[str(name)] = list(indices)

    structures: Dict[str, List[Atoms]] = {}
    for name, atoms_list in structures_by_filename.items():
        if isinstance(atoms_list, Atoms) or not isinstance(atoms_list, (list, tuple)):
            raise TypeError(
                f"structures_by_filename[{name!r}] must be a sequence of "
                f"Atoms, got {type(atoms_list).__name__}."
            )
        structures[str(name)] = list(atoms_list)

    missing = sorted(set(frames) - set(structures))
    extra = sorted(set(structures) - set(frames))
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing from structures_by_filename: {missing}")
        if extra:
            details.append(f"not referenced in frames_by_filename: {extra}")
        raise ValueError(
            "frames_by_filename and structures_by_filename must reference "
            f"exactly the same filenames; {'; '.join(details)}."
        )

    # Builds the canonical reference order (sorted filenames, then sorted frame
    # indices); FrameReference validates each reference's schema fields.
    references = get_references_from_frames(frames, domain)

    aligned: List[Atoms] = []
    for filename in sorted(frames):
        frame_indices = frames[filename]
        if not frame_indices:
            continue
        file_structures = structures[filename]
        largest_index = max(frame_indices)
        if len(file_structures) <= largest_index:
            raise ValueError(
                f"structures_by_filename[{filename!r}] has "
                f"{len(file_structures)} frames, but frames_by_filename "
                f"references frame index {largest_index} (0-based)."
            )
        for frame_index in sorted(frame_indices):
            atoms = file_structures[frame_index]
            if not isinstance(atoms, Atoms):
                raise TypeError(
                    f"structures_by_filename[{filename!r}][{frame_index}] "
                    f"must be an ase.Atoms, got {type(atoms).__name__}."
                )
            aligned.append(atoms)

    return references, aligned


def _normalize_method_sequence(
    train_val_split_method: SplitMethod | Sequence[SplitMethod],
) -> List[str]:
    """Validate and normalize the explicit method selection sequence.

    Accepts a single method or a sequence of methods. The sequence order is
    preserved in the returned list, and therefore in the schema's
    ``trajectories`` list. Unknown methods, an empty sequence, and duplicate
    methods are rejected.

    Parameters
    ----------
    train_val_split_method : SplitMethod | Sequence[SplitMethod]
        One of ``"random"``/``"quests"``, or a sequence of them.

    Returns
    -------
    list[str]
        The validated method names in the requested order.

    Raises
    ------
    ValueError
        If the sequence is empty, contains an unknown method, or contains
        duplicates.
    """
    if isinstance(train_val_split_method, str):
        methods: List[str] = [train_val_split_method]
    else:
        methods = list(train_val_split_method)
    if not methods:
        raise ValueError(
            "train_val_split_method must contain at least one method."
        )
    for index, method in enumerate(methods):
        if method not in ("random", "quests"):
            raise ValueError(
                f"train_val_split_method[{index}] must be 'random' or "
                f"'quests', got {method!r}."
            )
    duplicates = sorted({method for method in methods if methods.count(method) > 1})
    if duplicates:
        raise ValueError(
            "train_val_split_method must not contain duplicate methods; "
            f"got duplicates {duplicates}."
        )
    return methods


def split_data_group(
    *,
    frames_by_filename: Mapping[str, Sequence[int]],
    structures_by_filename: Mapping[str, Sequence[Atoms]],
    domain: str,
    grouping_strategy: str,
    group_name: str,
    split_seed: int,
    train_val_split_method: SplitMethod | Sequence[SplitMethod],
    quests_config: QuestsSplitConfig,
    test_ratio: float | None = None,
    test_size: int | None = None,
    requested_train_sizes: Sequence[float | int] | None = None,
    as_ratio: bool = True,
    max_train_size: int | None = None,
    random_seed: int | None = None,
    objective: QuestsObjective | None = None,
) -> List[SplitDataSchema]:
    """Split a data group into train/validation/test and persist the result.

    Composes the completed splitting components into one
    :class:`SplitDataSchema` per selected method (see the module docstring for
    the pipeline); each result has one singular trajectory.
    The train+validation vs test partition is always random (``split_seed``);
    the train vs validation split follows the selected trajectory method(s).

    Parameters
    ----------
    frames_by_filename : Mapping[str, Sequence[int]]
        Per-file frame indices of the group (see :func:`get_references_from_frames`).
    structures_by_filename : Mapping[str, Sequence[Atoms]]
        Per-file ordered ASE structures aligned with ``frames_by_filename``;
        ordering is verified against the generated references.
    domain : str
        Data domain name (persisted on every reference).
    grouping_strategy : str
        Grouping strategy name (persisted provenance).
    group_name : str
        Group name (persisted provenance).
    split_seed : int
        Random seed for the train+validation vs test partition.
    train_val_split_method : SplitMethod | Sequence[SplitMethod]
        Explicit method selection: ``"random"``, ``"quests"``, or a sequence
        of both (e.g. ``["random", "quests"]``) to generate one trajectory per
        method in the given order inside a single schema. No registry or
        opaque dispatch is used.
    quests_config : QuestsSplitConfig
        QUESTS descriptor/entropy/device configuration. Used to evaluate the
        entropy profile of every trajectory (including ``"random"``) and to
        drive ``"quests"`` selection; persisted as the schema's
        ``quests_config`` provenance.
    test_ratio : float | None, optional
        Requested ratio of the test set to the total pool. Mutually exclusive
        with ``test_size``. Defaults to
        :data:`src.temper.utils.env.DEFAULT_TEST_RATIO` when both are None.
    test_size : int | None, optional
        Requested exact number of test frames. Mutually exclusive with
        ``test_ratio``. The schema's persisted ``test_ratio`` is then the
        derived value ``test_size / total``.
    requested_train_sizes : Sequence[float | int] | None, optional
        Requested training sizes as pool ratios (``as_ratio=True``) or exact
        integer counts (``as_ratio=False``). Defaults to
        :data:`src.temper.utils.env.DEFAULT_TRAIN_RATIOS` (ratios) when None.
    as_ratio : bool, optional
        Whether ``requested_train_sizes`` are ratios of the train+validation
        pool size (True, default) or exact integer counts (False). Ignored
        when ``requested_train_sizes`` is None.
    max_train_size : int | None, optional
        Hard cap on every training size. Defaults to
        :data:`src.temper.utils.env.DEFAULT_MAX_N_TRAIN`. When ``as_ratio`` is
        True and the largest requested size would exceed the cap, all ratios
        are scaled down proportionally (see
        :func:`src.temper.splitting.common.normalize_requested_train_sizes`).
    random_seed : int | None, optional
        Seed of the ``"random"`` trajectory. Required (non-None) exactly when
        ``"random"`` is in ``train_val_split_method``; ignored otherwise.
    objective : QuestsObjective | None, optional
        Reusable QUESTS objective. When provided it is used for entropy
        evaluation and ``"quests"`` selection; otherwise a fresh
        ``QuestsObjective(quests_config)`` is constructed lazily. When
        provided its ``config`` must exactly equal ``quests_config`` (which
        remains the persisted provenance); a mismatch raises ``ValueError``
        before any computation.

    Returns
    -------
    list[SplitDataSchema]
        One persisted split result per selected method, each embedding its
        complete train+validation inventory in its singular trajectory.

    Raises
    ------
    ValueError
        If the method selection is invalid, ``frames_by_filename`` and
        ``structures_by_filename`` are misaligned, both/neither of
        ``test_ratio``/``test_size`` are provided with an invalid result, the
        requested training sizes are invalid (see
        :func:`src.temper.splitting.common.normalize_requested_train_sizes`),
        ``"random"`` is selected without ``random_seed``, or a supplied
        ``objective`` exposes a ``config`` that differs from ``quests_config``.
    """
    methods = _normalize_method_sequence(train_val_split_method)
    if "random" in methods and random_seed is None:
        raise ValueError(
            "random_seed must be provided when 'random' is in "
            "train_val_split_method."
        )

    # A reusable objective must be consistent with the persisted config, or
    # descriptors/entropy would be evaluated with different parameters than
    # the schema's quests_config provenance.
    _validate_objective_config(
        quests_config,
        objective,
        what="split_data_group",
    )

    references, aligned_structures = flatten_frames_and_structures(
        frames_by_filename,
        structures_by_filename,
        domain,
    )

    if test_ratio is not None and test_size is not None:
        raise ValueError(
            "Exactly one of test_ratio or test_size must be provided."
        )
    if test_ratio is None and test_size is None:
        test_ratio = DEFAULT_TEST_RATIO

    if test_size is not None:
        if test_size <= 0 or test_size >= len(references):
            raise ValueError(
                f"test_size must be in [1, {len(references) - 1}], got {test_size}."
            )
        # The authoritative common API accepts ratios only. This exact ratio
        # round-trips to the requested integer under its documented rounding.
        test_ratio = test_size / len(references)
        trainval_pool, test_set = partition_trainval_test(
            references,
            seed=split_seed,
            test_ratio=test_ratio,
        )
    else:
        trainval_pool, test_set = partition_trainval_test(
            references,
            seed=split_seed,
            test_ratio=test_ratio,
        )

    if requested_train_sizes is None:
        requested_train_sizes = DEFAULT_TRAIN_RATIOS
        as_ratio = True
    cap = max_train_size if max_train_size is not None else DEFAULT_MAX_N_TRAIN
    sizes = _normalize_requested_train_sizes(
        requested_train_sizes,
        len(trainval_pool),
        as_ratio=as_ratio,
        max_train_size=cap,
    )

    # Align the train+validation structures with the trainval pool, which
    # preserves the canonical reference order (partition_trainval_test keeps
    # the pool order).
    identity_to_structure: Dict[Tuple[str, str, int], Atoms] = {
        ref.identity: atoms
        for ref, atoms in zip(references, aligned_structures)
    }
    trainval_structures = [
        identity_to_structure[ref.identity]
        for ref in trainval_pool
    ]

    # A single QUESTS engine is shared by every selected method. When QUESTS
    # selection is requested, the full train+validation pool descriptors are
    # computed exactly once and reused for the QUESTS selection, the QUESTS
    # profile, and the random trajectory's entropy profile (via pool-index
    # projection), so no descriptor computation is repeated.
    quests_engine = objective if objective is not None else QuestsObjective(quests_config)
    pool_descriptors = None
    if "quests" in methods:
        pool_descriptors = quests_engine.compute_descriptors(trainval_structures)

    schemas: List[SplitDataSchema] = []
    for method in methods:
        if method == "random":
            trajectory = generate_random_trajectory(
                seed=random_seed,
                pool=trainval_pool,
                requested_train_sizes=sizes,
            )
            # The random trajectory gets a real QUESTS entropy profile,
            # computed with the exact same objective used for "quests" and,
            # when available, the shared pool descriptors.
            trajectory = populate_entropy_profile(
                trajectory=trajectory,
                pool=trainval_pool,
                structures=trainval_structures,
                config=quests_config,
                objective=quests_engine,
                descriptors=pool_descriptors,
            )
        else:
            trajectory = generate_quests_trajectory(
                pool=trainval_pool,
                structures=trainval_structures,
                requested_train_sizes=sizes,
                config=quests_config,
                objective=quests_engine,
                descriptors=pool_descriptors,
            )
        schemas.append(SplitDataSchema(
            domain=domain,
            grouping_strategy=grouping_strategy,
            group_name=group_name,
            test_set=test_set,
            test_ratio=test_ratio,
            trainval_test_split_seed=split_seed,
            train_val_split_trajectory=trajectory,
            quests_config=quests_config,
        ))

    return schemas


__all__ = [
    "SplitMethod",
    "flatten_frames_and_structures",
    "split_data_group",
]
