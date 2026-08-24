"""Resolves persisted frame references to source extxyz data, reconstructs split datasets, and exports them deterministically."""
from __future__ import annotations

import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Tuple

from ase import Atoms
from ase.io import read, write

from temper.schemas.split import SplitGroup
from temper.schemas.frame_refrence import FrameReference
from temper.schemas.train_unit import TrainingUnit
from temper.utils.defaults import (
    DEFAULT_SPLIT_RESULTS_DIR,
    DEFAULT_DATA_DIR,
)
from temper.logging import format_elapsed, progress_task


logger = logging.getLogger(__name__)


#: Dataset roles accepted in export filenames.
_ROLES: Tuple[str, ...] = ("train", "validation", "test")

#: Characters kept verbatim when sanitizing identity components for filenames.
_SAFE_FILENAME_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789._-"
)

def _sanitize_component(value: str) -> str:
    """Deterministically sanitize a string into a safe filename component.

    Every character that is not ASCII alphanumeric, ``.``, ``_``, or ``-`` is
    replaced with ``_``. Surrounding whitespace is stripped. An empty or
    fully-unsafe input yields ``"unnamed"`` so a component never vanishes.

    Parameters
    ----------
    value : str
        The raw identity component (e.g. a domain or group name).

    Returns
    -------
    str
        A deterministic, filesystem-safe component.
    """
    sanitized = "".join(
        ch if ch in _SAFE_FILENAME_CHARS else "_"
        for ch in value.strip()
    )
    return sanitized if sanitized else "unnamed"


class FrameReferenceResolver:
    """Resolve frame references to source extxyz files and cache their frames as List of ase.Atoms.

    Binds a single source root_path directory and provides two services:

    - Escape-guarded path resolution: each :class:`FrameReference` resolves to
      ``root_path / domain / filename`` and is verified to stay within the
      configured root_path and domain directory. This is defense in depth on top of
      the schema-level path validation already applied to
      :attr:`FrameReference.filename`.
    - Per-file caching: each source extxyz file is read at most once per
      resolver lifetime, so repeated references into the same file reuse the
      same in-memory frame list.
    """

    def __init__(self, root_path: Path | str) -> None:
        """Bind and validate a source root_path directory.

        Parameters
        ----------
        root_path : Path | str
            Source root_path directory that contains one subdirectory per domain.
            Defaults to ``DEFAULT_DATA_DIR``. See src.temper.utils.defaults.
            We recommend expanding user and resolving the path before passing its
            value in.
            Always expanded and resolved by this constructor.

        Raises
        ------
        NotADirectoryError
            If ``root_path`` does not exist or is not a directory.
        """
        self._root_path: Path = Path(root_path).expanduser().resolve()
        if not self._root_path.is_dir():
            raise NotADirectoryError(
                f"Source root_path must be an existing directory, got: {self._root_path}"
            )
        #: Cache of raw frame lists keyed by resolved source file path.
        self._cache: Dict[Path, List[Atoms]] = {}

    @property
    def root_path(self) -> Path:
        """The normalized source root_path directory."""
        return self._root_path

    def resolve_source_path(self, reference: FrameReference) -> Path:
        """Resolve a frame reference to its absolute source extxyz path.

        The domain must be a single, safe path component (no separators, no
        traversal, not absolute); the filename has already been validated as a
        relative, non-traversing ``.extxyz`` path by
        :class:`FrameReference`. The candidate path is resolved and verified to
        remain inside the configured domain directory.

        Parameters
        ----------
        reference : FrameReference
            The frame reference to resolve.

        Returns
        -------
        Path
            The resolved absolute source extxyz file path.

        Raises
        ------
        ValueError
            If the domain is not a single safe path component, or if the
            resolved path escapes the configured root_path/domain boundary.
        """
        domain = reference.domain
        if (
            not domain
            or domain in (".", "..")
            or "/" in domain
            or "\\" in domain
            or Path(domain).is_absolute()
            or Path(domain).root
        ):
            raise ValueError(
                f"Reference domain {domain!r} must be a single safe path "
                "component (no separators, no traversal)."
            )

        domain_dir = (self._root_path / domain).resolve()
        candidate = (domain_dir / reference.filename).resolve()
        if not candidate.is_relative_to(domain_dir):
            raise ValueError(
                f"Resolved source path {candidate} escapes the configured "
                f"domain directory {domain_dir} for reference "
                f"{reference.identity}."
            )
        return candidate

    def load_raw_frames(self, path: Path) -> List[Atoms]:
        """Return the cached raw frames of a source file, reading it at most once.

        The file is read with ``ase.io.read(path, index=":")`` on first access
        and the resulting frame list is cached for the lifetime of the
        resolver. The resolved path must stay within the configured source
        root_path.

        Parameters
        ----------
        path : Path
            The source file path, typically produced by
            :meth:`resolve_source_path`.

        Returns
        -------
        list[Atoms]
            The raw frames of the file in file order.

        Raises
        ------
        ValueError
            If ``path`` resolves outside the configured source root_path.
        FileNotFoundError
            If the source file does not exist.
        """
        resolved = Path(path).resolve()
        if not resolved.is_relative_to(self._root_path):
            raise ValueError(
                f"Source path {resolved} escapes the configured source root_path "
                f"{self._root_path}."
            )
        if resolved not in self._cache:
            if not resolved.is_file():
                raise FileNotFoundError(
                    f"Source extxyz file does not exist: {resolved}"
                )
            logger.debug("Reading source extxyz file %s.", resolved)
            read_started_at = time.monotonic()
            with progress_task(
                logger,
                f"Reading source file {resolved.name!r}",
                unit="frames",
            ) as progress:
                self._cache[resolved] = list(read(resolved, index=":"))
                progress.update(
                    completed=len(self._cache[resolved]),
                    detail=f"source {resolved.parent.name}/{resolved.name}",
                )
            logger.debug(
                "Read %d frame(s) from %s in %s.",
                len(self._cache[resolved]),
                resolved,
                format_elapsed(time.monotonic() - read_started_at),
            )

            # Check that all frames have energy and forces labels.
            try:
                for frame in self._cache[resolved]:
                    _ = frame.get_potential_energy()
                    _ = frame.get_forces()
            except Exception as e:
                raise ValueError(
                    f"Source file {resolved} contains a frame with missing "
                    "energy or forces labels and thus cannot be used to train or test"
                    " MLFF."
                ) from e

        return self._cache[resolved]


def _load_frames_with_resolver(
    references: List[FrameReference],
    resolver: FrameReferenceResolver,
) -> List[Atoms]:
    """Reconstruct frames for ordered references using a shared resolver.

    Each reference is resolved and bounds-checked against its source file,
    preserving reference order. Using a single resolver keeps the per-file read
    cache shared across all references of a reconstruction operation; returned
    objects may alias when references point to the same cached source frame.

    Parameters
    ----------
    references : List[FrameReference]
        Ordered frame references to reconstruct.
    resolver : FrameReferenceResolver
        The resolver providing safe path resolution and per-file caching.

    Returns
    -------
    list[Atoms]
        Labeled frames in reference order; repeated references may alias cached
        source objects.

    Raises
    ------
    FileNotFoundError
        If a referenced source file does not exist.
    IndexError
        If a reference's ``frame_index`` is out of range for its source file.
    ValueError
        If a referenced frame has no energy or forces labels.
    """
    frames: List[Atoms] = []
    update_interval = max(1, len(references) // 100)
    with progress_task(
        logger,
        "Resolving frame references",
        total=len(references),
        unit="frames",
    ) as progress:
        for index, reference in enumerate(references):
            source_path = resolver.resolve_source_path(reference)
            if index % update_interval == 0:
                progress.update(
                    completed=index,
                    detail=f"source {reference.domain}/{reference.filename}",
                )
            raw_frames = resolver.load_raw_frames(source_path)
            if reference.frame_index >= len(raw_frames):
                raise IndexError(
                    f"Frame reference {reference.identity} has frame_index "
                    f"{reference.frame_index} but source file {source_path} "
                    f"contains only {len(raw_frames)} frames."
                )
            raw = raw_frames[reference.frame_index]
            frames.append(raw)
        progress.update(completed=len(references))
    return frames


def load_frames_from_references(
    references: List[FrameReference],
    root_path: Path | str,
    resolver: FrameReferenceResolver | None = None,
) -> Tuple[List[Atoms], FrameReferenceResolver]:
    """Reconstruct independent labeled frames for ordered references.

    Each reference is resolved to its source extxyz file under ``root_path`` (see
    the module's source-root_path convention), bounds-checked against the file,
    and returned in reference order. All references share one resolver, so each
    source file is read at most once during the operation; returned objects may
    alias cached source frames and must be treated as read-only.

    Parameters
    ----------
    references : List[FrameReference]
        Ordered frame references to reconstruct.
    root_path : Path | str
        Source root_path directory beneath which each reference's
        ``domain / filename`` is located.
    resolver: FrameReferenceResolver | None
        Optional resolver to use instead of the default. Allows for
        shared cache reuse. Will not reuse if not provided or its
        ``root_path`` attribute does not match the provided ``root_path``.

    Returns
    -------
    Tuple[list[Atoms], FrameReferenceResolver]
        Independent labeled frames in reference order, and the resolver used to
        reconstruct them.

    Raises
    ------
    NotADirectoryError
        If ``root_path`` is not an existing directory.
    FileNotFoundError
        If a referenced source file does not exist.
    IndexError
        If a reference's ``frame_index`` is out of range for its source file.
    ValueError
        If a referenced frame has no energy or forces labels, or if a resolved
        path would escape the configured root_path/domain.
    """
    # Note: enforce identical convention of root_path with resolver.root_path.
    # This ensures that the resolver's cache can be reused given the same
    # root_path but with different conventions.
    root_path = Path(root_path).expanduser().resolve()
    if resolver is None or resolver.root_path != root_path:
        resolver = FrameReferenceResolver(root_path)
    return _load_frames_with_resolver(references, resolver), resolver


def load_frames_test(
    schema: SplitGroup,
    root_path: Path | str,
    resolver: FrameReferenceResolver | None = None,
) -> Tuple[List[Atoms], FrameReferenceResolver]:
    """Reconstruct the labeled test set of a split schema.

    Parameters
    ----------
    schema : SplitGroup
        The persisted split result whose
        :attr:`~SplitGroup.test_set` is reconstructed.
    root_path : Path | str
        Source root_path directory beneath which each reference's
        ``domain / filename`` is located.
    resolver: FrameReferenceResolver | None
        Optional resolver to use instead of the default. Allows for
        shared cache reuse. Will not reuse if not provided or its
        ``root_path`` attribute does not match the provided ``root_path``
        after expanduser and resolve.

    Returns
    -------
    Tuple[list[Atoms], FrameReferenceResolver]
        Independent labeled test frames in schema order, and the resolver used to
        reconstruct them.

    Raises
    ------
    NotADirectoryError
        If ``root_path`` is not an existing directory.
    FileNotFoundError
        If a referenced source file does not exist.
    IndexError
        If a reference's ``frame_index`` is out of range.
    ValueError
        If a referenced frame has no energy or forces labels.
    """
    root_path = Path(root_path).expanduser().resolve()
    if resolver is None or resolver.root_path != root_path:
        resolver = FrameReferenceResolver(root_path)
    return _load_frames_with_resolver(schema.test_set, resolver), resolver


def load_frames_train_validation(
    schema: SplitGroup,
    requested_size_index: int,
    root_path: Path | str,
    resolver: FrameReferenceResolver | None = None,
) -> Tuple[List[Atoms], List[Atoms], FrameReferenceResolver]:
    """Reconstruct the training and validation sets at a requested size.

    The training set is the prefix of the trajectory's
    :attr:`TrainValSplitTrajectory.selected_frames` of length ``requested_size``;
    validation is the remaining selected suffix followed by
    ``additional_trainval_frames``. The requested size is mapped to a checkpoint
    index before the trajectory accessors are called. Both sets share one
    resolver, so each source file is read at most once during the operation.

    Parameters
    ----------
    schema : SplitGroup
        The persisted split result.
    requested_size_index : int
        Index among the trajectory's requested training sizes.
    root_path : Path | str
        Source root_path directory beneath which each reference's
        ``domain / filename`` resolves.
        Defaults to ``DEFAULT_DATA_DIR``. See src.temper.utils.defaults.
    resolver: FrameReferenceResolver | None
        Optional resolver to use instead of the default. Allows for
        shared cache reuse. Will not reuse if not provided or its
        ``root_path`` attribute does not match the provided ``root_path``
        after expanduser and resolve.

    Returns
    -------
    tuple[list[Atoms], list[Atoms], FrameReferenceResolver]
        ``(train, validation)`` labeled frames; frames may alias the resolver's
        cached source objects and must be treated as read-only; followed by
        the resolver used to reconstruct them.

    Raises
    ------
    ValueError
        If ``method`` is unsupported, not represented by the schema, or
        ambiguous, or if ``requested_size`` is not a requested training size.
    NotADirectoryError
        If ``root_path`` is not an existing directory.
    FileNotFoundError
        If a referenced source file does not exist.
    IndexError
        If a reference's ``frame_index`` is out of range.
    """
    trajectory = schema.train_val_split_trajectory
    train_references = trajectory.get_train_set(requested_size_index)
    validation_references = trajectory.get_val_set(requested_size_index)
    root_path = Path(root_path).expanduser().resolve()
    if resolver is None or resolver.root_path != root_path:
        resolver = FrameReferenceResolver(root_path)
    train = _load_frames_with_resolver(train_references, resolver)
    validation = _load_frames_with_resolver(validation_references, resolver)
    return train, validation, resolver

######
# Utilities to export files.
######

def build_export_filename(
    domain: str,
    group_name: str,
    grouping_strategy: str | None,
    method: str,
    role: str,
    structure_count: int,
    repeat_id: int,
) -> str:
    """Build the deterministic, safe filename for an exported extxyz file.

    The filename has the form::

        <domain>__<strategy>__<group_name>__<method>__<role>__n<count>.extxyz

    Each identity component is sanitized deterministically (see
    :func:`_sanitize_component`); ``role`` is one of ``"train"``,
    ``"validation"``, or ``"test"``; and ``count`` is the number of structures
    the file contains. For the ``"train"`` role the structure count equals the
    training size. When ``grouping_strategy`` is None or empty it is replaced
    by ``"unknown"``.

    Parameters
    ----------
    domain : str
        Data domain name.
    group_name : str
        Group name.
    grouping_strategy : str | None
        Grouping strategy name, or None/empty if not represented by the schema.
    method : str
        Splitting method name.
    role : str
        Dataset role, one of ``"train"``, ``"validation"``, ``"test"``.
    structure_count : int
        Number of structures the file contains.
    repeat_id: int
        The id among repeated splits.

    Returns
    -------
    str
        The deterministic export filename including the ``.extxyz`` suffix.

    Raises
    ------
    ValueError
        If ``role`` is not one of the supported roles or if
        ``structure_count`` is negative.
    """
    if role not in _ROLES:
        raise ValueError(
            f"role must be one of {list(_ROLES)}, got {role!r}."
        )
    if structure_count < 0:
        raise ValueError(
            f"structure_count must be nonnegative, got {structure_count}."
        )
    strategy = grouping_strategy if grouping_strategy else "unknown_grouping"
    stem = "__".join(
        [
            _sanitize_component(domain),
            _sanitize_component(strategy),
            _sanitize_component(group_name),
            _sanitize_component(method),
            role,
            f"n{structure_count}",
            f"repeat{repeat_id}"
        ]
    )
    return f"{stem}.extxyz"


def _write_atoms_list_to_extxyz(
    dest_path: Path | str,
    atoms_list: List[Atoms],
) -> None:
    """Atomically write ``atoms_list`` to ``dest_path`` as a single extxyz file.

    Frames are written to a temporary sibling file and then published with
    :func:`os.replace`, so an interrupted write does not leave a partial target.
    Existing targets are intentionally replaced; export callers should treat
    the output directory as a generated-artifact directory.
    """
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=dest_path.parent,
        prefix=f".{dest_path.name}.",
        suffix=".tmp",
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        logger.debug(
            "Writing %d frame(s) to generated extxyz artifact %s.",
            len(atoms_list),
            dest_path,
        )
        write_started_at = time.monotonic()
        with progress_task(
            logger,
            f"Writing dataset {dest_path.name!r}",
            total=len(atoms_list),
            unit="frames",
        ) as progress:
            write(tmp_path, atoms_list, format="extxyz")
            progress.update(completed=len(atoms_list))
        os.replace(tmp_path, dest_path)
        logger.debug(
            "Published generated extxyz artifact %s in %s.",
            dest_path,
            format_elapsed(time.monotonic() - write_started_at),
        )
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def write_atoms_list_to_extxyz(
    atoms_list: List[Atoms],
    domain: str,
    group_name: str,
    grouping_strategy: str | None,
    method: str,
    role: str,
    repeat_id: int,
    output_dir: Path | str,
) -> Path:
    """Write a labeled frame list to one deterministic extxyz file.

    Builds the deterministic filename (see :func:`build_export_filename`) and
    atomically replaces the generated artifact in ``output_dir``.

    Parameters
    ----------
    atoms_list : list[Atoms]
        Independent labeled frames to export. Must not be empty.
    domain : str
        Data domain name (filename identity).
    group_name : str
        Group name (filename identity).
    grouping_strategy : str | None
        Grouping strategy name (filename identity).
    method : str
        Splitting method name (filename identity).
    role : str
        Dataset role (filename identity), ``'train'``, ``'validation'`` or ``'test'``.
    repeat_id: int
        Index among repeated splits for the same group.
    output_dir : Path | str
        Output directory; created if missing. Existing generated artifacts are
        replaced atomically.

    Returns
    -------
    Path
        The written file path.

    Raises
    ------
    ValueError
        If ``atoms_list`` is empty.
    """
    if not atoms_list:
        raise ValueError(
            f"Cannot export an empty {role} set: ASE extxyz semantics for an "
            "empty frame list are undefined."
        )
    filename = build_export_filename(
        domain=domain,
        group_name=group_name,
        grouping_strategy=grouping_strategy,
        method=method,
        role=role,
        structure_count=len(atoms_list),
        repeat_id=repeat_id
    )
    target = Path(output_dir) / filename
    _write_atoms_list_to_extxyz(target, atoms_list)
    return target


def write_all_sets_in_split_group_to_extxyz(
    split_group: SplitGroup,
    root_path: Path | str = DEFAULT_DATA_DIR,
    output_path: Path | str = DEFAULT_SPLIT_RESULTS_DIR,
    write_validation: bool = False,
    write_extra_tests: bool = True,
    all_split_groups: List[SplitGroup] | None = None,
    resolver: FrameReferenceResolver | None = None,
) -> Tuple[List[TrainingUnit], FrameReferenceResolver]:
    """Export training and testing sets, optionally including validation sets.

    Extra testing sets from other groups will also be written if required.

    Parameters
    ----------
    split_group : SplitGroup
        The split group to export.
    root_path : Path | str
        Source root_path directory beneath which each reference's
        ``domain / filename`` resolves.
        Defaults to ``DEFAULT_DATA_DIR``. See src.temper.utils.defaults.
    output_path : Path | str
        Output directory; created if missing. Existing generated artifacts are
        replaced atomically. Defaults to ``DEFAULT_SPLIT_RESULTS_DIR``.
        See src.temper.utils.defaults.
        Files will be written into each domain subfolder under ``output_path``.
    write_validation : bool, optional
        Whether to export non-empty validation sets at every checkpoint.
        Defaults to ``False``. The returned mapping always contains a
        ``"validation"`` key; it is empty when validation export is disabled
        or every validation set is empty.
    write_extra_tests : bool, optional
        Whether to export extra testing sets from other groups. Defaults to
        ``True``. All extra tests are written to the ``"extra_tests"`` subfolder
        under ``output_dir``.
    all_split_groups : list[SplitGroup], optional
        All split groups in the same domain. Required if ``write_extra_tests``
        is ``True``. Used to retrieve data of extra tests from.
    resolver: FrameReferenceResolver | None
        Optional resolver to use instead of the default. Allows for
        shared cache reuse. Will not reuse if not provided or its
        ``root_path`` attribute does not match the provided ``root_path``
        after expanduser and resolve.

    Returns
    -------
    List[TrainingUnit], FrameReferenceResolver
        A list of training unit objects, each containing the path to the training
        set, the path to the validation set (if any), and the path to the test
        set within a unitary training process, for future reference.
        Then returns the resolver used to load
        the frames, whose internal cache might be reused to save loading time.

    Raises
    ------
    ValueError
        If ``write_extra_tests`` is ``True`` but ``all_split_groups`` is
        ``None``.
    """
    root_path = Path(root_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()

    if write_extra_tests and (all_split_groups is None):
        raise ValueError(
            "all_split_groups must be provided if write_extra_tests is True."
        )

    train_files: List[str] = []
    val_files: List[str | None] = []
    test_files: List[str] = []

    # Write training sets and optionally validation sets.
    for i in range(len(split_group.train_val_split_trajectory.requested_train_sizes)):
        atoms_train, atoms_val, resolver = load_frames_train_validation(
            split_group, i, root_path=root_path, resolver=resolver
        )
        train_files.append(write_atoms_list_to_extxyz(
            atoms_list=atoms_train,
            domain=split_group.domain,
            group_name=split_group.group_name,
            grouping_strategy=split_group.grouping_strategy,
            method=split_group.train_val_split_trajectory.method,
            role="train",
            repeat_id=split_group.repeat_id,
            output_dir=output_path / split_group.domain,
        ).name)
        val_file = None
        if write_validation and atoms_val:
            val_file = write_atoms_list_to_extxyz(
                atoms_list=atoms_val,
                domain=split_group.domain,
                group_name=split_group.group_name,
                grouping_strategy=split_group.grouping_strategy,
                method=split_group.train_val_split_trajectory.method,
                role="validation",
                repeat_id=split_group.repeat_id,
                output_dir=output_path / split_group.domain,
            ).name
        val_files.append(val_file)
    # Write the testing set of the current group.
    atoms_test, resolver = load_frames_test(
        split_group, root_path, resolver=resolver
    )
    test_files.append(write_atoms_list_to_extxyz(
        atoms_list=atoms_test,
        domain=split_group.domain,
        group_name=split_group.group_name,
        grouping_strategy=split_group.grouping_strategy,
        method=split_group.train_val_split_trajectory.method,
        role="test",
        repeat_id=split_group.repeat_id,
        output_dir=output_path / split_group.domain,
    ).name)
    # Write extra tests. Since extra test files are already written, we only need to
    # write the paths to them into TrainingUnit objects.
    if write_extra_tests:
        # Find groups by matching domain, grouping strategy, group name and repeat_id.
        all_split_groups: List[SplitGroup] = [] if all_split_groups is None else all_split_groups
        for other_group in all_split_groups:
            if (
                other_group.domain == split_group.domain
                and other_group.grouping_strategy == split_group.grouping_strategy
                and other_group.group_name in split_group.extra_tested_groups
                and other_group.repeat_id == split_group.repeat_id
            ):
                filename = build_export_filename(
                    domain=other_group.domain,
                    group_name=other_group.group_name,
                    grouping_strategy=other_group.grouping_strategy,
                    method=other_group.train_val_split_trajectory.method,
                    role="test",
                    structure_count=len(other_group.test_set),
                    repeat_id=other_group.repeat_id,
                )
                filepath = (output_path / other_group.domain / filename).resolve()
                if not filepath.exists():  # Other group's file not yet written.
                    atoms_test, resolver = load_frames_test(
                        other_group, root_path, resolver=resolver
                    )
                    test_files.append(write_atoms_list_to_extxyz(
                        atoms_list=atoms_test,
                        domain=other_group.domain,
                        group_name=other_group.group_name,
                        grouping_strategy=other_group.grouping_strategy,
                        method=other_group.train_val_split_trajectory.method,
                        role="test",
                        repeat_id=other_group.repeat_id,
                        output_dir=output_path / other_group.domain,
                    ).name)
                else:  # Already written, just record file name.
                    test_files.append(filename)
    # Build TrainingUnit objects. All units share the same test sets.
    training_units: List[TrainingUnit] = [
        TrainingUnit(
            train_set=train_files[i],
            val_set=val_files[i],
            test_sets=test_files,
            root_path=output_path,
            domain=split_group.domain,
            group_name=split_group.group_name,
            grouping_strategy=split_group.grouping_strategy,
            method=split_group.train_val_split_trajectory.method,
            repeat_id=split_group.repeat_id,
            n_train=split_group.train_val_split_trajectory.requested_train_sizes[i],
            split_id=split_group.split_id,
        )
        for i in range(len(train_files))
    ]

    return training_units, resolver



__all__ = [
    "FrameReferenceResolver",
    "build_export_filename",
    "load_frames_from_references",
    "load_frames_test",
    "load_frames_train_validation",
    "write_atoms_list_to_extxyz",
    "write_all_sets_in_split_group_to_extxyz",
]
