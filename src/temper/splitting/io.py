"""Reconstruct split datasets and export them as deterministic extxyz files.

Persisted split data (see :class:`src.temper.schemas.split.SplitDataSchema`)
stores frame references — ``(domain, relative extxyz source filename,
nonnegative frame index)`` — rather than structures. This module resolves those
references to labeled ``ase.Atoms`` frames, reconstructs the train, validation,
and test datasets, and writes the resulting datasets to extxyz files.

Source paths and caching
------------------------
:class:`SourceResolver` resolves each reference using the path
``root_path / domain / filename``. Resolution includes guards against escaping
both the configured root and the domain directory. Source files are cached per
resolver, so each file is read at most once during that resolver's lifetime.
The returned ``Atoms`` objects and their ``SinglePointCalculator`` instances
may alias cached frames; callers must therefore treat returned frames and
calculators as read-only. Reconstruction validates that energy and forces are
available, while preserving stress and other metadata read from extxyz.

Reconstruction and export
-------------------------
:func:`load_frames_from_references` loads an ordered collection of referenced
frames. :func:`load_frames_test` reconstructs the test set, and
:func:`load_frames_train_validation` reconstructs the training and validation
sets. :func:`build_export_filename` produces the deterministic generated-file
name. :func:`write_single_dataset_to_extxyz` writes a nonempty role dataset to
that filename and atomically replaces an existing generated file.

:func:`write_all_sets_in_split_schema_to_extxyz` writes every training
checkpoint and test set. Validation exports are opt-in via the keyword-only
argument ``write_validation=False``; when enabled, nonempty validation datasets
are exported as well. Its returned mapping always contains ``train``,
``validation``, and ``test`` lists.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

from ase import Atoms
from ase.io import read, write

from src.temper.schemas.split import FrameReference, SplitDataSchema
from src.temper.utils.env import DEFAULT_DATA_DIR

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

class SourceResolver:
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

    def __init__(self, root_path: Path | str = DEFAULT_DATA_DIR) -> None:
        """Bind and validate a source root_path directory.

        Parameters
        ----------
        root_path : Path | str
            Source root_path directory that contains one subdirectory per domain.
            Defaults to ``DEFAULT_DATA_DIR``. See src.temper.utils.env.

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
            self._cache[resolved] = list(read(resolved, index=":"))

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
    resolver: SourceResolver,
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
    resolver : SourceResolver
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
    for reference in references:
        source_path = resolver.resolve_source_path(reference)
        raw_frames = resolver.load_raw_frames(source_path)
        if reference.frame_index >= len(raw_frames):
            raise IndexError(
                f"Frame reference {reference.identity} has frame_index "
                f"{reference.frame_index} but source file {source_path} "
                f"contains only {len(raw_frames)} frames."
            )
        raw = raw_frames[reference.frame_index]
        frames.append(raw)
    return frames

def load_frames_from_references(
    references: List[FrameReference],
    root_path: Path | str = DEFAULT_DATA_DIR,
) -> List[Atoms]:
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
        Defaults to ``DEFAULT_DATA_DIR``. See src.temper.utils.env.

    Returns
    -------
    list[Atoms]
        Independent labeled frames in reference order.

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
    resolver = SourceResolver(root_path)
    return _load_frames_with_resolver(references, resolver)

def load_frames_test(
    schema: SplitDataSchema,
    root_path: Path | str = DEFAULT_DATA_DIR,
) -> List[Atoms]:
    """Reconstruct the labeled test set of a split schema.

    Parameters
    ----------
    schema : SplitDataSchema
        The persisted split result whose
        :attr:`~SplitDataSchema.test_set` is reconstructed.
    root_path : Path | str
        Source root_path directory beneath which each reference's
        ``domain / filename`` is located.
        Defaults to ``DEFAULT_DATA_DIR``. See src.temper.utils.env.

    Returns
    -------
    list[Atoms]
        Independent labeled test frames in schema order.

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
    resolver = SourceResolver(root_path)
    return _load_frames_with_resolver(schema.test_set, resolver)

def load_frames_train_validation(
    schema: SplitDataSchema,
    requested_size_index: int,
    root_path: Path | str = DEFAULT_DATA_DIR,
) -> Tuple[List[Atoms], List[Atoms]]:
    """Reconstruct the training and validation sets at a requested size.

    The training set is the prefix of the trajectory's
    :attr:`TrainValSplitTrajectory.selected_frames` of length ``requested_size``;
    validation is the remaining selected suffix followed by
    ``additional_trainval_frames``. The requested size is mapped to a checkpoint
    index before the trajectory accessors are called. Both sets share one
    resolver, so each source file is read at most once during the operation.

    Parameters
    ----------
    schema : SplitDataSchema
        The persisted split result.
    requested_size_index : int
        Index among the trajectory's requested training sizes.
    root_path : Path | str
        Source root_path directory beneath which each reference's
        ``domain / filename`` resolves.
        Defaults to ``DEFAULT_DATA_DIR``. See src.temper.utils.env.

    Returns
    -------
    tuple[list[Atoms], list[Atoms]]
        ``(train, validation)`` labeled frames; frames may alias the resolver's
        cached source objects and must be treated as read-only.

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
    resolver = SourceResolver(root_path)
    train = _load_frames_with_resolver(train_references, resolver)
    validation = _load_frames_with_resolver(validation_references, resolver)
    return train, validation

######
# Utilities to export files.
######

def build_export_filename(
    *,
    domain: str,
    group_name: str,
    grouping_strategy: str | None,
    method: str,
    role: str,
    structure_count: int,
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
        ]
    )
    return f"{stem}.extxyz"

def _write_extxyz_atomic(
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
        write(tmp_path, atoms_list, format="extxyz")
        os.replace(tmp_path, dest_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

def write_single_dataset_to_extxyz(
    atoms_list: List[Atoms],
    *,
    domain: str,
    group_name: str,
    grouping_strategy: str | None,
    method: str,
    role: str,
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
    )
    target = Path(output_dir) / filename
    _write_extxyz_atomic(target, atoms_list)
    return target

def write_all_sets_in_split_schema_to_extxyz(
    schema: SplitDataSchema,
    output_dir: Path | str,
    root_path: Path | str = DEFAULT_DATA_DIR,
    *,
    write_validation: bool = False,
) -> Dict[str, List[Path]]:
    """Export training and testing sets, optionally including validation sets.

    Parameters
    ----------
    schema : SplitDataSchema
        The split schema to export.
    output_dir : Path | str
        Output directory; created if missing. Existing generated artifacts are
        replaced atomically.
    root_path : Path | str
        Source root_path directory beneath which each reference's
        ``domain / filename`` resolves.
        Defaults to ``DEFAULT_DATA_DIR``. See src.temper.utils.env.
    write_validation : bool, optional
        Whether to export non-empty validation sets at every checkpoint.
        Defaults to ``False``. The returned mapping always contains a
        ``"validation"`` key; it is empty when validation export is disabled
        or every validation set is empty.

    Returns
    -------
    dict[str, List[Path]]
        A mapping from dataset roles to the written file paths. Training and
        test paths are always included; validation paths are included only
        when ``write_validation=True``.
    """
    written_files = {
        "train": [],
        "validation": [],
        "test": [],
    }
    # Write training sets and optionally validation sets.
    for i in range(len(schema.train_val_split_trajectory.requested_train_sizes)):
        atoms_train, atoms_val = load_frames_train_validation(schema, i, root_path)
        written_files["train"].append(write_single_dataset_to_extxyz(
            atoms_list=atoms_train,
            domain=schema.domain,
            group_name=schema.group_name,
            grouping_strategy=schema.grouping_strategy,
            method=schema.train_val_split_trajectory.method,
            role="train",
            output_dir=output_dir,
        ))
        if write_validation and atoms_val:
            written_files["validation"].append(write_single_dataset_to_extxyz(
                atoms_list=atoms_val,
                domain=schema.domain,
                group_name=schema.group_name,
                grouping_strategy=schema.grouping_strategy,
                method=schema.train_val_split_trajectory.method,
                role="validation",
                output_dir=output_dir,
            ))
    # Write the testing set.
    written_files["test"].append(write_single_dataset_to_extxyz(
        atoms_list=load_frames_test(schema, root_path),
        domain=schema.domain,
        group_name=schema.group_name,
        grouping_strategy=schema.grouping_strategy,
        method=schema.train_val_split_trajectory.method,
        role="test",
        output_dir=output_dir,
    ))
    return written_files

__all__ = [
    "SourceResolver",
    "build_export_filename",
    "load_frames_from_references",
    "load_frames_test",
    "load_frames_train_validation",
    "write_single_dataset_to_extxyz",
    "write_all_sets_in_split_schema_to_extxyz",
]
