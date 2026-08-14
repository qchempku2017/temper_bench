"""QUESTS maximum-information-entropy splitting for MLFF datasets.

This module implements the QUESTS (Quick Uncertainty and Entropy from
STructural Similarity) entropy backend and the maximum-information-entropy
trajectory for the temper splitter. It builds on the shared :class:`src.temper.schemas.split.TrainValSplitTrajectory` output convention already

used by the random splitter and stores its results through the
:class:`src.temper.schemas.split.EntropyProfile` schema.

Verified QUESTS API (``quests==2026.2.22``, installed in the project virtual
environment; see ``requirements.txt``):

- ``quests.descriptor.get_descriptors(dset, k, cutoff, concat, dtype)``:
  computes per-atom descriptors and, with ``concat=True``, returns a single
  ``(total_atoms, 2*k - 1)`` float matrix (per-atom rows flattened across all
  structures, with no frame boundaries). CPU-only (numba) and never imports
  torch.
- ``quests.entropy.entropy(x, h, batch_size)``: scalar QUESTS entropy of the
  ``(N, d)`` descriptor matrix ``x`` (CPU, numba).
- ``quests.entropy.delta_entropy(y, x, h, batch_size)``: per-row differential
  entropy ``-log(p_y)`` of each test row ``y`` against the reference rows
  ``x`` (CPU, numba). The first argument is the *test/candidate* set and the
  second is the *reference* set.
- ``quests.gpu.entropy.entropy`` / ``quests.gpu.entropy.delta_entropy``:
  torch-based equivalents that take ``torch.Tensor`` inputs plus a ``device``
  string, with the same (candidate, reference) argument order. Importing
  ``quests.gpu.entropy`` imports torch, so this module is only ever imported
  lazily by the GPU route.

Design:

- **Explicit backend selection**: :class:`QuestsObjective` resolves the
  configured ``device`` ("cpu"/"gpu"/"auto") into a concrete route with
  :meth:`QuestsObjective.resolve_device`. The CPU route never imports or
  initializes CUDA/torch; an explicitly requested but unavailable GPU raises
  :class:`QuestsUnavailableError`; ``"auto"`` falls back to the CPU route
  only when CUDA is unavailable (a documented fallback).
- **One entropy objective**: the same ``entropy``/``delta_entropy`` objective
  is used to evaluate the entropy profile of *any* trajectory (random or
  QUESTS) and to drive the greedy QUESTS selection. No unrelated entropy
  proxy is invented.
- **Descriptor chunking and per-frame slices**: descriptors are computed once
  for the full pool with :class:`FrameDescriptors`, which stores the
  concatenated per-atom matrix and per-frame row offsets so any subset of
  frames can be sliced/concatenated cheaply.
- **Nested maximum-entropy selection**: :func:`generate_quests_trajectory`
  selects frames one at a time. Each candidate frame is scored by the sum of
  the per-atom differential entropy of its descriptors given the descriptors
  selected so far (``delta_entropy``); the first frame is scored by its own
  entropy (the empty-reference marginal information gain). Ties are broken by
  pool order, so the selection is deterministic and reproducible without a
  random seed.
- **Entropy profile**: each requested training size is associated with the
  cumulative QUESTS entropy of the selected prefix and the chunk information
  gain of the added chunk. The chunk gain uses the same ``delta_entropy``
  objective as selection: it is the sum over the chunk's atoms of the
  differential entropy of those atoms given the previously selected
  descriptors (for the first chunk, which has no reference set, it is the
  QUESTS entropy of the chunk itself). Verified against the real backend:
  cumulative entropy is not guaranteed to be non-decreasing (the objective
  normalizes by the set size) and the chunk gain is a differential entropy
  (``-log(p)`` with an unnormalized ``p``) that is not bounded below by zero,
  so the profile schema enforces only finiteness, strictly increasing sizes,
  and cumulative-entropy nonnegativity within ``config.entropy_tolerance``.
  The QUESTS backend can also produce non-finite values for degenerate
  kernels, so all backend output is checked for finiteness (raised as
  :class:`QuestsNumericalError`).

Structures are supplied as ASE ``Atoms`` objects aligned positionally with the
corresponding :class:`FrameReference` pool (``pool[i]`` refers to
``structures[i]``); only frame references are ever stored.
"""
# TODO: need further human review.
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Sequence, Tuple

import numpy as np
from ase import Atoms

from src.temper.schemas.split import (
    EntropyProfile,
    EntropyProfilePoint,
    FrameReference,
    QuestsSplitConfig,
    TrainValSplitTrajectory,
)


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


class QuestsUnavailableError(RuntimeError):
    """Raised when a requested QUESTS backend route is not available.

    Raised when the GPU route is explicitly requested but torch is not
    installed or no CUDA device is available, or when an ``"auto"`` route
    cannot fall back. The CPU route never raises this error.
    """


class QuestsNumericalError(RuntimeError):
    """Raised when the QUESTS backend returns invalid numerical output.

    Raised when ``entropy``/``delta_entropy`` return non-finite values
    (``NaN``/``inf``), which the QUESTS kernel can produce when the bandwidth
    is too small relative to the descriptor distances, or when a backend
    result has an unexpected shape.
    """


@dataclass(frozen=True, eq=False)
class FrameDescriptors:
    """Per-frame slices of concatenated per-atom QUESTS descriptors.

    Stores the concatenated per-atom descriptor matrix of a sequence of
    structures together with the row offsets of each structure, so any frame
    or any ordered subset of frames can be extracted as a contiguous
    descriptor matrix without re-running the descriptor computation.

    Attributes:
        values (np.ndarray): Concatenated per-atom descriptor matrix of shape
            ``(total_atoms, n_dims)``; frame ``i`` occupies rows
            ``frame_offsets[i]:frame_offsets[i + 1]``.
        frame_offsets (tuple[int, ...]): ``n_frames + 1`` non-decreasing row
            offsets; ``frame_offsets[0] == 0`` and
            ``frame_offsets[n_frames] == values.shape[0]``.
    """

    values: np.ndarray
    frame_offsets: Tuple[int, ...]

    def __post_init__(self) -> None:
        """Validate the descriptor matrix and the frame offsets."""
        if self.values.ndim != 2:
            raise ValueError(
                "FrameDescriptors values must be a 2D matrix, got "
                f"shape {self.values.shape}."
            )
        if len(self.frame_offsets) < 2:
            raise ValueError(
                "frame_offsets must contain at least 2 entries (one frame)."
            )
        offsets = list(self.frame_offsets)
        for i in range(1, len(offsets)):
            if offsets[i] <= offsets[i - 1]:
                raise ValueError(
                    "frame_offsets must be strictly increasing; "
                    f"entry {i} is {offsets[i]} after {offsets[i - 1]}."
                )
        if offsets[0] != 0:
            raise ValueError(
                f"frame_offsets must start at 0, got {offsets[0]}."
            )
        if offsets[-1] != self.values.shape[0]:
            raise ValueError(
                "The last frame offset must equal the number of descriptor "
                f"rows; got {offsets[-1]} vs {self.values.shape[0]}."
            )

    @property
    def n_frames(self) -> int:
        """Number of frames in this descriptor set."""
        return len(self.frame_offsets) - 1

    @property
    def n_dims(self) -> int:
        """Number of descriptor dimensions per atom."""
        return self.values.shape[1]

    @property
    def frame_atom_counts(self) -> Tuple[int, ...]:
        """Number of atoms in each frame, in frame order."""
        return tuple(
            self.frame_offsets[i + 1] - self.frame_offsets[i]
            for i in range(self.n_frames)
        )

    def _require_valid_index(self, frame_index: int) -> None:
        """Raise unless ``frame_index`` is a valid frame index."""
        if frame_index < 0 or frame_index >= self.n_frames:
            raise IndexError(
                f"frame_index {frame_index} is out of range for "
                f"{self.n_frames} frames."
            )

    def slice(self, frame_index: int) -> np.ndarray:
        """Return the descriptor rows of a single frame.

        Parameters
        ----------
        frame_index : int
            Zero-based frame index.

        Returns
        -------
        np.ndarray
            Per-atom descriptor matrix of shape ``(n_atoms, n_dims)`` for the
            frame.
        """
        self._require_valid_index(frame_index)
        return self.values[
            self.frame_offsets[frame_index] : self.frame_offsets[frame_index + 1]
        ]

    def concat(self, frame_indices: Sequence[int]) -> np.ndarray:
        """Return the concatenated descriptor rows of an ordered subset.

        Parameters
        ----------
        frame_indices : Sequence[int]
            Ordered frame indices; the result concatenates their per-atom
            descriptor rows in this order.

        Returns
        -------
        np.ndarray
            Per-atom descriptor matrix of shape
            ``(sum of atoms, n_dims)``.

        Raises
        ------
        ValueError
            If ``frame_indices`` is empty.
        """
        indices = list(frame_indices)
        if not indices:
            raise ValueError("concat requires at least one frame index.")
        for index in indices:
            self._require_valid_index(index)
        return np.concatenate([self.slice(index) for index in indices], axis=0)


class QuestsObjective:
    """Explicit CPU/GPU adapter for the QUESTS entropy objective.

    Wraps the verified ``quests`` import surface behind typed methods that
    resolve the configured device route, deliberately convert arrays/tensors,
    and reject non-finite backend output. All ``quests`` imports are lazy and
    performed only when a concrete route is used, so the CPU route never
    imports torch/CUDA.

    Attributes:
        config (QuestsSplitConfig): The persistent configuration for the
            descriptor, entropy, device, and reproducibility parameters.
    """

    def __init__(self, config: QuestsSplitConfig) -> None:
        """Construct an objective for a given configuration."""
        self.config = config
        self._cpu_backend: Tuple[Any, Any] | None = None
        self._gpu_entropy: Any | None = None
        self._numba_threads_configured: bool = False

    # -- backend import boundaries (explicit, mockable at this level) -------

    def _import_cpu_backend(self) -> Tuple[Any, Any]:
        """Lazily import and return ``(descriptor, entropy)`` CPU modules.

        Importing these modules pulls in numba but never torch, keeping the
        CPU route free of any CUDA/torch initialization.
        """
        if self._cpu_backend is None:
            import quests.descriptor as descriptor_module
            import quests.entropy as entropy_module

            self._cpu_backend = (descriptor_module, entropy_module)
        return self._cpu_backend

    def _import_gpu_entropy(self) -> Any:
        """Lazily import and return the ``quests.gpu.entropy`` module.

        This module imports torch, so it is imported only by the GPU route.
        """
        if self._gpu_entropy is None:
            import quests.gpu.entropy as gpu_entropy_module

            self._gpu_entropy = gpu_entropy_module
        return self._gpu_entropy

    # -- device resolution --------------------------------------------------

    def _gpu_device_string(self) -> str:
        """Return the torch device string for the GPU route."""
        return self.config.gpu_device if self.config.gpu_device is not None else "cuda"

    def _assert_gpu_available(self) -> None:
        """Raise unless the configured CUDA backend/device is available."""
        try:
            import torch  # noqa: PLC0415
        except ImportError as exc:
            raise QuestsUnavailableError(
                "The QUESTS GPU route requires torch, which is not installed. "
                "Install the 'quests[gpu]' extra (torch) or configure "
                "device='cpu' or device='auto'."
            ) from exc
        if not torch.cuda.is_available():
            raise QuestsUnavailableError(
                "torch is installed but no CUDA device is available. "
                "Configure device='cpu' or device='auto' to run on the CPU."
            )
        if self.config.gpu_device is not None:
            device_string = self.config.gpu_device
            if not device_string.startswith("cuda"):
                raise ValueError(
                    "The QUESTS GPU route only supports CUDA device strings, "
                    f"got {device_string!r}."
                )
            device_index_part = device_string[len("cuda") :].lstrip(":")
            if device_index_part and not device_index_part.isdigit():
                raise ValueError(
                    f"Invalid CUDA device string {device_string!r}."
                )
            device_index = int(device_index_part) if device_index_part else 0
            if device_index >= torch.cuda.device_count():
                raise ValueError(
                    f"CUDA device {device_string!r} is not available; "
                    f"torch reports {torch.cuda.device_count()} CUDA device(s)."
                )

    def resolve_device(self) -> Literal["cpu", "gpu"]:
        """Resolve the configured device to a concrete backend route.

        Returns
        -------
        Literal["cpu", "gpu"]
            ``"cpu"`` when ``config.device == "cpu"``; ``"gpu"`` when
            ``config.device == "gpu"`` (raising
            :class:`QuestsUnavailableError` if CUDA is unavailable); and, for
            ``config.device == "auto"``, ``"gpu"`` when CUDA is available with
            a documented fallback to ``"cpu"`` otherwise.

        Raises
        ------
        QuestsUnavailableError
            If ``config.device == "gpu"`` but CUDA/torch is unavailable.
        """
        if self.config.device == "cpu":
            return "cpu"
        if self.config.device == "gpu":
            self._assert_gpu_available()
            return "gpu"
        # config.device == "auto": documented CPU fallback when CUDA is
        # unavailable.
        try:
            self._assert_gpu_available()
        except QuestsUnavailableError:
            return "cpu"
        return "gpu"

    def _configure_cpu_threads(self) -> None:
        """Apply ``config.numba_threads`` to the numba CPU kernels once."""
        if self.config.numba_threads is None or self._numba_threads_configured:
            return
        import numba as nb  # noqa: PLC0415

        nb.set_num_threads(self.config.numba_threads)
        self._numba_threads_configured = True

    # -- descriptor computation ---------------------------------------------

    def compute_descriptors(self, structures: Sequence[Atoms]) -> FrameDescriptors:
        """Compute per-frame QUESTS descriptors for a sequence of structures.

        The descriptors are always computed with the CPU numba descriptor
        module (``quests.descriptor.get_descriptors``), which is how QUESTS
        itself produces descriptors regardless of the entropy backend.

        Parameters
        ----------
        structures : Sequence[Atoms]
            Non-empty sequence of ASE structures; each must contain at least
            one atom.

        Returns
        -------
        FrameDescriptors
            Per-frame slices of the concatenated per-atom descriptors, aligned
            with ``structures`` order.

        Raises
        ------
        ValueError
            If ``structures`` is empty or any structure has no atoms.
        QuestsNumericalError
            If the descriptor matrix contains non-finite values.
        """
        frames = list(structures)
        if not frames:
            raise ValueError("structures must not be empty.")
        for index, atoms in enumerate(frames):
            if len(atoms) < 1:
                raise ValueError(
                    f"structure {index} contains no atoms; "
                    "QUESTS descriptors require at least one atom per frame."
                )

        self._configure_cpu_threads()
        descriptor_module, _ = self._import_cpu_backend()
        values = descriptor_module.get_descriptors(
            frames,
            k=self.config.descriptor_k,
            cutoff=self.config.descriptor_cutoff,
            concat=True,
            dtype=self.config.descriptor_dtype,
        )
        values = np.ascontiguousarray(values)
        if not np.all(np.isfinite(values)):
            raise QuestsNumericalError(
                "QUESTS descriptor computation returned non-finite values."
            )

        offsets = [0]
        for atoms in frames:
            offsets.append(offsets[-1] + len(atoms))
        return FrameDescriptors(values=values, frame_offsets=tuple(offsets))

    # -- entropy objective --------------------------------------------------

    def _to_tensor(self, array: np.ndarray) -> Any:
        """Convert a descriptor array to a torch tensor on the GPU device.

        Deliberate conversion: the array is made contiguous and cast to the
        torch dtype matching ``config.descriptor_dtype`` on the resolved GPU
        device.
        """
        import torch  # noqa: PLC0415

        if self.config.descriptor_dtype == "float32":
            torch_dtype = torch.float32
        else:
            torch_dtype = torch.float64
        device = torch.device(self._gpu_device_string())
        return torch.as_tensor(
            np.ascontiguousarray(array, dtype=self.config.descriptor_dtype),
            dtype=torch_dtype,
            device=device,
        )

    @staticmethod
    def _require_scalar_finite(value: float, what: str) -> None:
        """Raise unless a scalar backend result is finite."""
        if not math.isfinite(value):
            raise QuestsNumericalError(
                f"QUESTS {what} returned a non-finite value {value}; "
                "adjust entropy_bandwidth/descriptor parameters."
            )

    @staticmethod
    def _require_array_finite(values: np.ndarray, what: str) -> None:
        """Raise unless all entries of a backend result array are finite."""
        if not np.all(np.isfinite(values)):
            raise QuestsNumericalError(
                f"QUESTS {what} returned non-finite values "
                f"({int(np.count_nonzero(~np.isfinite(values)))} of "
                f"{values.size}); adjust entropy_bandwidth/descriptor "
                "parameters."
            )

    def entropy(self, descriptors: np.ndarray) -> float:
        """Compute the scalar QUESTS entropy of a descriptor matrix.

        Parameters
        ----------
        descriptors : np.ndarray
            Per-atom descriptor matrix of shape ``(N, d)`` with at least one
            row.

        Returns
        -------
        float
            The QUESTS entropy (in nats) of the descriptor set.

        Raises
        ------
        ValueError
            If ``descriptors`` is not a 2D matrix or is empty.
        QuestsNumericalError
            If the backend returns a non-finite value.
        """
        matrix = np.ascontiguousarray(
            np.asarray(descriptors, dtype=self.config.descriptor_dtype)
        )
        if matrix.ndim != 2:
            raise ValueError(
                "descriptors must be a 2D matrix, got shape "
                f"{matrix.shape}."
            )
        if matrix.shape[0] == 0:
            raise ValueError("entropy requires at least one descriptor row.")

        if self.resolve_device() == "cpu":
            _, entropy_module = self._import_cpu_backend()
            value = entropy_module.entropy(
                matrix,
                h=self.config.entropy_bandwidth,
                batch_size=self.config.entropy_batch_size,
            )
            result = float(value)
        else:
            entropy_module = self._import_gpu_entropy()
            tensor = self._to_tensor(matrix)
            value = entropy_module.entropy(
                tensor,
                h=self.config.entropy_bandwidth,
                batch_size=self.config.entropy_batch_size,
                device=self._gpu_device_string(),
            )
            result = float(value.detach().cpu().item())

        self._require_scalar_finite(result, "entropy")
        return result

    def delta_entropy(
        self,
        candidate: np.ndarray,
        reference: np.ndarray,
    ) -> np.ndarray:
        """Compute per-atom differential entropy of candidates given a reference.

        Both the CPU and GPU ``quests`` backends take the test/candidate set
        as the first argument and the reference set as the second argument;
        this adapter preserves that order.

        Parameters
        ----------
        candidate : np.ndarray
            Candidate descriptor matrix of shape ``(M, d)``.
        reference : np.ndarray
            Reference (selected) descriptor matrix of shape ``(N, d)``.

        Returns
        -------
        np.ndarray
            Float64 array of shape ``(M,)`` with the differential entropy
            ``-log(p_y)`` of each candidate row given the reference.

        Raises
        ------
        ValueError
            If either matrix is not 2D, is empty, or has a different feature
            dimension than the other.
        QuestsNumericalError
            If the backend returns non-finite values or an unexpected shape.
        """
        candidate_matrix = np.ascontiguousarray(
            np.asarray(candidate, dtype=self.config.descriptor_dtype)
        )
        reference_matrix = np.ascontiguousarray(
            np.asarray(reference, dtype=self.config.descriptor_dtype)
        )
        if candidate_matrix.ndim != 2 or reference_matrix.ndim != 2:
            raise ValueError(
                "candidate and reference must be 2D matrices, got "
                f"{candidate_matrix.shape} and {reference_matrix.shape}."
            )
        if candidate_matrix.shape[0] == 0 or reference_matrix.shape[0] == 0:
            raise ValueError(
                "candidate and reference must each contain at least one row."
            )
        if candidate_matrix.shape[1] != reference_matrix.shape[1]:
            raise ValueError(
                "candidate and reference must have the same feature "
                f"dimension; got {candidate_matrix.shape[1]} vs "
                f"{reference_matrix.shape[1]}."
            )

        if self.resolve_device() == "cpu":
            _, entropy_module = self._import_cpu_backend()
            result = np.asarray(
                entropy_module.delta_entropy(
                    candidate_matrix,
                    reference_matrix,
                    h=self.config.entropy_bandwidth,
                    batch_size=self.config.entropy_batch_size,
                ),
                dtype=np.float64,
            )
        else:
            entropy_module = self._import_gpu_entropy()
            candidate_tensor = self._to_tensor(candidate_matrix)
            reference_tensor = self._to_tensor(reference_matrix)
            value = entropy_module.delta_entropy(
                candidate_tensor,
                reference_tensor,
                h=self.config.entropy_bandwidth,
                batch_size=self.config.entropy_batch_size,
                device=self._gpu_device_string(),
            )
            result = np.asarray(
                value.detach().cpu().numpy(),
                dtype=np.float64,
            )

        if result.shape != (candidate_matrix.shape[0],):
            raise QuestsNumericalError(
                "QUESTS delta_entropy returned shape "
                f"{result.shape}, expected {(candidate_matrix.shape[0],)}."
            )
        self._require_array_finite(result, "delta_entropy")
        return result


def _validate_objective_config(
    config: QuestsSplitConfig,
    objective: QuestsObjective | None,
    *,
    what: str,
) -> None:
    """Raise ``ValueError`` unless ``objective.config`` equals ``config``.

    Enforces the invariant that a reusable :class:`QuestsObjective` must be
    constructed from exactly the configuration supplied alongside it: a
    mismatch would silently evaluate descriptors/entropy with different
    parameters than the persisted ``config``. Objectives that do not expose a
    ``config`` attribute (e.g. deterministic test fakes) cannot be validated
    and are accepted unchanged, so production checks are never weakened.

    Parameters
    ----------
    config : QuestsSplitConfig
        The configuration supplied to the public API.
    objective : QuestsObjective | None
        Optional reusable objective supplied to the same API.
    what : str
        Name of the API for the error message.

    Raises
    ------
    ValueError
        If ``objective`` exposes a ``config`` that differs from ``config``.
    """
    if objective is None:
        return
    objective_config = getattr(objective, "config", None)
    if objective_config is None:
        return
    if objective_config != config:
        raise ValueError(
            f"{what}: the supplied objective's config does not match the "
            "supplied QUESTS configuration. Construct the objective from the "
            "same QuestsSplitConfig passed to this API."
        )


# -- descriptor / profile / trajectory entry points -------------------------


def build_frame_descriptors(
    structures: Sequence[Atoms],
    *,
    config: QuestsSplitConfig,
    objective: QuestsObjective | None = None,
) -> FrameDescriptors:
    """Compute per-frame QUESTS descriptors for a sequence of structures.

    Convenience wrapper around :meth:`QuestsObjective.compute_descriptors`
    that constructs a fresh objective from ``config`` when none is supplied.

    Parameters
    ----------
    structures : Sequence[Atoms]
        Non-empty sequence of ASE structures.
    config : QuestsSplitConfig
        QUESTS descriptor/entropy/device configuration.
    objective : QuestsObjective | None, optional
        Reusable objective; defaults to a new ``QuestsObjective(config)``.

    Returns
    -------
    FrameDescriptors
        Per-frame descriptor slices aligned with ``structures``.

    Raises
    ------
    ValueError
        If a supplied ``objective`` exposes a ``config`` that differs from
        ``config``.
    """
    _validate_objective_config(config, objective, what="build_frame_descriptors")
    engine = objective if objective is not None else QuestsObjective(config)
    return engine.compute_descriptors(structures)


def _identity_to_structure(
    pool: Sequence[FrameReference],
    structures: Sequence[Atoms],
) -> Dict[Tuple[str, str, int], Atoms]:
    """Map each pool frame identity to its aligned structure.

    Parameters
    ----------
    pool : Sequence[FrameReference]
        Pool of frame references.
    structures : Sequence[Atoms]
        Structures aligned positionally with ``pool``.

    Returns
    -------
    Dict[Tuple[str, str, int], Atoms]
        Identity-to-structure mapping.

    Raises
    ------
    ValueError
        If ``pool`` and ``structures`` have different lengths or contain
        duplicate identities.
    """
    pool_list = list(pool)
    structures_list = list(structures)
    if len(pool_list) != len(structures_list):
        raise ValueError(
            "pool and structures must be aligned (same length); got "
            f"{len(pool_list)} vs {len(structures_list)}."
        )
    mapping: Dict[Tuple[str, str, int], Atoms] = {}
    for reference, atoms in zip(pool_list, structures_list):
        identity = reference.identity
        if identity in mapping:
            raise ValueError(
                f"pool contains duplicate frame identity {identity}."
            )
        mapping[identity] = atoms
    return mapping


def _selected_structures(
    trajectory: TrainValSplitTrajectory,
    identity_to_structure: Dict[Tuple[str, str, int], Atoms],
) -> List[Atoms]:
    """Return the structures of a trajectory's selected frames in order.

    Parameters
    ----------
    trajectory : TrainValSplitTrajectory
        Trajectory whose ``selected_frames`` are looked up.
    identity_to_structure : Dict[Tuple[str, str, int], Atoms]
        Mapping built by :func:`_identity_to_structure`.

    Returns
    -------
    list[Atoms]
        Structures of ``trajectory.selected_frames`` in the same order.

    Raises
    ------
    ValueError
        If a selected frame is not present in the mapping.
    """
    structures: List[Atoms] = []
    for reference in trajectory.selected_frames:
        atoms = identity_to_structure.get(reference.identity)
        if atoms is None:
            raise ValueError(
                f"Trajectory selects frame {reference.identity}, which is "
                "not part of the provided pool."
            )
        structures.append(atoms)
    return structures


def _pool_selected_indices(
    trajectory: TrainValSplitTrajectory,
    pool: Sequence[FrameReference],
) -> List[int]:
    """Return the pool indices of a trajectory's selected frames, in order.

    This is the index-space counterpart of :func:`_selected_structures`: it
    projects a trajectory's ``selected_frames`` onto their positions in
    ``pool`` so the selection can be sliced out of a
    :class:`FrameDescriptors` computed for the whole pool.

    Parameters
    ----------
    trajectory : TrainValSplitTrajectory
        Trajectory whose ``selected_frames`` are looked up.
    pool : Sequence[FrameReference]
        Pool the trajectory was generated from (the descriptor order).

    Returns
    -------
    list[int]
        Pool indices of ``trajectory.selected_frames`` in the same order.

    Raises
    ------
    ValueError
        If a selected frame is not present in ``pool`` or the pool contains
        duplicate identities.
    """
    identity_to_index: Dict[Tuple[str, str, int], int] = {}
    for index, reference in enumerate(pool):
        identity = reference.identity
        if identity in identity_to_index:
            raise ValueError(
                f"pool contains duplicate frame identity {identity}."
            )
        identity_to_index[identity] = index
    indices: List[int] = []
    for reference in trajectory.selected_frames:
        index = identity_to_index.get(reference.identity)
        if index is None:
            raise ValueError(
                f"Trajectory selects frame {reference.identity}, which is "
                "not part of the provided pool."
            )
        indices.append(index)
    return indices


def _chunk_information_gain(
    descriptors: FrameDescriptors,
    objective: QuestsObjective,
    selected_indices: Sequence[int],
    start: int,
    end: int,
) -> float:
    """Return the QUESTS information gain of adding frames ``[start, end)``.

    ``descriptors`` are the descriptor slices of the full pool and
    ``selected_indices`` are the pool indices of the trajectory's selected
    frames in selection order, so the chunk (``[start, end)``) and the
    previously selected reference (``[:start]``) are projected onto the pool
    slices.

    For a non-empty reference (``start > 0``) the gain is the sum over the
    chunk's atoms of ``delta_entropy(chunk_atoms | previously selected
    atoms)``, which is the same objective used by the greedy selection. For
    the first chunk (``start == 0``) there is no previously selected set, so
    the gain is the QUESTS entropy of the chunk itself (the information
    gained from nothing), matching the first selection step.

    The returned value may legitimately be negative: QUESTS ``delta_entropy``
    is ``-log(p)`` with an unnormalized kernel sum ``p`` that can exceed one
    for redundant chunks, so the differential entropy is not bounded below by
    zero.

    Parameters
    ----------
    descriptors : FrameDescriptors
        Descriptor slices for the full pool.
    objective : QuestsObjective
        Entropy objective.
    selected_indices : Sequence[int]
        Pool indices of the trajectory's selected frames in selection order.
    start : int
        Number of frames selected before this chunk.
    end : int
        Number of frames selected after this chunk (``end > start``).

    Returns
    -------
    float
        The QUESTS information gain of the chunk (may be negative).

    Raises
    ------
    ValueError
        If the chunk is empty.
    QuestsNumericalError
        If the backend returns non-finite values.
    """
    if end <= start:
        raise ValueError(
            f"chunk [start, end) must be non-empty; got start={start}, "
            f"end={end}."
        )
    indices = list(selected_indices)
    chunk_descriptors = descriptors.concat(indices[start:end])
    if start == 0:
        return objective.entropy(chunk_descriptors)
    reference_descriptors = descriptors.concat(indices[:start])
    deltas = objective.delta_entropy(chunk_descriptors, reference_descriptors)
    return float(np.sum(deltas))


def _evaluate_profile(
    *,
    trajectory: TrainValSplitTrajectory,
    descriptors: FrameDescriptors,
    selected_indices: Sequence[int],
    config: QuestsSplitConfig,
    objective: QuestsObjective,
) -> EntropyProfile:
    """Evaluate an entropy profile from precomputed pool descriptors.

    The cumulative entropies and gains are computed after every selection step
    by projecting the requested-size prefixes of ``selected_indices`` (the
    pool indices of the trajectory's selected frames in selection order) onto
    ``descriptors``, so no descriptors are recomputed. A selection step may add
    multiple frames; it produces exactly one profile point. This is the shared
    core behind :func:`evaluate_entropy_profile` and
    :func:`generate_quests_trajectory`.
Parameters
    ----------
    trajectory : TrainValSplitTrajectory
        Trajectory whose requested training sizes define the selection steps
        and whose selected-frame prefixes define the corresponding profile
        points.
    descriptors : FrameDescriptors
        Descriptor slices aligned with the pool.
    selected_indices : Sequence[int]
        Pool indices of the trajectory's selected frames in selection order.
    config : QuestsSplitConfig
        QUESTS configuration; ``entropy_tolerance`` is applied to the profile.
    objective : QuestsObjective
        Entropy objective.

    Returns
    -------
    EntropyProfile
    A complete profile with one point for every requested selection step.
"""
    points: List[EntropyProfilePoint] = []
    previous_size = 0
    # Each requested size is one selection step.  Do not expand a multi-frame
    # chunk into per-frame profile points.
    for current_size in trajectory.requested_train_sizes:
        cumulative_entropy = objective.entropy(
            descriptors.concat(list(selected_indices)[:current_size])
        )
        chunk_information_gain = _chunk_information_gain(
            descriptors,
            objective,
            selected_indices,
            previous_size,
            current_size,
        )
        points.append(
            EntropyProfilePoint(
                training_size=current_size,
                cumulative_entropy=cumulative_entropy,
                information_gain=chunk_information_gain,
            )
        )
        previous_size = current_size

    return EntropyProfile(points=points)


def evaluate_entropy_profile(
    *,
    trajectory: TrainValSplitTrajectory,
    pool: Sequence[FrameReference],
    structures: Sequence[Atoms],
    config: QuestsSplitConfig,
    objective: QuestsObjective | None = None,
    descriptors: FrameDescriptors | None = None,
) -> EntropyProfile:
    """Evaluate the QUESTS entropy profile of an existing trajectory.

    Works on any trajectory (``"random"`` or ``"quests"``): for each requested
    training size, the cumulative entropy is the QUESTS entropy of the
    selected prefix descriptors, and the chunk information gain is the
    ``delta_entropy``-based information gain of the chunk added at that point
    (see :func:`_chunk_information_gain`).

    When ``descriptors`` is supplied, it must be the precomputed descriptor
    slices of the *full pool* (aligned positionally with ``pool``); the
    trajectory's selected frames are then projected onto those slices, so no
    descriptors are recomputed. When it is omitted, the descriptors of the
    selected frames are computed here (the backward-compatible path).

    Parameters
    ----------
    trajectory : TrainValSplitTrajectory
        Trajectory to evaluate; its ``selected_frames`` must be a subset of
        ``pool``.
    pool : Sequence[FrameReference]
        Pool the trajectory was generated from.
    structures : Sequence[Atoms]
        Structures aligned positionally with ``pool``.
    config : QuestsSplitConfig
        QUESTS configuration; ``entropy_tolerance`` is applied to the profile.
    objective : QuestsObjective | None, optional
        Reusable objective; defaults to a new ``QuestsObjective(config)``.
    descriptors : FrameDescriptors | None, optional
        Precomputed descriptor slices of the full pool, aligned with ``pool``.
        When omitted, descriptors of the selected frames are computed here.

    Returns
    -------
    EntropyProfile
    A complete profile with one point for every requested selection step.
Raises
    ------
    ValueError
        If a supplied ``objective`` exposes a ``config`` that differs from
        ``config``, if ``pool``/``structures`` are misaligned, if a selected
        frame is not in the pool, or if ``descriptors`` is not aligned with
        the pool.
    QuestsNumericalError
        If the backend returns non-finite values.
    pydantic.ValidationError
        If the resulting profile violates finite/nonnegativity validation
        beyond ``config.entropy_tolerance``.
    """
    _validate_objective_config(config, objective, what="evaluate_entropy_profile")
    engine = objective if objective is not None else QuestsObjective(config)

    if descriptors is None:
        # Backward-compatible path: compute descriptors of the selected frames
        # (which are then trivially indexed 0..n-1 in selection order).
        identity_to_structure = _identity_to_structure(pool, structures)
        selected = _selected_structures(trajectory, identity_to_structure)
        descriptors = build_frame_descriptors(
            selected,
            config=config,
            objective=engine,
        )
        selected_indices = list(range(descriptors.n_frames))
    else:
        # Precomputed path: descriptors are aligned with the full pool.
        pool_list = list(pool)
        if descriptors.n_frames != len(pool_list):
            raise ValueError(
                "descriptors must be aligned with the pool: descriptor "
                f"frames ({descriptors.n_frames}) must equal pool size "
                f"({len(pool_list)})."
            )
        selected_indices = _pool_selected_indices(trajectory, pool_list)

    return _evaluate_profile(
        trajectory=trajectory,
        descriptors=descriptors,
        selected_indices=selected_indices,
        config=config,
        objective=engine,
    )


def populate_entropy_profile(
    *,
    trajectory: TrainValSplitTrajectory,
    pool: Sequence[FrameReference],
    structures: Sequence[Atoms],
    config: QuestsSplitConfig,
    objective: QuestsObjective | None = None,
    descriptors: FrameDescriptors | None = None,
) -> TrainValSplitTrajectory:
    """Return a copy of a trajectory with its entropy profile populated.

    The original trajectory is left unchanged; the returned trajectory is a
    ``model_copy`` carrying ``entropy_profile``. When ``descriptors`` is
    supplied it is the precomputed descriptor slices of the full pool (aligned
    with ``pool``) and is reused instead of recomputing descriptors (see
    :func:`evaluate_entropy_profile`).

    Parameters
    ----------
    trajectory : TrainValSplitTrajectory
        Trajectory to populate.
    pool : Sequence[FrameReference]
        Pool the trajectory was generated from.
    structures : Sequence[Atoms]
        Structures aligned positionally with ``pool``.
    config : QuestsSplitConfig
        QUESTS configuration.
    objective : QuestsObjective | None, optional
        Reusable objective; defaults to a new ``QuestsObjective(config)``.
    descriptors : FrameDescriptors | None, optional
        Precomputed descriptor slices of the full pool, aligned with ``pool``.
        When omitted, descriptors of the selected frames are computed here.

    Returns
    -------
    TrainValSplitTrajectory
        A new trajectory equal to ``trajectory`` with ``entropy_profile`` set.

    Raises
    ------
    ValueError
        If a supplied ``objective`` exposes a ``config`` that differs from
        ``config``, or if ``descriptors`` is not aligned with ``pool``.
    """
    _validate_objective_config(config, objective, what="populate_entropy_profile")
    profile = evaluate_entropy_profile(
        trajectory=trajectory,
        pool=pool,
        structures=structures,
        config=config,
        objective=objective,
        descriptors=descriptors,
    )
    return trajectory.model_copy(update={"entropy_profile": profile})


def _candidate_scores(
    descriptors: FrameDescriptors,
    objective: QuestsObjective,
    candidate_indices: Sequence[int],
    selected_indices: Sequence[int],
) -> Dict[int, float]:
    """Score each candidate frame by summed per-atom delta entropy.

    Computes ``delta_entropy(candidate_descriptors, selected_descriptors)``
    once for all candidates and reduces the per-atom differential entropy to
    a per-candidate-frame sum, matching the QUESTS "summed per-atom delta
    entropy per candidate structure" criterion.

    Parameters
    ----------
    descriptors : FrameDescriptors
        Descriptor slices for the full pool.
    objective : QuestsObjective
        Entropy objective (its ``delta_entropy`` method).
    candidate_indices : Sequence[int]
        Pool indices of candidate frames.
    selected_indices : Sequence[int]
        Pool indices of already-selected frames (the reference).

    Returns
    -------
    Dict[int, float]
        Mapping from each candidate pool index to its summed per-atom delta
        entropy against the selected set.

    Raises
    ------
    QuestsNumericalError
        If the backend returns non-finite values.
    """
    candidates = list(candidate_indices)
    candidate_descriptors = descriptors.concat(candidates)
    selected_descriptors = descriptors.concat(list(selected_indices))
    deltas = objective.delta_entropy(candidate_descriptors, selected_descriptors)

    scores: Dict[int, float] = {}
    row = 0
    for frame_index in candidates:
        atom_count = (
            descriptors.frame_offsets[frame_index + 1]
            - descriptors.frame_offsets[frame_index]
        )
        scores[frame_index] = float(np.sum(deltas[row : row + atom_count]))
        row += atom_count
    return scores


def _greedy_select_indices(
    descriptors: FrameDescriptors,
    objective: QuestsObjective,
    total: int,
) -> List[int]:
    """Greedily select ``total`` frame indices maximizing information gain.

    At each step the candidate with the highest summed per-atom differential
    entropy given the currently selected descriptors is chosen; the first
    frame is chosen by its own entropy (the empty-reference marginal
    information gain). Ties are broken by pool order (lowest index), making
    the selection deterministic.

    Parameters
    ----------
    descriptors : FrameDescriptors
        Descriptor slices for the full pool.
    objective : QuestsObjective
        Entropy objective.
    total : int
        Number of frames to select (at most the pool size).

    Returns
    -------
    list[int]
        Selected pool indices in selection order.
    """
    if total > descriptors.n_frames:
        raise ValueError(
            f"Cannot select {total} frames from a pool of "
            f"{descriptors.n_frames}."
        )

    selected: List[int] = []
    remaining: List[int] = list(range(descriptors.n_frames))
    for _ in range(total):
        if not selected:
            scores = {
                frame_index: objective.entropy(descriptors.slice(frame_index))
                for frame_index in remaining
            }
        else:
            scores = _candidate_scores(
                descriptors,
                objective,
                candidate_indices=remaining,
                selected_indices=selected,
            )
        # max score with ties broken by the smallest pool index (deterministic).
        best = max(remaining, key=lambda frame_index: (scores[frame_index], -frame_index))
        selected.append(best)
        remaining.remove(best)
    return selected


def generate_quests_trajectory(
    *,
    pool: Sequence[FrameReference],
    structures: Sequence[Atoms],
    requested_train_sizes: Sequence[int],
    config: QuestsSplitConfig,
    objective: QuestsObjective | None = None,
    descriptors: FrameDescriptors | None = None,
) -> TrainValSplitTrajectory:
    """Generate a nested maximum-information-entropy QUESTS trajectory.

    Frames are selected greedily one at a time, each conditioned on the
    descriptors selected so far (see :func:`_greedy_select_indices`), and the
    resulting ordered ``selected_frames`` prefixes define the nested training
    sets at the requested sizes. The trajectory stores no seed (selection is
    deterministic) and carries a fully evaluated :class:`EntropyProfile`.

    The full-pool descriptors are computed exactly once and reused for both
    selection and profile evaluation. When ``descriptors`` is supplied (the
    precomputed descriptor slices of the full pool, aligned with ``pool``), it
    is used directly instead of being recomputed, so callers that already
    computed pool descriptors (e.g. the high-level orchestration) avoid a
    second descriptor pass.

    Parameters
    ----------
    pool : Sequence[FrameReference]
        The train+validation pool to select from (typically the output of
        :func:`src.temper.splitting.common.partition_trainval_test`).
    structures : Sequence[Atoms]
        Structures aligned positionally with ``pool``.
    requested_train_sizes : Sequence[int]
        Strictly increasing requested training sizes, each at most the pool
        size.
    config : QuestsSplitConfig
        QUESTS configuration.
    objective : QuestsObjective | None, optional
        Reusable objective; defaults to a new ``QuestsObjective(config)``.
    descriptors : FrameDescriptors | None, optional
        Precomputed descriptor slices of the full pool, aligned with ``pool``.
        When omitted, the pool descriptors are computed here.

    Returns
    -------
    TrainValSplitTrajectory
        A ``method="quests"`` trajectory with ``seed=None``, nested
        ``selected_frames``, and an evaluated ``entropy_profile``.

    Raises
    ------
    ValueError
    If a supplied ``objective`` exposes a ``config`` that differs from
    ``config``, if the requested sizes are invalid, if ``pool``/``structures``
    are misaligned or empty, or if ``descriptors`` is not aligned with ``pool``.
    QuestsNumericalError
        If the backend returns non-finite values.
    """
    _validate_objective_config(config, objective, what="generate_quests_trajectory")
    pool_list = list(pool)
    structures_list = list(structures)
    if not pool_list:
        raise ValueError("pool must not be empty.")
    if len(pool_list) != len(structures_list):
        raise ValueError(
            "pool and structures must be aligned (same length); got "
            f"{len(pool_list)} vs {len(structures_list)}."
        )

    sizes = _validate_requested_train_sizes(
        requested_train_sizes,
        len(pool_list),
    )

    engine = objective if objective is not None else QuestsObjective(config)
    if descriptors is None:
        descriptors = build_frame_descriptors(
            structures_list,
            config=config,
            objective=engine,
        )
    else:
        if descriptors.n_frames != len(pool_list):
            raise ValueError(
                "descriptors must be aligned with the pool: descriptor "
                f"frames ({descriptors.n_frames}) must equal pool size "
                f"({len(pool_list)})."
            )
    selected_indices = _greedy_select_indices(
        descriptors,
        engine,
        sizes[-1],
    )

    selected_frames = [pool_list[index] for index in selected_indices]
    selected_index_set = set(selected_indices)
    additional_trainval_frames = [
        ref for index, ref in enumerate(pool_list)
        if index not in selected_index_set
    ]
    trajectory = TrainValSplitTrajectory(
        method="quests",
        seed=None,
        requested_train_sizes=sizes,
        selected_frames=selected_frames,
        additional_trainval_frames=additional_trainval_frames,
        entropy_profile=None,
    )

    profile = _evaluate_profile(
        trajectory=trajectory,
        descriptors=descriptors,
        selected_indices=selected_indices,
        config=config,
        objective=engine,
    )
    return trajectory.model_copy(update={"entropy_profile": profile})
