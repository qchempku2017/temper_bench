"""Adapts the QUESTS backend to compute per-atom descriptors, entropy, and information gain. It validates persisted configuration and manages lazy CPU or GPU backend access."""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Tuple, List, Literal, Any
import math

import numpy as np
from ase import Atoms

from src.temper.schemas.quests_adapter import QuestsAdapterConfig


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
class QuestsDescriptorsStorage:
    """Storage of per-frame slices of concatenated per-atom QUESTS descriptors.

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
        quests_adapter_config (QuestsAdapterConfig): Configuration of the QUESTS
            adapter used to compute the descriptors. Used to validate that the
            descriptors can be reused.
    """

    values: np.ndarray
    frame_offsets: Tuple[int, ...]
    quests_adapter_config: QuestsAdapterConfig

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
        # Offsets delimit frames, not individual descriptor rows.  There is no
        # fixed relation between their count and ``values.shape[0]`` because a
        # frame may contain any positive number of atoms.
        if len(self.frame_offsets) < 2:
            raise ValueError("frame_offsets must delimit at least one frame.")
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

    @cached_property
    def frame_atom_counts(self) -> Tuple[int, ...]:
        """Number of atoms in each frame, in frame order.

        Notice: this is a cached property, so it is computed only once.
        """
        return tuple(
            self.frame_offsets[i + 1] - self.frame_offsets[i]
            for i in range(self.n_frames)
        )

    def get_one_frame(self, frame_index: int) -> np.ndarray:
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
        return self.values[
            self.frame_offsets[frame_index] : self.frame_offsets[frame_index + 1]
        ]

    def get_multiple_frames(self, frame_indices: List[int]) -> np.ndarray:
        """Return the concatenated descriptor rows of an ordered subset.

        Parameters
        ----------
        frame_indices : List[int]
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
        return np.concatenate([self.get_one_frame(index) for index in indices], axis=0)


class QuestsAdapter:
    """Explicit CPU/GPU adapter for the QUESTS entropy adapter.

    Wraps the verified ``quests`` import surface behind typed methods that
    resolve the configured device route, deliberately convert arrays/tensors,
    and reject non-finite backend output. All ``quests`` imports are lazy and
    performed only when a concrete route is used, so the CPU route never
    imports torch/CUDA.

    Attributes:
        config (QuestsAdapterConfig): The persistent configuration for the
            descriptor, entropy, device, and reproducibility parameters.
    """

    def __init__(self, config: QuestsAdapterConfig) -> None:
        """Construct an adapter for a given configuration."""
        self.config = config
        self._cpu_backend: Tuple[Any, Any] | None = None
        self._gpu_entropy: Any | None = None
        self._numba_threads_configured: bool = False

    # -- backend import boundaries (explicit, mockable at this level) -------

    def _import_cpu_backend(self) -> Tuple[Any, Any] | None:
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

    def _configure_numba_cpu_threads(self) -> None:
        """Apply ``config.numba_threads`` to the numba CPU kernels once."""
        if self._numba_threads_configured:
            return
        import numba as nb

        nb.set_num_threads(self.config.numba_threads)
        self._numba_threads_configured = True

        # Avoid multi-threading in OMP, MKL, OpenBLAS, NumExpr, BLIS.
        import os

        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["NUMEXPR_NUM_THREADS"] = "1"
        os.environ["BLIS_NUM_THREADS"] = "1"

    # -- descriptor computation ---------------------------------------------

    def compute_descriptors(self, frames: List[Atoms]) -> QuestsDescriptorsStorage:
        """Compute per-frame QUESTS descriptors for a sequence of structures.

        The descriptors are always computed with the CPU numba descriptor
        module (``quests.descriptor.get_descriptors``), which is how QUESTS
        itself produces descriptors regardless of the entropy backend.

        Parameters
        ----------
        frames : List[Atoms]
            Non-empty sequence of ASE structures; each must contain at least
            one atom.

        Returns
        -------
        QuestsDescriptorsStorage
            Per-frame slices of the concatenated per-atom descriptors, aligned
            with ``structures`` order.

        Raises
        ------
        ValueError
            If ``structures`` is empty or any structure has no atoms.
        QuestsNumericalError
            If the descriptor matrix contains non-finite values.
        """
        if not frames:
            raise ValueError("structures must not be empty.")
        for index, atoms in enumerate(frames):
            if len(atoms) < 1:
                raise ValueError(
                    f"structure {index} contains no atoms; "
                    "QUESTS descriptors require at least one atom per frame."
                )

        self._configure_numba_cpu_threads()
        descriptor_module, _ = self._import_cpu_backend()
        chunks = []
        for i in range(0, len(frames), self.config.compute_descriptor_chunk_size):
            chunks.append(descriptor_module.get_descriptors(
                frames[i:i+self.config.compute_descriptor_chunk_size],
                k=self.config.descriptor_k,
                cutoff=self.config.descriptor_cutoff,
                concat=True,
                dtype=self.config.descriptor_dtype,
            ))
        values = np.concatenate(chunks, axis=0)
        if not np.all(np.isfinite(values)):
            raise QuestsNumericalError(
                "QUESTS descriptor computation returned non-finite values."
            )

        offsets = [0]
        for atoms in frames:
            offsets.append(offsets[-1] + len(atoms))
        return QuestsDescriptorsStorage(
            values=values,
            frame_offsets=tuple(offsets),
            quests_adapter_config=self.config,
        )

    # -- entropy adapter --------------------------------------------------

    def _to_tensor(self, array: np.ndarray) -> Any:
        """Convert a descriptor array to a torch tensor on the GPU device.

        Deliberate conversion: the array is made contiguous and cast to the
        torch dtype matching ``config.descriptor_dtype`` on the resolved GPU
        device.
        """
        try:
            import torch
        except ImportError as e:
            raise ImportError(
                "QUESTS entropy computation GPU backend requires PyTorch, but not installed."
            ) from e

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

    def get_entropy(self, descriptors: np.ndarray) -> float:
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
            value = entropy_module.get_entropy(
                matrix,
                h=self.config.entropy_bandwidth,
                batch_size=self.config.entropy_batch_size,
            )
            result = float(value)
        else:
            entropy_module = self._import_gpu_entropy()
            tensor = self._to_tensor(matrix)
            value = entropy_module.get_entropy(
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


def compute_information_gain_per_candidate_frame(
    descriptors: QuestsDescriptorsStorage,
    adapter: QuestsAdapter,
    selected_frame_indices: List[int],
    candidate_frame_indices: List[int],
) -> np.ndarray:
    """Return the QUESTS information gain of adding each of the candidate frames.

    Parameters
    ----------
    descriptors : QuestsDescriptorsStorage
        Descriptor slices for the full pool.
    adapter : QuestsAdapter
        Entropy adapter.
    selected_frame_indices : List[int]
        Indices of selected frames in structure pool (i.e, descriptors),
        respecting selection order.
    candidate_frame_indices : List[int]
        Indices of candidate frames in structure pool (i.e, descriptors),

    Returns
    -------
    float
        The QUESTS information gain of the chunk (can be negative).

    Raises
    ------
    QuestsNumericalError
        If the backend returns non-finite values.
    """
    candidate_descriptors = descriptors.get_multiple_frames(candidate_frame_indices)

    reference_descriptors = descriptors.get_multiple_frames(selected_frame_indices)
    deltas_per_atom = adapter.delta_entropy(candidate_descriptors, reference_descriptors)

    deltas_per_structure = []
    atom_index = 0
    for i in candidate_frame_indices:
        n_atoms = descriptors.frame_atom_counts[i]
        deltas_per_structure.append(np.sum(deltas_per_atom[atom_index:atom_index + n_atoms]))
        atom_index += n_atoms
    return np.array(deltas_per_structure)


def compute_total_entropy_of_selected_frames(
    descriptors: QuestsDescriptorsStorage,
    adapter: QuestsAdapter,
    selected_frame_indices: List[int],
) -> float:
    """Evaluate entropy of all selected frames in a pool.

    The cumulative entropies and gains are computed after every selection step.

    Parameters
    ----------
    descriptors : QuestsDescriptorsStorage
        Descriptor slices for the full pool.
    adapter : QuestsAdapter
        Entropy adapter.
    selected_frame_indices : List[int]
        Indices of selected frames in structure pool (i.e, descriptors),
        respecting selection order.

    Returns
    -------
    float
        The QUESTS entropy of the selected frames.
    """
    return adapter.get_entropy(
        descriptors.get_multiple_frames(selected_frame_indices)
    )
