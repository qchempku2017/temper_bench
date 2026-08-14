"""Schema foundation for MLFF dataset splitting.

Defines the persisted, reference-only models used to represent the result of
splitting a data group into train/validation/test sets. Only references
(domain, relative extxyz source filename, and nonnegative frame index) are
stored; structures and descriptors are never stored.

This module contains:

- ``SplitSchema``: the legacy configuration-oriented schema, preserved for
  backward compatibility.
- ``FrameReference``: a single persisted reference to a structure frame.
- ``EntropyProfilePoint`` / ``EntropyProfile``: QUESTS maximum-entropy
  evaluation data associated with a trajectory.
- ``TrainValSplitTrajectory``: a train/validation trajectory (method ``"random"``
  or ``"quests"``) whose ordered selected-frame list, together with ordered
  additional train/validation frames, defines nested train/validation sets.
- ``QuestsSplitConfig``: typed configuration for the QUESTS entropy backend
  (descriptor, entropy, device, and reproducibility parameters).
- ``SplitDataSchema``: the persisted reference-only result of splitting a data
  group, with one singular train/validation trajectory.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Literal, Tuple, ClassVar

from pydantic import BaseModel, field_validator, model_validator, Field

from src.temper.schemas.utils import validate_relative_extxyz_path



class SplitSchema(BaseModel):
    """Schema to define the data-splitting in a specific fine-tuning experiment.

    Specifies the domain of data, grouping strategy name, group name,
    train/val splitting method ("random" or "max_entropy"),
    training set size, validation set size, test set size, random seed for split train+val against test,
    random seed for split train against val, and the content of train val and test sets.

    Attributes:
        domain (str): Name of data domain that the split dataset belongs to.
        grouping_strategy (str): Name of the grouping strategy used to group all datafiles in the domain.
        group_name (str): Name of the group this split dataset belongs to. Refer to `GroupEntry` for the
            concept of a group.
        train_val_split_method (str): Name of the method to split training set against validation set.
            Either "random" or "max_entropy". Splitting of train+val against test is always performed
            at random.
        train_size (int) : training set size. Do not provide, will be calculated from the dataset at model validation.
        val_size (int) : validation set size. Do not provide, will be calculated from the dataset at model validation.
        test_size (int): test set size. Do not provide, will be calculated from the dataset at model validation.
        train_val_test_split_seed (int): random seed used to perform train+val vs test split.
        train_val_split_seed (int | None, optional): random seed used to perform train vs val split.
            Default to None, as this is only used when `train_val_split_method` is "random".
        train_set (Dict[str, List[int]]): training set. Structure is {datafile_name: [list of indices of
            structure frames in the datafile]}.
        val_set (Dict[str, List[int]]): validation set. Structure is the same as `train_set`.
        test_set (Dict[str, List[int]]): test set. Structure is the same as `train_set`.
    """
    domain: str
    grouping_strategy: str
    group_name: str
    train_val_split_method: str
    train_val_test_split_seed: int
    train_val_split_seed: int | None = None
    train_set: Dict[str, List[int]]
    val_set: Dict[str, List[int]]
    test_set: Dict[str, List[int]]
    train_size: int | None = None
    val_size: int | None = None
    test_size: int | None = None

    @field_validator("train_val_split_method")
    @classmethod
    def validate_train_val_split_method(cls, v):
        """Validate the train_val_split_method."""
        if v not in ["random", "max_entropy"]:
            raise ValueError(
                f"train_val_split_method must be either 'random' or 'max_entropy',"
                f" but {v} provided."
            )
        return v

    @model_validator(mode="after")
    def validate_dataset_sizes(self):
        """Validate the sizes of the train, val, and test sets."""
        self.train_size = sum(len(indices) for indices in self.train_set.values())
        self.val_size = sum(len(indices) for indices in self.val_set.values())
        self.test_size = sum(len(indices) for indices in self.test_set.values())
        return self


class FrameReference(BaseModel):
    """Persisted reference to a single structure frame in a data group.

    References are lightweight: they only store the identity of a frame
    (domain, relative extxyz source filename, and nonnegative frame index).
    Structures and descriptors are never stored.

    Attributes:
        domain (str): Name of the data domain the frame belongs to.
        filename (str): Relative path to the extxyz source file, relative to
            the domain directory. Must be relative, end with ``.extxyz``, and
            must not contain directory-traversal segments.
        frame_index (int): Zero-based, nonnegative index of the frame within
            the source file.
    """

    domain: str
    filename: str
    frame_index: int

    @field_validator("domain")
    @classmethod
    def validate_domain(cls, value: str) -> str:
        """Require a non-empty domain name."""
        if not isinstance(value, str) or not value.strip():
            raise ValueError("domain must be a non-empty string.")
        return value

    @field_validator("filename")
    @classmethod
    def validate_filename(cls, value: str) -> str:
        """Require a safe relative extxyz source path."""
        return validate_relative_extxyz_path(value)

    @field_validator("frame_index")
    @classmethod
    def validate_frame_index(cls, value: int) -> int:
        """Reject negative frame indices."""
        if value < 0:
            raise ValueError(f"frame_index must be nonnegative, got {value}.")
        return value

    @property
    def identity(self) -> tuple[str, str, int]:
        """A hashable identity tuple ``(domain, filename, frame_index)``.

        Used for set-membership checks (e.g. duplicate detection and
        train/validation complement computation) without relying on model
        hashing.
        """
        return self.domain, self.filename, self.frame_index

    def as_dict(self) -> Dict[str, Any]:
        """Convert to a plain dictionary compatible with monty.serialization.dumpfn."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FrameReference":
        """Construct from a dictionary generated by :meth:`as_dict`."""
        return cls.model_validate(data)


class EntropyProfilePoint(BaseModel):
    """A single point of a QUESTS maximum-entropy entropy profile.

    Each point corresponds to one selection step. The step may add multiple
    frames; ``training_size`` is the selected-set size after that step. The
    "chunk" is the set of frames added since the previous point, so its size
    is the difference between consecutive ``training_size`` values.

    Attributes:
        training_size (int): Number of training frames selected at this point.
        cumulative_entropy (float): Cumulative QUESTS entropy of the training
            set at this point. Must be finite; the QUESTS entropy normalizes
            by the set size, so it is not guaranteed to be non-decreasing as
            frames are added. Nonnegativity is validated with a configurable
            tolerance by :class:`EntropyProfile`.
        information_gain (float): QUESTS information gain contributed by
            the chunk of frames added at this point, computed with the same
            ``delta_entropy`` objective used for selection. Must be finite.
            The QUESTS ``delta_entropy`` is an unnormalized differential
            entropy (``-log(p)``) that is not bounded below by zero, so this
            value may legitimately be negative for redundant chunks.
    """

    training_size: int
    cumulative_entropy: float
    information_gain: float

    @field_validator("training_size")
    @classmethod
    def validate_training_size(cls, value: int) -> int:
        """Reject non-positive training sizes."""
        if value <= 0:
            raise ValueError(f"training_size must be positive, got {value}.")
        return value

    @field_validator("cumulative_entropy", "information_gain")
    @classmethod
    def validate_finite(cls, value: float) -> float:
        """Reject NaN and infinite entropy values."""
        if not math.isfinite(value):
            raise ValueError(f"Entropy values must be finite, got {value}.")
        return value

    def as_dict(self) -> Dict[str, Any]:
        """Convert to a plain dictionary compatible with monty.serialization.dumpfn."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EntropyProfilePoint":
        """Construct from a dictionary generated by :meth:`as_dict`."""
        return cls.model_validate(data)


class EntropyProfile(BaseModel):
    """Ordered sequence of QUESTS entropy profile points.

    A trajectory may be persisted before evaluation with no profile
    (``entropy_profile=None``). When a profile is present it must be complete:
    the points must be ordered by strictly increasing ``training_size`` and all
    entropy values must be finite. Profile points represent the owning
    trajectory's selection steps, so one point may represent a multi-frame
    chunk.

    Validation is deliberately restricted to what the QUESTS objective
    guarantees (verified against ``quests==2026.2.22``):

    - Cumulative entropy normalizes by the set size and may increase or
      decrease as frames are added; no monotonicity or nonnegativity check is
      applied.
    - ``information_gain`` is a differential entropy (``-log(p)`` with an
      unnormalized kernel sum ``p``) and may legitimately be negative; only
      finiteness is enforced.

    Attributes:
        points (list[EntropyProfilePoint]): Ordered entropy profile points.
    """

    points: List[EntropyProfilePoint]

    @model_validator(mode="after")
    def validate_profile(self) -> "EntropyProfile":
        """Validate ordering and reject present-but-empty profiles."""
        if not self.points:
            raise ValueError("EntropyProfile.points must not be empty.")
        previous_size = 0
        for i, point in enumerate(self.points):
            if point.training_size <= previous_size:
                raise ValueError(
                    "EntropyProfile points must have strictly increasing "
                    f"training_size; point {i} has training_size "
                    f"{point.training_size} after {previous_size}."
                )
            previous_size = point.training_size
        return self

    def as_dict(self) -> Dict[str, Any]:
        """Convert to a plain dictionary compatible with monty.serialization.dumpfn."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EntropyProfile":
        """Construct from a dictionary generated by :meth:`as_dict`."""
        return cls.model_validate(data)


class TrainValSplitTrajectory(BaseModel):
    """A single train/validation trajectory for one splitting method.

    Both the ``"random"`` method and the ``"quests"`` (QUESTS
    maximum-entropy) method use this single output convention: an ordered
    ``selected_frames`` list plus ordered ``additional_trainval_frames`` form
    the complete train/validation inventory. Prefixes of ``selected_frames``
    define nested training sets at strictly increasing
    ``requested_train_sizes``.

    At checkpoint index ``i``, validation is the remaining selected suffix
    followed by ``additional_trainval_frames`` (see :meth:`get_val_set`).

    Attributes:
        method (Literal["random", "quests"]): Splitting method that produced
            this trajectory.
        seed (int | None): Random seed. Required for ``"random"``. Must be
            ``None`` for ``"quests"``, whose selection is deterministic.
        requested_train_sizes (list[int]): Strictly increasing requested
            training sizes. Each is the size of a nested training prefix.
        selected_frames (list[FrameReference]): Ordered selected-frame list;
            the first ``s`` frames form the training set of size ``s``.
        additional_trainval_frames (list[FrameReference] | None): Additional frames in
            the original train+val dataset but not selected in ``selected_frames``.
            Will go into the validation set. Empty list by default.
        entropy_profile (EntropyProfile | None): QUESTS maximum-entropy
            profile with one point per incremental selection step.
            ``None`` before evaluation; a complete profile once evaluated.
    """
    #: Supported train-val splitting methods.
    SUPPORTED_TRAIN_VAL_SPLIT_METHODS: ClassVar[Tuple[str, ...]] = ("random", "quests")

    method: Literal["random", "quests"]
    seed: int | None = None
    requested_train_sizes: List[int]
    selected_frames: List[FrameReference]
    additional_trainval_frames: List[FrameReference] = Field(default_factory=list)
    entropy_profile: EntropyProfile | None = None

    @model_validator(mode="after")
    def validate_trajectory(self) -> "TrainValSplitTrajectory":
        """Validate seed, requested sizes, selected frames, and profile."""
        if not self.method in self.SUPPORTED_TRAIN_VAL_SPLIT_METHODS:
            raise ValueError(
                f"method must be one of {self.SUPPORTED_TRAIN_VAL_SPLIT_METHODS}, "
                f"got {self.method}."
            )

        if self.method == "random" and self.seed is None:
            raise ValueError(
                "The 'random' method requires a non-None seed."
            )
        if self.method == "quests" and self.seed is not None:
            raise ValueError(
                "The 'quests' method is deterministic and must not store a seed."
            )

        requested = self.requested_train_sizes
        if not requested:
            raise ValueError("requested_train_sizes must not be empty.")
        for i, size in enumerate(requested):
            if size <= 0:
                raise ValueError(
                    f"requested_train_sizes[{i}] must be positive, got {size}."
                )
            if i > 0 and size <= requested[i - 1]:
                raise ValueError(
                    "requested_train_sizes must be strictly increasing; "
                    f"index {i} has {size} after {requested[i - 1]}."
                )
            if size > len(self.selected_frames):
                raise ValueError(
                    f"requested_train_sizes[{i}] = {size} exceeds the number of "
                    f"selected_frames ({len(self.selected_frames)})."
                )

        selected_identities = [ref.identity for ref in self.selected_frames]
        if len(set(selected_identities)) != len(selected_identities):
            raise ValueError("selected_frames must not contain duplicate frames.")
        additional_identities = [
            ref.identity for ref in self.additional_trainval_frames
        ]
        if len(set(additional_identities)) != len(additional_identities):
            raise ValueError(
                "additional_trainval_frames must not contain duplicate frames."
            )
        overlap = set(selected_identities) & set(additional_identities)
        if overlap:
            raise ValueError(
                "additional_trainval_frames must be disjoint from selected_frames; "
                f"overlap starts with {next(iter(overlap))!r}."
            )

        if self.entropy_profile is not None:
            profile_sizes = [point.training_size for point in self.entropy_profile.points]
            if profile_sizes != requested:
                raise ValueError(
                    "entropy_profile training_size values must exactly match "
                    "requested_train_sizes."
                )
        return self


    def get_train_set(self, requested_train_size_index: int) -> List[FrameReference]:
        """Return the training references for a requested training size.

        The training set is the prefix of :attr:`selected_frames` of length
        ``requested_size``.

        Parameters
        ----------
        requested_train_size_index : int
            Index of the requested training set size in :attr:`requested_train_sizes`.

        Returns
        -------
        list[FrameReference]
            Ordered training set references.
        """
        return self.selected_frames[:self.requested_train_sizes[requested_train_size_index]]

    def get_val_set(
        self,
        requested_train_size_index: int
    ) -> List[FrameReference]:
        """Return the validation references for a requested training size.

        The validation set is the complement of the training prefix within the
        trajectory's embedded complete train+validation inventory, preserving
        inventory order.

        Parameters
        ----------
        requested_train_size_index : int
            Index of the requested training set size in :attr:`requested_train_sizes`.

        Returns
        -------
        list[FrameReference]
            Validation references by concatenating selected_frames[requested_size:] and
            additional_trainval_frames.
        """
        size = self.requested_train_sizes[requested_train_size_index]
        return self.selected_frames[size:] + self.additional_trainval_frames

    def as_dict(self) -> Dict[str, Any]:
        """Convert to a plain dictionary compatible with monty.serialization.dumpfn."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainValSplitTrajectory":
        """Construct from a dictionary generated by :meth:`as_dict`."""
        return cls.model_validate(data)


class SplitDataSchema(BaseModel):
    """Persisted top-level result of splitting a data group.

    This is the persisted (reference-only) result schema for splitting a
    single data group into train/validation/test. It stores provenance (the
    identity of the domain/group and split configuration), a singular
    :class:`TrainValSplitTrajectory`, and the test references. The trajectory
    itself contains the complete train+validation inventory.

    Only frame references are stored; structures and descriptors are never
    stored.

    Attributes:
        domain (str): Name of the data domain.
        grouping_strategy (str): Name of the grouping strategy used.
        group_name (str): Name of the group that was split.
        test_set (list[FrameReference]): Test references within the current group
            and from extra cross-group tests.
        test_ratio (float): Requested ratio of the test set to the total
            dataset size. Must be in ``(0, 1)``.
        trainval_test_split_seed (int): Random seed used for the train+validation vs test
            partition (always performed at random).
        train_val_split_trajectory (TrainValSplitTrajectory): The single
            trajectory produced for this schema's splitting method.
        quests_config (QuestsSplitConfig | None): Provenance of the QUESTS
            descriptor/entropy/device configuration used to generate the
            ``"quests"`` trajectory and to evaluate the entropy profile of
            every trajectory (including ``"random"``). ``None`` when the
            schema was produced without any QUESTS evaluation (kept for
            backward compatibility with schemas persisted before this field
            was added).
    """
    domain: str
    grouping_strategy: str
    group_name: str
    test_set: List[FrameReference]
    test_ratio: float
    trainval_test_split_seed: int
    train_val_split_trajectory: TrainValSplitTrajectory
    quests_config: "QuestsSplitConfig | None" = None

    @field_validator("test_ratio")
    @classmethod
    def validate_test_ratio(cls, value: float) -> float:
        """Reject test ratios outside ``(0, 1)``."""
        if not 0.0 < value < 1.0:
            raise ValueError(f"test_ratio must be in (0, 1), got {value}.")
        return value

    @model_validator(mode="after")
    def validate_split_result(self) -> "SplitDataSchema":
        """Validate internal consistency of a persisted split result."""
        # Check testing set size.
        references = [
            *self.train_val_split_trajectory.selected_frames,
            *self.train_val_split_trajectory.additional_trainval_frames,
            *self.test_set,
        ]
        total = len(references)
        expected_test_size = round(total * self.test_ratio)
        if len(self.test_set) != expected_test_size:
            raise ValueError(
                f"test_set size ({len(self.test_set)}) does not match "
                f"round(total * test_ratio) = {expected_test_size} "
                f"for total={total}, test_ratio={self.test_ratio}."
            )

        # Check domain consistency.
        mismatched_domains = [
            ref.identity
            for ref in references
            if ref.domain != self.domain
        ]
        if mismatched_domains:
            raise ValueError(
                f"All frame references must belong to domain {self.domain!r}; "
                f"found {mismatched_domains[:5]}."
            )

        # Check frame duplication.
        identities = [ref.identity for ref in references]
        if len(set(identities)) != len(identities):
            raise ValueError("Train, val and test sets must not contain duplicate frames.")

        return self

    def as_dict(self) -> Dict[str, Any]:
        """Convert to a plain dictionary compatible with monty.serialization.dumpfn."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SplitDataSchema":
        """Construct from a dictionary generated by :meth:`as_dict`."""
        return cls.model_validate(data)


class QuestsSplitConfig(BaseModel):
    """Typed configuration for the QUESTS maximum-entropy splitting backend.

    Persists every descriptor, entropy, device, and reproducibility parameter
    consumed by the QUESTS adapter in ``src.temper.splitting.quests``. The
    QUESTS selection itself is deterministic (greedy maximum-information-gain
    selection conditioned on the descriptors selected so far), so no random
    seed is required or stored; see the adapter module documentation.

    Attributes:
        descriptor_k (int): Number of nearest neighbors ``k`` used by the
            QUESTS per-atom descriptor. The concatenated descriptor has
            ``2*k - 1`` columns. Must be at least 2.
        descriptor_cutoff (float): Cutoff radius (in Ångström) of the
            descriptor weight function. Must be positive.
        descriptor_dtype (Literal["float32", "float64"]): Floating-point dtype
            of the computed descriptors and, for the GPU route, of the torch
            tensors passed to the QUESTS backend.
        entropy_bandwidth (float): Bandwidth ``h`` of the Gaussian kernel used
            by the QUESTS entropy objective. Must be positive.
        entropy_batch_size (int): Maximum batch size used by the QUESTS
            backend when batching distance computations. Must be positive.
        entropy_tolerance (float): Absolute tolerance applied when validating
            that profile entropy values are nonnegative and that cumulative
            entropy is non-decreasing (see
            :attr:`EntropyProfile.entropy_tolerance`). The QUESTS backend can
            produce tiny negative values (e.g. ``-0.0``) for coincident or
            near-identical descriptors. Must be nonnegative.
        device (Literal["cpu", "gpu", "auto"]): Which QUESTS backend route to
            use. ``"cpu"`` never imports or initializes CUDA/torch;
            ``"gpu"`` requires an available CUDA device and raises
            :class:`src.temper.splitting.quests.QuestsUnavailableError`
            otherwise; ``"auto"`` uses the GPU route when a CUDA device is
            available and falls back to the CPU route otherwise (documented
            fallback).
        gpu_device (str | None): Optional torch device string (e.g.
            ``"cuda:0"``) used by the GPU route. Must be ``None`` when
            ``device == "cpu"``.
        numba_threads (int | None): Optional number of threads for the numba
            parallel sections of the CPU descriptor/entropy kernels. ``None``
            leaves numba's default thread count unchanged. The results are
            deterministic regardless of this value.
    """

    descriptor_k: int = 32
    descriptor_cutoff: float = 5.0
    descriptor_dtype: Literal["float32", "float64"] = "float64"
    entropy_bandwidth: float = 0.015
    entropy_batch_size: int = 20000
    entropy_tolerance: float = 1e-6
    device: Literal["cpu", "gpu", "auto"] = "auto"
    gpu_device: str | None = None
    numba_threads: int | None = None

    @field_validator("descriptor_k")
    @classmethod
    def validate_descriptor_k(cls, value: int) -> int:
        """Reject descriptor neighbor counts below 2."""
        if value < 2:
            raise ValueError(f"descriptor_k must be at least 2, got {value}.")
        return value

    @field_validator("descriptor_cutoff")
    @classmethod
    def validate_descriptor_cutoff(cls, value: float) -> float:
        """Reject non-finite or non-positive descriptor cutoffs."""
        if not math.isfinite(value) or value <= 0:
            raise ValueError(
                f"descriptor_cutoff must be finite and positive, got {value}."
            )
        return value

    @field_validator("entropy_bandwidth")
    @classmethod
    def validate_entropy_bandwidth(cls, value: float) -> float:
        """Reject non-finite or non-positive entropy bandwidths."""
        if not math.isfinite(value) or value <= 0:
            raise ValueError(
                f"entropy_bandwidth must be finite and positive, got {value}."
            )
        return value

    @field_validator("entropy_batch_size")
    @classmethod
    def validate_entropy_batch_size(cls, value: int) -> int:
        """Reject non-positive entropy batch sizes."""
        if value <= 0:
            raise ValueError(
                f"entropy_batch_size must be positive, got {value}."
            )
        return value

    @field_validator("entropy_tolerance")
    @classmethod
    def validate_entropy_tolerance(cls, value: float) -> float:
        """Reject non-finite or negative entropy tolerance values."""
        if not math.isfinite(value) or value < 0:
            raise ValueError(
                f"entropy_tolerance must be finite and nonnegative, got {value}."
            )
        return value

    @field_validator("numba_threads")
    @classmethod
    def validate_numba_threads(cls, value: int | None) -> int | None:
        """Reject non-positive numba thread counts."""
        if value is not None and value < 1:
            raise ValueError(
                f"numba_threads must be positive when set, got {value}."
            )
        return value

    @model_validator(mode="after")
    def validate_device_consistency(self) -> "QuestsSplitConfig":
        """Validate that ``gpu_device`` is consistent with ``device``."""
        if self.device == "cpu" and self.gpu_device is not None:
            raise ValueError(
                "gpu_device must be None when device == 'cpu'; "
                "the CPU route never initializes CUDA."
            )
        if self.gpu_device is not None and not self.gpu_device.strip():
            raise ValueError(
                "gpu_device must be a non-empty device string when set."
            )
        return self
