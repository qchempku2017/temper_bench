"""Defines persisted schemas for frame references and train, validation, and test splits. These models store references and split metadata, not atomic structures or descriptors."""
from __future__ import annotations

import math
from typing import List, Literal, Tuple, ClassVar

from pydantic import field_validator, model_validator, Field

from src.temper.schemas.utils import validate_relative_extxyz_path
from src.temper.splitting.quests_adapter import QuestsAdapterConfig

from src.temper.schemas.base import JsonIOModel


class FrameReference(JsonIOModel):
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


class EntropyProfilePoint(JsonIOModel):
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


class EntropyProfile(JsonIOModel):
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


class TrainValSplitTrajectory(JsonIOModel):
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
            this trajectory. Can be either ``"random"`` or ``"quests"``.
        seed (int | None): Random seed when partitioning train and val.
            Required for both methods as they both contain random initialization.
            And ``"random"`` further uses it at every incremental selection step.
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

        if self.seed is None:
            raise ValueError(
                "Require a non-None seed."
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
            for i, size in enumerate(profile_sizes):
                if i > 0 and size <= profile_sizes[i - 1]:
                    raise ValueError(
                        "entropy_profile points must be strictly increasing in "
                        f"training_size; index {i} has {size} after {profile_sizes[i - 1]}."
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


class SplitGroup(JsonIOModel):
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
            only. Does not include extra cross tests from other groups!
            Merging of extra cross tests is done by higher-level callers.
        extra_tested_groups (List(str)): Name of extra groups that should be tested
            on the model trained from the current group's split.
            Should belong to the same domain and same grouping strategy as the
            current group for correct reference.
        test_ratio (float): Requested ratio of the test set to the total
            dataset size. Must be in ``(0, 1)``.
        trainval_test_split_seed (int): Random seed used for the train+validation vs test
            partition (always performed at random).
        train_val_split_trajectory (TrainValSplitTrajectory): The single
            trajectory produced for this schema's splitting method.
        repeat_id (int): The index of repeated split on the same dataset.
        quests_adapter_config (QuestsAdapterConfig | None): Provenance of the QUESTS
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
    extra_tested_groups: List[str]
    test_ratio: float
    trainval_test_split_seed: int
    train_val_split_trajectory: TrainValSplitTrajectory
    repeat_id: int = 0
    quests_adapter_config: "QuestsAdapterConfig | None" = None

    @field_validator("test_ratio")
    @classmethod
    def validate_test_ratio(cls, value: float) -> float:
        """Reject test ratios outside ``(0, 1)``."""
        if not 0.0 < value < 1.0:
            raise ValueError(f"test_ratio must be in (0, 1), got {value}.")
        return value

    @field_validator("repeat_id")
    @classmethod
    def validate_repeat_id(cls, value: int) -> int:
        """Reject negative repeat IDs."""
        if value < 0:
            raise ValueError(f"repeat_id must be non-negative, got {value}.")
        return value

    @model_validator(mode="after")
    def validate_split_result(self) -> "SplitGroup":
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
