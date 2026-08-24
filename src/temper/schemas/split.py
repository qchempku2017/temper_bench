"""Defines persisted schemas for frame references and train, validation, and test splits. These models store references and split metadata, not atomic structures or descriptors."""
from __future__ import annotations

from pathlib import Path
from typing import List, Literal, Tuple, ClassVar, Any
from uuid import UUID

import numpy as np
from pydantic import (
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

from temper.schemas.base import ManagedIdentityModel, MSONableModel
from temper.schemas.quests_adapter import QuestsAdapterConfig
from temper.utils.defaults import (
    DEFAULT_DATA_DIR,
    DEFAULT_SPLIT_RESULTS_DIR,
    DEFAULT_SPLIT_REPEATS,
    DEFAULT_TEST_RATIO,
    DEFAULT_TRAIN_RATIOS,
    DEFAULT_MAX_N_TRAIN
)
from temper.schemas.entropy import EntropyProfile
from temper.schemas.frame_refrence import FrameReference


_SPLIT_ID_NAMESPACE = UUID("c04a0d94-c7f1-5ef4-b45e-1f9de274c99d")


class SplitConfig(MSONableModel):
    """Configuration for splitting a grouped domain into train/validation/test sets.

    Attributes
    ----------
    root_path: Path | str
        Path to the root directory of the dataset. Will always be expanded and
        resolved before assignment.
    output_path: Path | str
        Directory in which split artifacts and exported datasets are written.
    domains: list[str] | None
        Domain directory names to split. ``None`` discovers every domain beneath
        ``root_path`` that contains the configured metadata filename.
    split_repeats : int
        Number of times to repeat the split. See default in src.temper.utils.defaults.
    seed: int
       Random seed for reproducing the split. Must record a non-negative int value.
       If not provided, will generate a random seed.
    trainval_test_split_seeds : list[int] | None
        List of seeds for splitting the trainval and test sets. Not recommended to
        set directly as these will be automatically generated from the provided ``seed``.
    train_val_split_seeds : list[int] | None
        List of seeds for splitting the train and validation sets. Not recommended to
        set directly as these will be automatically generated from the provided ``seed``.
    test_ratio : float
        Ratio of the test set to the total number of frames.
        See default in src.temper.utils.defaults.
    requested_train_ratios : list[float] | None
        List of requested train ratios. If not provided, will use
        ``DEFAULT_TRAIN_RATIOS``. See default in src.temper.utils.defaults.
    max_train_size : float
        Maximum number of training frames. If not provided, will use
        ``DEFAULT_MAX_N_TRAIN``.
        See default in src.temper.utils.defaults.
    train_val_split_method : Literal["random", "quests"]
        Method to use for splitting the train and validation sets.
        Must be either ``"random"`` or ``"quests"``.
    quests_adapter_config: QuestsAdapterConfig
        Configuration for the Quests adapter. If not provided, will use
        all default settings. See src.temper.splitting.quests_adapter ``QuestsAdapterConfig``
        for details.
    write_validation: bool
        Whether to materialize validation datasets.
    write_extra_tests: bool
        Whether to materialize configured cross-test datasets.
    """
    model_config = ConfigDict(frozen=True)

    root_path: Path = Field(
        default=DEFAULT_DATA_DIR,
        validate_default=True,
    )  # Explicitly request conversion of str to Path.
    output_path: Path = Field(
        default=DEFAULT_SPLIT_RESULTS_DIR,
        validate_default=True,
    )
    domains: list[str] | None = None
    split_repeats: int = DEFAULT_SPLIT_REPEATS

    seed: int = Field(ge=0, default_factory=lambda: int(np.random.randint(0, 2**32)))
    trainval_test_split_seeds: list[int] = Field(default_factory=list)
    train_val_split_seeds: list[int] = Field(default_factory=list)

    test_ratio: float = DEFAULT_TEST_RATIO

    requested_train_ratios: list[float] = Field(
        default_factory=lambda: list(DEFAULT_TRAIN_RATIOS)
    )

    max_train_size: int = DEFAULT_MAX_N_TRAIN

    train_val_split_method: Literal["random", "quests"] = "quests"

    quests_adapter_config: QuestsAdapterConfig = Field(
        default_factory=QuestsAdapterConfig
    )

    write_validation: bool = False
    write_extra_tests: bool = True

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-compatible MSON data, including paths as strings."""
        return self.model_dump(mode="json")

    @field_serializer("root_path", "output_path")
    def serialize_paths(self, value: Path) -> str:
        """Serialize configuration paths portably instead of as Monty path objects."""
        return str(value)

    @field_validator("root_path", "output_path", mode="before")
    @classmethod
    def load_monty_paths(cls, value: Any) -> Any:
        """Accept path dictionaries written by earlier Monty/Pydantic encoders."""
        if isinstance(value, dict) and value.get("@module") == "pathlib":
            return value.get("string", value)
        return value

    @model_validator(mode="before")
    @classmethod
    def fill_defaults(cls, data: Any) -> Any:
        """Fill defaults that depend on other input fields."""
        if data is None:
            data = {}

        if not isinstance(data, dict):
            return data

        data = dict(data)

        split_repeats = data.get(
            "split_repeats",
            DEFAULT_SPLIT_REPEATS,
        )

        if data.get("seed") is None:
            data["seed"] = int(np.random.default_rng().integers(0, 2**32, dtype=np.uint32))
        rng = np.random.default_rng(data["seed"])

        if data.get("trainval_test_split_seeds") is None:
            data["trainval_test_split_seeds"] = [
                int(rng.integers(0, 2**32, dtype=np.uint32))
                for _ in range(split_repeats)
            ]

        if data.get("train_val_split_seeds") is None:
            data["train_val_split_seeds"] = [
                int(rng.integers(0, 2**32, dtype=np.uint32))
                for _ in range(split_repeats)
            ]

        # Check length of seeds are valid.
        if len(data["trainval_test_split_seeds"]) != split_repeats:
            raise ValueError(
                f"Length of trainval_test_split_seeds ({len(data['trainval_test_split_seeds'])}) "
                f"does not match split_repeats ({split_repeats})."
            )
        if len(data["train_val_split_seeds"]) != split_repeats:
            raise ValueError(
                f"Length of train_val_split_seeds ({len(data['train_val_split_seeds'])}) "
                f"does not match split_repeats ({split_repeats})."
            )

        return data


class TrainValSplitTrajectory(MSONableModel):
    """A single train/validation trajectory for one splitting method.

    Both the ``"random"`` method and the ``"quests"`` (QUESTS
    maximum-entropy) method use this single output convention: an ordered
    ``selected_frames`` tuple plus ordered ``additional_trainval_frames`` form
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
        requested_train_sizes (tuple[int, ...]): Strictly increasing requested
            training sizes. Each is the size of a nested training prefix.
        selected_frames (tuple[temper.schemas.frame_refrence.FrameReference, ...]): Ordered selected-frame tuple;
            the first ``s`` frames form the training set of size ``s``.
        additional_trainval_frames (tuple[temper.schemas.frame_refrence.FrameReference, ...]): Additional frames in
            the original train+val dataset but not selected in ``selected_frames``.
            Will go into the validation set. Empty tuple by default.
        entropy_profile (temper.schemas.entropy.EntropyProfile | None): QUESTS maximum-entropy
            profile with one point per incremental selection step.
            ``None`` before evaluation; a complete profile once evaluated.
    """
    #: Supported train-val splitting methods.
    SUPPORTED_TRAIN_VAL_SPLIT_METHODS: ClassVar[Tuple[str, ...]] = ("random", "quests")
    model_config = ConfigDict(validate_assignment=True)

    method: Literal["random", "quests"]
    seed: int | None = None

    requested_train_sizes: Tuple[int, ...]
    selected_frames: Tuple[FrameReference, ...]
    additional_trainval_frames: Tuple[FrameReference, ...] = Field(
        default_factory=tuple
    )
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
        return list(
            self.selected_frames[
                :self.requested_train_sizes[requested_train_size_index]
            ]
        )

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
        return [
            *self.selected_frames[size:],
            *self.additional_trainval_frames,
        ]


class SplitGroup(ManagedIdentityModel):
    """Persisted top-level result of splitting a data group.

    This is the persisted (reference-only) result schema for splitting a
    single data group into train/validation/test. It stores provenance (the
    identity of the domain/group and split configuration), a singular
    :class:`TrainValSplitTrajectory`, and the test references. The trajectory
    itself contains the complete train+validation inventory.

    Only frame references are stored; structures and descriptors are never
    stored.

    ``split_id`` is a stored, system-managed UUIDv5 identity for the complete
    persisted split definition. It is computed for legacy objects without an
    ID, verified once when an ID is loaded, and regenerated after validated
    reassignment of an identity-defining field.

    Identity-bearing collections are stored as tuples. To change trajectory
    membership on an existing split, build a replacement trajectory and
    assign it to ``train_val_split_trajectory`` so the parent can validate the
    change and regenerate ``split_id``.

    Attributes:
        domain (str): Name of the data domain.
        grouping_strategy (str): Name of the grouping strategy used.
        group_name (str): Name of the group that was split.
        test_set (tuple[FrameReference, ...]): Test references within the current group
            only. Does not include extra cross tests from other groups!
            Merging of extra cross tests is done by higher-level callers.
        extra_tested_groups (tuple[str, ...]): Names of extra groups that should be tested
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
        split_id (UUID | None): Stored, system-managed identity. ``None`` is
            accepted only as construction input for legacy records and is
            populated before a valid model is returned.
    """
    _IDENTITY_FIELD_NAME: ClassVar[str] = "split_id"
    _IDENTITY_SOURCE_FIELDS: ClassVar[tuple[str, ...]] = (
        "domain",
        "grouping_strategy",
        "group_name",
        "test_set",
        "extra_tested_groups",
        "test_ratio",
        "trainval_test_split_seed",
        "train_val_split_trajectory.method",
        "train_val_split_trajectory.seed",
        "train_val_split_trajectory.requested_train_sizes",
        "train_val_split_trajectory.selected_frames",
        "train_val_split_trajectory.additional_trainval_frames",
        "repeat_id",
        "quests_adapter_config",
    )
    _IDENTITY_SOURCE_NORMALIZERS: ClassVar[dict[str, Any]] = {
        "extra_tested_groups": lambda groups: sorted(set(groups)),
    }
    _IDENTITY_NAMESPACE: ClassVar[UUID] = _SPLIT_ID_NAMESPACE
    _IDENTITY_SCHEMA: ClassVar[str] = "temper.split-group.v2"
    _IDENTITY_LABEL: ClassVar[str] = "split"

    domain: str
    grouping_strategy: str
    group_name: str
    test_set: Tuple[FrameReference, ...]
    extra_tested_groups: Tuple[str, ...]
    test_ratio: float
    trainval_test_split_seed: int
    train_val_split_trajectory: TrainValSplitTrajectory
    repeat_id: int = 0
    quests_adapter_config: "QuestsAdapterConfig | None" = None
    split_id: UUID | None = None

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

    def _validate_before_identity(self) -> None:
        """Validate internal consistency before finalizing ``split_id``."""
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
