"""Defines persisted benchmark data units and their extxyz references."""

from typing import Any, ClassVar, Literal
from pathlib import Path
from uuid import UUID

from pydantic import (
    Field,
    field_serializer,
    field_validator,
)

from temper.schemas.base import ManagedIdentityModel
from temper.utils.defaults import DEFAULT_SPLIT_RESULTS_DIR


_TRAINING_UNIT_ID_NAMESPACE = UUID("a219bd97-5b63-5dc1-8543-38d74c746ecf")


class TrainingUnit(ManagedIdentityModel):
    """Persisted datasets for one fine-tuning or zero-shot benchmark.

    ``unit_type`` is derived from dataset shape rather than stored separately:
    a unit with ``train_set`` is ``"finetune"`` and a unit without one is
    ``"zeroshot"``. Fine-tuning units require a non-empty training dataset.
    Zero-shot units contain no training or validation dataset and reuse the
    same evaluation datasets as fine-tuning units from their ``SplitGroup``.

    A unit belongs to:
        - A specific data domain
        - A specific grouping strategy (The level of GroupedDomain)
        - A specific group produced by the grouping strategy.
        - A specific train-val split method.
        - A specific repeat_id among independent train-val-test splits on the group using the method.
            (The level of SplitGroup)
        - Either a specific training-frame checkpoint or the zero-shot
          evaluation derived from one independent train-val-test split.

    The unit contains extxyz file paths containing labeled data:
        - A training set (a single file path, fine-tuning only)
        - A validation set (a single file path, optional and fine-tuning only)
        - A test set (a list of file paths, each corresponding to a tested group)

    Usually produced by src.temper.splitting.io ``write_all_sets_in_split_group_to_extxyz``.
    It describes benchmark input data; it does not train or evaluate an MLFF.

    Fields remain mutable through validated reassignment. The system-managed
    ``training_unit_id`` is stored, verified when loaded, and regenerated when
    an identity-defining field changes. ``root_path`` can be relocated without
    changing the identity.

    Attributes:
        domain: str
            Name of the data domain.
        grouping_strategy: str
            Name of the grouping strategy.
        group_name: str
            Name of the group.
        method: str
            Name of the train-val split method.
        repeat_id: int
            Repeat id of the independent train-val-test split.
        unit_type: Literal["finetune", "zeroshot"]
            Read-only mode derived from whether ``train_set`` is present.
        train_n_frames: int
            Number of frames in the training dataset, or zero for zero-shot.
        val_n_frames: int
            Number of frames in the exported validation dataset, or zero when
            no validation dataset is exported.
        test_n_frames: int
            Total number of frames across all exported test datasets.
        train_n_atoms: int
            Total number of atoms across all training frames, or zero for
            zero-shot.
        val_n_atoms: int
            Total number of atoms across all frames in the exported validation
            dataset, or zero when no validation dataset is exported.
        test_n_atoms: int
            Total number of atoms across all frames in all exported test
            datasets.
        split_id: UUID | None
            Identity of the SplitGroup that produced this unit. ``None`` is
            accepted for training-unit manifests written before split
            identities were introduced.
        training_unit_id: UUID | None
            Stored, system-managed identity. ``None`` is accepted only as
            construction input for legacy records and is populated before a
            valid model is returned.
        train_set : str | None
            Filename of the training set, or ``None`` for zero-shot.
        test_sets : tuple[str, ...]
            Filenames of the test sets.
        val_set : str | None
            Filename of the validation set.
        root_path: Path
            Root path to the train, val and test files. Should be able to load from:
            rootpath / domain / train_set, rootpath / domain / val_set,
            rootpath / domain / test_sets.
            Defaults to ``DEFAULT_SPLIT_RESULTS_DIR``. See
            src.temper.utils.defaults.
    """
    _IDENTITY_FIELD_NAME: ClassVar[str] = "training_unit_id"
    _IDENTITY_SOURCE_FIELDS: ClassVar[tuple[str, ...]] = (
        "split_id",
        "domain",
        "grouping_strategy",
        "group_name",
        "method",
        "repeat_id",
        "train_n_frames",
        "train_set",
        "test_sets",
        "val_set",
    )
    _IDENTITY_NAMESPACE: ClassVar[UUID] = _TRAINING_UNIT_ID_NAMESPACE
    _IDENTITY_SCHEMA: ClassVar[str] = "temper.training-unit.v2"
    _IDENTITY_LABEL: ClassVar[str] = "training-unit"

    domain: str
    grouping_strategy: str
    group_name: str
    method: str

    repeat_id: int = Field(
        ge=0,
    )

    train_n_frames: int = Field(
        ge=0,
    )
    val_n_frames: int = Field(
        ge=0,
    )
    test_n_frames: int = Field(
        ge=0,
    )

    train_n_atoms: int = Field(
        ge=0,
    )
    val_n_atoms: int = Field(
        ge=0,
    )
    test_n_atoms: int = Field(
        ge=0,
    )

    split_id: UUID | None = None

    train_set: str | None

    test_sets: tuple[str, ...]

    val_set: str | None = None

    root_path: Path = Field(
        default=DEFAULT_SPLIT_RESULTS_DIR,
        validate_default=True,
    )
    training_unit_id: UUID | None = None

    @property
    def unit_type(self) -> Literal["finetune", "zeroshot"]:
        """Return the benchmark mode implied by the training-set shape."""
        return "finetune" if self.train_set is not None else "zeroshot"

    @field_serializer("root_path")
    def serialize_root_path(self, value: Path) -> str:
        """Serialize the movable root as a portable path string."""
        return str(value)

    @field_serializer("split_id")
    def serialize_split_id(self, value: UUID | None) -> str | None:
        """Serialize the parent identity as a standard UUID string."""
        return None if value is None else str(value)

    @field_validator("root_path", mode="before")
    @classmethod
    def load_monty_root_path(cls, value: Any) -> Any:
        """Accept path dictionaries written by earlier Monty encoders."""
        if isinstance(value, dict) and value.get("@module") == "pathlib":
            return value.get("string", value)
        return value

    @field_validator("split_id", mode="before")
    @classmethod
    def load_monty_split_id(cls, value: Any) -> Any:
        """Accept UUID dictionaries written by Monty encoders."""
        if isinstance(value, dict) and value.get("@module") == "uuid":
            return value.get("string", value)
        return value

    def _validate_before_identity(self) -> None:
        """Validate dataset shape and files before finalizing identity."""

        if self.unit_type == "finetune":
            if self.train_n_frames <= 0:
                raise ValueError(
                    "Fine-tuning TrainingUnit requires train_n_frames > 0."
                )
            if self.train_n_atoms <= 0:
                raise ValueError(
                    "Fine-tuning TrainingUnit requires train_n_atoms > 0."
                )
        else:
            if self.train_n_frames != 0:
                raise ValueError(
                    "Zero-shot TrainingUnit requires train_n_frames == 0."
                )
            if self.train_n_atoms != 0:
                raise ValueError(
                    "Zero-shot TrainingUnit requires train_n_atoms == 0."
                )
            if self.val_set is not None:
                raise ValueError(
                    "Zero-shot TrainingUnit cannot reference a validation set."
                )
            if self.val_n_frames != 0:
                raise ValueError(
                    "Zero-shot TrainingUnit requires val_n_frames == 0."
                )
            if self.val_n_atoms != 0:
                raise ValueError(
                    "Zero-shot TrainingUnit requires val_n_atoms == 0."
                )

        def _check_extxyz_file(f: str) -> None:
            file_path = self.root_path / self.domain / f

            if file_path.suffix != ".extxyz":
                raise ValueError(
                    f"Dataset file must have .extxyz extension, got: "
                    f"{f}."
                )

            if not file_path.is_file():
                raise ValueError(
                    f"Dataset file does not exist: {file_path}."
                )

        if self.train_set is not None:
            _check_extxyz_file(self.train_set)

        if self.val_set is not None:
            _check_extxyz_file(self.val_set)

        for filename in self.test_sets:
            _check_extxyz_file(filename)
