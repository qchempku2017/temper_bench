from typing import List
from pathlib import Path
from pydantic import model_validator, ConfigDict, Field

from src.temper.schemas.base import JsonIOModel
from src.temper.utils.defaults import DEFAULT_TRAIN_UNITS_DIR


class TrainingUnit(JsonIOModel):
    """Schema to define a unit containing a training set, validation set (optional) and test sets.

    A unit belongs to:
        - A specific data domain
        - A specific grouping strategy (The level of GroupedDomain)
        - A specific group produced by the grouping strategy.
        - A specific train-val split method.
        - A specific repeat_id among independent train-val-test splits on the group using the method.
            (The level of SplitGroup)
        - A specific N_train (number of requested training structures) in one independent train-val-test split.

    The unit contains extxyz file paths containing labeled data:
        - A training set (a single file path)
        - A validation set (a single file path, optional)
        - A test set (a list of file paths, each corresponding to a tested group)

    Usually produced by src.temper.splitting.io ``write_all_sets_in_split_group_to_extxyz``.
    Can be used to reference files when prepraing for MLFF training.

    All attributes except ``root_path`` are NOT allowed to change after initialization, because file names
    will not change after written, but the root path can be moved to different locations.

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
        n_train: int
            Number of training structures.
        train_set : str
            Filename of the training set.
        test_sets : List[str]
            List of filenames of the test sets.
        val_set : str | None
            Filename of the validation set.
        root_path: Path
            Root path to the train, val and test files. Should be able to load from:
            rootpath / train_set, rootpath / val_set, rootpath / test_sets.
            Defaults to ``DEFAULT_SPLIT_DATA_DIR``. See src.temper.utils.defaults.
    """
    model_config = ConfigDict(
        validate_assignment=True,
    )

    domain: str = Field(frozen=True)
    grouping_strategy: str = Field(frozen=True)
    group_name: str = Field(frozen=True)
    method: str = Field(frozen=True)

    repeat_id: int = Field(
        ge=0,
        frozen=True,
    )

    n_train: int = Field(
        ge=1,
        frozen=True,
    )

    train_set: str = Field(frozen=True)

    test_sets: List[str] = Field(
        frozen=True,
    )

    val_set: str | None = Field(
        default=None,
        frozen=True,
    )

    root_path: Path = Field(
        default=DEFAULT_TRAIN_UNITS_DIR,
        validate_default=True,
    )

    @model_validator(mode="after")
    def validate_dataset_files(self) -> "TrainingUnit":
        """Validate that all referenced dataset files exist and are extxyz."""

        def _check_extxyz_file(f: str) -> None:
            file_path = self.root_path / f

            if file_path.suffix != ".extxyz":
                raise ValueError(
                    f"Dataset file must have .extxyz extension, got: "
                    f"{f}."
                )

            if not file_path.is_file():
                raise ValueError(
                    f"Dataset file does not exist: {file_path}."
                )

        _check_extxyz_file(self.train_set)

        if self.val_set is not None:
            _check_extxyz_file(self.val_set)

        for filename in self.test_sets:
            _check_extxyz_file(filename)

        return self
