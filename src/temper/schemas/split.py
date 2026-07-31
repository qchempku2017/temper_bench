"""Schema to define the data-splitting in a specific fine-tuning experiment."""
from pydantic import BaseModel, model_validator, field_validator
from typing import Dict, List


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