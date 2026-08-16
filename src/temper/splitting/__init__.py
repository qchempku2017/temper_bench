"""Dataset splitting methods for the temper benchmark.

Implements the shared deterministic reference/partition logic, the
reproducible random splitting method, the QUESTS maximum-entropy method, and
the dataset-reconstruction/extxyz-export utilities. All methods reuse the same
output convention (see :class:`TrainValSplitTrajectory` and
:class:`SplitDataSchema`).
"""
from src.temper.splitting.utils import (
    get_references_from_frames,
    get_requested_train_sizes_from_ratios,
)
from temper.splitting.split import partition_trainval_test
from src.temper.splitting.io import (
    FrameReferenceResolver,
    build_export_filename,
    load_frames_from_references,
    load_frames_test,
    load_frames_train_validation,
    write_all_sets_in_split_group_to_extxyz,
    write_single_dataset_to_extxyz,
)
from src.temper.splitting.selectors import (
    QuestsIndicesSelector,
    RandomIndicesSelector
)
from temper.splitting.quests_adapter import (
    QuestsUnavailableError,
    QuestsNumericalError,
    QuestsDescriptorsStorage,
    QuestsAdapterConfig,
    QuestsAdapter,
    compute_information_gain_per_candidate_frame,
    compute_total_entropy_of_selected_frames
)

__all__ = [
    "get_references_from_frames",
    "get_requested_train_sizes_from_ratios",
    "partition_trainval_test",
    "FrameReferenceResolver",
    "build_export_filename",
    "load_frames_from_references",
    "load_frames_test",
    "load_frames_train_validation",
    "write_all_sets_in_split_group_to_extxyz",
    "write_single_dataset_to_extxyz",
    "QuestsIndicesSelector",
    "RandomIndicesSelector",
    "QuestsUnavailableError",
    "QuestsNumericalError",
    "QuestsDescriptorsStorage",
    "QuestsAdapterConfig",
    "QuestsAdapter",
    "compute_information_gain_per_candidate_frame",
    "compute_total_entropy_of_selected_frames"
]
