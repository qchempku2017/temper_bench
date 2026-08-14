"""Dataset splitting methods for the temper benchmark.

Implements the shared deterministic reference/partition logic, the
reproducible random splitting method, the QUESTS maximum-entropy method, and
the dataset-reconstruction/extxyz-export utilities. All methods reuse the same
output convention (see :class:`TrainValSplitTrajectory` and
:class:`SplitDataSchema`).
"""
from src.temper.schemas.split import FrameReference
from src.temper.splitting.common import (
    get_references_from_frames,
    get_requested_train_sizes_from_ratios,
    partition_trainval_test,
)
from src.temper.splitting.io import (
    SourceResolver,
    build_export_filename,
    load_frames_from_references,
    load_frames_test,
    load_frames_train_validation,
    write_all_sets_in_split_schema_to_extxyz,
    write_single_dataset_to_extxyz,
)
from src.temper.splitting.quests import (
    FrameDescriptors,
    QuestsNumericalError,
    QuestsObjective,
    QuestsSplitConfig,
    QuestsUnavailableError,
    build_frame_descriptors,
    evaluate_entropy_profile,
    generate_quests_trajectory,
    populate_entropy_profile,
)
from src.temper.splitting.random import generate_random_trajectory

__all__ = [
    "FrameDescriptors",
    "QuestsNumericalError",
    "QuestsObjective",
    "QuestsSplitConfig",
    "QuestsUnavailableError",
    "SourceResolver",
    "build_export_filename",
    "build_frame_descriptors",
    "evaluate_entropy_profile",
    "get_references_from_frames",
    "generate_quests_trajectory",
    "generate_random_trajectory",
    "get_requested_train_sizes_from_ratios",
    "partition_trainval_test",
    "populate_entropy_profile",
    "load_frames_from_references",
    "load_frames_test",
    "load_frames_train_validation",
    "write_all_sets_in_split_schema_to_extxyz",
    "write_single_dataset_to_extxyz",
]
