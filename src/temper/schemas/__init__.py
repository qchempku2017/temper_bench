"""Public persisted schemas for TEMPER data and local MLFF
experiments."""

from temper.schemas.artifact import LocalArtifactRef
from temper.schemas.mlff_spec import (
    MLFFFileLayout,
    MLFFImplementation,
    MLFFSpec,
    PretrainedMLFFSpec,
)
from temper.schemas.mlff_train_bundle import MLFFTrainBundle, MLFFBundleFiles
from temper.schemas.mlff_experiment_result import (
    MLFFExperimentResult, MLFFTestResult, MLFFTestManifest, MLFFExperimentManifest,
)
from temper.schemas.train_unit import TrainingUnit, load_training_units

__all__ = [
    "GroupedDomain",
    "LocalArtifactRef",
    "MLFFFileLayout",
    "MLFFImplementation",
    "MLFFSpec",
    "MLFFTestManifest",
    "MLFFTrainBundle",
    "MLFFBundleFiles",
    "PretrainedMLFFSpec",
    "MLFFExperimentManifest",
    "MLFFExperimentResult",
    "MLFFTestResult",
    "SplitGroup",
    "TrainingUnit",
    "load_training_units",
]


def __getattr__(name):
    """Keep preprocessing schemas out of runtime imports."""
    if name == "GroupedDomain":
        from temper.schemas.group import GroupedDomain

        return GroupedDomain
    if name == "SplitGroup":
        from temper.schemas.split import SplitGroup

        return SplitGroup
    raise AttributeError(name)
