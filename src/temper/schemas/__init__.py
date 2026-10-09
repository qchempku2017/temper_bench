"""Public persisted schemas for TEMPER data and local MLFF experiments."""
from temper.schemas.mlff_test_result import MLFFTestResult
from temper.schemas.artifact import LocalArtifactRef
from temper.schemas.mlff_spec import (
    MLFFImplementation,
    MLFFSpec,
    PretrainedMLFFSpec,
)
from temper.schemas.mlff_train_bundle import MLFFTrainBundle, MLFFBundleFiles
from temper.schemas.train_unit import TrainingUnit


__all__ = [
    "GroupedDomain",
    "LocalArtifactRef",
    "MLFFImplementation",
    "MLFFSpec",
    "MLFFTestResult",
    "MLFFTrainBundle",
    "MLFFBundleFiles",
    "PretrainedMLFFSpec",
    "SplitGroup",
    "TrainingUnit",
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
