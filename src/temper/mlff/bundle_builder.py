"""High-level creation of atomic MLFFTrainBundle Cartesian products."""

from __future__ import annotations

from collections.abc import Iterable

from temper.schemas.mlff_spec import MLFFSpec
from temper.schemas.mlff_train_bundle import MLFFTrainBundle
from temper.schemas.train_unit import TrainingUnit


def build_mlff_train_bundles(
    *,
    training_units: Iterable[TrainingUnit],
    mlff_specs: Iterable[MLFFSpec],
) -> list[MLFFTrainBundle]:
    """Build the ordered Cartesian product of data units and MLFF recipes.

    Parameters
    ----------
    training_units : Iterable[TrainingUnit]
        Exported benchmark units in desired output order.
    mlff_specs : Iterable[MLFFSpec]
        MLFF recipes in desired output order.

    Returns
    -------
    list[MLFFTrainBundle]
        One atomic bundle for each unit/specification pair, with units as the
        outer loop and specifications as the inner loop.
    """
    units = tuple(training_units)
    specs = tuple(mlff_specs)
    return [
        MLFFTrainBundle(training_unit=unit, mlff_spec=spec)
        for unit in units
        for spec in specs
    ]


__all__ = ["build_mlff_train_bundles"]
