"""Atomic pairings of benchmark datasets and MLFF specifications."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Literal
from uuid import UUID

from pydantic import ConfigDict

from temper.schemas.base import ManagedIdentityModel
from temper.schemas.mlff_spec import MLFFSpec
from temper.schemas.train_unit import TrainingUnit


_MLFF_TRAIN_BUNDLE_ID_NAMESPACE = UUID("cb75bc42-06f5-5eec-8d38-a711f707d4c2")


def _nested_identity(value: Any) -> str:
    """Reduce a nested persisted record to its managed identity."""
    if isinstance(value, TrainingUnit):
        if value.training_unit_id is None:
            raise ValueError("Nested TrainingUnit identity has not been initialized.")
        return str(value.training_unit_id)
    if isinstance(value, MLFFSpec):
        if value.mlff_spec_id is None:
            raise ValueError("Nested MLFFSpec identity has not been initialized.")
        return str(value.mlff_spec_id)
    return str(value)


class MLFFTrainBundle(ManagedIdentityModel):
    """Pair one exported data unit with one MLFF recipe.

    Constructing this object does not copy data or run third-party software.
    write_submit_folder performs the explicit copy and creates the
    package-specific training files plus the common evaluation runtime.

    Parameters
    ----------
    training_unit : TrainingUnit
        Exported train/validation/test dataset references.
    mlff_spec : MLFFSpec
        Model, implementation, training, and testing recipe.
    mlff_train_bundle_id : UUID or None
        Stored deterministic identity derived from the two nested identities.
    """

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    _IDENTITY_FIELD_NAME: ClassVar[str] = "mlff_train_bundle_id"
    _IDENTITY_SOURCE_FIELDS: ClassVar[tuple[str, ...]] = (
        "training_unit",
        "mlff_spec",
    )
    _IDENTITY_SOURCE_NORMALIZERS: ClassVar[dict[str, Any]] = {
        "training_unit": _nested_identity,
        "mlff_spec": _nested_identity,
    }
    _IDENTITY_NAMESPACE: ClassVar[UUID] = _MLFF_TRAIN_BUNDLE_ID_NAMESPACE
    _IDENTITY_SCHEMA: ClassVar[str] = "temper.mlff-train-bundle.v2"
    _IDENTITY_LABEL: ClassVar[str] = "MLFF train bundle"

    training_unit: TrainingUnit
    mlff_spec: MLFFSpec
    mlff_train_bundle_id: UUID | None = None

    @property
    def unit_type(self) -> Literal["finetune", "zeroshot"]:
        """Return the mode implied by the nested TrainingUnit."""
        return self.training_unit.unit_type

    def _validate_before_identity(self) -> None:
        """Reject a fine-tuning dataset paired with a test-only recipe."""
        if self.unit_type == "finetune" and self.mlff_spec.training is None:
            raise ValueError(
                "Fine-tuning TrainingUnit requires non-None MLFF training parameters."
            )

    def write_submit_folder(self, target_dir: str | Path | None = None) -> Path:
        """Copy all inputs into a self-contained local submit directory.

        Parameters
        ----------
        target_dir : str, pathlib.Path, or None, optional
            New directory to create. It must not already exist. When omitted,
            a caller-owned temporary directory is created and returned.

        Returns
        -------
        pathlib.Path
            Created submit directory containing ordinary copied files.

        Raises
        ------
        FileExistsError
            If target_dir already exists.
        ValueError
            If the bundle is inconsistent, a referenced dataset is invalid, or
            a pretrained artifact no longer matches its recorded hash.
        OSError
            If an input cannot be read or the destination cannot be written.
        """
        from temper.mlff.bundle_writers import _write_submit_folder

        return _write_submit_folder(self, target_dir)


__all__ = ["MLFFTrainBundle"]
