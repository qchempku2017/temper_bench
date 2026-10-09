"""Atomic pairings of benchmark datasets and MLFF specifications."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, ClassVar, Literal
from uuid import UUID

from pydantic import ConfigDict, field_validator
from temper._version import __version__

from temper.schemas.base import ManagedIdentityModel, MSONableModel
from temper.schemas.utils import validate_submit_relative_path
from temper.utils.defaults import (
    DEFAULT_MLFF_ARTIFACTS_DIR, DEFAULT_MLFF_OUTPUTS_DIR, DEFAULT_MLFF_TRAINING_DIR,
)
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


class MLFFBundleFiles(MSONableModel):
    """Portable input mapping and output directories, resolved by the writer.

    Dataset keys retain the original TrainingUnit filenames. All values are
    relative to bundle.json; local source paths remain provenance only.

    Attributes
    ----------
    datasets : dict[str, str]
        Original dataset filenames mapped to copied paths inside the bundle.
    models : dict[str, str]
        Pretrained artifact keys, such as ``model``, mapped to copied paths.
    training_stress : bool or None
        Whether train/validation data have stress labels; None for zero-shot.
    test_stress : list[bool]
        Stress availability in the same order as TrainingUnit.test_sets.
    trained_model_filename : str
        Backend's trained output filename, relative to artifacts_dir.
    training_dir : str
        Directory for native training configs and working files.
    artifacts_dir : str
        Directory for trained models and retained checkpoints.
    outputs_dir : str
        Directory for logs, prediction arrays, and result metadata.
    """

    model_config = ConfigDict(extra="forbid")

    datasets: dict[str, str]
    models: dict[str, str]
    training_stress: bool | None
    test_stress: list[bool]
    trained_model_filename: str
    training_dir: str = DEFAULT_MLFF_TRAINING_DIR
    artifacts_dir: str = DEFAULT_MLFF_ARTIFACTS_DIR
    outputs_dir: str = DEFAULT_MLFF_OUTPUTS_DIR

    @field_validator("datasets", "models")
    @classmethod
    def validate_paths(cls, value: dict[str, str]) -> dict[str, str]:
        for path in value.values():
            validate_submit_relative_path(path, label="bundle input")
        return value

    @field_validator("training_dir", "artifacts_dir", "outputs_dir", "trained_model_filename")
    @classmethod
    def validate_directory(cls, value: str) -> str:
        validate_submit_relative_path(value, label="bundle directory")
        return value


class MLFFTrainBundle(ManagedIdentityModel):
    """Pair one exported data unit with one MLFF recipe.

    Constructing this object does not copy data or run third-party software.
    write_submit_folder copies inputs and serializes a portable file mapping.
    The installed runtime creates native training files and evaluates models.
    Packaging fields do not change the experiment's deterministic identity.

    Attributes
    ----------
    training_unit : TrainingUnit
        Exported train/validation/test dataset references.
    mlff_spec : MLFFSpec
        Model, implementation, training, and testing recipe.
    mlff_train_bundle_id : UUID or None
        Stored deterministic identity derived from the two nested identities.
    schema_version : Literal[1]
        Bundle format version used when loading the serialized record.
    temper_version : str
        TEMPER release that wrote the bundle, retained for provenance.
        The runtime warns if its release differs.
    files : MLFFBundleFiles or None
        Packaged paths and label availability. The writer populates this on
        the serialized copy; a newly constructed local bundle has None.
    unit_type : Literal["finetune", "zeroshot"]
        Read-only mode derived from training_unit.
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
    schema_version: Literal[1] = 1
    temper_version: str = __version__
    files: MLFFBundleFiles | None = None

    @property
    def unit_type(self) -> Literal["finetune", "zeroshot"]:
        """Return the mode implied by the nested TrainingUnit."""
        return self.training_unit.unit_type

    def _validate_before_identity(self) -> None:
        """Reject a fine-tuning dataset paired with a test-only recipe."""
        if self.unit_type == "finetune" and self.mlff_spec.training_parameters is None:
            raise ValueError(
                "Fine-tuning TrainingUnit requires non-None MLFF training parameters."
            )

    def test_config(self) -> dict[str, Any]:
        """Describe evaluation using the recipe and packaged file mapping.

        Returns
        -------
        dict[str, Any]
            Calculator settings, selected model, per-dataset properties and
            output paths, and package requirements. Fine-tuning evaluates the
            trained model; zero-shot evaluates the pretrained model.

        Raises
        ------
        ValueError
            If the bundle has not been packaged yet.
        """
        if self.files is None:
            raise ValueError('Bundle has not been packaged; call write_submit_folder first.')
        model_path = (
            f"{self.files.artifacts_dir}/{self.files.trained_model_filename}"
            if self.unit_type == "finetune"
            else self.files.models["model"]
        )
        datasets = []
        for index, (filename, has_stress) in enumerate(
            zip(self.training_unit.test_sets, self.files.test_stress, strict=True)
        ):
            stem = f"test_{index:03d}"
            properties = ["energy", "forces"]
            if has_stress:
                properties.append("stress")
            datasets.append(
                {
                    "id": stem,
                    "path": self.files.datasets[filename],
                    "source_domain": self.training_unit.domain,
                    "source_filename": filename,
                    "properties": properties,
                    "output": f"{self.files.outputs_dir}/{stem}.npz",
                    "metadata_output": f"{self.files.outputs_dir}/{stem}.json",
                }
            )
        return {
            "schema_version": 2,
            "calculator": {
                "identifier": self.mlff_spec.mlff_type,
                "parameters": deepcopy(self.mlff_spec.testing_parameters),
            },
            "model": model_path,
            "test_datasets": datasets,
            "summary_output": f"{self.files.outputs_dir}/test_summary.json",
            "package_requirements": [
                {"name": item.name, "version": item.version}
                for item in self.mlff_spec.implementations
                if item.kind == "python_distribution"
            ],
        }

    def write_submit_folder(self, target_dir: str | Path | None = None) -> Path:
        """Copy inputs and a portable bundle.json into a new submit directory.

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
        from temper.mlff.bundle_writers import write_submit_folder

        return write_submit_folder(self, target_dir)


__all__ = ["MLFFTrainBundle", "MLFFBundleFiles"]
