"""Atomic pairings of benchmark datasets and MLFF specifications."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, ClassVar, Literal
from uuid import UUID

from pydantic import ConfigDict, Field, field_validator, model_validator
from temper._version import __version__

from temper.schemas.base import ManagedIdentityModel, MSONableModel
from temper.schemas.utils import validate_submit_relative_path
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
    """Record where a packaged experiment reads and writes its files.

    Every path is relative to the directory containing bundle.json, including
    paths to directories. These are planned locations: packaging copies input
    datasets and pretrained weights. Runtime preparation writes native configs
    and scripts; execution creates converted datasets, resolved backend configs
    and results.
    A recorded output therefore need not exist yet.

    Attributes
    ----------
    pretrained_model : str
        Copied pretrained weights. Fine-tuning starts from this file; zero-shot
        evaluation uses it directly. Execution does not modify it.
    training_input_files : dict[str, str]
        Training inputs named by purpose: train, optional validation, config,
        script, and any backend-specific converted datasets. Empty for zero-shot.
    training_output_files : dict[str, str]
        Training outputs named by purpose, including trained_model and
        training_log. Other entries describe backend configs, checkpoints,
        metrics, or directories containing them. Empty for zero-shot.
    test_input_files : dict[str, str]
        Test datasets keyed as test_000, test_001, and so on in TrainingUnit
        order, plus config for the generated test_config.json.
    test_output_files : dict[str, str]
        Prediction arrays and metadata keyed as test_000_predictions and
        test_000_metadata, plus summary for the combined test summary.
    evaluation_model : str or None
        Temporary export made for evaluation, such as a frozen DeepMD model.
        The runner removes it after evaluation. None means evaluation can use
        the pretrained or fine-tuned model directly.
    training_stress : bool or None
        Whether training and validation data have stress labels. None for zero-shot.
    test_stress : list[bool]
        Stress availability in the same order as TrainingUnit.test_sets.
    """

    model_config = ConfigDict(extra="forbid")

    pretrained_model: str
    training_input_files: dict[str, str] = Field(default_factory=dict)
    training_output_files: dict[str, str] = Field(default_factory=dict)
    test_input_files: dict[str, str]
    test_output_files: dict[str, str]
    evaluation_model: str | None = None
    training_stress: bool | None
    test_stress: list[bool]

    @field_validator("training_input_files", "training_output_files", "test_input_files", "test_output_files")
    @classmethod
    def validate_paths(cls, value: dict[str, str]) -> dict[str, str]:
        for path in value.values():
            validate_submit_relative_path(path, label="bundle file")
        return value

    @field_validator("pretrained_model", "evaluation_model")
    @classmethod
    def validate_model_path(cls, value: str | None) -> str | None:
        if value is not None:
            validate_submit_relative_path(value, label="bundle model")
        return value

    @model_validator(mode="after")
    def separate_temporary_model(self) -> MLFFBundleFiles:
        """Prevent evaluation cleanup from deleting an input or training result."""
        permanent_paths = {
            self.pretrained_model, *self.training_input_files.values(),
            *self.training_output_files.values(), *self.test_input_files.values(),
            *self.test_output_files.values(),
        }
        if self.evaluation_model in permanent_paths:
            raise ValueError("The temporary evaluation model must have its own path.")
        return self


class MLFFTrainBundle(ManagedIdentityModel):
    """Pair one exported training unit with one MLFF specification.

    Constructing this object does not copy data or run third-party software.
    write_submit_folder copies inputs and serializes a portable file mapping.
    The installed runtime writes missing configs and scripts, prepares
    backend-dependent data, trains when requested, and evaluates models.
    Packaging fields do not change the experiment's deterministic identity.

    Parameters
    ----------
    **data : object
        Values for the attributes below. New local bundles normally provide
        training_unit and mlff_spec only; files remains absent until packaging.

    Attributes
    ----------
    training_unit : TrainingUnit
        Exported train/validation/test dataset references.
    mlff_spec : MLFFSpec
        Specification of the model, software implementation, training, and testing settings.
    mlff_train_bundle_id : UUID or None
        Stored deterministic identity derived from the two nested identities.
    schema_version : Literal[2]
        Bundle format version used when loading the serialized record.
    temper_version : str
        TEMPER release that wrote the bundle, retained for provenance.
        The runtime warns if its release differs.
    files : MLFFBundleFiles or None
        Locations of copied datasets and pretrained weights, plus planned training
        and testing files. Every path is relative to the bundle directory.
        A newly constructed local bundle has None, and therefore
        when encountering a None value, the runtime cannot perform
        fine-tuning or evaluation.
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
    schema_version: Literal[2] = 2
    temper_version: str = __version__
    files: MLFFBundleFiles | None = None

    @property
    def unit_type(self) -> Literal["finetune", "zeroshot"]:
        """Return the mode implied by the nested TrainingUnit."""
        return self.training_unit.unit_type

    def _validate_before_identity(self) -> None:
        """Reject a fine-tuning dataset paired with a test-only specification."""
        if self.unit_type == "finetune" and self.mlff_spec.training_parameters is None:
            raise ValueError(
                "Fine-tuning TrainingUnit requires non-None MLFF training parameters."
            )

    def test_config(self) -> dict[str, Any]:
        """Describe evaluation using the bundle and packaged file mapping.

        Returns
        -------
        dict[str, Any]
            Calculator settings, selected model, per-dataset properties and
            output paths, and package requirements. Fine-tuning evaluates the
            trained model (or its temporary host-specific export); zero-shot
            evaluates the pretrained model.

        Raises
        ------
        ValueError
            If the bundle has not been packaged yet.
        """
        if self.files is None:
            raise ValueError('Bundle has not been packaged; call write_submit_folder first.')
        model_path = (
            self.files.evaluation_model or self.files.training_output_files["trained_model"]
            if self.unit_type == "finetune"
            else self.files.pretrained_model
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
                    "path": self.files.test_input_files[stem],
                    "source_domain": self.training_unit.domain,
                    "source_filename": filename,
                    "properties": properties,
                    "output": self.files.test_output_files[stem + "_predictions"],
                    "metadata_output": self.files.test_output_files[stem + "_metadata"],
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
            "summary_output": self.files.test_output_files["summary"],
            "package_requirements": [
                {"name": item.name, "version": item.version}
                for item in self.mlff_spec.implementations
                if item.kind == "python_distribution"
            ],
        }

    def write_submit_folder(
            self,
            target_dir: str | Path | None = None,
            *,
            bundle_write_root: str | Path | None = None
    ) -> tuple[Path, MLFFTrainBundle]:
        """Copy inputs and write bundle.json and a launcher to a new directory.

        Runtime preparation creates native configs, training scripts, and the
        test configuration at their recorded paths; packaging does not write them.

        Parameters
        ----------
        target_dir : str, pathlib.Path, or None, optional
            New directory to create. It must not already exist. When omitted,
            a readable model/domain/bundle/attempt directory is allocated.
        bundle_write_root : str, pathlib.Path or None, optional
            Automatic layout root, relative to the current directory. Defaults
            to DEFAULT_BUNDLE_WRITE_ROOT or ./bundles. Cannot accompany target_dir.

        Returns
        -------
        tuple[pathlib.Path, MLFFTrainBundle]
            Created directory and independent packaged record matching bundle.json.
            The original bundle is unchanged.

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

        return write_submit_folder(self, target_dir, bundle_write_root=bundle_write_root)


__all__ = ["MLFFTrainBundle", "MLFFBundleFiles"]
