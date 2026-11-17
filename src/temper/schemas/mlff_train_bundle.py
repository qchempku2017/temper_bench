"""Atomic pairings of benchmark datasets and MLFF specifications."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Literal
from uuid import UUID

from pydantic import ConfigDict, Field, field_validator, model_validator
from temper._version import __version__

from temper.schemas.base import ManagedIdentityModel, MSONableModel
from temper.utils.file_paths import validate_submit_relative_path
from temper.schemas.mlff_spec import MLFFSpec
from temper.schemas.train_unit import TrainingUnit

_MLFF_TRAIN_BUNDLE_ID_NAMESPACE = UUID("cb75bc42-06f5-5eec-8d38-a711f707d4c2")


def _nested_identity(value: Any) -> str:
    """Reduce a nested persisted record to its managed identity."""
    if isinstance(value, TrainingUnit):
        if value.training_unit_id is None:
            raise ValueError(
                "Nested TrainingUnit identity has not been initialized."
            )
        return str(value.training_unit_id)
    if isinstance(value, MLFFSpec):
        if value.mlff_spec_id is None:
            raise ValueError(
                "Nested MLFFSpec identity has not been initialized."
            )
        return str(value.mlff_spec_id)
    return str(value)


class MLFFBundleFiles(MSONableModel):
    """Record where a packaged experiment reads and writes its files.

    Every path is relative to the directory containing bundle.json,
    including paths to directories. These are planned locations:
    packaging copies input datasets and pretrained weights. Runtime
    preparation writes native configs and scripts; execution creates
    converted datasets, resolved backend configs and results. A recorded
    output therefore need not exist yet.

    Parameters
    ----------
    **data : object
        Values for the attributes below. Paths are portable relative paths;
        constructing this record does not create any files.

    Attributes
    ----------
    pretrained_model : str
        Copied pretrained weights. Fine-tuning starts from this file;
        zero-shot evaluation uses it directly. Execution does not modify
        it.
    training_input_files : dict[str, str]
        Training inputs named by purpose: train, optional validation,
        config, script, and any backend-specific converted datasets.
        Empty for zero-shot.
    training_output_files : dict[str, str]
        Training outputs named by purpose, including trained_model and
        training_log. Other entries describe backend configs,
        checkpoints, metrics, or directories containing them. Empty for
        zero-shot.
    test_input_files : dict[str, str]
        Copied test datasets keyed by their original TrainingUnit.test_sets
        filenames. Paths preserve those filenames below testing_inputs/.
    test_output_files : dict[str, str]
        Prediction NPZ files keyed by the same original test filenames.
        Paths preserve the source names with the .npz extension.
    evaluation_model : str or None
        Temporary export made for evaluation, such as a frozen DeepMD
        model. The runner removes it after evaluation. None means
        evaluation can use the pretrained or fine-tuned model directly.
    training_stress : bool or None
        Whether training and validation data have stress labels. None
        for zero-shot.
    test_stress : list[bool]
        Stress availability in the same order as TrainingUnit.test_sets.
    result_manifest : {'results.json'}
        Serialized MLFFExperimentManifest at the bundle root, including training status
        and training-call duration.
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
    result_manifest: Literal["results.json"] = "results.json"

    @field_validator(
        "training_input_files",
        "training_output_files",
        "test_input_files",
        "test_output_files",
    )
    @classmethod
    def validate_paths(cls, value: dict[str, str]) -> dict[str, str]:
        for path in value.values():
            validate_submit_relative_path(path, label="bundle file")
        return value

    @field_validator("pretrained_model", "evaluation_model")
    @classmethod
    def validate_model_path(cls, value: str | None) -> str | None:
        """Validate a model path inside the bundle.

        Parameters
        ----------
        value : str or None
            Portable relative path; None denotes an absent optional file.

        Returns
        -------
        str or None
            Unchanged valid value.

        Raises
        ------
        ValueError
            If the path is absolute or contains invalid path components.
        """
        if value is not None:
            validate_submit_relative_path(value, label="bundle model")
        return value

    @model_validator(mode="after")
    def separate_temporary_model(self) -> MLFFBundleFiles:
        """Prevent temporary-model cleanup from deleting a permanent file.

        Returns
        -------
        MLFFBundleFiles
            This validated file mapping.

        Raises
        ------
        ValueError
            If temporary-model or result-manifest paths
            collide with permanent inputs or outputs.
        """
        permanent_paths = {
            self.pretrained_model,
            *self.training_input_files.values(),
            *self.training_output_files.values(),
            *self.test_input_files.values(),
            *self.test_output_files.values(),
        }
        if self.result_manifest in permanent_paths:
            raise ValueError("Result manifest must have its own path.")
        permanent_paths.add(self.result_manifest)
        if self.evaluation_model in permanent_paths:
            raise ValueError(
                "The temporary evaluation model must have its own path."
            )
        return self


class MLFFTrainBundle(ManagedIdentityModel):
    """Pair one exported training unit with one MLFF specification.

    Constructing this object does not copy data or run third-party
    software. write_submit_folder copies inputs and serializes a
    portable file mapping. The installed runtime writes missing configs
    and scripts, prepares backend-dependent data, trains when requested,
    and evaluates models. Packaging fields do not change the
    experiment's deterministic identity.

    Parameters
    ----------
    **data : object
        Values for the attributes below. New local bundles normally
        provide training_unit and mlff_spec only; files remains absent
        until packaging.

    Attributes
    ----------
    training_unit : TrainingUnit
        Exported train/validation/test dataset references. Packaging records
        the absolute original split-data root for later local analysis.
        Users must keep those original datasets unchanged.
    mlff_spec : MLFFSpec
        Specification of the model, software implementation, training,
        and testing settings.
    mlff_train_bundle_id : UUID or None
        Stored deterministic identity derived from the two nested
        identities.
    schema_version : Literal[1]
        Bundle format version used when loading the serialized record.
    temper_version : str
        TEMPER release that wrote the bundle, retained for provenance.
        The runtime warns if its release differs.
    files : MLFFBundleFiles or None
        Locations of copied datasets and pretrained weights, plus
        planned training and testing files. Every path is relative to
        the bundle directory. A newly constructed local bundle has None,
        and therefore when encountering a None value, the runtime cannot
        perform fine-tuning or evaluation.
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
    _IDENTITY_SCHEMA: ClassVar[str] = "temper.mlff-train-bundle.v1"
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
        """Reject a fine-tuning dataset paired with a test-only
        specification."""
        if (
            self.unit_type == "finetune"
            and self.mlff_spec.training_parameters is None
        ):
            raise ValueError(
                "Fine-tuning TrainingUnit requires non-None MLFF training"
                " parameters."
            )

    def write_submit_folder(
        self,
        target_dir: str | Path | None = None,
        *,
        bundle_write_root: str | Path | None = None,
    ) -> tuple[Path, MLFFTrainBundle]:
        """Copy inputs and write bundle.json and a launcher to a new
        directory.

        Runtime preparation creates native training configs and scripts at
        their recorded paths. Execution writes results.json with pending
        test manifests, then fills their result fields as tests finish.
        Test copies preserve original filenames and relative subdirectories.
        The packaged training unit records an absolute source root. Users must
        keep the original split datasets unchanged there for later analysis.

        Parameters
        ----------
        target_dir : str, pathlib.Path, or None, optional
            New directory to create, relative to the working directory when
            not absolute. It must not already exist. When
            omitted, a readable model/domain/bundle/attempt directory is
            allocated.
        bundle_write_root : str, pathlib.Path or None, optional
            Automatic layout root, relative to the current directory.
            Defaults to DEFAULT_BUNDLE_WRITE_ROOT or ./bundles. Cannot
            accompany target_dir.

        Returns
        -------
        tuple[pathlib.Path, MLFFTrainBundle]
            Created directory and independent packaged record matching
            bundle.json. The original bundle is unchanged.

        Raises
        ------
        FileExistsError
            If target_dir already exists.
        ValueError
            If the bundle is inconsistent, a referenced dataset is
            invalid, or a pretrained artifact no longer matches its
            recorded hash.
        OSError
            If an input cannot be read or the destination cannot be
            written.
        """
        from temper.mlff.bundle_writers import write_submit_folder

        return write_submit_folder(
            self, target_dir, bundle_write_root=bundle_write_root
        )


__all__ = ["MLFFTrainBundle", "MLFFBundleFiles"]
