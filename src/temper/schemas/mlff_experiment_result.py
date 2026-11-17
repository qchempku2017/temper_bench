"""Persistent file descriptions and locally loaded MLFF experiment results."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, Self

from ase import Atoms
import numpy as np
from numpy.typing import NDArray
from pydantic import ConfigDict, Field, field_validator, model_validator

from temper.schemas.base import MSONableModel
from temper.schemas.mlff_train_bundle import MLFFTrainBundle
from temper.utils.file_paths import validate_submit_relative_path

Checksum = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
ExperimentStatus = Literal["prepared", "running", "completed", "failed"]
TrainingStatus = Literal[
    "pending", "running", "completed", "failed", "unavailable", "not_applicable"
]


class MLFFTestManifest(MSONableModel):
    """Describe one planned test and record its results after evaluation.

    A pending record supplies the dataset path and checksum, requested
    properties, units, and prediction output path. Evaluation uses these
    fields directly, then fills the prediction checksum, duration and status.
    Calculator settings and model selection come from the experiment's bundle
    and are shared by all its tests. No numerical arrays are stored here.

    Parameters
    ----------
    **data : object
        Values for the fields below. No scientific files are opened.

    Attributes
    ----------
    dataset_id : str
        Original test filename, such as test_own.extxyz, identifying the dataset.
    source_filename, source_domain : str
        Unchanged exported filename and original domain. Named datasets remain
        separate even when their bytes are identical.
    status : {'pending', 'completed', 'failed'}
        Evaluation state, initially pending.
    execution_input_path : str
        Copied test input relative to the bundle directory, used for local
        or remote execution. Local analysis instead reads the
        original file at bundle.training_unit.root_path/source_domain/source_filename.
        Cleanup may remove this temporary input after execution or upload.
    prediction_path : str
        NPZ file relative to results.json, containing numerical arrays only.
        This is a planned output location; the file need not exist while
        the test is pending or failed.
    reference_sha256 : str
        Expected reference-file checksum.
    prediction_sha256 : str or None
        Expected NPZ checksum; absent until evaluation completes.
    requested_properties : tuple of {'energy', 'forces', 'stress'}
        Physical properties requested for this dataset, in configuration order.
        Energy and forces are always requested; stress is optional.
    units : dict[str, str or None]
        Units of the prediction arrays: energy in eV, forces in eV/Angstrom,
        and stress in eV/Angstrom^3. The stress unit is None when not requested.
    wall_time_seconds : float or None
        Elapsed evaluation time, from reading the dataset through computing
        predictions and assembling arrays. Excludes NPZ writing and calculator
        construction. None means evaluation has not completed.
    """

    model_config = ConfigDict(extra="forbid")
    dataset_id: str
    source_filename: str
    source_domain: str
    status: Literal["pending", "completed", "failed"] = "pending"
    execution_input_path: str
    prediction_path: str
    reference_sha256: Checksum
    prediction_sha256: Checksum | None = None
    requested_properties: tuple[Literal["energy", "forces", "stress"], ...]
    units: dict[str, str | None]
    wall_time_seconds: float | None = Field(default=None, ge=0, allow_inf_nan=False)

    @field_validator("execution_input_path", "prediction_path")
    @classmethod
    def _relative_path(cls, value: str) -> str:
        validate_submit_relative_path(value, label="result file")
        return value

    @model_validator(mode="after")
    def _check_completed(self) -> Self:
        if self.status == "completed" and self.prediction_sha256 is None:
            raise ValueError("Completed tests require a prediction checksum.")
        return self


class MLFFExperimentManifest(MSONableModel):
    """Persist file references and training measurements in results.json.

    This stores the metadata needed to assemble an MLFFExperimentResult from
    downloaded predictions and the user's original local split datasets.
    The runner writes this model. Scientific data are assembled locally only
    when load_results is called.

    Parameters
    ----------
    **data : object
        Values for the fields below, validated without reading scientific files.

    Attributes
    ----------
    schema_version : {1}
        Saved manifest format version.
    status : {'prepared', 'running', 'completed', 'failed'}
        Execution state, initially prepared.
    bundle_path, bundle_sha256 : str
        Experiment settings file relative to results.json and its checksum.
        New runs reference bundle.json instead of embedding its contents.
    attempt_name : str or None
        Optional attempt label supplied by the writer. None means no label
        was recorded; collection does not fill or change this field.
    training_status : TrainingStatus
        pending, running, completed, failed, unavailable or not_applicable.
        Default unavailable means no measurement; not_applicable means zero-shot.
    training_wall_time_seconds : float or None
        Successful adapter.train() duration in seconds. Separate preparation,
        evaluation and transfers are excluded. None means no completed
        measurement; a later evaluation failure does not erase this value.
    test_manifests : tuple[MLFFTestManifest, ...]
        Test instructions and results in evaluation order, including
        unfinished tests. The runner writes pending entries before execution
        and updates them after each test without a separate test-config file.
    package_versions : dict[str, str or None]
        Versions observed in the evaluation process, recorded once for all
        tests. Empty before evaluation; a None value means that package's
        installed version could not be found.

    Notes
    -----
    Inherited from_json_file and to_json_file read and write field dictionaries
    through from_dict and to_dict. Monty's as_dict and to_json are also available.
    File locations are not remembered; load_results requires an explicit
    base_dir for resolving file references. The manifest itself is not checksummed.
    """

    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[1] = 1
    status: ExperimentStatus = "prepared"
    bundle_path: str = "bundle.json"
    bundle_sha256: Checksum
    attempt_name: str | None = None
    training_status: TrainingStatus = "unavailable"
    training_wall_time_seconds: float | None = Field(default=None, ge=0, allow_inf_nan=False, strict=True)
    test_manifests: tuple[MLFFTestManifest, ...]
    package_versions: dict[str, str | None] = Field(default_factory=dict)

    @field_validator("bundle_path")
    @classmethod
    def _bundle_relative(cls, value: str) -> str:
        validate_submit_relative_path(value, label="bundle file")
        return value

    @model_validator(mode="after")
    def _check_state(self) -> Self:
        if (self.training_status == "completed") != (self.training_wall_time_seconds is not None):
            raise ValueError("Only completed training has a duration.")
        keys = [entry.dataset_id for entry in self.test_manifests]
        if not keys or len(keys) != len(set(keys)):
            raise ValueError("Test dataset keys must be nonempty and unique.")
        if self.status == "completed" and any(t.status != "completed" for t in self.test_manifests):
            raise ValueError("Completed manifest contains unfinished tests.")
        return self

    def load_results(
        self, *,
        base_dir: str | Path, require_finite: bool = True,
    ) -> MLFFExperimentResult:
        """Load predictions and original labels for every completed test.

        Parameters
        ----------
        base_dir : str or Path
            Directory containing the manifest's referenced experiment settings
            and predictions. Pass the parent of results.json. Expands ~ and
            resolves relative paths against the current working directory.
            Ground-truth labels are read from the original split datasets at
            the root stored in bundle.training_unit, not from this directory.
            Users must keep those original files unchanged for later analysis.
        require_finite : bool, default True
            Reject NaN and infinity in predictions and reference labels.

        Returns
        -------
        MLFFExperimentResult
            Newly loaded settings, training measurements and all completed test
            data in manifest order. Pending and failed tests have no loaded
            result; their status remains available in this manifest.
            No structures or arrays are cached in this manifest.
            Uses this object's fields without rereading the manifest file.

        Raises
        ------
        ValueError
            If a completed test's original dataset is missing,
            or referenced file checksums and data disagree.
        OSError
            If a required file cannot be read.
        """
        from temper.results_loading.result_loading import (
            load_results_from_manifest,
        )

        return load_results_from_manifest(
            self, require_finite=require_finite, base_dir=base_dir,
        )


@dataclass(frozen=True)
class MLFFTestResult:
    """Hold the checked numerical data for one completed test.

    Not persistent. Loaded from an MLFFExperimentManifest upon request and
    discarded after use.

    Parameters
    ----------
    manifest : MLFFTestManifest
        Original names, relative file locations, checksums, properties, units
        and measured evaluation duration. No numerical arrays are stored here.
    frames : tuple[ase.Atoms, ...]
        Reference structures with saved labels, without an MLFF calculator.
    predictions, references : dict[str, numpy.ndarray]
        Energies (n_frames,), forces (total_atoms, 3), and optional stresses
        (n_frames, 3, 3), in eV, eV/Angstrom and eV/Angstrom^3 respectively.
    atom_offsets, frame_indices : numpy.ndarray
        Integer force boundaries and zero-based exported frame indices.

    Attributes
    ----------
    manifest : MLFFTestManifest
        The corresponding manifest used to build this result, including the
        test dataset domain, filename, checksums, units and evaluation duration,
        kept separately from the loaded scientific values.
    frames : tuple[ase.Atoms, ...]
        Structures in exported order.
    predictions, references : dict[str, numpy.ndarray]
        Aligned floating-point physical values. Arrays remain mutable.
    atom_offsets, frame_indices : numpy.ndarray
        Integer array boundaries and exported frame order.
    """

    manifest: MLFFTestManifest
    frames: tuple[Atoms, ...]
    predictions: dict[str, NDArray[np.float64]]
    references: dict[str, NDArray[np.float64]]
    atom_offsets: NDArray[np.int64]
    frame_indices: NDArray[np.int64]


@dataclass(frozen=True)
class MLFFExperimentResult:
    """Hold one experiment's locally assembled settings and scientific results.

    Not persistent. Loaded from an MLFFExperimentManifest upon request and
    discarded after use.

    Parameters
    ----------
    bundle : MLFFTrainBundle
        Loaded experiment settings. Original test datasets must be available
        when loading scientific data; metadata-only loading does not read them.
    status : ExperimentStatus
        Saved execution state: prepared, running, completed or failed.
    attempt_name : str or None
        Optional attempt label from the manifest; None if none was recorded.
    training_status : TrainingStatus
        Saved training state: pending, running, completed, failed, unavailable
        or not_applicable. Unavailable means no recorded measurement;
        not_applicable denotes zero-shot evaluation.
    training_wall_time_seconds : float or None
        Successful training-call duration in seconds. None means unavailable,
        not zero. Separate preparation, testing and transfers are excluded.
    test_results : tuple[MLFFTestResult, ...]
        Selected completed tests, in report order. Empty for metadata-only loading.
    file_sha256 : dict[str, str]
        Saved output checksums keyed by paths relative to the result JSON.
    package_versions : dict[str, str or None]
        Versions recorded once during evaluation. Empty if not recorded;
        None means a package version could not be found.

    Attributes
    ----------
    bundle : MLFFTrainBundle
        Experiment settings loaded from the manifest's referenced settings file.
    status : ExperimentStatus
        Recorded execution state, using the same values as the manifest.
    training_status : TrainingStatus
        Recorded training state, using the same values as the manifest.
    attempt_name : str or None
        Identity of the execution attempt, when recorded.
    training_wall_time_seconds : float or None
        Completed training duration, preserved even if a later test failed.
    test_results : tuple[MLFFTestResult, ...]
        Loaded test data; omitted or unfinished tests are not represented as
        empty arrays. The manifest still describes all named tests.
    file_sha256 : dict[str, str]
        Saved output checksums for recording which predictions a metric used.
    package_versions : dict[str, str or None]
        Recorded evaluation package versions, shared by all loaded tests.
    """

    bundle: MLFFTrainBundle
    status: ExperimentStatus
    attempt_name: str | None
    training_status: TrainingStatus
    training_wall_time_seconds: float | None
    test_results: tuple[MLFFTestResult, ...]
    file_sha256: dict[str, str]
    package_versions: dict[str, str | None]
