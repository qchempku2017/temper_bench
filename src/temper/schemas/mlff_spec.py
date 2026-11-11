"""Persisted MLFF specifications that are independent of benchmark
datasets."""

from __future__ import annotations

from typing import Any, ClassVar, Literal
from uuid import UUID

from pydantic import ConfigDict, Field, field_validator

from temper.schemas.artifact import LocalArtifactRef
from temper.schemas.base import MSONableModel, ManagedIdentityModel
from temper.schemas.utils import validate_submit_relative_path

_MLFF_SPEC_ID_NAMESPACE = UUID("4621b7bf-70f7-51bd-a3b3-7d5c46fa65a7")


class MLFFFileLayout(MSONableModel):
    """Specify backend filenames before a bundle is packaged.

    Parameters
    ----------
    pretrained_model_filename : str
        Destination name of the copied checkpoint, relative to the
        writer's pretrained-model directory, normally pretrained_models/.
        The local source remains in MLFFSpec.pretrained_model.artifacts.
    training_input_filenames : dict[str, str]
        Native input names keyed by purpose, such as config or script,
        relative to the training-input directory, normally training_inputs/.
        Converted-dataset directories may also be named here. The writer
        adds train and optional validation dataset paths separately.
    training_output_filenames : dict[str, str]
        Output names keyed by purpose, including trained_model and
        training_log for fine-tuning, relative to the training-output
        directory, normally training_outputs/. Values may name directories.
    evaluation_model_filename : str or None, optional
        Temporary model export name relative to the training-output
        directory. None, the default, means evaluation needs no separate
        export. The runtime creates this file and removes it after testing.

    Attributes
    ----------
    pretrained_model_filename : str
        Packaged checkpoint name, independent of its local source name.
    training_input_filenames : dict[str, str]
        Native input filenames and converted-data directories, relative
        to the writer's training-input directory.
    training_output_filenames : dict[str, str]
        Planned output files and directories, relative to the writer's
        training-output directory. They need not exist yet.
    evaluation_model_filename : str or None
        Temporary export name relative to the training-output directory,
        or None when the model can be evaluated directly.

    Raises
    ------
    pydantic.ValidationError
        If a required field is absent or a name is empty, absolute,
        drive-qualified, contains backslashes, or has unnormalized or
        parent-directory segments.

    Notes
    -----
    Names may contain subdirectories and use POSIX '/' separators. No
    files are read or written when constructing this object. Packaging
    expands these names into bundle-relative paths in MLFFBundleFiles;
    it omits training paths and the evaluation export for zero-shot runs.
    Changing names does not change the specification's scientific identity.
    """

    model_config = ConfigDict(
        validate_assignment=True, revalidate_instances="always", extra="forbid"
    )

    pretrained_model_filename: str
    training_input_filenames: dict[str, str]
    training_output_filenames: dict[str, str]
    evaluation_model_filename: str | None = None

    @field_validator("pretrained_model_filename", "evaluation_model_filename")
    @classmethod
    def _validate_model_filename(cls, value: str | None) -> str | None:
        """Reject model names that cannot be safely prefixed by a directory."""
        if value is not None:
            validate_submit_relative_path(value, label="model filename")
        return value

    @field_validator("training_input_filenames", "training_output_filenames")
    @classmethod
    def _validate_training_filenames(
        cls, value: dict[str, str]
    ) -> dict[str, str]:
        """Validate every native filename before directory expansion."""
        for name in value.values():
            validate_submit_relative_path(name, label="training filename")
        return value


class MLFFImplementation(MSONableModel):
    """Identify software whose exact version defines an MLFF workflow.

    A specification records every Python distribution or executable that
    can change its scientific result. Invocation details stay in the
    runtime adapter; this value object contains only stable provenance.

    Parameters
    ----------
    name : str
        Nonempty distribution or executable name.
    version : str
        Nonempty exact integration version supported by TEMPER.
    kind : {"python_distribution", "executable"}, optional
        Software type; defaults to Python distribution metadata.

    Attributes
    ----------
    name : str
        Distribution or executable name.
    version : str
        Exact integration version supported by TEMPER.
    kind : {"python_distribution", "executable"}
        Whether name is resolved as installed Python metadata or as an
        external executable.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    kind: Literal["python_distribution", "executable"] = "python_distribution"


class PretrainedMLFFSpec(MSONableModel):
    """Describe the local files that make up one pretrained model.

    Artifact keys are package-specific stable names such as model,
    config, and restart. Builders create the required mapping, so
    callers normally supply only file paths rather than constructing
    this object themselves.

    Parameters
    ----------
    name : str
        Human-readable model family name.
    version : str
        Version or release label of the pretrained weights.
    artifacts : dict[str, LocalArtifactRef]
        Local files required by the selected integration.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    artifacts: dict[str, LocalArtifactRef]


def _implementation_identity(value: Any) -> list[dict[str, Any]]:
    """Return order-independent implementation metadata for identity
    hashing."""
    items = [
        (
            item.model_dump(mode="json")
            if isinstance(item, MLFFImplementation)
            else item
        )
        for item in value
    ]
    return sorted(
        items, key=lambda item: (item["name"], item["version"], item["kind"])
    )


def _model_identity(value: PretrainedMLFFSpec) -> dict[str, Any]:
    """Return pretrained-model identity without machine-local source
    paths."""
    return {
        "name": value.name,
        "version": value.version,
        "artifacts": {
            key: artifact.sha256
            for key, artifact in sorted(value.artifacts.items())
        },
    }


class MLFFSpec(ManagedIdentityModel):
    """Persist one reproducible MLFF training and evaluation
    specification.

    The schema contains no dataset paths. It identifies the MLFF family,
    pinned software, content-addressed pretrained files, optional
    package-native training parameters, and package-native Calculator
    keyword arguments, together with the backend's packaging filenames.
    Local artifact paths and file_layout are deliberately excluded
    from mlff_spec_id so a model directory can move without changing
    scientific identity.

    Parameters
    ----------
    mlff_type : str
        MLFF family key such as dpa4 or mace, used to select the runtime
        adapter and calculator. Packaging does not resolve this key.
    implementations : tuple[MLFFImplementation, ...]
        Exact packages and executables used by the integration.
    pretrained_model : PretrainedMLFFSpec
        Local, content-addressed pretrained model files.
    file_layout : MLFFFileLayout
        Required packaging filenames supplied by the builder or caller.
        They are saved with the specification so packaging needs neither
        the original builder nor an installed runtime adapter.
    training_parameters : dict[str, Any] or None, optional
        Package-native fine-tuning parameters. None, the default, means
        no training.
    testing_parameters : dict[str, Any], optional
        Package-native ASE Calculator keyword arguments. Do not include
        device selection or evaluated properties; TEMPER manages them
        remotely. Defaults to an empty dictionary.
    mlff_spec_id : UUID or None, optional
        Stored deterministic identity. It is generated when omitted and
        verified when loading a persisted record.

    Attributes
    ----------
    mlff_type : str
        Family identifier used during execution.
    implementations : tuple[MLFFImplementation, ...]
        Required software and exact integration versions.
    pretrained_model : PretrainedMLFFSpec
        Source checkpoint identity, local artifact paths, and content hashes.
    file_layout : MLFFFileLayout
        Saved filename choices, relative to the writer's input/output
        directories. These are not yet the resolved bundle-relative paths.
    training_parameters : dict[str, Any] or None
        Native training settings, or None for evaluation without training.
    testing_parameters : dict[str, Any]
        Native calculator settings used for evaluation.
    mlff_spec_id : UUID or None
        Scientific identity, initialized during validation. Packaging names
        and machine-local source paths do not contribute to this identity.

    Raises
    ------
    pydantic.ValidationError
        If required fields are missing, filenames are invalid, or a supplied
        identity does not match the scientific settings and artifact hashes.
    """

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    _IDENTITY_FIELD_NAME: ClassVar[str] = "mlff_spec_id"
    _IDENTITY_SOURCE_FIELDS: ClassVar[tuple[str, ...]] = (
        "mlff_type",
        "implementations",
        "pretrained_model",
        "training_parameters",
        "testing_parameters",
    )
    _IDENTITY_SOURCE_NORMALIZERS: ClassVar[dict[str, Any]] = {
        "implementations": _implementation_identity,
        "pretrained_model": _model_identity,
    }
    _IDENTITY_NAMESPACE: ClassVar[UUID] = _MLFF_SPEC_ID_NAMESPACE
    _IDENTITY_SCHEMA: ClassVar[str] = "temper.mlff-spec.v3"
    _IDENTITY_LABEL: ClassVar[str] = "MLFF specification"

    mlff_type: str = Field(min_length=1)
    implementations: tuple[MLFFImplementation, ...] = Field(min_length=1)
    pretrained_model: PretrainedMLFFSpec
    file_layout: MLFFFileLayout
    training_parameters: dict[str, Any] | None = None
    testing_parameters: dict[str, Any] = Field(default_factory=dict)
    mlff_spec_id: UUID | None = None


__all__ = [
    "MLFFFileLayout",
    "MLFFImplementation",
    "MLFFSpec",
    "PretrainedMLFFSpec",
]
