"""Persisted MLFF recipes that are independent of benchmark datasets."""

from __future__ import annotations

from typing import Any, ClassVar, Literal
from uuid import UUID

from pydantic import ConfigDict, Field

from temper.schemas.artifact import LocalArtifactRef
from temper.schemas.base import MSONableModel, ManagedIdentityModel


_MLFF_SPEC_ID_NAMESPACE = UUID("4621b7bf-70f7-51bd-a3b3-7d5c46fa65a7")


class MLFFImplementation(MSONableModel):
    """Identify software whose exact version defines an MLFF workflow.

    A specification records every Python distribution or executable that can
    change its scientific result. Invocation details stay in the bundle
    writer; this value object contains only stable provenance.

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

    Artifact keys are package-specific stable names such as model, config, and
    restart. Builders create the required mapping, so callers normally supply
    only file paths rather than constructing this object themselves.

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
    """Return order-independent implementation metadata for identity hashing."""
    items = [
        item.model_dump(mode="json") if isinstance(item, MLFFImplementation) else item
        for item in value
    ]
    return sorted(items, key=lambda item: (item["name"], item["version"], item["kind"]))


def _model_identity(value: PretrainedMLFFSpec) -> dict[str, Any]:
    """Return pretrained-model identity without machine-local source paths."""
    return {
        "name": value.name,
        "version": value.version,
        "artifacts": {
            key: artifact.sha256
            for key, artifact in sorted(value.artifacts.items())
        },
    }


class MLFFSpec(ManagedIdentityModel):
    """Persist one reproducible MLFF training and evaluation recipe.

    The schema contains no dataset paths. It identifies the MLFF family,
    pinned software, content-addressed pretrained files, optional package-native
    training parameters, and package-native Calculator keyword arguments.
    Local artifact paths are deliberately excluded from mlff_spec_id so a
    model directory can move without changing scientific identity.

    Parameters
    ----------
    mlff_type : str
        Writer key such as dpa4 or mace.
    implementations : tuple[MLFFImplementation, ...]
        Exact packages and executables used by the integration.
    pretrained_model : PretrainedMLFFSpec
        Local, content-addressed pretrained model files.
    training_parameters : dict[str, Any] or None
        Package-native fine-tuning parameters. None means no training.
    testing_parameters : dict[str, Any]
        Package-native ASE Calculator keyword arguments. Do not include device
        selection or evaluated properties; TEMPER manages them remotely.
    mlff_spec_id : UUID or None
        Stored deterministic identity. It is generated when omitted and
        verified when loading a persisted record.
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
    training_parameters: dict[str, Any] | None = None
    testing_parameters: dict[str, Any] = Field(default_factory=dict)
    mlff_spec_id: UUID | None = None


__all__ = [
    "MLFFImplementation",
    "MLFFSpec",
    "PretrainedMLFFSpec",
]
