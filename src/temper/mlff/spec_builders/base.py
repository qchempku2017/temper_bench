"""Shared construction of content-addressed MLFF specifications."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Callable
from copy import deepcopy
from importlib.resources import files
from pathlib import Path
from typing import Any, ClassVar, TypeVar

from temper.schemas.artifact import LocalArtifactRef
from temper.schemas.mlff_spec import (
    MLFFImplementation,
    MLFFSpec,
    PretrainedMLFFSpec,
)
from temper.schemas.utils import validate_submit_relative_path
from temper.utils.defaults import DEFAULT_MLFF_PRETRAINED_MODELS_DIR

_Builder = TypeVar("_Builder", bound="BaseSpecBuilder")


# Utilities to load pretrained checkpoint metadata.
def pretrained_model_variants(mlff_type: str) -> dict[str, dict[str, Any]]:
    """Return an independent mapping of variant names to release
    metadata.

    ``mlff_type`` is a builder family key, such as ``mace``. Each entry
    contains the model name, release version, and checkpoint filename
    relative to the caller's pretrained model directory, together with
    upstream sources and usage notes and an estimated_num_parameters
    integer describing the full pretrained model before adaptation.
    Entries provide fine-tunable weights; callers must satisfy any
    backend prerequisites in their usage notes. Reading the catalog
    neither downloads nor opens checkpoints. Changes to the returned
    dictionaries do not affect subsequent calls.

    Parameters
    ----------
    mlff_type : str
        Registered family name, such as ``mace`` or ``dpa4``.

    Returns
    -------
    dict[str, dict[str, Any]]
        Variant names mapped to independent checkpoint metadata dictionaries.

    Raises
    ------
    ValueError
        If the family is not present in the packaged catalog.
    """
    catalog = json.loads(
        files("temper.mlff")
        .joinpath("default_pretrained_models.json")
        .read_text(encoding="utf-8")
    )
    try:
        return catalog["models"][mlff_type]
    except KeyError:
        raise ValueError(
            f"No pretrained model catalog for family {mlff_type!r}."
        ) from None


def pretrained_model_defaults(mlff_type: str, variant: str) -> dict[str, Any]:
    """Return release metadata for one exact, case-sensitive variant
    name.

    The result is independent of the packaged catalog. An unknown
    variant raises ``ValueError`` listing the available choices for this
    family.

    Parameters
    ----------
    mlff_type : str
        Registered family name, such as ``mace`` or ``dpa4``.
    variant : str
        Exact variant key returned by pretrained_model_variants.

    Returns
    -------
    dict[str, Any]
        Checkpoint identity, relative filename, source URLs and usage notes.
        No files are downloaded or validated by this lookup.

    Raises
    ------
    ValueError
        If the family or variant is absent from the packaged catalog.
    """
    variants = pretrained_model_variants(mlff_type)
    try:
        return variants[variant]
    except KeyError:
        raise ValueError(
            f"Unknown {mlff_type} variant {variant!r}. "
            f"Available variants: {', '.join(variants) or '(none)'}."
        ) from None


class BaseSpecBuilder(ABC):
    """Construct a specification from family metadata and native
    training defaults.

    Parameters
    ----------
    variant : str or None, optional
        Exact variant key from this family's packaged pretrained model
        catalog. Select either a variant or all three explicit model
        fields below. No checkpoint is downloaded; the selected file
        must exist at build time.
    model_name, model_version : str or None, optional
        Pretrained variant name and release identifier for a custom
        checkpoint. Required together with model_filename when variant
        is absent; forbidden when variant is supplied.
    pretrained_model_dir : str or Path, optional
        Directory containing pretrained files. Defaults to
        DEFAULT_MLFF_PRETRAINED_MODELS_DIR. A leading ~ is expanded;
        relative directories are resolved against the working directory
        at build time.
    model_filename : str or None, optional
        Checkpoint filename or POSIX subpath relative to
        pretrained_model_dir. Required for custom checkpoints and
        supplied by the catalog for variants.
    training_parameters : dict or None
        Native overrides; None selects zero-shot, {} enables training
        defaults.
    testing_parameters : dict or None
        ASE calculator options; TEMPER manages hardware selection.
        Variant options are included automatically. Conflicting task or
        model-type options are rejected so the selected model identity
        stays accurate.

    Attributes
    ----------
    variant : str or None
        Catalog selection used to initialize this builder; None means
        the caller supplied the checkpoint identity explicitly.
    model_name, model_version : str
        Pretrained model name and release identifier.
    model_filename : str
        Pretrained file name, relative to pretrained_model_dir. This
        points to pretrained model file on local submission machine.
    implementations : tuple[MLFFImplementation, ...]
        Software required for evaluation.
    training_implementations : tuple[MLFFImplementation, ...]
        Additional software required only for fine-tuning.
    training_defaults : dict[str, Any]
        Native controls copied and merged with caller overrides.
    epoch_key : str
        Native field specifying the positive training epoch count.
    patience_key : str or None
        Early-stopping field managed as epoch count plus one, when
        applicable.
    registry : dict[str, type[BaseSpecBuilder]]
        Builder classes indexed by explicit registration names and
        aliases.
    """

    registry: ClassVar[dict[str, type[BaseSpecBuilder]]] = {}
    implementations: tuple[MLFFImplementation, ...]
    training_implementations: tuple[MLFFImplementation, ...] = ()
    training_defaults: dict[str, Any]
    epoch_key: str
    patience_key: str | None = None

    @property
    @abstractmethod
    def mlff_type(self) -> str:
        """Family key; concrete builders supply this as a class
        attribute."""

    @classmethod
    def register(
        cls, name: str, alias: str | list[str] | None = None
    ) -> Callable[[type[_Builder]], type[_Builder]]:
        """Return a decorator that registers a builder class.

        Parameters
        ----------
        name : str
            Primary key accepted by mlff_spec_builder_factory.
        alias : str, list[str], or None, optional
            Additional key or keys referring to the same builder class.

        Returns
        -------
        Callable
            Decorator that records and returns the supplied class
            unchanged.
        """

        def decorate(subclass: type[_Builder]) -> type[_Builder]:
            cls.registry[name] = subclass
            if alias is not None:
                for key in ([alias] if isinstance(alias, str) else alias):
                    cls.registry[key] = subclass
            return subclass

        return decorate

    def __init__(
        self,
        *,
        variant: str | None = None,
        model_name: str | None = None,
        model_version: str | None = None,
        pretrained_model_dir: str | Path | None = None,
        model_filename: str | None = None,
        training_parameters: dict[str, Any] | None = None,
        testing_parameters: dict[str, Any] | None = None,
    ) -> None:
        explicit = (model_name, model_version, model_filename)
        preset: dict[str, Any] = {}
        if variant is not None:
            if any(value is not None for value in explicit):
                raise ValueError(
                    "Specify either variant or explicit model_name,"
                    " model_version and model_filename, not both."
                )
            preset = pretrained_model_defaults(self.mlff_type, variant)
            model_name = preset["model_name"]
            model_version = preset["model_version"]
            model_filename = preset["model_filename"]
        for name, value in (
            ("model_name", model_name),
            ("model_version", model_version),
            ("model_filename", model_filename),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"{name} must be a non-empty string; provide variant or"
                    " all three explicit model fields."
                )
        assert (
            model_name is not None
            and model_version is not None
            and model_filename is not None
        ), "Explicit model fields must be non-None after validation."

        self.variant = variant
        self.model_name: str = model_name
        self.model_version: str = model_version
        self.model_filename: str = validate_submit_relative_path(
            model_filename
        )
        self.pretrained_model_dir = Path(
            DEFAULT_MLFF_PRETRAINED_MODELS_DIR
            if pretrained_model_dir is None
            else pretrained_model_dir
        ).expanduser()
        self.training_parameters: dict[str, Any] | None = training_parameters
        options = deepcopy(preset.get("testing_parameters", {}))
        for key, value in (testing_parameters or {}).items():
            if key in options and options[key] != value:
                raise ValueError(
                    f"Variant {variant!r} requires testing_parameters[{key!r}]"
                    f" to be {options[key]!r}; select another variant or use"
                    " explicit model fields."
                )
            options[key] = deepcopy(value)
        self.testing_parameters = options

    @classmethod
    def available_variants(cls) -> tuple[str, ...]:
        """Return catalog variant keys without inspecting local
        checkpoint files."""
        return tuple(pretrained_model_variants(cls.mlff_type))

    def prepare_training(self) -> dict[str, Any]:
        """Return independent native defaults merged with caller
        overrides."""
        overrides = self.training_parameters or {}
        if self.patience_key and self.patience_key in overrides:
            raise ValueError(f"{self.patience_key!r} is managed by TEMPER.")
        parameters = deepcopy(self.training_defaults)
        parameters.update(deepcopy(overrides))

        # Effectively disable early stopping by setting patience to
        # epochs + 1.
        epochs = parameters[self.epoch_key]
        if (
            isinstance(epochs, bool)
            or not isinstance(epochs, int)
            or epochs <= 0
        ):
            raise ValueError(f"{self.epoch_key!r} must be a positive integer.")
        if self.patience_key:
            parameters[self.patience_key] = epochs + 1
        return parameters

    def build(self) -> MLFFSpec:
        """Read and hash the local checkpoint and return its MLFFSpec
        specification.

        The returned spec contains the selected model identity, local
        artifact reference, software requirements, and
        training/calculator parameters. No weights are downloaded or
        modified. Raises ValueError for missing files or invalid
        training controls. Building a specification does not execute the
        backend or verify that its checkpoint requirements are
        installed.
        """
        artifacts = {
            "model": LocalArtifactRef.from_path(
                self.pretrained_model_dir / self.model_filename
            )
        }
        training = (
            None
            if self.training_parameters is None
            else self.prepare_training()
        )
        return MLFFSpec(
            mlff_type=self.mlff_type,
            implementations=(
                self.implementations
                + (
                    self.training_implementations
                    if training is not None
                    else ()
                )
            ),
            pretrained_model=PretrainedMLFFSpec(
                name=self.model_name,
                version=self.model_version,
                artifacts=artifacts,
            ),
            training_parameters=training,
            testing_parameters=deepcopy(self.testing_parameters or {}),
        )
