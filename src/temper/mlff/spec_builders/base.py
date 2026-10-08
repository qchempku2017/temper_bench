"""Shared construction of content-addressed MLFF recipes."""
from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import Any, ClassVar

from temper.schemas.artifact import LocalArtifactRef
from temper.schemas.mlff_spec import MLFFImplementation, MLFFSpec, PretrainedMLFFSpec
from temper.utils.defaults import DEFAULT_MLFF_PRETRAINED_MODELS_DIR


class BaseSpecBuilder(ABC):
    """Construct a recipe from family metadata and native training defaults.

    Parameters
    ----------
    pretrained_model_dir : str or Path, optional
        Directory containing pretrained files. Defaults to
        DEFAULT_MLFF_PRETRAINED_MODELS_DIR.
    model_filename : str, optional
        Source filename; defaults to the subclass's release filename.
    training_parameters : dict or None
        Native overrides; None selects zero-shot, {} enables training defaults.
    testing_parameters : dict or None
        ASE calculator options; TEMPER manages hardware selection.

    Attributes
    ----------
    mlff_type : str
        Family identifier stored in the generated recipe.
    model_name, model_version : str
        Pretrained model name and release identifier.
    implementations : tuple[MLFFImplementation, ...]
        Software required for evaluation.
    training_implementations : tuple[MLFFImplementation, ...]
        Additional software required only for fine-tuning.
    training_defaults : dict[str, Any]
        Native controls copied and merged with caller overrides.
    epoch_key : str
        Native field specifying the positive training epoch count.
    patience_key : str or None
        Early-stopping field managed as epoch count plus one, when applicable.
    registry : dict[str, type[BaseSpecBuilder]]
        Builder classes indexed by explicit registration names and aliases.
    """

    registry: ClassVar[dict[str, type[BaseSpecBuilder]]] = {}
    model_name: str
    model_version: str
    model_filename: str
    implementations: tuple[MLFFImplementation, ...]
    training_implementations: tuple[MLFFImplementation, ...] = ()
    training_defaults: dict[str, Any]
    epoch_key: str
    patience_key: str | None = None

    @property
    @abstractmethod
    def mlff_type(self) -> str:
        """Family key; concrete builders supply this as a class attribute."""

    @classmethod
    def register(cls, name: str, alias: str | None = None):
        """Return a decorator that registers a builder class.

        Parameters
        ----------
        name : str
            Primary key accepted by mlff_spec_builder_factory.
        alias : str or None, optional
            Additional key referring to the same builder class.

        Returns
        -------
        Callable
            Decorator that records and returns the supplied class unchanged.
        """
        def decorate(subclass):
            cls.registry[name] = subclass
            if alias is not None:
                cls.registry[alias] = subclass
            return subclass

        return decorate

    def __init__(
        self,
        *,
        pretrained_model_dir: str | Path | None = None,
        model_filename: str | None = None,
        training_parameters: dict[str, Any] | None = None,
        testing_parameters: dict[str, Any] | None = None,
    ) -> None:
        self.pretrained_model_dir = Path(
            DEFAULT_MLFF_PRETRAINED_MODELS_DIR
            if pretrained_model_dir is None
            else pretrained_model_dir
        ).expanduser()
        self.model_filename = model_filename or type(self).model_filename
        self.training_parameters = training_parameters
        self.testing_parameters = testing_parameters

    def prepare_training(self) -> dict[str, Any]:
        """Return independent native defaults merged with caller overrides."""
        overrides = self.training_parameters or {}
        if self.patience_key and self.patience_key in overrides:
            raise ValueError(f"{self.patience_key!r} is managed by TEMPER.")
        parameters = deepcopy(self.training_defaults)
        parameters.update(deepcopy(overrides))
        epochs = parameters[self.epoch_key]
        if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0:
            raise ValueError(f"{self.epoch_key!r} must be a positive integer.")
        if self.patience_key:
            parameters[self.patience_key] = epochs + 1
        return parameters

    def build(self) -> MLFFSpec:
        """Return a persistable recipe, hashing each required local file.

        Raises ValueError for missing files or invalid training controls.
        """
        artifacts = {
            "model": LocalArtifactRef.from_path(
                self.pretrained_model_dir / self.model_filename
            )
        }
        training = (
            None if self.training_parameters is None else self.prepare_training()
        )
        return MLFFSpec(
            mlff_type=self.mlff_type,
            implementations=(
                self.implementations
                + (self.training_implementations if training is not None else ())
            ),
            pretrained_model=PretrainedMLFFSpec(
                name=self.model_name,
                version=self.model_version,
                artifacts=artifacts,
            ),
            training_parameters=training,
            testing_parameters=deepcopy(self.testing_parameters or {}),
        )
