"""Specification builders for DeepMD DPA-4 and DPA-4C workflows."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from temper.mlff.spec_builders._training import validate_epoch
from temper.schemas.mlff_spec import (
    LocalArtifactRef,
    MLFFImplementation,
    MLFFSpec,
    PretrainedMLFFSpec,
)
from temper.utils.defaults import DEFAULT_MLFF_PRETRAINED_MODELS_DIR


_TRAINING_DEFAULTS: dict[str, Any] = {
    "numb_epoch": 100,
    "save_freq": 1000,
    "disp_freq": 100,
    "seed": 42,
}
_DISALLOWED_LENGTH_KEYS = {
    "numb_steps",
    "stop_batch",
    "num_step",
    "num_steps",
    "numb_step",
    "num_epochs",
    "num_epoch",
    "numb_epochs",
}
_IMPLEMENTATIONS = (MLFFImplementation(name="deepmd-kit", version="3.2.0"),)


def _default_path(filename: str) -> Path:
    return Path(DEFAULT_MLFF_PRETRAINED_MODELS_DIR) / filename


def _build(
    *,
    mlff_type: str,
    name: str,
    model_filename: str,
    config_filename: str,
    pretrained_model_path: str | Path | None,
    pretrained_config_path: str | Path | None,
    training_parameters: dict[str, Any] | None,
    testing: dict[str, Any] | None,
) -> MLFFSpec:
    model = PretrainedMLFFSpec(
        name=name,
        version="2025.10",
        artifacts={
            "model": LocalArtifactRef.from_path(
                _default_path(model_filename)
                if pretrained_model_path is None
                else pretrained_model_path
            ),
            "config": LocalArtifactRef.from_path(
                _default_path(config_filename)
                if pretrained_config_path is None
                else pretrained_config_path
            ),
        },
    )
    training = None
    if training_parameters is not None:
        disallowed = sorted(_DISALLOWED_LENGTH_KEYS & training_parameters.keys())
        if disallowed:
            raise ValueError(
                "DeepMD training length must use canonical 'numb_epoch'; "
                f"unsupported keys: {disallowed!r}."
            )
        training = deepcopy(_TRAINING_DEFAULTS)
        training.update(deepcopy(training_parameters))
        validate_epoch(training, "numb_epoch", "DeepMD")
    return MLFFSpec(
        mlff_type=mlff_type,
        implementations=_IMPLEMENTATIONS,
        pretrained_model=model,
        training=training,
        testing=deepcopy(testing) if testing is not None else {},
    )


class DPA4SpecBuilder:
    """Build a DeepMD-kit 3.2.0 DPA-4 specification from local files.

    Attributes
    ----------
    pretrained_model_path : str, pathlib.Path, or None, optional
        DPA-4 checkpoint. Defaults to dpa4.pt below
        DEFAULT_MLFF_PRETRAINED_MODELS_DIR.
    pretrained_config_path : str, pathlib.Path, or None, optional
        DeepMD type-map and non-architecture training template. Defaults to
        dpa4.json in the same source directory.
    training_parameters : dict[str, Any] or None, optional
        Top-level DeepMD training overrides using canonical ``numb_epoch``.
        None creates a zero-shot recipe; an empty dictionary enables defaults.
    testing_parameters : dict[str, Any] or None, optional
        Keyword arguments forwarded to deepmd.calculator.DP, excluding device.
    """

    def __init__(
        self,
        *,
        pretrained_model_path: str | Path | None = None,
        pretrained_config_path: str | Path | None = None,
        training_parameters: dict[str, Any] | None = None,
        testing_parameters: dict[str, Any] | None = None,
    ) -> None:
        self.pretrained_model_path = pretrained_model_path
        self.pretrained_config_path = pretrained_config_path
        self.training_parameters = training_parameters
        self.testing_parameters = testing_parameters

    def build(self) -> MLFFSpec:
        """Build the content-addressed DPA-4 specification.

        Returns
        -------
        MLFFSpec
            Persistable zero-shot or fine-tuning recipe.

        Raises
        ------
        ValueError
            If a pretrained file is missing.
        """
        return _build(
            mlff_type="dpa4",
            name="DPA-4",
            model_filename="dpa4.pt",
            config_filename="dpa4.json",
            pretrained_model_path=self.pretrained_model_path,
            pretrained_config_path=self.pretrained_config_path,
            training_parameters=self.training_parameters,
            testing=self.testing_parameters,
        )


class DPA4CSpecBuilder:
    """Build a DeepMD-kit 3.2.0 DPA-4C specification from local files.

    Parameters
    ----------
    pretrained_model_path : str, pathlib.Path, or None, optional
        DPA-4C checkpoint. Defaults to dpa4c.pt below
        DEFAULT_MLFF_PRETRAINED_MODELS_DIR.
    pretrained_config_path : str, pathlib.Path, or None, optional
        DeepMD type-map and non-architecture training template. Defaults to
        dpa4c.json in the same source directory.
    training_parameters : dict[str, Any] or None, optional
        Top-level DeepMD training overrides using canonical ``numb_epoch``.
        None creates a zero-shot recipe; an empty dictionary enables defaults.
    testing_parameters : dict[str, Any] or None, optional
        Keyword arguments forwarded to deepmd.calculator.DP, excluding device.
    """

    def __init__(
        self,
        *,
        pretrained_model_path: str | Path | None = None,
        pretrained_config_path: str | Path | None = None,
        training_parameters: dict[str, Any] | None = None,
        testing_parameters: dict[str, Any] | None = None,
    ) -> None:
        self.pretrained_model_path = pretrained_model_path
        self.pretrained_config_path = pretrained_config_path
        self.training_parameters = training_parameters
        self.testing_parameters = testing_parameters

    def build(self) -> MLFFSpec:
        """Build the content-addressed DPA-4C specification.

        Returns
        -------
        MLFFSpec
            Persistable zero-shot or fine-tuning recipe.

        Raises
        ------
        ValueError
            If a pretrained file is missing.
        """
        return _build(
            mlff_type="dpa4c",
            name="DPA-4C",
            model_filename="dpa4c.pt",
            config_filename="dpa4c.json",
            pretrained_model_path=self.pretrained_model_path,
            pretrained_config_path=self.pretrained_config_path,
            training_parameters=self.training_parameters,
            testing=self.testing_parameters,
        )


__all__ = ["DPA4CSpecBuilder", "DPA4SpecBuilder"]
