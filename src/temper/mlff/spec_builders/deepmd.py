"""DeepMD fine-tuning policy, independent of submitted datasets."""

from __future__ import annotations

from abc import ABC
from copy import deepcopy
import json
from typing import Any

from temper.mlff.spec_builders.base import BaseSpecBuilder
from temper.schemas.mlff_spec import MLFFImplementation


class _DeepMDSpecBuilder(BaseSpecBuilder, ABC):
    """Share DPA training policy; instances select the pretrained
    release."""

    implementations = (MLFFImplementation(name="deepmd-kit", version="3.2.0"),)
    epoch_key = "numb_epoch"
    training_defaults = {
        "numb_epoch": 60,
        "save_freq": 400,
        "disp_freq": 100,
        "seed": 42,
        "gradient_max_norm": 1.0,
        "training_data": {"batch_size": "auto:128"},
        "validation_data": {"batch_size": "auto:128"},
        "optimizer": {"type": "HybridMuon", "weight_decay": 0.001},
        "learning_rate": {"type": "exp", "start_lr": 1e-4, "stop_lr": 1e-6},
        "loss": {
            "type": "ener",
            "loss_func": "mae",
            "f_use_norm": True,
            "start_pref_e": 20.0,
            "limit_pref_e": 20.0,
            "start_pref_f": 20.0,
            "limit_pref_f": 20.0,
            "start_pref_v": 5.0,
            "limit_pref_v": 5.0,
        },
    }

    def prepare_training(self) -> dict[str, Any]:
        """Combine training controls with the checkpoint's explicit
        model JSON.

        The matching JSON must be beside the checkpoint with the same
        filename stem. For example, ``model.pt`` requires
        ``model.json``. Its model section is copied into the
        specification, including the complete element order, so
        execution does not depend on this source JSON file.
        Alternatively, training_parameters may contain an explicit model
        mapping. Presets are not accepted: model settings must be
        readable in the configuration itself.

        Returns
        -------
        dict[str, Any]
            Independent model, loss, learning-rate, optimizer, and
            training settings. The adapter separates these into DeepMD's
            JSON sections.

        Raises
        ------
        OSError
            If the matching JSON cannot be read when no model was
            supplied.
        ValueError
            If the epoch count, JSON, model mapping, or element list is
            invalid, or the model relies on a preset rather than
            explicit settings.
        """
        aliases = {
            "numb_steps",
            "stop_batch",
            "num_step",
            "num_steps",
            "numb_step",
            "num_epochs",
            "num_epoch",
            "numb_epochs",
        }
        # Intercept illegal aliases for DeepMD training length, which
        # must be specified as 'numb_epoch'.
        if aliases & (self.training_parameters or {}).keys():
            raise ValueError(
                "DeepMD training length must use canonical 'numb_epoch'."
            )
        parameters = super().prepare_training()
        if "model" not in parameters:
            path = (
                self.pretrained_model_dir / self.model_filename
            ).with_suffix(".json")
            config = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(config, dict):
                raise ValueError("DeepMD configuration must be a JSON object.")
            parameters["model"] = deepcopy(config.get("model"))
        model = parameters["model"]
        if not isinstance(model, dict) or "preset" in model:
            raise ValueError(
                "Supply an explicit DeepMD model mapping, without preset."
            )
        type_map = model.get("type_map")
        if (
            not isinstance(type_map, list)
            or not type_map
            or not all(
                isinstance(element, str) and element for element in type_map
            )
            or len(set(type_map)) != len(type_map)
        ):
            raise ValueError(
                "DeepMD model.type_map must list distinct elements in"
                " checkpoint order."
            )
        return parameters


@BaseSpecBuilder.register(name="dpa4")
class DPA4SpecBuilder(_DeepMDSpecBuilder):
    """Build a selected DPA-4 checkpoint specification with 60 training
    epochs."""

    mlff_type = "dpa4"


@BaseSpecBuilder.register(name="dpa4c")
class DPA4CSpecBuilder(_DeepMDSpecBuilder):
    """Build a selected DPA-4C checkpoint specification with 100
    training epochs."""

    mlff_type = "dpa4c"
    training_defaults = {
        **_DeepMDSpecBuilder.training_defaults,
        "numb_epoch": 100,
    }
