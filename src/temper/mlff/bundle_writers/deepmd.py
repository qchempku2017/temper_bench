"""Submit-folder writers for DPA-4 and DPA-4C through DeepMD-kit."""

from __future__ import annotations

import json
from pathlib import PurePosixPath

from temper.mlff.bundle_writers.base import BaseMLFFBundleWriter, command
from temper.utils.defaults import (
    DEFAULT_MLFF_DATASETS_DIR,
    DEFAULT_MLFF_OUTPUTS_DIR,
    DEFAULT_MLFF_RUNTIME_DIR,
    DEFAULT_MLFF_TRAINING_DIR,
)


class _DeepMDBundleWriter(BaseMLFFBundleWriter):
    calculator_resource = "calculators/deepmd.py"
    backend_flag: str

    def extra_runtime_resources(self) -> dict[str, str]:
        return {"prepare_deepmd.py": "data_preparation/deepmd.py"}

    def generated_training_files(self, training_stress: bool) -> dict[str, str]:
        config_path = self._verify_artifact(self.artifact("config"))
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(
                f"Invalid DeepMD JSON configuration: {config_path}."
            ) from error
        if not isinstance(config, dict):
            raise ValueError("DeepMD configuration must contain a JSON object.")

        model = config.get("model")
        if not isinstance(model, dict):
            raise ValueError("DeepMD configuration must contain a model object.")
        type_map = model.get("type_map")
        if (
            not isinstance(type_map, list)
            or not type_map
            or any(not isinstance(symbol, str) or not symbol for symbol in type_map)
        ):
            raise ValueError(
                "DeepMD configuration model.type_map must be a nonempty list "
                "of element symbols."
            )
        config["model"] = {
            "type_map": type_map,
            "descriptor": {},
            "fitting_net": {},
        }

        length_aliases = {
            "numb_steps",
            "stop_batch",
            "num_step",
            "num_steps",
            "numb_step",
            "num_epochs",
            "num_epoch",
            "numb_epochs",
        }
        training = dict(config.get("training", {}))
        for key in length_aliases | {"numb_epoch"}:
            training.pop(key, None)
        overrides = dict(self.spec.training or {})
        disallowed = sorted(length_aliases & overrides.keys())
        if disallowed:
            raise ValueError(
                "DeepMD training length must use canonical 'numb_epoch'; "
                f"unsupported keys: {disallowed!r}."
            )
        epoch = overrides.get("numb_epoch")
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch <= 0:
            raise ValueError(
                "DeepMD training parameter 'numb_epoch' must be a positive "
                "integer."
            )
        training.update(overrides)
        training["training_data"] = {
            "systems": [f"{DEFAULT_MLFF_TRAINING_DIR}/data/train"],
            "batch_size": "auto",
        }
        if self.unit.val_set is None:
            training.pop("validation_data", None)
        else:
            training["validation_data"] = {
                "systems": [f"{DEFAULT_MLFF_TRAINING_DIR}/data/validation"],
                "batch_size": "auto",
            }
        training["save_ckpt"] = f"{DEFAULT_MLFF_TRAINING_DIR}/deepmd/model.ckpt"
        config["training"] = training

        loss = dict(config.get("loss", {}))
        stress_weight = 0.1 if training_stress else 0.0
        loss["start_pref_e"] = 1.0
        loss["limit_pref_e"] = 1.0
        loss["start_pref_f"] = 1.0
        loss["limit_pref_f"] = 1.0
        loss["start_pref_v"] = stress_weight
        loss["limit_pref_v"] = stress_weight
        config["loss"] = loss
        return {
            f"{DEFAULT_MLFF_TRAINING_DIR}/input.json": (
                json.dumps(config, indent=2, sort_keys=True) + "\n"
            )
        }

    def training_lines(self, training_stress: bool) -> tuple[str, ...]:
        del training_stress
        lines = [
            command(
                "$PYTHON_BIN",
                f"{DEFAULT_MLFF_RUNTIME_DIR}/prepare_deepmd.py",
                "--input",
                f"{DEFAULT_MLFF_DATASETS_DIR}/train.extxyz",
                "--output",
                f"{DEFAULT_MLFF_TRAINING_DIR}/data/train",
            )
        ]
        if self.unit.val_set is not None:
            lines.append(
                command(
                    "$PYTHON_BIN",
                    f"{DEFAULT_MLFF_RUNTIME_DIR}/prepare_deepmd.py",
                    "--input",
                    f"{DEFAULT_MLFF_DATASETS_DIR}/validation.extxyz",
                    "--output",
                    f"{DEFAULT_MLFF_TRAINING_DIR}/data/validation",
                )
            )
        lines.append(
            command(
                "mkdir", "-p", f"{DEFAULT_MLFF_TRAINING_DIR}/deepmd"
            )
        )
        lines.append(
            command(
                "dp",
                self.backend_flag,
                "train",
                f"{DEFAULT_MLFF_TRAINING_DIR}/input.json",
                "--finetune",
                self.artifact_path("model"),
                "--use-pretrain-script",
                "--output",
                f"{DEFAULT_MLFF_OUTPUTS_DIR}/train_adapted.json",
            )
        )
        output_without_suffix = str(
            PurePosixPath(self.trained_model_path).with_suffix("")
        )
        freeze = [
            "dp",
            self.backend_flag,
            "freeze",
            "-c",
            f"{DEFAULT_MLFF_TRAINING_DIR}/deepmd/model.ckpt.pt",
            "-o",
            output_without_suffix,
        ]
        if self.mlff_type == "dpa4c":
            freeze.extend(("--lower-kind", "graph"))
        lines.append(command(*freeze))
        return tuple(lines)


class DPA4BundleWriter(_DeepMDBundleWriter):
    """Write fixed-layout DPA-4 train-and-test bundles."""

    mlff_type = "dpa4"
    backend_flag = "--pt"
    model_filenames = {"model": "dpa4.pt", "config": "dpa4.json"}
    trained_model_filename = "dpa4.pt2"


class DPA4CBundleWriter(_DeepMDBundleWriter):
    """Write fixed-layout DPA-4C train-and-test bundles."""

    mlff_type = "dpa4c"
    backend_flag = "--pt-expt"
    model_filenames = {"model": "dpa4c.pt", "config": "dpa4c.json"}
    trained_model_filename = "dpa4c.pt2"


__all__ = ["DPA4BundleWriter", "DPA4CBundleWriter"]
