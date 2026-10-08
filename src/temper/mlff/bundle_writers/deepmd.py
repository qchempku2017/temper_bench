"""Submit-folder writers for DPA-4 and DPA-4C through DeepMD-kit."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import PurePosixPath

from temper.mlff.bundle_writers.base import BaseMLFFBundleWriter, command
from ase.data import chemical_symbols
from temper.utils.defaults import (
    DEFAULT_MLFF_ARTIFACTS_DIR,
    DEFAULT_MLFF_DATASETS_DIR,
    DEFAULT_MLFF_OUTPUTS_DIR,
    DEFAULT_MLFF_RUNTIME_DIR,
    DEFAULT_MLFF_TRAINING_DIR,
)


class _DeepMDBundleWriter(BaseMLFFBundleWriter):
    calculator_resource = "calculators/deepmd.py"
    backend_flag: str

    def extra_runtime_resources(self) -> dict[str, str]:
        """Return the extxyz-to-DeepMD data converter resource mapping."""
        return {"prepare_deepmd.py": "data_preparation/deepmd.py"}

    def generated_training_files(self, training_stress: bool) -> dict[str, str]:
        """Return native inputs with all 118 elements in atomic-number order."""
        training = deepcopy(self.spec.training_parameters)
        config = {
            "model": {"type_map": chemical_symbols[1:]},
            "loss": training.pop("loss"),
            "learning_rate": training.pop("learning_rate"),
        }
        if not training_stress:
            config["loss"]["start_pref_v"] = 0.0
            config["loss"]["limit_pref_v"] = 0.0
        training["training_data"]["systems"] = [
            f"{DEFAULT_MLFF_TRAINING_DIR}/data/train"
        ]
        if self.training_unit.val_set is not None:
            training["validation_data"]["systems"] = [
                f"{DEFAULT_MLFF_TRAINING_DIR}/data/validation"
            ]
        else:
            training.pop("validation_data", None)
        # DeepMD appends .pt to save_ckpt; freeze takes the resulting filename.
        training["save_ckpt"] = f"{DEFAULT_MLFF_TRAINING_DIR}/deepmd/model.ckpt"
        config["training"] = training

        return {
            f"{DEFAULT_MLFF_TRAINING_DIR}/input.json": (
                json.dumps(config, indent=2, sort_keys=True) + "\n"
            )
        }

    def training_lines(self, training_stress: bool) -> tuple[str, ...]:
        """Return CUDA training, freezing and portable-checkpoint copy commands."""
        del training_stress
        lines = [
            "export DEVICE=cuda",
            command(
                "$PYTHON_BIN",
                f"{DEFAULT_MLFF_RUNTIME_DIR}/prepare_deepmd.py",
                "--input",
                f"{DEFAULT_MLFF_DATASETS_DIR}/train.extxyz",
                "--output",
                f"{DEFAULT_MLFF_TRAINING_DIR}/data/train",
            )
        ]
        if self.training_unit.val_set is not None:
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
        lines.append(
            command(
                "cp",
                f"{DEFAULT_MLFF_TRAINING_DIR}/deepmd/model.ckpt.pt",
                f"{DEFAULT_MLFF_ARTIFACTS_DIR}/model.ckpt.pt",
            )
        )
        return tuple(lines)


@BaseMLFFBundleWriter.register(name="dpa4")
class DPA4BundleWriter(_DeepMDBundleWriter):
    """Write fixed-layout DPA-4 train-and-test bundles."""

    mlff_type = "dpa4"
    backend_flag = "--pt"
    model_filenames = {"model": "dpa4.pt"}
    trained_model_filename = "dpa4.pt2"


@BaseMLFFBundleWriter.register(name="dpa4c")
class DPA4CBundleWriter(_DeepMDBundleWriter):
    """Write fixed-layout DPA-4C train-and-test bundles."""

    mlff_type = "dpa4c"
    backend_flag = "--pt-expt"
    model_filenames = {"model": "dpa4c.pt"}
    trained_model_filename = "dpa4c.pt2"


__all__ = ["DPA4BundleWriter", "DPA4CBundleWriter"]
