"""Runtime adapters for DPA-4 and DPA-4C through DeepMD-kit."""

from __future__ import annotations

import json
import os
import subprocess
import shutil
from copy import deepcopy
from pathlib import PurePosixPath

from temper.mlff.pipeline.training_adapters.base import BaseMLFFAdapter, command
from ase.data import chemical_symbols


class _DeepMDAdapter(BaseMLFFAdapter):
    backend_flag: str
    freeze_options: tuple[str, ...] = ()
    evaluation_model_filename = "evaluation.pt2"
    training_output_filenames = {
        "trained_model": "model.ckpt.pt",
        "training_log": "training.log", "adapted_config": "train_adapted.json",
        "learning_curve": "lcurve.out",
        "checkpoints": "deepmd/checkpoints",
    }
    training_input_filenames = {
        "script": "run.sh", "config": "input.json",
        "converted_train": "data/train", "converted_validation": "data/validation",
    }

    def _training_file_contents(self, training_stress: bool) -> dict[str, str]:
        """Return native inputs with all 118 elements in atomic-number order."""
        assert self.spec.training_parameters is not None
        training = deepcopy(self.spec.training_parameters)
        assert "loss" in training and "learning_rate" in training
        config = {
            "model": {"type_map": chemical_symbols[1:]},
            "loss": training.pop("loss"),
            "learning_rate": training.pop("learning_rate"),
        }
        if not training_stress:
            config["loss"]["start_pref_v"] = 0.0
            config["loss"]["limit_pref_v"] = 0.0
        training["training_data"]["systems"] = [
            self.input_path("converted_train")
        ]
        if self.training_unit.val_set is not None:
            training["validation_data"]["systems"] = [
                self.input_path("converted_validation")
            ]
        else:
            training.pop("validation_data", None)
        # DeepMD appends .pt to save_ckpt; freeze takes the resulting filename.
        training["save_ckpt"] = str(PurePosixPath(self.trained_model_path).with_suffix(""))
        # Older packaged bundles predate the explicit checkpoint-directory map.
        training["save_dir"] = self.files.training_output_files.get("checkpoints", "deepmd/checkpoints")
        training["disp_file"] = self.output_path("learning_curve")
        config["training"] = training

        return {
            self.input_path("config"): (
                json.dumps(config, indent=2, sort_keys=True) + "\n"
            )
        }

    def prepare_training(self) -> None:
        """Convert packaged train/validation data before native training starts."""
        from temper.mlff.pipeline.data_preparation.deepmd import prepare_dataset

        for name in ("train", "validation"):
            assert self.files is not None
            if name in self.files.training_input_files:
                prepare_dataset(
                    self.root / self.input_path(name),
                    self.root / self.input_path("converted_" + name),
                )

    def _training_commands(self, training_stress: bool) -> tuple[str, ...]:
        """Train on CUDA and write portable weights to the recorded output."""
        del training_stress
        assert self.files is not None
        lines = [
            "export DEVICE=cuda",
            command(
            "mkdir", "-p", str(PurePosixPath(self.trained_model_path).parent)
            ),
            command(
                "dp",
                self.backend_flag,
                "train",
                self.input_path("config"),
                "--finetune",
                self.files.pretrained_model,
                "--use-pretrain-script",
                "--output",
                self.output_path("adapted_config"),
            )
        ]
        return tuple(lines)

    def train(self) -> None:
        """Train to the native epoch budget and materialize the final checkpoint.

        Notes
        -----
        DeepMD publishes its final checkpoint through a symlink. Replace that
        link with an ordinary file so exact-file collection can reject all
        remote symlinks while retaining a self-contained final model.

        Raises
        ------
        ValueError
            If the published checkpoint points outside the bundle directory.
        OSError
            If the final checkpoint is missing or cannot be copied.
        subprocess.CalledProcessError
            If native training fails before completing its budget.
        """
        super().train()
        model = self.root / self.trained_model_path
        if not model.resolve().is_relative_to(self.root.resolve()):
            raise ValueError("DeepMD checkpoint points outside the bundle.")
        if not model.is_file():
            raise FileNotFoundError(model)
        if model.is_symlink():
            temporary = model.with_name(model.name + ".final")
            shutil.copyfile(model, temporary)
            os.replace(temporary, model)

    def prepare_evaluation(self) -> None:
        """Freeze fine-tuned portable weights on the evaluation host only."""
        if self.bundle.unit_type != "finetune":
            return
        # Bundle must have been packaged with an evaluation model filename, and the file must exist.
        assert self.files is not None
        assert self.files.evaluation_model is not None
        output_without_suffix = str(
            PurePosixPath(self.files.evaluation_model).with_suffix("")
        )
        freeze = [
            "dp",
            self.backend_flag,
            "freeze",
            "-c",
            self.trained_model_path,
            "-o",
            output_without_suffix,
            *self.freeze_options,
        ]
        with (self.root / self.output_path("training_log")).open(
            "a", encoding="utf-8"
        ) as log:
            subprocess.run(
                freeze, cwd=self.root, env={**os.environ, "DEVICE": "cuda"},
                stdout=log, stderr=subprocess.STDOUT, check=True,
            )


@BaseMLFFAdapter.register(name="dpa4")
class DPA4Adapter(_DeepMDAdapter):
    """Prepare DPA-4 train-and-test bundles."""

    backend_flag = "--pt"
    pretrained_model_filename = "dpa4.pt"


@BaseMLFFAdapter.register(name="dpa4c")
class DPA4CAdapter(_DeepMDAdapter):
    """Prepare DPA-4C train-and-test bundles."""

    backend_flag = "--pt-expt"
    pretrained_model_filename = "dpa4c.pt"
    freeze_options = ("--lower-kind", "graph")
