"""Runtime adapters for DPA-4 and DPA-4C through DeepMD-kit."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import PurePosixPath

from temper.mlff.pipeline.training_adapters.base import BaseMLFFAdapter, command
from ase.data import chemical_symbols


class _DeepMDAdapter(BaseMLFFAdapter):
    calculator_module = "temper.mlff.pipeline.calculators.deepmd"
    backend_flag: str

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
            f"{self.files.training_dir}/data/train"
        ]
        if self.training_unit.val_set is not None:
            training["validation_data"]["systems"] = [
                f"{self.files.training_dir}/data/validation"
            ]
        else:
            training.pop("validation_data", None)
        # DeepMD appends .pt to save_ckpt; freeze takes the resulting filename.
        training["save_ckpt"] = f"{self.files.training_dir}/deepmd/model.ckpt"
        config["training"] = training

        return {
            f"{self.files.training_dir}/input.json": (
                json.dumps(config, indent=2, sort_keys=True) + "\n"
            )
        }

    def prepare_training(self) -> None:
        """Convert packaged train/validation data before native training starts."""
        from temper.mlff.pipeline.data_preparation.deepmd import prepare_dataset

        for filename, name in (
            (self.training_unit.train_set, "train"),
            (self.training_unit.val_set, "validation"),
        ):
            if filename is not None:
                prepare_dataset(
                    self.root / self.dataset_path(filename),
                    self.root / self.files.training_dir / "data" / name,
                )

    def training_lines(self, training_stress: bool) -> tuple[str, ...]:
        """Return CUDA training, freezing and portable-checkpoint copy commands."""
        del training_stress
        lines = ["export DEVICE=cuda"]
        lines.append(
            command(
                "mkdir", "-p", f"{self.files.training_dir}/deepmd"
            )
        )
        lines.append(
            command(
                "dp",
                self.backend_flag,
                "train",
                f"{self.files.training_dir}/input.json",
                "--finetune",
                self.artifact_path("model"),
                "--use-pretrain-script",
                "--output",
                f"{self.files.outputs_dir}/train_adapted.json",
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
            f"{self.files.training_dir}/deepmd/model.ckpt.pt",
            "-o",
            output_without_suffix,
        ]
        if self.mlff_type == "dpa4c":
            freeze.extend(("--lower-kind", "graph"))
        lines.append(command(*freeze))
        lines.append(
            command(
                "cp",
                f"{self.files.training_dir}/deepmd/model.ckpt.pt",
                f"{self.files.artifacts_dir}/model.ckpt.pt",
            )
        )
        return tuple(lines)


@BaseMLFFAdapter.register(name="dpa4")
class DPA4Adapter(_DeepMDAdapter):
    """Prepare DPA-4 train-and-test bundles."""

    mlff_type = "dpa4"
    backend_flag = "--pt"
    model_filenames = {"model": "dpa4.pt"}
    trained_model_filename = "dpa4.pt2"


@BaseMLFFAdapter.register(name="dpa4c")
class DPA4CAdapter(_DeepMDAdapter):
    """Prepare DPA-4C train-and-test bundles."""

    mlff_type = "dpa4c"
    backend_flag = "--pt-expt"
    model_filenames = {"model": "dpa4c.pt"}
    trained_model_filename = "dpa4c.pt2"


__all__ = ["DPA4Adapter", "DPA4CAdapter"]
