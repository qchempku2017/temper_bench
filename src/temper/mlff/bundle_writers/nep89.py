"""NEP-89 TorchNEP submit-folder writer."""

from __future__ import annotations

from math import isclose
from pathlib import Path

from ase.data import chemical_symbols
from ase.io import iread

from temper.mlff.bundle_writers.base import BaseMLFFBundleWriter, command
from temper.utils.defaults import (
    DEFAULT_MLFF_DATASETS_DIR,
    DEFAULT_MLFF_RUNTIME_DIR,
    DEFAULT_MLFF_TRAINING_DIR,
)


def _nep_architecture(path: Path) -> tuple[list[str], set[str]]:
    """Translate a GPUMD NEP4 header into TorchNEP ``nep.in`` lines."""
    with path.open(encoding="utf-8") as stream:
        rows = [stream.readline().split() for _ in range(7)]

    header = rows[0]
    if len(header) < 3 or header[0] not in {"nep4", "nep4_zbl"}:
        raise ValueError(
            "NEP-89 fine-tuning requires a GPUMD nep4 or nep4_zbl model."
        )
    try:
        type_count = int(header[1])
    except ValueError as error:
        raise ValueError("Invalid element count in pretrained NEP header.") from error
    symbols = header[2:]
    if type_count <= 0 or len(symbols) != type_count:
        raise ValueError(
            "Pretrained NEP header element count does not match its type list."
        )
    if len(set(symbols)) != len(symbols) or any(
        symbol not in chemical_symbols[1:] for symbol in symbols
    ):
        raise ValueError("Pretrained NEP header contains invalid element symbols.")

    cursor = 1

    def take(name: str, minimum_values: int) -> list[str]:
        nonlocal cursor
        row = rows[cursor] if cursor < len(rows) else []
        cursor += 1
        if len(row) < minimum_values + 1 or row[0].lower() != name.lower():
            raise ValueError(f"Pretrained NEP header is missing a valid {name} line.")
        return row[1:]

    lines = ["version 4", f"type {type_count} " + " ".join(symbols)]
    if header[0] == "nep4_zbl":
        zbl = take("zbl", 2)
        if len(zbl) not in {2, 3}:
            raise ValueError("Pretrained NEP zbl line has an unsupported shape.")
        try:
            inner, outer = map(float, zbl[:2])
            factor = float(zbl[2]) if len(zbl) == 3 else None
        except ValueError as error:
            raise ValueError("Pretrained NEP zbl cutoffs must be numeric.") from error
        if inner <= 0 or outer <= 0 or not isclose(
            inner * 2.0, outer, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise ValueError(
                "TorchNEP requires the pretrained ZBL inner cutoff to equal "
                "half the outer cutoff."
            )
        if factor is not None and factor <= 0:
            raise ValueError("Pretrained typewise ZBL factor must be positive.")
        lines.append(f"zbl {zbl[1]}")
        if factor is not None:
            lines.append(f"use_typewise_cutoff_zbl {zbl[2]}")

    cutoff = take("cutoff", 2)
    n_max = take("n_max", 2)
    basis_size = take("basis_size", 2)
    l_max = take("l_max", 1)
    ann = take("ANN", 1)
    try:
        if len(cutoff) not in {2, 4}:
            raise ValueError
        if len(n_max) != 2 or len(basis_size) != 2:
            raise ValueError
        if any(float(value) <= 0 for value in cutoff[:2]):
            raise ValueError
        if any(int(value) < 0 for value in n_max[:2] + basis_size[:2]):
            raise ValueError
        if (
            not 1 <= len(l_max) <= 7
            or int(l_max[0]) <= 0
            or any(int(value) < 0 for value in l_max[1:])
        ):
            raise ValueError
        if len(ann) != 2 or int(ann[0]) <= 0 or int(ann[1]) != 0:
            raise ValueError
    except ValueError as error:
        raise ValueError(
            "Pretrained NEP architecture contains invalid values."
        ) from error

    lines.extend(
        (
            "cutoff " + " ".join(cutoff[:2]),
            "n_max " + " ".join(n_max[:2]),
            "basis_size " + " ".join(basis_size[:2]),
            "l_max " + " ".join(l_max),
            f"neuron {ann[0]}",
        )
    )
    return lines, set(symbols)


def _parameter_line(key: str, value: str | int | float | bool) -> str:
    """Render one scalar TorchNEP hyperparameter."""
    if isinstance(value, bool):
        value = int(value)
    return f"{key} {value}"


class NEP89BundleWriter(BaseMLFFBundleWriter):
    """Write fixed-layout TorchNEP 1.0.2 and calorine 3.5 bundles."""

    mlff_type = "nep89"
    calculator_resource = "calculators/nep89.py"
    model_filenames = {"model": "nep89.txt"}
    trained_model_filename = "nep89.txt"

    @property
    def work_directory(self) -> str:
        """Return the TorchNEP working directory within the submit folder."""
        return f"{DEFAULT_MLFF_TRAINING_DIR}/torchnep"

    def validate_bundle(self) -> None:
        if "restart" in self.spec.pretrained_model.artifacts:
            raise ValueError(
                "NEP-89 TorchNEP fine-tuning no longer uses a restart artifact; "
                "rebuild this MLFF specification."
            )
        super().validate_bundle()
        if self.bundle.unit_type == "finetune" and self.unit.val_set is None:
            raise ValueError("NEP-89 fine-tuning requires a validation dataset.")

    def extra_runtime_resources(self) -> dict[str, str]:
        return {
            "device.py": "device.py",
            "prepare_nep89.py": "data_preparation/nep89.py",
            "train_nep89.py": "train_nep89.py",
        }

    def _training_symbols(self) -> set[str]:
        assert self.unit.train_set is not None
        filenames = [self.unit.train_set]
        if self.unit.val_set is not None:
            filenames.append(self.unit.val_set)
        symbols: set[str] = set()
        for filename in filenames:
            source = self._dataset_source(filename)
            for frame_index, atoms in enumerate(iread(source, index=":")):
                if not all(bool(value) for value in atoms.pbc):
                    raise ValueError(
                        "NEP-89 fine-tuning requires fully periodic structures; "
                        f"frame {frame_index} in {source} is not periodic in "
                        "every direction."
                    )
                symbols.update(
                    chemical_symbols[int(number)] for number in atoms.numbers
                )
        return symbols

    def generated_training_files(self, training_stress: bool) -> dict[str, str]:
        model_path = self._verify_artifact(self.artifact("model"))
        architecture, model_symbols = _nep_architecture(model_path)
        missing = sorted(self._training_symbols() - model_symbols)
        if missing:
            raise ValueError(
                "NEP training data contains elements absent from the pretrained "
                f"model: {missing!r}."
            )

        parameters = dict(self.spec.training or {})
        epoch = parameters.get("epoch")
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch <= 0:
            raise ValueError(
                "TorchNEP training parameter 'epoch' must be a positive integer."
            )
        parameters["early_stop"] = 0
        parameters["lambda_v"] = 0.01 if training_stress else 0.0
        parameters["stage2_lambda_v"] = 0.1 if training_stress else 0.0
        lines = architecture + [
            _parameter_line(key, value) for key, value in parameters.items()
        ]
        return {f"{self.work_directory}/nep.in": "\n".join(lines) + "\n"}

    def training_lines(self, training_stress: bool) -> tuple[str, ...]:
        prepare = [
            "$PYTHON_BIN",
            f"{DEFAULT_MLFF_RUNTIME_DIR}/prepare_nep89.py",
            "--train",
            f"{DEFAULT_MLFF_DATASETS_DIR}/train.extxyz",
            "--validation",
            f"{DEFAULT_MLFF_DATASETS_DIR}/validation.extxyz",
            "--output-directory",
            self.work_directory,
        ]
        if training_stress:
            prepare.append("--stress")
        output = f"{self.work_directory}/output"
        return (
            command(
                "$PYTHON_BIN",
                f"{DEFAULT_MLFF_RUNTIME_DIR}/device.py",
                "require-cuda",
            ),
            command(*prepare),
            command(
                "$PYTHON_BIN",
                f"{DEFAULT_MLFF_RUNTIME_DIR}/train_nep89.py",
                "--config",
                f"{self.work_directory}/nep.in",
                "--train",
                f"{self.work_directory}/train.xyz",
                "--validation",
                f"{self.work_directory}/test.xyz",
                "--model",
                self.artifact_path("model"),
                "--output-directory",
                output,
            ),
            command("cp", f"{output}/nep_best.txt", self.trained_model_path),
        )


__all__ = ["NEP89BundleWriter"]
