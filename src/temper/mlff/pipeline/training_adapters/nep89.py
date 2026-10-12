"""NEP-89 TorchNEP runtime adapter."""

from __future__ import annotations

from math import isclose
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr
import shutil

from ase.data import chemical_symbols
from temper.utils.extxyz import iter_extxyz_metadata

from temper.mlff.pipeline.training_adapters.base import BaseMLFFAdapter


def _nep_architecture(path: Path) -> tuple[list[str], set[str]]:
    """Translate a GPUMD NEP4 header into TorchNEP ``nep.in`` lines (architecture-relevant sections)."""
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


@BaseMLFFAdapter.register(name="nep89")
class NEP89Adapter(BaseMLFFAdapter):
    """Prepare TorchNEP 1.0.2 and calorine 3.5 bundles."""

    pretrained_model_filename = "nep89.txt"
    training_input_filenames = {"config": "nep.in"}
    training_output_filenames = {
        "trained_model": "finetuned_nep89.txt", "training_log": "training.log",
        "native_outputs": "torchnep",
    }

    def _check_chemical_symbols(self, model_symbols: set[str]) -> None:
        """Require periodic data whose species are covered by the pretrained model."""
        assert self.training_unit.train_set is not None
        filenames = [self.input_path("train")]
        if self.training_unit.val_set is not None:
            filenames.append(self.input_path("validation"))
        symbols: set[str] = set()
        for filename in filenames:
            source = self.root / filename
            for frame_index, frame in enumerate(iter_extxyz_metadata(source, read_symbols=True)):
                if not frame.fully_periodic:
                    raise ValueError(
                        "NEP-89 fine-tuning requires fully periodic structures; "
                        f"frame {frame_index} in {source} is not periodic in "
                        "every direction."
                    )
                symbols.update(frame.symbols)
        missing = sorted(symbols - model_symbols)
        if missing:
            raise ValueError(
                "NEP training data contains elements absent from the pretrained "
                f"model: {missing!r}."
            )

    def _training_file_contents(self, training_stress: bool) -> dict[str, str]:
        """Return nep.in with checkpoint architecture and recipe controls.

        The pretrained header determines species and network dimensions.
        Dataset species must be covered, periodicity must be complete, and
        validation data is required for selecting nep_best.txt.
        """
        if self.training_unit.val_set is None:
            raise ValueError("NEP-89 fine-tuning requires a validation dataset.")
        model_path = self.root / self.files.pretrained_model
        architecture, model_symbols = _nep_architecture(model_path)

        self._check_chemical_symbols(model_symbols)

        parameters = dict(self.spec.training_parameters or {})
        # run_seed belongs to train_nep's Python API, not the nep.in grammar.
        parameters.pop("run_seed", None)
        if not training_stress:
            parameters["lambda_v"] = 0.0
            parameters["stage2_lambda_v"] = 0.0
        lines = architecture + [
            _parameter_line(key, value) for key, value in parameters.items()
        ]
        return {self.input_path("config"): "\n".join(lines) + "\n"}

    def _training_commands(self, training_stress: bool) -> tuple[str, ...]:
        """Return no shell commands: TorchNEP is called through its Python API."""
        return ()

    def train(self) -> None:
        """Fine-tune with TorchNEP, save its log, and copy the selected model.

        All input and output paths resolve against the bundle root. Returns
        None after copying nep_best.txt to the bundle's trained-model path.
        Backend exceptions propagate to stop the pipeline before evaluation.
        """
        from torchnep import train_nep

        output = self.root / self.output_path("native_outputs")
        log_path = self.root / self.output_path("training_log")
        with log_path.open("w", encoding="utf-8") as log:
            with redirect_stdout(log), redirect_stderr(log):
                train_nep(
                    str(self.root / self.input_path("config")),
                    str(self.root / self.input_path("train")),
                    valid_file=str(self.root / self.input_path("validation")),
                    finetune_from=str(self.root / self.files.pretrained_model),
                    output_dir=str(output),
                    device="cuda", restart=False, recompute_q_scaler=False,
                    slim_types=True,
                    run_seed=(self.spec.training_parameters or {}).get("run_seed", 42),
                )
        shutil.copy2(output / "nep_best.txt", self.root / self.trained_model_path)
