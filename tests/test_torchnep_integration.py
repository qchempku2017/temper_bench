"""Optional smoke test against an installed TorchNEP 1.0.2."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path

import pytest
from ase.io import write

from conftest import make_frame


torchnep = pytest.importorskip("torchnep")


def test_one_epoch_finetune_writes_reloadable_gpumd_model(tmp_path: Path) -> None:
    if torchnep.__version__ != "1.0.2":
        pytest.skip("TEMPER's TorchNEP integration is pinned to 1.0.2")

    model = tmp_path / "source-nep.txt"
    model.write_text(
        "nep4 1 H\n"
        "cutoff 3 3 10 10\n"
        "n_max 0 0\n"
        "basis_size 0 0\n"
        "l_max 1 0 0\n"
        "ANN 1 0\n"
        + "0\n" * 7
        + "1\n" * 2,
        encoding="utf-8",
    )
    config = tmp_path / "nep.in"
    config.write_text(
        "version 4\n"
        "type 1 H\n"
        "cutoff 3 3\n"
        "n_max 0 0\n"
        "basis_size 0 0\n"
        "l_max 1 0 0\n"
        "neuron 1\n"
        "epoch 1\n"
        "batch 1\n"
        "lambda_v 0\n"
        "early_stop 0\n",
        encoding="utf-8",
    )

    def frame(tag: str):
        atoms = make_frame("H2", -1.0, tag, stress=False)
        atoms.positions[1, 0] = 1.0
        atoms.set_cell([10, 10, 10])
        atoms.set_pbc(True)
        return atoms

    train = tmp_path / "train.xyz"
    validation = tmp_path / "validation.xyz"
    write(train, [frame("train-0"), frame("train-1")], format="extxyz")
    write(validation, frame("validation"), format="extxyz")
    output = tmp_path / "output"

    torchnep.train_nep(
        str(config),
        str(train),
        output_dir=str(output),
        device="cpu",
        use_compile=False,
        finetune_from=str(model),
        restart=False,
        recompute_q_scaler=False,
        slim_types=True,
        run_seed=42,
        valid_file=str(validation),
    )

    trained = output / "nep_best.txt"
    assert trained.is_file()
    calculator = getattr(import_module("torchnep.nep"), "NEPCalculator")

    assert calculator(str(trained)).type_names == ["H"]
