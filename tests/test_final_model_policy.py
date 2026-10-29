"""Regression tests for final-epoch selection without GPU training."""

from collections.abc import Callable
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
from ruamel.yaml import YAML

from temper.mlff.pipeline import runner
from temper.mlff.pipeline.training_adapters.base import BaseMLFFAdapter
from temper.schemas.mlff_spec import MLFFSpec
from temper.schemas.mlff_train_bundle import MLFFTrainBundle
from temper.schemas.train_unit import TrainingUnit


@pytest.mark.parametrize(
    "option,value",
    [("ema", True), ("swa", True), ("save_all_checkpoints", False)],
)
def test_mace_rejects_best_or_averaged_export(
    option: str, value: bool, mlff_spec_factory: Callable[..., MLFFSpec]
) -> None:
    with pytest.raises(ValueError, match="final-epoch"):
        mlff_spec_factory("mace", training_parameters={option: value})


def test_sevennet_nonmultiple_epoch_budget_is_saved(
    tmp_path: Path,
    finetune_training_unit: TrainingUnit,
    mlff_spec_factory: Callable[..., MLFFSpec],
) -> None:
    root, bundle = MLFFTrainBundle(
        training_unit=finetune_training_unit,
        mlff_spec=mlff_spec_factory(
            "sevennet", training_parameters={"epoch": 7}
        ),
    ).write_submit_folder(tmp_path / "job")
    runner.prepare_bundle(root / "bundle.json")
    config = YAML(typ="safe").load(
        root / bundle.files.training_input_files["config"]
    )
    assert 7 % config["train"]["per_epoch"] == 0
    assert (
        "checkpoint_7.pth"
        in (root / bundle.files.training_input_files["script"]).read_text()
    )


def test_nep_early_final_model_is_rejected(
    tmp_path: Path,
    finetune_training_unit: TrainingUnit,
    mlff_spec_factory: Callable[..., MLFFSpec],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, bundle = MLFFTrainBundle(
        training_unit=finetune_training_unit,
        mlff_spec=mlff_spec_factory("nep89", training_parameters={"epoch": 7}),
    ).write_submit_folder(tmp_path / "job")
    adapter = runner.prepare_bundle(root / "bundle.json")
    native = ModuleType("torchnep")

    def train(*args: object, **kwargs: object) -> None:
        output = root / bundle.files.training_output_files["native_outputs"]
        output.mkdir(exist_ok=True)
        (output / "loss.out").write_text("3 0.1\n")
        (output / "nep_final.txt").write_text("early final")

    native.train_nep = train
    monkeypatch.setitem(sys.modules, "torchnep", native)
    with pytest.raises(RuntimeError, match="final epoch"):
        adapter.train()
    assert not (root / adapter.trained_model_path).exists()


@pytest.mark.parametrize("last_epoch", [5, 6])
def test_mattersim_checks_last_epoch(
    last_epoch: int,
    tmp_path: Path,
    finetune_training_unit: TrainingUnit,
    mlff_spec_factory: Callable[..., MLFFSpec],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _ = MLFFTrainBundle(
        training_unit=finetune_training_unit,
        mlff_spec=mlff_spec_factory(
            "mattersim", training_parameters={"epochs": 7}
        ),
    ).write_submit_folder(tmp_path / "job")
    adapter = runner.prepare_bundle(root / "bundle.json")
    monkeypatch.setattr(BaseMLFFAdapter, "train", lambda self: None)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            load=lambda *args, **kwargs: {"last_epoch": last_epoch}
        ),
    )
    if last_epoch == 6:
        adapter.train()
    else:
        with pytest.raises(RuntimeError, match="final epoch"):
            adapter.train()


def test_mattersim_launcher_disables_averaging_only_in_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from temper.mlff.pipeline import mattersim_final

    decay_values: list[float] = []

    class Average:
        def __init__(
            self,
            parameters: object,
            decay: float,
            use_num_updates: bool = True,
        ) -> None:
            decay_values.append(decay)

    potential = SimpleNamespace(ExponentialMovingAverage=Average)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "torch_ema",
        SimpleNamespace(ExponentialMovingAverage=Average),
    )
    monkeypatch.setitem(sys.modules, "mattersim", ModuleType("mattersim"))
    monkeypatch.setitem(
        sys.modules,
        "mattersim.forcefield",
        SimpleNamespace(potential=potential),
    )

    def run(module: str, run_name: str) -> None:
        assert (
            module == "mattersim.training.finetune_mattersim"
            and run_name == "__main__"
        )
        potential.ExponentialMovingAverage([], decay=0.99)

    monkeypatch.setattr(mattersim_final.runpy, "run_module", run)
    mattersim_final._main()
    assert decay_values == [0.0]
    assert potential.ExponentialMovingAverage is Average


def test_deepmd_materializes_final_symlink(
    tmp_path: Path,
    finetune_training_unit: TrainingUnit,
    mlff_spec_factory: Callable[..., MLFFSpec],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _ = MLFFTrainBundle(
        training_unit=finetune_training_unit,
        mlff_spec=mlff_spec_factory("dpa4"),
    ).write_submit_folder(tmp_path / "job")
    adapter = runner.prepare_bundle(root / "bundle.json")
    final = root / adapter.trained_model_path
    checkpoint = root / adapter.output_path("checkpoints") / "model.ckpt-7.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"last weights")
    try:
        final.symlink_to(checkpoint)
    except OSError:
        pytest.skip("Host does not permit creating symbolic links")
    monkeypatch.setattr(BaseMLFFAdapter, "train", lambda self: None)
    adapter.train()
    assert not final.is_symlink() and final.read_bytes() == b"last weights"
