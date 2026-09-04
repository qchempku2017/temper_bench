"""Tests for the deliberately small MLFF specification API."""

from __future__ import annotations

import importlib
import inspect
import json
from pathlib import Path

import pytest
from monty.serialization import dumpfn, loadfn
from pydantic import ValidationError

import temper.mlff as mlff
from temper.mlff import (
    DPA4CSpecBuilder,
    DPA4SpecBuilder,
    MACESpecBuilder,
    MatterSimSpecBuilder,
    NEP89SpecBuilder,
    SevenNetSpecBuilder,
)
from temper.schemas.mlff_spec import (
    LocalArtifactRef,
    MLFFImplementation,
    MLFFSpec,
    PretrainedMLFFSpec,
)
from temper.schemas.utils import validate_submit_relative_path


FAMILIES = ("dpa4", "dpa4c", "mattersim", "mace", "sevennet", "nep89")


def test_only_six_concrete_builders_are_public() -> None:
    assert not hasattr(mlff, "ORBSpecBuilder")
    assert not hasattr(mlff, "MLFFType")
    assert not hasattr(mlff, "MLFFBundleLayout")
    assert not hasattr(mlff, "get_mlff_spec_builder")


@pytest.mark.parametrize("family", FAMILIES)
def test_each_builder_records_fixed_implementations(
    family: str,
    mlff_spec_factory,
) -> None:
    spec = mlff_spec_factory(family)
    assert spec.mlff_type == family
    assert spec.training is not None
    assert all(
        set(type(item).model_fields) == {"name", "version", "kind"}
        for item in spec.implementations
    )


def test_model_paths_are_adjustable_and_content_addressed(tmp_path: Path) -> None:
    first = tmp_path / "first.model"
    second = tmp_path / "second.model"
    first.write_bytes(b"same")
    second.write_bytes(b"same")
    one = MACESpecBuilder(pretrained_model_path=first).build()
    two = MACESpecBuilder(pretrained_model_path=second).build()

    assert one.pretrained_model.artifacts["model"].path == first
    assert one.mlff_spec_id == two.mlff_spec_id

    second.write_bytes(b"different")
    changed = MACESpecBuilder(pretrained_model_path=second).build()
    assert changed.mlff_spec_id != one.mlff_spec_id


def test_default_model_paths_cover_every_family(
    monkeypatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "pretrained_models"
    root.mkdir()
    for filename in (
        "dpa4.pt",
        "dpa4c.pt",
        "mattersim.pth",
        "mace.model",
        "sevennet.pth",
        "nep89.txt",
    ):
        (root / filename).write_bytes(filename.encode())
    for filename in ("dpa4.json", "dpa4c.json"):
        (root / filename).write_text(
            json.dumps(
                {"model": {"type_map": ["H"]}, "training": {}}
            ),
            encoding="utf-8",
        )
    monkeypatch.chdir(tmp_path)

    specs = [
        DPA4SpecBuilder().build(),
        DPA4CSpecBuilder().build(),
        MatterSimSpecBuilder().build(),
        MACESpecBuilder().build(),
        SevenNetSpecBuilder().build(),
        NEP89SpecBuilder().build(),
    ]
    assert [spec.mlff_type for spec in specs] == list(FAMILIES)


def test_mlff_directory_defaults_are_read_from_environment(monkeypatch) -> None:
    import temper.utils.defaults as defaults

    with monkeypatch.context() as environment:
        environment.setenv(
            "DEFAULT_MLFF_DATASETS_DIR", "inputs/datasets"
        )
        environment.setenv("DEFAULT_MLFF_OUTPUTS_DIR", "results")
        reloaded = importlib.reload(defaults)
        assert reloaded.DEFAULT_MLFF_DATASETS_DIR == "inputs/datasets"
        assert reloaded.DEFAULT_MLFF_OUTPUTS_DIR == "results"
    importlib.reload(defaults)


def test_mlff_directory_defaults_reject_unsafe_environment(
    monkeypatch,
) -> None:
    import temper.utils.defaults as defaults

    with monkeypatch.context() as environment:
        environment.setenv("DEFAULT_MLFF_MODELS_DIR", "../outside")
        with pytest.raises(ValueError, match="normalized relative"):
            importlib.reload(defaults)
    importlib.reload(defaults)


def test_none_means_zeroshot_and_empty_dict_enables_defaults(
    tmp_path: Path,
) -> None:
    model = tmp_path / "mace.model"
    model.write_bytes(b"mace")
    zero = MACESpecBuilder(pretrained_model_path=model).build()
    fine = MACESpecBuilder(
        pretrained_model_path=model,
        training_parameters={},
    ).build()

    assert zero.training is None
    assert fine.training is not None
    assert fine.training["max_num_epochs"] == 100


@pytest.mark.parametrize(
    ("family", "epoch_key"),
    (
        ("dpa4", "numb_epoch"),
        ("dpa4c", "numb_epoch"),
        ("mattersim", "epochs"),
        ("mace", "max_num_epochs"),
        ("sevennet", "epoch"),
        ("nep89", "epoch"),
    ),
)
def test_every_gradient_trainer_defaults_to_one_hundred_epochs(
    family: str,
    epoch_key: str,
    mlff_spec_factory,
) -> None:
    spec = mlff_spec_factory(family)
    assert spec.training[epoch_key] == 100


def test_early_stopping_is_disabled_for_exact_epoch_runs(
    mlff_spec_factory,
) -> None:
    mace = mlff_spec_factory(
        "mace", training_parameters={"max_num_epochs": 7}
    )
    mattersim = mlff_spec_factory(
        "mattersim", training_parameters={"epochs": 7}
    )
    nep = mlff_spec_factory("nep89", training_parameters={"epoch": 7})

    assert mace.training["patience"] == 8
    assert mattersim.training["early_stop_patience"] == 8
    assert nep.training["early_stop"] == 0


@pytest.mark.parametrize(
    ("family", "parameters", "message"),
    (
        ("dpa4", {"numb_steps": 10}, "numb_epoch"),
        ("dpa4", {"num_epochs": 10}, "numb_epoch"),
        ("mace", {"patience": 2}, "managed"),
        ("mattersim", {"early_stop_patience": 2}, "managed"),
        ("nep89", {"generation": 10}, "GPUMD/SNES"),
        ("nep89", {"cutoff": "6 5"}, "architecture"),
        ("nep89", {"early_stop": 2}, "must be 0"),
    ),
)
def test_managed_training_controls_reject_conflicting_overrides(
    family: str,
    parameters: dict,
    message: str,
    mlff_spec_factory,
) -> None:
    with pytest.raises(ValueError, match=message):
        mlff_spec_factory(family, training_parameters=parameters)


@pytest.mark.parametrize(
    ("family", "epoch_key"),
    (
        ("dpa4", "numb_epoch"),
        ("mattersim", "epochs"),
        ("mace", "max_num_epochs"),
        ("sevennet", "epoch"),
        ("nep89", "epoch"),
    ),
)
def test_epoch_counts_must_be_positive_integers(
    family: str,
    epoch_key: str,
    mlff_spec_factory,
) -> None:
    for invalid in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="positive integer"):
            mlff_spec_factory(
                family, training_parameters={epoch_key: invalid}
            )


def test_nep_uses_torchnep_only_for_finetuning(mlff_spec_factory) -> None:
    fine = mlff_spec_factory("nep89")
    zero = mlff_spec_factory("nep89", with_training=False)

    assert set(fine.pretrained_model.artifacts) == {"model"}
    assert {item.name for item in fine.implementations} == {
        "torchnep",
        "gpumd",
        "calorine",
    }
    assert {item.name for item in zero.implementations} == {
        "gpumd",
        "calorine",
    }
    assert "pretrained_restart_path" not in inspect.signature(
        NEP89SpecBuilder
    ).parameters


def test_nep_defaults_match_torchnep_1_0_2(mlff_spec_factory) -> None:
    assert mlff_spec_factory("nep89").training == {
        "epoch": 100,
        "batch": 32,
        "lr": 0.01,
        "stop_lr": 1e-6,
        "lambda_e": 0.01,
        "lambda_f": 1.0,
        "lambda_v": 0.01,
        "max_grad_norm": 10.0,
        "lr_scheduler": "plateau",
        "scheduler_patience": 15,
        "early_stop": 0,
        "scheduler_factor": 0.7,
        "stage2": 0,
        "stage2_lr": 1e-3,
        "stage2_lambda_e": 1.0,
        "stage2_lambda_f": 0.05,
        "stage2_lambda_v": 0.1,
        "weight_decay": 1e-4,
    }


def test_parameter_dictionaries_are_plain_and_mutable(
    mlff_spec_factory,
) -> None:
    spec = mlff_spec_factory(
        "mace",
        training_parameters={"custom": {"values": [1, 2]}},
        testing_parameters={"default_dtype": "float32"},
    )
    assert isinstance(spec.training, dict)
    assert isinstance(spec.testing, dict)
    spec.training["custom"]["values"].append(3)
    spec.testing["default_dtype"] = "float64"
    assert spec.training["custom"]["values"] == [1, 2, 3]


def test_schema_and_monty_round_trip(tmp_path: Path, mlff_spec_factory) -> None:
    spec = mlff_spec_factory("nep89")
    path = tmp_path / "spec.json"
    dumpfn(spec, path, indent=2)
    assert loadfn(path) == spec
    assert MLFFSpec.model_validate(spec.model_dump(mode="json")) == spec


def test_local_artifact_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        LocalArtifactRef.from_path(tmp_path / "missing.model")


def test_small_value_objects_forbid_removed_metadata(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.write_bytes(b"x")
    artifact = LocalArtifactRef.from_path(model)
    with pytest.raises(ValidationError):
        MLFFImplementation(
            name="package",
            version="1",
            role="calculator",
        )
    with pytest.raises(ValidationError):
        PretrainedMLFFSpec(
            name="model",
            version="1",
            artifacts={"model": artifact},
            loader="named",
        )


@pytest.mark.parametrize(
    "unsafe",
    (
        "",
        ".",
        "../escape",
        "/absolute",
        "C:/absolute",
        "a\\b",
        "a//b",
        "a/./b",
    ),
)
def test_submit_relative_path_rejects_unsafe_values(unsafe: str) -> None:
    with pytest.raises((TypeError, ValueError)):
        validate_submit_relative_path(unsafe)
