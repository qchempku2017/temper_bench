"""Tests for atomic MLFF bundle construction and fixed folder writing."""

from __future__ import annotations

import inspect
import json
import shutil
from pathlib import Path

import pytest
from ase.io import write
from monty.serialization import dumpfn, loadfn
from ruamel.yaml import YAML

from temper.mlff import (
    DPA4SpecBuilder,
    MLFFTrainBundle,
    NEP89SpecBuilder,
    build_mlff_train_bundles,
)
import temper.mlff.bundle_writers.base as writer_base
from temper.mlff.bundle_writers.base import BaseMLFFBundleWriter
from temper.mlff.bundle_writers.nep89 import _nep_architecture
from temper.schemas.mlff_spec import LocalArtifactRef

from conftest import make_frame


FAMILIES = ("dpa4", "dpa4c", "mattersim", "mace", "sevennet", "nep89")
MODEL_FILES = {
    "dpa4": ("dpa4.pt", "dpa4.json"),
    "dpa4c": ("dpa4c.pt", "dpa4c.json"),
    "mattersim": ("mattersim.pth",),
    "mace": ("mace.model",),
    "sevennet": ("sevennet.pth",),
    "nep89": ("nep89.txt",),
}
TRAINING_MARKERS = {
    "dpa4": "dp --pt train",
    "dpa4c": "dp --pt-expt train",
    "mattersim": "mattersim.training.finetune_mattersim",
    "mace": "mace_run_train --config",
    "sevennet": "sevenn train",
    "nep89": "runtime/train_nep89.py",
}


def test_bundle_schema_contains_only_the_pair_and_identity(
    finetune_training_unit,
    mlff_spec_factory,
) -> None:
    bundle = MLFFTrainBundle(
        training_unit=finetune_training_unit,
        mlff_spec=mlff_spec_factory("mace"),
    )
    assert set(type(bundle).model_fields) == {
        "training_unit",
        "mlff_spec",
        "mlff_train_bundle_id",
    }
    assert bundle.unit_type == "finetune"
    assert "file_mode" not in inspect.signature(
        bundle.write_submit_folder
    ).parameters
    assert not hasattr(bundle, "materialize_submit_folder")


def test_cartesian_product_preserves_input_order(
    zeroshot_training_unit,
    mlff_spec_factory,
) -> None:
    specs = [
        mlff_spec_factory("mace", with_training=False),
        mlff_spec_factory("sevennet", with_training=False),
    ]
    bundles = build_mlff_train_bundles(
        training_units=[zeroshot_training_unit],
        mlff_specs=specs,
    )
    assert [bundle.mlff_spec.mlff_type for bundle in bundles] == [
        "mace",
        "sevennet",
    ]


def test_bundle_identity_and_monty_round_trip(
    tmp_path: Path,
    finetune_training_unit,
    mlff_spec_factory,
) -> None:
    spec = mlff_spec_factory("mace")
    first = MLFFTrainBundle(
        training_unit=finetune_training_unit,
        mlff_spec=spec,
    )
    repeated = MLFFTrainBundle(
        training_unit=finetune_training_unit.model_copy(),
        mlff_spec=spec.model_copy(),
    )
    assert first.mlff_train_bundle_id == repeated.mlff_train_bundle_id

    path = tmp_path / "bundle.json"
    dumpfn(first, path, indent=2)
    assert loadfn(path) == first


def test_finetune_requires_training_parameters(
    finetune_training_unit,
    mlff_spec_factory,
) -> None:
    with pytest.raises(ValueError, match="non-None"):
        MLFFTrainBundle(
            training_unit=finetune_training_unit,
            mlff_spec=mlff_spec_factory("mace", with_training=False),
        )


@pytest.mark.parametrize("family", FAMILIES)
def test_zeroshot_writes_fixed_copied_inputs(
    family: str,
    tmp_path: Path,
    zeroshot_training_unit,
    mlff_spec_factory,
) -> None:
    bundle = MLFFTrainBundle(
        training_unit=zeroshot_training_unit,
        mlff_spec=mlff_spec_factory(family, with_training=False),
    )
    target = tmp_path / f"zero-{family}"
    result = bundle.write_submit_folder(target)
    assert result == target.absolute()
    assert (target / "run.sh").is_file()
    assert (target / "test_config.json").is_file()
    assert (target / "runtime" / "run_test.py").is_file()
    assert (target / "runtime" / "calculator.py").is_file()
    assert not (target / "bundle_manifest.json").exists()
    assert not (target / "datasets" / "train.extxyz").exists()
    for filename in MODEL_FILES[family]:
        copied = target / "models" / filename
        assert copied.is_file()
        assert not copied.is_symlink()

    config = json.loads((target / "test_config.json").read_text())
    assert config["schema_version"] == 2
    assert config["calculator"]["parameters"] == {}
    assert config["test_datasets"][0]["properties"] == [
        "energy",
        "forces",
        "stress",
    ]
    assert "device" not in json.dumps(config)
    assert "training" not in (target / "run.sh").read_text().lower()
    if family == "nep89":
        assert {item["name"] for item in config["package_requirements"]} == {
            "calorine"
        }


@pytest.mark.parametrize("family", FAMILIES)
def test_finetune_writes_native_training_and_automatic_stress(
    family: str,
    tmp_path: Path,
    finetune_training_unit,
    mlff_spec_factory,
) -> None:
    bundle = MLFFTrainBundle(
        training_unit=finetune_training_unit,
        mlff_spec=mlff_spec_factory(family),
    )
    target = bundle.write_submit_folder(tmp_path / f"fine-{family}")
    run_script = (target / "run.sh").read_text()
    assert TRAINING_MARKERS[family] in run_script
    assert (target / "datasets" / "train.extxyz").is_file()
    assert (target / "datasets" / "validation.extxyz").is_file()
    if family == "nep89":
        test_config = json.loads((target / "test_config.json").read_text())
        assert {
            item["name"] for item in test_config["package_requirements"]
        } == {"torchnep", "calorine"}

    if family in {"dpa4", "dpa4c"}:
        config = json.loads((target / "training" / "input.json").read_text())
        assert config["loss"]["start_pref_v"] == 0.1
        assert config["model"] == {
            "type_map": ["H", "He"],
            "descriptor": {},
            "fitting_net": {},
        }
        assert config["training"]["numb_epoch"] == 100
        assert "--use-pretrain-script" in run_script
        assert "--output outputs/train_adapted.json" in run_script
        assert "model.ckpt.pt" in run_script
    elif family == "mattersim":
        assert "--include_stresses" in run_script
        assert "--batch_size 1" in run_script
        assert "--epochs 100" in run_script
        assert "--early_stop_patience 101" in run_script
    elif family == "mace":
        config = YAML(typ="safe").load(
            (target / "training" / "mace.yaml").read_text()
        )
        assert config["stress_weight"] == 1.0
        assert config["max_num_epochs"] == 100
        assert config["patience"] == 101
    elif family == "sevennet":
        config = YAML(typ="safe").load(
            (target / "training" / "sevennet.yaml").read_text()
        )
        assert config["train"]["is_train_stress"] is True
        assert config["train"]["epoch"] == 100
    else:
        config = (target / "training" / "torchnep" / "nep.in").read_text()
        assert "type 2 H He" in config
        assert "zbl 2" in config
        assert "l_max 4 2 1" in config
        assert "epoch 100" in config
        assert "early_stop 0" in config
        assert "lambda_v 0.01" in config
        assert "--model models/nep89.txt" in run_script
        assert "output/nep_best.txt artifacts/nep89.txt" in run_script


def test_no_stress_is_omitted_from_training_and_each_test(
    tmp_path: Path,
    finetune_training_unit,
    mlff_spec_factory,
) -> None:
    domain = (
        finetune_training_unit.root_path / finetune_training_unit.domain
    )
    for filename in (
        "train.extxyz",
        "validation.extxyz",
        "test_own.extxyz",
        "test_cross.extxyz",
    ):
        frame = make_frame("H2", -1.0, filename, stress=False)
        frame.set_cell([10, 10, 10])
        frame.set_pbc(True)
        write(domain / filename, frame, format="extxyz")

    target = MLFFTrainBundle(
        training_unit=finetune_training_unit,
        mlff_spec=mlff_spec_factory("mace"),
    ).write_submit_folder(tmp_path / "no-stress")
    training = YAML(typ="safe").load(
        (target / "training" / "mace.yaml").read_text()
    )
    testing = json.loads((target / "test_config.json").read_text())
    assert training["stress_weight"] == 0.0
    assert all(
        dataset["properties"] == ["energy", "forces"]
        for dataset in testing["test_datasets"]
    )


def test_deepmd_uses_sidecar_only_as_a_nonarchitecture_template(
    tmp_path: Path,
    finetune_training_unit,
) -> None:
    model = tmp_path / "dpa4.pt"
    model.write_bytes(b"checkpoint")
    sidecar = tmp_path / "dpa4.json"
    sidecar.write_text(
        json.dumps(
            {
                "model": {
                    "type_map": ["H", "He"],
                    "descriptor": {"type": "hardcoded"},
                    "fitting_net": {"neuron": [1]},
                },
                "training": {
                    "numb_steps": 999,
                    "num_epochs": 999,
                    "disp_freq": 12,
                },
                "learning_rate": {"type": "exp", "start_lr": 1e-4},
                "loss": {},
            }
        ),
        encoding="utf-8",
    )
    spec = DPA4SpecBuilder(
        pretrained_model_path=model,
        pretrained_config_path=sidecar,
        training_parameters={"numb_epoch": 7},
    ).build()

    target = MLFFTrainBundle(
        training_unit=finetune_training_unit,
        mlff_spec=spec,
    ).write_submit_folder(tmp_path / "deepmd-template")
    config = json.loads((target / "training" / "input.json").read_text())
    run_script = (target / "run.sh").read_text()

    assert config["model"] == {
        "type_map": ["H", "He"],
        "descriptor": {},
        "fitting_net": {},
    }
    assert config["training"]["numb_epoch"] == 7
    assert config["training"]["disp_freq"] == 100
    assert not {
        "numb_steps",
        "stop_batch",
        "num_step",
        "num_steps",
        "numb_step",
        "num_epochs",
        "num_epoch",
        "numb_epochs",
    } & config["training"].keys()
    assert config["learning_rate"]["start_lr"] == 1e-4
    assert "--finetune models/dpa4.pt --use-pretrain-script" in run_script
    assert "--output outputs/train_adapted.json" in run_script
    assert "-c training/deepmd/model.ckpt.pt" in run_script


def test_deepmd_requires_type_map_in_sidecar(
    tmp_path: Path,
    finetune_training_unit,
) -> None:
    model = tmp_path / "dpa4.pt"
    model.write_bytes(b"checkpoint")
    sidecar = tmp_path / "dpa4.json"
    sidecar.write_text(json.dumps({"model": {}, "training": {}}))
    spec = DPA4SpecBuilder(
        pretrained_model_path=model,
        pretrained_config_path=sidecar,
        training_parameters={},
    ).build()

    with pytest.raises(ValueError, match="model.type_map"):
        MLFFTrainBundle(
            training_unit=finetune_training_unit,
            mlff_spec=spec,
        ).write_submit_folder(tmp_path / "missing-type-map")


def test_nep_without_stress_disables_virial_losses(
    tmp_path: Path,
    finetune_training_unit,
    mlff_spec_factory,
) -> None:
    domain = finetune_training_unit.root_path / finetune_training_unit.domain
    for filename in ("train.extxyz", "validation.extxyz"):
        frame = make_frame("H", -1.0, filename, stress=False)
        frame.set_cell([10, 10, 10])
        frame.set_pbc(True)
        write(domain / filename, frame, format="extxyz")

    target = MLFFTrainBundle(
        training_unit=finetune_training_unit,
        mlff_spec=mlff_spec_factory("nep89"),
    ).write_submit_folder(tmp_path / "nep-no-stress")
    config = (target / "training" / "torchnep" / "nep.in").read_text()
    assert "lambda_v 0.0" in config
    assert "stage2_lambda_v 0.0" in config


def test_nep_rejects_unsupported_checkpoint_headers(
    tmp_path: Path,
    finetune_training_unit,
) -> None:
    model = tmp_path / "nep.txt"
    model.write_text("nep3 1 H\n", encoding="utf-8")
    spec = NEP89SpecBuilder(
        pretrained_model_path=model,
        training_parameters={},
    ).build()

    with pytest.raises(ValueError, match="nep4"):
        MLFFTrainBundle(
            training_unit=finetune_training_unit,
            mlff_spec=spec,
        ).write_submit_folder(tmp_path / "bad-nep")


@pytest.mark.parametrize(
    ("header", "message"),
    (
        (
            "nep4 2 H\ncutoff 6 5\nn_max 4 4\nbasis_size 8 8\n"
            "l_max 4 2 1\nANN 80 0\n",
            "element count",
        ),
        (
            "nep4_zbl 1 H\nzbl 1 3\ncutoff 6 5\nn_max 4 4\n"
            "basis_size 8 8\nl_max 4 2 1\nANN 80 0\n",
            "half the outer cutoff",
        ),
    ),
)
def test_nep_header_validation(
    tmp_path: Path,
    header: str,
    message: str,
) -> None:
    model = tmp_path / "invalid-nep.txt"
    model.write_text(header, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        _nep_architecture(model)


def test_nep_header_derives_typewise_zbl_and_full_type_order(
    tmp_path: Path,
) -> None:
    model = tmp_path / "nep.txt"
    model.write_text(
        "nep4_zbl 3 Cr Co Ni\n"
        "zbl 1.25 2.5 0.7\n"
        "cutoff 6 4 134 42\n"
        "n_max 8 8\n"
        "basis_size 12 12\n"
        "l_max 4 2 1\n"
        "ANN 80 0\n",
        encoding="utf-8",
    )

    lines, symbols = _nep_architecture(model)

    assert lines == [
        "version 4",
        "type 3 Cr Co Ni",
        "zbl 2.5",
        "use_typewise_cutoff_zbl 0.7",
        "cutoff 6 4",
        "n_max 8 8",
        "basis_size 12 12",
        "l_max 4 2 1",
        "neuron 80",
    ]
    assert symbols == {"Cr", "Co", "Ni"}


def test_nep_rejects_training_elements_absent_from_checkpoint(
    tmp_path: Path,
    finetune_training_unit,
) -> None:
    model = tmp_path / "nep.txt"
    model.write_text(
        "nep4 1 He\n"
        "cutoff 6 5\n"
        "n_max 4 4\n"
        "basis_size 8 8\n"
        "l_max 4 2 1\n"
        "ANN 80 0\n",
        encoding="utf-8",
    )
    spec = NEP89SpecBuilder(
        pretrained_model_path=model,
        training_parameters={},
    ).build()

    with pytest.raises(ValueError, match="absent.*'H'"):
        MLFFTrainBundle(
            training_unit=finetune_training_unit,
            mlff_spec=spec,
        ).write_submit_folder(tmp_path / "missing-nep-element")


def test_nep_legacy_restart_artifact_has_migration_error(
    tmp_path: Path,
    finetune_training_unit,
    mlff_spec_factory,
) -> None:
    spec = mlff_spec_factory("nep89")
    restart = tmp_path / "nep.restart"
    restart.write_bytes(b"legacy")
    spec.pretrained_model.artifacts["restart"] = LocalArtifactRef.from_path(
        restart
    )

    with pytest.raises(ValueError, match="rebuild this MLFF specification"):
        MLFFTrainBundle(
            training_unit=finetune_training_unit,
            mlff_spec=spec,
        ).write_submit_folder(tmp_path / "legacy-nep")


def test_mixed_stress_within_dataset_is_rejected(
    tmp_path: Path,
    zeroshot_training_unit,
    mlff_spec_factory,
) -> None:
    source = (
        zeroshot_training_unit.root_path
        / zeroshot_training_unit.domain
        / zeroshot_training_unit.test_sets[0]
    )
    write(
        source,
        [
            make_frame("H", -1, "with", stress=True),
            make_frame("H", -1, "without", stress=False),
        ],
        format="extxyz",
    )
    target = tmp_path / "mixed"
    bundle = MLFFTrainBundle(
        training_unit=zeroshot_training_unit,
        mlff_spec=mlff_spec_factory("mace", with_training=False),
    )
    with pytest.raises(ValueError, match="mixes frames"):
        bundle.write_submit_folder(target)
    assert not target.exists()


def test_train_validation_stress_disagreement_is_rejected(
    tmp_path: Path,
    finetune_training_unit,
    mlff_spec_factory,
) -> None:
    validation = (
        finetune_training_unit.root_path
        / finetune_training_unit.domain
        / finetune_training_unit.val_set
    )
    frame = make_frame("H", -1, "validation", stress=False)
    frame.set_cell([10, 10, 10])
    frame.set_pbc(True)
    write(validation, frame, format="extxyz")
    with pytest.raises(ValueError, match="disagree"):
        MLFFTrainBundle(
            training_unit=finetune_training_unit,
            mlff_spec=mlff_spec_factory("mace"),
        ).write_submit_folder(tmp_path / "mismatch")


def test_artifact_hash_is_checked_again_before_copy(
    tmp_path: Path,
    zeroshot_training_unit,
    mlff_spec_factory,
) -> None:
    spec = mlff_spec_factory("mace", with_training=False)
    spec.pretrained_model.artifacts["model"].path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed after"):
        MLFFTrainBundle(
            training_unit=zeroshot_training_unit,
            mlff_spec=spec,
        ).write_submit_folder(tmp_path / "changed")


def test_writer_uses_configured_submit_directories(
    monkeypatch,
    tmp_path: Path,
    zeroshot_training_unit,
    mlff_spec_factory,
) -> None:
    monkeypatch.setattr(
        writer_base, "DEFAULT_MLFF_DATASETS_DIR", "inputs/datasets"
    )
    monkeypatch.setattr(
        writer_base, "DEFAULT_MLFF_MODELS_DIR", "inputs/models"
    )
    monkeypatch.setattr(writer_base, "DEFAULT_MLFF_RUNTIME_DIR", "engine")
    monkeypatch.setattr(writer_base, "DEFAULT_MLFF_OUTPUTS_DIR", "results")
    monkeypatch.setattr(
        writer_base, "DEFAULT_MLFF_ARTIFACTS_DIR", "results/models"
    )
    target = MLFFTrainBundle(
        training_unit=zeroshot_training_unit,
        mlff_spec=mlff_spec_factory("mace", with_training=False),
    ).write_submit_folder(tmp_path / "custom-layout")

    assert (target / "inputs" / "datasets" / "test_000.extxyz").is_file()
    assert (target / "inputs" / "models" / "mace.model").is_file()
    assert (target / "engine" / "calculator.py").is_file()
    config = json.loads((target / "test_config.json").read_text())
    assert config["summary_output"] == "results/test_summary.json"


def test_temporary_submit_folder_is_caller_owned(
    zeroshot_training_unit,
    mlff_spec_factory,
) -> None:
    target = MLFFTrainBundle(
        training_unit=zeroshot_training_unit,
        mlff_spec=mlff_spec_factory("mace", with_training=False),
    ).write_submit_folder()
    try:
        assert target.is_dir()
    finally:
        shutil.rmtree(target)


def test_base_writer_has_no_copy_strategy_surface() -> None:
    assert "file_mode" not in inspect.signature(
        BaseMLFFBundleWriter.write_submit_folder
    ).parameters
