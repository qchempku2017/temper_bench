"""Regression coverage for builder policy, shared data and standalone results."""
from copy import deepcopy
import importlib
import inspect
import json
import sys

import pytest
from ruamel.yaml import YAML

from temper.mlff import MACESpecBuilder, MLFFTrainBundle, mlff_spec_builder_factory
from temper.mlff.bundle_writers import write_submit_folder
from temper.mlff.pipeline.training_adapters import mlff_adapter_factory
from temper.mlff.bundle_writers.base import MLFFBundleWriter
from temper.mlff.pipeline.training_adapters.base import BaseMLFFAdapter
from temper.mlff.spec_builders.base import BaseSpecBuilder
from temper.schemas import MLFFTestResult


def test_registered_release_needs_only_class_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(BaseSpecBuilder, "registry", dict(BaseSpecBuilder.registry))

    @BaseSpecBuilder.register(name="other-mace", alias="other")
    class OtherMACE(MACESpecBuilder):
        mlff_type = "other-mace"
        model_filename = "other.model"
        model_version = "2"

    (tmp_path / "other.model").write_bytes(b"weights")
    builder = mlff_spec_builder_factory("other-mace")(pretrained_model_dir=tmp_path)
    assert mlff_spec_builder_factory("other") is OtherMACE
    spec = builder.build()
    assert spec.mlff_type == "other-mace"
    assert spec.pretrained_model.version == "2"
    assert spec.pretrained_model.artifacts["model"].path == tmp_path / "other.model"


def test_writer_registry_and_abstract_contract():
    assert inspect.isabstract(BaseSpecBuilder)
    assert inspect.isabstract(BaseMLFFAdapter)
    for key in ("dpa4", "dpa4c", "mace", "mattersim", "sevennet", "nep89"):
        assert mlff_spec_builder_factory(key).mlff_type == key
        writer = mlff_adapter_factory(key)
        assert writer.mlff_type == key
        assert not inspect.isabstract(writer)


def test_writer_registration_uses_explicit_name_and_alias(monkeypatch):
    monkeypatch.setattr(BaseMLFFAdapter, "registry", {})
    writer = type("Writer", (), {"mlff_type": "different"})
    BaseMLFFAdapter.register(name="registered", alias="short")(writer)
    assert mlff_adapter_factory("registered") is writer
    assert mlff_adapter_factory("short") is writer
    assert "different" not in BaseMLFFAdapter.registry


def test_sevennet_resolves_architecture_from_checkpoint(tmp_path, monkeypatch):
    from types import ModuleType, SimpleNamespace
    from temper.mlff.pipeline.data_preparation.sevennet import prepare_config

    architectures = [
        {"chemical_species": "auto", "channel": 32, "lmax": 1},
        {"chemical_species": "auto", "channel": 64, "lmax": 3},
    ]
    native = ModuleType("sevenn.util")
    calls = []

    def load_checkpoint(path):
        calls.append(path)
        def yaml_dict(mode):
            assert mode == "continue"
            return {"model": dict(architectures[len(calls) - 1])}
        return SimpleNamespace(yaml_dict=yaml_dict)

    native.load_checkpoint = load_checkpoint
    monkeypatch.setitem(sys.modules, "sevenn", ModuleType("sevenn"))
    monkeypatch.setitem(sys.modules, "sevenn.util", native)
    config = {
        "model": {"train_shift_scale": False, "train_denominator": False},
        "train": {"continue": {"checkpoint": "models/sevennet.pth"}, "epoch": 7},
        "data": {"load_trainset_path": ["datasets/train.extxyz"]},
    }
    source = tmp_path / "input.yaml"
    output = tmp_path / "resolved.yaml"
    YAML().dump(config, source)
    for architecture in architectures:
        prepare_config(source, output, bundle_root=tmp_path)
        assert calls[-1] == str(tmp_path / "models/sevennet.pth")
        resolved = YAML(typ="safe").load(output)
        assert resolved["model"] == {**architecture, **config["model"]}
        assert resolved["train"] == config["train"]
        assert resolved["data"] == config["data"]


def test_parameter_fields_and_independent_defaults(mlff_spec_factory):
    first = mlff_spec_factory("dpa4")
    second = mlff_spec_factory("dpa4c")
    assert {"training_parameters", "testing_parameters"} <= first.model_dump().keys()
    assert not {"training", "testing"} & first.model_dump().keys()
    assert first._IDENTITY_SCHEMA == "temper.mlff-spec.v3"
    assert first.training_parameters["numb_epoch"] == 60
    assert first.training_parameters["training_data"] == {"batch_size": "auto:128"}
    assert first.training_parameters["validation_data"] == {"batch_size": "auto:128"}
    assert second.training_parameters["numb_epoch"] == 100
    first.training_parameters["loss"]["start_pref_e"] = 99
    assert second.training_parameters["loss"]["start_pref_e"] == 20
    assert mlff_spec_factory("dpa4").training_parameters["loss"]["start_pref_e"] == 20


@pytest.mark.parametrize(
    ("family", "overrides"),
    [
        ("dpa4", {"learning_rate": {"type": "exp", "start_lr": 2e-5, "stop_lr": 2e-7}}),
        ("dpa4c", {"gradient_max_norm": 0.5}),
        ("mace", {"energy_weight": 37.0, "stress_weight": 0.8, "batch_size": 8}),
        ("mattersim", {"batch_size": 4, "stress_loss_ratio": 0.7}),
        ("sevennet", {"force_loss_weight": 9.0, "stress_loss_weight": 0.6}),
        ("nep89", {"lambda_v": 0.4, "stage2_lambda_v": 0.3}),
    ],
)
def test_writers_preserve_recipe_overrides(
    family, overrides, tmp_path, finetune_training_unit, mlff_spec_factory,
):
    spec = mlff_spec_factory(family, training_parameters=overrides)
    before = deepcopy(spec.model_dump(mode="json"))
    bundle = MLFFTrainBundle(training_unit=finetune_training_unit, mlff_spec=spec)
    target = write_prepared(write_submit_folder(bundle, tmp_path / family))
    script = training_script(target)
    assert "set -euo pipefail" in script
    assert "MLFF_DEVICE" not in script
    assert spec.model_dump(mode="json") == before
    if family in {"dpa4", "dpa4c"}:
        config = json.loads((target / "training/input.json").read_text())
        flattened = {**config["training"], "learning_rate": config["learning_rate"]}
        assert all(flattened[key] == value for key, value in overrides.items())
        assert config["training"]["optimizer"] == {
            "type": "HybridMuon", "weight_decay": 0.001,
        }
        assert config["training"]["training_data"]["batch_size"] == "auto:128"
        assert config["training"]["validation_data"]["batch_size"] == "auto:128"
        assert config["loss"]["loss_func"] == "mae"
        assert config["loss"]["f_use_norm"] is True
        assert config["training"]["save_ckpt"] == "training/deepmd/model.ckpt"
        assert "cp training/deepmd/model.ckpt.pt artifacts/model.ckpt.pt" in script
    elif family == "mattersim":
        assert all(f"--{key} {value}" in script for key, value in overrides.items())
    elif family == "nep89":
        config = (target / "training/torchnep/nep.in").read_text()
        assert all(f"{key} {value}" in config for key, value in overrides.items())
        assert not (target / "training/run.sh").exists()
        assert not (target / "runtime/prepare_nep89.py").exists()
    else:
        config = YAML(typ="safe").load((target / f"training/{family}.yaml").read_text())
        if family == "sevennet":
            config = config["train"]
            assert config["continue"] == {
                "checkpoint": "models/sevennet.pth",
                "reset_optimizer": True, "reset_scheduler": True, "reset_epoch": True,
            }
            assert config["device"] == "cuda"
        assert all(config[key] == value for key, value in overrides.items())


def test_stress_inspection_is_shared_and_invalidated(
    tmp_path, finetune_training_unit, mlff_spec_factory, monkeypatch,
):
    module = importlib.import_module("temper.schemas.train_unit")
    original = module.iter_extxyz_metadata
    reads = []

    def counted(path, **kwargs):
        reads.append(path)
        return original(path, **kwargs)

    monkeypatch.setattr(module, "iter_extxyz_metadata", counted)
    unit = finetune_training_unit
    for family in ("mace", "mattersim"):
        write_prepared(MLFFTrainBundle(training_unit=unit, mlff_spec=mlff_spec_factory(family)).write_submit_folder(
            tmp_path / family
        ))
    assert len(reads) == 4
    assert unit.training_has_stress is True
    assert unit.test_has_stress == [True, True]
    assert "dataset_stress" not in unit.model_dump()
    unit.test_sets = tuple(reversed(unit.test_sets))
    assert "dataset_stress" not in unit.__dict__
    assert unit.test_has_stress == [True, True]
    assert len(reads) == 8


def test_written_bundle_uses_installed_runtime_and_result_schema(
    tmp_path, zeroshot_training_unit, mlff_spec_factory,
):
    target = write_prepared(MLFFTrainBundle(
        training_unit=zeroshot_training_unit,
        mlff_spec=mlff_spec_factory("mace", with_training=False),
    ).write_submit_folder(tmp_path / "standalone"))
    assert not (target / "runtime").exists()
    assert "temper_bench run_pipeline bundle.json" in (target / "run.sh").read_text()
    result = MLFFTestResult(
        schema_version=1, calculator_identifier="mace", model="models/mace.model",
        package_versions={"mace-torch": None}, dataset_id="test_000",
        source_domain="example", source_filename="test.extxyz",
        submit_path="datasets/test_000.extxyz", number_of_frames=2, total_atoms=3,
        requested_properties=["energy", "forces"],
        units={"energy": "eV", "forces": "eV/Angstrom", "stress": None},
        wall_time_seconds=0.1,
    )
    assert MLFFTestResult.from_dict(json.loads(json.dumps(result.as_dict()))) == result


def write_prepared(target):
    """Exercise backend generation on the runtime side of the bundle boundary."""
    from temper.mlff.pipeline.runner import prepare_bundle
    prepare_bundle(target / "bundle.json")
    return target


def training_script(target):
    path = target / "training/run.sh"
    return (path if path.exists() else target / "run.sh").read_text()
