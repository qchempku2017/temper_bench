"""Behavioral coverage for packaging and adapter review changes."""
import json
import logging
from types import ModuleType
import sys

import pytest
from pydantic import ValidationError

from temper.schemas.mlff_train_bundle import MLFFTrainBundle
from temper.mlff.bundle_writers import MLFFBundleWriter, write_submit_folder
from temper.mlff.pipeline import evaluate_mlff, runner
from temper.mlff.pipeline.training_adapters import BaseMLFFAdapter, mlff_adapter_factory
from temper.mlff.spec_builders.base import BaseSpecBuilder
from temper.schemas.mlff_train_bundle import MLFFBundleFiles


@pytest.mark.parametrize("entrypoint", ["bundle", "writer", "function"])
def test_packaging_returns_independent_serialized_record(
    entrypoint, tmp_path, finetune_training_unit, mlff_spec_factory,
):
    bundle = MLFFTrainBundle(
        training_unit=finetune_training_unit, mlff_spec=mlff_spec_factory("mace"),
    )
    before = bundle.model_dump()
    if entrypoint == "bundle":
        target, packaged = bundle.write_submit_folder(tmp_path / "job")
    elif entrypoint == "writer":
        target, packaged = MLFFBundleWriter(bundle).write_submit_folder(tmp_path / "job")
    else:
        target, packaged = write_submit_folder(bundle, tmp_path / "job")
    assert packaged == MLFFTrainBundle.model_validate_json((target / "bundle.json").read_text())
    assert packaged.mlff_train_bundle_id == bundle.mlff_train_bundle_id
    assert bundle.model_dump() == before
    packaged.mlff_spec.training_parameters["seed"] = 17
    assert bundle.mlff_spec.training_parameters["seed"] == 42


@pytest.mark.parametrize("base", [BaseMLFFAdapter, BaseSpecBuilder])
@pytest.mark.parametrize("aliases", [None, "short", ["short", "another"], []])
def test_registration_accepts_multiple_aliases(base, aliases, monkeypatch):
    monkeypatch.setattr(base, "registry", {})
    registered = type("Registered", (), {})
    assert base.register("canonical", alias=aliases)(registered) is registered
    keys = [aliases] if isinstance(aliases, str) else aliases or []
    assert base.registry == dict.fromkeys(["canonical", *keys], registered)


@pytest.mark.parametrize("family", ["dpa4", "dpa4c", "mace", "mattersim", "sevennet", "nep89"])
@pytest.mark.parametrize("finetune", [False, True])
def test_manifest_matches_results_and_native_inputs(
    family, finetune, tmp_path, finetune_training_unit, zeroshot_training_unit,
    mlff_spec_factory,
):
    bundle = MLFFTrainBundle(
        training_unit=finetune_training_unit if finetune else zeroshot_training_unit,
        mlff_spec=mlff_spec_factory(family, with_training=finetune),
    )
    target, packaged = bundle.write_submit_folder(tmp_path / "job")
    files = packaged.files
    adapter = runner.prepare_bundle(target / "bundle.json")
    config = packaged.test_config()
    assert config["summary_output"] == files.test_output_files["summary"]
    for dataset in config["test_datasets"]:
        for suffix, key in (("predictions", "output"), ("metadata", "metadata_output")):
            assert dataset[key] == files.test_output_files[dataset["id"] + "_" + suffix]
    if finetune:
        assert files.training_output_files["training_log"] == "training_outputs/training.log"
        for key in ("config", "script"):
            if key in files.training_input_files:
                assert (target / files.training_input_files[key]).is_file()
        if family.startswith("dpa"):
            assert adapter.trained_model_path == "training_outputs/model.ckpt.pt"
            assert config["model"] == "training_outputs/evaluation.pt2"
            assert files.evaluation_model not in files.training_output_files.values()
            native = json.loads((target / "training_inputs/input.json").read_text())
            assert native["training"]["disp_file"] == "training_outputs/lcurve.out"
    else:
        assert files.training_input_files == {}
        assert files.training_output_files == {}
        assert files.evaluation_model is None
        assert config["model"] == files.pretrained_model
        assert not (target / "training_inputs").exists()
        assert not (target / "training_outputs").exists()


@pytest.mark.parametrize("field", ["training_input_files", "training_output_files", "test_input_files", "test_output_files", "pretrained_model", "evaluation_model"])
@pytest.mark.parametrize("path", ["../escape", "/absolute", "C:/outside", "a\\b"])
def test_manifest_rejects_nonportable_paths(field, path):
    payload = dict(pretrained_model="pretrained_models/model", test_input_files={}, test_output_files={}, training_stress=None, test_stress=[])
    payload[field] = path if field in {"pretrained_model", "evaluation_model"} else {"result": path}
    with pytest.raises(ValidationError):
        MLFFBundleFiles(**payload)


@pytest.mark.parametrize("family", ["dpa4", "dpa4c"])
@pytest.mark.parametrize("failure", [None, "freeze", "evaluate"])
def test_deepmd_freezes_for_evaluation_and_cleans_up(
    family, failure, tmp_path, finetune_training_unit, mlff_spec_factory, monkeypatch,
):
    from temper.mlff.pipeline.training_adapters import deepmd

    target, packaged = MLFFTrainBundle(
        training_unit=finetune_training_unit, mlff_spec=mlff_spec_factory(family),
    ).write_submit_folder(tmp_path / "job")
    events = []
    adapter_type = mlff_adapter_factory(family)
    monkeypatch.setattr(runner, "require_cuda", lambda: None)
    monkeypatch.setattr(adapter_type, "prepare_training", lambda self: None)
    temporary = target / "training_outputs/evaluation.pt2"
    retained = target / "training_outputs/model.ckpt.pt"

    def train(self):
        events.append("train")
        retained.write_bytes(b"portable")

    def freeze(args, **kwargs):
        events.append("freeze")
        assert args[:3] == ["dp", "--pt" if family == "dpa4" else "--pt-expt", "freeze"]
        assert args[3:7] == ["-c", "training_outputs/model.ckpt.pt", "-o", "training_outputs/evaluation"]
        assert ("--lower-kind" in args) == (family == "dpa4c")
        assert kwargs["cwd"] == target and kwargs["check"]
        temporary.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_bytes(b"host-specific")
        if failure == "freeze":
            raise RuntimeError("freeze failed")

    def evaluate(path, *, bundle_root):
        assert bundle_root == target
        events.append("evaluate")
        assert (target / json.loads(path.read_text())["model"]) == temporary
        assert temporary.is_file()
        if failure == "evaluate":
            raise RuntimeError("evaluate failed")

    monkeypatch.setattr(adapter_type, "train", train)
    monkeypatch.setattr(deepmd.subprocess, "run", freeze)
    monkeypatch.setattr(runner, "run_test", evaluate)
    if failure:
        with pytest.raises(RuntimeError, match=failure):
            runner.run_pipeline(target / "bundle.json")
    else:
        runner.run_pipeline(target / "bundle.json")
    assert events == (["train", "freeze"] if failure == "freeze" else ["train", "freeze", "evaluate"])
    assert retained.read_bytes() == b"portable"
    assert not temporary.exists()


def test_packaging_and_preparation_log_levels(tmp_path, finetune_training_unit, mlff_spec_factory, caplog):
    with caplog.at_level(logging.DEBUG, logger="temper"):
        target, _ = MLFFTrainBundle(
            training_unit=finetune_training_unit, mlff_spec=mlff_spec_factory("mace"),
        ).write_submit_folder(tmp_path / "job")
        runner.prepare_bundle(target / "bundle.json")
    assert any(r.levelno == logging.INFO and "Packaging" in r.message for r in caplog.records)
    assert any(r.levelno == logging.DEBUG and "Wrote training input" in r.message for r in caplog.records)
    bundle_id = str(MLFFTrainBundle.model_validate_json((target / "bundle.json").read_text()).mlff_train_bundle_id)
    assert all(bundle_id in r.message for r in caplog.records if "Packaging" in r.message or "Wrote bundle" in r.message)


@pytest.mark.parametrize("family,key", [("dpa4", "seed"), ("dpa4c", "seed"), ("mace", "seed"), ("mattersim", "seed"), ("sevennet", "random_seed"), ("nep89", "run_seed")])
def test_default_seeds_are_42(family, key, mlff_spec_factory):
    spec = mlff_spec_factory(family)
    assert spec.training_parameters[key] == 42
    if family == "mace":
        assert "valid_fraction" not in spec.training_parameters
