"""Verify that recorded paths control execution after packaging and
relocation."""

import json
from pathlib import Path
import shlex

import pytest
from pydantic import ValidationError
from ruamel.yaml import YAML

from temper.schemas.mlff_train_bundle import MLFFTrainBundle, MLFFBundleFiles
from temper.mlff.pipeline import runner, evaluate_mlff
from temper.mlff.pipeline.runner import create_result_manifest, make_test_manifests
from test_mlff_runtime import FakeCalculator


@pytest.mark.parametrize(
    "family", ["dpa4", "dpa4c", "mace", "sevennet", "mattersim", "nep89"]
)
def test_adapters_use_recorded_paths_instead_of_rebuilding_defaults(
    family,
    tmp_path,
    finetune_training_unit,
    mlff_spec_factory,
    monkeypatch,
):
    target, bundle = MLFFTrainBundle(
        training_unit=finetune_training_unit,
        mlff_spec=mlff_spec_factory(family),
    ).write_submit_folder(tmp_path / "original")
    files = bundle.files

    def move_input(old, new):
        source = target / old
        if source.is_file():
            (target / new).parent.mkdir(parents=True, exist_ok=True)
            source.rename(target / new)

    move_input(files.pretrained_model, "custom weights/pretrained.model")
    files.pretrained_model = "custom weights/pretrained.model"
    for field in (
        "training_input_files",
        "training_output_files",
        "test_input_files",
        "test_output_files",
    ):
        mapping = getattr(files, field)
        for key, old in list(mapping.items()):
            # Every entry has its own parent, including directory
            # entries.
            new = f"custom {field}/{key}/{Path(old).name}"
            if field.endswith("input_files"):
                move_input(old, new)
            mapping[key] = new
    if files.evaluation_model:
        files.evaluation_model = "custom evaluation/model.pt2"
    bundle.to_json_file((target / "bundle.json"))
    remote = tmp_path / "remote"
    target.rename(remote)
    monkeypatch.chdir(tmp_path)
    adapter = runner.prepare_bundle(remote / "bundle.json")
    inputs, outputs = files.training_input_files, files.training_output_files
    tests = make_test_manifests(adapter.bundle, bundle_root=remote)
    assert tests == make_test_manifests(bundle, bundle_root=remote)
    assert all(
        (remote / test.execution_input_path).is_file() for test in tests
    )
    if family != "nep89":
        script = (remote / inputs["script"]).read_text()
        assert f"> {shlex.quote(outputs['training_log'])} 2>&1" in script
        assert "tee " not in script
        assert "set -euo pipefail" in script
    if family.startswith("dpa"):
        config = json.loads((remote / inputs["config"]).read_text())
        assert config["training"]["save_ckpt"] == outputs[
            "trained_model"
        ].removesuffix(".pt")
        assert config["training"]["disp_file"] == outputs["learning_curve"]
        assert config["training"]["training_data"]["systems"] == [
            inputs["converted_train"]
        ]
        assert config["training"]["validation_data"]["systems"] == [
            inputs["converted_validation"]
        ]
        assert shlex.quote(outputs["adapted_config"]) in script
        assert shlex.quote(files.pretrained_model) in script
        from temper.mlff.pipeline.data_preparation import deepmd

        calls = []
        monkeypatch.setattr(
            deepmd,
            "prepare_dataset",
            lambda source, output: calls.append((source, output)),
        )
        adapter.prepare_training()
        assert calls == [
            (remote / inputs[key], remote / inputs["converted_" + key])
            for key in ("train", "validation")
        ]
    elif family == "mace":
        config = YAML(typ="safe").load(remote / inputs["config"])
        assert config["foundation_model"] == files.pretrained_model
        assert config["train_file"] == inputs["train"]
        assert config["valid_file"] == inputs["validation"]
        for option, key in (
            ("model_dir", "native_models"),
            ("checkpoints_dir", "checkpoints"),
            ("results_dir", "metrics"),
            ("log_dir", "logs"),
        ):
            assert config[option] == outputs[key]
        assert shlex.quote(outputs["trained_model"]) in script
    elif family == "sevennet":
        config = YAML(typ="safe").load(remote / inputs["config"])
        assert (
            config["train"]["continue"]["checkpoint"] == files.pretrained_model
        )
        assert config["data"]["load_trainset_path"] == [inputs["train"]]
        from temper.mlff.pipeline.data_preparation import sevennet

        calls = []
        monkeypatch.setattr(
            sevennet,
            "prepare_config",
            lambda source, output, **kwargs: calls.append((source, output)),
        )
        adapter.prepare_training()
        assert calls == [
            (remote / inputs["config"], remote / outputs["resolved_config"])
        ]
        assert shlex.quote(outputs["native_outputs"]) in script
    elif family == "mattersim":
        for path in (
            files.pretrained_model,
            inputs["train"],
            inputs["validation"],
            outputs["native_outputs"],
            outputs["trained_model"],
        ):
            assert shlex.quote(path) in script
    else:
        from types import ModuleType
        import sys

        native = ModuleType("torchnep")

        def train(config, data, **kwargs):
            assert config == str(remote / inputs["config"])
            assert data == str(remote / inputs["train"])
            assert kwargs["valid_file"] == str(remote / inputs["validation"])
            assert kwargs["finetune_from"] == str(
                remote / files.pretrained_model
            )
            assert kwargs["output_dir"] == str(
                remote / outputs["native_outputs"]
            )
            output = Path(kwargs["output_dir"])
            output.mkdir()
            (output / "nep_best.txt").write_bytes(b"earlier best")
            (output / "nep_final.txt").write_bytes(b"trained")
            (output / "loss.out").write_text(
                f"{bundle.mlff_spec.training_parameters['epoch']} 0.1\n"
            )

        native.train_nep = train
        monkeypatch.setitem(sys.modules, "torchnep", native)
        adapter.train()
        assert (remote / outputs["trained_model"]).read_bytes() == b"trained"


def test_evaluation_resolves_paths_from_explicit_root(
    tmp_path,
    zeroshot_training_unit,
    mlff_spec_factory,
    monkeypatch,
):
    target, bundle = MLFFTrainBundle(
        training_unit=zeroshot_training_unit,
        mlff_spec=mlff_spec_factory("mace", with_training=False),
    ).write_submit_folder(tmp_path / "job")
    runner.prepare_bundle(target / "bundle.json")
    def calculator(*, identifier: str, model: Path, parameters: dict[str, object]) -> FakeCalculator:
        assert model == target / bundle.files.pretrained_model
        return FakeCalculator()

    monkeypatch.setattr(evaluate_mlff, "build_calculator", calculator)
    monkeypatch.chdir(tmp_path)
    manifest = create_result_manifest(bundle, target)
    manifest.status = "running"
    # Loading a manifest from elsewhere does not change its explicit base.
    relocated = tmp_path / "elsewhere.json"
    manifest.to_json_file(relocated)
    manifest = type(manifest).from_json_file(relocated)
    evaluate_mlff.run_test(manifest, bundle_root=target)
    for path in bundle.files.test_output_files.values():
        assert (target / path).is_file()


def test_unknown_bundle_format_is_rejected(
    zeroshot_training_unit, mlff_spec_factory
):
    bundle = MLFFTrainBundle(
        training_unit=zeroshot_training_unit,
        mlff_spec=mlff_spec_factory("mace", with_training=False),
    )
    payload = bundle.model_dump()
    payload["schema_version"] = 2
    with pytest.raises(ValidationError, match="schema_version"):
        MLFFTrainBundle.model_validate(payload)


def test_temporary_model_cannot_replace_permanent_model():
    with pytest.raises(ValidationError, match="own path"):
        MLFFBundleFiles(
            pretrained_model="pretrained_models/mace.model",
            training_output_files={
                "trained_model": "training_outputs/model.pt"
            },
            evaluation_model="training_outputs/model.pt",
            test_input_files={},
            test_output_files={},
            training_stress=True,
            test_stress=[],
        )
