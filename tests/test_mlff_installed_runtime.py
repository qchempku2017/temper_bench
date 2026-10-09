"""Portable bundle execution and optional dependency boundaries."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from temper.mlff import MLFFTrainBundle
from temper.mlff.pipeline import runner
from temper.mlff.pipeline.calculators import mace
from temper.mlff.pipeline.training_adapters.mace import MACEAdapter
from temper.mlff.pipeline.training_adapters import base as adapter_base
from test_mlff_runtime import FakeCalculator


@pytest.mark.parametrize("family", ["dpa4", "dpa4c", "mace", "mattersim", "sevennet", "nep89"])
def test_prepare_after_relocation_without_original_inputs(
    family, tmp_path, finetune_training_unit, mlff_spec_factory,
):
    spec = mlff_spec_factory(family)
    bundle = MLFFTrainBundle(training_unit=finetune_training_unit, mlff_spec=spec)
    before = bundle.model_dump(mode="json")
    source = bundle.write_submit_folder(tmp_path / "local")
    assert set(p.name for p in source.iterdir()) == {"bundle.json", "run.sh", "datasets", "models"}
    assert bundle.model_dump(mode="json") == before
    remote = tmp_path / "remote"
    source.rename(remote)
    shutil.rmtree(finetune_training_unit.root_path)
    for artifact in spec.pretrained_model.artifacts.values():
        artifact.path.unlink()

    adapter = runner.prepare_bundle(remote / "bundle.json")
    assert adapter.bundle.mlff_train_bundle_id == bundle.mlff_train_bundle_id
    assert adapter.bundle.training_unit.training_unit_id == finetune_training_unit.training_unit_id
    assert (remote / "training/run.sh").is_file() == (family != "nep89")
    config = json.loads((remote / "test_config.json").read_text())
    assert all((remote / item["path"]).is_file() for item in config["test_datasets"])
    assert not (remote / "runtime").exists()


@pytest.mark.parametrize("finetune", [False, True])
def test_runner_trains_then_evaluates_from_another_directory(
    finetune, tmp_path, finetune_training_unit, zeroshot_training_unit,
    mlff_spec_factory, monkeypatch,
):
    unit = finetune_training_unit if finetune else zeroshot_training_unit
    bundle = MLFFTrainBundle(training_unit=unit, mlff_spec=mlff_spec_factory("mace", with_training=finetune))
    target = bundle.write_submit_folder(tmp_path / "job")
    events = []
    monkeypatch.setattr(runner, "require_cuda", lambda: events.append("cuda"))
    monkeypatch.setattr(MACEAdapter, "prepare_training", lambda self: events.append("prepare"))

    def train(command, *, cwd, env, check):
        assert cwd == target
        assert command == ["bash", "training/run.sh"]
        assert env["PYTHON_BIN"] == sys.executable
        assert check
        events.append("train")
        (target / "artifacts/finetuned_mace.model").write_bytes(b"trained")

    def calculator(config):
        model = Path(config["model"])
        assert model.is_absolute() and model.is_file()
        assert model.name == ("finetuned_mace.model" if finetune else "mace.model")
        events.append("evaluate")
        return FakeCalculator()

    monkeypatch.setattr(adapter_base.subprocess, "run", train)
    monkeypatch.setattr(mace, "build_calculator", calculator)
    monkeypatch.chdir(tmp_path)
    runner.run_pipeline(target / "bundle.json")
    assert events == (["cuda", "prepare", "train", "evaluate"] if finetune else ["cuda", "evaluate"])
    summary = json.loads((target / "outputs/test_summary.json").read_text())
    assert len(summary["datasets"]) == len(unit.test_sets)
    assert "temper-bench" in summary["package_versions"]


def test_runtime_uses_serialized_directories_and_checks_version(
    tmp_path, finetune_training_unit, mlff_spec_factory,
):
    path = MLFFTrainBundle(training_unit=finetune_training_unit, mlff_spec=mlff_spec_factory("mace")).write_submit_folder(tmp_path / "job") / "bundle.json"
    bundle = MLFFTrainBundle.model_validate_json(path.read_text())
    bundle.files.training_dir = "work/native"
    bundle.files.outputs_dir = "results"
    bundle.files.artifacts_dir = "results/models"
    path.write_text(bundle.model_dump_json())
    runner.prepare_bundle(path)
    assert (path.parent / "work/native/mace.yaml").exists()
    assert "results/models/finetuned_mace.model" in (path.parent / "test_config.json").read_text()
    bundle.temper_version = "0.0.0"
    path.write_text(bundle.model_dump_json())
    with pytest.warns(RuntimeWarning, match="written by temper-bench 0.0.0"):
        runner.prepare_bundle(path)


def test_runtime_imports_without_preprocessing_or_model_backends():
    source = Path(__file__).resolve().parents[1] / "src"
    program = f'''
import sys
sys.path.insert(0, {str(source)!r})
class BlockPreprocessing:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {{'quests', 'numba', 'torch', 'mace', 'sevenn', 'deepmd', 'mattersim', 'torchnep', 'calorine'}}:
            raise AssertionError('Unexpected dependency: ' + fullname)
sys.meta_path.insert(0, BlockPreprocessing())
from temper.mlff.pipeline.runner import prepare_bundle
from temper.entrypoints.main import main_parser
assert main_parser().parse_args(['run_pipeline', 'bundle.json']).command == 'run_pipeline'
assert 'temper.splitting.split' not in sys.modules
assert 'temper.schemas.split' not in sys.modules
'''
    subprocess.run([sys.executable, "-I", "-c", program], check=True, capture_output=True, text=True)


@pytest.mark.parametrize("family", ["dpa4", "dpa4c", "sevennet"])
def test_backend_preparation_calls_functions_with_packaged_paths(
    family, tmp_path, finetune_training_unit, mlff_spec_factory, monkeypatch,
):
    from temper.mlff.pipeline.data_preparation import deepmd, sevennet

    target = MLFFTrainBundle(
        training_unit=finetune_training_unit, mlff_spec=mlff_spec_factory(family),
    ).write_submit_folder(tmp_path / "job")
    adapter = runner.prepare_bundle(target / "bundle.json")
    calls = []
    if family == "sevennet":
        monkeypatch.setattr(sevennet, "prepare_config", lambda source, output, **kwargs: calls.append((source, output, kwargs)))
    else:
        monkeypatch.setattr(deepmd, "prepare_dataset", lambda source, output: calls.append((source, output)))
    monkeypatch.chdir(tmp_path)
    adapter.prepare_training()
    if family == "sevennet":
        assert calls == [(target / "training/sevennet.yaml", target / "training/sevennet_resolved.yaml", {"bundle_root": target})]
    else:
        assert calls == [
            (target / "datasets/train.extxyz", target / "training/data/train"),
            (target / "datasets/validation.extxyz", target / "training/data/validation"),
        ]


def test_deepmd_conversion_uses_ase_reader_and_writes_systems(tmp_path, monkeypatch):
    from types import ModuleType, SimpleNamespace
    from temper.mlff.pipeline.data_preparation.deepmd import prepare_dataset

    calls = []
    module = ModuleType("dpdata")
    def from_file(source, *, fmt):
        calls.append((source, fmt))
        return SimpleNamespace(to=lambda fmt, output: calls.append((fmt, output)))
    module.MultiSystems = SimpleNamespace(from_file=from_file)
    monkeypatch.setitem(sys.modules, "dpdata", module)
    source, output = tmp_path / "train.extxyz", tmp_path / "systems"
    prepare_dataset(source, output)
    assert output.is_dir()
    assert calls == [(str(source), "ase/structure"), ("deepmd/npy", str(output))]


def test_training_failure_does_not_start_testing(
    tmp_path, finetune_training_unit, mlff_spec_factory, monkeypatch,
):
    target = MLFFTrainBundle(
        training_unit=finetune_training_unit, mlff_spec=mlff_spec_factory("mace"),
    ).write_submit_folder(tmp_path / "job")
    monkeypatch.setattr(runner, "require_cuda", lambda: None)
    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0])
    monkeypatch.setattr(adapter_base.subprocess, "run", fail)
    monkeypatch.setattr(runner.run_test, "run", lambda path: pytest.fail("Testing ran after failed training"))
    with pytest.raises(subprocess.CalledProcessError):
        runner.run_pipeline(target / "bundle.json")


def test_bundle_owns_test_configuration_after_serialization(
    tmp_path, finetune_training_unit, mlff_spec_factory, monkeypatch,
):
    target = MLFFTrainBundle(
        training_unit=finetune_training_unit, mlff_spec=mlff_spec_factory("mace"),
    ).write_submit_folder(tmp_path / "job")
    bundle = MLFFTrainBundle.model_validate_json((target / "bundle.json").read_text())
    config = bundle.test_config()
    # Changing an installed adapter's default cannot change a packaged model path.
    monkeypatch.setattr(MACEAdapter, "trained_model_filename", "another.model")
    adapter = runner.prepare_bundle(target / "bundle.json")
    assert config["model"] == adapter.trained_model_path == "artifacts/finetuned_mace.model"
    assert config == json.loads((target / "test_config.json").read_text())


def test_nep_training_is_a_function_call_before_evaluation(
    tmp_path, finetune_training_unit, mlff_spec_factory, monkeypatch,
):
    from temper.mlff.pipeline import train_nep89
    from temper.mlff.pipeline.calculators import nep89

    target = MLFFTrainBundle(
        training_unit=finetune_training_unit, mlff_spec=mlff_spec_factory("nep89"),
    ).write_submit_folder(tmp_path / "job")
    events = []
    monkeypatch.setattr(runner, "require_cuda", lambda: None)
    def train(**kwargs):
        events.append("train")
        assert kwargs["config"] == str(target / "training/torchnep/nep.in")
        assert kwargs["train"] == str(target / "datasets/train.extxyz")
        assert kwargs["validation"] == str(target / "datasets/validation.extxyz")
        assert kwargs["model"] == str(target / "models/nep89.txt")
        output = Path(kwargs["output_directory"])
        output.mkdir(parents=True)
        (output / "nep_best.txt").write_bytes(b"trained")
        print("training completed")
    def calculator(config):
        events.append("evaluate")
        assert Path(config["model"]).read_bytes() == b"trained"
        return FakeCalculator()
    monkeypatch.setattr(train_nep89, "run", train)
    monkeypatch.setattr(nep89, "build_calculator", calculator)
    monkeypatch.chdir(tmp_path)
    runner.run_pipeline(target / "bundle.json")
    assert events == ["train", "evaluate"]
    assert "training completed" in (target / "outputs/training.log").read_text()
    assert not (target / "training/run.sh").exists()


def test_pipeline_contains_no_internal_script_entrypoints():
    import ast
    from temper.mlff import pipeline

    root = Path(pipeline.__file__).parent
    assert not list(root.rglob("__main__.py"))
    for source in root.rglob("*.py"):
        tree = ast.parse(source.read_text(encoding="utf-8"))
        assert not any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "main" for node in ast.walk(tree)), source
