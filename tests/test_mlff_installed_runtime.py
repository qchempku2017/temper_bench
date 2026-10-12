"""Portable bundle execution and optional dependency boundaries."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from temper.schemas.mlff_train_bundle import MLFFTrainBundle
from temper.mlff.pipeline import runner
from temper.mlff.pipeline import evaluate_mlff
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
    source = bundle.write_submit_folder(tmp_path / "local")[0]
    assert set(p.name for p in source.iterdir()) == {"bundle.json", "run.sh", "training_inputs", "testing_inputs", "pretrained_models"}
    assert bundle.model_dump(mode="json") == before
    remote = tmp_path / "remote"
    source.rename(remote)
    shutil.rmtree(finetune_training_unit.root_path)
    for artifact in spec.pretrained_model.artifacts.values():
        artifact.path.unlink()

    adapter = runner.prepare_bundle(remote / "bundle.json")
    assert adapter.bundle.mlff_train_bundle_id == bundle.mlff_train_bundle_id
    assert adapter.bundle.training_unit.training_unit_id == finetune_training_unit.training_unit_id
    assert (remote / "training_inputs/run.sh").is_file() == (family != "nep89")
    config = json.loads((remote / "testing_inputs/test_config.json").read_text())
    assert all((remote / item["path"]).is_file() for item in config["test_datasets"])
    assert not (remote / "runtime").exists()


@pytest.mark.parametrize("finetune", [False, True])
def test_runner_trains_then_evaluates_from_another_directory(
    finetune, tmp_path, finetune_training_unit, zeroshot_training_unit,
    mlff_spec_factory, monkeypatch,
):
    unit = finetune_training_unit if finetune else zeroshot_training_unit
    bundle = MLFFTrainBundle(training_unit=unit, mlff_spec=mlff_spec_factory("mace", with_training=finetune))
    target = bundle.write_submit_folder(tmp_path / "job")[0]
    events = []
    monkeypatch.setattr(runner, "require_cuda", lambda: events.append("cuda"))
    monkeypatch.setattr(MACEAdapter, "prepare_training", lambda self: events.append("prepare"))

    def train(command, *, cwd, env, check):
        assert cwd == target
        assert command == ["bash", "training_inputs/run.sh"]
        assert env["PYTHON_BIN"] == sys.executable
        assert check
        events.append("train")
        (target / "training_outputs/finetuned_mace.model").write_bytes(b"trained")

    def calculator(config):
        model = Path(config["model"])
        assert model.is_absolute() and model.is_file()
        assert model.name == ("finetuned_mace.model" if finetune else "mace.model")
        events.append("evaluate")
        return FakeCalculator()

    monkeypatch.setattr(adapter_base.subprocess, "run", train)
    monkeypatch.setattr(evaluate_mlff, "build_calculator", calculator)
    monkeypatch.chdir(tmp_path)
    runner.run_pipeline(target / "bundle.json")
    assert events == (["cuda", "prepare", "train", "evaluate"] if finetune else ["cuda", "evaluate"])
    summary = json.loads((target / "testing_outputs/summary.json").read_text())
    assert len(summary["datasets"]) == len(unit.test_sets)
    assert "temper-bench" in summary["package_versions"]


def test_runtime_uses_serialized_directories_and_checks_version(
    tmp_path, finetune_training_unit, mlff_spec_factory,
):
    path = MLFFTrainBundle(training_unit=finetune_training_unit, mlff_spec=mlff_spec_factory("mace")).write_submit_folder(tmp_path / "job")[0] / "bundle.json"
    bundle = MLFFTrainBundle.model_validate_json(path.read_text())
    bundle.files.training_input_files["config"] = "work/native/mace.yaml"
    bundle.files.training_output_files["trained_model"] = "results/models/finetuned_mace.model"
    path.write_text(bundle.model_dump_json())
    runner.prepare_bundle(path)
    assert (path.parent / "work/native/mace.yaml").exists()
    assert "results/models/finetuned_mace.model" in (path.parent / "testing_inputs/test_config.json").read_text()
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
    ).write_submit_folder(tmp_path / "job")[0]
    adapter = runner.prepare_bundle(target / "bundle.json")
    calls = []
    if family == "sevennet":
        monkeypatch.setattr(sevennet, "prepare_config", lambda source, output, **kwargs: calls.append((source, output, kwargs)))
    else:
        monkeypatch.setattr(deepmd, "prepare_dataset", lambda source, output: calls.append((source, output)))
    monkeypatch.chdir(tmp_path)
    adapter.prepare_training()
    if family == "sevennet":
        assert calls == [(target / "training_inputs/sevennet.yaml", target / "training_outputs/sevennet_resolved.yaml", {"bundle_root": target})]
    else:
        assert calls == [
            (target / "training_inputs/train.extxyz", target / "training_inputs/data/train"),
            (target / "training_inputs/validation.extxyz", target / "training_inputs/data/validation"),
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
    ).write_submit_folder(tmp_path / "job")[0]
    monkeypatch.setattr(runner, "require_cuda", lambda: None)
    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0])
    monkeypatch.setattr(adapter_base.subprocess, "run", fail)
    monkeypatch.setattr(runner, "run_test", lambda path: pytest.fail("Testing ran after failed training"))
    with pytest.raises(subprocess.CalledProcessError):
        runner.run_pipeline(target / "bundle.json")


def test_bundle_owns_test_configuration_after_serialization(
    tmp_path, finetune_training_unit, mlff_spec_factory, monkeypatch,
):
    target = MLFFTrainBundle(
        training_unit=finetune_training_unit, mlff_spec=mlff_spec_factory("mace"),
    ).write_submit_folder(tmp_path / "job")[0]
    bundle = MLFFTrainBundle.model_validate_json((target / "bundle.json").read_text())
    config = bundle.test_config()
    # Changing an installed adapter's default cannot change a packaged model path.
    monkeypatch.setattr(MACEAdapter, "training_output_filenames", {"trained_model": "another.model"})
    adapter = runner.prepare_bundle(target / "bundle.json")
    assert config["model"] == adapter.trained_model_path == "training_outputs/finetuned_mace.model"
    assert config == json.loads((target / "testing_inputs/test_config.json").read_text())


@pytest.mark.parametrize("run_seed", [42, 23])
def test_nep_training_is_a_function_call_before_evaluation(
    run_seed, tmp_path, finetune_training_unit, mlff_spec_factory, monkeypatch,
):
    from types import ModuleType
    native = ModuleType("torchnep")
    monkeypatch.setitem(sys.modules, "torchnep", native)

    target = MLFFTrainBundle(
        training_unit=finetune_training_unit,
        mlff_spec=mlff_spec_factory("nep89", training_parameters={"run_seed": run_seed}),
    ).write_submit_folder(tmp_path / "job")[0]
    events = []
    monkeypatch.setattr(runner, "require_cuda", lambda: None)
    def train(config, train, **kwargs):
        events.append("train")
        assert config == str(target / "training_inputs/nep.in")
        assert train == str(target / "training_inputs/train.extxyz")
        assert kwargs["valid_file"] == str(target / "training_inputs/validation.extxyz")
        assert kwargs["finetune_from"] == str(target / "pretrained_models/nep89.txt")
        assert kwargs["device"] == "cuda"
        assert kwargs["restart"] is False
        assert kwargs["recompute_q_scaler"] is False
        assert kwargs["slim_types"] is True
        assert kwargs["run_seed"] == run_seed
        assert "run_seed" not in Path(config).read_text()
        output = Path(kwargs["output_dir"])
        output.mkdir(parents=True)
        (output / "nep_best.txt").write_bytes(b"trained")
        print("training completed")
    def calculator(config):
        events.append("evaluate")
        assert Path(config["model"]).read_bytes() == b"trained"
        return FakeCalculator()
    monkeypatch.setattr(native, "train_nep", train, raising=False)
    monkeypatch.setattr(evaluate_mlff, "build_calculator", calculator)
    monkeypatch.chdir(tmp_path)
    runner.run_pipeline(target / "bundle.json")
    assert events == ["train", "evaluate"]
    assert "training completed" in (target / "training_outputs/training.log").read_text()
    assert not (target / "training_inputs/run.sh").exists()


def test_pipeline_contains_no_internal_script_entrypoints():
    import ast
    from temper.mlff import pipeline

    root = Path(pipeline.__file__).parent
    assert not list(root.rglob("__main__.py"))
    for source in root.rglob("*.py"):
        tree = ast.parse(source.read_text(encoding="utf-8"))
        assert not any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "main" for node in ast.walk(tree)), source
