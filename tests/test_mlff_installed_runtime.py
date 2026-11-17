"""Portable bundle execution and optional dependency boundaries."""

from collections.abc import Callable
import json
from pathlib import Path
import shutil
import subprocess
import sys

from temper.mlff.pipeline.runner import make_test_manifests

import pytest
import numpy as np

from temper.schemas.mlff_train_bundle import MLFFTrainBundle
from temper.schemas.mlff_spec import MLFFSpec
from temper.schemas.train_unit import TrainingUnit
from temper.mlff.pipeline import runner
from temper.mlff.pipeline import evaluate_mlff
from temper.mlff.pipeline.training_adapters.mace import MACEAdapter
from temper.mlff.pipeline.training_adapters import base as adapter_base
from test_mlff_runtime import FakeCalculator


@pytest.mark.parametrize(
    "family", ["dpa4", "dpa4c", "mace", "mattersim", "sevennet", "nep89"]
)
def test_prepare_after_relocation_without_original_inputs(
    family,
    tmp_path,
    finetune_training_unit,
    mlff_spec_factory,
):
    spec = mlff_spec_factory(family)
    bundle = MLFFTrainBundle(
        training_unit=finetune_training_unit, mlff_spec=spec
    )
    before = bundle.model_dump(mode="json")
    source = bundle.write_submit_folder(tmp_path / "local")[0]
    assert set(p.name for p in source.iterdir()) == {
        "bundle.json",
        "run.sh",
        "training_inputs",
        "testing_inputs",
        "pretrained_models",
    }
    assert bundle.model_dump(mode="json") == before
    remote = tmp_path / "remote"
    source.rename(remote)
    shutil.rmtree(finetune_training_unit.root_path)
    for artifact in spec.pretrained_model.artifacts.values():
        artifact.path.unlink()

    adapter = runner.prepare_bundle(remote / "bundle.json")
    assert adapter.bundle.mlff_train_bundle_id == bundle.mlff_train_bundle_id
    assert (
        adapter.bundle.training_unit.training_unit_id
        == finetune_training_unit.training_unit_id
    )
    assert (remote / "training_inputs/run.sh").is_file() == (family != "nep89")
    tests = make_test_manifests(adapter.bundle, bundle_root=remote)
    assert all(
        (remote / test.execution_input_path).is_file() for test in tests
    )
    assert not (remote / "testing_inputs/test_config.json").exists()
    assert not (remote / "runtime").exists()


@pytest.mark.parametrize("finetune", [False, True])
def test_preparation_preserves_configs_and_restores_missing_files(
    finetune,
    tmp_path,
    finetune_training_unit,
    zeroshot_training_unit,
    mlff_spec_factory,
    monkeypatch,
):
    bundle = MLFFTrainBundle(
        training_unit=(
            finetune_training_unit if finetune else zeroshot_training_unit
        ),
        mlff_spec=mlff_spec_factory("mace", with_training=finetune),
    )
    target, packaged = bundle.write_submit_folder(tmp_path / "job")
    runner.prepare_bundle(target / "bundle.json")
    if finetune:
        config = target / packaged.files.training_input_files["config"]
        config.write_text(config.read_text() + "# caller edit\n")
    before = {
        path: path.read_bytes() for path in target.rglob("*") if path.is_file()
    }

    with monkeypatch.context() as patch:
        patch.setattr(
            MACEAdapter,
            "training_file_contents",
            lambda *args: pytest.fail("Existing configs were rendered again"),
        )
        runner.prepare_bundle(target / "bundle.json")
    assert all(
        path.read_bytes() == content for path, content in before.items()
    )

    # A missing script must not cause the edited native config to be
    # rewritten.
    if finetune:
        script = target / packaged.files.training_input_files["script"]
        script.unlink()
        runner.prepare_bundle(target / "bundle.json")
        assert script.read_bytes() == before[script]
        assert config.read_bytes() == before[config]
    adapter = runner.prepare_bundle(target / "bundle.json")
    runner.write_configuration_files(adapter)
    assert "config" not in packaged.files.test_input_files
    assert not (target / "testing_inputs/test_config.json").exists()
    if finetune:
        assert "# caller edit" not in config.read_text()


@pytest.mark.parametrize(
    "family", ["dpa4", "dpa4c", "mace", "mattersim", "sevennet", "nep89"]
)
def test_older_bundle_restores_configs_from_copied_inputs(
    family,
    tmp_path,
    finetune_training_unit,
    mlff_spec_factory,
):
    spec = mlff_spec_factory(family)
    target, packaged = MLFFTrainBundle(
        training_unit=finetune_training_unit,
        mlff_spec=spec,
    ).write_submit_folder(tmp_path / "old")
    paths = [
        path
        for key, path in packaged.files.training_input_files.items()
        if key in {"config", "script"}
    ]
    assert all(not (target / path).exists() for path in paths)
    runner.prepare_bundle(target / "bundle.json")
    expected = {path: (target / path).read_bytes() for path in paths}
    for path in paths:
        (target / path).unlink()
    remote = tmp_path / "remote"
    target.rename(remote)
    shutil.rmtree(finetune_training_unit.root_path)
    spec.pretrained_model.artifacts["model"].path.unlink()
    runner.prepare_bundle(remote / "bundle.json")
    assert all(
        (remote / path).read_bytes() == content
        for path, content in expected.items()
    )


@pytest.mark.parametrize("finetune", [False, True])
def test_runner_trains_then_evaluates_from_another_directory(
    finetune: bool,
    tmp_path: Path,
    finetune_training_unit: TrainingUnit,
    zeroshot_training_unit: TrainingUnit,
    mlff_spec_factory: Callable[..., MLFFSpec],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run from another directory and separate arrays from manifest metadata.

    Parameters
    ----------
    finetune : bool
        Whether training should run before evaluation.
    tmp_path : Path
        Isolated directory containing the packaged calculation.
    finetune_training_unit, zeroshot_training_unit : TrainingUnit
        Labeled datasets for the two execution modes.
    mlff_spec_factory : Callable[..., MLFFSpec]
        Creates a MACE specification without installing MACE.
    monkeypatch : pytest.MonkeyPatch
        Replaces training, CUDA checks and calculator construction.
    """
    unit = finetune_training_unit if finetune else zeroshot_training_unit
    bundle = MLFFTrainBundle(
        training_unit=unit,
        mlff_spec=mlff_spec_factory("mace", with_training=finetune),
    )
    target = bundle.write_submit_folder(tmp_path / "job")[0]
    events = []
    monkeypatch.setattr(runner, "require_cuda", lambda: events.append("cuda"))
    monkeypatch.setattr(
        MACEAdapter, "prepare_training", lambda self: events.append("prepare")
    )

    def train(command, *, cwd, env, check):
        assert cwd == target
        assert command == ["bash", "training_inputs/run.sh"]
        assert env["PYTHON_BIN"] == sys.executable
        assert check
        events.append("train")
        (target / "training_outputs/finetuned_mace.model").write_bytes(
            b"trained"
        )
        parameters = bundle.mlff_spec.training_parameters
        checkpoint = (
            target
            / "training_outputs/mace/checkpoints"
            / "temper_model_run-{}_epoch-{}.pt".format(
                parameters["seed"], parameters["max_num_epochs"] - 1
            )
        )
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"last epoch")

    def calculator(*, identifier: str, model: Path, parameters: dict[str, object]) -> FakeCalculator:
        assert model.is_absolute() and model.is_file()
        assert model.name == (
            "finetuned_mace.model" if finetune else "mace.model"
        )
        events.append("evaluate")
        return FakeCalculator()

    monkeypatch.setattr(adapter_base.subprocess, "run", train)
    monkeypatch.setattr(evaluate_mlff, "build_calculator", calculator)
    monkeypatch.chdir(tmp_path)
    runner.run_pipeline(target / "bundle.json")
    assert events == (
        ["cuda", "prepare", "train", "evaluate"]
        if finetune
        else ["cuda", "evaluate"]
    )
    summary = json.loads((target / "results.json").read_text())
    assert summary["status"] == "completed"
    assert len(summary["test_manifests"]) == len(unit.test_sets)
    with np.load(target / summary["test_manifests"][0]["prediction_path"], allow_pickle=False) as saved:
        assert "_metadata" not in saved.files
    assert "temper-bench" in summary["package_versions"]
    assert summary["training_wall_time_seconds"] is not None if finetune else (
        summary["training_wall_time_seconds"] is None
    )
    assert not list((target / "testing_outputs").glob("*.json"))
    assert not (target / "training_outputs/training_result.json").exists()


def test_runtime_uses_serialized_directories_and_checks_version(
    tmp_path,
    finetune_training_unit,
    mlff_spec_factory,
):
    path = (
        MLFFTrainBundle(
            training_unit=finetune_training_unit,
            mlff_spec=mlff_spec_factory("mace"),
        ).write_submit_folder(tmp_path / "job")[0]
        / "bundle.json"
    )
    bundle = MLFFTrainBundle.from_json_file(path)
    bundle.files.training_input_files["config"] = "work/native/mace.yaml"
    bundle.files.training_output_files["trained_model"] = (
        "results/models/finetuned_mace.model"
    )
    # Configs have not been generated yet, so preparation uses the new
    # mapping.
    bundle.to_json_file(path)
    adapter = runner.prepare_bundle(path)
    assert (path.parent / "work/native/mace.yaml").exists()
    assert (
        adapter.files.training_output_files["trained_model"]
        == "results/models/finetuned_mace.model"
    )
    bundle.temper_version = "0.0.0"
    bundle.to_json_file(path)
    with pytest.warns(RuntimeWarning, match="written by temper-bench 0.0.0"):
        runner.prepare_bundle(path)


def test_runtime_imports_without_preprocessing_or_model_backends() -> None:
    """Import shared utilities independently and keep optional backends lazy.

    A fresh interpreter checks that utilities do not import schemas or workflow
    modules, then verifies that runtime imports need no preprocessing or model
    packages. A failed import or dependency assertion fails the subprocess.
    """
    source = Path(__file__).resolve().parents[1] / "src"
    program = f"""
import sys
sys.path.insert(0, {str(source)!r})
class BlockPreprocessing:
    def find_spec(self, fullname, *args):
        blocked = {{
            'quests', 'numba', 'torch', 'mace', 'sevenn', 'deepmd',
            'mattersim', 'torchnep', 'calorine'
        }}
        if fullname.split('.')[0] in blocked:
            raise AssertionError('Unexpected dependency: ' + fullname)
sys.meta_path.insert(0, BlockPreprocessing())
from temper.utils import (
    cuda, dataset_sizes, defaults, extxyz, file_hashing,
    file_paths, file_writing,
)
for prefix in (
    'temper.schemas', 'temper.mlff', 'temper.splitting',
    'temper.job_management', 'temper.results_loading',
):
    assert not any(
        name == prefix or name.startswith(prefix + '.') for name in sys.modules
    ), prefix
from temper.mlff.pipeline.runner import prepare_bundle
from temper.entrypoints.main import main_parser
arguments = main_parser().parse_args(['run_pipeline', 'bundle.json'])
assert arguments.command == 'run_pipeline'
assert 'temper.splitting.split' not in sys.modules
assert 'temper.schemas.split' not in sys.modules
"""
    subprocess.run(
        [sys.executable, "-I", "-c", program],
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("family", ["dpa4", "dpa4c", "sevennet"])
def test_backend_preparation_calls_functions_with_packaged_paths(
    family,
    tmp_path,
    finetune_training_unit,
    mlff_spec_factory,
    monkeypatch,
):
    from temper.mlff.pipeline.data_preparation import deepmd, sevennet

    target = MLFFTrainBundle(
        training_unit=finetune_training_unit,
        mlff_spec=mlff_spec_factory(family),
    ).write_submit_folder(tmp_path / "job")[0]
    adapter = runner.prepare_bundle(target / "bundle.json")
    calls = []
    if family == "sevennet":
        monkeypatch.setattr(
            sevennet,
            "prepare_config",
            lambda source, output, **kwargs: calls.append(
                (source, output, kwargs)
            ),
        )
    else:
        monkeypatch.setattr(
            deepmd,
            "prepare_dataset",
            lambda source, output: calls.append((source, output)),
        )
    monkeypatch.chdir(tmp_path)
    adapter.prepare_training()
    if family == "sevennet":
        assert calls == [
            (
                target / "training_inputs/sevennet.yaml",
                target / "training_outputs/sevennet_resolved.yaml",
                {"bundle_root": target},
            )
        ]
    else:
        assert calls == [
            (
                target / "training_inputs/train.extxyz",
                target / "training_inputs/data/train",
            ),
            (
                target / "training_inputs/validation.extxyz",
                target / "training_inputs/data/validation",
            ),
        ]


def test_deepmd_conversion_uses_ase_reader_and_writes_systems(
    tmp_path, monkeypatch
):
    from types import ModuleType, SimpleNamespace
    from temper.mlff.pipeline.data_preparation.deepmd import prepare_dataset

    calls = []
    module = ModuleType("dpdata")

    def from_file(source, *, fmt):
        calls.append((source, fmt))
        return SimpleNamespace(
            to=lambda fmt, output: calls.append((fmt, output))
        )

    module.MultiSystems = SimpleNamespace(from_file=from_file)
    monkeypatch.setitem(sys.modules, "dpdata", module)
    source, output = tmp_path / "train.extxyz", tmp_path / "systems"
    prepare_dataset(source, output)
    assert output.is_dir()
    assert calls == [
        (str(source), "ase/structure"),
        ("deepmd/npy", str(output)),
    ]


def test_training_failure_does_not_start_testing(
    tmp_path,
    finetune_training_unit,
    mlff_spec_factory,
    monkeypatch,
):
    target = MLFFTrainBundle(
        training_unit=finetune_training_unit,
        mlff_spec=mlff_spec_factory("mace"),
    ).write_submit_folder(tmp_path / "job")[0]
    monkeypatch.setattr(runner, "require_cuda", lambda: None)

    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0])

    monkeypatch.setattr(adapter_base.subprocess, "run", fail)
    monkeypatch.setattr(
        runner,
        "run_test",
        lambda path: pytest.fail("Testing ran after failed training"),
    )
    with pytest.raises(subprocess.CalledProcessError):
        runner.run_pipeline(target / "bundle.json")


def test_bundle_owns_test_configuration_after_serialization(
    tmp_path: Path,
    finetune_training_unit: TrainingUnit,
    mlff_spec_factory: Callable[..., MLFFSpec],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep runtime paths fixed when installed builder defaults change.

    Parameters
    ----------
    tmp_path : Path
        Temporary bundle directory.
    finetune_training_unit : TrainingUnit
        Unit with readable training and testing datasets.
    mlff_spec_factory : Callable[..., MLFFSpec]
        Fixture constructing a specification with a local checkpoint.
    monkeypatch : pytest.MonkeyPatch
        Fixture replacing builder defaults only for this test.
    """
    from temper.mlff.spec_builders import MACESpecBuilder

    target = MLFFTrainBundle(
        training_unit=finetune_training_unit,
        mlff_spec=mlff_spec_factory("mace"),
    ).write_submit_folder(tmp_path / "job")[0]
    bundle = MLFFTrainBundle.from_json_file(target / "bundle.json")
    tests = make_test_manifests(bundle, bundle_root=target)
    # Changing an installed builder's default cannot change a packaged
    # model path.
    monkeypatch.setattr(
        MACESpecBuilder,
        "default_file_layout",
        MACESpecBuilder.default_file_layout.model_copy(
            update={
                "training_output_filenames": {"trained_model": "another.model"}
            },
            deep=True,
        ),
    )
    adapter = runner.prepare_bundle(target / "bundle.json")
    assert (
        bundle.files.training_output_files["trained_model"]
        == adapter.trained_model_path
        == "training_outputs/finetuned_mace.model"
    )
    assert tests == make_test_manifests(adapter.bundle, bundle_root=target)


@pytest.mark.parametrize("run_seed", [42, 23])
def test_nep_training_is_a_function_call_before_evaluation(
    run_seed,
    tmp_path,
    finetune_training_unit,
    mlff_spec_factory,
    monkeypatch,
):
    from types import ModuleType

    native = ModuleType("torchnep")
    monkeypatch.setitem(sys.modules, "torchnep", native)

    target = MLFFTrainBundle(
        training_unit=finetune_training_unit,
        mlff_spec=mlff_spec_factory(
            "nep89", training_parameters={"run_seed": run_seed}
        ),
    ).write_submit_folder(tmp_path / "job")[0]
    events = []
    monkeypatch.setattr(runner, "require_cuda", lambda: None)

    def train(config, train, **kwargs):
        events.append("train")
        assert config == str(target / "training_inputs/nep.in")
        assert train == str(target / "training_inputs/train.extxyz")
        assert kwargs["valid_file"] == str(
            target / "training_inputs/validation.extxyz"
        )
        assert kwargs["finetune_from"] == str(
            target / "pretrained_models/nep89.txt"
        )
        assert kwargs["device"] == "cuda"
        assert kwargs["restart"] is False
        assert kwargs["recompute_q_scaler"] is False
        assert kwargs["slim_types"] is True
        assert kwargs["run_seed"] == run_seed
        assert "run_seed" not in Path(config).read_text()
        output = Path(kwargs["output_dir"])
        output.mkdir(parents=True)
        (output / "nep_best.txt").write_bytes(b"earlier best")
        (output / "nep_final.txt").write_bytes(b"trained")
        (output / "loss.out").write_text("# epoch loss\n100 0.1\n")
        print("training completed")

    def calculator(*, identifier: str, model: Path, parameters: dict[str, object]) -> FakeCalculator:
        events.append("evaluate")
        assert model.read_bytes() == b"trained"
        return FakeCalculator()

    monkeypatch.setattr(native, "train_nep", train, raising=False)
    monkeypatch.setattr(evaluate_mlff, "build_calculator", calculator)
    monkeypatch.chdir(tmp_path)
    runner.run_pipeline(target / "bundle.json")
    assert events == ["train", "evaluate"]
    assert (
        "training completed"
        in (target / "training_outputs/training.log").read_text()
    )
    assert not (target / "training_inputs/run.sh").exists()


def test_pipeline_contains_no_internal_script_entrypoints():
    import ast
    from temper.mlff import pipeline

    root = Path(pipeline.__file__).parent
    assert not list(root.rglob("__main__.py"))
    for source in root.rglob("*.py"):
        tree = ast.parse(source.read_text(encoding="utf-8"))
        assert not any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "main"
            for node in ast.walk(tree)
        ), source
