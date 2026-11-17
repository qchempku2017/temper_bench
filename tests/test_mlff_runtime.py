"""Generic ASE test-runtime behavior without real MLFF installations."""

from __future__ import annotations

import json
import sys
from importlib import resources
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.io import read, write
from collections.abc import Callable
from temper.schemas.mlff_train_bundle import MLFFTrainBundle
from temper.schemas.mlff_experiment_result import MLFFExperimentManifest, MLFFTestManifest
from temper.schemas.mlff_spec import MLFFSpec
from temper.schemas.train_unit import TrainingUnit

from temper.mlff.pipeline import evaluate_mlff as run_test
from temper.mlff.pipeline.runner import create_result_manifest, make_test_manifests
from temper.utils.file_hashing import file_sha256


class FakeCalculator(Calculator):
    """Deterministic energy/force/stress calculator for runtime tests."""

    implemented_properties = ["energy", "forces", "stress"]

    def calculate(
        self,
        atoms=None,
        properties=("energy", "forces", "stress"),
        system_changes=all_changes,
    ) -> None:
        super().calculate(atoms, properties, system_changes)
        assert atoms is not None
        atom_count = len(atoms)
        self.results = {
            "energy": float(atom_count),
            "forces": np.full((atom_count, 3), float(atom_count)),
            # ASE's six-vector order is xx, yy, zz, yz, xz, xy. The
            # generic
            # runner requests voigt=False and persists the resulting
            # tensor.
            "stress": np.arange(6, dtype=float),
        }


class NoStressCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def calculate(
        self,
        atoms=None,
        properties=("energy", "forces"),
        system_changes=all_changes,
    ) -> None:
        super().calculate(atoms, properties, system_changes)
        assert atoms is not None
        self.results = {
            "energy": 0.0,
            "forces": np.zeros((len(atoms), 3)),
        }


def _write_runtime_manifest(
    root: Path, unit: TrainingUnit, spec_factory: Callable[..., MLFFSpec], *,
    stress: bool = True,
) -> tuple[Path, MLFFExperimentManifest]:
    if not stress:
        for name in unit.test_sets:
            path = unit.resolve_local_dataset_path(name)
            frames = read(path, index=":")
            for frame in frames:
                frame.calc.results.pop("stress")
            write(path, frames, format="extxyz")
    bundle = MLFFTrainBundle(
        training_unit=unit.copy_with_refreshed_dataset_hashes(),
        mlff_spec=spec_factory("mace", with_training=False),
    )
    target, packaged = bundle.write_submit_folder(root / "run")
    manifest = create_result_manifest(packaged, target)
    manifest.status = "running"
    manifest.to_json_file(target / "results.json")
    return target, manifest


def _install_fake_adapter(monkeypatch: pytest.MonkeyPatch, calculator: Calculator) -> None:
    module = ModuleType("calculator")
    module.build_calculator = lambda **kwargs: calculator
    monkeypatch.setattr(run_test, "build_calculator", module.build_calculator)


@pytest.mark.parametrize("recorded_hashes", [False, True])
def test_pending_manifests_preserve_bundle_and_roundtrip(
    tmp_path: Path, zeroshot_training_unit: TrainingUnit,
    mlff_spec_factory: Callable[..., MLFFSpec], recorded_hashes: bool,
) -> None:
    """Create independent pending instructions with checked reference identities.

    Parameters
    ----------
    tmp_path : Path
        Destination for copied inputs and a serialized experiment manifest.
    zeroshot_training_unit : TrainingUnit
        Test-only input record with automatically calculated checksums.
    mlff_spec_factory : Callable[..., MLFFSpec]
        Creates a model specification with readable dummy weights.
    recorded_hashes : bool
        Whether checksums remain present in the packaged metadata, or are
        omitted to exercise execution with an older metadata-only record.
    """
    bundle = MLFFTrainBundle(
        training_unit=zeroshot_training_unit,
        mlff_spec=mlff_spec_factory("mace", with_training=False),
    )
    with pytest.raises(ValueError, match="not been packaged"):
        make_test_manifests(bundle)
    root, packaged = bundle.write_submit_folder(tmp_path / "run")
    if not recorded_hashes:
        data = packaged.model_dump(exclude={"mlff_train_bundle_id"})
        data["training_unit"].pop("dataset_sha256")
        data["training_unit"].pop("training_unit_id")
        packaged = MLFFTrainBundle.model_validate(
            data, context={"verify_files": False}
        )
        packaged.to_json_file(root / "bundle.json", context={"verify_files": False})
    before = packaged.model_dump()
    if recorded_hashes:
        tests = make_test_manifests(packaged)
    else:
        with pytest.raises(ValueError, match="bundle_root is required"):
            make_test_manifests(packaged)
        tests = make_test_manifests(packaged, bundle_root=root)
    assert all(isinstance(test, MLFFTestManifest) for test in tests)
    for test in tests:
        assert test.status == "pending"
        assert test.prediction_sha256 is None
        assert test.wall_time_seconds is None
        assert test.reference_sha256 == file_sha256(root / test.execution_input_path)
        assert not (root / test.prediction_path).exists()
    manifest = create_result_manifest(packaged, root)
    assert manifest.test_manifests == tests
    manifest.to_json_file(root / "results.json")
    restored = MLFFExperimentManifest.from_json_file(root / "results.json")
    assert restored == manifest
    assert restored.load_results(base_dir=root).test_results == ()
    tests[0].units["energy"] = "changed"
    assert manifest.test_manifests[0].units["energy"] == "eV"
    assert packaged.model_dump() == before


@pytest.mark.parametrize("family", ["dpa4", "dpa4c", "mace", "mattersim", "sevennet", "nep89"])
@pytest.mark.parametrize("finetune", [False, True])
def test_evaluation_uses_shared_bundle_settings(
    tmp_path: Path, finetune_training_unit: TrainingUnit,
    zeroshot_training_unit: TrainingUnit,
    mlff_spec_factory: Callable[..., MLFFSpec], monkeypatch: pytest.MonkeyPatch,
    family: str, finetune: bool,
) -> None:
    """Select one calculator from saved bundle settings and ignore old configs.

    Parameters
    ----------
    tmp_path : Path
        Destination for the packaged experiment and simulated model outputs.
    finetune_training_unit, zeroshot_training_unit : TrainingUnit
        Input datasets for the selected execution mode.
    mlff_spec_factory : Callable[..., MLFFSpec]
        Creates a specification for each supported backend family.
    monkeypatch : pytest.MonkeyPatch
        Replaces calculator construction without importing backend packages.
    family : str
        Supported MLFF family whose model selection is checked.
    finetune : bool
        Whether to use a trained model or temporary export instead of weights.
    """
    unit = finetune_training_unit if finetune else zeroshot_training_unit
    root, bundle = MLFFTrainBundle(
        training_unit=unit,
        mlff_spec=mlff_spec_factory(
            family, with_training=finetune, testing_parameters={"custom": 17},
        ),
    ).write_submit_folder(tmp_path / "run")
    files = bundle.files
    assert files is not None
    if finetune:
        expected_model = root / (files.evaluation_model or files.training_output_files["trained_model"])
        expected_model.parent.mkdir(parents=True, exist_ok=True)
        expected_model.write_bytes(b"evaluation weights")
    else:
        expected_model = root / files.pretrained_model
    # A stale file from a previous preparation cannot override bundle settings.
    (root / "testing_inputs/test_config.json").write_text("invalid old config")
    manifest = create_result_manifest(bundle, root)
    if finetune:
        manifest.training_status = "completed"
        manifest.training_wall_time_seconds = 2.5
    manifest.to_json_file(root / "results.json")
    calls: list[str] = []

    def calculator(*, identifier: str, model: Path, parameters: dict[str, object]) -> Calculator:
        assert identifier == family
        assert model == expected_model and model.is_file()
        assert parameters == {"custom": 17}
        saved = MLFFExperimentManifest.from_json_file(root / "results.json")
        assert all(test.status == "pending" for test in saved.test_manifests)
        calls.append(identifier)
        return FakeCalculator()

    monkeypatch.setattr(run_test, "build_calculator", calculator)
    monkeypatch.chdir(tmp_path)
    restored = MLFFExperimentManifest.from_json_file(root / "results.json")
    run_test.run_test(restored, bundle_root=root)
    assert calls == [family]
    assert restored.status == "completed"
    assert restored.training_wall_time_seconds == (2.5 if finetune else None)
    assert len(restored.load_results(base_dir=root).test_results) == len(unit.test_sets)


@pytest.mark.parametrize("recorded_hashes", [False, True])
def test_evaluation_checks_manifest_hashes_before_each_dataset(
    tmp_path: Path, zeroshot_training_unit: TrainingUnit,
    mlff_spec_factory: Callable[..., MLFFSpec], monkeypatch: pytest.MonkeyPatch,
    recorded_hashes: bool,
) -> None:
    """Keep finished predictions when a later reference file changes.

    Parameters
    ----------
    tmp_path : Path
        Destination for copied reference data and predictions.
    zeroshot_training_unit : TrainingUnit
        Two named test datasets with distinct frames.
    mlff_spec_factory : Callable[..., MLFFSpec]
        Creates a zero-shot MACE specification with dummy weights.
    monkeypatch : pytest.MonkeyPatch
        Installs a deterministic calculator without requiring a GPU.
    recorded_hashes : bool
        Whether the packaged metadata includes hashes or manifest creation
        calculates them for an older metadata-only record.
    """
    root, bundle = MLFFTrainBundle(
        training_unit=zeroshot_training_unit,
        mlff_spec=mlff_spec_factory("mace", with_training=False),
    ).write_submit_folder(tmp_path / "run")
    if not recorded_hashes:
        data = bundle.model_dump(exclude={"mlff_train_bundle_id"})
        data["training_unit"].pop("dataset_sha256")
        data["training_unit"].pop("training_unit_id")
        bundle = MLFFTrainBundle.model_validate(
            data, context={"verify_files": False}
        )
        bundle.to_json_file(root / "bundle.json", context={"verify_files": False})
    manifest = create_result_manifest(bundle, root)
    manifest.to_json_file(root / "results.json")
    manifest = MLFFExperimentManifest.from_json_file(root / "results.json")
    second = manifest.test_manifests[1]
    (root / second.execution_input_path).write_bytes(b"changed reference")
    _install_fake_adapter(monkeypatch, FakeCalculator())
    with pytest.raises(ValueError, match="Test dataset bytes changed: test_cross.extxyz"):
        run_test.run_test(manifest, bundle_root=root)
    saved = MLFFExperimentManifest.from_json_file(root / "results.json")
    assert saved.status == "failed"
    assert [test.status for test in saved.test_manifests] == ["completed", "failed"]
    assert saved.test_manifests[0].prediction_sha256 is not None
    assert saved.test_manifests[0].wall_time_seconds is not None
    assert saved.test_manifests[1].prediction_sha256 is None
    assert saved.test_manifests[1].wall_time_seconds is None
    assert len(saved.load_results(base_dir=root).test_results) == 1


def test_generic_runner_preserves_frames_and_writes_ragged_arrays(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path, zeroshot_training_unit: TrainingUnit,
    mlff_spec_factory: Callable[..., MLFFSpec],
) -> None:
    """Store only numerical arrays in NPZ and descriptions in results.json.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Replaces the installed calculator with a deterministic test calculator.
    tmp_path : Path
        Isolated execution directory.
    zeroshot_training_unit : TrainingUnit
        Named datasets with known structure sizes.
    mlff_spec_factory : Callable[..., MLFFSpec]
        Creates the model specification used to package the test.
    """
    tmp_path, manifest = _write_runtime_manifest(tmp_path, zeroshot_training_unit, mlff_spec_factory)
    _install_fake_adapter(monkeypatch, FakeCalculator())
    with pytest.raises(TypeError, match="manifest"):
        run_test.run_test(bundle_root=tmp_path)
    run_test.run_test(manifest, bundle_root=tmp_path)
    assert manifest.status == "completed"
    assert all(entry.status == "completed" for entry in manifest.test_manifests)

    with np.load(
        tmp_path / "testing_outputs" / "test_own.npz", allow_pickle=False
    ) as data:
        assert data.files == [
            "energies",
            "forces",
            "atom_offsets",
            "frame_indices",
            "stresses",
        ]
        np.testing.assert_array_equal(data["energies"], [1.0, 2.0])
        np.testing.assert_array_equal(data["atom_offsets"], [0, 1, 3])
        np.testing.assert_array_equal(data["frame_indices"], [0, 1])
        assert data["forces"].shape == (3, 3)
        np.testing.assert_array_equal(data["forces"][0], [1.0, 1.0, 1.0])
        np.testing.assert_array_equal(data["forces"][1:], np.full((2, 3), 2.0))
        assert data["stresses"].shape == (2, 3, 3)
        np.testing.assert_array_equal(
            data["stresses"][0],
            [[0.0, 5.0, 4.0], [5.0, 1.0, 3.0], [4.0, 3.0, 2.0]],
        )
        assert all(data[name].dtype != object for name in data.files)

    with np.load(
        tmp_path / "testing_outputs" / "test_cross.npz", allow_pickle=False
    ) as data:
        np.testing.assert_array_equal(data["energies"], [2.0])
        np.testing.assert_array_equal(data["atom_offsets"], [0, 2])

    recorded = MLFFExperimentManifest.from_json_file(tmp_path / "results.json")
    first_test = recorded.test_manifests[0]
    assert first_test.dataset_id == "test_own.extxyz"
    assert first_test.source_filename == "test_own.extxyz"
    assert first_test.requested_properties == ("energy", "forces", "stress")
    assert first_test.wall_time_seconds is not None and first_test.wall_time_seconds >= 0
    assert first_test.units == {
        "energy": "eV",
        "forces": "eV/Angstrom",
        "stress": "eV/Angstrom^3",
    }
    assert "temper-bench" in recorded.package_versions
    loaded = recorded.load_results(base_dir=tmp_path)
    assert loaded.package_versions == recorded.package_versions
    assert loaded.test_results[0].manifest == first_test
    np.testing.assert_array_equal(loaded.test_results[0].predictions["energies"], [1.0, 2.0])

    summary = json.loads(
        (tmp_path / "results.json").read_text(
            encoding="utf-8"
        )
    )
    assert [item["dataset_id"] for item in summary["test_manifests"]] == [
        "test_own.extxyz",
        "test_cross.extxyz",
    ]


def test_generic_runner_omits_stress_when_not_requested(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path, zeroshot_training_unit: TrainingUnit,
    mlff_spec_factory: Callable[..., MLFFSpec],
) -> None:
    """Omit stress arrays and record the absence in the test manifest.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Installs the deterministic calculator.
    tmp_path : Path
        Isolated execution directory.
    zeroshot_training_unit : TrainingUnit
        Datasets whose stress labels are removed before packaging.
    mlff_spec_factory : Callable[..., MLFFSpec]
        Creates the model specification used for evaluation.
    """
    tmp_path, manifest = _write_runtime_manifest(tmp_path, zeroshot_training_unit, mlff_spec_factory, stress=False)
    _install_fake_adapter(monkeypatch, FakeCalculator())
    run_test.run_test(manifest, bundle_root=tmp_path)

    with np.load(
        tmp_path / "testing_outputs" / "test_own.npz", allow_pickle=False
    ) as data:
        assert "stresses" not in data.files
        assert "_metadata" not in data.files
    recorded = MLFFExperimentManifest.from_json_file(tmp_path / "results.json")
    assert recorded.test_manifests[0].units["stress"] is None
    assert recorded.test_manifests[0].requested_properties == ("energy", "forces")
    assert "stresses" not in recorded.load_results(base_dir=tmp_path).test_results[0].predictions


def test_generic_runner_reports_unsupported_stress_with_dataset_and_frame(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path, zeroshot_training_unit: TrainingUnit,
    mlff_spec_factory: Callable[..., MLFFSpec],
) -> None:
    tmp_path, manifest = _write_runtime_manifest(tmp_path, zeroshot_training_unit, mlff_spec_factory, stress=True)
    _install_fake_adapter(monkeypatch, NoStressCalculator())
    with pytest.raises(RuntimeError, match="test_own.extxyz frame 0"):
        run_test.run_test(manifest, bundle_root=tmp_path)


def test_generic_runner_rejects_dataset_escape(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, zeroshot_training_unit: TrainingUnit,
    mlff_spec_factory: Callable[..., MLFFSpec],
) -> None:
    tmp_path, manifest = _write_runtime_manifest(tmp_path, zeroshot_training_unit, mlff_spec_factory, stress=False)
    outside = tmp_path.parent / "outside.extxyz"
    write(outside, Atoms("H"), format="extxyz")
    manifest.test_manifests[0].execution_input_path = "../outside.extxyz"
    _install_fake_adapter(monkeypatch, FakeCalculator())
    with pytest.raises(ValueError, match="does not match its bundle"):
        run_test.run_test(manifest, bundle_root=tmp_path)
