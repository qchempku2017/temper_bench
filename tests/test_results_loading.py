"""Result recording and alignment with the user's original split datasets."""

from collections.abc import Callable
import json
from pathlib import Path
import shutil

from ase.io import read, write
import numpy as np
import pytest

from temper.job_management.download_utils import (
    get_download_files, verify_results, verify_result_manifest,
)
from temper.job_management.local import LocalBackend
from temper.job_management.schemas.records import JobRecord
from temper.utils.file_writing import write_json_atomically
from temper.mlff.pipeline import runner
from temper.mlff.pipeline.training_adapters.mace import MACEAdapter
from temper.schemas.mlff_spec import MLFFSpec
from temper.schemas.mlff_experiment_result import (
    MLFFExperimentResult, MLFFTestResult, MLFFTestManifest, MLFFExperimentManifest,
)
from temper.mlff.pipeline.runner import create_result_manifest
from temper.results_loading.manifest_io import load_bundle_from_manifest
from monty.serialization import dumpfn, loadfn
from monty.json import MontyDecoder
from pydantic import ValidationError
from temper.schemas.base import MSONableModel
from temper.schemas.mlff_train_bundle import MLFFTrainBundle
from temper.schemas.train_unit import TrainingUnit
from temper.utils.file_hashing import file_sha256


def _load_test(path: Path, dataset_id: str, *, require_finite: bool = True) -> MLFFTestResult:
    manifest = MLFFExperimentManifest.from_json_file(path).load_results(base_dir=path.parent, require_finite=require_finite)
    return next(result for result in manifest.test_results if result.manifest.dataset_id == dataset_id)

def _outputs(root: Path, bundle: MLFFTrainBundle) -> None:
    assert bundle.files is not None
    path = root / "results.json"
    report = MLFFExperimentManifest.from_json_file(path) if path.exists() else create_result_manifest(bundle, root)
    for index, entry in enumerate(report.test_manifests):
        frames = read(root / entry.execution_input_path, index=":")
        counts = np.array([len(a) for a in frames])
        arrays = {
            "energies": np.array([a.get_potential_energy() for a in frames]) + 2 + 4 * np.arange(len(frames)),
            "forces": np.concatenate([a.get_forces() for a in frames]) + 1,
            "atom_offsets": np.concatenate(([0], np.cumsum(counts))),
            "frame_indices": np.arange(len(frames)),
        }
        if bundle.files.test_stress[index]:
            arrays["stresses"] = np.array([a.get_stress(voigt=False) for a in frames]) + 0.5
        output = root / entry.prediction_path
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output, **arrays)
        entry.status = "completed"
        entry.prediction_sha256 = file_sha256(output)
        entry.wall_time_seconds = 1.0
    report.package_versions = {"numpy": np.__version__}
    report.status = "completed"
    if bundle.unit_type == "finetune":
        report.training_status = "completed"
        report.training_wall_time_seconds = 12.5
        model = root / bundle.files.training_output_files["trained_model"]
        model.parent.mkdir(parents=True, exist_ok=True)
        model.write_bytes(b"final weights")
    report.to_json_file(path)


@pytest.fixture
def saved_result(
        tmp_path: Path,
        finetune_training_unit: TrainingUnit,
        mlff_spec_factory: Callable[..., MLFFSpec],
) -> Path:
    """Create two named test results with known physical errors.

    Parameters
    ----------
    tmp_path : Path
        Isolated test directory.
    finetune_training_unit : TrainingUnit
        Small labeled training unit with own-group and cross-group tests.
    mlff_spec_factory : Callable
        Factory supplying a specification without installing model software.

    Returns
    -------
    Path
        Completed result-manifest path beside its unchanged bundle.json.
    """
    unit = finetune_training_unit
    bundle = MLFFTrainBundle(training_unit=unit, mlff_spec=mlff_spec_factory("mace"))
    root, packaged = bundle.write_submit_folder(tmp_path / "runs" / "attempt")
    _outputs(root, packaged)
    verify_result_manifest(root / "results.json", successful=True)
    return root / "results.json"


def test_original_filenames_and_root_survive_packaging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    finetune_training_unit: TrainingUnit,
    mlff_spec_factory: Callable[..., MLFFSpec],
) -> None:
    """Preserve nested filenames and resolve a relative source root once.

    Parameters
    ----------
    tmp_path : Path
        Isolated directory for source datasets and packaged results.
    monkeypatch : pytest.MonkeyPatch
        Changes the working directory during packaging and loading.
    finetune_training_unit : TrainingUnit
        Source unit with two differently sized test datasets.
    mlff_spec_factory : Callable[..., MLFFSpec]
        Factory providing a specification without model software.
    """
    unit = finetune_training_unit.model_copy(deep=True)
    filenames = ("own group/test.extxyz", "cross group/test.extxyz")
    for old_name, filename in zip(unit.test_sets, filenames, strict=True):
        destination = unit.root_path / unit.domain / filename
        destination.parent.mkdir()
        unit.resolve_local_dataset_path(old_name).rename(destination)
    unit = unit.model_copy(update={"test_sets": filenames, "dataset_sha256": {}})
    original_root = unit.root_path.resolve()
    monkeypatch.chdir(tmp_path)
    unit.root_path = original_root.relative_to(tmp_path)
    bundle = MLFFTrainBundle(
        training_unit=unit,
        mlff_spec=mlff_spec_factory("mace"),
    )
    root, packaged = bundle.write_submit_folder("attempt")
    assert packaged.training_unit.root_path == original_root
    assert not bundle.training_unit.root_path.is_absolute()
    assert packaged.mlff_train_bundle_id == bundle.mlff_train_bundle_id
    assert packaged.files is not None
    assert packaged.files.test_input_files == {
        filename: f"testing_inputs/{filename}" for filename in filenames
    }
    assert packaged.files.test_output_files == {
        "own group/test.extxyz": "testing_outputs/own group/test.npz",
        "cross group/test.extxyz": "testing_outputs/cross group/test.npz",
    }
    _outputs(root, packaged)
    shutil.rmtree(root / "testing_inputs")
    monkeypatch.chdir(root)
    manifest = MLFFExperimentManifest.from_json_file(root / "results.json")
    result = manifest.load_results(base_dir=root)
    assert tuple(test.manifest.dataset_id for test in result.test_results) == filenames
    assert [len(test.frames) for test in result.test_results] == [2, 1]


def test_cleanup_relocation_preserves_named_data(saved_result: Path) -> None:
    """Load named datasets and aligned values after deleting execution inputs.

    Parameters
    ----------
    saved_result : Path
        Completed calculation with two test datasets and training timing.
    """
    manifest = MLFFExperimentManifest.from_json_file(saved_result)
    root = saved_result.parent
    bundle = load_bundle_from_manifest(manifest, root)
    bundle.to_json_file(root.parent / "bundle.json")
    record = JobRecord(bundle_id=str(bundle.mlff_train_bundle_id), attempt_folder_name="attempt", resource={})
    backend = LocalBackend(record, root, lambda: None)
    hashes = backend.collect()
    verify_results(root, hashes, get_download_files(bundle)[1])
    backend.cleanup()
    assert bundle.training_unit.root_path.is_dir()
    assert not (root / "testing_inputs" / "test_own.extxyz").exists()
    assert (root / "bundle.json").is_file()
    bundle.mlff_spec.pretrained_model.artifacts["model"].path.unlink()
    moved = root.parent.with_name("moved")
    root.parent.rename(moved)
    path = moved / "attempt" / "results.json"
    own = _load_test(path, "test_own.extxyz")
    cross = _load_test(path, "test_cross.extxyz")
    assert own.manifest.source_filename == "test_own.extxyz"
    assert cross.manifest.source_filename == "test_cross.extxyz"
    assert MLFFExperimentManifest.from_json_file(path).training_wall_time_seconds == 12.5
    np.testing.assert_allclose(own.predictions["energies"], own.references["energies"] + [2, 6])
    np.testing.assert_allclose(own.predictions["forces"], own.references["forces"] + 1)
    np.testing.assert_allclose(own.predictions["stresses"], own.references["stresses"] + 0.5)
    np.testing.assert_array_equal(own.atom_offsets, [0, 1, 3])
    np.testing.assert_array_equal(own.frame_indices, [0, 1])


@pytest.mark.parametrize("damage",
                         ["reference", "prediction", "offsets", "indices", "units", "filename", "nan", "shape",
                          "version", "properties", "extra_array"])
def test_rejects_invalid_results(saved_result: Path, damage: str) -> None:
    """Reject changed files and scientifically inconsistent output.

    Parameters
    ----------
    saved_result : Path
        Valid result to damage in isolation.
    damage : str
        File change or numerical inconsistency introduced by this case.
    """
    report = json.loads(saved_result.read_text())
    dataset = report["test_manifests"][0]
    root = saved_result.parent
    if damage == "reference":
        saved_manifest = MLFFExperimentManifest.from_json_file(saved_result)
        bundle = load_bundle_from_manifest(saved_manifest, saved_result.parent)
        reference = bundle.training_unit.resolve_local_dataset_path(dataset["source_filename"])
        with reference.open("ab") as stream:
            stream.write(b"changed")
    elif damage == "prediction":
        (root / dataset["prediction_path"]).write_bytes(b"changed")
    elif damage == "filename":
        dataset["source_filename"] = "wrong.extxyz"
    elif damage == "version":
        report["schema_version"] = 999
    elif damage == "units":
        dataset["units"]["energy"] = "Hartree"
    elif damage == "properties":
        dataset["requested_properties"] = ["energy", "forces"]
    else:
        path = root / dataset["prediction_path"]
        with np.load(path) as source:
            arrays = {key: source[key] for key in source.files}
        if damage == "offsets":
            arrays["atom_offsets"][1] = 2
        elif damage == "indices":
            arrays["frame_indices"] = np.array([1, 0])
        elif damage == "nan":
            arrays["energies"][0] = np.nan
        elif damage == "extra_array":
            arrays["_metadata"] = np.asarray("{}")
        else:
            arrays["forces"] = arrays["forces"][:-1]
        np.savez_compressed(path, **arrays)
        dataset["prediction_sha256"] = file_sha256(path)
    write_json_atomically(saved_result, report)
    with pytest.raises(ValueError):
        _load_test(saved_result, "test_own.extxyz")
    if damage == "nan":
        invalid = _load_test(saved_result, "test_own.extxyz", require_finite=False)
        assert np.isnan(invalid.predictions["energies"][0])


def test_shared_bytes_keep_distinct_dataset_names(saved_result: Path) -> None:
    """Keep distinct original filenames even when dataset bytes are equal.

    Parameters
    ----------
    saved_result : Path
        Calculation supplying valid reference data and a packaged bundle.
    """
    saved_manifest = MLFFExperimentManifest.from_json_file(saved_result)
    bundle = load_bundle_from_manifest(saved_manifest, saved_result.parent)
    unit = bundle.training_unit
    own, cross = (unit.resolve_local_dataset_path(name) for name in unit.test_sets)
    shutil.copyfile(own, cross)
    unit = unit.copy_with_refreshed_dataset_hashes()
    replacement = MLFFTrainBundle(training_unit=unit, mlff_spec=bundle.mlff_spec)
    root, packaged = replacement.write_submit_folder(saved_result.parent.parent / "same_bytes")
    _outputs(root, packaged)
    path = root / "results.json"
    datasets = MLFFExperimentManifest.from_json_file(path).test_manifests
    assert datasets[0].reference_sha256 == datasets[1].reference_sha256
    assert datasets[0].execution_input_path != datasets[1].execution_input_path
    assert datasets[0].source_filename != datasets[1].source_filename


def test_missing_manifest_is_not_reconstructed(
    tmp_path: Path, zeroshot_training_unit: TrainingUnit,
    mlff_spec_factory: Callable[..., MLFFSpec],
) -> None:
    """Reject missing runner output without creating replacement metadata.

    Parameters
    ----------
    tmp_path : Path
        Isolated directory for packaged inputs.
    zeroshot_training_unit : TrainingUnit
        Test-only dataset record used to construct an unexecuted bundle.
    mlff_spec_factory : Callable[..., MLFFSpec]
        Creates a model specification with readable dummy weights.
    """
    root, bundle = MLFFTrainBundle(
        training_unit=zeroshot_training_unit,
        mlff_spec=mlff_spec_factory("mace", with_training=False),
    ).write_submit_folder(tmp_path / "run")
    before = {path.relative_to(root): path.read_bytes()
              for path in root.rglob("*") if path.is_file()}
    with pytest.raises(FileNotFoundError, match="results.json"):
        MLFFExperimentManifest.from_json_file(root / "results.json")
    assert not (tmp_path / "reference_datasets").exists()
    assert {path.relative_to(root): path.read_bytes()
            for path in root.rglob("*") if path.is_file()} == before


@pytest.mark.parametrize("status", ["completed", "failed", "cancelled"])
@pytest.mark.parametrize("stale_local_manifest", [False, True])
def test_collection_requires_downloaded_manifest_before_cleanup(
    saved_result: Path, status: str, stale_local_manifest: bool,
) -> None:
    """Reject missing downloaded results and allow a later successful collection.

    Parameters
    ----------
    saved_result : Path
        Experiment supplying readable inputs for the managed attempt.
    status : str
        Terminal computation state reported by the simulated remote backend.
    stale_local_manifest : bool
        Whether an old local results.json remains despite omission from downloads.
    """
    from test_job_management import FakeBackend
    from temper.job_management.manager import JobManager
    from temper.job_management.schemas.resources import RemoteResource

    class MissingManifestBackend(FakeBackend):
        """Simulate a download omitting the runner's manifest."""

        def collect(self) -> dict[str, str]:
            """Return available download hashes, optionally omitting results.json.

            Returns
            -------
            dict[str, str]
                Downloaded paths and checksums. The control flag omits the
                manifest; a local copy can remain to simulate stale output.
            """
            hashes = super().collect()
            if self.control["omit_manifest"]:
                hashes.pop("results.json")
                if not stale_local_manifest:
                    (self.root / "results.json").unlink()
            return hashes

    saved_manifest = MLFFExperimentManifest.from_json_file(saved_result)
    bundle = load_bundle_from_manifest(saved_manifest, saved_result.parent)
    control = {"uploads": 0, "submits": 0, "checks": 0, "downloads": 0,
               "cleanups": 0, "cancels": 0, "status": status, "omit_manifest": True}
    manager = JobManager(
        saved_result.parent.parent / "missing_managed",
        backend_factory=lambda r, p, c: MissingManifestBackend(r, p, c, control),
    )
    resource = RemoteResource(
        name="cluster", backend="Slurm", hostname="example.invalid",
        username="scientist", key_filename="key", remote_root="/scratch/temper",
        environments={"mace": {}},
    )
    record = manager.submit(bundle, resource)
    root = manager.store.path(record)
    manager.synchronize()
    collected = manager.list_attempts()[0]
    assert collected.computation_status == status
    assert collected.collection_status == "pending"
    assert collected.cleanup_status == "pending"
    assert "results.json was not downloaded" in collected.error
    assert control["cleanups"] == 0
    assert (root / "stderr.log").is_file()
    assert (root / "results.json").exists() == stale_local_manifest
    assert not list(root.glob("experiment_*.json"))
    assert not (root / "completed").exists()
    control["omit_manifest"] = False
    manager.synchronize()
    collected = manager.list_attempts()[0]
    assert collected.collection_status == "verified"
    assert collected.cleanup_status == "completed"
    assert collected.error is None
    assert control["cleanups"] == 1


@pytest.mark.parametrize("failure", [None, "training", "evaluation", "interrupt"])
def test_training_time_survives_evaluation_failure(
        saved_result: Path, monkeypatch: pytest.MonkeyPatch, failure: str | None,
) -> None:
    """Measure only successful train calls and preserve time after test failure.

    Parameters
    ----------
    saved_result : Path
        Packaged inputs used by the installed runner.
    monkeypatch : pytest.MonkeyPatch
        Replaces native training, CUDA checks and the duration clock.
    failure : str or None
        Stage to fail, or None for a successful pipeline.
    """
    root = saved_result.parent
    ticks = iter([10.0, 14.25])
    monkeypatch.setattr(runner.time, "perf_counter", lambda: next(ticks))
    monkeypatch.setattr(runner, "require_cuda", lambda: None)

    def train(adapter: MACEAdapter) -> None:
        if failure == "training":
            raise RuntimeError("training failed")
        if failure == "interrupt":
            raise KeyboardInterrupt

    def evaluate(manifest: MLFFExperimentManifest, *, bundle_root: Path) -> None:
        if failure == "evaluation":
            raise RuntimeError("evaluation failed")
        for entry in manifest.test_manifests:
            entry.status = "completed"
            entry.prediction_sha256 = file_sha256(root / entry.prediction_path)

    monkeypatch.setattr(MACEAdapter, "train", train)
    monkeypatch.setattr(runner, "run_test", evaluate)
    if failure is None:
        runner.run_pipeline(root / "bundle.json")
    else:
        with pytest.raises(KeyboardInterrupt if failure == "interrupt" else RuntimeError):
            runner.run_pipeline(root / "bundle.json")
    saved_manifest = MLFFExperimentManifest.from_json_file(saved_result)
    bundle = load_bundle_from_manifest(saved_manifest, saved_result.parent)
    timing = json.loads(saved_result.read_text())
    if failure in {"training", "interrupt"}:
        assert timing["training_status"] == "failed"
        assert timing["training_wall_time_seconds"] is None
    else:
        assert timing["training_status"] == "completed"
        assert timing["training_wall_time_seconds"] == 4.25


def test_zero_shot_without_stress_or_training_time(
        tmp_path: Path, zeroshot_training_unit: TrainingUnit,
        mlff_spec_factory: Callable[..., MLFFSpec], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep absent stress and training timing explicit in loaded results.

    Parameters
    ----------
    tmp_path : Path
        Destination for a standalone zero-shot experiment.
    zeroshot_training_unit : TrainingUnit
        Test-only unit with labeled input files.
    mlff_spec_factory : Callable
        Creates a test-only model specification.
    monkeypatch : pytest.MonkeyPatch
        Replaces CUDA checking and evaluation during the timing check.
    """
    for filename in zeroshot_training_unit.test_sets:
        source = zeroshot_training_unit.resolve_local_dataset_path(filename)
        frames = read(source, index=":")
        for frame in frames:
            frame.calc.results.pop("stress")
        write(source, frames, format="extxyz")
    bundle = MLFFTrainBundle(
        training_unit=zeroshot_training_unit.copy_with_refreshed_dataset_hashes(),
        mlff_spec=mlff_spec_factory("mace", with_training=False),
    )
    root, packaged = bundle.write_submit_folder(tmp_path / "zeroshot")
    monkeypatch.setattr(runner, "require_cuda", lambda: None)
    def evaluate(manifest: MLFFExperimentManifest, *, bundle_root: Path) -> None:
        _outputs(root, packaged)
        manifest.test_manifests = MLFFExperimentManifest.from_json_file(root / "results.json").test_manifests

    monkeypatch.setattr(runner, "run_test", evaluate)
    monkeypatch.setattr(MACEAdapter, "train", lambda *args: pytest.fail("Zero-shot trained"))
    runner.run_pipeline(root / "bundle.json")
    assert not list(root.rglob("training_result.json"))
    _outputs(root, packaged)
    result = _load_test(root / "results.json", "test_own.extxyz")
    assert MLFFExperimentManifest.from_json_file(root / "results.json").training_wall_time_seconds is None
    assert set(result.predictions) == {"energies", "forces"}
    assert set(result.references) == {"energies", "forces"}


def test_missing_references_and_incomplete_results_are_explicit(saved_result: Path) -> None:
    """Reject missing original labels when loading scientific results.

    Parameters
    ----------
    saved_result : Path
        Completed result changed into incomplete and missing-data cases.
    """
    report = json.loads(saved_result.read_text())
    report["status"] = "prepared"
    write_json_atomically(saved_result, report)
    recipe = MLFFExperimentManifest.from_json_file(saved_result)
    saved_manifest = MLFFExperimentManifest.from_json_file(saved_result)
    unit = load_bundle_from_manifest(saved_manifest, saved_result.parent).training_unit
    unit.resolve_local_dataset_path(recipe.test_manifests[0].source_filename).unlink()
    with pytest.raises(ValueError, match="incomplete"):
        verify_result_manifest(saved_result, successful=True)
    with pytest.raises(ValueError, match="does not exist"):
        recipe.load_results(base_dir=saved_result.parent)
    assert load_bundle_from_manifest(recipe, saved_result.parent).training_unit == unit


def test_unknown_versions_and_escaping_paths_are_rejected(saved_result: Path) -> None:
    """Reject unsupported formats and execution paths outside the bundle.

    Parameters
    ----------
    saved_result : Path
        Valid manifest used to exercise reader validation.
    """
    original = json.loads(saved_result.read_text())
    changed = {**original, "schema_version": 999}
    write_json_atomically(saved_result, changed)
    with pytest.raises(ValueError):
        MLFFExperimentManifest.from_json_file(saved_result)
    original["test_manifests"][0]["execution_input_path"] = "../outside.extxyz"
    write_json_atomically(saved_result, original)
    with pytest.raises(ValueError):
        MLFFExperimentManifest.from_json_file(saved_result)


def test_managed_collection_keeps_settings_without_archiving_labels(saved_result: Path) -> None:
    """Collect predictions without copying or reading local reference datasets.

    Parameters
    ----------
    saved_result : Path
        Packaged bundle used to submit an offline simulated remote attempt.
    """
    from test_job_management import FakeBackend
    from temper.job_management.manager import JobManager
    from temper.job_management.schemas.resources import RemoteResource

    saved_manifest = MLFFExperimentManifest.from_json_file(saved_result)
    bundle = load_bundle_from_manifest(saved_manifest, saved_result.parent)
    control = {"uploads": 0, "submits": 0, "checks": 0, "downloads": 0,
               "cleanups": 0, "cancels": 0, "status": "completed"}
    manager = JobManager(saved_result.parent.parent / "managed", backend_factory=lambda r, p, c: FakeBackend(r, p, c, control))
    resource = RemoteResource(name="cluster", backend="Slurm", hostname="example.invalid",
                              username="scientist", key_filename="key", remote_root="/scratch/temper",
                              environments={"mace": {}})
    record = manager.submit(bundle, resource)
    root = manager.store.path(record)
    path = root / "results.json"
    assert not path.exists()
    assert not (root / bundle.files.test_input_files["test_own.extxyz"]).exists()
    reference = bundle.training_unit.resolve_local_dataset_path(bundle.training_unit.test_sets[0])
    original = reference.read_bytes()
    reference.write_bytes(b"corrupt")
    manager.synchronize(download_only=True)
    assert MLFFExperimentManifest.from_json_file(path).status == "completed"
    assert control["cleanups"] == 0
    manager.synchronize()
    assert control["cleanups"] == 1
    assert (root / "bundle.json").is_file()
    assert not (manager.store.root / "reference_datasets").exists()
    assert not list(root.glob("experiment_*.json"))
    # Original labels are the user's responsibility and are checked at analysis.
    with pytest.raises(ValueError, match="checksum"):
        MLFFExperimentManifest.from_json_file(path).load_results(base_dir=root)
    reference.write_bytes(original)


def test_metadata_loading_is_explicit_and_serialization_excludes_arrays(
    saved_result: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep metadata inspection cheap and resolve numerical paths after chdir.

    Parameters
    ----------
    saved_result : Path
        Completed calculation with its original split datasets still available.
    monkeypatch : pytest.MonkeyPatch
        Temporarily prevents array reads and changes the working directory.
    """
    from temper.results_loading import result_loading

    with monkeypatch.context() as patch:
        patch.setattr(result_loading, "read", lambda *a, **k: pytest.fail("Read frames"))
        patch.setattr(np, "load", lambda *a, **k: pytest.fail("Read arrays"))
        manifest = MLFFExperimentManifest.from_json_file(saved_result)
        bundle = load_bundle_from_manifest(manifest, saved_result.parent)
    before = manifest.as_dict()
    assert "test_manifests" in before and "test_results" not in before
    assert isinstance(manifest, MSONableModel)
    assert json.loads(saved_result.read_text(encoding="utf-8")) == manifest.to_dict()
    assert MLFFExperimentManifest.from_dict(before).as_dict() == before
    assert json.loads(manifest.to_json(), cls=MontyDecoder).as_dict() == before
    copy_path = saved_result.with_name("copied_results.json")
    manifest.to_json_file(copy_path)
    restored = MLFFExperimentManifest.from_json_file(copy_path)
    restored.to_json_file(copy_path)
    assert MLFFExperimentManifest.from_json_file(copy_path).as_dict() == before
    assert all(isinstance(t, MLFFTestManifest) for t in manifest.test_manifests)
    assert "bundle" not in before and "result_path" not in before
    monty_path = saved_result.with_name("monty_results.json")
    dumpfn(manifest, monty_path)
    for restored_manifest in (manifest, MLFFExperimentManifest.from_dict(before), loadfn(monty_path)):
        assert restored_manifest == manifest
        assert restored_manifest.training_wall_time_seconds == 12.5
        with pytest.raises(TypeError, match="base_dir"):
            restored_manifest.load_results()
    report_bytes = saved_result.read_bytes()
    assert isinstance(bundle, MLFFTrainBundle)
    assert not hasattr(manifest, "bundle")
    assert manifest.training_wall_time_seconds == 12.5
    assert all(not hasattr(entry, "frames") for entry in manifest.test_manifests)
    monkeypatch.chdir(saved_result.parent.parent.parent)
    loaded = manifest.load_results(base_dir=saved_result.parent)
    assert isinstance(loaded, MLFFExperimentResult)
    assert all(isinstance(test, MLFFTestResult) for test in loaded.test_results)
    assert all(isinstance(test.manifest, MLFFTestManifest) for test in loaded.test_results)
    assert [test.manifest.dataset_id for test in loaded.test_results] == [
        "test_own.extxyz", "test_cross.extxyz",
    ]
    again = manifest.load_results(base_dir=saved_result.parent)
    assert len(loaded.test_results) == 2
    loaded.test_results[0].predictions["energies"][0] = 999
    assert again.test_results[0].predictions["energies"][0] != 999
    assert manifest.as_dict() == before
    assert saved_result.read_bytes() == report_bytes
    report = json.loads(report_bytes)
    report["attempt_name"] = "changed"
    write_json_atomically(saved_result, report)
    assert manifest.load_results(base_dir=saved_result.parent).attempt_name == before["attempt_name"]
    assert MLFFExperimentManifest.from_json_file(saved_result).load_results(base_dir=saved_result.parent).attempt_name == "changed"
    saved_result.unlink()
    assert len(manifest.load_results(base_dir=saved_result.parent).test_results) == 2


@pytest.mark.parametrize("state,duration", [
    ("completed", None), ("failed", 1.0), ("running", 1.0),
    ("completed", -1.0), ("completed", float("nan")),
    ("completed", float("inf")), ("completed", True),
])
def test_manifest_rejects_invalid_training_measurements(
    saved_result: Path, state: str, duration: float | bool | None,
) -> None:
    """Reject incomplete, nonfinite or invalid measurements before saving.

    Parameters
    ----------
    saved_result : Path
        Valid saved model used as a starting point.
    state : str
        Recorded training state to validate.
    duration : float, bool or None
        Deliberately invalid duration for the selected state.
    """
    manifest = MLFFExperimentManifest.from_json_file(saved_result)
    before = saved_result.read_bytes()
    manifest.training_status = state
    manifest.training_wall_time_seconds = duration
    with pytest.raises(ValidationError):
        manifest.to_json_file(saved_result)
    assert saved_result.read_bytes() == before


def test_runner_keeps_finished_tests_when_a_later_dataset_fails(
    saved_result: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Persist training and first-test results before a later evaluation fails.

    Parameters
    ----------
    saved_result : Path
        Packaged fine-tuning inputs used by the real evaluator with a fake calculator.
    monkeypatch : pytest.MonkeyPatch
        Replaces CUDA, native training and calculator construction.
    """
    from temper.mlff.pipeline import evaluate_mlff
    from test_mlff_installed_runtime import FakeCalculator

    root = saved_result.parent
    monkeypatch.setattr(runner, "require_cuda", lambda: None)
    monkeypatch.setattr(MACEAdapter, "train", lambda self: None)
    monkeypatch.setattr(evaluate_mlff, "build_calculator", lambda **kwargs: FakeCalculator())
    evaluate = evaluate_mlff._evaluate_dataset

    def fail_second(**kwargs: object) -> None:
        before = json.loads(saved_result.read_text())
        assert before["training_status"] == "completed"
        assert before["training_wall_time_seconds"] is not None
        test = kwargs["manifest"]
        assert isinstance(test, MLFFTestManifest)
        if test.dataset_id == "test_cross.extxyz":
            assert before["test_manifests"][0]["status"] == "completed"
            assert before["test_manifests"][0]["wall_time_seconds"] is not None
            assert "temper-bench" in before["package_versions"]
            raise RuntimeError("second dataset failed")
        return evaluate(**kwargs)

    monkeypatch.setattr(evaluate_mlff, "_evaluate_dataset", fail_second)
    # Execution and collection must not compose analysis objects or recipes.
    with monkeypatch.context() as patch:
        patch.setattr(MLFFExperimentResult, "__init__", lambda *a, **k: pytest.fail("Runtime data"))
        with pytest.raises(RuntimeError, match="second dataset"):
            runner.run_pipeline(root / "bundle.json")
    manifest = MLFFExperimentManifest.from_json_file(saved_result)
    experiment = manifest.load_results(base_dir=saved_result.parent)
    assert experiment.status == "failed"
    assert experiment.training_status == "completed"
    assert [entry["status"] for entry in json.loads(saved_result.read_text())["test_manifests"]] == ["completed", "failed"]
    assert len(experiment.test_results) == 1
    assert manifest.test_manifests[1].wall_time_seconds is None
    assert manifest.test_manifests[1].prediction_sha256 is None
    assert experiment.test_results[0].predictions is not None
    assert experiment.test_results[0].manifest.dataset_id == "test_own.extxyz"
    assert not list((root / "testing_outputs").glob("*.json"))
    assert not (root / "training_outputs/training_result.json").exists()


def test_root_manifest_is_required_for_successful_collection(saved_result: Path) -> None:
    """Require the root record and retain it even when training or testing fails.

    Parameters
    ----------
    saved_result : Path
        Completed bundle used to check the exact download file selection.
    """
    saved_manifest = MLFFExperimentManifest.from_json_file(saved_result)
    bundle = load_bundle_from_manifest(saved_manifest, saved_result.parent)
    selected, required = get_download_files(bundle)
    assert "results.json" in selected and "results.json" in required
    assert not any(name.endswith(".json") for name in bundle.files.test_output_files.values())
    root = saved_result.parent
    hashes = {name: file_sha256(root / name) for name in required}
    hashes.pop("results.json")
    with pytest.raises(ValueError, match="results.json"):
        verify_results(root, hashes, required)


def test_failed_collection_downloads_partial_manifest_without_rebuilding_results(
    saved_result: Path,
) -> None:
    """Retain a failed runner's manifest byte-for-byte during collection.

    Parameters
    ----------
    saved_result : Path
        Packaged inputs used to simulate an execution machine and its downloads.
    """
    from test_job_management import FakeBackend
    from temper.job_management.manager import JobManager
    from temper.job_management.schemas.resources import RemoteResource

    class PartialBackend(FakeBackend):
        """Simulate a remote calculation with a failed second test."""

        def upload(self) -> None:
            """Capture the result bytes before local copied inputs are removed."""
            super().upload()
            _outputs(self.root, self.bundle)
            report = json.loads((self.root / "results.json").read_text())
            report["status"] = "failed"
            second = report["test_manifests"][1]
            second["status"] = "failed"
            second["prediction_sha256"] = None
            write_json_atomically(self.root / "results.json", report)
            self.control["remote_files"] = {
                name: (self.root / name).read_bytes()
                for name in ("results.json", "bundle.json", report["test_manifests"][0]["prediction_path"])
            }
            (self.root / "results.json").unlink()
            for entry in report["test_manifests"]:
                (self.root / entry["prediction_path"]).unlink()

        def collect(self) -> dict[str, str]:
            """Copy available remote files and return their downloaded checksums.

            Returns
            -------
            dict[str, str]
                Exact relative downloaded names mapped to SHA-256 digests.
            """
            self.control["downloads"] += 1
            for name, content in self.control["remote_files"].items():
                (self.root / name).parent.mkdir(parents=True, exist_ok=True)
                (self.root / name).write_bytes(content)
            return {
                name: file_sha256(self.root / name)
                for name in self.control["remote_files"]
            }

    saved_manifest = MLFFExperimentManifest.from_json_file(saved_result)
    bundle = load_bundle_from_manifest(saved_manifest, saved_result.parent)
    control = {"uploads": 0, "submits": 0, "checks": 0, "downloads": 0,
               "cleanups": 0, "cancels": 0, "status": "failed"}
    manager = JobManager(
        saved_result.parent.parent / "partial_managed",
        backend_factory=lambda r, p, c: PartialBackend(r, p, c, control),
    )
    resource = RemoteResource(
        name="cluster", backend="Slurm", hostname="example.invalid",
        username="scientist", key_filename="key", remote_root="/scratch/temper",
        environments={"mace": {}},
    )
    record = manager.submit(bundle, resource)
    path = manager.store.path(record) / "results.json"
    assert not path.exists()
    manager.synchronize(download_only=True)
    collected = manager.list_attempts()[0]
    assert collected.collection_status == "verified"
    assert control["cleanups"] == 0
    manifest = MLFFExperimentManifest.from_json_file(path).load_results(base_dir=path.parent)
    assert manifest.status == "failed"
    assert manifest.training_wall_time_seconds == 12.5
    assert manifest.test_results[0].manifest.source_filename == "test_own.extxyz"
    assert manifest.test_results[0].predictions is not None
    assert len(manifest.test_results) == 1
    assert path.read_bytes() == control["remote_files"]["results.json"]
    assert collected.downloaded_file_hashes["results.json"] == file_sha256(path)
    manager.synchronize()
    assert control["cleanups"] == 1
    assert control["downloads"] == 1
    assert (path.parent / "bundle.json").read_bytes() == control["remote_files"]["bundle.json"]
    assert not (manager.store.root / "reference_datasets").exists()
    assert not list(path.parent.glob("experiment_*.json"))
