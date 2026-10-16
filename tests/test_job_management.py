"""Offline tests for durable attempts, explicit retries and safe collection."""
from __future__ import annotations

import json
import os
import shutil
import stat
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from collections.abc import Callable

import pytest

from temper.job_management.collection import result_selection, selected_file
from temper.job_management.inputs import input_hashes, verify_packaged_inputs
from temper.job_management.manager import JobManager
from temper.job_management.storage_utils import store_lock
from temper.job_management.records import JobRecord
from temper.schemas.mlff_train_bundle import MLFFTrainBundle
from temper.job_management.resources import RemoteResource
from temper.schemas.train_unit import TrainingUnit
from temper.utils.file_hashing import copy_verified, file_sha256


@pytest.fixture
def recipe(finetune_training_unit: TrainingUnit, mlff_spec_factory: Callable) -> MLFFTrainBundle:
    return MLFFTrainBundle(training_unit=finetune_training_unit.with_dataset_hashes(), mlff_spec=mlff_spec_factory("mace"))


@pytest.fixture
def resource() -> RemoteResource:
    return RemoteResource(name="cluster", backend="Slurm", hostname="example.invalid", username="scientist", key_filename="key", remote_root="/scratch/temper")


class FakeBackend:
    """Simulate persisted remote jobs, including communication failures."""
    def __init__(self, record: JobRecord, root: Path, checkpoint: Callable[[], None], control: dict) -> None:
        self.record = record
        self.root = root
        self.checkpoint = checkpoint
        self.control = control
        self.bundle = MLFFTrainBundle.model_validate_json((root / "bundle.json").read_text(), context={"verify_files": False})

    def upload(self) -> None:
        self.control["uploads"] += 1
        if self.control.get("upload_error"):
            raise ConnectionError("secret-password")
        self.record.dispatcher["uploaded"] = True

    def submit(self) -> None:
        self.control["submits"] += 1
        if self.control.get("power_loss"):
            raise KeyboardInterrupt
        if self.control.get("submit_error"):
            raise ConnectionError("secret-password")
        self.record.dispatcher["job_id"] = "123"

    def status(self) -> str:
        self.control["checks"] += 1
        if self.control.get("status_error"):
            raise ConnectionError("secret-password")
        return self.control["status"]

    def collect(self) -> dict[str, str]:
        self.control["downloads"] += 1
        if self.control.get("download_error"):
            raise ConnectionError("secret-signed-url")
        _, required = result_selection(self.bundle)
        names = ["stderr.log"]
        if self.control["status"] == "success" and not self.control.get("missing_output"):
            names += required
        manifest = {}
        for name in names:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"diagnostics or output")
            manifest[name] = file_sha256(path)
        return manifest

    def cleanup(self) -> None:
        self.control["cleanups"] += 1
        if self.control.get("cleanup_error"):
            raise ConnectionError("secret-password")

    def cancel(self) -> None:
        persisted = JobRecord.model_validate_json((self.root / "job.json").read_text())
        assert persisted.cancel_requested
        self.control["cancels"] += 1

    def recover(self, job_id: str | None = None) -> bool:
        self.record.dispatcher["job_id"] = job_id or "123"
        return True

    def close(self) -> None:
        pass


@pytest.fixture
def managed(tmp_path: Path) -> tuple[JobManager, dict]:
    control = {"uploads": 0, "submits": 0, "checks": 0, "downloads": 0, "cleanups": 0, "cancels": 0, "status": "running"}
    manager = JobManager(tmp_path / "jobs", backend_factory=lambda r, p, c: FakeBackend(r, p, c, control))
    return manager, control


def test_hash_load_detects_same_size_same_timestamp_edit(recipe: MLFFTrainBundle) -> None:
    unit = recipe.training_unit
    data = unit.model_dump_json()
    path = unit.dataset_source(unit.train_set)
    previous = path.stat()
    content = path.read_bytes()
    path.write_bytes(content.replace(b"H", b"C", 1))
    os.utime(path, ns=(previous.st_atime_ns, previous.st_mtime_ns))
    with pytest.raises(ValueError, match="bytes changed"):
        TrainingUnit.model_validate_json(data)
    historical = TrainingUnit.model_validate_json(data, context={"verify_files": False})
    assert historical.training_unit_id == unit.training_unit_id
    rebuilt = historical.with_dataset_hashes()
    assert rebuilt.training_unit_id != unit.training_unit_id


def test_hash_cache_shared_only_within_creation(recipe: MLFFTrainBundle, monkeypatch: pytest.MonkeyPatch) -> None:
    import temper.schemas.train_unit as module
    calls = []
    monkeypatch.setattr(module, "file_sha256", lambda path: calls.append(path) or file_sha256(path))
    cache = {}
    recipe.training_unit.with_dataset_hashes(cache)
    recipe.training_unit.with_dataset_hashes(cache)
    assert len(calls) == len(recipe.training_unit.dataset_sha256)
    TrainingUnit.model_validate_json(recipe.training_unit.model_dump_json())
    assert len(calls) == 2 * len(recipe.training_unit.dataset_sha256)


def test_copy_checks_written_bytes(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.write_bytes(b"old")
    expected = file_sha256(source)
    source.write_bytes(b"new")
    destination = tmp_path / "copied"
    with pytest.raises(ValueError, match="changed after"):
        copy_verified(source, destination, expected)
    assert not destination.exists()
    assert source.read_bytes() == b"new"


def test_remote_input_validation_ignores_original_paths(recipe: MLFFTrainBundle, tmp_path: Path) -> None:
    root, packaged = recipe.write_submit_folder(tmp_path / "attempt")
    for name in recipe.training_unit.dataset_sha256:
        recipe.training_unit.dataset_source(name).unlink()
    loaded = MLFFTrainBundle.model_validate_json((root / "bundle.json").read_text(), context={"verify_files": False})
    verify_packaged_inputs(loaded, root)
    (root / packaged.files.test_input_files["test_000"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="Packaged input changed"):
        verify_packaged_inputs(loaded, root)


def test_readable_attempt_paths_and_no_overwrite(recipe: MLFFTrainBundle, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEFAULT_BUNDLE_WRITE_ROOT", str(tmp_path / "bundles"))
    first, _ = recipe.write_submit_folder()
    second, _ = recipe.write_submit_folder()
    assert first.name == "attempt_001" and second.name == "attempt_002"
    assert first.parent == second.parent
    assert str(recipe.mlff_train_bundle_id) in first.parent.name
    assert "repeat0" in first.parent.name
    with pytest.raises(FileExistsError):
        recipe.write_submit_folder(first)
    with pytest.raises(ValueError, match="mutually exclusive"):
        recipe.write_submit_folder(tmp_path / "exact", bundle_write_root=tmp_path)


def test_nested_in_place_spec_edit_rejected(recipe: MLFFTrainBundle, tmp_path: Path) -> None:
    recipe.mlff_spec.testing_parameters["new_setting"] = True
    with pytest.raises(ValueError, match="does not match"):
        recipe.write_submit_folder(tmp_path / "bad")


def test_success_collection_cleanup_and_duplicate_guard(managed: tuple, recipe: MLFFTrainBundle, resource: RemoteResource) -> None:
    manager, control = managed
    record = manager.submit(recipe, resource)
    root = manager.store.path(record)
    packaged = MLFFTrainBundle.model_validate_json((root / "bundle.json").read_text(), context={"verify_files": False})
    assert all(not (root / name).exists() for name in input_hashes(packaged))
    assert recipe.training_unit.dataset_source(recipe.training_unit.train_set).exists()
    control["status"] = "success"
    result = manager.synchronize()[0]
    assert result.collection_status == "verified" and result.cleanup_status == "complete"
    assert (root / "success").exists()
    with pytest.raises(ValueError, match="already succeeded"):
        manager.submit(recipe, resource)
    assert control["submits"] == 1


def test_failed_jobs_no_automatic_retry_and_bulk_retry_once(managed: tuple, recipe: MLFFTrainBundle, resource: RemoteResource) -> None:
    manager, control = managed
    first = manager.submit(recipe, resource)
    control["status"] = "failed"
    result = manager.synchronize()[0]
    assert result.computation_status == "failed" and result.cleanup_status == "complete"
    assert (manager.store.path(first) / "failed").exists()
    manager.synchronize()
    assert control["submits"] == 1
    assert manager.resubmit(dry_run=True)[0]["action"] == "eligible"
    assert control["submits"] == 1
    assert manager.resubmit()[0]["attempt"].endswith("attempt_002")
    assert control["submits"] == 2
    control["status"] = "success"
    manager.synchronize()
    assert manager.resubmit()[0]["action"] == "skipped"
    assert control["submits"] == 2


@pytest.mark.parametrize("flag", ["download_error", "missing_output"])
def test_incomplete_success_never_cleanup(managed: tuple, recipe: MLFFTrainBundle, resource: RemoteResource, flag: str) -> None:
    manager, control = managed
    record = manager.submit(recipe, resource)
    control.update(status="success", **{flag: True})
    result = manager.synchronize()[0]
    assert result.collection_status == "pending"
    assert control["cleanups"] == 0
    assert not (manager.store.path(record) / "success").exists()
    control[flag] = False
    manager.synchronize()
    assert control["cleanups"] == 1


def test_cleanup_restart_does_not_redownload(managed: tuple, recipe: MLFFTrainBundle, resource: RemoteResource) -> None:
    manager, control = managed
    record = manager.submit(recipe, resource)
    control.update(status="success", cleanup_error=True)
    result = manager.synchronize()[0]
    assert result.collection_status == "verified" and result.cleanup_status == "pending"
    assert (manager.store.path(record) / "success").exists()
    control["cleanup_error"] = False
    restarted = JobManager(manager.store.root, backend_factory=manager._factory)
    assert restarted.synchronize()[0].cleanup_status == "complete"
    assert control["downloads"] == 1 and control["submits"] == 1


def test_refresh_only_has_no_transfer_or_cleanup(managed: tuple, recipe: MLFFTrainBundle, resource: RemoteResource) -> None:
    manager, control = managed
    manager.submit(recipe, resource)
    control["status"] = "success"
    manager.synchronize(refresh_only=True)
    assert control["uploads"] == 1 and control["submits"] == 1
    assert control["downloads"] == control["cleanups"] == 0


def test_lost_query_preserves_running_state_and_redacts_errors(managed: tuple, recipe: MLFFTrainBundle, resource: RemoteResource) -> None:
    manager, control = managed
    record = manager.submit(recipe, resource)
    manager.synchronize()
    control["status_error"] = True
    result = manager.synchronize()[0]
    assert result.computation_status == "running" and result.error
    assert "secret" not in (manager.store.path(record) / "job.json").read_text()
    assert control["submits"] == 1


def test_uncertain_submit_requires_recovery_not_resubmission(managed: tuple, recipe: MLFFTrainBundle, resource: RemoteResource) -> None:
    manager, control = managed
    control["submit_error"] = True
    record = manager.submit(recipe, resource)
    assert record.submission_status == "uncertain"
    manager.synchronize()
    assert control["checks"] == 0
    with pytest.raises(ValueError, match="uncertain"):
        manager.submit(recipe, resource)
    result = manager.recover(record.bundle_id)[0]
    assert result.submission_status == "submitted"
    assert control["submits"] == 1


def test_power_loss_retains_submission_intent(managed: tuple, recipe: MLFFTrainBundle, resource: RemoteResource) -> None:
    manager, control = managed
    control["power_loss"] = True
    with pytest.raises(KeyboardInterrupt):
        manager.submit(recipe, resource)
    record = manager.store.records()[0]
    assert record.submission_status == "submitting"
    assert manager.recover(record.bundle_id)[0].submission_status == "submitted"
    assert control["submits"] == 1


def test_interrupted_upload_retains_copies_and_can_resume(managed: tuple, recipe: MLFFTrainBundle, resource: RemoteResource) -> None:
    manager, control = managed
    control["upload_error"] = True
    record = manager.submit(recipe, resource)
    assert record.submission_status == "uploading"
    assert (manager.store.path(record) / "training_inputs/train.extxyz").exists()
    control["upload_error"] = False
    assert manager.recover(record.bundle_id)[0].submission_status == "submitted"
    assert control["uploads"] == 2 and control["submits"] == 1


def test_cancel_persisted_before_remote_request(managed: tuple, recipe: MLFFTrainBundle, resource: RemoteResource) -> None:
    manager, control = managed
    record = manager.submit(recipe, resource)
    manager.cancel(record.bundle_id)
    assert control["cancels"] == 1
    assert control["cleanups"] == 0
    control["status"] = "failed"
    result = manager.synchronize()[0]
    assert result.computation_status == "cancelled" and result.cleanup_status == "complete"


def test_markers_repaired_from_record(managed: tuple, recipe: MLFFTrainBundle, resource: RemoteResource) -> None:
    manager, control = managed
    record = manager.submit(recipe, resource)
    root = manager.store.path(record)
    (root / "success").touch()
    control["status"] = "failed"
    manager.synchronize()
    assert not (root / "success").exists()
    assert (root / "failed").exists()


def test_store_lock_excludes_second_writer(tmp_path: Path) -> None:
    with store_lock(tmp_path):
        with pytest.raises(OSError):
            with store_lock(tmp_path):
                pytest.fail("A second writer acquired the store lock.")


def test_result_selection_excludes_checkpoints_and_inputs(recipe: MLFFTrainBundle, tmp_path: Path) -> None:
    _, packaged = recipe.write_submit_folder(tmp_path / "bundle")
    patterns, required = result_selection(packaged)
    assert packaged.files.training_output_files["trained_model"] in required
    assert not selected_file("training_outputs/mace/checkpoints/large.pt", patterns)
    assert not selected_file("training_inputs/data/train/coord.npy", patterns)
    assert not selected_file("pretrained_models/mace.model", patterns)
    with pytest.raises(ValueError):
        selected_file("../outside", patterns)


def test_profile_relative_key_and_family_environment(tmp_path: Path) -> None:
    path = tmp_path / "cluster.yaml"
    path.write_text("name: test\nbackend: Slurm\nhostname: cluster\nusername: me\nkey_filename: keys/id_ed25519\nenvironments:\n  mace:\n    conda_init: /opt/conda/etc/profile.d/conda.sh\n    conda_environment: mace\n")
    profile = RemoteResource.from_file(path)
    assert profile.key_filename == str(tmp_path / "keys/id_ed25519")
    assert "conda activate mace" in profile.environments["mace"].shell_lines()


def test_cli_job_parser_defaults_and_alias() -> None:
    from temper.entrypoints.main import main_parser
    parser = main_parser()
    args = parser.parse_args(["job", "synchronize_remote"])
    assert not args.keep_watching and args.interval == 60
    assert parser.parse_args(["job", "synchronize_remote", "--keep_watching"]).keep_watching


def test_bohrium_status_has_no_download_side_effect() -> None:
    pytest.importorskip("temper.job_management.dispatcher")
    from temper.job_management.dispatcher import DispatcherBackend
    backend = DispatcherBackend.__new__(DispatcherBackend)
    backend.profile = RemoteResource(name="cloud", backend="Bohrium", project_id=1, machine_type="gpu")
    backend._job = SimpleNamespace(job_id=42)
    backend.record = SimpleNamespace(observation={})
    detail = {"status": 2, "exitCode": 0, "resultUrl": "must-not-download"}
    backend.submission = SimpleNamespace(machine=SimpleNamespace(job=SimpleNamespace(detail=lambda _: detail)))
    assert backend.status() == "success"
    detail["exitCode"] = 1
    assert backend.status() == "failed"
    detail.pop("exitCode")
    assert backend.status() == "unknown"


def test_dispatcher_roundtrip_uses_actual_dpdispatcher_without_connecting(recipe: MLFFTrainBundle, resource: RemoteResource, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("temper.job_management.dispatcher")
    import temper.job_management.dispatcher as module
    monkeypatch.setattr(module, "_VerifiedSession", lambda profile: SimpleNamespace())
    root, packaged = recipe.write_submit_folder(tmp_path / "bundle")
    record = JobRecord(bundle_id=str(recipe.mlff_train_bundle_id), attempt_folder_name="bundle", resource=resource.model_dump(mode="json"), recipe=recipe.model_dump(mode="json"))
    first = module.DispatcherBackend(record, root, lambda: None)
    assert first.submission.machine.retry_count == 0
    assert len(first.submission.belonging_jobs) == 1
    first._job.job_id = "1234"
    first._snapshot()
    restored = module.DispatcherBackend(JobRecord.model_validate_json(record.model_dump_json()), root, lambda: None)
    assert restored._job.job_id == "1234"
    assert restored.submission.machine.context.remote_root == first.submission.machine.context.remote_root
    assert restored._job.job_hash == first._job.job_hash
    assert restored.submission.belonging_tasks


@pytest.mark.parametrize(("backend_name", "queue_output", "expected"), [
    ("PBS", "Job Id: 123\n    job_state = E\n", "running"),
    ("PBS", "Job Id: 123\n    job_state = F\n", "failed"),
    ("Slurm", "COMPLETING\n", "running"),
    ("Slurm", "PENDING\n", "waiting"),
])
def test_scheduler_exiting_is_not_terminal(backend_name: str, queue_output: str, expected: str, resource: RemoteResource) -> None:
    pytest.importorskip("temper.job_management.dispatcher")
    from temper.job_management.dispatcher import DispatcherBackend
    backend = DispatcherBackend.__new__(DispatcherBackend)
    backend.profile = resource.model_copy(update={"backend": backend_name})
    backend._job = SimpleNamespace(job_id="123")
    backend.record = SimpleNamespace(observation={})
    context = SimpleNamespace(block_call=lambda _: (0, None, BytesIO(queue_output.encode()), BytesIO()))
    backend.submission = SimpleNamespace(machine=SimpleNamespace(context=context, check_finish_tag=lambda _: False))
    assert backend.status() == expected


def test_slurm_empty_queue_uses_accounting(resource: RemoteResource) -> None:
    pytest.importorskip("temper.job_management.dispatcher")
    from temper.job_management.dispatcher import DispatcherBackend
    backend = DispatcherBackend.__new__(DispatcherBackend)
    backend.profile = resource
    backend._job = SimpleNamespace(job_id="123")
    backend.record = SimpleNamespace(observation={})
    calls = []
    def query(command: str) -> tuple:
        calls.append(command)
        text = "" if command.startswith("squeue") else "123|TIMEOUT|0:15|\n"
        return 0, None, BytesIO(text.encode()), BytesIO()
    backend.submission = SimpleNamespace(machine=SimpleNamespace(context=SimpleNamespace(block_call=query), check_finish_tag=lambda _: False))
    assert backend.status() == "failed"
    assert len(calls) == 2 and "sacct" in calls[-1]


def test_bohrium_real_serialization_upload_checkpoint_and_secret_removal(recipe: MLFFTrainBundle, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("bohrium")
    from temper.job_management.dispatcher import DispatcherBackend
    import dpdispatcher.contexts.openapi_context as cloud_context
    monkeypatch.setattr(cloud_context, "DP_CLOUD_SERVER_HOME_DIR", str(tmp_path / "dp-history"))
    monkeypatch.setenv("TEST_CLOUD_KEY", "never-write-this-secret")
    resource = RemoteResource(name="cloud", backend="Bohrium", project_id=123, machine_type="gpu", access_key_env="TEST_CLOUD_KEY", environment={"image": "image:fixed"})
    root, _ = recipe.write_submit_folder(tmp_path / "cloud")
    record = JobRecord(bundle_id=str(recipe.mlff_train_bundle_id), attempt_folder_name="cloud", resource=resource.model_dump(mode="json"), recipe=recipe.model_dump(mode="json"))
    checkpoints = []
    backend = DispatcherBackend(record, root, lambda: checkpoints.append(record.model_dump_json()))
    assert "never-write-this-secret" not in record.model_dump_json()
    assert backend.submission.machine.client.max_retries == 1
    context = backend.submission.machine.context
    monkeypatch.setattr(context.job, "create", lambda **kwargs: {"jobId": 7, "storePath": "store/job7", "token": "upload-secret", "jobGroupId": 2})
    def write_upload(**kwargs: object) -> SimpleNamespace:
        assert "upload_path" in json.loads(checkpoints[-1])["dispatcher"]
        assert kwargs["token"] == "upload-secret"
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"code": 0, "data": {}}, status_code=200)
    monkeypatch.setattr(context.storage, "write", write_upload)
    backend.upload()
    assert checkpoints and all("upload-secret" not in item and "never-write-this-secret" not in item for item in checkpoints)
    params = []
    monkeypatch.setattr(backend.submission.machine.job, "insert", lambda **kwargs: params.append(kwargs) or {"jobId": 7})
    backend.submit()
    assert params[0]["image_name"] == "image:fixed"
    restored = DispatcherBackend(JobRecord.model_validate_json(record.model_dump_json()), root, lambda: None)
    assert restored._job.job_id == 7
    assert restored._job.upload_path == backend._job.upload_path


def test_bohrium_lost_create_reply_does_not_create_again(recipe: MLFFTrainBundle, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("bohrium")
    from temper.job_management.dispatcher import DispatcherBackend
    import dpdispatcher.contexts.openapi_context as cloud_context
    monkeypatch.setattr(cloud_context, "DP_CLOUD_SERVER_HOME_DIR", str(tmp_path / "dp-history"))
    monkeypatch.setenv("BOHRIUM_ACCESS_KEY", "test-key")
    resource = RemoteResource(name="cloud", backend="Bohrium", project_id=123, machine_type="gpu", environment={"image": "image:fixed"})
    root, _ = recipe.write_submit_folder(tmp_path / "cloud")
    record = JobRecord(bundle_id=str(recipe.mlff_train_bundle_id), attempt_folder_name="cloud", resource=resource.model_dump(mode="json"), recipe=recipe.model_dump(mode="json"))
    backend = DispatcherBackend(record, root, lambda: None)
    calls = []
    def create(**kwargs: object) -> dict:
        calls.append(kwargs)
        raise ConnectionError("lost response")
    monkeypatch.setattr(backend.submission.machine.context.job, "create", create)
    with pytest.raises(ConnectionError):
        backend.upload()
    with pytest.raises(RuntimeError, match="creation response was lost"):
        backend.upload()
    assert len(calls) == 1


def test_cancel_interrupted_upload_does_not_launch_on_recovery(managed: tuple, recipe: MLFFTrainBundle, resource: RemoteResource) -> None:
    manager, control = managed
    control["upload_error"] = True
    record = manager.submit(recipe, resource)
    manager.cancel(record.bundle_id)
    manager.recover(record.bundle_id)
    assert control["submits"] == 0
    assert manager.store.records()[0].computation_status == "cancelled"


def test_shared_dataset_batch_loading(recipe: MLFFTrainBundle, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import temper.schemas.train_unit as module
    path = tmp_path / "units.json"
    path.write_text(json.dumps([recipe.training_unit.model_dump(mode="json")] * 3))
    calls = []
    monkeypatch.setattr(module, "file_sha256", lambda path: calls.append(path) or file_sha256(path))
    assert len(module.load_training_units(path)) == 3
    assert len(calls) == len(recipe.training_unit.dataset_sha256)


def test_tampered_local_result_prevents_pending_cleanup(managed: tuple, recipe: MLFFTrainBundle, resource: RemoteResource) -> None:
    manager, control = managed
    record = manager.submit(recipe, resource)
    control.update(status="success", cleanup_error=True)
    result = manager.synchronize()[0]
    root = manager.store.path(record)
    (root / next(iter(result.manifest))).write_bytes(b"edited")
    control["cleanup_error"] = False
    cleanups = control["cleanups"]
    assert manager.synchronize()[0].cleanup_status == "pending"
    assert control["cleanups"] == cleanups


def test_environment_is_frozen_per_family(managed: tuple, recipe: MLFFTrainBundle, resource: RemoteResource) -> None:
    from temper.job_management.resources import RuntimeEnvironment
    manager, _ = managed
    resource.environments["mace"] = RuntimeEnvironment(conda_init="/conda/conda.sh", conda_environment="mace-env")
    record = manager.submit(recipe, resource)
    resource.environments["mace"].conda_environment = "changed-later"
    assert record.resource["environment"]["conda_environment"] == "mace-env"


def test_cli_upgrade_does_not_overwrite_original(recipe: MLFFTrainBundle, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from temper.entrypoints.main import main
    source = tmp_path / "old.json"
    data = recipe.model_dump(mode="json")
    data.pop("mlff_train_bundle_id")
    data["training_unit"].pop("training_unit_id")
    data["training_unit"].pop("dataset_sha256")
    legacy = MLFFTrainBundle.model_validate(data)
    original = legacy.model_dump_json()
    source.write_text(original)
    destination = tmp_path / "new.json"
    with pytest.raises(SystemExit) as result:
        main(["job", "upgrade_bundle", str(source), "--output", str(destination)])
    assert result.value.code == 0
    assert source.read_text() == original
    assert MLFFTrainBundle.model_validate_json(destination.read_text()).training_unit.dataset_sha256


def test_copy_does_not_remove_existing_destination(tmp_path: Path) -> None:
    source, destination = tmp_path / "source", tmp_path / "destination"
    source.write_bytes(b"input")
    destination.write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        copy_verified(source, destination, file_sha256(source))
    assert destination.read_bytes() == b"keep"


def test_actual_ssh_collector_checks_hashes_and_excludes_checkpoints(recipe: MLFFTrainBundle, resource: RemoteResource, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("temper.job_management.dispatcher")
    from temper.job_management.dispatcher import DispatcherBackend
    root, bundle = recipe.write_submit_folder(tmp_path / "local")
    backend = DispatcherBackend.__new__(DispatcherBackend)
    backend.profile, backend.root, backend._bundle = resource, root, bundle
    remote = tmp_path / "remote"
    remote.mkdir()
    (remote / "stderr.log").write_bytes(b"failure diagnosis")
    (remote / "training_outputs/mace/checkpoints").mkdir(parents=True)
    (remote / "training_outputs/mace/checkpoints/large.pt").write_bytes(b"do not transfer")
    def local_path(path: str) -> Path:
        return remote / path.removeprefix("/remote/").removeprefix("/remote")
    def listing(path: str) -> list:
        return [SimpleNamespace(filename=p.name, st_mode=stat.S_IFDIR if p.is_dir() else stat.S_IFREG) for p in local_path(path).iterdir()]
    transfers = []
    def download(source: str, destination: str) -> None:
        transfers.append(source)
        shutil.copyfile(local_path(source), destination)
    context = SimpleNamespace(remote_root="/remote", sftp=SimpleNamespace(listdir_attr=listing, get=download))
    backend.submission = SimpleNamespace(machine=SimpleNamespace(context=context))
    monkeypatch.setattr(backend, "_command", lambda _: file_sha256(remote / "stderr.log") + "  stderr.log")
    manifest = backend.collect()
    assert manifest == {"stderr.log": file_sha256(remote / "stderr.log")}
    assert transfers == ["/remote/stderr.log"]
    assert not list(root.rglob("*.partial"))


def test_ssh_cleanup_requires_matching_ownership(resource: RemoteResource, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("temper.job_management.dispatcher")
    from temper.job_management.dispatcher import DispatcherBackend
    backend = DispatcherBackend.__new__(DispatcherBackend)
    backend.profile = resource
    backend.record = SimpleNamespace(dispatcher={"token": "a" * 32})
    context = SimpleNamespace(remote_root="/scratch/temper/temper_" + "a" * 32, read_file=lambda _: "different-owner")
    backend.submission = SimpleNamespace(machine=SimpleNamespace(context=context))
    calls = []
    monkeypatch.setattr(backend, "_command", lambda command: calls.append(command))
    with pytest.raises(ValueError, match="ownership mismatch"):
        backend.cleanup()
    assert not calls


def test_zip_collection_rejects_traversal_without_writing(recipe: MLFFTrainBundle, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("temper.job_management.dispatcher")
    from zipfile import ZipFile
    import requests
    from temper.job_management.dispatcher import DispatcherBackend
    root, bundle = recipe.write_submit_folder(tmp_path / "local")
    archive = BytesIO()
    with ZipFile(archive, "w") as stream:
        stream.writestr("../stderr.log", b"escape")
    class Response:
        def __enter__(self) -> "Response":
            return self
        def __exit__(self, *args: object) -> None:
            pass
        def raise_for_status(self) -> None:
            pass
        def iter_content(self, size: int) -> list[bytes]:
            return [archive.getvalue()]
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: Response())
    backend = DispatcherBackend.__new__(DispatcherBackend)
    backend.profile = RemoteResource(name="cloud", backend="Bohrium", project_id=1, machine_type="gpu")
    backend.root, backend._bundle, backend._job = root, bundle, SimpleNamespace(job_id=1)
    backend.submission = SimpleNamespace(machine=SimpleNamespace(context=object(), job=SimpleNamespace(detail=lambda _: {"resultUrl": "https://example.invalid/results"})))
    with pytest.raises(ValueError, match="Unsafe remote result path"):
        backend.collect()
    assert not (tmp_path / "stderr.log").exists()
