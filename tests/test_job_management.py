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

from temper.job_management.collection import get_download_files, selected_file
from temper.job_management.input_utils import (
    get_dataset_and_model_hashes,
    load_bundle,
    verify_dataset_and_model_hashes,
)
from temper.job_management.manager import JobManager
from temper.job_management.paths import bundle_root
from temper.job_management.storage_utils import store_lock
from temper.job_management.records import JobRecord
from temper.schemas.mlff_train_bundle import MLFFTrainBundle
from temper.job_management.resources import RemoteResource
from temper.schemas.train_unit import TrainingUnit
from temper.utils.file_hashing import copy_verified, file_sha256


@pytest.fixture
def source_bundle(
    finetune_training_unit: TrainingUnit, mlff_spec_factory: Callable
) -> MLFFTrainBundle:
    return MLFFTrainBundle(
        training_unit=finetune_training_unit.with_dataset_hashes(),
        mlff_spec=mlff_spec_factory("mace"),
    )


@pytest.fixture
def resource() -> RemoteResource:
    return RemoteResource(
        name="cluster",
        backend="Slurm",
        hostname="example.invalid",
        username="scientist",
        key_filename="key",
        remote_root="/scratch/temper",
        environments={"mace": {}},
    )


class FakeBackend:
    """Simulate persisted remote jobs, including communication failures."""

    def __init__(
        self,
        record: JobRecord,
        root: Path,
        checkpoint: Callable[[], None],
        control: dict,
    ) -> None:
        self.record = record
        self.root = root
        self.checkpoint = checkpoint
        self.control = control
        self.bundle = load_bundle(record, root, require_packaged=True)

    def upload(self) -> None:
        self.control["uploads"] += 1
        if self.control.get("upload_error"):
            raise ConnectionError("secret-password")
        self.record.dispatcher["uploaded"] = True

    def submit(self) -> None:
        self.control["submits"] += 1
        if self.control.get("rejected"):
            from temper.job_management.errors import SubmissionRejected

            self.record.dispatcher["submit_result"] = {
                "exit_status": 1,
                "stdout": "",
                "stderr": "invalid resources",
            }
            raise SubmissionRejected("invalid resources")
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
        _, required = get_download_files(self.bundle)
        names = ["stderr.log"]
        if self.control["status"] == "success" and not self.control.get(
            "missing_output"
        ):
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
        persisted = JobRecord.model_validate_json(
            (self.root / "job.json").read_text()
        )
        assert persisted.cancel_requested
        self.control["cancels"] += 1

    def recover(self, job_id: str | None = None) -> bool:
        self.record.dispatcher["job_id"] = job_id or "123"
        return True

    def close(self) -> None:
        pass


@pytest.fixture
def managed(tmp_path: Path) -> tuple[JobManager, dict]:
    control = {
        "uploads": 0,
        "submits": 0,
        "checks": 0,
        "downloads": 0,
        "cleanups": 0,
        "cancels": 0,
        "status": "running",
    }
    manager = JobManager(
        tmp_path / "jobs",
        backend_factory=lambda r, p, c: FakeBackend(r, p, c, control),
    )
    return manager, control


def test_rejected_submission_can_retry_with_new_resources(
    managed: tuple[JobManager, dict],
    source_bundle: MLFFTrainBundle,
    resource: RemoteResource,
) -> None:
    """Retire a confirmed rejection under the store lock and preserve
    history."""
    manager, control = managed
    control["rejected"] = True
    record = manager.submit(source_bundle, resource)
    assert record.submission_status == "rejected"
    control["rejected"] = False
    corrected = resource.model_copy(
        update={"resources": {"tasks_per_node": 1, "cpus_per_task": 4}}
    )
    outcomes = manager.resubmit(replacement_resource=corrected)
    assert outcomes[0]["action"] == "submitted"
    new = manager.list_attempts(bundle_id=record.bundle_id)[-1]
    assert new.submission_status == "submitted"
    assert new.attempt_folder_name != record.attempt_folder_name
    old = manager.list_attempts(bundle_id=record.bundle_id)[0]
    assert old.cleanup_status == "complete"
    assert old.computation_status == "unknown"
    assert old.dispatcher["submit_result"]["stderr"] == "invalid resources"
    assert control["submits"] == 2


def test_uncertain_submission_cannot_retry(
    managed: tuple[JobManager, dict],
    source_bundle: MLFFTrainBundle,
    resource: RemoteResource,
) -> None:
    """A lost response cannot authorize another launch."""
    manager, control = managed
    control["submit_error"] = True
    record = manager.submit(source_bundle, resource)
    assert record.submission_status == "uncertain"
    outcomes = manager.resubmit()
    assert outcomes[0]["action"] == "skipped"
    assert control["submits"] == 1


def test_rejection_cleanup_failure_does_not_launch(
    managed: tuple[JobManager, dict],
    source_bundle: MLFFTrainBundle,
    resource: RemoteResource,
) -> None:
    """Cleanup failure retains the rejection and allows a later retry."""
    manager, control = managed
    control["rejected"] = True
    first = manager.submit(source_bundle, resource)
    control.update(rejected=False, cleanup_error=True)
    assert manager.resubmit()[0]["action"] == "error"
    assert control["submits"] == 1
    saved = manager.list_attempts()[0]
    assert saved.submission_status == "rejected"
    assert saved.cleanup_status == "pending"
    assert saved.error is not None
    events = (manager.store.path(saved) / "events.jsonl").read_text()
    assert json.loads(events.splitlines()[-1])["event"] == "operation_error"
    control["cleanup_error"] = False
    assert manager.resubmit()[0]["action"] == "submitted"
    assert manager.list_attempts()[0].dispatcher == first.dispatcher


def test_rejected_bulk_retry_dry_run_and_latest_resource_filter(
    managed: tuple[JobManager, dict],
    source_bundle: MLFFTrainBundle,
    resource: RemoteResource,
) -> None:
    """Discover rejected bundles, honor filters, and never retry stale
    history."""
    manager, control = managed
    control["rejected"] = True
    first = manager.submit(source_bundle, resource)
    data = source_bundle.model_dump()
    data.pop("mlff_train_bundle_id", None)
    data["mlff_spec"].pop("mlff_spec_id", None)
    data["mlff_spec"]["training_parameters"]["max_num_epochs"] += 1
    second = MLFFTrainBundle.model_validate(data)
    manager.submit(second, resource)
    preview = manager.resubmit(dry_run=True)
    assert len(preview) == 2
    assert all(item["action"] == "eligible" for item in preview)
    assert control["cleanups"] == 0 and control["submits"] == 2
    control["rejected"] = False
    replacement = resource.model_copy(update={"name": "corrected"})
    outcomes = manager.resubmit(replacement_resource=replacement)
    assert len(outcomes) == 2
    assert all(item["action"] == "submitted" for item in outcomes)
    assert control["submits"] == 4
    skipped = manager.resubmit(resource=resource.name)
    assert all(item["action"] == "skipped" for item in skipped)
    assert all("newer attempt" in item["reason"] for item in skipped)
    assert (
        manager.list_attempts(bundle_id=first.bundle_id)[0].computation_status
        == "unknown"
    )


def test_hash_load_detects_same_size_same_timestamp_edit(
    source_bundle: MLFFTrainBundle,
) -> None:
    unit = source_bundle.training_unit
    data = unit.model_dump_json()
    path = unit.dataset_source(unit.train_set)
    previous = path.stat()
    content = path.read_bytes()
    path.write_bytes(content.replace(b"H", b"C", 1))
    os.utime(path, ns=(previous.st_atime_ns, previous.st_mtime_ns))
    with pytest.raises(ValueError, match="bytes changed"):
        TrainingUnit.model_validate_json(data)
    historical = TrainingUnit.model_validate_json(
        data, context={"verify_files": False}
    )
    assert historical.training_unit_id == unit.training_unit_id
    rebuilt = historical.with_dataset_hashes()
    assert rebuilt.training_unit_id != unit.training_unit_id


def test_hash_cache_shared_only_within_creation(
    source_bundle: MLFFTrainBundle, monkeypatch: pytest.MonkeyPatch
) -> None:
    import temper.schemas.train_unit as module

    calls = []
    monkeypatch.setattr(
        module,
        "file_sha256",
        lambda path: calls.append(path) or file_sha256(path),
    )
    cache = {}
    source_bundle.training_unit.with_dataset_hashes(cache)
    source_bundle.training_unit.with_dataset_hashes(cache)
    assert len(calls) == len(source_bundle.training_unit.dataset_sha256)
    TrainingUnit.model_validate_json(
        source_bundle.training_unit.model_dump_json()
    )
    assert len(calls) == 2 * len(source_bundle.training_unit.dataset_sha256)


def test_load_bundle_packaging_and_dataset_checks_are_independent(
    source_bundle: MLFFTrainBundle,
    tmp_path: Path,
) -> None:
    """Allow inspection while requiring packaging or source verification on
    request.

    Parameters
    ----------
    source_bundle : MLFFTrainBundle
        Unpackaged bundle with verified dataset hashes.
    tmp_path : Path
        Temporary directory for the shared bundle and attempt.
    """
    root = tmp_path / "attempt"
    root.mkdir()
    (tmp_path / "bundle.json").write_text(
        source_bundle.model_dump_json(), encoding="utf-8"
    )
    record = JobRecord(
        bundle_id=str(source_bundle.mlff_train_bundle_id),
        attempt_folder_name="attempt",
        resource={},
    )
    assert load_bundle(record, root).files is None
    assert load_bundle(record, root, verify_files=True).files is None
    with pytest.raises(ValueError, match="not packaged"):
        load_bundle(record, root, require_packaged=True)
    for name in source_bundle.training_unit.dataset_sha256:
        source_bundle.training_unit.dataset_source(name).unlink()
    assert (
        load_bundle(record, root).mlff_train_bundle_id
        == source_bundle.mlff_train_bundle_id
    )
    with pytest.raises(OSError):
        load_bundle(record, root, verify_files=True)


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


def test_remote_input_validation_ignores_original_paths(
    source_bundle: MLFFTrainBundle, tmp_path: Path
) -> None:
    root, packaged = source_bundle.write_submit_folder(tmp_path / "attempt")
    for name in source_bundle.training_unit.dataset_sha256:
        source_bundle.training_unit.dataset_source(name).unlink()
    loaded = MLFFTrainBundle.model_validate_json(
        (root / "bundle.json").read_text(), context={"verify_files": False}
    )
    verify_dataset_and_model_hashes(loaded, root)
    (root / packaged.files.test_input_files["test_000"]).write_bytes(
        b"changed"
    )
    with pytest.raises(ValueError, match="Packaged input changed"):
        verify_dataset_and_model_hashes(loaded, root)


def test_readable_attempt_paths_and_no_overwrite(
    source_bundle: MLFFTrainBundle,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "temper.job_management.paths.DEFAULT_BUNDLE_WRITE_ROOT",
        str(tmp_path / "bundles"),
    )
    first, _ = source_bundle.write_submit_folder()
    second, _ = source_bundle.write_submit_folder()
    assert first.name == "attempt_001" and second.name == "attempt_002"
    assert first.parent == second.parent
    assert str(source_bundle.mlff_train_bundle_id) in first.parent.name
    assert "repeat0" in first.parent.name
    with pytest.raises(FileExistsError):
        source_bundle.write_submit_folder(first)
    with pytest.raises(ValueError, match="mutually exclusive"):
        source_bundle.write_submit_folder(
            tmp_path / "exact", bundle_write_root=tmp_path
        )


def test_nested_in_place_spec_edit_rejected(
    source_bundle: MLFFTrainBundle, tmp_path: Path
) -> None:
    source_bundle.mlff_spec.testing_parameters["new_setting"] = True
    with pytest.raises(ValueError, match="does not match"):
        source_bundle.write_submit_folder(tmp_path / "bad")


def test_success_collection_cleanup_and_duplicate_guard(
    managed: tuple, source_bundle: MLFFTrainBundle, resource: RemoteResource
) -> None:
    manager, control = managed
    record = manager.submit(source_bundle, resource)
    root = manager.store.path(record)
    packaged = load_bundle(record, root, require_packaged=True)
    assert not (root / "bundle.json").exists()
    assert all(
        not (root / name).exists()
        for name in get_dataset_and_model_hashes(packaged)
    )
    assert source_bundle.training_unit.dataset_source(
        source_bundle.training_unit.train_set
    ).exists()
    control["status"] = "success"
    result = manager.synchronize()[0]
    assert (
        result.collection_status == "verified"
        and result.cleanup_status == "complete"
    )
    assert (root / "success").exists()
    with pytest.raises(ValueError, match="already succeeded"):
        manager.submit(source_bundle, resource)
    assert control["submits"] == 1


@pytest.mark.parametrize("upload_fails", [False, True])
def test_shared_bundle_is_packaged_before_upload(
    managed: tuple[JobManager, dict[str, object]],
    source_bundle: MLFFTrainBundle,
    resource: RemoteResource,
    monkeypatch: pytest.MonkeyPatch,
    upload_fails: bool,
) -> None:
    """Save packaged locations before upload and retain copies on upload
    failure.

    Parameters
    ----------
    managed : tuple[JobManager, dict[str, object]]
        Manager with a simulated backend and its operation controls.
    source_bundle : MLFFTrainBundle
        Unpackaged bundle with accessible datasets and model weights.
    resource : RemoteResource
        Remote settings used by the simulated backend.
    monkeypatch : pytest.MonkeyPatch
        Replace upload with an observation of the files available at
        that time.
    upload_fails : bool
        Whether upload fails before submission can be confirmed.
    """
    from temper.job_management.paths import bundle_directory

    manager, control = managed
    shared = (
        bundle_directory(manager.store.root, source_bundle) / "bundle.json"
    )
    shared.parent.mkdir(parents=True)
    shared.write_text(source_bundle.model_dump_json(), encoding="utf-8")
    observed: list[tuple[dict[str, object], dict[str, object]]] = []
    original_upload = FakeBackend.upload

    def observe_upload(backend: FakeBackend) -> None:
        observed.append(
            (
                json.loads(shared.read_text(encoding="utf-8")),
                json.loads(
                    (backend.root / "bundle.json").read_text(encoding="utf-8")
                ),
            )
        )
        original_upload(backend)

    monkeypatch.setattr(FakeBackend, "upload", observe_upload)
    control["upload_error"] = upload_fails
    record = manager.submit(source_bundle, resource)
    assert len(observed) == 1
    shared_data, attempt_data = observed[0]
    assert shared_data == attempt_data
    assert shared_data["files"] is not None
    assert json.loads(shared.read_text(encoding="utf-8")) == shared_data
    assert source_bundle.files is None
    assert (
        manager.store.path(record) / "bundle.json"
    ).exists() is upload_fails
    assert record.submission_status == (
        "uploading" if upload_fails else "submitted"
    )
    assert control["submits"] == (0 if upload_fails else 1)


def test_local_bundle_remains_until_completion_cleanup(
    managed: tuple[JobManager, dict[str, object]],
    source_bundle: MLFFTrainBundle,
) -> None:
    """Keep the local runner's bundle until its results have been collected.

    Parameters
    ----------
    managed : tuple[JobManager, dict[str, object]]
        Manager with a simulated backend and its operation controls.
    source_bundle : MLFFTrainBundle
        Bundle with accessible datasets and model weights.
    """
    manager, control = managed
    profile = RemoteResource(
        name="local", backend="Local", environments={"mace": {}}
    )
    record = manager.submit(source_bundle, profile)
    path = manager.store.path(record) / "bundle.json"
    assert path.is_file()
    manager.synchronize(refresh_only=True)
    assert path.is_file()
    control["status"] = "success"
    manager.synchronize(download_only=True)
    assert path.is_file()
    assert manager.synchronize()[0].cleanup_status == "complete"
    assert not path.exists()
    assert manager.synchronize()[0].error is None


def test_collection_uses_shared_file_locations_without_source_datasets(
    managed: tuple[JobManager, dict[str, object]],
    source_bundle: MLFFTrainBundle,
    resource: RemoteResource,
) -> None:
    """Collect results using only the shared bundle after local inputs are
    removed.

    Parameters
    ----------
    managed : tuple[JobManager, dict[str, object]]
        Manager with a simulated backend and its operation controls.
    source_bundle : MLFFTrainBundle
        Bundle whose original datasets will be removed after submission.
    resource : RemoteResource
        Remote execution settings for the simulated submission.
    """
    manager, control = managed
    record = manager.submit(source_bundle, resource)
    path = manager.store.path(record) / "bundle.json"
    manager.synchronize(refresh_only=True)
    assert not path.exists()
    assert (
        load_bundle(record, path.parent, require_packaged=True).files
        is not None
    )
    saved = json.loads((path.parent / "job.json").read_text())
    assert "source_bundle" not in saved
    assert "packaged_files" not in saved
    assert saved["relative_source_bundle_path"] == "../bundle.json"
    for name in source_bundle.training_unit.dataset_sha256:
        source_bundle.training_unit.dataset_source(name).unlink()
    control["status"] = "success"
    assert manager.synchronize()[0].cleanup_status == "complete"


def test_attempt_bundle_reference_survives_store_move(
    managed: tuple[JobManager, dict[str, object]],
    source_bundle: MLFFTrainBundle,
    resource: RemoteResource,
    tmp_path: Path,
) -> None:
    """Keep one shared bundle and resolve it after relocating the job
    directory.

    Parameters
    ----------
    managed : tuple[JobManager, dict[str, object]]
        Manager with a simulated backend and its operation controls.
    source_bundle : MLFFTrainBundle
        Source bundle for the submission.
    resource : RemoteResource
        Remote settings used by the simulated backend.
    tmp_path : Path
        Temporary directory containing the relocated store.
    """
    manager, _ = managed
    record = manager.submit(source_bundle, resource)
    root = manager.store.path(record)
    saved = json.loads((root / "job.json").read_text())
    assert "source_bundle" not in saved
    assert saved["relative_source_bundle_path"] == "../bundle.json"
    assert (root.parent / "bundle.json").exists()
    assert not (root / "bundle.json").exists()
    moved = bundle_root(tmp_path / "moved")
    shutil.copytree(manager.store.root, moved)
    moved_root = moved / record.attempt_folder_name
    loaded = load_bundle(record, moved_root, require_packaged=True)
    assert str(loaded.mlff_train_bundle_id) == record.bundle_id
    assert loaded.files is not None
    assert "packaged_files" not in saved


def test_referenced_bundle_is_required_and_cannot_change_identity(
    managed: tuple[JobManager, dict[str, object]],
    source_bundle: MLFFTrainBundle,
    resource: RemoteResource,
) -> None:
    """Reject missing or changed shared bundles and protect them from CLI
    refresh.

    Parameters
    ----------
    managed : tuple[JobManager, dict[str, object]]
        Manager with a simulated backend and its operation controls.
    source_bundle : MLFFTrainBundle
        Original bundle for the submission.
    resource : RemoteResource
        Execution settings for the simulated backend.
    """
    from temper.entrypoints.job import job_cli
    from temper.entrypoints.main import main_parser

    manager, _ = managed
    record = manager.submit(source_bundle, resource)
    root = manager.store.path(record)
    shared = root.parent / "bundle.json"
    original = shared.read_bytes()
    args = main_parser().parse_args(
        ["job", "refresh_bundle_identity", str(shared)]
    )
    with pytest.raises(ValueError, match="existing attempts"):
        job_cli(args)
    assert shared.read_bytes() == original
    changed = source_bundle.model_copy(
        update={
            "training_unit": source_bundle.training_unit.model_copy(
                update={"repeat_id": 999}
            ),
        }
    )
    shared.write_text(changed.model_dump_json(), encoding="utf-8")
    with pytest.raises(ValueError, match="bundle ID"):
        load_bundle(record, root, require_packaged=True)
    shared.unlink()
    with pytest.raises(OSError):
        load_bundle(record, root, require_packaged=True)


def test_failed_jobs_no_automatic_retry_and_bulk_retry_once(
    managed: tuple, source_bundle: MLFFTrainBundle, resource: RemoteResource
) -> None:
    manager, control = managed
    first = manager.submit(source_bundle, resource)
    control["status"] = "failed"
    result = manager.synchronize()[0]
    assert (
        result.computation_status == "failed"
        and result.cleanup_status == "complete"
    )
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
def test_incomplete_success_never_cleanup(
    managed: tuple,
    source_bundle: MLFFTrainBundle,
    resource: RemoteResource,
    flag: str,
) -> None:
    manager, control = managed
    record = manager.submit(source_bundle, resource)
    control.update(status="success", **{flag: True})
    result = manager.synchronize()[0]
    assert result.collection_status == "pending"
    assert control["cleanups"] == 0
    assert not (manager.store.path(record) / "success").exists()
    control[flag] = False
    manager.synchronize()
    assert control["cleanups"] == 1


def test_cleanup_restart_does_not_redownload(
    managed: tuple, source_bundle: MLFFTrainBundle, resource: RemoteResource
) -> None:
    manager, control = managed
    record = manager.submit(source_bundle, resource)
    control.update(status="success", cleanup_error=True)
    result = manager.synchronize()[0]
    assert (
        result.collection_status == "verified"
        and result.cleanup_status == "pending"
    )
    assert (manager.store.path(record) / "success").exists()
    control["cleanup_error"] = False
    restarted = JobManager(
        manager.store.root, backend_factory=manager._factory
    )
    assert restarted.synchronize()[0].cleanup_status == "complete"
    assert control["downloads"] == 1 and control["submits"] == 1


def test_refresh_only_has_no_transfer_or_cleanup(
    managed: tuple, source_bundle: MLFFTrainBundle, resource: RemoteResource
) -> None:
    manager, control = managed
    manager.submit(source_bundle, resource)
    control["status"] = "success"
    manager.synchronize(refresh_only=True)
    assert control["uploads"] == 1 and control["submits"] == 1
    assert control["downloads"] == control["cleanups"] == 0


def test_lost_query_preserves_running_state_and_redacts_errors(
    managed: tuple, source_bundle: MLFFTrainBundle, resource: RemoteResource
) -> None:
    manager, control = managed
    record = manager.submit(source_bundle, resource)
    manager.synchronize()
    control["status_error"] = True
    result = manager.synchronize()[0]
    assert result.computation_status == "running" and result.error
    assert (
        "secret" not in (manager.store.path(record) / "job.json").read_text()
    )
    assert control["submits"] == 1


def test_uncertain_submit_requires_recovery_not_resubmission(
    managed: tuple, source_bundle: MLFFTrainBundle, resource: RemoteResource
) -> None:
    manager, control = managed
    control["submit_error"] = True
    record = manager.submit(source_bundle, resource)
    assert record.submission_status == "uncertain"
    assert (manager.store.path(record) / "bundle.json").exists()
    manager.synchronize()
    assert control["checks"] == 0
    with pytest.raises(ValueError, match="uncertain"):
        manager.submit(source_bundle, resource)
    result = manager.recover(record.bundle_id)[0]
    assert result.submission_status == "submitted"
    assert not (manager.store.path(record) / "bundle.json").exists()
    assert control["submits"] == 1


def test_power_loss_retains_submission_intent(
    managed: tuple, source_bundle: MLFFTrainBundle, resource: RemoteResource
) -> None:
    manager, control = managed
    control["power_loss"] = True
    with pytest.raises(KeyboardInterrupt):
        manager.submit(source_bundle, resource)
    record = manager.store.records()[0]
    assert record.submission_status == "submitting"
    assert (
        manager.recover(record.bundle_id)[0].submission_status == "submitted"
    )
    assert control["submits"] == 1


def test_interrupted_upload_retains_copies_and_can_resume(
    managed: tuple, source_bundle: MLFFTrainBundle, resource: RemoteResource
) -> None:
    manager, control = managed
    control["upload_error"] = True
    record = manager.submit(source_bundle, resource)
    assert record.submission_status == "uploading"
    assert (
        manager.store.path(record) / "training_inputs/train.extxyz"
    ).exists()
    control["upload_error"] = False
    assert (
        manager.recover(record.bundle_id)[0].submission_status == "submitted"
    )
    assert control["uploads"] == 2 and control["submits"] == 1


def test_cancel_persisted_before_remote_request(
    managed: tuple, source_bundle: MLFFTrainBundle, resource: RemoteResource
) -> None:
    manager, control = managed
    record = manager.submit(source_bundle, resource)
    manager.cancel(record.bundle_id)
    assert control["cancels"] == 1
    assert control["cleanups"] == 0
    control["status"] = "failed"
    result = manager.synchronize()[0]
    assert (
        result.computation_status == "cancelled"
        and result.cleanup_status == "complete"
    )


def test_markers_repaired_from_record(
    managed: tuple, source_bundle: MLFFTrainBundle, resource: RemoteResource
) -> None:
    manager, control = managed
    record = manager.submit(source_bundle, resource)
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


def test_get_download_files_excludes_checkpoints_and_inputs(
    source_bundle: MLFFTrainBundle, tmp_path: Path
) -> None:
    _, packaged = source_bundle.write_submit_folder(tmp_path / "bundle")
    patterns, required = get_download_files(packaged)
    assert packaged.files.training_output_files["trained_model"] in required
    assert not selected_file(
        "training_outputs/mace/checkpoints/large.pt", patterns
    )
    assert not selected_file("training_inputs/data/train/coord.npy", patterns)
    assert not selected_file("pretrained_models/mace.model", patterns)
    with pytest.raises(ValueError):
        selected_file("../outside", patterns)


def test_profile_relative_key_and_family_environment(tmp_path: Path) -> None:
    path = tmp_path / "cluster.yaml"
    path.write_text(
        "name: test\nbackend: Slurm\nhostname: cluster\nusername:"
        " me\nkey_filename: keys/id_ed25519\nenvironments:\n  mace:\n   "
        " conda_init: /opt/conda/etc/profile.d/conda.sh\n   "
        " conda_environment: mace\n"
    )
    profile = RemoteResource.from_file(path)
    assert profile.key_filename == str(tmp_path / "keys/id_ed25519")
    assert "conda activate mace" in profile.environments["mace"].shell_lines()


def test_cli_job_parser_defaults_and_alias() -> None:
    from temper.entrypoints.main import main_parser

    parser = main_parser()
    args = parser.parse_args(["job", "synchronize_remote"])
    assert not args.keep_watching and args.interval == 60
    assert parser.parse_args(
        ["job", "synchronize_remote", "--keep_watching"]
    ).keep_watching


def test_bohrium_status_has_no_download_side_effect() -> None:
    pytest.importorskip("temper.job_management.dispatcher")
    from temper.job_management.dispatcher import DispatcherBackend

    backend = DispatcherBackend.__new__(DispatcherBackend)
    backend.profile = RemoteResource(
        name="cloud", backend="Bohrium", project_id=1, machine_type="gpu"
    )
    backend._job = SimpleNamespace(job_id=42)
    backend.record = SimpleNamespace(observation={})
    detail = {"status": 2, "exitCode": 0, "resultUrl": "must-not-download"}
    backend.submission = SimpleNamespace(
        machine=SimpleNamespace(job=SimpleNamespace(detail=lambda _: detail))
    )
    assert backend.status() == "success"
    detail["exitCode"] = 1
    assert backend.status() == "failed"
    detail.pop("exitCode")
    assert backend.status() == "unknown"


def test_dispatcher_roundtrip_uses_actual_dpdispatcher_without_connecting(
    source_bundle: MLFFTrainBundle,
    resource: RemoteResource,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("temper.job_management.dispatcher")
    import temper.job_management.dispatcher as module

    monkeypatch.setattr(
        module, "_VerifiedSession", lambda profile: SimpleNamespace()
    )
    root, packaged = source_bundle.write_submit_folder(tmp_path / "bundle")
    record = JobRecord(
        bundle_id=str(source_bundle.mlff_train_bundle_id),
        attempt_folder_name="bundle",
        resource=resource.model_dump(mode="json"),
        relative_source_bundle_path="bundle.json",
    )
    first = module.DispatcherBackend(record, root, lambda: None)
    assert first.submission.machine.retry_count == 0
    assert len(first.submission.belonging_jobs) == 1
    first._job.job_id = "1234"
    first._snapshot()
    restored = module.DispatcherBackend(
        JobRecord.model_validate_json(record.model_dump_json()),
        root,
        lambda: None,
    )
    assert restored._job.job_id == "1234"
    assert (
        restored.submission.machine.context.remote_root
        == first.submission.machine.context.remote_root
    )
    assert restored._job.job_hash == first._job.job_hash
    assert restored.submission.belonging_tasks


@pytest.mark.parametrize(
    ("backend_name", "queue_output", "expected"),
    [
        ("PBS", "Job Id: 123\n    job_state = E\n", "running"),
        ("PBS", "Job Id: 123\n    job_state = F\n", "failed"),
        ("Slurm", "COMPLETING\n", "running"),
        ("Slurm", "PENDING\n", "waiting"),
    ],
)
def test_scheduler_exiting_is_not_terminal(
    backend_name: str,
    queue_output: str,
    expected: str,
    resource: RemoteResource,
) -> None:
    pytest.importorskip("temper.job_management.dispatcher")
    from temper.job_management.dispatcher import DispatcherBackend

    backend = DispatcherBackend.__new__(DispatcherBackend)
    backend.profile = resource.model_copy(update={"backend": backend_name})
    backend._job = SimpleNamespace(job_id="123")
    backend.record = SimpleNamespace(observation={})
    context = SimpleNamespace(
        block_call=lambda _: (
            0,
            None,
            BytesIO(queue_output.encode()),
            BytesIO(),
        )
    )
    backend.submission = SimpleNamespace(
        machine=SimpleNamespace(
            context=context, check_finish_tag=lambda _: False
        )
    )
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

    backend.submission = SimpleNamespace(
        machine=SimpleNamespace(
            context=SimpleNamespace(block_call=query),
            check_finish_tag=lambda _: False,
        )
    )
    assert backend.status() == "failed"
    assert len(calls) == 2 and "sacct" in calls[-1]


def test_bohrium_real_serialization_upload_checkpoint_and_secret_removal(
    source_bundle: MLFFTrainBundle,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("bohrium")
    from temper.job_management.dispatcher import DispatcherBackend
    import dpdispatcher.contexts.openapi_context as cloud_context

    monkeypatch.setattr(
        cloud_context, "DP_CLOUD_SERVER_HOME_DIR", str(tmp_path / "dp-history")
    )
    monkeypatch.setenv("TEST_CLOUD_KEY", "never-write-this-secret")
    resource = RemoteResource(
        name="cloud",
        backend="Bohrium",
        project_id=123,
        machine_type="gpu",
        access_key_env="TEST_CLOUD_KEY",
        environment={"image": "image:fixed"},
    )
    root, _ = source_bundle.write_submit_folder(tmp_path / "cloud")
    record = JobRecord(
        bundle_id=str(source_bundle.mlff_train_bundle_id),
        attempt_folder_name="cloud",
        resource=resource.model_dump(mode="json"),
        relative_source_bundle_path="bundle.json",
    )
    checkpoints = []
    backend = DispatcherBackend(
        record, root, lambda: checkpoints.append(record.model_dump_json())
    )
    assert "never-write-this-secret" not in record.model_dump_json()
    assert backend.submission.machine.client.max_retries == 1
    context = backend.submission.machine.context
    monkeypatch.setattr(
        context.job,
        "create",
        lambda **kwargs: {
            "jobId": 7,
            "storePath": "store/job7",
            "token": "upload-secret",
            "jobGroupId": 2,
        },
    )

    def write_upload(**kwargs: object) -> SimpleNamespace:
        assert "upload_path" in json.loads(checkpoints[-1])["dispatcher"]
        assert kwargs["token"] == "upload-secret"
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"code": 0, "data": {}},
            status_code=200,
        )

    monkeypatch.setattr(context.storage, "write", write_upload)
    backend.upload()
    assert checkpoints and all(
        "upload-secret" not in item and "never-write-this-secret" not in item
        for item in checkpoints
    )
    params = []
    monkeypatch.setattr(
        backend.submission.machine.job,
        "insert",
        lambda **kwargs: params.append(kwargs) or {"jobId": 7},
    )
    backend.submit()
    assert params[0]["image_name"] == "image:fixed"
    restored = DispatcherBackend(
        JobRecord.model_validate_json(record.model_dump_json()),
        root,
        lambda: None,
    )
    assert restored._job.job_id == 7
    assert restored._job.upload_path == backend._job.upload_path


def test_bohrium_lost_create_reply_does_not_create_again(
    source_bundle: MLFFTrainBundle,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("bohrium")
    from temper.job_management.dispatcher import DispatcherBackend
    import dpdispatcher.contexts.openapi_context as cloud_context

    monkeypatch.setattr(
        cloud_context, "DP_CLOUD_SERVER_HOME_DIR", str(tmp_path / "dp-history")
    )
    monkeypatch.setenv("BOHRIUM_ACCESS_KEY", "test-key")
    resource = RemoteResource(
        name="cloud",
        backend="Bohrium",
        project_id=123,
        machine_type="gpu",
        environment={"image": "image:fixed"},
    )
    root, _ = source_bundle.write_submit_folder(tmp_path / "cloud")
    record = JobRecord(
        bundle_id=str(source_bundle.mlff_train_bundle_id),
        attempt_folder_name="cloud",
        resource=resource.model_dump(mode="json"),
        relative_source_bundle_path="bundle.json",
    )
    backend = DispatcherBackend(record, root, lambda: None)
    calls = []

    def create(**kwargs: object) -> dict:
        calls.append(kwargs)
        raise ConnectionError("lost response")

    monkeypatch.setattr(
        backend.submission.machine.context.job, "create", create
    )
    with pytest.raises(ConnectionError):
        backend.upload()
    with pytest.raises(RuntimeError, match="creation response was lost"):
        backend.upload()
    assert len(calls) == 1


def test_cancel_interrupted_upload_does_not_launch_on_recovery(
    managed: tuple, source_bundle: MLFFTrainBundle, resource: RemoteResource
) -> None:
    manager, control = managed
    control["upload_error"] = True
    record = manager.submit(source_bundle, resource)
    manager.cancel(record.bundle_id)
    manager.recover(record.bundle_id)
    assert control["submits"] == 0
    assert manager.store.records()[0].computation_status == "cancelled"


def test_shared_dataset_batch_loading(
    source_bundle: MLFFTrainBundle,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import temper.schemas.train_unit as module

    path = tmp_path / "units.json"
    path.write_text(
        json.dumps([source_bundle.training_unit.model_dump(mode="json")] * 3)
    )
    calls = []
    monkeypatch.setattr(
        module,
        "file_sha256",
        lambda path: calls.append(path) or file_sha256(path),
    )
    assert len(module.load_training_units(path)) == 3
    assert len(calls) == len(source_bundle.training_unit.dataset_sha256)


def test_tampered_local_result_prevents_pending_cleanup(
    managed: tuple, source_bundle: MLFFTrainBundle, resource: RemoteResource
) -> None:
    manager, control = managed
    record = manager.submit(source_bundle, resource)
    control.update(status="success", cleanup_error=True)
    result = manager.synchronize()[0]
    root = manager.store.path(record)
    (root / next(iter(result.downloaded_file_hashes))).write_bytes(b"edited")
    control["cleanup_error"] = False
    cleanups = control["cleanups"]
    assert manager.synchronize()[0].cleanup_status == "pending"
    assert control["cleanups"] == cleanups


def test_environment_is_frozen_per_family(
    managed: tuple, source_bundle: MLFFTrainBundle, resource: RemoteResource
) -> None:
    from temper.job_management.resources import RuntimeEnvironment

    manager, _ = managed
    resource.environments["mace"] = RuntimeEnvironment(
        conda_init="/conda/conda.sh", conda_environment="mace-env"
    )
    record = manager.submit(source_bundle, resource)
    resource.environments["mace"].conda_environment = "changed-later"
    assert (
        record.resource["resolved_environment"]["conda_environment"]
        == "mace-env"
    )


def test_cli_refresh_bundle_identity_replaces_bundle(
    source_bundle: MLFFTrainBundle, tmp_path: Path
) -> None:
    """Replace a legacy bundle at its existing path with a hashed identity.

    Parameters
    ----------
    source_bundle : MLFFTrainBundle
        Valid bundle with accessible local datasets.
    tmp_path : Path
        Temporary directory for the bundle being replaced.
    """
    from temper.entrypoints.main import main

    source = tmp_path / "old.json"
    data = source_bundle.model_dump(mode="json")
    data.pop("mlff_train_bundle_id")
    data["training_unit"].pop("training_unit_id")
    data["training_unit"].pop("dataset_sha256")
    legacy = MLFFTrainBundle.model_validate(data)
    original = legacy.model_dump_json()
    source.write_text(original)
    with pytest.raises(SystemExit) as result:
        main(["job", "refresh_bundle_identity", str(source)])
    assert result.value.code == 0
    refreshed = MLFFTrainBundle.model_validate_json(source.read_text())
    assert source.read_text() != original
    assert (
        refreshed.training_unit.dataset_sha256
        == source_bundle.training_unit.dataset_sha256
    )
    assert refreshed.mlff_train_bundle_id != legacy.mlff_train_bundle_id
    assert refreshed.files is None
    assert list(tmp_path.glob("*.json")) == [source]


@pytest.mark.parametrize("failure", ["missing_dataset", "replacement"])
def test_cli_refresh_preserves_bundle_on_failure(
    source_bundle: MLFFTrainBundle,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    """Keep the original bundle if dataset reading or file replacement fails.

    Parameters
    ----------
    source_bundle : MLFFTrainBundle
        Valid bundle with accessible local datasets.
    tmp_path : Path
        Temporary directory for the bundle being replaced.
    monkeypatch : pytest.MonkeyPatch
        Fixture used to simulate a failed filesystem replacement.
    failure : str
        Either missing_dataset to remove an input, or replacement to
        fail the final filesystem operation after writing the temporary
        bundle.
    """
    import argparse
    import temper.job_management.storage_utils as storage
    from temper.entrypoints.job import add_job_parser, job_cli

    source = tmp_path / "bundle.json"
    source.write_text(source_bundle.model_dump_json(), encoding="utf-8")
    original = source.read_bytes()
    parser = argparse.ArgumentParser()
    add_job_parser(parser.add_subparsers())
    args = parser.parse_args(["job", "refresh_bundle_identity", str(source)])
    assert not hasattr(args, "bundle_write_root")
    assert not hasattr(args, "output")
    if failure == "missing_dataset":
        source_bundle.training_unit.dataset_source(
            source_bundle.training_unit.test_sets[0]
        ).unlink()
    else:

        def fail_replace(source_path: str, destination_path: Path) -> None:
            raise OSError("Simulated replacement failure")

        monkeypatch.setattr(storage.os, "replace", fail_replace)
    with pytest.raises((ValueError, OSError)):
        job_cli(args)
    assert source.read_bytes() == original
    assert not list(tmp_path.glob(".temper-*"))


def test_submit_discovers_bundles_but_not_attempts(
    managed: tuple,
    source_bundle: MLFFTrainBundle,
    resource: RemoteResource,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Submit each bundle ID once and exclude packaged attempt bundles.

    Parameters
    ----------
    managed : tuple
        Job manager and simulated backend counters; no real jobs are
        launched.
    source_bundle : MLFFTrainBundle
        Bundle with verified local inputs.
    resource : RemoteResource
        Resource settings used with the simulated backend.
    tmp_path : Path
        Temporary directory for the resource profile.
    monkeypatch : pytest.MonkeyPatch
        Fixture connecting the CLI to the simulated manager.
    """
    from temper.entrypoints import job
    from temper.entrypoints.main import main_parser
    from temper.job_management.paths import bundle_directory

    manager, control = managed
    directory = bundle_directory(manager.store.root, source_bundle)
    directory.mkdir(parents=True)
    (directory / "bundle.json").write_text(source_bundle.model_dump_json())
    # An invalid attempt bundle must never be read as an input bundle.
    attempt = directory / "attempt_001"
    attempt.mkdir()
    (attempt / "bundle.json").write_text("not JSON")
    duplicate = directory.with_name("bundle__duplicate")
    duplicate.mkdir()
    (duplicate / "bundle.json").write_text(source_bundle.model_dump_json())
    profile = tmp_path / "profile.json"
    profile.write_text(
        resource.model_dump_json(exclude={"resolved_environment"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(job, "JobManager", lambda root: manager)
    args = main_parser().parse_args(
        [
            "job",
            "submit",
            "--bundle-write-root",
            str(manager.store.root),
            "--resource-file",
            str(profile),
        ]
    )
    assert job.job_cli(args) == 0
    assert control["submits"] == 1
    assert (directory / "attempt_002" / "job.json").is_file()
    assert len(manager.list_attempts()) == 1
    assert job.job_cli(args) == 1  # Existing active attempt is not relaunched.
    assert control["submits"] == 1


def test_copy_does_not_remove_existing_destination(tmp_path: Path) -> None:
    source, destination = tmp_path / "source", tmp_path / "destination"
    source.write_bytes(b"input")
    destination.write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        copy_verified(source, destination, file_sha256(source))
    assert destination.read_bytes() == b"keep"


def test_actual_ssh_collector_checks_hashes_and_excludes_checkpoints(
    source_bundle: MLFFTrainBundle,
    resource: RemoteResource,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("temper.job_management.dispatcher")
    from temper.job_management.dispatcher import DispatcherBackend

    root, bundle = source_bundle.write_submit_folder(tmp_path / "local")
    backend = DispatcherBackend.__new__(DispatcherBackend)
    backend.profile, backend.root, backend._bundle = resource, root, bundle
    remote = tmp_path / "remote"
    remote.mkdir()
    (remote / "stderr.log").write_bytes(b"failure diagnosis")
    (remote / "training_outputs/mace/checkpoints").mkdir(parents=True)
    (remote / "training_outputs/mace/checkpoints/large.pt").write_bytes(
        b"do not transfer"
    )

    def local_path(path: str) -> Path:
        return remote / path.removeprefix("/remote/").removeprefix("/remote")

    def listing(path: str) -> list:
        return [
            SimpleNamespace(
                filename=p.name,
                st_mode=stat.S_IFDIR if p.is_dir() else stat.S_IFREG,
            )
            for p in local_path(path).iterdir()
        ]

    transfers = []

    def download(source: str, destination: str) -> None:
        transfers.append(source)
        shutil.copyfile(local_path(source), destination)

    context = SimpleNamespace(
        remote_root="/remote",
        sftp=SimpleNamespace(listdir_attr=listing, get=download),
    )
    backend.submission = SimpleNamespace(
        machine=SimpleNamespace(context=context)
    )
    monkeypatch.setattr(
        backend,
        "_command",
        lambda _: file_sha256(remote / "stderr.log") + "  stderr.log",
    )
    manifest = backend.collect()
    assert manifest == {"stderr.log": file_sha256(remote / "stderr.log")}
    assert transfers == ["/remote/stderr.log"]
    assert not list(root.rglob("*.partial"))


def test_ssh_cleanup_requires_matching_ownership(
    resource: RemoteResource, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("temper.job_management.dispatcher")
    from temper.job_management.dispatcher import DispatcherBackend

    backend = DispatcherBackend.__new__(DispatcherBackend)
    backend.profile = resource
    backend.record = SimpleNamespace(dispatcher={"token": "a" * 32})
    context = SimpleNamespace(
        remote_root="/scratch/temper/temper_" + "a" * 32,
        read_file=lambda _: "different-owner",
    )
    backend.submission = SimpleNamespace(
        machine=SimpleNamespace(context=context)
    )
    calls = []
    monkeypatch.setattr(
        backend, "_command", lambda command: calls.append(command)
    )
    with pytest.raises(ValueError, match="ownership mismatch"):
        backend.cleanup()
    assert not calls


def test_zip_collection_rejects_traversal_without_writing(
    source_bundle: MLFFTrainBundle,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("temper.job_management.dispatcher")
    from zipfile import ZipFile
    import requests
    from temper.job_management.dispatcher import DispatcherBackend

    root, bundle = source_bundle.write_submit_folder(tmp_path / "local")
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
    backend.profile = RemoteResource(
        name="cloud", backend="Bohrium", project_id=1, machine_type="gpu"
    )
    backend.root, backend._bundle, backend._job = (
        root,
        bundle,
        SimpleNamespace(job_id=1),
    )
    backend.submission = SimpleNamespace(
        machine=SimpleNamespace(
            context=object(),
            job=SimpleNamespace(
                detail=lambda _: {
                    "resultUrl": "https://example.invalid/results"
                }
            ),
        )
    )
    with pytest.raises(ValueError, match="Unsafe remote result path"):
        backend.collect()
    assert not (tmp_path / "stderr.log").exists()


def test_saved_record_legacy_names_are_read_but_new_names_are_written() -> (
    None
):
    record = JobRecord(
        bundle_id="id",
        attempt_folder_name="attempt",
        resource={},
        manifest={"stderr.log": "hash"},
    )
    assert record.downloaded_file_hashes == {"stderr.log": "hash"}
    saved = record.model_dump()
    assert "manifest" not in saved
    assert "source_bundle" not in saved
    assert "packaged_files" not in saved
    assert "source_bundle" not in JobRecord.model_fields
    assert "packaged_files" not in JobRecord.model_fields
    assert JobRecord.model_validate(saved) == record


@pytest.mark.parametrize("field", ["source_bundle", "packaged_files"])
def test_job_record_rejects_removed_bundle_fields(field: str) -> None:
    """Reject old embedded bundle fields instead of silently discarding them.

    Parameters
    ----------
    field : str
        Removed field whose presence must fail record validation.
    """
    with pytest.raises(ValueError, match="Extra inputs"):
        JobRecord.model_validate(
            {
                "bundle_id": "id",
                "attempt_folder_name": "attempt",
                "resource": {},
                field: {},
            }
        )


def test_retry_preserves_shared_file_layout(
    managed: tuple[JobManager, dict[str, object]],
    source_bundle: MLFFTrainBundle,
    resource: RemoteResource,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject changed packaging paths before creating another attempt
    directory.

    Parameters
    ----------
    managed : tuple[JobManager, dict[str, object]]
        Manager with simulated execution and its operation controls.
    source_bundle : MLFFTrainBundle
        Source bundle for the first attempt.
    resource : RemoteResource
        Settings for the simulated remote job.
    monkeypatch : pytest.MonkeyPatch
        Fixture changing a packaging directory setting for the retry.
    """
    import temper.mlff.bundle_writers.base as writers

    manager, control = managed
    record = manager.submit(source_bundle, resource)
    root = manager.store.path(record)
    original = (root.parent / "bundle.json").read_bytes()
    control["status"] = "failed"
    manager.synchronize()
    monkeypatch.setattr(
        writers, "DEFAULT_MLFF_TRAINING_INPUTS_DIR", "changed_inputs"
    )
    assert manager.resubmit()[0]["action"] == "error"
    assert (root.parent / "bundle.json").read_bytes() == original
    assert not (root.parent / "attempt_002").exists()


def test_environment_activation_and_conflicts() -> None:
    from temper.job_management.schemas.resources import RuntimeEnvironment

    conda = RuntimeEnvironment(conda_environment="model env").shell_lines()
    assert conda[0] == "temper_conda_hook=$(conda shell.bash hook)"
    assert conda[-1] == "conda activate 'model env'"
    assert RuntimeEnvironment(venv_path="/opt/model env").shell_lines() == [
        "source '/opt/model env/bin/activate'"
    ]
    with pytest.raises(ValueError, match="either venv or Conda"):
        RuntimeEnvironment(venv_path="/opt/venv", conda_environment="model")
    with pytest.raises(ValueError, match="absolute"):
        RuntimeEnvironment(venv_path="relative")


def test_new_profiles_reject_default_environment(tmp_path: Path) -> None:
    profile = tmp_path / "local.yaml"
    profile.write_text("name: local\nbackend: Local\nenvironment: {}\n")
    with pytest.raises(ValueError, match="environments keyed"):
        RemoteResource.from_file(profile)


def test_missing_family_and_remote_root(
    managed: tuple, source_bundle: MLFFTrainBundle, resource: RemoteResource
) -> None:
    manager, _ = managed
    resource.environments = {}
    with pytest.raises(ValueError, match="environments entry for mace"):
        manager.submit(source_bundle, resource)
    from temper.job_management.schemas.resources import RuntimeEnvironment

    resource.environments = {"mace": RuntimeEnvironment()}
    resource.remote_root = None
    with pytest.warns(RuntimeWarning, match="/tmp may be cleared"):
        record = manager.submit(source_bundle, resource)
    assert record.resource["remote_root"].startswith("/tmp/temper-")
    assert resource.remote_root is None


def test_password_requires_explicit_source_and_keys_are_not_packaged(
    source_bundle: MLFFTrainBundle, tmp_path: Path
) -> None:
    profile = RemoteResource(
        name="cluster",
        backend="Slurm",
        hostname="host",
        username="user",
        password_env="PASSWORD",
    )
    assert profile.key_filename is None
    with pytest.raises(ValueError, match="key_filename or password_env"):
        RemoteResource(
            name="cluster", backend="Slurm", hostname="host", username="user"
        )
    _, bundle = source_bundle.write_submit_folder(tmp_path / "bundle")
    names = get_dataset_and_model_hashes(bundle)
    assert not set(names) & {
        bundle.files.training_input_files["config"],
        bundle.files.training_input_files["script"],
    }


def test_download_selection_is_exact_and_diagnostics_can_be_empty(
    source_bundle: MLFFTrainBundle, tmp_path: Path
) -> None:
    from temper.job_management.collection import verify_results

    root, bundle = source_bundle.write_submit_folder(tmp_path / "bundle")
    names, required = get_download_files(bundle)
    assert all(not any(char in name for char in "*?[") for name in names)
    assert not selected_file("training_outputs/unexpected.json", names)
    assert not selected_file("training_outputs/nep_best.txt", names)
    (root / "stderr.log").touch()
    hashes = {"stderr.log": file_sha256(root / "stderr.log")}
    verify_results(root, hashes, [])
    with pytest.raises(ValueError, match="Required result"):
        verify_results(root, hashes, required)


def test_ssh_connection_retry_budget_and_auth_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("temper.job_management.dispatcher")
    import temper.job_management.dispatcher as module

    session = module._VerifiedSession.__new__(module._VerifiedSession)
    session.ssh = SimpleNamespace(close=lambda: None)
    attempts: list[int] = []
    sleeps: list[int] = []

    def fail() -> None:
        attempts.append(1)
        raise ConnectionError("offline")

    monkeypatch.setattr(session, "_connect_once", fail)
    monkeypatch.setattr(module.time, "sleep", sleeps.append)
    with pytest.raises(ConnectionError):
        session._setup_ssh()
    assert len(attempts) == 5 and sleeps == [10] * 4
    attempts.clear()
    sleeps.clear()

    def auth_failure() -> None:
        attempts.append(1)
        raise module.paramiko.AuthenticationException("wrong key")

    monkeypatch.setattr(session, "_connect_once", auth_failure)
    with pytest.raises(module.paramiko.AuthenticationException):
        session._setup_ssh()
    assert len(attempts) == 1 and not sleeps


def test_local_collection_cleanup_and_uncertain_recovery(
    source_bundle: MLFFTrainBundle, tmp_path: Path
) -> None:
    from temper.job_management.local import LocalBackend

    root, bundle = source_bundle.write_submit_folder(tmp_path / "bundle")
    profile = RemoteResource(
        name="local", backend="Local", environments={"mace": {}}
    )
    record = JobRecord(
        bundle_id=str(bundle.mlff_train_bundle_id),
        attempt_folder_name="bundle",
        resource=profile.model_dump(mode="json"),
        relative_source_bundle_path="bundle.json",
    )
    backend = LocalBackend(record, root, lambda: None)
    config = root / bundle.files.training_input_files["config"]
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text("user configuration")
    assert backend.status() == "unknown" and not backend.recover()
    receipt = root / ".local-receipt.json"
    for contents in ("", "{}", "[]", '{"pid": false}', '{"pid": -1}'):
        receipt.write_text(contents)
        assert not backend.recover()
    receipt.write_text('{"pid": 1234}')
    assert backend.recover()
    backend.cancel()
    assert (root / ".local-cancel").exists()
    (root / ".local-exit.json").write_text(
        '{"returncode": 0, "cancelled": false}'
    )
    assert backend.status() == "success"
    names, required = get_download_files(bundle)
    for name in required:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"result")
    intermediate = (
        root / bundle.files.training_output_files["checkpoints"] / "epoch.pt"
    )
    intermediate.parent.mkdir(parents=True, exist_ok=True)
    intermediate.write_bytes(b"checkpoint")
    converted = root / "converted-data"
    converted.mkdir()
    (converted / "arrays.npy").write_bytes(b"converted")
    backend._bundle.files.training_input_files["converted"] = "converted-data"
    hashes = backend.collect()
    assert set(required) <= hashes.keys()
    backend.cleanup()
    assert not intermediate.exists()
    assert not converted.exists()
    assert all((root / name).is_file() for name in required)
    assert (root / bundle.files.training_input_files["config"]).is_file()
    assert all(
        not (root / name).exists()
        for name in get_dataset_and_model_hashes(bundle)
    )
    record.dispatcher["local_launch_intent"] = True
    with pytest.raises(RuntimeError, match="uncertain"):
        backend.submit()


@pytest.mark.skipif(
    os.name != "posix", reason="Local MLFF execution requires POSIX Bash"
)
def test_local_supervisor_records_completion_and_cancellation(
    tmp_path: Path,
) -> None:
    import subprocess
    import sys
    import time

    for cancel in (False, True):
        root = tmp_path / str(cancel)
        root.mkdir()
        (root / "temper_job.sh").write_text(
            "sleep 60\n" if cancel else "echo done\n"
        )
        process = subprocess.Popen(
            [sys.executable, "-m", "temper.job_management.local", str(root)]
        )
        try:
            if cancel:
                deadline = time.monotonic() + 10
                while (
                    not (root / ".local-heartbeat").exists()
                    and time.monotonic() < deadline
                ):
                    time.sleep(0.05)
                assert (root / ".local-heartbeat").exists()
                (root / ".local-cancel").touch()
            process.wait(timeout=20)
            result = json.loads((root / ".local-exit.json").read_text())
            assert result["cancelled"] is cancel
            if not cancel:
                assert result["returncode"] == 0
                assert (root / "stdout.log").read_text().strip() == "done"
            assert (root / ".local-receipt.json").exists()
        finally:
            if process.poll() is None:
                (root / ".local-cancel").touch()
                process.wait(timeout=20)
