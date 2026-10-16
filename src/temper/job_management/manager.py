"""Submission, restart recovery and explicit retries of complete calculations."""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import logging
import time
from typing import Protocol
from uuid import uuid4

from temper.job_management.collection import result_selection, verify_results
from temper.job_management.inputs import input_hashes, verify_packaged_inputs
from temper.job_management.paths import bundle_directory, next_attempt
from temper.job_management.storage_utils import store_lock
from temper.job_management.store import JobStore
from temper.job_management.records import JobRecord, utc_now
from temper.schemas.mlff_train_bundle import MLFFTrainBundle
from temper.job_management.resources import RemoteResource
from temper.utils.file_hashing import hashing_session

logger = logging.getLogger(__name__)


class _Backend(Protocol):
    def upload(self) -> None: ...
    def submit(self) -> None: ...
    def status(self) -> str: ...
    def recover(self, job_id: str | None = None) -> bool: ...
    def collect(self) -> dict[str, str]: ...
    def cleanup(self) -> None: ...
    def cancel(self) -> None: ...
    def close(self) -> None: ...


def _backend(record: JobRecord, root: Path, checkpoint: Callable[[], None]) -> _Backend:
    try:
        from temper.job_management.dispatcher import DispatcherBackend
    except ImportError as error:
        raise RuntimeError("Install temper-bench[remote] to use SSH or [bohrium] for Bohrium.") from error
    return DispatcherBackend(record, root, checkpoint)


class JobManager:
    """Coordinate durable job history and bounded remote operations.

    Parameters
    ----------
    root : str, Path or None, optional
        Bundle store root; defaults to DEFAULT_BUNDLE_WRITE_ROOT or ./bundles.
    backend_factory : Callable or None, optional
        Backend constructor, principally for offline tests. None selects the
        pinned dpdispatcher integration. Construction must not launch a job.

    Attributes
    ----------
    store : JobStore
        Local job history. All mutations use an operating-system lock so two
        CLI processes sharing this root cannot submit the same bundle together.
    """
    def __init__(self, root: str | Path | None = None,
                 backend_factory: Callable[[JobRecord, Path, Callable[[], None]], _Backend] | None = None) -> None:
        self.store: JobStore = JobStore(root)
        self._factory = backend_factory or _backend

    @hashing_session()
    def submit(self, bundle: MLFFTrainBundle, resource: RemoteResource) -> JobRecord:
        """Create and launch a new attempt, leaving previous attempts untouched.

        Parameters
        ----------
        bundle : MLFFTrainBundle
            Local recipe with dataset hashes. Sources are reread and validated;
            nested mutations with stale identities are rejected.
        resource : RemoteResource
            Resource and MLFF-specific runtime environment to freeze in history.

        Returns
        -------
        JobRecord
            Submitted attempt, or a recoverable attempt containing an operation
            error. Calculation failures are never retried automatically.

        Raises
        ------
        ValueError
            If inputs changed, hashes are absent, or this bundle has a successful,
            active, uncertain or not-yet-collected earlier attempt.
        OSError
            If packaging or persistence fails.
        """
        with store_lock(self.store.root):
            return self._submit(bundle, resource)

    def _submit(self, bundle: MLFFTrainBundle, resource: RemoteResource) -> JobRecord:
        bundle = MLFFTrainBundle.model_validate(bundle.model_dump())
        if not bundle.training_unit.dataset_sha256:
            raise ValueError("Legacy dataset identity: upgrade the bundle with job upgrade_bundle before submission.")
        bundle.training_unit.root_path = bundle.training_unit.root_path.expanduser().resolve()
        for artifact in bundle.mlff_spec.pretrained_model.artifacts.values():
            artifact.path = artifact.path.expanduser().resolve()
        identity = str(bundle.mlff_train_bundle_id)
        reason = self._blocked(identity, self.store.records())
        if reason:
            raise ValueError(reason)
        effective = resource.model_copy(deep=True)
        effective.environment = effective.environments.get(bundle.mlff_spec.mlff_type, effective.environment)
        effective.environments = {}
        if effective.backend == "Bohrium" and not effective.environment.image:
            raise ValueError("Bohrium requires an image for this MLFF family.")
        target = next_attempt(bundle_directory(self.store.root, bundle))
        root, packaged = bundle.write_submit_folder(target)
        record = JobRecord(bundle_id=identity, attempt_folder_name=root.relative_to(self.store.root).as_posix(),
                           recipe=bundle.model_dump(mode="json"), resource=effective.model_dump(mode="json"),
                           dispatcher={"token": uuid4().hex})
        self.store.save(record, "prepared")
        self._launch(record, packaged)
        return record

    @staticmethod
    def _blocked(identity: str, records: list[JobRecord]) -> str | None:
        previous = [record for record in records if record.bundle_id == identity]
        if any(record.computation_status == "success" for record in previous):
            return "Bundle already succeeded; only changed scientific content permits a new submission."
        if any(record.computation_status not in {"failed", "cancelled"} or record.cleanup_status != "complete" for record in previous):
            return "Bundle has an active, uncertain or uncollected attempt; synchronize or recover it first."
        return None

    def _connect(self, record: JobRecord) -> _Backend:
        return self._factory(record, self.store.path(record), lambda: self.store.save(record, "backend_checkpoint"))

    def _launch(self, record: JobRecord, bundle: MLFFTrainBundle) -> None:
        backend = None
        try:
            backend = self._connect(record)
            if record.submission_status in {"prepared", "uploading"}:
                verify_packaged_inputs(bundle, self.store.path(record))
                record.submission_status = "uploading"
                self.store.save(record, "uploading")
                backend.upload()
                record.submission_status = "uploaded"
                self.store.save(record, "uploaded")
            self._remove_copies(record, bundle)
            record.submission_status = "submitting"
            self.store.save(record, "submission_intent")
            backend.submit()
            record.submission_status = "submitted"
            record.computation_status = "waiting"
            record.error = None
            self.store.save(record, "submitted")
        except Exception as error:
            if record.submission_status == "submitting":
                record.submission_status = "uncertain"
            self._error(record, error)
        finally:
            if backend:
                backend.close()

    def _remove_copies(self, record: JobRecord, bundle: MLFFTrainBundle) -> None:
        root = self.store.path(record)
        for name in input_hashes(bundle):
            (root / name).unlink(missing_ok=True)
        # Dispatcher archives contain the same large copies, not new results.
        for suffix in ("zip", "tgz", "tar"):
            for path in root.glob("*." + suffix):
                if len(path.stem) == 40 and all(c in "0123456789abcdef" for c in path.stem):
                    path.unlink()

    def _error(self, record: JobRecord, error: Exception) -> None:
        # Backend exceptions may contain credentials or signed download URLs.
        record.error = f"{type(error).__name__} during {record.submission_status}/{record.collection_status}/{record.cleanup_status}; retry synchronization or inspect the remote resource."
        self.store.save(record, "operation_error")

    def synchronize(self, *, bundle_id: str | None = None, resource: str | None = None,
                    domain: str | None = None, refresh_only: bool = False,
                    download_only: bool = False) -> list[JobRecord]:
        """Check selected attempts once and optionally collect and clean them.

        Parameters
        ----------
        bundle_id, resource, domain : str or None, optional
            Exact filters. None selects all values. Resource is the profile name.
        refresh_only : bool, default False
            Query and persist status only; no upload, submission, cancellation,
            result download or remote deletion.
        download_only : bool, default False
            Collect terminal results but postpone remote cleanup.

        Returns
        -------
        list[JobRecord]
            Selected refreshed records, including independent operation errors.
            One unreachable resource does not stop processing other attempts.
        """
        with store_lock(self.store.root):
            records = self.select(bundle_id=bundle_id, resource=resource, domain=domain)
            for record in records:
                self.store.repair_markers(record)
                if record.cleanup_status == "complete":
                    continue
                self._synchronize(record, refresh_only, download_only)
            return records

    def _synchronize(self, record: JobRecord, refresh_only: bool, download_only: bool) -> None:
        if record.submission_status != "submitted":
            return  # Recovery is explicit, never a hidden calculation retry.
        backend = None
        try:
            backend = self._connect(record)
            if record.computation_status not in {"success", "failed", "cancelled"}:
                observation = backend.status()
                record.checked_at = utc_now()
                if observation != "unknown":
                    record.computation_status = observation
                if record.cancel_requested and observation == "failed":
                    record.computation_status = "cancelled"
                record.error = "Remote status is unknown; retaining previous observation." if observation == "unknown" else None
                self.store.save(record, "status_checked")
            if refresh_only:
                return
            if record.cancel_requested and record.computation_status not in {"success", "failed", "cancelled"}:
                backend.cancel()
                self.store.save(record, "cancellation_requested")
                return
            if record.computation_status not in {"success", "failed", "cancelled"}:
                return
            root = self.store.path(record)
            bundle = MLFFTrainBundle.model_validate_json((root / "bundle.json").read_text(encoding="utf-8"), context={"verify_files": False})
            _, required = result_selection(bundle)
            required = required if record.computation_status == "success" else []
            if record.collection_status != "verified":
                record.manifest = backend.collect()
                verify_results(root, record.manifest, required)
                record.collection_status = "verified"
                record.error = None
                self.store.save(record, "collected_and_verified")
            if not download_only:
                verify_results(root, record.manifest, required)
                backend.cleanup()
                record.cleanup_status = "complete"
                record.error = None
                self.store.save(record, "remote_cleanup_complete")
        except Exception as error:
            self._error(record, error)
        finally:
            if backend:
                backend.close()

    def select(self, *, bundle_id: str | None = None, resource: str | None = None,
               domain: str | None = None, failed: bool = False) -> list[JobRecord]:
        """Read local attempts matching exact filters.

        Parameters
        ----------
        bundle_id, resource, domain : str or None, optional
            Full bundle ID, profile name and data domain. None leaves a filter unset.
        failed : bool, default False
            Return failed computations only; operational errors are separate.

        Returns
        -------
        list[JobRecord]
            Matching attempts in chronological order, without remote queries.
        """
        return [r for r in self.store.records()
                if (bundle_id is None or r.bundle_id == bundle_id)
                and (resource is None or r.resource["name"] == resource)
                and (domain is None or r.recipe["training_unit"]["domain"] == domain)
                and (not failed or r.computation_status == "failed")]

    @hashing_session()
    def resubmit(self, *, bundle_id: str | None = None, resource: str | None = None,
                 domain: str | None = None, dry_run: bool = False,
                 replacement_resource: RemoteResource | None = None) -> list[dict[str, str]]:
        """Retry eligible failed bundles once each, with fresh attempt directories.

        Parameters
        ----------
        bundle_id, resource, domain : str or None, optional
            Exact filters on failed history; None includes all values.
        dry_run : bool, default False
            Report eligible and skipped bundle IDs without packaging or launching.
        replacement_resource : RemoteResource or None, optional
            Alternative execution resource; None reuses each frozen profile.
            Resource changes do not change the scientific identity.

        Returns
        -------
        list[dict[str, str]]
            One result per bundle: ID, action and reason or attempt path. A later
            successful attempt always excludes earlier failures of that bundle.
        """
        with store_lock(self.store.root):
            all_records = self.store.records()
            failed = self.select(bundle_id=bundle_id, resource=resource, domain=domain, failed=True)
            latest = {record.bundle_id: record for record in failed}
            outcomes = []
            for identity, record in latest.items():
                reason = self._blocked(identity, all_records)
                if reason:
                    outcomes.append({"bundle_id": identity, "action": "skipped", "reason": reason})
                    continue
                if dry_run:
                    outcomes.append({"bundle_id": identity, "action": "eligible"})
                    continue
                try:
                    bundle = MLFFTrainBundle.model_validate(record.recipe)
                    profile = replacement_resource or RemoteResource.model_validate(record.resource)
                    attempt = self._submit(bundle, profile)
                    outcomes.append({"bundle_id": identity, "action": attempt.submission_status, "attempt": attempt.attempt_folder_name})
                    all_records.append(attempt)
                except Exception as error:
                    outcomes.append({"bundle_id": identity, "action": "error", "reason": type(error).__name__ + ": input validation or packaging failed"})
            return outcomes

    def cancel(self, bundle_id: str) -> list[JobRecord]:
        """Persist cancellation intent and request it for active attempts.

        Parameters
        ----------
        bundle_id : str
            Exact full bundle ID. Completed attempts are left alone.

        Returns
        -------
        list[JobRecord]
            Updated attempts. Synchronization confirms termination and collects
            diagnostics; stopping a watcher never calls this method.
        """
        with store_lock(self.store.root):
            records = self.select(bundle_id=bundle_id)
            for record in records:
                if record.computation_status in {"success", "failed", "cancelled"}:
                    continue
                record.cancel_requested = True
                self.store.save(record, "cancellation_intent")
                if record.submission_status in {"prepared", "uploading", "uploaded"}:
                    self._cancel_unsubmitted(record)
                else:
                    self._synchronize(record, False, False)
            return records

    def _cancel_unsubmitted(self, record: JobRecord) -> None:
        backend = None
        try:
            if record.submission_status != "prepared":
                backend = self._connect(record)
                if record.resource["backend"] == "Bohrium" and not backend.recover():
                    raise ValueError("Recover the cloud draft ID before deleting its upload.")
                backend.cleanup()
            record.computation_status = "cancelled"
            record.collection_status = "verified"
            record.cleanup_status = "complete"
            record.error = None
            self.store.save(record, "cancelled_before_launch")
            bundle = MLFFTrainBundle.model_validate_json((self.store.path(record) / "bundle.json").read_text(encoding="utf-8"), context={"verify_files": False})
            self._remove_copies(record, bundle)
        except Exception as error:
            self._error(record, error)
        finally:
            if backend:
                backend.close()

    def recover(self, bundle_id: str, job_id: str | None = None) -> list[JobRecord]:
        """Resume interrupted submission operations without rerunning a calculation.

        Parameters
        ----------
        bundle_id : str
            Exact full bundle ID.
        job_id : str or None, optional
            Existing remote ID verified by the user when the receipt is absent.
            This attaches an uncertain attempt; it never launches a replacement.

        Returns
        -------
        list[JobRecord]
            Recovered or still-uncertain records. Prepared/uploaded SSH attempts
            can finish their original submission; an uncertain launch only
            reads remote evidence. A failed calculation requires resubmit.
        """
        with store_lock(self.store.root):
            records = self.select(bundle_id=bundle_id)
            for record in records:
                if record.submission_status == "submitted" or record.computation_status in {"success", "failed", "cancelled"}:
                    continue
                if record.submission_status == "uploading" and record.resource["backend"] == "Bohrium" and job_id:
                    backend = self._connect(record)
                    try:
                        backend.recover(job_id)
                        self.store.save(record, "recovered_cloud_draft")
                    finally:
                        backend.close()
                if record.cancel_requested and record.submission_status in {"prepared", "uploading", "uploaded"}:
                    self._cancel_unsubmitted(record)
                    continue
                if record.submission_status in {"prepared", "uploaded", "uploading"}:
                    bundle = MLFFTrainBundle.model_validate_json((self.store.path(record) / "bundle.json").read_text(encoding="utf-8"), context={"verify_files": False})
                    self._launch(record, bundle)
                    continue
                backend = None
                try:
                    backend = self._connect(record)
                    if backend.recover(job_id) and record.submission_status != "uploading":
                        record.submission_status = "submitted"
                        record.error = None
                        self.store.save(record, "recovered_existing_job")
                    else:
                        record.error = "No confirmed launch receipt; inspect the remote job before recovery."
                        self.store.save(record, "recovery_inconclusive")
                except Exception as error:
                    self._error(record, error)
                finally:
                    if backend:
                        backend.close()
            return records

    def watch(self, interval: float = 60, *, bundle_id: str | None = None,
              resource: str | None = None, domain: str | None = None) -> None:
        """Synchronize repeatedly in the foreground until all selected jobs are clean.

        Parameters
        ----------
        interval : float, default 60
            Seconds between synchronization passes; must be positive.
        bundle_id, resource, domain : str or None, optional
            Exact selection filters. None includes all values.

        Raises
        ------
        ValueError
            If interval is not positive.
        KeyboardInterrupt
            When stopped by the user. Remote calculations continue, and another
            watch or synchronization command resumes from JSON history.
        """
        if interval <= 0:
            raise ValueError("Watch interval must be positive.")
        while True:
            records = self.synchronize(bundle_id=bundle_id, resource=resource, domain=domain)
            logger.info("Job synchronization: %d attempts, %d cleaned, %d operational errors",
                        len(records), sum(r.cleanup_status == "complete" for r in records), sum(r.error is not None for r in records))
            if all(record.cleanup_status == "complete" for record in records):
                return
            time.sleep(interval)
