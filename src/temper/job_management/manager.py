"""Submission, restart recovery and explicit retries of complete
calculations."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import logging
import hashlib
import json
import warnings
import time
from typing import Protocol
from uuid import uuid4

from pydantic import JsonValue

from temper.job_management.collection import get_download_files, verify_results
from temper.job_management.errors import SubmissionRejected
from temper.job_management.input_utils import (
    get_dataset_and_model_hashes,
    load_bundle,
    verify_dataset_and_model_hashes,
)
from temper.job_management.paths import bundle_directory, next_attempt
from temper.job_management.storage_utils import write_json_atomically, store_lock
from temper.job_management.store import JobStore
from temper.job_management.schemas.records import (
    ComputationStatus,
    JobRecord,
    utc_now,
)
from temper.schemas.mlff_train_bundle import MLFFTrainBundle
from temper.job_management.schemas.resources import RemoteResource
from temper.job_management.schemas.submission import SubmissionDecision
from temper.job_management.schemas.abandonment import AbandonmentCandidate
from temper.utils.file_hashing import cache_dataset_hashes

logger = logging.getLogger(__name__)


class _Backend(Protocol):
    def upload(self) -> None: ...
    def submit(self) -> None: ...
    def status(self) -> ComputationStatus: ...
    def recover(self, job_id: str | None = None) -> bool: ...
    def collect(self) -> dict[str, str]: ...
    def cleanup(self) -> None: ...
    def cancel(self) -> None: ...
    def close(self) -> None: ...


def _backend(
    record: JobRecord, root: Path, checkpoint: Callable[[], None]
) -> _Backend:
    if record.resource["backend"] == "Local":
        from temper.job_management.local import LocalBackend

        return LocalBackend(record, root, checkpoint)
    try:
        from temper.job_management.dispatcher import DispatcherBackend
    except ImportError as error:
        raise RuntimeError(
            "Install temper-bench[remote] to use SSH or [bohrium] for Bohrium."
        ) from error
    return DispatcherBackend(record, root, checkpoint)


class JobManager:
    """Coordinate durable job history and bounded remote operations.

    Parameters
    ----------
    root : str, Path or None, optional
        Bundle store root; defaults to DEFAULT_BUNDLE_WRITE_ROOT or
        ./bundles.
    backend_factory : Callable or None, optional
        Backend constructor, principally for offline tests. None selects
        the pinned dpdispatcher integration. Construction must not
        launch a job.

    Attributes
    ----------
    store : JobStore
        Local job history. All mutations use an operating-system lock so
        two CLI processes sharing this root cannot submit the same
        bundle together.
    """

    def __init__(
        self,
        root: str | Path | None = None,
        backend_factory: (
            Callable[[JobRecord, Path, Callable[[], None]], _Backend] | None
        ) = None,
    ) -> None:
        self.store: JobStore = JobStore(root)
        self._factory = backend_factory or _backend

    @staticmethod
    def _abandonment_reason(record: JobRecord) -> str | None:
        """Explain unresolved history, excluding known active or ended jobs."""
        if (
            record.abandoned_at is not None
            or record.computation_status != "unknown"
        ):
            return None
        if record.submission_status in {"submitting", "uncertain"}:
            return (
                "Submission may have succeeded, but no outcome is confirmed."
            )
        if record.submission_status == "submitted":
            return (
                "A submission was recorded, but its calculation status is"
                " unknown."
            )
        if (
            record.resource["backend"] == "Bohrium"
            and record.submission_status == "uploading"
            and record.dispatcher.get("cloud_create_intent")
            and record.error
            and not JobManager._saved_remote_ids(record.dispatcher)
        ):
            return (
                "Cloud draft creation is unresolved; its ID could not be"
                " recovered."
            )
        return None

    @staticmethod
    def _abandonment_token(record: JobRecord) -> str:
        """Identify changes in saved state relevant to abandonment."""
        data = record.model_dump(
            mode="json",
            exclude={"updated_at", "checked_at", "error", "observation"},
        )
        return hashlib.sha256(
            json.dumps(data, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()

    @staticmethod
    def _saved_remote_ids(value: JsonValue) -> set[str]:
        """Find saved job IDs without exposing other connection settings."""
        result: set[str] = set()
        if isinstance(value, dict):
            for key, item in value.items():
                if key in {"job_id", "jobId"} and isinstance(item, (str, int)):
                    if str(item).strip():
                        result.add(str(item))
                elif isinstance(item, (dict, list)):
                    result.update(JobManager._saved_remote_ids(item))
        elif isinstance(value, list):
            for item in value:
                result.update(JobManager._saved_remote_ids(item))
        return result

    def plan_abandonment(
        self,
        *,
        bundle_id: str | None = None,
        resource: str | None = None,
        domain: str | None = None,
    ) -> list[AbandonmentCandidate]:
        """Check unresolved attempts and list those a user may abandon.

        Parameters
        ----------
        bundle_id, resource, domain : str or None, optional
            Exact filters on saved attempts: full bundle ID, saved resource
            name, and training-unit domain. None includes all values.

        Returns
        -------
        list[AbandonmentCandidate]
            Current unresolved attempts for review. Known queued, running,
            or ended calculations are excluded, as are ordinary interrupted
            uploads that can be resumed. Missing resource settings and
            collection failures alone do not make an attempt eligible.
            Empty means no matching attempt is eligible for abandonment.

        Raises
        ------
        OSError
            If history or the store lock cannot be accessed.
        ValueError
            If saved records or bundle references are invalid.

        Notes
        -----
        Checks remote receipts and status and may save updated observations
        locally. It does not upload, submit, cancel, download, or delete.
        Present the list and the duplicate-calculation warning to the user
        before passing confirmed candidates to abandon_attempts.
        """
        candidates: list[AbandonmentCandidate] = []
        with store_lock(self.store.root):
            for record in self.list_attempts(
                bundle_id=bundle_id, resource=resource, domain=domain
            ):
                if record.abandoned_at is not None:
                    continue
                self._observe_submission(record, preserve_files=True)
                reason = self._abandonment_reason(record)
                if reason is not None:
                    ids = sorted(self._saved_remote_ids(record.dispatcher))
                    candidates.append(
                        AbandonmentCandidate(
                            bundle_id=record.bundle_id,
                            attempt=record.attempt_folder_name,
                            resource=str(record.resource["name"]),
                            job_id=", ".join(ids) or None,
                            reason=reason,
                            review_token=self._abandonment_token(record),
                        )
                    )
        return candidates

    def abandon_attempts(
        self,
        candidates: list[AbandonmentCandidate],
    ) -> list[JobRecord]:
        """Stop tracking reviewed unresolved attempts without deleting work.

        Parameters
        ----------
        candidates : list[AbandonmentCandidate]
            Exact selection returned by plan_abandonment and confirmed by
            the caller. Empty performs no work. The caller must explain that
            abandonment cannot cancel an unknown remote job and that later
            submission may duplicate a calculation which is still running.

        Returns
        -------
        list[JobRecord]
            Attempts marked with abandoned_at and abandoned_reason. Original
            statuses, remote IDs, errors, and files remain available. These
            attempts no longer block new submissions or participate in
            automatic monitoring, cancellation, recovery, or cleanup.

        Raises
        ------
        ValueError
            If the selection contains duplicates, relevant saved state
            changed since review, or fresh remote evidence makes any attempt
            ineligible.
            No attempt is abandoned in these cases; review a fresh list.
        OSError
            If the store lock, saved history, or a record write fails.
            Records are written individually, so a write failure can leave
            earlier records abandoned. Inspect history before retrying.

        Notes
        -----
        Rechecks all selected records under the store lock before marking
        any. Fresh observations may be saved even when rechecking rejects
        the selection. Statuses, resource settings, saved identifiers, and
        other decision-relevant fields must still match the reviewed record.
        Updated query timestamps, error messages, and observation details do
        not invalidate review by themselves; eligibility is checked again.
        Abandonment records a user decision, not evidence that a remote job
        stopped or that its files were cleaned. No replacement is launched;
        run submit separately when ready.
        """
        if not candidates:
            return []
        with store_lock(self.store.root):
            records = {r.attempt_folder_name: r for r in self.store.records()}
            selected: list[JobRecord] = []
            seen: set[str] = set()
            for candidate in candidates:
                record = records.get(candidate.attempt)
                if (
                    candidate.attempt in seen
                    or record is None
                    or record.bundle_id != candidate.bundle_id
                    or (
                        self._abandonment_token(record)
                        != candidate.review_token
                    )
                    or self._abandonment_reason(record) is None
                ):
                    raise ValueError(
                        "Abandonment selection changed; review a fresh list."
                    )
                seen.add(candidate.attempt)
                selected.append(record)
            for record, candidate in zip(selected, candidates, strict=True):
                self._observe_submission(record, preserve_files=True)
                if (
                    self._abandonment_reason(record) is None
                    or (
                        self._abandonment_token(record)
                        != candidate.review_token
                    )
                ):
                    raise ValueError(
                        "Remote evidence changed; review a fresh abandonment"
                        " list."
                    )
            for record in selected:
                record.abandoned_at = utc_now()
                record.abandoned_reason = (
                    "User explicitly abandoned this unresolved attempt after"
                    " reviewing the warning about possible duplicate jobs."
                )
                self.store.save(record, "abandoned_by_user")
            return selected

    @cache_dataset_hashes()
    def submit_bundles(
        self,
        *,
        resource: RemoteResource | None = None,
        bundle_id: str | None = None,
        resource_name: str | None = None,
        domain: str | None = None,
        preview: bool = False,
        retry_cancelled: bool = False,
    ) -> list[SubmissionDecision]:
        """Start or resume bundles without repeating uncertain launches.

        Parameters
        ----------
        resource : RemoteResource or None, optional
            Settings for new attempts. Required for a bundle with no history;
            otherwise the latest attempt's settings are reused when absent.
            Recovery always uses the interrupted attempt's saved settings.
        bundle_id, resource_name, domain : str or None, optional
            Exact filters; None includes all values. Resource names refer to
            the latest saved attempt, or the supplied profile for new bundles.
            All attempts of each selected bundle are checked for conflicts.
        preview : bool, default False
            Query status and recovery evidence and save local observations,
            but do not upload, launch, cancel, collect, or delete remote files.
            Eligibility does not guarantee that inputs or resources will still
            be available at execution time.
        retry_cancelled : bool, default False
            Allow a new attempt after confirmed cancellation. By default,
            deliberately cancelled work stays cancelled.

        Returns
        -------
        list[SubmissionDecision]
            One explanation per selected bundle, including independent errors.
            No matches returns an empty list. A failed input is reported by
            its path because its bundle ID could not be trusted.

        Raises
        ------
        BlockingIOError
            If another process holds the store lock; retry later.
        ValueError
            If saved job history is corrupt and cannot safely be checked.
        OSError
            If the store or its lock cannot be accessed. On Windows, this
            can also mean another command holds the store lock; retry later.

        Notes
        -----
        Reads root/model/domain/bundle__*/bundle.json under this manager's
        store. Checking and execution share the store lock. Each call creates
        at most one new attempt per bundle. Unknown outcomes block another
        launch unless the user explicitly abandons those attempts, accepting
        possible duplicate calculations. Keep one shared store: independent
        copies cannot coordinate submissions with each other. Completed
        attempts are collected and cleaned before retrying; collection
        errors do not rerun calculations.
        """
        outcomes: list[SubmissionDecision] = []
        with store_lock(self.store.root):
            records = self.store.records()
            seen: set[str] = set()
            for path in sorted(
                self.store.root.glob("*/*/bundle__*/bundle.json")
            ):
                identity = str(path)
                try:
                    bundle = MLFFTrainBundle.model_validate_json(
                        path.read_text(encoding="utf-8"),
                        context={"verify_files": False},
                    )
                    identity = str(bundle.mlff_train_bundle_id)
                    if identity in seen:
                        continue
                    seen.add(identity)
                    history = [r for r in records if r.bundle_id == identity]
                    latest = history[-1] if history else None
                    profile_name = (
                        str(latest.resource["name"])
                        if latest
                        else resource.name if resource else None
                    )
                    if (
                        (bundle_id is not None and identity != bundle_id)
                        or (
                            domain is not None
                            and bundle.training_unit.domain != domain
                        )
                        or (
                            resource_name is not None
                            and profile_name != resource_name
                        )
                    ):
                        continue
                    outcomes.append(
                        self._submit_selected(
                            bundle, history, resource, preview, retry_cancelled
                        )
                    )
                except Exception as error:
                    outcomes.append(
                        SubmissionDecision(
                            bundle_id=identity,
                            action="error",
                            reason=(
                                f"{type(error).__name__}: check bundle inputs"
                                " and saved history."
                            ),
                        )
                    )
        return outcomes

    def _observe_submission(
        self, record: JobRecord, *, preserve_files: bool = False,
    ) -> None:
        """Read remote evidence without launching, collecting, or deleting."""
        if record.abandoned_at is not None:
            return
        cloud_draft = (
            record.resource["backend"] == "Bohrium"
            and record.submission_status == "uploading"
            and bool(record.dispatcher.get("cloud_create_intent"))
        )
        if record.submission_status not in {
            "submitted", "submitting", "uncertain"
        } and not cloud_draft:
            return
        if record.computation_status in {"success", "failed", "cancelled"}:
            return
        backend = None
        try:
            backend = self._connect(record, update_markers=not preserve_files)
            if cloud_draft:
                record.error = None if backend.recover() else (
                    "Cloud draft creation outcome is unknown; attach the"
                    " existing draft ID before resuming."
                )
                self.store.save(
                    record, "cloud_draft_checked",
                    update_markers=not preserve_files,
                )
                return
            if record.submission_status != "submitted":
                if not backend.recover():
                    record.error = (
                        "Submission outcome unknown; inspect the remote job"
                        " and attach its ID."
                    )
                    self.store.save(
                        record, "recovery_inconclusive",
                        update_markers=not preserve_files,
                    )
                    return
                record.submission_status = "submitted"
                self.store.save(
                    record, "recovered_existing_job",
                    update_markers=not preserve_files,
                )
            observation = backend.status()
            record.checked_at = utc_now()
            if observation == "unknown":
                record.error = (
                    "Remote status is unknown; no replacement will be"
                    " launched."
                )
            else:
                record.computation_status = (
                    "cancelled"
                    if record.cancel_requested and observation == "failed"
                    else observation
                )
                record.error = None
            self.store.save(
                record, "submission_checked",
                update_markers=not preserve_files,
            )
        except SubmissionRejected:
            record.submission_status = "rejected"
            record.error = None
            self.store.save(
                record, "rejection_confirmed",
                update_markers=not preserve_files,
            )
        except Exception as error:
            self._error(record, error, update_markers=not preserve_files)
        finally:
            if backend:
                backend.close()

    def _submit_selected(
        self,
        bundle: MLFFTrainBundle,
        history: list[JobRecord],
        resource: RemoteResource | None,
        preview: bool,
        retry_cancelled: bool,
    ) -> SubmissionDecision:
        """Check every attempt and execute at most one launch for a bundle."""
        identity = str(bundle.mlff_train_bundle_id)
        previous_latest = history[-1] if history else None
        history = [r for r in history if r.abandoned_at is None]
        latest = history[-1] if history else previous_latest
        decision = SubmissionDecision(
            bundle_id=identity,
            action="blocked",
            reason="",
            attempt=latest.attempt_folder_name if latest else None,
            resource=(
                str(latest.resource["name"])
                if latest
                else resource.name if resource else None
            ),
        )
        for record in history:
            if (
                not preview
                and record.cancel_requested
                and record.submission_status in {
                    "prepared", "uploading", "uploaded"
                }
            ):
                self._cancel_unsubmitted(record)
            self._observe_submission(record)
        if any(r.computation_status == "success" for r in history):
            if any(
                r.submission_status != "rejected"
                and r.computation_status not in {
                    "success", "failed", "cancelled"
                }
                for r in history
            ):
                decision.reason = (
                    "A successful attempt coexists with unfinished work;"
                    " inspect the other attempts. No new launch."
                )
                return decision
            decision.action = "succeeded"
            decision.reason = "Already succeeded; no new calculation."
            if not preview:
                for record in history:
                    if (
                        record.computation_status == "success"
                        and record.cleanup_status != "complete"
                    ):
                        self._synchronize_one(record, False, False)
                        if record.error:
                            decision.action = "error"
                            decision.reason = (
                                "Calculation succeeded; collection or cleanup"
                                " needs another synchronization."
                            )
            return decision
        uncertain = [
            r for r in history
            if r.submission_status in {"submitting", "uncertain"}
            or (
                r.submission_status == "submitted"
                and (
                    r.computation_status == "unknown"
                    or (
                        r.error
                        and r.computation_status not in {"failed", "cancelled"}
                    )
                )
            )
        ]
        if uncertain:
            decision.reason = (
                "Remote outcome is unresolved; check connectivity or attach"
                " the existing job ID."
            )
            return decision
        active = [
            r for r in history
            if r.submission_status != "rejected"
            and r.computation_status not in {"failed", "cancelled"}
        ]
        if len(active) > 1:
            decision.reason = (
                "Multiple unfinished attempts exist; inspect them before"
                " continuing."
            )
            return decision
        if active:
            record = active[0]
            decision.attempt = record.attempt_folder_name
            decision.resource = str(record.resource["name"])
            if (
                record.resource["backend"] == "Bohrium"
                and record.submission_status == "uploading"
                and record.dispatcher.get("cloud_create_intent")
                and record.error
            ):
                decision.reason = record.error
                return decision
            if record.cancel_requested:
                decision.reason = (
                    "Cancellation is pending; run synchronize_remote to"
                    " confirm it."
                )
                return decision
            if record.submission_status == "submitted":
                decision.action = "monitor"
                decision.reason = (
                    "Existing job is queued or running; no new submission."
                )
                return decision
            if record is not latest:
                decision.reason = (
                    "An older interrupted attempt conflicts with newer"
                    " history; inspect it before resuming."
                )
                return decision
            decision.action = "resume"
            decision.reason = (
                "Continue the interrupted attempt using its saved resource."
            )
            if not preview:
                packaged = load_bundle(
                    record, self.store.path(record), require_packaged=True
                )
                self._launch(record, packaged)
                decision.action = (
                    "resumed"
                    if (
                        record.submission_status == "submitted"
                        and not record.error
                    )
                    else "error"
                )
                if record.error:
                    decision.reason = record.error
            return decision
        if (
            latest
            and latest.abandoned_at is None
            and (
                latest.cancel_requested
                or latest.computation_status == "cancelled"
            )
            and not retry_cancelled
        ):
            decision.action = "cancelled"
            decision.reason = (
                "Cancelled by request; use --retry-cancelled to start a new"
                " attempt."
            )
            return decision
        profile = resource or (
            RemoteResource.model_validate(latest.resource) if latest else None
        )
        if profile is None:
            decision.reason = (
                "No execution profile; supply --resource-file for the first"
                " attempt."
            )
            return decision
        decision.resource = profile.name
        decision.action = "retry" if previous_latest else "new"
        decision.reason = (
            "Collect and clean previous attempts, then create one new attempt."
            if history else "Create a new attempt; abandoned history is retained."
            if previous_latest else "Create the first attempt."
        )
        # In preview mode, the new attempt is not created and no previous work is deleted.
        # Function exits with a decision.
        if preview:
            return decision
        # Check inputs before deleting any previous diagnostics or remote work.
        MLFFTrainBundle.model_validate(bundle.model_dump())
        for record in history:
            if record.cleanup_status == "complete":
                continue
            if record.submission_status == "rejected":
                backend = self._connect(record)
                try:
                    backend.cleanup()
                finally:
                    backend.close()
                record.collection_status = "verified"
                record.cleanup_status = "complete"
                record.error = None
                self.store.save(record, "rejected_submission_retired")
            else:
                self._synchronize_one(record, False, False)
            if record.error or record.cleanup_status != "complete":
                decision.action = "blocked"
                decision.reason = (
                    "Previous results could not be collected or cleaned;"
                    " no new attempt."
                )
                return decision
        # Perform actual submission here.
        attempt = self._submit(bundle, profile)
        decision.attempt = attempt.attempt_folder_name
        decision.action = (
            "submitted"
            if attempt.submission_status == "submitted" and not attempt.error
            else "error"
        )
        decision.reason = (
            "New attempt submitted."
            if decision.action == "submitted"
            else attempt.error or "Submission was not confirmed."
        )
        return decision

    def attach(self, bundle_id: str, attempt: str, job_id: str) -> JobRecord:
        """Associate one interrupted attempt with a verified remote job ID.

        Parameters
        ----------
        bundle_id : str
            Full scientific bundle ID from job list.
        attempt : str
            Exact attempt directory name, such as attempt_001, not a path.
        job_id : str
            Existing scheduler job or Bohrium draft ID. Verify on the remote
            resource that it belongs to this attempt before supplying it.

        Returns
        -------
        JobRecord
            Updated attempt. No upload, launch, cancellation, or deletion is
            performed. A Bohrium upload draft stays in uploading state; run
            submit afterward to continue uploading it.

        Raises
        ------
        ValueError
            If the attempt is absent, ambiguous, local, already resolved,
            or the ID conflicts with available remote evidence.
        OSError
            If saved history or remote evidence cannot be accessed, or the
            store lock is held by another command.
        RuntimeError
            If the backend cannot confirm the supplied association.
        """
        if not job_id.strip():
            raise ValueError("Supply a nonempty remote job ID.")
        with store_lock(self.store.root):
            matches = [
                r for r in self.store.records()
                if r.bundle_id == bundle_id
                and Path(r.attempt_folder_name).name == attempt
            ]
            if len(matches) != 1 or Path(attempt).name != attempt:
                raise ValueError(
                    "Select exactly one existing attempt by bundle ID and"
                    " attempt_NNN name."
                )
            record = matches[0]
            if record.abandoned_at is not None:
                raise ValueError("An abandoned attempt cannot be attached.")
            draft = (
                record.resource["backend"] == "Bohrium"
                and record.submission_status == "uploading"
            )
            if record.resource["backend"] == "Local" or (
                record.submission_status not in {"submitting", "uncertain"}
                and not draft
            ):
                raise ValueError(
                    "Only unresolved remote submissions or Bohrium upload"
                    " drafts can be attached."
                )
            backend = self._connect(record)
            try:
                if not backend.recover(job_id):
                    raise RuntimeError(
                        "Remote job association could not be confirmed."
                    )
                if not draft:
                    record.submission_status = "submitted"
                record.error = None
                self.store.save(record, "attached_existing_job")
            finally:
                backend.close()
            return record

    def watch_submissions(
        self,
        decisions: list[SubmissionDecision],
        interval: float = 60,
    ) -> None:
        """Monitor selected submissions without retrying failed calculations.

        Parameters
        ----------
        decisions : list[SubmissionDecision]
            Results from submit_bundles. Only confirmed submissions, running
            jobs and successful jobs are watched; blocked cases are excluded.
            The eligible attempts are selected once when watching begins.
            Attempts created later are not added to this watch operation.
            Abandoned attempts are excluded, including attempts abandoned
            while this watch operation is running.
        interval : float, default 60
            Seconds between passes; must be positive.

        Raises
        ------
        ValueError
            If interval is not positive or saved history is invalid.
        OSError
            If saved history cannot be accessed or another command holds
            the store lock. Remote operation errors are saved in each
            attempt's record and retried on a later pass.
        KeyboardInterrupt
            If interrupted; remote jobs continue and can be monitored later.
        """
        if interval <= 0:
            raise ValueError("Watch interval must be positive.")
        identities = {
            d.bundle_id for d in decisions
            if d.action in {"submitted", "resumed", "monitor", "succeeded"}
        }
        # Freeze the attempts being watched. Later launches and unresolved
        # historical submissions must not keep this invocation alive forever.
        with store_lock(self.store.root):
            pending = {
                r.attempt_folder_name for r in self.store.records()
                if r.bundle_id in identities
                and r.abandoned_at is None
                and r.submission_status == "submitted"
                and r.cleanup_status != "complete"
            }
        while pending:
            with store_lock(self.store.root):
                for record in self.store.records():
                    if record.attempt_folder_name not in pending:
                        continue
                    if record.abandoned_at is not None:
                        pending.remove(record.attempt_folder_name)
                        continue
                    self._synchronize_one(record, False, False)
                    if record.cleanup_status == "complete":
                        pending.remove(record.attempt_folder_name)
            if pending:
                time.sleep(interval)

    @cache_dataset_hashes()
    def submit(
        self, bundle: MLFFTrainBundle, resource: RemoteResource
    ) -> JobRecord:
        """Create and launch a new attempt, leaving previous attempts
        untouched.

        Parameters
        ----------
        bundle : MLFFTrainBundle
            Local bundle with dataset hashes. Sources are reread and
            validated; nested mutations with stale identities are
            rejected.
        resource : RemoteResource
            Resource and MLFF-specific runtime environment to freeze in
            history.

        Returns
        -------
        JobRecord
            Submitted attempt, or a recoverable attempt containing an
            operation error. Calculation failures are never retried
            automatically. After confirmed remote submission, the local
            attempt's bundle.json is removed; its file locations remain
            in the shared bundle. Local execution retains bundle.json
            until completion cleanup. A single bundle is kept in the
            parent bundle directory and referenced by relative path from
            each job record. It must remain available and keep the same
            bundle identity while those attempts are retained.

        Raises
        ------
        ValueError
            If inputs changed, hashes are absent, or this bundle has a
            successful, active, uncertain or not-yet-collected earlier
            attempt.
        OSError
            If packaging or persistence fails.
        """
        with store_lock(self.store.root):
            return self._submit(bundle, resource)

    def _submit(
        self, bundle: MLFFTrainBundle, resource: RemoteResource
    ) -> JobRecord:
        bundle = MLFFTrainBundle.model_validate(bundle.model_dump())
        if not bundle.training_unit.dataset_sha256:
            raise ValueError(
                "Legacy dataset identity: run job refresh_bundle_identity"
                " before submission."
            )
        bundle.training_unit.root_path = (
            bundle.training_unit.root_path.expanduser().resolve()
        )
        for artifact in bundle.mlff_spec.pretrained_model.artifacts.values():
            artifact.path = artifact.path.expanduser().resolve()
        identity = str(bundle.mlff_train_bundle_id)
        reason = self._blocked(identity, self.store.records())
        if reason:
            raise ValueError(reason)
        effective = resource.model_copy(deep=True)
        family = bundle.mlff_spec.mlff_type
        if effective.resolved_environment is None:
            if family not in effective.environments:
                raise ValueError(
                    f"Resource requires an environments entry for {family}."
                )
            effective.resolved_environment = effective.environments[family]
        if (
            effective.backend in {"Slurm", "PBS"}
            and effective.remote_root is None
        ):
            user_token = hashlib.sha256(
                (effective.username or "").encode()
            ).hexdigest()[:16]
            effective.remote_root = f"/tmp/temper-{user_token}"
            warnings.warn(
                f"No remote_root configured; using {effective.remote_root}."
                " Configure an absolute home or shared scratch directory:"
                " /tmp may be cleared or unavailable to compute nodes.",
                RuntimeWarning,
                stacklevel=2,
            )
        effective.environments = {}
        if (
            effective.backend == "Bohrium"
            and not effective.resolved_environment.image
        ):
            raise ValueError("Bohrium requires an image for this MLFF family.")
        directory = bundle_directory(self.store.root, bundle)
        directory.mkdir(parents=True, exist_ok=True)
        shared = directory / "bundle.json"
        if shared.exists():
            saved = MLFFTrainBundle.model_validate_json(
                shared.read_text(encoding="utf-8")
            )
            if str(saved.mlff_train_bundle_id) != identity:
                raise ValueError(
                    "Existing shared bundle has a different bundle ID."
                )
            bundle = saved
        else:
            bundle.files = None
            write_json_atomically(shared, bundle.model_dump(mode="json"))
        target = next_attempt(directory)
        root, packaged = bundle.write_submit_folder(target)
        if bundle.files is None:
            write_json_atomically(shared, packaged.model_dump(mode="json"))
        record = JobRecord(
            bundle_id=identity,
            attempt_folder_name=root.relative_to(self.store.root).as_posix(),
            relative_source_bundle_path="../bundle.json",
            resource=effective.model_dump(mode="json"),
            dispatcher={"token": uuid4().hex},
        )
        self.store.save(record, "prepared")
        self._launch(record, packaged)
        return record

    @staticmethod
    def _blocked(identity: str, records: list[JobRecord]) -> str | None:
        previous = [
            record for record in records
            if record.bundle_id == identity and record.abandoned_at is None
        ]
        if any(record.computation_status == "success" for record in previous):
            return (
                "Bundle already succeeded; only changed scientific content"
                " permits a new submission."
            )
        if any(
            (
                record.submission_status != "rejected"
                and record.computation_status not in {"failed", "cancelled"}
            )
            or record.cleanup_status != "complete"
            for record in previous
        ):
            return (
                "Bundle has an active, uncertain or uncollected attempt;"
                " synchronize or recover it first."
            )
        return None

    def _connect(
        self, record: JobRecord, *, update_markers: bool = True,
    ) -> _Backend:
        return self._factory(
            record,
            self.store.path(record),
            lambda: self.store.save(
                record, "backend_checkpoint", update_markers=update_markers,
            ),
        )

    def _launch(self, record: JobRecord, bundle: MLFFTrainBundle) -> None:
        if record.abandoned_at is not None:
            raise ValueError("An abandoned attempt cannot be launched.")
        backend = None
        try:
            backend = self._connect(record)
            if record.submission_status in {"prepared", "uploading"}:
                verify_dataset_and_model_hashes(
                    bundle, self.store.path(record)
                )
                record.submission_status = "uploading"
                self.store.save(record, "uploading")
                backend.upload()
                record.submission_status = "uploaded"
                self.store.save(record, "uploaded")
            if record.resource["backend"] != "Local":
                self._remove_copies(record, bundle)
            record.submission_status = "submitting"
            self.store.save(record, "submission_intent")
            backend.submit()
            record.submission_status = "submitted"
            record.computation_status = "waiting"
            record.error = None
            self.store.save(record, "submitted")
            if record.resource["backend"] != "Local":
                self._remove_bundle_copy(record)
        except Exception as error:
            if record.submission_status == "submitting":
                record.submission_status = (
                    "rejected"
                    if isinstance(error, SubmissionRejected)
                    else "uncertain"
                )
            self._error(record, error)
        finally:
            if backend:
                backend.close()

    def _remove_copies(
        self, record: JobRecord, bundle: MLFFTrainBundle
    ) -> None:
        """Remove copied datasets/models and upload archives, retaining
        scripts/configs."""
        if record.abandoned_at is not None:
            return
        root = self.store.path(record)
        for name in get_dataset_and_model_hashes(bundle):
            (root / name).unlink(missing_ok=True)
        # Dispatcher archives contain the same large copies, not new
        # results.
        for suffix in ("zip", "tgz", "tar"):
            for path in root.glob("*." + suffix):
                if len(path.stem) == 40 and all(
                    c in "0123456789abcdef" for c in path.stem
                ):
                    path.unlink()

    def _remove_bundle_copy(self, record: JobRecord) -> None:
        """Remove the temporary copy once the shared bundle can serve
        the attempt."""
        if record.abandoned_at is not None:
            return
        root = self.store.path(record)
        load_bundle(record, root, require_packaged=True)
        (root / "bundle.json").unlink(missing_ok=True)

    def _error(
        self, record: JobRecord, error: Exception, *,
        update_markers: bool = True,
    ) -> None:
        # Backend exceptions may contain credentials or signed download
        # URLs.
        record.error = (
            f"{type(error).__name__} during"
            f" {record.submission_status}/{record.collection_status}/"
            f"{record.cleanup_status};"
            " retry synchronization or inspect the remote resource."
        )
        self.store.save(
            record, "operation_error", update_markers=update_markers,
        )

    def synchronize(
        self,
        *,
        bundle_id: str | None = None,
        resource: str | None = None,
        domain: str | None = None,
        refresh_only: bool = False,
        download_only: bool = False,
    ) -> list[JobRecord]:
        """Check selected attempts once and optionally collect and clean
        them.

        Parameters
        ----------
        bundle_id, resource, domain : str or None, optional
            Exact filters. None selects all values. Resource is the
            profile name.
        refresh_only : bool, default False
            Query and persist status only; no upload, submission,
            cancellation, result download or remote deletion.
        download_only : bool, default False
            Collect results from remote but postpone remote running
            directory cleanup.

        Returns
        -------
        list[JobRecord]
            Selected refreshed records, including independent operation
            errors. One unreachable resource does not stop processing
            other attempts. Abandoned attempts remain in the returned history
            but are not queried, changed, or cleaned up.

        Notes
        -----
        Confirmed remote submissions no longer need a local bundle.json;
        it is removed even when only refreshing status. For local
        execution, that file is retained until completion cleanup.
        """
        with store_lock(self.store.root):
            records = self.list_attempts(
                bundle_id=bundle_id, resource=resource, domain=domain
            )
            for record in records:
                if record.abandoned_at is not None:
                    continue
                self.store.repair_markers(record)
                if record.cleanup_status == "complete":
                    try:
                        self._remove_bundle_copy(record)
                    except Exception as error:
                        self._error(record, error)
                    continue
                self._synchronize_one(record, refresh_only, download_only)
            return records

    def _synchronize_one(
        self, record: JobRecord, refresh_only: bool, download_only: bool
    ) -> None:
        """Synchronize one attempt.

        Refresh the record content with current status.
        """
        if record.abandoned_at is not None:
            return
        if record.submission_status != "submitted":
            if (
                record.cancel_requested
                and record.submission_status in {"prepared", "uploading", "uploaded"}
                and not refresh_only
                and not download_only
            ):
                self._cancel_unsubmitted(record)
            return  # Recovery is explicit, never a hidden calculation retry.
        backend = None
        try:
            backend = self._connect(record)
            if record.resource["backend"] != "Local":
                self._remove_bundle_copy(record)
            if record.computation_status not in {
                "success",
                "failed",
                "cancelled",
            }:
                observation = backend.status()
                record.checked_at = utc_now()
                if observation != "unknown":
                    record.computation_status = observation
                if record.cancel_requested and observation == "failed":
                    record.computation_status = "cancelled"
                record.error = (
                    "Remote status is unknown; retaining previous observation."
                    if observation == "unknown"
                    else None
                )
                self.store.save(record, "status_checked")
            if refresh_only:
                return
            if record.cancel_requested and record.computation_status not in {
                "success",
                "failed",
                "cancelled",
            }:
                backend.cancel()
                self.store.save(record, "cancellation_requested")
                return
            if record.computation_status not in {
                "success",
                "failed",
                "cancelled",
            }:
                return
            root = self.store.path(record)
            bundle = load_bundle(record, root, require_packaged=True)
            _, required = get_download_files(bundle)
            required = (
                required if record.computation_status == "success" else []
            )
            if record.collection_status != "verified":
                record.downloaded_file_hashes = backend.collect()
                verify_results(root, record.downloaded_file_hashes, required)
                record.collection_status = "verified"
                record.error = None
                self.store.save(record, "collected_and_verified")
            if not download_only:
                verify_results(root, record.downloaded_file_hashes, required)
                backend.cleanup()
                self._remove_bundle_copy(record)
                record.cleanup_status = "complete"
                record.error = None
                self.store.save(record, "remote_cleanup_complete")
        except Exception as error:
            self._error(record, error)
        finally:
            if backend:
                backend.close()

    def list_attempts(
        self,
        *,
        bundle_id: str | None = None,
        resource: str | None = None,
        domain: str | None = None,
        failed: bool = False,
    ) -> list[JobRecord]:
        """Read local attempts matching exact filters.

        Parameters
        ----------
        bundle_id, resource, domain : str or None, optional
            Full bundle ID, profile name and data domain. None leaves a
            filter unset.
        failed : bool, default False
            Return failed computations only; operational errors are
            separate.

        Returns
        -------
        list[JobRecord]
            Matching attempts in chronological order, without remote
            queries.
        """
        return [
            r
            for r in self.store.records()
            if (bundle_id is None or r.bundle_id == bundle_id)
            and (resource is None or r.resource["name"] == resource)
            and (
                domain is None
                or load_bundle(r, self.store.path(r)).training_unit.domain
                == domain
            )
            and (not failed or r.computation_status == "failed")
        ]

    @cache_dataset_hashes()
    def resubmit(
        self,
        *,
        bundle_id: str | None = None,
        resource: str | None = None,
        domain: str | None = None,
        dry_run: bool = False,
        replacement_resource: RemoteResource | None = None,
    ) -> list[dict[str, str]]:
        """Retry rejected submissions and failed calculations once per
        bundle.

        Parameters
        ----------
        bundle_id, resource, domain : str or None, optional
            Exact filters on saved attempts; None includes all values. A
            later attempt outside these filters prevents retrying an
            older attempt.
        dry_run : bool, default False
            Report eligibility from saved history without contacting a
            resource, cleaning files, or launching jobs. Input
            availability is not checked.
        replacement_resource : RemoteResource or None, optional
            Alternative execution resource; None reuses each saved
            profile. Resource changes do not change the scientific
            identity.

        Returns
        -------
        list[dict[str, str]]
            One result per bundle: ID, action and reason or attempt
            path. A later successful attempt always excludes earlier
            failures of that bundle. Unknown submission outcomes are
            skipped: use recover to establish whether a job exists
            before retrying. Rejected submissions retain their
            diagnostics and are cleaned before a new attempt is created.
            Abandoned attempts are excluded from retry selection and do not
            prevent another attempt; their saved records remain unchanged.

        Notes
        -----
        Selection, cleanup, and creation run under the job-store lock. A
        failure for one bundle is reported without stopping the
        remaining bundles. Rejection means no calculation started; its
        computation status remains unknown rather than being recorded as
        a failed calculation.
        """
        with store_lock(self.store.root):
            all_records = [
                record for record in self.store.records()
                if record.abandoned_at is None
            ]
            selected = [
                record for record in self.list_attempts(
                    bundle_id=bundle_id, resource=resource, domain=domain
                )
                if record.abandoned_at is None
            ]
            latest = {record.bundle_id: record for record in selected}
            latest_all = {record.bundle_id: record for record in all_records}
            outcomes = []
            for identity, record in latest.items():
                rejected = record.submission_status == "rejected"
                reason = self._blocked(
                    identity,
                    [
                        r
                        for r in all_records
                        if not (
                            rejected
                            and r.attempt_folder_name
                            == record.attempt_folder_name
                            and r.bundle_id == identity
                        )
                    ],
                )
                if (
                    latest_all[identity].attempt_folder_name
                    != record.attempt_folder_name
                ):
                    reason = (
                        "A newer attempt does not match the selected filters."
                    )
                elif not rejected and record.computation_status != "failed":
                    reason = (
                        reason
                        or "The latest attempt is not a confirmed failure."
                    )
                if reason:
                    outcomes.append(
                        {
                            "bundle_id": identity,
                            "action": "skipped",
                            "reason": reason,
                        }
                    )
                    continue
                if dry_run:
                    outcomes.append(
                        {"bundle_id": identity, "action": "eligible"}
                    )
                    continue
                try:
                    bundle = load_bundle(
                        record, self.store.path(record), verify_files=True
                    )
                    profile = (
                        replacement_resource
                        or RemoteResource.model_validate(record.resource)
                    )
                    if rejected and record.cleanup_status != "complete":
                        backend = self._connect(record)
                        try:
                            backend.cleanup()
                        finally:
                            backend.close()
                        record.collection_status = "verified"
                        record.cleanup_status = "complete"
                        self.store.save(record, "rejected_submission_retired")
                    attempt = self._submit(bundle, profile)
                    outcomes.append(
                        {
                            "bundle_id": identity,
                            "action": attempt.submission_status,
                            "attempt": attempt.attempt_folder_name,
                        }
                    )
                    all_records.append(attempt)
                except Exception as error:
                    self._error(record, error)
                    outcomes.append(
                        {
                            "bundle_id": identity,
                            "action": "error",
                            "reason": (
                                type(error).__name__
                                + ": input validation, cleanup or submission"
                                " preparation failed"
                            ),
                        }
                    )
            return outcomes

    def cancel(self, bundle_id: str) -> list[JobRecord]:
        """Persist cancellation intent and request it for active
        attempts.

        Parameters
        ----------
        bundle_id : str
            Exact full bundle ID. Completed and abandoned attempts are left
            alone, including their saved statuses and remote jobs.

        Returns
        -------
        list[JobRecord]
            Updated attempts. Synchronization confirms termination and
            collects diagnostics; stopping a watcher never calls this
            method. Abandoned attempts are returned unchanged for inspection.
        """
        with store_lock(self.store.root):
            records = self.list_attempts(bundle_id=bundle_id)
            for record in records:
                if record.abandoned_at is not None:
                    continue
                if record.computation_status in {
                    "success",
                    "failed",
                    "cancelled",
                }:
                    continue
                record.cancel_requested = True
                self.store.save(record, "cancellation_intent")
                if record.submission_status in {
                    "prepared",
                    "uploading",
                    "uploaded",
                }:
                    self._cancel_unsubmitted(record)
                else:
                    self._synchronize_one(record, False, False)
            return records

    def _cancel_unsubmitted(self, record: JobRecord) -> None:
        if record.abandoned_at is not None:
            return
        backend = None
        try:
            if record.submission_status != "prepared":
                backend = self._connect(record)
                if (
                    record.resource["backend"] == "Bohrium"
                    and not backend.recover()
                ):
                    raise ValueError(
                        "Recover the cloud draft ID before deleting its"
                        " upload."
                    )
                backend.cleanup()
            record.computation_status = "cancelled"
            record.collection_status = "verified"
            record.cleanup_status = "complete"
            record.error = None
            self.store.save(record, "cancelled_before_launch")
            bundle = load_bundle(
                record, self.store.path(record), require_packaged=True
            )
            self._remove_copies(record, bundle)
            self._remove_bundle_copy(record)
        except Exception as error:
            self._error(record, error)
        finally:
            if backend:
                backend.close()

    def recover(
        self, bundle_id: str, job_id: str | None = None
    ) -> list[JobRecord]:
        """Resume interrupted submission operations without rerunning a
        calculation.

        Parameters
        ----------
        bundle_id : str
            Exact full bundle ID.
        job_id : str or None, optional
            Existing remote ID verified by the user when the receipt is
            absent. This attaches an uncertain attempt; it never
            launches a replacement.

        Returns
        -------
        list[JobRecord]
            Recovered or still-uncertain records. Prepared/uploaded SSH
            attempts can finish their original submission; an uncertain
            launch only reads remote evidence. A failed calculation
            requires resubmit. Abandoned attempts are returned unchanged;
            no remote operation is performed for them.
        """
        with store_lock(self.store.root):
            records = self.list_attempts(bundle_id=bundle_id)
            for record in records:
                if record.abandoned_at is not None:
                    continue
                if record.submission_status in {
                    "submitted",
                    "rejected",
                } or record.computation_status in {
                    "success",
                    "failed",
                    "cancelled",
                }:
                    continue
                if (
                    record.submission_status == "uploading"
                    and record.resource["backend"] == "Bohrium"
                    and job_id
                ):
                    backend = self._connect(record)
                    try:
                        backend.recover(job_id)
                        self.store.save(record, "recovered_cloud_draft")
                    finally:
                        backend.close()
                if record.cancel_requested and record.submission_status in {
                    "prepared",
                    "uploading",
                    "uploaded",
                }:
                    self._cancel_unsubmitted(record)
                    continue
                if record.submission_status in {
                    "prepared",
                    "uploaded",
                    "uploading",
                }:
                    bundle = load_bundle(
                        record, self.store.path(record), require_packaged=True
                    )
                    self._launch(record, bundle)
                    continue
                backend = None
                try:
                    backend = self._connect(record)
                    if (
                        backend.recover(job_id)
                        and record.submission_status != "uploading"
                    ):
                        record.submission_status = "submitted"
                        record.error = None
                        self.store.save(record, "recovered_existing_job")
                        if record.resource["backend"] != "Local":
                            self._remove_bundle_copy(record)
                    else:
                        record.error = (
                            "No confirmed launch receipt; inspect the remote"
                            " job before recovery."
                        )
                        self.store.save(record, "recovery_inconclusive")
                except Exception as error:
                    if isinstance(error, SubmissionRejected):
                        record.submission_status = "rejected"
                    self._error(record, error)
                finally:
                    if backend:
                        backend.close()
            return records

    def watch(
        self,
        interval: float = 60,
        *,
        bundle_id: str | None = None,
        resource: str | None = None,
        domain: str | None = None,
        keep_remote_files: bool = False,
    ) -> None:
        """Synchronize repeatedly in the foreground until all selected
        jobs are collected and optionally cleaned up.

        Parameters
        ----------
        interval : float, default 60
            Seconds between synchronization passes; must be positive.
        bundle_id, resource, domain : str or None, optional
            Exact selection filters. None includes all values. Abandoned
            attempts are excluded from monitoring and the completion check.
        keep_remote_files : bool, default False
            Retain remote working files after collecting and verifying
            results. When True, stop once every selected attempt has
            verified results; otherwise wait for cleanup to complete.
            A later synchronization without download_only performs cleanup.

        Raises
        ------
        ValueError
            If interval is not positive.
        KeyboardInterrupt
            When stopped by the user. Remote calculations continue, and
            another watch or synchronization command resumes from JSON
            history.
        """
        if interval <= 0:
            raise ValueError("Watch interval must be positive.")
        while True:
            records = [
                record for record in self.synchronize(
                    bundle_id=bundle_id,
                    resource=resource,
                    domain=domain,
                    download_only=keep_remote_files,
                )
                if record.abandoned_at is None
            ]
            logger.info(
                "Job synchronization: %d attempts, %d collected, %d cleaned,"
                " %d operational errors",
                len(records),
                sum(r.collection_status == "verified" for r in records),
                sum(r.cleanup_status == "complete" for r in records),
                sum(r.error is not None for r in records),
            )
            if all(
                record.collection_status == "verified"
                if keep_remote_files
                else record.cleanup_status == "complete"
                for record in records
            ):
                return
            time.sleep(interval)
