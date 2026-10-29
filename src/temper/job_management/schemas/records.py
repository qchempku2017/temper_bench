"""Persisted execution state; records contain no authentication
credentials."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, JsonValue


def utc_now() -> str:
    """Return the current UTC timestamp.

    Returns
    -------
    str
        ISO 8601 timestamp including the UTC offset.
    """
    return datetime.now(timezone.utc).isoformat()


ComputationStatus = Literal[
    "unknown", "waiting", "running", "success", "failed", "cancelled"
]


class JobRecord(BaseModel):
    """Store one attempt independently of the remote scheduler.

    One job record describes one submission attempt of a bundle and its
    attempt directory below the bundle root.

    Parameters
    ----------
    **data : object
        Validated values for the attributes below. Paths in dispatcher
        state are implementation details; attempt paths are relative to
        the store.

    Attributes
    ----------
    schema_version : int
        Record format version, currently 3. Bundles and their file
        locations are stored in the shared bundle.json, not in this
        record.
    bundle_id : str
        Full MLFF train bundle identity; unchanged between retries. Used
        for validating whether the bundled dataset and model have
        changed.
    attempt_folder_name : str
        Relative attempt directory below the bundle root.
    resource : dict[str, JsonValue]
        Resource settings saved when the attempt was created. Secrets
        are excluded; credentials are referenced by environment variable
        name so they can be supplied again when reconnecting.
    relative_source_bundle_path : str
        Bundle path relative to the attempt directory, normally
        ../bundle.json. The bundle is shared by all attempts and must
        retain this bundle ID.
    submission_status : str
        prepared, uploading, uploaded, submitting, submitted, rejected
        or uncertain. Rejected means failed preflight or a confirmed
        scheduler policy/input rejection. A nonzero client exit with
        lost-response diagnostics remains uncertain because the
        scheduler may already have accepted the job. Uncertain
        submissions must be reconciled before another launch.
    computation_status : str
        unknown, waiting, running, success, failed or cancelled.
    collection_status : str
        pending or verified. Verified means selected files passed hash
        checks.
    cleanup_status : str
        pending or complete; remote deletion follows verified
        collection.
    dispatcher : dict[str, JsonValue]
        Saved DPDispatcher connection and scheduler state, with secrets
        removed. Used to reconnect to the same attempt after restarting
        the submission process.
    downloaded_file_hashes : dict[str, str]
        Retained bundle-relative filenames and SHA-256 values, including
        files verified in place for local jobs. Empty until collection
        succeeds.
    cancel_requested : bool
        Durable cancellation intent, written before contacting the
        scheduler.
    created_at, updated_at, checked_at : str or None
        UTC timestamps; checked_at is absent until a successful
        observation.
    error : str or None
        Last operational error without credentials; absent after
        success.
    observation : dict[str, JsonValue]
        Latest non-secret scheduler state or Bohrium status/exit code.
        Signed result URLs and full provider responses are deliberately
        excluded.
    """

    # Model is not frozen so that store can update status.
    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    schema_version: Literal[3] = 3
    bundle_id: str
    attempt_folder_name: str
    resource: dict[str, JsonValue]
    relative_source_bundle_path: str = "../bundle.json"
    submission_status: Literal[
        "prepared",
        "uploading",
        "uploaded",
        "submitting",
        "submitted",
        "rejected",
        "uncertain",
    ] = "prepared"
    computation_status: ComputationStatus = "unknown"
    collection_status: Literal["pending", "verified"] = "pending"
    cleanup_status: Literal["pending", "complete"] = "pending"
    dispatcher: dict[str, JsonValue] = Field(default_factory=dict)
    downloaded_file_hashes: dict[str, str] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("downloaded_file_hashes", "manifest"),
    )
    cancel_requested: bool = False
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)
    checked_at: str | None = None
    error: str | None = None
    observation: dict[str, JsonValue] = Field(default_factory=dict)
