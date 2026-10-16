"""Persisted execution state; records contain no authentication credentials."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, JsonValue


def utc_now() -> str:
    """Return the current UTC timestamp.

    Returns
    -------
    str
        ISO 8601 timestamp including the UTC offset.
    """
    return datetime.now(timezone.utc).isoformat()


class JobRecord(BaseModel):
    """Store one attempt independently of the remote scheduler.

    One job record correspond to one submission attempt of a bundle, and therefore
     one attempt directory below the bundle root.

    Parameters
    ----------
    **data : object
        Validated values for the attributes below. Paths in dispatcher state
        are implementation details; attempt paths are relative to the store.

    Attributes
    ----------
    schema_version : int
        Record format version, currently 1.
    bundle_id : str
        Full scientific bundle identity; unchanged between retries.
    attempt_folder_name : str
        Relative attempt directory below the bundle root.
    resource : dict[str, JsonValue]
        Frozen non-secret resource settings; credential environment names only.
    recipe : dict[str, JsonValue]
        Original local recipe, including source paths and content hashes.
    submission_status : str
        prepared, uploading, uploaded, submitting, submitted or uncertain.
        Uncertain submissions must be reconciled before another launch.
    computation_status : str
        unknown, waiting, running, success, failed or cancelled.
    collection_status : str
        pending or verified. Verified means selected files passed hash checks.
    cleanup_status : str
        pending or complete; remote deletion follows verified collection.
    dispatcher : dict[str, JsonValue]
        Reconstructible dpdispatcher state with secrets removed.
    manifest : dict[str, str]
        Downloaded bundle-relative filenames and SHA-256 values.
    cancel_requested : bool
        Durable cancellation intent, written before contacting the scheduler.
    created_at, updated_at, checked_at : str or None
        UTC timestamps; checked_at is absent until a successful observation.
    error : str or None
        Last operational error without credentials; absent after success.
    observation : dict[str, JsonValue]
        Latest non-secret scheduler state or Bohrium status/exit code. Signed
        result URLs and full provider responses are deliberately excluded.
    """
    # Model is not frozen so that store can update status.
    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    schema_version: Literal[1] = 1
    bundle_id: str
    attempt_folder_name: str
    resource: dict[str, JsonValue]
    # Comment: what does a recipe mean in this context? must explain in docstring.
    recipe: dict[str, JsonValue]
    submission_status: Literal["prepared", "uploading", "uploaded", "submitting", "submitted", "uncertain"] = "prepared"
    computation_status: Literal["unknown", "waiting", "running", "success", "failed", "cancelled"] = "unknown"
    collection_status: Literal["pending", "verified"] = "pending"
    cleanup_status: Literal["pending", "complete"] = "pending"
    dispatcher: dict[str, JsonValue] = Field(default_factory=dict)
    # Comment: the name manifest is too professional, should be something more pedagogical and straightforward.
    manifest: dict[str, str] = Field(default_factory=dict)
    cancel_requested: bool = False
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)
    checked_at: str | None = None
    error: str | None = None
    observation: dict[str, JsonValue] = Field(default_factory=dict)
