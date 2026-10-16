"""Local JSON job history. The per-attempt job.json is authoritative."""
from __future__ import annotations

import json
import os
from pathlib import Path
from temper.job_management.paths import bundle_root
from temper.job_management.storage_utils import atomic_json
from temper.job_management.records import JobRecord, utc_now


class JobStore:
    """Read and write job records below one bundle root.

    Parameters
    ----------
    root : str, Path or None, optional
        Root resolved by bundle_root; defaults to DEFAULT_BUNDLE_WRITE_ROOT or
        ./bundles. Keep this directory to preserve successful-bundle history.

    Attributes
    ----------
    root : Path
        Absolute local store directory. Mutating operations require callers
        to hold store_lock(root); JobManager does this automatically.
    """
    def __init__(self, root: str | Path | None = None) -> None:
        self.root: Path = bundle_root(root)

    def path(self, record: JobRecord) -> Path:
        """Resolve an attempt and reject paths outside the store.

        Parameters
        ----------
        record : JobRecord
            Record containing a relative attempt directory.

        Returns
        -------
        Path
            Absolute attempt directory.

        Raises
        ------
        ValueError
            If the record points outside the store or to its root.
        """
        relative = Path(record.attempt_folder_name)
        result = (self.root / relative).resolve()
        if relative.is_absolute() or result == self.root or not result.is_relative_to(self.root):
            raise ValueError("Attempt path must remain below the job store root.")
        return result

    # Comment: I'm not sure whether the current records reading and updating mechanism is still efficient enough
    #   when the numeber of bundles and attempts grows to thousands. Maybe we need to consider more efficient
    #   storage solutions, such as one unified jsonl file for all attempts, while keeping the current job.json
    #   and events.jsonl under each attempt folder for convenient debugging of the users only.
    def records(self) -> list[JobRecord]:
        """Read all attempts without contacting remote resources.

        Returns
        -------
        list[JobRecord]
            Records sorted by creation time. Corrupt records raise an error
            instead of being skipped, so duplicate-submission checks remain safe.

        Raises
        ------
        ValueError
            If a record is invalid or stored at a different path than declared.
        """
        result = []
        for path in self.root.glob("*/*/bundle__*/attempt_*/job.json"):
            record = JobRecord.model_validate_json(path.read_text(encoding="utf-8"))
            if self.path(record) != path.parent.resolve():
                raise ValueError(f"Job record path mismatch: {path}")
            result.append(record)
        return sorted(result, key=lambda item: (item.created_at, item.attempt_folder_name))

    def save(self, record: JobRecord, event: str) -> None:
        """Commit one record, repair markers, and append a small event entry.

        Parameters
        ----------
        record : JobRecord
            State to persist. Its attempt directory must already exist.
        event : str
            Non-secret description, such as uploaded or collected. No backend
            exception bodies or credential values should be passed here.

        Raises
        ------
        OSError
            If persistence fails. A later synchronization repairs markers from
            job.json if interruption occurred between writing these files.
        """
        record.updated_at = utc_now()
        root = self.path(record)
        atomic_json(root / "job.json", record.model_dump(mode="json"))
        self.repair_markers(record)
        with (root / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"time": record.updated_at, "event": event}) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def repair_markers(self, record: JobRecord) -> None:
        """Make convenient status marker files agree with the job record.

        Parameters
        ----------
        record : JobRecord
            Confirmed computation and collection status. Success requires
            verified results; failed and cancelled can precede collection.
        """
        expected = record.computation_status if record.computation_status in {"failed", "cancelled"} else None
        if record.computation_status == "success" and record.collection_status == "verified":
            expected = "success"
        for name in ("success", "failed", "cancelled"):
            marker = self.path(record) / name
            if name == expected:
                marker.touch(exist_ok=True)
            else:
                marker.unlink(missing_ok=True)
