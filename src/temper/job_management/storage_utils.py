"""Small, crash-recoverable JSON records and operating-system file
locks."""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import tempfile
from collections.abc import Iterator


@contextmanager
def store_lock(root: Path) -> Iterator[None]:
    """Exclusively lock a local bundle root until the caller exits.

    This is only an advisory lock. Therefore, all TEMPER job status
    writers must use it to avoid racing.

    Parameters
    ----------
    root : Path
        Store directory. It is created if necessary. All TEMPER writers
        must use this lock; the operating system releases it after a
        process dies.

    Yields
    ------
    None
        Permission to read and change records without another writer
        racing.

    Raises
    ------
    BlockingIOError
        If another TEMPER command holds the lock; retry that command
        later.
    """
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".temper.lock").open("a+b") as stream:
        # Adds an exclusive lock file here. If a process acquires the
        # lock, the OS
        # will prohibit any other process that explicitly uses this
        # context manager
        # from acquiring the lock again. Therefore, this other process
        # will raise a BlockingIOError.
        # And any code wrapped in this context manager will not be
        # executed until the lock is released,
        # i.e., the process that holds the lock exits or releases it.
        stream.seek(0, 2)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            # Lock the job store until the caller exits. The lock is
            # released if the process dies.
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def write_json_atomically(path: Path, value: object) -> None:
    """Write JSON through a temporary file before replacing the destination.

    Parameters
    ----------
    path : Path
        Destination to create or replace, whose parent must already exist.
        Relative paths are resolved against the current working directory.
    value : object
        JSON-serializable data, written as UTF-8 with two-space indentation
        and sorted keys. NaN and infinite floating-point values are rejected.

    Raises
    ------
    TypeError
        If the value contains an object that cannot be serialized as JSON.
    ValueError
        If the value contains NaN, infinity, or a circular reference.
    OSError
        If writing, flushing or replacing fails; the previous file is
        retained until the replacement succeeds. Filesystem durability
        still depends on the operating system and storage hardware.

    Notes
    -----
    The temporary file is written in the destination directory and flushed
    and synchronized before replacement. Readers see either the previous
    file or the complete replacement, never a partially written replacement.
    On non-Windows systems, the parent directory is also synchronized after
    replacement; failure at that stage leaves the new file in place.
    Temporary files are removed on exit. This function does not lock out
    other writers; use ``store_lock`` when coordinating changes to the store.
    """
    descriptor, temporary = tempfile.mkstemp(
        prefix=".temper-", dir=path.parent
    )
    try:
        with os.fdopen(
            descriptor, "w", encoding="utf-8", newline="\n"
        ) as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if os.name != "nt":
            descriptor = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    finally:
        Path(temporary).unlink(missing_ok=True)
