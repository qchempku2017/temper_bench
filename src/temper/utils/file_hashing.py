"""Read and verify scientific inputs without retaining file contents in
memory."""

from __future__ import annotations

import hashlib
from pathlib import Path
from contextlib import contextmanager
from contextvars import ContextVar
from collections.abc import Iterator

_verification_cache: ContextVar[dict[Path, str] | None] = ContextVar(
    "dataset_hashes", default=None
)


@contextmanager
def hashing_session() -> Iterator[dict[Path, str]]:
    """Share dataset hashes only for one packaging or submission
    operation.

    Yields
    ------
    dict[Path, str]
        Per-operation digest cache. Nested operations share it; leaving
        the outer operation discards it. File copies still hash the
        bytes written, and a later independent JSON load reads every
        referenced file again.
    """
    current = _verification_cache.get()
    if current is not None:
        yield current
        return
    cache: dict[Path, str] = {}
    token = _verification_cache.set(cache)
    try:
        yield cache
    finally:
        _verification_cache.reset(token)


def file_sha256(path: str | Path) -> str:
    """Hash the complete bytes of a file.

    Parameters
    ----------
    path : str or Path
        File to read; relative paths use the current working directory.

    Returns
    -------
    str
        Lowercase SHA-256 digest. Each call reads the file again, even
        when its modification time and size have not changed.

    Raises
    ------
    OSError
        If the file cannot be read.
    """
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def copy_verified(source: Path, destination: Path, expected: str) -> None:
    """Copy input bytes and check the bytes actually written.

    Parameters
    ----------
    source : Path
        Original input, resolved by the caller.
    destination : Path
        New copied input. Parent directories are created automatically.
    expected : str
        SHA-256 recorded when the scientific bundle was created.

    Raises
    ------
    ValueError
        If the copied bytes differ; the incomplete destination is
        removed.
    OSError
        If reading or writing fails. The original is never modified.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    created = False
    try:
        with source.open("rb") as incoming, destination.open("xb") as outgoing:
            created = True
            while block := incoming.read(1024 * 1024):
                digest.update(block)
                outgoing.write(block)
        if digest.hexdigest() != expected:
            raise ValueError(
                f"Input changed after its identity was recorded: {source}"
            )
    except BaseException:
        if created:
            destination.unlink(missing_ok=True)
        raise
