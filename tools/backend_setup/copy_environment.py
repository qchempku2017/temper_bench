"""Copy a relocated environment onto storage with slow file operations.

Use this optional installation tool after unpacking a conda-pack archive on
local scratch. The archive must already have been created with --dest-prefix
set to the permanent destination. This tool copies files; it does not rewrite
embedded environment paths or validate GPU execution.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import shutil


def copy_environment(
    source: Path, destination: Path, workers: int = 16
) -> int:
    """Copy an environment into a new directory using concurrent file IO.

    Parameters
    ----------
    source : Path
        Unpacked environment directory. Relative paths resolve against
        the current working directory. Its contents must remain unchanged
        during copying. It is never deleted or modified.
    destination : Path
        New permanent directory, resolved against the working directory.
        It must not exist or be inside source. Embedded paths must already
        refer to this destination, for example through conda-pack's
        --dest-prefix option before unpacking the source tree.
    workers : int, default 16
        Number of concurrent file operations, between 1 and 64. Reduce this
        on sites with stricter filesystem concurrency limits.

    Returns
    -------
    int
        Number of regular files and symbolic links copied. Empty directories
        are also copied. Permissions and timestamps are preserved. Symbolic
        links retain their target text; hard links become independent files.

    Raises
    ------
    ValueError
        If workers is outside the accepted range or destination is inside
        source.
    FileExistsError
        If destination already exists; existing installations are never
        merged or overwritten.
    OSError
        If source cannot be traversed or a copy fails. A partial destination
        remains for diagnosis and must not be activated as a complete
        environment. No installation checks are suppressed.
    """
    source = source.resolve(strict=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    destination = destination.resolve()
    if not 1 <= workers <= 64 or destination.is_relative_to(source):
        raise ValueError("Use 1–64 workers and a destination outside source.")
    if not source.is_dir():
        raise NotADirectoryError(source)
    destination.mkdir(parents=True, exist_ok=False)
    directories = [source]
    entries: list[Path] = []

    def traversal_error(error: OSError) -> None:
        raise error

    for parent, names, files in os.walk(
        source, followlinks=False, onerror=traversal_error
    ):
        parent_path = Path(parent)
        for name in list(names):
            path = parent_path / name
            if path.is_symlink():
                entries.append(path)
                names.remove(name)
            else:
                directories.append(path)
        entries.extend(parent_path / name for name in files)

    def make_directory(path: Path) -> None:
        (destination / path.relative_to(source)).mkdir(
            parents=True, exist_ok=True
        )

    def copy_entry(path: Path) -> None:
        shutil.copy2(
            path, destination / path.relative_to(source), follow_symlinks=False
        )

    def copy_directory_metadata(path: Path) -> None:
        shutil.copystat(path, destination / path.relative_to(source))

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for _ in executor.map(make_directory, directories):
            pass
        for index, _ in enumerate(executor.map(copy_entry, entries), 1):
            if index % 1000 == 0:
                print(
                    f"Copied {index}/{len(entries)} files and links",
                    flush=True,
                )
        for _ in executor.map(copy_directory_metadata, directories):
            pass
    return len(entries)


def main() -> None:
    """Copy command-line source and destination paths, reporting completion.

    Accept SOURCE DESTINATION and optional --workers. Copy errors propagate
    with a nonzero exit status; see copy_environment for failure semantics.

    Raises
    ------
    OSError
        If source cannot be read, destination exists, or copying fails.
    ValueError
        If concurrency or destination placement is invalid.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--workers", type=int, default=16)
    arguments = parser.parse_args()
    count = copy_environment(
        arguments.source, arguments.destination, arguments.workers
    )
    print(f"Completed {count} files and links")


if __name__ == "__main__":
    main()
