"""Record installed versions and source modifications without importing
models.

Run with the execution environment's interpreter, passing an output JSON
path. This is an installation audit and can run on a login node; it
performs no GPU operations. RECORD hashes describe wheel installation
state, not a trusted signature. Compare reported changes with the
preserved compatibility patches.
"""

from __future__ import annotations

import base64
import hashlib
from importlib.metadata import distributions
import json
from pathlib import Path
import shutil
import sys


def main() -> None:
    """Write package versions and modified backend Python source hashes.

    The output path is the first command-line argument and resolves
    against the current working directory. Missing RECORD hashes are
    reported as unverified; they are never interpreted as evidence of an
    unmodified build.

    Raises
    ------
    OSError
        If installed source files or the output path cannot be accessed.
    """
    selected = {
        "mace-torch",
        "mattersim",
        "calorine",
        "torchnep",
        "sevenn",
        "deepmd-kit",
        "temper-bench",
    }
    records: dict[str, object] = {"python": sys.version, "prefix": sys.prefix}
    versions: dict[str, str] = {}
    changes: dict[str, str] = {}
    unverified: list[str] = []
    for package in distributions():
        name = package.metadata["Name"]
        if name in versions:
            continue  # First distribution wins in an environment overlay.
        versions[name] = package.version
        if name.lower() not in selected:
            continue
        for entry in package.files or []:
            if entry.suffix != ".py":
                continue
            path = Path(package.locate_file(entry))
            key = name + ":" + str(entry)
            if not entry.hash or entry.hash.mode != "sha256":
                unverified.append(key)
                continue
            digest = hashlib.sha256(path.read_bytes()).digest()
            if (
                base64.urlsafe_b64encode(digest).decode().rstrip("=")
                != entry.hash.value
            ):
                changes[key] = digest.hex()
    records.update(
        packages=versions,
        modified_python_files=changes,
        unverified_python_files=unverified,
    )
    gpumd = shutil.which("gpumd")
    if gpumd:
        records["gpumd"] = {
            "path": gpumd,
            "sha256": hashlib.sha256(Path(gpumd).read_bytes()).hexdigest(),
        }
    Path(sys.argv[1]).write_text(
        json.dumps(records, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
