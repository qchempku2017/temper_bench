"""Build auditable MACE/MatterSim wheels from hash-pinned upstream
archives.

Run with an isolated build environment containing ``build`` and Git.
This tool never edits installed packages. Install its wheel with
ordinary pip dependency resolution, then run pip check and scheduled GPU
validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
from urllib.request import urlopen


def main() -> None:
    """Build the selected patched wheel and a provenance JSON beside it.

    Command-line parameters select mace-torch or mattersim, an output
    directory relative to the current directory, and optionally an
    already downloaded archive. Without an archive, download the exact
    recorded upstream source.

    Raises
    ------
    ValueError
        If the source archive checksum differs from the recorded
        upstream hash.
    subprocess.CalledProcessError
        If the patch does not apply exactly or isolated wheel building
        fails.
    OSError
        If downloading, reading source or writing output fails.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=("mace-torch", "mattersim"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    artifacts = json.loads(
        (here / "upstream-artifacts.json").read_text(encoding="utf-8-sig")
    )
    artifact = next(item for item in artifacts if item["name"] == args.backend)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="temper-backend-") as work:
        root = Path(work)
        archive = args.archive
        if archive is None:
            archive = root / "source.tar.gz"
            with urlopen(artifact["url"], timeout=120) as response:
                archive.write_bytes(response.read())
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        if digest != artifact["sha256"]:
            raise ValueError("Upstream archive SHA256 mismatch.")
        with tarfile.open(archive) as stream:
            stream.extractall(root / "source", filter="data")
        (source,) = (root / "source").iterdir()
        patch = (
            here
            / "patches"
            / (
                "mace-local-checkpoint-dtype.patch"
                if args.backend == "mace-torch"
                else "mattersim-training.patch"
            )
        )
        options = ["--directory=src"] if args.backend == "mattersim" else []
        subprocess.run(
            ["git", "apply", "--check", *options, str(patch)],
            cwd=source,
            check=True,
        )
        subprocess.run(
            ["git", "apply", *options, str(patch)], cwd=source, check=True
        )
        patched_version = artifact["version"] + "+temper.1"
        env = dict(os.environ)
        if args.backend == "mace-torch":
            version_file = source / "mace" / "__version__.py"
            version_file.write_text(
                version_file.read_text().replace(
                    '"0.3.16"', f'"{patched_version}"'
                )
            )
        else:
            env["SETUPTOOLS_SCM_PRETEND_VERSION_FOR_MATTERSIM"] = (
                patched_version
            )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "build",
                "--wheel",
                "--outdir",
                str(output),
            ],
            cwd=source,
            env=env,
            check=True,
        )
        wheels = list(
            output.glob(
                f"{args.backend.replace('-', '_')}-{patched_version}-*.whl"
            )
        )
        if len(wheels) != 1:
            raise RuntimeError("Expected one versioned backend wheel.")
        record = {
            **artifact,
            "patched_version": patched_version,
            "patch_sha256": hashlib.sha256(patch.read_bytes()).hexdigest(),
            "wheel_sha256": hashlib.sha256(wheels[0].read_bytes()).hexdigest(),
            "wheel": wheels[0].name,
        }
        (output / f"{args.backend}-build.json").write_text(
            json.dumps(record, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
