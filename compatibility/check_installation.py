"""Check the tested backend/checkpoint recipe without executing GPU
code.

Usage: python compatibility/check_installation.py FAMILY CHECKPOINT
OUTPUT.json. Paths resolve against the current working directory. Only
the checkpoint/JSON pairs recorded in docs/deployment/pretrained-models.json
have measured backend requirements. Selection uses both the requested family
and checkpoint hash, so several validated variants can share one family.
Use a scheduled functional test after installation checks pass.
"""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import distribution, version
import json
from pathlib import Path
import shutil
import subprocess
import sys


def main() -> None:
    """Validate package requirements, checkpoint identity and patch
    contents.

    Write a JSON report only after all checks pass. No model, calculator
    or CUDA library is imported. A successful report establishes
    installation consistency, not training, inference or scientific
    accuracy.

    Raises
    ------
    ValueError
        If the checkpoint or installed recipe differs from the tested
        one.
    importlib.metadata.PackageNotFoundError
        If a required distribution is missing.
    subprocess.CalledProcessError
        If pip reports incompatible installed dependencies.
    OSError
        If checkpoint, source files or output cannot be accessed.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "family",
        choices=("mace", "mattersim", "nep89", "sevennet", "dpa4", "dpa4c"),
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--configuration",
        type=Path,
        help="DeepMD JSON; defaults to the checkpoint's .json sibling.",
    )
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    models = json.loads(
        (here.parent / "docs/deployment/pretrained-models.json").read_text()
    )
    with args.checkpoint.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    model = next(
        (
            item for item in models
            if item["family"] == args.family and item["sha256"] == digest
        ),
        None,
    )
    if model is None:
        raise ValueError(
            "Checkpoint hash is not the tested release; establish its own"
            " backend requirements."
        )
    configuration_hash = None
    if args.family in {"dpa4", "dpa4c"}:
        configuration = args.configuration or args.checkpoint.with_suffix(
            ".json"
        )
        configuration_hash = hashlib.sha256(
            configuration.read_bytes()
        ).hexdigest()
        if configuration_hash != model["configuration_sha256"]:
            raise ValueError(
                "Configuration differs from the tested checkpoint/JSON pair."
            )
    requirements = {
        "mace": {"mace-torch": "0.3.16", "e3nn": "0.4.4"},
        "mattersim": {"mattersim": "1.2.5"},
        "nep89": {"torchnep": "1.0.2", "calorine": "4.1"},
        "sevennet": {"sevenn": "0.13.0"},
        "dpa4": {
            "deepmd-kit": "3.2.0",
            "torch": "2.12.0",
            "vesin": "0.6.1",
            "vesin-torch": "0.6.1",
        },
        "dpa4c": {
            "deepmd-kit": "3.2.0",
            "torch": "2.12.0",
            "vesin": "0.6.1",
            "vesin-torch": "0.6.1",
        },
    }[args.family]
    installed = {name: version(name) for name in requirements}
    for name, expected in requirements.items():
        allowed = (
            {expected, expected + "+temper.1"}
            if name in {"mace-torch", "mattersim"}
            else {expected}
        )
        if installed[name] not in allowed:
            raise ValueError(
                f"{name}: expected audited recipe {sorted(allowed)}, found"
                f" {installed[name]}."
            )
    patch_files = json.loads(
        (here / "patched-source-sha256.json").read_text(encoding="utf-8-sig")
    )
    checked: dict[str, str] = {}
    package = {"mace": "mace-torch", "mattersim": "mattersim"}.get(args.family)
    if package:
        for relative, expected in patch_files.items():
            if not relative.startswith(args.family + "/"):
                continue
            path = Path(distribution(package).locate_file(relative))
            actual = hashlib.sha256(
                path.read_bytes().replace(b"\r\n", b"\n")
            ).hexdigest()
            if actual != expected:
                raise ValueError(
                    f"Unverified compatibility source: {path}. Rebuild from"
                    " the recorded patch."
                )
            checked[relative] = actual
    executable = {
        "mace": "mace_run_train",
        "nep89": "gpumd",
        "dpa4": "dp",
        "dpa4c": "dp",
    }.get(args.family)
    if executable and shutil.which(executable) is None:
        raise ValueError(f"Required executable is not on PATH: {executable}.")
    subprocess.run([sys.executable, "-m", "pip", "check"], check=True)
    args.output.write_text(
        json.dumps(
            {
                "family": args.family,
                "checkpoint": model,
                "configuration_sha256": configuration_hash,
                "packages": installed,
                "verified_patch_files": checked,
                "gpu_functional_test": "not performed",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
