"""Select scientific results and verify downloads before remote
deletion."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from temper.mlff.pipeline.training_adapters import mlff_adapter_factory
from temper.schemas.mlff_train_bundle import MLFFTrainBundle
from temper.utils.file_hashing import file_sha256


def get_download_files(bundle: MLFFTrainBundle) -> tuple[list[str], list[str]]:
    """Describe retained files and required successful outputs.

    Parameters
    ----------
    bundle : MLFFTrainBundle
        Packaged bundle with its configured output locations.

    Returns
    -------
    tuple[list[str], list[str]]
        Exact bundle-relative download paths and required success paths.
        Failed attempts may lack any calculation output; available
        diagnostics are still collected. Converted datasets and
        intermediate checkpoints are never selected by broad
        output-directory recursion.

    Raises
    ------
    ValueError
        If the bundle has not been packaged.
    """
    files = bundle.files
    if files is None:
        raise ValueError("Bundle has not been packaged.")
    adapter = mlff_adapter_factory(bundle.mlff_spec.mlff_type)
    required = list(files.test_output_files.values())
    if bundle.unit_type == "finetune":
        required.append(files.training_output_files["trained_model"])
    selected = [
        "stdout.log",
        "stderr.log",
        "scheduler.stdout.log",
        "scheduler.stderr.log",
        "run.sh",
        "temper_job.sh",
    ]
    selected += list(files.test_output_files.values())
    selected += [
        value
        for key, value in files.training_output_files.items()
        if key in adapter.retained_output_keys
    ]
    selected += [
        value
        for key, value in files.training_input_files.items()
        if key in {"script", "config"}
    ]
    selected += [files.test_input_files["config"]]
    for name in selected:
        validate_result_path(name)
    return sorted(set(selected)), required


def validate_result_path(name: str) -> None:
    """Reject paths that cannot safely identify a file inside an
    attempt.

    Parameters
    ----------
    name : str
        POSIX path from an archive or remote directory listing.

    Raises
    ------
    ValueError
        If a remote name could escape the local attempt directory.
    """
    path = PurePosixPath(name)
    if (
        not path.parts
        or path.is_absolute()
        or ".." in path.parts
        or "\\" in name
        or ":" in name
    ):
        raise ValueError("Unsafe remote result path.")


def selected_file(name: str, filenames: list[str]) -> bool:
    """Check an exact relative filename; wildcard expansion is not
    performed.

    Parameters
    ----------
    name : str
        Bundle-relative POSIX filename from a remote listing or archive.
    filenames : list[str]
        Exact permitted filenames from get_download_files.

    Returns
    -------
    bool
        Whether the normalized filename is explicitly included.

    Raises
    ------
    ValueError
        If name is absolute or contains parent traversal or a Windows
        drive.
    """
    validate_result_path(name)
    return PurePosixPath(name).as_posix() in filenames


def verify_results(
    root: Path, manifest: dict[str, str], required: list[str]
) -> None:
    """Verify all collected bytes and require the successful result set.

    Parameters
    ----------
    root : Path
        Attempt directory containing collected files.
    manifest : dict[str, str]
        Relative downloaded paths and their expected SHA-256 digests.
    required : list[str]
        Required nonempty successful output files; use [] for a failed
        job.

    Raises
    ------
    ValueError
        If a required output is absent/empty or downloaded content
        changed.
    OSError
        If a downloaded file cannot be read. Remote deletion must be
        postponed.
    """
    for name in [*required, *manifest]:
        validate_result_path(name)
        if not (root / name).resolve().is_relative_to(root.resolve()):
            raise ValueError("Result path escapes the attempt directory.")
    for name in required:
        if name not in manifest or (root / name).stat().st_size == 0:
            raise ValueError(f"Required result missing or empty: {name}")
    for name, expected in manifest.items():
        if file_sha256(root / name) != expected:
            raise ValueError(f"Downloaded result changed: {name}")
