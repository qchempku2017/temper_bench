"""Select scientific results and verify downloads before remote deletion."""
from __future__ import annotations

import fnmatch
from pathlib import Path, PurePosixPath
from temper.mlff.pipeline.training_adapters import mlff_adapter_factory
from temper.schemas.mlff_train_bundle import MLFFTrainBundle
from temper.utils.file_hashing import file_sha256


def result_selection(bundle: MLFFTrainBundle) -> tuple[list[str], list[str]]:
    """Describe retained files and required successful outputs.

    Parameters
    ----------
    bundle : MLFFTrainBundle
        Packaged recipe with its configured output locations.

    Returns
    -------
    tuple[list[str], list[str]]
        Bundle-relative selection patterns and exact required success paths.
        Failed attempts may lack any calculation output; available diagnostics
        are still collected. Converted datasets and intermediate checkpoints
        are never selected by broad output-directory recursion.

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
    selected = ["stdout.log", "stderr.log", "slurm-*.out", "*.sub.o*", "*.sub.e*", "*.sub", "*.sub.run"]
    selected += list(files.test_output_files.values())
    selected += [value for key, value in files.training_output_files.items() if key in adapter.retained_output_keys]
    selected += [value for key, value in files.training_input_files.items() if key in {"script", "config"}]
    selected += [files.test_input_files["config"], *adapter.retained_output_patterns]
    selected += [f"{files.training_output_files[key]}/{pattern}"
                 for key, globs in adapter.retained_output_globs.items()
                 if key in files.training_output_files for pattern in globs]
    return sorted(set(selected)), required


def selected_file(name: str, patterns: list[str]) -> bool:
    """Check a safe relative filename against retained-output patterns.

    Parameters
    ----------
    name : str
        POSIX path from an archive or remote directory listing.
    patterns : list[str]
        Patterns returned by result_selection.

    Returns
    -------
    bool
        Whether the file is selected. Absolute, parent-traversing and Windows
        drive paths raise ValueError, even when they would not be selected.

    Raises
    ------
    ValueError
        If a remote name could escape the local attempt directory.
    """
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name:
        raise ValueError("Unsafe remote result path.")
    return any(fnmatch.fnmatchcase(path.as_posix(), pattern) for pattern in patterns)


def verify_results(root: Path, manifest: dict[str, str], required: list[str]) -> None:
    """Verify all collected bytes and require the successful result set.

    Parameters
    ----------
    root : Path
        Attempt directory containing collected files.
    manifest : dict[str, str]
        Relative downloaded paths and their expected SHA-256 digests.
    required : list[str]
        Required nonempty successful output files; use [] for a failed job.

    Raises
    ------
    ValueError
        If a required output is absent/empty or downloaded content changed.
    OSError
        If a downloaded file cannot be read. Remote deletion must be postponed.
    """
    for name in required:
        if name not in manifest or (root / name).stat().st_size == 0:
            raise ValueError(f"Required result missing or empty: {name}")
    for name, expected in manifest.items():
        selected_file(name, [])
        if file_sha256(root / name) != expected:
            raise ValueError(f"Downloaded result changed: {name}")
