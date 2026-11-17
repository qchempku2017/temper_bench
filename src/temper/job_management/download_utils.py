"""Select download files and verify their paths and contents before cleanup."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from temper.mlff.pipeline.training_adapters import mlff_adapter_factory
from temper.schemas.mlff_train_bundle import MLFFTrainBundle
from temper.utils.file_hashing import file_sha256


def get_download_files(bundle: MLFFTrainBundle) -> tuple[list[str], list[str]]:
    """Get files to download and required successful outputs.

    Parameters
    ----------
    bundle : MLFFTrainBundle
        Packaged bundle with its configured output locations.

    Returns
    -------
    tuple[list[str], list[str]]
        Exact bundle-relative download paths and required success paths.
        Failed attempts may lack models and predictions; available
        diagnostics are still collected. Converted datasets and
        intermediate checkpoints are never selected by broad
        output-directory recursion.
        Collection requires the runner's results.json even for failed or
        cancelled calculations. Missing manifests prevent verified collection
        and cleanup; completed training time and partial tests are retained.

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
    required.extend([files.result_manifest, "bundle.json"])
    if bundle.unit_type == "finetune":
        required.append(files.training_output_files["trained_model"])
    selected = [
        "results.json",
        "bundle.json",
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
    for name in selected:
        validate_result_path(name)
    return sorted(set(selected)), required


def required_result_files(
    bundle: MLFFTrainBundle, *, successful: bool
) -> list[str]:
    """Return the files required to verify a collected calculation.

    Parameters
    ----------
    bundle : MLFFTrainBundle
        Experiment with packaged output locations. Its files mapping must
        be populated when successful is True.
    successful : bool
        True for a completed calculation, requiring predictions and the
        trained model for fine-tuning. False for a failed or cancelled
        calculation, requiring results.json and bundle.json.

    Returns
    -------
    list[str]
        Required paths relative to the attempt directory. Every outcome
        requires results.json and bundle.json. Successful calculations also
        require every prediction file and, for fine-tuning, the trained model.
        Returning this list does not mark collection verified or create files.

    Raises
    ------
    ValueError
        If successful is True and the bundle has no packaged file mapping.
    KeyError
        If a successful fine-tuning bundle lacks its trained_model output path.
    """
    if not successful:
        return ["results.json", "bundle.json"]
    return get_download_files(bundle)[1]


def verify_result_manifest(path: str | Path, *, successful: bool) -> None:
    """Check a runner's saved settings and predictions before cleanup.

    Parameters
    ----------
    path : str or pathlib.Path
        Existing results.json. Expands ~ and resolves relative paths against
        the working directory. Referenced files resolve from its parent.
    successful : bool
        Require a completed manifest when True. False permits unfinished
        tests from a failed or cancelled calculation, while still verifying
        every prediction with a recorded checksum.

    Raises
    ------
    ValueError
        If settings, test descriptions, prediction bytes or completion status
        disagree with the manifest.
    OSError
        If a required manifest, settings file or prediction cannot be read.

    Notes
    -----
    Does not modify files or read original reference datasets. Ground-truth
    labels are checked when the user loads results for analysis.
    """
    from temper.results_loading.manifest_io import load_bundle_from_manifest
    from temper.schemas.mlff_experiment_result import MLFFExperimentManifest

    path = Path(path).expanduser().resolve()
    report = MLFFExperimentManifest.from_json_file(path)
    load_bundle_from_manifest(report, path.parent)
    if successful and report.status != "completed":
        raise ValueError("Calculation manifest is incomplete.")
    expected = {
        test.prediction_path: test.prediction_sha256
        for test in report.test_manifests
        if test.prediction_sha256 is not None
    }
    expected[report.bundle_path] = report.bundle_sha256
    verify_results(path.parent, expected, list(expected))


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


def is_selected_for_download(name: str, filenames: list[str]) -> bool:
    """Return whether a relative filename is selected for download.

    Matches exact filenames; wildcard expansion is not performed.

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
    """Verify collected bytes and require the specified output files.

    Parameters
    ----------
    root : Path
        Attempt directory containing collected files. Relative paths use the
        current working directory.
    manifest : dict[str, str]
        Relative downloaded paths and their expected SHA-256 digests.
    required : list[str]
        Required nonempty paths relative to root. Use required_result_files
        to obtain the appropriate list for a completed, failed or cancelled job.

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
