"""Run local attempts under a detached supervisor with durable receipts."""
from __future__ import annotations

from collections.abc import Callable
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from temper.job_management.collection import get_download_files, validate_result_path
from temper.job_management.inputs import get_dataset_and_model_hashes
from temper.job_management.schemas.records import ComputationStatus, JobRecord
from temper.job_management.schemas.resources import RemoteResource
from temper.job_management.storage_utils import atomic_json
from temper.schemas.mlff_train_bundle import MLFFTrainBundle
from temper.utils.file_hashing import file_sha256


class LocalBackend:
    """Execute a packaged bundle locally without transferring its inputs.

    Parameters
    ----------
    record : JobRecord
        Mutable attempt record. Submission stores a durable launch intent.
    root : Path
        Absolute attempt directory containing bundle.json and run.sh.
    checkpoint : Callable[[], None]
        Persist record changes before starting the supervisor.

    Attributes
    ----------
    record : JobRecord
        Attempt whose status and launch intent are updated.
    root : Path
        Local attempt directory; results and configuration files are retained.

    Notes
    -----
    Local execution requires POSIX Bash and the selected MLFF environment.
    The supervisor survives the submitting process. Cancellation is requested
    through a file, so a reused process ID is never signalled by a later CLI.
    A lost supervisor without an exit receipt leaves status unknown.
    """

    def __init__(self, record: JobRecord, root: Path, checkpoint: Callable[[], None]) -> None:
        self.record: JobRecord = record
        self.root: Path = root
        self._checkpoint: Callable[[], None] = checkpoint
        self._bundle: MLFFTrainBundle = MLFFTrainBundle.model_validate_json(
            (root / "bundle.json").read_text(encoding="utf-8"), context={"verify_files": False})

    def upload(self) -> None:
        """Prepare the local Bash wrapper without copying or deleting inputs.

        Raises
        ------
        RuntimeError
            If Bash is unavailable or the host is not POSIX.
        ValueError
            If the attempt has no resolved runtime environment.
        OSError
            If the wrapper cannot be written.
        """
        if os.name != "posix" or shutil.which("bash") is None:
            raise RuntimeError("Local jobs require a POSIX host with Bash; use a Linux/WSL runner.")
        environment = RemoteResource.model_validate(self.record.resource).resolved_environment
        if environment is None:
            raise ValueError("Resolve the runtime environment before submitting.")
        script = "#!/usr/bin/env bash\nset -eo pipefail\n" + "\n".join(environment.shell_lines()) + "\nexec bash run.sh\n"
        (self.root / "temper_job.sh").write_text(script, encoding="utf-8", newline="\n")

    def submit(self) -> None:
        """Start one detached supervisor after persisting launch intent.

        Raises
        ------
        RuntimeError
            If a previous launch is uncertain; recover it instead of replaying.
        OSError
            If the supervisor cannot be started. The intent remains uncertain.
        """
        if self.record.dispatcher.get("local_launch_intent"):
            raise RuntimeError("Local launch is uncertain; recover its receipt.")
        self.record.dispatcher["local_launch_intent"] = True
        self._checkpoint()
        subprocess.Popen(
            [sys.executable, "-m", "temper.job_management.local", str(self.root)],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True, cwd=self.root,
        )

    def recover(self, job_id: str | None = None) -> bool:
        """Read the supervisor receipt without launching another calculation.

        Parameters
        ----------
        job_id : str or None, optional
            Unsupported for local attempts; None reads the local receipt.

        Returns
        -------
        bool
            Whether the supervisor has durably acknowledged the launch.

        Raises
        ------
        ValueError
            If an external job ID is supplied.
        """
        if job_id is not None:
            raise ValueError("Local recovery uses the attempt receipt, not a job ID.")
        return (self.root / ".local-receipt.json").is_file()

    def status(self) -> ComputationStatus:
        """Read the local supervisor's latest durable observation.

        Returns
        -------
        ComputationStatus
            Terminal result if an exit receipt exists; running only while a
            heartbeat is recent, otherwise unknown. No process is signalled.

        Raises
        ------
        ValueError
            If a receipt is malformed.
        OSError
            If a receipt cannot be read.
        """
        exit_path = self.root / ".local-exit.json"
        if exit_path.exists():
            result = json.loads(exit_path.read_text(encoding="utf-8"))
            if result.get("cancelled"):
                return "cancelled"
            return "success" if result["returncode"] == 0 else "failed"
        heartbeat = self.root / ".local-heartbeat"
        if heartbeat.exists() and time.time() - heartbeat.stat().st_mtime < 30:
            return "running"
        return "unknown"

    def collect(self) -> dict[str, str]:
        """Hash existing retained files in place without copying checkpoints.

        Returns
        -------
        dict[str, str]
            Exact relative result paths mapped to their SHA-256 digests.

        Raises
        ------
        ValueError
            If a retained path escapes the attempt directory.
        OSError
            If a retained file cannot be read.
        """
        names, _ = get_download_files(self._bundle)
        result: dict[str, str] = {}
        for name in names:
            validate_result_path(name)
            path = self.root / name
            if not path.resolve().is_relative_to(self.root.resolve()):
                raise ValueError("Result path escapes the attempt directory.")
            if path.is_file():
                result[name] = file_sha256(path)
        return result

    def cleanup(self) -> None:
        """Remove copied inputs, converted datasets, and intermediate outputs.

        Notes
        -----
        The manager calls this only after terminal status and result verification.
        Configuration files, logs, final weights, and evaluation results remain.

        Raises
        ------
        ValueError
            If an output or input path escapes the attempt directory.
        OSError
            If a temporary input or intermediate output cannot be removed.
        """
        retained, _ = get_download_files(self._bundle)
        for name in get_dataset_and_model_hashes(self._bundle):
            path = self.root / name
            if not path.resolve().is_relative_to(self.root.resolve()):
                raise ValueError("Input path escapes the attempt directory.")
            path.unlink(missing_ok=True)
        files = self._bundle.files
        assert files is not None
        temporary_names = {
            *files.training_input_files.values(), *files.test_input_files.values(),
            *files.training_output_files.values(),
        }
        if files.evaluation_model is not None:
            temporary_names.add(files.evaluation_model)
        retained_paths = [(self.root / name).resolve() for name in retained]
        for name in temporary_names:
            path = self.root / name
            if not path.resolve().is_relative_to(self.root.resolve()):
                raise ValueError("Output path escapes the attempt directory.")
            # Never remove a directory containing a retained result or config.
            if not any(kept.is_relative_to(path.resolve()) for kept in retained_paths):
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink(missing_ok=True)

    def cancel(self) -> None:
        """Ask the live supervisor to terminate its calculation process group.

        Raises
        ------
        OSError
            If the cancellation request cannot be written.
        """
        (self.root / ".local-cancel").touch()

    def close(self) -> None:
        """Leave the detached supervisor running; no connection is held."""


def _supervise(root: Path) -> None:
    # Exclusive creation prevents accidental duplicate supervisors from launching.
    with (root / ".local-started").open("x", encoding="utf-8"):
        pass
    atomic_json(root / ".local-receipt.json", {"pid": os.getpid()})
    with (root / "stdout.log").open("wb") as stdout, (root / "stderr.log").open("wb") as stderr:
        try:
            child = subprocess.Popen(["bash", "temper_job.sh"], cwd=root, stdout=stdout, stderr=stderr, start_new_session=True)
        except OSError:
            stderr.write(b"Could not start Bash for the local calculation.\n")
            atomic_json(root / ".local-exit.json", {"returncode": 127, "cancelled": False})
            return
        cancelled = False
        while child.poll() is None:
            (root / ".local-heartbeat").touch()
            if (root / ".local-cancel").exists():
                cancelled = True
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                # The leader may exit before its descendants. Terminate any
                # remaining members of this child's process group as well.
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait()
                break
            time.sleep(1)
        atomic_json(root / ".local-exit.json", {"returncode": child.wait(), "cancelled": cancelled})


if __name__ == "__main__":
    _supervise(Path(sys.argv[1]).resolve())
