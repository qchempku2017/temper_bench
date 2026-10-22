"""Controlled dpdispatcher operations without its automatic retry lifecycle.

Only this module imports the optional remote dependencies. dpdispatcher owns
scheduler script generation and upload; TEMPER owns when each operation runs.
"""
from __future__ import annotations

from collections.abc import Callable
from importlib.metadata import version
import json
import logging
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import stat
import time
from typing import Any, TYPE_CHECKING, cast
from uuid import uuid4
from zipfile import ZipFile

# dpdispatcher initializes a cwd file logger during import. Its lifecycle logs
# may include serialized backend details, so TEMPER supplies its own status
# history and disables that independent logger before it can open a file.
_dispatcher_log = logging.getLogger("dpdispatcher")
_dispatcher_log.disabled = True
from dpdispatcher import Resources, Submission, Task
from dpdispatcher.contexts.ssh_context import SSHContext, SSHSession
from dpdispatcher.machines.slurm import Slurm
from dpdispatcher.machines.pbs import PBS
from dpdispatcher.utils.job_status import JobStatus
import paramiko

if TYPE_CHECKING:
    from dpdispatcher.machines.openapi import OpenAPI
    from dpdispatcher.contexts.openapi_context import OpenAPIContext

for _handler in list(_dispatcher_log.handlers):
    _dispatcher_log.removeHandler(_handler)
    _handler.close()

from temper.job_management.collection import get_download_files, selected_file
from temper.job_management.inputs import get_dataset_and_model_hashes
from temper.job_management.schemas.records import ComputationStatus, JobRecord
from temper.schemas.mlff_train_bundle import MLFFTrainBundle
from temper.job_management.schemas.resources import RemoteResource
from temper.utils.file_hashing import file_sha256


# Overwrite default SSH session implementation in DPDispatcher.
class _VerifiedSession(SSHSession):
    def __init__(self, profile: RemoteResource) -> None:
        self._known_hosts: str | None = profile.known_hosts
        self._password: str | None = os.environ[profile.password_env] if not profile.key_filename and profile.password_env else None
        super().__init__(hostname=profile.hostname, username=profile.username,
                         port=profile.port, key_filename=profile.key_filename,
                         passphrase=os.environ[profile.passphrase_env] if profile.passphrase_env else None,
                         timeout=profile.timeout, proxy_command=profile.proxy_command,
                         look_for_keys=False)

    def _setup_ssh(self) -> None:
        for attempt in range(5):
            try:
                self._connect_once()
                return
            except (paramiko.AuthenticationException, paramiko.BadHostKeyException,
                    FileNotFoundError, PermissionError):
                self.ssh.close()
                raise
            except (OSError, EOFError):
                self.ssh.close()
                if attempt == 4:
                    raise
                time.sleep(10)

    def _connect_once(self) -> None:
        if self.ssh is not None:
            self.ssh.close()
        self.ssh = paramiko.SSHClient()
        self.ssh.load_system_host_keys()
        if self._known_hosts:
            self.ssh.load_host_keys(self._known_hosts)
        self.ssh.set_missing_host_key_policy(paramiko.RejectPolicy())
        proxy = paramiko.ProxyCommand(self.proxy_command) if self.proxy_command else None
        self.ssh.connect(self.hostname, port=self.port, username=self.username,
                         key_filename=self.key_filename, passphrase=self.passphrase, password=self._password,
                         timeout=self.timeout, auth_timeout=self.timeout, banner_timeout=self.timeout,
                         allow_agent=False, look_for_keys=False, sock=proxy)
        transport = self.ssh.get_transport()
        if transport is None:
            raise ConnectionError("SSH connection has no transport.")
        transport.set_keepalive(30)
        self._sftp = None

    def ensure_alive(self, max_check: int = 5, sleep_time: int = 10) -> None:
        """Reconnect with up to five attempts, ten seconds apart, if necessary.

        Parameters
        ----------
        max_check, sleep_time : int
            Compatibility arguments; unused. Connection policy is five total
            attempts with ten seconds between transient failures.
        """
        try:
            alive = self._check_alive()
        except (EOFError, OSError, paramiko.SSHException):
            alive = False
        if not alive:
            self._setup_ssh()

    def exec_command(self, cmd: str) -> tuple[paramiko.channel.ChannelStdinFile, paramiko.channel.ChannelFile, paramiko.channel.ChannelStderrFile]:
        """Execute once, without replaying a command after a lost response.

        Parameters
        ----------
        cmd : str
            Complete quoted remote shell command.

        Returns
        -------
        tuple[ChannelStdinFile, ChannelFile, ChannelStderrFile]
            Paramiko stdin, stdout and stderr streams.

        Raises
        ------
        OSError
            If connection attempts fail or command execution times out.
        paramiko.SSHException
            If authentication, host verification, or channel creation fails.
        """
        self.ensure_alive()
        return self.ssh.exec_command(cmd, timeout=self.timeout)

    @property
    def sftp(self) -> paramiko.SFTPClient:
        """Return an SFTP channel with the configured operation timeout.

        This is used for all file transfers.
        Returns
        -------
        paramiko.SFTPClient
            Transfer client on the verified SSH connection.
        """
        client = super().sftp
        client.get_channel().settimeout(self.timeout)
        return client

    @property
    def rsync_available(self) -> bool:
        """Return False to keep all transfers on the verified SSH connection.

        Returns
        -------
        bool
            False; SFTP also supports Windows submission hosts.
        """
        return False


# Overwrite default SSH context implementation in DPDispatcher.
class _VerifiedContext(SSHContext):
    # The parent constructor creates its own SSH session and remote directory.
    # TEMPER supplies the verified session and defers directory creation to upload.
    # noinspection PyMissingConstructor
    def __init__(self, root: Path, profile: RemoteResource, token: str) -> None:
        self.submission: Submission | None = None
        self.submission_hash: str | None = None
        if profile.remote_root is None:
            raise ValueError("Resolve remote_root before constructing the SSH context.")
        self.init_local_root = self.temp_local_root = self.local_root = str(root)
        self.init_remote_root = self.temp_remote_root = profile.remote_root
        self.remote_root = str(PurePosixPath(profile.remote_root) / ("temper_" + token))
        self.remote_profile = {"hostname": profile.hostname, "username": profile.username,
                               "port": profile.port, "key_filename": profile.key_filename,
                               "tar_compress": True}
        self.clean_asynchronously = False
        self.ssh_session = _VerifiedSession(profile)

    def bind_submission(self, submission: Submission) -> None:
        """Associate state without creating or modifying remote directories.

        Parameters
        ----------
        submission : Submission
            Single-task submission using this fixed attempt directory.
        """
        self.submission = submission
        self.submission_hash = submission.submission_hash


class DispatcherBackend:
    """Execute individual operations for one persisted attempt.

    Parameters
    ----------
    record : JobRecord
        Durable attempt, including sanitized dispatcher state when resuming.
    root : Path
        Absolute local attempt directory with bundle.json.
    checkpoint : Callable[[], None]
        Persist the current record immediately, including cloud upload IDs.

    Attributes
    ----------
    record : JobRecord
        The same mutable record supplied by the caller.
    root : Path
        Local attempt directory; uploaded copies can be removed after upload.
    profile : RemoteResource
        Frozen resource settings, with credentials resolved only in memory.
    submission : Submission
        dpdispatcher submission containing exactly one job and task.
    """
    def __init__(self, record: JobRecord, root: Path, checkpoint: Callable[[], None]) -> None:
        if version("dpdispatcher") != "1.0.3":
            raise RuntimeError("TEMPER remote execution requires dpdispatcher==1.0.3.")
        self.record: JobRecord = record
        self.root: Path = root
        self.profile: RemoteResource = RemoteResource.model_validate(record.resource)
        self._checkpoint = checkpoint
        self._bundle = MLFFTrainBundle.model_validate_json((root / "bundle.json").read_text(encoding="utf-8"), context={"verify_files": False})
        state = record.dispatcher
        token = str(state.setdefault("token", uuid4().hex))
        if not re.fullmatch(r"[0-9a-f]{32}", token):
            raise ValueError("Invalid remote attempt token.")
        if self.profile.backend == "Bohrium":
            if version("bohrium-sdk") != "0.15.0":
                raise RuntimeError("TEMPER Bohrium execution requires bohrium-sdk==0.15.0.")
            from dpdispatcher.contexts.openapi_context import OpenAPIContext
            from dpdispatcher.machines.openapi import OpenAPI
            # The SDK logs complete API responses (including signed URLs).
            logging.getLogger("bohrium").setLevel(logging.CRITICAL)
            logging.getLogger("httpx").setLevel(logging.CRITICAL)
            logging.getLogger("httpcore").setLevel(logging.CRITICAL)
            if self.profile.resolved_environment is None:
                raise ValueError("Bohrium requires a resolved runtime environment.")
            profile = {"access_key": os.environ[self.profile.bohrium_access_key_env],
                       "project_id": self.profile.bohrium_project_id, "machine_type": self.profile.bohrium_machine_type,
                       "image_address": self.profile.resolved_environment.image, "platform": self.profile.bohrium_platform,
                       "ignore_exit_code": False, "job_name": "temper-" + token}
            if not profile["image_address"]:
                raise ValueError("Bohrium requires an environment image.")
            context = OpenAPIContext(str(root), remote_profile=profile)
            machine = OpenAPI(context, retry_count=0)
            # In SDK 0.15.0 this value is the number of attempts, despite its
            # name. One prevents replaying a creation/insertion after a lost reply.
            context.client.max_retries = machine.client.max_retries = 1
            context.storage.client.max_retries = machine.storage.client.max_retries = 1
            # Authentication remains in SDK clients, not in serialized profiles.
            context.remote_profile = {k: v for k, v in profile.items() if k != "access_key"}
            machine.remote_profile = dict(context.remote_profile)
        else:
            context = _VerifiedContext(root, self.profile, token)
            machine = (Slurm if self.profile.backend == "Slurm" else PBS)(context=context, retry_count=0)
        if state.get("submission"):
            saved_submission = state["submission"]
            if not isinstance(saved_submission, dict):
                raise ValueError("Saved submission must be a JSON object.")
            self.submission: Submission = Submission.deserialize(saved_submission, machine=machine)
            self.submission.belonging_tasks = self.submission.belonging_jobs[0].job_task_list
        else:
            patterns, _ = get_download_files(self._bundle)
            task = Task(command="bash temper_job.sh", task_work_path=".",
                        forward_files=["bundle.json", "run.sh", "temper_job.sh", *get_dataset_and_model_hashes(self._bundle)],
                        backward_files=patterns, outlog="stdout.log", errlog="stderr.log")
            resources = Resources(**{"number_node": 1, "cpu_per_node": 1, "gpu_per_node": 0,
                                     "queue_name": "", "group_size": 1, **self.profile.resources})
            self.submission = Submission(work_base=".", machine=machine, resources=resources, task_list=[task])
            self.submission.generate_jobs()
        self._job = self.submission.belonging_jobs[0]
        for key in ("upload_path", "jgid"):
            if key in state:
                setattr(self._job, key, state[key])
        self._snapshot()

    def _snapshot(self) -> None:
        state = self.record.dispatcher
        state["submission"] = json.loads(json.dumps(self.submission.serialize()))
        for key in ("upload_path", "jgid"):
            if hasattr(self._job, key):
                state[key] = getattr(self._job, key)

    def _cloud_machine(self) -> OpenAPI:
        if self.profile.backend != "Bohrium":
            raise TypeError("This operation requires the Bohrium backend.")
        # Construction above pairs this discriminator with OpenAPI. Persisted
        # submissions are reconstructed with the same concrete machine.
        return cast("OpenAPI", self.submission.machine)

    def _cloud_context(self) -> OpenAPIContext:
        return cast("OpenAPIContext", self._cloud_machine().context)

    def _command(self, command: str) -> str:
        context = self.submission.machine.context
        stdin, stdout, stderr = context.ssh_session.exec_command(command)
        output = stdout.read().decode("utf-8")
        stderr.read()  # Drain diagnostics without persisting possibly secret text.
        if stdout.channel.recv_exit_status() != 0:
            raise RuntimeError("Remote command failed; inspect scheduler diagnostics.")
        return output

    def upload(self) -> None:
        """Upload copied inputs and persist cloud-created IDs before data transfer.

        Raises
        ------
        Exception
            If upload fails; the manager preserves input copies and recovery
            state. This method never starts a calculation.
        """
        environment = self.profile.resolved_environment
        if environment is None:
            raise ValueError("Resolve the runtime environment before uploading.")
        script = "#!/usr/bin/env bash\nset -eo pipefail\n" + "\n".join(environment.shell_lines()) + "\nexec bash run.sh\n"
        (self.root / "temper_job.sh").write_text(script, encoding="utf-8", newline="\n")
        context = self.submission.machine.context
        if self.profile.backend != "Bohrium":
            self._command("mkdir -p -- " + shlex.quote(context.temp_remote_root))
            self._command("mkdir -p -- " + shlex.quote(context.remote_root))
            context.write_file(".temper-owner", str(self.record.dispatcher["token"]))
            self.submission.upload_jobs()
        else:
            from dpdispatcher.contexts.openapi_context import zip_file_list
            machine = self._cloud_machine()
            machine.gen_local_script(self._job)
            archive = self._job.job_hash + ".zip"
            files = [self._job.script_file_name, self._job.script_file_name + ".run", *self._job.job_task_list[0].forward_files]
            zip_file_list(str(self.root), archive, files)
            if self._job.job_id:
                # A durable uploading state precedes every insert/launch. This
                # ID therefore names an unlaunched draft, whose expired upload
                # token can be replaced by deleting and recreating the draft.
                self._cloud_action("del")
                self._job.job_id = ""
                self.record.dispatcher.pop("upload_path", None)
                self.record.dispatcher["cloud_create_intent"] = False
                if hasattr(self._job, "upload_path"):
                    del self._job.upload_path
                self._snapshot()
                self._checkpoint()
            token: str | None = None
            if not self._job.job_id:
                if self.record.dispatcher.get("cloud_create_intent"):
                    raise RuntimeError("Cloud draft creation response was lost; recover its ID before retrying upload.")
                self.record.dispatcher["cloud_create_intent"] = True
                self._checkpoint()
                data = context.job.create(project_id=self.profile.bohrium_project_id, name="temper-" + str(self.record.dispatcher["token"]))
                self._job.job_id = data["jobId"]
                self._job.upload_path = str(PurePosixPath(data["storePath"]) / archive)
                self._job.jgid = data.get("jobGroupId", "")
                self._snapshot()
                self._checkpoint()
                token = data["token"]
            if token is None:
                raise RuntimeError("Cloud draft has no upload token.")
            self._upload_cloud_archive(self.root / archive, token)
        self._snapshot()

    @staticmethod
    def _confirmed_response(response: Any) -> dict[str, Any]:
        if response is None:
            raise RuntimeError("Cloud operation returned no confirmation.")
        response.raise_for_status()
        body = response.json()
        if body.get("code") != 0:
            raise RuntimeError("Cloud operation did not report success.")
        return body.get("data") or {}

    def _upload_cloud_archive(self, archive: Path, token: str) -> None:
        storage = self.submission.machine.context.storage
        key = self._job.upload_path
        chunk_size = 50 * 1024 * 1024
        with archive.open("rb") as stream:
            if archive.stat().st_size <= chunk_size:
                self._confirmed_response(storage.write(object_key=key, data=stream.read(), token=token))
                return
            initial = self._confirmed_response(storage.init_upload_by_part(key, token=token))["initialKey"]
            parts = []
            number = 1
            while block := stream.read(chunk_size):
                result = self._confirmed_response(
                    storage.upload_by_part(
                        key, initial, chunk_size=len(block), number=number, body=block, token=token
                    )
                )
                parts.append(result["partString"])
                number += 1
            self._confirmed_response(storage.complete_upload_by_part(key, initial, parts, token))

    def submit(self) -> None:
        """Launch once and preserve a remote receipt for SSH recovery.

        Raises
        ------
        Exception
            If the response is lost or rejected. The caller must mark the
            submission uncertain and must not automatically launch it again.
        """
        machine = self.submission.machine
        if self.profile.backend == "Bohrium":
            self._job.job_id = machine.do_submit(self._job)
        else:
            context = machine.context
            context.write_file(self._job.script_file_name, machine.gen_script(self._job))
            context.write_file(self._job.script_file_name + ".run", machine.gen_script_command(self._job))
            output = shlex.quote(str(PurePosixPath(context.remote_root) / "scheduler.stdout.log"))
            error = shlex.quote(str(PurePosixPath(context.remote_root) / "scheduler.stderr.log"))
            submit_command = (f"sbatch --parsable --output={output} --error={error}"
                              if self.profile.backend == "Slurm" else f"qsub -o {output} -e {error}")
            # The receipt is written remotely even if the local connection dies.
            output = self._command(
                f"cd {shlex.quote(context.remote_root)}"
                f" && {submit_command} {shlex.quote(self._job.script_file_name)} > .temper-receipt.tmp"
                f" && mv .temper-receipt.tmp .temper-receipt"
                f" && cat .temper-receipt"
            )
            self._job.job_id = self._parse_id(output)
        if not self._job.job_id:
            raise RuntimeError("Scheduler returned no job identifier.")
        self._job.job_state = JobStatus.waiting
        self._snapshot()

    @staticmethod
    def _parse_id(value: str) -> str:
        candidate = value.strip().split(";")[0]
        if not re.fullmatch(r"[0-9]+(?:[.][A-Za-z0-9_.-]+)?", candidate):
            raise ValueError("Scheduler receipt contains an invalid job identifier.")
        return candidate

    def recover(self, job_id: str | None = None) -> bool:
        """Attach an uncertain submission to an existing remote job.

        This handles the case where a previous launch succeeded but the local receipt was lost.
        Will reassign the uncertain submission to the existing job ID if it is valid.
        Parameters
        ----------
        job_id : str or None, optional
            User-confirmed existing scheduler ID when a receipt was lost. None
            reads the SSH receipt or uses the previously persisted Bohrium ID.

        Returns
        -------
        bool
            True if an ID is available. False never authorizes a fresh launch.
        """
        if job_id:
            parsed = self._parse_id(job_id)
            if self.profile.backend == "Bohrium" and not parsed.isdigit():
                raise ValueError("Bohrium job IDs must be numeric.")
            self._job.job_id = int(parsed) if self.profile.backend == "Bohrium" else parsed
        elif self.profile.backend != "Bohrium" and not self._job.job_id:
            try:
                receipt = self.submission.machine.context.read_file(".temper-receipt")
            except FileNotFoundError:
                return False
            self._job.job_id = self._parse_id(receipt)
        self._snapshot()
        return bool(self._job.job_id)

    def status(self) -> ComputationStatus:
        """Query computation state without submitting, downloading or deleting.

        Returns
        -------
        str
            unknown, waiting, running, success or failed. Unknown remains
            inconclusive and must never trigger resubmission or deletion.
        """
        if not self._job.job_id:
            return "unknown"
        if self.profile.backend == "Bohrium":
            detail = self._cloud_machine().job.detail(self._job.job_id)
            if not isinstance(detail, dict):
                raise RuntimeError("Bohrium returned no job details.")
            status_code = detail.get("status")
            self.record.observation = {"status": status_code, "exit_code": detail.get("exitCode")}
            if status_code == 2:
                exit_code = detail.get("exitCode")
                return "unknown" if exit_code is None else ("success" if exit_code == 0 else "failed")
            states: dict[int, ComputationStatus] = {-1: "failed", 5: "failed", 0: "waiting", 3: "waiting", 9: "waiting", 1: "running", 4: "running", 6: "running"}
            return states.get(status_code, "unknown") if type(status_code) is int else "unknown"
        return self._scheduler_status()

    def _scheduler_status(self) -> ComputationStatus:
        context = self.submission.machine.context
        job_id = self._parse_id(str(self._job.job_id))
        if self.profile.backend == "Slurm":
            code, _, stdout, stderr = context.block_call(f"squeue -h -o '%T' -j {shlex.quote(job_id)}")
            output = stdout.read().decode().strip()
            error = stderr.read().decode()
            if code and "Invalid job id" not in error:
                raise RuntimeError("Slurm status query failed.")
            if output:
                states = set(output.split())
                self.record.observation = {"scheduler": "Slurm", "states": sorted(states)}
                if states & {"RUNNING", "COMPLETING", "CONFIGURING", "STAGE_OUT"}:
                    return "running"
                if states & {"PENDING", "SUSPENDED", "REQUEUED", "RESIZING"}:
                    return "waiting"
            if self.submission.machine.check_finish_tag(self._job):
                return "success"
            code, _, stdout, _ = context.block_call(f"sacct -n -X -P -j {shlex.quote(job_id)} --format=JobIDRaw,State,ExitCode")
            if code:
                return "unknown"  # Accounting may be disabled at this site.
            for line in stdout.read().decode().splitlines():
                columns = line.split("|")
                if len(columns) >= 3 and columns[0] == job_id:
                    state = columns[1].split()[0].rstrip("+")
                    self.record.observation = {"scheduler": "Slurm", "state": state, "exit_code": columns[2]}
                    if state in {"FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY", "PREEMPTED", "BOOT_FAIL", "DEADLINE", "COMPLETED"}:
                        # Without the task completion tag, even scheduler exit 0
                        # does not prove the TEMPER command completed.
                        return "failed"
            return "unknown"
        code, _, stdout, stderr = context.block_call(f"qstat -x -f {shlex.quote(job_id)}")
        output = stdout.read().decode()
        error = stderr.read().decode()
        if code and ("invalid option" in error.lower() or "illegal option" in error.lower()):
            code, _, stdout, stderr = context.block_call(f"qstat -f {shlex.quote(job_id)}")
            output, error = stdout.read().decode(), stderr.read().decode()
        if code:
            if "Unknown Job Id" not in error and "Job has finished" not in error:
                raise RuntimeError("PBS status query failed.")
            return "success" if self.submission.machine.check_finish_tag(self._job) else "unknown"
        match = re.search(r"\bjob_state\s*=\s*(\w+)", output)
        state = match.group(1) if match else ""
        self.record.observation = {"scheduler": "PBS", "state": state}
        if state in {"R", "E", "B"}:
            return "running"  # E is still exiting: deleting here would race PBS.
        if state in {"Q", "H", "W", "S", "T"}:
            return "waiting"
        if state in {"C", "F"}:
            return "success" if self.submission.machine.check_finish_tag(self._job) else "failed"
        return "unknown"

    def collect(self) -> dict[str, str]:
        """Download selected existing outputs into temporary files, then install them.

        Returns
        -------
        dict[str, str]
            Downloaded relative paths and verified SHA-256 digests. SSH checks
            remote hashes; Bohrium checks archive CRCs before local hashing.

        Raises
        ------
        Exception
            On transport, checksum, unsafe-path or archive errors. Missing
            failed-calculation outputs are allowed; transfer errors are not.
        """
        patterns, _ = get_download_files(self._bundle)
        manifest: dict[str, str] = {}
        context = self.submission.machine.context
        if self.profile.backend == "Bohrium":
            detail = self._cloud_machine().job.detail(self._job.job_id)
            url = detail.get("resultUrl")
            if not url:
                raise RuntimeError("Bohrium result archive is not available yet; retaining remote job.")
            archive = self.root / ".results.zip.partial"
            import requests
            # SDK 0.15.0's downloader has no timeout and can return after all
            # connection attempts fail. Require a fresh successful response.
            with requests.get(url, stream=True, timeout=(self.profile.timeout, self.profile.timeout)) as response:
                response.raise_for_status()
                # Download results as zip files in 1MB blocks.
                with archive.open("wb") as stream:
                    for block in response.iter_content(1024 * 1024):
                        stream.write(block)
            with ZipFile(archive) as zipped:
                for entry in zipped.infolist():
                    name = PurePosixPath(entry.filename).as_posix()
                    if entry.is_dir() or not selected_file(name, patterns):
                        continue
                    if stat.S_ISLNK(entry.external_attr >> 16):
                        raise ValueError("Result archives must not contain symbolic links.")
                    destination = self._destination(name)
                    temporary = destination.with_name(destination.name + ".partial")
                    with zipped.open(entry) as incoming, temporary.open("wb") as outgoing:
                        while block := incoming.read(1024 * 1024):
                            outgoing.write(block)
                        outgoing.flush()
                        os.fsync(outgoing.fileno())
                    os.replace(temporary, destination)
                    manifest[name] = file_sha256(destination)
            archive.unlink()
        else:
            def visit(relative: str) -> None:
                remote = str(PurePosixPath(context.remote_root) / relative)
                for entry in context.sftp.listdir_attr(remote):
                    name = str(PurePosixPath(relative) / entry.filename)
                    selected = selected_file(name, patterns)
                    if stat.S_ISDIR(entry.st_mode):
                        if any(pattern.startswith(name + "/") for pattern in patterns):
                            visit(name)
                    elif selected:
                        if not stat.S_ISREG(entry.st_mode):
                            raise ValueError("Selected remote output is not a regular file.")
                        remote_file = str(PurePosixPath(context.remote_root) / name)
                        expected = self._command("sha256sum -- " + shlex.quote(remote_file)).split()[0]
                        destination = self._destination(name)
                        temporary = destination.with_name(destination.name + ".partial")
                        context.sftp.get(remote_file, str(temporary))
                        if file_sha256(temporary) != expected:
                            raise ValueError("Downloaded output checksum mismatch.")
                        with temporary.open("rb+") as stream:
                            os.fsync(stream.fileno())
                        os.replace(temporary, destination)
                        manifest[name] = expected
            visit(".")
        if not manifest:
            raise RuntimeError("No diagnostics or outputs available; retaining remote data.")
        return manifest

    def _destination(self, name: str) -> Path:
        destination = (self.root / name).resolve()
        if not destination.is_relative_to(self.root.resolve()):
            raise ValueError("Output path escapes the attempt directory.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        return destination

    def cancel(self) -> None:
        """Request cancellation of the existing remote job.

        Raises
        ------
        Exception
            If cancellation cannot be sent. Terminal status must still be
            observed before collecting results or deleting the remote directory.
        """
        if not self._job.job_id:
            raise ValueError("Recover the remote job ID before cancelling.")
        if self.profile.backend == "Bohrium":
            self._cloud_action("kill")
        else:
            command = "scancel -Q" if self.profile.backend == "Slurm" else "qdel"
            self._command(command + " " + shlex.quote(str(self._job.job_id)))

    def _cloud_action(self, action: str) -> None:
        from bohrium.resources.job.job import USE_SANDBOX
        prefix = "/openapi/v1/sandbox/job" if USE_SANDBOX else "/openapi/v1/job"
        response = self._cloud_machine().client.post(f"{prefix}/{action}/{self._job.job_id}")
        if action == "del" and response is not None and response.status_code == 404:
            return  # A repeated deletion may observe the already-removed job.
        self._confirmed_response(response)

    def cleanup(self) -> None:
        """Delete this attempt's remote data after verified local collection.

        Raises
        ------
        Exception
            If ownership or deletion cannot be confirmed. Cleanup remains
            pending and may be repeated without rerunning the calculation.
        """
        if self.profile.backend == "Bohrium":
            self._cloud_action("del")
            return
        context = self.submission.machine.context
        remote = PurePosixPath(context.remote_root)
        if self.profile.remote_root is None:
            raise ValueError("Remote cleanup requires a resolved remote_root.")
        parent = PurePosixPath(self.profile.remote_root)
        if remote.parent != parent or remote.name != "temper_" + str(self.record.dispatcher["token"]):
            raise ValueError("Refusing to remove an unowned remote directory.")
        try:
            owner = context.read_file(".temper-owner")
        except FileNotFoundError:
            try:
                context.sftp.stat(str(remote))
            except FileNotFoundError:
                return  # Previous cleanup succeeded before the local commit.
            raise ValueError("Remote directory has no TEMPER ownership record.")
        if owner.strip() != str(self.record.dispatcher["token"]):
            raise ValueError("Remote directory ownership mismatch.")
        self._command("rm -rf -- " + shlex.quote(str(remote)))
        try:
            context.sftp.stat(str(remote))
        except FileNotFoundError:
            return
        raise RuntimeError("Remote directory still exists after cleanup.")

    def close(self) -> None:
        """Release SSH/HTTP connections without cancelling a remote calculation."""
        if self.profile.backend != "Bohrium":
            self.submission.machine.context.close()
        else:
            machine = self._cloud_machine()
            context = self._cloud_context()
            for client in (machine.client, context.client, machine.storage.client, context.storage.client):
                client._client.close()
