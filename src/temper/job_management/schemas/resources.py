"""Execution resource and runtime environment configuration schemas."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
import re
from typing import Literal, Self
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    model_validator,
)


class RuntimeEnvironment(BaseModel):
    """Select the software environment used to run an MLFF calculation.

    These settings generate Bash commands that run before the
    calculation. Paths refer to the machine running the calculation, not
    the workstation submitting it. A failed setup command stops the job
    and appears in its log.

    Parameters
    ----------
    **data : object
        Values for the attributes below. Unknown setting names are
        rejected.

    Attributes
    ----------
    modules : list[str]
        Software modules to load, in order, using the cluster's ``module
        load`` command. Modules commonly provide compilers or CUDA. An
        empty list leaves the job's initially available software
        unchanged.
    conda_init : str or None
        Full path to Conda's ``etc/profile.d/conda.sh`` activation
        script. Requires conda_environment. If omitted, TEMPER asks the
        ``conda`` executable to initialize Bash; that executable must
        already be available.
    conda_environment : str or None
        Name or full directory path of the Conda environment to
        activate. None skips Conda activation. Use a full path when the
        environment is installed outside Conda's usual environment
        directories.
    venv_path : str or None
        Full directory path of a Python virtual environment. TEMPER
        loads its ``bin/activate`` script into Bash. None skips this
        activation. Choose either this setting or Conda settings, so the
        selected Python is clear.
    variables : dict[str, str]
        Environment variable names and values to set before the
        calculation, for example thread limits. These override inherited
        values and are saved in job records, so do not put passwords or
        access tokens here.
    image : str or None
        Container image containing the software for a Bohrium job.
        Required for Bohrium; other execution resources ignore this
        setting.
    cache_root : str or None
        Full path to a directory for temporary compiler caches. Each job
        creates a separate directory inside it for Triton, Torch
        Inductor, extensions, CUDA, and XDG caches. Existing cache
        variables take precedence. None leaves cache locations
        unchanged. Successful jobs remove their own directory; failed
        jobs retain it and report its path. Cluster cleanup may remove
        temporary storage independently, so later jobs never depend on
        these files being present.
    required_executables : list[str]
        Executable names, such as ``dp``, that must be available after
        activation. TEMPER checks their presence without running them.
        An empty list skips this check. Presence alone does not
        establish that GPU execution works.
    check_dependencies : bool
        Whether to run ``python -m pip check`` after activation; default
        False. This detects missing or conflicting declared package
        dependencies. It does not load models or test GPU execution.
    """

    model_config = ConfigDict(extra="forbid")
    modules: list[str] = Field(default_factory=list)
    conda_init: str | None = None
    conda_environment: str | None = None
    venv_path: str | None = None
    variables: dict[str, str] = Field(default_factory=dict)
    image: str | None = None
    cache_root: str | None = None
    required_executables: list[str] = Field(default_factory=list)
    check_dependencies: bool = False

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.conda_init and not self.conda_environment:
            raise ValueError("conda_init requires conda_environment.")
        if self.venv_path and (self.conda_init or self.conda_environment):
            raise ValueError("Select either venv or Conda activation.")
        for value in (self.venv_path, self.conda_init, self.cache_root):
            if value and not PurePosixPath(value).is_absolute():
                raise ValueError(
                    "Activation paths must be absolute runner paths."
                )
        if any(
            not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key)
            for key in self.variables
        ):
            raise ValueError(
                "Environment variable names must be shell identifiers."
            )
        return self

    def shell_lines(self) -> list[str]:
        """Render quoted setup commands for a Bash task.

        Returns
        -------
        list[str]
            Module, Conda or venv activation, and export commands. Run
            with Bash set -e enabled; a missing activation tool stops
            the job.
        """
        from temper.job_management.resources import render_environment

        return render_environment(self)


class RemoteResource(BaseModel):
    """Describe one local, SSH scheduler, or Bohrium execution resource.

    Parameters
    ----------
    **data : object
        Values for the attributes below. Unknown settings are rejected.

    Attributes
    ----------
    name : str
        Human-readable resource name used in CLI filters.
    backend : str
        Where the job runs: Local starts it on this machine; Slurm and
        PBS submit it to a cluster through SSH; Bohrium uses its cloud
        service. Local requires Linux or WSL with Bash, not native
        Windows execution.
    remote_root : str or None
        Full Linux path to the parent directory for uploaded jobs on an
        SSH resource, for example ``/shared/my-account/temper-jobs``. It
        must be accessible to both the login node and compute nodes.
        None selects a user-specific directory under /tmp and emits a
        warning, because /tmp may be private to each node or job. Each
        attempt gets its own child directory; TEMPER deletes only
        children it created and can identify. Local and Bohrium ignore
        this setting.
    hostname, username : str or None
        Address of the SSH server and the account used to log in. Both
        are required for Slurm and PBS resources.
    port : int
        SSH port, default 22.
    key_filename, known_hosts : str or None
        Files on the submission machine containing the SSH private key
        and trusted server keys. Server-key verification protects
        against connecting to the wrong server. Relative paths in a
        profile file are resolved against that file's directory, not
        against the remote working directory.
    passphrase_env : str or None
        Name of the local environment variable containing the key
        passphrase. The passphrase itself is never stored in this model.
    password_env : str or None
        Local variable containing an SSH password, used only when no key
        file is configured. Passwords and key files are never copied
        into bundles.
    proxy_command : str or None
        Optional OpenSSH connection command for reaching a server
        through an intermediate host. TEMPER passes it unchanged to the
        SSH connection. None connects directly.
    timeout : int
        Connection and SSH command timeout in seconds; default 30.
    bohrium_project_id : int or None
        Bohrium project identifier; required for Bohrium.
    bohrium_access_key_env : str
        Local Bohrium credential variable name, default
        BOHRIUM_ACCESS_KEY.
    bohrium_machine_type, bohrium_platform : str
        Bohrium machine type and platform (default ali).
    resources : dict[str, JsonValue]
        Scheduler requests: number_node is the node count; gpu_per_node
        is the GPU count on each node; queue_name selects a queue or
        partition. For Slurm, specify tasks_per_node and cpus_per_task
        together to distinguish processes from CPUs assigned to each
        process. For example, values 1 and 4 request one task with four
        CPUs. Older cpu_per_node profiles keep DPDispatcher's original
        interpretation. custom_flags contains extra scheduler header
        lines. Omit memory flags to use cluster defaults.
    slurm_preflight : bool
        Whether to ask Slurm to check the job request before submitting
        it; default False. TEMPER runs ``sbatch --test-only`` and stops
        if Slurm rejects the request. If the check passes, TEMPER
        submits the real job. The check reserves no resources, cannot
        guarantee later acceptance, and does not test software or
        calculations. Only valid for Slurm. The previous configuration
        key slurm_test_only is still accepted.
    resolved_environment : RuntimeEnvironment or None
        Environment settings saved for this particular submission. None
        means no environment has been selected yet. TEMPER fills this
        field when submitting; users should configure environments
        instead.
    environments : dict[str, RuntimeEnvironment]
        Software settings for each MLFF family, keyed by names such as
        mace or dpa4. Submission requires an entry for the selected
        family. An empty entry uses the job's existing environment
        without activation commands.
    """

    model_config = ConfigDict(extra="forbid")
    name: str
    backend: Literal["Local", "Slurm", "PBS", "Bohrium"]
    remote_root: str | None = None
    hostname: str | None = None
    username: str | None = None
    port: int = Field(default=22, ge=1, le=65535)
    key_filename: str | None = None
    known_hosts: str | None = None
    passphrase_env: str | None = None
    password_env: str | None = None
    proxy_command: str | None = None
    timeout: int = Field(default=30, ge=1)
    bohrium_project_id: int | None = Field(
        default=None,
        validation_alias=AliasChoices("bohrium_project_id", "project_id"),
    )
    bohrium_access_key_env: str = Field(
        default="BOHRIUM_ACCESS_KEY",
        validation_alias=AliasChoices(
            "bohrium_access_key_env", "access_key_env"
        ),
    )
    bohrium_machine_type: str = Field(
        default="",
        validation_alias=AliasChoices("bohrium_machine_type", "machine_type"),
    )
    bohrium_platform: str = Field(
        default="ali",
        validation_alias=AliasChoices("bohrium_platform", "platform"),
    )
    resources: dict[str, JsonValue] = Field(default_factory=dict)
    slurm_preflight: bool = Field(
        default=False,
        validation_alias=AliasChoices("slurm_preflight", "slurm_test_only"),
    )
    resolved_environment: RuntimeEnvironment | None = Field(
        default=None,
        validation_alias=AliasChoices("resolved_environment", "environment"),
    )
    environments: dict[str, RuntimeEnvironment] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.backend in {"Slurm", "PBS"}:
            if not (
                self.hostname
                and self.username
                and (self.key_filename or self.password_env)
            ):
                raise ValueError(
                    "SSH resources require hostname, username and key_filename"
                    " or password_env."
                )
            remote = PurePosixPath(self.remote_root or "/tmp")
            if (
                not remote.is_absolute()
                or remote == PurePosixPath("/")
                or ".." in remote.parts
                or "\\" in str(remote)
            ):
                raise ValueError(
                    "remote_root must be an absolute POSIX scratch directory."
                )
        elif self.backend == "Bohrium" and (
            self.bohrium_project_id is None or not self.bohrium_machine_type
        ):
            raise ValueError("Bohrium requires project_id and machine_type.")
        allowed = {
            "number_node",
            "cpu_per_node",
            "gpu_per_node",
            "queue_name",
            "custom_flags",
            "kwargs",
            "tasks_per_node",
            "cpus_per_task",
        }
        if set(self.resources) - allowed:
            raise ValueError(
                "Unsupported scheduler options:"
                f" {sorted(set(self.resources) - allowed)}"
            )
        for name in (
            "number_node",
            "cpu_per_node",
            "gpu_per_node",
            "tasks_per_node",
            "cpus_per_task",
        ):
            if name in self.resources:
                value = self.resources[name]
                minimum = 0 if name == "gpu_per_node" else 1
                if type(value) is not int or value < minimum:
                    raise ValueError(
                        f"{name} must be an integer >= {minimum}."
                    )
        flags: list[str] = []
        if "custom_flags" in self.resources:
            raw_flags = self.resources["custom_flags"]
            if not isinstance(raw_flags, list):
                raise ValueError("custom_flags must be a list of strings.")
            for raw_flag in raw_flags:
                if not isinstance(raw_flag, str):
                    raise ValueError(
                        "Each custom_flags entry must be a string."
                    )
                flags.append(raw_flag)
        explicit = {"tasks_per_node", "cpus_per_task"} & self.resources.keys()
        if self.slurm_preflight and self.backend != "Slurm":
            raise ValueError("slurm_preflight requires Slurm.")
        if explicit:
            if self.backend != "Slurm" or len(explicit) != 2:
                raise ValueError(
                    "Slurm requires tasks_per_node and cpus_per_task together."
                )
            if "cpu_per_node" in self.resources:
                raise ValueError(
                    "Use tasks_per_node/cpus_per_task instead of cpu_per_node."
                )
            for flag in flags:
                if re.search(
                    r"(?:--(?:ntasks(?:-per-node)?|cpus-per-task)|"
                    r"(?:^|\s)-[nc])(?:[=\s]|\d|$)",
                    flag,
                ):
                    raise ValueError(
                        "Task/CPU custom_flags conflict with explicit Slurm"
                        " resources."
                    )
        return self

    @classmethod
    def from_file(cls, path: str | Path) -> Self:
        """Read a JSON or YAML resource profile.

        Parameters
        ----------
        path : str or Path
            Profile file, relative to the current working directory.

        Returns
        -------
        RemoteResource
            Validated profile with local key paths resolved against the
            file.

        Raises
        ------
        ValueError
            If settings are invalid or contain unsupported options.
        OSError
            If the profile cannot be read.
        """
        from temper.job_management.resources import load_resource_profile

        return cls.model_validate(load_resource_profile(path).model_dump())
