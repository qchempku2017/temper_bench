"""Execution resource and runtime environment configuration schemas."""
from __future__ import annotations

from pathlib import Path, PurePosixPath
import re
from typing import Literal, Self
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, JsonValue, model_validator


class RuntimeEnvironment(BaseModel):
    """Describe how a job activates its MLFF software.

    Parameters
    ----------
    **data : object
        Values for the attributes below. Setup commands run inside the task
        before TEMPER; any failed command stops the calculation.

    Attributes
    ----------
    modules : list[str]
        Module names to load in order; empty means no module commands.
    conda_init : str or None
        Optional absolute runner path to conda.sh. If absent, initialize Bash
        using ``conda shell.bash hook``; conda must be on the job's PATH.
    conda_environment : str or None
        Conda environment name or remote absolute prefix; absent skips conda.
    venv_path : str or None
        Absolute runner path to a Python virtual environment. Its bin/activate
        is sourced in Bash. Cannot be combined with Conda settings.
    variables : dict[str, str]
        Non-secret shell environment values. They are retained in job records.
    image : str or None
        Bohrium container image; required for Bohrium, unused on SSH hosts.
    """
    model_config = ConfigDict(extra="forbid")
    modules: list[str] = Field(default_factory=list)
    conda_init: str | None = None
    conda_environment: str | None = None
    venv_path: str | None = None
    variables: dict[str, str] = Field(default_factory=dict)
    image: str | None = None

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.conda_init and not self.conda_environment:
            raise ValueError("conda_init requires conda_environment.")
        if self.venv_path and (self.conda_init or self.conda_environment):
            raise ValueError("Select either venv or Conda activation.")
        for value in (self.venv_path, self.conda_init):
            if value and not PurePosixPath(value).is_absolute():
                raise ValueError("Activation paths must be absolute runner paths.")
        if any(not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) for key in self.variables):
            raise ValueError("Environment variable names must be shell identifiers.")
        return self

    def shell_lines(self) -> list[str]:
        """Render quoted setup commands for a Bash task.

        Returns
        -------
        list[str]
            Module, Conda or venv activation, and export commands. Run with
            Bash set -e enabled; a missing activation tool stops the job.
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
        Local, Slurm, PBS or Bohrium. Local jobs require a POSIX Bash runner.
    remote_root : str or None
        Absolute POSIX working directory for SSH attempts; ignored by Local
        and Bohrium. Prefer a configured home or shared scratch directory.
        None selects a user-specific /tmp directory with a submission warning.
        TEMPER deletes only the attempt directory it owns beneath this root.
    hostname, username : str or None
        SSH destination and account; required for Slurm/PBS.
    port : int
        SSH port, default 22.
    key_filename, known_hosts : str or None
        Local private-key and verified host-key files. Relative paths in a
        configuration file resolve against that file's directory.
    passphrase_env : str or None
        Name of the local environment variable containing the key passphrase.
        The passphrase itself is never stored in this model.
    password_env : str or None
        Local variable containing an SSH password, used only when no key file
        is configured. Passwords and key files are never copied into bundles.
    proxy_command : str or None
        Optional OpenSSH ProxyCommand, used verbatim for jump-host connections.
    timeout : int
        Connection and SSH command timeout in seconds; default 30.
    bohrium_project_id : int or None
        Bohrium project identifier; required for Bohrium.
    bohrium_access_key_env : str
        Local Bohrium credential variable name, default BOHRIUM_ACCESS_KEY.
    bohrium_machine_type, bohrium_platform : str
        Bohrium machine type and platform (default ali).
    resources : dict[str, JsonValue]
        dpdispatcher scheduler options, such as number_node, cpu_per_node,
        gpu_per_node, queue_name and custom_flags. One task per job is enforced.
    resolved_environment : RuntimeEnvironment or None
        Internal frozen environment for one attempt. None before submission.
        User profile files must use environments instead.
    environments : dict[str, RuntimeEnvironment]
        Complete environment replacements by MLFF family, such as mace or dpa4.
        A matching family entry is required at submission, even if empty.
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
    bohrium_project_id: int | None = Field(default=None, validation_alias=AliasChoices("bohrium_project_id", "project_id"))
    bohrium_access_key_env: str = Field(default="BOHRIUM_ACCESS_KEY", validation_alias=AliasChoices("bohrium_access_key_env", "access_key_env"))
    bohrium_machine_type: str = Field(default="", validation_alias=AliasChoices("bohrium_machine_type", "machine_type"))
    bohrium_platform: str = Field(default="ali", validation_alias=AliasChoices("bohrium_platform", "platform"))
    resources: dict[str, JsonValue] = Field(default_factory=dict)
    resolved_environment: RuntimeEnvironment | None = Field(default=None, validation_alias=AliasChoices("resolved_environment", "environment"))
    environments: dict[str, RuntimeEnvironment] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.backend in {"Slurm", "PBS"}:
            if not (self.hostname and self.username and (self.key_filename or self.password_env)):
                raise ValueError("SSH resources require hostname, username and key_filename or password_env.")
            remote = PurePosixPath(self.remote_root or "/tmp")
            if not remote.is_absolute() or remote == PurePosixPath("/") or ".." in remote.parts or "\\" in str(remote):
                raise ValueError("remote_root must be an absolute POSIX scratch directory.")
        elif self.backend == "Bohrium" and (self.bohrium_project_id is None or not self.bohrium_machine_type):
            raise ValueError("Bohrium requires project_id and machine_type.")
        allowed = {"number_node", "cpu_per_node", "gpu_per_node", "queue_name", "custom_flags", "kwargs"}
        if set(self.resources) - allowed:
            raise ValueError(f"Unsupported scheduler options: {sorted(set(self.resources) - allowed)}")
        for name in ("number_node", "cpu_per_node", "gpu_per_node"):
            if name in self.resources:
                value = self.resources[name]
                minimum = 0 if name == "gpu_per_node" else 1
                if type(value) is not int or value < minimum:
                    raise ValueError(f"{name} must be an integer >= {minimum}.")
        flags = self.resources.get("custom_flags")
        if "custom_flags" in self.resources and (
            not isinstance(flags, list)
            or not all(isinstance(flag, str) for flag in flags)
        ):
            raise ValueError("custom_flags must be a list of scheduler header lines.")
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
            Validated profile with local key paths resolved against the file.

        Raises
        ------
        ValueError
            If settings are invalid or contain unsupported options.
        OSError
            If the profile cannot be read.
        """
        from temper.job_management.resources import load_resource_profile
        return cls.model_validate(load_resource_profile(path).model_dump())
