"""User-facing remote resource and runtime environment configuration."""
from __future__ import annotations

from pathlib import Path, PurePosixPath
import re
import shlex
from typing import Literal, Self
from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator
from ruamel.yaml import YAML


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
        Absolute remote path to conda.sh, required with conda_environment.
    conda_environment : str or None
        Conda environment name or remote absolute prefix; absent skips conda.
    variables : dict[str, str]
        Non-secret shell environment values. They are retained in job records.
    image : str or None
        Bohrium container image; required for Bohrium, unused on SSH hosts.
    """
    model_config = ConfigDict(extra="forbid")
    modules: list[str] = Field(default_factory=list)
    conda_init: str | None = None
    conda_environment: str | None = None
    variables: dict[str, str] = Field(default_factory=dict)
    image: str | None = None

    @model_validator(mode="after")
    def _check(self) -> Self:
        if bool(self.conda_init) != bool(self.conda_environment):
            raise ValueError("Specify both conda_init and conda_environment.")
        if any(not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) for key in self.variables):
            raise ValueError("Environment variable names must be shell identifiers.")
        return self

    def shell_lines(self) -> list[str]:
        """Render quoted setup commands for a Bash task.

        Returns
        -------
        list[str]
            Module, conda and export commands. Run with Bash set -e enabled.
        """
        lines = [f"module load {shlex.quote(module)}" for module in self.modules]
        if self.conda_init:
            lines += [f"source {shlex.quote(self.conda_init)}", f"conda activate {shlex.quote(self.conda_environment or '')}"]
        return lines + [f"export {key}={shlex.quote(value)}" for key, value in self.variables.items()]


class RemoteResource(BaseModel):
    """Describe one SSH scheduler or Bohrium execution resource.

    Parameters
    ----------
    **data : object
        Values for the attributes below. Unknown settings are rejected.

    Attributes
    ----------
    name : str
        Human-readable resource name used in CLI filters.
    backend : str
        Slurm, PBS or Bohrium. SSH uses key-file authentication.
    remote_root : str
        Absolute POSIX scratch directory for SSH attempts; ignored by Bohrium.
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
    proxy_command : str or None
        Optional OpenSSH ProxyCommand, used verbatim for jump-host connections.
    timeout : int
        Connection and SSH command timeout in seconds; default 30.
    project_id : int or None
        Bohrium project identifier; required for Bohrium.
    access_key_env : str
        Local Bohrium credential variable name, default BOHRIUM_ACCESS_KEY.
    machine_type, platform : str
        Bohrium machine type and platform (default ali).
    resources : dict[str, JsonValue]
        dpdispatcher scheduler options, such as number_node, cpu_per_node,
        gpu_per_node, queue_name and custom_flags. One task per job is enforced.
    environment : RuntimeEnvironment
        Default runtime setup.
    environments : dict[str, RuntimeEnvironment]
        Complete environment replacements by MLFF family, such as mace or dpa4.
        The chosen replacement is frozen in each attempt's environment field.
    """
    model_config = ConfigDict(extra="forbid")
    name: str
    backend: Literal["Slurm", "PBS", "Bohrium"]
    remote_root: str = "/tmp/temper"
    hostname: str | None = None
    username: str | None = None
    port: int = Field(default=22, ge=1, le=65535)
    key_filename: str | None = None
    known_hosts: str | None = None
    passphrase_env: str | None = None
    proxy_command: str | None = None
    timeout: int = Field(default=30, ge=1)
    project_id: int | None = None
    access_key_env: str = "BOHRIUM_ACCESS_KEY"
    machine_type: str = ""
    platform: str = "ali"
    resources: dict[str, JsonValue] = Field(default_factory=dict)
    environment: RuntimeEnvironment = Field(default_factory=RuntimeEnvironment)
    environments: dict[str, RuntimeEnvironment] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.backend != "Bohrium":
            if not all((self.hostname, self.username, self.key_filename)):
                raise ValueError("SSH resources require hostname, username and key_filename.")
            remote = PurePosixPath(self.remote_root)
            if not remote.is_absolute() or remote == PurePosixPath("/") or ".." in remote.parts or "\\" in self.remote_root:
                raise ValueError("remote_root must be an absolute POSIX scratch directory.")
        elif self.project_id is None or not self.machine_type:
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
        if "custom_flags" in self.resources and (
            not isinstance(self.resources["custom_flags"], list)
            or not all(isinstance(flag, str) for flag in self.resources["custom_flags"])
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
        path = Path(path).expanduser().resolve()
        data = YAML(typ="safe").load(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("A resource profile must contain a mapping of settings.")
        for key in ("key_filename", "known_hosts"):
            if data.get(key):
                local = Path(data[key]).expanduser()
                data[key] = str((path.parent / local).resolve())
        return cls.model_validate(data)
