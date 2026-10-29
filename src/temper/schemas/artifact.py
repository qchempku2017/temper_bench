"""Content-addressed references to local input files."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Self

from pydantic import ConfigDict, Field, field_serializer, field_validator
from temper.schemas.base import MSONableModel


class LocalArtifactRef(MSONableModel):
    """Identify a local input file by its path and contents.

    Parameters
    ----------
    path : pathlib.Path
        Absolute path to the source file that TEMPER will copy into a
        bundle. Use from_path to expand ``~`` and resolve relative paths
        automatically.
    sha256 : str
        SHA-256 checksum: a 64-character hexadecimal value computed from
        the file's bytes. It identifies the contents independently of
        the filename and lets TEMPER detect changes made after this
        reference was created.

    Attributes
    ----------
    path : pathlib.Path
        Location of the source file on the submission machine.
    sha256 : str
        Expected checksum, used when verifying and copying the source
        file.
    """

    model_config = ConfigDict(extra="forbid")
    path: Path
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def from_path(cls, path: str | Path) -> Self:
        """Resolve a local file and compute its content checksum.

        Parameters
        ----------
        path : str or pathlib.Path
            Source file. A leading ``~`` is expanded; relative paths are
            resolved against the current working directory.

        Returns
        -------
        LocalArtifactRef
            Absolute source path and SHA-256 checksum of its current
            bytes. This reads the file but does not copy or change it.

        Raises
        ------
        ValueError
            If the resolved path is not an existing regular file.
        OSError
            If the file cannot be read.
        """
        local_path = Path(path).expanduser().resolve()
        if not local_path.is_file():
            raise ValueError(f"Local artifact does not exist: {local_path}.")
        with local_path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        return cls(path=local_path, sha256=digest)

    @field_validator("path", mode="before")
    @classmethod
    def _load_monty_path(cls, value: Any) -> Any:
        """Accept paths serialized by Monty encoders."""
        if isinstance(value, dict) and value.get("@module") == "pathlib":
            return value.get("string", value)
        return value

    @field_serializer("path")
    def _serialize_path(self, value: Path) -> str:
        """Return the local path as a plain string."""
        return str(value)
