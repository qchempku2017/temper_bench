"""Content-addressed references to local input files."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Self

from pydantic import ConfigDict, Field, field_serializer, field_validator
from temper.schemas.base import MSONableModel


class LocalArtifactRef(MSONableModel):
    """Reference a local file by absolute path and SHA-256 content digest.

    path locates the source to copy; sha256 supplies path-independent identity
    and allows consumers to detect changes after the reference was created.
    """

    model_config = ConfigDict(extra="forbid")
    path: Path
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def from_path(cls, path: str | Path) -> Self:
        """Return a reference to an existing file, expanding and resolving path.

        Raises ValueError if path is not a file, or OSError if reading fails.
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
