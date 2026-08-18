from __future__ import annotations

from pydantic import field_validator

from src.temper.schemas.base import JsonIOModel
from src.temper.schemas.utils import validate_relative_extxyz_path


class FrameReference(JsonIOModel):
    """Persisted reference to a single structure frame in a data group.

    References are lightweight: they only store the identity of a frame
    (domain, relative extxyz source filename, and nonnegative frame index).
    Structures and descriptors are never stored.

    Attributes:
        domain (str): Name of the data domain the frame belongs to.
        filename (str): Relative path to the extxyz source file, relative to
            the domain directory. Must be relative, end with ``.extxyz``, and
            must not contain directory-traversal segments.
        frame_index (int): Zero-based, nonnegative index of the frame within
            the source file.
    """

    domain: str
    filename: str
    frame_index: int

    @field_validator("domain")
    @classmethod
    def validate_domain(cls, value: str) -> str:
        """Require a non-empty domain name."""
        if not isinstance(value, str) or not value.strip():
            raise ValueError("domain must be a non-empty string.")
        return value

    @field_validator("filename")
    @classmethod
    def validate_filename(cls, value: str) -> str:
        """Require a safe relative extxyz source path."""
        return validate_relative_extxyz_path(value)

    @field_validator("frame_index")
    @classmethod
    def validate_frame_index(cls, value: int) -> int:
        """Reject negative frame indices."""
        if value < 0:
            raise ValueError(f"frame_index must be nonnegative, got {value}.")
        return value

    @property
    def identity(self) -> tuple[str, str, int]:
        """A hashable identity tuple ``(domain, filename, frame_index)``.

        Used for set-membership checks (e.g. duplicate detection and
        train/validation complement computation) without relying on model
        hashing.
        """
        return self.domain, self.filename, self.frame_index
