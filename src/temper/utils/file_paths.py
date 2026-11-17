"""Validate relative dataset paths and resolve paths inside bundles."""

from __future__ import annotations

from pathlib import Path, PurePosixPath, PureWindowsPath


def validate_submit_relative_path(value: str, *, label: str = "path") -> str:
    """Validate a portable path that is safe to place below a submit
    directory.

    Submit bundles are created locally and may later be copied to a
    different operating system. Their stored paths therefore use POSIX
    ``/`` separators and must be relative to the bundle root. This
    helper rejects absolute paths, Windows drive paths, traversal,
    redundant segments, and other forms whose meaning could change after
    transport.

    Parameters
    ----------
    value : str
        Path to validate, written with POSIX separators. It may contain
        subdirectories, for example ``"datasets/train.extxyz"``.
    label : str, optional
        Human-readable name used in validation errors. It does not
        affect the returned path.

    Returns
    -------
    str
        ``value`` unchanged after validation.

    Raises
    ------
    TypeError
        If ``value`` or ``label`` is not a string.
    ValueError
        If ``value`` is empty, absolute, drive-qualified, uses
        backslashes, or contains empty, current-directory,
        parent-directory, or unnormalized path segments.
    """
    if not isinstance(value, str):
        raise TypeError(
            f"{label} must be a string, got {type(value).__name__}."
        )
    if not isinstance(label, str):
        raise TypeError(f"label must be a string, got {type(label).__name__}.")
    if not value:
        raise ValueError(f"{label} must be a non-empty string.")
    if "\\" in value:
        raise ValueError(f"{label} must use '/' separators, got {value!r}.")

    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if posix.is_absolute() or windows.is_absolute() or windows.drive:
        raise ValueError(f"{label} must be relative, got {value!r}.")
    if not posix.parts or any(part in {"", ".", ".."} for part in posix.parts):
        raise ValueError(
            f"{label} must not contain empty, '.' or '..' segments: {value!r}."
        )
    if str(posix) != value:
        raise ValueError(f"{label} must be normalized, got {value!r}.")
    return value


def resolve_path_in_bundle(root: Path, name: str) -> Path:
    """Resolve a relative path and require it to stay within the bundle.

    Parameters
    ----------
    root : pathlib.Path
        Bundle directory. A relative root is interpreted against the current
        working directory; user-home abbreviations are not expanded.
    name : str
        Normalized path relative to root, using POSIX '/' separators.
        May identify an input, output, or settings file in a subdirectory.

    Returns
    -------
    pathlib.Path
        Absolute path with symbolic links resolved. The target need not exist;
        this function neither reads file contents nor creates files.

    Raises
    ------
    TypeError
        If name is not a string.
    ValueError
        If name is not a valid bundle-relative path or the resolved path
        leaves root, including through a symbolic link.
    OSError
        If filesystem path resolution fails.
    """
    validate_submit_relative_path(name, label="result file")
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Result path escapes its directory: {name}")
    return path


def validate_relative_extxyz_path(path: str) -> str:
    """Validate that a path is a relative extxyz path without directory
    traversal.

    Used to validate persisted source filenames. The path must be
    relative (not absolute, no drive/root), must not contain ``..``
    traversal segments, and must end with the ``.extxyz`` suffix.
    Subdirectories (e.g. ``subdir/file.extxyz``) are allowed as long as
    they are relative.

    Parameters
    ----------
    path : str
        The source filename to validate.

    Returns
    -------
    str
        The validated path, unchanged.

    Raises
    ------
    TypeError
        If ``path`` is not a string.
    ValueError
        If ``path`` is absolute, contains traversal, or does not end
        with ``.extxyz``.
    """
    if not isinstance(path, str):
        raise TypeError(f"path must be a str, got {type(path).__name__}.")

    filepath = Path(path)

    # Reject absolute paths. On Windows a leading-root path such as
    # "/a.extxyz" is drive-relative (is_absolute() is False) but is
    # still
    # rooted, so a non-empty ``root`` is rejected as well.
    if filepath.is_absolute() or filepath.root:
        raise ValueError(
            f"Source filename must be relative, got absolute path: {path}."
        )

    if ".." in filepath.parts:
        raise ValueError(
            "Source filename must not contain directory traversal, got:"
            f" {path}."
        )

    if filepath.suffix != ".extxyz":
        raise ValueError(
            f"Source filename must have the '.extxyz' extension, got: {path}."
        )

    return path
