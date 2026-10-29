"""Check the optional installation copy tool without model dependencies."""

from pathlib import Path
import os

import pytest

from compatibility.copy_environment import copy_environment


def test_environment_copy_preserves_inputs_and_empty_directories(
    tmp_path: Path,
) -> None:
    """Copy a tree with spaces while preserving source bytes and metadata."""
    source = tmp_path / "source"
    (source / "empty directory").mkdir(parents=True)
    binary = source / "data file.bin"
    binary.write_bytes(bytes(range(256)))
    os.utime(binary, (1000000000, 1000000000))
    destination = tmp_path / "permanent environment"
    assert copy_environment(source, destination, workers=2) == 1
    assert (destination / "empty directory").is_dir()
    assert (destination / binary.name).read_bytes() == binary.read_bytes()
    copied_time = (destination / binary.name).stat().st_mtime
    assert copied_time == binary.stat().st_mtime
    with pytest.raises(FileExistsError):
        copy_environment(source, destination)
    with pytest.raises(ValueError):
        copy_environment(source, source / "nested")
    assert binary.read_bytes() == bytes(range(256))


@pytest.mark.skipif(os.name == "nt", reason="Requires unprivileged symlinks")
def test_environment_copy_does_not_follow_symbolic_links(
    tmp_path: Path,
) -> None:
    """Preserve external and dangling links without copying their targets."""
    source = tmp_path / "source"
    source.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "untouched").write_text("not part of the environment")
    (source / "external").symlink_to(outside, target_is_directory=True)
    (source / "dangling").symlink_to("missing")
    destination = tmp_path / "copy"
    assert copy_environment(source, destination) == 2
    assert (destination / "external").is_symlink()
    assert os.readlink(destination / "external") == str(outside)
    assert os.readlink(destination / "dangling") == "missing"
