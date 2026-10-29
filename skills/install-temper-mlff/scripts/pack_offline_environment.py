"""Relocate an offline Conda prefix using its retained prefix-rewrite
metadata."""

from pathlib import Path
import json
import sys
import tarfile

from conda_pack import CondaEnv
from conda_pack.core import File

prefix, destination, archive = map(Path, sys.argv[1:4])
assert prefix.is_dir() and not archive.exists()
assert len(str(destination)) <= len(str(prefix))
rewrites = {}
python_record = next((prefix / "conda-meta").glob("python-3*.json"))
python_version = ".".join(
    json.loads(python_record.read_text())["version"].split(".")[:2]
)
for record in (prefix / "conda-meta").glob("*.json"):
    data = json.loads(record.read_text())
    cache = data.get("link", {}).get("source", "") if data.get("link") else ""
    if cache and Path(cache).exists():
        continue
    assert data.get(
        "paths_data"
    ), f"Missing relocation metadata: {record.name}"
    for entry in data["paths_data"]["paths"]:
        if entry.get("prefix_placeholder"):
            assert entry["file_mode"] in ("text", "binary")
            target = entry["_path"]
            if target.startswith("python-scripts/"):
                target = "bin/" + target.removeprefix("python-scripts/")
            elif target.startswith("site-packages/"):
                target = f"lib/python{python_version}/" + target
            rewrites[target] = entry["file_mode"]

environment = CondaEnv.from_prefix(str(prefix))
files = []
handled = set()
for file in environment.files:
    if file.target in rewrites:
        assert Path(file.source) == prefix / file.target
        file = File(
            file.source,
            file.target,
            is_conda=True,
            file_mode=rewrites[file.target],
            prefix_placeholder=str(prefix),
        )
        handled.add(file.target)
    elif (
        file.file_mode == "unknown"
        and Path(file.source).is_file()
        and not Path(file.source).is_symlink()
    ):
        source = Path(file.source)
        if source.stat().st_size > 16 * 1024 * 1024:
            with source.open("rb") as stream:
                first = stream.read(1024 * 1024)
                if b"\0" in first:
                    needle = str(prefix).encode()
                    tail = b""
                    block = first
                    while block:
                        assert (
                            needle not in tail + block
                        ), f"Unmanaged binary embeds prefix: {file.target}"
                        tail = block[-len(needle) :]
                        block = stream.read(1024 * 1024)
                    file = File(
                        file.source,
                        file.target,
                        is_conda=file.is_conda,
                        file_mode=None,
                        prefix_placeholder=None,
                    )
    files.append(file)
assert handled == set(rewrites), f"Unmapped paths: {set(rewrites)-handled}"
print(f"RESTORED_PREFIX_METADATA {len(handled)}", flush=True)
CondaEnv(str(prefix), files).pack(
    output=str(archive), format="tar", dest_prefix=str(destination)
)
with tarfile.open(archive, "r:") as packed:
    for name in handled:
        member = packed.getmember(name)
        if member.isfile():
            with packed.extractfile(member) as stream:
                needle = str(prefix).encode()
                tail = b""
                while block := stream.read(1024 * 1024):
                    assert (
                        needle not in tail + block
                    ), f"Old prefix retained in {name}"
                    tail = block[-len(needle) :]
print("DEEPMD_PACK_COMPLETE_AND_PREFIX_VERIFIED", flush=True)
