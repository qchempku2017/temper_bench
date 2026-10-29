"""Regression coverage for deployment boundaries without GPU dependencies."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
import json
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import stat
import tarfile
from types import SimpleNamespace

import pytest
from ruamel.yaml import YAML

from temper.job_management.errors import SubmissionRejected
from temper.job_management.schemas.resources import (
    RemoteResource,
    RuntimeEnvironment,
)
from temper.mlff.pipeline.runner import prepare_bundle
from temper.schemas.mlff_train_bundle import MLFFTrainBundle
from temper.schemas.train_unit import TrainingUnit


@pytest.mark.parametrize(
    "family,model_type", [("dpa4", "dpa4"), ("dpa4c", "standard")]
)
def test_deepmd_legacy_normalization(
    tmp_path: Path,
    finetune_training_unit: TrainingUnit,
    mlff_spec_factory: Callable,
    family: str,
    model_type: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserve optimizer values, reject ambiguity, and normalize
    idempotently."""
    bundle = MLFFTrainBundle(
        training_unit=finetune_training_unit,
        mlff_spec=mlff_spec_factory(family),
    )
    root, _ = bundle.write_submit_folder(tmp_path / family)
    adapter = prepare_bundle(root / "bundle.json")
    path = root / adapter.input_path("config")
    config = json.loads(path.read_text())
    assert config["model"]["type"] == model_type
    config["model"].pop("type")
    config["training"]["optimizer"] = config.pop("optimizer")
    config["training"]["optimizer"]["weight_decay"] = 0.123
    original = deepcopy(config)
    normalized = adapter._normalize_config(config)
    assert config == original
    assert normalized["optimizer"]["weight_decay"] == 0.123
    assert adapter._normalize_config(normalized) == normalized
    config["optimizer"] = {"type": "Adam"}
    with pytest.raises(ValueError, match="Conflicting"):
        adapter._normalize_config(config)
    normalized["model"]["type"] = "wrong"
    with pytest.raises(ValueError, match="model.type"):
        adapter._normalize_config(normalized)
    path.write_text(json.dumps(original))
    monkeypatch.setattr(
        "temper.mlff.pipeline.data_preparation.deepmd.prepare_dataset",
        lambda *args: None,
    )
    adapter.prepare_training()
    assert json.loads(path.read_text())["optimizer"]["weight_decay"] == 0.123


@pytest.mark.parametrize("extended", [False, True])
def test_native_archive_upload(tmp_path: Path, extended: bool) -> None:
    """Exercise real archive IO with spaces, dot components and long paths."""
    from temper.job_management.dispatcher import _VerifiedContext

    if extended and os.name != "nt":
        pytest.skip("Native Windows prefix test")
    root = tmp_path / "space in root"
    if extended:
        root = (
            Path("\\\\?\\" + str(root))
            / ("long component " * 8).strip()
            / ("nested component " * 8).strip()
        )
    root.mkdir(parents=True)
    (root / "input file.txt").write_bytes(b"archive payload")
    native = str(root)
    context = object.__new__(_VerifiedContext)
    context.local_root = native
    context.remote_root = "/remote/space in directory"
    observed = []

    def put(local: str, remote: str) -> None:
        assert local == os.path.normpath(local)
        if extended:
            assert local.startswith("\\\\?\\")
        assert remote == "/remote/space in directory/abc.tgz"
        with tarfile.open(local) as archive:
            assert archive.getnames() == ["input file.txt"]
            assert (
                archive.extractfile("input file.txt").read()
                == b"archive payload"
            )
        observed.append(local)

    context.ssh_session = SimpleNamespace(put=put)
    context.block_checkcall = lambda cmd: None
    context.upload(
        SimpleNamespace(
            submission_hash="abc",
            forward_common_files=[],
            belonging_tasks=[
                SimpleNamespace(
                    task_work_path=".", forward_files=["./input file.txt"]
                )
            ],
        )
    )
    assert len(observed) == 1


def test_deepmd_explicit_model_preserves_order_and_alias(
    tmp_path: Path,
    finetune_training_unit: TrainingUnit,
    mlff_spec_factory: Callable,
) -> None:
    """Keep user model settings; the pretrain flag does not copy every
    field."""
    model = {
        "type": "SeZM",
        "type_map": ["Cu", "H"],
        "descriptor": {"type": "dpa4", "rcut": 5.0},
        "use_compile": False,
    }
    original = deepcopy(model)
    spec = mlff_spec_factory("dpa4", training_parameters={"model": model})
    bundle = MLFFTrainBundle(
        training_unit=finetune_training_unit, mlff_spec=spec
    )
    root, _ = bundle.write_submit_folder(tmp_path / "explicit")
    adapter = prepare_bundle(root / "bundle.json")
    config = json.loads((root / adapter.input_path("config")).read_text())
    assert config["model"] == original
    assert model == original
    assert adapter._normalize_config(config) == config


@pytest.mark.parametrize(
    "model",
    [
        {"preset": "dpa4-nano", "type_map": ["Cu"]},
        {"type_map": []},
        {"type_map": ["Cu", "Cu"]},
        {"type_map": [1]},
        {"type_map": "Cu"},
    ],
)
def test_deepmd_rejects_presets_and_invalid_type_map(
    mlff_spec_factory: Callable,
    model: dict,
) -> None:
    """Require explicit, unambiguous element indexing at bundle creation."""
    with pytest.raises(ValueError):
        mlff_spec_factory("dpa4", training_parameters={"model": model})


def test_explicit_slurm_resources() -> None:
    """Generate a single unambiguous task and CPU request; reject
    duplicates."""
    from dpdispatcher import Resources
    from temper.job_management.dispatcher import _ExplicitSlurm

    profile = dict(
        name="test",
        backend="Slurm",
        hostname="host",
        username="user",
        key_filename="key",
    )
    RemoteResource(
        **profile, resources={"tasks_per_node": 1, "cpus_per_task": 4}
    )
    machine = object.__new__(_ExplicitSlurm)
    machine.tasks_per_node, machine.cpus_per_task = 1, 4
    job = SimpleNamespace(
        resources=Resources(
            number_node=1,
            cpu_per_node=4,
            gpu_per_node=1,
            queue_name="gpu",
            group_size=1,
        )
    )
    header = machine.gen_script_header(job)
    assert header.count("--ntasks-per-node") == 1
    assert "--ntasks-per-node 1" in header and "--cpus-per-task 4" in header
    assert "--mem" not in header
    with pytest.raises(ValueError, match="conflict"):
        RemoteResource(
            **profile,
            resources={
                "tasks_per_node": 1,
                "cpus_per_task": 4,
                "custom_flags": ["#SBATCH --ntasks-per-node=2"],
            },
        )


@pytest.mark.parametrize("value", [None, False, 2, "--exclusive", {}, [1]])
def test_custom_flags_require_list_of_strings(value: object) -> None:
    """Reject malformed flags, including explicit null rather than absence."""
    with pytest.raises(ValueError, match="custom_flags"):
        RemoteResource(
            name="local", backend="Local", resources={"custom_flags": value}
        )


def test_slurm_preflight_legacy_key() -> None:
    """Read previous profiles while writing the clearer current key."""
    profile = dict(
        name="cluster",
        backend="Slurm",
        hostname="host",
        username="user",
        key_filename="key",
    )
    current = RemoteResource(**profile, slurm_preflight=True)
    legacy = RemoteResource(**profile, slurm_test_only=True)
    assert current == legacy
    assert current.model_dump()["slurm_preflight"] is True
    assert "slurm_test_only" not in current.model_dump()


@pytest.mark.parametrize(
    "code,output",
    [(1, ""), (0, ""), (0, "123;cluster\n"), (0, "123.server\n")],
)
def test_scheduler_receipts(tmp_path: Path, code: int, output: str) -> None:
    """Retain diagnostics and distinguish rejection, empty receipts and IDs."""
    from temper.job_management.dispatcher import DispatcherBackend

    backend = object.__new__(DispatcherBackend)
    files = {
        ".temper-submit.status": str(code),
        ".temper-submit.stdout": output,
        ".temper-submit.stderr": "invalid resource request",
    }
    backend.submission = SimpleNamespace(
        machine=SimpleNamespace(
            context=SimpleNamespace(read_file=files.__getitem__)
        )
    )
    backend.root = tmp_path
    backend.record = SimpleNamespace(dispatcher={})
    backend._checkpoint = lambda: None
    if code:
        with pytest.raises(SubmissionRejected):
            backend._read_submission_result("submit")
    else:
        text = backend._read_submission_result("submit")
        if output:
            assert backend._parse_id(text) in {"123", "123.server"}
        else:
            with pytest.raises(ValueError):
                backend._parse_id(text)
    assert backend.record.dispatcher["submit_result"] == {
        "exit_status": code,
        "stdout": output,
        "stderr": "invalid resource request",
    }


@pytest.mark.parametrize(
    "diagnostic",
    ["Socket timed out on send/recv", "unrecognized scheduler failure"],
)
def test_scheduler_client_failure_can_be_uncertain(
    tmp_path: Path, diagnostic: str
) -> None:
    """A nonzero client exit alone cannot prove the scheduler rejected a
    job."""
    from temper.job_management.dispatcher import DispatcherBackend

    backend = object.__new__(DispatcherBackend)
    files = {
        ".temper-submit.status": "1",
        ".temper-submit.stdout": "",
        ".temper-submit.stderr": diagnostic,
    }
    backend.submission = SimpleNamespace(
        machine=SimpleNamespace(
            context=SimpleNamespace(read_file=files.__getitem__)
        )
    )
    backend.root = tmp_path
    backend.record = SimpleNamespace(dispatcher={})
    backend._checkpoint = lambda: None
    with pytest.raises(RuntimeError) as error:
        backend._read_submission_result("submit")
    assert not isinstance(error.value, SubmissionRejected)


def test_cache_environment() -> None:
    """Cache defaults expand inside each job and preserve user variables."""
    lines = RuntimeEnvironment(
        cache_root="/tmp", required_executables=["dp"], check_dependencies=True
    ).shell_lines()
    assert any("mktemp -d" in line for line in lines)
    assert (
        'export TRITON_CACHE_DIR="${TRITON_CACHE_DIR-$temper_cache/triton}"'
        in lines
    )
    assert any("trap " in line and "Retained cache" in line for line in lines)
    assert "command -v dp >/dev/null" in lines
    assert "python -m pip check" in lines


@pytest.mark.parametrize("exit_status", [0, 7])
def test_cache_cleanup_shell(tmp_path: Path, exit_status: int) -> None:
    """Remove only owned successful caches and retain failed-job
    diagnostics."""
    bash = (
        "C:/Program Files/Git/bin/bash.exe"
        if os.name == "nt"
        else shutil.which("bash")
    )
    if not bash or not Path(bash).is_file():
        pytest.skip("Bash is required for the runtime shell integration check")
    # Bash and Python agree on relative paths, including on Git Bash.
    # /tmp is
    # the runtime scratch parent; Python reads its path through Bash
    # assertions.
    setup = RuntimeEnvironment(cache_root="/tmp").shell_lines()
    script = "\n".join(
        [
            "set -e",
            *setup,
            'printf "%s" "$temper_cache" > cache-path.txt',
            'test "$TRITON_CACHE_DIR" = "user-owned-cache"',
            'touch "$temper_cache/owned"',
            f"exit {exit_status}",
        ]
    )
    env = {**os.environ, "TRITON_CACHE_DIR": "user-owned-cache"}
    result = subprocess.run(
        [bash, "-c", script],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == exit_status, result.stderr
    check = (
        'cache=$(cat cache-path.txt); case "$cache" in /tmp/temper-cache.*) ;;'
        ' *) exit 1;; esac; test -f "$cache/owned"; rm -r -- "$cache"'
        if exit_status
        else 'cache=$(cat cache-path.txt); test ! -e "$cache"'
    )
    subprocess.run([bash, "-ec", check], cwd=tmp_path, check=True)
    assert ("Retained cache diagnostics:" in result.stderr) == bool(
        exit_status
    )


@pytest.mark.parametrize("extended", [False, True])
def test_native_download_paths(
    tmp_path: Path, extended: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Download through native paths and verify bytes before atomic
    replacement."""
    from temper.job_management.dispatcher import DispatcherBackend

    if extended and os.name != "nt":
        pytest.skip("Native Windows prefix test")
    root = tmp_path / "download space"
    if extended:
        root = Path("\\\\?\\" + str(root)) / ("long path " * 18).strip()
    root.mkdir(parents=True)
    backend = object.__new__(DispatcherBackend)
    backend.root = root
    backend.profile = SimpleNamespace(backend="Slurm")
    backend._bundle = None
    payload = b"downloaded bytes"
    digest = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(
        "temper.job_management.dispatcher.get_download_files",
        lambda bundle: (["stderr.log"], []),
    )

    def download(remote: str, local: str) -> None:
        assert remote == "/remote/attempt/stderr.log"
        assert local == os.path.normpath(local)
        if extended:
            assert local.startswith("\\\\?\\")
        Path(local).write_bytes(payload)

    context = SimpleNamespace(
        remote_root="/remote/attempt",
        sftp=SimpleNamespace(
            listdir_attr=lambda remote: [
                SimpleNamespace(filename="stderr.log", st_mode=stat.S_IFREG)
            ],
            get=download,
        ),
    )
    backend.submission = SimpleNamespace(
        machine=SimpleNamespace(context=context)
    )
    backend._command = lambda command: digest + "  stderr.log"
    assert backend.collect() == {"stderr.log": digest}
    assert (root / "stderr.log").read_bytes() == payload
    assert not (root / "stderr.log.partial").exists()


def test_pbs_resource_header() -> None:
    """Keep PBS node, CPU and GPU semantics independent of Slurm fields."""
    from dpdispatcher import Resources
    from dpdispatcher.machines.pbs import PBS

    machine = object.__new__(PBS)
    job = SimpleNamespace(
        resources=Resources(
            number_node=1,
            cpu_per_node=4,
            gpu_per_node=1,
            queue_name="gpu",
            group_size=1,
        )
    )
    header = machine.gen_script_header(job)
    assert "#PBS -l select=1:ncpus=4:ngpus=1" in header
    assert "#SBATCH" not in header
    with pytest.raises(ValueError, match="Slurm"):
        RemoteResource(
            name="pbs",
            backend="PBS",
            hostname="host",
            username="user",
            key_filename="key",
            resources={"tasks_per_node": 1, "cpus_per_task": 4},
        )


def test_missing_receipt_reconciles_scheduler(tmp_path: Path) -> None:
    """Find a lost Slurm receipt by its unique working directory, never
    retry."""
    from temper.job_management.dispatcher import DispatcherBackend

    backend = object.__new__(DispatcherBackend)

    def missing(name: str) -> str:
        raise FileNotFoundError(name)

    backend.profile = SimpleNamespace(backend="Slurm")
    backend._job = SimpleNamespace(job_id="")
    backend.submission = SimpleNamespace(
        machine=SimpleNamespace(
            context=SimpleNamespace(
                read_file=missing, remote_root="/scratch/temper_unique"
            )
        )
    )
    backend.record = SimpleNamespace(created_at="2026-10-27T00:00:00Z")
    backend._snapshot = lambda: None
    backend._command = (
        lambda command: "123|/scratch/temper_unique\n456|/scratch/other\n"
    )
    assert backend.recover()
    assert backend._job.job_id == "123"
    backend._job.job_id = ""
    backend._command = lambda command: ""
    assert not backend.recover()


def test_network_interruption_is_not_scheduler_rejection() -> None:
    """A missing command response stays uncertain and executes only once."""
    from temper.job_management.dispatcher import DispatcherBackend

    backend = object.__new__(DispatcherBackend)
    backend.profile = SimpleNamespace(backend="Slurm", slurm_preflight=False)
    context = SimpleNamespace(
        remote_root="/scratch/attempt", write_file=lambda *args: None
    )
    backend.submission = SimpleNamespace(
        machine=SimpleNamespace(
            context=context,
            gen_script=lambda job: "script",
            gen_script_command=lambda job: "run",
        )
    )
    backend._job = SimpleNamespace(script_file_name="job.sub")
    calls = []

    def interrupted(command: str) -> str:
        calls.append(command)
        raise ConnectionError("lost response")

    backend._command = interrupted
    with pytest.raises(ConnectionError):
        backend.submit()
    assert len(calls) == 1
