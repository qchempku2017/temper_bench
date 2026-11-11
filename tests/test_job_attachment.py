"""Check explicit job attachment without launching or changing remote jobs."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Literal

import pytest

from temper.job_management.schemas.records import JobRecord
from temper.job_management.schemas.resources import RemoteResource

if TYPE_CHECKING:
    from temper.job_management.dispatcher import DispatcherBackend


def _attachment_backend(
    name: Literal["Slurm", "Bohrium"], monkeypatch: pytest.MonkeyPatch,
) -> tuple[DispatcherBackend, list[str]]:
    """Build an offline backend that records snapshots and forbids launches."""
    pytest.importorskip("temper.job_management.dispatcher")
    from temper.job_management.dispatcher import DispatcherBackend

    backend = DispatcherBackend.__new__(DispatcherBackend)
    backend.profile = RemoteResource(
        name="cluster", backend=name, hostname="example.invalid",
        username="scientist", key_filename="key", remote_root="/scratch/temper",
        bohrium_project_id=7, bohrium_machine_type="test-machine",
    )
    backend.record = JobRecord(
        bundle_id="bundle", attempt_folder_name="attempt_001",
        resource=backend.profile.model_dump(mode="json"),
        dispatcher={"token": "a" * 32},
    )
    backend._job = SimpleNamespace(job_id="")
    snapshots: list[str] = []

    def snapshot() -> None:
        snapshots.append(str(backend._job.job_id))

    def forbidden_launch() -> None:
        pytest.fail("Attaching an existing job must never launch a calculation")

    monkeypatch.setattr(backend, "_snapshot", snapshot)
    monkeypatch.setattr(backend, "submit", forbidden_launch)
    return backend, snapshots


@pytest.mark.parametrize("evidence", ["saved", "receipt", "legacy", "queue"])
@pytest.mark.parametrize("matches", [False, True])
def test_ssh_attachment_checks_existing_evidence(
    monkeypatch: pytest.MonkeyPatch, evidence: str, matches: bool,
) -> None:
    """Reject contradictory IDs from saved state or remote launch records.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture replacing remote reads and recording local snapshots.
    evidence : str
        Source of an existing ID: saved state, current receipt, older
        .temper-receipt file, or queue/accounting lookup.
    matches : bool
        Whether the existing ID agrees with the supplied ID, 123.
    """
    backend, snapshots = _attachment_backend("Slurm", monkeypatch)
    existing = "123" if matches else "456"
    if evidence == "saved":
        backend._job.job_id = existing

    def read_result(phase: str) -> str:
        assert phase == "submit"
        if evidence == "receipt":
            return existing
        raise FileNotFoundError

    def read_file(name: str) -> str:
        assert name == ".temper-receipt"
        if evidence == "legacy":
            return existing
        raise FileNotFoundError

    def find_job() -> str:
        return existing if evidence == "queue" else ""

    backend.submission = SimpleNamespace(
        machine=SimpleNamespace(context=SimpleNamespace(read_file=read_file))
    )
    monkeypatch.setattr(backend, "_read_submission_result", read_result)
    monkeypatch.setattr(backend, "_find_scheduler_job", find_job)
    if matches:
        assert backend.recover("123")
        assert backend._job.job_id == "123"
        assert snapshots == ["123"]
    else:
        with pytest.raises(ValueError, match="conflicts"):
            backend.recover("123")
        assert backend._job.job_id == (existing if evidence == "saved" else "")
        assert snapshots == []


@pytest.mark.parametrize(
    "scenario", ["wrong_name", "missing_name", "wrong_project", "matching", "alternate_name"],
)
def test_cloud_attachment_checks_attempt_ownership(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    scenario: str,
) -> None:
    """Associate only matching cloud jobs without exposing their full response.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture replacing the cloud detail request and local snapshots.
    capsys : pytest.CaptureFixture[str]
        Captured output used to check that a signed result URL is not printed.
    scenario : str
        Wrong or absent job name, conflicting project, or matching ownership
        using either accepted job-name field.
    """
    backend, snapshots = _attachment_backend("Bohrium", monkeypatch)
    secret = "https://example.invalid/results?token=secret-download-token"
    detail: dict[str, str | int] = {
        "jobName": "temper-" + "a" * 32, "projectId": 7, "resultUrl": secret,
    }
    if scenario == "wrong_name":
        detail["jobName"] = "unrelated-job"
    elif scenario == "missing_name":
        detail.pop("jobName")
    elif scenario == "wrong_project":
        detail["projectId"] = 8
    elif scenario == "alternate_name":
        detail["name"] = detail.pop("jobName")
        detail.pop("projectId")

    def read_detail(job_id: int) -> dict[str, str | int]:
        assert job_id == 123
        return detail

    backend.submission = SimpleNamespace(
        machine=SimpleNamespace(job=SimpleNamespace(detail=read_detail))
    )
    if scenario in {"matching", "alternate_name"}:
        assert backend.recover("123")
        assert backend._job.job_id == 123
        assert snapshots == ["123"]
    else:
        with pytest.raises(ValueError) as error:
            backend.recover("123")
        assert secret not in str(error.value)
        assert backend._job.job_id == ""
        assert snapshots == []
    output = capsys.readouterr()
    assert secret not in output.out + output.err
    assert secret not in backend.record.model_dump_json()
