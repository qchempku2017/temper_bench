"""Command-line entry point for remote jobs and their local history."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from temper.job_management.manager import JobManager
from temper.schemas.mlff_train_bundle import MLFFTrainBundle
from temper.job_management.schemas.resources import RemoteResource
from temper.utils.file_hashing import hashing_session


def add_job_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the job command and its subcommands.

    Parameters
    ----------
    subparsers : argparse._SubParsersAction
        Top-level parser's subcommand collection. This only configures CLI
        arguments; it does not import dpdispatcher or contact a remote service.
    """
    jobs = subparsers.add_parser("job", help="Submit bundles and manage persistent remote attempts.")
    commands = jobs.add_subparsers(dest="job_command", required=True)
    for name in ("submit", "list", "inspect", "status", "synchronize_remote", "logs", "download", "cancel", "recover", "resubmit", "upgrade_bundle"):
        command = commands.add_parser(name)
        command.add_argument("--bundle-write-root", type=Path, help="Store root; defaults to DEFAULT_BUNDLE_WRITE_ROOT or ./bundles.")
        if name == "upgrade_bundle":
            command.add_argument("bundle", type=Path)
            command.add_argument("--output", required=True, type=Path, help="New recipe path; existing files are never overwritten.")
            continue
        if name == "submit":
            command.add_argument("bundles", nargs="+", type=Path, help="Bundle JSON files, or JSON lists of recipe file paths.")
            command.add_argument("--resource-file", required=True, type=Path)
        else:
            command.add_argument("--bundle-id", required=name in {"inspect", "logs", "cancel", "recover"})
            command.add_argument("--resource", help="Filter by saved resource name.")
            command.add_argument("--domain", help="Filter by training-unit domain.")
        if name in {"submit", "synchronize_remote"}:
            command.add_argument("--keep-watching", "--keep_watching", action="store_true", help="Keep synchronizing in this process; Ctrl+C leaves remote jobs running.")
            command.add_argument("--interval", type=float, default=60, help="Polling interval in seconds.")
        if name in {"list", "logs", "resubmit"}:
            command.add_argument("--failed", action="store_true", help="Select failed computations (resubmit always selects failures).")
        if name == "status":
            command.add_argument("--refresh", action="store_true", help="Query remote status without downloading or deleting.")
        if name == "recover":
            command.add_argument("--job-id", help="Existing remote job ID, verified by the user.")
        if name == "resubmit":
            command.add_argument("--dry-run", action="store_true")
            command.add_argument("--resource-file", type=Path, help="Use a different execution resource for new attempts.")
        command.add_argument("--json", action="store_true", help="Print structured records instead of a concise listing.")


def _recipe_paths(paths: list[Path]) -> list[Path]:
    result = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            if not all(isinstance(item, str) for item in data):
                raise ValueError("A submission list must contain recipe file paths.")
            result.extend((path.parent / item).resolve() for item in data)
        else:
            result.append(path)
    return list(dict.fromkeys(result))


def job_cli(args: argparse.Namespace) -> int:
    """Execute one job-management command.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments from add_job_parser.

    Returns
    -------
    int
        Zero on an error-free operation, one if any selected operation failed.
        Confirmed failed calculations remain inspectable and are not relaunched
        unless the resubmit command was explicitly requested.

    Raises
    ------
    ValueError
        If the command input is invalid or an upgrade destination exists.
    KeyboardInterrupt
        If watching is stopped; remote calculations continue.
    """
    manager = JobManager(args.bundle_write_root)
    action = args.job_command
    if getattr(args, "keep_watching", False) and args.interval <= 0:
        raise ValueError("Watch interval must be positive.")
    if action == "upgrade_bundle":
        data = json.loads(args.bundle.read_text(encoding="utf-8"))
        bundle = MLFFTrainBundle.model_validate(data, context={"verify_files": False})
        unit = bundle.training_unit.with_dataset_hashes()
        upgraded = bundle.model_copy(update={"training_unit": unit, "files": None})
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(upgraded.model_dump_json(indent=2) + "\n")
        print(f"Wrote {upgraded.mlff_train_bundle_id} to {args.output.resolve()}")
        return 0
    filters = {name: getattr(args, name, None) for name in ("bundle_id", "resource", "domain")}
    failures = []
    if action == "submit":
        resource = RemoteResource.from_file(args.resource_file)
        records = []
        with hashing_session():
            for path in _recipe_paths(args.bundles):
                try:
                    bundle = MLFFTrainBundle.model_validate_json(path.read_text(encoding="utf-8"))
                    records.append(manager.submit(bundle, resource))
                except Exception as error:
                    failures.append({"recipe": str(path), "error": type(error).__name__, "reason": str(error)})
    elif action == "resubmit":
        outcomes = manager.resubmit(**filters, dry_run=args.dry_run,
                                    replacement_resource=RemoteResource.from_file(args.resource_file) if args.resource_file else None)
        print(json.dumps(outcomes, indent=2))
        return int(any(item["action"] in {"error", "uncertain", "uploading", "prepared"} for item in outcomes))
    elif action in {"synchronize_remote", "download"} or (action == "status" and args.refresh):
        records = manager.synchronize(**filters, refresh_only=action == "status", download_only=action == "download")
    elif action == "cancel":
        records = manager.cancel(args.bundle_id)
    elif action == "recover":
        records = manager.recover(args.bundle_id, args.job_id)
    else:
        records = manager.list_attempts(**filters, failed=getattr(args, "failed", False))
    if args.json or action == "inspect":
        print(json.dumps({"attempts": [record.model_dump(mode="json") for record in records], "errors": failures}, indent=2))
    else:
        for record in records:
            root = manager.store.path(record)
            print(f"{record.bundle_id}  {root.name}  {record.submission_status}/{record.computation_status}  results={record.collection_status} cleanup={record.cleanup_status}\n  {root}")
            if record.error:
                print(f"  {record.error}")
            if action == "logs":
                for path in sorted(root.rglob("*")):
                    if path.is_file() and (path.suffix in {".log", ".out", ".err"} or path.name == "events.jsonl"):
                        print(f"  {path}")
        for error in failures:
            print(f"{error['recipe']}: {error['error']}: {error['reason']}")
    if getattr(args, "keep_watching", False):
        manager.watch(args.interval, **filters)
    return int(bool(failures) or any(record.error is not None for record in records))
