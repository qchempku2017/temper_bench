"""Command-line entry point for remote jobs and their local history."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from temper.job_management.manager import JobManager
from temper.job_management.storage_utils import atomic_json
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
    jobs = subparsers.add_parser(
        "job",
        help="Submit training bundles, check progress, and collect results.",
        description=(
            "Run training bundles on a configured execution resource and manage "
            "their saved job history. Each submission creates an attempt: one "
            "run of a bundle, with its own status, files, and logs. Use list to "
            "find bundle IDs, status --refresh to check progress, and "
            "synchronize_remote to collect finished results and clean up remote files."
        ),
    )
    commands = jobs.add_subparsers(dest="job_command", required=True, title="commands")
    command_help: dict[str, tuple[str, str]] = {
        "submit": (
            "Submit all bundles under the local bundle root.",
            "Read every model/domain/bundle directory's bundle.json under "
            "--bundle-write-root, as written by create_bundles. Packaged bundles in "
            "attempt directories are excluded. Package inputs and launch one job "
            "per eligible bundle. Save each attempt inside its bundle directory "
            "and return after submission unless --keep-watching is set. Bundles need "
            "dataset hashes; use refresh_bundle_identity for older bundles. A bundle with a "
            "successful, unfinished, uncertain, or not-yet-cleaned attempt cannot be submitted.",
        ),
        "list": (
            "List saved attempts and their last recorded status.",
            "Show bundle IDs, submission and calculation status, result collection "
            "and cleanup status, and local attempt directories. Read local history "
            "only; use status --refresh to check for remote changes.",
        ),
        "inspect": (
            "Show the full saved job records for a bundle as JSON.",
            "Read all matching attempts for a bundle from local history and print "
            "their complete records as JSON, including saved inputs, resource settings, "
            "status, and errors. This does not contact the execution resource.",
        ),
        "status": (
            "Show saved job status, optionally checking the execution resource.",
            "Show the last recorded status of matching attempts. Add --refresh to "
            "query the execution resource and save updated status. Neither mode "
            "downloads results or removes remote files.",
        ),
        "synchronize_remote": (
            "Update status, collect finished results, and clean up remote files.",
            "Check matching attempts once. For finished jobs (including failed or "
            "cancelled jobs), download available results and logs, verify the downloads, "
            "then delete the remote attempt's working files. Also retry pending "
            "cancellation requests. Interrupted submissions require recover; failed "
            "calculations are not restarted. Use --keep-watching to repeat until "
            "all selected attempts are cleaned up.",
        ),
        "logs": (
            "List locally available log file paths for a bundle.",
            "Show saved attempt status and paths to local .log, .out, .err, and "
            "events.jsonl files. This lists paths, not file contents, and does not "
            "fetch remote logs. Use download to collect files from finished jobs.",
        ),
        "download": (
            "Collect finished results and logs while keeping remote files.",
            "Update status and download and verify available results and logs for "
            "matching finished jobs, including failed or cancelled jobs. Running "
            "jobs are checked but their files are not downloaded. Pending cancellation "
            "requests may be retried. Remote working files are retained; use "
            "synchronize_remote when ready to clean them up.",
        ),
        "cancel": (
            "Request cancellation of unfinished attempts for a bundle.",
            "Record a cancellation request for each unfinished attempt of a bundle "
            "and stop jobs where possible. Completed attempts are left alone. Run "
            "synchronize_remote afterward to confirm termination, collect available "
            "logs, and clean up remote files.",
        ),
        "recover": (
            "Resume an interrupted submission or reconnect to an existing job.",
            "Use after an upload or submission was interrupted. Finish the original "
            "submission where possible, or look for evidence that the job was already "
            "launched. If its remote ID was not saved, supply a verified --job-id. "
            "This does not restart failed calculations; use resubmit for those.",
        ),
        "resubmit": (
            "Retry eligible failed bundles in new attempt directories.",
            "Select bundles with failed calculations in local history and submit "
            "each eligible bundle once, keeping earlier attempts. Skip bundles with "
            "a successful, unfinished, uncertain, or not-yet-cleaned attempt. Original "
            "input files must still be available and unchanged. Print a JSON list "
            "of outcomes and skip reasons. Use --dry-run to preview eligibility.",
        ),
        "refresh_bundle_identity": (
            "Recompute dataset hashes and replace the bundle in place.",
            "Read the bundle's local dataset files, recompute their hashes and the "
            "training-unit and bundle IDs, and replace the bundle at the same path. "
            "Bundles with existing attempts cannot be refreshed; create a separate "
            "bundle for the changed experiment. "
            "Use after intentionally changing dataset contents or for bundles without "
            "dataset hashes. Clear packaged file locations; the refreshed bundle must "
            "be packaged again before training or evaluation. Dataset files must be "
            "accessible. The original file remains intact until the replacement is "
            "ready. No job is submitted. Print the resulting bundle ID and bundle path.",
        ),
    }
    for name, (summary, description) in command_help.items():
        command = commands.add_parser(name, help=summary, description=description)
        if name == "refresh_bundle_identity":
            command.add_argument(
                "bundle", type=Path, metavar="BUNDLE_JSON",
                help="Existing bundle JSON to replace in place. Relative paths "
                     "use the current working directory, as does the training unit's "
                     "root_path when relative; dataset paths are not relative to this bundle.",
            )
            continue
        command.add_argument(
            "--bundle-write-root", type=Path, metavar="DIR",
            help="Local directory containing job history, packaged inputs, and collected "
                 "results. Use the same directory for submission and later commands. "
                 "Defaults to the DEFAULT_BUNDLE_WRITE_ROOT environment variable, or "
                 "./bundles if unset. Relative paths use the current working directory.",
        )
        if name == "submit":
            command.add_argument(
                "--resource-file", required=True, type=Path, metavar="PROFILE",
                help="JSON or YAML profile defining where jobs run (Local, Slurm, PBS, or "
                     "Bohrium), requested compute resources, and software environments. "
                     "Must include an environments entry for each submitted MLFF family. "
                     "Relative profile paths use the current working directory.",
            )
        else:
            required_bundle = name in {"inspect", "logs", "cancel", "recover"}
            command.add_argument(
                "--bundle-id", required=required_bundle, metavar="ID",
                help="Full bundle ID shown by job list; must match exactly, not a prefix. "
                     + ("Selects attempts for this bundle." if required_bundle else
                        "If omitted, include all bundle IDs that match the other filters."),
            )
            for option, value_help in (
                ("resource", "the resource profile's saved name field (not its file path)"),
                ("domain", "the training unit's domain value"),
            ):
                command.add_argument(
                    f"--{option}", metavar="NAME",
                    help=(
                        "Ignored by this command; --bundle-id alone selects the attempts."
                        if name in {"cancel", "recover"} else
                        f"Select attempts whose {option} exactly matches {value_help}. "
                        "If omitted, include all values. Combine filters to narrow the selection."
                    ),
                )
        if name in {"submit", "synchronize_remote"}:
            command.add_argument(
                "--keep-watching", "--keep_watching", action="store_true",
                help="Continue checking status, downloading finished results, and cleaning "
                     "remote files until all selected attempts are cleaned up. "
                     + ("Watches ALL attempts in the local history directory, including "
                        "earlier submissions. " if name == "submit" else
                        "Uses the bundle, resource, and domain filters. ")
                     + "Ctrl+C stops watching and leaves remote jobs running.",
            )
            command.add_argument(
                "--interval", type=float, default=60, metavar="SECONDS",
                help="Wait this many seconds between synchronization passes with "
                     "--keep-watching (default: 60). Must be positive when watching; "
                     "has no effect otherwise.",
            )
        if name in {"list", "logs", "resubmit"}:
            command.add_argument(
                "--failed", action="store_true",
                help=(
                    "Optional and redundant: resubmit always selects failed calculations."
                    if name == "resubmit" else
                    "Show only attempts whose saved calculation status is failed. "
                    "Connection, upload, or download errors alone do not count as "
                    "failed calculations; cancelled attempts are excluded."
                ),
            )
        if name == "status":
            command.add_argument(
                "--refresh", action="store_true",
                help="Contact the execution resource and save updated job status before "
                     "displaying it. Without this flag, read saved local status only. "
                     "Does not download results, cancel jobs, or delete remote files.",
            )
        if name == "recover":
            command.add_argument(
                "--job-id", metavar="REMOTE_ID",
                help="ID of an existing scheduler job or Bohrium job/draft when its ID "
                     "was not saved locally. First verify on the execution resource "
                     "that it belongs to this attempt. If omitted, recovery uses saved "
                     "submission information and available remote evidence.",
            )
        if name == "resubmit":
            command.add_argument(
                "--dry-run", action="store_true",
                help="Print which failed bundles are eligible and why others are skipped, "
                     "without packaging inputs or launching jobs. This checks saved "
                     "history, not input availability or resource connectivity.",
            )
            command.add_argument(
                "--resource-file", type=Path, metavar="PROFILE",
                help="JSON or YAML execution profile for all new attempts, relative to "
                     "the current working directory. If omitted, reuse the settings "
                     "saved with each bundle's latest matching failed attempt. "
                     "--resource still filters the old attempts by their saved name.",
            )
        command.add_argument(
            "--json", action="store_true",
            help=(
                "Optional and redundant: this command always prints JSON."
                if name in {"inspect", "resubmit"} else
                "Print full job records and submission errors as a JSON object with "
                "attempts and errors fields instead of a status listing."
                + (" Omits the log file path listing." if name == "logs" else "")
            ),
        )


def _bundle_paths(root: Path) -> list[Path]:
    """Find bundle-level bundles, excluding packaged attempt records."""
    paths = sorted(root.glob("*/*/bundle__*/bundle.json"))
    if not paths:
        raise ValueError(f"No bundles found under {root}; run create_bundles first.")
    return paths


def job_cli(args: argparse.Namespace) -> int:
    """Execute one job-management command.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments from add_job_parser. The refresh_bundle_identity command
        replaces the selected bundle in place after recomputing dataset hashes
        and identities, and clears its packaged file locations. The submit
        command reads all model/domain/bundle-level bundles under the selected
        bundle-write root; it does not read bundles inside attempt directories.

    Returns
    -------
    int
        Zero on an error-free operation, one if any selected operation failed.
        Confirmed failed calculations remain inspectable and are not relaunched
        unless the resubmit command was explicitly requested.

    Raises
    ------
    ValueError
        If the command input is invalid, a required dataset is missing, or a
        refresh would replace a bundle used by existing attempts.
    OSError
        If an input cannot be read or the refreshed bundle cannot be written.
        The original bundle is retained until its replacement is ready.
    KeyboardInterrupt
        If watching is stopped; remote calculations continue.
    """
    action = args.job_command
    if getattr(args, "keep_watching", False) and args.interval <= 0:
        raise ValueError("Watch interval must be positive.")
    if action == "refresh_bundle_identity":
        if any(args.bundle.parent.glob("attempt_*/job.json")) or (args.bundle.parent / "job.json").exists():
            raise ValueError("Cannot refresh a bundle used by existing attempts. "
                             "Create a separate bundle for the changed experiment.")
        data = json.loads(args.bundle.read_text(encoding="utf-8"))
        bundle = MLFFTrainBundle.model_validate(data, context={"verify_files": False})
        unit = bundle.training_unit.with_dataset_hashes()
        refreshed = bundle.model_copy(update={"training_unit": unit, "files": None})
        atomic_json(args.bundle, refreshed.model_dump(mode="json"))
        print(f"Refreshed {refreshed.mlff_train_bundle_id} at {args.bundle.resolve()}")
        return 0
    manager = JobManager(args.bundle_write_root)
    filters = {name: getattr(args, name, None) for name in ("bundle_id", "resource", "domain")}
    failures = []
    if action == "submit":
        resource = RemoteResource.from_file(args.resource_file)
        records = []
        submitted_ids: set[str] = set()
        with hashing_session():
            for path in _bundle_paths(manager.store.root):
                try:
                    bundle = MLFFTrainBundle.model_validate_json(path.read_text(encoding="utf-8"))
                    identity = str(bundle.mlff_train_bundle_id)
                    if identity in submitted_ids:
                        continue
                    submitted_ids.add(identity)
                    records.append(manager.submit(bundle, resource))
                except Exception as error:
                    failures.append({"bundle": str(path), "error": type(error).__name__, "reason": str(error)})
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
            print(f"{error['bundle']}: {error['error']}: {error['reason']}")
    if getattr(args, "keep_watching", False):
        manager.watch(args.interval, **filters)
    return int(bool(failures) or any(record.error is not None for record in records))
