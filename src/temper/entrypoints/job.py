"""Command-line entry point for remote jobs and their local history."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys

from temper.job_management.manager import JobManager
from temper.job_management.paths import bundle_directory
from temper.job_management.storage_utils import store_lock, write_json_atomically
from temper.schemas.mlff_train_bundle import MLFFTrainBundle
from temper.job_management.schemas.resources import RemoteResource


def add_job_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the job command and its subcommands.

    Parameters
    ----------
    subparsers : argparse._SubParsersAction
        Top-level parser's subcommand collection. This only configures
        CLI arguments; it does not import dpdispatcher or contact a
        remote service.
    """
    jobs = subparsers.add_parser(
        "job",
        help="Submit training bundles, check progress, and collect results.",
        description=(
            "Run training bundles on a configured execution resource and"
            " manage their saved job history. Each new calculation creates an"
            " attempt: one run of a bundle, with its own status, files, and"
            " logs. Use list to find bundle IDs, status --refresh to check"
            " progress, and synchronize_remote to collect finished results and"
            " clean up remote files."
        ),
    )
    commands = jobs.add_subparsers(
        dest="job_command", required=True, title="commands"
    )
    command_help: dict[str, tuple[str, str]] = {
        "submit": (
            "Start or safely resume selected bundles under the local root.",
            (
                "Read every model/domain/bundle directory's bundle.json under"
                " --bundle-write-root, as written by create_bundles. Packaged"
                " bundles in attempt directories are excluded. Check saved"
                " history and remote status, resume interrupted submissions,"
                " and start at most one new attempt per eligible bundle."
                " Successful and running calculations are not launched again."
                " Uncertain submissions are reported without launching a"
                " replacement. Confirmed failures are collected and cleaned"
                " before retrying; cancelled jobs require --retry-cancelled."
                " Use --preview to inspect these decisions without changing"
                " remote jobs or files. Bundles need dataset hashes; use"
                " refresh_bundle_identity for older bundles."
            ),
        ),
        "list": (
            "List saved attempts and their last recorded status.",
            (
                "Show bundle IDs, submission and calculation status, result"
                " collection and cleanup status, and local attempt"
                " directories. Read local history only; use status --refresh"
                " to check for remote changes."
            ),
        ),
        "inspect": (
            "Show the full saved job records for a bundle as JSON.",
            (
                "Read all matching attempts for a bundle from local history"
                " and print their complete records as JSON, including saved"
                " inputs, resource settings, status, and errors. This does not"
                " contact the execution resource."
            ),
        ),
        "status": (
            (
                "Show saved job status, optionally checking the execution"
                " resource."
            ),
            (
                "Show the last recorded status of matching attempts. Add"
                " --refresh to query the execution resource and save updated"
                " status. Neither mode downloads results or removes remote"
                " files."
            ),
        ),
        "synchronize_remote": (
            (
                "Update status, collect finished results, and clean up remote"
                " files."
            ),
            (
                "Check matching attempts once. For finished jobs (including"
                " failed or cancelled jobs), download available results and"
                " logs, verify the downloads, then delete the remote attempt's"
                " working files unless --keep-remote-files is set. Also retry"
                " pending cancellation requests."
                " Run submit again to resume interrupted submissions; failed"
                " calculations"
                " are not restarted. Use --keep-watching to repeat until all"
                " selected attempts are cleaned up, or their results are"
                " collected and verified with --keep-remote-files."
            ),
        ),
        "logs": (
            "List locally available log file paths for a bundle.",
            (
                "Show saved attempt status and paths to local .log, .out,"
                " .err, and events.jsonl files. This lists paths, not file"
                " contents, and does not fetch remote logs. Use"
                " synchronize_remote --keep-remote-files to collect files"
                " from finished jobs without deleting remote files."
            ),
        ),
        "cancel": (
            "Request cancellation of unfinished attempts for a bundle.",
            (
                "Record a cancellation request for each unfinished attempt of"
                " a bundle and stop jobs where possible. Completed attempts"
                " are left alone. Run synchronize_remote afterward to confirm"
                " termination, collect available logs, and clean up remote"
                " files."
            ),
        ),
        "attach": (
            "Associate an uncertain attempt with an existing remote job.",
            (
                "Use only when automatic recovery cannot find the job."
                " Identify one saved attempt and the remote job that belongs"
                " to it. This command checks available evidence and saves"
                " the association; it never launches a calculation. Run"
                " submit or synchronize_remote afterward to continue."
            ),
        ),
        "abandon": (
            "Stop tracking selected unresolved attempts after confirmation.",
            (
                "List unresolved attempts that prevent another submission."
                " Check the scheduler or cloud service before proceeding: a"
                " missing queue entry does not prove that a job never started."
                " Abandoning changes local history only; it does not cancel"
                " jobs or delete files. A later submit may duplicate a job"
                " that is still running. The default answer is no. Use"
                " --preview to inspect candidates, or --yes to confirm them"
                " without an interactive question."
            ),
        ),
        "refresh_bundle_identity": (
            "Recompute dataset hashes and update the bundle and directory ID.",
            (
                "Read the bundle's local dataset files, recompute their hashes"
                " and the training-unit and bundle IDs, and replace the bundle"
                " and rename its bundle__ directory to include the new ID."
                " Standalone JSON files keep their paths. Bundles with existing"
                " attempts cannot be"
                " refreshed; create a separate bundle for the changed"
                " experiment. Use after intentionally changing dataset"
                " contents or for bundles without dataset hashes. Clear"
                " packaged file locations; the refreshed bundle must be"
                " packaged again before training or evaluation. Dataset files"
                " must be accessible. The original file remains intact until"
                " the replacement is ready. No job is submitted. Print the"
                " resulting bundle ID and bundle path."
            ),
        ),
    }
    for name, (summary, description) in command_help.items():
        command = commands.add_parser(
            name, help=summary, description=description
        )
        if name == "refresh_bundle_identity":
            command.add_argument(
                "bundle",
                type=Path,
                metavar="BUNDLE_JSON",
                help=(
                    "Path to the existing bundle JSON to refresh."
                    " inspect, logs, cancel, and attach use a bundle ID to"
                    " select saved job attempts; this command edits the bundle"
                    " file itself and may change its ID. A bundle.json in a"
                    " bundle__ directory moves with that directory to its new"
                    " ID-based name; standalone JSON files keep their paths."
                    " Relative paths"
                    " use the current working directory, as does the training"
                    " unit's root_path when relative; dataset paths are not"
                    " relative to this bundle."
                ),
            )
            continue
        command.add_argument(
            "--bundle-write-root",
            type=Path,
            metavar="DIR",
            help=(
                "Local directory containing job history, packaged inputs, and"
                " collected results. Use the same directory for submission and"
                " later commands. Defaults to the DEFAULT_BUNDLE_WRITE_ROOT"
                " environment variable, or ./bundles if unset. Relative paths"
                " use the current working directory."
            ),
        )
        if name == "submit":
            command.add_argument(
                "--resource-file",
                type=Path,
                metavar="PROFILE",
                help=(
                    "JSON or YAML profile defining where jobs run (Local,"
                    " Slurm, PBS, or Bohrium), requested compute resources,"
                    " and software environments. Must include an environments"
                    " entry for each submitted MLFF family. Required for"
                    " bundles with no previous attempt. If omitted, retries"
                    " reuse saved settings. Supplied settings apply only to"
                    " new attempts; recovery always uses the original settings."
                    " Relative paths use the current working directory."
                ),
            )
        require_bundle_id = name in {"inspect", "logs", "cancel", "attach"}
        command.add_argument(
            "--bundle-id",
            required=require_bundle_id,
            metavar="ID",
            help=(
                "Full bundle ID shown by job list; must match exactly, not"
                " a prefix. "
            )
            + (
                "Selects attempts for this bundle."
                if require_bundle_id
                else (
                    "If omitted, include all bundle IDs that match the"
                    " other filters."
                )
            ),
        )
        if name not in {"cancel", "attach"}:
            for option, value_help in (
                (
                    "resource",
                    (
                        "the resource profile's saved name field (not its file"
                        " path)"
                    ),
                ),
                ("domain", "the training unit's domain value"),
            ):
                command.add_argument(
                    f"--{option}",
                    metavar="NAME",
                    help=(
                        (
                            "Select bundles by their latest attempt's saved"
                            " resource name. For bundles with no attempts,"
                            " use the supplied resource profile's name."
                            " A replacement profile does not change selection"
                            " of bundles with existing attempts."
                        )
                        if name == "submit" and option == "resource"
                        else (
                            f"Select {'bundles' if name == 'submit' else 'attempts'}"
                            f" whose {option} exactly matches {value_help}."
                            " If omitted, include all values. Combine filters"
                            " to narrow the selection."
                        )
                    ),
                )
        if name == "synchronize_remote":
            command.add_argument(
                "--keep-remote-files",
                action="store_true",
                help=(
                    "Collect and verify finished results and logs without"
                    " deleting remote working files. With --keep-watching,"
                    " stop when all selected attempts have verified results."
                    " Run again without this option to clean up remote files."
                ),
            )
        if name in {"submit", "synchronize_remote"}:
            command.add_argument(
                "--keep-watching",
                "--keep_watching",
                action="store_true",
                help=(
                    "Continue checking status, downloading finished results,"
                    " and cleaning remote files until all selected attempts"
                    " are cleaned up (or results are verified when"
                    " --keep-remote-files is set on synchronize_remote). "
                )
                + (
                    "Watches only the bundles selected by this submission;"
                    " watching never starts another retry. "
                    if name == "submit"
                    else "Uses the bundle, resource, and domain filters. "
                )
                + "Ctrl+C stops watching and leaves remote jobs running.",
            )
            command.add_argument(
                "--interval",
                type=float,
                default=60,
                metavar="SECONDS",
                help=(
                    "Wait this many seconds between synchronization passes"
                    " with --keep-watching (default: 60). Must be positive"
                    " when watching; has no effect otherwise."
                ),
            )
        if name in {"list", "logs"}:
            command.add_argument(
                "--failed",
                action="store_true",
                help=(
                    "Show only attempts whose saved calculation status is"
                    " failed. Connection, upload, or download errors alone"
                    " do not count as failed calculations; cancelled"
                    " attempts are excluded."
                ),
            )
        if name == "status":
            command.add_argument(
                "--refresh",
                action="store_true",
                help=(
                    "Contact the execution resource and save updated job"
                    " status before displaying it. Without this flag, read"
                    " saved local status only. Does not download results,"
                    " cancel jobs, or delete remote files."
                ),
            )
        if name == "attach":
            command.add_argument(
                "--job-id",
                required=True,
                metavar="REMOTE_ID",
                help=(
                    "ID of the existing remote job. First verify on the"
                    " execution resource that it belongs to the selected"
                    " attempt; the command rejects conflicting evidence."
                ),
            )
            command.add_argument(
                "--attempt",
                required=True,
                metavar="ATTEMPT",
                help="Exact attempt directory name, for example attempt_001.",
            )
        if name == "submit":
            command.add_argument(
                "--preview",
                "-p",
                action="store_true",
                help=(
                    "Check remote status and report proposed actions. Local"
                    " observations may be updated, but no files are uploaded"
                    " or downloaded and no remote jobs or files are changed."
                    " Cannot be combined with --keep-watching."
                ),
            )
            command.add_argument(
                "--retry-cancelled",
                action="store_true",
                help=(
                    "Allow a new attempt after confirmed cancellation and"
                    " completed collection/cleanup. By default, cancelled"
                    " bundles remain cancelled."
                ),
            )
        if name == "abandon":
            command.add_argument(
                "--preview",
                "-p",
                action="store_true",
                help=(
                    "List eligible attempts and the warning without asking"
                    " for confirmation or changing any records."
                ),
            )
            command.add_argument(
                "--yes",
                action="store_true",
                help=(
                    "Confirm the listed attempts without a question. The"
                    " warning is still shown and the records are rechecked"
                    " before changing them. Required for noninteractive use."
                ),
            )
        command.add_argument(
            "--json",
            action="store_true",
            help=(
                "Optional and redundant: this command always prints JSON."
                if name == "inspect"
                else "Print submission decisions as a JSON list."
                if name == "submit"
                else (
                    "Print candidates and updated records as JSON. The"
                    " candidate list, warning, and question go to stderr."
                )
                if name == "abandon"
                else (
                    "Print full job records and submission errors as a JSON"
                    " object with attempts and errors fields instead of a"
                    " status listing."
                )
                + (
                    " Omits the log file path listing."
                    if name == "logs"
                    else ""
                )
            ),
        )


def _refresh_bundle_identity(path: Path) -> tuple[MLFFTrainBundle, Path]:
    """Recompute dataset hashes and IDs, then save the updated bundle.

    Use this after intentionally changing dataset contents, or to add hashes
    to an older bundle. The dataset hashes describe the files' current bytes;
    they determine the training-unit ID, which in turn affects the bundle ID.
    Bundles already used by recorded job attempts cannot be refreshed because
    those attempts must remain associated with their original experiment.

    Parameters
    ----------
    path : pathlib.Path
        Existing bundle JSON file. Relative paths use the current working
        directory. Dataset paths follow the training unit's stored root_path
        and domain; a relative root_path also uses the working directory,
        not the directory containing this JSON file.
        A file named bundle.json inside a directory starting with bundle__
        is treated as part of the root/model/domain/bundle directory layout.
        Its store root is locked, and its containing directory is renamed
        to the name derived from the refreshed bundle if necessary. Other
        files in that directory move with it. Standalone JSON files keep
        their paths and do not acquire a store lock.

    Returns
    -------
    refreshed : MLFFTrainBundle
        Saved bundle with freshly calculated dataset hashes and identities.
        Its files attribute is reset to None because previously recorded
        packaged file locations no longer describe a prepared experiment.
        Package it again before training or evaluation. No input datasets
        are copied and no calculation is launched by this function.
    destination : pathlib.Path
        Absolute path to the saved JSON file after any directory rename.
        Use this path for subsequent operations; the input path may no
        longer exist.

    Raises
    ------
    ValueError
        If the bundle is invalid, a job.json exists beside it or inside an
        attempt_* subdirectory, or the new directory name is already occupied.
    OSError
        If an input cannot be read, the store lock cannot be acquired, or
        renaming, writing, or synchronizing files to storage fails.

    Notes
    -----
    Stored dataset hashes are not checked against the current files before
    refreshing: changed contents are the reason to call this function.
    The JSON is replaced only after its complete replacement is written.
    Directory renaming and JSON replacement are separate operations. If
    writing fails while the original JSON remains, the function attempts to
    restore the old directory name. If replacement has already succeeded,
    it keeps the new name. Abrupt termination can interrupt these steps;
    rerun refresh using the file's current path to reconcile the name and ID.
    """
    path = path.resolve()
    managed = (
        path.name == "bundle.json" and path.parent.name.startswith("bundle__")
    )
    root = path.parents[3] if managed and len(path.parents) > 3 else None
    with store_lock(root) if root is not None else nullcontext():
        if (
            any(path.parent.glob("attempt_*/job.json"))
            or (path.parent / "job.json").exists()
        ):
            raise ValueError(
                "Cannot refresh a bundle used by existing attempts. "
                "Create a separate bundle for the changed experiment."
            )
        original = path.read_bytes()
        bundle = MLFFTrainBundle.model_validate_json(
            original, context={"verify_files": False}
        )
        # This will recompute data set hashes and the training-unit and bundle IDs.
        unit = bundle.training_unit.copy_with_dataset_hashes()
        refreshed = bundle.model_copy(
            update={"training_unit": unit, "files": None}
        )
        destination = path
        if root is not None:
            destination = path.parent.with_name(
                bundle_directory(root, refreshed).name
            ) / path.name
        payload = refreshed.model_dump(mode="json")
        if destination != path:
            if destination.parent.exists() or destination.parent.is_symlink():
                raise ValueError(
                    f"Refreshed bundle directory already exists: {destination.parent}"
                )
            path.parent.rename(destination.parent)
        try:
            write_json_atomically(destination, payload)
        except BaseException:
            # A directory fsync can fail after the JSON replacement succeeds.
            # In that case keep the new directory name with the new contents.
            if destination != path and destination.read_bytes() == original:
                destination.parent.rename(path.parent)
            raise
        return refreshed, destination


def _abandon_cli(args: argparse.Namespace, manager: JobManager) -> int:
    """Review abandonment candidates and apply only explicit confirmation.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed abandon options, including selection filters, preview, yes,
        and json. Preview never prompts or marks attempts abandoned; local
        observations may be saved.
    manager : JobManager
        Manager for the selected store. It checks that the reviewed attempts
        are unchanged before recording abandonment.

    Returns
    -------
    int
        Zero after preview, an empty selection, or successful abandonment.
        One when confirmation is declined or unavailable.

    Raises
    ------
    ValueError
        If an attempt changed after review or is no longer eligible.
    OSError
        If saved history cannot be read or updated, or its lock is unavailable.
    """
    candidates = manager.plan_abandonment(
        bundle_id=args.bundle_id, resource=args.resource, domain=args.domain
    )
    stream = sys.stderr if args.json else sys.stdout
    visible = [
        {
            "bundle_id": candidate.bundle_id,
            "attempt": candidate.attempt,
            "resource": candidate.resource,
            "job_id": candidate.job_id,
            "reason": candidate.reason,
        }
        for candidate in candidates
    ]
    if not candidates:
        print("No unresolved attempts match the selected filters.", file=stream)
        if args.json:
            print(json.dumps({"candidates": [], "attempts": [], "errors": []}))
        return 0
    print("Attempts proposed for abandonment:", file=stream)
    for candidate in candidates:
        print(
            f"  {candidate.bundle_id}  {candidate.attempt}\n"
            f"    resource={candidate.resource}"
            f"  job_id={candidate.job_id or 'unknown'}\n"
            f"    {candidate.reason}",
            file=stream,
        )
    print(
        "Warning: check the scheduler or cloud service first. A missing queue"
        " entry is not proof that a job never started. Abandoning changes local"
        " tracking only: it does not cancel jobs or delete files. Running job"
        " submit afterward may duplicate a job that is still running.",
        file=stream,
    )
    if args.preview:
        if args.json:
            print(
                json.dumps(
                    {"candidates": visible, "attempts": [], "errors": []},
                    indent=2,
                )
            )
        return 0
    confirmed = args.yes
    if not confirmed and sys.stdin.isatty():
        print("Abandon these attempts? [y/N] ", end="", file=stream, flush=True)
        try:
            confirmed = input().strip().lower() in {"y", "yes"}
        except (EOFError, KeyboardInterrupt):
            print(file=stream)
            confirmed = False
    if not confirmed:
        print(
            "No attempts were abandoned. Explicit confirmation is required;"
            " use --yes for noninteractive execution.",
            file=stream,
        )
        if args.json:
            print(
                json.dumps(
                    {"candidates": visible, "attempts": [], "errors": []},
                    indent=2,
                )
            )
        return 1
    records = manager.abandon_attempts(candidates)
    if args.json:
        print(
            json.dumps(
                {
                    "candidates": visible,
                    "attempts": [
                        record.model_dump(mode="json") for record in records
                    ],
                    "errors": [],
                },
                indent=2,
            )
        )
    print(
        f"Abandoned {len(records)} attempts."
        " Run job submit to create replacements.",
        file=stream,
    )
    return 0


def job_cli(args: argparse.Namespace) -> int:
    """Execute one job-management command.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments from add_job_parser. The
        refresh_bundle_identity command replaces the selected bundle
        after recomputing dataset hashes and identities, renames managed
        bundle directories to include the new ID, and
        clears its packaged file locations. The submit command reads all
        model/domain/bundle-level bundles under the selected
        bundle-write root; it does not read bundles inside attempt
        directories. Repeating submit checks existing attempts before
        resuming an interrupted submission or starting one new attempt for
        a confirmed failure. Submit preview checks status without changing
        remote jobs or files. Attach associates an existing remote job with one
        saved attempt and never launches a calculation. Abandon lists
        unresolved attempts and requires confirmation before ending their
        local tracking. It does not cancel jobs or delete remote files.

    Returns
    -------
    int
        Zero on an error-free operation, one if any selected operation
        failed or a selected submission remains blocked by an uncertain
        outcome. Already successful or running calculations are normal
        outcomes, not errors. Watching collects results without starting
        further retries. Abandon returns one if confirmation is declined
        or unavailable; preview and an empty selection return zero.
        Errors retained in abandoned history do not affect this exit code.

    Raises
    ------
    ValueError
        If the command input is invalid, a required dataset is missing,
        a refresh would replace a bundle used by existing attempts, or the
        refreshed directory name is already occupied. Also raised when
        preview is combined with watching, or reviewed abandonment candidates
        changed before confirmation was applied.
    OSError
        If an input or saved job record cannot be read or written, or the
        store lock is unavailable. When refreshing a bundle, its original
        contents are retained until replacement is ready.
    KeyboardInterrupt
        If watching is stopped; remote calculations continue.

    Notes
    -----
    Refresh treats ``bundle.json`` inside a ``bundle__`` directory as a
    managed bundle in the root/model/domain/bundle layout and locks that
    root. It prints the final path after renaming. The directory rename
    and JSON replacement are separate filesystem operations. A failed write
    before replacement restores the old directory name when possible;
    abrupt process termination can leave the rename incomplete. Rerunning
    refresh at the file's current path reconciles its directory name.
    """
    action = args.job_command
    if action == "submit" and args.preview and args.keep_watching:
        raise ValueError("--preview cannot be combined with --keep-watching.")
    if getattr(args, "keep_watching", False) and args.interval <= 0:
        raise ValueError("Watch interval must be positive.")
    if action == "refresh_bundle_identity":
        refreshed, destination = _refresh_bundle_identity(args.bundle)
        print(f"Refreshed {refreshed.mlff_train_bundle_id} at {destination}")
        return 0
    manager = JobManager(args.bundle_write_root)
    if action == "abandon":
        return _abandon_cli(args, manager)
    filters = {
        name: getattr(args, name, None)
        for name in ("bundle_id", "resource", "domain")
    }
    if action == "submit":
        decisions = manager.submit_bundles(
            resource=(
                RemoteResource.from_file(args.resource_file)
                if args.resource_file
                else None
            ),
            bundle_id=args.bundle_id,
            resource_name=args.resource,
            domain=args.domain,
            preview=args.preview,
            retry_cancelled=args.retry_cancelled,
        )
        if args.json:
            print(
                json.dumps(
                    [decision.model_dump(mode="json") for decision in decisions],
                    indent=2,
                )
            )
        else:
            for decision in decisions:
                location = f"  {decision.attempt}" if decision.attempt else ""
                resource = (
                    f"  resource={decision.resource}"
                    if decision.resource
                    else ""
                )
                print(
                    f"{decision.bundle_id}  {decision.action}{resource}"
                    f"{location}\n  {decision.reason}"
                )
            if not decisions:
                print("No bundles match the selected filters.")
        if args.keep_watching:
            manager.watch_submissions(decisions, args.interval)
        return int(
            any(
                decision.action
                in {"blocked", "error", "rejected", "uncertain", "uploading", "prepared"}
                for decision in decisions
            )
        )
    elif action == "synchronize_remote" or (
        action == "status" and args.refresh
    ):
        records = manager.synchronize(
            **filters,
            refresh_only=action == "status",
            download_only=getattr(args, "keep_remote_files", False),
        )
    elif action == "cancel":
        records = manager.cancel(args.bundle_id)
    elif action == "attach":
        records = [manager.attach(args.bundle_id, args.attempt, args.job_id)]
    else:
        records = manager.list_attempts(
            **filters, failed=getattr(args, "failed", False)
        )
    if args.json or action == "inspect":
        print(
            json.dumps(
                {
                    "attempts": [
                        record.model_dump(mode="json") for record in records
                    ],
                    "errors": [],
                },
                indent=2,
            )
        )
    else:
        for record in records:
            root = manager.store.path(record)
            abandonment = (
                f" abandoned_at={record.abandoned_at}"
                if record.abandoned_at is not None else ""
            )
            print(
                f"{record.bundle_id}  {root.name} "
                f" {record.submission_status}/{record.computation_status} "
                f" results={record.collection_status}"
                f" cleanup={record.cleanup_status}{abandonment}\n  {root}"
            )
            if record.error:
                print(f"  {record.error}")
            if action == "logs":
                for path in sorted(root.rglob("*")):
                    if path.is_file() and (
                        path.suffix in {".log", ".out", ".err"}
                        or path.name == "events.jsonl"
                    ):
                        print(f"  {path}")
    if getattr(args, "keep_watching", False):
        manager.watch(
            args.interval,
            **filters,
            keep_remote_files=getattr(args, "keep_remote_files", False),
        )
    return int(
        any(
            record.error is not None and record.abandoned_at is None
            for record in records
        )
    )
