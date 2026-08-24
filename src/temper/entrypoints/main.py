"""Main CLI of TEMPER."""
from __future__ import annotations

import argparse
import logging
from typing import Sequence

from temper.entrypoints.split import add_split_parser, split_cli
from temper.logging import (
    LOG_LEVEL_NAMES,
    PROGRESS_MODES,
    configure_cli_logging,
    shutdown_progress,
)


# Keep module execution (``python -m temper.entrypoints.main``) inside the
# package logger hierarchy instead of using the special ``__main__`` name.
logger = logging.getLogger("temper.entrypoints.main")


def main_parser() -> argparse.ArgumentParser:
    """TEMPER bench main CLI options and arguments parser.

    Returns
    ------
    ArgumentParser
        TEMPER bench main CLI options and arguments parser.
    """
    parser = argparse.ArgumentParser(
        description="Temper bench: benchmarking tool for evaluating the performance of"
        " fine-tuning on machine-learned force fields.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    level_options = parser.add_mutually_exclusive_group()
    verbose_action = level_options.add_argument(
        "-v",
        "--verbose",
        action="store_const",
        const="DEBUG",
        dest="log_level",
        help="Show developer diagnostics and tracebacks (equivalent to --log-level DEBUG).",
    )
    quiet_action = level_options.add_argument(
        "-q",
        "--quiet",
        action="store_const",
        const="WARNING",
        dest="log_level",
        help="Show warnings and errors only (equivalent to --log-level WARNING).",
    )
    explicit_level_action = level_options.add_argument(
        "--log-level",
        type=str.upper,
        choices=LOG_LEVEL_NAMES,
        help="Minimum diagnostic level for every subcommand (default: INFO).",
    )
    parser.set_defaults(log_level="INFO")
    # The three actions share one destination. Keep the parser-wide INFO
    # default without showing a misleading "default: INFO" next to the
    # --verbose and --quiet shortcuts in generated help.
    for action in (verbose_action, quiet_action, explicit_level_action):
        action.default = argparse.SUPPRESS
    parser.add_argument(
        "--progress",
        choices=PROGRESS_MODES,
        default="auto",
        help=(
            "Live progress rendering: automatic one-line terminal status, "
            "plain rate-limited heartbeats, or disabled."
        ),
    )

    subparsers = parser.add_subparsers(title="Valid subcommands", dest="command")

    _ = add_split_parser(subparsers)

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """TEMPER bench main CLI."""
    parser = main_parser()
    args = parser.parse_args(argv)
    configure_cli_logging(args.log_level, progress_mode=args.progress)

    try:
        if args.command == "split":
            sysexit = split_cli(args.config_file)
        else:
            parser.print_help()
            sysexit = 1
    except KeyboardInterrupt:
        logger.warning("Command %r was interrupted by the user.", args.command)
        sysexit = 130
    except Exception as exc:  # CLI boundary: Python APIs continue to raise.
        debug = logger.isEnabledFor(logging.DEBUG)
        retry_hint = "" if debug else " Rerun with --verbose for a traceback."
        logger.error(
            "Command %r failed: %s: %s.%s",
            args.command,
            type(exc).__name__,
            exc,
            retry_hint,
            exc_info=debug,
        )
        sysexit = 1
    finally:
        shutdown_progress()
    raise SystemExit(sysexit)


if __name__ == "__main__":
    main()
