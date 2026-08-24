"""CLI logging configuration and formatting for TEMPER."""

from __future__ import annotations

from datetime import datetime
import logging
from pathlib import Path
import re
import sys
from typing import IO

from temper.logging._progress import (
    DEFAULT_HEARTBEAT_INTERVAL,
    DEFAULT_REFRESH_INTERVAL,
    ProgressManager,
    ProgressMode,
    _PROGRESS_MANAGER,
)


LOG_LEVEL_NAMES: tuple[str, ...] = ("DEBUG", "INFO", "WARNING", "ERROR")

_WARNING_RECORD = re.compile(
    r"^(?P<filename>.*):(?P<lineno>\d+): "
    r"(?P<category>[A-Za-z_][\w.]*Warning): (?P<body>.*)$",
    flags=re.DOTALL,
)


def normalize_log_level(value: str | int) -> int:
    """Return a supported numeric logging level."""
    if isinstance(value, int):
        if value in {
            logging.DEBUG,
            logging.INFO,
            logging.WARNING,
            logging.ERROR,
        }:
            return value
        raise ValueError(f"Unsupported logging level: {value}.")

    normalized = value.strip().upper()
    if normalized not in LOG_LEVEL_NAMES:
        raise ValueError(
            f"Unsupported logging level {value!r}; choose from "
            f"{', '.join(LOG_LEVEL_NAMES)}."
        )
    return int(getattr(logging, normalized))


class TemperFormatter(logging.Formatter):
    """Render concise user records and source-rich debug records."""

    def __init__(self, *, developer_detail: bool = False) -> None:
        super().__init__()
        self.developer_detail = developer_detail

    def _message(self, record: logging.LogRecord) -> str:
        message = record.getMessage().rstrip()
        if record.name != "py.warnings":
            return message

        match = _WARNING_RECORD.match(message)
        if match is None:
            return message

        body = match.group("body").splitlines()[0].strip()
        cleaned = f"{match.group('category')}: {body}"
        if self.developer_detail:
            filename = Path(match.group("filename")).name
            cleaned += f" ({filename}:{match.group('lineno')})"
        return cleaned

    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.fromtimestamp(record.created).astimezone()
        if record.levelno == logging.DEBUG:
            time_text = timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        else:
            time_text = timestamp.strftime("%Y-%m-%d %H:%M:%S")

        prefix = f"{time_text} | {record.levelname:<7}"
        if record.levelno == logging.DEBUG:
            prefix += f" | {record.name}:{record.lineno}"

        message = self._message(record)
        continuation = " " * (len(prefix) + 3)
        rendered_message = ("\n" + continuation).join(message.splitlines())
        rendered = f"{prefix} | {rendered_message}"

        if record.exc_info:
            exception_text = self.formatException(record.exc_info)
            rendered += "\n" + ("\n" + continuation).join(
                exception_text.splitlines()
            )
        return rendered


class ProgressAwareStreamHandler(logging.StreamHandler):
    """A stream handler that does not corrupt an in-place progress line."""

    def __init__(self, stream: IO[str], manager: ProgressManager) -> None:
        super().__init__(stream)
        self._progress_manager = manager
        self._temper_owned_handler = True

    def emit(self, record: logging.LogRecord) -> None:
        # Hold the renderer lock across the permanent write so the monitor
        # cannot redraw halfway through a log record.
        with self._progress_manager.suspend_status():
            super().emit(record)


def configure_cli_logging(
    level: str | int = "INFO",
    *,
    progress_mode: ProgressMode = "auto",
    stream: IO[str] | None = None,
    heartbeat_interval: float = DEFAULT_HEARTBEAT_INTERVAL,
    refresh_interval: float = DEFAULT_REFRESH_INTERVAL,
) -> None:
    """Configure TEMPER logging once for a command-line invocation."""
    stream = stream or sys.stderr
    numeric_level = normalize_log_level(level)
    _PROGRESS_MANAGER.configure(
        mode=progress_mode,
        stream=stream,
        heartbeat_interval=heartbeat_interval,
        refresh_interval=refresh_interval,
    )

    handler = ProgressAwareStreamHandler(stream, _PROGRESS_MANAGER)
    handler.setLevel(numeric_level)
    handler.setFormatter(
        TemperFormatter(developer_detail=numeric_level == logging.DEBUG)
    )

    for logger_name in ("temper", "py.warnings"):
        configured_logger = logging.getLogger(logger_name)
        for existing in list(configured_logger.handlers):
            if getattr(existing, "_temper_owned_handler", False):
                configured_logger.removeHandler(existing)
                existing.close()
        configured_logger.addHandler(handler)
        configured_logger.setLevel(numeric_level)
        configured_logger.propagate = False

    logging.captureWarnings(True)


def shutdown_progress() -> None:
    """Clear progress and restore logging state owned by the CLI."""
    _PROGRESS_MANAGER.close()
    logging.captureWarnings(False)
    closed_handlers: set[logging.Handler] = set()
    for logger_name in ("temper", "py.warnings"):
        configured_logger = logging.getLogger(logger_name)
        for existing in list(configured_logger.handlers):
            if getattr(existing, "_temper_owned_handler", False):
                configured_logger.removeHandler(existing)
                if existing not in closed_handlers:
                    existing.close()
                    closed_handlers.add(existing)
        configured_logger.setLevel(logging.NOTSET)
        configured_logger.propagate = True


__all__ = [
    "LOG_LEVEL_NAMES",
    "ProgressAwareStreamHandler",
    "TemperFormatter",
    "configure_cli_logging",
    "normalize_log_level",
    "shutdown_progress",
]
