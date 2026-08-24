"""Tests for CLI logging, warning formatting, and bounded progress output."""
from __future__ import annotations

from io import StringIO
import logging
import time
import warnings

from temper.logging import (
    DataQualityWarning,
    TemperFormatter,
    TemperWarning,
    configure_cli_logging,
    normalize_log_level,
    progress_task,
    shutdown_progress,
)


class _TTYBuffer(StringIO):
    def isatty(self) -> bool:
        return True


def test_logging_package_exposes_public_api() -> None:
    import temper.logging as temper_logging
    import temper.logging.warnings as temper_warnings

    expected = {
        "BackendFallbackWarning",
        "DataQualityWarning",
        "PerformanceWarning",
        "ProgressManager",
        "ProgressMode",
        "ProgressTask",
        "TemperFormatter",
        "TemperWarning",
        "configure_cli_logging",
        "progress_task",
        "shutdown_progress",
    }
    assert expected.issubset(temper_logging.__all__)
    assert temper_logging.TemperWarning is temper_warnings.TemperWarning
    assert temper_logging.DataQualityWarning is temper_warnings.DataQualityWarning


def test_levels_and_formatter_separate_user_and_developer_detail() -> None:
    assert normalize_log_level("debug") == logging.DEBUG
    assert normalize_log_level(" WARNING ") == logging.WARNING

    user_record = logging.LogRecord(
        "temper.unit",
        logging.INFO,
        __file__,
        10,
        "Loaded %d groups",
        (3,),
        None,
    )
    debug_record = logging.LogRecord(
        "temper.unit",
        logging.DEBUG,
        __file__,
        20,
        "chunk %d",
        (2,),
        None,
    )
    formatter = TemperFormatter(developer_detail=True)
    assert "Loaded 3 groups" in formatter.format(user_record)
    assert "temper.unit" not in formatter.format(user_record)
    assert "DEBUG" in formatter.format(debug_record)
    assert "temper.unit:20" in formatter.format(debug_record)


def test_plain_progress_is_rate_limited_and_file_safe() -> None:
    stream = StringIO()
    configure_cli_logging(
        "INFO",
        progress_mode="plain",
        stream=stream,
        heartbeat_interval=0.02,
        refresh_interval=0.005,
    )
    logger = logging.getLogger("temper.test.progress")
    try:
        with progress_task(
            logger,
            "Processing many updates",
            total=1000,
            unit="items",
        ) as progress:
            for completed in range(1000):
                progress.update(completed=completed, detail="current batch")
            time.sleep(0.055)
    finally:
        shutdown_progress()

    output = stream.getvalue()
    heartbeat_count = output.count("Still Processing many updates")
    assert 1 <= heartbeat_count <= 4
    assert "\r" not in output
    assert "\x1b" not in output
    assert output.count("current batch") == heartbeat_count


def test_auto_tty_progress_reuses_a_line_and_preserves_logs() -> None:
    stream = _TTYBuffer()
    configure_cli_logging(
        "INFO",
        progress_mode="auto",
        stream=stream,
        heartbeat_interval=1.0,
        refresh_interval=0.005,
    )
    logger = logging.getLogger("temper.test.tty")
    try:
        with progress_task(
            logger,
            "Interactive work",
            total=4,
            unit="steps",
        ) as progress:
            progress.update(completed=2, detail="halfway")
            time.sleep(0.02)
            logger.info("Permanent checkpoint.")
            time.sleep(0.01)
    finally:
        shutdown_progress()

    output = stream.getvalue()
    assert "\r" in output
    assert "Interactive work" in output
    assert output.count("Permanent checkpoint.") == 1
    assert "\x1b" not in output


def test_warning_level_suppresses_info_progress() -> None:
    stream = StringIO()
    configure_cli_logging(
        "WARNING",
        progress_mode="plain",
        stream=stream,
        heartbeat_interval=0.01,
        refresh_interval=0.005,
    )
    logger = logging.getLogger("temper.test.quiet")
    try:
        with progress_task(logger, "Hidden progress", total=1) as progress:
            progress.update(completed=1)
            time.sleep(0.025)
    finally:
        shutdown_progress()
    assert stream.getvalue() == ""


def test_captured_temper_warning_uses_clean_cli_format() -> None:
    stream = StringIO()
    configure_cli_logging("INFO", progress_mode="off", stream=stream)
    try:
        with warnings.catch_warnings(record=False):
            warnings.simplefilter("always")
            warnings.warn("input may be incomplete", DataQualityWarning)
    finally:
        shutdown_progress()

    output = stream.getvalue()
    assert "DataQualityWarning: input may be incomplete" in output
    assert __file__ not in output
    assert issubclass(DataQualityWarning, TemperWarning)
