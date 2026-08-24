"""Public logging, progress, and warning API for TEMPER."""

from __future__ import annotations

from temper.logging._configuration import (
    LOG_LEVEL_NAMES,
    ProgressAwareStreamHandler,
    TemperFormatter,
    configure_cli_logging,
    normalize_log_level,
    shutdown_progress,
)
from temper.logging._progress import (
    DEFAULT_HEARTBEAT_INTERVAL,
    PROGRESS_MODES,
    ProgressManager,
    ProgressMode,
    ProgressTask,
    format_elapsed,
    progress_task,
)
from temper.logging.warnings import (
    BackendFallbackWarning,
    DataQualityWarning,
    PerformanceWarning,
    TemperWarning,
)


__all__ = [
    "BackendFallbackWarning",
    "DEFAULT_HEARTBEAT_INTERVAL",
    "DataQualityWarning",
    "LOG_LEVEL_NAMES",
    "PROGRESS_MODES",
    "PerformanceWarning",
    "ProgressAwareStreamHandler",
    "ProgressManager",
    "ProgressMode",
    "ProgressTask",
    "TemperFormatter",
    "TemperWarning",
    "configure_cli_logging",
    "format_elapsed",
    "normalize_log_level",
    "progress_task",
    "shutdown_progress",
]
