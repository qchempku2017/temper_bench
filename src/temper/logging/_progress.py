"""Bounded live-progress reporting for TEMPER logging."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import logging
import shutil
import threading
import time
from typing import IO, Iterator, Literal, cast


PROGRESS_MODES: tuple[str, ...] = ("auto", "plain", "off")
DEFAULT_HEARTBEAT_INTERVAL = 60.0
DEFAULT_REFRESH_INTERVAL = 0.25

ProgressMode = Literal["auto", "plain", "off"]


def format_elapsed(seconds: float) -> str:
    """Format a non-negative duration as ``HH:MM:SS``."""
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


@dataclass
class ProgressTask:
    """Mutable state for one progress operation."""

    manager: "ProgressManager" = field(repr=False)
    logger: logging.Logger = field(repr=False)
    description: str
    total: int | None = None
    unit: str = "items"
    completed: int = 0
    detail: str | None = None
    started_at: float = 0.0
    enabled: bool = False

    def update(
        self,
        *,
        completed: int | None = None,
        advance: int | None = None,
        detail: str | None = None,
    ) -> None:
        """Update progress without writing one log record per call."""
        if completed is not None and advance is not None:
            raise ValueError("Specify either completed or advance, not both.")
        self.manager.update_task(
            self,
            completed=completed,
            advance=advance,
            detail=detail,
        )

    def advance(self, amount: int = 1, *, detail: str | None = None) -> None:
        """Advance the completed count."""
        self.update(advance=amount, detail=detail)


@dataclass(frozen=True)
class _ProgressSnapshot:
    logger: logging.Logger
    path: str
    completed: int
    total: int | None
    unit: str
    detail: str | None
    elapsed: float


class ProgressManager:
    """Coordinate one TTY status line or rate-limited plain heartbeats."""

    def __init__(
        self,
        *,
        mode: ProgressMode = "plain",
        stream: IO[str] | None = None,
        heartbeat_interval: float = DEFAULT_HEARTBEAT_INTERVAL,
        refresh_interval: float = DEFAULT_REFRESH_INTERVAL,
    ) -> None:
        self._condition = threading.Condition(threading.RLock())
        self._mode: ProgressMode = mode
        self._stream = stream
        self._heartbeat_interval = heartbeat_interval
        self._refresh_interval = refresh_interval
        self._active: list[ProgressTask] = []
        self._thread: threading.Thread | None = None
        self._stop = False
        self._next_heartbeat: float | None = None
        self._last_rendered_width = 0
        self._spinner_index = 0
        self._tty = self._detect_tty(mode, stream)

    @staticmethod
    def _detect_tty(mode: ProgressMode, stream: IO[str] | None) -> bool:
        if mode != "auto" or stream is None:
            return False
        try:
            return bool(stream.isatty())
        except (AttributeError, OSError):
            return False

    def configure(
        self,
        *,
        mode: ProgressMode,
        stream: IO[str] | None,
        heartbeat_interval: float = DEFAULT_HEARTBEAT_INTERVAL,
        refresh_interval: float = DEFAULT_REFRESH_INTERVAL,
    ) -> None:
        """Reset and configure the renderer before command dispatch."""
        if mode not in PROGRESS_MODES:
            raise ValueError(
                f"Unsupported progress mode {mode!r}; choose from "
                f"{', '.join(PROGRESS_MODES)}."
            )
        if heartbeat_interval <= 0 or refresh_interval <= 0:
            raise ValueError("Progress intervals must be positive.")

        self.close()
        with self._condition:
            self._mode = mode
            self._stream = stream
            self._heartbeat_interval = heartbeat_interval
            self._refresh_interval = refresh_interval
            self._tty = self._detect_tty(mode, stream)
            self._stop = False
            self._next_heartbeat = None
            self._last_rendered_width = 0

    def _ensure_thread_locked(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop = False
        thread = threading.Thread(
            target=self._monitor,
            name="temper-progress",
            daemon=True,
        )
        self._thread = thread
        thread.start()

    def start_task(
        self,
        logger: logging.Logger,
        description: str,
        *,
        total: int | None,
        unit: str,
        detail: str | None,
    ) -> ProgressTask:
        """Register a progress task when INFO output is enabled."""
        if total is not None and total < 0:
            raise ValueError("Progress total must be non-negative.")

        task = ProgressTask(
            manager=self,
            logger=logger,
            description=description,
            total=total,
            unit=unit,
            detail=detail,
            started_at=time.monotonic(),
        )
        task.enabled = self._mode != "off" and logger.isEnabledFor(logging.INFO)
        if not task.enabled:
            return task

        with self._condition:
            if not self._active:
                self._next_heartbeat = (
                    task.started_at + self._heartbeat_interval
                )
            self._active.append(task)
            self._ensure_thread_locked()
            self._condition.notify_all()
        return task

    def update_task(
        self,
        task: ProgressTask,
        *,
        completed: int | None,
        advance: int | None,
        detail: str | None,
    ) -> None:
        """Update a registered task under the renderer lock."""
        if not task.enabled:
            if completed is not None:
                task.completed = completed
            elif advance is not None:
                task.completed += advance
            if detail is not None:
                task.detail = detail
            return

        with self._condition:
            if completed is not None:
                task.completed = completed
            elif advance is not None:
                task.completed += advance
            if task.completed < 0:
                raise ValueError("Progress completed count must be non-negative.")
            if task.total is not None:
                task.completed = min(task.completed, task.total)
            if detail is not None:
                task.detail = detail
            if task.enabled:
                self._condition.notify_all()

    def finish_task(self, task: ProgressTask) -> None:
        """Remove a task and clear a terminal line when the stack empties."""
        if not task.enabled:
            return
        with self._condition:
            if task in self._active:
                self._active.remove(task)
            if not self._active:
                self._next_heartbeat = None
                self._clear_status_locked()
            self._condition.notify_all()

    def _snapshot_locked(self) -> _ProgressSnapshot | None:
        if not self._active:
            return None
        current = self._active[-1]
        root = self._active[0]
        return _ProgressSnapshot(
            logger=current.logger,
            path=" > ".join(task.description for task in self._active),
            completed=current.completed,
            total=current.total,
            unit=current.unit,
            detail=current.detail,
            elapsed=time.monotonic() - root.started_at,
        )

    @staticmethod
    def _count_text(snapshot: _ProgressSnapshot) -> str:
        if snapshot.total is None:
            return f"{snapshot.completed} {snapshot.unit}"
        if snapshot.total == 0:
            return f"0/0 {snapshot.unit} (100%)"
        percentage = 100.0 * snapshot.completed / snapshot.total
        return (
            f"{snapshot.completed}/{snapshot.total} {snapshot.unit} "
            f"({percentage:.0f}%)"
        )

    def _plain_message(self, snapshot: _ProgressSnapshot) -> str:
        parts = [
            f"Still {snapshot.path}",
            self._count_text(snapshot),
        ]
        if snapshot.detail:
            parts.append(snapshot.detail)
        parts.append(f"elapsed {format_elapsed(snapshot.elapsed)}")
        return "; ".join(parts)

    def _terminal_message(self, snapshot: _ProgressSnapshot) -> str:
        if snapshot.total is None or snapshot.total == 0:
            spinner = "|/-\\"[self._spinner_index % 4]
            self._spinner_index += 1
            lead = spinner
        else:
            fraction = min(1.0, snapshot.completed / snapshot.total)
            width = 16
            filled = round(width * fraction)
            lead = f"[{'#' * filled}{'-' * (width - filled)}]"

        parts = [lead, snapshot.path, self._count_text(snapshot)]
        if snapshot.detail:
            parts.append(snapshot.detail)
        parts.append(f"elapsed {format_elapsed(snapshot.elapsed)}")
        return " | ".join(parts)

    def _clear_status_locked(self) -> None:
        if not self._tty or self._stream is None or self._last_rendered_width == 0:
            return
        try:
            self._stream.write(
                "\r" + (" " * self._last_rendered_width) + "\r"
            )
            self._stream.flush()
        except (OSError, ValueError):
            self._tty = False
        self._last_rendered_width = 0

    def _render_snapshot(self, snapshot: _ProgressSnapshot | None) -> None:
        if not self._tty or self._stream is None:
            return
        with self._condition:
            if snapshot is None:
                self._clear_status_locked()
                return
            text = self._terminal_message(snapshot)
            terminal_width = shutil.get_terminal_size(fallback=(100, 24)).columns
            text = text[: max(1, terminal_width - 1)]
            try:
                padding = max(0, self._last_rendered_width - len(text))
                self._stream.write("\r" + text + (" " * padding))
                self._stream.flush()
                self._last_rendered_width = len(text)
            except (OSError, ValueError):
                self._tty = False
                self._last_rendered_width = 0

    @contextmanager
    def suspend_status(self) -> Iterator[None]:
        """Clear and atomically restore status around permanent output."""
        with self._condition:
            self._clear_status_locked()
            try:
                yield
            finally:
                snapshot = self._snapshot_locked()
                self._render_snapshot(snapshot)

    def _monitor(self) -> None:
        while True:
            action: Literal["render", "heartbeat"] | None = None
            snapshot: _ProgressSnapshot | None = None
            with self._condition:
                if self._stop:
                    return
                if not self._active:
                    self._thread = None
                    return

                if self._tty:
                    self._condition.wait(timeout=self._refresh_interval)
                    if self._stop:
                        return
                    snapshot = self._snapshot_locked()
                    action = "render"
                else:
                    now = time.monotonic()
                    next_heartbeat = self._next_heartbeat or (
                        now + self._heartbeat_interval
                    )
                    if now < next_heartbeat:
                        self._condition.wait(timeout=next_heartbeat - now)
                        continue
                    snapshot = self._snapshot_locked()
                    self._next_heartbeat = now + self._heartbeat_interval
                    action = "heartbeat"

            if action == "render":
                self._render_snapshot(snapshot)
            elif action == "heartbeat" and snapshot is not None:
                snapshot.logger.info(self._plain_message(snapshot))

    def close(self) -> None:
        """Stop monitoring and clear any active terminal status."""
        thread: threading.Thread | None
        with self._condition:
            self._stop = True
            self._condition.notify_all()
            thread = self._thread
        if (
            isinstance(thread, threading.Thread)
            and thread is not threading.current_thread()
        ):
            cast(threading.Thread, thread).join(
                timeout=max(1.0, self._refresh_interval * 4)
            )
        with self._condition:
            self._clear_status_locked()
            self._active.clear()
            self._thread = None
            self._next_heartbeat = None


_PROGRESS_MANAGER = ProgressManager()


@contextmanager
def progress_task(
    logger: logging.Logger,
    description: str,
    *,
    total: int | None = None,
    unit: str = "items",
    detail: str | None = None,
) -> Iterator[ProgressTask]:
    """Track a task without producing a permanent record for every update."""
    task = _PROGRESS_MANAGER.start_task(
        logger,
        description,
        total=total,
        unit=unit,
        detail=detail,
    )
    logger.debug("Started %s.", description)
    started_at = time.monotonic()
    try:
        yield task
    finally:
        _PROGRESS_MANAGER.finish_task(task)
        logger.debug(
            "Finished %s in %s.",
            description,
            format_elapsed(time.monotonic() - started_at),
        )


__all__ = [
    "DEFAULT_HEARTBEAT_INTERVAL",
    "DEFAULT_REFRESH_INTERVAL",
    "PROGRESS_MODES",
    "ProgressManager",
    "ProgressMode",
    "ProgressTask",
    "format_elapsed",
    "progress_task",
    "_PROGRESS_MANAGER",
]
