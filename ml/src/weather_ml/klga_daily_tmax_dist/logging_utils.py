from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol


def format_duration(seconds: float) -> str:
    seconds = max(float(seconds), 0.0)
    s = int(round(seconds))
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{sec:02d}"
    return f"{m:02d}:{sec:02d}"


class _LoggerProto(Protocol):
    def info(self, msg: str, *args) -> None: ...


@dataclass
class ProgressTracker:
    logger: _LoggerProto
    name: str
    total: int
    log_every_rows: int = 1000
    log_every_seconds: float = 20.0

    def __post_init__(self) -> None:
        self._start = time.perf_counter()
        self._last_log = self._start
        self._last_n = 0

    def maybe_log(self, n_done: int, *, extra: str = "") -> None:
        now = time.perf_counter()
        should_log_rows = self.log_every_rows > 0 and (n_done - self._last_n) >= self.log_every_rows
        should_log_time = (now - self._last_log) >= self.log_every_seconds
        if not should_log_rows and not should_log_time and n_done < self.total:
            return

        elapsed = now - self._start
        pct = (100.0 * n_done / self.total) if self.total > 0 else 100.0
        rate = (n_done / elapsed) if elapsed > 0 else 0.0
        remaining = max(self.total - n_done, 0)
        eta_s = (remaining / rate) if rate > 0 else float("inf")
        eta_txt = format_duration(eta_s) if eta_s < float("inf") else "?:??"
        msg = (
            f"{self.name} progress={pct:6.2f}% rows={n_done}/{self.total} "
            f"elapsed={format_duration(elapsed)} eta={eta_txt} rps={rate:,.1f}"
        )
        if extra:
            msg = f"{msg} {extra}"
        self.logger.info(msg)
        self._last_log = now
        self._last_n = n_done

    def done(self, *, extra: str = "") -> None:
        now = time.perf_counter()
        elapsed = now - self._start
        rate = (self.total / elapsed) if elapsed > 0 else 0.0
        msg = (
            f"{self.name} progress=100.00% rows={self.total}/{self.total} "
            f"elapsed={format_duration(elapsed)} rps={rate:,.1f}"
        )
        if extra:
            msg = f"{msg} {extra}"
        self.logger.info(msg)

