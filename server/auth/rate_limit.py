"""M106: Sliding-window rate limiter — no extra dependencies."""
from __future__ import annotations

import asyncio
import os
import time
from collections import deque
from typing import Deque

_lock = asyncio.Lock()
# Maps identity string → deque of request timestamps (monotonic)
_windows: dict[str, Deque[float]] = {}


def _get_rpm() -> int:
    return int(os.environ.get("QUANTX_RATE_LIMIT_RPM", "60"))


def reset_all() -> None:
    """Clear all rate-limit windows (test helper)."""
    _windows.clear()


async def check_rate_limit(identity: str) -> tuple[bool, int]:
    """Check whether *identity* is within the allowed request rate.

    Returns
    -------
    (allowed, retry_after_seconds)
        *allowed* is ``True`` when the request may proceed.
        *retry_after_seconds* is how long the caller should wait before
        retrying (only meaningful when *allowed* is ``False``).
    """
    rpm = _get_rpm()
    if rpm <= 0:
        return True, 0  # rate limiting disabled

    now = time.monotonic()
    window_start = now - 60.0

    async with _lock:
        if identity not in _windows:
            _windows[identity] = deque()
        window = _windows[identity]
        # Evict timestamps outside the rolling 60-second window
        while window and window[0] < window_start:
            window.popleft()
        if len(window) >= rpm:
            # Oldest timestamp determines when the window opens a slot
            retry_after = max(1, int(window[0] - window_start) + 1)
            return False, retry_after
        window.append(now)
        return True, 0
