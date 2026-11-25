"""Client-side pacing against a tokens-per-minute limit.

Groq's free tier allows ~8k tokens/min per model, i.e. about two calls a
minute. Waiting before a call is cheaper and more predictable than sending it
and backing off from a 429.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable


class _Ticket:
    __slots__ = ("ts", "tokens")

    def __init__(self, ts: float, tokens: int):
        self.ts = ts
        self.tokens = tokens


class TokenPacer:
    """Sliding 60-second window of token usage for one model."""

    def __init__(
        self,
        tokens_per_minute: int,
        *,
        window_s: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.tokens_per_minute = tokens_per_minute
        self.window_s = window_s
        self._clock = clock
        self._sleep = sleep
        self._entries: deque[_Ticket] = deque()
        self._blocked_until = 0.0

    def _used(self, now: float) -> int:
        while self._entries and now - self._entries[0].ts >= self.window_s:
            self._entries.popleft()
        return sum(e.tokens for e in self._entries)

    def acquire(self, estimate: int) -> tuple[_Ticket, float]:
        """Block until `estimate` tokens fit in the window. Returns (ticket, seconds waited).

        A single request larger than the whole budget is let through once the
        window is empty; the provider will reject it if it truly can't fit.
        """
        waited = 0.0
        while True:
            now = self._clock()
            if now < self._blocked_until:
                delay = self._blocked_until - now
                self._sleep(delay)
                waited += delay
                continue
            used = self._used(now)
            if not self._entries or used + estimate <= self.tokens_per_minute:
                ticket = _Ticket(now, estimate)
                self._entries.append(ticket)
                return ticket, waited
            delay = max(self._entries[0].ts + self.window_s - now, 0.05)
            self._sleep(delay)
            waited += delay

    @staticmethod
    def settle(ticket: _Ticket, actual_tokens: int) -> None:
        """Replace the estimate with the provider-reported usage."""
        ticket.tokens = actual_tokens

    def penalize(self, seconds: float) -> None:
        """After a 429, block all calls for `seconds` (the provider's stated wait)."""
        self._blocked_until = max(self._blocked_until, self._clock() + seconds)
