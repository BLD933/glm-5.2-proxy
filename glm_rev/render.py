"""Shared terminal rendering helpers (used by both ui.py and tools.py)."""
import sys

_DIM = "\x1b[2m"
_RESET = "\x1b[0m"


class ReasoningStream:
    """Streams GLM's thinking deltas to stderr in realtime (dim ANSI, line-buffered).
    `started` is True once the first thinking chunk arrived; call finish() when the
    stream ends to flush the remainder and close the block."""

    def __init__(self):
        self._buf = ""
        self.started = False
        self.finished = False

    def write(self, chunk):
        if not chunk:
            return
        if not self.started:
            self.started = True
            print(f"{_DIM}── reasoning ──{_RESET}", file=sys.stderr)
        self._buf += chunk
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            print(f"{_DIM}{line}{_RESET}", file=sys.stderr)

    def finish(self):
        if not self.started or self.finished:
            return
        self.finished = True
        if self._buf:
            print(f"{_DIM}{self._buf}{_RESET}", file=sys.stderr)
            self._buf = ""
        print(f"{_DIM}{'─' * 14}{_RESET}", file=sys.stderr)