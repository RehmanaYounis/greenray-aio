"""Streaming output for sampling tools (``nvidia-smi dmon``, ``dcgmi dmon``).

On a terminal each sample is printed as it is produced, with a real pause between
samples like the real tools. In pipes and tests the samples are returned at once.
"""

from __future__ import annotations

import time

from . import clock


class Stream:
    def __init__(self, ctx, header: list[str], delay: float = 1.0):
        self.ctx = ctx
        self.delay = delay
        self.lines: list[str] = list(header)
        if ctx.streaming and header:
            ctx.emit("\n".join(header))

    def sample(self, rows: list[str], last: bool) -> None:
        self.lines += rows
        if not self.ctx.streaming:
            return
        self.ctx.emit("\n".join(rows))
        if not last:
            if self.ctx.live_delay:
                time.sleep(self.delay)
            clock.advance(self.ctx.state, self.delay)

    def text(self) -> str:
        """What the command returns: everything in a pipe, nothing extra when already streamed."""
        return "" if self.ctx.streaming else "\n".join(self.lines) + "\n"
