"""Drive a PE mesh (``PEArray`` or ``SystolicArray``) with staggered streams.

The mesh has no skew registers: the L1 stream buffers stagger their streams, so
row ``i``'s stream starts one clock after row ``i - 1``'s. These helpers build
that schedule, play it clock by clock and record what leaves the edges.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from systars.core.pe import OUTPUT_STATIONARY, STATIONARY

from ._sim import run_sim


@dataclass
class Schedule:
    """Edge inputs by clock: ``inputs[t][port] = value`` (ports not set are 0)."""

    inputs: dict = field(default_factory=lambda: defaultdict(dict))
    clocks: int = 0

    def put(self, t: int, port: str, value: int) -> None:
        self.inputs[t][port] = int(value)
        self.clocks = max(self.clocks, t + 1)

    def os_stream(self, s: int, A, B, *, first=True, last=True) -> None:
        """One OS stream: row ``i`` presents ``A[i, k]`` and column ``j`` ``B[k, j]``
        from clock ``s + i`` / ``s + j``. ``first`` / ``last`` mark the reduction's
        first and last products (a reduction may span several streams)."""
        rows, k_len = A.shape
        cols = B.shape[1]
        for k in range(k_len):
            for i in range(rows):
                t = s + i + k
                self.put(t, f"in_a_{i}", A[i, k])
                self.put(t, f"in_valid_{i}", 1)
                self.put(t, f"in_first_{i}", first and k == 0)
                self.put(t, f"in_last_{i}", last and k == k_len - 1)
            for j in range(cols):
                self.put(s + j + k, f"in_b_{j}", B[k, j])

    def stationary_stream(self, s: int, A, *, swap=True) -> None:
        """One stationary stream: row ``k`` presents ``A[m, k]`` from clock ``s + k``.
        ``swap`` marks element 0 ``first``: the preloaded weights become active."""
        m_len, rows = A.shape
        for m in range(m_len):
            for k in range(rows):
                self.put(s + k + m, f"in_a_{k}", A[m, k])
                self.put(s + k + m, f"in_valid_{k}", 1)
                self.put(s + k + m, f"in_first_{k}", swap and m == 0)

    def preload(self, p: int, W) -> int:
        """Stream ``W`` down the weight chain: column ``j`` presents ``W[R-1-e, j]``
        in clock ``p + j + e`` (bottom row first, the L1 stagger). Returns the first
        clock a stream may start and swap the weights in (``p + R``)."""
        rows, cols = W.shape
        for j in range(cols):
            for e in range(rows):
                self.put(p + j + e, f"in_w_{j}", W[rows - 1 - e, j])
                self.put(p + j + e, f"in_w_valid_{j}", 1)
        return p + rows


@dataclass
class Trace:
    """What left the mesh: per column, ``(clock, value)`` of every valid result."""

    north: dict = field(default_factory=lambda: defaultdict(list))
    south: dict = field(default_factory=lambda: defaultdict(list))
    collisions: list = field(default_factory=list)


def play(dut, cols: int, sched: Schedule, *, stationary: bool, extra: int) -> Trace:
    """Run ``sched`` (plus ``extra`` idle clocks) on ``dut`` and record its edges."""
    trace = Trace()
    ports = {name for t in sched.inputs.values() for name in t}

    async def tb(ctx):
        ctx.set(dut.in_control_dataflow, STATIONARY if stationary else OUTPUT_STATIONARY)
        for t in range(sched.clocks + extra):
            now = sched.inputs.get(t, {})
            for name in ports:
                ctx.set(getattr(dut, name), now.get(name, 0))
            for j in range(cols):
                if ctx.get(getattr(dut, f"out_c_valid_{j}")):
                    trace.north[j].append((t, ctx.get(getattr(dut, f"out_c_{j}"))))
                if ctx.get(getattr(dut, f"out_psum_valid_{j}")):
                    trace.south[j].append((t, ctx.get(getattr(dut, f"out_psum_{j}"))))
            if ctx.get(dut.out_collision):
                trace.collisions.append(t)
            await ctx.tick()

    run_sim(dut, tb, max_cycles=sched.clocks + extra + 10)
    return trace


def column_values(trace_edge: dict, cols: int) -> np.ndarray:
    """Values that left each column, in order: ``[n, j]`` is column ``j``'s ``n``-th."""
    return np.array([[v for _, v in trace_edge[j]] for j in range(cols)]).T


def column_clocks(trace_edge: dict, cols: int) -> np.ndarray:
    """Clocks of those values, laid out as ``column_values``."""
    return np.array([[t for t, _ in trace_edge[j]] for j in range(cols)]).T
