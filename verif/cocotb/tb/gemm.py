"""cocotb helpers for matrix tests: per-cycle driving and packed matrix rows.

Standard library and cocotb only, so array-level tests do not need the generator package.
"""

from __future__ import annotations

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ReadOnly, RisingEdge


async def start(dut, reset_cycles: int = 2) -> None:
    """Start the clock and reset the design."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    dut.rst.value = 1
    for _ in range(reset_cycles):
        await RisingEdge(dut.clk)
    dut.rst.value = 0
    await RisingEdge(dut.clk)


async def cycle(dut, **inputs) -> None:
    """Drive ``inputs`` for one cycle (they are sampled at the next rising edge)."""
    for name, value in inputs.items():
        getattr(dut, name).value = int(value)
    await ReadOnly()


async def tick(dut) -> None:
    await RisingEdge(dut.clk)


def signed(dut, name: str) -> int:
    return getattr(dut, name).value.to_signed()


def pack(values, bits: int) -> int:
    """Pack ``values`` into one integer, element 0 in the low bits."""
    word = 0
    for i, v in enumerate(values):
        word |= (int(v) & ((1 << bits) - 1)) << (i * bits)
    return word


def unpack(word: int, bits: int, n: int) -> list[int]:
    out = []
    for i in range(n):
        v = (int(word) >> (i * bits)) & ((1 << bits) - 1)
        out.append(v - (1 << bits) if v >> (bits - 1) else v)
    return out


# ---------------------------------------------------------------------------
# Staggered streams for a PE mesh (PEArray, SystolicArray): the L1 stream buffers
# start lane ``i``'s stream one clock after lane ``i - 1``'s, so the mesh needs no
# skew registers.
# ---------------------------------------------------------------------------


def os_schedule(A, B, *, start: int = 0, first: bool = True, last: bool = True) -> dict:
    """``{clock: {port: value}}`` streaming ``A @ B`` (one OS reduction) into a mesh."""
    sched: dict = {}
    rows, k_len, cols = len(A), len(B), len(B[0])
    for k in range(k_len):
        for i in range(rows):
            at = sched.setdefault(start + i + k, {})
            at.update({f"in_a_{i}": A[i][k], f"in_valid_{i}": 1,
                       f"in_first_{i}": first and k == 0, f"in_last_{i}": last and k == k_len - 1})  # fmt: skip
        for j in range(cols):
            sched.setdefault(start + j + k, {})[f"in_b_{j}"] = B[k][j]
    return sched


def stationary_schedule(A, W, *, start: int = 0) -> dict:
    """Shift W into the stationary chain (last row first), then stream A's rows."""
    sched: dict = {}
    rows, cols = len(W), len(W[0])
    for n in range(rows):
        at = sched.setdefault(start + n, {})
        at["in_shift"] = 1
        at.update({f"in_w_{j}": W[rows - 1 - n][j] for j in range(cols)})
    s = start + rows
    for m, row in enumerate(A):
        for r in range(rows):
            at = sched.setdefault(s + r + m, {})
            at.update({f"in_a_{r}": row[r], f"in_valid_{r}": 1})
    return sched


async def play(dut, cols: int, sched: dict, clocks: int, *, dataflow: int):
    """Drive ``sched``; return each column's (north OS results, south partial sums)."""
    ports = {p for at in sched.values() for p in at}
    north = [[] for _ in range(cols)]
    south = [[] for _ in range(cols)]
    for t in range(clocks):
        at = sched.get(t, {})
        await cycle(dut, in_control_dataflow=dataflow, **{p: at.get(p, 0) for p in ports})
        for j in range(cols):
            if int(getattr(dut, f"out_c_valid_{j}").value):
                north[j].append(signed(dut, f"out_c_{j}"))
            if int(getattr(dut, f"out_psum_valid_{j}").value):
                south[j].append(signed(dut, f"out_psum_{j}"))
        assert not int(dut.out_collision.value), f"collision at clock {t}"
        await tick(dut)
    return north, south


def rows_of(columns) -> list[list[int]]:
    """Columns of results (each in row order) as rows."""
    return [list(row) for row in zip(*columns, strict=True)]
