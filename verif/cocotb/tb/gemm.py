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
