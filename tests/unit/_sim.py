"""Shared Amaranth simulation helper for unit tests."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from amaranth.sim import Simulator

CLOCK_PERIOD = 1e-6


def run_sim(
    dut: Any,
    testbench: Callable[[Any], Awaitable[None]],
    *,
    max_cycles: int = 20_000,
    processes: Sequence[Callable[[Any], Awaitable[None]]] = (),
) -> None:
    """Run ``testbench`` against ``dut`` with a cycle deadline.

    A handshake that never completes makes ``ctx.tick().until(...)`` wait
    forever; the deadline turns that hang into a test failure. ``processes``
    are background monitors (``Simulator.add_process``) that may run forever:
    the simulation ends as soon as the testbench returns.
    """
    done = False

    async def wrapped(ctx):
        nonlocal done
        await testbench(ctx)
        done = True

    async def watchdog(ctx):
        for _ in range(max_cycles):
            await ctx.tick()
        if not done:
            raise AssertionError(
                f"testbench did not finish within {max_cycles} cycles (stalled handshake?)"
            )

    sim = Simulator(dut)
    sim.add_clock(CLOCK_PERIOD)
    sim.add_testbench(wrapped)
    sim.add_process(watchdog)
    for process in processes:
        sim.add_process(process)
    sim.run()
    assert done, "testbench did not finish"


def skip_without_yosys() -> None:
    """Skip the calling test when Yosys (needed for Verilog conversion) is missing."""
    import pytest
    from amaranth._toolchain.yosys import YosysError, find_yosys

    try:
        find_yosys(lambda ver: ver >= (0, 40))
    except YosysError:
        pytest.skip("Yosys not found")
