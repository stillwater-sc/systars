"""Tests for SRAMReadScheduler against a scratchpad model with a one-cycle read_valid pulse."""

import pytest
from amaranth.sim import Simulator

from systars.config import SystolicConfig
from systars.memory.skew_buffer import SRAMReadScheduler


def to_signed(v: int, bits: int) -> int:
    v %= 1 << bits
    return v - (1 << bits) if v >> (bits - 1) else v


def row_value(addr: int, n: int, bits: int) -> int:
    """Packed row whose element i is (addr * 16 + i) mod 2**bits."""
    return sum(((addr * 16 + i) % (1 << bits)) << (i * bits) for i in range(n))


@pytest.mark.parametrize("delay", [1, 2, 4])
def test_scheduler_streams_every_k(delay):
    cfg = SystolicConfig(grid_rows=2, grid_cols=2, spad_read_delay=delay)
    dut = SRAMReadScheduler(cfg)
    k_dim, a_base, b_base = 3, 10, 40
    outputs, state = [], {"done": False, "finished": False}

    async def sram(ctx):
        """Scratchpad model matching Scratchpad: a request in cycle N makes
        read_valid pulse for exactly one cycle, in cycle N + delay."""
        pending = []  # (cycles until visible, a_addr, b_addr)
        async for _clk, _rst, a_req, a_addr, b_addr in ctx.tick().sample(
            dut.sp_a_read_req, dut.sp_a_read_addr, dut.sp_b_read_addr
        ):
            # Values set after this edge are visible in cycle N + 1
            pending = [(c - 1, a, b) for c, a, b in pending]
            if a_req:
                pending.append((delay - 1, a_addr, b_addr))
            ready = [p for p in pending if p[0] == 0]
            pending = [p for p in pending if p[0] > 0]
            if ready:
                _, a, b = ready[0]
                ctx.set(dut.sp_a_read_data, row_value(a, 2, cfg.input_bits))
                ctx.set(dut.sp_b_read_data, row_value(b, 2, cfg.weight_bits))
            ctx.set(dut.sp_a_read_valid, bool(ready))
            ctx.set(dut.sp_b_read_valid, bool(ready))

    async def testbench(ctx):
        ctx.set(dut.a_base_addr, a_base)
        ctx.set(dut.b_base_addr, b_base)
        ctx.set(dut.k_dim, k_dim)
        ctx.set(dut.start, 1)
        await ctx.tick()
        ctx.set(dut.start, 0)
        for _ in range(200):
            await ctx.delay(1e-8)  # settle combinational outputs
            if ctx.get(dut.out_valid):
                outputs.append(
                    (
                        [ctx.get(getattr(dut, f"out_a_{i}")) for i in range(2)],
                        [ctx.get(getattr(dut, f"out_b_{j}")) for j in range(2)],
                    )
                )
            if ctx.get(dut.done):
                state["done"] = True
                break
            await ctx.tick()
        state["finished"] = True

    sim = Simulator(dut)
    sim.add_clock(1e-6)
    sim.add_process(sram)
    sim.add_testbench(testbench)
    sim.run_until(400e-6)

    assert state["finished"]
    assert state["done"], "scheduler never finished (missed the read_valid pulse?)"
    assert outputs == [
        (
            [to_signed((a_base + k) * 16 + i, cfg.input_bits) for i in range(2)],
            [to_signed((b_base + k) * 16 + j, cfg.weight_bits) for j in range(2)],
        )
        for k in range(k_dim)
    ]
