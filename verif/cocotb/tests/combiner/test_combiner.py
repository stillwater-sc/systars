"""
Cocotb tests for the Combiner (memory/combiner.py, #74): the merge of result rows
into L2's result group, on the generated RTL.

The ``combiner`` target: 2 banks (the bank is address bit 0), 4 KiB, read latency 2.
The activation is STORE's (controller/blockmover_out.py), not the Combiner's.
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import NextTimeStep, ReadOnly, RisingEdge

COMBINER_BIT = 1 << 31
LATENCY = 2
LANES, BITS = 16, 32  # the target's rows: 16 results of 32 bits (512 bits)


def pack(values):
    return sum((v & ((1 << BITS) - 1)) << (i * BITS) for i, v in enumerate(values))


def unpack(word):
    vals = [(word >> (i * BITS)) & ((1 << BITS) - 1) for i in range(LANES)]
    return [v - (1 << BITS) if v >> (BITS - 1) else v for v in vals]


def row_addr(bank, row):
    return COMBINER_BIT | bank | (row << 1)


async def reset_dut(dut):
    dut.rst.value = 1
    dut.write_req.value = 0
    dut.read_req.value = 0
    dut.accumulate.value = 0
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst.value = 0
    await RisingEdge(dut.clk)


async def start(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)


async def request(dut, addr, value, accumulate):
    """One clock's merge request."""
    dut.write_req.value = 1
    dut.write_addr.value = addr
    dut.write_data.value = value
    dut.accumulate.value = int(accumulate)
    await RisingEdge(dut.clk)
    dut.write_req.value = 0
    dut.accumulate.value = 0


async def idle(dut, clocks=LATENCY + 2):
    for _ in range(clocks):
        await RisingEdge(dut.clk)


async def read(dut, addr):
    """STORE's read: held until taken (read_ready), then its data."""
    dut.read_req.value = 1
    dut.read_addr.value = addr
    for _ in range(50):
        await ReadOnly()
        taken = bool(dut.read_ready.value)
        await RisingEdge(dut.clk)
        if taken:
            break
    else:
        raise AssertionError("the read was never taken")
    dut.read_req.value = 0
    for _ in range(LATENCY + 2):
        await ReadOnly()
        if dut.read_valid.value:
            value = dut.read_data.value.to_unsigned()  # the whole row, every lane
            await NextTimeStep()
            return value
        await RisingEdge(dut.clk)
    raise AssertionError("no read data")


@cocotb.test()
async def test_combiner_reset(dut):
    await start(dut)
    assert dut.read_valid.value == 0


@cocotb.test()
async def test_combiner_overwrite_and_read(dut):
    await start(dut)
    await request(dut, row_addr(0, 3), 0xDEAD_BEEF, accumulate=False)
    await idle(dut)
    assert await read(dut, row_addr(0, 3)) == 0xDEAD_BEEF


@cocotb.test()
async def test_combiner_accumulate(dut):
    await start(dut)
    await request(dut, row_addr(1, 2), 1000, accumulate=False)
    await idle(dut)
    await request(dut, row_addr(1, 2), 234, accumulate=True)
    await idle(dut)
    assert await read(dut, row_addr(1, 2)) == 1234


@cocotb.test()
async def test_combiner_back_to_back_accumulates(dut):
    """Accumulates onto one row on consecutive clocks: each sees the last one's sum,
    through the pipeline's forwarding."""
    await start(dut)
    addr = row_addr(0, 5)
    await request(dut, addr, 10, accumulate=False)
    for n in range(1, 6):
        await request(dut, addr, n, accumulate=True)
    await idle(dut)
    assert await read(dut, addr) == 10 + 1 + 2 + 3 + 4 + 5


@cocotb.test()
async def test_combiner_store_waits_while_the_merge_reads(dut):
    """STORE's read isn't taken while accumulates use the read port."""
    await start(dut)
    await request(dut, row_addr(0, 0), 7, accumulate=False)
    await idle(dut)
    dut.read_req.value = 1
    dut.read_addr.value = row_addr(0, 0)
    for _ in range(4):
        dut.write_req.value = 1
        dut.write_addr.value = row_addr(1, 1)
        dut.write_data.value = 1
        dut.accumulate.value = 1
        await ReadOnly()
        assert dut.read_ready.value == 0, "STORE taken while the merge reads"
        await RisingEdge(dut.clk)
    dut.write_req.value = 0
    dut.accumulate.value = 0
    dut.read_req.value = 0
    await idle(dut)
    assert await read(dut, row_addr(0, 0)) == 7


@cocotb.test()
async def test_combiner_every_lane(dut):
    """A row with every lane distinct and nonzero, accumulated twice back to back: each
    lane adds on its own, and a lane that wraps leaves its neighbours alone."""
    await start(dut)
    addr = row_addr(1, 7)
    big = (1 << 31) - 1
    base = [big if i == 5 else (i + 1) * 1000 * (-1) ** i for i in range(LANES)]
    step = [1 if i == 5 else i - 8 for i in range(LANES)]
    await request(dut, addr, pack(base), accumulate=False)
    await request(dut, addr, pack(step), accumulate=True)
    await request(dut, addr, pack(step), accumulate=True)
    await idle(dut)
    expect = [
        ((b + 2 * s + (1 << 31)) % (1 << 32)) - (1 << 31) for b, s in zip(base, step, strict=True)
    ]
    assert expect[5] == -(1 << 31) + 1  # lane 5 wrapped
    assert unpack(await read(dut, addr)) == expect


@cocotb.test()
async def test_combiner_bank_selection(dut):
    await start(dut)
    await request(dut, row_addr(0, 4), 0x1111, accumulate=False)
    await request(dut, row_addr(1, 4), 0x2222, accumulate=False)
    await idle(dut)
    assert await read(dut, row_addr(0, 4)) == 0x1111
    assert await read(dut, row_addr(1, 4)) == 0x2222
