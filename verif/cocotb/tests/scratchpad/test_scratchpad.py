"""
Cocotb tests for the Scratchpad memory.

These tests verify the Scratchpad module behavior using cycle-accurate RTL simulation.
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import NextTimeStep, ReadOnly, RisingEdge


async def reset_dut(dut):
    """Reset the DUT."""
    dut.rst.value = 1
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst.value = 0
    await RisingEdge(dut.clk)


@cocotb.test()
async def test_scratchpad_reset(dut):
    """Test Scratchpad initial state after reset."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    # After reset, read_valid should be low
    assert dut.read_valid.value == 0, "read_valid should be 0 after reset"
    dut._log.info("Scratchpad reset complete")


@cocotb.test()
async def test_scratchpad_write_read(dut):
    """Test basic write and read operations."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    # Write to bank 0, row 0
    test_data = 0xDEADBEEF
    test_addr = 0  # Bank 0, row 0

    dut.write_req.value = 1
    dut.write_addr.value = test_addr
    dut.write_data.value = test_data
    dut.write_mask.value = 0xFFFF  # All bytes enabled (assuming 128-bit width = 16 bytes)
    dut.read_req.value = 0

    await RisingEdge(dut.clk)

    # Disable write
    dut.write_req.value = 0

    await RisingEdge(dut.clk)

    # Read back from same address
    dut.read_req.value = 1
    dut.read_addr.value = test_addr

    await RisingEdge(dut.clk)
    dut.read_req.value = 0

    # Wait for read latency (configured as 2 cycles)
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)

    # Check read valid and data
    assert dut.read_valid.value == 1, "read_valid should be high after read latency"

    read_data = dut.read_data.value.to_unsigned()
    # Data is in lower bits
    assert (read_data & 0xFFFFFFFF) == test_data, (
        f"Read data mismatch: got {hex(read_data)}, expected {hex(test_data)}"
    )

    dut._log.info(
        f"Write/Read test passed: wrote {hex(test_data)}, read {hex(read_data & 0xFFFFFFFF)}"
    )


# The ``scratchpad`` target: a 4 KiB L2 of 16-byte rows, 256 rows in two bank groups
# of 128 (A below, B above), each two 64-row banks (memory/scratchpad.py, #24)
GROUP_ROWS = 128


async def write_rows(dut, rows):
    dut.write_mask.value = 0xFFFF
    for addr, value in rows.items():
        dut.write_req.value = 1
        dut.write_addr.value = addr
        dut.write_data.value = value
        await RisingEdge(dut.clk)
    dut.write_req.value = 0


@cocotb.test()
async def test_scratchpad_two_reads_per_clock_from_the_two_groups(dut):
    """An A-group row on port A and a B-group row on port B, every clock, walking the
    banks: every row comes back, in order, and nothing conflicts."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    rows = {r: 0x1000_0000 + r for r in range(8)}
    rows.update({GROUP_ROWS + r: 0x2000_0000 + r for r in range(8)})
    await write_rows(dut, rows)

    got_a, got_b = [], []
    for i in range(8 + 3):
        issue = i < 8
        dut.read_req.value = int(issue)
        dut.read_addr.value = i if issue else 0
        dut.read_b_req.value = int(issue)
        dut.read_b_addr.value = GROUP_ROWS + i if issue else 0
        await RisingEdge(dut.clk)
        await ReadOnly()
        if dut.read_valid.value:
            got_a.append(dut.read_data.value.to_unsigned() & 0xFFFF_FFFF)
        if dut.read_b_valid.value:
            got_b.append(dut.read_b_data.value.to_unsigned() & 0xFFFF_FFFF)
        await NextTimeStep()
    assert got_a == [0x1000_0000 + r for r in range(8)], got_a
    assert got_b == [0x2000_0000 + r for r in range(8)], got_b
    assert dut.error.value == 0, "no access the macros can't take"


@cocotb.test()
async def test_scratchpad_flags_two_reads_in_one_group(dut):
    """Both reads of a clock in the A group: the macros can't take it, and the
    sticky ``error`` says so."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    assert dut.error.value == 0
    dut.read_req.value = 1
    dut.read_addr.value = 1
    dut.read_b_req.value = 1
    dut.read_b_addr.value = 2
    await RisingEdge(dut.clk)
    dut.read_req.value = 0
    dut.read_b_req.value = 0
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    assert dut.error.value == 1, "error is set"
    await RisingEdge(dut.clk)
    assert dut.error.value == 1, "and stays set"


@cocotb.test()
async def test_scratchpad_read_latency(dut):
    """Test that read data appears after the configured latency."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    # Write test data
    test_data = 0xCAFEBABE
    dut.write_req.value = 1
    dut.write_addr.value = 0
    dut.write_data.value = test_data
    dut.write_mask.value = 0xFFFF
    dut.read_req.value = 0
    await RisingEdge(dut.clk)
    dut.write_req.value = 0
    await RisingEdge(dut.clk)

    # Issue read request
    dut.read_req.value = 1
    dut.read_addr.value = 0
    await RisingEdge(dut.clk)
    dut.read_req.value = 0

    # Check read_valid is low immediately (latency = 2)
    assert dut.read_valid.value == 0, "read_valid should be low before latency"

    await RisingEdge(dut.clk)
    # After 1 cycle, still waiting (latency = 2)

    await RisingEdge(dut.clk)
    # After 2 cycles, should be valid
    assert dut.read_valid.value == 1, "read_valid should be high after read latency"

    dut._log.info("Read latency test passed")
