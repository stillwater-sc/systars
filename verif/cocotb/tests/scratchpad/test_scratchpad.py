"""
Cocotb tests for the Scratchpad memory.

These tests verify the Scratchpad module behavior using cycle-accurate RTL simulation.
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge


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


@cocotb.test()
async def test_scratchpad_bank_selection(dut):
    """Test that different banks can be accessed independently."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    # Write different data to bank 0 and bank 1
    # Address bit 0 selects bank (with 2 banks)
    data_bank0 = 0x11111111
    data_bank1 = 0x22222222

    # Write to bank 0 (address with bit 0 = 0)
    dut.write_req.value = 1
    dut.write_addr.value = 0  # Bank 0, row 0
    dut.write_data.value = data_bank0
    dut.write_mask.value = 0xFFFF
    dut.read_req.value = 0
    await RisingEdge(dut.clk)

    # Write to bank 1 (address with bit 0 = 1)
    dut.write_addr.value = 1  # Bank 1, row 0
    dut.write_data.value = data_bank1
    await RisingEdge(dut.clk)

    dut.write_req.value = 0
    await RisingEdge(dut.clk)

    # Read from bank 0
    dut.read_req.value = 1
    dut.read_addr.value = 0
    await RisingEdge(dut.clk)
    dut.read_req.value = 0

    # Wait for latency
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)

    read_bank0 = dut.read_data.value.to_unsigned() & 0xFFFFFFFF
    assert read_bank0 == data_bank0, f"Bank 0 data mismatch: got {hex(read_bank0)}"

    # Read from bank 1
    dut.read_req.value = 1
    dut.read_addr.value = 1
    await RisingEdge(dut.clk)
    dut.read_req.value = 0

    # Wait for latency
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)

    read_bank1 = dut.read_data.value.to_unsigned() & 0xFFFFFFFF
    assert read_bank1 == data_bank1, f"Bank 1 data mismatch: got {hex(read_bank1)}"

    dut._log.info(f"Bank selection test passed: bank0={hex(read_bank0)}, bank1={hex(read_bank1)}")


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
