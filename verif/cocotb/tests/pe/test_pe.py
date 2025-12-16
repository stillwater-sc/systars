"""
Cocotb tests for the Processing Element (PE).

These tests verify the PE module behavior using cycle-accurate RTL simulation.
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge


@cocotb.test()
async def test_pe_reset(dut):
    """Test PE initial state after reset."""
    # Create clock
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    # Reset
    dut.rst.value = 1
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst.value = 0
    await RisingEdge(dut.clk)

    # Check initial state
    dut._log.info("PE reset complete")


@cocotb.test()
async def test_pe_simple_mac(dut):
    """Test a simple multiply-accumulate operation."""
    # Create clock
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    # Reset
    dut.rst.value = 1
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst.value = 0
    await RisingEdge(dut.clk)

    # Set up inputs: a=3, b=4
    # In WS mode with d=0, result should be 3*4=12
    dut.in_a.value = 3
    dut.in_b.value = 4
    dut.in_d.value = 0
    dut.in_control_dataflow.value = 1  # WS mode
    dut.in_control_propagate.value = 0
    dut.in_control_shift.value = 0
    dut.in_valid.value = 1
    dut.in_id.value = 1
    dut.in_last.value = 0

    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)

    # Check output valid propagates
    assert dut.out_valid.value == 1, "Output should be valid"

    dut._log.info(f"out_a = {dut.out_a.value}")
    dut._log.info(f"out_b = {dut.out_b.value}")
    dut._log.info(f"out_c = {dut.out_c.value}")


@cocotb.test()
async def test_pe_passthrough(dut):
    """Test that control signals pass through correctly."""
    # Create clock
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    # Reset
    dut.rst.value = 1
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst.value = 0
    await RisingEdge(dut.clk)

    # Set specific values
    dut.in_a.value = 42
    dut.in_id.value = 123
    dut.in_last.value = 1
    dut.in_valid.value = 1
    dut.in_control_shift.value = 7
    dut.in_control_dataflow.value = 0
    dut.in_control_propagate.value = 1
    dut.in_b.value = 0
    dut.in_d.value = 0

    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)

    # Check pass-through
    assert dut.out_a.value == 42, f"out_a should be 42, got {dut.out_a.value}"
    assert dut.out_id.value == 123, f"out_id should be 123, got {dut.out_id.value}"
    assert dut.out_last.value == 1, f"out_last should be 1, got {dut.out_last.value}"
    assert dut.out_control_shift.value == 7, "out_control_shift should be 7"

    dut._log.info("Pass-through test passed")


@cocotb.test()
async def test_pe_accumulation(dut):
    """Test accumulation over multiple cycles."""
    # Create clock
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    # Reset
    dut.rst.value = 1
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst.value = 0
    await RisingEdge(dut.clk)

    # Output-stationary mode: accumulate internally
    dut.in_control_dataflow.value = 0  # OS mode
    dut.in_control_propagate.value = 0  # Use c1
    dut.in_control_shift.value = 0
    dut.in_valid.value = 1
    dut.in_id.value = 1
    dut.in_last.value = 0

    # First MAC: 2 * 3 = 6
    dut.in_a.value = 2
    dut.in_b.value = 3
    dut.in_d.value = 0
    await RisingEdge(dut.clk)

    # Second MAC: 4 * 5 = 20, should accumulate with 6 -> 26
    dut.in_a.value = 4
    dut.in_b.value = 5
    await RisingEdge(dut.clk)

    # Wait for output
    await RisingEdge(dut.clk)

    dut._log.info(f"Accumulated result: out_c = {dut.out_c.value}")


@cocotb.test()
async def test_pe_signed_values(dut):
    """Test PE with signed values."""
    # Create clock
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    # Reset
    dut.rst.value = 1
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst.value = 0
    await RisingEdge(dut.clk)

    # Test: -3 * 4 = -12 (in WS mode)
    dut.in_control_dataflow.value = 1  # WS mode
    dut.in_control_propagate.value = 0
    dut.in_control_shift.value = 0
    dut.in_valid.value = 1
    dut.in_id.value = 1
    dut.in_last.value = 0

    # -3 in 8-bit signed = 0xFD
    dut.in_a.value = -3 & 0xFF
    dut.in_b.value = 4
    dut.in_d.value = 0

    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)

    dut._log.info(f"Signed result: out_c = {dut.out_c.value.to_signed()}")
