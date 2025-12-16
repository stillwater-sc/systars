"""
Cocotb tests for the PEArray module.

These tests verify the PEArray module behavior using cycle-accurate RTL simulation.
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge


@cocotb.test()
async def test_pe_array_reset(dut):
    """Test PEArray initial state after reset."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    # Reset
    dut.rst.value = 1
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst.value = 0
    await RisingEdge(dut.clk)

    dut._log.info("PEArray reset complete")


@cocotb.test()
async def test_pe_array_simple_operation(dut):
    """Test simple operation through 1x1 PEArray (single PE)."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    # Reset
    dut.rst.value = 1
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst.value = 0
    await RisingEdge(dut.clk)

    # Set up inputs: a=3, b=4, d=0 in WS mode
    dut.in_a_0.value = 3
    dut.in_b_0.value = 4
    dut.in_d_0.value = 0
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

    dut._log.info(f"out_a_0 = {dut.out_a_0.value}")
    dut._log.info(f"out_b_0 = {dut.out_b_0.value}")
    dut._log.info(f"out_c_0 = {dut.out_c_0.value}")


@cocotb.test()
async def test_pe_array_passthrough(dut):
    """Test that control signals pass through the PEArray correctly."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    # Reset
    dut.rst.value = 1
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst.value = 0
    await RisingEdge(dut.clk)

    # Set specific control values
    dut.in_a_0.value = 42
    dut.in_b_0.value = 0
    dut.in_d_0.value = 0
    dut.in_id.value = 99
    dut.in_last.value = 1
    dut.in_valid.value = 1
    dut.in_control_shift.value = 12
    dut.in_control_dataflow.value = 1
    dut.in_control_propagate.value = 1

    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)

    # Check pass-through
    assert dut.out_a_0.value == 42, f"out_a_0 should be 42, got {dut.out_a_0.value}"
    assert dut.out_id.value == 99, f"out_id should be 99, got {dut.out_id.value}"
    assert dut.out_last.value == 1, f"out_last should be 1, got {dut.out_last.value}"
    assert dut.out_control_shift.value == 12, "out_control_shift should be 12"
    assert dut.out_control_dataflow.value == 1, "out_control_dataflow should be 1"
    assert dut.out_control_propagate.value == 1, "out_control_propagate should be 1"

    dut._log.info("PEArray pass-through test passed")


@cocotb.test()
async def test_pe_array_a_flow(dut):
    """Test that A data flows through the PEArray horizontally."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    # Reset
    dut.rst.value = 1
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst.value = 0
    await RisingEdge(dut.clk)

    # Set A input
    dut.in_a_0.value = 77
    dut.in_b_0.value = 1
    dut.in_d_0.value = 0
    dut.in_control_dataflow.value = 1
    dut.in_control_propagate.value = 0
    dut.in_control_shift.value = 0
    dut.in_valid.value = 1
    dut.in_id.value = 0
    dut.in_last.value = 0

    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)

    # A should pass through (registered)
    assert dut.out_a_0.value == 77, f"out_a_0 should be 77, got {dut.out_a_0.value}"

    dut._log.info("PEArray A-flow test passed")
