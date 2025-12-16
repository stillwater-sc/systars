"""
Cocotb tests for the SystolicArray module.

These tests verify the SystolicArray module behavior using cycle-accurate RTL simulation.
The SystolicArray adds pipeline registers between PEArrays, so data takes multiple cycles
to propagate through.
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge


@cocotb.test()
async def test_array_reset(dut):
    """Test SystolicArray initial state after reset."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    # Reset
    dut.rst.value = 1
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst.value = 0
    await RisingEdge(dut.clk)

    dut._log.info("SystolicArray reset complete")


@cocotb.test()
async def test_array_simple_operation(dut):
    """Test simple operation through 2x2 SystolicArray."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    # Reset
    dut.rst.value = 1
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst.value = 0
    await RisingEdge(dut.clk)

    # Set up inputs for row 0 and row 1
    dut.in_a_0.value = 3
    dut.in_a_1.value = 5
    dut.in_b_0.value = 4
    dut.in_b_1.value = 2
    dut.in_d_0.value = 0
    dut.in_d_1.value = 0
    dut.in_control_dataflow.value = 1  # WS mode
    dut.in_control_propagate.value = 0
    dut.in_control_shift.value = 0
    dut.in_valid.value = 1
    dut.in_id.value = 1
    dut.in_last.value = 0

    # Wait for data to propagate through pipeline
    for _ in range(6):
        await RisingEdge(dut.clk)

    # Check output valid propagates
    assert dut.out_valid.value == 1, "Output should be valid"

    dut._log.info(f"out_a_0 = {dut.out_a_0.value}")
    dut._log.info(f"out_a_1 = {dut.out_a_1.value}")
    dut._log.info(f"out_b_0 = {dut.out_b_0.value}")
    dut._log.info(f"out_b_1 = {dut.out_b_1.value}")


@cocotb.test()
async def test_array_control_passthrough(dut):
    """Test that control signals pass through the SystolicArray correctly."""
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
    dut.in_a_1.value = 43
    dut.in_b_0.value = 0
    dut.in_b_1.value = 0
    dut.in_d_0.value = 0
    dut.in_d_1.value = 0
    dut.in_id.value = 99
    dut.in_last.value = 1
    dut.in_valid.value = 1
    dut.in_control_shift.value = 12
    dut.in_control_dataflow.value = 1
    dut.in_control_propagate.value = 1

    # Wait for control to propagate through array pipeline
    for _ in range(6):
        await RisingEdge(dut.clk)

    # Check pass-through
    assert dut.out_id.value == 99, f"out_id should be 99, got {dut.out_id.value}"
    assert dut.out_last.value == 1, f"out_last should be 1, got {dut.out_last.value}"
    assert dut.out_control_shift.value == 12, "out_control_shift should be 12"
    assert dut.out_control_dataflow.value == 1, "out_control_dataflow should be 1"
    assert dut.out_control_propagate.value == 1, "out_control_propagate should be 1"

    dut._log.info("SystolicArray control pass-through test passed")


@cocotb.test()
async def test_array_a_flow(dut):
    """Test that A data flows through the array horizontally."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    # Reset
    dut.rst.value = 1
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst.value = 0
    await RisingEdge(dut.clk)

    # Set A inputs
    dut.in_a_0.value = 77
    dut.in_a_1.value = 88
    dut.in_b_0.value = 1
    dut.in_b_1.value = 1
    dut.in_d_0.value = 0
    dut.in_d_1.value = 0
    dut.in_control_dataflow.value = 1
    dut.in_control_propagate.value = 0
    dut.in_control_shift.value = 0
    dut.in_valid.value = 1
    dut.in_id.value = 0
    dut.in_last.value = 0

    # Wait for pipeline propagation
    for _ in range(6):
        await RisingEdge(dut.clk)

    # A should pass through (with pipeline delay)
    assert dut.out_a_0.value == 77, f"out_a_0 should be 77, got {dut.out_a_0.value}"
    assert dut.out_a_1.value == 88, f"out_a_1 should be 88, got {dut.out_a_1.value}"

    dut._log.info("SystolicArray A-flow test passed")


@cocotb.test()
async def test_array_pipeline_latency(dut):
    """Test that array has expected pipeline latency between PEArrays."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    # Reset
    dut.rst.value = 1
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst.value = 0
    await RisingEdge(dut.clk)

    # Initialize to zero
    dut.in_a_0.value = 0
    dut.in_a_1.value = 0
    dut.in_b_0.value = 0
    dut.in_b_1.value = 0
    dut.in_d_0.value = 0
    dut.in_d_1.value = 0
    dut.in_control_dataflow.value = 1
    dut.in_control_propagate.value = 0
    dut.in_control_shift.value = 0
    dut.in_valid.value = 1
    dut.in_id.value = 0
    dut.in_last.value = 0

    await RisingEdge(dut.clk)

    # Inject a pulse on row 0
    dut.in_a_0.value = 100
    await RisingEdge(dut.clk)
    dut.in_a_0.value = 0

    # Track when the pulse arrives at output
    pulse_arrived = False
    cycles = 0
    for _ in range(10):
        await RisingEdge(dut.clk)
        cycles += 1
        if int(dut.out_a_0.value) == 100:
            pulse_arrived = True
            break

    assert pulse_arrived, "Pulse should propagate through array"
    dut._log.info(f"Pulse took {cycles} cycles to propagate through array")
