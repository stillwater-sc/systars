"""
Cocotb tests for the Accumulator memory.

These tests verify the Accumulator module behavior using cycle-accurate RTL simulation.
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

# Activation function constants (must match config.py)
ACTIVATION_NONE = 0
ACTIVATION_RELU = 1
ACTIVATION_RELU6 = 2


async def reset_dut(dut):
    """Reset the DUT."""
    dut.rst.value = 1
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst.value = 0
    await RisingEdge(dut.clk)


@cocotb.test()
async def test_accumulator_reset(dut):
    """Test Accumulator initial state after reset."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    # After reset, read_valid should be low
    assert dut.read_valid.value == 0, "read_valid should be 0 after reset"
    dut._log.info("Accumulator reset complete")


@cocotb.test()
async def test_accumulator_write_read(dut):
    """Test basic write and read operations (overwrite mode)."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    # Write to bank 0, row 0 (overwrite mode)
    test_data = 0x12345678
    test_addr = 0  # Bank 0, row 0

    dut.write_req.value = 1
    dut.write_addr.value = test_addr
    dut.write_data.value = test_data
    dut.accumulate.value = 0  # Overwrite mode
    dut.read_req.value = 0
    dut.read_activation.value = ACTIVATION_NONE

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

    read_data = dut.read_data.value.to_signed()
    assert read_data == test_data, f"Read data mismatch: got {read_data}, expected {test_data}"

    dut._log.info(f"Write/Read test passed: wrote {hex(test_data)}, read {hex(read_data)}")


@cocotb.test()
async def test_accumulator_accumulate_mode(dut):
    """Test accumulate mode (add to existing value)."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    # First, write initial value (overwrite mode)
    initial_value = 100
    test_addr = 0

    dut.write_req.value = 1
    dut.write_addr.value = test_addr
    dut.write_data.value = initial_value
    dut.accumulate.value = 0
    dut.read_req.value = 0
    dut.read_activation.value = ACTIVATION_NONE

    await RisingEdge(dut.clk)
    dut.write_req.value = 0
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)

    # Now accumulate (add to existing)
    add_value = 50
    dut.write_req.value = 1
    dut.write_addr.value = test_addr
    dut.write_data.value = add_value
    dut.accumulate.value = 1  # Accumulate mode

    await RisingEdge(dut.clk)
    dut.write_req.value = 0

    # Wait for accumulate pipeline
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)

    # Read back
    dut.read_req.value = 1
    dut.read_addr.value = test_addr
    await RisingEdge(dut.clk)
    dut.read_req.value = 0

    # Wait for read latency
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)

    expected = initial_value + add_value
    read_data = dut.read_data.value.to_signed()
    assert read_data == expected, f"Accumulate mismatch: got {read_data}, expected {expected}"

    dut._log.info(f"Accumulate test passed: {initial_value} + {add_value} = {read_data}")


@cocotb.test()
async def test_accumulator_relu_activation(dut):
    """Test ReLU activation function on read."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    # Write a negative value (use signed assignment)
    # For signed signals in cocotb, assign negative values directly
    test_addr = 0
    acc_bits = len(dut.write_data)

    # Create negative value in two's complement
    negative_value = (1 << acc_bits) - 50  # -50 in two's complement

    dut.write_req.value = 1
    dut.write_addr.value = test_addr
    dut.write_data.value = negative_value
    dut.accumulate.value = 0
    dut.read_req.value = 0
    dut.read_activation.value = ACTIVATION_NONE

    await RisingEdge(dut.clk)
    dut.write_req.value = 0
    await RisingEdge(dut.clk)

    # First verify we stored a negative value (read without activation)
    dut.read_req.value = 1
    dut.read_addr.value = test_addr
    dut.read_activation.value = ACTIVATION_NONE

    await RisingEdge(dut.clk)
    dut.read_req.value = 0

    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)

    raw_value = dut.read_data.value.to_signed()
    dut._log.info(f"Stored value (no activation): {raw_value}")

    # Now read with ReLU activation
    dut.read_req.value = 1
    dut.read_addr.value = test_addr
    dut.read_activation.value = ACTIVATION_RELU

    await RisingEdge(dut.clk)
    dut.read_req.value = 0

    # Wait for read latency
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)

    # ReLU should clamp negative to 0
    read_data = dut.read_data.value.to_signed()
    assert read_data == 0, f"ReLU failed: got {read_data}, expected 0 for negative input"

    dut._log.info("ReLU activation test passed: negative value clamped to 0")


@cocotb.test()
async def test_accumulator_relu_positive(dut):
    """Test that ReLU passes positive values unchanged."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    # Write a positive value
    positive_value = 42
    test_addr = 2  # Different address

    dut.write_req.value = 1
    dut.write_addr.value = test_addr
    dut.write_data.value = positive_value
    dut.accumulate.value = 0
    dut.read_req.value = 0
    dut.read_activation.value = ACTIVATION_NONE

    await RisingEdge(dut.clk)
    dut.write_req.value = 0
    await RisingEdge(dut.clk)

    # Read with ReLU activation
    dut.read_req.value = 1
    dut.read_addr.value = test_addr
    dut.read_activation.value = ACTIVATION_RELU

    await RisingEdge(dut.clk)
    dut.read_req.value = 0

    # Wait for read latency
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)

    # ReLU should pass positive values through
    read_data = dut.read_data.value.to_signed()
    assert read_data == positive_value, f"ReLU failed: got {read_data}, expected {positive_value}"

    dut._log.info(f"ReLU positive test passed: {positive_value} passed through unchanged")


@cocotb.test()
async def test_accumulator_bank_selection(dut):
    """Test that different banks can be accessed independently."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    # Write different data to bank 0 and bank 1
    data_bank0 = 111
    data_bank1 = 222

    dut.read_activation.value = ACTIVATION_NONE

    # Write to bank 0
    dut.write_req.value = 1
    dut.write_addr.value = 0  # Bank 0
    dut.write_data.value = data_bank0
    dut.accumulate.value = 0
    dut.read_req.value = 0
    await RisingEdge(dut.clk)

    # Write to bank 1
    dut.write_addr.value = 1  # Bank 1
    dut.write_data.value = data_bank1
    await RisingEdge(dut.clk)

    dut.write_req.value = 0
    await RisingEdge(dut.clk)

    # Read from bank 0
    dut.read_req.value = 1
    dut.read_addr.value = 0
    await RisingEdge(dut.clk)
    dut.read_req.value = 0

    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)

    read_bank0 = dut.read_data.value.to_signed()
    assert read_bank0 == data_bank0, f"Bank 0 mismatch: got {read_bank0}"

    # Read from bank 1
    dut.read_req.value = 1
    dut.read_addr.value = 1
    await RisingEdge(dut.clk)
    dut.read_req.value = 0

    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)

    read_bank1 = dut.read_data.value.to_signed()
    assert read_bank1 == data_bank1, f"Bank 1 mismatch: got {read_bank1}"

    dut._log.info(f"Bank selection test passed: bank0={read_bank0}, bank1={read_bank1}")
