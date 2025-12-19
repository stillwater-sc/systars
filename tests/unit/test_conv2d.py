"""
Unit tests for the Conv2D ISA instruction.

Tests cover:
- Configuration interface
- Dimension calculation (M, N, K mapping)
- Address calculation for different tile positions
- Edge tile size handling
- Dataflow selection heuristics
- Command emission
"""

import pytest
from amaranth.sim import Simulator, Tick

from systars.config import SystolicConfig
from systars.isa.conv2d import Conv2d, Conv2dCmd


@pytest.fixture
def config():
    """Create a small test configuration."""
    return SystolicConfig(
        grid_rows=4,
        grid_cols=4,
        tile_rows=1,
        tile_cols=1,
        input_bits=8,
        acc_bits=32,
    )


@pytest.fixture
def conv2d(config):
    """Create a Conv2d instance."""
    return Conv2d(config)


class TestConv2dConfiguration:
    """Test configuration interface."""

    def test_config_input_dims(self, conv2d):
        """Test input dimension configuration."""

        def testbench():
            # Configure input dimensions: batch=1, height=8, width=8, channels=3
            yield conv2d.cfg_valid.eq(1)
            yield conv2d.cfg_cmd.eq(Conv2dCmd.CONFIG_INPUT_DIMS)
            # Pack: in_c[55:40] | in_w[39:24] | in_h[23:8] | batch[7:0]
            yield conv2d.cfg_data.eq((3 << 40) | (8 << 24) | (8 << 8) | 1)
            yield Tick()
            yield conv2d.cfg_valid.eq(0)
            yield Tick()
            yield Tick()

            # Verify dimensions were latched
            assert (yield conv2d.param_batch) == 1
            assert (yield conv2d.param_in_h) == 8
            assert (yield conv2d.param_in_w) == 8
            assert (yield conv2d.param_in_c) == 3

        sim = Simulator(conv2d)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()

    def test_config_output_dims(self, conv2d):
        """Test output dimension configuration."""

        def testbench():
            # Configure output dimensions: out_h=6, out_w=6, out_c=16
            yield conv2d.cfg_valid.eq(1)
            yield conv2d.cfg_cmd.eq(Conv2dCmd.CONFIG_OUTPUT_DIMS)
            # Pack: out_c[47:32] | out_w[31:16] | out_h[15:0]
            yield conv2d.cfg_data.eq((16 << 32) | (6 << 16) | 6)
            yield Tick()
            yield conv2d.cfg_valid.eq(0)
            yield Tick()
            yield Tick()

            # Verify dimensions were latched
            assert (yield conv2d.param_out_h) == 6
            assert (yield conv2d.param_out_w) == 6
            assert (yield conv2d.param_out_c) == 16

        sim = Simulator(conv2d)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()

    def test_config_kernel_dims(self, conv2d):
        """Test kernel dimension configuration."""

        def testbench():
            # Configure kernel dimensions: kernel_h=3, kernel_w=3
            yield conv2d.cfg_valid.eq(1)
            yield conv2d.cfg_cmd.eq(Conv2dCmd.CONFIG_KERNEL_DIMS)
            # Pack: kernel_w[15:8] | kernel_h[7:0]
            yield conv2d.cfg_data.eq((3 << 8) | 3)
            yield Tick()
            yield conv2d.cfg_valid.eq(0)
            yield Tick()
            yield Tick()

            # Verify dimensions were latched
            assert (yield conv2d.param_kernel_h) == 3
            assert (yield conv2d.param_kernel_w) == 3

        sim = Simulator(conv2d)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()

    def test_config_addresses(self, conv2d):
        """Test address configuration."""

        def testbench():
            # Configure X address
            yield conv2d.cfg_valid.eq(1)
            yield conv2d.cfg_cmd.eq(Conv2dCmd.CONFIG_X_ADDR)
            yield conv2d.cfg_data.eq(0x1000_0000)
            yield Tick()
            yield conv2d.cfg_valid.eq(0)
            yield Tick()

            # Configure F address
            yield conv2d.cfg_valid.eq(1)
            yield conv2d.cfg_cmd.eq(Conv2dCmd.CONFIG_F_ADDR)
            yield conv2d.cfg_data.eq(0x2000_0000)
            yield Tick()
            yield conv2d.cfg_valid.eq(0)
            yield Tick()

            # Configure Y address
            yield conv2d.cfg_valid.eq(1)
            yield conv2d.cfg_cmd.eq(Conv2dCmd.CONFIG_Y_ADDR)
            yield conv2d.cfg_data.eq(0x3000_0000)
            yield Tick()
            yield conv2d.cfg_valid.eq(0)
            yield Tick()

        sim = Simulator(conv2d)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()


class TestDimensionMapping:
    """Test M, N, K dimension mapping from conv2d parameters."""

    def test_dimension_calculation(self, conv2d):
        """Test that M, N, K are calculated correctly after START."""

        def testbench():
            yield conv2d.cmd_ready.eq(1)

            # Configure for a simple 8x8 input, 3x3 kernel, 16 output channels
            # Input: batch=1, height=8, width=8, channels=3
            yield conv2d.cfg_valid.eq(1)
            yield conv2d.cfg_cmd.eq(Conv2dCmd.CONFIG_INPUT_DIMS)
            yield conv2d.cfg_data.eq((3 << 40) | (8 << 24) | (8 << 8) | 1)
            yield Tick()
            yield conv2d.cfg_valid.eq(0)
            yield Tick()

            # Output: out_h=6, out_w=6, out_c=16 (valid padding)
            yield conv2d.cfg_valid.eq(1)
            yield conv2d.cfg_cmd.eq(Conv2dCmd.CONFIG_OUTPUT_DIMS)
            yield conv2d.cfg_data.eq((16 << 32) | (6 << 16) | 6)
            yield Tick()
            yield conv2d.cfg_valid.eq(0)
            yield Tick()

            # Kernel: 3x3
            yield conv2d.cfg_valid.eq(1)
            yield conv2d.cfg_cmd.eq(Conv2dCmd.CONFIG_KERNEL_DIMS)
            yield conv2d.cfg_data.eq((3 << 8) | 3)
            yield Tick()
            yield conv2d.cfg_valid.eq(0)
            yield Tick()

            # Addresses
            yield conv2d.cfg_valid.eq(1)
            yield conv2d.cfg_cmd.eq(Conv2dCmd.CONFIG_X_ADDR)
            yield conv2d.cfg_data.eq(0x1000)
            yield Tick()
            yield conv2d.cfg_valid.eq(0)
            yield Tick()

            yield conv2d.cfg_valid.eq(1)
            yield conv2d.cfg_cmd.eq(Conv2dCmd.CONFIG_F_ADDR)
            yield conv2d.cfg_data.eq(0x2000)
            yield Tick()
            yield conv2d.cfg_valid.eq(0)
            yield Tick()

            yield conv2d.cfg_valid.eq(1)
            yield conv2d.cfg_cmd.eq(Conv2dCmd.CONFIG_Y_ADDR)
            yield conv2d.cfg_data.eq(0x3000)
            yield Tick()
            yield conv2d.cfg_valid.eq(0)
            yield Tick()

            # Start execution
            yield conv2d.cfg_valid.eq(1)
            yield conv2d.cfg_cmd.eq(Conv2dCmd.START)
            yield Tick()
            yield conv2d.cfg_valid.eq(0)

            # Wait for INIT and CALC_TILES to complete
            for _ in range(5):
                yield Tick()

            # Verify M, N, K calculation:
            # M = batch * out_h * out_w = 1 * 6 * 6 = 36
            # N = out_c = 16
            # K = kernel_h * kernel_w * in_c = 3 * 3 * 3 = 27
            assert (yield conv2d.param_M) == 36
            assert (yield conv2d.param_N) == 16
            assert (yield conv2d.param_K) == 27

        sim = Simulator(conv2d)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()


class TestDataflowSelection:
    """Test automatic dataflow selection."""

    def test_output_stationary_default(self, conv2d):
        """Test that output-stationary is selected for balanced dimensions."""

        def testbench():
            yield conv2d.cmd_ready.eq(1)

            # Configure for balanced dimensions
            # Input: batch=1, height=16, width=16, channels=8
            yield conv2d.cfg_valid.eq(1)
            yield conv2d.cfg_cmd.eq(Conv2dCmd.CONFIG_INPUT_DIMS)
            yield conv2d.cfg_data.eq((8 << 40) | (16 << 24) | (16 << 8) | 1)
            yield Tick()
            yield conv2d.cfg_valid.eq(0)
            yield Tick()

            # Output: out_h=16, out_w=16, out_c=8 (same padding)
            yield conv2d.cfg_valid.eq(1)
            yield conv2d.cfg_cmd.eq(Conv2dCmd.CONFIG_OUTPUT_DIMS)
            yield conv2d.cfg_data.eq((8 << 32) | (16 << 16) | 16)
            yield Tick()
            yield conv2d.cfg_valid.eq(0)
            yield Tick()

            # Kernel: 1x1 (so K is small)
            yield conv2d.cfg_valid.eq(1)
            yield conv2d.cfg_cmd.eq(Conv2dCmd.CONFIG_KERNEL_DIMS)
            yield conv2d.cfg_data.eq((1 << 8) | 1)
            yield Tick()
            yield conv2d.cfg_valid.eq(0)
            yield Tick()

            # Addresses
            yield conv2d.cfg_valid.eq(1)
            yield conv2d.cfg_cmd.eq(Conv2dCmd.CONFIG_X_ADDR)
            yield conv2d.cfg_data.eq(0x1000)
            yield Tick()
            yield conv2d.cfg_valid.eq(0)
            yield Tick()

            yield conv2d.cfg_valid.eq(1)
            yield conv2d.cfg_cmd.eq(Conv2dCmd.CONFIG_F_ADDR)
            yield conv2d.cfg_data.eq(0x2000)
            yield Tick()
            yield conv2d.cfg_valid.eq(0)
            yield Tick()

            yield conv2d.cfg_valid.eq(1)
            yield conv2d.cfg_cmd.eq(Conv2dCmd.CONFIG_Y_ADDR)
            yield conv2d.cfg_data.eq(0x3000)
            yield Tick()
            yield conv2d.cfg_valid.eq(0)
            yield Tick()

            # Start execution
            yield conv2d.cfg_valid.eq(1)
            yield conv2d.cfg_cmd.eq(Conv2dCmd.START)
            yield Tick()
            yield conv2d.cfg_valid.eq(0)

            # Wait for initialization
            for _ in range(5):
                yield Tick()

            # Should select output-stationary (0)
            # M = 1 * 16 * 16 = 256, N = 8, K = 1 * 1 * 8 = 8
            # K (8) is NOT > M (256) << 1, so should be OS
            assert (yield conv2d.selected_dataflow) == 0

        sim = Simulator(conv2d)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()

    def test_weight_stationary_large_kernel(self, conv2d):
        """Test that weight-stationary is selected when K is large."""

        def testbench():
            yield conv2d.cmd_ready.eq(1)

            # Configure for small spatial, large kernel
            # Input: batch=1, height=4, width=4, channels=64
            yield conv2d.cfg_valid.eq(1)
            yield conv2d.cfg_cmd.eq(Conv2dCmd.CONFIG_INPUT_DIMS)
            yield conv2d.cfg_data.eq((64 << 40) | (4 << 24) | (4 << 8) | 1)
            yield Tick()
            yield conv2d.cfg_valid.eq(0)
            yield Tick()

            # Output: out_h=2, out_w=2, out_c=4
            yield conv2d.cfg_valid.eq(1)
            yield conv2d.cfg_cmd.eq(Conv2dCmd.CONFIG_OUTPUT_DIMS)
            yield conv2d.cfg_data.eq((4 << 32) | (2 << 16) | 2)
            yield Tick()
            yield conv2d.cfg_valid.eq(0)
            yield Tick()

            # Kernel: 3x3
            yield conv2d.cfg_valid.eq(1)
            yield conv2d.cfg_cmd.eq(Conv2dCmd.CONFIG_KERNEL_DIMS)
            yield conv2d.cfg_data.eq((3 << 8) | 3)
            yield Tick()
            yield conv2d.cfg_valid.eq(0)
            yield Tick()

            # Addresses
            yield conv2d.cfg_valid.eq(1)
            yield conv2d.cfg_cmd.eq(Conv2dCmd.CONFIG_X_ADDR)
            yield conv2d.cfg_data.eq(0x1000)
            yield Tick()
            yield conv2d.cfg_valid.eq(0)
            yield Tick()

            yield conv2d.cfg_valid.eq(1)
            yield conv2d.cfg_cmd.eq(Conv2dCmd.CONFIG_F_ADDR)
            yield conv2d.cfg_data.eq(0x2000)
            yield Tick()
            yield conv2d.cfg_valid.eq(0)
            yield Tick()

            yield conv2d.cfg_valid.eq(1)
            yield conv2d.cfg_cmd.eq(Conv2dCmd.CONFIG_Y_ADDR)
            yield conv2d.cfg_data.eq(0x3000)
            yield Tick()
            yield conv2d.cfg_valid.eq(0)
            yield Tick()

            # Start execution
            yield conv2d.cfg_valid.eq(1)
            yield conv2d.cfg_cmd.eq(Conv2dCmd.START)
            yield Tick()
            yield conv2d.cfg_valid.eq(0)

            # Wait for initialization
            for _ in range(5):
                yield Tick()

            # M = 1 * 2 * 2 = 4, N = 4, K = 3 * 3 * 64 = 576
            # K (576) > M (4) << 1 = 8, so should be WS (1)
            assert (yield conv2d.selected_dataflow) == 1

        sim = Simulator(conv2d)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()


class TestCommandEmission:
    """Test command emission from state machine."""

    def test_config_command_emitted(self, conv2d):
        """Test that CONFIG command is emitted after START."""
        from systars.isa.conv2d import InternalOpcode

        commands_seen = []

        def testbench():
            yield conv2d.cmd_ready.eq(1)

            # Configure for simple conv2d
            yield conv2d.cfg_valid.eq(1)
            yield conv2d.cfg_cmd.eq(Conv2dCmd.CONFIG_INPUT_DIMS)
            yield conv2d.cfg_data.eq((3 << 40) | (4 << 24) | (4 << 8) | 1)
            yield Tick()
            yield conv2d.cfg_valid.eq(0)
            yield Tick()

            yield conv2d.cfg_valid.eq(1)
            yield conv2d.cfg_cmd.eq(Conv2dCmd.CONFIG_OUTPUT_DIMS)
            yield conv2d.cfg_data.eq((4 << 32) | (4 << 16) | 4)
            yield Tick()
            yield conv2d.cfg_valid.eq(0)
            yield Tick()

            yield conv2d.cfg_valid.eq(1)
            yield conv2d.cfg_cmd.eq(Conv2dCmd.CONFIG_KERNEL_DIMS)
            yield conv2d.cfg_data.eq((1 << 8) | 1)
            yield Tick()
            yield conv2d.cfg_valid.eq(0)
            yield Tick()

            yield conv2d.cfg_valid.eq(1)
            yield conv2d.cfg_cmd.eq(Conv2dCmd.CONFIG_X_ADDR)
            yield conv2d.cfg_data.eq(0x1000)
            yield Tick()
            yield conv2d.cfg_valid.eq(0)
            yield Tick()

            yield conv2d.cfg_valid.eq(1)
            yield conv2d.cfg_cmd.eq(Conv2dCmd.CONFIG_F_ADDR)
            yield conv2d.cfg_data.eq(0x2000)
            yield Tick()
            yield conv2d.cfg_valid.eq(0)
            yield Tick()

            yield conv2d.cfg_valid.eq(1)
            yield conv2d.cfg_cmd.eq(Conv2dCmd.CONFIG_Y_ADDR)
            yield conv2d.cfg_data.eq(0x3000)
            yield Tick()
            yield conv2d.cfg_valid.eq(0)
            yield Tick()

            # Start execution
            yield conv2d.cfg_valid.eq(1)
            yield conv2d.cfg_cmd.eq(Conv2dCmd.START)
            yield Tick()
            yield conv2d.cfg_valid.eq(0)

            # Collect commands for several cycles
            for _ in range(15):
                yield Tick()
                valid = yield conv2d.cmd_valid
                if valid:
                    opcode = yield conv2d.cmd_opcode
                    rs1 = yield conv2d.cmd_rs1
                    rs2 = yield conv2d.cmd_rs2
                    commands_seen.append((opcode, rs1, rs2))

        sim = Simulator(conv2d)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()

        # Verify we saw commands
        assert len(commands_seen) > 0, "No commands were emitted"

        # First command should be CONFIG
        assert commands_seen[0][0] == InternalOpcode.EXEC_CONFIG

        # Should see LOAD_X and LOAD_F
        opcodes = [cmd[0] for cmd in commands_seen]
        assert InternalOpcode.LOAD_X in opcodes
        assert InternalOpcode.LOAD_F in opcodes


class TestErrorHandling:
    """Test error handling."""

    def test_start_without_config(self, conv2d):
        """Test that starting without configuration raises an error."""
        from systars.isa.conv2d import Conv2dError

        def testbench():
            # Try to start without configuring anything
            yield conv2d.cfg_valid.eq(1)
            yield conv2d.cfg_cmd.eq(Conv2dCmd.START)
            yield Tick()
            yield conv2d.cfg_valid.eq(0)

            # Check error immediately after START
            yield Tick()
            error_code = yield conv2d.error_code
            assert error_code == Conv2dError.CONFIG_INCOMPLETE

        sim = Simulator(conv2d)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()

    def test_partial_config_error(self, conv2d):
        """Test that starting with partial configuration raises an error."""
        from systars.isa.conv2d import Conv2dError

        def testbench():
            # Configure only input dimensions
            yield conv2d.cfg_valid.eq(1)
            yield conv2d.cfg_cmd.eq(Conv2dCmd.CONFIG_INPUT_DIMS)
            yield conv2d.cfg_data.eq((3 << 40) | (8 << 24) | (8 << 8) | 1)
            yield Tick()
            yield conv2d.cfg_valid.eq(0)
            yield Tick()

            # Try to start without output dims, kernel dims, or addresses
            yield conv2d.cfg_valid.eq(1)
            yield conv2d.cfg_cmd.eq(Conv2dCmd.START)
            yield Tick()
            yield conv2d.cfg_valid.eq(0)

            yield Tick()
            error_code = yield conv2d.error_code
            assert error_code == Conv2dError.CONFIG_INCOMPLETE

        sim = Simulator(conv2d)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()


class TestBackpressure:
    """Test backpressure handling for command interface."""

    def test_stall_on_backpressure(self, conv2d):
        """Test that state machine stalls when cmd_ready is low."""
        from systars.isa.conv2d import InternalOpcode

        def testbench():
            # Start with cmd_ready low
            yield conv2d.cmd_ready.eq(0)

            # Full configuration
            yield conv2d.cfg_valid.eq(1)
            yield conv2d.cfg_cmd.eq(Conv2dCmd.CONFIG_INPUT_DIMS)
            yield conv2d.cfg_data.eq((3 << 40) | (4 << 24) | (4 << 8) | 1)
            yield Tick()
            yield conv2d.cfg_valid.eq(0)
            yield Tick()

            yield conv2d.cfg_valid.eq(1)
            yield conv2d.cfg_cmd.eq(Conv2dCmd.CONFIG_OUTPUT_DIMS)
            yield conv2d.cfg_data.eq((4 << 32) | (4 << 16) | 4)
            yield Tick()
            yield conv2d.cfg_valid.eq(0)
            yield Tick()

            yield conv2d.cfg_valid.eq(1)
            yield conv2d.cfg_cmd.eq(Conv2dCmd.CONFIG_KERNEL_DIMS)
            yield conv2d.cfg_data.eq((1 << 8) | 1)
            yield Tick()
            yield conv2d.cfg_valid.eq(0)
            yield Tick()

            yield conv2d.cfg_valid.eq(1)
            yield conv2d.cfg_cmd.eq(Conv2dCmd.CONFIG_X_ADDR)
            yield conv2d.cfg_data.eq(0x1000)
            yield Tick()
            yield conv2d.cfg_valid.eq(0)
            yield Tick()

            yield conv2d.cfg_valid.eq(1)
            yield conv2d.cfg_cmd.eq(Conv2dCmd.CONFIG_F_ADDR)
            yield conv2d.cfg_data.eq(0x2000)
            yield Tick()
            yield conv2d.cfg_valid.eq(0)
            yield Tick()

            yield conv2d.cfg_valid.eq(1)
            yield conv2d.cfg_cmd.eq(Conv2dCmd.CONFIG_Y_ADDR)
            yield conv2d.cfg_data.eq(0x3000)
            yield Tick()
            yield conv2d.cfg_valid.eq(0)
            yield Tick()

            # Start execution
            yield conv2d.cfg_valid.eq(1)
            yield conv2d.cfg_cmd.eq(Conv2dCmd.START)
            yield Tick()
            yield conv2d.cfg_valid.eq(0)

            # Wait for INIT and CALC_TILES
            yield Tick()
            yield Tick()
            yield Tick()

            # Should be stuck in CONFIG state with cmd_valid high
            valid = yield conv2d.cmd_valid
            opcode = yield conv2d.cmd_opcode
            assert valid == 1, "cmd_valid should be high in CONFIG state"
            assert opcode == InternalOpcode.EXEC_CONFIG

            # Run several cycles with cmd_ready=0 - should stay stuck
            for _ in range(3):
                yield Tick()
                valid = yield conv2d.cmd_valid
                opcode = yield conv2d.cmd_opcode
                assert valid == 1
                assert opcode == InternalOpcode.EXEC_CONFIG

            # Assert cmd_ready - should advance
            yield conv2d.cmd_ready.eq(1)
            yield Tick()

            # Should have moved to LOAD_X state
            opcode = yield conv2d.cmd_opcode
            assert opcode == InternalOpcode.LOAD_X

        sim = Simulator(conv2d)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()


class TestDoubleBuffering:
    """Test double buffering bank selection."""

    def test_bank_toggling(self, conv2d):
        """Test that scratchpad banks toggle after each K iteration."""
        from systars.isa.conv2d import InternalOpcode

        bank_history = []

        def testbench():
            yield conv2d.cmd_ready.eq(1)

            # Configure for conv2d with multiple K tiles
            # Input: batch=1, h=4, w=4, c=8 (K = 1*1*8 = 8, 2 tiles at dim=4)
            yield conv2d.cfg_valid.eq(1)
            yield conv2d.cfg_cmd.eq(Conv2dCmd.CONFIG_INPUT_DIMS)
            yield conv2d.cfg_data.eq((8 << 40) | (4 << 24) | (4 << 8) | 1)
            yield Tick()
            yield conv2d.cfg_valid.eq(0)
            yield Tick()

            yield conv2d.cfg_valid.eq(1)
            yield conv2d.cfg_cmd.eq(Conv2dCmd.CONFIG_OUTPUT_DIMS)
            yield conv2d.cfg_data.eq((4 << 32) | (4 << 16) | 4)
            yield Tick()
            yield conv2d.cfg_valid.eq(0)
            yield Tick()

            yield conv2d.cfg_valid.eq(1)
            yield conv2d.cfg_cmd.eq(Conv2dCmd.CONFIG_KERNEL_DIMS)
            yield conv2d.cfg_data.eq((1 << 8) | 1)
            yield Tick()
            yield conv2d.cfg_valid.eq(0)
            yield Tick()

            yield conv2d.cfg_valid.eq(1)
            yield conv2d.cfg_cmd.eq(Conv2dCmd.CONFIG_X_ADDR)
            yield conv2d.cfg_data.eq(0x1000)
            yield Tick()
            yield conv2d.cfg_valid.eq(0)
            yield Tick()

            yield conv2d.cfg_valid.eq(1)
            yield conv2d.cfg_cmd.eq(Conv2dCmd.CONFIG_F_ADDR)
            yield conv2d.cfg_data.eq(0x2000)
            yield Tick()
            yield conv2d.cfg_valid.eq(0)
            yield Tick()

            yield conv2d.cfg_valid.eq(1)
            yield conv2d.cfg_cmd.eq(Conv2dCmd.CONFIG_Y_ADDR)
            yield conv2d.cfg_data.eq(0x3000)
            yield Tick()
            yield conv2d.cfg_valid.eq(0)
            yield Tick()

            # Start execution
            yield conv2d.cfg_valid.eq(1)
            yield conv2d.cfg_cmd.eq(Conv2dCmd.START)
            yield Tick()
            yield conv2d.cfg_valid.eq(0)

            # Collect bank values at each LOAD_X command
            for _ in range(50):
                yield Tick()
                valid = yield conv2d.cmd_valid
                opcode = yield conv2d.cmd_opcode
                if valid and opcode == InternalOpcode.LOAD_X:
                    bank_x = yield conv2d.dbg_sp_bank_X
                    bank_f = yield conv2d.dbg_sp_bank_F
                    bank_history.append((bank_x, bank_f))

                done = yield conv2d.done
                if done:
                    break

        sim = Simulator(conv2d)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()

        # With K=8 and dim=4, we should have 2 K iterations
        # First should use banks (0, 1), second should use (2, 3)
        assert len(bank_history) >= 2, f"Expected at least 2 LOAD_X, got {len(bank_history)}"
        assert bank_history[0] == (0, 1), (
            f"First LOAD_X should use banks (0,1), got {bank_history[0]}"
        )
        assert bank_history[1] == (2, 3), (
            f"Second LOAD_X should use banks (2,3), got {bank_history[1]}"
        )


class TestCompleteTiledConv2d:
    """Test complete tiled conv2d execution."""

    def test_simple_conv2d_execution(self, conv2d):
        """Test execution of simple 4x4 conv2d with 1x1 kernel."""
        from systars.isa.conv2d import InternalOpcode

        command_sequence = []

        def testbench():
            yield conv2d.cmd_ready.eq(1)

            # Configure for 4x4 input, 1x1 kernel, 4 output channels
            # M = 1 * 4 * 4 = 16, N = 4, K = 1 * 1 * 3 = 3
            # With dim=4: tiles_M=4, tiles_N=1, tiles_K=1
            yield conv2d.cfg_valid.eq(1)
            yield conv2d.cfg_cmd.eq(Conv2dCmd.CONFIG_INPUT_DIMS)
            yield conv2d.cfg_data.eq((3 << 40) | (4 << 24) | (4 << 8) | 1)
            yield Tick()
            yield conv2d.cfg_valid.eq(0)
            yield Tick()

            yield conv2d.cfg_valid.eq(1)
            yield conv2d.cfg_cmd.eq(Conv2dCmd.CONFIG_OUTPUT_DIMS)
            yield conv2d.cfg_data.eq((4 << 32) | (4 << 16) | 4)
            yield Tick()
            yield conv2d.cfg_valid.eq(0)
            yield Tick()

            yield conv2d.cfg_valid.eq(1)
            yield conv2d.cfg_cmd.eq(Conv2dCmd.CONFIG_KERNEL_DIMS)
            yield conv2d.cfg_data.eq((1 << 8) | 1)
            yield Tick()
            yield conv2d.cfg_valid.eq(0)
            yield Tick()

            yield conv2d.cfg_valid.eq(1)
            yield conv2d.cfg_cmd.eq(Conv2dCmd.CONFIG_X_ADDR)
            yield conv2d.cfg_data.eq(0x1000)
            yield Tick()
            yield conv2d.cfg_valid.eq(0)
            yield Tick()

            yield conv2d.cfg_valid.eq(1)
            yield conv2d.cfg_cmd.eq(Conv2dCmd.CONFIG_F_ADDR)
            yield conv2d.cfg_data.eq(0x2000)
            yield Tick()
            yield conv2d.cfg_valid.eq(0)
            yield Tick()

            yield conv2d.cfg_valid.eq(1)
            yield conv2d.cfg_cmd.eq(Conv2dCmd.CONFIG_Y_ADDR)
            yield conv2d.cfg_data.eq(0x3000)
            yield Tick()
            yield conv2d.cfg_valid.eq(0)
            yield Tick()

            # Start execution
            yield conv2d.cfg_valid.eq(1)
            yield conv2d.cfg_cmd.eq(Conv2dCmd.START)
            yield Tick()
            yield conv2d.cfg_valid.eq(0)

            # Run until done
            for _ in range(200):
                yield Tick()
                valid = yield conv2d.cmd_valid
                if valid:
                    opcode = yield conv2d.cmd_opcode
                    command_sequence.append(opcode)

                done = yield conv2d.done
                if done:
                    break

        sim = Simulator(conv2d)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()

        # Verify command sequence structure
        # M=16, N=4, K=3 with dim=4: tiles_M=4, tiles_N=1, tiles_K=1
        # Each output tile: LOAD_X, LOAD_F, PRELOAD, COMPUTE, STORE
        # 4 output tiles total = 4 STORE_Y commands
        store_count = sum(1 for op in command_sequence if op == InternalOpcode.STORE_Y)
        assert store_count == 4, f"Expected 4 STORE_Y commands, got {store_count}"

        # 4 LOAD_X commands (1 per tile * 4 tiles)
        load_x_count = sum(1 for op in command_sequence if op == InternalOpcode.LOAD_X)
        assert load_x_count == 4, f"Expected 4 LOAD_X commands, got {load_x_count}"

    def test_with_bias(self, conv2d):
        """Test execution with bias loading."""
        from systars.isa.conv2d import InternalOpcode

        load_b_count = 0

        def testbench():
            nonlocal load_b_count
            yield conv2d.cmd_ready.eq(1)

            # Configure for simple conv2d with bias
            yield conv2d.cfg_valid.eq(1)
            yield conv2d.cfg_cmd.eq(Conv2dCmd.CONFIG_INPUT_DIMS)
            yield conv2d.cfg_data.eq((3 << 40) | (4 << 24) | (4 << 8) | 1)
            yield Tick()
            yield conv2d.cfg_valid.eq(0)
            yield Tick()

            yield conv2d.cfg_valid.eq(1)
            yield conv2d.cfg_cmd.eq(Conv2dCmd.CONFIG_OUTPUT_DIMS)
            yield conv2d.cfg_data.eq((4 << 32) | (4 << 16) | 4)
            yield Tick()
            yield conv2d.cfg_valid.eq(0)
            yield Tick()

            yield conv2d.cfg_valid.eq(1)
            yield conv2d.cfg_cmd.eq(Conv2dCmd.CONFIG_KERNEL_DIMS)
            yield conv2d.cfg_data.eq((1 << 8) | 1)
            yield Tick()
            yield conv2d.cfg_valid.eq(0)
            yield Tick()

            yield conv2d.cfg_valid.eq(1)
            yield conv2d.cfg_cmd.eq(Conv2dCmd.CONFIG_X_ADDR)
            yield conv2d.cfg_data.eq(0x1000)
            yield Tick()
            yield conv2d.cfg_valid.eq(0)
            yield Tick()

            yield conv2d.cfg_valid.eq(1)
            yield conv2d.cfg_cmd.eq(Conv2dCmd.CONFIG_F_ADDR)
            yield conv2d.cfg_data.eq(0x2000)
            yield Tick()
            yield conv2d.cfg_valid.eq(0)
            yield Tick()

            yield conv2d.cfg_valid.eq(1)
            yield conv2d.cfg_cmd.eq(Conv2dCmd.CONFIG_Y_ADDR)
            yield conv2d.cfg_data.eq(0x3000)
            yield Tick()
            yield conv2d.cfg_valid.eq(0)
            yield Tick()

            # Set bias address (non-zero to enable bias loading)
            yield conv2d.cfg_valid.eq(1)
            yield conv2d.cfg_cmd.eq(Conv2dCmd.CONFIG_B_ADDR)
            yield conv2d.cfg_data.eq(0x4000)
            yield Tick()
            yield conv2d.cfg_valid.eq(0)
            yield Tick()

            # Start execution
            yield conv2d.cfg_valid.eq(1)
            yield conv2d.cfg_cmd.eq(Conv2dCmd.START)
            yield Tick()
            yield conv2d.cfg_valid.eq(0)

            # Run until done
            for _ in range(200):
                yield Tick()
                valid = yield conv2d.cmd_valid
                if valid:
                    opcode = yield conv2d.cmd_opcode
                    if opcode == InternalOpcode.LOAD_B:
                        load_b_count += 1

                done = yield conv2d.done
                if done:
                    break

        sim = Simulator(conv2d)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()

        # Should have 4 LOAD_B commands (one per output tile)
        assert load_b_count == 4, f"Expected 4 LOAD_B commands, got {load_b_count}"
