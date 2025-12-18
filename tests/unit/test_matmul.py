"""
Unit tests for the Matmul ISA instruction.

Tests cover:
- Configuration interface
- Address calculation for different tile positions
- Edge tile size handling
- Dataflow selection heuristics
"""

import pytest
from amaranth.sim import Simulator, Tick

from systars.config import SystolicConfig
from systars.isa.matmul import Matmul, MatmulCmd


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
def matmul(config):
    """Create a Matmul instance."""
    return Matmul(config)


class TestMatmulConfiguration:
    """Test configuration interface."""

    def test_config_dims(self, matmul):
        """Test dimension configuration."""

        def testbench():
            # Configure dimensions: M=16, N=8, K=32
            yield matmul.cfg_valid.eq(1)
            yield matmul.cfg_cmd.eq(MatmulCmd.CONFIG_DIMS)
            # Pack: K[47:32] | N[31:16] | M[15:0]
            yield matmul.cfg_data.eq((32 << 32) | (8 << 16) | 16)
            yield Tick()
            yield matmul.cfg_valid.eq(0)
            yield Tick()
            yield Tick()

            # Verify dimensions were latched
            assert (yield matmul.param_M) == 16
            assert (yield matmul.param_N) == 8
            assert (yield matmul.param_K) == 32

        sim = Simulator(matmul)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()

    def test_config_addresses(self, matmul):
        """Test address configuration."""

        def testbench():
            # Configure A address
            yield matmul.cfg_valid.eq(1)
            yield matmul.cfg_cmd.eq(MatmulCmd.CONFIG_A_ADDR)
            yield matmul.cfg_data.eq(0x1000_0000)
            yield Tick()
            yield matmul.cfg_valid.eq(0)
            yield Tick()

            # Configure B address
            yield matmul.cfg_valid.eq(1)
            yield matmul.cfg_cmd.eq(MatmulCmd.CONFIG_B_ADDR)
            yield matmul.cfg_data.eq(0x2000_0000)
            yield Tick()
            yield matmul.cfg_valid.eq(0)
            yield Tick()

            # Configure C address
            yield matmul.cfg_valid.eq(1)
            yield matmul.cfg_cmd.eq(MatmulCmd.CONFIG_C_ADDR)
            yield matmul.cfg_data.eq(0x3000_0000)
            yield Tick()
            yield matmul.cfg_valid.eq(0)
            yield Tick()

        sim = Simulator(matmul)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()


class TestAddressCalculation:
    """Test address calculation for tiles."""

    def test_first_tile_address(self, matmul):
        """Test address calculation for the first tile (0, 0, 0)."""
        # dim = config.grid_rows * _config.tile_rows = 4

        def testbench():
            # Configure dimensions and addresses
            yield matmul.cfg_valid.eq(1)
            yield matmul.cfg_cmd.eq(MatmulCmd.CONFIG_DIMS)
            yield matmul.cfg_data.eq((8 << 32) | (8 << 16) | 8)  # 8x8 @ 8x8
            yield Tick()
            yield matmul.cfg_valid.eq(0)
            yield Tick()

            yield matmul.cfg_valid.eq(1)
            yield matmul.cfg_cmd.eq(MatmulCmd.CONFIG_A_ADDR)
            yield matmul.cfg_data.eq(0x1000)
            yield Tick()
            yield matmul.cfg_valid.eq(0)
            yield Tick()

            yield matmul.cfg_valid.eq(1)
            yield matmul.cfg_cmd.eq(MatmulCmd.CONFIG_B_ADDR)
            yield matmul.cfg_data.eq(0x2000)
            yield Tick()
            yield matmul.cfg_valid.eq(0)
            yield Tick()

            yield matmul.cfg_valid.eq(1)
            yield matmul.cfg_cmd.eq(MatmulCmd.CONFIG_C_ADDR)
            yield matmul.cfg_data.eq(0x3000)
            yield Tick()
            yield matmul.cfg_valid.eq(0)
            yield Tick()

            # Configure strides (row stride in bytes)
            # A is 8 cols * 1 byte = 8 bytes per row
            yield matmul.cfg_valid.eq(1)
            yield matmul.cfg_cmd.eq(MatmulCmd.CONFIG_A_STRIDE)
            yield matmul.cfg_data.eq(8)
            yield Tick()
            yield matmul.cfg_valid.eq(0)
            yield Tick()

            yield matmul.cfg_valid.eq(1)
            yield matmul.cfg_cmd.eq(MatmulCmd.CONFIG_B_STRIDE)
            yield matmul.cfg_data.eq(8)
            yield Tick()
            yield matmul.cfg_valid.eq(0)
            yield Tick()

            yield matmul.cfg_valid.eq(1)
            yield matmul.cfg_cmd.eq(MatmulCmd.CONFIG_C_STRIDE)
            yield matmul.cfg_data.eq(32)  # 8 cols * 4 bytes
            yield Tick()
            yield matmul.cfg_valid.eq(0)
            yield Tick()

            # Start execution
            yield matmul.cfg_valid.eq(1)
            yield matmul.cfg_cmd.eq(MatmulCmd.START)
            yield Tick()
            yield matmul.cfg_valid.eq(0)
            yield Tick()

            # Wait for INIT state to complete
            yield Tick()
            yield Tick()

            # Check addresses for first tile (i=0, j=0, k=0)
            # A tile: base + 0 = 0x1000
            # B tile: base + 0 = 0x2000
            # C tile: base + 0 = 0x3000
            assert (yield matmul.dbg_addr_A) == 0x1000
            assert (yield matmul.dbg_addr_B) == 0x2000
            assert (yield matmul.dbg_addr_C) == 0x3000

            # Check tile sizes (should be full tiles for 8x8 with dim=4)
            assert (yield matmul.dbg_tile_M) == 4
            assert (yield matmul.dbg_tile_N) == 4
            assert (yield matmul.dbg_tile_K) == 4

        sim = Simulator(matmul)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()

    def test_edge_tile_sizes(self, matmul):
        """Test tile size calculation for edge tiles."""
        # dim = config.grid_rows * _config.tile_rows = 4

        def testbench():
            # Configure dimensions: 6x6 matrix (not multiple of dim=4)
            yield matmul.cfg_valid.eq(1)
            yield matmul.cfg_cmd.eq(MatmulCmd.CONFIG_DIMS)
            yield matmul.cfg_data.eq((6 << 32) | (6 << 16) | 6)
            yield Tick()
            yield matmul.cfg_valid.eq(0)
            yield Tick()

            yield matmul.cfg_valid.eq(1)
            yield matmul.cfg_cmd.eq(MatmulCmd.CONFIG_A_ADDR)
            yield matmul.cfg_data.eq(0x1000)
            yield Tick()
            yield matmul.cfg_valid.eq(0)
            yield Tick()

            yield matmul.cfg_valid.eq(1)
            yield matmul.cfg_cmd.eq(MatmulCmd.CONFIG_B_ADDR)
            yield matmul.cfg_data.eq(0x2000)
            yield Tick()
            yield matmul.cfg_valid.eq(0)
            yield Tick()

            yield matmul.cfg_valid.eq(1)
            yield matmul.cfg_cmd.eq(MatmulCmd.CONFIG_C_ADDR)
            yield matmul.cfg_data.eq(0x3000)
            yield Tick()
            yield matmul.cfg_valid.eq(0)
            yield Tick()

            # Strides
            yield matmul.cfg_valid.eq(1)
            yield matmul.cfg_cmd.eq(MatmulCmd.CONFIG_A_STRIDE)
            yield matmul.cfg_data.eq(6)
            yield Tick()
            yield matmul.cfg_valid.eq(0)
            yield Tick()

            yield matmul.cfg_valid.eq(1)
            yield matmul.cfg_cmd.eq(MatmulCmd.CONFIG_B_STRIDE)
            yield matmul.cfg_data.eq(6)
            yield Tick()
            yield matmul.cfg_valid.eq(0)
            yield Tick()

            yield matmul.cfg_valid.eq(1)
            yield matmul.cfg_cmd.eq(MatmulCmd.CONFIG_C_STRIDE)
            yield matmul.cfg_data.eq(24)
            yield Tick()
            yield matmul.cfg_valid.eq(0)
            yield Tick()

            # Start execution
            yield matmul.cfg_valid.eq(1)
            yield matmul.cfg_cmd.eq(MatmulCmd.START)
            yield Tick()
            yield matmul.cfg_valid.eq(0)
            yield Tick()

            # Wait for INIT
            yield Tick()
            yield Tick()

            # First tile (0,0,0) should be full size (4x4)
            assert (yield matmul.dbg_tile_M) == 4
            assert (yield matmul.dbg_tile_N) == 4
            assert (yield matmul.dbg_tile_K) == 4

            # Progress would be at tile (0,0,0)
            assert (yield matmul.progress_i) == 0
            assert (yield matmul.progress_j) == 0
            assert (yield matmul.progress_k) == 0

        sim = Simulator(matmul)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()


class TestDataflowSelection:
    """Test automatic dataflow selection."""

    def test_output_stationary_default(self, matmul):
        """Test that output-stationary is selected for balanced dimensions."""

        def testbench():
            # Configure balanced dimensions: 16x16x16
            yield matmul.cfg_valid.eq(1)
            yield matmul.cfg_cmd.eq(MatmulCmd.CONFIG_DIMS)
            yield matmul.cfg_data.eq((16 << 32) | (16 << 16) | 16)
            yield Tick()
            yield matmul.cfg_valid.eq(0)
            yield Tick()
            yield Tick()

            # Should select output-stationary (0)
            assert (yield matmul.selected_dataflow) == 0

        sim = Simulator(matmul)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()

    def test_weight_stationary_large_k(self, matmul):
        """Test that B-stationary is selected when K is dominant."""

        def testbench():
            # Configure K >> M, N: M=4, N=4, K=64
            yield matmul.cfg_valid.eq(1)
            yield matmul.cfg_cmd.eq(MatmulCmd.CONFIG_DIMS)
            yield matmul.cfg_data.eq((64 << 32) | (4 << 16) | 4)
            yield Tick()
            yield matmul.cfg_valid.eq(0)
            yield Tick()
            yield Tick()

            # Should select B-stationary (2) because K >> M and K >> N
            assert (yield matmul.selected_dataflow) == 2

        sim = Simulator(matmul)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()


class TestCommandEmission:
    """Test command emission from state machine."""

    def test_config_command_emitted(self, matmul):
        """Test that CONFIG command is emitted after START."""
        from systars.isa.matmul import InternalOpcode

        commands_seen = []

        def testbench():
            # Always ready to accept commands (for backpressure testing)
            yield matmul.cmd_ready.eq(1)

            # Configure dimensions
            yield matmul.cfg_valid.eq(1)
            yield matmul.cfg_cmd.eq(MatmulCmd.CONFIG_DIMS)
            yield matmul.cfg_data.eq((8 << 32) | (8 << 16) | 8)  # 8x8x8
            yield Tick()
            yield matmul.cfg_valid.eq(0)
            yield Tick()

            # Configure addresses
            yield matmul.cfg_valid.eq(1)
            yield matmul.cfg_cmd.eq(MatmulCmd.CONFIG_A_ADDR)
            yield matmul.cfg_data.eq(0x1000)
            yield Tick()
            yield matmul.cfg_valid.eq(0)
            yield Tick()

            yield matmul.cfg_valid.eq(1)
            yield matmul.cfg_cmd.eq(MatmulCmd.CONFIG_B_ADDR)
            yield matmul.cfg_data.eq(0x2000)
            yield Tick()
            yield matmul.cfg_valid.eq(0)
            yield Tick()

            yield matmul.cfg_valid.eq(1)
            yield matmul.cfg_cmd.eq(MatmulCmd.CONFIG_C_ADDR)
            yield matmul.cfg_data.eq(0x3000)
            yield Tick()
            yield matmul.cfg_valid.eq(0)
            yield Tick()

            # Configure strides
            yield matmul.cfg_valid.eq(1)
            yield matmul.cfg_cmd.eq(MatmulCmd.CONFIG_A_STRIDE)
            yield matmul.cfg_data.eq(8)
            yield Tick()
            yield matmul.cfg_valid.eq(0)
            yield Tick()

            yield matmul.cfg_valid.eq(1)
            yield matmul.cfg_cmd.eq(MatmulCmd.CONFIG_B_STRIDE)
            yield matmul.cfg_data.eq(8)
            yield Tick()
            yield matmul.cfg_valid.eq(0)
            yield Tick()

            yield matmul.cfg_valid.eq(1)
            yield matmul.cfg_cmd.eq(MatmulCmd.CONFIG_C_STRIDE)
            yield matmul.cfg_data.eq(32)
            yield Tick()
            yield matmul.cfg_valid.eq(0)
            yield Tick()

            # Start execution
            yield matmul.cfg_valid.eq(1)
            yield matmul.cfg_cmd.eq(MatmulCmd.START)
            yield Tick()
            yield matmul.cfg_valid.eq(0)

            # Collect commands for several cycles
            for _ in range(10):
                yield Tick()
                valid = yield matmul.cmd_valid
                if valid:
                    opcode = yield matmul.cmd_opcode
                    rs1 = yield matmul.cmd_rs1
                    rs2 = yield matmul.cmd_rs2
                    commands_seen.append((opcode, rs1, rs2))

        sim = Simulator(matmul)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()

        # Verify we saw commands
        assert len(commands_seen) > 0, "No commands were emitted"

        # First command should be CONFIG
        assert commands_seen[0][0] == InternalOpcode.EXEC_CONFIG

        # Should see LOAD_A and LOAD_B
        opcodes = [cmd[0] for cmd in commands_seen]
        assert InternalOpcode.LOAD_A in opcodes
        assert InternalOpcode.LOAD_B in opcodes

    def test_load_addresses_correct(self, matmul):
        """Test that LOAD commands have correct addresses."""
        from systars.isa.matmul import InternalOpcode

        load_a_addr = None
        load_b_addr = None

        def testbench():
            nonlocal load_a_addr, load_b_addr

            # Always ready to accept commands (for backpressure testing)
            yield matmul.cmd_ready.eq(1)

            # Configure for 8x8 matmul
            yield matmul.cfg_valid.eq(1)
            yield matmul.cfg_cmd.eq(MatmulCmd.CONFIG_DIMS)
            yield matmul.cfg_data.eq((8 << 32) | (8 << 16) | 8)
            yield Tick()
            yield matmul.cfg_valid.eq(0)
            yield Tick()

            # Set specific addresses
            yield matmul.cfg_valid.eq(1)
            yield matmul.cfg_cmd.eq(MatmulCmd.CONFIG_A_ADDR)
            yield matmul.cfg_data.eq(0xAAAA_0000)
            yield Tick()
            yield matmul.cfg_valid.eq(0)
            yield Tick()

            yield matmul.cfg_valid.eq(1)
            yield matmul.cfg_cmd.eq(MatmulCmd.CONFIG_B_ADDR)
            yield matmul.cfg_data.eq(0xBBBB_0000)
            yield Tick()
            yield matmul.cfg_valid.eq(0)
            yield Tick()

            yield matmul.cfg_valid.eq(1)
            yield matmul.cfg_cmd.eq(MatmulCmd.CONFIG_C_ADDR)
            yield matmul.cfg_data.eq(0xCCCC_0000)
            yield Tick()
            yield matmul.cfg_valid.eq(0)
            yield Tick()

            # Set strides
            yield matmul.cfg_valid.eq(1)
            yield matmul.cfg_cmd.eq(MatmulCmd.CONFIG_A_STRIDE)
            yield matmul.cfg_data.eq(8)
            yield Tick()
            yield matmul.cfg_valid.eq(0)
            yield Tick()

            yield matmul.cfg_valid.eq(1)
            yield matmul.cfg_cmd.eq(MatmulCmd.CONFIG_B_STRIDE)
            yield matmul.cfg_data.eq(8)
            yield Tick()
            yield matmul.cfg_valid.eq(0)
            yield Tick()

            yield matmul.cfg_valid.eq(1)
            yield matmul.cfg_cmd.eq(MatmulCmd.CONFIG_C_STRIDE)
            yield matmul.cfg_data.eq(32)
            yield Tick()
            yield matmul.cfg_valid.eq(0)
            yield Tick()

            # Start
            yield matmul.cfg_valid.eq(1)
            yield matmul.cfg_cmd.eq(MatmulCmd.START)
            yield Tick()
            yield matmul.cfg_valid.eq(0)

            # Collect LOAD commands (stop after finding both)
            for _ in range(10):
                yield Tick()
                valid = yield matmul.cmd_valid
                if valid:
                    opcode = yield matmul.cmd_opcode
                    rs1 = yield matmul.cmd_rs1
                    if opcode == InternalOpcode.LOAD_A and load_a_addr is None:
                        load_a_addr = rs1
                    elif opcode == InternalOpcode.LOAD_B and load_b_addr is None:
                        load_b_addr = rs1
                # Stop once we have both addresses
                if load_a_addr is not None and load_b_addr is not None:
                    break

        sim = Simulator(matmul)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()

        # LOAD_A should use A base address (first tile at offset 0)
        assert load_a_addr == 0xAAAA_0000, f"Expected 0xAAAA0000, got {load_a_addr:#x}"
        # LOAD_B should use B base address
        assert load_b_addr == 0xBBBB_0000, f"Expected 0xBBBB0000, got {load_b_addr:#x}"


class TestBackpressure:
    """Test backpressure handling for command interface."""

    def test_stall_on_backpressure(self, matmul):
        """Test that state machine stalls when cmd_ready is low."""
        from systars.isa.matmul import InternalOpcode

        states_seen = []

        def testbench():
            # Start with cmd_ready low (not accepting commands)
            yield matmul.cmd_ready.eq(0)

            # Configure dimensions
            yield matmul.cfg_valid.eq(1)
            yield matmul.cfg_cmd.eq(MatmulCmd.CONFIG_DIMS)
            yield matmul.cfg_data.eq((4 << 32) | (4 << 16) | 4)  # 4x4x4
            yield Tick()
            yield matmul.cfg_valid.eq(0)
            yield Tick()

            # Configure addresses
            yield matmul.cfg_valid.eq(1)
            yield matmul.cfg_cmd.eq(MatmulCmd.CONFIG_A_ADDR)
            yield matmul.cfg_data.eq(0x1000)
            yield Tick()
            yield matmul.cfg_valid.eq(0)
            yield Tick()

            yield matmul.cfg_valid.eq(1)
            yield matmul.cfg_cmd.eq(MatmulCmd.CONFIG_B_ADDR)
            yield matmul.cfg_data.eq(0x2000)
            yield Tick()
            yield matmul.cfg_valid.eq(0)
            yield Tick()

            yield matmul.cfg_valid.eq(1)
            yield matmul.cfg_cmd.eq(MatmulCmd.CONFIG_C_ADDR)
            yield matmul.cfg_data.eq(0x3000)
            yield Tick()
            yield matmul.cfg_valid.eq(0)
            yield Tick()

            # Configure strides
            yield matmul.cfg_valid.eq(1)
            yield matmul.cfg_cmd.eq(MatmulCmd.CONFIG_A_STRIDE)
            yield matmul.cfg_data.eq(4)
            yield Tick()
            yield matmul.cfg_valid.eq(0)
            yield Tick()

            yield matmul.cfg_valid.eq(1)
            yield matmul.cfg_cmd.eq(MatmulCmd.CONFIG_B_STRIDE)
            yield matmul.cfg_data.eq(4)
            yield Tick()
            yield matmul.cfg_valid.eq(0)
            yield Tick()

            yield matmul.cfg_valid.eq(1)
            yield matmul.cfg_cmd.eq(MatmulCmd.CONFIG_C_STRIDE)
            yield matmul.cfg_data.eq(16)
            yield Tick()
            yield matmul.cfg_valid.eq(0)
            yield Tick()

            # Start execution
            yield matmul.cfg_valid.eq(1)
            yield matmul.cfg_cmd.eq(MatmulCmd.START)
            yield Tick()
            yield matmul.cfg_valid.eq(0)
            yield Tick()

            # Wait for INIT to complete, then we should be stuck in CONFIG
            yield Tick()

            # Should be stuck in CONFIG state, cmd_valid should be high
            valid = yield matmul.cmd_valid
            opcode = yield matmul.cmd_opcode
            assert valid == 1, "cmd_valid should be high in CONFIG state"
            assert opcode == InternalOpcode.EXEC_CONFIG, "Should be emitting CONFIG"

            # Run several more cycles with cmd_ready=0 - should stay in CONFIG
            for _ in range(3):
                yield Tick()
                valid = yield matmul.cmd_valid
                opcode = yield matmul.cmd_opcode
                states_seen.append((valid, opcode))

            # Verify we're still stuck (same state)
            assert all(v == 1 for v, _ in states_seen), "Should remain valid"
            assert all(o == InternalOpcode.EXEC_CONFIG for _, o in states_seen)

            # Now assert cmd_ready - should advance
            yield matmul.cmd_ready.eq(1)
            yield Tick()

            # Should have moved to next state (LOAD_A since no bias)
            # Check immediately (before the next handshake completes)
            valid = yield matmul.cmd_valid
            opcode = yield matmul.cmd_opcode
            assert opcode == InternalOpcode.LOAD_A, f"Expected LOAD_A, got {opcode}"

            # Deassert cmd_ready to verify we stall on LOAD_A
            yield matmul.cmd_ready.eq(0)
            yield Tick()
            yield Tick()
            opcode2 = yield matmul.cmd_opcode
            assert opcode2 == InternalOpcode.LOAD_A, "Should still be on LOAD_A"

        sim = Simulator(matmul)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()


class TestDoubleBuffering:
    """Test double buffering bank selection."""

    def test_bank_toggling(self, matmul):
        """Test that scratchpad banks toggle after each K iteration."""
        from systars.isa.matmul import InternalOpcode

        bank_history = []

        def testbench():
            # Always ready to accept commands
            yield matmul.cmd_ready.eq(1)

            # Configure for 4x4x8 matmul (dim=4, so K=8 means 2 K iterations)
            yield matmul.cfg_valid.eq(1)
            yield matmul.cfg_cmd.eq(MatmulCmd.CONFIG_DIMS)
            yield matmul.cfg_data.eq((8 << 32) | (4 << 16) | 4)  # K=8, N=4, M=4
            yield Tick()
            yield matmul.cfg_valid.eq(0)
            yield Tick()

            # Configure addresses
            yield matmul.cfg_valid.eq(1)
            yield matmul.cfg_cmd.eq(MatmulCmd.CONFIG_A_ADDR)
            yield matmul.cfg_data.eq(0x1000)
            yield Tick()
            yield matmul.cfg_valid.eq(0)
            yield Tick()

            yield matmul.cfg_valid.eq(1)
            yield matmul.cfg_cmd.eq(MatmulCmd.CONFIG_B_ADDR)
            yield matmul.cfg_data.eq(0x2000)
            yield Tick()
            yield matmul.cfg_valid.eq(0)
            yield Tick()

            yield matmul.cfg_valid.eq(1)
            yield matmul.cfg_cmd.eq(MatmulCmd.CONFIG_C_ADDR)
            yield matmul.cfg_data.eq(0x3000)
            yield Tick()
            yield matmul.cfg_valid.eq(0)
            yield Tick()

            # Configure strides
            yield matmul.cfg_valid.eq(1)
            yield matmul.cfg_cmd.eq(MatmulCmd.CONFIG_A_STRIDE)
            yield matmul.cfg_data.eq(8)
            yield Tick()
            yield matmul.cfg_valid.eq(0)
            yield Tick()

            yield matmul.cfg_valid.eq(1)
            yield matmul.cfg_cmd.eq(MatmulCmd.CONFIG_B_STRIDE)
            yield matmul.cfg_data.eq(4)
            yield Tick()
            yield matmul.cfg_valid.eq(0)
            yield Tick()

            yield matmul.cfg_valid.eq(1)
            yield matmul.cfg_cmd.eq(MatmulCmd.CONFIG_C_STRIDE)
            yield matmul.cfg_data.eq(16)
            yield Tick()
            yield matmul.cfg_valid.eq(0)
            yield Tick()

            # Start execution
            yield matmul.cfg_valid.eq(1)
            yield matmul.cfg_cmd.eq(MatmulCmd.START)
            yield Tick()
            yield matmul.cfg_valid.eq(0)

            # Run until done, recording bank values at each LOAD_A command
            for _ in range(50):  # Enough cycles to complete
                yield Tick()
                valid = yield matmul.cmd_valid
                opcode = yield matmul.cmd_opcode
                if valid and opcode == InternalOpcode.LOAD_A:
                    bank_a = yield matmul.dbg_sp_bank_A
                    bank_b = yield matmul.dbg_sp_bank_B
                    bank_history.append((bank_a, bank_b))

                done = yield matmul.done
                if done:
                    break

        sim = Simulator(matmul)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()

        # Should have 2 LOAD_A commands (2 K iterations)
        assert len(bank_history) >= 2, f"Expected at least 2 LOAD_A, got {len(bank_history)}"

        # First iteration should use banks (0, 1)
        assert bank_history[0] == (0, 1), (
            f"First LOAD_A should use banks (0,1), got {bank_history[0]}"
        )

        # Second iteration should use banks (2, 3)
        assert bank_history[1] == (2, 3), (
            f"Second LOAD_A should use banks (2,3), got {bank_history[1]}"
        )


class TestCompleteTiledMatmul:
    """Test complete tiled matmul execution."""

    def test_multi_tile_execution(self, matmul):
        """Test execution of 8x8 matmul with 4x4 tiles (2x2 output tiles)."""
        from systars.isa.matmul import InternalOpcode

        command_sequence = []

        def testbench():
            yield matmul.cmd_ready.eq(1)

            # Configure for 8x8 @ 8x8 matmul (2x2 output tiles with dim=4)
            yield matmul.cfg_valid.eq(1)
            yield matmul.cfg_cmd.eq(MatmulCmd.CONFIG_DIMS)
            yield matmul.cfg_data.eq((8 << 32) | (8 << 16) | 8)  # M=8, N=8, K=8
            yield Tick()
            yield matmul.cfg_valid.eq(0)
            yield Tick()

            # Configure addresses
            yield matmul.cfg_valid.eq(1)
            yield matmul.cfg_cmd.eq(MatmulCmd.CONFIG_A_ADDR)
            yield matmul.cfg_data.eq(0x1000)
            yield Tick()
            yield matmul.cfg_valid.eq(0)
            yield Tick()

            yield matmul.cfg_valid.eq(1)
            yield matmul.cfg_cmd.eq(MatmulCmd.CONFIG_B_ADDR)
            yield matmul.cfg_data.eq(0x2000)
            yield Tick()
            yield matmul.cfg_valid.eq(0)
            yield Tick()

            yield matmul.cfg_valid.eq(1)
            yield matmul.cfg_cmd.eq(MatmulCmd.CONFIG_C_ADDR)
            yield matmul.cfg_data.eq(0x3000)
            yield Tick()
            yield matmul.cfg_valid.eq(0)
            yield Tick()

            # Configure strides
            yield matmul.cfg_valid.eq(1)
            yield matmul.cfg_cmd.eq(MatmulCmd.CONFIG_A_STRIDE)
            yield matmul.cfg_data.eq(8)
            yield Tick()
            yield matmul.cfg_valid.eq(0)
            yield Tick()

            yield matmul.cfg_valid.eq(1)
            yield matmul.cfg_cmd.eq(MatmulCmd.CONFIG_B_STRIDE)
            yield matmul.cfg_data.eq(8)
            yield Tick()
            yield matmul.cfg_valid.eq(0)
            yield Tick()

            yield matmul.cfg_valid.eq(1)
            yield matmul.cfg_cmd.eq(MatmulCmd.CONFIG_C_STRIDE)
            yield matmul.cfg_data.eq(32)
            yield Tick()
            yield matmul.cfg_valid.eq(0)
            yield Tick()

            # Start execution
            yield matmul.cfg_valid.eq(1)
            yield matmul.cfg_cmd.eq(MatmulCmd.START)
            yield Tick()
            yield matmul.cfg_valid.eq(0)

            # Run until done
            for _ in range(200):  # Enough cycles to complete
                yield Tick()
                valid = yield matmul.cmd_valid
                if valid:
                    opcode = yield matmul.cmd_opcode
                    command_sequence.append(opcode)

                done = yield matmul.done
                if done:
                    break

        sim = Simulator(matmul)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()

        # Verify command sequence structure
        # For 2x2 output tiles with 2 K iterations each:
        # Each output tile: CONFIG, LOAD_A, LOAD_B, PRELOAD, COMPUTE, LOAD_A, LOAD_B, COMPUTE, STORE
        # 4 output tiles total = 4 STORE commands
        store_count = sum(1 for op in command_sequence if op == InternalOpcode.STORE_C)
        assert store_count == 4, f"Expected 4 STORE_C commands for 2x2 tiles, got {store_count}"

        # 8 LOAD_A commands (2 per tile * 4 tiles)
        load_a_count = sum(1 for op in command_sequence if op == InternalOpcode.LOAD_A)
        assert load_a_count == 8, f"Expected 8 LOAD_A commands, got {load_a_count}"

    def test_with_bias(self, matmul):
        """Test execution with bias loading."""
        from systars.isa.matmul import InternalOpcode

        load_d_count = 0

        def testbench():
            nonlocal load_d_count
            yield matmul.cmd_ready.eq(1)

            # Configure for 4x4x4 matmul with bias
            yield matmul.cfg_valid.eq(1)
            yield matmul.cfg_cmd.eq(MatmulCmd.CONFIG_DIMS)
            yield matmul.cfg_data.eq((4 << 32) | (4 << 16) | 4)
            yield Tick()
            yield matmul.cfg_valid.eq(0)
            yield Tick()

            # Configure addresses (including bias)
            yield matmul.cfg_valid.eq(1)
            yield matmul.cfg_cmd.eq(MatmulCmd.CONFIG_A_ADDR)
            yield matmul.cfg_data.eq(0x1000)
            yield Tick()
            yield matmul.cfg_valid.eq(0)
            yield Tick()

            yield matmul.cfg_valid.eq(1)
            yield matmul.cfg_cmd.eq(MatmulCmd.CONFIG_B_ADDR)
            yield matmul.cfg_data.eq(0x2000)
            yield Tick()
            yield matmul.cfg_valid.eq(0)
            yield Tick()

            yield matmul.cfg_valid.eq(1)
            yield matmul.cfg_cmd.eq(MatmulCmd.CONFIG_C_ADDR)
            yield matmul.cfg_data.eq(0x3000)
            yield Tick()
            yield matmul.cfg_valid.eq(0)
            yield Tick()

            # Set bias address (non-zero to enable bias loading)
            yield matmul.cfg_valid.eq(1)
            yield matmul.cfg_cmd.eq(MatmulCmd.CONFIG_D_ADDR)
            yield matmul.cfg_data.eq(0x4000)
            yield Tick()
            yield matmul.cfg_valid.eq(0)
            yield Tick()

            # Configure strides
            yield matmul.cfg_valid.eq(1)
            yield matmul.cfg_cmd.eq(MatmulCmd.CONFIG_A_STRIDE)
            yield matmul.cfg_data.eq(4)
            yield Tick()
            yield matmul.cfg_valid.eq(0)
            yield Tick()

            yield matmul.cfg_valid.eq(1)
            yield matmul.cfg_cmd.eq(MatmulCmd.CONFIG_B_STRIDE)
            yield matmul.cfg_data.eq(4)
            yield Tick()
            yield matmul.cfg_valid.eq(0)
            yield Tick()

            yield matmul.cfg_valid.eq(1)
            yield matmul.cfg_cmd.eq(MatmulCmd.CONFIG_C_STRIDE)
            yield matmul.cfg_data.eq(16)
            yield Tick()
            yield matmul.cfg_valid.eq(0)
            yield Tick()

            yield matmul.cfg_valid.eq(1)
            yield matmul.cfg_cmd.eq(MatmulCmd.CONFIG_D_STRIDE)
            yield matmul.cfg_data.eq(16)
            yield Tick()
            yield matmul.cfg_valid.eq(0)
            yield Tick()

            # Start execution
            yield matmul.cfg_valid.eq(1)
            yield matmul.cfg_cmd.eq(MatmulCmd.START)
            yield Tick()
            yield matmul.cfg_valid.eq(0)

            # Run until done
            for _ in range(50):
                yield Tick()
                valid = yield matmul.cmd_valid
                if valid:
                    opcode = yield matmul.cmd_opcode
                    if opcode == InternalOpcode.LOAD_D:
                        load_d_count += 1

                done = yield matmul.done
                if done:
                    break

        sim = Simulator(matmul)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()

        # Should have 1 LOAD_D for single tile matmul
        assert load_d_count == 1, f"Expected 1 LOAD_D command, got {load_d_count}"


class TestErrorHandling:
    """Test error handling."""

    def test_start_without_config(self, matmul):
        """Test that starting without configuration raises an error."""
        from systars.isa.matmul import MatmulError

        def testbench():
            # Try to start without configuring anything
            yield matmul.cfg_valid.eq(1)
            yield matmul.cfg_cmd.eq(MatmulCmd.START)
            yield Tick()
            yield matmul.cfg_valid.eq(0)

            # Check error immediately after START (error state is transient)
            yield Tick()
            error_code = yield matmul.error_code
            # Should have CONFIG_INCOMPLETE error code
            assert error_code == MatmulError.CONFIG_INCOMPLETE

        sim = Simulator(matmul)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()
