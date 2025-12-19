"""
Unit tests for the Stencil Machine module.

These tests verify:
1. StencilConfig instantiation and validation
2. LineBufferBank basic operations
3. LineBufferUnit circular buffer management
4. WindowFormer sliding window extraction
5. MACBank dot product computation
6. ChannelParallelMAC parallel processing
7. StencilMachine integration
8. Verilog generation
"""

import pytest
from amaranth.sim import Simulator, Tick

from systars.stencil import (
    ChannelParallelMAC,
    LineBufferBank,
    LineBufferUnit,
    MACBank,
    StencilActivation,
    StencilConfig,
    StencilDataflow,
    StencilMachine,
    StencilState,
    WindowFormer,
)

# =============================================================================
# StencilConfig Tests
# =============================================================================


class TestStencilConfig:
    """Test suite for StencilConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        cfg = StencilConfig()
        assert cfg.max_width == 224
        assert cfg.max_height == 224
        assert cfg.max_kernel_h == 7
        assert cfg.max_kernel_w == 7
        assert cfg.input_bits == 8
        assert cfg.weight_bits == 8
        assert cfg.acc_bits == 32
        assert cfg.parallel_channels == 32

    def test_custom_config(self):
        """Test custom configuration."""
        cfg = StencilConfig(
            max_width=56,
            max_height=56,
            max_kernel_h=3,
            max_kernel_w=3,
            parallel_channels=16,
        )
        assert cfg.max_width == 56
        assert cfg.max_kernel_h == 3
        assert cfg.parallel_channels == 16

    def test_computed_properties(self):
        """Test computed properties."""
        cfg = StencilConfig(max_width=56, max_kernel_h=3, max_kernel_w=3)

        # Line buffer width
        assert cfg.line_buffer_width == 56 * 8  # 56 pixels × 8 bits

        # Window size
        assert cfg.window_size == 3 * 3  # 9 elements

        # Total MACs
        assert cfg.total_macs == cfg.parallel_channels * cfg.macs_per_bank

    def test_line_buffer_size_kb(self):
        """Test line buffer size calculation."""
        cfg = StencilConfig(max_width=224, max_kernel_h=3)
        # 3 rows × 224 pixels × 8 bits = 5376 bits = 672 bytes = 0.656 KB
        expected_kb = (3 * 224 * 8) / 8 / 1024
        assert abs(cfg.line_buffer_size_kb - expected_kb) < 0.001

    def test_validation_max_kernel(self):
        """Test that kernel size is validated."""
        with pytest.raises(AssertionError):
            StencilConfig(max_kernel_h=8)  # Max is 7

    def test_validation_parallel_channels_power_of_2(self):
        """Test that parallel_channels must be power of 2."""
        with pytest.raises(AssertionError):
            StencilConfig(parallel_channels=15)  # Not power of 2

    def test_activation_enum(self):
        """Test activation function enum."""
        assert StencilActivation.NONE.value == 0
        assert StencilActivation.RELU.value == 1
        assert StencilActivation.RELU6.value == 2

    def test_dataflow_enum(self):
        """Test dataflow mode enum."""
        assert StencilDataflow.OUTPUT_STATIONARY.value == 0
        assert StencilDataflow.WEIGHT_STATIONARY.value == 1


# =============================================================================
# LineBufferBank Tests
# =============================================================================


class TestLineBufferBank:
    """Test suite for LineBufferBank."""

    @pytest.fixture
    def config(self):
        """Small configuration for tests."""
        return StencilConfig(
            max_width=16,
            max_height=16,
            max_kernel_h=3,
            max_kernel_w=3,
            parallel_channels=4,
        )

    @pytest.fixture
    def bank(self, config):
        """Create a LineBufferBank instance."""
        return LineBufferBank(config, bank_id=0)

    def test_bank_instantiation(self, bank, config):
        """Test that LineBufferBank can be instantiated."""
        assert bank is not None
        assert bank.config == config
        assert bank.bank_id == 0

    def test_bank_has_correct_ports(self, bank):
        """Test that bank has all required ports."""
        assert hasattr(bank, "read_addr")
        assert hasattr(bank, "read_en")
        assert hasattr(bank, "read_data")
        assert hasattr(bank, "read_valid")
        assert hasattr(bank, "write_addr")
        assert hasattr(bank, "write_en")
        assert hasattr(bank, "write_data")

    def test_bank_write_read(self, config):
        """Test basic write then read operation."""
        bank = LineBufferBank(config, bank_id=0)
        results = []

        def testbench():
            # Write to address 0
            yield bank.write_addr.eq(0)
            yield bank.write_data.eq(0x42)
            yield bank.write_en.eq(1)
            yield Tick()

            # Write to address 1
            yield bank.write_addr.eq(1)
            yield bank.write_data.eq(0x55)
            yield Tick()

            # Disable write
            yield bank.write_en.eq(0)
            yield Tick()

            # Read from address 0
            yield bank.read_addr.eq(0)
            yield bank.read_en.eq(1)
            yield Tick()

            # Read valid after 1 cycle
            yield bank.read_en.eq(0)
            valid = yield bank.read_valid
            data = yield bank.read_data
            results.append({"addr": 0, "valid": valid, "data": data})
            yield Tick()

            # Read from address 1
            yield bank.read_addr.eq(1)
            yield bank.read_en.eq(1)
            yield Tick()

            yield bank.read_en.eq(0)
            valid = yield bank.read_valid
            data = yield bank.read_data
            results.append({"addr": 1, "valid": valid, "data": data})

        sim = Simulator(bank)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()

        assert results[0]["valid"] == 1
        assert results[0]["data"] == 0x42
        assert results[1]["valid"] == 1
        assert results[1]["data"] == 0x55

    def test_bank_elaboration(self, bank):
        """Test that LineBufferBank elaborates without errors."""

        def testbench():
            yield Tick()

        sim = Simulator(bank)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()


# =============================================================================
# LineBufferUnit Tests
# =============================================================================


class TestLineBufferUnit:
    """Test suite for LineBufferUnit."""

    @pytest.fixture
    def config(self):
        """Small configuration for tests."""
        return StencilConfig(
            max_width=8,
            max_height=8,
            max_kernel_h=3,
            max_kernel_w=3,
            parallel_channels=4,
        )

    @pytest.fixture
    def unit(self, config):
        """Create a LineBufferUnit instance."""
        return LineBufferUnit(config)

    def test_unit_instantiation(self, unit, config):
        """Test that LineBufferUnit can be instantiated."""
        assert unit is not None
        assert unit.config == config

    def test_unit_has_correct_ports(self, unit):
        """Test that unit has all required ports."""
        # Input stream
        assert hasattr(unit, "in_valid")
        assert hasattr(unit, "in_ready")
        assert hasattr(unit, "in_data")
        assert hasattr(unit, "in_last_col")
        assert hasattr(unit, "in_last_row")

        # Output
        assert hasattr(unit, "out_valid")
        assert hasattr(unit, "out_ready")
        assert hasattr(unit, "out_data")
        assert hasattr(unit, "out_col")

        # Configuration
        assert hasattr(unit, "cfg_kernel_h")
        assert hasattr(unit, "cfg_width")

        # Status
        assert hasattr(unit, "row_count")
        assert hasattr(unit, "ready_for_compute")

    def test_unit_not_ready_initially(self, config):
        """Test that unit is not ready for compute initially."""
        unit = LineBufferUnit(config)
        result = {"ready": None}

        def testbench():
            # Configure for 3x3 kernel
            yield unit.cfg_kernel_h.eq(3)
            yield unit.cfg_width.eq(8)
            yield Tick()

            result["ready"] = yield unit.ready_for_compute

        sim = Simulator(unit)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()

        assert result["ready"] == 0

    def test_unit_ready_after_k_rows(self, config):
        """Test that unit becomes ready after K_h rows are filled."""
        unit = LineBufferUnit(config)
        result = {"ready_history": [], "row_count_history": []}

        def testbench():
            kernel_h = 3
            width = 8

            yield unit.cfg_kernel_h.eq(kernel_h)
            yield unit.cfg_width.eq(width)
            yield unit.out_ready.eq(1)
            yield Tick()

            # Fill rows
            for row in range(kernel_h + 1):
                for col in range(width):
                    yield unit.in_valid.eq(1)
                    yield unit.in_data.eq((row << 4) | col)  # Unique value
                    yield unit.in_last_col.eq(col == width - 1)
                    yield unit.in_last_row.eq(0)
                    yield Tick()

                # Record status after each row
                ready = yield unit.ready_for_compute
                row_count = yield unit.row_count
                result["ready_history"].append(ready)
                result["row_count_history"].append(row_count)

            yield unit.in_valid.eq(0)
            yield Tick()

        sim = Simulator(unit)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()

        # Should be ready after 3 rows (indices 0, 1, 2)
        assert result["ready_history"][0] == 0  # After row 0
        assert result["ready_history"][1] == 0  # After row 1
        assert result["ready_history"][2] == 1  # After row 2 (have 3 rows)

    def test_unit_elaboration(self, unit):
        """Test that LineBufferUnit elaborates without errors."""

        def testbench():
            yield unit.cfg_kernel_h.eq(3)
            yield unit.cfg_width.eq(8)
            yield Tick()

        sim = Simulator(unit)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()


# =============================================================================
# WindowFormer Tests
# =============================================================================


class TestWindowFormer:
    """Test suite for WindowFormer."""

    @pytest.fixture
    def config(self):
        """Small configuration for tests."""
        return StencilConfig(
            max_width=8,
            max_height=8,
            max_kernel_h=3,
            max_kernel_w=3,
            parallel_channels=4,
        )

    @pytest.fixture
    def window_former(self, config):
        """Create a WindowFormer instance."""
        return WindowFormer(config)

    def test_window_former_instantiation(self, window_former, config):
        """Test that WindowFormer can be instantiated."""
        assert window_former is not None
        assert window_former.config == config

    def test_window_former_has_correct_ports(self, window_former):
        """Test that WindowFormer has all required ports."""
        # Input
        assert hasattr(window_former, "in_valid")
        assert hasattr(window_former, "in_ready")
        assert hasattr(window_former, "in_data")

        # Output
        assert hasattr(window_former, "out_valid")
        assert hasattr(window_former, "out_ready")
        assert hasattr(window_former, "out_window")
        assert hasattr(window_former, "out_col")

        # Configuration
        assert hasattr(window_former, "cfg_kernel_h")
        assert hasattr(window_former, "cfg_kernel_w")
        assert hasattr(window_former, "cfg_stride_w")
        assert hasattr(window_former, "cfg_width")

        # Status
        assert hasattr(window_former, "window_valid")
        assert hasattr(window_former, "filling")

    def test_window_former_filling_state(self, config):
        """Test that window former starts in filling state."""
        wf = WindowFormer(config)
        result = {"filling": None, "window_valid": None}

        def testbench():
            yield wf.cfg_kernel_h.eq(3)
            yield wf.cfg_kernel_w.eq(3)
            yield wf.cfg_stride_w.eq(1)
            yield wf.cfg_width.eq(8)
            yield wf.out_ready.eq(1)
            yield Tick()

            result["filling"] = yield wf.filling
            result["window_valid"] = yield wf.window_valid

        sim = Simulator(wf)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()

        assert result["filling"] == 1
        assert result["window_valid"] == 0

    def test_window_former_becomes_valid(self, config):
        """Test that window becomes valid after K_w columns."""
        wf = WindowFormer(config)
        valid_history = []

        def testbench():
            kernel_w = 3
            yield wf.cfg_kernel_h.eq(3)
            yield wf.cfg_kernel_w.eq(kernel_w)
            yield wf.cfg_stride_w.eq(1)
            yield wf.cfg_width.eq(8)
            yield wf.out_ready.eq(1)
            yield Tick()

            # Feed columns
            for col in range(kernel_w + 2):
                yield wf.in_valid.eq(1)
                # Pack 3 rows of data (one from each line buffer)
                data = col | (col << 8) | (col << 16)
                yield wf.in_data.eq(data)
                yield Tick()

                # Check valid after the tick (fill_count updates on clock edge)
                valid = yield wf.window_valid
                valid_history.append(valid)

            yield wf.in_valid.eq(0)
            yield Tick()

        sim = Simulator(wf)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()

        # Window valid after K_w columns fed
        # fill_count increments on each in_valid cycle
        # After col 0: fill_count=1, not valid
        # After col 1: fill_count=2, not valid
        # After col 2: fill_count=3, valid
        assert valid_history[0] == 0  # After col 0 (fill_count = 1)
        assert valid_history[1] == 0  # After col 1 (fill_count = 2)
        assert valid_history[2] == 1  # After col 2 (fill_count = 3, valid!)

    def test_window_former_elaboration(self, window_former):
        """Test that WindowFormer elaborates without errors."""

        def testbench():
            yield window_former.cfg_kernel_h.eq(3)
            yield window_former.cfg_kernel_w.eq(3)
            yield window_former.cfg_stride_w.eq(1)
            yield window_former.cfg_width.eq(8)
            yield Tick()

        sim = Simulator(window_former)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()


# =============================================================================
# MACBank Tests
# =============================================================================


class TestMACBank:
    """Test suite for MACBank."""

    @pytest.fixture
    def config(self):
        """Small configuration for tests."""
        return StencilConfig(
            max_width=8,
            max_height=8,
            max_kernel_h=3,
            max_kernel_w=3,
            parallel_channels=4,
        )

    @pytest.fixture
    def mac_bank(self, config):
        """Create a MACBank instance."""
        return MACBank(config, bank_id=0)

    def test_mac_bank_instantiation(self, mac_bank, config):
        """Test that MACBank can be instantiated."""
        assert mac_bank is not None
        assert mac_bank.config == config
        assert mac_bank.bank_id == 0

    def test_mac_bank_has_correct_ports(self, mac_bank):
        """Test that MACBank has all required ports."""
        # Window input
        assert hasattr(mac_bank, "in_window")
        assert hasattr(mac_bank, "in_valid")

        # Filter
        assert hasattr(mac_bank, "filter_load")
        assert hasattr(mac_bank, "filter_data")

        # Accumulator control
        assert hasattr(mac_bank, "clear_accum")
        assert hasattr(mac_bank, "accum_en")

        # Output
        assert hasattr(mac_bank, "out_sum")
        assert hasattr(mac_bank, "out_valid")

    def test_mac_bank_filter_load(self, config):
        """Test that filters can be loaded."""
        bank = MACBank(config, bank_id=0)

        def testbench():
            # Load filter: 9 coefficients for 3x3
            # Pack as 9 × 8 bits = 72 bits
            filter_val = 0
            for i in range(9):
                filter_val |= (i + 1) << (i * 8)

            yield bank.filter_data.eq(filter_val)
            yield bank.filter_load.eq(1)
            yield Tick()

            yield bank.filter_load.eq(0)
            yield Tick()

        sim = Simulator(bank)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()

    def test_mac_bank_accumulation(self, config):
        """Test basic MAC accumulation."""
        bank = MACBank(config, bank_id=0)
        result = {"sum": None}

        def testbench():
            # Load all-ones filter (each coeff = 1)
            filter_val = 0
            for i in range(9):
                filter_val |= 1 << (i * 8)
            yield bank.filter_data.eq(filter_val)
            yield bank.filter_load.eq(1)
            yield Tick()
            yield bank.filter_load.eq(0)
            yield Tick()

            # Clear accumulator
            yield bank.clear_accum.eq(1)
            yield Tick()
            yield bank.clear_accum.eq(0)
            yield Tick()

            # Apply window with all-twos (each pixel = 2)
            window_val = 0
            for i in range(9):
                window_val |= 2 << (i * 8)
            yield bank.in_window.eq(window_val)
            yield bank.in_valid.eq(1)
            yield bank.accum_en.eq(1)
            yield Tick()

            yield bank.in_valid.eq(0)
            yield bank.accum_en.eq(0)
            yield Tick()

            # Sum should be 9 × (2 × 1) = 18
            result["sum"] = yield bank.out_sum

        sim = Simulator(bank)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()

        assert result["sum"] == 18

    def test_mac_bank_elaboration(self, mac_bank):
        """Test that MACBank elaborates without errors."""

        def testbench():
            yield Tick()

        sim = Simulator(mac_bank)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()


# =============================================================================
# ChannelParallelMAC Tests
# =============================================================================


class TestChannelParallelMAC:
    """Test suite for ChannelParallelMAC."""

    @pytest.fixture
    def config(self):
        """Small configuration for tests."""
        return StencilConfig(
            max_width=8,
            max_height=8,
            max_kernel_h=3,
            max_kernel_w=3,
            parallel_channels=4,
        )

    @pytest.fixture
    def mac_array(self, config):
        """Create a ChannelParallelMAC instance."""
        return ChannelParallelMAC(config)

    def test_mac_array_instantiation(self, mac_array, config):
        """Test that ChannelParallelMAC can be instantiated."""
        assert mac_array is not None
        assert mac_array.config == config

    def test_mac_array_has_correct_ports(self, mac_array):
        """Test that ChannelParallelMAC has all required ports."""
        # Window input
        assert hasattr(mac_array, "in_window_valid")
        assert hasattr(mac_array, "in_window")
        assert hasattr(mac_array, "in_last_channel")

        # Filter loading
        assert hasattr(mac_array, "filter_load")
        assert hasattr(mac_array, "filter_data")
        assert hasattr(mac_array, "filter_bank")

        # Output
        assert hasattr(mac_array, "out_valid")
        assert hasattr(mac_array, "out_ready")
        assert hasattr(mac_array, "out_data")

        # Control
        assert hasattr(mac_array, "clear_accum")
        assert hasattr(mac_array, "cfg_kernel_h")
        assert hasattr(mac_array, "cfg_kernel_w")

    def test_mac_array_parallel_output_width(self, config):
        """Test that output width matches P_c × acc_bits."""
        mac = ChannelParallelMAC(config)
        expected_width = config.parallel_channels * config.acc_bits
        assert mac.out_width == expected_width

    def test_mac_array_filter_bank_routing(self, config):
        """Test that filters can be loaded to specific banks."""
        mac = ChannelParallelMAC(config)

        def testbench():
            # Load filter to bank 0
            yield mac.filter_bank.eq(0)
            yield mac.filter_data.eq(0x0101010101010101)  # 8 bytes
            yield mac.filter_load.eq(1)
            yield Tick()

            # Load filter to bank 1
            yield mac.filter_bank.eq(1)
            yield mac.filter_data.eq(0x0202020202020202)
            yield Tick()

            yield mac.filter_load.eq(0)
            yield Tick()

        sim = Simulator(mac)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()

    def test_mac_array_elaboration(self, mac_array):
        """Test that ChannelParallelMAC elaborates without errors."""

        def testbench():
            yield mac_array.cfg_kernel_h.eq(3)
            yield mac_array.cfg_kernel_w.eq(3)
            yield Tick()

        sim = Simulator(mac_array)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()


# =============================================================================
# StencilMachine Tests
# =============================================================================


class TestStencilMachine:
    """Test suite for StencilMachine."""

    @pytest.fixture
    def config(self):
        """Small configuration for tests."""
        return StencilConfig(
            max_width=8,
            max_height=8,
            max_kernel_h=3,
            max_kernel_w=3,
            parallel_channels=4,
        )

    @pytest.fixture
    def stencil(self, config):
        """Create a StencilMachine instance."""
        return StencilMachine(config)

    def test_stencil_instantiation(self, stencil, config):
        """Test that StencilMachine can be instantiated."""
        assert stencil is not None
        assert stencil.config == config

    def test_stencil_has_correct_ports(self, stencil):
        """Test that StencilMachine has all required ports."""
        # Input stream
        assert hasattr(stencil, "in_valid")
        assert hasattr(stencil, "in_ready")
        assert hasattr(stencil, "in_data")
        assert hasattr(stencil, "in_last_col")
        assert hasattr(stencil, "in_last_row")

        # Filter stream
        assert hasattr(stencil, "filter_valid")
        assert hasattr(stencil, "filter_ready")
        assert hasattr(stencil, "filter_data")
        assert hasattr(stencil, "filter_bank")

        # Output stream
        assert hasattr(stencil, "out_valid")
        assert hasattr(stencil, "out_ready")
        assert hasattr(stencil, "out_data")

        # Control
        assert hasattr(stencil, "start")
        assert hasattr(stencil, "done")

        # Configuration
        assert hasattr(stencil, "cfg_in_height")
        assert hasattr(stencil, "cfg_in_width")
        assert hasattr(stencil, "cfg_in_channels")
        assert hasattr(stencil, "cfg_out_channels")
        assert hasattr(stencil, "cfg_kernel_h")
        assert hasattr(stencil, "cfg_kernel_w")
        assert hasattr(stencil, "cfg_stride_h")
        assert hasattr(stencil, "cfg_stride_w")
        assert hasattr(stencil, "cfg_activation")

        # Status
        assert hasattr(stencil, "state")
        assert hasattr(stencil, "rows_processed")
        assert hasattr(stencil, "channels_processed")

    def test_stencil_idle_state(self, config):
        """Test that stencil starts in IDLE state."""
        stencil = StencilMachine(config)
        result = {"state": None, "done": None}

        def testbench():
            yield Tick()
            result["state"] = yield stencil.state
            result["done"] = yield stencil.done

        sim = Simulator(stencil)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()

        assert result["state"] == StencilState.IDLE
        assert result["done"] == 0

    def test_stencil_start_transition(self, config):
        """Test that stencil transitions from IDLE on start."""
        stencil = StencilMachine(config)
        states = []

        def testbench():
            # Configure
            yield stencil.cfg_in_height.eq(8)
            yield stencil.cfg_in_width.eq(8)
            yield stencil.cfg_in_channels.eq(4)
            yield stencil.cfg_out_channels.eq(4)
            yield stencil.cfg_kernel_h.eq(3)
            yield stencil.cfg_kernel_w.eq(3)
            yield stencil.cfg_stride_h.eq(1)
            yield stencil.cfg_stride_w.eq(1)
            yield stencil.out_ready.eq(1)
            yield Tick()

            states.append((yield stencil.state))

            # Start
            yield stencil.start.eq(1)
            yield Tick()

            yield stencil.start.eq(0)
            states.append((yield stencil.state))
            yield Tick()

            states.append((yield stencil.state))

        sim = Simulator(stencil)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()

        # Should transition from IDLE to FILL_BUFFER
        assert states[0] == StencilState.IDLE
        assert states[1] == StencilState.FILL_BUFFER

    def test_stencil_elaboration(self, stencil):
        """Test that StencilMachine elaborates without errors."""

        def testbench():
            yield stencil.cfg_kernel_h.eq(3)
            yield stencil.cfg_kernel_w.eq(3)
            yield Tick()

        sim = Simulator(stencil)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()


# =============================================================================
# Verilog Generation Tests
# =============================================================================


class TestStencilVerilogGeneration:
    """Test Verilog generation for Stencil Machine modules."""

    @pytest.fixture
    def config(self):
        """Small configuration for Verilog tests."""
        return StencilConfig(
            max_width=16,
            max_height=16,
            max_kernel_h=3,
            max_kernel_w=3,
            parallel_channels=4,
        )

    def test_generate_line_buffer_bank_verilog(self, config, tmp_path):
        """Test that LineBufferBank can generate valid Verilog."""
        from amaranth._toolchain.yosys import find_yosys
        from amaranth.back import verilog

        try:
            find_yosys(lambda ver: ver >= (0, 40))
        except Exception:
            pytest.skip("Yosys not found")

        bank = LineBufferBank(config, bank_id=0)
        output = verilog.convert(bank, name="LineBufferBank")

        assert "module LineBufferBank" in output
        assert "read_addr" in output
        assert "write_data" in output

        verilog_file = tmp_path / "line_buffer_bank.v"
        verilog_file.write_text(output)
        assert verilog_file.exists()

    def test_generate_line_buffer_unit_verilog(self, config, tmp_path):
        """Test that LineBufferUnit can generate valid Verilog."""
        from amaranth._toolchain.yosys import find_yosys
        from amaranth.back import verilog

        try:
            find_yosys(lambda ver: ver >= (0, 40))
        except Exception:
            pytest.skip("Yosys not found")

        unit = LineBufferUnit(config)
        output = verilog.convert(unit, name="LineBufferUnit")

        assert "module LineBufferUnit" in output
        assert "in_valid" in output
        assert "out_data" in output

        verilog_file = tmp_path / "line_buffer_unit.v"
        verilog_file.write_text(output)
        assert verilog_file.exists()

    def test_generate_window_former_verilog(self, config, tmp_path):
        """Test that WindowFormer can generate valid Verilog."""
        from amaranth._toolchain.yosys import find_yosys
        from amaranth.back import verilog

        try:
            find_yosys(lambda ver: ver >= (0, 40))
        except Exception:
            pytest.skip("Yosys not found")

        wf = WindowFormer(config)
        output = verilog.convert(wf, name="WindowFormer")

        assert "module WindowFormer" in output
        assert "out_window" in output

        verilog_file = tmp_path / "window_former.v"
        verilog_file.write_text(output)
        assert verilog_file.exists()

    def test_generate_mac_bank_verilog(self, config, tmp_path):
        """Test that MACBank can generate valid Verilog."""
        from amaranth._toolchain.yosys import find_yosys
        from amaranth.back import verilog

        try:
            find_yosys(lambda ver: ver >= (0, 40))
        except Exception:
            pytest.skip("Yosys not found")

        bank = MACBank(config, bank_id=0)
        output = verilog.convert(bank, name="MACBank")

        assert "module MACBank" in output
        assert "in_window" in output
        assert "out_sum" in output

        verilog_file = tmp_path / "mac_bank.v"
        verilog_file.write_text(output)
        assert verilog_file.exists()

    def test_generate_channel_parallel_mac_verilog(self, config, tmp_path):
        """Test that ChannelParallelMAC can generate valid Verilog."""
        from amaranth._toolchain.yosys import find_yosys
        from amaranth.back import verilog

        try:
            find_yosys(lambda ver: ver >= (0, 40))
        except Exception:
            pytest.skip("Yosys not found")

        mac = ChannelParallelMAC(config)
        output = verilog.convert(mac, name="ChannelParallelMAC")

        assert "module ChannelParallelMAC" in output
        assert "mac_bank_0" in output

        verilog_file = tmp_path / "channel_parallel_mac.v"
        verilog_file.write_text(output)
        assert verilog_file.exists()

    def test_generate_stencil_machine_verilog(self, config, tmp_path):
        """Test that StencilMachine can generate valid Verilog."""
        from amaranth._toolchain.yosys import find_yosys
        from amaranth.back import verilog

        try:
            find_yosys(lambda ver: ver >= (0, 40))
        except Exception:
            pytest.skip("Yosys not found")

        stencil = StencilMachine(config)
        output = verilog.convert(stencil, name="StencilMachine")

        assert "module StencilMachine" in output
        assert "line_buffer" in output
        assert "window_former" in output
        assert "mac_array" in output

        verilog_file = tmp_path / "stencil_machine.v"
        verilog_file.write_text(output)
        assert verilog_file.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
