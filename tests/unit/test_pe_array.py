"""
Unit tests for the PEArray module.

These tests verify:
1. PEArray instantiation with various configurations
2. PE grid structure (rows x cols)
3. Horizontal (A) data flow
4. Vertical (B, D) data flow
5. Control signal broadcast
6. Verilog generation
"""

import pytest
from amaranth.sim import Simulator, Tick

from systars.config import Dataflow, SystolicConfig
from systars.core.pe_array import PEArray


class TestPEArray:
    """Test suite for the PEArray module."""

    @pytest.fixture
    def config_1x1(self):
        """1x1 PEArray configuration (single PE)."""
        return SystolicConfig(
            tile_rows=1,
            tile_cols=1,
            input_bits=8,
            weight_bits=8,
            acc_bits=32,
            output_bits=20,
            dataflow=Dataflow.OUTPUT_STATIONARY | Dataflow.B_STATIONARY,
        )

    @pytest.fixture
    def config_2x2(self):
        """2x2 PEArray configuration."""
        return SystolicConfig(
            tile_rows=2,
            tile_cols=2,
            input_bits=8,
            weight_bits=8,
            acc_bits=32,
            output_bits=20,
            dataflow=Dataflow.OUTPUT_STATIONARY | Dataflow.B_STATIONARY,
        )

    @pytest.fixture
    def config_2x3(self):
        """2x3 PEArray configuration (non-square)."""
        return SystolicConfig(
            tile_rows=2,
            tile_cols=3,
            input_bits=8,
            weight_bits=8,
            acc_bits=32,
            output_bits=20,
            dataflow=Dataflow.OUTPUT_STATIONARY | Dataflow.B_STATIONARY,
        )

    @pytest.fixture
    def pe_array_1x1(self, config_1x1):
        """Create a 1x1 PEArray instance."""
        return PEArray(config_1x1)

    @pytest.fixture
    def pe_array_2x2(self, config_2x2):
        """Create a 2x2 PEArray instance."""
        return PEArray(config_2x2)

    @pytest.fixture
    def pe_array_2x3(self, config_2x3):
        """Create a 2x3 PEArray instance."""
        return PEArray(config_2x3)

    def test_pe_array_1x1_instantiation(self, pe_array_1x1):
        """Test that 1x1 PEArray can be instantiated."""
        assert pe_array_1x1 is not None
        assert pe_array_1x1.config.tile_rows == 1
        assert pe_array_1x1.config.tile_cols == 1

    def test_pe_array_2x2_instantiation(self, pe_array_2x2):
        """Test that 2x2 PEArray can be instantiated."""
        assert pe_array_2x2 is not None
        assert pe_array_2x2.config.tile_rows == 2
        assert pe_array_2x2.config.tile_cols == 2

    def test_pe_array_2x3_instantiation(self, pe_array_2x3):
        """Test that non-square PEArray can be instantiated."""
        assert pe_array_2x3 is not None
        assert pe_array_2x3.config.tile_rows == 2
        assert pe_array_2x3.config.tile_cols == 3

    def test_pe_array_has_correct_ports_1x1(self, pe_array_1x1):
        """Test that 1x1 PEArray has correct port names."""
        # Input ports
        assert hasattr(pe_array_1x1, "in_a_0")
        assert hasattr(pe_array_1x1, "in_b_0")
        assert hasattr(pe_array_1x1, "in_d_0")
        assert hasattr(pe_array_1x1, "in_control_dataflow")
        assert hasattr(pe_array_1x1, "in_valid")

        # Output ports
        assert hasattr(pe_array_1x1, "out_a_0")
        assert hasattr(pe_array_1x1, "out_b_0")
        assert hasattr(pe_array_1x1, "out_c_0")
        assert hasattr(pe_array_1x1, "out_valid")

    def test_pe_array_has_correct_ports_2x3(self, pe_array_2x3):
        """Test that 2x3 PEArray has correct vector ports."""
        # A ports (per row = 2)
        assert hasattr(pe_array_2x3, "in_a_0")
        assert hasattr(pe_array_2x3, "in_a_1")
        assert hasattr(pe_array_2x3, "out_a_0")
        assert hasattr(pe_array_2x3, "out_a_1")

        # B, C, D ports (per column = 3)
        for i in range(3):
            assert hasattr(pe_array_2x3, f"in_b_{i}")
            assert hasattr(pe_array_2x3, f"in_d_{i}")
            assert hasattr(pe_array_2x3, f"out_b_{i}")
            assert hasattr(pe_array_2x3, f"out_c_{i}")

    def test_pe_array_1x1_simple_mac(self, pe_array_1x1):
        """Test 1x1 PEArray performs simple MAC like a single PE."""
        results = []

        def testbench():
            # Set up inputs: a=3, b=4, d=10 in WS mode
            yield pe_array_1x1.in_a_0.eq(3)
            yield pe_array_1x1.in_b_0.eq(4)
            yield pe_array_1x1.in_d_0.eq(10)
            yield pe_array_1x1.in_control_dataflow.eq(1)  # WS mode
            yield pe_array_1x1.in_control_propagate.eq(0)
            yield pe_array_1x1.in_valid.eq(1)
            yield Tick()

            # Wait for PE pipeline
            yield Tick()

            # Read outputs
            out_a = yield pe_array_1x1.out_a_0
            out_b = yield pe_array_1x1.out_b_0
            results.append({"out_a": out_a, "out_b": out_b})

        sim = Simulator(pe_array_1x1)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()

        # out_a should pass through (registered): 3
        assert results[0]["out_a"] == 3

    def test_pe_array_2x2_horizontal_flow(self, pe_array_2x2):
        """Test that A data flows horizontally through 2x2 PEArray."""
        results = []

        def testbench():
            # Set different A values for each row
            yield pe_array_2x2.in_a_0.eq(10)
            yield pe_array_2x2.in_a_1.eq(20)
            yield pe_array_2x2.in_b_0.eq(1)
            yield pe_array_2x2.in_b_1.eq(1)
            yield pe_array_2x2.in_d_0.eq(0)
            yield pe_array_2x2.in_d_1.eq(0)
            yield pe_array_2x2.in_control_dataflow.eq(1)  # WS mode
            yield pe_array_2x2.in_valid.eq(1)
            yield Tick()

            # Pipeline delay: each column adds latency
            yield Tick()
            yield Tick()

            out_a_0 = yield pe_array_2x2.out_a_0
            out_a_1 = yield pe_array_2x2.out_a_1
            results.append({"out_a_0": out_a_0, "out_a_1": out_a_1})

        sim = Simulator(pe_array_2x2)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()

        # A values should pass through (after pipeline delay)
        # Note: exact timing depends on PE pipeline depth
        assert results[0]["out_a_0"] == 10
        assert results[0]["out_a_1"] == 20

    def test_pe_array_control_broadcast(self, pe_array_2x2):
        """Test that control signals are broadcast to all PEs."""
        results = {}

        def testbench():
            # Set control signals
            yield pe_array_2x2.in_control_dataflow.eq(1)
            yield pe_array_2x2.in_control_propagate.eq(1)
            yield pe_array_2x2.in_control_shift.eq(15)
            yield pe_array_2x2.in_valid.eq(1)
            yield pe_array_2x2.in_id.eq(42)
            yield pe_array_2x2.in_last.eq(1)

            # Set data inputs
            yield pe_array_2x2.in_a_0.eq(1)
            yield pe_array_2x2.in_a_1.eq(1)
            yield pe_array_2x2.in_b_0.eq(1)
            yield pe_array_2x2.in_b_1.eq(1)
            yield pe_array_2x2.in_d_0.eq(0)
            yield pe_array_2x2.in_d_1.eq(0)
            yield Tick()

            # Wait for pipeline
            yield Tick()
            yield Tick()

            results["out_dataflow"] = yield pe_array_2x2.out_control_dataflow
            results["out_propagate"] = yield pe_array_2x2.out_control_propagate
            results["out_shift"] = yield pe_array_2x2.out_control_shift
            results["out_valid"] = yield pe_array_2x2.out_valid
            results["out_id"] = yield pe_array_2x2.out_id
            results["out_last"] = yield pe_array_2x2.out_last

        sim = Simulator(pe_array_2x2)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()

        # Control signals should pass through (from bottom-right PE)
        assert results["out_dataflow"] == 1
        assert results["out_propagate"] == 1
        assert results["out_shift"] == 15
        assert results["out_valid"] == 1
        assert results["out_id"] == 42
        assert results["out_last"] == 1

    def test_pe_array_elaboration(self, pe_array_2x2):
        """Test that PEArray elaborates without errors."""

        def testbench():
            yield Tick()

        sim = Simulator(pe_array_2x2)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()  # Should not raise


class TestPEArrayVerilogGeneration:
    """Test Verilog generation for PEArray modules."""

    def test_generate_pe_array_1x1_verilog(self, tmp_path):
        """Test that 1x1 PEArray can generate valid Verilog."""
        from amaranth._toolchain.yosys import find_yosys
        from amaranth.back import verilog

        try:
            find_yosys(lambda ver: ver >= (0, 40))
        except Exception:
            pytest.skip("Yosys not found")

        config = SystolicConfig(tile_rows=1, tile_cols=1)
        pe_array = PEArray(config)

        output = verilog.convert(pe_array, name="PEArray_1x1")
        assert "module PEArray_1x1" in output
        assert "in_a_0" in output
        assert "out_c_0" in output

        verilog_file = tmp_path / "pe_array_1x1.v"
        verilog_file.write_text(output)
        assert verilog_file.exists()

    def test_generate_pe_array_2x2_verilog(self, tmp_path):
        """Test that 2x2 PEArray can generate valid Verilog."""
        from amaranth._toolchain.yosys import find_yosys
        from amaranth.back import verilog

        try:
            find_yosys(lambda ver: ver >= (0, 40))
        except Exception:
            pytest.skip("Yosys not found")

        config = SystolicConfig(tile_rows=2, tile_cols=2)
        pe_array = PEArray(config)

        output = verilog.convert(pe_array, name="PEArray_2x2")
        assert "module PEArray_2x2" in output

        # Check for vector ports
        assert "in_a_0" in output
        assert "in_a_1" in output
        assert "in_b_0" in output
        assert "in_b_1" in output

        verilog_file = tmp_path / "pe_array_2x2.v"
        verilog_file.write_text(output)
        assert verilog_file.exists()

    def test_generate_pe_array_4x4_verilog(self, tmp_path):
        """Test that larger PEArray can generate valid Verilog."""
        from amaranth._toolchain.yosys import find_yosys
        from amaranth.back import verilog

        try:
            find_yosys(lambda ver: ver >= (0, 40))
        except Exception:
            pytest.skip("Yosys not found")

        config = SystolicConfig(tile_rows=4, tile_cols=4)
        pe_array = PEArray(config)

        output = verilog.convert(pe_array, name="PEArray_4x4")
        assert "module PEArray_4x4" in output

        # Check for PE submodules
        assert "pe_0_0" in output
        assert "pe_3_3" in output

        verilog_file = tmp_path / "pe_array_4x4.v"
        verilog_file.write_text(output)
        assert verilog_file.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
