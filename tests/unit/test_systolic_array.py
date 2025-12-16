"""
Unit tests for the SystolicArray module.

These tests verify:
1. SystolicArray instantiation with various configurations
2. PEArray grid structure (grid_rows x grid_cols)
3. Pipeline registers between PEArray boundaries
4. Control signal synchronization with data flow
5. Verilog generation
"""

import pytest
from amaranth.sim import Simulator, Tick

from systars.config import Dataflow, SystolicConfig
from systars.core.systolic_array import SystolicArray


class TestSystolicArray:
    """Test suite for the SystolicArray module."""

    @pytest.fixture
    def config_1x1_array_1x1_pe_array(self):
        """1x1 SystolicArray with 1x1 PEArrays (single PE total)."""
        return SystolicConfig(
            grid_rows=1,
            grid_cols=1,
            tile_rows=1,
            tile_cols=1,
            input_bits=8,
            weight_bits=8,
            acc_bits=32,
            output_bits=20,
            dataflow=Dataflow.OUTPUT_STATIONARY | Dataflow.B_STATIONARY,
        )

    @pytest.fixture
    def config_2x2_array_1x1_pe_array(self):
        """2x2 SystolicArray with 1x1 PEArrays (4 PEs total, tests inter-array pipelining)."""
        return SystolicConfig(
            grid_rows=2,
            grid_cols=2,
            tile_rows=1,
            tile_cols=1,
            input_bits=8,
            weight_bits=8,
            acc_bits=32,
            output_bits=20,
            dataflow=Dataflow.OUTPUT_STATIONARY | Dataflow.B_STATIONARY,
        )

    @pytest.fixture
    def config_2x2_array_2x2_pe_array(self):
        """2x2 SystolicArray with 2x2 PEArrays (16 PEs total)."""
        return SystolicConfig(
            grid_rows=2,
            grid_cols=2,
            tile_rows=2,
            tile_cols=2,
            input_bits=8,
            weight_bits=8,
            acc_bits=32,
            output_bits=20,
            dataflow=Dataflow.OUTPUT_STATIONARY | Dataflow.B_STATIONARY,
        )

    @pytest.fixture
    def config_2x3_array_1x1_pe_array(self):
        """2x3 SystolicArray (non-square) with 1x1 PEArrays."""
        return SystolicConfig(
            grid_rows=2,
            grid_cols=3,
            tile_rows=1,
            tile_cols=1,
            input_bits=8,
            weight_bits=8,
            acc_bits=32,
            output_bits=20,
            dataflow=Dataflow.OUTPUT_STATIONARY | Dataflow.B_STATIONARY,
        )

    @pytest.fixture
    def array_1x1(self, config_1x1_array_1x1_pe_array):
        """Create a minimal SystolicArray instance (single PEArray)."""
        return SystolicArray(config_1x1_array_1x1_pe_array)

    @pytest.fixture
    def array_2x2(self, config_2x2_array_1x1_pe_array):
        """Create a 2x2 SystolicArray instance with 1x1 PEArrays."""
        return SystolicArray(config_2x2_array_1x1_pe_array)

    @pytest.fixture
    def array_2x2_pe_arrays(self, config_2x2_array_2x2_pe_array):
        """Create a 2x2 SystolicArray instance with 2x2 PEArrays."""
        return SystolicArray(config_2x2_array_2x2_pe_array)

    @pytest.fixture
    def array_2x3(self, config_2x3_array_1x1_pe_array):
        """Create a 2x3 SystolicArray instance."""
        return SystolicArray(config_2x3_array_1x1_pe_array)

    def test_array_1x1_instantiation(self, array_1x1):
        """Test that minimal SystolicArray (1x1 PEArrays) can be instantiated."""
        assert array_1x1 is not None
        assert array_1x1.config.grid_rows == 1
        assert array_1x1.config.grid_cols == 1
        assert array_1x1.config.tile_rows == 1
        assert array_1x1.config.tile_cols == 1

    def test_array_2x2_instantiation(self, array_2x2):
        """Test that 2x2 SystolicArray can be instantiated."""
        assert array_2x2 is not None
        assert array_2x2.config.grid_rows == 2
        assert array_2x2.config.grid_cols == 2

    def test_array_2x2_pe_arrays_instantiation(self, array_2x2_pe_arrays):
        """Test that SystolicArray with multi-PE PEArrays can be instantiated."""
        assert array_2x2_pe_arrays is not None
        assert array_2x2_pe_arrays.config.grid_rows == 2
        assert array_2x2_pe_arrays.config.grid_cols == 2
        assert array_2x2_pe_arrays.config.tile_rows == 2
        assert array_2x2_pe_arrays.config.tile_cols == 2

    def test_array_2x3_instantiation(self, array_2x3):
        """Test that non-square SystolicArray can be instantiated."""
        assert array_2x3 is not None
        assert array_2x3.config.grid_rows == 2
        assert array_2x3.config.grid_cols == 3

    def test_array_has_correct_ports_1x1(self, array_1x1):
        """Test that minimal SystolicArray has correct port names."""
        # Input ports (single PE array)
        assert hasattr(array_1x1, "in_a_0")
        assert hasattr(array_1x1, "in_b_0")
        assert hasattr(array_1x1, "in_d_0")
        assert hasattr(array_1x1, "in_control_dataflow")
        assert hasattr(array_1x1, "in_valid")

        # Output ports
        assert hasattr(array_1x1, "out_a_0")
        assert hasattr(array_1x1, "out_b_0")
        assert hasattr(array_1x1, "out_c_0")
        assert hasattr(array_1x1, "out_valid")

    def test_array_has_correct_ports_2x2(self, array_2x2):
        """Test that 2x2 SystolicArray (1x1 PEArrays) has correct vector ports."""
        # A ports (per total row = 2)
        assert hasattr(array_2x2, "in_a_0")
        assert hasattr(array_2x2, "in_a_1")
        assert hasattr(array_2x2, "out_a_0")
        assert hasattr(array_2x2, "out_a_1")

        # B, C, D ports (per total column = 2)
        for i in range(2):
            assert hasattr(array_2x2, f"in_b_{i}")
            assert hasattr(array_2x2, f"in_d_{i}")
            assert hasattr(array_2x2, f"out_b_{i}")
            assert hasattr(array_2x2, f"out_c_{i}")

    def test_array_has_correct_ports_2x2_pe_arrays(self, array_2x2_pe_arrays):
        """Test that 2x2 SystolicArray with 2x2 PEArrays has correct port count."""
        # Total rows = grid_rows * tile_rows = 2 * 2 = 4
        # Total cols = grid_cols * tile_cols = 2 * 2 = 4
        for i in range(4):
            assert hasattr(array_2x2_pe_arrays, f"in_a_{i}")
            assert hasattr(array_2x2_pe_arrays, f"out_a_{i}")
            assert hasattr(array_2x2_pe_arrays, f"in_b_{i}")
            assert hasattr(array_2x2_pe_arrays, f"in_d_{i}")
            assert hasattr(array_2x2_pe_arrays, f"out_b_{i}")
            assert hasattr(array_2x2_pe_arrays, f"out_c_{i}")

    def test_array_1x1_simple_mac(self, array_1x1):
        """Test minimal SystolicArray performs simple MAC like a single PE."""
        results = []

        def testbench():
            # Set up inputs: a=3, b=4, d=10 in WS mode
            yield array_1x1.in_a_0.eq(3)
            yield array_1x1.in_b_0.eq(4)
            yield array_1x1.in_d_0.eq(10)
            yield array_1x1.in_control_dataflow.eq(1)  # WS mode
            yield array_1x1.in_control_propagate.eq(0)
            yield array_1x1.in_valid.eq(1)
            yield Tick()

            # Wait for PE pipeline
            yield Tick()

            # Read outputs
            out_a = yield array_1x1.out_a_0
            out_b = yield array_1x1.out_b_0
            results.append({"out_a": out_a, "out_b": out_b})

        sim = Simulator(array_1x1)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()

        # out_a should pass through (registered): 3
        assert results[0]["out_a"] == 3

    def test_array_2x2_pipeline_delay(self, array_2x2):
        """Test that 2x2 SystolicArray has pipeline delay between PEArrays."""
        results = []

        def testbench():
            # Set A values for each row
            yield array_2x2.in_a_0.eq(10)
            yield array_2x2.in_a_1.eq(20)

            # Set B, D values for columns
            yield array_2x2.in_b_0.eq(1)
            yield array_2x2.in_b_1.eq(1)
            yield array_2x2.in_d_0.eq(0)
            yield array_2x2.in_d_1.eq(0)

            yield array_2x2.in_control_dataflow.eq(1)  # WS mode
            yield array_2x2.in_valid.eq(1)
            yield Tick()

            # Record outputs over several cycles to observe pipeline
            for _ in range(5):
                yield Tick()
                out_a_0 = yield array_2x2.out_a_0
                out_a_1 = yield array_2x2.out_a_1
                results.append({"out_a_0": out_a_0, "out_a_1": out_a_1})

        sim = Simulator(array_2x2)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()

        # Due to pipeline registers, data takes multiple cycles to reach output
        # Eventually we should see the input values propagate through
        final = results[-1]
        assert final["out_a_0"] == 10
        assert final["out_a_1"] == 20

    def test_array_control_signal_propagation(self, array_2x2):
        """Test that control signals propagate through SystolicArray."""
        results = {}

        def testbench():
            # Set control signals
            yield array_2x2.in_control_dataflow.eq(1)
            yield array_2x2.in_control_propagate.eq(1)
            yield array_2x2.in_control_shift.eq(15)
            yield array_2x2.in_valid.eq(1)
            yield array_2x2.in_id.eq(42)
            yield array_2x2.in_last.eq(1)

            # Set minimal data inputs
            yield array_2x2.in_a_0.eq(1)
            yield array_2x2.in_a_1.eq(1)
            yield array_2x2.in_b_0.eq(1)
            yield array_2x2.in_b_1.eq(1)
            yield array_2x2.in_d_0.eq(0)
            yield array_2x2.in_d_1.eq(0)

            # Wait for pipeline (control propagates with data)
            for _ in range(5):
                yield Tick()

            results["out_dataflow"] = yield array_2x2.out_control_dataflow
            results["out_propagate"] = yield array_2x2.out_control_propagate
            results["out_shift"] = yield array_2x2.out_control_shift
            results["out_valid"] = yield array_2x2.out_valid
            results["out_id"] = yield array_2x2.out_id
            results["out_last"] = yield array_2x2.out_last

        sim = Simulator(array_2x2)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()

        # Control signals should propagate through array
        assert results["out_dataflow"] == 1
        assert results["out_propagate"] == 1
        assert results["out_shift"] == 15
        assert results["out_valid"] == 1
        assert results["out_id"] == 42
        assert results["out_last"] == 1

    def test_array_elaboration(self, array_2x2):
        """Test that SystolicArray elaborates without errors."""

        def testbench():
            yield Tick()

        sim = Simulator(array_2x2)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()  # Should not raise

    def test_array_2x2_pe_arrays_elaboration(self, array_2x2_pe_arrays):
        """Test that SystolicArray with multi-PE PEArrays elaborates."""

        def testbench():
            yield Tick()

        sim = Simulator(array_2x2_pe_arrays)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()  # Should not raise


class TestSystolicArrayVerilogGeneration:
    """Test Verilog generation for SystolicArray modules."""

    def test_generate_array_1x1_verilog(self, tmp_path):
        """Test that minimal SystolicArray can generate valid Verilog."""
        from amaranth._toolchain.yosys import find_yosys
        from amaranth.back import verilog

        try:
            find_yosys(lambda ver: ver >= (0, 40))
        except Exception:
            pytest.skip("Yosys not found")

        config = SystolicConfig(grid_rows=1, grid_cols=1, tile_rows=1, tile_cols=1)
        array = SystolicArray(config)

        output = verilog.convert(array, name="SystolicArray_1x1")
        assert "module SystolicArray_1x1" in output
        assert "in_a_0" in output
        assert "out_c_0" in output

        verilog_file = tmp_path / "systolic_array_1x1.v"
        verilog_file.write_text(output)
        assert verilog_file.exists()

    def test_generate_array_2x2_verilog(self, tmp_path):
        """Test that 2x2 SystolicArray can generate valid Verilog."""
        from amaranth._toolchain.yosys import find_yosys
        from amaranth.back import verilog

        try:
            find_yosys(lambda ver: ver >= (0, 40))
        except Exception:
            pytest.skip("Yosys not found")

        config = SystolicConfig(grid_rows=2, grid_cols=2, tile_rows=1, tile_cols=1)
        array = SystolicArray(config)

        output = verilog.convert(array, name="SystolicArray_2x2")
        assert "module SystolicArray_2x2" in output

        # Check for vector ports
        assert "in_a_0" in output
        assert "in_a_1" in output
        assert "in_b_0" in output
        assert "in_b_1" in output

        # Check for pe_array submodules
        assert "pe_array_0_0" in output
        assert "pe_array_1_1" in output

        verilog_file = tmp_path / "systolic_array_2x2.v"
        verilog_file.write_text(output)
        assert verilog_file.exists()

    def test_generate_array_4x4_verilog(self, tmp_path):
        """Test that larger SystolicArray can generate valid Verilog."""
        from amaranth._toolchain.yosys import find_yosys
        from amaranth.back import verilog

        try:
            find_yosys(lambda ver: ver >= (0, 40))
        except Exception:
            pytest.skip("Yosys not found")

        config = SystolicConfig(grid_rows=4, grid_cols=4, tile_rows=1, tile_cols=1)
        array = SystolicArray(config)

        output = verilog.convert(array, name="SystolicArray_4x4")
        assert "module SystolicArray_4x4" in output

        # Check for pe_array submodules
        assert "pe_array_0_0" in output
        assert "pe_array_3_3" in output

        verilog_file = tmp_path / "systolic_array_4x4.v"
        verilog_file.write_text(output)
        assert verilog_file.exists()

    def test_generate_array_2x2_with_2x2_pe_arrays_verilog(self, tmp_path):
        """Test SystolicArray with multi-PE PEArrays generates valid Verilog."""
        from amaranth._toolchain.yosys import find_yosys
        from amaranth.back import verilog

        try:
            find_yosys(lambda ver: ver >= (0, 40))
        except Exception:
            pytest.skip("Yosys not found")

        config = SystolicConfig(grid_rows=2, grid_cols=2, tile_rows=2, tile_cols=2)
        array = SystolicArray(config)

        output = verilog.convert(array, name="SystolicArray_2x2_pe_arrays_2x2")
        assert "module SystolicArray_2x2_pe_arrays_2x2" in output

        # Should have 4 total rows/cols of ports (2 array * 2 PEArray)
        assert "in_a_3" in output
        assert "in_b_3" in output
        assert "out_c_3" in output

        verilog_file = tmp_path / "systolic_array_2x2_pe_arrays_2x2.v"
        verilog_file.write_text(output)
        assert verilog_file.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
