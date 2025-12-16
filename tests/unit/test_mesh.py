"""
Unit tests for the Mesh module.

These tests verify:
1. Mesh instantiation with various configurations
2. Tile grid structure (mesh_rows x mesh_cols)
3. Pipeline registers between tile boundaries
4. Control signal synchronization with data flow
5. Verilog generation
"""

import pytest
from amaranth.sim import Simulator, Tick

from systars.config import Dataflow, SystolicConfig
from systars.core.mesh import Mesh


class TestMesh:
    """Test suite for the Mesh module."""

    @pytest.fixture
    def config_1x1_mesh_1x1_tile(self):
        """1x1 mesh with 1x1 tiles (single PE total)."""
        return SystolicConfig(
            mesh_rows=1,
            mesh_cols=1,
            tile_rows=1,
            tile_cols=1,
            input_bits=8,
            weight_bits=8,
            acc_bits=32,
            output_bits=20,
            dataflow=Dataflow.OW,
        )

    @pytest.fixture
    def config_2x2_mesh_1x1_tile(self):
        """2x2 mesh with 1x1 tiles (4 PEs total, tests inter-tile pipelining)."""
        return SystolicConfig(
            mesh_rows=2,
            mesh_cols=2,
            tile_rows=1,
            tile_cols=1,
            input_bits=8,
            weight_bits=8,
            acc_bits=32,
            output_bits=20,
            dataflow=Dataflow.OW,
        )

    @pytest.fixture
    def config_2x2_mesh_2x2_tile(self):
        """2x2 mesh with 2x2 tiles (16 PEs total)."""
        return SystolicConfig(
            mesh_rows=2,
            mesh_cols=2,
            tile_rows=2,
            tile_cols=2,
            input_bits=8,
            weight_bits=8,
            acc_bits=32,
            output_bits=20,
            dataflow=Dataflow.OW,
        )

    @pytest.fixture
    def config_2x3_mesh_1x1_tile(self):
        """2x3 mesh (non-square) with 1x1 tiles."""
        return SystolicConfig(
            mesh_rows=2,
            mesh_cols=3,
            tile_rows=1,
            tile_cols=1,
            input_bits=8,
            weight_bits=8,
            acc_bits=32,
            output_bits=20,
            dataflow=Dataflow.OW,
        )

    @pytest.fixture
    def mesh_1x1(self, config_1x1_mesh_1x1_tile):
        """Create a minimal Mesh instance (single tile)."""
        return Mesh(config_1x1_mesh_1x1_tile)

    @pytest.fixture
    def mesh_2x2(self, config_2x2_mesh_1x1_tile):
        """Create a 2x2 Mesh instance with 1x1 tiles."""
        return Mesh(config_2x2_mesh_1x1_tile)

    @pytest.fixture
    def mesh_2x2_tiles(self, config_2x2_mesh_2x2_tile):
        """Create a 2x2 Mesh instance with 2x2 tiles."""
        return Mesh(config_2x2_mesh_2x2_tile)

    @pytest.fixture
    def mesh_2x3(self, config_2x3_mesh_1x1_tile):
        """Create a 2x3 Mesh instance."""
        return Mesh(config_2x3_mesh_1x1_tile)

    def test_mesh_1x1_instantiation(self, mesh_1x1):
        """Test that minimal Mesh (1x1 tiles) can be instantiated."""
        assert mesh_1x1 is not None
        assert mesh_1x1.config.mesh_rows == 1
        assert mesh_1x1.config.mesh_cols == 1
        assert mesh_1x1.config.tile_rows == 1
        assert mesh_1x1.config.tile_cols == 1

    def test_mesh_2x2_instantiation(self, mesh_2x2):
        """Test that 2x2 Mesh can be instantiated."""
        assert mesh_2x2 is not None
        assert mesh_2x2.config.mesh_rows == 2
        assert mesh_2x2.config.mesh_cols == 2

    def test_mesh_2x2_tiles_instantiation(self, mesh_2x2_tiles):
        """Test that Mesh with multi-PE tiles can be instantiated."""
        assert mesh_2x2_tiles is not None
        assert mesh_2x2_tiles.config.mesh_rows == 2
        assert mesh_2x2_tiles.config.mesh_cols == 2
        assert mesh_2x2_tiles.config.tile_rows == 2
        assert mesh_2x2_tiles.config.tile_cols == 2

    def test_mesh_2x3_instantiation(self, mesh_2x3):
        """Test that non-square Mesh can be instantiated."""
        assert mesh_2x3 is not None
        assert mesh_2x3.config.mesh_rows == 2
        assert mesh_2x3.config.mesh_cols == 3

    def test_mesh_has_correct_ports_1x1(self, mesh_1x1):
        """Test that minimal Mesh has correct port names."""
        # Input ports (single PE mesh)
        assert hasattr(mesh_1x1, "in_a_0")
        assert hasattr(mesh_1x1, "in_b_0")
        assert hasattr(mesh_1x1, "in_d_0")
        assert hasattr(mesh_1x1, "in_control_dataflow")
        assert hasattr(mesh_1x1, "in_valid")

        # Output ports
        assert hasattr(mesh_1x1, "out_a_0")
        assert hasattr(mesh_1x1, "out_b_0")
        assert hasattr(mesh_1x1, "out_c_0")
        assert hasattr(mesh_1x1, "out_valid")

    def test_mesh_has_correct_ports_2x2(self, mesh_2x2):
        """Test that 2x2 Mesh (1x1 tiles) has correct vector ports."""
        # A ports (per total row = 2)
        assert hasattr(mesh_2x2, "in_a_0")
        assert hasattr(mesh_2x2, "in_a_1")
        assert hasattr(mesh_2x2, "out_a_0")
        assert hasattr(mesh_2x2, "out_a_1")

        # B, C, D ports (per total column = 2)
        for i in range(2):
            assert hasattr(mesh_2x2, f"in_b_{i}")
            assert hasattr(mesh_2x2, f"in_d_{i}")
            assert hasattr(mesh_2x2, f"out_b_{i}")
            assert hasattr(mesh_2x2, f"out_c_{i}")

    def test_mesh_has_correct_ports_2x2_tiles(self, mesh_2x2_tiles):
        """Test that 2x2 Mesh with 2x2 tiles has correct port count."""
        # Total rows = mesh_rows * tile_rows = 2 * 2 = 4
        # Total cols = mesh_cols * tile_cols = 2 * 2 = 4
        for i in range(4):
            assert hasattr(mesh_2x2_tiles, f"in_a_{i}")
            assert hasattr(mesh_2x2_tiles, f"out_a_{i}")
            assert hasattr(mesh_2x2_tiles, f"in_b_{i}")
            assert hasattr(mesh_2x2_tiles, f"in_d_{i}")
            assert hasattr(mesh_2x2_tiles, f"out_b_{i}")
            assert hasattr(mesh_2x2_tiles, f"out_c_{i}")

    def test_mesh_1x1_simple_mac(self, mesh_1x1):
        """Test minimal Mesh performs simple MAC like a single PE."""
        results = []

        def testbench():
            # Set up inputs: a=3, b=4, d=10 in WS mode
            yield mesh_1x1.in_a_0.eq(3)
            yield mesh_1x1.in_b_0.eq(4)
            yield mesh_1x1.in_d_0.eq(10)
            yield mesh_1x1.in_control_dataflow.eq(1)  # WS mode
            yield mesh_1x1.in_control_propagate.eq(0)
            yield mesh_1x1.in_valid.eq(1)
            yield Tick()

            # Wait for PE pipeline
            yield Tick()

            # Read outputs
            out_a = yield mesh_1x1.out_a_0
            out_b = yield mesh_1x1.out_b_0
            results.append({"out_a": out_a, "out_b": out_b})

        sim = Simulator(mesh_1x1)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()

        # out_a should pass through (registered): 3
        assert results[0]["out_a"] == 3

    def test_mesh_2x2_pipeline_delay(self, mesh_2x2):
        """Test that 2x2 Mesh has pipeline delay between tiles."""
        results = []

        def testbench():
            # Set A values for each row
            yield mesh_2x2.in_a_0.eq(10)
            yield mesh_2x2.in_a_1.eq(20)

            # Set B, D values for columns
            yield mesh_2x2.in_b_0.eq(1)
            yield mesh_2x2.in_b_1.eq(1)
            yield mesh_2x2.in_d_0.eq(0)
            yield mesh_2x2.in_d_1.eq(0)

            yield mesh_2x2.in_control_dataflow.eq(1)  # WS mode
            yield mesh_2x2.in_valid.eq(1)
            yield Tick()

            # Record outputs over several cycles to observe pipeline
            for _ in range(5):
                yield Tick()
                out_a_0 = yield mesh_2x2.out_a_0
                out_a_1 = yield mesh_2x2.out_a_1
                results.append({"out_a_0": out_a_0, "out_a_1": out_a_1})

        sim = Simulator(mesh_2x2)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()

        # Due to pipeline registers, data takes multiple cycles to reach output
        # Eventually we should see the input values propagate through
        final = results[-1]
        assert final["out_a_0"] == 10
        assert final["out_a_1"] == 20

    def test_mesh_control_signal_propagation(self, mesh_2x2):
        """Test that control signals propagate through mesh."""
        results = {}

        def testbench():
            # Set control signals
            yield mesh_2x2.in_control_dataflow.eq(1)
            yield mesh_2x2.in_control_propagate.eq(1)
            yield mesh_2x2.in_control_shift.eq(15)
            yield mesh_2x2.in_valid.eq(1)
            yield mesh_2x2.in_id.eq(42)
            yield mesh_2x2.in_last.eq(1)

            # Set minimal data inputs
            yield mesh_2x2.in_a_0.eq(1)
            yield mesh_2x2.in_a_1.eq(1)
            yield mesh_2x2.in_b_0.eq(1)
            yield mesh_2x2.in_b_1.eq(1)
            yield mesh_2x2.in_d_0.eq(0)
            yield mesh_2x2.in_d_1.eq(0)

            # Wait for pipeline (control propagates with data)
            for _ in range(5):
                yield Tick()

            results["out_dataflow"] = yield mesh_2x2.out_control_dataflow
            results["out_propagate"] = yield mesh_2x2.out_control_propagate
            results["out_shift"] = yield mesh_2x2.out_control_shift
            results["out_valid"] = yield mesh_2x2.out_valid
            results["out_id"] = yield mesh_2x2.out_id
            results["out_last"] = yield mesh_2x2.out_last

        sim = Simulator(mesh_2x2)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()

        # Control signals should propagate through mesh
        assert results["out_dataflow"] == 1
        assert results["out_propagate"] == 1
        assert results["out_shift"] == 15
        assert results["out_valid"] == 1
        assert results["out_id"] == 42
        assert results["out_last"] == 1

    def test_mesh_elaboration(self, mesh_2x2):
        """Test that Mesh elaborates without errors."""

        def testbench():
            yield Tick()

        sim = Simulator(mesh_2x2)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()  # Should not raise

    def test_mesh_2x2_tiles_elaboration(self, mesh_2x2_tiles):
        """Test that Mesh with multi-PE tiles elaborates."""

        def testbench():
            yield Tick()

        sim = Simulator(mesh_2x2_tiles)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()  # Should not raise


class TestMeshVerilogGeneration:
    """Test Verilog generation for Mesh modules."""

    def test_generate_mesh_1x1_verilog(self, tmp_path):
        """Test that minimal Mesh can generate valid Verilog."""
        from amaranth._toolchain.yosys import find_yosys
        from amaranth.back import verilog

        try:
            find_yosys(lambda ver: ver >= (0, 40))
        except Exception:
            pytest.skip("Yosys not found")

        config = SystolicConfig(mesh_rows=1, mesh_cols=1, tile_rows=1, tile_cols=1)
        mesh = Mesh(config)

        output = verilog.convert(mesh, name="Mesh_1x1")
        assert "module Mesh_1x1" in output
        assert "in_a_0" in output
        assert "out_c_0" in output

        verilog_file = tmp_path / "mesh_1x1.v"
        verilog_file.write_text(output)
        assert verilog_file.exists()

    def test_generate_mesh_2x2_verilog(self, tmp_path):
        """Test that 2x2 Mesh can generate valid Verilog."""
        from amaranth._toolchain.yosys import find_yosys
        from amaranth.back import verilog

        try:
            find_yosys(lambda ver: ver >= (0, 40))
        except Exception:
            pytest.skip("Yosys not found")

        config = SystolicConfig(mesh_rows=2, mesh_cols=2, tile_rows=1, tile_cols=1)
        mesh = Mesh(config)

        output = verilog.convert(mesh, name="Mesh_2x2")
        assert "module Mesh_2x2" in output

        # Check for vector ports
        assert "in_a_0" in output
        assert "in_a_1" in output
        assert "in_b_0" in output
        assert "in_b_1" in output

        # Check for tile submodules
        assert "tile_0_0" in output
        assert "tile_1_1" in output

        verilog_file = tmp_path / "mesh_2x2.v"
        verilog_file.write_text(output)
        assert verilog_file.exists()

    def test_generate_mesh_4x4_verilog(self, tmp_path):
        """Test that larger Mesh can generate valid Verilog."""
        from amaranth._toolchain.yosys import find_yosys
        from amaranth.back import verilog

        try:
            find_yosys(lambda ver: ver >= (0, 40))
        except Exception:
            pytest.skip("Yosys not found")

        config = SystolicConfig(mesh_rows=4, mesh_cols=4, tile_rows=1, tile_cols=1)
        mesh = Mesh(config)

        output = verilog.convert(mesh, name="Mesh_4x4")
        assert "module Mesh_4x4" in output

        # Check for tile submodules
        assert "tile_0_0" in output
        assert "tile_3_3" in output

        verilog_file = tmp_path / "mesh_4x4.v"
        verilog_file.write_text(output)
        assert verilog_file.exists()

    def test_generate_mesh_2x2_with_2x2_tiles_verilog(self, tmp_path):
        """Test Mesh with multi-PE tiles generates valid Verilog."""
        from amaranth._toolchain.yosys import find_yosys
        from amaranth.back import verilog

        try:
            find_yosys(lambda ver: ver >= (0, 40))
        except Exception:
            pytest.skip("Yosys not found")

        config = SystolicConfig(mesh_rows=2, mesh_cols=2, tile_rows=2, tile_cols=2)
        mesh = Mesh(config)

        output = verilog.convert(mesh, name="Mesh_2x2_tiles_2x2")
        assert "module Mesh_2x2_tiles_2x2" in output

        # Should have 4 total rows/cols of ports (2 mesh * 2 tile)
        assert "in_a_3" in output
        assert "in_b_3" in output
        assert "out_c_3" in output

        verilog_file = tmp_path / "mesh_2x2_tiles_2x2.v"
        verilog_file.write_text(output)
        assert verilog_file.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
