"""
Unit tests for the LoadController.

These tests verify:
1. LoadController instantiation
2. MEMCPY command acceptance (for load operations)
3. BlockMover request generation
4. Data reception and scratchpad write
5. Completion signaling
6. Verilog generation
"""

import pytest
from amaranth.sim import Simulator, Tick

from systars.config import SystolicConfig
from systars.controller.load import LoadController
from systars.util.commands import BlockMoverOpcode


class TestLoadController:
    """Test suite for LoadController."""

    @pytest.fixture
    def config(self):
        """Configuration for tests."""
        return SystolicConfig(
            blockmover_buswidth=128,
            grid_rows=4,
            grid_cols=4,
            tile_rows=1,
            tile_cols=1,
        )

    @pytest.fixture
    def controller(self, config):
        """Create a LoadController instance."""
        return LoadController(config)

    def test_instantiation(self, controller, config):
        """Test that LoadController can be instantiated."""
        assert controller is not None
        assert controller.config == config

    def test_has_correct_ports(self, controller):
        """Test that LoadController has all required ports."""
        # Command interface
        assert hasattr(controller, "cmd_valid")
        assert hasattr(controller, "cmd_ready")
        assert hasattr(controller, "cmd_opcode")
        assert hasattr(controller, "cmd_dram_addr")
        assert hasattr(controller, "cmd_sp_addr")
        assert hasattr(controller, "cmd_len")
        assert hasattr(controller, "cmd_id")

        # BlockMover interface
        assert hasattr(controller, "engine_req_valid")
        assert hasattr(controller, "engine_req_ready")
        assert hasattr(controller, "engine_req_addr")
        assert hasattr(controller, "engine_req_len")
        assert hasattr(controller, "engine_resp_valid")
        assert hasattr(controller, "engine_resp_ready")
        assert hasattr(controller, "engine_resp_data")
        assert hasattr(controller, "engine_resp_last")

        # Scratchpad interface
        assert hasattr(controller, "sp_write_en")
        assert hasattr(controller, "sp_write_addr")
        assert hasattr(controller, "sp_write_data")
        assert hasattr(controller, "sp_write_mask")

        # Status
        assert hasattr(controller, "busy")
        assert hasattr(controller, "completed")
        assert hasattr(controller, "completed_id")

    def test_idle_ready(self, config):
        """Test that LoadController is ready in idle state."""
        ctrl = LoadController(config)
        results = []

        def testbench():
            ready = yield ctrl.cmd_ready
            busy = yield ctrl.busy
            results.append({"ready": ready, "busy": busy})
            yield Tick()

        sim = Simulator(ctrl)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()

        assert results[0]["ready"] == 1, "Should be ready in idle state"
        assert results[0]["busy"] == 0, "Should not be busy in idle state"

    def test_memcpy_command_acceptance(self, config):
        """Test that LoadController accepts MEMCPY commands."""
        ctrl = LoadController(config)
        results = {"cmd_accepted": False, "dma_issued": False}

        def testbench():
            # Issue MEMCPY command for load
            yield ctrl.cmd_valid.eq(1)
            yield ctrl.cmd_opcode.eq(BlockMoverOpcode.MEMCPY)
            yield ctrl.cmd_dram_addr.eq(0x80000000)
            yield ctrl.cmd_sp_addr.eq(0x100)
            yield ctrl.cmd_len.eq(4)  # 4 beats
            yield ctrl.cmd_id.eq(1)
            yield Tick()

            # Check command accepted
            ready = yield ctrl.cmd_ready
            if ready == 0:
                results["cmd_accepted"] = True

            yield ctrl.cmd_valid.eq(0)
            yield Tick()

            # Should see BlockMover request
            dma_valid = yield ctrl.engine_req_valid
            engine_addr = yield ctrl.engine_req_addr
            dma_len = yield ctrl.engine_req_len
            if dma_valid:
                results["dma_issued"] = True
                assert engine_addr == 0x80000000
                assert dma_len == 3  # AXI4 style: 4 beats = len 3

            yield Tick()

        sim = Simulator(ctrl)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()

        assert results["cmd_accepted"], "Command should be accepted"
        assert results["dma_issued"], "BlockMover request should be issued"

    def test_single_beat_transfer(self, config):
        """Test single-beat MEMCPY transfer."""
        ctrl = LoadController(config)
        results = {"sp_write": False, "completed": False}

        def testbench():
            # Issue MEMCPY command for single beat
            yield ctrl.cmd_valid.eq(1)
            yield ctrl.cmd_opcode.eq(BlockMoverOpcode.MEMCPY)
            yield ctrl.cmd_dram_addr.eq(0x1000)
            yield ctrl.cmd_sp_addr.eq(0x10)
            yield ctrl.cmd_len.eq(1)
            yield ctrl.cmd_id.eq(5)
            yield Tick()

            yield ctrl.cmd_valid.eq(0)
            yield Tick()

            # Accept BlockMover request
            yield ctrl.engine_req_ready.eq(1)
            yield Tick()
            yield ctrl.engine_req_ready.eq(0)
            yield Tick()

            # Provide BlockMover response data
            # Note: sp_width = grid_cols * tile_cols * input_bits = 4*1*8 = 32 bits
            # so only lower 32 bits of engine_resp_data will be written
            yield ctrl.engine_resp_valid.eq(1)
            yield ctrl.engine_resp_data.eq(0xCAFEBABE)
            yield ctrl.engine_resp_last.eq(1)

            # Check scratchpad write BEFORE Tick (combinational)
            sp_write_en = yield ctrl.sp_write_en
            sp_write_addr = yield ctrl.sp_write_addr
            sp_write_data = yield ctrl.sp_write_data
            if sp_write_en:
                results["sp_write"] = True
                assert sp_write_addr == 0x10
                assert sp_write_data == 0xCAFEBABE

            yield Tick()  # State transitions to DONE

            yield ctrl.engine_resp_valid.eq(0)

            # Check completion BEFORE Tick (combinational in DONE state)
            # After Tick, state transitions to IDLE
            completed = yield ctrl.completed
            completed_id = yield ctrl.completed_id
            if completed:
                results["completed"] = True
                assert completed_id == 5

            yield Tick()  # State transitions to IDLE
            yield Tick()

        sim = Simulator(ctrl)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()

        assert results["sp_write"], "Should write to scratchpad"
        assert results["completed"], "Should signal completion"

    def test_multi_beat_transfer(self, config):
        """Test multi-beat MEMCPY transfer."""
        ctrl = LoadController(config)
        write_count = [0]
        write_addrs = []

        def testbench():
            # Issue MEMCPY command for 4 beats
            yield ctrl.cmd_valid.eq(1)
            yield ctrl.cmd_opcode.eq(BlockMoverOpcode.MEMCPY)
            yield ctrl.cmd_dram_addr.eq(0x2000)
            yield ctrl.cmd_sp_addr.eq(0x20)
            yield ctrl.cmd_len.eq(4)
            yield ctrl.cmd_id.eq(2)
            yield Tick()

            yield ctrl.cmd_valid.eq(0)
            yield Tick()

            # Accept BlockMover request
            yield ctrl.engine_req_ready.eq(1)
            yield Tick()
            yield ctrl.engine_req_ready.eq(0)
            yield Tick()

            # Provide 4 beats of data
            for i in range(4):
                yield ctrl.engine_resp_valid.eq(1)
                yield ctrl.engine_resp_data.eq(0x1000 + i)
                yield ctrl.engine_resp_last.eq(1 if i == 3 else 0)

                # Check scratchpad write BEFORE Tick
                sp_write_en = yield ctrl.sp_write_en
                sp_write_addr = yield ctrl.sp_write_addr
                if sp_write_en:
                    write_count[0] += 1
                    write_addrs.append(sp_write_addr)

                yield Tick()

            yield ctrl.engine_resp_valid.eq(0)
            yield Tick()
            yield Tick()

        sim = Simulator(ctrl)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()

        assert write_count[0] == 4, f"Expected 4 writes, got {write_count[0]}"
        assert write_addrs == [0x20, 0x21, 0x22, 0x23], f"Wrong addresses: {write_addrs}"

    def test_busy_signal(self, config):
        """Test that busy signal is asserted during transfer."""
        ctrl = LoadController(config)
        busy_states = []

        def testbench():
            # Check idle
            busy = yield ctrl.busy
            busy_states.append(("idle", busy))
            yield Tick()

            # Issue command
            yield ctrl.cmd_valid.eq(1)
            yield ctrl.cmd_opcode.eq(BlockMoverOpcode.MEMCPY)
            yield ctrl.cmd_dram_addr.eq(0x3000)
            yield ctrl.cmd_sp_addr.eq(0x30)
            yield ctrl.cmd_len.eq(1)
            yield Tick()
            yield ctrl.cmd_valid.eq(0)
            yield Tick()

            # Check busy during transfer
            busy = yield ctrl.busy
            busy_states.append(("transfer", busy))

            # Complete transfer
            yield ctrl.engine_req_ready.eq(1)
            yield Tick()
            yield ctrl.engine_req_ready.eq(0)
            yield Tick()

            yield ctrl.engine_resp_valid.eq(1)
            yield ctrl.engine_resp_last.eq(1)
            yield Tick()
            yield ctrl.engine_resp_valid.eq(0)
            yield Tick()
            yield Tick()

            # Check idle again
            busy = yield ctrl.busy
            busy_states.append(("done", busy))

        sim = Simulator(ctrl)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()

        assert busy_states[0] == ("idle", 0), "Should not be busy in idle"
        assert busy_states[1] == ("transfer", 1), "Should be busy during transfer"
        assert busy_states[2] == ("done", 0), "Should not be busy after done"

    def test_elaboration(self, controller):
        """Test that LoadController elaborates without errors."""

        def testbench():
            yield Tick()

        sim = Simulator(controller)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()


class TestLoadControllerVerilogGeneration:
    """Test Verilog generation for LoadController."""

    def test_generate_verilog(self, tmp_path):
        """Test that LoadController can generate valid Verilog."""
        from amaranth._toolchain.yosys import find_yosys
        from amaranth.back import verilog

        try:
            find_yosys(lambda ver: ver >= (0, 40))
        except Exception:
            pytest.skip("Yosys not found")

        config = SystolicConfig(blockmover_buswidth=512)
        ctrl = LoadController(config)

        output = verilog.convert(ctrl, name="LoadController")
        assert "module LoadController" in output
        assert "cmd_valid" in output
        assert "engine_req_valid" in output
        assert "sp_write_en" in output

        verilog_file = tmp_path / "load_controller.v"
        verilog_file.write_text(output)
        assert verilog_file.exists()


class TestStridedAndTransposeElaboration:
    """StridedStreamReader / MatrixLoader must elaborate and convert to Verilog."""

    @pytest.mark.parametrize("component", ["StridedStreamReader", "MatrixLoader"])
    def test_elaborates_to_verilog(self, component):
        from amaranth._toolchain.yosys import YosysError, find_yosys

        try:
            find_yosys(lambda ver: ver >= (0, 40))
        except YosysError:
            pytest.skip("Yosys not found")
        from amaranth.back import verilog

        from systars.blockmover.strided_reader import StridedStreamReader
        from systars.controller.load_transpose import MatrixLoader

        cls = {"StridedStreamReader": StridedStreamReader, "MatrixLoader": MatrixLoader}[component]
        config = SystolicConfig(grid_rows=4, grid_cols=4, blockmover_buswidth=128)
        output = verilog.convert(cls(config), name=component)
        assert f"module {component}" in output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
