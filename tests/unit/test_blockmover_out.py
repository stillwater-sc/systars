"""
Unit tests for the BlockMoverOut.

These tests verify:
1. BlockMoverOut instantiation
2. MEMCPY command acceptance (for store operations)
3. Combiner read requests
4. BlockMover write data transmission
5. Activation function (ReLU)
6. Completion signaling
7. Verilog generation
"""

import pytest
from amaranth.sim import Simulator, Tick

from systars.config import SystolicConfig
from systars.controller.blockmover_out import BlockMoverOut
from systars.util.commands import BlockMoverOpcode


class TestBlockMoverOut:
    """Test suite for BlockMoverOut."""

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
        """Create a BlockMoverOut instance."""
        return BlockMoverOut(config)

    def test_instantiation(self, controller, config):
        """Test that BlockMoverOut can be instantiated."""
        assert controller is not None
        assert controller.config == config

    def test_has_correct_ports(self, controller):
        """Test that BlockMoverOut has all required ports."""
        # Command interface
        assert hasattr(controller, "cmd_valid")
        assert hasattr(controller, "cmd_ready")
        assert hasattr(controller, "cmd_opcode")
        assert hasattr(controller, "cmd_combiner_addr")
        assert hasattr(controller, "cmd_dram_addr")
        assert hasattr(controller, "cmd_len")
        assert hasattr(controller, "cmd_id")
        assert hasattr(controller, "cmd_activation")

        # Combiner interface
        assert hasattr(controller, "combiner_read_req")
        assert hasattr(controller, "combiner_read_addr")
        assert hasattr(controller, "combiner_read_data")
        assert hasattr(controller, "combiner_read_valid")

        # BlockMover interface
        assert hasattr(controller, "engine_req_valid")
        assert hasattr(controller, "engine_req_ready")
        assert hasattr(controller, "engine_req_addr")
        assert hasattr(controller, "engine_req_len")
        assert hasattr(controller, "engine_data_valid")
        assert hasattr(controller, "engine_data_ready")
        assert hasattr(controller, "engine_data")
        assert hasattr(controller, "engine_data_last")
        assert hasattr(controller, "engine_done")

        # Status
        assert hasattr(controller, "busy")
        assert hasattr(controller, "completed")
        assert hasattr(controller, "completed_id")

    def test_idle_ready(self, config):
        """Test that BlockMoverOut is ready in idle state."""
        ctrl = BlockMoverOut(config)
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
        """Test that BlockMoverOut accepts MEMCPY commands."""
        ctrl = BlockMoverOut(config)
        results = {"cmd_accepted": False, "dma_issued": False}

        def testbench():
            # Issue MEMCPY command for store
            yield ctrl.cmd_valid.eq(1)
            yield ctrl.cmd_opcode.eq(BlockMoverOpcode.MEMCPY)
            yield ctrl.cmd_combiner_addr.eq(0x100)
            yield ctrl.cmd_dram_addr.eq(0x80000000)
            yield ctrl.cmd_len.eq(2)  # 2 beats
            yield ctrl.cmd_id.eq(1)
            yield ctrl.cmd_activation.eq(0)
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
                assert dma_len == 1  # AXI4 style: 2 beats = len 1

            yield Tick()

        sim = Simulator(ctrl)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()

        assert results["cmd_accepted"], "Command should be accepted"
        assert results["dma_issued"], "BlockMover request should be issued"

    def test_single_beat_transfer(self, config):
        """Test single-beat MEMCPY transfer."""
        ctrl = BlockMoverOut(config)
        results = {"combiner_read": False, "engine_data": False, "completed": False}

        def testbench():
            # Issue MEMCPY command for single beat store
            yield ctrl.cmd_valid.eq(1)
            yield ctrl.cmd_opcode.eq(BlockMoverOpcode.MEMCPY)
            yield ctrl.cmd_combiner_addr.eq(0x10)
            yield ctrl.cmd_dram_addr.eq(0x1000)
            yield ctrl.cmd_len.eq(1)
            yield ctrl.cmd_id.eq(5)
            yield ctrl.cmd_activation.eq(0)
            yield Tick()

            yield ctrl.cmd_valid.eq(0)
            yield Tick()

            # Accept BlockMover request
            yield ctrl.engine_req_ready.eq(1)
            yield Tick()  # State transitions to READ_COMBINER
            yield ctrl.engine_req_ready.eq(0)

            # Check Combiner read request BEFORE Tick (in READ_COMBINER state)
            # The state immediately transitions to WAIT_VALID on next Tick
            combiner_read_req = yield ctrl.combiner_read_req
            combiner_read_addr = yield ctrl.combiner_read_addr
            if combiner_read_req:
                results["combiner_read"] = True
                assert combiner_read_addr == 0x10

            yield Tick()  # State transitions to WAIT_VALID

            # Provide Combiner read data
            yield ctrl.combiner_read_valid.eq(1)
            yield ctrl.combiner_read_data.eq(0xDEADBEEF)
            yield Tick()  # State transitions to SEND_DATA with buffered data
            yield ctrl.combiner_read_valid.eq(0)
            yield Tick()

            # Check BlockMover data BEFORE Tick
            dma_valid = yield ctrl.engine_data_valid
            engine_data = yield ctrl.engine_data
            dma_last = yield ctrl.engine_data_last
            if dma_valid:
                results["engine_data"] = True
                assert engine_data == 0xDEADBEEF
                assert dma_last == 1

            # Accept BlockMover data
            yield ctrl.engine_data_ready.eq(1)
            yield Tick()  # State transitions to WAIT_DONE
            yield ctrl.engine_data_ready.eq(0)
            yield Tick()

            # Signal BlockMover done
            yield ctrl.engine_done.eq(1)
            yield Tick()  # State transitions to DONE
            yield ctrl.engine_done.eq(0)

            # Check completion (combinational in DONE state)
            completed = yield ctrl.completed
            completed_id = yield ctrl.completed_id
            if completed:
                results["completed"] = True
                assert completed_id == 5

            yield Tick()

        sim = Simulator(ctrl)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()

        assert results["combiner_read"], "Should read from Combiner"
        assert results["engine_data"], "Should send data to BlockMover"
        assert results["completed"], "Should signal completion"

    def test_multi_beat_transfer(self, config):
        """Test multi-beat MEMCPY transfer."""
        ctrl = BlockMoverOut(config)
        read_addrs = []
        data_values = []

        def testbench():
            # Issue MEMCPY command for 3 beats store
            yield ctrl.cmd_valid.eq(1)
            yield ctrl.cmd_opcode.eq(BlockMoverOpcode.MEMCPY)
            yield ctrl.cmd_combiner_addr.eq(0x20)
            yield ctrl.cmd_dram_addr.eq(0x2000)
            yield ctrl.cmd_len.eq(3)
            yield ctrl.cmd_id.eq(2)
            yield ctrl.cmd_activation.eq(0)
            yield Tick()

            yield ctrl.cmd_valid.eq(0)
            yield Tick()

            # Accept BlockMover request
            yield ctrl.engine_req_ready.eq(1)
            yield Tick()  # State transitions to READ_COMBINER
            yield ctrl.engine_req_ready.eq(0)

            # Transfer 3 beats
            for i in range(3):
                # Check Combiner read BEFORE Tick (in READ_COMBINER state)
                combiner_read_req = yield ctrl.combiner_read_req
                combiner_read_addr = yield ctrl.combiner_read_addr
                if combiner_read_req:
                    read_addrs.append(combiner_read_addr)

                yield Tick()  # State transitions to WAIT_VALID

                # Provide Combiner data
                yield ctrl.combiner_read_valid.eq(1)
                yield ctrl.combiner_read_data.eq(0x1000 + i)
                yield Tick()  # State transitions to SEND_DATA
                yield ctrl.combiner_read_valid.eq(0)
                yield Tick()

                # Check BlockMover data BEFORE accepting
                dma_valid = yield ctrl.engine_data_valid
                engine_data = yield ctrl.engine_data
                if dma_valid:
                    data_values.append(engine_data)

                # Accept BlockMover data
                yield ctrl.engine_data_ready.eq(1)
                yield Tick()  # State transitions to READ_COMBINER or WAIT_DONE
                yield ctrl.engine_data_ready.eq(0)

            yield Tick()

            # Signal BlockMover done
            yield ctrl.engine_done.eq(1)
            yield Tick()
            yield ctrl.engine_done.eq(0)
            yield Tick()
            yield Tick()

        sim = Simulator(ctrl)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()

        assert read_addrs == [0x20, 0x21, 0x22], f"Wrong read addresses: {read_addrs}"
        assert data_values == [0x1000, 0x1001, 0x1002], f"Wrong data: {data_values}"

    def test_relu_activation(self, config):
        """Test ReLU activation on negative values."""
        ctrl = BlockMoverOut(config)
        results = {"positive_passed": False, "negative_zeroed": False}

        def testbench():
            # Issue MEMCPY command with ReLU activation
            yield ctrl.cmd_valid.eq(1)
            yield ctrl.cmd_opcode.eq(BlockMoverOpcode.MEMCPY)
            yield ctrl.cmd_combiner_addr.eq(0x40)
            yield ctrl.cmd_dram_addr.eq(0x4000)
            yield ctrl.cmd_len.eq(2)
            yield ctrl.cmd_id.eq(3)
            yield ctrl.cmd_activation.eq(1)  # ReLU
            yield Tick()

            yield ctrl.cmd_valid.eq(0)
            yield Tick()

            # Accept BlockMover request
            yield ctrl.engine_req_ready.eq(1)
            yield Tick()
            yield ctrl.engine_req_ready.eq(0)
            yield Tick()

            # First beat: positive value
            yield Tick()  # Read request
            yield ctrl.combiner_read_valid.eq(1)
            yield ctrl.combiner_read_data.eq(100)  # Positive
            yield Tick()
            yield ctrl.combiner_read_valid.eq(0)
            yield Tick()

            engine_data = yield ctrl.engine_data
            if engine_data == 100:
                results["positive_passed"] = True

            yield ctrl.engine_data_ready.eq(1)
            yield Tick()
            yield ctrl.engine_data_ready.eq(0)
            yield Tick()

            # Second beat: negative value
            yield Tick()  # Read request
            yield ctrl.combiner_read_valid.eq(1)
            # Set negative value (signed representation)
            yield ctrl.combiner_read_data.eq(-50 & ((1 << config.combiner_width) - 1))
            yield Tick()
            yield ctrl.combiner_read_valid.eq(0)
            yield Tick()

            engine_data = yield ctrl.engine_data
            if engine_data == 0:
                results["negative_zeroed"] = True

            yield ctrl.engine_data_ready.eq(1)
            yield Tick()
            yield ctrl.engine_data_ready.eq(0)
            yield Tick()

            # Signal BlockMover done
            yield ctrl.engine_done.eq(1)
            yield Tick()

        sim = Simulator(ctrl)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()

        assert results["positive_passed"], "Positive values should pass through"
        assert results["negative_zeroed"], "Negative values should be zeroed by ReLU"

    def test_busy_signal(self, config):
        """Test that busy signal is asserted during transfer."""
        ctrl = BlockMoverOut(config)
        busy_states = []

        def testbench():
            # Check idle
            busy = yield ctrl.busy
            busy_states.append(("idle", busy))
            yield Tick()

            # Issue command
            yield ctrl.cmd_valid.eq(1)
            yield ctrl.cmd_opcode.eq(BlockMoverOpcode.MEMCPY)
            yield ctrl.cmd_combiner_addr.eq(0x50)
            yield ctrl.cmd_dram_addr.eq(0x5000)
            yield ctrl.cmd_len.eq(1)
            yield ctrl.cmd_activation.eq(0)
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
            yield Tick()

            yield ctrl.combiner_read_valid.eq(1)
            yield Tick()
            yield ctrl.combiner_read_valid.eq(0)
            yield Tick()

            yield ctrl.engine_data_ready.eq(1)
            yield Tick()
            yield ctrl.engine_data_ready.eq(0)
            yield Tick()

            yield ctrl.engine_done.eq(1)
            yield Tick()
            yield ctrl.engine_done.eq(0)
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
        """Test that BlockMoverOut elaborates without errors."""

        def testbench():
            yield Tick()

        sim = Simulator(controller)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()


class TestBlockMoverOutVerilogGeneration:
    """Test Verilog generation for BlockMoverOut."""

    def test_generate_verilog(self, tmp_path):
        """Test that BlockMoverOut can generate valid Verilog."""
        from amaranth._toolchain.yosys import find_yosys
        from amaranth.back import verilog

        try:
            find_yosys(lambda ver: ver >= (0, 40))
        except Exception:
            pytest.skip("Yosys not found")

        # 4 x 32-bit Combiner row = 128 bits: one row per 128-bit BlockMover beat
        config = SystolicConfig(grid_rows=4, grid_cols=4, blockmover_buswidth=128)
        ctrl = BlockMoverOut(config)

        output = verilog.convert(ctrl, name="BlockMoverOut")
        assert "module BlockMoverOut" in output
        assert "cmd_valid" in output
        assert "combiner_read_req" in output
        assert "engine_data_valid" in output

        verilog_file = tmp_path / "store_controller.v"
        verilog_file.write_text(output)
        assert verilog_file.exists()


def test_rejects_rows_wider_than_dma_bus():
    """A row wider than the BlockMover bus would be silently truncated: reject the config."""
    with pytest.raises(ValueError, match="wider than blockmover_buswidth"):
        BlockMoverOut(SystolicConfig(grid_rows=16, grid_cols=16, blockmover_buswidth=128))


def test_default_config_dma_width_fits_rows():
    """The default config picks a power-of-two DMA width that fits a row."""
    config = SystolicConfig()
    assert config.blockmover_buswidth >= max(config.sp_width, config.combiner_width)
    assert config.blockmover_buswidth & (config.blockmover_buswidth - 1) == 0
    assert (
        SystolicConfig(blockmover_buswidth=128, grid_rows=4, grid_cols=4).blockmover_buswidth == 128
    )


def test_default_config_systolic_top_converts():
    """With the automatic DMA width, the default SystolicTop converts to Verilog."""
    from amaranth._toolchain.yosys import YosysError, find_yosys

    try:
        find_yosys(lambda ver: ver >= (0, 40))
    except YosysError:
        pytest.skip("Yosys not found")
    from amaranth.back import verilog

    from systars.top import SystolicTop

    output = verilog.convert(SystolicTop(SystolicConfig()), name="SystolicTop")
    assert "module SystolicTop" in output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
