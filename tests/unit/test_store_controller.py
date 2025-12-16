"""
Unit tests for the StoreController.

These tests verify:
1. StoreController instantiation
2. MEMCPY command acceptance (for store operations)
3. Accumulator read requests
4. DMA write data transmission
5. Activation function (ReLU)
6. Completion signaling
7. Verilog generation
"""

import pytest
from amaranth.sim import Simulator, Tick

from systars.config import SystolicConfig
from systars.controller.store import StoreController
from systars.util.commands import DmaOpcode


class TestStoreController:
    """Test suite for StoreController."""

    @pytest.fixture
    def config(self):
        """Configuration for tests."""
        return SystolicConfig(
            dma_buswidth=128,
            dma_maxbytes=64,
            grid_rows=4,
            grid_cols=4,
            tile_rows=1,
            tile_cols=1,
        )

    @pytest.fixture
    def controller(self, config):
        """Create a StoreController instance."""
        return StoreController(config)

    def test_instantiation(self, controller, config):
        """Test that StoreController can be instantiated."""
        assert controller is not None
        assert controller.config == config

    def test_has_correct_ports(self, controller):
        """Test that StoreController has all required ports."""
        # Command interface
        assert hasattr(controller, "cmd_valid")
        assert hasattr(controller, "cmd_ready")
        assert hasattr(controller, "cmd_opcode")
        assert hasattr(controller, "cmd_acc_addr")
        assert hasattr(controller, "cmd_dram_addr")
        assert hasattr(controller, "cmd_len")
        assert hasattr(controller, "cmd_id")
        assert hasattr(controller, "cmd_activation")

        # Accumulator interface
        assert hasattr(controller, "acc_read_req")
        assert hasattr(controller, "acc_read_addr")
        assert hasattr(controller, "acc_read_data")
        assert hasattr(controller, "acc_read_valid")

        # DMA interface
        assert hasattr(controller, "dma_req_valid")
        assert hasattr(controller, "dma_req_ready")
        assert hasattr(controller, "dma_req_addr")
        assert hasattr(controller, "dma_req_len")
        assert hasattr(controller, "dma_data_valid")
        assert hasattr(controller, "dma_data_ready")
        assert hasattr(controller, "dma_data")
        assert hasattr(controller, "dma_data_last")
        assert hasattr(controller, "dma_done")

        # Status
        assert hasattr(controller, "busy")
        assert hasattr(controller, "completed")
        assert hasattr(controller, "completed_id")

    def test_idle_ready(self, config):
        """Test that StoreController is ready in idle state."""
        ctrl = StoreController(config)
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
        """Test that StoreController accepts MEMCPY commands."""
        ctrl = StoreController(config)
        results = {"cmd_accepted": False, "dma_issued": False}

        def testbench():
            # Issue MEMCPY command for store
            yield ctrl.cmd_valid.eq(1)
            yield ctrl.cmd_opcode.eq(DmaOpcode.MEMCPY)
            yield ctrl.cmd_acc_addr.eq(0x100)
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

            # Should see DMA request
            dma_valid = yield ctrl.dma_req_valid
            dma_addr = yield ctrl.dma_req_addr
            dma_len = yield ctrl.dma_req_len
            if dma_valid:
                results["dma_issued"] = True
                assert dma_addr == 0x80000000
                assert dma_len == 1  # AXI4 style: 2 beats = len 1

            yield Tick()

        sim = Simulator(ctrl)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()

        assert results["cmd_accepted"], "Command should be accepted"
        assert results["dma_issued"], "DMA request should be issued"

    def test_single_beat_transfer(self, config):
        """Test single-beat MEMCPY transfer."""
        ctrl = StoreController(config)
        results = {"acc_read": False, "dma_data": False, "completed": False}

        def testbench():
            # Issue MEMCPY command for single beat store
            yield ctrl.cmd_valid.eq(1)
            yield ctrl.cmd_opcode.eq(DmaOpcode.MEMCPY)
            yield ctrl.cmd_acc_addr.eq(0x10)
            yield ctrl.cmd_dram_addr.eq(0x1000)
            yield ctrl.cmd_len.eq(1)
            yield ctrl.cmd_id.eq(5)
            yield ctrl.cmd_activation.eq(0)
            yield Tick()

            yield ctrl.cmd_valid.eq(0)
            yield Tick()

            # Accept DMA request
            yield ctrl.dma_req_ready.eq(1)
            yield Tick()  # State transitions to READ_ACC
            yield ctrl.dma_req_ready.eq(0)

            # Check accumulator read request BEFORE Tick (in READ_ACC state)
            # The state immediately transitions to WAIT_VALID on next Tick
            acc_read_req = yield ctrl.acc_read_req
            acc_read_addr = yield ctrl.acc_read_addr
            if acc_read_req:
                results["acc_read"] = True
                assert acc_read_addr == 0x10

            yield Tick()  # State transitions to WAIT_VALID

            # Provide accumulator read data
            yield ctrl.acc_read_valid.eq(1)
            yield ctrl.acc_read_data.eq(0xDEADBEEF)
            yield Tick()  # State transitions to SEND_DATA with buffered data
            yield ctrl.acc_read_valid.eq(0)
            yield Tick()

            # Check DMA data BEFORE Tick
            dma_valid = yield ctrl.dma_data_valid
            dma_data = yield ctrl.dma_data
            dma_last = yield ctrl.dma_data_last
            if dma_valid:
                results["dma_data"] = True
                assert dma_data == 0xDEADBEEF
                assert dma_last == 1

            # Accept DMA data
            yield ctrl.dma_data_ready.eq(1)
            yield Tick()  # State transitions to WAIT_DONE
            yield ctrl.dma_data_ready.eq(0)
            yield Tick()

            # Signal DMA done
            yield ctrl.dma_done.eq(1)
            yield Tick()  # State transitions to DONE
            yield ctrl.dma_done.eq(0)

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

        assert results["acc_read"], "Should read from accumulator"
        assert results["dma_data"], "Should send data to DMA"
        assert results["completed"], "Should signal completion"

    def test_multi_beat_transfer(self, config):
        """Test multi-beat MEMCPY transfer."""
        ctrl = StoreController(config)
        read_addrs = []
        data_values = []

        def testbench():
            # Issue MEMCPY command for 3 beats store
            yield ctrl.cmd_valid.eq(1)
            yield ctrl.cmd_opcode.eq(DmaOpcode.MEMCPY)
            yield ctrl.cmd_acc_addr.eq(0x20)
            yield ctrl.cmd_dram_addr.eq(0x2000)
            yield ctrl.cmd_len.eq(3)
            yield ctrl.cmd_id.eq(2)
            yield ctrl.cmd_activation.eq(0)
            yield Tick()

            yield ctrl.cmd_valid.eq(0)
            yield Tick()

            # Accept DMA request
            yield ctrl.dma_req_ready.eq(1)
            yield Tick()  # State transitions to READ_ACC
            yield ctrl.dma_req_ready.eq(0)

            # Transfer 3 beats
            for i in range(3):
                # Check accumulator read BEFORE Tick (in READ_ACC state)
                acc_read_req = yield ctrl.acc_read_req
                acc_read_addr = yield ctrl.acc_read_addr
                if acc_read_req:
                    read_addrs.append(acc_read_addr)

                yield Tick()  # State transitions to WAIT_VALID

                # Provide accumulator data
                yield ctrl.acc_read_valid.eq(1)
                yield ctrl.acc_read_data.eq(0x1000 + i)
                yield Tick()  # State transitions to SEND_DATA
                yield ctrl.acc_read_valid.eq(0)
                yield Tick()

                # Check DMA data BEFORE accepting
                dma_valid = yield ctrl.dma_data_valid
                dma_data = yield ctrl.dma_data
                if dma_valid:
                    data_values.append(dma_data)

                # Accept DMA data
                yield ctrl.dma_data_ready.eq(1)
                yield Tick()  # State transitions to READ_ACC or WAIT_DONE
                yield ctrl.dma_data_ready.eq(0)

            yield Tick()

            # Signal DMA done
            yield ctrl.dma_done.eq(1)
            yield Tick()
            yield ctrl.dma_done.eq(0)
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
        ctrl = StoreController(config)
        results = {"positive_passed": False, "negative_zeroed": False}

        def testbench():
            # Issue MEMCPY command with ReLU activation
            yield ctrl.cmd_valid.eq(1)
            yield ctrl.cmd_opcode.eq(DmaOpcode.MEMCPY)
            yield ctrl.cmd_acc_addr.eq(0x40)
            yield ctrl.cmd_dram_addr.eq(0x4000)
            yield ctrl.cmd_len.eq(2)
            yield ctrl.cmd_id.eq(3)
            yield ctrl.cmd_activation.eq(1)  # ReLU
            yield Tick()

            yield ctrl.cmd_valid.eq(0)
            yield Tick()

            # Accept DMA request
            yield ctrl.dma_req_ready.eq(1)
            yield Tick()
            yield ctrl.dma_req_ready.eq(0)
            yield Tick()

            # First beat: positive value
            yield Tick()  # Read request
            yield ctrl.acc_read_valid.eq(1)
            yield ctrl.acc_read_data.eq(100)  # Positive
            yield Tick()
            yield ctrl.acc_read_valid.eq(0)
            yield Tick()

            dma_data = yield ctrl.dma_data
            if dma_data == 100:
                results["positive_passed"] = True

            yield ctrl.dma_data_ready.eq(1)
            yield Tick()
            yield ctrl.dma_data_ready.eq(0)
            yield Tick()

            # Second beat: negative value
            yield Tick()  # Read request
            yield ctrl.acc_read_valid.eq(1)
            # Set negative value (signed representation)
            yield ctrl.acc_read_data.eq(-50 & ((1 << config.acc_width) - 1))
            yield Tick()
            yield ctrl.acc_read_valid.eq(0)
            yield Tick()

            dma_data = yield ctrl.dma_data
            if dma_data == 0:
                results["negative_zeroed"] = True

            yield ctrl.dma_data_ready.eq(1)
            yield Tick()
            yield ctrl.dma_data_ready.eq(0)
            yield Tick()

            # Signal DMA done
            yield ctrl.dma_done.eq(1)
            yield Tick()

        sim = Simulator(ctrl)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()

        assert results["positive_passed"], "Positive values should pass through"
        assert results["negative_zeroed"], "Negative values should be zeroed by ReLU"

    def test_busy_signal(self, config):
        """Test that busy signal is asserted during transfer."""
        ctrl = StoreController(config)
        busy_states = []

        def testbench():
            # Check idle
            busy = yield ctrl.busy
            busy_states.append(("idle", busy))
            yield Tick()

            # Issue command
            yield ctrl.cmd_valid.eq(1)
            yield ctrl.cmd_opcode.eq(DmaOpcode.MEMCPY)
            yield ctrl.cmd_acc_addr.eq(0x50)
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
            yield ctrl.dma_req_ready.eq(1)
            yield Tick()
            yield ctrl.dma_req_ready.eq(0)
            yield Tick()
            yield Tick()

            yield ctrl.acc_read_valid.eq(1)
            yield Tick()
            yield ctrl.acc_read_valid.eq(0)
            yield Tick()

            yield ctrl.dma_data_ready.eq(1)
            yield Tick()
            yield ctrl.dma_data_ready.eq(0)
            yield Tick()

            yield ctrl.dma_done.eq(1)
            yield Tick()
            yield ctrl.dma_done.eq(0)
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
        """Test that StoreController elaborates without errors."""

        def testbench():
            yield Tick()

        sim = Simulator(controller)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()


class TestStoreControllerVerilogGeneration:
    """Test Verilog generation for StoreController."""

    def test_generate_verilog(self, tmp_path):
        """Test that StoreController can generate valid Verilog."""
        from amaranth._toolchain.yosys import find_yosys
        from amaranth.back import verilog

        try:
            find_yosys(lambda ver: ver >= (0, 40))
        except Exception:
            pytest.skip("Yosys not found")

        config = SystolicConfig(dma_buswidth=128, dma_maxbytes=64)
        ctrl = StoreController(config)

        output = verilog.convert(ctrl, name="StoreController")
        assert "module StoreController" in output
        assert "cmd_valid" in output
        assert "acc_read_req" in output
        assert "dma_data_valid" in output

        verilog_file = tmp_path / "store_controller.v"
        verilog_file.write_text(output)
        assert verilog_file.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
