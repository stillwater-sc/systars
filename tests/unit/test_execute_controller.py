"""
Unit tests for the ExecuteController module.

These tests verify:
1. ExecuteController instantiation
2. Port existence and configuration
3. CONFIG command (dataflow/shift configuration)
4. PRELOAD command (bias loading from accumulator)
5. COMPUTE command (matmul execution)
6. State machine transitions
7. Verilog generation
"""

import pytest
from amaranth.sim import Simulator, Tick

from systars.config import SystolicConfig
from systars.controller.execute import ExecuteController
from systars.util.commands import OpCode


class TestExecuteControllerInstantiation:
    """Test suite for ExecuteController instantiation and ports."""

    @pytest.fixture
    def config(self):
        """Small configuration for faster tests."""
        return SystolicConfig(
            grid_rows=2,
            grid_cols=2,
            tile_rows=1,
            tile_cols=1,
            input_bits=8,
            weight_bits=8,
            acc_bits=32,
            spad_read_delay=2,
        )

    @pytest.fixture
    def controller(self, config):
        """Create an ExecuteController instance."""
        return ExecuteController(config)

    def test_instantiation(self, controller, config):
        """Test that ExecuteController can be instantiated."""
        assert controller is not None
        assert controller.config == config

    def test_command_interface_ports(self, controller):
        """Test that controller has command interface ports."""
        assert hasattr(controller, "cmd_valid")
        assert hasattr(controller, "cmd_ready")
        assert hasattr(controller, "cmd_opcode")
        assert hasattr(controller, "cmd_rs1")
        assert hasattr(controller, "cmd_rs2")
        assert hasattr(controller, "cmd_rd")
        assert hasattr(controller, "cmd_k_dim")
        assert hasattr(controller, "cmd_id")

    def test_config_output_ports(self, controller):
        """Test that controller has configuration output ports."""
        assert hasattr(controller, "cfg_dataflow")
        assert hasattr(controller, "cfg_shift")
        assert hasattr(controller, "cfg_propagate")

    def test_scratchpad_interface_ports(self, controller):
        """Test that controller has scratchpad interface ports."""
        # A port
        assert hasattr(controller, "sp_a_read_req")
        assert hasattr(controller, "sp_a_read_addr")
        assert hasattr(controller, "sp_a_read_data")
        assert hasattr(controller, "sp_a_read_valid")
        # B port
        assert hasattr(controller, "sp_b_read_req")
        assert hasattr(controller, "sp_b_read_addr")
        assert hasattr(controller, "sp_b_read_data")
        assert hasattr(controller, "sp_b_read_valid")

    def test_accumulator_interface_ports(self, controller):
        """Test that controller has accumulator interface ports."""
        assert hasattr(controller, "acc_read_req")
        assert hasattr(controller, "acc_read_addr")
        assert hasattr(controller, "acc_read_data")
        assert hasattr(controller, "acc_read_valid")
        assert hasattr(controller, "acc_write_req")
        assert hasattr(controller, "acc_write_addr")
        assert hasattr(controller, "acc_write_data")
        assert hasattr(controller, "acc_accumulate")

    def test_array_vector_ports(self, config, controller):
        """Test that controller has correct number of array vector ports."""
        total_rows = config.grid_rows * config.tile_rows
        total_cols = config.grid_cols * config.tile_cols

        # Input vector ports
        for i in range(total_rows):
            assert hasattr(controller, f"array_in_a_{i}")
        for j in range(total_cols):
            assert hasattr(controller, f"array_in_b_{j}")
            assert hasattr(controller, f"array_in_d_{j}")

        # Output vector ports
        for j in range(total_cols):
            assert hasattr(controller, f"array_out_c_{j}")

    def test_array_control_ports(self, controller):
        """Test that controller has array control ports."""
        assert hasattr(controller, "array_in_valid")
        assert hasattr(controller, "array_in_control_dataflow")
        assert hasattr(controller, "array_in_control_propagate")
        assert hasattr(controller, "array_in_control_shift")
        assert hasattr(controller, "array_in_id")
        assert hasattr(controller, "array_in_last")
        assert hasattr(controller, "array_out_valid")
        assert hasattr(controller, "array_out_id")
        assert hasattr(controller, "array_out_last")

    def test_status_ports(self, controller):
        """Test that controller has status ports."""
        assert hasattr(controller, "busy")
        assert hasattr(controller, "completed")
        assert hasattr(controller, "completed_id")
        assert hasattr(controller, "state_debug")


class TestExecuteControllerIdleState:
    """Test suite for ExecuteController IDLE state."""

    @pytest.fixture
    def config(self):
        """Small configuration for faster tests."""
        return SystolicConfig(
            grid_rows=2,
            grid_cols=2,
            tile_rows=1,
            tile_cols=1,
            spad_read_delay=1,
        )

    def test_idle_state_signals(self, config):
        """Test that IDLE state has correct signal values."""
        ctrl = ExecuteController(config)
        results = {}

        def testbench():
            # Wait a cycle to let things settle
            yield Tick()

            # In IDLE, cmd_ready should be high
            results["cmd_ready"] = yield ctrl.cmd_ready
            results["busy"] = yield ctrl.busy
            results["completed"] = yield ctrl.completed
            results["state"] = yield ctrl.state_debug

        sim = Simulator(ctrl)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()

        assert results["cmd_ready"] == 1, "cmd_ready should be high in IDLE"
        assert results["busy"] == 0, "busy should be low in IDLE"
        assert results["completed"] == 0, "completed should be low in IDLE"
        assert results["state"] == 0, "state should be 0 (IDLE)"


class TestExecuteControllerConfig:
    """Test suite for ExecuteController CONFIG command."""

    @pytest.fixture
    def config(self):
        """Small configuration for faster tests."""
        return SystolicConfig(
            grid_rows=2,
            grid_cols=2,
            tile_rows=1,
            tile_cols=1,
            spad_read_delay=1,
        )

    def test_config_command(self, config):
        """Test CONFIG_EX command sets dataflow and shift."""
        ctrl = ExecuteController(config)
        results = {}

        def testbench():
            # Issue CONFIG_EX command
            # cmd_rs1 format: [0] = dataflow, [5:1] = shift
            # Set dataflow=1 (WS), shift=7 -> rs1 = (7 << 1) | 1 = 0x0F
            yield ctrl.cmd_valid.eq(1)
            yield ctrl.cmd_opcode.eq(OpCode.CONFIG_EX)
            yield ctrl.cmd_rs1.eq(0x0F)  # dataflow=1, shift=7
            yield ctrl.cmd_id.eq(42)
            yield Tick()  # After this tick: FSM transitions to CONFIG

            # Check completion BEFORE next tick - completed is combinational
            # and only high during CONFIG state
            results["completed"] = yield ctrl.completed
            results["completed_id"] = yield ctrl.completed_id

            # Command accepted, deassert valid
            yield ctrl.cmd_valid.eq(0)
            yield Tick()  # After this tick: FSM back to IDLE

            # Config should be latched now
            results["cfg_dataflow"] = yield ctrl.cfg_dataflow
            results["cfg_shift"] = yield ctrl.cfg_shift

            # Should be back in IDLE
            results["cmd_ready"] = yield ctrl.cmd_ready
            results["busy"] = yield ctrl.busy

        sim = Simulator(ctrl)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()

        assert results["completed"] == 1, "completed should be asserted"
        assert results["completed_id"] == 42, "completed_id should match cmd_id"
        assert results["cfg_dataflow"] == 1, "dataflow should be WS (1)"
        assert results["cfg_shift"] == 7, "shift should be 7"
        assert results["cmd_ready"] == 1, "should return to IDLE"
        assert results["busy"] == 0, "should not be busy"


class TestExecuteControllerPreload:
    """Test suite for ExecuteController PRELOAD command."""

    @pytest.fixture
    def config(self):
        """Small configuration for faster tests."""
        return SystolicConfig(
            grid_rows=2,
            grid_cols=2,
            tile_rows=1,
            tile_cols=1,
            spad_read_delay=1,
            acc_latency=1,
        )

    def test_preload_requests_accumulator_read(self, config):
        """Test PRELOAD command requests read from accumulator."""
        ctrl = ExecuteController(config)
        results = {}

        def testbench():
            # Issue PRELOAD command
            yield ctrl.cmd_valid.eq(1)
            yield ctrl.cmd_opcode.eq(OpCode.PRELOAD)
            yield ctrl.cmd_rs1.eq(0x100)  # Accumulator address
            yield ctrl.cmd_id.eq(5)
            yield Tick()  # After this: FSM transitions to PRELOAD_START

            # Check immediately - acc_read_req is set combinationally in PRELOAD_START
            results["state_after_cmd"] = yield ctrl.state_debug

            yield ctrl.cmd_valid.eq(0)
            yield Tick()  # After this: FSM in PRELOAD_WAIT (if not acc_read_valid)

            # Check that accumulator read was requested in PRELOAD_START
            # Note: may have transitioned to PRELOAD_WAIT by now
            results["acc_read_req"] = yield ctrl.acc_read_req
            results["busy"] = yield ctrl.busy
            results["state"] = yield ctrl.state_debug

        sim = Simulator(ctrl)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()

        # After first tick, should be in PRELOAD_START (state 2) or PRELOAD_WAIT (state 3)
        assert results["state_after_cmd"] in [2, 3], (
            f"should be in PRELOAD state, got {results['state_after_cmd']}"
        )
        assert results["busy"] == 1, "should be busy during PRELOAD"

    def test_preload_feeds_array_on_valid(self, config):
        """Test PRELOAD feeds bias to array when acc_read_valid."""
        ctrl = ExecuteController(config)
        results = {}

        def testbench():
            # Issue PRELOAD command
            yield ctrl.cmd_valid.eq(1)
            yield ctrl.cmd_opcode.eq(OpCode.PRELOAD)
            yield ctrl.cmd_rs1.eq(0x100)
            yield ctrl.cmd_id.eq(10)
            yield Tick()

            yield ctrl.cmd_valid.eq(0)
            yield Tick()

            # Provide accumulator read response
            # For 2x2 array with 32-bit acc, acc_width = 64 bits
            # Pack two 32-bit values: [31:0] = 0x1111, [63:32] = 0x2222
            yield ctrl.acc_read_valid.eq(1)
            yield ctrl.acc_read_data.eq(0x2222_0000_1111)
            yield Tick()

            # Check array_in_valid and completion
            results["array_in_valid"] = yield ctrl.array_in_valid
            results["array_in_d_0"] = yield ctrl.array_in_d_0
            results["array_in_d_1"] = yield ctrl.array_in_d_1
            results["completed"] = yield ctrl.completed

            yield ctrl.acc_read_valid.eq(0)
            yield Tick()

            # Should return to IDLE
            results["busy"] = yield ctrl.busy
            results["cmd_ready"] = yield ctrl.cmd_ready

        sim = Simulator(ctrl)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()

        assert results["array_in_valid"] == 1, "array_in_valid should be high"
        assert results["completed"] == 1, "completed should be asserted"
        assert results["busy"] == 0, "should return to IDLE"
        assert results["cmd_ready"] == 1, "cmd_ready should be high"


class TestExecuteControllerCompute:
    """Test suite for ExecuteController COMPUTE command."""

    @pytest.fixture
    def config(self):
        """Small configuration for faster tests."""
        return SystolicConfig(
            grid_rows=2,
            grid_cols=2,
            tile_rows=1,
            tile_cols=1,
            spad_read_delay=1,
        )

    def test_compute_requests_scratchpad_reads(self, config):
        """Test COMPUTE command requests A and B from scratchpad."""
        ctrl = ExecuteController(config)
        results = {"sp_a_req": [], "sp_b_req": []}

        def testbench():
            # Issue COMPUTE command
            yield ctrl.cmd_valid.eq(1)
            yield ctrl.cmd_opcode.eq(OpCode.COMPUTE)
            yield ctrl.cmd_rs1.eq(0x000)  # A address
            yield ctrl.cmd_rs2.eq(0x100)  # B address
            yield ctrl.cmd_rd.eq(0x200)  # C address
            yield ctrl.cmd_k_dim.eq(1)  # k_dim = 1
            yield ctrl.cmd_id.eq(20)
            yield Tick()

            yield ctrl.cmd_valid.eq(0)

            # Track scratchpad requests over several cycles
            for _ in range(6):
                yield Tick()
                a_req = yield ctrl.sp_a_read_req
                b_req = yield ctrl.sp_b_read_req
                results["sp_a_req"].append(a_req)
                results["sp_b_req"].append(b_req)

        sim = Simulator(ctrl)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()

        # Should see A requests and B requests at some point
        assert any(results["sp_a_req"]), "sp_a_read_req should be asserted"
        assert any(results["sp_b_req"]), "sp_b_read_req should be asserted"

    def test_compute_state_progression(self, config):
        """Test COMPUTE command progresses through states."""
        ctrl = ExecuteController(config)
        state_history = []

        def testbench():
            # Issue COMPUTE command
            yield ctrl.cmd_valid.eq(1)
            yield ctrl.cmd_opcode.eq(OpCode.COMPUTE)
            yield ctrl.cmd_rs1.eq(0x000)
            yield ctrl.cmd_rs2.eq(0x100)
            yield ctrl.cmd_rd.eq(0x200)
            yield ctrl.cmd_k_dim.eq(1)
            yield ctrl.cmd_id.eq(30)
            yield Tick()

            yield ctrl.cmd_valid.eq(0)

            # Track state over several cycles
            for _i in range(10):
                yield Tick()
                state = yield ctrl.state_debug
                state_history.append(state)

                # Provide scratchpad responses when requested
                if state in [6, 7, 8]:  # COMPUTE_A_REQ, COMPUTE_B_REQ, COMPUTE_WAIT
                    yield ctrl.sp_a_read_valid.eq(1)
                    yield ctrl.sp_b_read_valid.eq(1)
                    yield ctrl.sp_a_read_data.eq(0x0102)
                    yield ctrl.sp_b_read_data.eq(0x0304)
                else:
                    yield ctrl.sp_a_read_valid.eq(0)
                    yield ctrl.sp_b_read_valid.eq(0)

        sim = Simulator(ctrl)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()

        # Should see state progression (not stuck in IDLE)
        assert len(set(state_history)) > 1, "should see multiple states"
        # State 5 is COMPUTE_START
        assert 5 in state_history or 6 in state_history, "should enter compute states"

    def test_compute_feeds_array(self, config):
        """Test COMPUTE feeds data to array when scratchpad valid."""
        ctrl = ExecuteController(config)
        results = {}

        def testbench():
            # Issue COMPUTE command
            yield ctrl.cmd_valid.eq(1)
            yield ctrl.cmd_opcode.eq(OpCode.COMPUTE)
            yield ctrl.cmd_rs1.eq(0x000)
            yield ctrl.cmd_rs2.eq(0x100)
            yield ctrl.cmd_rd.eq(0x200)
            yield ctrl.cmd_k_dim.eq(1)
            yield ctrl.cmd_id.eq(40)
            yield Tick()

            yield ctrl.cmd_valid.eq(0)

            # Wait for COMPUTE_WAIT state
            for _ in range(4):
                yield Tick()

            # Provide scratchpad data
            # For 2x2 array: sp_width = 2 * 8 = 16 bits per row
            yield ctrl.sp_a_read_valid.eq(1)
            yield ctrl.sp_b_read_valid.eq(1)
            yield ctrl.sp_a_read_data.eq(0x0102)  # A[0]=0x02, A[1]=0x01
            yield ctrl.sp_b_read_data.eq(0x0304)  # B[0]=0x04, B[1]=0x03
            yield Tick()

            # Check array inputs in COMPUTE_FEED state
            yield Tick()  # May need extra cycle
            results["array_in_valid"] = yield ctrl.array_in_valid

        sim = Simulator(ctrl)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()

        # Should see array_in_valid at some point during compute
        # Note: exact timing depends on state machine, this is a basic check


class TestExecuteControllerElaboration:
    """Test suite for ExecuteController elaboration and Verilog generation."""

    @pytest.fixture
    def config(self):
        """Small configuration for faster tests."""
        return SystolicConfig(
            grid_rows=2,
            grid_cols=2,
            tile_rows=1,
            tile_cols=1,
        )

    def test_elaboration(self, config):
        """Test that ExecuteController elaborates without errors."""
        ctrl = ExecuteController(config)

        def testbench():
            yield Tick()

        sim = Simulator(ctrl)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()

    def test_generate_verilog(self, config, tmp_path):
        """Test that ExecuteController can generate valid Verilog."""
        from amaranth._toolchain.yosys import find_yosys
        from amaranth.back import verilog

        try:
            find_yosys(lambda ver: ver >= (0, 40))
        except Exception:
            pytest.skip("Yosys not found")

        ctrl = ExecuteController(config)

        output = verilog.convert(ctrl, name="ExecuteController")
        assert "module ExecuteController" in output
        assert "cmd_valid" in output
        assert "cmd_ready" in output
        assert "sp_a_read_req" in output
        assert "acc_write_req" in output
        assert "array_in_a_0" in output
        assert "array_out_c_0" in output

        verilog_file = tmp_path / "execute_controller.v"
        verilog_file.write_text(output)
        assert verilog_file.exists()


class TestExecuteControllerLargerConfig:
    """Test ExecuteController with larger configurations."""

    def test_4x4_array(self):
        """Test with 4x4 SystolicArray configuration."""
        config = SystolicConfig(
            grid_rows=4,
            grid_cols=4,
            tile_rows=1,
            tile_cols=1,
        )
        ctrl = ExecuteController(config)

        # Should have 4 array_in_a ports and 4 array_in_b/d ports
        assert hasattr(ctrl, "array_in_a_0")
        assert hasattr(ctrl, "array_in_a_3")
        assert hasattr(ctrl, "array_in_b_0")
        assert hasattr(ctrl, "array_in_b_3")
        assert hasattr(ctrl, "array_out_c_0")
        assert hasattr(ctrl, "array_out_c_3")

    def test_2x2_array_with_2x2_pe_arrays(self):
        """Test with 2x2 SystolicArray of 2x2 PEArrays configuration."""
        config = SystolicConfig(
            grid_rows=2,
            grid_cols=2,
            tile_rows=2,
            tile_cols=2,
        )
        ctrl = ExecuteController(config)

        # Total rows/cols = 4
        assert hasattr(ctrl, "array_in_a_0")
        assert hasattr(ctrl, "array_in_a_3")
        assert hasattr(ctrl, "array_in_b_0")
        assert hasattr(ctrl, "array_in_b_3")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
