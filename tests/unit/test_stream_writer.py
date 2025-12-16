"""
Unit tests for the StreamWriter DMA engine.

These tests verify:
1. StreamWriter instantiation
2. Request acceptance
3. AXI address channel handshake
4. AXI data channel transmission
5. AXI response handling
6. Burst completion
7. Verilog generation
"""

import pytest
from amaranth.sim import Simulator, Tick

from systars.config import SystolicConfig
from systars.dma.writer import StreamWriter


class TestStreamWriter:
    """Test suite for StreamWriter."""

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
    def writer(self, config):
        """Create a StreamWriter instance."""
        return StreamWriter(config)

    def test_instantiation(self, writer, config):
        """Test that StreamWriter can be instantiated."""
        assert writer is not None
        assert writer.config == config

    def test_has_correct_ports(self, writer):
        """Test that StreamWriter has all required ports."""
        # Request interface
        assert hasattr(writer, "req_valid")
        assert hasattr(writer, "req_ready")
        assert hasattr(writer, "req_addr")
        assert hasattr(writer, "req_len")

        # Data interface
        assert hasattr(writer, "data_valid")
        assert hasattr(writer, "data_ready")
        assert hasattr(writer, "data")
        assert hasattr(writer, "data_last")

        # Status
        assert hasattr(writer, "busy")
        assert hasattr(writer, "done")

        # AXI AW channel
        assert hasattr(writer, "mem_awvalid")
        assert hasattr(writer, "mem_awready")
        assert hasattr(writer, "mem_awaddr")
        assert hasattr(writer, "mem_awlen")

        # AXI W channel
        assert hasattr(writer, "mem_wvalid")
        assert hasattr(writer, "mem_wready")
        assert hasattr(writer, "mem_wdata")
        assert hasattr(writer, "mem_wstrb")
        assert hasattr(writer, "mem_wlast")

        # AXI B channel
        assert hasattr(writer, "mem_bvalid")
        assert hasattr(writer, "mem_bready")
        assert hasattr(writer, "mem_bresp")

    def test_idle_ready(self, config):
        """Test that StreamWriter is ready in idle state."""
        writer = StreamWriter(config)
        results = []

        def testbench():
            # Check initial state
            ready = yield writer.req_ready
            busy = yield writer.busy
            results.append({"ready": ready, "busy": busy})
            yield Tick()

        sim = Simulator(writer)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()

        assert results[0]["ready"] == 1, "Should be ready in idle state"
        assert results[0]["busy"] == 0, "Should not be busy in idle state"

    def test_single_beat_write(self, config):
        """Test single-beat write transaction."""
        writer = StreamWriter(config)
        results = {"addr_issued": False, "data_sent": False, "done": False}

        def testbench():
            # Issue request for single beat
            yield writer.req_valid.eq(1)
            yield writer.req_addr.eq(0x1000)
            yield writer.req_len.eq(0)  # 1 beat
            yield Tick()

            # Request captured, state transitions to ADDR
            yield writer.req_valid.eq(0)
            yield Tick()

            # Should see address on AW channel
            awvalid = yield writer.mem_awvalid
            awaddr = yield writer.mem_awaddr
            awlen = yield writer.mem_awlen
            if awvalid:
                results["addr_issued"] = True
                assert awaddr == 0x1000
                assert awlen == 0

            # Accept address - state will transition to DATA
            yield writer.mem_awready.eq(1)
            yield Tick()
            yield writer.mem_awready.eq(0)
            yield Tick()  # Let state transition settle

            # Now in DATA state - provide data
            yield writer.data_valid.eq(1)
            yield writer.data.eq(0xCAFEBABE)
            yield writer.data_last.eq(1)
            yield writer.mem_wready.eq(1)

            # Check W channel BEFORE Tick (combinational passthrough)
            # After Tick, state transitions to RESP on data_last
            wvalid = yield writer.mem_wvalid
            wdata = yield writer.mem_wdata
            wlast = yield writer.mem_wlast
            if wvalid:
                results["data_sent"] = True
                assert wdata == 0xCAFEBABE
                assert wlast == 1

            yield Tick()  # State transitions to RESP

            yield writer.data_valid.eq(0)
            yield Tick()
            yield Tick()  # State transitions to RESP

            # Now in RESP state - provide response
            yield writer.mem_bvalid.eq(1)
            yield writer.mem_bresp.eq(0)  # OKAY

            # Check done BEFORE Tick - it's combinational in RESP state
            # After Tick, state transitions to IDLE and done goes low
            done = yield writer.done
            if done:
                results["done"] = True

            yield Tick()  # State transitions to IDLE

            yield writer.mem_bvalid.eq(0)
            yield Tick()
            yield Tick()

            # Should be back to idle
            ready = yield writer.req_ready
            busy = yield writer.busy
            assert ready == 1
            assert busy == 0

        sim = Simulator(writer)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()

        assert results["addr_issued"], "Address should have been issued"
        assert results["data_sent"], "Data should have been sent"
        assert results["done"], "Done should have been signaled"

    def test_multi_beat_burst(self, config):
        """Test multi-beat burst write transaction."""
        writer = StreamWriter(config)
        beat_count = [0]

        def testbench():
            # Issue request for 4 beats
            yield writer.req_valid.eq(1)
            yield writer.req_addr.eq(0x2000)
            yield writer.req_len.eq(3)  # 4 beats
            yield Tick()

            yield writer.req_valid.eq(0)
            yield Tick()

            # Accept address
            yield writer.mem_awready.eq(1)
            yield Tick()
            yield writer.mem_awready.eq(0)
            yield Tick()  # Let state transition settle

            # Now in DATA state - send 4 data beats
            yield writer.mem_wready.eq(1)
            for i in range(4):
                yield writer.data_valid.eq(1)
                yield writer.data.eq(0x1000 + i)
                yield writer.data_last.eq(1 if i == 3 else 0)

                # Check wvalid BEFORE Tick (combinational passthrough)
                wvalid = yield writer.mem_wvalid
                if wvalid:
                    beat_count[0] += 1

                yield Tick()

            yield writer.data_valid.eq(0)
            yield Tick()

            # Provide response
            yield writer.mem_bvalid.eq(1)
            yield Tick()
            yield writer.mem_bvalid.eq(0)
            yield Tick()

        sim = Simulator(writer)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()

        assert beat_count[0] == 4, f"Expected 4 beats, got {beat_count[0]}"

    def test_busy_signal(self, config):
        """Test that busy signal is asserted during transaction."""
        writer = StreamWriter(config)
        busy_states = []

        def testbench():
            # Check idle
            busy = yield writer.busy
            busy_states.append(("idle", busy))
            yield Tick()

            # Issue request
            yield writer.req_valid.eq(1)
            yield writer.req_addr.eq(0x3000)
            yield writer.req_len.eq(0)
            yield Tick()
            yield writer.req_valid.eq(0)
            yield Tick()

            # Check busy during transaction
            busy = yield writer.busy
            busy_states.append(("addr", busy))

            # Complete transaction
            yield writer.mem_awready.eq(1)
            yield Tick()
            yield writer.mem_awready.eq(0)
            yield Tick()

            yield writer.data_valid.eq(1)
            yield writer.data_last.eq(1)
            yield writer.mem_wready.eq(1)
            yield Tick()
            yield writer.data_valid.eq(0)
            yield Tick()

            yield writer.mem_bvalid.eq(1)
            yield Tick()
            yield writer.mem_bvalid.eq(0)
            yield Tick()

            # Check idle again
            busy = yield writer.busy
            busy_states.append(("done", busy))

        sim = Simulator(writer)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()

        assert busy_states[0] == ("idle", 0), "Should not be busy in idle"
        assert busy_states[1] == ("addr", 1), "Should be busy during addr phase"
        assert busy_states[2] == ("done", 0), "Should not be busy after done"

    def test_write_strobe(self, config):
        """Test that write strobes are set correctly."""
        writer = StreamWriter(config)
        strb_width = config.dma_buswidth // 8
        expected_strb = (1 << strb_width) - 1
        results = {"strb": None}

        def testbench():
            yield writer.req_valid.eq(1)
            yield writer.req_addr.eq(0x4000)
            yield writer.req_len.eq(0)
            yield Tick()
            yield writer.req_valid.eq(0)
            yield Tick()

            yield writer.mem_awready.eq(1)
            yield Tick()
            yield writer.mem_awready.eq(0)
            yield Tick()  # Let state transition settle

            # Now in DATA state
            yield writer.data_valid.eq(1)
            yield writer.data_last.eq(1)
            yield writer.mem_wready.eq(1)

            # Strobe is combinational output in DATA state - check BEFORE Tick
            results["strb"] = yield writer.mem_wstrb

            yield Tick()  # State transitions to RESP

            yield writer.data_valid.eq(0)
            yield Tick()

        sim = Simulator(writer)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()

        assert results["strb"] == expected_strb, f"Expected strb={expected_strb:#x}"

    def test_elaboration(self, writer):
        """Test that StreamWriter elaborates without errors."""

        def testbench():
            yield Tick()

        sim = Simulator(writer)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()


class TestStreamWriterVerilogGeneration:
    """Test Verilog generation for StreamWriter."""

    def test_generate_verilog(self, tmp_path):
        """Test that StreamWriter can generate valid Verilog."""
        from amaranth._toolchain.yosys import find_yosys
        from amaranth.back import verilog

        try:
            find_yosys(lambda ver: ver >= (0, 40))
        except Exception:
            pytest.skip("Yosys not found")

        config = SystolicConfig(dma_buswidth=128, dma_maxbytes=64)
        writer = StreamWriter(config)

        output = verilog.convert(writer, name="StreamWriter")
        assert "module StreamWriter" in output
        assert "req_valid" in output
        assert "mem_awvalid" in output
        assert "mem_wdata" in output
        assert "mem_bvalid" in output

        verilog_file = tmp_path / "stream_writer.v"
        verilog_file.write_text(output)
        assert verilog_file.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
