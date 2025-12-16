"""
Unit tests for the StreamReader DMA engine.

These tests verify:
1. StreamReader instantiation
2. Request acceptance
3. AXI address channel handshake
4. AXI data channel forwarding
5. Burst completion
6. Verilog generation
"""

import pytest
from amaranth.sim import Simulator, Tick

from systars.config import SystolicConfig
from systars.dma.reader import StreamReader


class TestStreamReader:
    """Test suite for StreamReader."""

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
    def reader(self, config):
        """Create a StreamReader instance."""
        return StreamReader(config)

    def test_instantiation(self, reader, config):
        """Test that StreamReader can be instantiated."""
        assert reader is not None
        assert reader.config == config

    def test_has_correct_ports(self, reader):
        """Test that StreamReader has all required ports."""
        # Request interface
        assert hasattr(reader, "req_valid")
        assert hasattr(reader, "req_ready")
        assert hasattr(reader, "req_addr")
        assert hasattr(reader, "req_len")

        # Response interface
        assert hasattr(reader, "resp_valid")
        assert hasattr(reader, "resp_ready")
        assert hasattr(reader, "resp_data")
        assert hasattr(reader, "resp_last")

        # AXI AR channel
        assert hasattr(reader, "mem_arvalid")
        assert hasattr(reader, "mem_arready")
        assert hasattr(reader, "mem_araddr")
        assert hasattr(reader, "mem_arlen")

        # AXI R channel
        assert hasattr(reader, "mem_rvalid")
        assert hasattr(reader, "mem_rready")
        assert hasattr(reader, "mem_rdata")
        assert hasattr(reader, "mem_rlast")

    def test_idle_ready(self, config):
        """Test that StreamReader is ready in idle state."""
        reader = StreamReader(config)
        results = []

        def testbench():
            # Check initial state
            ready = yield reader.req_ready
            results.append(ready)
            yield Tick()

        sim = Simulator(reader)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()

        assert results[0] == 1, "Should be ready in idle state"

    def test_single_beat_read(self, config):
        """Test single-beat read transaction."""
        reader = StreamReader(config)
        results = {"addr_issued": False, "data_received": False}

        def testbench():
            # Issue request for single beat (len=0)
            yield reader.req_valid.eq(1)
            yield reader.req_addr.eq(0x1000)
            yield reader.req_len.eq(0)  # 1 beat
            yield Tick()

            # Request captured, state transitions to ADDR
            yield reader.req_valid.eq(0)
            yield Tick()

            # Should see address on AR channel
            arvalid = yield reader.mem_arvalid
            araddr = yield reader.mem_araddr
            arlen = yield reader.mem_arlen
            if arvalid:
                results["addr_issued"] = True
                assert araddr == 0x1000
                assert arlen == 0

            # Accept address - state will transition to DATA
            yield reader.mem_arready.eq(1)
            yield Tick()
            yield reader.mem_arready.eq(0)
            yield Tick()  # Let state transition settle

            # Now in DATA state - provide data on R channel
            yield reader.mem_rvalid.eq(1)
            yield reader.mem_rdata.eq(0xDEADBEEF)
            yield reader.mem_rlast.eq(1)
            yield reader.resp_ready.eq(1)

            # Check response BEFORE Tick - outputs are combinational passthrough
            # and after Tick the state will transition to IDLE on rlast
            resp_valid = yield reader.resp_valid
            resp_data = yield reader.resp_data
            resp_last = yield reader.resp_last
            if resp_valid:
                results["data_received"] = True
                assert resp_data == 0xDEADBEEF
                assert resp_last == 1

            yield Tick()  # State transitions to IDLE on rlast

            yield reader.mem_rvalid.eq(0)
            yield Tick()
            yield Tick()

            # Should be back to idle
            ready = yield reader.req_ready
            assert ready == 1

        sim = Simulator(reader)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()

        assert results["addr_issued"], "Address should have been issued"
        assert results["data_received"], "Data should have been received"

    def test_multi_beat_burst(self, config):
        """Test multi-beat burst read transaction."""
        reader = StreamReader(config)
        beat_count = [0]

        def testbench():
            # Issue request for 4 beats (len=3)
            yield reader.req_valid.eq(1)
            yield reader.req_addr.eq(0x2000)
            yield reader.req_len.eq(3)  # 4 beats
            yield Tick()

            yield reader.req_valid.eq(0)
            yield Tick()

            # Accept address
            yield reader.mem_arready.eq(1)
            yield Tick()
            yield reader.mem_arready.eq(0)
            yield Tick()  # Let state transition settle

            # Now in DATA state - provide 4 data beats
            yield reader.resp_ready.eq(1)
            for i in range(4):
                yield reader.mem_rvalid.eq(1)
                yield reader.mem_rdata.eq(0x1000 + i)
                yield reader.mem_rlast.eq(1 if i == 3 else 0)

                # Check resp_valid BEFORE Tick (combinational passthrough)
                resp_valid = yield reader.resp_valid
                if resp_valid:
                    beat_count[0] += 1

                yield Tick()

            yield reader.mem_rvalid.eq(0)
            yield Tick()

        sim = Simulator(reader)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()

        assert beat_count[0] == 4, f"Expected 4 beats, got {beat_count[0]}"

    def test_backpressure(self, config):
        """Test that reader handles backpressure correctly."""
        reader = StreamReader(config)
        results = {"stalled": False, "resumed": False}

        def testbench():
            # Issue request
            yield reader.req_valid.eq(1)
            yield reader.req_addr.eq(0x3000)
            yield reader.req_len.eq(1)  # 2 beats
            yield Tick()
            yield reader.req_valid.eq(0)
            yield Tick()

            # Accept address
            yield reader.mem_arready.eq(1)
            yield Tick()
            yield reader.mem_arready.eq(0)
            yield Tick()

            # Provide first beat, but don't accept it
            yield reader.mem_rvalid.eq(1)
            yield reader.mem_rdata.eq(0xAAAA)
            yield reader.mem_rlast.eq(0)
            yield reader.resp_ready.eq(0)  # Backpressure
            yield Tick()

            # Check rready is low (backpressure propagates)
            rready = yield reader.mem_rready
            if rready == 0:
                results["stalled"] = True

            # Now accept
            yield reader.resp_ready.eq(1)
            yield Tick()

            rready = yield reader.mem_rready
            if rready == 1:
                results["resumed"] = True

            # Complete burst
            yield reader.mem_rlast.eq(1)
            yield Tick()
            yield reader.mem_rvalid.eq(0)
            yield Tick()

        sim = Simulator(reader)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()

        assert results["stalled"], "Should stall on backpressure"
        assert results["resumed"], "Should resume when ready"

    def test_elaboration(self, reader):
        """Test that StreamReader elaborates without errors."""

        def testbench():
            yield Tick()

        sim = Simulator(reader)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()


class TestStreamReaderVerilogGeneration:
    """Test Verilog generation for StreamReader."""

    def test_generate_verilog(self, tmp_path):
        """Test that StreamReader can generate valid Verilog."""
        from amaranth._toolchain.yosys import find_yosys
        from amaranth.back import verilog

        try:
            find_yosys(lambda ver: ver >= (0, 40))
        except Exception:
            pytest.skip("Yosys not found")

        config = SystolicConfig(dma_buswidth=128, dma_maxbytes=64)
        reader = StreamReader(config)

        output = verilog.convert(reader, name="StreamReader")
        assert "module StreamReader" in output
        assert "req_valid" in output
        assert "mem_arvalid" in output
        assert "mem_rdata" in output

        verilog_file = tmp_path / "stream_reader.v"
        verilog_file.write_text(output)
        assert verilog_file.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
