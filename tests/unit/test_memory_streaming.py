"""
Memory Streaming Tests.

This test verifies the memory streaming data paths:
1. BlockMoverIn -> StreamReader -> Scratchpad data path
2. BlockMoverOut -> StreamWriter -> DRAM data path
3. BlockMover engine roundtrip data integrity

The test simulates simplified data flows without actual memory:
- Mock BlockMover responses for BlockMoverIn
- Verify BlockMoverIn writes to scratchpad interface
- Mock Combiner data for BlockMoverOut
- Verify BlockMoverOut sends data through BlockMover interface
"""

import pytest
from amaranth.sim import Simulator, Tick

from systars.blockmover.reader import StreamReader
from systars.blockmover.writer import StreamWriter
from systars.config import SystolicConfig
from systars.controller.blockmover_in import BlockMoverIn
from systars.controller.blockmover_out import BlockMoverOut
from systars.util.commands import BlockMoverOpcode


class TestMemoryStreaming:
    """Integration tests for memory streaming components."""

    @pytest.fixture
    def config(self):
        """Configuration for tests."""
        return SystolicConfig(
            blockmover_buswidth=128,
            grid_rows=2,
            grid_cols=2,
            tile_rows=1,
            tile_cols=1,
        )

    def test_blockmover_in_with_stream_reader(self, config):
        """Test BlockMoverIn issuing commands to StreamReader."""
        bm_in = BlockMoverIn(config)
        # StreamReader would be connected in full integration
        results = {"engine_req": False, "transfer_complete": False}

        def testbench():
            # Issue MEMCPY command to BlockMoverIn
            yield bm_in.cmd_valid.eq(1)
            yield bm_in.cmd_opcode.eq(BlockMoverOpcode.MEMCPY)
            yield bm_in.cmd_dram_addr.eq(0x1000)
            yield bm_in.cmd_sp_addr.eq(0x10)
            yield bm_in.cmd_len.eq(2)
            yield bm_in.cmd_id.eq(1)
            yield Tick()
            yield bm_in.cmd_valid.eq(0)
            yield Tick()

            # BlockMoverIn should issue BlockMover request
            dma_valid = yield bm_in.engine_req_valid
            engine_addr = yield bm_in.engine_req_addr
            if dma_valid:
                results["engine_req"] = True
                assert engine_addr == 0x1000

            # Simulate BlockMover request acceptance
            yield bm_in.engine_req_ready.eq(1)
            yield Tick()
            yield bm_in.engine_req_ready.eq(0)
            yield Tick()

            # Simulate BlockMover response with 2 beats of data
            for i in range(2):
                yield bm_in.engine_resp_valid.eq(1)
                yield bm_in.engine_resp_data.eq(0xABCD0000 + i)
                yield bm_in.engine_resp_last.eq(1 if i == 1 else 0)

                # Check scratchpad write
                sp_write_en = yield bm_in.sp_write_en
                sp_write_addr = yield bm_in.sp_write_addr
                if sp_write_en:
                    assert sp_write_addr == 0x10 + i

                yield Tick()

            yield bm_in.engine_resp_valid.eq(0)

            # Check completion
            completed = yield bm_in.completed
            if completed:
                results["transfer_complete"] = True

            yield Tick()
            yield Tick()

        sim = Simulator(bm_in)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()

        assert results["engine_req"], "BlockMoverIn should issue BlockMover request"
        assert results["transfer_complete"], "Transfer should complete"

    def test_blockmover_out_with_stream_writer(self, config):
        """Test BlockMoverOut issuing commands to StreamWriter."""
        bm_out = BlockMoverOut(config)
        # StreamWriter would be connected in full integration
        results = {"engine_req": False, "data_sent": False, "transfer_complete": False}

        def testbench():
            # Issue MEMCPY command to BlockMoverOut
            yield bm_out.cmd_valid.eq(1)
            yield bm_out.cmd_opcode.eq(BlockMoverOpcode.MEMCPY)
            yield bm_out.cmd_combiner_addr.eq(0x20)
            yield bm_out.cmd_dram_addr.eq(0x2000)
            yield bm_out.cmd_len.eq(2)
            yield bm_out.cmd_id.eq(2)
            yield bm_out.cmd_activation.eq(0)
            yield Tick()
            yield bm_out.cmd_valid.eq(0)
            yield Tick()

            # BlockMoverOut should issue BlockMover request
            dma_valid = yield bm_out.engine_req_valid
            engine_addr = yield bm_out.engine_req_addr
            if dma_valid:
                results["engine_req"] = True
                assert engine_addr == 0x2000

            # Accept BlockMover request
            yield bm_out.engine_req_ready.eq(1)
            yield Tick()
            yield bm_out.engine_req_ready.eq(0)

            # Process 2 beats: read from acc, send to BlockMover
            for i in range(2):
                # Check Combiner read request
                combiner_read_req = yield bm_out.combiner_read_req
                if combiner_read_req:
                    pass  # Request issued
                yield Tick()

                # Provide Combiner data
                yield bm_out.combiner_read_valid.eq(1)
                yield bm_out.combiner_read_data.eq(0x5000 + i)
                yield Tick()
                yield bm_out.combiner_read_valid.eq(0)
                yield Tick()

                # Check BlockMover data output
                engine_data_valid = yield bm_out.engine_data_valid
                engine_data = yield bm_out.engine_data
                if engine_data_valid:
                    results["data_sent"] = True
                    assert engine_data == 0x5000 + i

                # Accept BlockMover data
                yield bm_out.engine_data_ready.eq(1)
                yield Tick()
                yield bm_out.engine_data_ready.eq(0)

            yield Tick()

            # Signal BlockMover done
            yield bm_out.engine_done.eq(1)
            yield Tick()
            yield bm_out.engine_done.eq(0)

            # Check completion
            completed = yield bm_out.completed
            if completed:
                results["transfer_complete"] = True

            yield Tick()

        sim = Simulator(bm_out)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()

        assert results["engine_req"], "BlockMoverOut should issue BlockMover request"
        assert results["data_sent"], "Data should be sent to BlockMover"
        assert results["transfer_complete"], "Transfer should complete"

    def test_dma_engines_roundtrip(self, config):
        """Test data flowing through both BlockMover engines."""
        reader = StreamReader(config)
        writer = StreamWriter(config)
        test_data = [0x1111, 0x2222, 0x3333, 0x4444]
        received_data = []
        sent_data = []

        def reader_testbench():
            """Test StreamReader receiving data from memory."""
            # Issue read request
            yield reader.req_valid.eq(1)
            yield reader.req_addr.eq(0x1000)
            yield reader.req_len.eq(3)  # 4 beats
            yield Tick()
            yield reader.req_valid.eq(0)
            yield Tick()

            # Accept address
            yield reader.mem_arready.eq(1)
            yield Tick()
            yield reader.mem_arready.eq(0)
            yield Tick()

            # Provide data
            yield reader.resp_ready.eq(1)
            for i, data in enumerate(test_data):
                yield reader.mem_rvalid.eq(1)
                yield reader.mem_rdata.eq(data)
                yield reader.mem_rlast.eq(1 if i == len(test_data) - 1 else 0)

                resp_valid = yield reader.resp_valid
                resp_data = yield reader.resp_data
                if resp_valid:
                    received_data.append(resp_data)

                yield Tick()

            yield reader.mem_rvalid.eq(0)
            yield Tick()

        def writer_testbench():
            """Test StreamWriter sending data to memory."""
            # Issue write request
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
            yield Tick()

            # Send data
            yield writer.mem_wready.eq(1)
            for i, data in enumerate(test_data):
                yield writer.data_valid.eq(1)
                yield writer.data.eq(data)
                yield writer.data_last.eq(1 if i == len(test_data) - 1 else 0)

                wvalid = yield writer.mem_wvalid
                wdata = yield writer.mem_wdata
                if wvalid:
                    sent_data.append(wdata)

                yield Tick()

            yield writer.data_valid.eq(0)
            yield Tick()

            # Provide response
            yield writer.mem_bvalid.eq(1)
            yield Tick()
            yield writer.mem_bvalid.eq(0)
            yield Tick()

        # Run reader test
        sim_reader = Simulator(reader)
        sim_reader.add_clock(1e-6)
        sim_reader.add_testbench(reader_testbench)
        sim_reader.run()

        # Run writer test
        sim_writer = Simulator(writer)
        sim_writer.add_clock(1e-6)
        sim_writer.add_testbench(writer_testbench)
        sim_writer.run()

        assert received_data == test_data, f"Reader data mismatch: {received_data}"
        assert sent_data == test_data, f"Writer data mismatch: {sent_data}"

    def test_component_instantiation(self, config):
        """Test that all streaming components can be instantiated together."""
        # BlockMover engines
        reader = StreamReader(config)
        writer = StreamWriter(config)

        # Controllers
        bm_in = BlockMoverIn(config)
        bm_out = BlockMoverOut(config)

        # All should have correct config
        assert reader.config == config
        assert writer.config == config
        assert bm_in.config == config
        assert bm_out.config == config


class TestMemoryStreamingVerilogGeneration:
    """Test Verilog generation for memory streaming components."""

    def test_generate_all_components(self, tmp_path):
        """Test that all streaming components generate valid Verilog."""
        from amaranth._toolchain.yosys import find_yosys
        from amaranth.back import verilog

        try:
            find_yosys(lambda ver: ver >= (0, 40))
        except Exception:
            pytest.skip("Yosys not found")

        config = SystolicConfig(
            blockmover_buswidth=128,
            grid_rows=2,
            grid_cols=2,
        )

        components = [
            ("StreamReader", StreamReader(config)),
            ("StreamWriter", StreamWriter(config)),
            ("BlockMoverIn", BlockMoverIn(config)),
            ("BlockMoverOut", BlockMoverOut(config)),
        ]

        for name, component in components:
            output = verilog.convert(component, name=name)
            assert f"module {name}" in output

            verilog_file = tmp_path / f"{name.lower()}.v"
            verilog_file.write_text(output)
            assert verilog_file.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
