"""
Memory Streaming Tests.

This test verifies the memory streaming data paths:
1. LoadController -> StreamReader -> Scratchpad data path
2. StoreController -> StreamWriter -> DRAM data path
3. DMA engine roundtrip data integrity

The test simulates simplified data flows without actual memory:
- Mock DMA responses for LoadController
- Verify LoadController writes to scratchpad interface
- Mock accumulator data for StoreController
- Verify StoreController sends data through DMA interface
"""

import pytest
from amaranth.sim import Simulator, Tick

from systars.config import SystolicConfig
from systars.controller.load import LoadController
from systars.controller.store import StoreController
from systars.dma.reader import StreamReader
from systars.dma.writer import StreamWriter
from systars.util.commands import DmaOpcode


class TestMemoryStreaming:
    """Integration tests for memory streaming components."""

    @pytest.fixture
    def config(self):
        """Configuration for tests."""
        return SystolicConfig(
            dma_buswidth=128,
            dma_maxbytes=64,
            grid_rows=2,
            grid_cols=2,
            tile_rows=1,
            tile_cols=1,
        )

    def test_load_controller_with_stream_reader(self, config):
        """Test LoadController issuing commands to StreamReader."""
        load_ctrl = LoadController(config)
        # StreamReader would be connected in full integration
        results = {"dma_req": False, "transfer_complete": False}

        def testbench():
            # Issue MEMCPY command to LoadController
            yield load_ctrl.cmd_valid.eq(1)
            yield load_ctrl.cmd_opcode.eq(DmaOpcode.MEMCPY)
            yield load_ctrl.cmd_dram_addr.eq(0x1000)
            yield load_ctrl.cmd_sp_addr.eq(0x10)
            yield load_ctrl.cmd_len.eq(2)
            yield load_ctrl.cmd_id.eq(1)
            yield Tick()
            yield load_ctrl.cmd_valid.eq(0)
            yield Tick()

            # LoadController should issue DMA request
            dma_valid = yield load_ctrl.dma_req_valid
            dma_addr = yield load_ctrl.dma_req_addr
            if dma_valid:
                results["dma_req"] = True
                assert dma_addr == 0x1000

            # Simulate DMA request acceptance
            yield load_ctrl.dma_req_ready.eq(1)
            yield Tick()
            yield load_ctrl.dma_req_ready.eq(0)
            yield Tick()

            # Simulate DMA response with 2 beats of data
            for i in range(2):
                yield load_ctrl.dma_resp_valid.eq(1)
                yield load_ctrl.dma_resp_data.eq(0xABCD0000 + i)
                yield load_ctrl.dma_resp_last.eq(1 if i == 1 else 0)

                # Check scratchpad write
                sp_write_en = yield load_ctrl.sp_write_en
                sp_write_addr = yield load_ctrl.sp_write_addr
                if sp_write_en:
                    assert sp_write_addr == 0x10 + i

                yield Tick()

            yield load_ctrl.dma_resp_valid.eq(0)

            # Check completion
            completed = yield load_ctrl.completed
            if completed:
                results["transfer_complete"] = True

            yield Tick()
            yield Tick()

        sim = Simulator(load_ctrl)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()

        assert results["dma_req"], "LoadController should issue DMA request"
        assert results["transfer_complete"], "Transfer should complete"

    def test_store_controller_with_stream_writer(self, config):
        """Test StoreController issuing commands to StreamWriter."""
        store_ctrl = StoreController(config)
        # StreamWriter would be connected in full integration
        results = {"dma_req": False, "data_sent": False, "transfer_complete": False}

        def testbench():
            # Issue MEMCPY command to StoreController
            yield store_ctrl.cmd_valid.eq(1)
            yield store_ctrl.cmd_opcode.eq(DmaOpcode.MEMCPY)
            yield store_ctrl.cmd_acc_addr.eq(0x20)
            yield store_ctrl.cmd_dram_addr.eq(0x2000)
            yield store_ctrl.cmd_len.eq(2)
            yield store_ctrl.cmd_id.eq(2)
            yield store_ctrl.cmd_activation.eq(0)
            yield Tick()
            yield store_ctrl.cmd_valid.eq(0)
            yield Tick()

            # StoreController should issue DMA request
            dma_valid = yield store_ctrl.dma_req_valid
            dma_addr = yield store_ctrl.dma_req_addr
            if dma_valid:
                results["dma_req"] = True
                assert dma_addr == 0x2000

            # Accept DMA request
            yield store_ctrl.dma_req_ready.eq(1)
            yield Tick()
            yield store_ctrl.dma_req_ready.eq(0)

            # Process 2 beats: read from acc, send to DMA
            for i in range(2):
                # Check accumulator read request
                acc_read_req = yield store_ctrl.acc_read_req
                if acc_read_req:
                    pass  # Request issued
                yield Tick()

                # Provide accumulator data
                yield store_ctrl.acc_read_valid.eq(1)
                yield store_ctrl.acc_read_data.eq(0x5000 + i)
                yield Tick()
                yield store_ctrl.acc_read_valid.eq(0)
                yield Tick()

                # Check DMA data output
                dma_data_valid = yield store_ctrl.dma_data_valid
                dma_data = yield store_ctrl.dma_data
                if dma_data_valid:
                    results["data_sent"] = True
                    assert dma_data == 0x5000 + i

                # Accept DMA data
                yield store_ctrl.dma_data_ready.eq(1)
                yield Tick()
                yield store_ctrl.dma_data_ready.eq(0)

            yield Tick()

            # Signal DMA done
            yield store_ctrl.dma_done.eq(1)
            yield Tick()
            yield store_ctrl.dma_done.eq(0)

            # Check completion
            completed = yield store_ctrl.completed
            if completed:
                results["transfer_complete"] = True

            yield Tick()

        sim = Simulator(store_ctrl)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()

        assert results["dma_req"], "StoreController should issue DMA request"
        assert results["data_sent"], "Data should be sent to DMA"
        assert results["transfer_complete"], "Transfer should complete"

    def test_dma_engines_roundtrip(self, config):
        """Test data flowing through both DMA engines."""
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
        # DMA engines
        reader = StreamReader(config)
        writer = StreamWriter(config)

        # Controllers
        load_ctrl = LoadController(config)
        store_ctrl = StoreController(config)

        # All should have correct config
        assert reader.config == config
        assert writer.config == config
        assert load_ctrl.config == config
        assert store_ctrl.config == config


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
            dma_buswidth=128,
            dma_maxbytes=64,
            grid_rows=2,
            grid_cols=2,
        )

        components = [
            ("StreamReader", StreamReader(config)),
            ("StreamWriter", StreamWriter(config)),
            ("LoadController", LoadController(config)),
            ("StoreController", StoreController(config)),
        ]

        for name, component in components:
            output = verilog.convert(component, name=name)
            assert f"module {name}" in output

            verilog_file = tmp_path / f"{name.lower()}.v"
            verilog_file.write_text(output)
            assert verilog_file.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
