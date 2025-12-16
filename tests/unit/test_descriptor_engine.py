"""
Unit tests for the DescriptorEngine.

These tests verify:
1. DescriptorEngine instantiation
2. Idle state behavior
3. Descriptor fetch sequence
4. MEMCPY execution
5. FILL execution
6. FENCE/NOP handling
7. Descriptor chain following
8. Completion and interrupt signaling
9. Verilog generation
"""

import pytest
from amaranth.sim import Simulator, Tick

from systars.config import SystolicConfig
from systars.dma import DescriptorEngine
from systars.util.commands import (
    DmaDescriptor,
    DmaFlags,
    DmaOpcode,
)


class TestDescriptorEngine:
    """Test suite for DescriptorEngine."""

    @pytest.fixture
    def config(self):
        """Configuration for tests."""
        return SystolicConfig(
            dma_buswidth=128,  # 16 bytes per beat
            dma_maxbytes=64,
            grid_rows=4,
            grid_cols=4,
        )

    @pytest.fixture
    def engine(self, config):
        """Create a DescriptorEngine instance."""
        return DescriptorEngine(config)

    def test_instantiation(self, engine, config):
        """Test that DescriptorEngine can be instantiated."""
        assert engine is not None
        assert engine.config == config

    def test_has_correct_ports(self, engine):
        """Test that DescriptorEngine has all required ports."""
        # Command interface
        assert hasattr(engine, "start")
        assert hasattr(engine, "desc_addr")
        assert hasattr(engine, "busy")
        assert hasattr(engine, "done")
        assert hasattr(engine, "error")

        # Descriptor fetch interface
        assert hasattr(engine, "fetch_req_valid")
        assert hasattr(engine, "fetch_req_ready")
        assert hasattr(engine, "fetch_req_addr")
        assert hasattr(engine, "fetch_resp_valid")
        assert hasattr(engine, "fetch_resp_data")
        assert hasattr(engine, "fetch_resp_last")

        # Data read interface
        assert hasattr(engine, "read_req_valid")
        assert hasattr(engine, "read_req_ready")
        assert hasattr(engine, "read_req_addr")
        assert hasattr(engine, "read_resp_valid")
        assert hasattr(engine, "read_resp_data")
        assert hasattr(engine, "read_resp_last")

        # Data write interface
        assert hasattr(engine, "write_req_valid")
        assert hasattr(engine, "write_req_ready")
        assert hasattr(engine, "write_req_addr")
        assert hasattr(engine, "write_data_valid")
        assert hasattr(engine, "write_data_ready")
        assert hasattr(engine, "write_data")
        assert hasattr(engine, "write_data_last")
        assert hasattr(engine, "write_done")

        # Status and interrupt
        assert hasattr(engine, "status_write_valid")
        assert hasattr(engine, "interrupt")

    def test_idle_state(self, config):
        """Test that DescriptorEngine starts in idle state."""
        engine = DescriptorEngine(config)
        results = {"busy": None, "done": None}

        def testbench():
            results["busy"] = yield engine.busy
            results["done"] = yield engine.done
            yield Tick()

        sim = Simulator(engine)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()

        assert results["busy"] == 0, "Should not be busy in idle"
        assert results["done"] == 0, "Should not signal done in idle"

    def test_start_triggers_fetch(self, config):
        """Test that start signal triggers descriptor fetch."""
        engine = DescriptorEngine(config)
        results = {"fetch_req": False, "fetch_addr": None}

        def testbench():
            # Issue start command
            yield engine.start.eq(1)
            yield engine.desc_addr.eq(0x1000)
            yield Tick()
            yield engine.start.eq(0)
            yield Tick()

            # Should see fetch request
            fetch_valid = yield engine.fetch_req_valid
            fetch_addr = yield engine.fetch_req_addr
            if fetch_valid:
                results["fetch_req"] = True
                results["fetch_addr"] = fetch_addr

            yield Tick()

        sim = Simulator(engine)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()

        assert results["fetch_req"], "Should issue fetch request after start"
        assert results["fetch_addr"] == 0x1000, "Fetch address should match desc_addr"

    def test_busy_during_operation(self, config):
        """Test that busy is asserted during operation."""
        engine = DescriptorEngine(config)
        busy_states = []

        def testbench():
            # Check idle
            busy = yield engine.busy
            busy_states.append(("idle", busy))
            yield Tick()

            # Start operation
            yield engine.start.eq(1)
            yield engine.desc_addr.eq(0x2000)
            yield Tick()
            yield engine.start.eq(0)
            yield Tick()

            # Check busy during fetch
            busy = yield engine.busy
            busy_states.append(("fetching", busy))

            yield Tick()
            yield Tick()

        sim = Simulator(engine)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()

        assert busy_states[0] == ("idle", 0), "Should not be busy in idle"
        assert busy_states[1] == ("fetching", 1), "Should be busy during fetch"

    def test_nop_descriptor_completion(self, config):
        """Test that NOP descriptor completes quickly."""
        engine = DescriptorEngine(config)
        buswidth = config.dma_buswidth
        bytes_per_beat = buswidth // 8
        desc_beats = 64 // bytes_per_beat  # 4 beats for 128-bit bus

        # Create NOP descriptor
        desc = DmaDescriptor(opcode=DmaOpcode.NOP)
        desc_bytes = desc.to_bytes()

        results = {"done": False}

        def testbench():
            # Start with NOP descriptor
            yield engine.start.eq(1)
            yield engine.desc_addr.eq(0x1000)
            yield Tick()
            yield engine.start.eq(0)
            yield Tick()

            # Accept fetch request
            yield engine.fetch_req_ready.eq(1)
            yield Tick()
            yield engine.fetch_req_ready.eq(0)
            yield Tick()

            # Provide descriptor data (4 beats for 128-bit bus)
            for i in range(desc_beats):
                yield engine.fetch_resp_valid.eq(1)
                # Extract 16 bytes per beat
                start = i * bytes_per_beat
                end = start + bytes_per_beat
                beat_data = int.from_bytes(desc_bytes[start:end], "little")
                yield engine.fetch_resp_data.eq(beat_data)
                yield engine.fetch_resp_last.eq(1 if i == desc_beats - 1 else 0)
                yield Tick()

            yield engine.fetch_resp_valid.eq(0)

            # Wait for state machine to process through PARSE_DESC -> CHECK_CHAIN -> DONE
            for _ in range(10):
                done = yield engine.done
                if done:
                    results["done"] = True
                    break
                yield Tick()

        sim = Simulator(engine)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()

        assert results["done"], "NOP should complete"

    def test_fence_descriptor_completion(self, config):
        """Test that FENCE descriptor completes."""
        engine = DescriptorEngine(config)
        buswidth = config.dma_buswidth
        bytes_per_beat = buswidth // 8
        desc_beats = 64 // bytes_per_beat

        # Create FENCE descriptor
        desc = DmaDescriptor(opcode=DmaOpcode.FENCE)
        desc_bytes = desc.to_bytes()

        results = {"done": False}

        def testbench():
            # Start with FENCE descriptor
            yield engine.start.eq(1)
            yield engine.desc_addr.eq(0x1000)
            yield Tick()
            yield engine.start.eq(0)
            yield Tick()

            # Accept fetch request
            yield engine.fetch_req_ready.eq(1)
            yield Tick()
            yield engine.fetch_req_ready.eq(0)
            yield Tick()

            # Provide descriptor data
            for i in range(desc_beats):
                yield engine.fetch_resp_valid.eq(1)
                start = i * bytes_per_beat
                end = start + bytes_per_beat
                beat_data = int.from_bytes(desc_bytes[start:end], "little")
                yield engine.fetch_resp_data.eq(beat_data)
                yield engine.fetch_resp_last.eq(1 if i == desc_beats - 1 else 0)
                yield Tick()

            yield engine.fetch_resp_valid.eq(0)

            # Wait for state machine to process
            for _ in range(10):
                done = yield engine.done
                if done:
                    results["done"] = True
                    break
                yield Tick()

        sim = Simulator(engine)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()

        assert results["done"], "FENCE should complete"

    def test_memcpy_issues_read_write(self, config):
        """Test that MEMCPY issues both read and write requests."""
        engine = DescriptorEngine(config)
        buswidth = config.dma_buswidth
        bytes_per_beat = buswidth // 8
        desc_beats = 64 // bytes_per_beat

        # Create MEMCPY descriptor: copy 32 bytes from 0x1000 to 0x2000
        desc = DmaDescriptor(
            opcode=DmaOpcode.MEMCPY,
            length=32,
            src_addr=0x1000,
            dst_addr=0x2000,
        )
        desc_bytes = desc.to_bytes()

        results = {"read_req": False, "write_req": False, "read_addr": None, "write_addr": None}

        def testbench():
            # Start with MEMCPY descriptor
            yield engine.start.eq(1)
            yield engine.desc_addr.eq(0x3000)
            yield Tick()
            yield engine.start.eq(0)
            yield Tick()

            # Accept fetch request
            yield engine.fetch_req_ready.eq(1)
            yield Tick()
            yield engine.fetch_req_ready.eq(0)
            yield Tick()

            # Provide descriptor data
            for i in range(desc_beats):
                yield engine.fetch_resp_valid.eq(1)
                start = i * bytes_per_beat
                end = start + bytes_per_beat
                beat_data = int.from_bytes(desc_bytes[start:end], "little")
                yield engine.fetch_resp_data.eq(beat_data)
                yield engine.fetch_resp_last.eq(1 if i == desc_beats - 1 else 0)
                yield Tick()

            yield engine.fetch_resp_valid.eq(0)
            yield Tick()
            yield Tick()  # Parse

            # Check for read and write requests
            for _ in range(5):
                read_valid = yield engine.read_req_valid
                write_valid = yield engine.write_req_valid
                if read_valid:
                    results["read_req"] = True
                    results["read_addr"] = yield engine.read_req_addr
                if write_valid:
                    results["write_req"] = True
                    results["write_addr"] = yield engine.write_req_addr
                yield Tick()

        sim = Simulator(engine)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()

        assert results["read_req"], "MEMCPY should issue read request"
        assert results["write_req"], "MEMCPY should issue write request"
        assert results["read_addr"] == 0x1000, (
            f"Read addr should be 0x1000, got {results['read_addr']}"
        )
        assert results["write_addr"] == 0x2000, (
            f"Write addr should be 0x2000, got {results['write_addr']}"
        )

    def test_fill_issues_write_only(self, config):
        """Test that FILL issues write request but not read."""
        engine = DescriptorEngine(config)
        buswidth = config.dma_buswidth
        bytes_per_beat = buswidth // 8
        desc_beats = 64 // bytes_per_beat

        # Create FILL descriptor: fill 32 bytes at 0x2000 with pattern
        desc = DmaDescriptor(
            opcode=DmaOpcode.FILL,
            flags=DmaFlags.SRC_IS_PATTERN,
            length=32,
            src_addr=0xDEADBEEF,  # Pattern
            dst_addr=0x2000,
        )
        desc_bytes = desc.to_bytes()

        results = {"read_req": False, "write_req": False, "write_addr": None}

        def testbench():
            # Start with FILL descriptor
            yield engine.start.eq(1)
            yield engine.desc_addr.eq(0x3000)
            yield Tick()
            yield engine.start.eq(0)
            yield Tick()

            # Accept fetch request
            yield engine.fetch_req_ready.eq(1)
            yield Tick()
            yield engine.fetch_req_ready.eq(0)
            yield Tick()

            # Provide descriptor data
            for i in range(desc_beats):
                yield engine.fetch_resp_valid.eq(1)
                start = i * bytes_per_beat
                end = start + bytes_per_beat
                beat_data = int.from_bytes(desc_bytes[start:end], "little")
                yield engine.fetch_resp_data.eq(beat_data)
                yield engine.fetch_resp_last.eq(1 if i == desc_beats - 1 else 0)
                yield Tick()

            yield engine.fetch_resp_valid.eq(0)
            yield Tick()
            yield Tick()  # Parse

            # Check for write request (not read)
            for _ in range(5):
                read_valid = yield engine.read_req_valid
                write_valid = yield engine.write_req_valid
                if read_valid:
                    results["read_req"] = True
                if write_valid:
                    results["write_req"] = True
                    results["write_addr"] = yield engine.write_req_addr
                yield Tick()

        sim = Simulator(engine)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()

        assert not results["read_req"], "FILL should NOT issue read request"
        assert results["write_req"], "FILL should issue write request"
        assert results["write_addr"] == 0x2000, "Write addr should be 0x2000"

    def test_chained_descriptors(self, config):
        """Test that CHAIN flag causes fetch of next descriptor."""
        engine = DescriptorEngine(config)
        buswidth = config.dma_buswidth
        bytes_per_beat = buswidth // 8
        desc_beats = 64 // bytes_per_beat

        # Create first descriptor with CHAIN flag pointing to second
        desc1 = DmaDescriptor(
            opcode=DmaOpcode.NOP,
            flags=DmaFlags.CHAIN,
            next_desc=0x2000,  # Next descriptor at 0x2000
        )
        desc1_bytes = desc1.to_bytes()

        fetch_addresses = []

        def testbench():
            # Start with first descriptor
            yield engine.start.eq(1)
            yield engine.desc_addr.eq(0x1000)
            yield Tick()
            yield engine.start.eq(0)
            yield Tick()

            # First fetch
            yield engine.fetch_req_ready.eq(1)
            fetch_addr = yield engine.fetch_req_addr
            fetch_addresses.append(fetch_addr)
            yield Tick()
            yield engine.fetch_req_ready.eq(0)
            yield Tick()

            # Provide first descriptor
            for i in range(desc_beats):
                yield engine.fetch_resp_valid.eq(1)
                start = i * bytes_per_beat
                end = start + bytes_per_beat
                beat_data = int.from_bytes(desc1_bytes[start:end], "little")
                yield engine.fetch_resp_data.eq(beat_data)
                yield engine.fetch_resp_last.eq(1 if i == desc_beats - 1 else 0)
                yield Tick()

            yield engine.fetch_resp_valid.eq(0)
            yield Tick()
            yield Tick()  # Parse
            yield Tick()  # Check chain -> fetch next

            # Check for second fetch
            for _ in range(5):
                fetch_valid = yield engine.fetch_req_valid
                if fetch_valid:
                    fetch_addr = yield engine.fetch_req_addr
                    fetch_addresses.append(fetch_addr)
                    break
                yield Tick()

        sim = Simulator(engine)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()

        assert len(fetch_addresses) == 2, f"Should fetch 2 descriptors, got {len(fetch_addresses)}"
        assert fetch_addresses[0] == 0x1000, "First fetch should be at 0x1000"
        assert fetch_addresses[1] == 0x2000, "Second fetch should be at 0x2000 (next_desc)"

    def test_interrupt_on_complete(self, config):
        """Test that INTERRUPT_ON_COMPLETE flag triggers interrupt."""
        engine = DescriptorEngine(config)
        buswidth = config.dma_buswidth
        bytes_per_beat = buswidth // 8
        desc_beats = 64 // bytes_per_beat

        # Create NOP descriptor with interrupt flag
        desc = DmaDescriptor(
            opcode=DmaOpcode.NOP,
            flags=DmaFlags.INTERRUPT_ON_COMPLETE,
        )
        desc_bytes = desc.to_bytes()

        results = {"interrupt": False, "done": False}

        def testbench():
            # Start with NOP+interrupt descriptor
            yield engine.start.eq(1)
            yield engine.desc_addr.eq(0x1000)
            yield Tick()
            yield engine.start.eq(0)
            yield Tick()

            # Accept fetch
            yield engine.fetch_req_ready.eq(1)
            yield Tick()
            yield engine.fetch_req_ready.eq(0)
            yield Tick()

            # Provide descriptor
            for i in range(desc_beats):
                yield engine.fetch_resp_valid.eq(1)
                start = i * bytes_per_beat
                end = start + bytes_per_beat
                beat_data = int.from_bytes(desc_bytes[start:end], "little")
                yield engine.fetch_resp_data.eq(beat_data)
                yield engine.fetch_resp_last.eq(1 if i == desc_beats - 1 else 0)
                yield Tick()

            yield engine.fetch_resp_valid.eq(0)

            # Wait for state machine and check for interrupt
            for _ in range(10):
                done = yield engine.done
                interrupt = yield engine.interrupt
                if done:
                    results["done"] = True
                if interrupt:
                    results["interrupt"] = True
                if done:
                    break
                yield Tick()

        sim = Simulator(engine)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()

        assert results["done"], "Should complete"
        assert results["interrupt"], "Should trigger interrupt on completion"

    def test_elaboration(self, engine):
        """Test that DescriptorEngine elaborates without errors."""

        def testbench():
            yield Tick()

        sim = Simulator(engine)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()


class TestDescriptorEngineVerilogGeneration:
    """Test Verilog generation for DescriptorEngine."""

    def test_generate_verilog(self, tmp_path):
        """Test that DescriptorEngine can generate valid Verilog."""
        from amaranth._toolchain.yosys import find_yosys
        from amaranth.back import verilog

        try:
            find_yosys(lambda ver: ver >= (0, 40))
        except Exception:
            pytest.skip("Yosys not found")

        config = SystolicConfig(dma_buswidth=128, dma_maxbytes=64)
        engine = DescriptorEngine(config)

        output = verilog.convert(engine, name="DescriptorEngine")
        assert "module DescriptorEngine" in output
        assert "start" in output
        assert "desc_addr" in output
        assert "fetch_req_valid" in output
        assert "read_req_valid" in output
        assert "write_req_valid" in output

        verilog_file = tmp_path / "descriptor_engine.v"
        verilog_file.write_text(output)
        assert verilog_file.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
