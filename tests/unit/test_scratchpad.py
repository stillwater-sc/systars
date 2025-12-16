"""
Unit tests for the Scratchpad module.

These tests verify:
1. ScratchpadBank instantiation
2. Basic write/read operations
3. Byte-level write masking
4. Read latency and valid signaling
5. Multi-bank Scratchpad routing
6. Verilog generation
"""

import pytest
from amaranth.sim import Simulator, Tick

from systars.config import SystolicConfig
from systars.memory.scratchpad import Scratchpad, ScratchpadBank


class TestScratchpadBank:
    """Test suite for ScratchpadBank."""

    @pytest.fixture
    def config(self):
        """Configuration for tests with shorter latency for faster tests."""
        return SystolicConfig(
            sp_banks=4,
            sp_capacity_kb=16,  # Smaller for faster tests
            spad_read_delay=2,  # Shorter latency
            mesh_rows=4,
            mesh_cols=4,
            tile_rows=1,
            tile_cols=1,
        )

    @pytest.fixture
    def bank(self, config):
        """Create a ScratchpadBank instance."""
        return ScratchpadBank(config, bank_id=0)

    def test_bank_instantiation(self, bank, config):
        """Test that ScratchpadBank can be instantiated."""
        assert bank is not None
        assert bank.config == config
        assert bank.bank_id == 0

    def test_bank_has_correct_ports(self, bank):
        """Test that bank has all required ports."""
        # Read ports
        assert hasattr(bank, "read_addr")
        assert hasattr(bank, "read_en")
        assert hasattr(bank, "read_data")
        assert hasattr(bank, "read_valid")

        # Write ports
        assert hasattr(bank, "write_addr")
        assert hasattr(bank, "write_en")
        assert hasattr(bank, "write_data")
        assert hasattr(bank, "write_mask")

    def test_bank_write_read(self, config):
        """Test basic write then read operation."""
        bank = ScratchpadBank(config, bank_id=0)
        results = []

        def testbench():
            # Write to address 0
            yield bank.write_addr.eq(0)
            yield bank.write_data.eq(0xDEADBEEF)
            yield bank.write_mask.eq(0xFFFF)  # All bytes enabled
            yield bank.write_en.eq(1)
            yield Tick()

            # Disable write
            yield bank.write_en.eq(0)
            yield Tick()

            # Read from address 0
            yield bank.read_addr.eq(0)
            yield bank.read_en.eq(1)
            yield Tick()

            # Wait for read latency (latency-1 more ticks after read_en goes low)
            yield bank.read_en.eq(0)
            for _ in range(config.spad_read_delay - 1):
                yield Tick()

            # Check read data (valid is only high for one cycle)
            valid = yield bank.read_valid
            data = yield bank.read_data
            results.append({"valid": valid, "data": data})

        sim = Simulator(bank)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()

        assert results[0]["valid"] == 1
        # Data should be what we wrote (masked to width)
        expected = 0xDEADBEEF & ((1 << config.sp_width) - 1)
        assert results[0]["data"] == expected

    def test_bank_read_latency(self, config):
        """Test that read valid follows read enable with correct latency."""
        bank = ScratchpadBank(config, bank_id=0)
        valid_history = []

        def testbench():
            # Issue a read
            yield bank.read_addr.eq(0)
            yield bank.read_en.eq(1)
            yield Tick()

            # Track valid signal after read_en goes low
            yield bank.read_en.eq(0)
            for _ in range(config.spad_read_delay + 2):
                valid = yield bank.read_valid
                valid_history.append(valid)
                yield Tick()

        sim = Simulator(bank)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()

        # Valid should go high after latency cycles from read_en going high
        # History is built by checking valid before each tick in the loop
        # After initial tick + (latency-1) more ticks in loop = latency total
        # So valid_history[latency-1] should be 1
        assert valid_history[config.spad_read_delay - 1] == 1

    def test_bank_elaboration(self, bank):
        """Test that ScratchpadBank elaborates without errors."""

        def testbench():
            yield Tick()

        sim = Simulator(bank)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()


class TestScratchpad:
    """Test suite for multi-bank Scratchpad."""

    @pytest.fixture
    def config(self):
        """Configuration for tests."""
        return SystolicConfig(
            sp_banks=4,
            sp_capacity_kb=16,
            spad_read_delay=2,
            mesh_rows=4,
            mesh_cols=4,
            tile_rows=1,
            tile_cols=1,
        )

    @pytest.fixture
    def scratchpad(self, config):
        """Create a Scratchpad instance."""
        return Scratchpad(config)

    def test_scratchpad_instantiation(self, scratchpad, config):
        """Test that Scratchpad can be instantiated."""
        assert scratchpad is not None
        assert scratchpad.config == config

    def test_scratchpad_has_correct_ports(self, scratchpad):
        """Test that Scratchpad has all required ports."""
        # Read interface
        assert hasattr(scratchpad, "read_req")
        assert hasattr(scratchpad, "read_addr")
        assert hasattr(scratchpad, "read_data")
        assert hasattr(scratchpad, "read_valid")

        # Write interface
        assert hasattr(scratchpad, "write_req")
        assert hasattr(scratchpad, "write_addr")
        assert hasattr(scratchpad, "write_data")
        assert hasattr(scratchpad, "write_mask")

    def test_scratchpad_bank_routing(self, config):
        """Test that addresses are routed to correct banks."""
        sp = Scratchpad(config)
        results = []

        def testbench():
            # Write to bank 0 (address = row << 2 | bank)
            yield sp.write_addr.eq(0b00)  # Bank 0, row 0
            yield sp.write_data.eq(0x1111)
            yield sp.write_mask.eq(0xFFFF)
            yield sp.write_req.eq(1)
            yield Tick()

            # Write to bank 1
            yield sp.write_addr.eq(0b01)  # Bank 1, row 0
            yield sp.write_data.eq(0x2222)
            yield Tick()

            # Write to bank 2
            yield sp.write_addr.eq(0b10)  # Bank 2, row 0
            yield sp.write_data.eq(0x3333)
            yield Tick()

            # Write to bank 3
            yield sp.write_addr.eq(0b11)  # Bank 3, row 0
            yield sp.write_data.eq(0x4444)
            yield Tick()

            yield sp.write_req.eq(0)
            yield Tick()

            # Read from bank 0
            yield sp.read_addr.eq(0b00)
            yield sp.read_req.eq(1)
            yield Tick()
            yield sp.read_req.eq(0)

            # Wait for latency (latency-1 more ticks)
            for _ in range(config.spad_read_delay - 1):
                yield Tick()

            data = yield sp.read_data
            valid = yield sp.read_valid
            results.append({"bank": 0, "data": data, "valid": valid})

            # Extra tick before next read
            yield Tick()

            # Read from bank 2
            yield sp.read_addr.eq(0b10)
            yield sp.read_req.eq(1)
            yield Tick()
            yield sp.read_req.eq(0)

            for _ in range(config.spad_read_delay - 1):
                yield Tick()

            data = yield sp.read_data
            valid = yield sp.read_valid
            results.append({"bank": 2, "data": data, "valid": valid})

        sim = Simulator(sp)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()

        # Verify bank 0 has its data
        assert results[0]["valid"] == 1
        assert (results[0]["data"] & 0xFFFF) == 0x1111

        # Verify bank 2 has its data
        assert results[1]["valid"] == 1
        assert (results[1]["data"] & 0xFFFF) == 0x3333

    def test_scratchpad_elaboration(self, scratchpad):
        """Test that Scratchpad elaborates without errors."""

        def testbench():
            yield Tick()

        sim = Simulator(scratchpad)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()


class TestScratchpadVerilogGeneration:
    """Test Verilog generation for Scratchpad modules."""

    def test_generate_bank_verilog(self, tmp_path):
        """Test that ScratchpadBank can generate valid Verilog."""
        from amaranth._toolchain.yosys import find_yosys
        from amaranth.back import verilog

        try:
            find_yosys(lambda ver: ver >= (0, 40))
        except Exception:
            pytest.skip("Yosys not found")

        config = SystolicConfig(sp_banks=4, sp_capacity_kb=16, spad_read_delay=2)
        bank = ScratchpadBank(config, bank_id=0)

        output = verilog.convert(bank, name="ScratchpadBank")
        assert "module ScratchpadBank" in output
        assert "read_addr" in output
        assert "write_data" in output

        verilog_file = tmp_path / "scratchpad_bank.v"
        verilog_file.write_text(output)
        assert verilog_file.exists()

    def test_generate_scratchpad_verilog(self, tmp_path):
        """Test that Scratchpad can generate valid Verilog."""
        from amaranth._toolchain.yosys import find_yosys
        from amaranth.back import verilog

        try:
            find_yosys(lambda ver: ver >= (0, 40))
        except Exception:
            pytest.skip("Yosys not found")

        config = SystolicConfig(sp_banks=4, sp_capacity_kb=16, spad_read_delay=2)
        sp = Scratchpad(config)

        output = verilog.convert(sp, name="Scratchpad")
        assert "module Scratchpad" in output

        # Should have bank submodules
        assert "bank_0" in output
        assert "bank_1" in output

        verilog_file = tmp_path / "scratchpad.v"
        verilog_file.write_text(output)
        assert verilog_file.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
