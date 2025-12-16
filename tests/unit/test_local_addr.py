"""
Unit tests for the LocalAddr module.

These tests verify:
1. Address field extraction (is_acc, accumulate, read_full_row)
2. Scratchpad bank/row decoding
3. Accumulator bank/row decoding
4. Address construction utilities
5. Garbage address detection
"""

import pytest
from amaranth import Elaboratable, Module, Signal, unsigned
from amaranth.sim import Simulator, Tick

from systars.config import SystolicConfig
from systars.memory.local_addr import LocalAddr


class TestLocalAddr:
    """Test suite for LocalAddr utilities."""

    @pytest.fixture
    def config(self):
        """Default configuration for tests."""
        return SystolicConfig(
            sp_banks=4,
            acc_banks=2,
            sp_capacity_kb=256,
            acc_capacity_kb=64,
        )

    @pytest.fixture
    def addr_util(self, config):
        """Create LocalAddr instance."""
        return LocalAddr(config)

    def test_instantiation(self, addr_util, config):
        """Test that LocalAddr can be instantiated."""
        assert addr_util is not None
        assert addr_util.config == config
        assert addr_util.sp_bank_bits == 2  # log2(4) = 2
        assert addr_util.acc_bank_bits == 1  # log2(2) = 1

    def test_is_acc_flag(self):
        """Test is_acc bit extraction."""

        class TestModule(Elaboratable):
            def __init__(self):
                self.addr = Signal(unsigned(32))
                self.is_acc = Signal()
                self.dummy = Signal()  # Create sync domain

            def elaborate(self, _platform):
                m = Module()
                m.d.comb += self.is_acc.eq(LocalAddr.is_acc(self.addr))
                m.d.sync += self.dummy.eq(~self.dummy)  # Dummy sync to create domain
                return m

        dut = TestModule()
        results = []

        def testbench():
            # Test scratchpad address (bit 31 = 0)
            yield dut.addr.eq(0x0000_0000)
            yield Tick()
            is_acc = yield dut.is_acc
            results.append(("sp", is_acc))

            # Test accumulator address (bit 31 = 1)
            yield dut.addr.eq(0x8000_0000)
            yield Tick()
            is_acc = yield dut.is_acc
            results.append(("acc", is_acc))

        sim = Simulator(dut)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()

        assert results[0] == ("sp", 0)
        assert results[1] == ("acc", 1)

    def test_accumulate_flag(self):
        """Test accumulate bit extraction."""

        class TestModule(Elaboratable):
            def __init__(self):
                self.addr = Signal(unsigned(32))
                self.accumulate = Signal()
                self.dummy = Signal()

            def elaborate(self, _platform):
                m = Module()
                m.d.comb += self.accumulate.eq(LocalAddr.accumulate(self.addr))
                m.d.sync += self.dummy.eq(~self.dummy)
                return m

        dut = TestModule()
        results = []

        def testbench():
            # Test no accumulate (bit 30 = 0)
            yield dut.addr.eq(0x0000_0000)
            yield Tick()
            acc = yield dut.accumulate
            results.append(("no_acc", acc))

            # Test accumulate (bit 30 = 1)
            yield dut.addr.eq(0x4000_0000)
            yield Tick()
            acc = yield dut.accumulate
            results.append(("acc", acc))

        sim = Simulator(dut)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()

        assert results[0] == ("no_acc", 0)
        assert results[1] == ("acc", 1)

    def test_garbage_detection(self):
        """Test garbage address detection."""

        class TestModule(Elaboratable):
            def __init__(self):
                self.addr = Signal(unsigned(32))
                self.is_garbage = Signal()
                self.dummy = Signal()

            def elaborate(self, _platform):
                m = Module()
                m.d.comb += self.is_garbage.eq(LocalAddr.is_garbage(self.addr))
                m.d.sync += self.dummy.eq(~self.dummy)
                return m

        dut = TestModule()
        results = []

        def testbench():
            # Test valid address
            yield dut.addr.eq(0x0000_0100)
            yield Tick()
            garbage = yield dut.is_garbage
            results.append(("valid", garbage))

            # Test garbage address (all flag bits set)
            garbage_addr = LocalAddr.make_garbage_addr()
            yield dut.addr.eq(garbage_addr)
            yield Tick()
            garbage = yield dut.is_garbage
            results.append(("garbage", garbage))

        sim = Simulator(dut)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()

        assert results[0] == ("valid", 0)
        assert results[1] == ("garbage", 1)

    def test_make_sp_addr(self):
        """Test scratchpad address construction."""
        # Bank 0, row 0
        addr = LocalAddr.make_sp_addr(0, 0)
        assert addr & (1 << 31) == 0  # Not accumulator

        # Bank 2, row 100
        addr = LocalAddr.make_sp_addr(2, 100)
        assert addr & (1 << 31) == 0  # Not accumulator

        # With read_full_row
        addr = LocalAddr.make_sp_addr(0, 0, read_full_row=True)
        assert addr & (1 << 29) != 0  # read_full_row set

    def test_make_acc_addr(self):
        """Test accumulator address construction."""
        # Bank 0, row 0
        addr = LocalAddr.make_acc_addr(0, 0)
        assert addr & (1 << 31) != 0  # Is accumulator

        # With accumulate flag
        addr = LocalAddr.make_acc_addr(1, 50, accumulate=True)
        assert addr & (1 << 31) != 0  # Is accumulator
        assert addr & (1 << 30) != 0  # Accumulate set

    def test_make_garbage_addr(self):
        """Test garbage address construction."""
        addr = LocalAddr.make_garbage_addr()
        assert addr & (1 << 31) != 0  # is_acc set
        assert addr & (1 << 30) != 0  # accumulate set
        assert addr & (1 << 29) != 0  # read_full_row set

    def test_sp_bank_extraction(self):
        """Test scratchpad bank index extraction."""

        class TestModule(Elaboratable):
            def __init__(self, config):
                self.addr = Signal(unsigned(32))
                self.bank = Signal(unsigned(2))
                self.addr_util = LocalAddr(config)
                self.dummy = Signal()

            def elaborate(self, _platform):
                m = Module()
                m.d.comb += self.bank.eq(self.addr_util.sp_bank(self.addr))
                m.d.sync += self.dummy.eq(~self.dummy)
                return m

        config = SystolicConfig(sp_banks=4)
        dut = TestModule(config)
        results = []

        def testbench():
            # Bank 0
            yield dut.addr.eq(0b00)
            yield Tick()
            bank = yield dut.bank
            results.append(bank)

            # Bank 1
            yield dut.addr.eq(0b01)
            yield Tick()
            bank = yield dut.bank
            results.append(bank)

            # Bank 2
            yield dut.addr.eq(0b10)
            yield Tick()
            bank = yield dut.bank
            results.append(bank)

            # Bank 3
            yield dut.addr.eq(0b11)
            yield Tick()
            bank = yield dut.bank
            results.append(bank)

        sim = Simulator(dut)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()

        assert results == [0, 1, 2, 3]

    def test_bit_widths(self, config):
        """Test computed bit widths."""
        addr_util = LocalAddr(config)

        # With 4 banks, need 2 bits
        assert addr_util.sp_bank_bits == 2

        # With 2 banks, need 1 bit
        assert addr_util.acc_bank_bits == 1

        # Row bits depend on bank entries
        assert addr_util.sp_row_bits > 0
        assert addr_util.acc_row_bits > 0


class TestLocalAddrEdgeCases:
    """Edge case tests for LocalAddr."""

    def test_single_bank(self):
        """Test with single bank configuration."""
        config = SystolicConfig(sp_banks=1, acc_banks=1)
        addr_util = LocalAddr(config)

        # Single bank still needs at least 1 bit
        assert addr_util.sp_bank_bits >= 1
        assert addr_util.acc_bank_bits >= 1

    def test_large_banks(self):
        """Test with many banks."""
        config = SystolicConfig(sp_banks=16, acc_banks=8)
        addr_util = LocalAddr(config)

        # 16 banks needs 4 bits
        assert addr_util.sp_bank_bits == 4
        # 8 banks needs 3 bits
        assert addr_util.acc_bank_bits == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
