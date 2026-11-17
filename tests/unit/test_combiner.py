"""
Unit tests for the Combiner module.

These tests verify:
1. CombinerBank instantiation
2. Basic write/read operations
3. Accumulate mode (add to existing)
4. Activation functions (RELU)
5. Multi-bank Combiner routing
6. Verilog generation
"""

import pytest
from amaranth.sim import Simulator, Tick

from systars.config import Activation, SystolicConfig
from systars.memory.combiner import Combiner, CombinerBank


class TestCombinerBank:
    """Test suite for CombinerBank."""

    @pytest.fixture
    def config(self):
        """Configuration for tests with shorter latency."""
        return SystolicConfig(
            combiner_banks=2,
            combiner_capacity_kb=8,  # Smaller for faster tests
            combiner_latency=2,  # Short latency
            grid_rows=4,
            grid_cols=4,
            tile_rows=1,
            tile_cols=1,
        )

    @pytest.fixture
    def bank(self, config):
        """Create an CombinerBank instance."""
        return CombinerBank(config, bank_id=0)

    def test_bank_instantiation(self, bank, config):
        """Test that CombinerBank can be instantiated."""
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
        assert hasattr(bank, "read_activation")

        # Write ports
        assert hasattr(bank, "write_addr")
        assert hasattr(bank, "write_en")
        assert hasattr(bank, "write_data")
        assert hasattr(bank, "accumulate")

    def test_bank_write_overwrite(self, config):
        """Test basic write (overwrite mode)."""
        bank = CombinerBank(config, bank_id=0)
        results = []

        def testbench():
            # Write to address 0 (overwrite mode)
            yield bank.write_addr.eq(0)
            yield bank.write_data.eq(100)
            yield bank.accumulate.eq(0)
            yield bank.write_en.eq(1)
            yield Tick()

            yield bank.write_en.eq(0)
            yield Tick()

            # Read back
            yield bank.read_addr.eq(0)
            yield bank.read_activation.eq(Activation.NONE.value)
            yield bank.read_en.eq(1)
            yield Tick()

            yield bank.read_en.eq(0)
            # Wait for latency-1 more ticks (valid is high for only one cycle)
            for _ in range(config.combiner_latency - 1):
                yield Tick()

            valid = yield bank.read_valid
            data = yield bank.read_data
            results.append({"valid": valid, "data": data})

        sim = Simulator(bank)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()

        assert results[0]["valid"] == 1
        assert results[0]["data"] == 100

    def test_bank_write_accumulate(self, config):
        """Test accumulate mode (add to existing)."""
        bank = CombinerBank(config, bank_id=0)
        results = []

        def testbench():
            # First write: 100
            yield bank.write_addr.eq(0)
            yield bank.write_data.eq(100)
            yield bank.accumulate.eq(0)  # Overwrite first
            yield bank.write_en.eq(1)
            yield Tick()

            yield bank.write_en.eq(0)
            yield Tick()
            yield Tick()

            # Accumulate: add 50
            yield bank.write_addr.eq(0)
            yield bank.write_data.eq(50)
            yield bank.accumulate.eq(1)  # Accumulate mode
            yield bank.write_en.eq(1)
            yield Tick()

            # Drop accumulate together with write_en, as Streamer does
            yield bank.write_en.eq(0)
            yield bank.accumulate.eq(0)
            yield Tick()
            yield Tick()

            # Read back
            yield bank.read_addr.eq(0)
            yield bank.read_activation.eq(Activation.NONE.value)
            yield bank.read_en.eq(1)
            yield Tick()

            yield bank.read_en.eq(0)
            for _ in range(config.combiner_latency - 1):
                yield Tick()

            valid = yield bank.read_valid
            data = yield bank.read_data
            results.append({"valid": valid, "data": data})

        sim = Simulator(bank)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()

        assert results[0]["valid"] == 1
        # Should be 100 + 50 = 150
        assert results[0]["data"] == 150

    def test_bank_relu_activation(self, config):
        """Test RELU activation function."""
        bank = CombinerBank(config, bank_id=0)
        results = []

        def testbench():
            # Write negative value to address 0
            yield bank.write_addr.eq(0)
            # Use signed representation for -50
            yield bank.write_data.eq((-50) & ((1 << config.combiner_width) - 1))
            yield bank.accumulate.eq(0)
            yield bank.write_en.eq(1)
            yield Tick()

            # Write positive value to address 1
            yield bank.write_addr.eq(1)
            yield bank.write_data.eq(100)
            yield Tick()

            yield bank.write_en.eq(0)
            yield Tick()

            # Read address 0 with RELU
            yield bank.read_addr.eq(0)
            yield bank.read_activation.eq(Activation.RELU.value)
            yield bank.read_en.eq(1)
            yield Tick()

            yield bank.read_en.eq(0)
            for _ in range(config.combiner_latency - 1):
                yield Tick()

            valid = yield bank.read_valid
            data = yield bank.read_data
            results.append({"addr": 0, "valid": valid, "data": data})

            # Extra tick before next read
            yield Tick()

            # Read address 1 with RELU
            yield bank.read_addr.eq(1)
            yield bank.read_en.eq(1)
            yield Tick()

            yield bank.read_en.eq(0)
            for _ in range(config.combiner_latency - 1):
                yield Tick()

            valid = yield bank.read_valid
            data = yield bank.read_data
            results.append({"addr": 1, "valid": valid, "data": data})

        sim = Simulator(bank)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()

        # Address 0: negative value should become 0 with RELU
        assert results[0]["valid"] == 1
        assert results[0]["data"] == 0

        # Address 1: positive value stays positive
        assert results[1]["valid"] == 1
        assert results[1]["data"] == 100

    def test_bank_elaboration(self, bank):
        """Test that CombinerBank elaborates without errors."""

        def testbench():
            yield Tick()

        sim = Simulator(bank)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()


class TestCombiner:
    """Test suite for multi-bank Combiner."""

    @pytest.fixture
    def config(self):
        """Configuration for tests."""
        return SystolicConfig(
            combiner_banks=2,
            combiner_capacity_kb=8,
            combiner_latency=2,
            grid_rows=4,
            grid_cols=4,
            tile_rows=1,
            tile_cols=1,
        )

    @pytest.fixture
    def combiner(self, config):
        """Create an Combiner instance."""
        return Combiner(config)

    def test_combiner_instantiation(self, combiner, config):
        """Test that Combiner can be instantiated."""
        assert combiner is not None
        assert combiner.config == config

    def test_combiner_has_correct_ports(self, combiner):
        """Test that Combiner has all required ports."""
        # Read interface
        assert hasattr(combiner, "read_req")
        assert hasattr(combiner, "read_addr")
        assert hasattr(combiner, "read_data")
        assert hasattr(combiner, "read_valid")
        assert hasattr(combiner, "read_activation")

        # Write interface
        assert hasattr(combiner, "write_req")
        assert hasattr(combiner, "write_addr")
        assert hasattr(combiner, "write_data")
        assert hasattr(combiner, "accumulate")

    def test_combiner_bank_routing(self, config):
        """Test that addresses are routed to correct banks."""
        acc = Combiner(config)
        results = []

        def testbench():
            # Write to bank 0 (address = row << 1 | bank)
            yield acc.write_addr.eq(0b0)  # Bank 0, row 0
            yield acc.write_data.eq(111)
            yield acc.accumulate.eq(0)
            yield acc.write_req.eq(1)
            yield Tick()

            # Write to bank 1
            yield acc.write_addr.eq(0b1)  # Bank 1, row 0
            yield acc.write_data.eq(222)
            yield Tick()

            yield acc.write_req.eq(0)
            yield Tick()

            # Read from bank 0
            yield acc.read_addr.eq(0b0)
            yield acc.read_activation.eq(Activation.NONE.value)
            yield acc.read_req.eq(1)
            yield Tick()
            yield acc.read_req.eq(0)

            for _ in range(config.combiner_latency - 1):
                yield Tick()

            valid = yield acc.read_valid
            data = yield acc.read_data
            results.append({"bank": 0, "valid": valid, "data": data})

            # Extra tick before next read
            yield Tick()

            # Read from bank 1
            yield acc.read_addr.eq(0b1)
            yield acc.read_req.eq(1)
            yield Tick()
            yield acc.read_req.eq(0)

            for _ in range(config.combiner_latency - 1):
                yield Tick()

            valid = yield acc.read_valid
            data = yield acc.read_data
            results.append({"bank": 1, "valid": valid, "data": data})

        sim = Simulator(acc)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()

        assert results[0]["valid"] == 1
        assert results[0]["data"] == 111

        assert results[1]["valid"] == 1
        assert results[1]["data"] == 222

    def test_combiner_elaboration(self, combiner):
        """Test that Combiner elaborates without errors."""

        def testbench():
            yield Tick()

        sim = Simulator(combiner)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()


class TestCombinerVerilogGeneration:
    """Test Verilog generation for Combiner modules."""

    def test_generate_bank_verilog(self, tmp_path):
        """Test that CombinerBank can generate valid Verilog."""
        from amaranth._toolchain.yosys import find_yosys
        from amaranth.back import verilog

        try:
            find_yosys(lambda ver: ver >= (0, 40))
        except Exception:
            pytest.skip("Yosys not found")

        config = SystolicConfig(combiner_banks=2, combiner_capacity_kb=8, combiner_latency=2)
        bank = CombinerBank(config, bank_id=0)

        output = verilog.convert(bank, name="CombinerBank")
        assert "module CombinerBank" in output
        assert "read_addr" in output
        assert "write_data" in output
        assert "accumulate" in output

        verilog_file = tmp_path / "combiner_bank.v"
        verilog_file.write_text(output)
        assert verilog_file.exists()

    def test_generate_accumulator_verilog(self, tmp_path):
        """Test that Combiner can generate valid Verilog."""
        from amaranth._toolchain.yosys import find_yosys
        from amaranth.back import verilog

        try:
            find_yosys(lambda ver: ver >= (0, 40))
        except Exception:
            pytest.skip("Yosys not found")

        config = SystolicConfig(combiner_banks=2, combiner_capacity_kb=8, combiner_latency=2)
        acc = Combiner(config)

        output = verilog.convert(acc, name="Combiner")
        assert "module Combiner" in output

        # Should have bank submodules
        assert "bank_0" in output
        assert "bank_1" in output

        verilog_file = tmp_path / "combiner.v"
        verilog_file.write_text(output)
        assert verilog_file.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestCombinerBankRowSemantics:
    """Accumulate and activation operate on each acc_bits element of a packed row."""

    @pytest.fixture
    def cfg(self):
        # 2 elements of 32 bits per row
        return SystolicConfig(
            grid_rows=2, grid_cols=2, combiner_banks=2, combiner_capacity_kb=1, combiner_latency=1
        )

    @staticmethod
    def pack(values, bits=32):
        return sum((v % (1 << bits)) << (i * bits) for i, v in enumerate(values))

    @staticmethod
    def unpack(value, n, bits=32):
        out = []
        for i in range(n):
            v = (value >> (i * bits)) & ((1 << bits) - 1)
            out.append(v - (1 << bits) if v >> (bits - 1) else v)
        return out

    def _run(self, bank, writes, activation=Activation.NONE.value):
        """Issue (data, accumulate) writes to row 0 on consecutive cycles, then read row 0."""
        out = {}

        async def testbench(ctx):
            for data, acc in writes:
                ctx.set(bank.write_addr, 0)
                ctx.set(bank.write_data, data)
                ctx.set(bank.accumulate, acc)
                ctx.set(bank.write_en, 1)
                await ctx.tick()
            ctx.set(bank.write_en, 0)
            ctx.set(bank.accumulate, 0)
            await ctx.tick()
            await ctx.tick()
            ctx.set(bank.read_addr, 0)
            ctx.set(bank.read_activation, activation)
            ctx.set(bank.read_en, 1)
            await ctx.tick()
            ctx.set(bank.read_en, 0)
            out["valid"] = ctx.get(bank.read_valid)
            out["data"] = ctx.get(bank.read_data)

        sim = Simulator(bank)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()
        assert out["valid"] == 1
        return out["data"] % (1 << bank.config.combiner_width)

    def test_back_to_back_accumulates_same_row(self, cfg):
        bank = CombinerBank(cfg, bank_id=0)
        p = self.pack
        data = self._run(bank, [(p([100, 1]), 0), (p([50, 2]), 1), (p([25, 3]), 1)])
        assert self.unpack(data, 2) == [175, 6]

    def test_accumulate_does_not_carry_between_elements(self, cfg):
        bank = CombinerBank(cfg, bank_id=0)
        p = self.pack
        data = self._run(bank, [(p([-1, 10]), 0), (p([1, 0]), 1)])
        assert self.unpack(data, 2) == [0, 10]

    @pytest.mark.parametrize(
        ("activation", "expected"),
        [
            (Activation.NONE.value, [-5, 9]),
            (Activation.RELU.value, [0, 9]),
            (Activation.RELU6.value, [0, 6]),
        ],
    )
    def test_activation_is_per_element(self, cfg, activation, expected):
        bank = CombinerBank(cfg, bank_id=0)
        data = self._run(bank, [(self.pack([-5, 9]), 0)], activation=activation)
        assert self.unpack(data, 2) == expected
