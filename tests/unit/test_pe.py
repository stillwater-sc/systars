"""
Unit tests for the Processing Element (PE).

These tests verify:
1. Basic MAC operation
2. Output-stationary dataflow
3. Weight-stationary dataflow
4. Propagate signal behavior
5. Rounding shift operation
"""

import pytest
from amaranth.sim import Simulator, Tick

from systars.config import Dataflow, SystolicConfig
from systars.core.pe import PE, PEWithShift


async def _mac(ctx, pe, a, b, d=0, *, dataflow, propagate):
    """Apply one valid MAC cycle."""
    ctx.set(pe.in_control_dataflow, dataflow)
    ctx.set(pe.in_control_propagate, propagate)
    ctx.set(pe.in_a, a)
    ctx.set(pe.in_b, b)
    ctx.set(pe.in_d, d)
    ctx.set(pe.in_valid, 1)
    await ctx.tick()
    ctx.set(pe.in_valid, 0)


def _read_reg(ctx, pe, propagate):
    """out_c shows the register *not* selected by propagate: c1 when propagate=1."""
    ctx.set(pe.in_control_propagate, propagate)
    return ctx.get(pe.out_c)


def _run(pe, testbench):
    sim = Simulator(pe)
    sim.add_clock(1e-6)
    sim.add_testbench(testbench)
    sim.run()


class TestPE:
    """Test suite for the basic PE module."""

    @pytest.fixture
    def config(self):
        """Default configuration for testing."""
        return SystolicConfig(
            input_bits=8,
            weight_bits=8,
            acc_bits=32,
            output_bits=20,
            dataflow=Dataflow.OUTPUT_STATIONARY | Dataflow.B_STATIONARY,
        )

    @pytest.fixture
    def pe(self, config):
        """Create a PE instance."""
        return PE(config)

    def test_pe_instantiation(self, pe):
        """Test that PE can be instantiated."""
        assert pe is not None
        assert pe.config.input_bits == 8
        assert pe.config.acc_bits == 32

    def test_simple_mac(self, pe):
        """WS mode: c1 <= in_d + a * b = 10 + 3 * 4 = 22."""
        out = {}

        async def testbench(ctx):
            await _mac(ctx, pe, 3, 4, 10, dataflow=1, propagate=0)
            out["c1"] = _read_reg(ctx, pe, propagate=1)

        _run(pe, testbench)
        assert out["c1"] == 22

    def test_accumulation_os_mode(self, pe):
        """OS mode accumulates into c1: 2 * 3 + 4 * 5 = 26."""
        out = {}

        async def testbench(ctx):
            await _mac(ctx, pe, 2, 3, dataflow=0, propagate=0)
            out["after_first"] = _read_reg(ctx, pe, propagate=1)
            await _mac(ctx, pe, 4, 5, dataflow=0, propagate=0)
            out["after_second"] = _read_reg(ctx, pe, propagate=1)
            out["c2"] = _read_reg(ctx, pe, propagate=0)  # untouched

        _run(pe, testbench)
        assert out == {"after_first": 6, "after_second": 26, "c2": 0}

    def test_propagate_toggle(self, pe):
        """propagate selects the register written (c1/c2); out_c shows the other one."""
        out = {}

        async def testbench(ctx):
            await _mac(ctx, pe, 1, 1, dataflow=0, propagate=0)  # c1 = 1
            await _mac(ctx, pe, 2, 2, dataflow=0, propagate=1)  # c2 = 4
            out["c1"] = _read_reg(ctx, pe, propagate=1)
            out["c2"] = _read_reg(ctx, pe, propagate=0)

        _run(pe, testbench)
        assert out == {"c1": 1, "c2": 4}

    def test_passthrough_signals(self, pe):
        """Test that control signals pass through correctly."""
        results = {}

        def testbench():
            yield pe.in_a.eq(42)
            yield pe.in_id.eq(123)
            yield pe.in_last.eq(1)
            yield pe.in_valid.eq(1)
            yield pe.in_control_shift.eq(7)
            yield Tick()

            # Wait for registered output
            yield Tick()
            results["out_a"] = yield pe.out_a
            results["out_id"] = yield pe.out_id
            results["out_last"] = yield pe.out_last
            results["out_shift"] = yield pe.out_control_shift

        sim = Simulator(pe)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()

        assert results["out_a"] == 42
        assert results["out_id"] == 123
        assert results["out_last"] == 1
        assert results["out_shift"] == 7

    def test_signed_multiplication(self, pe):
        """Signed operands: -3 * 4 + 0 = -12, and -3 * -4 + -5 = 7."""
        out = {}

        async def testbench(ctx):
            await _mac(ctx, pe, -3, 4, 0, dataflow=1, propagate=0)
            out["neg"] = _read_reg(ctx, pe, propagate=1)
            await _mac(ctx, pe, -3, -4, -5, dataflow=1, propagate=0)
            out["pos"] = _read_reg(ctx, pe, propagate=1)

        _run(pe, testbench)
        assert out == {"neg": -12, "pos": 7}


class TestPEWithShift:
    """Test suite for PE with rounding shift."""

    @pytest.fixture
    def config(self):
        return SystolicConfig(
            input_bits=8,
            weight_bits=8,
            acc_bits=32,
            output_bits=8,  # Narrower output for shift testing
        )

    @pytest.fixture
    def pe(self, config):
        return PEWithShift(config)

    def test_shift_instantiation(self, pe):
        """Test that PEWithShift can be instantiated."""
        assert pe is not None

    def test_rounding_shift(self, pe):
        """Rounding right shift: (24 + (1 << 3)) >> 4 = 2."""
        out = {}

        async def testbench(ctx):
            ctx.set(pe.in_control_shift, 4)
            await _mac(ctx, pe, 6, 4, 0, dataflow=1, propagate=0)  # c1 = 24
            _read_reg(ctx, pe, propagate=1)
            await ctx.tick()  # shift amount is registered (out_control_shift)
            out["shifted"] = ctx.get(pe.out_c_shifted)

        _run(pe, testbench)
        assert out["shifted"] == 2

    def test_saturation(self, pe):
        """100 * 100 = 10000 saturates to the 8-bit output range: +127 / -128."""
        out = {}

        async def testbench(ctx):
            ctx.set(pe.in_control_shift, 0)
            await _mac(ctx, pe, 100, 100, 0, dataflow=1, propagate=0)
            _read_reg(ctx, pe, propagate=1)
            await ctx.tick()
            out["pos"] = ctx.get(pe.out_c_shifted)
            ctx.set(pe.in_control_propagate, 0)
            await _mac(ctx, pe, -100, 100, 0, dataflow=1, propagate=0)
            _read_reg(ctx, pe, propagate=1)
            await ctx.tick()
            out["neg"] = ctx.get(pe.out_c_shifted)

        _run(pe, testbench)
        assert out == {"pos": 127, "neg": -128}


class TestPEVerilogGeneration:
    """Test Verilog generation for PE modules."""

    def test_generate_verilog(self, tmp_path):
        """Test that PE can generate valid Verilog."""
        from amaranth._toolchain.yosys import find_yosys
        from amaranth.back import verilog

        try:
            find_yosys(lambda ver: ver >= (0, 40))
        except Exception:
            pytest.skip("Yosys not found")

        config = SystolicConfig()
        pe = PE(config)

        output = verilog.convert(pe, name="PE")
        assert "module PE" in output
        assert "in_a" in output
        assert "out_c" in output

        # Write to file for inspection
        verilog_file = tmp_path / "pe.v"
        verilog_file.write_text(output)
        assert verilog_file.exists()

    def test_generate_pe_with_shift_verilog(self, tmp_path):
        """Test that PEWithShift can generate valid Verilog."""
        from amaranth._toolchain.yosys import find_yosys
        from amaranth.back import verilog

        try:
            find_yosys(lambda ver: ver >= (0, 40))
        except Exception:
            pytest.skip("Yosys not found")

        config = SystolicConfig()
        pe = PEWithShift(config)

        output = verilog.convert(pe, name="PEWithShift")
        assert "module PEWithShift" in output

        verilog_file = tmp_path / "pe_with_shift.v"
        verilog_file.write_text(output)
        assert verilog_file.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
