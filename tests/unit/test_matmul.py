"""
Unit tests for the Matmul ISA instruction.

- Its command stream, checked numerically: a Python model of SystolicTop runs
  the commands Matmul issues against numpy (``_isa.TopModel``), on beat-aligned
  layouts and, with the layout check off, on layouts tiled in M, N and K.
- Matmul driving a real SystolicTop through an AXI memory model.
- Configuration errors.
"""

import numpy as np
import pytest
from amaranth import Elaboratable, Module

from systars.config import Dataflow, SystolicConfig
from systars.isa.command import CTRL_EXEC, CTRL_LOAD, CTRL_STORE, connect_to_top
from systars.isa.matmul import Matmul, MatmulCmd, MatmulError
from systars.top import SystolicTop
from systars.util.commands import DmaOpcode, ExecOpcode

from ._isa import TopModel, from_bytes, run_isa, to_unsigned_bytes
from ._sim import run_sim, skip_without_yosys
from .test_gemm_systolic import axi_dram, pack, unpack

A_ADDR, B_ADDR, C_ADDR = 0x1000, 0x8000, 0x10000


def config(grid=4, tile=1, **overrides):
    fields = {"grid_rows": grid, "grid_cols": grid, "tile_rows": tile, "tile_cols": tile}
    fields.update(sp_capacity_kb=4, acc_capacity_kb=4)
    fields.update(overrides)
    return SystolicConfig(**fields)


def setup(M, N, K, a_stride, b_stride, c_stride, *, d_addr=0, options=0, a_addr=A_ADDR):
    return [
        (MatmulCmd.CONFIG_DIMS, M | N << 16 | K << 32),
        (MatmulCmd.CONFIG_A_ADDR, a_addr),
        (MatmulCmd.CONFIG_B_ADDR, B_ADDR),
        (MatmulCmd.CONFIG_C_ADDR, C_ADDR),
        (MatmulCmd.CONFIG_D_ADDR, d_addr),
        (MatmulCmd.CONFIG_A_STRIDE, a_stride),
        (MatmulCmd.CONFIG_B_STRIDE, b_stride),
        (MatmulCmd.CONFIG_C_STRIDE, c_stride),
        (MatmulCmd.CONFIG_OPTIONS, options),
    ]


def place(model, base, matrix, stride, bits):
    for r, row in enumerate(matrix):
        model.write(base + r * stride, to_unsigned_bytes(row, bits))


def read_back(model, base, rows, cols, stride, bits):
    return np.array([model.read_elems(base + r * stride, bits, cols) for r in range(rows)])


def check_numerically(cfg, M, N, K, a_stride, b_stride, c_stride, *, aligned=True, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.integers(-128, 128, (M, K))
    B = rng.integers(-128, 128, (K, N))
    isa = Matmul(cfg, require_beat_aligned=aligned)
    result = run_isa(isa, setup(M, N, K, a_stride, b_stride, c_stride), MatmulCmd.START)
    assert result.done and not result.error, f"error {result.error_code:#x}"
    model = TopModel(cfg)
    place(model, A_ADDR, A, a_stride, 8)
    place(model, B_ADDR, B, b_stride, 8)
    model.run(result.commands)
    np.testing.assert_array_equal(read_back(model, C_ADDR, M, N, c_stride, 32), A @ B)
    return result


class TestConstruction:
    def test_requires_square_array(self):
        with pytest.raises(ValueError, match="square"):
            Matmul(SystolicConfig(grid_rows=2, grid_cols=4))

    def test_requires_power_of_two(self):
        with pytest.raises(ValueError, match="power-of-two"):
            Matmul(config(grid=3))

    def test_requires_weight_stationary(self):
        with pytest.raises(ValueError, match="weight-stationary"):
            Matmul(config(dataflow=Dataflow.OUTPUT_STATIONARY))

    def test_verilog(self):
        skip_without_yosys()
        from amaranth.back import verilog

        assert "module Matmul" in verilog.convert(Matmul(config()), name="Matmul")


class TestCommandStream:
    def test_commands_target_controllers_with_their_own_opcodes(self):
        cfg = config()
        beat = cfg.dma_buswidth // 8
        result = run_isa(Matmul(cfg), setup(6, 4, 3, beat, beat, beat), MatmulCmd.START)
        assert result.done
        cmds = result.commands
        assert (cmds[0].type, cmds[0].opcode, cmds[0].rs1) == (CTRL_EXEC, ExecOpcode.CONFIG, 1)
        loads = [c for c in cmds if c.type == CTRL_LOAD]
        stores = [c for c in cmds if c.type == CTRL_STORE]
        execs = [c.opcode for c in cmds[1:] if c.type == CTRL_EXEC]
        assert all(c.opcode == DmaOpcode.MEMCPY and c.len == 1 for c in loads + stores)
        # Two M tiles (4 + 2 rows), one N and one K tile
        assert execs == [ExecOpcode.PRELOAD, ExecOpcode.COMPUTE] * 2
        assert len(loads) == (4 + 3) + (2 + 3)
        assert [c.dram_addr for c in stores] == [C_ADDR + r * beat for r in range(6)]
        assert [c.id for c in cmds] == list(range(len(cmds)))

    def test_scratchpad_addresses_use_low_bank_bits(self):
        """Tile rows are consecutive local addresses (banks in the low bits), not bank << 16."""
        cfg = config()
        beat = cfg.dma_buswidth // 8
        cmds = run_isa(Matmul(cfg), setup(4, 4, 4, beat, beat, beat), MatmulCmd.START).commands
        loads = [c.local_addr for c in cmds if c.type == CTRL_LOAD]
        assert loads == [0, 1, 2, 3, 4, 5, 6, 7]  # A rows, then B rows at dim
        preload = next(c for c in cmds if c.opcode == ExecOpcode.PRELOAD and c.type == CTRL_EXEC)
        assert preload.rs1 == 4 and preload.k_dim == 4

    def test_activation_reaches_store(self):
        cfg = config()
        beat = cfg.dma_buswidth // 8
        cmds = run_isa(
            Matmul(cfg), setup(2, 2, 2, beat, beat, beat, options=1 << 1), MatmulCmd.START
        ).commands
        assert {c.activation for c in cmds if c.type == CTRL_STORE} == {1}

    def test_backpressure_does_not_change_the_stream(self):
        cfg = config()
        beat = cfg.dma_buswidth // 8
        cmds = setup(5, 3, 4, beat, beat, beat)
        fast = run_isa(Matmul(cfg), cmds, MatmulCmd.START).commands
        slow = run_isa(Matmul(cfg), cmds, MatmulCmd.START, ready_every=3, latency=5).commands
        assert fast == slow


class TestNumerical:
    @pytest.mark.parametrize("grid,tile", [(2, 1), (4, 1), (2, 2)])
    @pytest.mark.parametrize("M", [1, 4, 10])
    def test_beat_aligned_layout(self, grid, tile, M):
        """The layouts the tile's DMA can move: N, K <= dim, rows one beat apart."""
        cfg = config(grid=grid, tile=tile)
        dim = grid * tile
        beat = cfg.dma_buswidth // 8
        check_numerically(cfg, M, dim, dim, beat, beat, beat)
        check_numerically(cfg, M, dim - 1, max(1, dim - 1), 2 * beat, beat, beat, seed=1)

    @pytest.mark.parametrize("M,N,K", [(7, 9, 6), (4, 4, 8), (9, 2, 13), (1, 11, 1)])
    def test_tiled_in_m_n_and_k(self, M, N, K):
        """With the alignment check off, dense A and B tiled in every dimension."""
        cfg = config()
        n_padded = -(-N // 4) * 4  # C rows hold whole N tiles (STORE writes tile rows)
        check_numerically(cfg, M, N, K, K, N, n_padded * 4, aligned=False, seed=M + N + K)


class TestErrors:
    def run(self, cfg, cmds):
        return run_isa(Matmul(cfg), cmds, MatmulCmd.START)

    def test_start_without_config(self):
        result = self.run(config(), [])
        assert result.error and result.error_code == MatmulError.CONFIG_INCOMPLETE
        assert not result.commands

    def test_zero_dimension(self):
        beat = config().dma_buswidth // 8
        result = self.run(config(), setup(0, 4, 4, beat, beat, beat))
        assert result.error_code == MatmulError.INVALID_DIMS

    def test_bias_is_unsupported(self):
        beat = config().dma_buswidth // 8
        result = self.run(config(), setup(4, 4, 4, beat, beat, beat, d_addr=0x4000))
        assert result.error_code == MatmulError.UNSUPPORTED_BIAS

    def test_accumulate_is_unsupported(self):
        """C is never loaded into the accumulator, so accumulating onto it would add stale rows."""
        beat = config().dma_buswidth // 8
        result = self.run(config(), setup(4, 4, 4, beat, beat, beat, options=1))
        assert result.error_code == MatmulError.UNSUPPORTED_ACCUMULATE
        assert not result.commands

    @pytest.mark.parametrize(
        "case",
        [
            "unaligned_base",
            "unaligned_stride",
            "zero_stride",
            "two_n_tiles",
            "two_k_tiles",
            "short_c_stride",
        ],
    )
    def test_layouts_the_dma_cannot_move(self, case):
        cfg = config()
        beat = cfg.dma_buswidth // 8
        args = {"M": 4, "N": 4, "K": 4, "a_stride": beat, "b_stride": beat, "c_stride": beat}
        kw = {}
        if case == "unaligned_base":
            kw["a_addr"] = A_ADDR + 4
        elif case == "unaligned_stride":
            args["a_stride"] = beat + 4
        elif case == "zero_stride":
            args["b_stride"] = 0
        elif case == "two_n_tiles":
            args.update(N=8, c_stride=4 * beat)
        elif case == "two_k_tiles":
            args["K"] = 8
        elif case == "short_c_stride":
            args.update(N=8, c_stride=beat)
        result = self.run(cfg, setup(**args, **kw))
        assert result.error_code == MatmulError.UNSUPPORTED_LAYOUT
        assert not result.commands


class MatmulOnTop(Elaboratable):
    def __init__(self, cfg):
        self.top = SystolicTop(cfg)
        self.isa = Matmul(cfg)

    def elaborate(self, _platform):
        m = Module()
        m.submodules.top = self.top
        m.submodules.isa = self.isa
        connect_to_top(m, self.isa, self.top)
        return m


@pytest.mark.parametrize("grid,tile", [(4, 1), (2, 2)])
def test_matmul_drives_systolic_top(grid, tile):
    """Matmul issues every command to a real SystolicTop; C in DRAM equals A @ B."""
    cfg = config(grid=grid, tile=tile)
    dim = grid * tile
    beat = cfg.dma_buswidth // 8
    rng = np.random.default_rng(grid * 10 + tile)
    M, N, K = 2 * dim + 1, dim - 1, dim
    A = rng.integers(-128, 128, (M, K))
    B = rng.integers(-128, 128, (K, N))
    memory = {}
    for r, row in enumerate(A):
        memory[A_ADDR + r * beat] = pack(row, 8)
    for r, row in enumerate(B):
        memory[B_ADDR + r * beat] = pack(row, 8)

    dut = MatmulOnTop(cfg)
    isa, top = dut.isa, dut.top
    status = {}

    async def testbench(ctx):
        for cmd, data in [*setup(M, N, K, beat, beat, beat), (MatmulCmd.START, 0)]:
            ctx.set(isa.cfg_valid, 1)
            ctx.set(isa.cfg_cmd, cmd)
            ctx.set(isa.cfg_data, data)
            await ctx.tick()
        ctx.set(isa.cfg_valid, 0)
        done, error = await ctx.tick().sample(isa.done, isa.error).until(isa.done | isa.error)
        status.update(done=done, error=error)

    run_sim(dut, testbench, processes=[axi_dram(top, memory, beat)], max_cycles=50_000)
    assert status == {"done": 1, "error": 0}
    C = np.array([unpack(memory.get(C_ADDR + r * beat, 0), 32, dim)[:N] for r in range(M)])
    np.testing.assert_array_equal(C, A @ B)


def test_helpers_round_trip():
    data = to_unsigned_bytes([-1, 2, -128, 127], 8)
    assert from_bytes(data, 8, 4) == [-1, 2, -128, 127]
