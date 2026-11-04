"""
Unit tests for the Matmul ISA instruction.

- Its command stream, checked numerically: a Python model of SystolicTop runs
  the commands Matmul issues against numpy (``_isa.TopModel``), with A, B and C
  dense row-major (or strided) at any address, tiled in M, N and K.
- Matmul driving a real SystolicTop through a byte-addressed AXI memory model:
  dense operands, any M, N and K, at unaligned addresses.
- Configuration errors.
"""

import numpy as np
import pytest
from amaranth import Elaboratable, Module

from systars.config import Dataflow, SystolicConfig
from systars.controller.sync import Sync
from systars.isa.command import CTRL_EXEC, CTRL_LOAD, CTRL_STORE, connect_to_top
from systars.isa.matmul import IS, OS, WS, Matmul, MatmulCmd, MatmulError
from systars.top import SystolicTop
from systars.util.commands import DmaOpcode, ExecOpcode

from ._axi import axi_memory
from ._isa import TopModel, from_bytes, run_isa, to_unsigned_bytes
from ._sim import run_sim, skip_without_yosys

A_ADDR, B_ADDR, C_ADDR = 0x1000, 0x8000, 0x10000
ALL = Dataflow.OUTPUT_STATIONARY | Dataflow.B_STATIONARY | Dataflow.A_STATIONARY
DATAFLOWS = [OS, WS, IS]
DF_IDS = ["os", "ws", "is"]


def config(grid=4, tile=1, **overrides):
    fields = {"grid_rows": grid, "grid_cols": grid, "tile_rows": tile, "tile_cols": tile}
    fields.update(sp_capacity_kb=4, acc_capacity_kb=4, dataflow=ALL)
    fields.update(overrides)
    return SystolicConfig(**fields)


def setup(
    M, N, K, a_stride, b_stride, c_stride, *, d_addr=0, options=0, a_addr=A_ADDR, dataflow=None
):
    extra = [] if dataflow is None else [(MatmulCmd.CONFIG_DATAFLOW, dataflow)]
    return extra + [
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


def check_numerically(
    cfg, M, N, K, a_stride, b_stride, c_stride, *, a_addr=A_ADDR, seed=0, buffers=2,
    dataflow=None,
):  # fmt: skip
    rng = np.random.default_rng(seed)
    A = rng.integers(-128, 128, (M, K))
    B = rng.integers(-128, 128, (K, N))
    cmds = setup(M, N, K, a_stride, b_stride, c_stride, a_addr=a_addr, dataflow=dataflow)
    result = run_isa(Matmul(cfg, buffers=buffers), cmds, MatmulCmd.START)
    assert result.done and not result.error, f"error {result.error_code:#x}"
    model = TopModel(cfg)
    place(model, a_addr, A, a_stride, 8)
    place(model, B_ADDR, B, b_stride, 8)
    model.run(result.commands)  # also checks the credits and tokens
    model.assert_all_free()
    np.testing.assert_array_equal(read_back(model, C_ADDR, M, N, c_stride, 32), A @ B)
    return result


class TestConstruction:
    def test_requires_square_array(self):
        with pytest.raises(ValueError, match="square"):
            Matmul(SystolicConfig(grid_rows=2, grid_cols=4))

    def test_requires_power_of_two(self):
        with pytest.raises(ValueError, match="power-of-two"):
            Matmul(config(grid=3))

    def test_requires_a_dataflow(self):
        with pytest.raises(ValueError, match="OS, WS or IS"):
            Matmul(config(dataflow=Dataflow(0)))

    @pytest.mark.parametrize(
        "dataflow,default",
        [
            (ALL, OS),
            (Dataflow.B_STATIONARY, WS),
            (Dataflow.A_STATIONARY, IS),
            (Dataflow.B_STATIONARY | Dataflow.A_STATIONARY, WS),
        ],
    )
    def test_default_dataflow(self, dataflow, default):
        """OS when the array has it (bubble-free with K tiles chained), else WS, else IS."""
        assert Matmul(config(dataflow=dataflow)).default_dataflow == default

    def test_verilog(self):
        skip_without_yosys()
        from amaranth.back import verilog

        assert "module Matmul" in verilog.convert(Matmul(config()), name="Matmul")


class TestCommandStream:
    @pytest.mark.parametrize("dataflow", DATAFLOWS, ids=DF_IDS)
    def test_commands_target_controllers_with_their_own_opcodes(self, dataflow):
        cfg = config()
        result = run_isa(Matmul(cfg), setup(6, 4, 3, 3, 4, 16, dataflow=dataflow), MatmulCmd.START)
        assert result.done
        cmds = result.commands
        assert (cmds[0].type, cmds[0].opcode, cmds[0].rs1) == (
            CTRL_EXEC, ExecOpcode.CONFIG, dataflow,
        )  # fmt: skip
        loads = [c for c in cmds if c.type == CTRL_LOAD]
        stores = [c for c in cmds if c.type == CTRL_STORE]
        execs = [c.opcode for c in cmds[1:] if c.type == CTRL_EXEC]
        # The corner turns lay the tiles out as the Streamer streams them
        T, M_ = DmaOpcode.TRANSPOSE, DmaOpcode.MEMCPY
        a_op, b_op = (M_, T) if dataflow == OS else (T, M_)
        assert [c.opcode for c in loads] == [a_op, b_op] * 2
        assert {c.opcode for c in stores} == {T if dataflow == IS else M_}
        # Two M tiles (4 + 2 rows), one N and one K tile: one LOAD per A and B tile,
        # one STORE per C tile
        if dataflow == OS:
            assert execs == [ExecOpcode.COMPUTE] * 2
            assert all(c.chain == 0 for c in cmds if c.type == CTRL_EXEC)  # one K tile each
        else:
            assert execs == [ExecOpcode.PRELOAD, ExecOpcode.COMPUTE] * 2
        assert [(c.len, c.stride, c.row_bytes) for c in loads] == [
            (4, 3, 3), (3, 4, 4), (2, 3, 3), (3, 4, 4),
        ]  # fmt: skip
        assert [(c.dram_addr, c.len, c.stride, c.row_bytes) for c in stores] == [
            (C_ADDR, 4, 16, 16), (C_ADDR + 4 * 16, 2, 16, 16),
        ]  # fmt: skip
        assert [c.id for c in cmds] == list(range(len(cmds)))

    def test_edge_tiles_move_only_their_columns(self):
        """N = 6 on a 4-wide array: the second N tile's B rows and C rows are 2 wide."""
        cfg = config()
        cmds = run_isa(Matmul(cfg), setup(4, 6, 4, 4, 6, 24, dataflow=WS), MatmulCmd.START).commands
        loads_b = [c for c in cmds if c.type == CTRL_LOAD][1::2]
        stores = [c for c in cmds if c.type == CTRL_STORE]
        assert [(c.dram_addr, c.row_bytes) for c in loads_b] == [(B_ADDR, 4), (B_ADDR + 4, 2)]
        assert [(c.dram_addr, c.row_bytes) for c in stores] == [(C_ADDR, 16), (C_ADDR + 16, 8)]

    def test_scratchpad_addresses_use_low_bank_bits(self):
        """Tile rows are consecutive local addresses (banks in the low bits), not bank << 16.

        Two K tiles use the two A/B slot pairs: A slots at rows 0 and dim, B slots at
        rows 2*dim and 3*dim."""
        cfg = config()
        cmds = run_isa(Matmul(cfg), setup(4, 4, 8, 8, 4, 16, dataflow=WS), MatmulCmd.START).commands
        loads = [c.local_addr for c in cmds if c.type == CTRL_LOAD]
        assert loads == [0, 8, 4, 12]
        preloads = [c for c in cmds if c.opcode == ExecOpcode.PRELOAD and c.type == CTRL_EXEC]
        assert [(c.rs1, c.k_dim) for c in preloads] == [(8, 4), (12, 4)]

    @pytest.mark.parametrize("dataflow", DATAFLOWS, ids=DF_IDS)
    def test_credits_and_tokens(self, dataflow):
        """Each tile is a slot: LOAD takes its credit and emits its token; PRELOAD and
        COMPUTE wait for the token and return the credit; COMPUTE takes the accumulator
        credit on the first K tile and emits the result token on the last; STORE waits
        for it and returns the credit. OS reads both tiles of a pair in one COMPUTE, so
        the pair is one slot (the A slot): its A LOAD takes the credit and its B LOAD
        emits the token."""
        cfg = config()
        # Two K tiles, two N tiles: the slot pairs and accumulator slots rotate
        cmds = run_isa(
            Matmul(cfg), setup(4, 8, 8, 8, 8, 32, dataflow=dataflow), MatmulCmd.START
        ).commands[1:]
        W, R, A, S = Sync.WAIT, Sync.RELEASE, Sync.ACQUIRE, Sync.SIGNAL

        def load(dst, flags=A | S):
            return (CTRL_LOAD, dst << Sync.DST | flags)

        def preload(src):
            return (CTRL_EXEC, src | W | R)

        def compute(src, acc, flags):
            return (CTRL_EXEC, src | acc << Sync.DST | W | R | flags)

        def store(acc):
            return (CTRL_STORE, acc | W | R)

        expected = []
        for n_tile in range(2):
            for k in range(2):
                acc = (A if k == 0 else 0) | (S if k == 1 else 0)
                if dataflow == OS:
                    expected += [load(k, A), load(k, S), compute(k, n_tile, acc)]
                elif dataflow == WS:
                    expected += [load(k), load(2 + k), preload(2 + k), compute(k, n_tile, acc)]
                else:
                    expected += [load(k), load(2 + k), preload(k), compute(2 + k, n_tile, acc)]
            expected += [store(n_tile)]
        assert [(c.type, c.sync) for c in cmds] == expected
        if dataflow == OS:  # the two K tiles are one reduction in the PEs
            chains = [c.chain for c in cmds if c.type == CTRL_EXEC]
            assert chains == [0b10, 0b01] * 2

    @pytest.mark.parametrize("dataflow", DATAFLOWS, ids=DF_IDS)
    def test_one_buffer_gives_the_same_results(self, dataflow):
        cfg = config()
        beat = cfg.dma_buswidth // 8
        for buffers in (1, 2):
            check_numerically(
                cfg, 9, 6, 10, 2 * beat, beat, 6 * 4, buffers=buffers, dataflow=dataflow
            )

    def test_buffers_must_fit(self):
        with pytest.raises(ValueError, match="L2 slots"):
            Matmul(config(), buffers=3)
        with pytest.raises(ValueError, match="scratchpad rows"):
            Matmul(config(grid=16, sp_capacity_kb=1, l2_slots=16, acc_slots=8), buffers=8)

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


@pytest.mark.parametrize("dataflow", DATAFLOWS, ids=DF_IDS)
class TestNumerical:
    @pytest.mark.parametrize("grid,tile", [(2, 1), (4, 1), (2, 2)])
    @pytest.mark.parametrize("M", [1, 4, 10])
    def test_beat_spaced_rows(self, grid, tile, M, dataflow):
        cfg = config(grid=grid, tile=tile)
        dim = grid * tile
        beat = cfg.dma_buswidth // 8
        check_numerically(cfg, M, dim, dim, beat, beat, beat, dataflow=dataflow)
        check_numerically(
            cfg, M, dim - 1, max(1, dim - 1), 2 * beat, beat, beat, seed=1, dataflow=dataflow
        )

    @pytest.mark.parametrize("M,N,K", [(7, 9, 6), (4, 4, 8), (9, 2, 13), (1, 11, 1)])
    def test_dense_tiled_in_m_n_and_k(self, M, N, K, dataflow):
        """Dense row-major A, B and C, tiled in every dimension."""
        check_numerically(config(), M, N, K, K, N, N * 4, seed=M + N + K, dataflow=dataflow)

    def test_unaligned_and_strided(self, dataflow):
        check_numerically(
            config(), 9, 7, 10, 13, 11, 31, a_addr=A_ADDR + 3, seed=5, dataflow=dataflow
        )


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

    @pytest.mark.parametrize("case", ["zero_a_stride", "zero_b_stride", "short_c_stride"])
    def test_rejected_layouts(self, case):
        args = {"M": 4, "N": 8, "K": 4, "a_stride": 4, "b_stride": 8, "c_stride": 32}
        if case == "zero_a_stride":
            args["a_stride"] = 0
        elif case == "zero_b_stride":
            args["b_stride"] = 0
        elif case == "short_c_stride":
            args["c_stride"] = 31  # C rows of 8 results would overlap
        result = self.run(config(), setup(**args))
        assert result.error_code == MatmulError.UNSUPPORTED_LAYOUT
        assert not result.commands

    @pytest.mark.parametrize(
        "dataflow,selected",
        [(Dataflow.OUTPUT_STATIONARY | Dataflow.B_STATIONARY, IS),
         (Dataflow.B_STATIONARY, OS), (Dataflow.OUTPUT_STATIONARY, WS)],
    )  # fmt: skip
    def test_dataflow_must_be_built(self, dataflow, selected):
        beat = config().dma_buswidth // 8
        cfg = config(dataflow=dataflow)
        result = self.run(cfg, setup(4, 4, 4, beat, beat, beat, dataflow=selected))
        assert result.error_code == MatmulError.UNSUPPORTED_DATAFLOW
        assert not result.commands

    def test_unaligned_layouts_are_accepted(self):
        """The BlockMover moves rows from any byte address: no alignment requirement."""
        result = self.run(config(), setup(5, 6, 7, 7, 6, 24, a_addr=A_ADDR + 1))
        assert result.done and not result.error


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


@pytest.mark.parametrize("dataflow", DATAFLOWS, ids=DF_IDS)
@pytest.mark.parametrize(
    "grid,tile,M,N,K",
    [(4, 1, 9, 7, 10), (2, 2, 4, 4, 4), (2, 2, 5, 9, 3), (2, 1, 3, 5, 6)],
)
def test_matmul_drives_systolic_top(grid, tile, M, N, K, dataflow):
    """Matmul issues every command to a real SystolicTop; dense C in L3 equals A @ B.

    A, B and C are dense row-major at odd addresses, tiled in M, N and K, with edge
    tiles in each; every byte outside C keeps its value."""
    cfg = config(grid=grid, tile=tile)
    beat = cfg.dma_buswidth // 8
    rng = np.random.default_rng(grid * 100 + M * 10 + N + K)
    A = rng.integers(-128, 128, (M, K))
    B = rng.integers(-128, 128, (K, N))
    a_addr, b_addr, c_addr = A_ADDR + 1, B_ADDR + 7, C_ADDR + 3
    memory = bytearray(rng.integers(0, 256, 1 << 17, dtype=np.uint8).tobytes())
    memory[a_addr : a_addr + M * K] = to_unsigned_bytes(A, 8)
    memory[b_addr : b_addr + K * N] = to_unsigned_bytes(B, 8)
    before = bytes(memory)

    dut = MatmulOnTop(cfg)
    isa, top = dut.isa, dut.top
    status = {}

    async def testbench(ctx):
        cmds = setup(M, N, K, K, N, 4 * N, a_addr=a_addr, dataflow=dataflow)
        cmds = [
            (
                c,
                b_addr
                if c == MatmulCmd.CONFIG_B_ADDR
                else c_addr
                if c == MatmulCmd.CONFIG_C_ADDR
                else d,
            )
            for c, d in cmds
        ]
        for cmd, data in [*cmds, (MatmulCmd.START, 0)]:
            ctx.set(isa.cfg_valid, 1)
            ctx.set(isa.cfg_cmd, cmd)
            ctx.set(isa.cfg_data, data)
            await ctx.tick()
        ctx.set(isa.cfg_valid, 0)
        done, error = await ctx.tick().sample(isa.done, isa.error).until(isa.done | isa.error)
        status.update(done=done, error=error)

    run_sim(dut, testbench, processes=[axi_memory(top, memory, beat)], max_cycles=200_000)
    assert status == {"done": 1, "error": 0}
    c_bytes = M * N * 4
    C = np.array(from_bytes(memory[c_addr : c_addr + c_bytes], 32, M * N)).reshape(M, N)
    np.testing.assert_array_equal(C, A @ B)
    assert (
        memory[:c_addr] == before[:c_addr]
        and memory[c_addr + c_bytes :] == before[c_addr + c_bytes :]
    )


def test_helpers_round_trip():
    data = to_unsigned_bytes([-1, 2, -128, 127], 8)
    assert from_bytes(data, 8, 4) == [-1, 2, -128, 127]
