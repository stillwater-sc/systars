"""
End-to-end GEMM through SystolicTop: LOAD the operand tiles from L3, run the
matmul through L1 and the fabric, STORE C to L3, and check C against ``A @ B``.

Runs output-stationary, weight-stationary and input-stationary, on 2x2 and 4x4
arrays, with PEArrays of 1x1 and 2x2 PEs, and the default (multi-cycle)
scratchpad and accumulator latencies. The commands are untracked (``cmd_sync`` 0):
the host issues one at a time and waits for each completion.

The host lays each L2 tile out in L3 as the Streamer reads it (one L2 row per
block, ``controller/execute.py``) and moves it with MEMCPY:

- OS: per K tile, A row-major (row ``i`` = ``A[i, k0:k0+KB]``) and B column-major
  (row ``j`` = ``B[k0:k0+KB, j]``); the K tiles are one reduction (``cmd_chain``).
- WS: B row-major (PRELOAD); per M tile, A column-major (row ``k`` = ``A[m0:, k]``).
- IS: A column-major (PRELOAD: the stationary tile is Aᵀ); B row-major streamed
  (row ``k`` = ``B[k, n0:]``); the accumulator holds Cᵀ and STORE TRANSPOSE writes C.
"""

import numpy as np
import pytest

from systars.config import SystolicConfig
from systars.controller.execute import CHAIN_CONTINUES, CHAIN_MORE
from systars.top import SystolicTop
from systars.util.commands import DmaOpcode, ExecOpcode

from ._axi import axi_memory
from ._sim import run_sim

OS, WS, IS = 0, 1, 2

# (grid_rows, grid_cols, tile_rows, tile_cols): 2x2 and 4x4 PE arrays
SHAPES = [(2, 2, 1, 1), (1, 1, 2, 2), (4, 4, 1, 1), (2, 2, 2, 2)]
SHAPE_IDS = ["2x2-grid", "2x2-pearray", "4x4-grid", "4x4-2x2pearrays"]


def make_config(grid_rows, grid_cols, tile_rows, tile_cols):
    return SystolicConfig(
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        tile_rows=tile_rows,
        tile_cols=tile_cols,
        sp_capacity_kb=4,
        acc_capacity_kb=4,
    )


def pack(values, bits):
    """Pack ``values`` into one integer, element 0 in the low bits."""
    word = 0
    for i, v in enumerate(values):
        word |= (int(v) & ((1 << bits) - 1)) << (i * bits)
    return word


def unpack(word, bits, n):
    out = []
    for i in range(n):
        v = (word >> (i * bits)) & ((1 << bits) - 1)
        out.append(v - (1 << bits) if v >> (bits - 1) else v)
    return out


class Host:
    """Issues SystolicTop commands and waits for their completions."""

    COMPLETED = {
        SystolicTop.CTRL_LOAD: "load_completed",
        SystolicTop.CTRL_STORE: "store_completed",
        SystolicTop.CTRL_EXEC: "exec_completed",
    }
    FIELDS = ("dram_addr", "local_addr", "len", "rs1", "rs2", "rd", "k_dim", "chain",
              "activation")  # fmt: skip
    OPTIONAL = ("stride", "row_bytes")  # left as the caller set them unless given

    def __init__(self, ctx, dut):
        self.ctx, self.dut, self.next_id = ctx, dut, 1

    async def run(self, cmd_type, opcode, **fields):
        ctx, dut = self.ctx, self.dut
        cmd_id = self.next_id
        self.next_id = self.next_id % 255 + 1
        for name in self.FIELDS:
            ctx.set(getattr(dut, f"cmd_{name}"), fields.get(name, 0))
        for name in self.OPTIONAL:
            if name in fields:
                ctx.set(getattr(dut, f"cmd_{name}"), fields[name])
        ctx.set(dut.cmd_type, cmd_type)
        ctx.set(dut.cmd_opcode, opcode)
        ctx.set(dut.cmd_id, cmd_id)
        ctx.set(dut.cmd_valid, 1)
        await ctx.tick().until(dut.cmd_ready)
        ctx.set(dut.cmd_valid, 0)
        done = getattr(dut, self.COMPLETED[cmd_type])
        *_, completed_id = await ctx.tick().sample(getattr(dut, f"{done.name}_id")).until(done)
        assert completed_id == cmd_id
        assert ctx.get(dut.l1_error) == 0

    async def load(self, l3_addr, sp_addr, rows, row_bytes=None):
        """MEMCPY ``rows`` L3 rows (``row_bytes`` apart, when given) into L2."""
        layout = {} if row_bytes is None else {"stride": row_bytes, "row_bytes": row_bytes}
        await self.run(SystolicTop.CTRL_LOAD, DmaOpcode.MEMCPY, dram_addr=l3_addr,
                       local_addr=sp_addr, len=rows, **layout)  # fmt: skip

    async def store(self, acc_addr, l3_addr, rows, row_bytes=None, *, opcode=DmaOpcode.MEMCPY):
        layout = {} if row_bytes is None else {"stride": row_bytes, "row_bytes": row_bytes}
        await self.run(SystolicTop.CTRL_STORE, opcode, dram_addr=l3_addr,
                       local_addr=acc_addr, len=rows, **layout)  # fmt: skip

    async def execute(self, opcode, **fields):
        await self.run(SystolicTop.CTRL_EXEC, opcode, **fields)


A_L3, B_L3, C_L3 = 0x1000, 0x4000, 0x8000
A_SP, B_SP = 0, 128


class Layout:
    """L2 rows as the host lays them out in L3, and moved to L2 with one LOAD."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.rows = cfg.grid_rows * cfg.tile_rows
        self.cols = cfg.grid_cols * cfg.tile_cols
        self.kb = min(cfg.sp_width // cfg.input_bits, cfg.sp_width // cfg.weight_bits)
        self.row_bytes = cfg.sp_width // 8

    def image(self, l2_rows, bits):
        """Bytes of ``l2_rows`` (each a list of at most KB elements, zero-padded)."""
        out = bytearray()
        for row in l2_rows:
            out += pack(row, bits).to_bytes(self.row_bytes, "little")
        return out


def tiles(n, size):
    """``(start, length)`` of the ``size``-element pieces of ``range(n)``."""
    return [(s, min(size, n - s)) for s in range(0, n, size)]


def gemm(cfg, mode, A, B, *, weight_rows=None, repeat=1, poison=None):
    """Run ``A @ B`` through SystolicTop and return C as read back from L3.

    OS: A is R x K and B is K x C. WS: A is M x K and B is K x C with K <= R.
    IS: A is M x K and B is K x N with K <= R and M, N <= C. With ``repeat`` > 1
    the computation is issued again with COMPUTE_ACCUMULATE. With ``poison`` (a
    numpy Generator) every scratchpad and accumulator word starts random, not
    zero, as on hardware with no power-up contents.
    """
    lay = Layout(cfg)
    R, C, KB = lay.rows, lay.cols, lay.kb
    beat = cfg.dma_buswidth // 8
    acc_bytes = cfg.acc_width // 8
    memory = bytearray(1 << 16)
    exec_cmds = []  # (opcode, fields) per repeat-independent command
    if mode == OS:
        a_rows, b_rows = [], []
        k_tiles = tiles(A.shape[1], KB)
        for k0, kn in k_tiles:
            a_rows += [list(A[i, k0 : k0 + kn]) for i in range(R)]
            b_rows += [list(B[k0 : k0 + kn, j]) for j in range(C)]
        for t, (_, kn) in enumerate(k_tiles):
            chain = (CHAIN_CONTINUES if t else 0) | (CHAIN_MORE if t < len(k_tiles) - 1 else 0)
            exec_cmds.append(
                {"rs1": A_SP + t * R, "rs2": B_SP + t * C, "rd": 0, "k_dim": kn, "chain": chain}
            )
        preload, out_rows = None, R
    elif mode == WS:
        b_rows = [list(row) for row in B]
        a_rows = []
        m_tiles = tiles(A.shape[0], KB)
        for m0, mn in m_tiles:
            a_rows += [list(A[m0 : m0 + mn, k]) if k < A.shape[1] else [0] * mn for k in range(R)]
        for t, (m0, mn) in enumerate(m_tiles):
            exec_cmds.append({"rs1": A_SP + t * R, "rd": m0, "k_dim": mn})
        preload, out_rows = weight_rows or B.shape[0], A.shape[0]
    else:  # IS: the stationary tile is Aᵀ (row k = A[:, k]); B rows stream
        b_rows = [list(A[:, k]) for k in range(A.shape[1])]
        a_rows = []
        n_tiles = tiles(B.shape[1], KB)
        for n0, nn in n_tiles:
            a_rows += [list(B[k, n0 : n0 + nn]) if k < B.shape[0] else [0] * nn for k in range(R)]
        for t, (n0, nn) in enumerate(n_tiles):
            exec_cmds.append({"rs1": A_SP + t * R, "rd": n0, "k_dim": nn})
        preload, out_rows = A.shape[1], A.shape[0]

    a_image = lay.image(a_rows, cfg.input_bits)
    b_image = lay.image(b_rows, cfg.weight_bits)
    memory[A_L3 : A_L3 + len(a_image)] = a_image
    memory[B_L3 : B_L3 + len(b_image)] = b_image
    dut = SystolicTop(cfg)

    async def testbench(ctx):
        if poison is not None:
            for bank in [*dut.scratchpad.banks, *dut.accumulator.banks]:
                width = bank.mem.shape.width
                for addr in range(bank.mem.depth):
                    word = int.from_bytes(poison.bytes((width + 7) // 8), "little")
                    ctx.set(bank.mem.data[addr], word % (1 << width))  # every bit random
        host = Host(ctx, dut)
        await host.load(A_L3, A_SP, len(a_rows), lay.row_bytes)
        await host.load(B_L3, B_SP, len(b_rows), lay.row_bytes)
        await host.execute(ExecOpcode.CONFIG, rs1=mode)
        if preload is not None:
            await host.execute(ExecOpcode.PRELOAD, rs1=B_SP, k_dim=preload)
        for n in range(repeat):
            opcode = ExecOpcode.COMPUTE if n == 0 else ExecOpcode.COMPUTE_ACCUMULATE
            for fields in exec_cmds:
                await host.execute(opcode, **fields)
        if mode == IS:
            # N accumulator rows of M results (Cᵀ) -> M L3 rows of N results (C)
            n_cols = B.shape[1]
            await host.store(0, C_L3, out_rows, 4 * n_cols, opcode=DmaOpcode.TRANSPOSE)
        else:
            await host.store(0, C_L3, out_rows, acc_bytes)

    run_sim(dut, testbench, processes=[axi_memory(dut, memory, beat)], max_cycles=20_000)
    width = B.shape[1] if mode == IS else C
    row = 4 * width if mode == IS else acc_bytes
    return np.array(
        [
            unpack(int.from_bytes(memory[C_L3 + n * row : C_L3 + (n + 1) * row], "little"),
                   cfg.acc_bits, width)
            for n in range(out_rows)
        ]
    )  # fmt: skip


def operands(rng, rows_a, k, cols):
    A = rng.integers(-128, 128, (rows_a, k))
    B = rng.integers(-128, 128, (k, cols))
    return A, B


@pytest.mark.parametrize("shape", SHAPES, ids=SHAPE_IDS)
class TestGemm:
    def test_output_stationary(self, shape):
        """K = 7: a reduction chained across several K tiles."""
        cfg = make_config(*shape)
        lay = Layout(cfg)
        A, B = operands(np.random.default_rng(1), lay.rows, 7, lay.cols)
        np.testing.assert_array_equal(gemm(cfg, OS, A, B), A @ B)

    def test_weight_stationary(self, shape):
        cfg = make_config(*shape)
        lay = Layout(cfg)
        A, B = operands(np.random.default_rng(2), 5, lay.rows, lay.cols)
        np.testing.assert_array_equal(gemm(cfg, WS, A, B), A @ B)

    def test_input_stationary(self, shape):
        cfg = make_config(*shape)
        lay = Layout(cfg)
        A, B = operands(np.random.default_rng(3), lay.cols, lay.rows, lay.cols)
        np.testing.assert_array_equal(gemm(cfg, IS, A, B), A @ B)


class TestGemmOptions:
    CFG = (2, 2, 1, 1)

    def test_os_accumulate(self):
        cfg = make_config(*self.CFG)
        A, B = operands(np.random.default_rng(3), 2, 3, 2)
        np.testing.assert_array_equal(gemm(cfg, OS, A, B, repeat=2), 2 * (A @ B))

    def test_ws_accumulate(self):
        cfg = make_config(*self.CFG)
        A, B = operands(np.random.default_rng(4), 3, 2, 2)
        np.testing.assert_array_equal(gemm(cfg, WS, A, B, repeat=2), 2 * (A @ B))

    def test_ws_fewer_weight_rows_than_pe_rows(self):
        """K < R: the PE rows below K get zero weights, so A needs no padding."""
        cfg = make_config(4, 4, 1, 1)
        rng = np.random.default_rng(5)
        A = rng.integers(-128, 128, (3, 4))  # columns 2..3 are garbage
        B = rng.integers(-128, 128, (2, 4))
        np.testing.assert_array_equal(gemm(cfg, WS, A, B, weight_rows=2), A[:, :2] @ B)

    def test_os_extreme_values(self):
        cfg = make_config(*self.CFG)
        A = np.full((2, 4), -128)
        B = np.full((4, 2), -128)
        np.testing.assert_array_equal(gemm(cfg, OS, A, B), A @ B)

    def test_is_narrow_tile(self):
        """IS with M, N < C and K < R: edge tiles."""
        cfg = make_config(4, 4, 1, 1)
        A, B = operands(np.random.default_rng(6), 3, 2, 2)
        np.testing.assert_array_equal(gemm(cfg, IS, A, B), A @ B)


@pytest.mark.parametrize("mode", [OS, WS, IS], ids=["os", "ws", "is"])
def test_no_read_before_write(mode):
    """The design never reads a scratchpad or accumulator word it hasn't written.

    Every word starts random (as on hardware: `just synth polarfire` drops the
    memories' zero initial values, which only simulation has), and C is still exact,
    with partial L2 rows (K and M not multiples of the block) and COMPUTE_ACCUMULATE.
    """
    cfg = make_config(2, 2, 2, 2)
    lay = Layout(cfg)
    rng = np.random.default_rng(7)
    if mode == OS:
        A, B = operands(rng, lay.rows, 7, lay.cols)
    elif mode == WS:
        A, B = operands(rng, 5, lay.rows, lay.cols)
    else:
        A, B = operands(rng, lay.cols, lay.rows, lay.cols)
    C = gemm(cfg, mode, A, B, repeat=2, poison=np.random.default_rng(8))
    np.testing.assert_array_equal(C, 2 * (A @ B))


@pytest.mark.parametrize("dim", [8, 16])
@pytest.mark.parametrize("mode", [OS, WS, IS], ids=["os", "ws", "is"])
def test_larger_arrays(dim, mode):
    """8x8 and 16x16 arrays: OS with two K tiles chained, WS and IS one full tile."""
    cfg = SystolicConfig(grid_rows=dim, grid_cols=dim, sp_capacity_kb=16, acc_capacity_kb=16)
    rng = np.random.default_rng(dim + mode)
    if mode == OS:
        A, B = operands(rng, dim, 2 * dim, dim)
    else:
        A, B = operands(rng, dim, dim, dim)
    np.testing.assert_array_equal(gemm(cfg, mode, A, B), A @ B)
