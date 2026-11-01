"""
End-to-end GEMM through SystolicTop: LOAD A and B from DRAM, run the matmul on
the systolic array, STORE C to DRAM, and check C against ``A @ B``.

Runs output-stationary and weight-stationary, on 2x2 and 4x4 arrays, with
PEArrays of 1x1 and 2x2 PEs, and the default (multi-cycle) scratchpad and
accumulator latencies.
"""

import numpy as np
import pytest

from systars.config import SystolicConfig
from systars.top import SystolicTop
from systars.util.commands import DmaOpcode, ExecOpcode

from ._sim import run_sim

OS, WS = 0, 1

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


def axi_dram(dut, memory, beat_bytes):
    """AXI slave over ``memory`` (beat-aligned byte address -> beat value)."""

    async def process(ctx):
        reads = []  # [address, beats left]
        write = None  # [address, beat index]
        bresp_pending = False
        while True:
            ctx.set(dut.axi_arready, 1)
            ctx.set(dut.axi_awready, write is None and not bresp_pending)
            ctx.set(dut.axi_wready, write is not None)
            ctx.set(dut.axi_bvalid, bresp_pending)
            if reads:
                addr, left = reads[0]
                ctx.set(dut.axi_rvalid, 1)
                ctx.set(dut.axi_rdata, memory.get(addr, 0))
                ctx.set(dut.axi_rlast, left == 1)
            else:
                ctx.set(dut.axi_rvalid, 0)
                ctx.set(dut.axi_rlast, 0)
            (
                _,
                _,
                arvalid,
                araddr,
                arlen,
                rready,
                awvalid,
                awaddr,
                wvalid,
                wdata,
                wlast,
                bready,
            ) = await ctx.tick().sample(
                dut.axi_arvalid,
                dut.axi_araddr,
                dut.axi_arlen,
                dut.axi_rready,
                dut.axi_awvalid,
                dut.axi_awaddr,
                dut.axi_wvalid,
                dut.axi_wdata,
                dut.axi_wlast,
                dut.axi_bready,
            )
            if reads and rready:
                reads[0][0] += beat_bytes
                reads[0][1] -= 1
                if reads[0][1] == 0:
                    reads.pop(0)
            if arvalid:
                reads.append([araddr, arlen + 1])
            if bresp_pending and bready:
                bresp_pending = False
            if write is not None and wvalid:
                memory[write[0] + write[1] * beat_bytes] = wdata
                write[1] += 1
                if wlast:
                    write, bresp_pending = None, True
            elif write is None and not bresp_pending and awvalid:
                write = [awaddr, 0]

    return process


class Host:
    """Issues SystolicTop commands and waits for their completions."""

    COMPLETED = {
        SystolicTop.CTRL_LOAD: "load_completed",
        SystolicTop.CTRL_STORE: "store_completed",
        SystolicTop.CTRL_EXEC: "exec_completed",
    }

    def __init__(self, ctx, dut):
        self.ctx, self.dut, self.next_id = ctx, dut, 1

    async def run(self, cmd_type, opcode, **fields):
        ctx, dut = self.ctx, self.dut
        cmd_id = self.next_id
        self.next_id += 1
        for name in ("dram_addr", "local_addr", "len", "rs1", "rs2", "rd", "k_dim", "activation"):
            ctx.set(getattr(dut, f"cmd_{name}"), fields.get(name, 0))
        ctx.set(dut.cmd_type, cmd_type)
        ctx.set(dut.cmd_opcode, opcode)
        ctx.set(dut.cmd_id, cmd_id)
        ctx.set(dut.cmd_valid, 1)
        await ctx.tick().until(dut.cmd_ready)
        ctx.set(dut.cmd_valid, 0)
        done = getattr(dut, self.COMPLETED[cmd_type])
        *_, completed_id = await ctx.tick().sample(getattr(dut, f"{done.name}_id")).until(done)
        assert completed_id == cmd_id

    async def load(self, dram_addr, sp_addr, rows):
        await self.run(
            SystolicTop.CTRL_LOAD,
            DmaOpcode.MEMCPY,
            dram_addr=dram_addr,
            local_addr=sp_addr,
            len=rows,
        )

    async def store(self, acc_addr, dram_addr, rows):
        await self.run(
            SystolicTop.CTRL_STORE,
            DmaOpcode.MEMCPY,
            dram_addr=dram_addr,
            local_addr=acc_addr,
            len=rows,
        )

    async def execute(self, opcode, **fields):
        await self.run(SystolicTop.CTRL_EXEC, opcode, **fields)


A_DRAM, B_DRAM, C_DRAM = 0x1000, 0x2000, 0x3000
A_SP, B_SP = 0, 64


def gemm(cfg, dataflow, A, B, *, weight_rows=None, repeat=1):
    """Run ``A @ B`` through SystolicTop and return C as read back from DRAM.

    OS: A is R x K and B is K x C. WS: A is M x K and B is K x C with K <= R.
    With ``repeat`` > 1 the COMPUTE is issued again as COMPUTE_ACCUMULATE.
    """
    rows = cfg.grid_rows * cfg.tile_rows
    cols = cfg.grid_cols * cfg.tile_cols
    beat = cfg.dma_buswidth // 8
    memory = {}

    # A rows as stored: OS keeps Aᵀ (row k = column k of A), WS keeps A
    a_rows = A.T if dataflow == OS else A
    for n, row in enumerate(a_rows):
        memory[A_DRAM + n * beat] = pack(row, cfg.input_bits)
    for n, row in enumerate(B):
        memory[B_DRAM + n * beat] = pack(row, cfg.weight_bits)
    out_rows = rows if dataflow == OS else A.shape[0]

    dut = SystolicTop(cfg)

    async def testbench(ctx):
        host = Host(ctx, dut)
        await host.load(A_DRAM, A_SP, len(a_rows))
        await host.load(B_DRAM, B_SP, len(B))
        await host.execute(ExecOpcode.CONFIG, rs1=dataflow)
        if dataflow == WS:
            await host.execute(ExecOpcode.PRELOAD, rs1=B_SP, k_dim=weight_rows or len(B))
        for n in range(repeat):
            opcode = ExecOpcode.COMPUTE if n == 0 else ExecOpcode.COMPUTE_ACCUMULATE
            await host.execute(opcode, rs1=A_SP, rs2=B_SP, rd=0, k_dim=len(a_rows))
        await host.store(0, C_DRAM, out_rows)

    run_sim(dut, testbench, processes=[axi_dram(dut, memory, beat)], max_cycles=10_000)
    return np.array(
        [unpack(memory.get(C_DRAM + n * beat, 0), cfg.acc_bits, cols) for n in range(out_rows)]
    )


def operands(rng, rows_a, k, cols):
    A = rng.integers(-128, 128, (rows_a, k))
    B = rng.integers(-128, 128, (k, cols))
    return A, B


@pytest.mark.parametrize("shape", SHAPES, ids=SHAPE_IDS)
class TestGemm:
    def test_output_stationary(self, shape):
        cfg = make_config(*shape)
        rows, cols = cfg.grid_rows * cfg.tile_rows, cfg.grid_cols * cfg.tile_cols
        A, B = operands(np.random.default_rng(1), rows, 7, cols)
        np.testing.assert_array_equal(gemm(cfg, OS, A, B), A @ B)

    def test_weight_stationary(self, shape):
        cfg = make_config(*shape)
        rows, cols = cfg.grid_rows * cfg.tile_rows, cfg.grid_cols * cfg.tile_cols
        A, B = operands(np.random.default_rng(2), 5, rows, cols)
        np.testing.assert_array_equal(gemm(cfg, WS, A, B), A @ B)


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
