"""
The tile's BlockMover legs (kpu-sim's CSP model; issue #10): LOAD moves a tile
from L3 into the scratchpad, STORE moves a result tile from the accumulator to
L3, each from or to any byte address, ``stride`` bytes between rows, with
``row_bytes`` valid bytes per row.

- LoadController + StreamReader, and StoreController + StreamWriter, over a
  byte-addressed AXI memory: every byte offset in a beat, rows spilling into the
  next beat, partial rows, strides, streamed bursts longer than 256 beats, and
  zero-length commands.
- A GEMM through SystolicTop with A, B and C at unaligned, strided addresses,
  C written with partial rows between canary bytes.
- Bank counts: one bank works, a count that is not a power of two is rejected.
"""

import numpy as np
import pytest
from amaranth import Elaboratable, Module, Signal

from systars.config import SystolicConfig
from systars.controller.load import LoadController
from systars.controller.store import StoreController
from systars.dma.reader import StreamReader
from systars.dma.writer import StreamWriter
from systars.top import SystolicTop
from systars.util.commands import DmaOpcode, ExecOpcode

from ._axi import axi_memory
from ._sim import run_sim
from .test_gemm_systolic import Host, pack, unpack

WS = 1


def config(dim=4, **overrides):
    fields = {"grid_rows": dim, "grid_cols": dim, "sp_capacity_kb": 4, "acc_capacity_kb": 4}
    fields.update(overrides)
    return SystolicConfig(**fields)


class _AxiPorts:
    """Alias a DMA engine's ``mem_*`` ports as ``axi_*``, with idle stand-ins for the
    other direction, so ``axi_memory`` can serve one engine."""

    READ = ("arvalid", "arready", "araddr", "arlen", "rvalid", "rready", "rdata", "rlast")
    WRITE = ("awvalid", "awready", "awaddr", "wvalid", "wready", "wdata", "wstrb", "wlast")
    WRITE += ("bvalid", "bready")

    def _alias(self, engine, names, present):
        for name in names:
            port = getattr(engine, f"mem_{name}") if present else Signal(64, name=f"idle_{name}")
            setattr(self, f"axi_{name}", port)


class LoadPath(Elaboratable, _AxiPorts):
    """LoadController + StreamReader; scratchpad writes are recorded by the testbench."""

    def __init__(self, cfg):
        self.ctrl = LoadController(cfg)
        self.reader = StreamReader(cfg)
        self._alias(self.reader, self.READ, True)
        self._alias(None, self.WRITE, False)

    def elaborate(self, _platform):
        m = Module()
        m.submodules.ctrl, m.submodules.reader = self.ctrl, self.reader
        c, r = self.ctrl, self.reader
        m.d.comb += [
            r.req_valid.eq(c.dma_req_valid),
            c.dma_req_ready.eq(r.req_ready),
            r.req_addr.eq(c.dma_req_addr),
            r.req_len.eq(c.dma_req_len),
            c.dma_resp_valid.eq(r.resp_valid),
            r.resp_ready.eq(c.dma_resp_ready),
            c.dma_resp_data.eq(r.resp_data),
            c.dma_resp_last.eq(r.resp_last),
        ]
        return m


class StorePath(Elaboratable, _AxiPorts):
    """StoreController + StreamWriter, reading a model accumulator."""

    def __init__(self, cfg):
        self.ctrl = StoreController(cfg)
        self.writer = StreamWriter(cfg)
        self._alias(self.writer, self.WRITE, True)
        self._alias(None, self.READ, False)

    def elaborate(self, _platform):
        m = Module()
        m.submodules.ctrl, m.submodules.writer = self.ctrl, self.writer
        c, w = self.ctrl, self.writer
        m.d.comb += [
            w.req_valid.eq(c.dma_req_valid),
            c.dma_req_ready.eq(w.req_ready),
            w.req_addr.eq(c.dma_req_addr),
            w.req_len.eq(c.dma_req_len),
            w.data_valid.eq(c.dma_data_valid),
            c.dma_data_ready.eq(w.data_ready),
            w.data.eq(c.dma_data),
            w.data_strb.eq(c.dma_data_strb),
            w.data_last.eq(c.dma_data_last),
            c.dma_done.eq(w.done),
        ]
        return m


async def issue(ctx, ctrl, local_field, *, dram_addr, local_addr, rows, stride=0, row_bytes=0):
    ctx.set(ctrl.cmd_opcode, DmaOpcode.MEMCPY)
    ctx.set(ctrl.cmd_dram_addr, dram_addr)
    ctx.set(getattr(ctrl, local_field), local_addr)
    ctx.set(ctrl.cmd_len, rows)
    ctx.set(ctrl.cmd_stride, stride)
    ctx.set(ctrl.cmd_row_bytes, row_bytes)
    ctx.set(ctrl.cmd_id, 7)
    ctx.set(ctrl.cmd_valid, 1)
    await ctx.tick().until(ctrl.cmd_ready)
    ctx.set(ctrl.cmd_valid, 0)
    await ctx.tick().until(ctrl.completed)


def run_load(cfg, memory, **cmd):
    """Run one LOAD; return {scratchpad row: bytes written}."""
    dut = LoadPath(cfg)
    sp_bytes = cfg.sp_width // 8
    written = {}

    async def recorder(ctx):
        async for _, _, en, addr, data in ctx.tick().sample(
            dut.ctrl.sp_write_en, dut.ctrl.sp_write_addr, dut.ctrl.sp_write_data
        ):
            if en:
                assert addr not in written, f"scratchpad row {addr} written twice"
                written[addr] = data.to_bytes(sp_bytes, "little")

    async def testbench(ctx):
        await issue(ctx, dut.ctrl, "cmd_sp_addr", **cmd)

    beat = cfg.dma_buswidth // 8
    run_sim(dut, testbench, processes=[axi_memory(dut, memory, beat), recorder])
    return written


def run_store(cfg, memory, acc_rows, *, activation=0, **cmd):
    """Run one STORE of ``acc_rows`` (lists of acc_bits values) into ``memory``."""
    dut = StorePath(cfg)

    async def accumulator(ctx):
        pending = None
        while True:
            ctx.set(dut.ctrl.acc_read_valid, pending is not None)
            ctx.set(
                dut.ctrl.acc_read_data,
                pack(acc_rows[pending], cfg.acc_bits) if pending is not None else 0,
            )
            _, _, req, addr = await ctx.tick().sample(dut.ctrl.acc_read_req, dut.ctrl.acc_read_addr)
            pending = addr if req else None

    async def testbench(ctx):
        ctx.set(dut.ctrl.cmd_activation, activation)
        await issue(ctx, dut.ctrl, "cmd_acc_addr", **cmd)

    beat = cfg.dma_buswidth // 8
    run_sim(dut, testbench, processes=[axi_memory(dut, memory, beat), accumulator])


def pattern(n, seed=0):
    return bytearray(np.random.default_rng(seed).integers(0, 256, n, dtype=np.uint8).tobytes())


class TestLoad:
    @pytest.mark.parametrize("offset", [0, 1, 5, 12, 13, 15])
    def test_any_byte_offset(self, offset):
        """A row at any offset in a beat, including rows that spill into the next beat."""
        cfg = config()  # 4-byte scratchpad rows, 16-byte beats
        memory = pattern(1 << 12)
        base = 0x100 + offset
        written = run_load(cfg, memory, dram_addr=base, local_addr=3, rows=3, stride=40)
        assert written == {
            3 + r: bytes(memory[base + 40 * r : base + 40 * r + 4]) for r in range(3)
        }

    def test_partial_rows_are_zero_filled(self):
        cfg = config()
        memory = pattern(1 << 12, seed=1)
        written = run_load(
            cfg, memory, dram_addr=0x203, local_addr=0, rows=2, stride=7, row_bytes=3
        )
        assert written == {
            r: bytes(memory[0x203 + 7 * r : 0x203 + 7 * r + 3]) + b"\0" for r in range(2)
        }

    def test_dense_rows_closer_than_a_beat(self):
        """Dense row-major int8: rows dim bytes apart, many to a beat."""
        cfg = config(dim=8)  # 8-byte rows, 16-byte beats
        memory = pattern(1 << 12, seed=2)
        written = run_load(cfg, memory, dram_addr=0x40 + 6, local_addr=0, rows=5, stride=8)
        assert written == {r: bytes(memory[0x46 + 8 * r : 0x46 + 8 * r + 8]) for r in range(5)}

    def test_consecutive_beats_stream_past_256(self):
        """Aligned, beat-spaced rows stream; more than 256 rows take several bursts."""
        cfg = config(sp_capacity_kb=4)  # 1024 scratchpad rows
        beat = cfg.dma_buswidth // 8
        memory = pattern(300 * beat + 0x1000, seed=3)
        written = run_load(cfg, memory, dram_addr=0x1000, local_addr=0, rows=300)
        assert len(written) == 300
        assert all(
            written[r] == bytes(memory[0x1000 + r * beat : 0x1000 + r * beat + 4])
            for r in range(300)
        )

    def test_zero_rows_complete_without_a_transfer(self):
        assert run_load(config(), pattern(256), dram_addr=0x10, local_addr=0, rows=0) == {}

    @pytest.mark.parametrize("offset", [1, 13])
    def test_beat_spaced_rows_off_a_boundary_do_not_stream(self, offset):
        """Rows one beat apart but mid-beat are not consecutive beats: each is its own burst."""
        cfg = config()
        beat = cfg.dma_buswidth // 8
        memory = pattern(1 << 12, seed=9)
        base = 0x500 + offset
        written = run_load(cfg, memory, dram_addr=base, local_addr=0, rows=3, stride=beat)
        assert written == {
            r: bytes(memory[base + r * beat : base + r * beat + 4]) for r in range(3)
        }


class TestStore:
    ROWS = [[1, -2, 3, -4], [0x11223344, -1, 7, 8], [-(1 << 31), (1 << 31) - 1, 0, 5]]

    @pytest.mark.parametrize("offset", [0, 4, 9, 15])
    def test_any_byte_offset_leaves_neighbours(self, offset):
        cfg = config()  # 16-byte accumulator rows, 16-byte beats
        memory = pattern(1 << 12, seed=4)
        before = bytes(memory)
        base = 0x300 + offset
        run_store(cfg, memory, self.ROWS, dram_addr=base, local_addr=0, rows=3, stride=24)
        expected = bytearray(before)
        for r, row in enumerate(self.ROWS):
            expected[base + 24 * r : base + 24 * r + 16] = pack(row, 32).to_bytes(16, "little")
        assert memory == expected

    @pytest.mark.parametrize("offset", [4, 9])
    def test_beat_spaced_rows_off_a_boundary_do_not_stream(self, offset):
        cfg = config()
        beat = cfg.dma_buswidth // 8
        memory = pattern(1 << 12, seed=10)
        before = bytes(memory)
        base = 0x600 + offset
        run_store(cfg, memory, self.ROWS, dram_addr=base, local_addr=0, rows=3, stride=beat)
        expected = bytearray(before)
        for r, row in enumerate(self.ROWS):
            expected[base + beat * r : base + beat * r + 16] = pack(row, 32).to_bytes(16, "little")
        assert memory == expected

    def test_partial_rows_write_only_their_bytes(self):
        """An edge N tile: 2 of the 4 results per row, C rows dense."""
        cfg = config()
        memory = pattern(1 << 12, seed=5)
        before = bytes(memory)
        run_store(
            cfg, memory, self.ROWS, dram_addr=0x402, local_addr=0, rows=3, stride=8, row_bytes=8
        )
        expected = bytearray(before)
        for r, row in enumerate(self.ROWS):
            expected[0x402 + 8 * r : 0x402 + 8 * r + 8] = pack(row[:2], 32).to_bytes(8, "little")
        assert memory == expected

    def test_activation_applies_per_element(self):
        cfg = config()
        memory = bytearray(1 << 10)
        run_store(
            cfg, memory, self.ROWS[:1], activation=1, dram_addr=0x20 + 3, local_addr=0, rows=1
        )
        assert unpack(int.from_bytes(memory[0x23 : 0x23 + 16], "little"), 32, 4) == [1, 0, 3, 0]

    def test_consecutive_beats_stream_past_256(self):
        cfg = config(acc_capacity_kb=8)  # 512 accumulator rows
        beat = cfg.dma_buswidth // 8
        rows = [[r, -r, 2 * r, 3] for r in range(300)]
        memory = bytearray(300 * beat + 0x1000)
        run_store(cfg, memory, rows, dram_addr=0x1000, local_addr=0, rows=300)
        stored = [
            unpack(
                int.from_bytes(memory[0x1000 + r * beat : 0x1000 + (r + 1) * beat], "little"), 32, 4
            )
            for r in range(300)
        ]
        assert stored == rows

    def test_zero_rows_complete_without_a_transfer(self):
        memory = pattern(256, seed=6)
        before = bytes(memory)
        run_store(config(), memory, self.ROWS, dram_addr=0x10, local_addr=0, rows=0)
        assert memory == before


A_ADDR, B_ADDR, C_ADDR = 0x1003, 0x2009, 0x3005


def gemm_any_layout(cfg, A, B, *, a_stride, b_stride, c_stride):
    """WS GEMM (one tile) through SystolicTop, operands and C at unaligned, strided addresses.

    C is written with row_bytes = N results, so the bytes past each row keep their canary."""
    dim = cfg.grid_rows * cfg.tile_rows
    beat = cfg.dma_buswidth // 8
    M, K = A.shape
    N = B.shape[1]
    memory = pattern(1 << 16, seed=7)
    for r, row in enumerate(A):
        memory[A_ADDR + r * a_stride : A_ADDR + r * a_stride + K] = pack(row, 8).to_bytes(
            K, "little"
        )
    for r, row in enumerate(B):
        memory[B_ADDR + r * b_stride : B_ADDR + r * b_stride + N] = pack(row, 8).to_bytes(
            N, "little"
        )
    before = bytes(memory)
    dut = SystolicTop(cfg)
    sp_b = dim  # B rows follow the A rows

    async def testbench(ctx):
        host = Host(ctx, dut)
        ctx.set(dut.cmd_stride, a_stride)
        ctx.set(dut.cmd_row_bytes, K)
        await host.load(A_ADDR, 0, M)
        ctx.set(dut.cmd_stride, b_stride)
        ctx.set(dut.cmd_row_bytes, N)
        await host.load(B_ADDR, sp_b, K)
        await host.execute(ExecOpcode.CONFIG, rs1=WS)
        await host.execute(ExecOpcode.PRELOAD, rs1=sp_b, k_dim=K)
        await host.execute(ExecOpcode.COMPUTE, rs1=0, rd=0, k_dim=M)
        ctx.set(dut.cmd_stride, c_stride)
        ctx.set(dut.cmd_row_bytes, 4 * N)
        await host.store(0, C_ADDR, M)

    run_sim(dut, testbench, processes=[axi_memory(dut, memory, beat)], max_cycles=20_000)
    C = np.array(
        [
            unpack(
                int.from_bytes(
                    memory[C_ADDR + r * c_stride : C_ADDR + r * c_stride + 4 * N], "little"
                ),
                32,
                N,
            )
            for r in range(M)
        ]
    )
    # Everything outside C's rows is untouched
    expected = bytearray(memory)
    for r in range(M):
        expected[C_ADDR + r * c_stride : C_ADDR + r * c_stride + 4 * N] = before[
            C_ADDR + r * c_stride : C_ADDR + r * c_stride + 4 * N
        ]
    assert expected == before, "STORE wrote outside C's rows"
    return C


class TestGemmAnyLayout:
    @pytest.mark.parametrize("dim", [4, 8])
    def test_dense_unaligned(self, dim):
        """Dense row-major A (M x K), B (K x N), C (M x N) at odd addresses, edge-tile sized."""
        cfg = config(dim)
        rng = np.random.default_rng(dim)
        M, K, N = dim - 1, dim - 1, dim - 2
        A = rng.integers(-128, 128, (M, K))
        B = rng.integers(-128, 128, (K, N))
        C = gemm_any_layout(cfg, A, B, a_stride=K, b_stride=N, c_stride=4 * N)
        np.testing.assert_array_equal(C, A @ B)

    def test_wide_strides(self):
        cfg = config(4)
        rng = np.random.default_rng(11)
        A = rng.integers(-128, 128, (4, 4))
        B = rng.integers(-128, 128, (4, 3))
        C = gemm_any_layout(cfg, A, B, a_stride=37, b_stride=19, c_stride=53)
        np.testing.assert_array_equal(C, A @ B)

    def test_one_bank(self):
        """One scratchpad and one accumulator bank: consecutive rows stay in bank 0."""
        cfg = config(4, sp_banks=1, acc_banks=1)
        rng = np.random.default_rng(12)
        A = rng.integers(-128, 128, (4, 4))
        B = rng.integers(-128, 128, (4, 4))
        C = gemm_any_layout(cfg, A, B, a_stride=4, b_stride=4, c_stride=16)
        np.testing.assert_array_equal(C, A @ B)


@pytest.mark.parametrize("field", ["sp_banks", "acc_banks"])
@pytest.mark.parametrize("banks", [3, 6])
def test_bank_count_must_be_a_power_of_two(field, banks):
    with pytest.raises(AssertionError, match="power of two"):
        config(**{field: banks})


def test_cmd_fields_default_to_whole_rows_one_beat_apart():
    """stride 0 / row_bytes 0 keep the old meaning: whole rows, consecutive beats."""
    cfg = config()
    beat = cfg.dma_buswidth // 8
    memory = pattern(1 << 12, seed=8)
    written = run_load(cfg, memory, dram_addr=0x200, local_addr=0, rows=3)
    assert written == {r: bytes(memory[0x200 + r * beat : 0x200 + r * beat + 4]) for r in range(3)}
