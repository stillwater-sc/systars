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
- 4 KB boundaries (#18): no burst crosses one. A row spilling across one is two
  1-beat bursts, a streamed tile ends a burst at it, alone and in a GEMM through
  SystolicTop. The AXI model asserts it for every test.
- TRANSPOSE (#21): both legs with a corner turn, alone and through SystolicTop.
- LOAD into the accumulator (#17): rows written over the shared write port in the
  clocks it is free, overwriting or adding; CLEAR (zero rows, no L3 read);
  BROADCAST (one L3 read, every row); the accumulator corner turn.
"""

import numpy as np
import pytest
from amaranth import Elaboratable, Module, Signal

from systars.config import Dataflow, SystolicConfig
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
    WRITE = ("awvalid", "awready", "awaddr", "awlen", "wvalid", "wready", "wdata", "wstrb")
    WRITE += ("wlast",)
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


async def issue(
    ctx,
    ctrl,
    local_field,
    *,
    dram_addr,
    local_addr,
    rows,
    stride=0,
    row_bytes=0,
    opcode=DmaOpcode.MEMCPY,
):
    ctx.set(ctrl.cmd_opcode, opcode)
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


def always_free(_cycle):
    return True


def run_load(cfg, memory, *, then=(), log=None, acc=None, acc_free=always_free, **cmd):
    """Run one LOAD, then the commands in ``then`` (dicts of the same fields); return
    {scratchpad row: bytes written}. With ``acc`` (a dict), accumulator writes are
    recorded into it as {row: (bytes, accumulate)}; ``acc_free(cycle)`` says when the
    shared accumulator write port is free."""
    dut = LoadPath(cfg)
    sp_bytes = cfg.sp_width // 8
    acc_bytes = cfg.acc_width // 8
    written = {}

    async def recorder(ctx):
        c = dut.ctrl
        cycle = 0
        ctx.set(c.acc_write_ready, acc_free(0))
        async for _, _, en, addr, data, a_en, a_addr, a_data, a_add in ctx.tick().sample(
            c.sp_write_en, c.sp_write_addr, c.sp_write_data,
            c.acc_write_en, c.acc_write_addr, c.acc_write_data, c.acc_write_accumulate,
        ):  # fmt: skip
            if en:
                assert addr not in written, f"scratchpad row {addr} written twice"
                written[addr] = data.to_bytes(sp_bytes, "little")
            if a_en:
                assert acc is not None, "an accumulator write from a scratchpad LOAD"
                assert a_addr >> 31 & 1, f"accumulator write to {a_addr:#x} without the flag"
                row = a_addr & ((1 << 29) - 1)
                assert row not in acc, f"accumulator row {row} written twice"
                acc[row] = (a_data.to_bytes(acc_bytes, "little"), a_add)
            cycle += 1
            ctx.set(c.acc_write_ready, acc_free(cycle))

    async def testbench(ctx):
        for fields in (cmd, *then):
            await issue(ctx, dut.ctrl, "cmd_sp_addr", **fields)

    beat = cfg.dma_buswidth // 8
    run_sim(dut, testbench, processes=[axi_memory(dut, memory, beat, log=log), recorder])
    return written


def run_store(cfg, memory, acc_rows, *, activation=0, then=(), log=None, **cmd):
    """Run one STORE of ``acc_rows`` (lists of acc_bits values) into ``memory``, then the
    commands in ``then`` (dicts of the same fields)."""
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
        for fields in (cmd, *then):
            await issue(ctx, dut.ctrl, "cmd_acc_addr", **fields)

    beat = cfg.dma_buswidth // 8
    run_sim(dut, testbench, processes=[axi_memory(dut, memory, beat, log=log), accumulator])


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


def gemm_any_layout(
    cfg, A, B, *, a_stride, b_stride, c_stride, a_addr=A_ADDR, b_addr=B_ADDR, c_addr=C_ADDR,
    log=None,
):  # fmt: skip
    """WS GEMM (one tile) through SystolicTop, operands and C at unaligned, strided addresses.

    A is row-major in L3; the LOAD's corner turn lays it column-major in L2, as the
    Streamer streams it. C is written with row_bytes = N results, so the bytes past
    each row keep their canary."""
    dim = cfg.grid_rows * cfg.tile_rows
    beat = cfg.dma_buswidth // 8
    M, K = A.shape
    N = B.shape[1]
    memory = pattern(1 << 16, seed=7)
    for r, row in enumerate(A):
        memory[a_addr + r * a_stride : a_addr + r * a_stride + K] = pack(row, 8).to_bytes(
            K, "little"
        )
    for r, row in enumerate(B):
        memory[b_addr + r * b_stride : b_addr + r * b_stride + N] = pack(row, 8).to_bytes(
            N, "little"
        )
    before = bytes(memory)
    dut = SystolicTop(cfg)
    sp_b = dim  # B rows follow the A rows

    async def testbench(ctx):
        host = Host(ctx, dut)
        ctx.set(dut.cmd_stride, a_stride)
        ctx.set(dut.cmd_row_bytes, K)
        # M rows of K elements -> K L2 rows of M: A's columns
        await host.run(SystolicTop.CTRL_LOAD, TRANSPOSE, dram_addr=a_addr, local_addr=0, len=M)
        ctx.set(dut.cmd_stride, b_stride)
        ctx.set(dut.cmd_row_bytes, N)
        await host.load(b_addr, sp_b, K)
        await host.execute(ExecOpcode.CONFIG, rs1=WS)
        await host.execute(ExecOpcode.PRELOAD, rs1=sp_b, k_dim=K)
        await host.execute(ExecOpcode.COMPUTE, rs1=0, rd=0, k_dim=M)
        ctx.set(dut.cmd_stride, c_stride)
        ctx.set(dut.cmd_row_bytes, 4 * N)
        await host.store(0, c_addr, M)

    run_sim(dut, testbench, processes=[axi_memory(dut, memory, beat, log=log)], max_cycles=20_000)
    C = np.array(
        [
            unpack(
                int.from_bytes(
                    memory[c_addr + r * c_stride : c_addr + r * c_stride + 4 * N], "little"
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
        expected[c_addr + r * c_stride : c_addr + r * c_stride + 4 * N] = before[
            c_addr + r * c_stride : c_addr + r * c_stride + 4 * N
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


class TestFourKBoundaries:
    """No burst crosses a 4 KB boundary (#18). The AXI model asserts it for every burst;
    these tests also check the bursts themselves, and the values."""

    PAGE = 0x2000  # a 4 KB boundary

    def test_load_row_spilling_across_a_boundary(self):
        """8-byte rows 40 bytes apart: the row at PAGE - 4 is in two pages, and is read
        as two 1-beat bursts; the rows around it are ordinary bursts."""
        cfg = config(dim=8)  # 8-byte scratchpad rows, 32-byte beats
        beat = cfg.dma_buswidth // 8
        assert beat == 32
        memory = pattern(1 << 14, seed=20)
        base, log = self.PAGE - 4 - 40, []
        written = run_load(cfg, memory, dram_addr=base, local_addr=0, rows=3, stride=40, log=log)
        assert written == {r: bytes(memory[base + 40 * r : base + 40 * r + 8]) for r in range(3)}
        # rows at PAGE - 44 (1 beat), PAGE - 4 (split), PAGE + 36 (1 beat)
        p = self.PAGE
        assert log == [("r", p - 64, 1), ("r", p - 32, 1), ("r", p, 1), ("r", p + 32, 1)]

    def test_load_stream_crossing_a_boundary(self):
        """40 consecutive beats from 8 beats before PAGE: one burst to PAGE, one after."""
        cfg = config()
        beat = cfg.dma_buswidth // 8
        memory = pattern(1 << 14, seed=21)
        base, log = self.PAGE - 8 * beat, []
        written = run_load(cfg, memory, dram_addr=base, local_addr=0, rows=40, log=log)
        assert written == {
            r: bytes(memory[base + r * beat : base + r * beat + 4]) for r in range(40)
        }
        assert log == [("r", base, 8), ("r", self.PAGE, 32)]

    def test_load_stream_of_300_rows_from_mid_page(self):
        """300 beats from mid-page: bursts end at each boundary and at 256 beats."""
        cfg = config()
        beat = cfg.dma_buswidth // 8
        memory = pattern(1 << 15, seed=22)
        base, log = self.PAGE + 0x800, []
        written = run_load(cfg, memory, dram_addr=base, local_addr=0, rows=300, log=log)
        assert len(written) == 300
        assert all(
            written[r] == bytes(memory[base + r * beat : base + r * beat + 4]) for r in range(300)
        )
        assert log == [("r", base, 128), ("r", base + 0x800, 172)]

    def test_store_row_spilling_across_a_boundary(self):
        """A 16-byte row at PAGE - 4: two 1-beat bursts, its bytes only."""
        cfg = config()  # 16-byte accumulator rows, 16-byte beats
        memory = pattern(1 << 14, seed=23)
        before = bytes(memory)
        rows = TestStore.ROWS
        base, log = self.PAGE - 4 - 24, []
        run_store(cfg, memory, rows, dram_addr=base, local_addr=0, rows=3, stride=24, log=log)
        expected = bytearray(before)
        for r, row in enumerate(rows):
            expected[base + 24 * r : base + 24 * r + 16] = pack(row, 32).to_bytes(16, "little")
        assert memory == expected
        # rows at PAGE - 28 (2 beats), PAGE - 4 (split), PAGE + 20 (2 beats)
        p = self.PAGE
        assert log == [("w", p - 32, 2), ("w", p - 16, 1), ("w", p, 1), ("w", p + 16, 2)]

    def test_store_stream_crossing_a_boundary(self):
        cfg = config()
        beat = cfg.dma_buswidth // 8
        rows = [[r, -r, 2 * r, 3] for r in range(40)]
        memory = bytearray(1 << 14)
        base, log = self.PAGE - 8 * beat, []
        run_store(cfg, memory, rows, dram_addr=base, local_addr=0, rows=40, log=log)
        stored = [
            unpack(int.from_bytes(memory[base + r * beat : base + (r + 1) * beat], "little"), 32, 4)
            for r in range(40)
        ]
        assert stored == rows
        assert log == [("w", base, 8), ("w", self.PAGE, 32)]

    def test_gemm_with_every_operand_across_a_boundary(self):
        """A, B and C each have a row straddling a 4 KB boundary (A's second row at
        0x1000 - 2, B's at 0x2000 - 1, C's at 0x3000 - 6), at odd strides: each such row
        is two 1-beat bursts, and C is right."""
        cfg = config(4)
        beat = cfg.dma_buswidth // 8
        rng = np.random.default_rng(24)
        A = rng.integers(-128, 128, (4, 4))
        B = rng.integers(-128, 128, (4, 3))
        log = []
        C = gemm_any_layout(
            cfg, A, B, a_stride=37, b_stride=19, c_stride=53,
            a_addr=0x1000 - 2 - 37, b_addr=0x2000 - 1 - 19, c_addr=0x3000 - 6 - 53, log=log,
        )  # fmt: skip
        np.testing.assert_array_equal(C, A @ B)
        for kind, page in (("r", 0x1000), ("r", 0x2000), ("w", 0x3000)):
            n = log.index((kind, page - beat, 1))
            assert log[n + 1] == (kind, page, 1)


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


# ---------------------------------------------------------------------------
# TRANSPOSE: the corner turns (#21)
# ---------------------------------------------------------------------------

TRANSPOSE = DmaOpcode.TRANSPOSE


def turned(rows, n_out, width):
    """The corner turn of ``rows`` (lists of elements): ``n_out`` rows, element ``n`` of
    output row ``j`` being element ``j`` of input row ``n``, zero past the inputs."""
    return [[row[j] for row in rows] + [0] * (width - len(rows)) for j in range(n_out)]


class TestLoadTranspose:
    @pytest.mark.parametrize("offset", [0, 3, 13])
    def test_tile_lands_column_major(self, offset):
        """A 4x4 int8 tile at any offset: L3 row n becomes element n of every L2 row."""
        cfg = config()  # 4 lanes of int8
        memory = pattern(1 << 12, seed=20)
        base = 0x100 + offset
        written = run_load(
            cfg, memory, dram_addr=base, local_addr=5, rows=4, stride=29, opcode=TRANSPOSE
        )
        tile = [list(memory[base + 29 * r : base + 29 * r + 4]) for r in range(4)]
        assert written == {5 + j: bytes(row) for j, row in enumerate(turned(tile, 4, 4))}

    def test_edge_tile_is_zero_filled(self):
        """3 rows of 2 elements: 2 L2 rows, each with 3 elements and a zero."""
        cfg = config()
        memory = pattern(1 << 12, seed=21)
        written = run_load(
            cfg, memory, dram_addr=0x205, local_addr=0, rows=3, stride=7, row_bytes=2,
            opcode=TRANSPOSE,
        )  # fmt: skip
        tile = [list(memory[0x205 + 7 * r : 0x205 + 7 * r + 2]) for r in range(3)]
        assert written == {j: bytes(row) for j, row in enumerate(turned(tile, 2, 4))}

    def test_consecutive_beats_stream(self):
        """Aligned, beat-spaced rows take the streaming path into the buffer."""
        cfg = config(dim=8)
        beat = cfg.dma_buswidth // 8
        memory = pattern(1 << 12, seed=22)
        written = run_load(cfg, memory, dram_addr=0x400, local_addr=0, rows=8, opcode=TRANSPOSE)
        tile = [list(memory[0x400 + beat * r : 0x400 + beat * r + 8]) for r in range(8)]
        assert written == {j: bytes(row) for j, row in enumerate(turned(tile, 8, 8))}

    def test_short_tile_after_a_full_one_is_zero_filled(self):
        """A 2-row tile after a 4-row one has zeros past its second element: the
        first TRANSPOSE (2 of 4 columns) leaves nothing behind in the buffer."""
        cfg = config()
        memory = pattern(1 << 12, seed=25)
        written = run_load(
            cfg, memory, dram_addr=0x100, local_addr=0, rows=4, stride=4, row_bytes=2,
            opcode=TRANSPOSE,
            then=[{"dram_addr": 0x200, "local_addr": 8, "rows": 2, "stride": 4, "opcode": TRANSPOSE}],
        )  # fmt: skip
        tile = [list(memory[0x200 + 4 * r : 0x200 + 4 * r + 4]) for r in range(2)]
        assert {a: written[a] for a in range(8, 12)} == {
            8 + j: bytes(row) for j, row in enumerate(turned(tile, 4, 4))
        }

    def test_moves_at_most_one_tile(self):
        """len is clamped to the lanes: rows past the tile are not read."""
        cfg = config()
        memory = pattern(1 << 12, seed=23)
        written = run_load(
            cfg, memory, dram_addr=0x300, local_addr=0, rows=7, stride=4, opcode=TRANSPOSE
        )
        tile = [list(memory[0x300 + 4 * r : 0x300 + 4 * r + 4]) for r in range(4)]
        assert written == {j: bytes(row) for j, row in enumerate(turned(tile, 4, 4))}

    def test_zero_rows_complete_without_a_transfer(self):
        assert (
            run_load(config(), pattern(256), dram_addr=0x10, local_addr=0, rows=0, opcode=TRANSPOSE)
            == {}
        )


class TestStoreTranspose:
    ROWS = TestStore.ROWS  # 3 accumulator rows of 4 int32

    def stored(self, memory, base, stride, rows, n):
        return [
            unpack(int.from_bytes(memory[base + stride * r : base + stride * r + 4 * n], "little"),
                   32, n)
            for r in range(rows)
        ]  # fmt: skip

    @pytest.mark.parametrize("offset", [0, 4, 9])
    def test_columns_become_rows(self, offset):
        """3 accumulator rows (row_bytes = 3 results): L3 row i is element i of each."""
        cfg = config()
        memory = pattern(1 << 12, seed=24)
        before = bytes(memory)
        base = 0x300 + offset
        run_store(
            cfg, memory, self.ROWS, dram_addr=base, local_addr=0, rows=4, stride=20,
            row_bytes=12, opcode=TRANSPOSE,
        )  # fmt: skip
        assert self.stored(memory, base, 20, 4, 3) == turned(self.ROWS, 4, 3)
        expected = bytearray(before)
        for r in range(4):
            expected[base + 20 * r : base + 20 * r + 12] = memory[
                base + 20 * r : base + 20 * r + 12
            ]
        assert memory == expected, "STORE wrote outside its rows"

    def test_consecutive_beats_stream(self):
        cfg = config()
        beat = cfg.dma_buswidth // 8
        rows = [[r * 10 + c for c in range(4)] for r in range(4)]
        memory = bytearray(1 << 12)
        run_store(cfg, memory, rows, dram_addr=0x800, local_addr=0, rows=4, opcode=TRANSPOSE)
        assert self.stored(memory, 0x800, beat, 4, 4) == turned(rows, 4, 4)

    def test_activation_applies_per_element(self):
        cfg = config()
        memory = bytearray(1 << 10)
        run_store(
            cfg, memory, self.ROWS, activation=1, dram_addr=0x40, local_addr=0, rows=2,
            stride=16, row_bytes=12, opcode=TRANSPOSE,
        )  # fmt: skip
        relu = [[max(0, x) for x in row] for row in turned(self.ROWS, 2, 3)]
        assert self.stored(memory, 0x40, 16, 2, 3) == relu

    def test_narrow_tile_after_a_wide_one_writes_only_its_bytes(self):
        """A 2-column tile after a 4-column one: the first leaves accumulator rows in
        the buffer, which only reach elements past row_bytes, so nothing is written there."""
        cfg = config()
        rows = [[r * 10 + c + 1 for c in range(4)] for r in range(4)]
        memory = bytearray(1 << 12)
        run_store(
            cfg, memory, rows, dram_addr=0x100, local_addr=0, rows=2, stride=16, opcode=TRANSPOSE,
            then=[{"dram_addr": 0x400, "local_addr": 0, "rows": 4, "stride": 16,
                   "row_bytes": 8, "opcode": TRANSPOSE}],
        )  # fmt: skip
        assert self.stored(memory, 0x400, 16, 4, 2) == turned(rows[:2], 4, 2)
        assert not any(any(memory[0x400 + 16 * r + 8 : 0x400 + 16 * r + 16]) for r in range(4))

    def test_moves_at_most_one_tile(self):
        cfg = config()
        rows = [[r * 10 + c for c in range(4)] for r in range(4)]
        memory = bytearray(1 << 12)
        run_store(
            cfg, memory, rows, dram_addr=0x100, local_addr=0, rows=9, stride=16, opcode=TRANSPOSE
        )
        assert self.stored(memory, 0x100, 16, 4, 4) == turned(rows, 4, 4)
        assert not any(memory[0x100 + 64 : 0x100 + 9 * 16]), "rows past the tile were written"


@pytest.mark.parametrize("dim", [4, 8])
def test_round_trip_through_both_corner_turns(dim):
    """Through SystolicTop: LOAD TRANSPOSE of row-major A puts A's columns in L2 (the
    layout WS streams), a WS GEMM computes A @ B, and STORE TRANSPOSE writes Cᵀ, all
    at unaligned addresses."""
    cfg = config(dim)
    beat = cfg.dma_buswidth // 8
    rng = np.random.default_rng(30 + dim)
    M, K, N = dim, dim - 1, dim - 1
    A = rng.integers(-128, 128, (M, K))
    B = rng.integers(-128, 128, (K, N))
    a_addr, b_addr, ct_addr = 0x1003, 0x2009, 0x3005
    memory = pattern(1 << 16, seed=31)
    for r, row in enumerate(A):  # M rows of K
        memory[a_addr + r * K : a_addr + (r + 1) * K] = pack(row, 8).to_bytes(K, "little")
    for r, row in enumerate(B):
        memory[b_addr + r * N : b_addr + (r + 1) * N] = pack(row, 8).to_bytes(N, "little")
    dut = SystolicTop(cfg)
    sp_b = dim

    async def testbench(ctx):
        host = Host(ctx, dut)
        ctx.set(dut.cmd_stride, K)
        ctx.set(dut.cmd_row_bytes, K)
        # M rows of A, K elements each -> K L2 rows of M elements: A's columns
        await host.run(SystolicTop.CTRL_LOAD, TRANSPOSE, dram_addr=a_addr, local_addr=0, len=M)
        ctx.set(dut.cmd_stride, N)
        ctx.set(dut.cmd_row_bytes, N)
        await host.load(b_addr, sp_b, K)
        await host.execute(ExecOpcode.CONFIG, rs1=WS)
        await host.execute(ExecOpcode.PRELOAD, rs1=sp_b, k_dim=K)
        await host.execute(ExecOpcode.COMPUTE, rs1=0, rd=0, k_dim=M)
        # M accumulator rows of N results -> N L3 rows of M: Cᵀ
        ctx.set(dut.cmd_stride, 4 * M)
        ctx.set(dut.cmd_row_bytes, 4 * M)
        await host.run(SystolicTop.CTRL_STORE, TRANSPOSE, dram_addr=ct_addr, local_addr=0, len=N)

    run_sim(dut, testbench, processes=[axi_memory(dut, memory, beat)], max_cycles=20_000)
    CT = np.array(
        [
            unpack(int.from_bytes(memory[ct_addr + r * 4 * M : ct_addr + (r + 1) * 4 * M],
                                  "little"), 32, M)
            for r in range(N)
        ]
    )  # fmt: skip
    np.testing.assert_array_equal(CT, (A @ B).T)


# ---------------------------------------------------------------------------
# LOAD into the accumulator, CLEAR and BROADCAST (#17)
# ---------------------------------------------------------------------------

ACC, ADD = 1 << 31, 1 << 30


def busy_two_of_three(cycle):
    """The Streamer holds the accumulator write port two clocks in three."""
    return cycle % 3 == 0


class TestLoadAccumulator:
    @pytest.mark.parametrize("offset", [0, 5, 13])
    @pytest.mark.parametrize("free", [always_free, busy_two_of_three], ids=["free", "busy"])
    def test_rows_land_in_the_accumulator(self, offset, free):
        """Accumulator rows (4 int32: 16 bytes) at any byte address, written only in the
        clocks the Streamer leaves the port free; the scratchpad is not touched."""
        cfg = config()
        memory = pattern(1 << 12, seed=30)
        base, acc = 0x300 + offset, {}
        written = run_load(
            cfg, memory, dram_addr=base, local_addr=ACC | 6, rows=3, stride=40, row_bytes=12,
            acc=acc, acc_free=free,
        )  # fmt: skip
        assert written == {}
        assert acc == {
            6 + r: (bytes(memory[base + 40 * r : base + 40 * r + 12]) + bytes(4), 0)
            for r in range(3)
        }

    def test_streamed_rows_wait_for_the_port(self):
        """Consecutive beats stream; a busy port holds the stream (no beat is lost)."""
        cfg = config()
        beat = cfg.dma_buswidth // 8
        memory = pattern(1 << 13, seed=31)
        acc: dict = {}
        log: list = []
        run_load(
            cfg, memory, dram_addr=0x400, local_addr=ACC, rows=20, acc=acc,
            acc_free=busy_two_of_three, log=log,
        )  # fmt: skip
        assert log == [("r", 0x400, 20)]
        assert acc == {r: (bytes(memory[0x400 + r * beat : 0x400 + (r + 1) * beat]), 0)
                       for r in range(20)}  # fmt: skip

    def test_add_flag_accumulates(self):
        cfg = config()
        memory = pattern(1 << 12, seed=32)
        acc: dict = {}
        run_load(cfg, memory, dram_addr=0x101, local_addr=ACC | ADD | 2, rows=2, stride=16,
                 acc=acc)  # fmt: skip
        assert {row: add for row, (_, add) in acc.items()} == {2: 1, 3: 1}

    @pytest.mark.parametrize("dest", [0, ACC])
    def test_clear_writes_zero_rows_without_a_read(self, dest):
        cfg = config()
        acc: dict = {}
        log: list = []
        written = run_load(
            cfg, pattern(1 << 12), dram_addr=0x200, local_addr=dest | 4, rows=3,
            opcode=DmaOpcode.CLEAR, acc=acc, acc_free=busy_two_of_three, log=log,
        )  # fmt: skip
        assert log == []
        if dest:
            assert written == {} and acc == {4 + r: (bytes(16), 0) for r in range(3)}
        else:
            assert written == {4 + r: bytes(4) for r in range(3)} and acc == {}

    @pytest.mark.parametrize("dest", [0, ACC])
    def test_broadcast_reads_once_and_writes_every_row(self, dest):
        """One read, written to 5 rows. The 8-byte accumulator row spills into the next
        beat (a 2-beat read); the scratchpad row is clamped to its 4 bytes (1 beat)."""
        cfg = config()
        memory = pattern(1 << 12, seed=33)
        acc: dict = {}
        log: list = []
        written = run_load(
            cfg, memory, dram_addr=0x10B, local_addr=dest | 1, rows=5, stride=64,
            row_bytes=8, opcode=DmaOpcode.BROADCAST, acc=acc, log=log,
        )  # fmt: skip
        assert log == [("r", 0x100, 2 if dest else 1)]
        row = bytes(memory[0x10B : 0x10B + 8])
        if dest:
            assert written == {} and acc == {1 + r: (row + bytes(8), 0) for r in range(5)}
        else:
            assert written == {1 + r: row[:4] for r in range(5)} and acc == {}

    def test_corner_turn_into_the_accumulator(self):
        """IS holds Cᵀ: L3 row n (int32 elements) becomes element n of each accumulator row."""
        cfg = config(dataflow=Dataflow.A_STATIONARY | Dataflow.B_STATIONARY)
        memory = pattern(1 << 12, seed=34)
        acc: dict = {}
        run_load(
            cfg, memory, dram_addr=0x203, local_addr=ACC | 8, rows=3, stride=24, row_bytes=8,
            opcode=TRANSPOSE, acc=acc, acc_free=busy_two_of_three,
        )  # fmt: skip
        tile = [unpack(int.from_bytes(memory[0x203 + 24 * r : 0x203 + 24 * r + 8], "little"),
                       32, 2) for r in range(3)]  # fmt: skip
        assert {row: unpack(int.from_bytes(data, "little"), 32, 4) for row, (data, _) in
                acc.items()} == {8 + j: row for j, row in enumerate(turned(tile, 2, 4))}  # fmt: skip

    def test_broadcast_corner_turn(self):
        """Every accumulator row holds ``len`` copies of one element of the row."""
        cfg = config(dataflow=Dataflow.A_STATIONARY | Dataflow.B_STATIONARY)
        memory = pattern(1 << 12, seed=35)
        acc: dict = {}
        run_load(
            cfg, memory, dram_addr=0x200, local_addr=ACC, rows=3, row_bytes=12,
            opcode=DmaOpcode.BROADCAST_TRANSPOSE, acc=acc,
        )  # fmt: skip
        row = unpack(int.from_bytes(memory[0x200:0x20C], "little"), 32, 3)
        assert {r: unpack(int.from_bytes(d, "little"), 32, 4) for r, (d, _) in acc.items()} == {
            j: [row[j]] * 3 + [0] for j in range(3)
        }

    def test_corner_turn_needs_input_stationary(self):
        """Without IS the accumulator corner turn is not built: the LOAD moves nothing."""
        cfg = config(dataflow=Dataflow.OUTPUT_STATIONARY)
        acc: dict = {}
        log: list = []
        written = run_load(
            cfg, pattern(1 << 12), dram_addr=0x200, local_addr=ACC, rows=3, opcode=TRANSPOSE,
            acc=acc, log=log,
        )  # fmt: skip
        assert (written, acc, log) == ({}, {}, [])
