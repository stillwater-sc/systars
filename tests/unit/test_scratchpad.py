"""
Unit tests for L2 (memory/scratchpad.py): the banked scratchpad of #24.

These tests verify:
1. A bank: a 1R1W pipelined macro with a byte-masked write port
2. The layout: A and B bank groups, rows interleaved mod N over a group's banks,
   N from the occupancy and the macro depth, the hardware decode equal to the model
3. L2: two reads per clock from the two groups, writes beside them, the latency
4. The bank-conflict monitor: both reads in one group, a busy bank, a row past L2
5. Equivalence with the idealized two-read-port model (IdealScratchpad)
6. Verilog generation
"""

import random

import pytest
from amaranth import Elaboratable, Module, Signal
from amaranth.sim import Simulator, Tick

from systars.config import SystolicConfig
from systars.errors import L2Error
from systars.memory.scratchpad import (
    IdealScratchpad,
    L2Layout,
    Scratchpad,
    ScratchpadBank,
    make_l2,
)

from ._sim import run_sim


def small(**fields):
    """A 4x4 int8 tile (4-byte L2 rows) with a small L2."""
    base = {"grid_rows": 4, "grid_cols": 4, "tile_rows": 1, "tile_cols": 1, "sp_capacity_kb": 1}
    return SystolicConfig(**{**base, **fields})


def banked(n, **fields):
    """``small`` with exactly ``n`` pipelined banks per group (set by the macro depth)."""
    rows = small().l2_group_rows
    cfg = small(l2_macro_rows=-(-rows // n), **fields)
    assert cfg.l2_group_banks == n
    return cfg


class TestScratchpadBank:
    """One bank: a 1R1W macro."""

    @pytest.fixture
    def config(self):
        return SystolicConfig(
            sp_capacity_kb=16, spad_read_delay=2, grid_rows=4, grid_cols=4, tile_rows=1, tile_cols=1
        )

    def test_bank_has_one_read_and_one_write_port(self, config):
        bank = ScratchpadBank(config, bank_id=0)
        for port in ("read_addr", "read_en", "read_data", "read_valid"):
            assert hasattr(bank, port)
        for port in ("write_addr", "write_en", "write_data", "write_mask"):
            assert hasattr(bank, port)
        assert not hasattr(bank, "read_b_addr")

    def test_bank_write_read(self, config):
        bank = ScratchpadBank(config, bank_id=0)
        results = []

        def testbench():
            yield bank.write_addr.eq(0)
            yield bank.write_data.eq(0xDEADBEEF)
            yield bank.write_mask.eq(0xFFFF)
            yield bank.write_en.eq(1)
            yield Tick()
            yield bank.write_en.eq(0)
            yield Tick()
            yield bank.read_addr.eq(0)
            yield bank.read_en.eq(1)
            yield Tick()
            yield bank.read_en.eq(0)
            for _ in range(config.spad_read_delay - 1):
                yield Tick()
            results.append(((yield bank.read_valid), (yield bank.read_data)))

        sim = Simulator(bank)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()
        assert results == [(1, 0xDEADBEEF)]

    def test_bank_read_latency(self, config):
        bank = ScratchpadBank(config, bank_id=0)
        valid_cycles = []

        def testbench():
            yield bank.read_addr.eq(0)
            yield bank.read_en.eq(1)
            yield Tick()
            yield bank.read_en.eq(0)
            for cycle in range(10):
                if (yield bank.read_valid):
                    valid_cycles.append(cycle)
                yield Tick()

        sim = Simulator(bank)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()
        assert valid_cycles and valid_cycles[0] == config.spad_read_delay - 1


class TestLayout:
    """Where an L2 row lives."""

    def test_groups_are_the_two_halves(self):
        cfg = small()
        lay = L2Layout(cfg)
        assert cfg.sp_total_rows == 256 and cfg.l2_group_rows == 128
        assert lay.place(0)[0] == 0 and lay.place(127)[0] == 0
        assert lay.place(128)[0] == 1 and lay.place(255)[0] == 1

    @pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
    def test_rows_interleave_and_every_row_has_its_own_place(self, n):
        cfg = banked(n)
        lay = L2Layout(cfg)
        assert lay.banks == n
        places = [lay.place(r) for r in range(cfg.sp_total_rows)]
        assert len(set(places)) == len(places)
        for row, (group, bank, bank_row) in enumerate(places):
            r = row - group * cfg.l2_group_rows
            assert bank == r % n and bank_row == r // n
            assert bank_row < cfg.sp_bank_entries
        # Consecutive rows walk the banks: a bank is revisited only every N rows
        for g in range(2):
            banks = [places[g * cfg.l2_group_rows + i][1] for i in range(2 * n)]
            assert banks == [i % n for i in range(2 * n)]

    def test_bank_count_follows_the_macro_depth(self):
        cfg = small(l2_macro_rows=32)  # 128-row groups of at most 32-row macros
        assert cfg.l2_group_banks == 4 and cfg.sp_bank_entries == 32
        assert small(l2_macro_rows=0).l2_group_banks == 1
        assert small(l2_macro_rows=43).l2_group_banks == 3  # pipelined: any count

    @pytest.mark.parametrize(
        "dim,occupancy,macro_rows,banks",
        [
            (4, 2, 0, 2),
            (4, 3, 0, 4),  # 3 doesn't divide 4: the next divisor
            (4, 4, 0, 4),
            (6, 3, 0, 3),
            (12, 5, 0, 6),
            (4, 2, 8, 16),  # the depth needs 16: a multiple of dim
            (12, 2, 20, 3),  # the depth needs 3, a divisor of 12
        ],
    )
    def test_with_occupancy_the_banks_divide_or_are_a_multiple_of_dim(
        self, dim, occupancy, macro_rows, banks
    ):
        """A busy macro needs every command's ``dim`` reads to cycle the banks with
        no break across commands (CodeRabbit on #73: 3 banks under a 4x4 array would
        reuse bank 0 one clock after the wrap)."""
        cfg = small(grid_rows=dim, grid_cols=dim, l2_bank_occupancy=occupancy,
                    l2_macro_rows=macro_rows)  # fmt: skip
        n = cfg.l2_group_banks
        assert n >= occupancy and (dim % n == 0 or n % dim == 0)
        assert n == banks

    def test_occupancy_past_the_array_dimension_is_refused(self):
        with pytest.raises(AssertionError, match="at most the array dimension"):
            small(l2_bank_occupancy=5)

    def test_a_row_past_l2_has_no_place(self):
        lay = L2Layout(small())
        with pytest.raises(ValueError, match="past L2"):
            lay.place(256)

    @pytest.mark.parametrize("banks", [1, 2, 3, 4])
    def test_the_hardware_decode_is_the_model(self, banks):
        """The decode signals equal ``place`` for every row, and flag rows past L2."""
        cfg = banked(banks)
        lay = L2Layout(cfg)

        class Decode(Elaboratable):
            def __init__(self):
                self.addr = Signal(32)

            def elaborate(self, _platform):
                m = Module()
                self.out = lay.decode(m, self.addr, "t")
                tick = Signal()
                m.d.sync += tick.eq(~tick)
                return m

        dut = Decode()

        async def tb(ctx):
            group, bank, bank_row, out = dut.out
            for row in range(cfg.sp_total_rows + 3):
                ctx.set(dut.addr, row)
                await ctx.delay(1e-9)
                if row < cfg.sp_total_rows:
                    assert (ctx.get(group), ctx.get(bank), ctx.get(bank_row)) == lay.place(row)
                    assert not ctx.get(out)
                else:
                    assert ctx.get(out)

        sim = Simulator(dut)
        sim.add_testbench(tb)
        sim.run()


def fill(cfg):
    """Distinct row contents."""
    return {r: (0x5A00_0000 + 7919 * r) % (1 << cfg.sp_width) for r in range(cfg.sp_total_rows)}


async def write_all(ctx, dut, cfg, rows):
    ctx.set(dut.write_mask, (1 << (cfg.sp_width // 8)) - 1)
    for addr, value in rows.items():
        ctx.set(dut.write_req, 1)
        ctx.set(dut.write_addr, addr)
        ctx.set(dut.write_data, value)
        await ctx.tick()
    ctx.set(dut.write_req, 0)


async def stream(ctx, dut, cfg, pairs, *, writes=(), clocks=False):
    """Issue the read pairs ``(a_row, b_row)`` (None = no read) one per clock, and the
    writes ``(row, value)`` beside them; return what each port returned, in order
    (with ``clocks``, as ``(clock, data)`` pairs)."""
    got_a, got_b = [], []
    writes = list(writes)
    n = len(pairs) + cfg.spad_read_delay + 1
    for i in range(n):
        if i < len(pairs):
            a, b = pairs[i]
            ctx.set(dut.read_req, a is not None)
            ctx.set(dut.read_addr, a or 0)
            ctx.set(dut.read_b_req, b is not None)
            ctx.set(dut.read_b_addr, b or 0)
        else:
            ctx.set(dut.read_req, 0)
            ctx.set(dut.read_b_req, 0)
        if i < len(writes):
            row, value = writes[i]
            ctx.set(dut.write_req, 1)
            ctx.set(dut.write_addr, row)
            ctx.set(dut.write_data, value)
            ctx.set(dut.write_mask, (1 << (cfg.sp_width // 8)) - 1)
        else:
            ctx.set(dut.write_req, 0)
        await ctx.tick()
        if ctx.get(dut.read_valid):
            data = ctx.get(dut.read_data)
            got_a.append((i, data) if clocks else data)
        if ctx.get(dut.read_b_valid):
            data = ctx.get(dut.read_b_data)
            got_b.append((i, data) if clocks else data)
    return got_a, got_b


class TestL2:
    """The banked L2."""

    @pytest.mark.parametrize("delay", [1, 2, 4])
    @pytest.mark.parametrize("banks", [1, 2, 3])
    def test_two_reads_per_clock_from_the_two_groups(self, delay, banks):
        """An A row and a B row every clock, rows consecutive in each group, as the
        Streamer reads them: every row comes back, in order, with no error."""
        cfg = banked(banks, spad_read_delay=delay)
        dut = Scratchpad(cfg)
        rows = fill(cfg)
        g = cfg.l2_group_rows
        pairs = [(i, g + i) for i in range(40)]

        async def tb(ctx):
            await write_all(ctx, dut, cfg, rows)
            got_a, got_b = await stream(ctx, dut, cfg, pairs)
            assert got_a == [rows[a] for a, _ in pairs]
            assert got_b == [rows[b] for _, b in pairs]
            assert not ctx.get(dut.error)

        run_sim(dut, tb)

    def test_the_read_latency(self):
        cfg = small(spad_read_delay=3)
        dut = Scratchpad(cfg)

        async def tb(ctx):
            await write_all(ctx, dut, cfg, {5: 0x1234, 200: 0x5678})
            ctx.set(dut.read_req, 1)
            ctx.set(dut.read_addr, 5)
            ctx.set(dut.read_b_req, 1)
            ctx.set(dut.read_b_addr, 200)
            await ctx.tick()
            ctx.set(dut.read_req, 0)
            ctx.set(dut.read_b_req, 0)
            for _ in range(cfg.spad_read_delay - 1):
                assert not ctx.get(dut.read_valid) and not ctx.get(dut.read_b_valid)
                await ctx.tick()
            assert ctx.get(dut.read_valid) and ctx.get(dut.read_b_valid)
            assert ctx.get(dut.read_data) == 0x1234 and ctx.get(dut.read_b_data) == 0x5678

        run_sim(dut, tb)

    def test_writes_beside_the_reads(self):
        """A write every clock while both ports read: the reads see the old rows,
        the writes land, and nothing conflicts (each bank has its own write port)."""
        cfg = small(l2_bank_occupancy=2, l2_macro_rows=0)
        dut = Scratchpad(cfg)
        rows = fill(cfg)
        g = cfg.l2_group_rows
        pairs = [(i, g + i) for i in range(32)]
        # Writes to other rows of both groups, the same banks the reads walk
        writes = [((64 + i) if i % 2 else (g + 64 + i), 0xABC0_0000 + i) for i in range(32)]

        async def tb(ctx):
            await write_all(ctx, dut, cfg, rows)
            got_a, got_b = await stream(ctx, dut, cfg, pairs, writes=writes)
            assert got_a == [rows[a] for a, _ in pairs]
            assert got_b == [rows[b] for _, b in pairs]
            check = [(row, None) for row, _ in writes]
            got_w, _ = await stream(ctx, dut, cfg, check)
            assert got_w == [value for _, value in writes]
            assert not ctx.get(dut.error)

        run_sim(dut, tb)

    def test_the_masked_write(self):
        cfg = small()
        dut = Scratchpad(cfg)

        async def tb(ctx):
            await write_all(ctx, dut, cfg, {9: 0x11223344})
            ctx.set(dut.write_req, 1)
            ctx.set(dut.write_addr, 9)
            ctx.set(dut.write_data, 0xAABBCCDD)
            ctx.set(dut.write_mask, 0b0101)
            await ctx.tick()
            ctx.set(dut.write_req, 0)
            got, _ = await stream(ctx, dut, cfg, [(9, None)])
            assert got == [0x11BB33DD]

        run_sim(dut, tb)


class TestConflictMonitor:
    """``error`` flags every access the macros can't take, and stays set; ``error_kind``
    names the first one (#84)."""

    def run_pairs(self, cfg, pairs, *, expect_error, kind=None):
        dut = Scratchpad(cfg)
        rows = fill(cfg)

        async def tb(ctx):
            await write_all(ctx, dut, cfg, rows)
            assert not ctx.get(dut.error)
            got_a, _ = await stream(ctx, dut, cfg, pairs)
            assert ctx.get(dut.error) == expect_error
            # Port A is always served correctly
            assert got_a == [rows[a] for a, _ in pairs if a is not None]
            await ctx.tick().repeat(3)
            assert ctx.get(dut.error) == expect_error  # sticky
            if kind is not None:
                assert ctx.get(dut.error_kind) == kind

        run_sim(dut, tb)

    def test_both_reads_in_the_a_group(self):
        self.run_pairs(
            small(), [(0, None), (1, 4)], expect_error=True, kind=L2Error.READS_SAME_GROUP
        )

    def test_both_reads_in_the_b_group(self):
        g = small().l2_group_rows
        self.run_pairs(small(), [(g, g + 1)], expect_error=True, kind=L2Error.READS_SAME_GROUP)

    def test_reads_in_different_groups_in_either_order(self):
        g = small().l2_group_rows
        self.run_pairs(small(), [(0, g), (g + 1, 1), (2, g + 2)], expect_error=False)

    def test_a_busy_bank(self):
        """Occupancy 2: two reads of one bank two clocks apart are fine, one apart not."""
        cfg = small(l2_bank_occupancy=2, l2_macro_rows=0)
        self.run_pairs(cfg, [(0, None), (1, None), (2, None)], expect_error=False)
        self.run_pairs(cfg, [(0, None), (2, None)], expect_error=True, kind=L2Error.READ_BANK_BUSY)

    def test_a_row_past_l2(self):
        cfg = small()
        self.run_pairs(
            cfg, [(None, cfg.sp_total_rows)], expect_error=True, kind=L2Error.READ_B_PAST_L2
        )

    @pytest.mark.parametrize(
        "first,kind",
        [("write", L2Error.WRITE_PAST_L2), ("read_a", L2Error.READ_A_PAST_L2)],
    )
    def test_the_first_error_names_the_kind(self, first, kind):
        """A write or a port-A read past L2, then both reads in one group: the kind stays
        the first error's."""
        cfg = small()
        dut = Scratchpad(cfg)

        async def tb(ctx):
            port, addr = {"write": (dut.write_req, dut.write_addr),
                          "read_a": (dut.read_req, dut.read_addr)}[first]  # fmt: skip
            ctx.set(port, 1)
            ctx.set(addr, cfg.sp_total_rows)
            await ctx.tick()
            ctx.set(port, 0)
            assert ctx.get(dut.error) and ctx.get(dut.error_kind) == kind
            ctx.set(dut.read_req, 1)
            ctx.set(dut.read_b_req, 1)
            ctx.set(dut.read_addr, 0)
            ctx.set(dut.read_b_addr, 1)
            await ctx.tick().repeat(2)
            assert ctx.get(dut.error_kind) == kind

        run_sim(dut, tb)


@pytest.mark.parametrize("seed", range(3))
def test_bit_identical_to_the_idealized_model(seed):
    """A random legal access sequence (A-group and B-group reads, writes anywhere)
    returns the same data, on the same clocks, from the banked L2 and the ideal one."""
    cfg = small(spad_read_delay=2, l2_bank_occupancy=1, l2_macro_rows=32)
    rng = random.Random(seed)
    g = cfg.l2_group_rows
    pairs = [
        (rng.choice([None, rng.randrange(g)]), rng.choice([None, g + rng.randrange(g)]))
        for _ in range(120)
    ]
    writes = [(rng.randrange(cfg.sp_total_rows), rng.getrandbits(cfg.sp_width)) for _ in range(120)]
    rows = fill(cfg)
    out = {}
    for model in (Scratchpad, IdealScratchpad):
        dut = model(cfg)

        async def tb(ctx, dut=dut, model=model):
            await write_all(ctx, dut, cfg, rows)
            out[model] = await stream(ctx, dut, cfg, pairs, writes=writes, clocks=True)
            assert not ctx.get(dut.error)

        run_sim(dut, tb)
    assert out[Scratchpad] == out[IdealScratchpad]


def test_make_l2_picks_the_model():
    assert isinstance(make_l2(small()), Scratchpad)
    assert isinstance(make_l2(small(l2_ideal=True)), IdealScratchpad)


class TestScratchpadVerilogGeneration:
    @pytest.mark.parametrize("banks", [2, 3])
    def test_generate_scratchpad_verilog(self, banks):
        from amaranth._toolchain.yosys import find_yosys
        from amaranth.back import verilog

        try:
            find_yosys(lambda ver: ver >= (0, 40))
        except Exception:
            pytest.skip("Yosys not found")
        cfg = banked(banks)
        output = verilog.convert(Scratchpad(cfg), name="Scratchpad")
        assert "module Scratchpad" in output
        for g in range(2):
            for b in range(banks):
                assert f"g{g}_bank_{b}" in output
