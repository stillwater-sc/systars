"""
Unit tests for the Combiner (memory/combiner.py, #74): the merge of result rows into
L2's result group.

These tests verify:
1. A bank of the result group: a 1R1W macro with the read latency
2. The merge: overwrite, accumulate (each lane wraps on its own), commit order
3. Forwarding: accumulates onto one row at every spacing, back to back included
4. STORE's read: it waits while the merge reads, and for a row with a request pending
5. A random request and read sequence against a Python model
6. Verilog generation
"""

import random

import pytest

from systars.config import SystolicConfig
from systars.memory.combiner import Combiner, CombinerBank
from systars.memory.local_addr import LocalAddr

from ._sim import run_sim, skip_without_yosys


def config(latency=2, **fields):
    base = {"grid_rows": 4, "grid_cols": 4, "tile_rows": 1, "tile_cols": 1,
            "combiner_capacity_kb": 1, "combiner_banks": 2, "combiner_latency": latency}  # fmt: skip
    return SystolicConfig(**{**base, **fields})


def pack(values, bits=32):
    return sum((v & ((1 << bits) - 1)) << (i * bits) for i, v in enumerate(values))


def unpack(word, bits=32, n=4):
    vals = [(word >> (i * bits)) & ((1 << bits) - 1) for i in range(n)]
    return [v - (1 << bits) if v >> (bits - 1) else v for v in vals]


def addr(cfg, row):
    """The local address of result row ``row`` (bank in the low bits)."""
    la = LocalAddr(cfg)
    return la.make_combiner_addr(row % cfg.combiner_banks, row // cfg.combiner_banks)


def model_add(a, b, n=4):
    return [((x + y + (1 << 31)) % (1 << 32)) - (1 << 31) for x, y in zip(a, b, strict=True)][:n]


class Driver:
    """Drives the Combiner one clock at a time and models it."""

    def __init__(self, ctx, dut, cfg):
        self.ctx, self.dut, self.cfg = ctx, dut, cfg
        self.lanes = cfg.combiner_width // cfg.acc_bits
        self.model = {}
        self.forwards = 0  # clocks with forward_hit (an accumulate fed by forwarding)

    async def clock(self, req=None, read=None):
        """One clock: ``req`` = (row, values, accumulate) or None; ``read`` = a row for
        STORE to read (held by the caller until taken). Returns whether it was taken."""
        ctx, dut = self.ctx, self.dut
        if req is not None:
            row, values, acc = req
            ctx.set(dut.write_req, 1)
            ctx.set(dut.write_addr, addr(self.cfg, row))
            ctx.set(dut.write_data, pack(values))
            ctx.set(dut.accumulate, acc)
            old = self.model.get(row, [0] * self.lanes)
            self.model[row] = model_add(old, values, self.lanes) if acc else list(values)
        else:
            ctx.set(dut.write_req, 0)
            ctx.set(dut.accumulate, 0)
        ctx.set(dut.read_req, read is not None)
        ctx.set(dut.read_addr, addr(self.cfg, read) if read is not None else 0)
        taken = read is not None and ctx.get(dut.read_ready)
        self.forwards += ctx.get(dut.forward_hit)
        await ctx.tick()
        return taken

    async def drain(self):
        for _ in range(self.cfg.combiner_latency + 2):
            await self.clock()

    async def read(self, row):
        """STORE's read of ``row``: wait for it to be taken, then for its data."""
        while not await self.clock(read=row):
            pass
        for _ in range(self.cfg.combiner_latency + 2):
            if self.ctx.get(self.dut.read_valid):
                return unpack(self.ctx.get(self.dut.read_data), n=self.lanes)
            await self.clock()
        raise AssertionError("no read data")


class TestBank:
    def test_one_read_and_one_write_port(self):
        bank = CombinerBank(config(), 0)
        for port in ("read_addr", "read_en", "read_data", "read_valid",
                     "write_addr", "write_en", "write_data"):  # fmt: skip
            assert hasattr(bank, port)
        assert not hasattr(bank, "accumulate") and not hasattr(bank, "read_activation")

    @pytest.mark.parametrize("latency", [1, 2, 3])
    def test_the_read_latency(self, latency):
        cfg = config(latency)
        bank = CombinerBank(cfg, 0)

        async def tb(ctx):
            ctx.set(bank.write_en, 1)
            ctx.set(bank.write_addr, 3)
            ctx.set(bank.write_data, 0x1234)
            await ctx.tick()
            ctx.set(bank.write_en, 0)
            ctx.set(bank.read_en, 1)
            ctx.set(bank.read_addr, 3)
            await ctx.tick()
            ctx.set(bank.read_en, 0)
            for _ in range(latency - 1):
                assert not ctx.get(bank.read_valid)
                await ctx.tick()
            assert ctx.get(bank.read_valid) and ctx.get(bank.read_data) == 0x1234

        run_sim(bank, tb)


class TestMerge:
    @pytest.mark.parametrize("latency", [1, 2, 4])
    def test_overwrite_then_accumulate(self, latency):
        cfg = config(latency)
        dut = Combiner(cfg)

        async def tb(ctx):
            d = Driver(ctx, dut, cfg)
            await d.clock((5, [1, -2, 3, -4], 0))
            await d.drain()
            await d.clock((5, [10, 20, 30, 40], 1))
            await d.drain()
            assert await d.read(5) == [11, 18, 33, 36]

        run_sim(dut, tb)

    def test_each_lane_wraps_on_its_own(self):
        cfg = config()
        dut = Combiner(cfg)
        big = (1 << 31) - 1

        async def tb(ctx):
            d = Driver(ctx, dut, cfg)
            await d.clock((2, [big, -1, 0, -(1 << 31)], 0))
            await d.clock((2, [1, 1, -1, -1], 1))
            await d.drain()
            assert await d.read(2) == [-(1 << 31), 0, -1, big]

        run_sim(dut, tb)

    @pytest.mark.parametrize("latency", [1, 2, 3, 4])
    @pytest.mark.parametrize("gap", [0, 1, 2, 3, 4, 5])
    def test_accumulates_onto_one_row_at_every_spacing(self, latency, gap):
        """Five accumulates onto one row, ``gap`` idle clocks apart (0: back to back),
        with an unrelated row written between: each sees the last one's sum."""
        cfg = config(latency)
        dut = Combiner(cfg)

        async def tb(ctx):
            d = Driver(ctx, dut, cfg)
            await d.clock((7, [100, 200, 300, 400], 0))
            for n in range(5):
                await d.clock((7, [n, -n, 2 * n, 1], 1))
                for g in range(gap):
                    await d.clock((1, [g, g, g, g], g % 2) if g == 1 else None)
            await d.drain()
            assert await d.read(7) == d.model[7]
            assert d.model[7] == [110, 190, 320, 405]
            # forward_hit (#85): an accumulate is fed by forwarding exactly when the row's
            # last commit is within `latency` clocks. Spaced wider, only the first one,
            # right behind the overwrite, is
            if gap < latency:
                assert d.forwards >= 5
            else:
                assert d.forwards == 1

        run_sim(dut, tb)


class TestStoreRead:
    def test_store_waits_while_the_merge_reads(self):
        """An accumulate stream holds STORE off the read port; overwrites don't."""
        cfg = config()
        dut = Combiner(cfg)
        seen = {}

        async def tb(ctx):
            d = Driver(ctx, dut, cfg)
            await d.clock((0, [9, 9, 9, 9], 0))
            await d.drain()
            taken = [await d.clock((1, [1, 1, 1, 1], 1), read=0) for _ in range(6)]
            seen["during_accumulates"] = any(taken)
            taken = [await d.clock((2 + i, [1, 1, 1, 1], 0), read=0) for i in range(3)]
            seen["during_overwrites"] = taken[0]

        run_sim(dut, tb)
        assert seen == {"during_accumulates": False, "during_overwrites": True}

    def test_store_waits_for_a_pending_row(self):
        """STORE asks for a row whose write is still in the pipeline: it is taken only
        once the row is committed, and reads the new value."""
        cfg = config(latency=3)
        dut = Combiner(cfg)

        async def tb(ctx):
            d = Driver(ctx, dut, cfg)
            assert not await d.clock((4, [5, 6, 7, 8], 0), read=4)
            assert await d.read(4) == [5, 6, 7, 8]

        run_sim(dut, tb)


@pytest.mark.parametrize("latency", [1, 2, 3])
@pytest.mark.parametrize("seed", range(3))
def test_random_requests_and_reads_match_the_model(latency, seed):
    """Requests every clock (overwrites, accumulates, idle) over a few rows, with STORE
    reads among them: every read returns the model's row as of when it was taken."""
    cfg = config(latency)
    dut = Combiner(cfg)
    rng = random.Random(seed)

    async def tb(ctx):
        d = Driver(ctx, dut, cfg)
        for row in range(6):
            await d.clock((row, [rng.randrange(-50, 50) for _ in range(4)], 0))
        await d.drain()
        checks = 0
        for _ in range(300):
            req = None
            if rng.random() < 0.8:
                vals = [rng.randrange(-1000, 1000) for _ in range(4)]
                req = (rng.randrange(6), vals, rng.random() < 0.6)
            read = rng.randrange(6) if rng.random() < 0.15 else None
            want = dict(d.model)  # before this clock's request
            if await d.clock(req, read):
                # Taken: nothing pending on the row, so its value is the model's
                # before this clock's request (which can't be to this row)
                expect = want.get(read, [0] * 4)
                for _ in range(latency + 2):
                    if ctx.get(dut.read_valid):
                        assert unpack(ctx.get(dut.read_data)) == expect
                        checks += 1
                        break
                    await d.clock()
                else:
                    raise AssertionError(f"a read of row {read} was taken but returned no data")
        await d.drain()
        for row in range(6):
            assert await d.read(row) == d.model.get(row, [0] * 4)
        assert checks > 5

    run_sim(dut, tb, max_cycles=50_000)


def test_verilog():
    from amaranth.back import verilog

    skip_without_yosys()
    output = verilog.convert(Combiner(config()), name="Combiner")
    assert "module Combiner" in output and "bank_1" in output
