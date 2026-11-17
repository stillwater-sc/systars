"""
Unit tests for the Combiner's epilogue (memory/epilogue.py, #75).

These tests verify:
1. Requantization: the hardware equals the model, round-to-nearest-even ties and
   saturation included
2. The SFU bank's functions: NONE, ReLU, ReLU6, CLIP
3. The Combiner applies the epilogue only on a FINAL request's commit, after the merge,
   in either order, and packs requantized rows in their low bytes
"""

import random
from dataclasses import replace

import pytest
from amaranth import Elaboratable, Module, Signal, signed
from amaranth.sim import Simulator

from systars.config import Activation
from systars.memory.combiner import Combiner
from systars.memory.epilogue import (
    EPILOGUE_FINAL,
    Requant,
    epilogue_model,
    requant_model,
    requantize,
    sfu,
    sfu_model,
)

from ._sim import run_sim
from .test_combiner import Driver, config, pack


class Unit(Elaboratable):
    def __init__(self):
        self.x = Signal(signed(32))
        self.mul = Signal(signed(32))
        self.shift = Signal(6)
        self.code = Signal(3)
        self.lo = Signal(signed(16))
        self.hi = Signal(signed(16))
        self.rq = Signal(signed(8))
        self.act = Signal(signed(32))

    def elaborate(self, _platform):
        m = Module()
        m.d.comb += [
            self.rq.eq(requantize(self.x, self.mul, self.shift, 8)),
            self.act.eq(sfu(self.x, self.code, self.lo, self.hi)),
        ]
        return m


def run_comb(cases, check):
    dut = Unit()

    async def tb(ctx):
        for case in cases:
            for name, value in case.items():
                ctx.set(getattr(dut, name), value)
            await ctx.delay(1e-9)
            check(ctx, dut, case)

    sim = Simulator(dut)
    sim.add_testbench(tb)
    sim.run()


def test_the_model_rounds_ties_to_even_and_saturates():
    def rq(x, mul=1, shift=1):
        return requant_model(x, Requant(mul=mul, shift=shift, enable=True), 8)

    assert [rq(x) for x in (1, 3, 5, 7, -1, -3, -5, -7)] == [0, 2, 2, 4, 0, -2, -2, -4]
    assert rq(6, shift=2) == 2 and rq(10, shift=2) == 2 and rq(14, shift=2) == 4  # 1.5, 2.5, 3.5
    assert rq(7, shift=2) == 2  # 1.75
    assert rq(1000, shift=0) == 127 and rq(-1000, shift=0) == -128
    assert rq(2**31 - 1, mul=2**15 - 1, shift=63) == 0
    assert rq(-(2**31), mul=-(2**15), shift=0) == 127
    assert rq(-(2**31), mul=2**15 - 1, shift=62) == 0  # past the product's width
    assert rq(100, mul=1 << 16) == 0  # the multiplier is 16 bits: 1 << 16 is 0


def test_requantize_is_the_model():
    rng = random.Random(75)
    ties = [{"x": x, "mul": 1, "shift": s} for s in range(1, 8) for x in range(-300, 300)]
    edges = [{"x": x, "mul": mu, "shift": s} for x in (2**31 - 1, -(2**31), 0, 1, -1)
             for mu in (2**15 - 1, -(2**15), 2**31 - 1, 1, -1)
             for s in (0, 1, 31, 32, 46, 47, 48, 49, 62, 63)]  # fmt: skip
    rand = [{"x": rng.randrange(-(2**31), 2**31), "mul": rng.randrange(-(2**31), 2**31),
             "shift": rng.randrange(64)} for _ in range(2000)]  # fmt: skip

    def check(ctx, dut, c):
        want = requant_model(c["x"], Requant(mul=c["mul"], shift=c["shift"], enable=True), 8)
        assert ctx.get(dut.rq) == want, c

    run_comb(ties + edges + rand, check)


@pytest.mark.parametrize("code", [a.value for a in (Activation.NONE, Activation.RELU,
                                                    Activation.RELU6, Activation.CLIP)])  # fmt: skip
def test_the_sfu_is_the_model(code):
    rng = random.Random(code)
    cases = [{"x": x, "code": code, "lo": -10, "hi": 20} for x in range(-40, 40)]
    cases += [{"x": rng.randrange(-(2**31), 2**31), "code": code, "lo": rng.randrange(-300, 0),
               "hi": rng.randrange(0, 300)} for _ in range(500)]  # fmt: skip

    def check(ctx, dut, c):
        want = sfu_model(c["x"], code, Requant(clip_lo=c["lo"], clip_hi=c["hi"]))
        assert ctx.get(dut.act) == want, c

    run_comb(cases, check)


def narrow(word, lanes=4):
    """A requantized row's int8 results, from its low bytes."""
    vals = [(word >> (8 * i)) & 0xFF for i in range(lanes)]
    return [v - 256 if v & 0x80 else v for v in vals]


def set_profile(ctx, dut, n, profile):
    """Write profile ``n``'s configuration on the Combiner's ports (the tile's CSRs)."""
    ctx.set(getattr(dut, f"scale_{n}"), profile.scale_word())
    ctx.set(getattr(dut, f"control_{n}"), profile.control_word())
    ctx.set(getattr(dut, f"clip_{n}"), profile.clip_word())


@pytest.mark.parametrize(
    "code,rq",
    [
        (Activation.RELU.value, Requant()),
        (Activation.CLIP.value, Requant(clip_lo=-50, clip_hi=60)),
        (0, Requant(mul=3, shift=5, enable=True)),
        (Activation.RELU6.value, Requant(mul=-2, shift=3, enable=True)),
        (
            Activation.CLIP.value,
            Requant(mul=1, shift=2, enable=True, act_first=True, clip_lo=-100, clip_hi=40),
        ),  # fmt: skip
        # CLIP bounds outside int8 after requantizing: the result saturates, not wraps
        (Activation.CLIP.value, Requant(enable=True, clip_lo=200, clip_hi=300)),
        (Activation.CLIP.value, Requant(enable=True, clip_lo=-300, clip_hi=-200)),
    ],
)
def test_the_combiner_applies_it_on_the_final_commit_only(code, rq):
    """Two partials, then a FINAL accumulate naming profile 3: the earlier commits stay
    raw sums (a non-final row is still a partial), the final one is merged, then
    finished with profile 3, whatever profile 0 holds."""
    cfg = config()
    dut = Combiner(cfg)
    a, b, c = [100, -200, 300, -400], [5, 6, -7, 8], [-1000, 900, 13, 27]
    prof = replace(rq, sfu=code)

    async def tb(ctx):
        set_profile(ctx, dut, 3, prof)
        set_profile(ctx, dut, 0, Requant(enable=True, mul=9, shift=1, sfu=1))  # a decoy
        d = Driver(ctx, dut, cfg)
        ctx.set(dut.write_epilogue, 3)  # no FINAL bit: no epilogue
        await d.clock((3, a, 0))
        await d.clock((3, b, 1))
        await d.drain()
        assert await d.read(3) == [x + y for x, y in zip(a, b, strict=True)]
        ctx.set(dut.write_epilogue, EPILOGUE_FINAL | 3)
        await d.clock((3, c, 1))
        ctx.set(dut.write_epilogue, 0)
        await d.drain()
        merged = [x + y + z for x, y, z in zip(a, b, c, strict=True)]
        want = epilogue_model(merged, prof, cfg.input_bits)
        got = await d.read(3)
        if prof.enable:
            got = narrow(pack(got))
        assert got == want

    run_sim(dut, tb)


def test_each_request_gets_the_profile_it_names():
    """#77: four back-to-back FINAL commits, each naming another profile, on four rows:
    each row is finished with its own profile, though they share the pipeline."""
    cfg = config(latency=3)
    dut = Combiner(cfg)
    profiles = [
        Requant(),
        Requant(sfu=Activation.RELU.value),
        Requant(enable=True, mul=3, shift=4),
        Requant(
            enable=True,
            mul=-1,
            shift=2,
            act_first=True,
            sfu=Activation.CLIP.value,
            clip_lo=-30,
            clip_hi=10,
        ),  # fmt: skip
    ]
    rows = [[100, -200, 3000, -4], [-5, 6, -70, 800], [1000, -900, 13, 27], [64, -64, 32, -32]]

    async def tb(ctx):
        for n, prof in enumerate(profiles):
            set_profile(ctx, dut, n, prof)
        d = Driver(ctx, dut, cfg)
        for n, row in enumerate(rows):
            ctx.set(dut.write_epilogue, EPILOGUE_FINAL | n)
            await d.clock((n, row, 0))
        ctx.set(dut.write_epilogue, 0)
        await d.drain()
        for n, (row, prof) in enumerate(zip(rows, profiles, strict=True)):
            got = await d.read(n)
            if prof.enable:
                got = narrow(pack(got))
            assert got == epilogue_model(row, prof, cfg.input_bits), n

    run_sim(dut, tb)


def test_busy_covers_the_pipeline():
    """``busy`` holds while a request is in the Combiner's pipeline, so the tile's
    STATUS.busy clears only once every finished row has committed with the epilogue
    configuration it was computed under."""
    cfg = config(latency=3)
    dut = Combiner(cfg)
    seen = []

    async def tb(ctx):
        d = Driver(ctx, dut, cfg)
        assert not ctx.get(dut.busy)
        await d.clock((1, [1, 2, 3, 4], 0))
        for _ in range(5):
            seen.append(ctx.get(dut.busy))
            await d.clock()

    run_sim(dut, tb)
    assert seen == [1, 1, 1, 0, 0]  # the request, then latency - 1 more stages
