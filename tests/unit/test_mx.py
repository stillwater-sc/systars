"""
Unit tests for MXINT8 (memory/mx.py, #76).

These tests verify:
1. The model against the OCP MX v1.0 rules: the shared scale, round-to-nearest-even,
   saturation, the zero block, and that decoding recovers the values within half an
   element step
2. The blocks of a row: k = min(32, C), a short last block, the packed row's size
3. The hardware encoder equals the model, for rows of 4, 16, 48 and 64 results
4. The Combiner encodes a FINAL commit only, after the merge and the SFU
"""

import random

import pytest
from amaranth.sim import Simulator

from systars.config import Activation
from systars.csr import CSR
from systars.memory.combiner import Combiner
from systars.memory.epilogue import EPILOGUE_FINAL, Requant, epilogue_model
from systars.memory.mx import (
    E8M0_BIAS,
    MXInt8Encoder,
    mx_blocks,
    mx_row_bytes,
    mxint8_block_model,
    mxint8_decode_model,
    mxint8_row_bytes_model,
)

from ._sim import run_sim
from .test_combiner import Driver, config, pack


class TestModel:
    def test_the_shared_scale_is_the_largest_magnitudes_power_of_two(self):
        # amax 64 = 2**6: scale 2**6 (e = 6 + 127), elements V / 2**6 * 2**6 = V
        assert mxint8_block_model([64, -64, 1, 0]) == ([64, -64, 1, 0], E8M0_BIAS + 6)
        # amax 1000 -> 2**9: elements V / 8, 5 / 8 = 0.625 rounds to 1
        assert mxint8_block_model([1000, 5]) == ([125, 1], E8M0_BIAS + 9)
        # small values: amax 3 -> 2**1, elements V * 2**5, exact
        assert mxint8_block_model([3, -1]) == ([96, -32], E8M0_BIAS + 1)

    def test_ties_round_to_even_and_the_top_saturates(self):
        # amax 255 -> 2**7: 255 / 2 = 127.5, to even 128, saturated to 127
        assert mxint8_block_model([255]) == ([127], E8M0_BIAS + 7)
        # 129 / 2 = 64.5 -> 64; 131 / 2 = 65.5 -> 66; -129 / 2 = -64.5 -> -64
        assert mxint8_block_model([129, 131, -129, 200]) == ([64, 66, -64, 100], E8M0_BIAS + 7)
        assert mxint8_block_model([-(2**31)]) == ([-64], E8M0_BIAS + 31)

    def test_a_zero_block_has_the_smallest_scale(self):
        assert mxint8_block_model([0] * 8) == ([0] * 8, 0)

    def test_decoding_recovers_the_values(self):
        rng = random.Random(76)
        for _ in range(500):
            block = [rng.randrange(-(2**20), 2**20) for _ in range(32)]
            elems, scale = mxint8_block_model(block)
            step = 2.0 ** (scale - E8M0_BIAS - 6)
            for v, p in zip(block, elems, strict=True):
                assert abs(mxint8_decode_model(p, scale) - v) <= step  # half a step + saturation


class TestBlocks:
    @pytest.mark.parametrize(
        "lanes,blocks",
        [
            (4, [range(0, 4)]),
            (16, [range(0, 16)]),
            (48, [range(0, 32), range(32, 48)]),
            (64, [range(0, 32), range(32, 64)]),
        ],  # fmt: skip
    )
    def test_k_is_min_32_c(self, lanes, blocks):
        assert mx_blocks(lanes) == blocks
        assert mx_row_bytes(lanes) == lanes + len(blocks)


@pytest.mark.parametrize("lanes", [4, 16, 48, 64])
def test_the_encoder_is_the_model(lanes):
    width = lanes * 32
    dut = MXInt8Encoder(lanes, 32, width)
    rng = random.Random(lanes)
    cases = [[0] * lanes, [2**31 - 1] * lanes, [-(2**31)] * lanes, [255] + [0] * (lanes - 1),
             [129, 131, -129] + [1] * (lanes - 3)]  # fmt: skip
    for _ in range(300):
        mag = rng.choice([3, 7, 12, 20, 31])
        cases.append([rng.randrange(-(2**mag), 2**mag) for _ in range(lanes)])

    async def tb(ctx):
        for vals in cases:
            ctx.set(dut.row, sum((v & 0xFFFF_FFFF) << (32 * i) for i, v in enumerate(vals)))
            await ctx.delay(1e-9)
            got = ctx.get(dut.out).to_bytes(width // 8, "little")[: mx_row_bytes(lanes)]
            assert got == mxint8_row_bytes_model(vals), vals[:6]

    sim = Simulator(dut)
    sim.add_testbench(tb)
    sim.run()


@pytest.mark.parametrize("code", [0, Activation.RELU.value])
def test_the_combiner_encodes_the_final_commit_only(code):
    """Two partials stay raw sums; the FINAL accumulate is merged, run through the SFU
    and encoded: its row's low bytes are the elements, then the scale."""
    cfg = config()  # 4 results a row: one block
    dut = Combiner(cfg)
    a, b, c = [100, -2000, 300, -4], [5, 6, -7, 8], [-1000, 900, 13, 27]

    async def tb(ctx):
        ctx.set(dut.control, CSR.CMB_MXINT8)
        d = Driver(ctx, dut, cfg)
        ctx.set(dut.write_epilogue, code)
        await d.clock((3, a, 0))
        await d.clock((3, b, 1))
        await d.drain()
        assert await d.read(3) == [x + y for x, y in zip(a, b, strict=True)]
        ctx.set(dut.write_epilogue, EPILOGUE_FINAL | code)
        await d.clock((3, c, 1))
        ctx.set(dut.write_epilogue, 0)
        await d.drain()
        merged = [x + y + z for x, y, z in zip(a, b, c, strict=True)]
        want = epilogue_model(merged, code, Requant(mx=True), cfg.input_bits)
        got = pack(await d.read(3)).to_bytes(cfg.combiner_width // 8, "little")
        assert list(got[: mx_row_bytes(4)]) == [v & 0xFF for v in want]

    run_sim(dut, tb)
