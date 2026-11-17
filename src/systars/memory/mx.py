"""
MXINT8 encoding (#76): OCP Microscaling (MX v1.0) blocks on a finished result row.

An MX block is ``k`` elements sharing one 8-bit power-of-two scale. MXINT8's elements
are int8 with an implicit scale of 2**-6 (2 integer bits, 6 fractional: -2 to
1.984375), so the format's largest power of two is 2**0 (``emax = 0``). The scale is
E8M0: an exponent ``e``, value ``2**(e - 127)``. A block of values ``V`` encodes as
(spec, "Conversion"):

- ``shared_exp = floor(log2(max |V_i|)) - emax``, the scale ``X = 2**shared_exp``;
- each element ``P_i`` is ``V_i / X`` in the element format, rounded to nearest even
  and saturated (here: ``round(V_i * 2**(6 - shared_exp))`` clamped to ``[-128, 127]``);
- a block of zeros gets the smallest scale (``e = 0``) and zero elements.

The Combiner's results are integers, so ``shared_exp`` is 0..31: ``e = shared_exp +
127``. In hardware ``floor(log2(max |V_i|))`` is the highest set bit of the OR of the
block's magnitudes, so no comparator tree is needed, and an element is a left shift
(``shared_exp < 6``, exact) or one rounding right shift.

**The layout** (decided 2026-11-15): blocks run along the result row, the next layer's
K axis, ``k = min(32, C)``: a row of ``C >= 32`` results holds ``ceil(C / 32)`` blocks
(the last one shorter when ``C`` isn't a multiple of 32), and a shorter row is one
block (``k = C``, below the spec's 32). The row is packed in place: its ``C`` int8
elements in the low bytes, element 0 lowest, then one scale byte per block. STORE moves
both with ``row_bytes = C + blocks``.

**Decode** (L2 -> L1 reading MX tiles for a next layer) is not here: it waits for
MXFP8, whose element decode it would share.
"""

from __future__ import annotations

from amaranth import Cat, Module, Mux, Signal, signed, unsigned
from amaranth.lib.wiring import Component, In, Out

MX_BLOCK = 32
"""The spec's block size."""
MX_FRAC_BITS = 6
"""MXINT8's elements have 6 fractional bits (scale 2**-6)."""
E8M0_BIAS = 127


def mx_blocks(lanes: int) -> list[range]:
    """The blocks of a row of ``lanes`` results: ``k = min(32, lanes)``."""
    k = min(MX_BLOCK, lanes)
    return [range(b, min(b + k, lanes)) for b in range(0, lanes, k)]


def mx_row_bytes(lanes: int) -> int:
    """Bytes an MXINT8 row occupies: its elements, then a scale per block."""
    return lanes + len(mx_blocks(lanes))


# ---------------------------------------------------------------------------
# Reference model
# ---------------------------------------------------------------------------


def _rne_shift(x: int, s: int) -> int:
    """``x / 2**s`` rounded to nearest, ties to even (``s >= 0``)."""
    if s == 0:
        return x
    q, rem, half = x >> s, x & ((1 << s) - 1), 1 << (s - 1)
    return q + (rem > half or (rem == half and q & 1))


def mxint8_block_model(values: list[int]) -> tuple[list[int], int]:
    """One block: its int8 elements and its E8M0 scale byte."""
    amax = max(abs(v) for v in values)
    if amax == 0:
        return [0] * len(values), 0
    shared = amax.bit_length() - 1  # floor(log2(amax)), emax = 0
    shift = shared - MX_FRAC_BITS
    out = []
    for v in values:
        p = v << -shift if shift < 0 else _rne_shift(v, shift)
        out.append(min(127, max(-128, p)))
    return out, shared + E8M0_BIAS


def mxint8_row_model(values: list[int]) -> tuple[list[int], list[int]]:
    """A row: every element, and the scale of each block (``mx_blocks``)."""
    elems, scales = [], []
    for block in mx_blocks(len(values)):
        e, s = mxint8_block_model([values[i] for i in block])
        elems += e
        scales.append(s)
    return elems, scales


def mxint8_decode_model(element: int, scale: int) -> float:
    """The value an element and its block's scale stand for."""
    return element * 2.0 ** (-MX_FRAC_BITS) * 2.0 ** (scale - E8M0_BIAS)


def mxint8_row_bytes_model(values: list[int]) -> bytes:
    """A row's bytes as the Combiner packs them: elements, then scales."""
    elems, scales = mxint8_row_model(values)
    return bytes(e & 0xFF for e in elems) + bytes(scales)


# ---------------------------------------------------------------------------
# Hardware
# ---------------------------------------------------------------------------


class MXElement(Component):
    """One element: ``x * 2**(6 - shared)`` rounded to nearest even, saturated to int8."""

    def __init__(self, wide: int):
        self.wide = wide
        super().__init__({"x": In(signed(wide)), "shared": In(range(wide)), "p": Out(signed(8))})

    def elaborate(self, _platform):
        m = Module()
        w = self.wide
        x, s = self.x, self.shared
        left = Signal(signed(w + MX_FRAC_BITS + 1))
        m.d.comb += left.eq(x << (MX_FRAC_BITS - s)[: (MX_FRAC_BITS).bit_length()])
        # s >= 6: one rounding right shift by r = s - 6 (the epilogue's rule:
        # (x + 2**(r-1) - 1 + t[0]) >> r, the bias a thermometer of r)
        r = Signal(range(w))
        m.d.comb += r.eq(s - MX_FRAC_BITS)
        lsb = x.as_unsigned().bit_select(r, 1)
        # 2**(r-1) - 1: bit i is set when i + 1 < r; r <= w - 1, so bits w - 2 and up never are
        below = Cat(*[(i + 1) < r for i in range(w - 2)])
        bias = Signal(unsigned(w))
        m.d.comb += bias.eq(Mux(r == 0, 0, below + lsb))
        right = Signal(signed(w + 1))
        m.d.comb += right.eq((x + bias) >> r)
        v = Signal(signed(w + MX_FRAC_BITS + 1))
        m.d.comb += v.eq(Mux(s < MX_FRAC_BITS, left, right))
        m.d.comb += self.p.eq(Mux(v < -128, -128, Mux(v > 127, 127, v)))
        return m


class MXInt8Encoder(Component):
    """A row of ``lanes`` ``wide``-bit results as MXINT8 (see the module docstring):
    ``out`` holds the elements in the low bytes, then a scale byte per block."""

    def __init__(self, lanes: int, wide: int, width: int):
        self.lanes, self.wide, self.width = lanes, wide, width
        assert mx_row_bytes(lanes) * 8 <= width, "the packed MX row must fit the row"
        super().__init__({"row": In(width), "out": Out(width)})

    def elaborate(self, _platform):
        m = Module()
        w, lanes = self.wide, self.lanes
        xs = [self.row[i * w : (i + 1) * w].as_signed() for i in range(lanes)]
        elems, scales = [None] * lanes, []
        for b, block in enumerate(mx_blocks(lanes)):
            # floor(log2(max |x|)): the highest set bit of the OR of the magnitudes
            mags = [Mux(xs[i] < 0, -xs[i], xs[i]).as_unsigned()[:w] for i in block]
            union = Signal(w, name=f"mx{b}_or")
            acc = mags[0]
            for mag in mags[1:]:
                acc = acc | mag
            m.d.comb += union.eq(acc)
            shared = Signal(range(w), name=f"mx{b}_shared")
            for bit in range(w):  # the last true assignment wins: the highest bit
                with m.If(union[bit]):
                    m.d.comb += shared.eq(bit)
            scale = Signal(8, name=f"mx{b}_scale")
            m.d.comb += scale.eq(Mux(union == 0, 0, shared + E8M0_BIAS))
            scales.append(scale)
            for i in block:
                el = MXElement(w)
                m.submodules[f"mx_el_{i}"] = el
                m.d.comb += [el.x.eq(xs[i]), el.shared.eq(shared)]
                elems[i] = el.p
        m.d.comb += self.out.eq(Cat(*[e[:8] for e in elems], *scales))
        return m
