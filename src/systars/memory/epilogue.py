"""
The Combiner's epilogue (#75): what it fuses onto a reduction's finished rows.

A COMPUTE's epilogue code (its 4-bit ``activation`` field, ``EPILOGUE_*``) marks the
command that finishes a reduction (``EPILOGUE_FINAL``) and names one of the Combiner's
``PROFILES`` **profiles** (#77), each a set of the tile's ``CMB_*`` CSRs: the format,
the order, the scale and shift, the clip bounds and the SFU function. On that command's
commits the Combiner applies its profile, after the merge:

- **Requantization** (when ``Profile.enable``): each ``acc_bits`` result times a
  signed 16-bit multiplier, shifted right with round-to-nearest-even, saturated to
  ``input_bits`` (int32 -> int8), the usual fixed-point rescale of an integer layer. A
  16-bit multiplier and one rounding shifter keep a lane small enough to build one per
  result (a 32-bit one took a 4x4 tile's Combiner past 10 GB in ice40 synthesis).
- **The SFU banks:** one special-function unit per result lane, computing NONE, ReLU,
  ReLU6 or CLIP(lo, hi) (``Activation``). This is where kpu-sim's Vector Engine (the
  Combiner, stillwater-sc/kpu-sim#429) runs its fused activations; a piecewise-linear
  table is the next function the bank takes.

The order is the profile's (``Profile.act_first``): requantize then the SFU (the SFU
works on the narrow values), or the SFU then requantize. A bias is merged before
both, as a LOAD into the result group the reduction then accumulates onto.

A finished, requantized row keeps its slot and row address: its ``C`` narrow results
are packed in the row's low bytes, element 0 lowest (STORE it with ``row_bytes = C *
input_bits / 8`` and ``STORE_NARROW``). The configuration is read as each finished row
commits: change it only while the tile is idle (``STATUS.busy``, which includes the
Combiner's pipeline).

The ``*_model`` functions are the reference the tests check the RTL against.
"""

from __future__ import annotations

from dataclasses import dataclass

from amaranth import Cat, Const, Module, Mux, Signal, Value, signed
from amaranth.lib.wiring import Component, In, Out

from ..config import Activation
from .mx import MXInt8Encoder, mxint8_row_model

MUL_BITS = 16
"""Width of the requantization multiplier (``CMB_SCALE[15:0]``, signed)."""

PROFILES = 4
"""Combiner profiles: the epilogue configurations a COMPUTE can name."""
EPILOGUE_FINAL = 1 << 3
"""Epilogue code bit: this command finishes the reduction; apply its profile."""
EPILOGUE_PROFILE_MASK = PROFILES - 1
"""Epilogue code bits: the profile (bit 2 is reserved)."""
SFU_SHIFT = 12
"""Where a profile's control word holds its SFU function (``Activation``, 3 bits)."""


def epilogue_code(profile: int, *, final: bool = True) -> int:
    """A COMPUTE's ``activation`` field: ``EPILOGUE_FINAL`` and the profile index."""
    if not 0 <= profile < PROFILES:
        raise ValueError(f"profile {profile}: the Combiner has {PROFILES}")
    return (EPILOGUE_FINAL if final else 0) | profile


STORE_NARROW = 1 << 3
"""A STORE's activation-field bit: its rows are finished and requantized, ``input_bits``
elements in the low bytes (for the activation and the TRANSPOSE corner turn)."""

SFU_FUNCTIONS = (Activation.NONE, Activation.RELU, Activation.RELU6, Activation.CLIP)


@dataclass(frozen=True)
class Profile:
    """A Combiner profile: the epilogue a finished row gets (the tile's ``CMB_*`` CSRs
    for that profile; ``control_word``, ``scale_word`` and ``clip_word`` encode them)."""

    mul: int = 1
    """Signed 16-bit multiplier."""
    shift: int = 0
    """Right shift, 0..63, rounding to nearest even."""
    enable: bool = False
    """Requantize to ``input_bits``; otherwise the results stay ``acc_bits`` wide."""
    mx: bool = False
    """MXINT8 (#76): the SFU, then MX blocks (``memory/mx.py``); takes precedence over
    ``enable``."""
    act_first: bool = False
    """Apply the SFU before requantizing (on the wide results)."""
    clip_lo: int = 0
    """CLIP's lower bound (signed 16-bit), in the domain the SFU works in: the narrow
    results when requantizing first, else the wide ones. A bound outside the narrow
    range saturates the narrow result (it doesn't wrap)."""
    clip_hi: int = 0
    """CLIP's upper bound (signed 16-bit)."""
    sfu: int = 0
    """The SFU function, an ``Activation`` code (NONE, RELU, RELU6, CLIP)."""

    def scale_word(self) -> int:
        """``CMB_SCALE``: the multiplier in [15:0]."""
        return self.mul & 0xFFFF

    def control_word(self) -> int:
        """``CMB_CONTROL``: shift[5:0], requant[8], act_first[9], mxint8[10], sfu[14:12]."""
        return (
            (self.shift & 0x3F)
            | (int(self.enable) << 8)
            | (int(self.act_first) << 9)
            | (int(self.mx) << 10)
            | ((self.sfu & 0b111) << SFU_SHIFT)
        )

    def clip_word(self) -> int:
        """``CMB_CLIP``: lo[15:0], hi[31:16]."""
        return (self.clip_lo & 0xFFFF) | (self.clip_hi & 0xFFFF) << 16


Requant = Profile
"""The name a profile had before there were several (#75)."""


# ---------------------------------------------------------------------------
# Reference models
# ---------------------------------------------------------------------------


def wrap(x: int, bits: int) -> int:
    """``x`` as a signed ``bits``-bit value (two's complement wrap)."""
    x &= (1 << bits) - 1
    return x - (1 << bits) if x >> (bits - 1) else x


def sfu_model(x: int, code: int, cfg: Profile) -> int:
    if code == Activation.RELU.value:
        return max(0, x)
    if code == Activation.RELU6.value:
        return min(6, max(0, x))
    if code == Activation.CLIP.value:
        return min(cfg.clip_hi, max(cfg.clip_lo, x))
    return x


def requant_model(x: int, cfg: Profile, out_bits: int) -> int:
    """``x * mul``, shifted right ``shift`` with round-to-nearest-even, saturated."""
    p = x * wrap(cfg.mul, MUL_BITS)
    s = cfg.shift
    if s:
        q, rem, half = p >> s, p & ((1 << s) - 1), 1 << (s - 1)
        p = q + (rem > half or (rem == half and q & 1))
    lo, hi = -(1 << (out_bits - 1)), (1 << (out_bits - 1)) - 1
    return min(hi, max(lo, p))


def epilogue_model(values: list[int], cfg: Profile, out_bits: int) -> list[int]:
    """A finished row's results after profile ``cfg``. For MXINT8 (``cfg.mx``): the
    elements, then the block scales (``mxint8_row_model``)."""
    sfu = cfg.sfu
    if cfg.mx:
        elems, scales = mxint8_row_model([sfu_model(x, sfu, cfg) for x in values])
        return elems + scales
    out = []
    for x in values:
        if not cfg.enable:
            out.append(sfu_model(x, sfu, cfg))
        elif cfg.act_first:
            out.append(requant_model(sfu_model(x, sfu, cfg), cfg, out_bits))
        else:  # the SFU on the narrow result; CLIP's bounds may lie outside it
            lo, hi = -(1 << (out_bits - 1)), (1 << (out_bits - 1)) - 1
            out.append(min(hi, max(lo, sfu_model(requant_model(x, cfg, out_bits), sfu, cfg))))
    return out


# ---------------------------------------------------------------------------
# Hardware
# ---------------------------------------------------------------------------


def sfu(x: Value, code: Value, clip_lo: Value, clip_hi: Value) -> Value:
    """One SFU: ``Activation`` ``code`` on the signed lane ``x`` (same width out)."""
    relu = Mux(x < 0, 0, x)
    relu6 = Mux(relu > 6, 6, relu)
    clip = Mux(x < clip_lo, clip_lo, Mux(x > clip_hi, clip_hi, x))
    return Mux(
        code == Activation.RELU.value,
        relu,
        Mux(code == Activation.RELU6.value, relu6, Mux(code == Activation.CLIP.value, clip, x)),
    )


def requantize(x: Value, mul: Value, shift: Value, out_bits: int) -> Value:
    """``x * mul >> shift`` (round to nearest even), saturated to ``out_bits`` signed.

    One shifter: with ``t = prod >> s``, round-to-nearest-even is
    ``(prod + 2**(s-1) - 1 + t[0]) >> s``; ``t[0]`` is bit ``s`` of the product, and
    ``2**(s-1) - 1`` is the thermometer of the ``s - 1`` bits below it."""
    width = x.shape().width + MUL_BITS + 1
    prod = (x.as_signed() * mul[:MUL_BITS].as_signed()).as_signed()
    # Past the product's width every result rounds to 0 (|prod| < 2**(width - 2)), as
    # it does at width - 1: clamp there
    s = Mux(shift > width - 1, width - 1, shift)
    lsb = prod.as_unsigned().bit_select(s, 1)
    below = Cat(*[Const(i + 1, 7) < s for i in range(width)])  # 2**(s-1) - 1
    bias = Mux(s == 0, 0, below + lsb)  # unsigned
    r = ((prod + bias) >> s).as_signed()
    lo, hi = -(1 << (out_bits - 1)), (1 << (out_bits - 1)) - 1
    return Mux(r < lo, lo, Mux(r > hi, hi, r))


class EpilogueLane(Component):
    """The epilogue on one result: the SFU and, with ``enable``, requantization to
    ``narrow`` bits. One submodule per lane, all alike, so a hierarchical synthesis
    builds the lane once."""

    def __init__(self, wide: int, narrow: int):
        self.wide, self.narrow = wide, narrow
        super().__init__(
            {
                "x": In(signed(wide)),
                "code": In(3),
                "mul": In(signed(MUL_BITS)),
                "shift": In(6),
                "act_first": In(1),
                "clip_lo": In(signed(16)),
                "clip_hi": In(signed(16)),
                "wide_out": Out(signed(wide)),
                "narrow_out": Out(signed(narrow)),
            }
        )

    def elaborate(self, _platform):
        m = Module()
        act = Signal(signed(self.wide))
        m.d.comb += act.eq(sfu(self.x, self.code, self.clip_lo, self.clip_hi))
        src = Signal(signed(self.wide))
        m.d.comb += src.eq(Mux(self.act_first, act, self.x))
        rq = Signal(signed(self.narrow))
        m.d.comb += rq.eq(requantize(src, self.mul, self.shift, self.narrow))
        # The SFU after requantizing works on the narrow result, but CLIP's 16-bit bounds
        # may lie outside it: saturate, as the model does, rather than wrap
        post = Signal(signed(max(16, self.narrow)))  # CLIP's bounds, or a wider narrow
        m.d.comb += post.eq(sfu(rq, self.code, self.clip_lo, self.clip_hi))
        lo, hi = -(1 << (self.narrow - 1)), (1 << (self.narrow - 1)) - 1
        post_sat = Mux(post < lo, lo, Mux(post > hi, hi, post))
        m.d.comb += [
            self.wide_out.eq(act),
            self.narrow_out.eq(Mux(self.act_first, rq, post_sat)),
        ]
        return m


def apply_epilogue(
    m, row: Value, cfg: dict, *, lanes: int, wide: int, narrow: int, width: int
) -> Value:
    """The epilogue on a merged row of ``lanes`` ``wide``-bit results, one
    ``EpilogueLane`` each: the SFU and, when ``cfg["enable"]``, requantization to
    ``narrow`` bits packed in the low bytes; or, when ``cfg["mx"]``, the SFU then MXINT8
    (``memory/mx.py``). ``cfg`` holds the configuration signals (``mul``, ``shift``,
    ``enable``, ``act_first``, ``mx``, ``sfu``, ``clip_lo``, ``clip_hi``): the profile the row's
    request names."""
    wide_out, narrow_out = [], []
    for i in range(lanes):
        lane = EpilogueLane(wide, narrow)
        m.submodules[f"epilogue_{i}"] = lane
        m.d.comb += [
            lane.x.eq(row[i * wide : (i + 1) * wide]),
            lane.code.eq(cfg["sfu"]),
            lane.mul.eq(cfg["mul"][:MUL_BITS]),
            lane.shift.eq(cfg["shift"]),
            lane.act_first.eq(cfg["act_first"]),
            lane.clip_lo.eq(cfg["clip_lo"]),
            lane.clip_hi.eq(cfg["clip_hi"]),
        ]
        wide_out.append(lane.wide_out[:wide])
        narrow_out.append(lane.narrow_out[:narrow])
    packed = Signal(width, name="ep_narrow")
    m.d.comb += packed.eq(Cat(*narrow_out))
    # MXINT8 (#76, memory/mx.py): the SFU's wide results, encoded as MX blocks
    mx = MXInt8Encoder(lanes, wide, width)
    m.submodules.mxint8 = mx
    m.d.comb += mx.row.eq(Cat(*wide_out))
    return Mux(cfg["mx"], mx.out, Mux(cfg["enable"], packed, Cat(*wide_out)))
