"""
Random programs for the differential harness (#83): a Matmul on the systolic tile.

``generate(seed)`` draws a ``Program`` from the space the machine documents: the tile
config, the dataflow, the shapes (edge tiles included), bias and accumulate, the
buffering, the Combiner profile (its format, SFU function and order) and which of the
four profile slots it sits in (the others hold decoys), the layout (odd addresses,
slack in the strides) and the AXI memory's backpressure. A program follows the rules
(``violation`` is None) unless it was drawn to break one, which the machine must refuse
rather than run wrong.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field, replace
from typing import Any

from systars.config import Activation
from systars.memory.epilogue import PROFILES, Profile

OS, WS, IS = 0, 1, 2
SFUS = [a.value for a in (Activation.NONE, Activation.RELU, Activation.RELU6, Activation.CLIP)]
FORMATS = ("wide", "int8", "mxint8")
OPERAND_RANGES = (128, 8, 2)

# The rules a program can break (each must end in UNSUPPORTED_LAYOUT, nothing run)
VIOLATIONS = ("mx_in_is", "mx_edge_n", "mx_accumulate", "int8_accumulate")


@dataclass
class Program:
    seed: int
    dim: int = 4
    occupancy: int = 1
    combiner_latency: int = 2
    dataflow: int = OS
    M: int = 4
    N: int = 4
    K: int = 4
    bias: str | None = None  # None, "row", "matrix"
    accumulate: bool = False
    buffers: int = 2
    profile_slot: int = 0
    fmt: str = "wide"
    sfu: int = 0
    act_first: bool = False
    mul: int = 1
    shift: int = 0
    clip_lo: int = 0
    clip_hi: int = 0
    stall: float = 0.0
    operands: int = 128  # A and B elements in [-operands, operands): small values make ties
    a_off: int = 1
    b_off: int = 7
    c_off: int = 3
    d_off: int = 5
    stride_slack: int = 0
    violation: str | None = None
    decoys: list[dict[str, Any]] = field(default_factory=list)

    def profile(self) -> Profile:
        return Profile(
            mul=self.mul, shift=self.shift, enable=self.fmt == "int8", mx=self.fmt == "mxint8",
            act_first=self.act_first, clip_lo=self.clip_lo, clip_hi=self.clip_hi, sfu=self.sfu,
        )  # fmt: skip

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_json(d: dict[str, Any]) -> Program:
        """A program from its JSON (``to_json``). It must carry a decoy for each of the
        ``PROFILES`` slots, as the generator writes them (``ValueError`` otherwise)."""
        prog = Program(**d)
        if len(prog.decoys) != PROFILES:
            raise ValueError(
                f"a program needs {PROFILES} decoys (one per profile slot), got {len(prog.decoys)}"
            )
        return prog

    def bins(self) -> set[str]:
        """The coverage bins this program hits. A program that breaks a rule is refused
        before it runs, so it hits only its violation's bin, never an execution bin."""
        if self.violation:
            return {f"violation={self.violation}"}
        df = ("os", "ws", "is")[self.dataflow]
        b = {
            f"dataflow={df}", f"format={self.fmt}", f"sfu={self.sfu}", f"dim={self.dim}",
            f"bias={self.bias}", f"accumulate={self.accumulate}", f"buffers={self.buffers}",
            f"slot={self.profile_slot}", f"latency={self.combiner_latency}",
            f"occupancy={self.occupancy}", f"stall={self.stall > 0}",
            f"edge_m={self.M % self.dim != 0}", f"edge_n={self.N % self.dim != 0}",
            f"edge_k={self.K % self.dim != 0}", f"multi_k={self.dim < self.K}",
            f"operands={self.operands}",
            f"dataflow×format={df}×{self.fmt}",
            f"format×sfu×order={self.fmt}×{self.sfu}×{'act_first' if self.act_first else 'scale_first'}",
        }  # fmt: skip
        return b


def all_bins() -> set[str]:
    """Every bin a full run should hit (``Program.bins`` of the legal space and every
    violation): the coverage report lists the ones it missed."""
    out = set()
    for df in ("os", "ws", "is"):
        out.add(f"dataflow={df}")
        for fmt in FORMATS:
            if not (df == "is" and fmt == "mxint8"):
                out.add(f"dataflow×format={df}×{fmt}")
    for fmt in FORMATS:
        out.add(f"format={fmt}")
        for sfu in SFUS:
            # The order matters only when requantizing: the wide format has no scale,
            # and MXINT8 always runs the SFU first
            for order in ("act_first", "scale_first") if fmt == "int8" else ("scale_first",):
                out.add(f"format×sfu×order={fmt}×{sfu}×{order}")
    out |= {f"sfu={s}" for s in SFUS}
    out |= {f"bias={b}" for b in (None, "row", "matrix")}
    out |= {f"accumulate={a}" for a in (True, False)} | {f"buffers={b}" for b in (1, 2)}
    out |= {f"slot={n}" for n in range(PROFILES)} | {f"latency={n}" for n in (1, 2, 3)}
    out |= {f"occupancy={n}" for n in (1, 2)} | {f"stall={s}" for s in (True, False)}
    out |= {f"dim={d}" for d in (2, 4)}
    out |= {f"operands={n}" for n in OPERAND_RANGES}
    for e in ("edge_m", "edge_n", "edge_k", "multi_k"):
        out |= {f"{e}=True", f"{e}=False"}
    out |= {f"violation={v}" for v in VIOLATIONS}
    return out


def _random_profile(rng: random.Random, fmt: str, magnitude: int = 1 << 20) -> dict[str, Any]:
    """A profile; ``magnitude`` (the results' expected size) aims the requantization
    shift so results land in int8's range, not all at its saturation."""
    sfu = rng.choice(SFUS)
    act_first = rng.random() < 0.4
    mul = rng.choice([1, 1, -1, rng.randrange(-64, 64), rng.randrange(-(2**15), 2**15)])
    aimed = max(0, (magnitude * max(1, abs(mul))).bit_length() - 7 + rng.randrange(-2, 2))
    narrow_sfu = fmt == "int8" and not act_first  # the SFU runs on the int8 results
    lim = 120 if narrow_sfu else 30000
    lo = rng.randrange(-lim, 0)
    return {
        "fmt": fmt, "sfu": sfu, "act_first": act_first and fmt == "int8",
        "mul": mul,
        "shift": rng.choice([aimed, aimed, aimed + 1, rng.randrange(0, 4), rng.randrange(0, 64)]),
        "clip_lo": lo, "clip_hi": rng.randrange(0, lim),
    }  # fmt: skip


def generate(seed: int) -> Program:
    """A random program: legal, or (one in ten) breaking one documented rule."""
    rng = random.Random(seed)
    dim = rng.choice([2, 4, 4])
    fmt = rng.choice(["wide", "wide", "int8", "mxint8"])
    dataflow = rng.choice([OS, WS, IS])
    if fmt == "mxint8" and dataflow == IS:
        dataflow = rng.choice([OS, WS])
    M, K = rng.randrange(1, 3 * dim + 1), rng.randrange(1, 3 * dim + 1)
    N = rng.randrange(1, 3 * dim + 1)
    if fmt == "mxint8":
        N = dim * rng.randrange(1, 4)  # no edge tile in N
    accumulate = fmt == "wide" and rng.random() < 0.3
    operands = rng.choice(OPERAND_RANGES)
    # Half the int8 programs are tie-prone: small operands, a small multiplier and a
    # shift of 1-3, so exact halves are common (round-to-nearest-even shows) and the
    # results rarely saturate
    tie_prone = fmt == "int8" and rng.random() < 0.5
    if tie_prone:
        operands = 8
    magnitude = operands * operands * K + (5000 if operands == 128 else 40)
    prog = Program(
        seed=seed, dim=dim, occupancy=rng.choice([1, 1, 2]),
        combiner_latency=rng.choice([1, 2, 3]), dataflow=dataflow, M=M, N=N, K=K,
        bias=rng.choice([None, None, "row", "matrix"]), accumulate=accumulate,
        buffers=rng.choice([1, 2, 2]), profile_slot=rng.randrange(PROFILES),
        stall=rng.choice([0.0, 0.0, 0.3]), a_off=rng.randrange(0, 16),
        b_off=rng.randrange(0, 16), c_off=rng.randrange(0, 16), d_off=rng.randrange(0, 16),
        stride_slack=rng.choice([0, 0, rng.randrange(1, 9)]), operands=operands,
        **_random_profile(rng, fmt, magnitude),
    )  # fmt: skip
    if tie_prone:
        prog.mul, prog.shift = rng.choice([1, -1, 3]), rng.randrange(1, 4)
    prog.decoys = [_random_profile(rng, rng.choice(FORMATS)) for _ in range(PROFILES)]
    if rng.random() < 0.1:
        prog = violate(prog, rng.choice(VIOLATIONS))
    return prog


def violate(prog: Program, kind: str) -> Program:
    """``prog`` changed to break rule ``kind``."""
    if kind == "mx_in_is":
        # N whole tiles, so the IS rule is the only one broken
        n = prog.dim * max(1, prog.N // prog.dim)
        return replace(prog, fmt="mxint8", N=n, dataflow=IS, accumulate=False, violation=kind)
    if kind == "mx_edge_n":
        n = prog.dim * max(1, prog.N // prog.dim) + 1
        df = WS if prog.dataflow == IS else prog.dataflow
        return replace(prog, fmt="mxint8", N=n, dataflow=df, accumulate=False, violation=kind)
    if kind == "mx_accumulate":
        df = WS if prog.dataflow == IS else prog.dataflow
        return replace(prog, fmt="mxint8", N=prog.dim, dataflow=df, accumulate=True, violation=kind)
    if kind == "int8_accumulate":
        return replace(prog, fmt="int8", accumulate=True, violation=kind)
    raise ValueError(kind)


def legal(prog: Program) -> bool:
    """Whether ``prog`` follows the documented rules (``violation`` aside)."""
    if prog.fmt == "mxint8":
        return prog.dataflow != IS and prog.N % prog.dim == 0 and not prog.accumulate
    return not (prog.fmt == "int8" and prog.accumulate)


def shrink_candidates(prog: Program):
    """Smaller programs to try when ``prog`` fails, simplest change first. Each stays
    on the same side of the rules: a legal program stays legal, and a violation keeps
    its violation."""
    for cand in _candidates(prog):
        if prog.violation or legal(cand):
            yield cand


def _candidates(prog: Program):
    if prog.stall:
        yield replace(prog, stall=0.0)
    if prog.bias:
        yield replace(prog, bias=None)
    if prog.accumulate and not prog.violation:
        yield replace(prog, accumulate=False)
    if prog.stride_slack:
        yield replace(prog, stride_slack=0)
    if any((prog.a_off, prog.b_off, prog.c_off, prog.d_off)):
        yield replace(prog, a_off=0, b_off=0, c_off=0, d_off=0)
    if prog.sfu:
        yield replace(prog, sfu=0)
    if prog.act_first:
        yield replace(prog, act_first=False)
    if prog.profile_slot:
        yield replace(prog, profile_slot=0)
    if prog.combiner_latency > 1:
        yield replace(prog, combiner_latency=prog.combiner_latency - 1)
    if prog.occupancy > 1:
        yield replace(prog, occupancy=1)
    for dim_name in ("M", "K", "N"):
        if dim_name == "N" and prog.violation == "mx_edge_n":
            continue
        v = getattr(prog, dim_name)
        step = prog.dim if (dim_name == "N" and prog.fmt == "mxint8") else 1
        for smaller in sorted({max(step, (v // 2) // step * step), v - step}):
            if step <= smaller < v:
                yield replace(prog, **{dim_name: smaller})
    if prog.buffers > 1:
        yield replace(prog, buffers=1)
