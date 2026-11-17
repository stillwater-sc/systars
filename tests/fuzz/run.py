"""
Run a random program on the systolic tile and check it against the reference models.

``run(prog)`` builds the tile the program asks for, runs its Matmul on ``SystolicTop``
with the AXI memory model (with backpressure), and returns the problems it found:

- a legal program must finish with every sticky monitor clear (``l1_error``,
  ``l2_error``, ``sync_error``, the instruction's error; a monitor that fired is reported
  with the tile's decoded first error, #84), C must equal numpy's result
  through the epilogue model (``epilogue_model``, ``mxint8_*_model``), and no byte
  outside C may change;
- a program that breaks a rule must be refused (``UNSUPPORTED_LAYOUT``) with nothing
  written.

``shrink(prog)`` reduces a failing program to the smallest one that still fails.
``python -m tests.fuzz.run '<program json>'`` (or ``--seed N``) replays one, and with
``--trace BUNDLE`` records the tile's trace (#27): its commands, first error and internal
lanes, which ``python -m systars.trace BUNDLE --svg DIR`` reports on and draws.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from amaranth import Elaboratable, Module

from systars.config import Dataflow, SystolicConfig
from systars.errors import FirstError
from systars.isa.command import connect_to_top
from systars.isa.matmul import Matmul, MatmulCmd, MatmulError
from systars.memory.epilogue import PROFILES, Profile, epilogue_model, sfu_model
from systars.memory.mx import mx_row_bytes, mxint8_row_bytes_model
from systars.top import SystolicTop
from systars.trace.recorder import TileRecorder

from ..unit._axi import axi_memory
from ..unit._isa import from_bytes, to_unsigned_bytes
from ..unit._sim import run_sim
from .program import Program, generate, shrink_candidates

A_ADDR, B_ADDR, C_ADDR, D_ADDR = 0x1000, 0x8000, 0x10000, 0x18000
ALL = Dataflow.OUTPUT_STATIONARY | Dataflow.B_STATIONARY | Dataflow.A_STATIONARY


@dataclass
class Result:
    problems: list[str] = field(default_factory=list)
    cycles: int = 0
    values: str = ""  # a digest of C as the machine wrote it (the perf gate's, #85)

    @property
    def ok(self) -> bool:
        return not self.problems


class _OnTop(Elaboratable):
    def __init__(self, cfg, buffers):
        self.top = SystolicTop(cfg)
        self.isa = Matmul(cfg, buffers=buffers)

    def elaborate(self, _platform):
        m = Module()
        m.submodules.top = self.top
        m.submodules.isa = self.isa
        connect_to_top(m, self.isa, self.top)
        return m


def _config(prog: Program) -> SystolicConfig:
    return SystolicConfig(
        grid_rows=prog.dim, grid_cols=prog.dim, tile_rows=1, tile_cols=1,
        sp_capacity_kb=4, combiner_capacity_kb=4, dataflow=ALL,
        l2_bank_occupancy=prog.occupancy, l2_macro_rows=0,
        combiner_latency=prog.combiner_latency,
    )  # fmt: skip


def _expected(prog: Program, prof: Profile, raw: np.ndarray, dim: int, input_bits: int):
    if prof.mx:
        tiles_n = -(-prog.N // dim)
        rows = [[sfu_model(int(x), prof.sfu, prof) for x in row] for row in raw]
        return np.array(
            [
                [list(mxint8_row_bytes_model(r[j * dim : (j + 1) * dim])) for j in range(tiles_n)]
                for r in rows
            ]  # fmt: skip
        )
    return np.array([epilogue_model(list(map(int, row)), prof, input_bits) for row in raw])


def run(prog: Program, trace: str | Path | None = None, observe=None) -> Result:
    """Run ``prog`` and check it; with ``trace``, also record the tile's commands, errors
    and internal lanes there (a systars.trace bundle: ``python -m systars.trace`` draws it).
    ``observe(top)``, if given, returns one more simulator process (for tests)."""
    cfg = _config(prog)
    dim = prog.dim
    prof = prog.profile()
    rng = np.random.default_rng(prog.seed)
    M, N, K = prog.M, prog.N, prog.K
    lim = prog.operands
    bias_mag = 5000 if lim == 128 else 40
    A = rng.integers(-lim, lim, (M, K))
    B = rng.integers(-lim, lim, (K, N))
    D = {None: None, "matrix": rng.integers(-bias_mag, bias_mag, (M, N)),
         "row": rng.integers(-bias_mag, bias_mag, (1, N))}[prog.bias]  # fmt: skip
    C0 = rng.integers(-bias_mag, bias_mag, (M, N)) if prog.accumulate else None
    raw = A @ B + (0 if D is None else D) + (0 if C0 is None else C0)

    c_elem = 1 if prof.enable else 4
    mx_bytes, tiles_n = mx_row_bytes(dim), -(-N // dim)
    slack = prog.stride_slack
    a_stride, b_stride = K + slack, N + slack
    c_stride = (tiles_n * mx_bytes if prof.mx else c_elem * N) + slack * c_elem
    d_stride = 0 if prog.bias == "row" else N * 4 + slack * 4
    a_addr, b_addr = A_ADDR + prog.a_off, B_ADDR + prog.b_off
    c_addr, d_addr = C_ADDR + prog.c_off, D_ADDR + prog.d_off

    memory = bytearray(rng.integers(0, 256, 1 << 17, dtype=np.uint8).tobytes())
    for r in range(M):
        memory[a_addr + r * a_stride : a_addr + r * a_stride + K] = to_unsigned_bytes(A[r], 8)
    for r in range(K):
        memory[b_addr + r * b_stride : b_addr + r * b_stride + N] = to_unsigned_bytes(B[r], 8)
    if D is not None:
        for r in range(D.shape[0]):
            base = d_addr + r * d_stride
            memory[base : base + N * 4] = to_unsigned_bytes(D[r], 32)
    if C0 is not None:
        for r in range(M):
            memory[c_addr + r * c_stride : c_addr + r * c_stride + N * 4] = to_unsigned_bytes(
                C0[r], 32
            )
    before = bytes(memory)

    dut = _OnTop(cfg, prog.buffers)
    isa, top = dut.isa, dut.top
    status: dict = {}

    async def testbench(ctx):
        for n in range(PROFILES):  # the program's profile, and decoys in the other slots
            p = prof if n == prog.profile_slot else Program(seed=0, **prog.decoys[n]).profile()
            ctx.set(getattr(top, f"cmb_scale_{n}"), p.scale_word())
            ctx.set(getattr(top, f"cmb_control_{n}"), p.control_word())
            ctx.set(getattr(top, f"cmb_clip_{n}"), p.clip_word())
        options = int(prog.accumulate) | prog.profile_slot << 1
        cmds = [
            (MatmulCmd.CONFIG_DATAFLOW, prog.dataflow),
            (MatmulCmd.CONFIG_DIMS, M | N << 16 | K << 32),
            (MatmulCmd.CONFIG_A_ADDR, a_addr), (MatmulCmd.CONFIG_B_ADDR, b_addr),
            (MatmulCmd.CONFIG_C_ADDR, c_addr), (MatmulCmd.CONFIG_D_ADDR, d_addr if D is not None else 0),
            (MatmulCmd.CONFIG_A_STRIDE, a_stride), (MatmulCmd.CONFIG_B_STRIDE, b_stride),
            (MatmulCmd.CONFIG_C_STRIDE, c_stride), (MatmulCmd.CONFIG_D_STRIDE, d_stride),
            (MatmulCmd.CONFIG_OPTIONS, options), (MatmulCmd.START, 0),
        ]  # fmt: skip
        for cmd, data in cmds:
            ctx.set(isa.cfg_valid, 1)
            ctx.set(isa.cfg_cmd, cmd)
            ctx.set(isa.cfg_data, data)
            await ctx.tick()
        ctx.set(isa.cfg_valid, 0)
        clocks, done, error = 0, 0, 0
        async for _, _, done, error in ctx.tick().sample(isa.done, isa.error):
            clocks += 1
            if done or error:  # done is a one-clock pulse: keep the sampled values
                break
        status.update(
            done=done, error=error, code=ctx.get(isa.error_code),
            l1=ctx.get(top.l1_error), sync=ctx.get(top.sync_error), l2=ctx.get(top.l2_error),
            cycles=clocks, first=FirstError.decode(ctx.get(top.err_info), ctx.get(top.err_cycle)),
        )  # fmt: skip

    beat = cfg.blockmover_buswidth // 8
    proc = axi_memory(top, memory, beat, stall=prog.stall, seed=prog.seed)
    recorder = TileRecorder(top) if trace is not None else None
    result = Result()
    try:
        extra = ([recorder.process()] if recorder else []) + ([observe(top)] if observe else [])
        run_sim(dut, testbench, processes=[proc, *extra], max_cycles=400_000)
    except AssertionError as exc:
        result.problems.append(f"simulation: {exc}")
        return result
    finally:
        if recorder is not None:
            recorder.trace(program=prog.to_json()).save(trace)
    result.cycles = status.get("cycles", 0)

    if prog.violation:
        if not (status["error"] and status["code"] == MatmulError.UNSUPPORTED_LAYOUT):
            result.problems.append(f"{prog.violation}: not refused ({status})")
        for name in ("l1", "sync", "l2"):  # a refusal must not trip a monitor either
            if status[name]:
                result.problems.append(
                    f"{prog.violation}: refused, but {name} set: the first error was {status['first']}"
                )
        if bytes(memory) != before:
            result.problems.append(f"{prog.violation}: refused, but memory changed")
        return result

    for name in ("l1", "sync", "l2"):
        if status[name]:
            result.problems.append(f"{name} set: the first error was {status['first']}")
    if status["error"]:
        result.problems.append(f"the instruction failed ({status})")
    if not status["done"]:
        result.problems.append(f"not done ({status})")
    expected = _expected(prog, prof, raw, dim, cfg.input_bits)
    if prof.mx:
        got = np.array([list(memory[c_addr + r * c_stride : c_addr + r * c_stride + tiles_n * mx_bytes])
                        for r in range(M)]).reshape(M, tiles_n, mx_bytes)  # fmt: skip
        row_len = tiles_n * mx_bytes
    else:
        got = np.array([from_bytes(memory[c_addr + r * c_stride : c_addr + r * c_stride + N * c_elem],
                                   8 * c_elem, N) for r in range(M)])  # fmt: skip
        row_len = N * c_elem
    result.values = hashlib.sha256(json.dumps(got.tolist()).encode()).hexdigest()[:16]
    if got.shape != expected.shape or not np.array_equal(got, expected):
        bad = np.argwhere(got != expected) if got.shape == expected.shape else "shape"
        result.problems.append(
            f"C differs from the model at {bad if isinstance(bad, str) else bad[:4].tolist()}"
        )
    # Nothing outside C's rows may change
    touched = bytearray(before)
    for r in range(M):
        lo = c_addr + r * c_stride
        touched[lo : lo + row_len] = memory[lo : lo + row_len]
    if bytes(touched) != bytes(memory):
        result.problems.append("bytes outside C changed")
    return result


def shrink(prog: Program, budget: int = 60) -> tuple[Program, Result]:
    """The smallest program, by ``shrink_candidates``, that still fails."""
    best, best_result = prog, run(prog)
    tries, improved = 0, True
    while improved and tries < budget:
        improved = False
        for cand in shrink_candidates(best):
            tries += 1
            res = run(cand)
            if not res.ok:
                best, best_result, improved = cand, res, True
                break
            if tries >= budget:
                break
    return best, best_result


def repro(prog: Program) -> str:
    return f"python -m tests.fuzz.run '{json.dumps(prog.to_json())}'"


def main(argv: list[str]) -> int:
    """``python -m tests.fuzz.run '<program json>' | --seed N  [--trace BUNDLE]``"""
    parser = argparse.ArgumentParser(prog="python -m tests.fuzz.run")
    parser.add_argument("program", nargs="?", help="a program as JSON (a failure's replay)")
    parser.add_argument("--seed", type=int, help="the program a seed draws, instead")
    parser.add_argument("--trace", type=Path, help="record the run's trace bundle here")
    args = parser.parse_args(argv)
    if (args.program is None) == (args.seed is None):
        parser.error("give a program or --seed")
    prog = (
        generate(args.seed)
        if args.seed is not None
        else Program.from_json(json.loads(args.program))
    )
    if args.trace is not None:
        args.trace.parent.mkdir(parents=True, exist_ok=True)
    res = run(prog, trace=args.trace)
    print("ok" if res.ok else "\n".join(res.problems), f"({res.cycles} cycles)")
    if args.trace is not None:
        print(f"trace: {args.trace}")
    return 0 if res.ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
