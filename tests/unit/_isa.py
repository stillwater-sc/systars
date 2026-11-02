"""
Test helpers for the ISA instructions (Matmul, Conv2d).

- ``run_isa`` configures an instruction, starts it, and records every command it
  issues, standing in for ``SystolicTop`` (each command is accepted and
  completed after a few cycles).
- ``TopModel`` executes those commands the way ``SystolicTop`` does (LOAD,
  CONFIG, PRELOAD, COMPUTE[_ACCUMULATE], STORE), over a byte-addressed DRAM, so a
  command stream can be checked numerically against numpy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from systars.isa.command import CTRL_EXEC, CTRL_LOAD, CTRL_STORE
from systars.util.commands import ExecOpcode

from ._sim import run_sim

FIELDS = ("dram_addr", "local_addr", "len", "rs1", "rs2", "rd", "k_dim", "activation")
DATA_MASK = (1 << 29) - 1  # LocalAddr data bits (the flags are ignored by the memories)


@dataclass
class Command:
    type: int
    opcode: int
    id: int
    dram_addr: int
    local_addr: int
    len: int
    rs1: int
    rs2: int
    rd: int
    k_dim: int
    activation: int


@dataclass
class IsaResult:
    commands: list[Command]
    done: bool
    error: bool
    error_code: int


def run_isa(isa, config_cmds, start_cmd, *, latency=2, max_cycles=200_000, ready_every=1):
    """Configure ``isa`` with ``(cmd, data)`` pairs, START it, and record its commands.

    ``ready_every`` > 1 holds ``cmd_ready`` low on the other cycles (backpressure).
    """
    log: list[Command] = []
    status = {}

    async def controller(ctx):
        cycle = 0
        pending = None  # cycles until the accepted command completes
        while True:
            ready = cycle % ready_every == 0 and pending is None
            ctx.set(isa.cmd_ready, ready)
            ctx.set(isa.cmd_completed, pending == 0)
            sampled = await ctx.tick().sample(
                isa.cmd_valid, isa.cmd_type, isa.cmd_opcode, isa.cmd_id,
                *[getattr(isa, f"cmd_{f}") for f in FIELDS],
            )  # fmt: skip
            valid, ctype, opcode, cid, *fields = sampled[2:]
            if pending is not None:
                pending = None if pending == 0 else pending - 1
            if valid and ready:
                log.append(Command(ctype, opcode, cid, *fields))
                pending = latency
            cycle += 1

    async def testbench(ctx):
        for cmd, data in [*config_cmds, (start_cmd, 0)]:
            ctx.set(isa.cfg_valid, 1)
            ctx.set(isa.cfg_cmd, cmd)
            ctx.set(isa.cfg_data, data)
            await ctx.tick().until(isa.cfg_ready)
        ctx.set(isa.cfg_valid, 0)
        while True:
            _, _, done, error, code = await ctx.tick().sample(isa.done, isa.error, isa.error_code)
            if done or error:
                status.update(done=bool(done), error=bool(error), error_code=code)
                return

    run_sim(isa, testbench, processes=[controller], max_cycles=max_cycles)
    return IsaResult(log, **status)


def to_unsigned_bytes(values, bits):
    values = np.asarray(values, dtype=np.int64).ravel()
    out = bytearray()
    for v in values:
        out += (int(v) & ((1 << bits) - 1)).to_bytes(bits // 8, "little")
    return bytes(out)


def from_bytes(data, bits, n):
    return [
        int.from_bytes(data[i * bits // 8 : (i + 1) * bits // 8], "little", signed=True)
        for i in range(n)
    ]


class TopModel:
    """SystolicTop's command semantics over a byte-addressed DRAM (dim x dim array)."""

    def __init__(self, cfg, dram_bytes=1 << 20):
        self.cfg = cfg
        self.dim = cfg.grid_rows * cfg.tile_rows
        self.beat = cfg.dma_buswidth // 8
        self.dram = bytearray(dram_bytes)
        self.sp: dict[int, list[int]] = {}
        self.acc: dict[int, list[int]] = {}
        self.weights = np.zeros((self.dim, self.dim), dtype=np.int64)
        self.dataflow = 0

    def write(self, addr, data):
        self.dram[addr : addr + len(data)] = data

    def read_elems(self, addr, bits, n):
        return from_bytes(self.dram[addr : addr + n * bits // 8], bits, n)

    def run(self, commands):
        cfg, dim = self.cfg, self.dim
        for c in commands:
            if c.type == CTRL_LOAD:
                for n in range(c.len):
                    row = self.read_elems(c.dram_addr + n * self.beat, cfg.input_bits, dim)
                    self.sp[(c.local_addr & DATA_MASK) + n] = row
            elif c.type == CTRL_STORE:
                for n in range(c.len):
                    row = self.acc.get((c.local_addr & DATA_MASK) + n, [0] * dim)
                    self.write(c.dram_addr + n * self.beat, to_unsigned_bytes(row, cfg.acc_bits))
            elif c.type == CTRL_EXEC:
                self.execute(c)
            else:
                raise AssertionError(f"unknown command type {c.type}")

    def execute(self, c):
        dim = self.dim
        if c.opcode == ExecOpcode.CONFIG:
            self.dataflow = c.rs1 & 1
        elif c.opcode == ExecOpcode.PRELOAD:
            assert self.dataflow == 1, "PRELOAD in OS does nothing; Matmul/Conv2d use WS"
            k = min(c.k_dim, dim)
            self.weights[:] = 0
            for r in range(k):
                self.weights[r] = self.sp[(c.rs1 & DATA_MASK) + r]
        elif c.opcode in (ExecOpcode.COMPUTE, ExecOpcode.COMPUTE_ACCUMULATE):
            assert self.dataflow == 1
            accumulate = c.opcode == ExecOpcode.COMPUTE_ACCUMULATE
            for mrow in range(c.k_dim):
                a = np.array(self.sp[(c.rs1 & DATA_MASK) + mrow], dtype=np.int64)
                out = a @ self.weights
                key = (c.rd & DATA_MASK) + mrow
                if accumulate:
                    out = out + np.array(self.acc.get(key, [0] * dim))
                self.acc[key] = [int(v) for v in out]
        else:
            raise AssertionError(f"unexpected exec opcode {c.opcode:#x}")
