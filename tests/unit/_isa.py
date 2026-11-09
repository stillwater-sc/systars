"""
Test helpers for the ISA instructions (Matmul, Conv2d).

- ``run_isa`` configures an instruction, starts it, and records every command it
  issues, standing in for ``SystolicTop``: commands are accepted as pushed (no
  round trip), and each STORE reports completion a few cycles later.
- ``TopModel`` executes those commands the way ``SystolicTop`` does (LOAD,
  CONFIG, PRELOAD, COMPUTE[_ACCUMULATE], STORE, in OS and the stationary mode),
  over a byte-addressed L3, so a command stream can be checked numerically against
  numpy. LOAD and STORE move ``len`` rows ``stride`` bytes apart (0 = one beat),
  ``row_bytes`` per row (0 = a whole row): a loaded row is zero-filled past them, a
  stored row writes only them. TRANSPOSE turns the tile's rows into its columns.
  It also checks the commands' credits and tile-ready tokens (``cmd_sync``): in
  program order, a well-formed stream only waits on tokens that are there and takes
  credits that are free, and every slot is free again at the end.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from systars.controller.sync import SlotState, Sync
from systars.isa.command import CTRL_EXEC, CTRL_LOAD, CTRL_STORE
from systars.util.commands import DmaOpcode, ExecOpcode

from ._sim import run_sim

FIELDS = (
    "dram_addr", "local_addr", "len", "stride", "row_bytes", "rs1", "rs2", "rd", "k_dim",
    "activation", "sync", "chain",
)  # fmt: skip
DATA_MASK = (1 << 29) - 1  # LocalAddr data bits (the flags are ignored by the memories)


@dataclass
class Command:
    type: int
    opcode: int
    id: int
    dram_addr: int
    local_addr: int
    len: int
    stride: int
    row_bytes: int
    rs1: int
    rs2: int
    rd: int
    k_dim: int
    activation: int
    sync: int = 0
    chain: int = 0


@dataclass
class IsaResult:
    commands: list[Command]
    done: bool
    error: bool
    error_code: int


def run_isa(isa, config_cmds, start_cmd, *, latency=2, max_cycles=200_000, ready_every=1):
    """Configure ``isa`` with ``(cmd, data)`` pairs, START it, and record its commands.

    ``ready_every`` > 1 holds ``cmd_ready`` low on the other cycles (backpressure).
    Each accepted STORE pulses ``cmd_store_completed`` ``latency`` cycles later.
    """
    log: list[Command] = []
    status = {}

    async def controller(ctx):
        cycle = 0
        stores = []  # cycles until each accepted STORE completes
        while True:
            ready = cycle % ready_every == 0
            ctx.set(isa.cmd_ready, ready)
            ctx.set(isa.cmd_store_completed, bool(stores) and stores[0] == 0)
            sampled = await ctx.tick().sample(
                isa.cmd_valid, isa.cmd_type, isa.cmd_opcode, isa.cmd_id,
                *[getattr(isa, f"cmd_{f}") for f in FIELDS],
            )  # fmt: skip
            valid, ctype, opcode, cid, *fields = sampled[2:]
            if stores and stores[0] == 0:
                stores.pop(0)
            stores = [n - 1 for n in stores]
            if valid and ready:
                log.append(Command(ctype, opcode, cid, *fields))
                if ctype == CTRL_STORE:
                    stores.append(latency)
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
        self.held = np.zeros((self.dim, self.dim), dtype=np.int64)  # OS: C in the PEs
        self.dataflow = 0  # 0 = OS, 1 = stationary (WS and IS)
        self.l2_slots = [SlotState.FREE] * cfg.l2_slots
        self.acc_slots = [SlotState.FREE] * cfg.acc_slots

    def write(self, addr, data):
        self.dram[addr : addr + len(data)] = data

    def read_elems(self, addr, bits, n):
        return from_bytes(self.dram[addr : addr + n * bits // 8], bits, n)

    def _sync(self, c, phase):
        """Apply ``c``'s credits and tokens: ``phase`` "start" (WAIT, ACQUIRE) or "end"."""
        src, dst = c.sync & 0xF, (c.sync >> Sync.DST) & 0xF
        produces = {CTRL_LOAD: self.l2_slots, CTRL_EXEC: self.acc_slots}.get(c.type)
        if c.type == CTRL_LOAD and c.local_addr >> 31 & 1:  # a LOAD into the accumulator
            produces = self.acc_slots
        consumes = {CTRL_EXEC: self.l2_slots, CTRL_STORE: self.acc_slots}.get(c.type)

        def move(pool, slot, frm, to, what):
            assert pool is not None and slot < len(pool), f"{what}: slot {slot} of {c}"
            assert pool[slot] == frm, f"{what}: slot {slot} is {pool[slot]}, not {frm}: {c}"
            pool[slot] = to

        taken = SlotState.READING if c.sync & Sync.RELEASE else SlotState.READY
        if phase == "start":
            if c.sync & Sync.WAIT:  # with RELEASE, the consumer takes the token
                move(consumes, src, SlotState.READY, taken, "WAIT")
            if c.sync & Sync.ACQUIRE:
                move(produces, dst, SlotState.FREE, SlotState.OWNED, "ACQUIRE")
        else:
            if c.sync & Sync.RELEASE:
                held = SlotState.READING if c.sync & Sync.WAIT else SlotState.READY
                move(consumes, src, held, SlotState.FREE, "RELEASE")
            if c.sync & Sync.SIGNAL:
                move(produces, dst, SlotState.OWNED, SlotState.READY, "SIGNAL")

    def assert_all_free(self):
        assert all(s == SlotState.FREE for s in self.l2_slots + self.acc_slots), (
            f"slots left in use: L2 {self.l2_slots}, accumulator {self.acc_slots}"
        )

    def run(self, commands):
        for c in commands:
            self._sync(c, "start")
            self._run_one(c)
            self._sync(c, "end")

    def _run_one(self, c):
        cfg, dim = self.cfg, self.dim
        stride = c.stride or self.beat
        if c.type == CTRL_LOAD:
            to_acc, add = c.local_addr >> 31 & 1, c.local_addr >> 30 & 1
            bits = cfg.acc_bits if to_acc else cfg.input_bits
            row_bytes = c.row_bytes or dim * bits // 8
            n_elems = min(dim, row_bytes * 8 // bits)
            base = c.local_addr & DATA_MASK
            turn = c.opcode in (DmaOpcode.TRANSPOSE, DmaOpcode.BROADCAST_TRANSPOSE)
            n_rows = min(c.len, dim) if turn else c.len
            if c.opcode == DmaOpcode.CLEAR:
                rows = [[0] * dim for _ in range(n_rows)]
            elif c.opcode in (DmaOpcode.BROADCAST, DmaOpcode.BROADCAST_TRANSPOSE):
                rows = [self.read_elems(c.dram_addr, bits, n_elems)] * n_rows
            else:
                rows = [
                    self.read_elems(c.dram_addr + n * stride, bits, n_elems) for n in range(n_rows)
                ]
            if turn:  # L3 row n -> element n of every local row
                rows = [[row[j] for row in rows] for j in range(n_elems)]
            for n, row in enumerate(rows):
                row = row + [0] * (dim - len(row))
                if not to_acc:
                    self.sp[base + n] = row
                elif add:
                    old = self.acc.get(base + n, [0] * dim)
                    self.acc[base + n] = [a + b for a, b in zip(old, row, strict=True)]
                else:
                    self.acc[base + n] = row
        elif c.type == CTRL_STORE:
            row_bytes = c.row_bytes or dim * cfg.acc_bits // 8
            base = c.local_addr & DATA_MASK
            if c.opcode == DmaOpcode.TRANSPOSE:  # element n of each gathered row -> row n
                gather = min(dim, row_bytes * 8 // cfg.acc_bits)
                tile = [self.acc.get(base + g, [0] * dim) for g in range(gather)]
                rows = [[tile[g][n] for g in range(gather)] for n in range(min(c.len, dim))]
            else:
                rows = [self.acc.get(base + n, [0] * dim) for n in range(c.len)]
            for n, row in enumerate(rows):
                data = to_unsigned_bytes(row, cfg.acc_bits)[:row_bytes]
                self.write(c.dram_addr + n * stride, data)
        elif c.type == CTRL_EXEC:
            self.execute(c)
        else:
            raise AssertionError(f"unknown command type {c.type}")

    def execute(self, c):
        dim = self.dim
        sp = self.sp
        accumulate = c.opcode == ExecOpcode.COMPUTE_ACCUMULATE

        def row(addr):
            # Rows past an edge tile were never loaded: they feed only results no
            # STORE writes, so zeros stand in for whatever L2 holds there
            return sp.get(addr & DATA_MASK, [0] * dim)

        def write(key, out):
            if accumulate:
                out = out + np.array(self.acc.get(key, [0] * dim))
            self.acc[key] = [int(v) for v in out]

        if c.opcode == ExecOpcode.CONFIG:
            self.dataflow = int(c.rs1 != 0)
        elif c.opcode == ExecOpcode.PRELOAD:
            if self.dataflow:  # the stationary tile, row-major; zero rows below k_dim
                k = min(c.k_dim, dim)
                self.weights[:] = 0
                for r in range(k):
                    self.weights[r] = row(c.rs1 + r)
        elif c.opcode in (ExecOpcode.COMPUTE, ExecOpcode.COMPUTE_ACCUMULATE):
            k = min(c.k_dim, dim)
            if k == 0:
                return
            if self.dataflow == 0:  # OS: A tile row-major, B tile column-major
                A = np.array([row(c.rs1 + i)[:k] for i in range(dim)])
                Bt = np.array([row(c.rs2 + j)[:k] for j in range(dim)])
                part = A.astype(np.int64) @ Bt.T.astype(np.int64)
                self.held = part if not c.chain & 1 else self.held + part
                if not c.chain & 2:
                    for i in range(dim):
                        write((c.rd & DATA_MASK) + i, self.held[i])
            else:  # stationary: the streamed tile column-major (row r = column r)
                X = np.array([row(c.rs1 + r)[:k] for r in range(dim)]).T
                out = X.astype(np.int64) @ self.weights
                for mrow in range(k):
                    write((c.rd & DATA_MASK) + mrow, out[mrow])
        else:
            raise AssertionError(f"unexpected exec opcode {c.opcode:#x}")
