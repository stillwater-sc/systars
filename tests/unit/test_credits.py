"""
Credits and tile-ready tokens (controller/sync.py, SystolicTop's command queues).

- ``SlotPool`` on its own: the FREE -> OWNED -> READY -> FREE cycle, the sticky
  protocol error, and the accumulator's hand-off (a loader fills a slot first:
  FREE -> OWNED -> LOADED, and the producer resumes it: LOADED -> OWNED).
- Matmul and Conv2d on a real ``SystolicTop``, with a monitor that replays the
  observable events (commands pushed, ``*_started``, ``*_completed``, and the
  Streamer's ``exec_result``) through an independent slot model, every cycle. The
  Streamer has several commands in flight: each returns its L2 credit on its
  completion (its last block is in L1) and emits its result token on its result,
  matched by command ID:
  - credit conservation: the slot state bits always match the model, so every slot
    is in exactly one state and credits plus occupied slots equal capacity;
  - no tile is consumed before its TILE_READY, and no credit is taken twice;
  - every slot is free at the end, and the protocol error never fires;
  - results equal numpy, and are identical with one buffer per operand (the
    processes serialized) and with two (double buffering);
  - with two buffers the BlockMover fills the next tile's slot while the Streamer
    works (measured cycles), and L3 backpressure does not change values;
  - with a bias or accumulation, each accumulator LOAD hands its slot to the
    Streamer, and no STORE starts on the hand-off token.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import pytest
from amaranth import Elaboratable, Module

from systars.config import SystolicConfig
from systars.controller.sync import SlotPool, SlotState, Sync
from systars.isa.command import CTRL_EXEC, CTRL_LOAD, CTRL_STORE, connect_to_top
from systars.isa.conv2d import Conv2d, Conv2dCmd
from systars.isa.matmul import Matmul, MatmulCmd
from systars.top import SystolicTop

from ._axi import axi_memory
from ._isa import from_bytes, to_unsigned_bytes
from ._sim import run_sim
from .test_conv2d import B_ADDR as CONV_B_ADDR
from .test_conv2d import F_ADDR, X_ADDR, Y_ADDR, reference
from .test_conv2d import setup as conv_setup

FREE, OWNED, READY, READING, LOADED = (
    SlotState.FREE, SlotState.OWNED, SlotState.READY, SlotState.READING, SlotState.LOADED,
)  # fmt: skip


# =============================================================================
# SlotPool
# =============================================================================


def test_slot_pool_cycle_and_errors():
    pool = SlotPool(2)

    async def testbench(ctx):
        def state():
            free, ready = ctx.get(pool.free), ctx.get(pool.ready)
            return [FREE if free >> s & 1 else READY if ready >> s & 1 else OWNED for s in (0, 1)]

        async def pulse(port, slot_port=None, slot=0):
            if slot_port is not None:
                ctx.set(slot_port, slot)
            ctx.set(port, 1)
            await ctx.tick()
            ctx.set(port, 0)

        assert state() == [FREE, FREE]
        ctx.set(pool.prod_slot, 1)
        assert ctx.get(pool.prod_credit)
        await pulse(pool.prod_take)
        assert state() == [FREE, OWNED] and not ctx.get(pool.prod_credit)
        ctx.set(pool.cons_slot, 1)
        assert not ctx.get(pool.cons_ready)
        await pulse(pool.prod_signal, pool.prod_signal_slot, 1)
        assert state() == [FREE, READY] and ctx.get(pool.cons_ready)
        await pulse(pool.cons_release, pool.cons_release_slot, 1)
        assert state() == [FREE, FREE] and not ctx.get(pool.error)
        # Releasing a slot that holds no tile is a protocol error (sticky)
        await pulse(pool.cons_release, pool.cons_release_slot, 0)
        assert state() == [FREE, FREE] and ctx.get(pool.error)
        await ctx.tick()
        assert ctx.get(pool.error)

    run_sim(pool, testbench)


def test_slot_pool_hand_off():
    """The loader fills a slot and hands it on (LOADED); the producer resumes it with
    RESUME instead of a free credit, and only then does the consumer see a token."""
    pool = SlotPool(2, loader=True)

    async def testbench(ctx):
        async def pulse(*ports):
            for port in ports:
                ctx.set(port, 1)
            await ctx.tick()
            for port in ports:
                ctx.set(port, 0)

        ctx.set(pool.load_slot, 1)
        ctx.set(pool.prod_slot, 1)
        ctx.set(pool.cons_slot, 1)
        assert ctx.get(pool.load_credit)
        await pulse(pool.load_take)
        assert not ctx.get(pool.load_credit) and ctx.get(pool.free) == 0b01
        ctx.set(pool.load_signal_slot, 1)
        await pulse(pool.load_signal)
        # LOADED: no free credit, no token for the consumer; a hand-off for RESUME
        assert ctx.get(pool.free) == 0b01 and ctx.get(pool.ready) == 0
        assert not ctx.get(pool.cons_ready)
        assert not ctx.get(pool.prod_credit)
        ctx.set(pool.prod_resume, 1)
        assert ctx.get(pool.prod_credit)
        await pulse(pool.prod_take)
        ctx.set(pool.prod_resume, 0)
        assert not ctx.get(pool.prod_credit)  # OWNED by the producer now
        ctx.set(pool.prod_signal_slot, 1)
        await pulse(pool.prod_signal)
        assert ctx.get(pool.cons_ready) and ctx.get(pool.ready) == 0b10
        ctx.set(pool.cons_release_slot, 1)
        await pulse(pool.cons_release)
        assert ctx.get(pool.free) == 0b11 and not ctx.get(pool.error)
        # RESUME on a slot that was not handed over is a protocol error
        ctx.set(pool.prod_resume, 1)
        assert not ctx.get(pool.prod_credit)
        await pulse(pool.prod_take)
        assert ctx.get(pool.error)

    run_sim(pool, testbench)


def test_slot_pool_rejects_out_of_range_tags():
    pool = SlotPool(3)

    async def testbench(ctx):
        ctx.set(pool.prod_slot, 3)
        ctx.set(pool.prod_acquire, 1)
        assert not ctx.get(pool.prod_credit)  # never dispatched: the queue would stall
        await ctx.tick()
        assert ctx.get(pool.error)

    run_sim(pool, testbench)


def test_sync_encoding():
    value = Sync.encode(src=3, dst=5, wait=True, signal=True)
    assert value == 3 | 5 << 4 | Sync.WAIT | Sync.SIGNAL
    assert Sync.encode(dst=1, acquire=True, resume=True) == 1 << 4 | Sync.ACQUIRE | Sync.RESUME
    with pytest.raises(ValueError):
        Sync.encode(dst=16)


# =============================================================================
# SystolicTop: an independent replay of the observable events
# =============================================================================

PROCESSES = {CTRL_LOAD: "load", CTRL_STORE: "store", CTRL_EXEC: "exec"}


class Monitor:
    """Replays pushed / started / completed events through a slot model."""

    def __init__(self, cfg: SystolicConfig):
        self.pools = {"l2": [FREE] * cfg.l2_slots, "acc": [FREE] * cfg.acc_slots}
        # Which pool each process fills (dst) and reads (src)
        self.produces = {CTRL_LOAD: "l2", CTRL_EXEC: "acc"}
        self.consumes = {CTRL_EXEC: "l2", CTRL_STORE: "acc"}
        self.queued = {t: deque() for t in PROCESSES}
        self.inflight = {t: {} for t in PROCESSES}  # command ID -> cmd_sync
        self.awaiting_result = {}  # Streamer: command ID -> cmd_sync, until its result
        self.cycles = 0
        self.overlap = 0  # cycles with the BlockMover's LOAD and the Streamer both running

    def signals(self, top):
        sigs = [top.cmd_valid, top.cmd_ready, top.cmd_type, top.cmd_sync, top.cmd_local_addr]
        for name in PROCESSES.values():
            sigs += [getattr(top, f"{name}_started"), getattr(top, f"{name}_started_id"),
                     getattr(top, f"{name}_completed"), getattr(top, f"{name}_completed_id")]  # fmt: skip
        sigs += [top.exec_result, top.exec_result_id]
        return [*sigs, top.l2_free, top.l2_ready, top.acc_free, top.acc_ready, top.sync_error,
                top.l1_error]  # fmt: skip

    def check_state(self, name, free, ready):
        for s, expected in enumerate(self.pools[name]):
            bits = (free >> s & 1, ready >> s & 1)
            states = {FREE: (1, 0), OWNED: (0, 0), READY: (0, 1), READING: (0, 0), LOADED: (0, 0)}
            assert bits == states[expected], (
                f"cycle {self.cycles}: {name} slot {s} is {bits} (free, ready), model {expected}"
            )

    def step(self, values):
        valid, ready, ctype, sync, local_addr, *rest = values
        events, (result, result_id), state = rest[:12], rest[12:14], rest[14:]
        l2_free, l2_ready, acc_free, acc_ready, error, l1_error = state
        assert not error, f"cycle {self.cycles}: protocol error"
        assert not l1_error, f"cycle {self.cycles}: L1 schedule or protocol error"
        # The state this cycle reflects every earlier cycle's events
        self.check_state("l2", l2_free, l2_ready)
        self.check_state("acc", acc_free, acc_ready)
        if valid and ready:  # a LOAD into the accumulator fills an accumulator slot
            self.queued[ctype].append((sync, ctype == CTRL_LOAD and bool(local_addr >> 31)))

        moves = []
        for i, t in enumerate(PROCESSES):
            started, started_id, completed, completed_id = events[4 * i : 4 * i + 4]
            if started:
                s, to_acc = self.queued[t].popleft()
                src, dst = s & 0xF, s >> Sync.DST & 0xF
                if s & Sync.WAIT:
                    pool = self.pools[self.consumes[t]]
                    assert pool[src] == READY, (
                        f"cycle {self.cycles}: {PROCESSES[t]} started before slot {src}'s token"
                    )
                    if s & Sync.RELEASE:  # the consumer takes the token
                        moves.append((self.consumes[t], src, READY, READING))
                if s & Sync.ACQUIRE:  # with RESUME, the hand-off token
                    pool = "acc" if to_acc else self.produces[t]
                    moves.append((pool, dst, LOADED if s & Sync.RESUME else FREE, OWNED))
                assert started_id not in self.inflight[t], f"cycle {self.cycles}: ID reused"
                self.inflight[t][started_id] = (s, to_acc)
            if completed:
                s, to_acc = self.inflight[t].pop(completed_id, (None, False))
                assert s is not None, f"cycle {self.cycles}: {PROCESSES[t]} completed unstarted"
                src, dst = s & 0xF, s >> Sync.DST & 0xF
                if s & Sync.RELEASE:
                    moves.append((self.consumes[t], src, READING if s & Sync.WAIT else READY, FREE))
                if t == CTRL_EXEC:
                    self.awaiting_result[completed_id] = s
                elif s & Sync.SIGNAL and to_acc:  # hands the tile to the Streamer
                    moves.append(("acc", dst, OWNED, LOADED))
                elif s & Sync.SIGNAL:
                    moves.append((self.produces[t], dst, OWNED, READY))
        if result:
            s = self.awaiting_result.pop(result_id, None)
            assert s is not None, f"cycle {self.cycles}: a result for no command"
            if s & Sync.SIGNAL:
                moves.append(("acc", s >> Sync.DST & 0xF, OWNED, READY))
        if self.inflight[CTRL_LOAD] and self.inflight[CTRL_EXEC]:
            self.overlap += 1
        for pool, slot, frm, to in moves:
            assert self.pools[pool][slot] == frm, (
                f"cycle {self.cycles}: {pool} slot {slot} is {self.pools[pool][slot]}, not {frm}"
            )
            self.pools[pool][slot] = to
        self.cycles += 1

    def assert_drained(self):
        assert all(s == FREE for pool in self.pools.values() for s in pool), self.pools
        assert all(not q for q in self.queued.values())


class OnTop(Elaboratable):
    def __init__(self, cfg, isa):
        self.top = SystolicTop(cfg)
        self.isa = isa

    def elaborate(self, _platform):
        m = Module()
        m.submodules.top = self.top
        m.submodules.isa = self.isa
        connect_to_top(m, self.isa, self.top)
        return m


def run_on_top(cfg, isa, config_cmds, start, memory, *, stall=0.0, max_cycles=400_000):
    dut = OnTop(cfg, isa)
    monitor = Monitor(cfg)
    status = {}

    async def watch(ctx):
        while True:
            values = await ctx.tick().sample(*monitor.signals(dut.top))
            monitor.step(values[2:])

    async def testbench(ctx):
        for cmd, data in [*config_cmds, (start, 0)]:
            ctx.set(isa.cfg_valid, 1)
            ctx.set(isa.cfg_cmd, cmd)
            ctx.set(isa.cfg_data, data)
            await ctx.tick()
        ctx.set(isa.cfg_valid, 0)
        done, error = await ctx.tick().sample(isa.done, isa.error).until(isa.done | isa.error)
        status.update(done=done, error=error)

    beat = cfg.dma_buswidth // 8
    memory_process = axi_memory(dut.top, memory, beat, stall=stall, seed=len(memory))
    run_sim(dut, testbench, processes=[memory_process, watch], max_cycles=max_cycles)
    assert status == {"done": 1, "error": 0}
    monitor.assert_drained()
    return monitor


def config(grid=4):
    return SystolicConfig(
        grid_rows=grid, grid_cols=grid, sp_capacity_kb=4, acc_capacity_kb=4
    )  # 2 A/B slot pairs and 2 accumulator slots by default


A_ADDR, B_ADDR, C_ADDR, D_ADDR = 0x1001, 0x4007, 0x8003, 0xC005


def matmul_on_top(M, N, K, *, buffers=2, stall=0.0, seed=0, start=None):
    """``start``: None, "bias" (a dense D matrix) or "accumulate" (onto C)."""
    cfg = config()
    rng = np.random.default_rng(seed)
    A = rng.integers(-128, 128, (M, K))
    B = rng.integers(-128, 128, (K, N))
    D = rng.integers(-(1 << 20), 1 << 20, (M, N)) if start else np.zeros((M, N), dtype=int)
    memory = bytearray(rng.integers(0, 256, 1 << 16, dtype=np.uint8).tobytes())
    memory[A_ADDR : A_ADDR + M * K] = to_unsigned_bytes(A, 8)
    memory[B_ADDR : B_ADDR + K * N] = to_unsigned_bytes(B, 8)
    extra = []
    if start == "bias":
        memory[D_ADDR : D_ADDR + M * N * 4] = to_unsigned_bytes(D, 32)
        extra = [(MatmulCmd.CONFIG_D_ADDR, D_ADDR), (MatmulCmd.CONFIG_D_STRIDE, 4 * N)]
    elif start == "accumulate":
        memory[C_ADDR : C_ADDR + M * N * 4] = to_unsigned_bytes(D, 32)
        extra = [(MatmulCmd.CONFIG_OPTIONS, 1)]
    cmds = extra + [
        (MatmulCmd.CONFIG_DIMS, M | N << 16 | K << 32),
        (MatmulCmd.CONFIG_A_ADDR, A_ADDR),
        (MatmulCmd.CONFIG_B_ADDR, B_ADDR),
        (MatmulCmd.CONFIG_C_ADDR, C_ADDR),
        (MatmulCmd.CONFIG_A_STRIDE, K),
        (MatmulCmd.CONFIG_B_STRIDE, N),
        (MatmulCmd.CONFIG_C_STRIDE, 4 * N),
    ]
    monitor = run_on_top(
        cfg, Matmul(cfg, buffers=buffers), cmds, MatmulCmd.START, memory, stall=stall
    )
    C = np.array(from_bytes(memory[C_ADDR : C_ADDR + M * N * 4], 32, M * N)).reshape(M, N)
    np.testing.assert_array_equal(C, A @ B + D)
    return C, monitor


class TestMatmul:
    SHAPE = (8, 8, 16)  # 2 x 2 output tiles, 4 K tiles each, on a 4 x 4 array

    def test_double_buffering_overlaps_load_and_compute(self):
        C1, one = matmul_on_top(*self.SHAPE, buffers=1)
        C2, two = matmul_on_top(*self.SHAPE, buffers=2)
        np.testing.assert_array_equal(C1, C2)  # bit-identical to the serialized run
        # With one slot per operand a LOAD into the A slot waits for the COMPUTE
        # that reads it; with two, the next K tile's tiles load while the Streamer
        # still has the current ones in flight
        assert two.overlap > one.overlap == 0
        # Measured (OS, the default): 907 -> 744 cycles. The Streamer returns a
        # tile's L2 credit once its last block is in L1, a few clocks after it
        # starts, so even one buffer overlaps most of the next LOAD; the run is
        # bound by the BlockMover's L3 traffic (WS and IS measure the same)
        assert two.cycles < 0.85 * one.cycles, (one.cycles, two.cycles)

    @pytest.mark.parametrize("buffers", [1, 2])
    def test_l3_backpressure_does_not_change_values(self, buffers):
        C, fast = matmul_on_top(5, 9, 7, buffers=buffers, seed=1)
        C_slow, slow = matmul_on_top(5, 9, 7, buffers=buffers, seed=1, stall=0.4)
        np.testing.assert_array_equal(C, C_slow)
        assert slow.cycles > fast.cycles

    @pytest.mark.parametrize("start", ["bias", "accumulate"])
    @pytest.mark.parametrize("buffers", [1, 2])
    def test_accumulator_loads_hand_off_to_the_streamer(self, start, buffers):
        """Each C tile's accumulator LOAD takes its slot's credit and hands it to the
        Streamer; the monitor checks every slot's state each cycle, and that no STORE
        starts before the result token (a STORE on the hand-off token would store D)."""
        matmul_on_top(*self.SHAPE, buffers=buffers, seed=3, start=start, stall=0.2)


@pytest.mark.parametrize("buffers", [1, 2])
def test_conv2d_credits_on_top(buffers):
    """Conv2d's X tiles are filled by one LOAD (or CLEAR) per position: the credit is
    taken by the first, the token emitted by the last, and the Streamer waits for it.
    Each output tile starts from the bias row, whose accumulator LOAD hands the slot to
    the Streamer."""
    cfg = config()
    X_shape, F_shape, stride, pad = (1, 4, 4, 3), (3, 3, 3, 5), (1, 1), (1, 1)
    rng = np.random.default_rng(7)
    X = rng.integers(-128, 128, X_shape)
    F = rng.integers(-128, 128, F_shape)
    bias = rng.integers(-(1 << 20), 1 << 20, F_shape[3])
    expected = reference(X, F, stride, pad, (1, 1)) + bias  # with a bias row
    memory = bytearray(rng.integers(0, 256, 1 << 20, dtype=np.uint8).tobytes())
    memory[X_ADDR : X_ADDR + X.size] = to_unsigned_bytes(X, 8)
    memory[F_ADDR : F_ADDR + F.size] = to_unsigned_bytes(F, 8)
    memory[CONV_B_ADDR : CONV_B_ADDR + bias.size * 4] = to_unsigned_bytes(bias, 32)
    run_on_top(
        cfg,
        Conv2d(cfg, buffers=buffers),
        conv_setup(X_shape, F_shape, stride, pad, bias=CONV_B_ADDR),
        Conv2dCmd.START,
        memory,
        stall=0.2,
        max_cycles=600_000,
    )
    Y = np.array(from_bytes(memory[Y_ADDR : Y_ADDR + expected.size * 4], 32, expected.size))
    np.testing.assert_array_equal(Y.reshape(expected.shape), expected)
