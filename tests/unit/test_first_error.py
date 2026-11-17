"""
The first-error capture (#84): each monitor names its error's kind, and ``SystolicTop``
latches the first error event with its source, kind, command and cycle, re-armed on a
write. The runtime's decoded exception is in test_runtime_trace.py (an AXI error); the L2,
L1 and buffer kinds are also checked in test_scratchpad.py and test_streamer.py.
"""

from __future__ import annotations

import pytest

from systars.config import SystolicConfig
from systars.controller.sync import SlotPool, Sync
from systars.errors import (
    ErrorSource,
    FirstError,
    L1Error,
    L2Error,
    SlotError,
    SlotPoolId,
    axi_kind,
    kind_name,
    pack,
    sync_kind,
)
from systars.top import SystolicTop
from systars.util.commands import ExecOpcode

from ._sim import run_sim

E = SlotError


async def _pulse(ctx, *settings):
    """Hold ``(signal, value)`` pairs for one clock, then drop the strobes."""
    for sig, value in settings:
        ctx.set(sig, value)
    await ctx.tick()
    for sig, _ in settings:
        if len(sig) == 1:
            ctx.set(sig, 0)


def _take(p):
    return [(p.prod_slot, 0), (p.prod_acquire, 1), (p.prod_take, 1)]


def _signal(p):
    return [(p.prod_signal, 1), (p.prod_signal_slot, 0)]


# Each error, provoked on a fresh pool of two slots (two producers for the fill cases):
# the clocks before it, then the clock that breaks the protocol
SLOT_CASES = {
    E.TAKE_NOT_FREE: (1, lambda p: [_take(p)], lambda p: _take(p)),
    E.SIGNAL_NOT_OWNED: (1, lambda _p: [], _signal),
    E.FILL_TAKE_NOT_FREE: (
        2, lambda p: [[(p.fill_slot, 0), (p.fill_take, 1)]],
        lambda p: [(p.fill_slot, 0), (p.fill_take, 1)],
    ),
    E.FILL_SIGNAL_NOT_OWNED: (2, lambda _p: [], lambda p: [(p.fill_signal, 1)]),
    E.DOUBLE_SIGNAL: (
        2, lambda p: [_take(p)],
        lambda p: [*_signal(p), (p.fill_signal, 1), (p.fill_signal_slot, 0)],
    ),
    E.WAIT_NOT_READY: (1, lambda _p: [], lambda p: [(p.cons_slot, 0), (p.cons_take, 1)]),
    E.RELEASE_NOT_HELD: (1, lambda _p: [], lambda p: [(p.cons_release, 1)]),
    E.PROD_NO_SUCH_SLOT: (1, lambda _p: [], lambda p: [(p.prod_slot, 3), (p.prod_acquire, 1)]),
    E.FILL_NO_SUCH_SLOT: (2, lambda _p: [], lambda p: [(p.fill_slot, 3), (p.fill_acquire, 1)]),
    E.CONS_NO_SUCH_SLOT: (1, lambda _p: [], lambda p: [(p.cons_slot, 3), (p.cons_wait, 1)]),
    E.TAKE_AND_RELEASE: (
        1, lambda p: [_take(p), _signal(p)],
        lambda p: [(p.cons_slot, 0), (p.cons_take, 1), (p.cons_release, 1),
                   (p.cons_release_slot, 0)],
    ),
}  # fmt: skip


def test_every_slot_error_kind_is_provoked():
    assert set(SLOT_CASES) == set(SlotError)


# The kinds a completion raises (a signal or a release), so the tile names the command that
# completed; the others come from a start or a queue-head check
AT_COMPLETION = {E.SIGNAL_NOT_OWNED, E.FILL_SIGNAL_NOT_OWNED, E.DOUBLE_SIGNAL, E.RELEASE_NOT_HELD}


@pytest.mark.parametrize("kind", list(SLOT_CASES), ids=lambda k: k.name)
def test_the_slot_pool_names_the_protocol_error(kind):
    producers, before, breaking = SLOT_CASES[kind]
    pool = SlotPool(2, producers=producers)

    async def tb(ctx):
        for step in before(pool):
            await _pulse(ctx, *step)
            assert not ctx.get(pool.error), f"an error before the {kind.name} step"
        await _pulse(ctx, *breaking(pool))
        await ctx.tick()
        assert ctx.get(pool.error)
        assert ctx.get(pool.error_kind) == kind
        assert ctx.get(pool.error_at_completion) == (kind in AT_COMPLETION)

    run_sim(pool, tb)


# =============================================================================
# The capture on SystolicTop
# =============================================================================


def _config() -> SystolicConfig:
    return SystolicConfig(grid_rows=2, grid_cols=2, tile_rows=1, tile_cols=1)


async def _issue(ctx, top, **fields):
    for name, value in fields.items():
        ctx.set(getattr(top, f"cmd_{name}"), value)
    ctx.set(top.cmd_valid, 1)
    await ctx.tick().until(top.cmd_ready)
    ctx.set(top.cmd_valid, 0)


async def _until_error(ctx, top, bit, limit=200):
    """Clock until ``bit`` reads 1; return ``top.cycle`` in that clock (what the capture
    latches)."""
    for _ in range(limit):
        if ctx.get(bit):
            return ctx.get(top.cycle)
        await ctx.tick()
    raise AssertionError(f"{bit} never rose")


def test_an_l2_conflict_is_captured_with_its_command_and_cycle():
    """An OS COMPUTE with A and B in L2's A group: the first error is L2's
    READS_SAME_GROUP, the Streamer's command, in the clock ``l2_error`` rose."""
    cfg = _config()
    top = SystolicTop(cfg)
    rows = cfg.grid_rows
    out = {}

    async def tb(ctx):
        assert ctx.get(top.err_info) == 0
        await _issue(ctx, top, type=SystolicTop.CTRL_EXEC, opcode=ExecOpcode.COMPUTE, rs1=0,
                     rs2=2 * rows, rd=0, k_dim=rows, id=0x5A)  # fmt: skip
        out["rose"] = await _until_error(ctx, top, top.l2_error)
        await ctx.tick()
        out["info"], out["at"] = ctx.get(top.err_info), ctx.get(top.err_cycle)
        # Re-arming forgets it; the bit is still set, so nothing new is captured
        ctx.set(top.err_rearm, 1)
        await ctx.tick()
        ctx.set(top.err_rearm, 0)
        await ctx.tick().repeat(4 * rows)
        out["after"] = ctx.get(top.err_info), ctx.get(top.l2_error)

    run_sim(top, tb, max_cycles=2_000)
    assert out["info"] == pack(ErrorSource.L2, L2Error.READS_SAME_GROUP,
                               cmd_type=SystolicTop.CTRL_EXEC, cmd_id=0x5A)  # fmt: skip
    assert out["at"] == out["rose"]
    assert out["after"] == (0, 1)
    first = FirstError.decode(out["info"], out["at"])
    assert str(first) == f"L2 READS_SAME_GROUP at cycle {out['rose']}, EXECUTE command 90"


def test_a_slot_protocol_error_names_its_pool_and_command():
    """A COMPUTE that RELEASEs an L2 slot it never waited for: SYNC, the L2 pool's
    consumer (the Streamer) broke the protocol, with that command."""
    cfg = _config()
    top = SystolicTop(cfg)
    rows = cfg.grid_rows
    g = cfg.l2_group_rows
    out = {}

    async def tb(ctx):
        await _issue(ctx, top, type=SystolicTop.CTRL_EXEC, opcode=ExecOpcode.COMPUTE, rs1=0,
                     rs2=g, rd=0, k_dim=rows, id=7, sync=Sync.encode(src=1, release=True))  # fmt: skip
        out["rose"] = await _until_error(ctx, top, top.sync_error, limit=400)
        await ctx.tick()
        out["info"], out["at"] = ctx.get(top.err_info), ctx.get(top.err_cycle)
        out["l2"] = ctx.get(top.l2_error)

    run_sim(top, tb, max_cycles=2_000)
    assert out["l2"] == 0
    kind = sync_kind(SlotPoolId.L2, SlotError.RELEASE_NOT_HELD)
    assert out["info"] == pack(ErrorSource.SYNC, kind, cmd_type=SystolicTop.CTRL_EXEC, cmd_id=7)
    assert out["at"] == out["rose"]
    assert kind_name(ErrorSource.SYNC, kind) == "L2.RELEASE_NOT_HELD"


def test_decoding():
    assert FirstError.decode(0, 5) is None
    info = pack(ErrorSource.AXI, axi_kind(True, 2), cmd_type=1, cmd_id=3, multi=True)
    first = FirstError.decode(info, 1234)
    assert (first.source, first.kind_name, first.cmd_type, first.cmd_id, first.multi) == (
        ErrorSource.AXI, "WRITE_SLVERR", 1, 3, True,
    )  # fmt: skip
    assert str(first) == (
        "AXI WRITE_SLVERR at cycle 1234, STORE command 3 (another source in the same clock)"
    )
    assert str(FirstError.decode(pack(ErrorSource.L1, L1Error.NORTH_COLLISION), 9)) == (
        "L1 NORTH_COLLISION at cycle 9"
    )
    assert kind_name(ErrorSource.L2, 15) == "kind 15"


def test_an_l1_error_is_captured_with_the_streamers_command():
    """A COMPUTE with SIGNAL that writes no result (k = 0): L1's SIGNAL_NO_RESULT, the
    Streamer's command."""
    cfg = _config()
    top = SystolicTop(cfg)
    out = {}

    async def tb(ctx):
        await _issue(ctx, top, type=SystolicTop.CTRL_EXEC, opcode=ExecOpcode.COMPUTE, rs1=0,
                     rs2=cfg.l2_group_rows, rd=0, k_dim=0, id=0x21,
                     sync=Sync.encode(dst=1, acquire=True, signal=True))  # fmt: skip
        out["rose"] = await _until_error(ctx, top, top.l1_error)
        await ctx.tick()
        out["info"], out["at"] = ctx.get(top.err_info), ctx.get(top.err_cycle)

    run_sim(top, tb, max_cycles=2_000)
    assert out["info"] == pack(ErrorSource.L1, L1Error.SIGNAL_NO_RESULT,
                               cmd_type=SystolicTop.CTRL_EXEC, cmd_id=0x21)  # fmt: skip
    assert out["at"] == out["rose"]


def test_a_command_that_never_starts_is_named_from_its_queue():
    """A LOAD that acquires a slot past the pool can never start: the fault names that
    queued command, though no LOAD has started (not "LOAD command 0")."""
    cfg = _config()
    top = SystolicTop(cfg)
    out = {}

    async def tb(ctx):
        await _issue(ctx, top, type=SystolicTop.CTRL_LOAD, opcode=0, dram_addr=0, local_addr=0,
                     len=1, id=9, sync=Sync.encode(dst=15, acquire=True))  # fmt: skip
        await _until_error(ctx, top, top.sync_error)
        await ctx.tick()
        out["info"] = ctx.get(top.err_info)

    run_sim(top, tb, max_cycles=2_000)
    kind = sync_kind(SlotPoolId.L2, SlotError.PROD_NO_SUCH_SLOT)
    assert out["info"] == pack(ErrorSource.SYNC, kind, cmd_type=SystolicTop.CTRL_LOAD, cmd_id=9)


def _completion_fault(first_sync, second):
    """COMPUTE 7 with ``first_sync``, then ``second`` (the fields of command 8) pushed
    behind it: the capture after the first sync error."""
    cfg = SystolicConfig(grid_rows=2, grid_cols=2, tile_rows=1, tile_cols=1, spad_read_delay=4)
    top = SystolicTop(cfg)
    rows = cfg.grid_rows
    g = cfg.l2_group_rows
    out = {}

    async def tb(ctx):
        compute = {"type": SystolicTop.CTRL_EXEC, "opcode": ExecOpcode.COMPUTE, "rs1": 0,
                   "rs2": g, "k_dim": rows}  # fmt: skip
        await _issue(ctx, top, **compute, rd=0, id=7, sync=first_sync)
        await _issue(ctx, top, **compute, rd=rows, id=8, **second)
        await _until_error(ctx, top, top.sync_error, limit=400)
        await ctx.tick()
        out["info"] = ctx.get(top.err_info)

    run_sim(top, tb, max_cycles=2_000)
    return out["info"]


def test_a_release_past_the_pool_names_the_command_that_completed():
    """COMPUTE 7 RELEASEs slot 15 without a WAIT, so it starts, and the fault comes at its
    completion, while an untracked COMPUTE 8 waits at the queue head: the capture names 7,
    the command whose release broke the protocol, not the queued 8."""
    info = _completion_fault(Sync.encode(src=15, release=True), {"sync": 0})
    kind = sync_kind(SlotPoolId.L2, SlotError.CONS_NO_SUCH_SLOT)
    assert info == pack(ErrorSource.SYNC, kind, cmd_type=SystolicTop.CTRL_EXEC, cmd_id=7)


def test_a_completion_fault_names_its_command_though_a_later_one_started():
    """COMPUTE 7 RELEASEs an L2 slot it never waited for; COMPUTE 8 starts before 7
    completes. The fault comes at 7's completion: the capture names 7, not the command the
    Streamer started last."""
    info = _completion_fault(Sync.encode(src=1, release=True), {"sync": Sync.encode(dst=1)})
    kind = sync_kind(SlotPoolId.L2, SlotError.RELEASE_NOT_HELD)
    assert info == pack(ErrorSource.SYNC, kind, cmd_type=SystolicTop.CTRL_EXEC, cmd_id=7)


# =============================================================================
# FirstErrorKind: how Streamer combines its own errors and its L1 buffers'
# =============================================================================


def test_every_l1_buffer_has_its_kind():
    """Each L1 buffer reports its own kind; together they are every buffer kind."""
    from systars.controller.streamer import Streamer

    names = [name for name, _ in Streamer.L1_UNITS]
    kinds = [k for _, k in Streamer.L1_UNITS]
    assert names == ["a_buf", "b_buf", "w_buf", "north", "south"]
    assert sorted(kinds) == sorted(set(L1Error) - {L1Error.SIGNAL_NO_RESULT,
                                                    L1Error.RESULT_WITHOUT_COMMAND,
                                                    L1Error.NORTH_COLLISION})  # fmt: skip


@pytest.mark.parametrize("first", range(1, 6), ids=lambda i: L1Error(i + 3).name)
def test_the_first_buffer_to_fail_names_the_kind(first):
    """Six sources as in Streamer (its own errors, then the five buffers in
    L1_UNITS order): the first to rise names the kind from that very clock, and keeps it
    when another rises later."""
    from systars.controller.streamer import Streamer
    from systars.errors import FirstErrorKind

    kinds = [L1Error.RESULT_WITHOUT_COMMAND] + [k for _, k in Streamer.L1_UNITS]
    dut = FirstErrorKind(len(kinds), 4)

    async def tb(ctx):
        for i, k in enumerate(kinds):
            ctx.set(getattr(dut, f"kind_{i}"), k)
        assert not ctx.get(dut.error)
        ctx.set(getattr(dut, f"error_{first}"), 1)
        assert ctx.get(dut.error) and ctx.get(dut.kind) == kinds[first]  # the first clock
        await ctx.tick()
        ctx.set(dut.error_0, 1)  # the controller's own error, later
        await ctx.tick().repeat(2)
        assert ctx.get(dut.kind) == kinds[first]

    run_sim(dut, tb)


def test_sources_rising_together_name_the_lowest():
    from systars.errors import FirstErrorKind

    dut = FirstErrorKind(3, 4)

    async def tb(ctx):
        for i, k in enumerate((7, 8, 9)):
            ctx.set(getattr(dut, f"kind_{i}"), k)
        ctx.set(dut.error_1, 1)
        ctx.set(dut.error_2, 1)
        await ctx.tick().repeat(2)
        assert ctx.get(dut.kind) == 8

    run_sim(dut, tb)
