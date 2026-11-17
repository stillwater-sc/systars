"""The tile's hooks for a trace (#18, the systolic part of kpugen esg#27).

kpugen's trace (``kpu_gen.trace``) reads the tile's internal state through probes: a slot's
READING state, and signals inside the submodules of ``SystolicTop`` and the Streamer, which
must exist before a simulation starts. systars has no trace package yet; these tests keep
the hooks the RTL provides for one.
"""

from amaranth.sim import Simulator

from systars.config import Dataflow, SystolicConfig
from systars.controller.sync import SlotPool, SlotState
from systars.top import SystolicTop


def test_a_slot_pool_reports_each_slots_state():
    """``free``, ``ready`` and ``reading``: one bit per state, none for OWNED."""
    pool = SlotPool(2)
    seen = {}

    async def tb(ctx):
        async def pulse(**values):
            for name, value in values.items():
                ctx.set(getattr(pool, name), value)
            await ctx.tick()
            for name in values:
                ctx.set(getattr(pool, name), 0)

        def state():
            return (ctx.get(pool.free[0]), ctx.get(pool.ready[0]), ctx.get(pool.reading[0]))

        seen[SlotState.FREE] = state()
        await pulse(prod_slot=0, prod_acquire=1, prod_take=1)
        seen[SlotState.OWNED] = state()
        await pulse(prod_signal=1, prod_signal_slot=0)
        seen[SlotState.READY] = state()
        await pulse(cons_slot=0, cons_wait=1, cons_take=1)
        seen[SlotState.READING] = state()

    sim = Simulator(pool)
    sim.add_clock(1e-8)
    sim.add_testbench(tb)
    sim.run()
    assert seen == {
        SlotState.FREE: (1, 0, 0),
        SlotState.OWNED: (0, 0, 0),
        SlotState.READY: (0, 1, 0),
        SlotState.READING: (0, 0, 1),
    }


def test_the_tiles_submodules_exist_before_elaboration():
    """A probe can name a submodule's signal before a simulation elaborates the tile."""
    cfg = SystolicConfig(
        grid_rows=2,
        grid_cols=2,
        dataflow=Dataflow.OUTPUT_STATIONARY | Dataflow.B_STATIONARY,
    )
    top = SystolicTop(cfg)
    for name in ("systolic_array", "scratchpad", "combiner", "blockmover", "bm_in", "bm_out",
                 "streamer", "l2_slots", "combiner_slots"):  # fmt: skip
        assert getattr(top, name) is not None, name
    # The BlockMover holds both legs and their engines (#44)
    assert top.bm_in is top.blockmover.bm_in and top.bm_out is top.blockmover.bm_out
    assert top.blockmover.stream_reader is not None and top.blockmover.stream_writer is not None
    assert top.l2_slots.reading.shape().width == cfg.l2_slots
    streamer = top.streamer
    # L1: every buffer the dataflows built need, None for the ones they don't
    for name in ("a_buf", "b_buf", "north", "south", "w_buf"):
        assert getattr(streamer, name) is not None, name
    os_only = SystolicTop(SystolicConfig(grid_rows=2, grid_cols=2,
                                         dataflow=Dataflow.OUTPUT_STATIONARY)).streamer  # fmt: skip
    assert os_only.south is None and os_only.w_buf is None and os_only.north is not None


def test_queued_holds_while_a_command_waits_at_its_queue_head():
    """``*_queued`` (#52): a process has a command at its queue's head. An untracked CONFIG
    passes through at once; a COMPUTE that WAITs for a token no one emits stays queued,
    and its process never starts it."""
    from systars.controller.sync import Sync
    from systars.util.commands import ExecOpcode

    cfg = SystolicConfig(grid_rows=2, grid_cols=2, dataflow=Dataflow.OUTPUT_STATIONARY)
    top = SystolicTop(cfg)
    seen = {}

    async def tb(ctx):
        async def issue(**fields):
            for name, value in fields.items():
                ctx.set(getattr(top, f"cmd_{name}"), value)
            ctx.set(top.cmd_valid, 1)
            await ctx.tick().until(top.cmd_ready)
            ctx.set(top.cmd_valid, 0)

        exec_type = SystolicTop.CTRL_EXEC
        await issue(type=exec_type, opcode=ExecOpcode.CONFIG, rs1=0, id=1, sync=0)
        started = 0
        for _ in range(20):
            started += ctx.get(top.streamer_started)
            await ctx.tick()
        seen["config"] = (started, ctx.get(top.streamer_queued))
        await issue(type=exec_type, opcode=ExecOpcode.COMPUTE, rs1=0, rs2=cfg.l2_group_rows,
                    rd=0, k_dim=2, id=2, sync=Sync.encode(src=0, wait=True))  # fmt: skip
        queued = started = 0
        for _ in range(50):
            queued += ctx.get(top.streamer_queued)
            started += ctx.get(top.streamer_started)
            await ctx.tick()
        seen["blocked"] = (queued, started)
        seen["others"] = (ctx.get(top.bm_in_queued), ctx.get(top.bm_out_queued))

    sim = Simulator(top)
    sim.add_clock(1e-8)
    sim.add_testbench(tb)
    sim.run()
    assert seen["config"] == (1, 0)  # started once, then nothing queued
    assert seen["blocked"] == (50, 0)  # queued every clock, never started
    assert seen["others"] == (0, 0)
