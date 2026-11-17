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
    for name in ("systolic_array", "scratchpad", "combiner", "stream_reader", "stream_writer",
                 "bm_in", "bm_out", "streamer", "l2_slots", "combiner_slots"):  # fmt: skip
        assert getattr(top, name) is not None, name
    assert top.l2_slots.reading.shape().width == cfg.l2_slots
    streamer = top.streamer
    # L1: every buffer the dataflows built need, None for the ones they don't
    for name in ("a_buf", "b_buf", "north", "south", "w_buf"):
        assert getattr(streamer, name) is not None, name
    os_only = SystolicTop(SystolicConfig(grid_rows=2, grid_cols=2,
                                         dataflow=Dataflow.OUTPUT_STATIONARY)).streamer  # fmt: skip
    assert os_only.south is None and os_only.w_buf is None and os_only.north is not None
