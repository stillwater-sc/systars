"""The command stream of a systolic ISA instruction, for a host to replay on ``SystolicTop``.

``isa_commands`` runs an ISA instruction (``Matmul``, ``Conv2d``) in the Amaranth
simulator with a command port that accepts every command, and returns the commands
it issued, in order. A cocotb test then acts as the host: it pushes each command onto
the tile's command port (``push_commands``), its credits and tile-ready tokens
included, without waiting for completions, so the Verilated tile's queues, credits and
BlockMover run the same schedule the instruction would drive.

Port of kpugen's ``tb/isa.py``: ``isa_commands`` is the same; kpugen's
``push_commands`` stages each command in the tile's CSRs and rings a doorbell, while
``SystolicTop`` takes commands on its ``cmd_*`` port, so this one drives that port.

Needs Amaranth (the instruction is simulated), which every cocotb run has, since the
RTL is generated from it.
"""

from __future__ import annotations

from amaranth.sim import Simulator

import cocotb
from cocotb.triggers import ReadOnly, RisingEdge

FIELDS = (
    "type", "opcode", "id", "dram_addr", "local_addr", "len", "stride", "row_bytes",
    "rs1", "rs2", "rd", "k_dim", "activation", "chain", "sync",
)  # fmt: skip

STORE = 1  # SystolicTop.CTRL_STORE


def isa_commands(isa, config_cmds, start_cmd, *, max_cycles=200_000):
    """Configure ``isa`` with ``(cmd, data)`` pairs, START it, and return its commands
    (dicts of ``FIELDS``). Raises if it reports an error or does not finish."""
    log: list[dict] = []
    status: dict = {}

    async def controller(ctx):
        stores = []  # cycles until each accepted STORE completes
        ctx.set(isa.cmd_ready, 1)
        while True:
            ctx.set(isa.cmd_store_completed, bool(stores) and stores[0] == 0)
            ports = [getattr(isa, f"cmd_{f}") for f in ("valid", *FIELDS)]
            valid, *values = (await ctx.tick().sample(*ports))[2:]
            if stores and stores[0] == 0:
                stores.pop(0)
            stores = [n - 1 for n in stores]
            if valid:
                log.append(dict(zip(FIELDS, (int(v) for v in values), strict=True)))
                if log[-1]["type"] == STORE:
                    stores.append(2)

    async def testbench(ctx):
        for cmd, data in [*config_cmds, (start_cmd, 0)]:
            ctx.set(isa.cfg_valid, 1)
            ctx.set(isa.cfg_cmd, cmd)
            ctx.set(isa.cfg_data, data)
            await ctx.tick().until(isa.cfg_ready)
        ctx.set(isa.cfg_valid, 0)
        for _ in range(max_cycles):
            _, _, done, error, code = await ctx.tick().sample(isa.done, isa.error, isa.error_code)
            if done or error:
                status.update(done=done, error=error, code=code)
                return

    sim = Simulator(isa)
    sim.add_clock(1e-8)
    sim.add_process(controller)
    sim.add_testbench(testbench)
    sim.run()
    if not status.get("done"):
        raise AssertionError(f"{type(isa).__name__} did not finish: {status}")
    return log


def idle_commands(dut) -> None:
    """Drive the command port idle (every field 0)."""
    dut.cmd_valid.value = 0
    for name in FIELDS:
        getattr(dut, f"cmd_{name}").value = 0


async def completions(dut, counts: dict) -> None:
    """Count every command completion (runs forever): ``counts["load"]``,
    ``["store"]`` and ``["exec"]``, from the per-controller ``*_completed`` pulses."""
    while True:
        await RisingEdge(dut.clk)
        await ReadOnly()
        for kind in ("load", "store", "exec"):
            counts[kind] += int(getattr(dut, f"{kind}_completed").value)


async def push_commands(dut, commands, *, timeout=200_000) -> None:
    """Push ``commands`` onto the tile's command port as a host would: present each
    one and go on once the tile has taken it into its queue (``cmd_ready``), never
    waiting for a completion. Then wait until every command has completed and the
    tile is idle, with no sticky credit or L1 error."""
    counts = {"load": 0, "store": 0, "exec": 0}
    monitor = cocotb.start_soon(completions(dut, counts))
    for cmd in commands:
        dut.cmd_valid.value = 1
        for name in FIELDS:
            getattr(dut, f"cmd_{name}").value = cmd[name]
        for _ in range(timeout):
            await ReadOnly()
            ready = int(dut.cmd_ready.value)
            await RisingEdge(dut.clk)
            if ready:
                break
        else:
            raise AssertionError(f"command {cmd} never entered its queue")
    idle_commands(dut)
    for _ in range(timeout):
        await RisingEdge(dut.clk)
        await ReadOnly()
        done = sum(counts.values())
        if done == len(commands) and not int(dut.busy.value):
            break
    else:
        raise AssertionError(f"{counts} of {len(commands)} commands completed")
    monitor.cancel()
    assert not int(dut.sync_error.value), "sticky credit/token error"
    assert not int(dut.l1_error.value), "sticky L1 schedule error"
    await RisingEdge(dut.clk)  # out of ReadOnly, so the caller can drive inputs
