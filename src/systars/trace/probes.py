"""
The probe map: which signals of a ``SystolicTop`` the recorder samples, and what each one
means. Kept in one place, so the recorder never names a signal path.

The Amaranth simulator reads any signal, so the probes tap internal state (the command
port, the AXI master, the first-error capture, and the internal lanes of trace/lanes.py)
without adding debug ports to the RTL. The paths reach submodules ``SystolicTop`` builds in
``__init__`` (#18).

This is kpu-gen's probe map (branes-ai/embodied-ai-soc-generator, ``kpu_gen/trace``) for its
standalone-tile path: systars has no SoC, so the NoC links, the SoC DMA and the memory cells'
ports are not here.
"""

from __future__ import annotations

from typing import Any

# The tile's processes (#97), by the names the trace bundle gives their commands and lanes
# (``"load"``, ``"store"``, ``"exec"``, the command types of ``cmd_type``): the bundle keeps
# those, and the probes take the processes' own
PROCESSES = {"load": "bm_in", "store": "bm_out", "exec": "streamer"}

TILE_FIELDS = (
    "cmd_valid", "cmd_ready", "cmd_type", "cmd_opcode", "cmd_id", "cmd_dram_addr",
    "cmd_local_addr", "cmd_len", "cmd_rs1", "cmd_rd", "cmd_k_dim", "cmd_activation",
    "bm_in_completed", "bm_in_completed_id", "bm_out_completed", "bm_out_completed_id",
    "streamer_completed", "streamer_completed_id",
    "axi_arvalid", "axi_arready", "axi_araddr", "axi_arlen",
    "axi_rvalid", "axi_rready", "axi_rlast",
    "axi_awvalid", "axi_awready", "axi_awaddr", "axi_awlen",
    "axi_bvalid", "axi_bready", "axi_bresp",
    # The first-error capture (tiles/systolic/errors.py, #84)
    "cycle", "err_info", "err_cycle",
)  # fmt: skip


# The systolic tile's internal lanes (trace/lanes.py, #27): name -> path from the core
# (a SystolicTop). The paths reach submodules SystolicTop builds in __init__.
TILE_LANES = {
    "l2_read_req": "scratchpad.read_req", "l2_read_addr": "scratchpad.read_addr",
    "l2_read_b_req": "scratchpad.read_b_req", "l2_read_b_addr": "scratchpad.read_b_addr",
    "l2_write_req": "scratchpad.write_req", "l2_write_addr": "scratchpad.write_addr",
    "l2_free": "l2_free", "l2_ready": "l2_ready", "l2_reading": "l2_slots.reading",
    "cmb_free": "combiner_free", "cmb_ready": "combiner_ready",
    "cmb_reading": "combiner_slots.reading",
    "cmb_write_req": "combiner.write_req", "exec_cmb_write_req": "result_path.combiner_write_req",
    "cmb_accumulate": "combiner.accumulate", "cmb_write_epilogue": "combiner.write_epilogue",
    "cmb_read_req": "combiner.read_req", "cmb_read_ready": "combiner.read_ready",
    "result": "result", "result_id": "result_id",
    # A process started a command (its queue's head was dispatched)
    "bm_in_started": "bm_in_started", "bm_in_started_id": "bm_in_started_id",
    "bm_out_started": "bm_out_started", "bm_out_started_id": "bm_out_started_id",
    "streamer_started": "streamer_started", "streamer_started_id": "streamer_started_id",
    # Why a process waits (#85): a command queued, the controller's view of it
    "bm_in_queued": "bm_in_queued", "bm_in_cmd_valid": "blockmover.bm_in.cmd_valid",
    "bm_in_cmd_ready": "blockmover.bm_in.cmd_ready",
    "bm_out_queued": "bm_out_queued", "bm_out_cmd_valid": "blockmover.bm_out.cmd_valid",
    "bm_out_cmd_ready": "blockmover.bm_out.cmd_ready",
    "streamer_queued": "streamer_queued", "streamer_cmd_valid": "streamer.cmd_valid",
    "streamer_cmd_ready": "streamer.cmd_ready",
    "cmb_forward_hit": "combiner.forward_hit",
}  # fmt: skip


CORE_FIELDS = (*TILE_FIELDS, *TILE_LANES)
"""The names of ``core_signals``, in order."""


def _path(obj: Any, path: str) -> Any:
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def core_signals(core: Any) -> list[Any]:
    """A tile core's probes: ``TILE_FIELDS``, then ``TILE_LANES``, in order."""
    return [getattr(core, name) for name in TILE_FIELDS] + [
        _path(core, path) for path in TILE_LANES.values()
    ]


def fine_signals(core: Any) -> dict[str, Any]:
    """A tile core's fine probes (trace/lanes.py's ``FineLanes``, #27): each L1 stream
    buffer's credit, stream markers and loads, each result buffer column's arrivals and
    the rows leaving, and the fabric's north collision. Named per lane, so the set
    follows the array's size and dataflows."""
    ex, res = core.streamer, core.result_path
    out: dict[str, Any] = {"collision": res.array_out_collision}
    for buf, unit in (("A", ex.a_buf), ("B", ex.b_buf), ("W", ex.w_buf)):
        if unit is None:
            continue
        out[f"l1_{buf}_load_valid"] = unit.load_valid
        out[f"l1_{buf}_load_index"] = unit.load_index
        out[f"l1_{buf}_load_len"] = unit.load_len
        for i in range(unit.count):
            for name in ("credit", "valid", "first", "last"):
                port = name if name == "credit" else f"out_{name}"
                out[f"l1_{buf}{i}_{name}"] = getattr(unit, f"{port}_{i}")
    for side, unit in (("N", res.north), ("S", res.south)):
        if unit is None:
            continue
        out[f"res_{side}_out"] = unit.out_valid
        for j in range(unit.lanes):
            out[f"res_{side}_in_{j}"] = getattr(unit, f"in_valid_{j}")
            out[f"res_{side}_credit_{j}"] = getattr(unit, f"credit_{j}")
    return out
