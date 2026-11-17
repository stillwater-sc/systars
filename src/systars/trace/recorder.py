"""
``TileRecorder``: taps a ``SystolicTop`` simulation and writes a ``Trace``.

Usage, in an Amaranth simulation::

    rec = TileRecorder(top)
    sim.add_process(rec.process())             # samples every probe each cycle
    ...                                        # the program runs
    trace = rec.trace()                        # tables and intervals

What it records, from the RTL's ports (``probes.py``)
    * every tile command from accept to completion, with its AXI transactions
      (``commands``);
    * resource occupancy derived from those, and the tile's internal lanes (``lanes.py``:
      L2 ports, credit slots, the Combiner, each process's queue; with ``fine``, each L1
      stream and result column) (``intervals``);
    * the tile's first error (``errors``): its capture's ports, the cycle it happened on
      the trace's clock, and the ``commands`` row it names.

This is kpu-gen's recorder (branes-ai/embodied-ai-soc-generator, ``kpu_gen/trace``) for its
standalone-tile path (#18): systars has no SoC, so there are no NoC packets, SoC DMA
transfers or memory-cell accesses, and no residencies (kpu-gen derives them from its
runtime's annotations). The bundle keeps kpu-gen's tables, with those left empty, so the
two read alike.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, is_dataclass, replace
from typing import Any

from systars.errors import FirstError
from systars.util.commands import ExecOpcode

from . import probes
from .lanes import FineLanes, LaneRuns, TileLanes
from .record import Trace, producer

COMMAND_TYPES = {0: "load", 1: "store", 2: "exec"}


def _plain(value: Any) -> Any:
    """Annotations as JSON-able values (dataclasses become dicts)."""
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(v) for v in value]
    return value


def _runs(addrs: list[int]) -> list[list[int]]:
    """Sorted unique addresses as ``[first, count]`` runs."""
    out: list[list[int]] = []
    for a in sorted(set(addrs)):
        if out and out[-1][0] + out[-1][1] == a:
            out[-1][1] += 1
        else:
            out.append([a, 1])
    return out


class TileRecorder:
    """Records one standalone ``SystolicTop`` run (the ISA's ``Matmul`` on a tile, the
    differential harness, tests/fuzz) as site 0: its ``commands``, ``errors`` and internal
    lanes (``lanes.py``), the fine ones too unless ``fine=False``. There is no NoC, DMA or
    memory cell to record (see the module docstring).

    Usage, in an Amaranth simulation of a design containing ``top``::

        rec = TileRecorder(top)
        sim.add_process(rec.process())
        ...
        trace = rec.trace()
    """

    SCOPE = "tile"

    def __init__(self, top: Any, *, fine: bool = True):
        self.top = top
        cfg = top.config
        self.rows = self.cols = 1
        self.word_bytes = cfg.blockmover_buswidth // 8
        self.row_words = {0: 1}
        self._tiles = {0: probes.core_signals(top)}
        self._init_tables({0: cfg}, {0: top} if fine else {})

    def _init_tables(self, tile_configs: dict[int, Any], fine_cores: dict[int, Any]) -> None:
        """The recording state, and the internal lanes of each tile (lanes.py): the coarse
        ones for every tile, the fine ones for the cores in ``fine_cores``."""
        self.cycle = 0
        self.lane_runs = LaneRuns()
        self._lanes = {s: TileLanes(s, cfg, self.lane_runs) for s, cfg in tile_configs.items()}
        self._tile_fields = {s: list(probes.CORE_FIELDS) for s in self._tiles}
        self._fine: dict[int, FineLanes] = {}
        for s, core in fine_cores.items():
            fine = probes.fine_signals(core)
            self._tiles[s] = self._tiles[s] + list(fine.values())
            self._tile_fields[s] += list(fine)
            self._fine[s] = FineLanes(s, list(fine), self.lane_runs)

        self.commands: list[dict[str, Any]] = []
        self.annotations: list[dict[str, Any]] = []
        self.values: list[dict[str, Any]] = []
        self.errors: list[dict[str, Any]] = []
        # Per site, the capture last seen: its ERR_INFO and its cycle (a capture re-armed
        # and made again by a like error in one clock keeps ERR_INFO but not the cycle)
        self._err_info: dict[int, tuple[int, int]] = {}
        self.problems: list[str] = []  # what the recorder could not make sense of
        self._cmd_open: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
        self._axi_open: dict[tuple[int, str], dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Issuer side
    # ------------------------------------------------------------------

    def annotate(self, event: dict[str, Any]) -> None:
        """An issuer's observer: record what it asked for (``annotations``)."""
        self.annotations.append({"cycle": self.cycle, **_plain(event)})

    def add_values(self, tile: Any, actual: list, expected: list, **attrs: Any) -> None:
        """Record a result block's values beside the expected ones."""
        self.values.append({"tile": _plain(tile), "actual": actual, "expected": expected, **attrs})

    # ------------------------------------------------------------------
    # Simulation side
    # ------------------------------------------------------------------

    def process(self) -> Callable[[Any], Any]:
        """An Amaranth simulator process (``Simulator.add_process``) that samples
        every probe once per cycle."""
        groups = list(self._tiles.items())
        flat = [sig for _, sigs in groups for sig in sigs]

        async def run(ctx):
            async for _, _, *values in ctx.tick().sample(*flat):
                i = 0
                for site, sigs in groups:
                    v = values[i : i + len(sigs)]
                    i += len(sigs)
                    self._tile_step(site, dict(zip(self._tile_fields[site], v, strict=True)))
                self.cycle += 1

        return run

    def _error_step(self, site: int, v: dict[str, Any]) -> None:
        """A new capture (a first error, or one after a re-arm) is a row of ``errors``."""
        info = int(v["err_info"])
        seen = (info, int(v["err_cycle"]))
        if info and seen != self._err_info.get(site, (0, 0)):
            first = FirstError.decode(info, int(v["err_cycle"]))
            assert first is not None
            # The tile's clock counter and the trace's differ by a constant: place the
            # event on the trace's clock
            ago = (int(v["cycle"]) - first.cycle) % (1 << 32)
            first = replace(first, cycle=self.cycle - ago)
            ctype = None if first.cmd_type is None else COMMAND_TYPES[first.cmd_type]
            named = next(
                (c["id"] for c in reversed(self.commands)
                 if c["site"] == site and c["type"] == ctype and c["cmd_id"] == first.cmd_id),
                None,
            )  # fmt: skip
            self.errors.append({
                "id": len(self.errors), "site": site, "cycle": first.cycle,
                "source": first.source.name, "kind": first.kind_name, "code": first.kind,
                "cmd_type": ctype, "cmd_id": first.cmd_id if ctype else None,
                "command": named, "multi": first.multi, "text": str(first),
            })  # fmt: skip
        self._err_info[site] = seen

    def _result_step(self, site: int, rid: int) -> None:
        """The Streamer's ``result`` for command ``rid``: the oldest COMPUTE with that ID
        and no result yet (IDs are 8 bits and wrap; a PRELOAD or CONFIG has no result)."""
        cmd = next((c for c in self.commands if c["site"] == site and c["type"] == "exec"
                    and c["opcode"] in _COMPUTES and c["cmd_id"] == rid
                    and c["result"] == -1), None)  # fmt: skip
        if cmd is not None:
            cmd["result"] = self.cycle

    def _tile_step(self, site: int, v: dict[str, Any]) -> None:
        self._error_step(site, v)
        self._lanes[site].step(self.cycle, v)
        if site in self._fine:
            self._fine[site].step(self.cycle, v)
        # A command opens when its process starts it (accepted earlier into its queue)
        for t, name in COMMAND_TYPES.items():
            if v[f"{probes.PROCESSES[name]}_started"]:
                sid = int(v[f"{probes.PROCESSES[name]}_started_id"])
                cmd = next((c for c in self._cmd_open[(site, t)]
                            if c["cmd_id"] == sid and not c["_started"]), None)  # fmt: skip
                if cmd is not None:
                    cmd["open"], cmd["_started"] = self.cycle, True
        if v["result"]:  # a COMPUTE's results are in the Combiner: its drain ends
            self._result_step(site, int(v["result_id"]))
        for t, name in COMMAND_TYPES.items():
            if v[f"{probes.PROCESSES[name]}_completed"]:
                queue = self._cmd_open[(site, t)]
                cid = int(v[f"{probes.PROCESSES[name]}_completed_id"])
                cmd = next((c for c in queue if c["cmd_id"] == cid), None)
                if cmd is None:
                    self.problems.append(f"cycle {self.cycle}: site {site} completed {name} {cid}")
                else:
                    queue.remove(cmd)
                    cmd["close"] = self.cycle
        if v["cmd_valid"] and v["cmd_ready"]:
            t = int(v["cmd_type"])
            cmd = {
                "id": len(self.commands),
                "site": site,
                "cmd_id": int(v["cmd_id"]),
                "type": COMMAND_TYPES.get(t, str(t)),
                "opcode": int(v["cmd_opcode"]),
                "dram_addr": int(v["cmd_dram_addr"]),
                "local_addr": int(v["cmd_local_addr"]),
                "len": int(v["cmd_len"]),
                "rs1": int(v["cmd_rs1"]),
                "rd": int(v["cmd_rd"]),
                "k_dim": int(v["cmd_k_dim"]),
                # A COMPUTE's Combiner epilogue code (FINAL and the profile, #77); a
                # STORE's activation field
                "activation": int(v["cmd_activation"]),
                "accept": self.cycle,  # into its process's queue
                "open": self.cycle,  # started (the accept cycle until the process starts it)
                "close": -1,
                "result": -1,  # exec: the cycle its results were in the Combiner
                "_started": False,
                "axi": [],
                "_local": [],
            }
            self.commands.append(cmd)
            self._cmd_open[(site, t)].append(cmd)
        for kind, t, valid, ready, addr, beats in (
            ("r", 0, v["axi_arvalid"], v["axi_arready"], v["axi_araddr"], v["axi_arlen"]),
            ("w", 1, v["axi_awvalid"], v["axi_awready"], v["axi_awaddr"], v["axi_awlen"]),
        ):
            if valid and ready:
                txn = {"kind": kind, "addr": int(addr), "beats": int(beats) + 1,
                       "open": self.cycle, "close": -1, "resp": 0}  # fmt: skip
                self._axi_open[(site, kind)] = txn
                queue = self._cmd_open[(site, t)]
                if queue:
                    queue[0]["axi"].append(txn)
                else:
                    self.problems.append(
                        f"cycle {self.cycle}: site {site} AXI {kind} outside a command"
                    )
        if v["axi_rvalid"] and v["axi_rready"] and v["axi_rlast"] and (site, "r") in self._axi_open:
            self._axi_open.pop((site, "r"))["close"] = self.cycle
        if v["axi_bvalid"] and v["axi_bready"] and (site, "w") in self._axi_open:
            txn = self._axi_open.pop((site, "w"))
            txn["close"], txn["resp"] = self.cycle, int(v["axi_bresp"])

    # ------------------------------------------------------------------
    # The bundle
    # ------------------------------------------------------------------

    def geometry(self) -> dict[str, Any]:
        """The tile's shape, for the manifest."""
        cfg = self.top.config
        return {
            "rows": 1, "cols": 1, "word_bytes": self.word_bytes, "l3_words": None,
            "present": [True], "row_words": {"0": 1}, "layers": ["L2", "CMB"], "links": [],
            "tile": {
                "array": [cfg.grid_rows * cfg.tile_rows, cfg.grid_cols * cfg.tile_cols],
                "l2_rows": cfg.sp_total_rows, "l2_group_rows": cfg.l2_group_rows,
                "l2_slots": cfg.l2_slots, "combiner_slots": cfg.combiner_slots,
            },
        }  # fmt: skip

    def trace(self, **manifest: Any) -> Trace:
        """The run so far as a ``Trace`` (``manifest`` entries are added to its manifest)."""
        t = Trace(
            {"level": "RTL", "scope": self.SCOPE, "producer": producer(), "cycles": self.cycle,
             "geometry": self.geometry(), "problems": list(self.problems), **manifest}
        )  # fmt: skip
        for cmd in self.commands:
            row = {k: v for k, v in cmd.items() if not k.startswith("_")}
            row["axi"] = [dict(x) for x in cmd["axi"]]
            row["local_runs"] = _runs([a for a, _ in cmd["_local"]])
            t.tables["commands"].append(row)
        t.tables["annotations"] = [dict(a) for a in self.annotations]
        t.tables["values"] = [dict(v) for v in self.values]
        t.tables["errors"] = [dict(e) for e in self.errors]
        t.tables["intervals"] = self._intervals()
        return t

    def _intervals(self) -> list[dict[str, Any]]:
        out = []
        for cmd in self.commands:
            out.append({"resource": f"site {cmd['site']}:{cmd['type']}",
                        "client": f"cmd {cmd['cmd_id']}", "open": cmd["open"],
                        "close": cmd["close"], "event": f"command:{cmd['id']}",
                        "what": cmd["type"], "accesses": 0})  # fmt: skip
            # The Streamer's drain: done with L2, until its results are in the Combiner
            if cmd["type"] == "exec" and cmd["close"] != -1 and cmd["result"] >= cmd["close"]:
                out.append({"resource": f"site {cmd['site']}:exec drain",
                            "client": f"cmd {cmd['cmd_id']}", "open": cmd["close"],
                            "close": cmd["result"], "event": f"command:{cmd['id']}",
                            "what": "drain", "accesses": 0})  # fmt: skip
        out += [dict(r) for r in self.lane_runs.rows]
        return out


# The Streamer's commands with results in the Combiner (controller/streamer.py's is_compute)
_COMPUTES = (int(ExecOpcode.COMPUTE), int(ExecOpcode.COMPUTE_ACCUMULATE))
