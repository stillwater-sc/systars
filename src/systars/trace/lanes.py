"""
The systolic tile's internal occupancy lanes (#27, part 1: the coarse lanes).

``TileLanes`` turns one tile's probed internal signals (``probes.TILE_LANES``) into
``intervals`` rows, one lane per resource, as each clock is sampled:

``site N:l2 A rd``, ``site N:l2 B rd``
    each L2 bank group's read port, by client (``streamer a`` / ``streamer b``, the
    Streamer's two read ports); ``conflict`` when both read one group in a clock (the
    L2 monitor's ``READS_SAME_GROUP``)
``site N:l2 A wr``, ``site N:l2 B wr``
    each group's write port (``blockmover``: a LOAD's rows)
``site N:l2 slot S``, ``site N:cmb slot S``
    each credit slot's state: ``owned`` (filling), ``ready``, ``reading`` (FREE: no bar)
``site N:cmb wr``
    the Combiner's write port: ``overwrite`` or ``accumulate`` rows from the Streamer
    (``final pN``: the reduction's last, with its epilogue profile) or a LOAD's
``site N:cmb rd``
    the result group's read port: a STORE's ``read``, or its ``wait`` while the
    Combiner holds the port (a row still in its merge pipeline)
``site N:cmb fwd``
    an accumulate the merge pipeline fed by forwarding, not from the bank (#85)
``site N:load wait``, ``site N:exec wait``, ``site N:store wait``
    a process with a command queued that it has not started: ``sync`` (its credits or
    tokens are not there) or ``busy`` (its controller is still on the last one)

The Streamer's commands and each one's drain (from done with L2 to its results in the
Combiner) come from the ``commands`` table (``recorder.py``).

Each lane is run-length encoded: back-to-back clocks with the same activity and client
are one interval, so a long run stays small.
"""

from __future__ import annotations

from typing import Any

from systars.memory.epilogue import EPILOGUE_FINAL, EPILOGUE_PROFILE_MASK

from . import probes


class LaneRuns:
    """Run-length encoded lanes: ``step`` a lane's activity each clock it has one."""

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []
        self._open: dict[str, dict[str, Any]] = {}

    def step(self, cycle: int, resource: str, what: str, client: str = "", *,
             merge: bool = True, **attrs: Any) -> None:  # fmt: skip
        """``merge=False``: a run of its own even beside a like one (an event, such as a
        load, whose ``attrs`` must not be lost)."""
        run = self._open.get(resource)
        if (
            merge
            and run is not None
            and run["close"] == cycle - 1
            and (run["what"], run["client"]) == (what, client)
        ):
            run["close"] = cycle
            run["accesses"] += 1
            return
        run = {"resource": resource, "client": client, "open": cycle, "close": cycle,
               "event": "", "what": what, "accesses": 1, **attrs}  # fmt: skip
        self._open[resource] = run
        self.rows.append(run)


class TileLanes:
    """One systolic tile's coarse lanes (see the module docstring)."""

    def __init__(self, site: int, config: Any, runs: LaneRuns):
        self.site = site
        self.group_rows = config.l2_group_rows
        self.l2_slots = config.l2_slots
        self.combiner_slots = config.combiner_slots
        self.runs = runs

    def _lane(self, name: str) -> str:
        return f"site {self.site}:{name}"

    def step(self, cycle: int, v: dict[str, Any]) -> None:
        """One clock's samples: ``v`` maps ``probes.TILE_LANES`` names to values."""
        runs, lane = self.runs, self._lane

        # L2: each group's read port (the Streamer's ports A and B) and write port
        readers: dict[str, list[str]] = {"A": [], "B": []}
        for req, addr, client in (("l2_read_req", "l2_read_addr", "streamer a"),
                                  ("l2_read_b_req", "l2_read_b_addr", "streamer b")):  # fmt: skip
            if v[req]:
                readers["A" if v[addr] < self.group_rows else "B"].append(client)
        for group, clients in readers.items():
            if clients:
                what = "conflict" if len(clients) > 1 else "read"
                runs.step(cycle, lane(f"l2 {group} rd"), what, " + ".join(clients))
        if v["l2_write_req"]:
            group = "A" if v["l2_write_addr"] < self.group_rows else "B"
            runs.step(cycle, lane(f"l2 {group} wr"), "write", "blockmover")

        # The credit slots of both pools
        for pool, n in (("l2", self.l2_slots), ("cmb", self.combiner_slots)):
            free, ready, reading = (int(v[f"{pool}_{k}"]) for k in ("free", "ready", "reading"))
            for s in range(n):
                if free >> s & 1:
                    continue
                state = "ready" if ready >> s & 1 else "reading" if reading >> s & 1 else "owned"
                runs.step(cycle, lane(f"{pool} slot {s}"), state)

        # The Combiner: its write port (the Streamer's rows win; a LOAD's fill the rest)
        if v["cmb_write_req"]:
            if v["exec_cmb_write_req"]:
                epilogue = int(v["cmb_write_epilogue"])
                what = "accumulate" if v["cmb_accumulate"] else "overwrite"
                if epilogue & EPILOGUE_FINAL:
                    what += f" final p{epilogue & EPILOGUE_PROFILE_MASK}"
                runs.step(cycle, lane("cmb wr"), what, "streamer")
            else:
                what = "accumulate" if v["cmb_accumulate"] else "overwrite"
                runs.step(cycle, lane("cmb wr"), what, "load")
        # ... and its result group's read port, a STORE's
        if v["cmb_read_req"]:
            runs.step(cycle, lane("cmb rd"), "read" if v["cmb_read_ready"] else "wait", "store")
        if v["cmb_forward_hit"]:  # an accumulate fed from the merge pipeline, not the bank
            runs.step(cycle, lane("cmb fwd"), "forward", "streamer")

        # Why each process waits with a command queued: its credits or tokens (`sync`),
        # or its controller, still busy with the last one (`busy`)
        for name in ("load", "exec", "store"):
            proc = probes.PROCESSES[name]
            if v[f"{proc}_queued"]:
                if not v[f"{proc}_cmd_valid"]:
                    runs.step(cycle, lane(f"{name} wait"), "sync")
                elif not v[f"{proc}_cmd_ready"]:
                    runs.step(cycle, lane(f"{name} wait"), "busy")


class FineLanes:
    """One systolic tile's fine lanes (#27, part 2): its L1 and the fabric's result
    edges, a lane per stream and per column. ``probes.fine_signals`` names the samples.

    ``site N:l1 A<i>`` (``B<j>``, ``W<j>``)
        one L1 stream buffer: ``load`` (a block arrives, with its ``len``; its clock
        shows the load even
        when the previous block's last element leaves then), ``stream`` as its elements
        go into the fabric, one a clock, marked ``first`` / ``last`` at an OS reduction's
        ends (``first+last``: both), and ``held`` should a loaded block not stream
    ``site N:res N<j>`` (``S<j>``)
        a result buffer column's depth (``1``, ``2``, ...): the OS north drain's or the
        WS/IS south partial sums' rows that arrived and have not left as a row
    ``site N:res rows``
        a result row leaving for the Combiner (``north`` or ``south``)
    ``site N:res N<j> overflow``
        a row arriving at a full column, which refuses it (never, in a correct schedule)
    ``site N:collision``
        two OS results meeting on a north link (never, in a correct schedule)
    """

    def __init__(self, site: int, names: list[str], runs: LaneRuns):
        self.site, self.runs = site, runs
        self.streams = sorted({n.split("_")[1] for n in names if n.startswith("l1_") and n.endswith("_credit")},
                              key=lambda s: (s[0], int(s[1:])))  # fmt: skip
        # Only the result buffers this tile has (an OS-only tile has no south one)
        self.columns = {side: sorted(int(n.rsplit("_", 1)[1]) for n in names
                                     if n.startswith(f"res_{side}_in_"))
                        for side in ("N", "S") if f"res_{side}_out" in names}  # fmt: skip
        self.depth = {(side, j): 0 for side, cols in self.columns.items() for j in cols}

    def step(self, cycle: int, v: dict[str, Any]) -> None:
        runs, lane = self.runs, lambda name: f"site {self.site}:{name}"
        loads: dict[str, int] = {}  # stream -> the loaded block's length
        for buf in {s[0] for s in self.streams}:
            if v[f"l1_{buf}_load_valid"]:
                loads[f"{buf}{int(v[f'l1_{buf}_load_index'])}"] = int(v[f"l1_{buf}_load_len"])
        for s in self.streams:
            p = f"l1_{s}_"
            if s in loads:
                runs.step(cycle, lane(f"l1 {s}"), "load", merge=False, len=loads[s])
                continue
            if v[p + "valid"]:
                first, last = v[p + "first"], v[p + "last"]
                what = (
                    "first+last"
                    if first and last
                    else "first"
                    if first
                    else "last"
                    if last
                    else "stream"
                )
            elif not v[p + "credit"]:
                what = "held"
            else:
                continue
            runs.step(cycle, lane(f"l1 {s}"), what)
        for side, cols in self.columns.items():
            out = bool(v[f"res_{side}_out"])
            if out:
                runs.step(cycle, lane("res rows"), "row", "north" if side == "N" else "south")
            for j in cols:
                arrives = bool(v[f"res_{side}_in_{j}"])
                taken = arrives and bool(v[f"res_{side}_credit_{j}"])  # a full column refuses
                if arrives and not taken:
                    runs.step(cycle, lane(f"res {side}{j} overflow"), "overflow")
                d = self.depth[(side, j)] + taken - out
                self.depth[(side, j)] = d
                if d:
                    runs.step(cycle, lane(f"res {side}{j}"), str(d))
        if v["collision"]:
            runs.step(cycle, lane("collision"), "collision")
