"""
A run's performance summary (#85): where its cycles went, from a ``Trace``.

``summary(trace)`` returns a JSON-able dict (``python -m systars.trace BUNDLE --perf`` prints it):

``cycles``
    the run's sampled clocks; ``to_result``: the clock after the last result was in
    place: the last command's completion, or a COMPUTE's results reaching the Combiner,
    whichever is later (a COMPUTE completes once done with L2, before its results land)
``sites[N]``, per tile
    ``commands``
        per process (load, exec, store): how many, and its ``busy`` clocks (from each
        command's start to its completion, overlaps counted once)
    ``waits``
        per process, the clocks it had a command queued and did not start it: ``sync``
        (its credits or tokens were not there) or ``busy`` (its controller was on the last
        one). An idle process with nothing queued is not waiting
    ``blockmover``
        ``busy``: the clocks a LOAD or a STORE was running (both are its commands; a clock
        with both counts once)
    ``streamer``
        ``busy`` and ``idle`` clocks between its first start and its last completion, and
        ``drain``: the clocks its COMPUTEs spent between done with L2 and their results in
        the Combiner
    ``l2``
        reads and writes per bank group, the clocks both groups were read
        (``both_read``) and the clocks both reads hit one group (``conflicts``)
    ``combiner``
        its writes (``accumulates`` and ``finals`` among them), ``forwards`` (accumulates
        fed by the merge pipeline), and STORE's ``reads`` and ``waits`` on the result
        group's read port
    ``fabric``, where the fine lanes were recorded
        ``busy``: the clocks an A operand entered the array; ``utilization``: of the
        clocks from the first such clock to the last
``soc``, for a SoC's trace
    the DMA's ``busy`` clocks, and the NoC's flits: in all, and on the busiest link

Every number counts clocks or events, so two runs of one design and program give the
same summary, and any change to the schedule changes it: the reference runs
(``tests/perf/``) are gated on it exactly.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from typing import Any

from .record import Trace

_STREAMING = ("first", "stream", "last", "first+last")


def _clocks(ivs: list[dict[str, Any]], whats: tuple[str, ...] | None = None) -> set[int]:
    """The clocks the intervals cover (only those whose ``what`` is in ``whats``)."""
    out: set[int] = set()
    for iv in ivs:
        if whats is None or iv["what"] in whats:
            out.update(range(iv["open"], iv["close"] + 1))
    return out


def _accesses(ivs: list[dict[str, Any]], pred: Callable[[str], bool] | None = None) -> int:
    """The events the intervals count (only those whose ``what`` passes ``pred``)."""
    return sum(iv["accesses"] for iv in ivs if pred is None or pred(iv["what"]))


def summary(trace: Trace) -> dict[str, Any]:
    """The run's performance summary (see the module docstring)."""
    lanes: dict[int, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for iv in trace["intervals"]:
        res = iv["resource"]
        if res.startswith("site "):
            site, _, kind = res[5:].partition(":")
            lanes[int(site)][kind].append(iv)
    cmds = trace["commands"]
    done = [c["close"] for c in cmds if c["close"] != -1]
    done += [c["result"] for c in cmds if c.get("result", -1) != -1]
    out: dict[str, Any] = {
        "cycles": trace.manifest["cycles"],
        "to_result": max(done) + 1 if done else None,
        "sites": {},
    }
    for site in sorted({c["site"] for c in cmds} | set(lanes)):
        out["sites"][str(site)] = _site(site, cmds, lanes[site])
    if trace["transfers"] or trace["packets"]:
        out["soc"] = _soc(trace)
    return out


def _site(site: int, cmds: list[dict[str, Any]], ln: dict[str, list[dict[str, Any]]]) -> dict:
    mine = [c for c in cmds if c["site"] == site]
    commands, waits, spans_of = {}, {}, {}
    for kind in ("load", "exec", "store"):
        ks = [c for c in mine if c["type"] == kind]
        spans = [{"open": c["open"], "close": c["close"], "what": kind} for c in ks
                 if c["close"] != -1]  # fmt: skip
        spans_of[kind] = spans
        commands[kind] = {"count": len(ks), "busy": len(_clocks(spans))}
        w = ln.get(f"{kind} wait", [])
        waits[kind] = {"sync": len(_clocks(w, ("sync",))), "busy": len(_clocks(w, ("busy",)))}

    execs = [c for c in mine if c["type"] == "exec" and c["close"] != -1]
    streamer: dict[str, Any] = {
        "busy": 0,
        "idle": 0,
        "drain": len(_clocks(ln.get("exec drain", []))),
    }
    if execs:
        busy = _clocks([{"open": c["open"], "close": c["close"], "what": ""} for c in execs])
        span = max(c["close"] for c in execs) - min(c["open"] for c in execs) + 1
        streamer.update(busy=len(busy), idle=span - len(busy))

    reads = {g: _clocks(ln.get(f"l2 {g} rd", [])) for g in "AB"}
    l2: dict[str, Any] = {
        g: {"reads": _accesses(ln.get(f"l2 {g} rd", []))
                     + _accesses(ln.get(f"l2 {g} rd", []), lambda w: w == "conflict"),
            "writes": _accesses(ln.get(f"l2 {g} wr", []))}
        for g in "AB"
    }  # fmt: skip
    l2["both_read"] = len(reads["A"] & reads["B"])
    l2["conflicts"] = sum(
        _accesses(ln.get(f"l2 {g} rd", []), lambda w: w == "conflict") for g in "AB"
    )

    wr, rd = ln.get("cmb wr", []), ln.get("cmb rd", [])
    combiner = {
        "writes": _accesses(wr),
        "accumulates": _accesses(wr, lambda w: w.startswith("accumulate")),
        "finals": _accesses(wr, lambda w: "final" in w),
        "forwards": _accesses(ln.get("cmb fwd", [])),
        "store_reads": _accesses(rd, lambda w: w == "read"),
        "store_waits": _accesses(rd, lambda w: w == "wait"),
    }
    # The BlockMover runs both: busy while either a LOAD or a STORE is
    blockmover = {"busy": len(_clocks(spans_of["load"] + spans_of["store"]))}
    result: dict[str, Any] = {"commands": commands, "waits": waits, "streamer": streamer,
                              "blockmover": blockmover, "l2": l2, "combiner": combiner}  # fmt: skip

    a_lanes = [ivs for kind, ivs in ln.items() if kind.startswith("l1 A")]
    if a_lanes:  # the fine lanes were recorded
        ivs = [iv for lane in a_lanes for iv in lane]
        # A load clock where the previous block's last element also left streamed too
        busy = _clocks(ivs, _STREAMING) | _clocks([iv for iv in ivs if iv.get("streaming")])
        span = (max(busy) - min(busy) + 1) if busy else 0
        result["fabric"] = {"busy": len(busy),
                            "utilization": round(len(busy) / span, 4) if span else 0.0}  # fmt: skip
    return result


def _soc(trace: Trace) -> dict[str, Any]:
    dma = _clocks([{"open": t["open"], "close": t["close"], "what": ""} for t in trace["transfers"]
                   if t["close"] != -1])  # fmt: skip
    flits: dict[str, int] = defaultdict(int)
    by_id = {p["id"]: p for p in trace["packets"]}
    for h in trace["hops"]:
        flits[h["link"]] += by_id.get(h["packet"], {}).get("flits", 0)
    busiest = max(sorted(flits), key=lambda k: flits[k]) if flits else None
    return {
        "dma_busy": len(dma),
        "noc_flits": sum(flits.values()),
        "busiest_link": busiest,
        "busiest_link_flits": flits[busiest] if busiest else 0,
    }
