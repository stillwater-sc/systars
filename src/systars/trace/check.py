"""
Checks of a ``Trace``: the NoC, the DMA and the tiles did what the issuer asked.

``check_trace`` returns a list of problems (empty when the trace is clean).
Per kpu-sim#386 (D3) these are debugging checks of one run, not a golden
event trace: they hold for any correct run, so CI can run them on every
recording without goldens to regenerate.

The checks
    * **The recorder understood the run** (``manifest.problems`` is empty).
    * **Routes.** Every packet entered at its source endpoint, followed the XY
      route hop by hop, and left at its destination endpoint, with as many
      flits as its type says (a head, plus ``words`` for a ``WR`` or a good
      ``RD_RSP``).
    * **Requests are answered.** Every request has exactly one response from
      the endpoint it was sent to, for the same address, words and tag, and
      none is an error response unless ``allow_errors``.
    * **No tile error.** No tile recorded a first error (the ``errors`` table, #84),
      unless ``allow_errors``.
    * **Wormhole links.** No link carries two packets at once.
    * **One at a time.** The DMA runs one transfer at a time, and each tile
      controller (load, store, exec) one command at a time, from its start to its
      completion. The Streamer runs a PRELOAD beside the COMPUTE issued ahead of it,
      so its PRELOADs and its other commands are checked apart.
    * **Requests match the RTL.** Every DMA request the issuer made has a
      transfer with the same fields, which touched exactly the L3 words it
      asked for at the site the issuer expected, without an error; every tile
      command was accepted with the fields it was sent with, and completed. (A
      tile's own trace, ``TileRecorder``, has no issuer to compare with.)
    * **Residencies are where the issuer expected.** Every recorded residency
      (at the addresses seen on the ports) has the site, address and size
      its annotation expected, and so has the memory a LOAD read.
    * **Values.** Every recorded result block equals its expected values.
    * **The tile's lanes** (trace/lanes.py, #27), where they were recorded:
      - every credit slot goes owned -> ready (-> reading) and is free again before the
        end: a credit taken is returned;
      - every L1 stream buffer streams each block it loads, all of its elements, one a
        clock from the clock after the load, with no bubble, and nothing it did not load;
      - every result buffer column drains (no rows left in it at the end) and refuses no
        row for being full;
      - no collision on a north link.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from systars.util.commands import ExecOpcode

from .record import Trace

_PAYLOAD = {"WR", "RD_RSP"}  # packet types that carry ``words`` payload flits
_RESPONSE = {"RD_REQ": "RD_RSP", "WR": "WR_ACK"}
_ENDPOINT = {0: "tile", 1: "cell"}


def xy_route(
    net: str, src: tuple[int, int], dst: tuple[int, int], src_ep: int, dst_ep: int
) -> list[str]:
    """The link names of the XY route from endpoint ``src_ep`` at ``src`` to ``dst_ep``
    at ``dst``: along the row to the destination column, then along the column."""
    path = [f"{net}:{src[0]},{src[1]}:from_{_ENDPOINT[src_ep]}"]
    r, c = src
    while c != dst[1]:
        step = 1 if dst[1] > c else -1
        path.append(f"{net}:{r},{c}:{'east' if step > 0 else 'west'}")
        c += step
    while r != dst[0]:
        step = 1 if dst[0] > r else -1
        path.append(f"{net}:{r},{c}:{'south' if step > 0 else 'north'}")
        r += step
    path.append(f"{net}:{r},{c}:to_{_ENDPOINT[dst_ep]}")
    return path


def _overlaps(intervals: list[tuple[int, int, str]]) -> list[tuple[str, str]]:
    """Pairs of ``(open, close, label)`` intervals that overlap (``close`` -1: open)."""
    out = []
    ordered = sorted(intervals)
    for (_o1, c1, l1), (o2, _c2, l2) in zip(ordered, ordered[1:], strict=False):
        if c1 == -1 or o2 <= c1:
            out.append((l1, l2))
    return out


def check_trace(trace: Trace, *, allow_errors: bool = False) -> list[str]:
    """Every problem found in ``trace`` (see the module docstring)."""
    problems = [f"recorder: {p}" for p in trace.manifest.get("problems", [])]
    problems += _check_packets(trace, allow_errors)
    if not allow_errors:
        problems += [f"site {e['site']}: {e['text']}" for e in trace["errors"]]
    problems += _check_links(trace)
    problems += _check_serial(trace)
    problems += _check_requests(trace)
    problems += _check_residencies(trace)
    problems += _check_values(trace)
    problems += _check_lanes(trace)
    return problems


def _check_packets(trace: Trace, allow_errors: bool) -> list[str]:
    problems = []
    packets = trace["packets"]
    for p in packets:
        label = f"packet {p['id']} ({p['type']} {tuple(p['src'])} -> {tuple(p['dst'])})"
        src, dst = tuple(p["src"][:2]), tuple(p["dst"][:2])
        expected = xy_route(p["net"], src, dst, p["src"][2], p["dst"][2])
        if p["path"] != expected:
            problems.append(f"{label}: took {p['path']}, XY is {expected}")
        flits = 1 + (p["words"] if p["type"] in _PAYLOAD and not p["error"] else 0)
        if p["flits"] != flits:
            problems.append(f"{label}: {p['flits']} flits, expected {flits}")
        if p["eject"] == -1:
            problems.append(f"{label}: never left the network")
        if p["error"] and not allow_errors:
            problems.append(f"{label}: an error response")

    # Each request answered once, by the endpoint it went to
    responses = defaultdict(list)
    for p in packets:
        if p["net"] == "rsp":
            key = (p["type"], tuple(p["src"]), tuple(p["dst"]), p["addr"], p["words"], p["tag"])
            responses[key].append(p)
    for p in packets:
        if p["net"] != "req":
            continue
        key = (_RESPONSE[p["type"]], tuple(p["dst"]), tuple(p["src"]), p["addr"], p["words"],
               p["tag"])  # fmt: skip
        later = [r for r in responses[key] if r["inject"] > p["inject"]]
        if not later:
            problems.append(f"packet {p['id']} ({p['type']}): no response")
            continue
        responses[key].remove(min(later, key=lambda r: r["inject"]))
    for rest in responses.values():
        for r in rest:
            problems.append(f"packet {r['id']} ({r['type']}): a response to no request")
    return problems


def _check_links(trace: Trace) -> list[str]:
    by_link = defaultdict(list)
    for h in trace["hops"]:
        by_link[h["link"]].append((h["head"], h["last"], f"packet {h['packet']}"))
    return [
        f"link {link}: {a} and {b} overlap"
        for link, hops in by_link.items()
        for a, b in _overlaps(hops)
    ]


def _check_serial(trace: Trace) -> list[str]:
    problems = []
    dma = [(t["open"], t["close"], f"transfer {t['id']}") for t in trace["transfers"]]
    problems += [f"DMA: {a} and {b} overlap" for a, b in _overlaps(dma)]
    lanes = defaultdict(list)
    for c in trace["commands"]:
        preload = c["type"] == "exec" and c["opcode"] == ExecOpcode.PRELOAD
        lanes[(c["site"], c["type"], preload)].append((c["open"], c["close"], f"command {c['id']}"))
    for (site, kind, _), cmds in lanes.items():
        problems += [f"site {site} {kind}: {a} and {b} overlap" for a, b in _overlaps(cmds)]
    return problems


def _check_requests(trace: Trace) -> list[str]:
    problems = []
    dmas = [a for a in trace["annotations"] if a["kind"] == "dma"]
    transfers = trace["transfers"]
    if len(dmas) != len(transfers):
        problems.append(f"{len(dmas)} DMA requests, {len(transfers)} transfers")
    for n, (ann, tr) in enumerate(zip(dmas, transfers, strict=False)):
        label = f"DMA request {n}"
        for name in ("store", "dram_addr", "l3_addr", "len"):
            if ann[name] != tr[name]:
                problems.append(f"{label}: {name} {tr[name]!r}, asked for {ann[name]!r}")
        if tr["error"] or tr["close"] == -1:
            problems.append(f"{label}: {'failed' if tr['error'] else 'never finished'}")
        if tr["site"] != ann["site"]:
            problems.append(f"{label}: went to site {tr['site']}, expected {ann['site']}")
        runs = [[ann["l3_addr"], ann["len"]]] if ann["len"] else []
        if tr["l3_sites"] != [ann["site"]] or tr["l3_runs"] != runs:
            problems.append(
                f"{label}: touched {tr['l3_runs']} at sites {tr['l3_sites']}, "
                f"expected {runs} at site {ann['site']}"
            )

    if trace.manifest.get("scope") == "tile":
        # A tile's own trace has no issuer to compare with; every command must still finish
        return problems + [f"site {c['site']} {c['type']} command {c['cmd_id']}: never completed"
                           for c in trace["commands"] if c["close"] == -1]  # fmt: skip
    sent = defaultdict(list)
    for a in trace["annotations"]:
        if a["kind"] == "command":
            sent[a["site"]].append(a)
    got = defaultdict(list)
    for c in trace["commands"]:
        got[c["site"]].append(c)
    for site in sorted(set(sent) | set(got)):
        if len(sent[site]) != len(got[site]):
            problems.append(
                f"site {site}: sent {len(sent[site])} commands, {len(got[site])} accepted"
            )
        for ann, cmd in zip(sent[site], got[site], strict=False):
            label = f"site {site} command {ann['id']}"
            names = ("id", "opcode", "dram_addr", "local_addr", "len", "rs1", "rd", "k_dim")
            for name in names:
                field = "cmd_id" if name == "id" else name
                if ann[name] != cmd[field]:
                    problems.append(f"{label}: {name} {cmd[field]!r}, sent {ann[name]!r}")
            if cmd["close"] == -1:
                problems.append(f"{label}: never completed")
    return problems


def _check_residencies(trace: Trace) -> list[str]:
    problems = []
    for r in trace["residencies"]:
        tile = r["tile"]
        label = f"{tile['operand']}[{tile['i']},{tile['j']}] in {r['layer']} ({r['event']})"
        for name in ("site", "addr", "words"):
            if r[name] != r[f"expected_{name}"]:
                problems.append(f"{label}: {name} {r[name]}, expected {r[f'expected_{name}']}")
            if f"from_{name}" in r and r[f"from_{name}"] != r[f"expected_from_{name}"]:
                problems.append(
                    f"{label}: read {name} {r[f'from_{name}']}, "
                    f"expected {r[f'expected_from_{name}']}"
                )
    return problems


def _check_values(trace: Trace) -> list[str]:
    problems = []
    for v in trace["values"]:
        if v["actual"] != v["expected"]:
            tile = v["tile"]
            problems.append(f"{tile['operand']}[{tile['i']},{tile['j']}]: values differ")
    return problems


def summary(trace: Trace) -> dict[str, Any]:
    """Counts of what the trace holds (for reports)."""
    return {name: len(rows) for name, rows in trace.tables.items()} | {
        "cycles": trace.manifest.get("cycles")
    }


_SLOT_ORDER = {"owned": 0, "ready": 1, "reading": 2}


def _check_lanes(trace: Trace) -> list[str]:
    lanes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for iv in trace["intervals"]:
        if iv["resource"].startswith("site "):
            lanes[iv["resource"]].append(iv)
    last = trace.manifest["cycles"] - 1  # the last cycle sampled
    problems: list[str] = []
    for name, ivs in sorted(lanes.items()):
        ivs.sort(key=lambda iv: iv["open"])
        kind = name.partition(":")[2]
        if kind == "collision":
            problems += [f"{name}: a collision at cycle {iv['open']}" for iv in ivs]
        elif " slot " in kind:
            problems += _check_slot(name, ivs, last)
        elif kind.startswith("l1 "):
            problems += _check_stream(name, ivs)
        elif kind.endswith(" overflow"):
            problems += [f"{name}: a row refused at cycle {iv['open']} (the column was full)"
                         for iv in ivs]  # fmt: skip
        elif kind.startswith("res ") and kind != "res rows" and ivs and ivs[-1]["close"] >= last:
            problems.append(f"{name}: {ivs[-1]['what']} rows left at the end")
    return problems


def _check_slot(name: str, ivs: list[dict[str, Any]], last: int) -> list[str]:
    """Runs on back-to-back clocks are one round of the slot, from taken to free."""
    problems: list[str] = []
    rounds: list[list[dict[str, Any]]] = []
    for iv in ivs:
        if rounds and rounds[-1][-1]["close"] + 1 == iv["open"]:
            rounds[-1].append(iv)
        else:
            rounds.append([iv])
    for rnd in rounds:
        states = [iv["what"] for iv in rnd]
        order = [_SLOT_ORDER.get(s, -1) for s in states]
        at = rnd[0]["open"]
        if (states[0] != "owned" or "ready" not in states or order != sorted(set(order))
                or states[-1] == "owned"):  # fmt: skip
            problems.append(f"{name}: {' -> '.join(states)} from cycle {at} (a credit's round "
                            "is owned -> ready, then reading or not, then free)")  # fmt: skip
        if rnd[-1]["close"] >= last:
            problems.append(f"{name}: still {states[-1]} at the end (from cycle {at})")
    return problems


_STREAMING = ("first", "stream", "last", "first+last")


def _check_stream(name: str, ivs: list[dict[str, Any]]) -> list[str]:
    """Walk the lane clock by clock (memory/stream_buffer.py), counting each block's
    elements. A block of ``len`` elements streams them one a clock from the clock after
    its load: each of those clocks shows an element, the last of them may show the next
    block's ``load`` instead (a buffer loaded in the clock its last element leaves), and
    no element streams past them. A ``held`` clock (a block loaded but not streaming) is a
    bubble. (The ``first`` / ``last`` markers are not checked: their meaning depends on the
    dataflow, an OS reduction's ends or a WS/IS stream's start.)"""
    at: dict[int, dict[str, Any]] = {}
    for iv in ivs:
        for c in range(iv["open"], iv["close"] + 1):
            at[c] = iv
    if not at:
        return []
    problems: list[str] = []
    remaining, start, length = 0, 0, 0

    def missing(c: int) -> None:
        k = length - remaining + 1
        problems.append(f"{name}: the {length}-element block loaded at cycle {start} has no "
                        f"element {k} at cycle {c}")  # fmt: skip

    prev = None
    for c in sorted(at):  # the recorded clocks only: idle stretches cost nothing
        if remaining > 0 and prev is not None and c > prev + 1:
            missing(prev + 1)  # the block went quiet with elements still to stream
            remaining = 0
        prev = c
        here = at[c]
        what = here["what"]
        if what == "load":
            if remaining > 1:  # cut short (its last element may share this clock)
                missing(c)
            length = int(here.get("len") or 0)
            remaining, start = length, c
        elif what in _STREAMING:
            if remaining == 0:
                problems.append(f"{name}: an element at cycle {c} that no load brought")
            else:
                remaining -= 1
        elif what == "held":
            problems.append(f"{name}: a bubble at cycle {c} (a block loaded, not streaming)")
    if remaining > 0 and prev is not None:  # the lane ends with elements still to stream
        missing(prev + 1)
    return problems
