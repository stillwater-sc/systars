"""
A tile's own trace (#27, part 1): ``TileRecorder`` on a standalone ``SystolicTop`` running a
differential-harness program, its internal lanes (``trace/lanes.py``) checked against an
independent count of the RTL's port activity, and the bundle's round trip and renderings.
(kpu-gen also checks its SoC recorder's lanes; systars has no SoC.)
"""

from __future__ import annotations

import copy
import json
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict

import pytest

from systars.memory.epilogue import EPILOGUE_FINAL, EPILOGUE_PROFILE_MASK
from systars.trace import Trace
from systars.trace.check import check_trace
from systars.trace.render import svg_timeline, text_report, write_all
from tests.fuzz.program import IS, OS, WS, Program, generate
from tests.fuzz.run import run

ORDER = {"owned": 0, "ready": 1, "reading": 2}


def _observer(counts):
    """A process counting, per clock, what the lanes should show."""

    def observe(top):
        sp, cmb, ex, res = top.scratchpad, top.combiner, top.streamer, top.result_path
        g = top.config.l2_group_rows

        bufs = [(n, b) for n, b in (("A", ex.a_buf), ("B", ex.b_buf), ("W", ex.w_buf)) if b]
        results = [(n, r) for n, r in (("N", res.north), ("S", res.south)) if r]

        async def proc(ctx):
            sigs = (sp.read_req, sp.read_addr, sp.read_b_req, sp.read_b_addr, sp.write_req,
                    sp.write_addr, cmb.write_req, res.combiner_write_req, cmb.write_epilogue,
                    cmb.read_req, cmb.read_ready)  # fmt: skip
            fine = [b.load_valid for _, b in bufs] + [r.out_valid for _, r in results]
            async for _, _, ra, aa, rb, ab, w, wa, cw, cx, ce, cr, crr, *f in ctx.tick().sample(
                *sigs, *fine
            ):
                for (name, _), load in zip(bufs, f, strict=False):
                    counts[f"loads {name}"] += load
                for (name, _), out in zip(results, f[len(bufs) :], strict=True):
                    counts[f"rows {name}"] += out
                for req, addr in ((ra, aa), (rb, ab)):
                    if req:
                        counts["l2 A rd" if addr < g else "l2 B rd"] += 1
                if w:
                    counts["l2 A wr" if wa < g else "l2 B wr"] += 1
                if cw:
                    counts["cmb wr"] += 1
                    if cx and ce & EPILOGUE_FINAL:
                        counts[f"final p{ce & EPILOGUE_PROFILE_MASK}"] += 1
                if cr:
                    counts["cmb rd" if crr else "cmb wait"] += 1

        return proc

    return observe


@pytest.fixture(scope="module", params=[OS, WS, IS], ids=["os", "ws", "is"])
def traced(request, tmp_path_factory):
    """A multi-tile int8 Matmul (two K blocks, so the Combiner merges) with its trace."""
    prog = Program(seed=27, dim=4, dataflow=request.param, M=6, N=8, K=8, buffers=2,
                   profile_slot=2, fmt="int8", mul=3, shift=9, operands=8,
                   decoys=[generate(n).decoys[0] for n in range(4)])  # fmt: skip
    path = tmp_path_factory.mktemp("tile") / "trace.json"
    counts: Counter = Counter()
    result = run(prog, trace=path, observe=_observer(counts))
    assert result.ok, result.problems
    return prog, Trace.load(path), counts


def _lanes(trace):
    by = defaultdict(list)
    for iv in trace["intervals"]:
        by[iv["resource"].partition(":")[2]].append(iv)
    return by


def test_the_trace_is_a_tile_trace_and_checks_clean(traced):
    prog, trace, _ = traced
    assert trace.manifest["scope"] == "tile" and trace.manifest["program"]["seed"] == prog.seed
    assert trace.manifest["geometry"]["tile"]["l2_group_rows"] > 0
    assert check_trace(trace) == []
    assert trace["errors"] == []


def test_each_port_lane_counts_every_active_clock(traced):
    """Each L2 and Combiner port lane's accesses equal the clocks the port was active."""
    _, trace, counts = traced
    lanes = _lanes(trace)
    for lane in ("l2 A rd", "l2 B rd", "l2 A wr", "l2 B wr", "cmb wr"):
        assert sum(iv["accesses"] for iv in lanes[lane]) == counts[lane], lane
        assert counts[lane] > 0, f"{lane}: nothing to check"
    rd = Counter(iv["what"] for iv in lanes["cmb rd"] for _ in range(iv["accesses"]))
    assert (rd["read"], rd["wait"]) == (counts["cmb rd"], counts["cmb wait"])
    # No conflict: the Streamer reads A and B from their own groups
    assert all(iv["what"] == "read" for g in "AB" for iv in lanes[f"l2 {g} rd"])
    # The final rows name the program's profile slot
    finals = [iv for iv in lanes["cmb wr"] if "final" in iv["what"]]
    assert finals and {iv["what"].split()[-1] for iv in finals} == {f"p{2}"}
    assert sum(iv["accesses"] for iv in finals) == counts["final p2"]


def test_runs_do_not_overlap_and_are_back_to_back_merged(traced):
    _, trace, _ = traced
    for name, ivs in _lanes(trace).items():
        if name in ("load", "exec", "store", "exec drain"):
            continue
        ivs = sorted(ivs, key=lambda iv: iv["open"])
        for a, b in zip(ivs, ivs[1:], strict=False):
            assert a["close"] < b["open"], (name, a, b)
            if a["close"] + 1 == b["open"]:  # adjacent runs differ, or they'd be one
                assert (a["what"], a["client"]) != (b["what"], b["client"]), (name, a, b)


def test_each_slot_goes_owned_ready_reading(traced):
    """Every slot lane cycles owned -> ready -> reading (each state at most once a
    cycle round, in order), and both pools are used."""
    _, trace, _ = traced
    lanes = _lanes(trace)
    slots = {name: sorted(ivs, key=lambda iv: iv["open"]) for name, ivs in lanes.items()
             if " slot " in name}  # fmt: skip
    assert any(n.startswith("l2 slot") for n in slots) and any(
        n.startswith("cmb slot") for n in slots
    )
    for name, ivs in slots.items():
        for a, b in zip(ivs, ivs[1:], strict=False):
            if a["close"] + 1 == b["open"]:  # within one round: forward only
                assert ORDER[b["what"]] > ORDER[a["what"]], (name, a, b)


def test_commands_open_when_started_and_drains_follow(traced):
    _, trace, _ = traced
    cmds = trace["commands"]
    assert all(c["accept"] <= c["open"] <= c["close"] for c in cmds)
    queued = [c for c in cmds if c["open"] > c["accept"]]
    assert queued, "Matmul pushes commands ahead: some wait in their queue"
    drains = [iv for iv in trace["intervals"] if iv["resource"].endswith(":exec drain")]
    with_result = [c for c in cmds if c["type"] == "exec" and c["result"] != -1]
    assert drains and len(drains) == len(with_result)
    for c in with_result:
        assert c["result"] >= c["close"]


def test_renderings(traced, tmp_path):
    _, trace, _ = traced
    report = text_report(trace)
    assert report.splitlines()[1].startswith("one tile: 4x4 array")
    assert "NoC links" not in report
    svg = svg_timeline(trace)
    ET.fromstring(svg)
    for label in ("L2 port", "credit slot", "Combiner", "tile command"):
        assert label in svg
    assert "DMA" not in svg and "NoC link" not in svg  # only the groups on the plot
    paths = write_all(trace, tmp_path, "t")
    assert sorted(p.name for p in paths) == ["t.txt", "t_timeline.svg"]
    assert Trace.from_json(json.loads(json.dumps(trace.to_json()))).tables == trace.tables


def test_the_replay_command_writes_a_trace(tmp_path, capsys):
    """``python -m tests.fuzz.run --seed N --trace BUNDLE``: what ``just tile-trace`` and a
    failing fuzz seed use."""
    from tests.fuzz.run import main

    bundle = tmp_path / "deep" / "seed-1.json"
    assert main(["--seed", "1", "--trace", str(bundle)]) == 0
    trace = Trace.load(bundle)
    assert trace.manifest["program"] == generate(1).to_json()
    assert f"trace: {bundle}" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main([])  # a program or --seed


def test_a_result_goes_to_the_oldest_compute_with_its_id():
    """IDs are 8 bits and wrap: a COMPUTE's result never lands on an older PRELOAD (which
    has no result) with the same ID, and goes to the oldest COMPUTE still waiting."""
    from types import SimpleNamespace

    from systars.trace.recorder import TileRecorder
    from systars.util.commands import ExecOpcode

    def cmd(opcode, cid):
        return {"site": 0, "type": "exec", "opcode": int(opcode), "cmd_id": cid, "result": -1}

    cmds = [cmd(ExecOpcode.PRELOAD, 7), cmd(ExecOpcode.COMPUTE, 7), cmd(ExecOpcode.COMPUTE, 7)]
    rec = SimpleNamespace(cycle=40, commands=cmds)
    TileRecorder._result_step(rec, 0, 7)
    assert [c["result"] for c in cmds] == [-1, 40, -1]


def test_a_tile_trace_reports_a_command_that_never_completed():
    trace = Trace({"level": "RTL", "scope": "tile", "producer": "t", "cycles": 10,
                   "geometry": {"tile": {}}})  # fmt: skip
    trace.tables["commands"] = [{"id": 0, "site": 0, "type": "load", "cmd_id": 3, "opcode": 0,
                                 "accept": 1, "open": 2, "close": -1}]  # fmt: skip
    assert "site 0 load command 3: never completed" in check_trace(trace)


def test_the_coverage_report_skips_a_failed_seeds_trace(tmp_path):
    from tests.fuzz.coverage import report

    (tmp_path / "seed-0.json").write_text('{"bins": ["dim=2"], "ok": false, "cycles": 1}')
    (tmp_path / "seed-0.trace.json").write_text('{"format": "kpu-gen-trace"}')
    _, _, runs, failed = report(tmp_path)
    assert (runs, failed) == (1, 1)


# =============================================================================
# The fine lanes (#27, part 2): L1, the result buffers, and the invariant checks
# =============================================================================


def test_the_fine_lanes_count_every_load_and_row(traced):
    """Each L1 buffer group's loads, and each result buffer's rows, as an independent
    process counted them."""
    prog, trace, counts = traced
    lanes = _lanes(trace)
    for buf in "ABW":
        names = [n for n in lanes if n.startswith(f"l1 {buf}")]
        loads = sum(iv["accesses"] for n in names for iv in lanes[n] if iv["what"] == "load")
        assert loads == counts[f"loads {buf}"], buf
    assert counts["loads A"] > 0
    rows = Counter(iv["client"] for iv in lanes["res rows"] for _ in range(iv["accesses"]))
    assert (rows["north"], rows["south"]) == (counts["rows N"], counts["rows S"])
    assert rows["north" if prog.dataflow == OS else "south"] > 0
    # A result column never holds more than its depth (lanes + 1)
    depths = [int(iv["what"]) for n, ivs in lanes.items() if n.startswith("res ") and n != "res rows"
              for iv in ivs]  # fmt: skip
    assert depths and max(depths) <= prog.dim + 1
    assert "collision" not in lanes


def _corrupt(trace, lane_kind, edit):
    """A copy of ``trace`` with ``edit`` applied to its first lane named like ``lane_kind``."""
    bad = copy.deepcopy(trace)
    name = next(n for n in sorted({iv["resource"] for iv in bad["intervals"]})
                if n.partition(":")[2].startswith(lane_kind))  # fmt: skip
    ivs = sorted((iv for iv in bad["intervals"] if iv["resource"] == name), key=lambda i: i["open"])
    edit(bad, ivs)
    return check_trace(bad), name


def test_the_lane_checks_catch_each_fault(traced):
    """Each invariant, broken in a copy of a clean trace, is reported."""
    _, trace, _ = traced
    assert check_trace(trace) == []

    def skip_ready(bad, ivs):  # a slot that is never READY: owned, then reading at once
        ready = next(iv for iv in ivs if iv["what"] == "ready")
        owned = max((iv for iv in ivs if iv["close"] < ready["open"]), key=lambda iv: iv["close"])
        assert owned["what"] == "owned" and owned["close"] + 1 == ready["open"]
        owned["close"] = ready["close"]
        bad.tables["intervals"].remove(ready)

    def lose_elements(bad, ivs):  # a block loaded, its elements never streamed
        load = next(iv for iv in ivs if iv["what"] == "load" and iv["len"] > 0)
        for iv in [iv for iv in ivs if iv["open"] == load["open"] + 1]:
            bad.tables["intervals"].remove(iv)

    def hold(_bad, ivs):  # a clock a loaded block does not stream
        first = next(iv for iv in ivs if iv["what"] != "load")
        first["what"] = "held"

    def unloaded(bad, ivs):  # an element no load brought
        bad.tables["intervals"].remove(next(iv for iv in ivs if iv["what"] == "load"))

    def extra_element(bad, ivs):  # an element past its block's length
        load = next(iv for iv in ivs if iv["what"] == "load" and iv["len"] > 0)
        after = load["open"] + load["len"] + 1
        assert not any(iv["open"] <= after <= iv["close"] for iv in ivs)
        bad.tables["intervals"].append(dict(load, open=after, close=after, what="stream", len=None))

    def stuck(bad, ivs):  # rows left at the end
        ivs[-1]["close"] = bad.manifest["cycles"] - 1

    def collide(bad, ivs):
        bad.tables["intervals"].append(dict(ivs[0], resource=ivs[0]["resource"].split(":")[0]
                                            + ":collision", what="collision"))  # fmt: skip

    for lane, edit, expect in (
        ("l2 slot", skip_ready, "a credit's round"),
        ("l1 A", lose_elements, "has no element 1"),
        ("l1 A", extra_element, "that no load brought"),
        ("l1 A", hold, "a bubble"),
        ("l1 A", unloaded, "that no load brought"),
        ("res ", stuck, "rows left at the end"),
        ("l2 A rd", collide, "a collision"),
    ):
        problems, name = _corrupt(trace, lane, edit)
        assert any(expect in p for p in problems), (lane, expect, problems[:3])


def test_a_full_result_column_refuses_and_is_reported():
    """An arrival at a column with no credit is refused: the depth does not count it, an
    overflow lane records it, and the check reports it."""
    from systars.trace.lanes import FineLanes, LaneRuns

    names = ["collision", "res_S_out", "res_S_in_0", "res_S_credit_0", "res_S_in_1",
             "res_S_credit_1"]  # fmt: skip
    runs = LaneRuns()
    lanes = FineLanes(0, names, runs)
    base = dict.fromkeys(names, 0)
    for cycle, (in0, credit0) in enumerate([(1, 1), (1, 1), (1, 0), (0, 0)]):
        lanes.step(cycle, {**base, "res_S_in_0": in0, "res_S_credit_0": credit0})
    assert lanes.depth[("S", 0)] == 2  # the third row was refused
    over = [r for r in runs.rows if r["resource"] == "site 0:res S0 overflow"]
    assert [(r["open"], r["what"]) for r in over] == [(2, "overflow")]
    trace = Trace({"level": "RTL", "scope": "tile", "producer": "t", "cycles": 10,
                   "geometry": {"tile": {}}})  # fmt: skip
    trace.tables["intervals"] = runs.rows
    assert any("a row refused at cycle 2" in p for p in check_trace(trace))


@pytest.mark.parametrize("dataflow", ["OUTPUT_STATIONARY", "B_STATIONARY"])
def test_a_tile_with_one_dataflow_records_its_fine_lanes(dataflow):
    """An OS-only tile has no south result buffer, a stationary-only one no north one or B
    buffers: the fine lanes record only what the tile has."""
    from systars.config import Dataflow, SystolicConfig
    from systars.top import SystolicTop
    from systars.trace.recorder import TileRecorder

    from ._sim import run_sim

    cfg = SystolicConfig(grid_rows=2, grid_cols=2, tile_rows=1, tile_cols=1,
                         dataflow=Dataflow[dataflow])  # fmt: skip
    top = SystolicTop(cfg)
    rec = TileRecorder(top)

    async def tb(ctx):
        await ctx.tick().repeat(5)

    run_sim(top, tb, processes=[rec.process()], max_cycles=100)
    fine = rec._fine[0]
    assert set(fine.columns) == ({"N"} if dataflow == "OUTPUT_STATIONARY" else {"S"})
    assert rec.trace().manifest["cycles"] >= 5


# =============================================================================
# The performance summary (#85)
# =============================================================================


def test_the_summary_counts_what_the_trace_holds(traced):
    from systars.trace.perf import summary

    prog, trace, counts = traced
    s = summary(trace)
    site = s["sites"]["0"]
    assert s["cycles"] == trace.manifest["cycles"] and s["to_result"] <= s["cycles"]
    for kind in ("load", "exec", "store"):
        cmds = [c for c in trace["commands"] if c["type"] == kind]
        assert site["commands"][kind]["count"] == len(cmds)
        assert 0 < site["commands"][kind]["busy"] <= s["cycles"]
    assert (
        site["l2"]["A"]["reads"] + site["l2"]["B"]["reads"] == counts["l2 A rd"] + counts["l2 B rd"]
    )
    assert (
        site["l2"]["A"]["writes"] + site["l2"]["B"]["writes"]
        == counts["l2 A wr"] + counts["l2 B wr"]
    )
    assert site["combiner"]["writes"] == counts["cmb wr"]
    assert site["combiner"]["finals"] == counts["final p2"] > 0
    assert (site["combiner"]["store_reads"], site["combiner"]["store_waits"]) == (
        counts["cmb rd"], counts["cmb wait"],
    )  # fmt: skip
    # The BlockMover's busy clocks: each clock a LOAD or a STORE ran, counted once
    moves = [c for c in trace["commands"] if c["type"] in ("load", "store")]
    clocks = {cyc for c in moves for cyc in range(c["open"], c["close"] + 1)}
    assert site["blockmover"]["busy"] == len(clocks)
    assert site["blockmover"]["busy"] >= max(site["commands"][k]["busy"] for k in ("load", "store"))
    st = site["streamer"]
    assert st["busy"] == site["commands"]["exec"]["busy"] and st["idle"] >= 0
    assert 0 < site["fabric"]["utilization"] <= 1
    assert "soc" not in s


def test_the_trace_cli_perf_prints_the_summary(traced, tmp_path, capsys):
    from systars.trace.__main__ import main
    from systars.trace.perf import summary

    _, trace, _ = traced
    bundle = trace.save(tmp_path / "t.json")
    assert main([str(bundle), "--perf"]) == 0
    assert json.loads(capsys.readouterr().out) == json.loads(json.dumps(summary(trace)))


def test_to_result_waits_for_a_computes_results(traced):
    """A COMPUTE completes once done with L2: if its results land in the Combiner after
    every completion, ``to_result`` is the clock after they land."""
    from systars.trace.perf import summary

    _, trace, _ = traced
    late = copy.deepcopy(trace)
    last = max(c["close"] for c in late["commands"])
    compute = next(c for c in late["commands"] if c["type"] == "exec" and c["result"] != -1)
    compute["result"] = last + 7
    assert summary(late)["to_result"] == last + 8


def test_perf_does_not_combine_with_check_or_svg(traced, tmp_path, capsys):
    from systars.trace.__main__ import main

    _, trace, _ = traced
    bundle = trace.save(tmp_path / "t.json")
    for extra in (["--check"], ["--svg", str(tmp_path / "svg")]):
        assert main([str(bundle), "--perf", *extra]) == 1
        assert "does not combine" in capsys.readouterr().err
