"""
Tests for the RTL wavefront trace and its animations: the SystolicArray's per-PE,
per-clock behaviour matches the predicted wavefront (one PE per clock, the OS
results draining north), tiles run back to back on the Streamer's schedule, a
stagger bug is caught, and the animations the docs embed are correct, small and
current. (That ``streamer_starts`` is the Streamer's schedule is checked against the
Streamer in ``test_execute_controller.py``.)
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from systars import wavefront
from systars.config import SystolicConfig
from systars.wavefront import (
    OS,
    STAGES,
    WS,
    operands,
    record,
    render_svg,
    render_text,
    stage_svg,
    stage_trace,
    streamer_starts,
)

DOCS = Path(__file__).parents[2] / "docs" / "wavefront"

SHAPES = [
    (1, 1, 1, 1),
    (2, 2, 1, 1),
    (1, 1, 2, 2),
    (2, 2, 2, 2),
    (4, 4, 1, 1),
    (2, 3, 2, 1),
    (3, 2, 1, 2),
]


def config(shape):
    gr, gc, tr, tc = shape
    return SystolicConfig(grid_rows=gr, grid_cols=gc, tile_rows=tr, tile_cols=tc)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dataflow", [OS, WS], ids=["os", "ws"])
def test_rtl_matches_the_predicted_wavefront(shape, dataflow):
    cfg = config(shape)
    A, B = operands(cfg, dataflow, 4)
    trace = record(cfg, dataflow, A, B)
    assert trace.check() == []
    # Every (i, j, k) meeting happens exactly once
    stream = [f for f in trace.frames if f.phase == "stream"]
    meetings = sum(int(f.valid.sum()) for f in stream)
    assert meetings == trace.rows * trace.cols * 4


def test_wavefront_moves_one_pe_per_cycle():
    """Every link is registered: PEArray boundaries make no difference to the timing."""
    cfg = config((2, 2, 2, 2))
    A, B = operands(cfg, OS, 3)
    trace = record(cfg, OS, A, B)
    frame = next(f for f in trace.frames if f.phase == "stream" and f.cycle == 3)
    ks = [[trace.observed(frame, i, j)[0] for j in range(4)] for i in range(4)]
    # PE(i, j) holds k = 3 - 1 - i - j
    assert ks == [[2, 1, 0, None], [1, 0, None, None], [0, None, None, None], [None] * 4]
    # ... and the L1 stream buffers feed row i's element k in clock i + k (the stagger;
    # row 0's three elements have all gone in)
    assert [e[1] if e else None for e in frame.west] == [None, 2, 1, 0]


def test_results_drain_north_every_other_cycle():
    cfg = config((4, 4, 1, 1))
    A, B = operands(cfg, OS, 2)
    trace = record(cfg, OS, A, B)
    column_0 = sorted(t for (n, i, j), t in trace.result_clocks.items() if j == 0)
    assert column_0 == [2 + 2 * i for i in range(4)]
    drain = [f for f in trace.frames if f.c_valid.any()]
    assert "c" in render_text(trace) and drain


def test_streamer_starts():
    # OS: max(K, R, C), and 2R - 1 between reductions' results
    assert streamer_starts(OS, 4, 4, [8, 8, 8]) == [0, 8, 16]
    assert streamer_starts(OS, 4, 4, [1, 1, 1]) == [0, 7, 14]
    assert streamer_starts(OS, 4, 4, [8, 3]) == [0, 12]  # results 2R - 1 apart
    assert streamer_starts(OS, 2, 6, [2, 2]) == [0, 6]  # the wider edge's lanes
    # Stationary: max(M, R)
    assert streamer_starts(WS, 4, 4, [6, 2, 4], first=3) == [3, 9, 13]


@pytest.mark.parametrize("dataflow", [OS, WS], ids=["os", "ws"])
def test_tiles_back_to_back_without_bubbles(dataflow):
    """K >= 2R - 1 (OS) or M >= R (stationary): the L1 buffers feed every clock."""
    cfg = config((4, 4, 1, 1))
    R = 4
    length = 2 * R - 1 if dataflow == OS else R
    A, B = operands(cfg, dataflow, length, tiles=3)
    trace = record(cfg, dataflow, A, B)
    assert trace.check() == []
    assert [t.start for t in trace.tiles] == [0, length, 2 * length]
    feeding = [f.cycle for f in trace.frames if f.phase == "stream" and f.west[0]]
    assert feeding == list(range(3 * length))
    # PE(0, 0) is never idle from its first product to its last
    busy = [f.cycle for f in trace.frames if f.phase == "stream" and f.valid[0, 0]]
    assert busy == list(range(1, 3 * length + 1))
    assert trace.C.shape == (3 * R if dataflow == OS else 3 * length, 4)


def test_short_reductions_wait_for_the_drain():
    """K < 2R - 1: each next reduction's results leave 2R - 1 clocks after the last's."""
    cfg = config((4, 4, 1, 1))
    A, B = operands(cfg, OS, 2, tiles=3)
    trace = record(cfg, OS, A, B)
    assert trace.check() == []
    for n in range(2):
        assert trace.result_clocks[(n + 1, 0, 0)] - trace.result_clocks[(n, 0, 0)] == 7
        # the last result of a reduction and the first of the next never share a clock
        assert trace.result_clocks[(n + 1, 0, 0)] > trace.result_clocks[(n, 3, 0)]


def test_a_stagger_bug_is_caught(monkeypatch):
    """Starting one lane's stream a clock late breaks the wavefront."""
    real = wavefront.stream_start
    monkeypatch.setattr(wavefront, "stream_start", lambda lane: real(lane) + (lane == 1))
    cfg = config((2, 2, 1, 1))
    A, B = operands(cfg, OS, 3)
    trace = record(cfg, OS, A, B)
    problems = trace.check()
    assert any("expected" in p for p in problems)
    assert "result differs from A @ B" in problems
    assert "X" in render_text(trace)  # the mismatching PEs are flagged
    for view in ("value", "index", "front"):
        assert wavefront._MISMATCH in render_svg(trace, view=view)
    assert "departs from the prediction" in render_svg(trace, view="value")


@pytest.mark.parametrize("dataflow", [OS, WS], ids=["os", "ws"])
def test_value_view_shows_the_reduction(dataflow):
    """Stage 1: the running sums are the RTL's, C fills in, and the check is on screen."""
    stage = STAGES[1]
    trace = stage_trace(stage, dataflow)
    assert trace.check() == []
    svg = render_svg(trace, view="value")
    ET.fromstring(svg)
    C = trace.expected()
    for value in C.flat:
        assert f">{value}<" in svg
    assert ("C == A @ B" if dataflow == OS else "C == A @ W") in svg
    if dataflow == OS:
        # PE(i, j)'s running sum after its last product is C[i, j]
        for i in range(2):
            for j in range(2):
                assert f"c = {C[i, j]}" in svg
    else:
        assert "Σ" in svg and "w " in svg


def test_views_and_auto():
    cfg = config((2, 2, 1, 1))
    trace = record(cfg, OS, *operands(cfg, OS, 3))
    assert wavefront.default_view(trace) == "value"
    big = config((4, 4, 2, 2))
    assert wavefront.default_view(record(big, OS, *operands(big, OS, 1))) == "index"
    with pytest.raises(ValueError, match="view"):
        render_svg(trace, view="nope")


@pytest.mark.parametrize("dataflow", [OS, WS], ids=["os", "ws"])
def test_renderings(dataflow):
    cfg = config((2, 2, 2, 2))
    A, B = operands(cfg, dataflow, 3)
    trace = record(cfg, dataflow, A, B)
    text = render_text(trace)
    assert ("k0" if dataflow == OS else "m0") in text
    assert text.count("cycle") == len(trace.frames)
    for view in ("value", "index", "front"):
        svg = ET.fromstring(render_svg(trace, view=view))
        animations = list(svg.iter("{http://www.w3.org/2000/svg}animate"))
        assert animations
        for a in animations:
            times = [float(x) for x in a.get("keyTimes").split(";")]
            assert times[0] == 0 and times == sorted(set(times))
            assert len(times) == len(a.get("values").split(";"))


@pytest.mark.parametrize("dataflow", [OS, WS], ids=["os", "ws"])
def test_the_schedule_stage(dataflow):
    """Stage 3: 16x16, three tiles back to back, checked, and the SVG stays small."""
    stage = STAGES[3]
    trace = stage_trace(stage, dataflow)
    assert (trace.rows, trace.cols) == (16, 16)
    assert trace.check() == []
    starts = [t.start for t in trace.tiles]
    assert np.diff(starts).tolist() == [stage.length] * 2  # no bubble: K >= 2R - 1
    if dataflow == OS:
        for n in range(2):  # reductions' results at least 2R - 1 apart
            assert trace.result_clocks[(n + 1, 0, 0)] - trace.result_clocks[(n, 0, 0)] >= 31
    steady = [f for f in trace.frames if trace.phase_label(f) == "steady"]
    assert steady and all(f.west[0] is not None for f in steady[: 2 * stage.length])
    svg = render_svg(trace, view="front")
    ET.fromstring(svg)
    assert len(svg) < 1_000_000


@pytest.mark.parametrize("dataflow", [OS, WS], ids=["os", "ws"])
@pytest.mark.parametrize("number", sorted(STAGES))
def test_committed_animations_are_current(number, dataflow):
    """docs/wavefront/ holds what ``just wavefronts`` generates (rerun it if this fails)."""
    stage = STAGES[number]
    path = DOCS / stage.filename(dataflow)
    assert path.exists(), f"{path} is missing: run `just wavefronts`"
    assert path.read_text(encoding="utf-8") == stage_svg(stage, dataflow), (
        f"{path} is stale: run `just wavefronts`"
    )


@pytest.mark.parametrize("dataflow", [OS, WS], ids=["os", "ws"])
def test_zero_length_run_terminates(dataflow):
    cfg = config((2, 2, 1, 1))
    A, B = operands(cfg, dataflow, 0)
    trace = record(cfg, dataflow, A, B)
    assert trace.check() == []
    assert trace.C.shape == (0, 2)
    render_svg(trace, view="value")


def test_a_lost_last_marker_is_reported_not_hung(monkeypatch):
    """Without the last marker no PE launches its result: recording still ends, and
    check() reports the missing results."""
    real = wavefront.stream_start
    # Start row 0 so late that its last element never arrives in the recorded window
    monkeypatch.setattr(wavefront, "stream_start", lambda lane: real(lane) + 100 * (lane == 0))
    cfg = config((2, 2, 1, 1))
    A, B = operands(cfg, OS, 3)
    problems = record(cfg, OS, A, B).check()
    assert any("wrong clocks" in p for p in problems)


def test_os_needs_a_b_per_a():
    cfg = config((2, 2, 1, 1))
    A, B = operands(cfg, OS, 2, tiles=2)
    with pytest.raises(ValueError, match="one B tile per A tile"):
        record(cfg, OS, A, B[:1])


def test_narrow_operand_dtypes_dont_wrap_the_expected_product():
    """int8 operand arrays: ``A @ B`` is checked in int64, as the RTL's accumulator sums."""
    cfg = config((2, 2, 1, 1))
    A = np.full((2, 2), 100, dtype=np.int8)
    B = np.full((2, 2), 100, dtype=np.int8)
    trace = record(cfg, OS, A, B)
    assert trace.check() == []
    assert (trace.C == 20000).all()
    W = np.full((2, 2), 100, dtype=np.int8)
    assert record(cfg, WS, A, W).check() == []
