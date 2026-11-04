"""
Tests for the RTL wavefront trace: the SystolicArray's per-PE, per-cycle
behaviour matches the predicted wavefront (one PE per cycle, the OS results
draining north), a stagger bug is caught, and the renderings are well formed.
"""

import xml.etree.ElementTree as ET

import pytest

from systars import wavefront
from systars.config import SystolicConfig
from systars.wavefront import OS, WS, label_operands, record, render_svg, render_text

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
    A, B = label_operands(cfg, dataflow, 4)
    trace = record(cfg, dataflow, A, B)
    assert trace.check() == []
    # Every (i, j, k) meeting happens exactly once
    stream = [f for f in trace.frames if f.phase == "stream"]
    meetings = sum(int(f.valid.sum()) for f in stream)
    assert meetings == trace.rows * trace.cols * 4


def test_wavefront_moves_one_pe_per_cycle():
    """Every link is registered: PEArray boundaries make no difference to the timing."""
    cfg = config((2, 2, 2, 2))
    A, B = label_operands(cfg, OS, 3)
    trace = record(cfg, OS, A, B)
    frame = next(f for f in trace.frames if f.phase == "stream" and f.cycle == 3)
    ks = [[trace.decode(frame, i, j)[0] for j in range(4)] for i in range(4)]
    # PE(i, j) holds k = 3 - 1 - i - j
    assert ks == [[2, 1, 0, None], [1, 0, None, None], [0, None, None, None], [None] * 4]


def test_results_drain_north_every_other_cycle():
    cfg = config((4, 4, 1, 1))
    A, B = label_operands(cfg, OS, 2)
    trace = record(cfg, OS, A, B)
    column_0 = sorted(t for (i, j), t in trace.result_clocks.items() if j == 0)
    assert column_0 == [2 + 2 * i for i in range(4)]
    drain = [f for f in trace.frames if f.c_valid.any()]
    assert "c" in render_text(trace) and drain


def test_a_stagger_bug_is_caught(monkeypatch):
    """Starting one lane's stream a cycle late breaks the wavefront."""
    real = wavefront.stream_start
    monkeypatch.setattr(wavefront, "stream_start", lambda lane: real(lane) + (lane == 1))
    cfg = config((2, 2, 1, 1))
    A, B = label_operands(cfg, OS, 3)
    trace = record(cfg, OS, A, B)
    problems = trace.check()
    assert any("expected" in p for p in problems)
    assert "result differs from A @ B" in problems
    assert "X" in render_text(trace)  # the mismatching PEs are flagged


@pytest.mark.parametrize("dataflow", [OS, WS], ids=["os", "ws"])
def test_renderings(dataflow):
    cfg = config((2, 2, 2, 2))
    A, B = label_operands(cfg, dataflow, 3)
    trace = record(cfg, dataflow, A, B)
    text = render_text(trace)
    assert ("k0" if dataflow == OS else "m0") in text
    assert text.count("cycle") == len(trace.frames)
    svg = ET.fromstring(render_svg(trace))
    frames = list(svg.iter("{http://www.w3.org/2000/svg}g"))
    assert len(frames) == len(trace.frames)


def test_label_operands_limits():
    cfg = config((4, 4, 2, 2))  # 8x8 PEs
    with pytest.raises(ValueError, match="rows \\* n"):
        label_operands(cfg, OS, 20)


@pytest.mark.parametrize("dataflow", [OS, WS], ids=["os", "ws"])
def test_zero_length_run_terminates(dataflow):
    cfg = config((2, 2, 1, 1))
    A, B = label_operands(cfg, dataflow, 0)
    trace = record(cfg, dataflow, A, B)
    assert trace.check() == []
    assert trace.C.shape == (0, 2)


def test_a_lost_last_marker_is_reported_not_hung(monkeypatch):
    """Without the last marker no PE launches its result: recording still ends, and
    check() reports the missing results."""
    real = wavefront.stream_start
    # Start row 0 so late that its last element never arrives in the recorded window
    monkeypatch.setattr(wavefront, "stream_start", lambda lane: real(lane) + 100 * (lane == 0))
    cfg = config((2, 2, 1, 1))
    A, B = label_operands(cfg, OS, 3)
    problems = record(cfg, OS, A, B).check()
    assert any("wrong clocks" in p for p in problems)
