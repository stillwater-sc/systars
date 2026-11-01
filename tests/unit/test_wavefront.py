"""
Tests for the RTL wavefront trace: the SystolicArray's per-PE, per-cycle
behaviour matches the predicted wavefront, a skew bug is caught, and the
renderings are well formed.
"""

import xml.etree.ElementTree as ET

import pytest

from systars.config import SystolicConfig
from systars.core import systolic_array
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


def test_wavefront_moves_one_pearray_per_cycle():
    """Inside a PEArray everything is combinational: a whole PEArray holds the same k."""
    cfg = config((2, 2, 2, 2))
    A, B = label_operands(cfg, OS, 3)
    trace = record(cfg, OS, A, B)
    first = next(f for f in trace.frames if f.phase == "stream" and f.cycle == 1)
    ks = [[trace.decode(first, i, j)[0] for j in range(4)] for i in range(4)]
    assert ks == [[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, None, None], [0, 0, None, None]]


def test_a_skew_bug_is_caught(monkeypatch):
    """Delaying the B lanes of one PEArray column by an extra cycle breaks the wavefront."""
    real = systolic_array._delayed

    def late_b(m, value, cycles, name):
        if name.startswith("skew_b_") and cycles:
            cycles += 1
        return real(m, value, cycles, name)

    monkeypatch.setattr(systolic_array, "_delayed", late_b)
    cfg = config((2, 2, 1, 1))
    A, B = label_operands(cfg, OS, 3)
    trace = record(cfg, OS, A, B)
    problems = trace.check()
    assert any("expected A[" in p for p in problems)
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
