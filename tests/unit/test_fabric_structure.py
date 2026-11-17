"""The fabric is a pure nearest-neighbour array, checked on the elaborated design.

For a PEArray (its PEs) and a SystolicArray (its PEArrays) every input of a cell is
driven, by a plain wire, from the matching output of its neighbour, or from the
matching edge port when the cell is on that edge. No signal drives more than one
cell input. The one exception is ``in_control_dataflow``, documented static
configuration: the PEArray registers it once and fans it out to its PEs.
"""

from collections import defaultdict

import pytest
from amaranth.hdl import Fragment, Signal
from amaranth.hdl._ast import Assign
from amaranth.lib.wiring import Flow

from systars.config import SystolicConfig
from systars.core.pe_array import EAST, NORTH, SOUTH, PEArray
from systars.core.systolic_array import SystolicArray

STATIC = "control_dataflow"


def cell_inputs(cell):
    """``{port name without in_: signal}`` of a cell's inputs."""
    return {
        name[3:]: getattr(cell, name)
        for name, member in cell.signature.members.items()
        if member.flow == Flow.In
    }


def drivers(outer, cells):
    """``{(r, c, port): rhs}`` for every cell input assigned in ``outer``'s module."""
    fragment = Fragment.get(outer, None)
    owner = {}
    for r, row in enumerate(cells()):
        for c, cell in enumerate(row):
            for port, sig in cell_inputs(cell).items():
                owner[id(sig)] = (r, c, port)
    found = {}
    for domain, statements in fragment.statements.items():
        for stmt in statements:
            if isinstance(stmt, Assign) and id(stmt.lhs) in owner:
                assert domain == "comb", f"{owner[id(stmt.lhs)]} is driven in {domain}"
                found[owner[id(stmt.lhs)]] = stmt.rhs
    return found, cells()


def split(port):
    """``"a_3"`` -> ``("a", 3)``; ``"a"`` -> ``("a", None)`` (a PE's own port)."""
    name, _, index = port.rpartition("_")
    if index.isdigit():
        return name, int(index)
    return port, None


def expected_source(outer, cells, r, c, port, cell_rows, cell_cols):
    """The one signal allowed to drive cell (r, c)'s input ``port``."""
    name, index = split(port)
    rows = len(cells)

    def out(cell, local):
        """The output of ``cell`` that matches the input ``name`` (lane ``local``)."""
        return getattr(cell, f"out_{name}" if local is None else f"out_{name}_{local}")

    if name in EAST:
        lane = r * cell_rows + (index or 0)
        return getattr(outer, f"in_{name}_{lane}") if c == 0 else out(cells[r][c - 1], index)
    lane = c * cell_cols + (index or 0)
    if name in SOUTH:
        return getattr(outer, f"in_{name}_{lane}") if r == 0 else out(cells[r - 1][c], index)
    if name in NORTH:
        return getattr(outer, f"in_{name}_{lane}") if r == rows - 1 else out(cells[r + 1][c], index)
    raise AssertionError(f"unexpected cell input {port}")


def check_mesh(outer, cells, cell_rows, cell_cols):
    """Assert that ``outer`` wires ``cells`` as a pure nearest-neighbour mesh."""
    found, grid = drivers(outer, cells)
    fanout = defaultdict(list)
    for (r, c, port), rhs in found.items():
        assert isinstance(rhs, Signal), f"cell ({r},{c}).in_{port} is driven by logic: {rhs!r}"
        fanout[id(rhs)].append((r, c, port))
    static = set()
    for r, row in enumerate(grid):
        for c, cell in enumerate(row):
            for port in cell_inputs(cell):
                assert (r, c, port) in found, f"cell ({r},{c}).in_{port} is not driven"
                rhs = found[(r, c, port)]
                if port == STATIC:
                    static.add(id(rhs))
                    continue
                want = expected_source(outer, grid, r, c, port, cell_rows, cell_cols)
                assert rhs is want, f"cell ({r},{c}).in_{port} <- {rhs!r}, expected {want!r}"
    for sig, users in fanout.items():
        if sig in static:
            assert {port for _, _, port in users} == {STATIC}
        else:
            assert len(users) == 1, f"one signal drives {users}"
    assert len(static) == 1


SHAPES = [(1, 1, 2, 3), (2, 3, 1, 1), (2, 2, 2, 2), (3, 2, 2, 3)]  # tile rows/cols, grid


@pytest.mark.parametrize("shape", SHAPES)
def test_pe_array_is_nearest_neighbour(shape):
    tr, tc, gr, gc = shape
    array = PEArray(SystolicConfig(tile_rows=tr, tile_cols=tc, grid_rows=gr, grid_cols=gc))
    check_mesh(array, lambda: array.pes, 1, 1)


@pytest.mark.parametrize("shape", SHAPES)
def test_systolic_array_is_nearest_neighbour(shape):
    tr, tc, gr, gc = shape
    array = SystolicArray(SystolicConfig(tile_rows=tr, tile_cols=tc, grid_rows=gr, grid_cols=gc))
    check_mesh(array, lambda: array.pe_arrays, tr, tc)


def test_no_broadcast_weight_shift():
    """The weight chain has no global enable: a PE has no ``in_shift``."""
    array = PEArray(SystolicConfig(tile_rows=2, tile_cols=2))
    Fragment.get(array, None)
    assert "in_shift" not in array.signature.members
    assert "in_w_valid" in array.pes[0][0].signature.members


class _Broadcast(PEArray):
    """A PEArray with one input broadcast past the edge: the check must catch it."""

    def elaborate(self, platform):
        m = super().elaborate(platform)
        m.d.comb += self.pes[1][1].in_valid.eq(self.in_valid_0)
        return m


def test_the_check_catches_a_broadcast():
    """A PE input driven past its neighbour fails the check."""
    array = _Broadcast(SystolicConfig(tile_rows=2, tile_cols=2))
    with pytest.raises(AssertionError, match="in_valid"):
        check_mesh(array, lambda: array.pes, 1, 1)
