"""
PEArray - a grid of Processing Elements.

A PEArray is a ``tile_rows x tile_cols`` block of PEs, wired to their neighbours.
Every PE registers its own inputs (see ``PE``), so the PEArray adds no registers
and has no timing role: it is a structural grouping (the unit a floorplan or a
generator places), and a value moves one PE per clock across PEArray boundaries
exactly as inside one.

Links (``r`` is a PE row, ``c`` a PE column, inside the PEArray):

- West -> east, per row: ``a``, ``valid``, ``first``, ``last``.
- North -> south, per column: ``b``, ``psum``, ``w`` (the stationary chain).
- South -> north, per column: ``c``, ``c_valid`` (OS results).
- At the south edge, ``out_psum_valid`` per column: ``out_psum`` includes a valid
  product (the bottom PE's A marker).

``in_shift`` and ``in_control_dataflow`` are broadcast to every PE;
``out_collision`` is any PE's collision.

Example 2x3 PEArray:
                       in_b/psum/w[0..2]   out_c[0..2]
                    |  ^       |  ^       |  ^
    in_a[0] --> [PE(0,0)] -> [PE(0,1)] -> [PE(0,2)] --> out_a[0]
                    |  ^       |  ^       |  ^
    in_a[1] --> [PE(1,0)] -> [PE(1,1)] -> [PE(1,2)] --> out_a[1]
                    |  ^       |  ^       |  ^
                  out_b/psum/w[0..2]   in_c[0..2]
"""

from amaranth import Cat, Module, signed
from amaranth.lib.wiring import Component, In, Out

from ..config import SystolicConfig
from .pe import PE

EAST = ("a", "valid", "first", "last")  # west -> east, per row
SOUTH = ("b", "psum", "w")  # north -> south, per column
NORTH = ("c", "c_valid")  # south -> north, per column


def link_shapes(config: SystolicConfig) -> dict:
    """The shape of every link signal."""
    acc = signed(config.acc_bits)
    return {
        "a": signed(config.input_bits),
        "valid": 1,
        "first": 1,
        "last": 1,
        "b": signed(config.weight_bits),
        "psum": acc,
        "w": signed(config.weight_bits),
        "c": acc,
        "c_valid": 1,
    }


def edge_ports(config: SystolicConfig, rows: int, cols: int) -> dict:
    """Signature entries for the edges of a ``rows x cols`` mesh of PEs."""
    shapes = link_shapes(config)
    ports = {}
    for r in range(rows):
        for name in EAST:
            ports[f"in_{name}_{r}"] = In(shapes[name])
            ports[f"out_{name}_{r}"] = Out(shapes[name])
    for c in range(cols):
        for name in SOUTH + NORTH:
            ports[f"in_{name}_{c}"] = In(shapes[name])
            ports[f"out_{name}_{c}"] = Out(shapes[name])
        ports[f"out_psum_valid_{c}"] = Out(1)
    ports["in_shift"] = In(1)
    ports["in_control_dataflow"] = In(1)
    ports["out_collision"] = Out(1)
    return ports


def wire_mesh(m: Module, outer, cells, port, cell_rows: int, cell_cols: int) -> None:
    """Connect a grid of ``cells`` to each other and to the edges of ``outer``.

    ``cells[r][c]`` are PEs or PEArrays, each ``cell_rows x cell_cols`` PEs;
    ``port(cell, direction, name, index)`` returns a cell's port (``index`` is the
    row or column inside the cell, ``None`` for a broadcast or status port).
    """
    grid_rows, grid_cols = len(cells), len(cells[0])
    row_base = [tr * cell_rows for tr in range(grid_rows)]
    col_base = [tc * cell_cols for tc in range(grid_cols)]

    collisions = []
    for tr in range(grid_rows):
        for tc in range(grid_cols):
            cell = cells[tr][tc]
            m.d.comb += [
                port(cell, "in", "shift", None).eq(outer.in_shift),
                port(cell, "in", "control_dataflow", None).eq(outer.in_control_dataflow),
            ]
            collisions.append(port(cell, "out", "collision", None))
            for r in range(cell_rows):
                for name in EAST:
                    src = (
                        getattr(outer, f"in_{name}_{row_base[tr] + r}")
                        if tc == 0
                        else port(cells[tr][tc - 1], "out", name, r)
                    )
                    m.d.comb += port(cell, "in", name, r).eq(src)
                    if tc == grid_cols - 1:
                        m.d.comb += getattr(outer, f"out_{name}_{row_base[tr] + r}").eq(
                            port(cell, "out", name, r)
                        )
            for c in range(cell_cols):
                lane = col_base[tc] + c
                for name in SOUTH:
                    src = (
                        getattr(outer, f"in_{name}_{lane}")
                        if tr == 0
                        else port(cells[tr - 1][tc], "out", name, c)
                    )
                    m.d.comb += port(cell, "in", name, c).eq(src)
                    if tr == grid_rows - 1:
                        m.d.comb += getattr(outer, f"out_{name}_{lane}").eq(
                            port(cell, "out", name, c)
                        )
                if tr == grid_rows - 1:
                    m.d.comb += getattr(outer, f"out_psum_valid_{lane}").eq(
                        port(cell, "out", "psum_valid", c)
                    )
                for name in NORTH:
                    src = (
                        getattr(outer, f"in_{name}_{lane}")
                        if tr == grid_rows - 1
                        else port(cells[tr + 1][tc], "out", name, c)
                    )
                    m.d.comb += port(cell, "in", name, c).eq(src)
                    if tr == 0:
                        m.d.comb += getattr(outer, f"out_{name}_{lane}").eq(
                            port(cell, "out", name, c)
                        )
    # One wide OR, not a chain: a chain of rows * cols terms is as deep as the array
    m.d.comb += outer.out_collision.eq(Cat(*collisions).any())


def _pe_port(pe, direction, name, _index):
    return getattr(pe, f"{direction}_{name}")


class PEArray(Component):
    """
    ``tile_rows x tile_cols`` grid of PEs (see the module docstring for the links).
    """

    def __init__(self, config: SystolicConfig):
        self.config = config
        super().__init__(edge_ports(config, config.tile_rows, config.tile_cols))

    def elaborate(self, _platform):
        m = Module()
        rows, cols = self.config.tile_rows, self.config.tile_cols

        pes = [[PE(self.config) for _ in range(cols)] for _ in range(rows)]
        self.pes = pes  # the PEs of the last elaboration (read by ``wavefront`` tracing)
        for r in range(rows):
            for c in range(cols):
                m.submodules[f"pe_{r}_{c}"] = pes[r][c]
        wire_mesh(m, self, pes, _pe_port, 1, 1)
        return m


def pe_array_port(array, direction, name, index):
    """``wire_mesh`` port accessor for a PEArray."""
    if index is None:
        return getattr(array, f"{direction}_{name}")
    return getattr(array, f"{direction}_{name}_{index}")
