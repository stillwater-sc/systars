"""
SystolicArray - a pipelined grid of PEArrays.

The array is ``grid_rows x grid_cols`` PEArrays. Every signal that crosses a
PEArray boundary passes through exactly one pipeline register: A going right;
B, the partial sum and the control (``valid``, ``last``) going down or right.
So PEArray ``(ti, tj)`` sees the control presented at the array inputs in cycle
``t`` in cycle ``t + ti + tj``.

The array skews its own inputs and deskews its outputs, so callers present
unskewed vectors:

- A lane ``i`` (PEArray row ``ti = i // tile_rows``) is delayed by ``ti``, and B
  lane ``j`` (PEArray column ``tj = j // tile_cols``) by ``tj``. A[i, k] and
  B[k, j] presented in the same cycle then meet at every PE(i, j), together
  with their ``valid``.
- The WS partial sums leave the bottom edge with column ``j`` ``tj`` cycles late;
  each is delayed by ``grid_cols - 1 - tj`` so that a whole row leaves together.

Timing contract (``latency = (grid_rows - 1) + (grid_cols - 1)``):

- ``out_psum_*`` (WS) for the A vector presented in cycle ``t`` is valid in
  cycle ``t + latency``, together with that vector's ``out_valid`` and
  ``out_last``.
- In OS, the PE at the bottom-right accumulates the last product in the cycle
  ``out_last`` is high, so every result is complete from the next cycle.
- The shift chain (``in_shift``, ``in_d_*`` -> ``out_c_*``) is not pipelined:
  it is one register per PE row, shifted everywhere in the same cycle.
  ``out_c_*`` is the bottom row's chain register.
"""

from amaranth import Module, Signal, signed
from amaranth.lib.wiring import Component, In, Out

from ..config import SystolicConfig
from .pe_array import PEArray


def _delayed(m: Module, value, cycles: int, name: str):
    """``value`` delayed by ``cycles`` registers (the value itself for 0)."""
    for stage in range(cycles):
        reg = Signal.like(value, name=f"{name}_d{stage}")
        m.d.sync += reg.eq(value)
        value = reg
    return value


class SystolicArray(Component):
    """
    Pipelined grid of PEArrays with input skew and output deskew.

    Ports:
        in_a_i: A lane per array row (unskewed)
        in_b_j: B lane per array column (OS operand, unskewed)
        in_d_j: Shift-chain input per column (top edge)
        in_valid, in_last: Control, pipelined with the data
        in_shift: Shift the chain this cycle (broadcast)
        in_control_dataflow: 0 = OS, 1 = WS (static during an operation)

        out_psum_j: WS partial sums at the bottom edge (deskewed)
        out_c_j: Bottom row's shift-chain register (OS results, WS weights)
        out_valid, out_last: Control as it reaches the bottom-right PEArray

    Attributes:
        latency: Cycles from ``in_valid`` to ``out_valid``
    """

    def __init__(self, config: SystolicConfig):
        self.config = config
        self.latency = (config.grid_rows - 1) + (config.grid_cols - 1)
        total_rows = config.grid_rows * config.tile_rows
        total_cols = config.grid_cols * config.tile_cols

        ports = {}
        for i in range(total_rows):
            ports[f"in_a_{i}"] = In(signed(config.input_bits))
        for j in range(total_cols):
            ports[f"in_b_{j}"] = In(signed(config.weight_bits))
            ports[f"in_d_{j}"] = In(signed(config.acc_bits))
        ports["in_valid"] = In(1)
        ports["in_last"] = In(1)
        ports["in_shift"] = In(1)
        ports["in_control_dataflow"] = In(1)
        for j in range(total_cols):
            ports[f"out_psum_{j}"] = Out(signed(config.acc_bits))
            ports[f"out_c_{j}"] = Out(signed(config.acc_bits))
        ports["out_valid"] = Out(1)
        ports["out_last"] = Out(1)
        super().__init__(ports)

    def elaborate(self, _platform):
        m = Module()
        cfg = self.config
        grid_rows, grid_cols = cfg.grid_rows, cfg.grid_cols
        tile_rows, tile_cols = cfg.tile_rows, cfg.tile_cols

        tiles = [[PEArray(cfg) for _ in range(grid_cols)] for _ in range(grid_rows)]
        for tr in range(grid_rows):
            for tc in range(grid_cols):
                m.submodules[f"pe_array_{tr}_{tc}"] = tiles[tr][tc]

        # Control: enters PEArray (0, 0) unregistered, then one register per hop,
        # down the first column and right along each row.
        valid = [[None] * grid_cols for _ in range(grid_rows)]
        last = [[None] * grid_cols for _ in range(grid_rows)]
        for tr in range(grid_rows):
            for tc in range(grid_cols):
                if tr == 0 and tc == 0:
                    v, lst = self.in_valid, self.in_last
                else:
                    src = (tr - 1, 0) if tc == 0 else (tr, tc - 1)
                    v = _delayed(m, valid[src[0]][src[1]], 1, f"valid_{tr}_{tc}")
                    lst = _delayed(m, last[src[0]][src[1]], 1, f"last_{tr}_{tc}")
                valid[tr][tc], last[tr][tc] = v, lst
                m.d.comb += [
                    tiles[tr][tc].in_valid.eq(v),
                    tiles[tr][tc].in_shift.eq(self.in_shift),
                    tiles[tr][tc].in_control_dataflow.eq(self.in_control_dataflow),
                ]
        m.d.comb += [
            self.out_valid.eq(valid[grid_rows - 1][grid_cols - 1]),
            self.out_last.eq(last[grid_rows - 1][grid_cols - 1]),
        ]

        # A: skewed by PEArray row, then registered at each column boundary
        for tr in range(grid_rows):
            for r in range(tile_rows):
                lane = tr * tile_rows + r
                a = _delayed(m, getattr(self, f"in_a_{lane}"), tr, f"skew_a_{lane}")
                for tc in range(grid_cols):
                    if tc > 0:
                        a = _delayed(
                            m, getattr(tiles[tr][tc - 1], f"out_a_{r}"), 1, f"a_{tr}_{tc}_{r}"
                        )
                    m.d.comb += getattr(tiles[tr][tc], f"in_a_{r}").eq(a)

        # B and partial sums: B skewed by PEArray column, both registered at each
        # row boundary. The shift chain crosses boundaries unregistered.
        for tc in range(grid_cols):
            for c in range(tile_cols):
                lane = tc * tile_cols + c
                for tr in range(grid_rows):
                    tile = tiles[tr][tc]
                    if tr == 0:
                        b = _delayed(m, getattr(self, f"in_b_{lane}"), tc, f"skew_b_{lane}")
                        psum = 0
                        d = getattr(self, f"in_d_{lane}")
                    else:
                        above = tiles[tr - 1][tc]
                        b = _delayed(m, getattr(above, f"out_b_{c}"), 1, f"b_{tr}_{tc}_{c}")
                        psum = _delayed(
                            m, getattr(above, f"out_psum_{c}"), 1, f"psum_{tr}_{tc}_{c}"
                        )
                        d = getattr(above, f"out_c_{c}")
                    m.d.comb += [
                        getattr(tile, f"in_b_{c}").eq(b),
                        getattr(tile, f"in_psum_{c}").eq(psum),
                        getattr(tile, f"in_d_{c}").eq(d),
                    ]
                bottom = tiles[grid_rows - 1][tc]
                psum_out = _delayed(
                    m, getattr(bottom, f"out_psum_{c}"), grid_cols - 1 - tc, f"deskew_{lane}"
                )
                m.d.comb += [
                    getattr(self, f"out_psum_{lane}").eq(psum_out),
                    getattr(self, f"out_c_{lane}").eq(getattr(bottom, f"out_c_{c}")),
                ]
        return m
