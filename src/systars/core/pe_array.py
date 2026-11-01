"""
PEArray - a combinational grid of Processing Elements.

A PEArray is a ``tile_rows x tile_cols`` block of PEs with no registers between
them: A passes right, and B and the partial sum pass down, combinationally. The
shift chain (``in_d`` -> ``out_c``) runs down each column through the PEs' chain
registers. Control (``in_valid``, ``in_shift``, ``in_control_dataflow``) is
broadcast to every PE.

``SystolicArray`` places pipeline registers between PEArrays, so the PEArray
size trades combinational depth against pipeline latency.

Example 2x3 PEArray:
                 in_b/psum/d[0..2]
                    |          |          |
    in_a[0] --> [PE(0,0)] -> [PE(0,1)] -> [PE(0,2)] --> out_a[0]
                    |          |          |
    in_a[1] --> [PE(1,0)] -> [PE(1,1)] -> [PE(1,2)] --> out_a[1]
                    |          |          |
                 out_b/psum/c[0..2]
"""

from amaranth import Module, signed
from amaranth.lib.wiring import Component, In, Out

from ..config import SystolicConfig
from .pe import PE


class PEArray(Component):
    """
    Combinational ``tile_rows x tile_cols`` grid of PEs.

    Ports:
        in_a_i, out_a_i: A operand per row (left edge in, right edge out)
        in_b_j, out_b_j: B operand per column (OS)
        in_psum_j, out_psum_j: Partial sum per column (WS)
        in_d_j, out_c_j: Shift chain per column
        in_valid, in_shift, in_control_dataflow: Broadcast to every PE
    """

    def __init__(self, config: SystolicConfig):
        self.config = config
        ports = {}
        for i in range(config.tile_rows):
            ports[f"in_a_{i}"] = In(signed(config.input_bits))
            ports[f"out_a_{i}"] = Out(signed(config.input_bits))
        for j in range(config.tile_cols):
            ports[f"in_b_{j}"] = In(signed(config.weight_bits))
            ports[f"in_psum_{j}"] = In(signed(config.acc_bits))
            ports[f"in_d_{j}"] = In(signed(config.acc_bits))
            ports[f"out_b_{j}"] = Out(signed(config.weight_bits))
            ports[f"out_psum_{j}"] = Out(signed(config.acc_bits))
            ports[f"out_c_{j}"] = Out(signed(config.acc_bits))
        ports["in_valid"] = In(1)
        ports["in_shift"] = In(1)
        ports["in_control_dataflow"] = In(1)
        super().__init__(ports)

    def elaborate(self, _platform):
        m = Module()
        rows, cols = self.config.tile_rows, self.config.tile_cols

        pes = [[PE(self.config) for _ in range(cols)] for _ in range(rows)]
        self.pes = pes  # the PEs of the last elaboration (read by ``wavefront`` tracing)
        for r in range(rows):
            for c in range(cols):
                pe = pes[r][c]
                m.submodules[f"pe_{r}_{c}"] = pe
                m.d.comb += [
                    pe.in_valid.eq(self.in_valid),
                    pe.in_shift.eq(self.in_shift),
                    pe.in_control_dataflow.eq(self.in_control_dataflow),
                    pe.in_a.eq(getattr(self, f"in_a_{r}") if c == 0 else pes[r][c - 1].out_a),
                ]
                if r == 0:
                    m.d.comb += [
                        pe.in_b.eq(getattr(self, f"in_b_{c}")),
                        pe.in_psum.eq(getattr(self, f"in_psum_{c}")),
                        pe.in_d.eq(getattr(self, f"in_d_{c}")),
                    ]
                else:
                    above = pes[r - 1][c]
                    m.d.comb += [
                        pe.in_b.eq(above.out_b),
                        pe.in_psum.eq(above.out_psum),
                        pe.in_d.eq(above.out_c),
                    ]

        for r in range(rows):
            m.d.comb += getattr(self, f"out_a_{r}").eq(pes[r][cols - 1].out_a)
        for c in range(cols):
            bottom = pes[rows - 1][c]
            m.d.comb += [
                getattr(self, f"out_b_{c}").eq(bottom.out_b),
                getattr(self, f"out_psum_{c}").eq(bottom.out_psum),
                getattr(self, f"out_c_{c}").eq(bottom.out_c),
            ]
        return m
