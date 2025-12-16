"""
PEArray - A combinational grid of Processing Elements.

A PEArray is the fundamental compute block of the systolic matmul array,
containing a configurable grid of PEs (tile_rows x tile_cols). Data flows
through the array in a systolic pattern for D = A × B + C:

- A (left operand): flows horizontally (left to right)
- B (right operand/partial sums): flows vertically (top to bottom)
- D (bias/preload): enters from top, chains through via out_c -> in_d

Control signals are broadcast to all PEs within the array.

Example 2x3 PEArray:
                 in_b[0]    in_b[1]    in_b[2]
                    |          |          |
    in_a[0] --> [PE(0,0)] -> [PE(0,1)] -> [PE(0,2)] --> out_a[0]
                    |          |          |
    in_a[1] --> [PE(1,0)] -> [PE(1,1)] -> [PE(1,2)] --> out_a[1]
                    |          |          |
                 out_b[0]   out_b[1]   out_b[2]
                 out_c[0]   out_c[1]   out_c[2]
"""

from amaranth import Module, signed
from amaranth.lib.wiring import Component, In, Out

from ..config import SystolicConfig
from .pe import PE


class PEArray(Component):
    """
    PEArray - A combinational grid of Processing Elements.

    The PEArray instantiates a tile_rows x tile_cols grid of PEs and wires them
    together in a systolic pattern. All internal connections are combinational.

    Ports:
        in_a_0..N: Input A operands (one per row, flows right)
        in_b_0..M: Input B operands/partial sums (one per column, flows down)
        in_d_0..M: Preload values (one per column, enters at top)
        in_control_*: Control signals (broadcast to all PEs)
        in_valid: Data valid signal
        in_id: Operation tag
        in_last: Last element in sequence

        out_a_0..N: Output A operands (right edge)
        out_b_0..M: Output B operands/partial sums (bottom edge)
        out_c_0..M: Accumulated results (bottom edge)
        out_control_*: Pass-through control signals
        out_valid, out_id, out_last: Pass-through metadata

    Parameters:
        config: SystolicConfig with array dimensions and data widths
    """

    def __init__(self, config: SystolicConfig):
        self.config = config
        rows = config.tile_rows
        cols = config.tile_cols

        # Build port signature dynamically based on tile dimensions
        ports = {}

        # Vector inputs - A (one per row)
        for i in range(rows):
            ports[f"in_a_{i}"] = In(signed(config.input_bits))

        # Vector inputs - B and D (one per column)
        for j in range(cols):
            ports[f"in_b_{j}"] = In(signed(config.weight_bits))
            ports[f"in_d_{j}"] = In(signed(config.acc_bits))

        # Control inputs (broadcast to all PEs)
        ports["in_control_dataflow"] = In(1)
        ports["in_control_propagate"] = In(1)
        ports["in_control_shift"] = In(5)
        ports["in_valid"] = In(1)
        ports["in_id"] = In(8)
        ports["in_last"] = In(1)

        # Vector outputs - A (one per row, right edge)
        for i in range(rows):
            ports[f"out_a_{i}"] = Out(signed(config.input_bits))

        # Vector outputs - B and C (one per column, bottom edge)
        for j in range(cols):
            ports[f"out_b_{j}"] = Out(signed(config.output_bits))
            ports[f"out_c_{j}"] = Out(signed(config.acc_bits))

        # Control outputs (pass-through from bottom-right PE)
        ports["out_control_dataflow"] = Out(1)
        ports["out_control_propagate"] = Out(1)
        ports["out_control_shift"] = Out(5)
        ports["out_valid"] = Out(1)
        ports["out_id"] = Out(8)
        ports["out_last"] = Out(1)

        super().__init__(ports)

    def elaborate(self, _platform):
        m = Module()
        cfg = self.config
        rows = cfg.tile_rows
        cols = cfg.tile_cols

        # Create PE grid
        pes = [[PE(cfg) for _ in range(cols)] for _ in range(rows)]

        # Register all PEs as submodules
        for r in range(rows):
            for c in range(cols):
                m.submodules[f"pe_{r}_{c}"] = pes[r][c]

        # =================================================================
        # Horizontal (A) Wiring - flows left to right
        # =================================================================
        for r in range(rows):
            # First column gets external input
            m.d.comb += pes[r][0].in_a.eq(getattr(self, f"in_a_{r}"))

            # Chain through columns
            for c in range(1, cols):
                m.d.comb += pes[r][c].in_a.eq(pes[r][c - 1].out_a)

            # Last column outputs to tile edge
            m.d.comb += getattr(self, f"out_a_{r}").eq(pes[r][cols - 1].out_a)

        # =================================================================
        # Vertical (B, D) Wiring - flows top to bottom
        # =================================================================
        for c in range(cols):
            # First row gets external inputs
            m.d.comb += pes[0][c].in_b.eq(getattr(self, f"in_b_{c}"))
            m.d.comb += pes[0][c].in_d.eq(getattr(self, f"in_d_{c}"))

            # Chain through rows (D chains from out_c to in_d)
            for r in range(1, rows):
                m.d.comb += pes[r][c].in_b.eq(pes[r - 1][c].out_b)
                m.d.comb += pes[r][c].in_d.eq(pes[r - 1][c].out_c)

            # Last row outputs to tile edge
            m.d.comb += getattr(self, f"out_b_{c}").eq(pes[rows - 1][c].out_b)
            m.d.comb += getattr(self, f"out_c_{c}").eq(pes[rows - 1][c].out_c)

        # =================================================================
        # Control Signal Broadcast - same signal to all PEs
        # =================================================================
        for r in range(rows):
            for c in range(cols):
                m.d.comb += [
                    pes[r][c].in_control_dataflow.eq(self.in_control_dataflow),
                    pes[r][c].in_control_propagate.eq(self.in_control_propagate),
                    pes[r][c].in_control_shift.eq(self.in_control_shift),
                    pes[r][c].in_valid.eq(self.in_valid),
                    pes[r][c].in_id.eq(self.in_id),
                    pes[r][c].in_last.eq(self.in_last),
                ]

        # =================================================================
        # Control Output - from bottom-right PE (longest path)
        # =================================================================
        bottom_right = pes[rows - 1][cols - 1]
        m.d.comb += [
            self.out_control_dataflow.eq(bottom_right.out_control_dataflow),
            self.out_control_propagate.eq(bottom_right.out_control_propagate),
            self.out_control_shift.eq(bottom_right.out_control_shift),
            self.out_valid.eq(bottom_right.out_valid),
            self.out_id.eq(bottom_right.out_id),
            self.out_last.eq(bottom_right.out_last),
        ]

        return m
