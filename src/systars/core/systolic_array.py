"""
SystolicArray - A pipelined grid of PEArrays forming the systolic matmul array.

The SystolicArray instantiates a grid_rows x grid_cols grid of PEArrays and adds
pipeline registers between boundaries to enable proper systolic data flow for
the operation D = A × B + C.

Key difference from PEArray:
- PEArray: All PE connections are combinational (within-array)
- SystolicArray: Pipeline registers between PEArray boundaries (inter-array)

Data flows through the array in a systolic pattern:
- A (left operand): flows horizontally (left to right) with inter-array registers
- B (right operand/partial sums): flows vertically (top to bottom) with registers
- D (bias/preload): chains through via out_c -> in_d with inter-array registers

Example 2x2 SystolicArray (each box is a PEArray):

           in_b[0..N]    in_b[N..2N]
              |              |
in_a[0..M] -> [PEArray 0,0] -R-> [PEArray 0,1] -> out_a[0..M]
              |                  |
             [R]                [R]           (R = pipeline register)
              |                  |
in_a[M..] ->  [PEArray 1,0] -R-> [PEArray 1,1] -> out_a[M..]
              |                  |
           out_b[0..N]       out_b[N..2N]
           out_c[0..N]       out_c[N..2N]
"""

from amaranth import Module, Signal, signed
from amaranth.lib.wiring import Component, In, Out

from ..config import SystolicConfig
from .pe_array import PEArray


class SystolicArray(Component):
    """
    SystolicArray - A pipelined grid of PEArrays for systolic matmul.

    The SystolicArray instantiates a grid_rows x grid_cols grid of PEArrays
    and wires them together with pipeline registers at boundaries.

    Ports:
        in_a_0..N: Input A operands (one per total row, flows right)
        in_b_0..M: Input B operands/partial sums (one per total column, flows down)
        in_d_0..M: Preload values (one per total column, enters at top)
        in_control_*: Control signals (broadcast/pipelined to arrays)
        in_valid: Data valid signal
        in_id: Operation tag
        in_last: Last element in sequence

        out_a_0..N: Output A operands (right edge)
        out_b_0..M: Output B operands/partial sums (bottom edge)
        out_c_0..M: Accumulated results (bottom edge)
        out_control_*: Pass-through control signals
        out_valid, out_id, out_last: Pass-through metadata

    Parameters:
        config: SystolicConfig with grid dimensions and data widths
    """

    def __init__(self, config: SystolicConfig):
        self.config = config

        # Grid dimensions (in PEArrays)
        grid_rows = config.grid_rows
        grid_cols = config.grid_cols

        # PEArray dimensions (PEs per array)
        tile_rows = config.tile_rows
        tile_cols = config.tile_cols

        # Total dimensions (in PEs)
        total_rows = grid_rows * tile_rows
        total_cols = grid_cols * tile_cols

        # Build port signature dynamically based on grid dimensions
        ports = {}

        # Vector inputs - A (one per total row)
        for i in range(total_rows):
            ports[f"in_a_{i}"] = In(signed(config.input_bits))

        # Vector inputs - B and D (one per total column)
        for j in range(total_cols):
            ports[f"in_b_{j}"] = In(signed(config.weight_bits))
            ports[f"in_d_{j}"] = In(signed(config.acc_bits))

        # Control inputs (broadcast to all tiles)
        ports["in_control_dataflow"] = In(1)
        ports["in_control_propagate"] = In(1)
        ports["in_control_shift"] = In(5)
        ports["in_valid"] = In(1)
        ports["in_id"] = In(8)
        ports["in_last"] = In(1)

        # Vector outputs - A (one per total row, right edge)
        for i in range(total_rows):
            ports[f"out_a_{i}"] = Out(signed(config.input_bits))

        # Vector outputs - B and C (one per total column, bottom edge)
        for j in range(total_cols):
            ports[f"out_b_{j}"] = Out(signed(config.output_bits))
            ports[f"out_c_{j}"] = Out(signed(config.acc_bits))

        # Control outputs (pass-through from bottom-right tile)
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

        grid_rows = cfg.grid_rows
        grid_cols = cfg.grid_cols
        tile_rows = cfg.tile_rows
        tile_cols = cfg.tile_cols

        # Create PEArray grid
        pe_arrays = [[PEArray(cfg) for _ in range(grid_cols)] for _ in range(grid_rows)]

        # Register all PEArrays as submodules
        for mr in range(grid_rows):
            for mc in range(grid_cols):
                m.submodules[f"pe_array_{mr}_{mc}"] = pe_arrays[mr][mc]

        # =================================================================
        # Horizontal (A) Wiring - flows left to right
        # =================================================================
        for mr in range(grid_rows):
            # First tile column gets external inputs
            for tr in range(tile_rows):
                global_row = mr * tile_rows + tr
                m.d.comb += getattr(pe_arrays[mr][0], f"in_a_{tr}").eq(
                    getattr(self, f"in_a_{global_row}")
                )

            # Chain through tile columns with pipeline registers
            for mc in range(1, grid_cols):
                for tr in range(tile_rows):
                    # Pipeline register between tiles
                    pipe_a = Signal(signed(cfg.input_bits), name=f"pipe_a_{mr}_{mc}_{tr}")
                    m.d.sync += pipe_a.eq(getattr(pe_arrays[mr][mc - 1], f"out_a_{tr}"))
                    m.d.comb += getattr(pe_arrays[mr][mc], f"in_a_{tr}").eq(pipe_a)

            # Last tile column outputs to grid edge
            for tr in range(tile_rows):
                global_row = mr * tile_rows + tr
                m.d.comb += getattr(self, f"out_a_{global_row}").eq(
                    getattr(pe_arrays[mr][grid_cols - 1], f"out_a_{tr}")
                )

        # =================================================================
        # Vertical (B, D) Wiring - flows top to bottom
        # =================================================================
        for mc in range(grid_cols):
            # First tile row gets external inputs
            for tc in range(tile_cols):
                global_col = mc * tile_cols + tc
                m.d.comb += getattr(pe_arrays[0][mc], f"in_b_{tc}").eq(
                    getattr(self, f"in_b_{global_col}")
                )
                m.d.comb += getattr(pe_arrays[0][mc], f"in_d_{tc}").eq(
                    getattr(self, f"in_d_{global_col}")
                )

            # Chain through tile rows with pipeline registers
            for mr in range(1, grid_rows):
                for tc in range(tile_cols):
                    # Pipeline registers between tiles
                    pipe_b = Signal(signed(cfg.output_bits), name=f"pipe_b_{mr}_{mc}_{tc}")
                    pipe_d = Signal(signed(cfg.acc_bits), name=f"pipe_d_{mr}_{mc}_{tc}")

                    m.d.sync += pipe_b.eq(getattr(pe_arrays[mr - 1][mc], f"out_b_{tc}"))
                    m.d.sync += pipe_d.eq(getattr(pe_arrays[mr - 1][mc], f"out_c_{tc}"))

                    m.d.comb += getattr(pe_arrays[mr][mc], f"in_b_{tc}").eq(pipe_b)
                    m.d.comb += getattr(pe_arrays[mr][mc], f"in_d_{tc}").eq(pipe_d)

            # Last tile row outputs to grid edge
            for tc in range(tile_cols):
                global_col = mc * tile_cols + tc
                m.d.comb += getattr(self, f"out_b_{global_col}").eq(
                    getattr(pe_arrays[grid_rows - 1][mc], f"out_b_{tc}")
                )
                m.d.comb += getattr(self, f"out_c_{global_col}").eq(
                    getattr(pe_arrays[grid_rows - 1][mc], f"out_c_{tc}")
                )

        # =================================================================
        # Control Signal Distribution
        # Row 0: direct connection (no delay)
        # Row r > 0: use previous row's output (already registered in tile)
        # This synchronizes control with data as it flows down
        # =================================================================
        for mc in range(grid_cols):
            # First row gets direct external control
            m.d.comb += [
                pe_arrays[0][mc].in_control_dataflow.eq(self.in_control_dataflow),
                pe_arrays[0][mc].in_control_propagate.eq(self.in_control_propagate),
                pe_arrays[0][mc].in_control_shift.eq(self.in_control_shift),
                pe_arrays[0][mc].in_valid.eq(self.in_valid),
                pe_arrays[0][mc].in_id.eq(self.in_id),
                pe_arrays[0][mc].in_last.eq(self.in_last),
            ]

            # Subsequent rows chain from previous row's output
            for mr in range(1, grid_rows):
                m.d.comb += [
                    pe_arrays[mr][mc].in_control_dataflow.eq(
                        pe_arrays[mr - 1][mc].out_control_dataflow
                    ),
                    pe_arrays[mr][mc].in_control_propagate.eq(
                        pe_arrays[mr - 1][mc].out_control_propagate
                    ),
                    pe_arrays[mr][mc].in_control_shift.eq(pe_arrays[mr - 1][mc].out_control_shift),
                    pe_arrays[mr][mc].in_valid.eq(pe_arrays[mr - 1][mc].out_valid),
                    pe_arrays[mr][mc].in_id.eq(pe_arrays[mr - 1][mc].out_id),
                    pe_arrays[mr][mc].in_last.eq(pe_arrays[mr - 1][mc].out_last),
                ]

        # =================================================================
        # Control Output - from bottom-right tile (longest path)
        # =================================================================
        bottom_right = pe_arrays[grid_rows - 1][grid_cols - 1]
        m.d.comb += [
            self.out_control_dataflow.eq(bottom_right.out_control_dataflow),
            self.out_control_propagate.eq(bottom_right.out_control_propagate),
            self.out_control_shift.eq(bottom_right.out_control_shift),
            self.out_valid.eq(bottom_right.out_valid),
            self.out_id.eq(bottom_right.out_id),
            self.out_last.eq(bottom_right.out_last),
        ]

        return m
