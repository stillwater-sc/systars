"""
SystolicArray - the compute fabric: a grid of PEArrays.

The array is ``grid_rows x grid_cols`` PEArrays of ``tile_rows x tile_cols`` PEs,
so ``R = grid_rows * tile_rows`` PE rows and ``C = grid_cols * tile_cols`` PE
columns. Every PE registers its inputs, so a value moves one PE per clock in
every direction, across PEArray boundaries as inside them (see ``PE``).

The array has no skew or deskew registers. Its callers, the L1 stream buffers,
stagger their streams themselves: stream buffer ``i`` starts one clock after
buffer ``i - 1``, which is the skew. At the far edges the L1 result buffers turn
the results back into rows.

Edges (``i`` is a PE row, ``j`` a PE column):

- West, per row: ``in_a_i`` with ``in_valid_i``, ``in_first_i``, ``in_last_i``.
- North, per column: ``in_b_j`` (OS), ``in_w_j`` with ``in_w_valid_j`` (the weight
  chain), and the OS results ``out_c_j`` with ``out_c_valid_j``.
- South, per column: the partial sums ``out_psum_j`` (stationary), valid with
  ``out_psum_valid_j`` (the ``valid`` marker of the bottom row's PE).

Timing (all from the clock a value is on an input edge):

- A presented on row ``i`` in clock ``t`` is in PE(i, j)'s register in clock
  ``t + 1 + j``; B presented on column ``j`` in clock ``t`` is in PE(i, j)'s in
  ``t + 1 + i``. So with row ``i`` streaming ``A[i, k]`` from clock ``s + i`` and
  column ``j`` streaming ``B[k, j]`` from ``s + j``, PE(i, j) multiplies
  ``A[i, k] * B[k, j]`` in clock ``s + 1 + i + j + k``.
- OS: PE(i, j) launches its result in the clock it multiplies the product marked
  ``last``; the result is on ``out_c_j`` ``i`` clocks later. For a reduction of
  ``K`` products starting as above: clock ``s + K + j + 2 i``.
- Stationary: with row ``k`` streaming element ``m`` in clock ``s + k + m``, the
  partial sum over all ``R`` rows for element ``m`` is on ``out_psum_j`` in clock
  ``s + R + j + m``.
- The weight chain: with column ``j`` streaming ``W[R-1-e, j]`` (bottom row first)
  with ``in_w_valid_j`` in clock ``p + j + e``, ``e < R``, PE(i, j)'s shadow
  register holds ``W[i, j]`` from clock ``p + R + 1 + i + j``
  (``weight_ready_clock``). A stationary stream whose first element is marked
  ``first`` swaps it in at PE(i, j) in clock ``s + 1 + i + j``, so a stream may
  start at ``s >= p + R``; the next weights may start loading at ``p' >= s``
  (their first write lands after the swap), while the stream runs.
- ``in_control_dataflow`` is static configuration: each PEArray registers it, so
  it reaches the PEs one clock after the edge (as A does).

``out_collision`` is high (two clocks late: the OR tree is registered) when a PE
launched a result while a result from the south was passing it: never, in a valid
schedule.
"""

from amaranth import Module
from amaranth.lib.wiring import Component

from ..config import SystolicConfig
from .pe_array import PEArray, edge_ports, pe_array_port, wire_mesh


class SystolicArray(Component):
    """
    Grid of PEArrays with a register on every PE-to-PE link.

    Ports: see the module docstring; the port names are those of ``PEArray``
    (``core.pe_array.edge_ports``) for ``R`` rows and ``C`` columns.

    Attributes:
        rows, cols: PE rows and columns (``R`` and ``C``)
    """

    def __init__(self, config: SystolicConfig):
        self.config = config
        self.rows = config.grid_rows * config.tile_rows
        self.cols = config.grid_cols * config.tile_cols
        super().__init__(edge_ports(config, self.rows, self.cols))

    @staticmethod
    def os_result_clock(s: int, k_len: int, i: int, j: int) -> int:
        """Clock PE(i, j)'s OS result is on ``out_c_j`` (streams started as documented)."""
        return s + k_len + j + 2 * i

    def stationary_result_clock(self, s: int, m: int, j: int) -> int:
        """Clock element ``m``'s partial sum is on ``out_psum_j``."""
        return s + self.rows + j + m

    @staticmethod
    def weight_ready_clock(p: int, i: int, j: int, rows: int) -> int:
        """First clock PE(i, j)'s shadow holds its weight (chain streamed from ``p``)."""
        return p + rows + 1 + i + j

    def elaborate(self, _platform):
        m = Module()
        cfg = self.config
        tiles = [[PEArray(cfg) for _ in range(cfg.grid_cols)] for _ in range(cfg.grid_rows)]
        self.pe_arrays = tiles  # the PEArrays of the last elaboration (``wavefront`` tracing)
        for tr in range(cfg.grid_rows):
            for tc in range(cfg.grid_cols):
                m.submodules[f"pe_array_{tr}_{tc}"] = tiles[tr][tc]
        wire_mesh(
            m, self, tiles, pe_array_port, cfg.tile_rows, cfg.tile_cols,
            dataflow=self.in_control_dataflow,
        )  # fmt: skip
        return m
