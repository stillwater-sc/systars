"""
ResultPath - the L1 -> L2 hop: the fabric's results into the Combiner.

In the KPU's CSP model (kpu-sim, ``docs/01-architecture/kpu-execution-model.md``) the
result path takes the fabric's results from L1 into L2, where the Combiner merges them
(#68). This process holds the tile's L1 result buffers and writes their rows into the
Combiner (``memory/combiner.py``): its write port, shared with the BlockMover's inbound
leg, which writes only in the clocks this process leaves free.

- The **north** result buffers (output-stationary) take the C tile the fabric sends
  north at the end of a reduction; the **south** ones (weight- and input-stationary)
  take the partial sums leaving the bottom row. Each turns its columns back into
  row-major rows (``memory/stream_buffer.py``). ``stationary`` picks the side.
- A **result tile** is described by the Streamer when it starts the COMPUTE that
  produces it (``tile_*``): its Combiner rows (``rd``, ``rows``), merge or overwrite
  (``accumulate``), the epilogue code, and the command's ID and ``cmd_sync``. The
  descriptors wait in an in-order queue (the Streamer starts a producing command only
  when the queue has room, ``tile_ready``), and the rows that arrive fill the oldest.

**The process:**

- *Inputs:* the result tiles' descriptors, in order; the fabric's result rows, through
  the L1 result buffers. Its Combiner slot was taken by the Streamer's ``ACQUIRE`` as the
  COMPUTE started.
- *Fires* on every result row as it leaves an L1 result buffer: the row is written into
  the Combiner at once. Results never wait (the fabric has no flow control), so a row
  with no descriptor is an error (``RESULT_WITHOUT_COMMAND``).
- *Outputs:* the rows in the Combiner; once a tile's last row is in, the ``result``
  event (``result_id``, ``result_sync``), on which ``SystolicTop`` emits the Combiner
  slot's tile-ready token (``SIGNAL``) to the BlockMover's outbound leg. A tile of an
  untracked command (``hold``) also completes that command then (``held``).
- *Credits returned:* none (the L1 result buffers have no credits: the schedule keeps
  them from overflowing, and an overflow is an error).

``error`` (sticky) and ``error_kind`` (``L1Error``, from the clock ``error`` rises): a
row without a descriptor, two OS results meeting on a north link
(``array_out_collision``), or a result buffer overflowing. A valid schedule never
raises it.
"""

from amaranth import Cat, Module, Mux, Signal, signed
from amaranth.lib.fifo import SyncFIFO
from amaranth.lib.wiring import Component, In, Out

from ..config import Dataflow, SystolicConfig
from ..errors import FirstErrorKind, L1Error
from ..memory.stream_buffer import ResultBuffers
from .sync import Sync


def result_tile_layout(config: SystolicConfig) -> list[tuple[str, object]]:
    """A result tile's descriptor fields, in the order the queue packs them."""
    rows = config.grid_rows * config.tile_rows
    block = min(config.sp_width // config.input_bits, config.sp_width // config.weight_bits)
    return [("rd", 32), ("accumulate", 1), ("rows", range(block + rows + 1)),
            ("hold", 1), ("id", 8), ("sync", Sync.BITS), ("epilogue", 4)]  # fmt: skip


class ResultPath(Component):
    """The result path (see the module docstring).

    Ports:
        tile_valid, tile_ready, tile_<field>: a result tile's descriptor
            (``result_tile_layout``), taken in the clock ``tile_valid`` is high (the
            Streamer raises it only with ``tile_ready``)
        stationary: The fabric's mode (0 = OS: the north buffers; 1 = the south ones)
        array_out_c_*, array_out_c_valid_*, array_out_psum_*, array_out_psum_valid_*,
            array_out_collision: The fabric's result edges (see ``SystolicArray``)
        combiner_write_req, combiner_write_addr, combiner_write_data,
            combiner_accumulate, combiner_write_epilogue: The Combiner's write port
        result, result_id, result_sync: A tile's results are in the Combiner
        held, held_id, held_sync: An untracked command's results are in: it completes
        busy: A tile is queued or a result buffer holds rows
        error, error_kind: Sticky schedule error, and the first one's ``L1Error``
    """

    # Each result buffer and the kind its error reports (an overflow)
    L1_UNITS = (
        ("north", L1Error.NORTH_RESULT_OVERFLOW),
        ("south", L1Error.SOUTH_RESULT_OVERFLOW),
    )

    def __init__(self, config: SystolicConfig):
        self.config = config
        rows = config.grid_rows * config.tile_rows
        cols = config.grid_cols * config.tile_cols
        self.rows, self.cols = rows, cols
        self.block = min(
            config.sp_width // config.input_bits, config.sp_width // config.weight_bits
        )
        self.supports_os = bool(config.dataflow & Dataflow.OUTPUT_STATIONARY)
        self.supports_stationary = bool(
            config.dataflow & (Dataflow.A_STATIONARY | Dataflow.B_STATIONARY)
        )
        self.layout = result_tile_layout(config)
        ports: dict[str, object] = {"tile_valid": In(1), "tile_ready": Out(1)}
        for name, shape in self.layout:
            ports[f"tile_{name}"] = In(shape)
        ports.update(
            {
                "stationary": In(1),
                "combiner_write_req": Out(1),
                "combiner_write_addr": Out(32),
                "combiner_write_data": Out(signed(config.combiner_width)),
                "combiner_write_epilogue": Out(4),
                "combiner_accumulate": Out(1),
                "result": Out(1),
                "result_id": Out(8),
                "result_sync": Out(Sync.BITS),
                "held": Out(1),
                "held_id": Out(8),
                "held_sync": Out(Sync.BITS),
                "busy": Out(1),
                "error": Out(1),
                "error_kind": Out(4),
                "array_out_collision": In(1),
            }
        )
        for j in range(cols):
            ports[f"array_out_c_{j}"] = In(signed(config.acc_bits))
            ports[f"array_out_c_valid_{j}"] = In(1)
            ports[f"array_out_psum_{j}"] = In(signed(config.acc_bits))
            ports[f"array_out_psum_valid_{j}"] = In(1)
        super().__init__(ports)  # type: ignore[arg-type]

        # L1's result buffers, built here so kpu_gen.trace's probes reach them (None: the
        # dataflow lacks the side)
        C = cols
        self.north = ResultBuffers(C, C + 1, config.acc_bits) if self.supports_os else None
        self.south = ResultBuffers(C, C + 1, config.acc_bits) if self.supports_stationary else None

    def elaborate(self, _platform):
        m = Module()
        cfg = self.config
        R, C, KB = self.rows, self.cols, self.block
        north, south = self.north, self.south
        if north is not None:
            m.submodules.north_results = north
            for j in range(C):
                m.d.comb += [
                    getattr(north, f"in_{j}").eq(getattr(self, f"array_out_c_{j}")),
                    getattr(north, f"in_valid_{j}").eq(
                        ~self.stationary & getattr(self, f"array_out_c_valid_{j}")
                    ),
                ]
        if south is not None:
            m.submodules.south_results = south
            for j in range(C):
                m.d.comb += [
                    getattr(south, f"in_{j}").eq(getattr(self, f"array_out_psum_{j}")),
                    getattr(south, f"in_valid_{j}").eq(
                        self.stationary & getattr(self, f"array_out_psum_valid_{j}")
                    ),
                ]

        # Result tiles in flight: where their rows go. Deep enough for every tile in
        # flight: a command's last row is written about D + R + C + KB clocks after it
        # starts, and streams start R or more apart
        width = sum(Signal(shape).shape().width for _, shape in self.layout)
        depth = 2 + -(-(cfg.spad_read_delay + 6 + R + C + KB) // R)
        m.submodules.results = res_q = SyncFIFO(width=width, depth=max(4, depth))
        m.d.comb += [
            res_q.w_en.eq(self.tile_valid),
            res_q.w_data.eq(Cat(*(getattr(self, f"tile_{name}") for name, _ in self.layout))),
            self.tile_ready.eq(res_q.w_rdy),
        ]
        tile, offset = {}, 0
        for name, shape in self.layout:
            w = Signal(shape).shape().width
            tile[name] = res_q.r_data[offset : offset + w]
            offset += w

        error = Signal()
        local_kind = Signal(4)  # the first of this process's own errors

        def fault(k: L1Error):
            m.d.sync += error.eq(1)
            with m.If(~error):
                m.d.sync += local_kind.eq(k)

        # The rows: L1 result buffers -> Combiner rows
        row_count = Signal(range(KB + R + 1))
        row_valid = Signal()
        row_data = Signal(cfg.combiner_width)
        if north is not None and south is not None:
            m.d.comb += [
                row_valid.eq(Mux(self.stationary, south.out_valid, north.out_valid)),
                row_data.eq(Mux(self.stationary, south.out_data, north.out_data)),
            ]
        else:
            rb = north if north is not None else south
            m.d.comb += [row_valid.eq(rb.out_valid), row_data.eq(rb.out_data)]

        with m.If(row_valid):
            with m.If(res_q.r_rdy):
                m.d.comb += [
                    self.combiner_write_req.eq(1),
                    self.combiner_write_addr.eq(tile["rd"] + row_count),
                    self.combiner_write_data.eq(row_data),
                    self.combiner_accumulate.eq(tile["accumulate"]),
                    self.combiner_write_epilogue.eq(tile["epilogue"]),
                ]
                with m.If(row_count == tile["rows"] - 1):
                    m.d.sync += row_count.eq(0)
                    m.d.comb += [
                        res_q.r_en.eq(1),
                        self.result.eq(1),
                        self.result_id.eq(tile["id"]),
                        self.result_sync.eq(tile["sync"]),
                        self.held.eq(tile["hold"]),
                        self.held_id.eq(tile["id"]),
                        self.held_sync.eq(tile["sync"]),
                    ]
                with m.Else():
                    m.d.sync += row_count.eq(row_count + 1)
            with m.Else():
                fault(L1Error.RESULT_WITHOUT_COMMAND)

        with m.If(self.array_out_collision):
            fault(L1Error.NORTH_COLLISION)

        busy = res_q.r_rdy
        for rb in (north, south):
            if rb is not None:
                busy = busy | rb.busy
        m.d.comb += self.busy.eq(busy)

        # This process's own errors, then each result buffer's: the first to rise names
        # the kind
        buffers = {"north": north, "south": south}
        units = [(error, local_kind)] + [
            (buffers[name].error, k) for name, k in self.L1_UNITS if buffers[name] is not None
        ]
        m.submodules.first_kind = first = FirstErrorKind(len(units), 4)
        for i, (err, k) in enumerate(units):
            m.d.comb += [getattr(first, f"error_{i}").eq(err), getattr(first, f"kind_{i}").eq(k)]
        m.d.comb += [self.error.eq(first.error), self.error_kind.eq(first.kind)]
        return m
