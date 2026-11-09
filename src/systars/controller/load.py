"""
LoadController - moves a tile from L3 into the scratchpad or the accumulator (the
BlockMover's L3 -> L2 leg).

In the KPU's CSP model (kpu-sim, docs/01-architecture/kpu-execution-model.md) the
tile's AXI port is its link to L3: a memory tile in a SoC, the AXI memory model
in tests. A LOAD command moves one tile: ``len`` rows, row ``n`` read from L3 byte
address ``dram_addr + n * stride`` into local row ``sp_addr + n``.

- The destination is the local address (``LocalAddr``): the scratchpad, or, with
  bit 31 set, the accumulator (a bias or an existing C loaded under the results).
  With bit 30 also set, each row is added to the accumulator row instead of
  overwriting it. The accumulator's write port is shared with the Streamer, which
  never waits: an accumulator row is written in a clock the Streamer leaves free.
- Any byte address: a row that starts mid-beat, or spills into the next beat, is
  read as a 1- or 2-beat burst from the beat boundary and shifted into place.
- ``row_bytes`` valid bytes per row (0 = a whole local row); the rest of the local
  row is written with zeros.
- ``stride`` 0 means rows one bus beat apart (consecutive beats).
- No burst crosses a 4 KB boundary (AXI forbids it): a streamed burst ends at the
  boundary and the next one starts there, and a row that spills from the last beat
  of a 4 KB page into the next page is read as two 1-beat bursts.

Opcodes:

- ``MEMCPY``: as above.
- ``TRANSPOSE``: a MEMCPY with a corner turn, for the layouts the Streamer reads
  (one L2 row per L1 stream buffer, #21) and the input-stationary accumulator
  (which holds Cᵀ). The rows land in a corner-turn buffer of one tile (``lanes``
  rows of ``lanes`` elements, where an element is ``input_bits`` wide in the
  scratchpad and ``acc_bits`` in the accumulator) instead of local memory. Once
  the last row is in, local row ``sp_addr + j`` is written with element ``j`` of
  every L3 row: L3 row ``n`` becomes element ``n`` of each local row. That is one
  local row per element of an L3 row, ``ceil(row_bytes * 8 / element bits)`` rows,
  zero past the ``len``-th element. A TRANSPOSE moves at most one tile: ``len`` is
  clamped to ``lanes``. An accumulator corner turn is built only for an array with
  the input-stationary dataflow (the one that needs it); without it, the command
  completes without moving anything.
- ``CLEAR``: writes ``len`` zero rows, with no L3 read (ADR 0002's ``clear``): a
  padded position of a convolution, or an accumulator tile.
- ``BROADCAST``: reads the row at ``dram_addr`` once and writes it to all ``len``
  rows (``stride`` is ignored): a bias row under every output position.
  ``BROADCAST_TRANSPOSE`` is the same with a corner turn, so every local row holds
  ``len`` copies of one element.

State Machine (per row):
    IDLE -> ISSUE_DMA -> STREAM (consecutive beats from a beat boundary: one
                             burst of up to 256 rows, within one 4 KB page)
                       -> RECV_LO [-> (ISSUE_HI ->) RECV_HI] -> WRITE_SP (any other
                             row; ISSUE_HI when its second beat is in the next page;
                             a BROADCAST stays in WRITE_SP for every row)
         -> (next burst | TURN (TRANSPOSE: write the tile's columns) | DONE)
    IDLE -> CLEAR (one zero row per clock) -> DONE

Data Flow:
    L3 (via StreamReader) --> LoadController --> Scratchpad | Accumulator
"""

from amaranth import Cat, Const, Module, Mux, Signal
from amaranth.lib.wiring import Component, In, Out

from ..config import Dataflow, SystolicConfig
from ..dma.axi import AXI_4KB
from ..memory.local_addr import LocalAddr
from ..util.commands import DmaOpcode


class LoadController(Component):
    """
    Load controller for DRAM -> scratchpad transfers.

    Coordinates data movement from external memory through the StreamReader
    DMA engine into the scratchpad memory.

    Ports:
        Command Interface:
            cmd_valid: Command available
            cmd_ready: Controller ready to accept command
            cmd_opcode: Operation code (MEMCPY, TRANSPOSE, CLEAR, BROADCAST,
                BROADCAST_TRANSPOSE; see the module docstring)
            cmd_dram_addr: Source byte address of row 0 in L3 (64-bit, any alignment)
            cmd_sp_addr: Destination local address (32-bit local format: bit 31 =
                the accumulator, bit 30 = add to it)
            cmd_len: Rows to move (0 completes at once)
            cmd_stride: Bytes between source rows (0 = one bus beat)
            cmd_row_bytes: Valid bytes per row (0 = a whole local row)
            cmd_id: Command ID for tracking

        DMA Interface (to StreamReader):
            dma_req_valid: Request valid
            dma_req_ready: DMA ready to accept
            dma_req_addr: DRAM address (64-bit)
            dma_req_len: Burst length (AXI4 style: 0=1 beat)

            dma_resp_valid: Response data valid
            dma_resp_ready: Controller ready to accept
            dma_resp_data: Read data (dma_buswidth bits)
            dma_resp_last: Last beat of burst

        Scratchpad Interface:
            sp_write_en: Write enable
            sp_write_addr: Write address (local format)
            sp_write_data: Write data (sp_width bits)
            sp_write_mask: Byte mask for partial writes

        Accumulator Interface (shared with the Streamer, which has priority):
            acc_write_en: Write request
            acc_write_ready: The port is free this clock (the write happens)
            acc_write_addr: Write address (local format)
            acc_write_data: Write data (acc_width bits)
            acc_write_accumulate: Add to the row instead of overwriting it

        Status:
            busy: Controller is processing a command
            completed: Command completed this cycle
            completed_id: ID of completed command
    """

    def __init__(self, config: SystolicConfig):
        self.config = config
        buswidth = config.dma_buswidth
        sp_width = config.sp_width
        if sp_width > buswidth:
            # One scratchpad row is written per DMA beat; a wider row would only
            # receive the low dma_buswidth bits
            raise ValueError(
                f"LoadController: scratchpad row ({sp_width} bits) is wider than "
                f"dma_buswidth ({buswidth}); set dma_buswidth >= {sp_width}"
            )
        beat = buswidth // 8
        if beat < 2 or beat & (beat - 1):
            raise ValueError(
                f"LoadController: dma_buswidth must be a power-of-two bytes, got {buswidth}"
            )
        if beat > AXI_4KB:
            raise ValueError(f"LoadController: a {beat}-byte beat is wider than a 4 KB page")
        if config.acc_width > buswidth:
            raise ValueError(
                f"LoadController: accumulator row ({config.acc_width} bits) is wider than "
                f"dma_buswidth ({buswidth}); set dma_buswidth >= {config.acc_width}"
            )
        mask_width = sp_width // 8
        self.lanes = sp_width // config.input_bits  # elements per scratchpad row
        self.acc_lanes = config.acc_width // config.acc_bits  # elements per accumulator row
        # The accumulator corner turn is for the input-stationary accumulator (Cᵀ)
        self.acc_turn = bool(config.dataflow & Dataflow.A_STATIONARY)

        super().__init__(
            {
                # Command interface
                "cmd_valid": In(1),
                "cmd_ready": Out(1),
                "cmd_opcode": In(8),
                "cmd_dram_addr": In(64),
                "cmd_sp_addr": In(32),
                "cmd_len": In(16),  # Rows to move
                "cmd_stride": In(32),  # Bytes between source rows (0 = one beat)
                "cmd_row_bytes": In(16),  # Valid bytes per row (0 = whole row)
                "cmd_id": In(8),
                # DMA interface (to StreamReader)
                "dma_req_valid": Out(1),
                "dma_req_ready": In(1),
                "dma_req_addr": Out(64),
                "dma_req_len": Out(8),  # AXI4 style: 0=1 beat
                "dma_resp_valid": In(1),
                "dma_resp_ready": Out(1),
                "dma_resp_data": In(buswidth),
                "dma_resp_last": In(1),
                # Scratchpad write interface
                "sp_write_en": Out(1),
                "sp_write_addr": Out(32),
                "sp_write_data": Out(sp_width),
                "sp_write_mask": Out(mask_width),
                # Accumulator write interface
                "acc_write_en": Out(1),
                "acc_write_ready": In(1),
                "acc_write_addr": Out(32),
                "acc_write_data": Out(config.acc_width),
                "acc_write_accumulate": Out(1),
                # Status
                "busy": Out(1),
                "completed": Out(1),
                "completed_id": Out(8),
            }
        )

    def elaborate(self, _platform):
        m = Module()
        cfg = self.config

        beat = cfg.dma_buswidth // 8
        sp_bytes = cfg.sp_width // 8
        acc_bytes = cfg.acc_width // 8
        row_width = max(cfg.sp_width, cfg.acc_width)  # the widest local row
        rw_bytes = row_width // 8
        off_bits = (beat - 1).bit_length()

        # Registered command parameters
        row_addr = Signal(64)  # byte address of the current row
        sp_addr_reg = Signal(LocalAddr.DATA_BITS)  # first local row (no flags)
        len_reg = Signal(16)
        stride_reg = Signal(32)
        row_bytes = Signal(16)
        cmd_id_reg = Signal(8)
        row = Signal(16)
        to_acc = Signal()  # the destination is the accumulator
        add = Signal()  # add the rows to the accumulator's
        broadcast = Signal()  # one L3 row, written to every row

        # TRANSPOSE: the corner-turn buffer (one tile) and the columns left to write.
        # It is never cleared: a row is zero past row_bytes, and the turn shifts out
        # every element up to row_bytes, so the buffer is empty after each TRANSPOSE
        eb, acc_eb = cfg.input_bits, cfg.acc_bits
        lanes, acc_lanes = self.lanes, self.acc_lanes
        turn_rows = max(lanes, acc_lanes) if self.acc_turn else lanes
        turn_width = row_width if self.acc_turn else cfg.sp_width
        transpose = Signal()
        turn = [Signal(turn_width, name=f"turn_{n}") for n in range(turn_rows)]
        col = Signal(16)
        cols = Signal(16)  # local rows a TRANSPOSE writes

        lo = Signal(cfg.dma_buswidth)  # first beat of the row
        hi = Signal(cfg.dma_buswidth)  # second beat, when the row spills over

        offset = Signal(off_bits)  # row's byte offset within its first beat
        two_beats = Signal()
        m.d.comb += [
            offset.eq(row_addr[:off_bits]),
            two_beats.eq(offset + row_bytes > beat),
        ]

        # The row: shift the two beats down by the byte offset, keep row_bytes bytes
        window = Signal(2 * cfg.dma_buswidth)
        m.d.comb += window.eq(Cat(lo, hi) >> (offset * 8))
        keep = Cat(*[Mux(i < row_bytes, Const(0xFF, 8), Const(0, 8)) for i in range(rw_bytes)])
        row_data = Signal(row_width)
        m.d.comb += row_data.eq(window[:row_width] & keep)

        m.d.comb += [
            self.cmd_ready.eq(0),
            self.dma_req_valid.eq(0),
            self.dma_req_len.eq(0),
            self.dma_resp_ready.eq(0),
            self.sp_write_en.eq(0),
            self.sp_write_addr.eq(0),
            self.sp_write_data.eq(0),
            self.sp_write_mask.eq(0),
            self.acc_write_en.eq(0),
            self.acc_write_addr.eq(0),
            self.acc_write_data.eq(0),
            self.acc_write_accumulate.eq(0),
            self.completed.eq(0),
            self.completed_id.eq(0),
        ]

        # Rows that are consecutive beats from a beat boundary stream as bursts of up
        # to 256 beats, ending at the next 4 KB boundary; any other row is its own 1-
        # or 2-beat burst, or two 1-beat bursts when its beats are in different pages
        contiguous = (offset == 0) & (stride_reg == beat) & ~broadcast
        remaining = Signal(16)
        m.d.comb += remaining.eq(len_reg - row)
        page_beats = Signal(range(AXI_4KB // beat + 1))  # beats left in row_addr's page
        m.d.comb += page_beats.eq((AXI_4KB >> off_bits) - row_addr[off_bits:12])
        capped = Mux(remaining > 256, 256, remaining)  # AXI4: at most 256 beats
        burst_rows = Signal(range(257))
        m.d.comb += burst_rows.eq(Mux(page_beats < capped, page_beats, capped))
        split = Signal()  # a 2-beat row whose second beat starts a new page
        m.d.comb += split.eq(two_beats & (page_beats == 1))
        aligned = Cat(Const(0, off_bits), row_addr[off_bits:])
        m.d.comb += self.dma_req_addr.eq(aligned)

        # A local row can be written this clock: the scratchpad always, the
        # accumulator when the Streamer leaves its write port free
        writable = Signal()
        m.d.comb += writable.eq(~to_acc | self.acc_write_ready)

        def write_row(addr, data):
            """Write local row ``addr`` (lands only when ``writable``)."""
            with m.If(to_acc):
                m.d.comb += [
                    self.acc_write_en.eq(self.acc_write_ready),
                    self.acc_write_addr.eq(Cat(addr[: LocalAddr.DATA_BITS], Const(0b100, 3))),
                    self.acc_write_data.eq(data),
                    self.acc_write_accumulate.eq(add),
                ]
            with m.Else():
                m.d.comb += [
                    self.sp_write_en.eq(1),
                    self.sp_write_addr.eq(addr),
                    self.sp_write_data.eq(data),
                    self.sp_write_mask.eq((1 << sp_bytes) - 1),
                ]

        def put_row(data):
            """Row ``row`` arrived: into local memory, or the corner-turn buffer."""
            with m.If(transpose):
                for n in range(turn_rows):
                    with m.If(row == n):
                        m.d.sync += turn[n].eq(data)
            with m.Else():
                write_row(sp_addr_reg + row, data)

        # A row is taken once it is written (or buffered for the corner turn)
        taken = transpose | writable

        def finish():
            """After the last row: write the tile's columns (TRANSPOSE), or complete."""
            with m.If(transpose):
                m.next = "TURN"
            with m.Else():
                m.next = "DONE"

        op = self.cmd_opcode
        is_transpose = (op == DmaOpcode.TRANSPOSE) | (op == DmaOpcode.BROADCAST_TRANSPOSE)
        is_broadcast = (op == DmaOpcode.BROADCAST) | (op == DmaOpcode.BROADCAST_TRANSPOSE)
        is_clear = op == DmaOpcode.CLEAR
        accepted = (op == DmaOpcode.MEMCPY) | is_transpose | is_broadcast | is_clear
        cmd_acc = LocalAddr.is_acc(self.cmd_sp_addr)
        # A corner turn into the accumulator needs its buffer (see the docstring)
        unbuilt = is_transpose & cmd_acc if not self.acc_turn else Const(0)
        width_bytes = Mux(cmd_acc, acc_bytes, sp_bytes)
        cmd_row_bytes = Mux(
            (self.cmd_row_bytes == 0) | (self.cmd_row_bytes > width_bytes),
            width_bytes,
            self.cmd_row_bytes,
        )
        cmd_lanes = Mux(cmd_acc, acc_lanes, lanes)
        cmd_eb = Mux(cmd_acc, acc_eb, eb)

        with m.FSM(init="IDLE") as fsm:
            with m.State("IDLE"):
                m.d.comb += self.cmd_ready.eq(1)
                with m.If(self.cmd_valid & accepted):
                    cmd_len = Mux(
                        is_transpose & (self.cmd_len > cmd_lanes), cmd_lanes, self.cmd_len
                    )
                    m.d.sync += [
                        transpose.eq(is_transpose),
                        broadcast.eq(is_broadcast),
                        to_acc.eq(cmd_acc),
                        add.eq(cmd_acc & LocalAddr.accumulate(self.cmd_sp_addr)),
                        col.eq(0),
                        cols.eq((cmd_row_bytes * 8 + cmd_eb - 1) // cmd_eb),
                        row_addr.eq(self.cmd_dram_addr),
                        sp_addr_reg.eq(LocalAddr.data(self.cmd_sp_addr)),
                        len_reg.eq(cmd_len),
                        stride_reg.eq(Mux(self.cmd_stride == 0, beat, self.cmd_stride)),
                        row_bytes.eq(cmd_row_bytes),
                        cmd_id_reg.eq(self.cmd_id),
                        row.eq(0),
                    ]
                    with m.If((self.cmd_len == 0) | unbuilt):
                        m.next = "DONE"
                    with m.Elif(is_clear):
                        m.next = "CLEAR"
                    with m.Else():
                        m.next = "ISSUE_DMA"

            with m.State("ISSUE_DMA"):
                m.d.comb += [
                    self.dma_req_valid.eq(1),
                    self.dma_req_len.eq(Mux(contiguous, burst_rows - 1, two_beats & ~split)),
                ]
                with m.If(self.dma_req_ready):
                    with m.If(contiguous):
                        m.next = "STREAM"
                    with m.Else():
                        m.next = "RECV_LO"

            with m.State("STREAM"):
                # One beat per row, written as it arrives (held while the
                # accumulator's port is busy)
                m.d.comb += self.dma_resp_ready.eq(taken)
                with m.If(self.dma_resp_valid & taken):
                    put_row(self.dma_resp_data[:row_width] & keep)
                    m.d.sync += [row.eq(row + 1), row_addr.eq(row_addr + beat)]
                    with m.If(self.dma_resp_last):
                        with m.If(row + 1 >= len_reg):
                            finish()
                        with m.Else():
                            m.next = "ISSUE_DMA"

            with m.State("RECV_LO"):
                m.d.comb += self.dma_resp_ready.eq(1)
                with m.If(self.dma_resp_valid):
                    m.d.sync += [lo.eq(self.dma_resp_data), hi.eq(0)]
                    with m.If(split):
                        m.next = "ISSUE_HI"
                    with m.Elif(two_beats):
                        m.next = "RECV_HI"
                    with m.Else():
                        m.next = "WRITE_SP"

            # The row's second beat starts the next 4 KB page: a burst of its own
            with m.State("ISSUE_HI"):
                m.d.comb += [
                    self.dma_req_valid.eq(1),
                    self.dma_req_addr.eq(aligned + beat),
                    self.dma_req_len.eq(0),
                ]
                with m.If(self.dma_req_ready):
                    m.next = "RECV_HI"

            with m.State("RECV_HI"):
                m.d.comb += self.dma_resp_ready.eq(1)
                with m.If(self.dma_resp_valid):
                    m.d.sync += hi.eq(self.dma_resp_data)
                    m.next = "WRITE_SP"

            # A BROADCAST writes its one row to every row from here
            with m.State("WRITE_SP"):
                put_row(row_data)
                with m.If(taken):
                    m.d.sync += row.eq(row + 1)
                    with m.If(row + 1 >= len_reg):
                        finish()
                    with m.Elif(~broadcast):
                        m.d.sync += row_addr.eq(row_addr + stride_reg)
                        m.next = "ISSUE_DMA"

            # TRANSPOSE: local row sp_addr + col is element 0 of every buffered row;
            # the buffer then shifts down one element
            with m.State("TURN"):
                with m.If(to_acc):
                    if self.acc_turn:
                        write_row(sp_addr_reg + col, Cat(*[t[:acc_eb] for t in turn[:acc_lanes]]))
                        with m.If(writable):
                            m.d.sync += [t.eq(t >> acc_eb) for t in turn]
                with m.Else():
                    write_row(sp_addr_reg + col, Cat(*[t[:eb] for t in turn[:lanes]]))
                    m.d.sync += [t.eq(t >> eb) for t in turn]
                with m.If(writable):
                    m.d.sync += col.eq(col + 1)
                    with m.If(col + 1 >= cols):
                        m.next = "DONE"

            # CLEAR: zero rows, no L3 read
            with m.State("CLEAR"):
                write_row(sp_addr_reg + row, 0)
                with m.If(writable):
                    m.d.sync += row.eq(row + 1)
                    with m.If(row + 1 >= len_reg):
                        m.next = "DONE"

            with m.State("DONE"):
                m.d.comb += [self.completed.eq(1), self.completed_id.eq(cmd_id_reg)]
                m.next = "IDLE"

        m.d.comb += self.busy.eq(~fsm.ongoing("IDLE"))
        return m
