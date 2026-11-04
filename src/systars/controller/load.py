"""
LoadController - moves a tile from L3 into the scratchpad (the BlockMover's L3 -> L2 leg).

In the KPU's CSP model (kpu-sim, docs/01-architecture/kpu-execution-model.md) the
tile's AXI port is its link to L3: a memory tile in a SoC, the AXI memory model
in tests. A LOAD command moves one tile: ``len`` rows, row ``n`` read from L3 byte
address ``dram_addr + n * stride`` into scratchpad row ``sp_addr + n``.

- Any byte address: a row that starts mid-beat, or spills into the next beat, is
  read as a 1- or 2-beat burst from the beat boundary and shifted into place.
- ``row_bytes`` valid bytes per row (0 = a whole scratchpad row); the rest of
  the scratchpad row is written with zeros.
- ``stride`` 0 means rows one bus beat apart (consecutive beats).

``TRANSPOSE`` is a MEMCPY with a corner turn, for the layouts the Streamer reads
(one L2 row per L1 stream buffer, #21). The rows land in a corner-turn buffer of
one tile (``lanes`` rows of ``lanes`` elements, ``input_bits`` each, where
``lanes = sp_width // input_bits``) instead of the scratchpad. Once the last row
is in, scratchpad row ``sp_addr + j`` is written with element ``j`` of every L3
row: L3 row ``n`` becomes element ``n`` of each scratchpad row. That is one
scratchpad row per element of an L3 row, ``ceil(row_bytes * 8 / input_bits)``
rows, zero past the ``len``-th element. A TRANSPOSE moves at most one tile:
``len`` is clamped to ``lanes``.

State Machine (per row):
    IDLE -> ISSUE_DMA -> STREAM (consecutive beats from a beat boundary: one
                             burst of up to 256 rows)
                       -> RECV_LO [-> RECV_HI] -> WRITE_SP (any other row)
         -> (next burst | TURN (TRANSPOSE: write the tile's columns) | DONE)

Data Flow:
    L3 (via StreamReader) --> LoadController --> Scratchpad
"""

from amaranth import Cat, Const, Module, Mux, Signal
from amaranth.lib.wiring import Component, In, Out

from ..config import SystolicConfig
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
            cmd_opcode: Operation code (MEMCPY, or TRANSPOSE: with a corner turn)
            cmd_dram_addr: Source byte address of row 0 in L3 (64-bit, any alignment)
            cmd_sp_addr: Destination address in scratchpad (32-bit local format)
            cmd_len: Rows to move (0 completes at once)
            cmd_stride: Bytes between source rows (0 = one bus beat)
            cmd_row_bytes: Valid bytes per row (0 = a whole scratchpad row)
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
        mask_width = sp_width // 8
        self.lanes = sp_width // config.input_bits  # elements per scratchpad row

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
        off_bits = (beat - 1).bit_length()

        # Registered command parameters
        row_addr = Signal(64)  # byte address of the current row
        sp_addr_reg = Signal(32)
        len_reg = Signal(16)
        stride_reg = Signal(32)
        row_bytes = Signal(16)
        cmd_id_reg = Signal(8)
        row = Signal(16)

        # TRANSPOSE: the corner-turn buffer (one tile) and the columns left to write.
        # It is never cleared: a row is zero past row_bytes, and the turn shifts out
        # every element up to row_bytes, so the buffer is empty after each TRANSPOSE
        eb = cfg.input_bits
        lanes = self.lanes
        transpose = Signal()
        turn = [Signal(cfg.sp_width, name=f"turn_{n}") for n in range(lanes)]
        col = Signal(16)
        cols = Signal(16)  # scratchpad rows a TRANSPOSE writes

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
        keep = Cat(*[Mux(i < row_bytes, Const(0xFF, 8), Const(0, 8)) for i in range(sp_bytes)])
        row_data = Signal(cfg.sp_width)
        m.d.comb += row_data.eq(window[: cfg.sp_width] & keep)

        m.d.comb += [
            self.cmd_ready.eq(0),
            self.dma_req_valid.eq(0),
            self.dma_req_len.eq(0),
            self.dma_resp_ready.eq(0),
            self.sp_write_en.eq(0),
            self.sp_write_addr.eq(0),
            self.sp_write_data.eq(0),
            self.sp_write_mask.eq(0),
            self.completed.eq(0),
            self.completed_id.eq(0),
        ]

        # Rows that are consecutive beats from a beat boundary stream as bursts of up
        # to 256 beats; any other row is its own 1- or 2-beat burst
        contiguous = (offset == 0) & (stride_reg == beat)
        remaining = Signal(16)
        m.d.comb += remaining.eq(len_reg - row)
        burst_rows = Mux(remaining > 256, 256, remaining)
        m.d.comb += self.dma_req_addr.eq(Cat(Const(0, off_bits), row_addr[off_bits:]))

        def write_sp(addr, data):
            m.d.comb += [
                self.sp_write_en.eq(1),
                self.sp_write_addr.eq(addr),
                self.sp_write_data.eq(data),
                self.sp_write_mask.eq((1 << sp_bytes) - 1),
            ]

        def put_row(data):
            """Row ``row`` arrived: into the scratchpad, or the corner-turn buffer."""
            with m.If(transpose):
                for n in range(lanes):
                    with m.If(row == n):
                        m.d.sync += turn[n].eq(data)
            with m.Else():
                write_sp(sp_addr_reg + row, data)

        def finish():
            """After the last row: write the tile's columns (TRANSPOSE), or complete."""
            with m.If(transpose):
                m.next = "TURN"
            with m.Else():
                m.next = "DONE"

        accepted = (self.cmd_opcode == DmaOpcode.MEMCPY) | (self.cmd_opcode == DmaOpcode.TRANSPOSE)
        is_transpose = self.cmd_opcode == DmaOpcode.TRANSPOSE
        cmd_row_bytes = Mux(
            (self.cmd_row_bytes == 0) | (self.cmd_row_bytes > sp_bytes),
            sp_bytes,
            self.cmd_row_bytes,
        )

        with m.FSM(init="IDLE") as fsm:
            with m.State("IDLE"):
                m.d.comb += self.cmd_ready.eq(1)
                with m.If(self.cmd_valid & accepted):
                    cmd_len = Mux(is_transpose & (self.cmd_len > lanes), lanes, self.cmd_len)
                    m.d.sync += [
                        transpose.eq(is_transpose),
                        col.eq(0),
                        cols.eq((cmd_row_bytes * 8 + eb - 1) // eb),
                        row_addr.eq(self.cmd_dram_addr),
                        sp_addr_reg.eq(self.cmd_sp_addr),
                        len_reg.eq(cmd_len),
                        stride_reg.eq(Mux(self.cmd_stride == 0, beat, self.cmd_stride)),
                        row_bytes.eq(cmd_row_bytes),
                        cmd_id_reg.eq(self.cmd_id),
                        row.eq(0),
                    ]
                    with m.If(self.cmd_len == 0):
                        m.next = "DONE"
                    with m.Else():
                        m.next = "ISSUE_DMA"

            with m.State("ISSUE_DMA"):
                m.d.comb += [
                    self.dma_req_valid.eq(1),
                    self.dma_req_len.eq(Mux(contiguous, burst_rows - 1, two_beats)),
                ]
                with m.If(self.dma_req_ready):
                    with m.If(contiguous):
                        m.next = "STREAM"
                    with m.Else():
                        m.next = "RECV_LO"

            with m.State("STREAM"):
                # One beat per row, written as it arrives
                m.d.comb += self.dma_resp_ready.eq(1)
                with m.If(self.dma_resp_valid):
                    put_row(self.dma_resp_data[: cfg.sp_width] & keep)
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
                    with m.If(two_beats):
                        m.next = "RECV_HI"
                    with m.Else():
                        m.next = "WRITE_SP"

            with m.State("RECV_HI"):
                m.d.comb += self.dma_resp_ready.eq(1)
                with m.If(self.dma_resp_valid):
                    m.d.sync += hi.eq(self.dma_resp_data)
                    m.next = "WRITE_SP"

            with m.State("WRITE_SP"):
                put_row(row_data)
                m.d.sync += [row.eq(row + 1), row_addr.eq(row_addr + stride_reg)]
                with m.If(row + 1 >= len_reg):
                    finish()
                with m.Else():
                    m.next = "ISSUE_DMA"

            # TRANSPOSE: scratchpad row sp_addr + col is element 0 of every buffered
            # row; the buffer then shifts down one element
            with m.State("TURN"):
                write_sp(sp_addr_reg + col, Cat(*[t[:eb] for t in turn]))
                m.d.sync += [t.eq(t >> eb) for t in turn]
                m.d.sync += col.eq(col + 1)
                with m.If(col + 1 >= cols):
                    m.next = "DONE"

            with m.State("DONE"):
                m.d.comb += [self.completed.eq(1), self.completed_id.eq(cmd_id_reg)]
                m.next = "IDLE"

        m.d.comb += self.busy.eq(~fsm.ongoing("IDLE"))
        return m
