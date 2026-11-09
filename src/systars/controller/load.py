"""
LoadController - moves a tile from L3 into the scratchpad or the accumulator (the
BlockMover's L3 -> L2 leg, and its L3 -> accumulator leg).

In the KPU's CSP model (kpu-sim, docs/01-architecture/kpu-execution-model.md) the
tile's AXI port is its link to L3: a memory tile in a SoC, the AXI memory model
in tests. A LOAD command moves one tile: ``len`` rows, row ``n`` read from L3 byte
address ``dram_addr + n * stride`` into scratchpad row ``sp_addr + n``.

- Any byte address: a row that starts mid-beat, or spills into the next beat, is
  read as a 1- or 2-beat burst from the beat boundary and shifted into place.
- ``row_bytes`` valid bytes per row (0 = a whole scratchpad row); the rest of
  the scratchpad row is written with zeros.
- ``stride`` 0 means rows one bus beat apart (consecutive beats).
- A ``sp_addr`` with the accumulator flag (``LocalAddr``, bit 31) writes accumulator
  rows instead: the starting values of a result tile, a bias or the C being
  accumulated onto. ``row_bytes`` 0 is then a whole accumulator row. The
  accumulator's write port is the Streamer's first: a row waits for a clock in
  which the Streamer writes none (the fabric never stalls).
- No burst crosses a 4 KB boundary (AXI forbids it): a streamed burst ends at the
  boundary and the next one starts there, and a row that spills from the last beat
  of a 4 KB page into the next page is read as two 1-beat bursts.

``BROADCAST`` reads one L3 row and writes it to all ``len`` rows (kpu-sim's
``STR_BROADCAST_ROW``): a bias row for every output row. ``CLEAR`` writes ``len``
zero rows and reads nothing from L3 (a state operation, ADR 0002): a padded
position's row of zeros.

``TRANSPOSE`` is a MEMCPY with a corner turn, for the layouts the Streamer reads
(one L2 row per L1 stream buffer, #21). The rows land in a corner-turn buffer of
one tile (``lanes`` rows of ``lanes`` elements, ``input_bits`` each, where
``lanes = sp_width // input_bits``) instead of the scratchpad. Once the last row
is in, scratchpad row ``sp_addr + j`` is written with element ``j`` of every L3
row: L3 row ``n`` becomes element ``n`` of each scratchpad row. That is one
scratchpad row per element of an L3 row, ``ceil(row_bytes * 8 / input_bits)``
rows, zero past the ``len``-th element. A TRANSPOSE moves at most one tile:
``len`` is clamped to ``lanes``. The corner turn writes only the scratchpad: a
TRANSPOSE into the accumulator is not accepted.

State Machine (per row):
    IDLE -> ISSUE_DMA -> STREAM (consecutive beats from a beat boundary: one
                             burst of up to 256 rows, within one 4 KB page)
                       -> RECV_LO [-> (ISSUE_HI ->) RECV_HI] -> WRITE_SP (any other
                             row; ISSUE_HI when its second beat is in the next page)
         -> (next burst | TURN (TRANSPOSE: write the tile's columns)
             | REPEAT (BROADCAST: write the row to every row) | DONE)
    IDLE -> REPEAT (CLEAR: zero rows) -> DONE

Data Flow:
    L3 (via StreamReader) --> LoadController --> Scratchpad | Accumulator
"""

from amaranth import Cat, Const, Module, Mux, Signal
from amaranth.lib.wiring import Component, In, Out

from ..config import SystolicConfig
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
            cmd_opcode: Operation code (MEMCPY, TRANSPOSE: with a corner turn,
                BROADCAST: one row to every row, CLEAR: zero rows)
            cmd_dram_addr: Source byte address of row 0 in L3 (64-bit, any alignment)
            cmd_sp_addr: Destination address in the scratchpad or, with the
                accumulator flag, the accumulator (32-bit local format)
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

        Accumulator Interface:
            acc_write_en: Write enable (a whole row, overwritten)
            acc_write_addr: Write address (local format)
            acc_write_data: Write data (acc_width bits)
            acc_write_ready: The write port is free this clock (the Streamer is not
                writing)

        Status:
            busy: Controller is processing a command
            completed: Command completed this cycle
            completed_id: ID of completed command
    """

    def __init__(self, config: SystolicConfig):
        self.config = config
        buswidth = config.dma_buswidth
        sp_width = config.sp_width
        for what, width in (("scratchpad", sp_width), ("accumulator", config.acc_width)):
            if width > buswidth:
                # One row is written per DMA beat; a wider row would only receive
                # the low dma_buswidth bits
                raise ValueError(
                    f"LoadController: {what} row ({width} bits) is wider than "
                    f"dma_buswidth ({buswidth}); set dma_buswidth >= {width}"
                )
        beat = buswidth // 8
        if beat < 2 or beat & (beat - 1):
            raise ValueError(
                f"LoadController: dma_buswidth must be a power-of-two bytes, got {buswidth}"
            )
        if beat > AXI_4KB:
            raise ValueError(f"LoadController: a {beat}-byte beat is wider than a 4 KB page")
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
                # Accumulator write interface
                "acc_write_en": Out(1),
                "acc_write_addr": Out(32),
                "acc_write_data": Out(config.acc_width),
                "acc_write_ready": In(1),
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
        row_width = max(cfg.sp_width, cfg.acc_width)  # the widest destination row
        off_bits = (beat - 1).bit_length()

        # Registered command parameters
        row_addr = Signal(64)  # byte address of the current row
        sp_addr_reg = Signal(32)
        len_reg = Signal(16)  # L3 rows to read
        stride_reg = Signal(32)
        row_bytes = Signal(16)
        cmd_id_reg = Signal(8)
        row = Signal(16)
        to_acc = Signal()  # the destination is the accumulator

        # TRANSPOSE: the corner-turn buffer (one tile) and the columns left to write.
        # It is never cleared: a row is zero past row_bytes, and the turn shifts out
        # every element up to row_bytes, so the buffer is empty after each TRANSPOSE
        eb = cfg.input_bits
        lanes = self.lanes
        transpose = Signal()
        turn = [Signal(cfg.sp_width, name=f"turn_{n}") for n in range(lanes)]
        col = Signal(16)
        cols = Signal(16)  # rows TURN or REPEAT writes

        # BROADCAST and CLEAR: the row REPEAT writes to every destination row
        broadcast = Signal()
        fill_row = Signal(row_width)

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
        keep = Cat(
            *[Mux(i < row_bytes, Const(0xFF, 8), Const(0, 8)) for i in range(row_width // 8)]
        )
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
            self.completed.eq(0),
            self.completed_id.eq(0),
        ]

        # Rows that are consecutive beats from a beat boundary stream as bursts of up
        # to 256 beats, ending at the next 4 KB boundary; any other row is its own 1-
        # or 2-beat burst, or two 1-beat bursts when its beats are in different pages
        contiguous = (offset == 0) & (stride_reg == beat)
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

        # A row can be put down this clock: the scratchpad always, the accumulator
        # when the Streamer is not writing it (nor a BROADCAST, which only keeps it)
        can_put = Signal()
        m.d.comb += can_put.eq(~to_acc | transpose | broadcast | self.acc_write_ready)

        def write_row(addr, data):
            with m.If(to_acc):
                m.d.comb += [
                    self.acc_write_en.eq(1),
                    self.acc_write_addr.eq(addr),
                    self.acc_write_data.eq(data[: cfg.acc_width]),
                ]
            with m.Else():
                m.d.comb += [
                    self.sp_write_en.eq(1),
                    self.sp_write_addr.eq(addr),
                    self.sp_write_data.eq(data[: cfg.sp_width]),
                    self.sp_write_mask.eq((1 << sp_bytes) - 1),
                ]

        def put_row(data):
            """Row ``row`` arrived (when ``can_put``): into its destination row, the
            corner-turn buffer, or the BROADCAST row."""
            with m.If(transpose):
                for n in range(lanes):
                    with m.If(row == n):
                        m.d.sync += turn[n].eq(data[: cfg.sp_width])
            with m.Elif(broadcast):
                m.d.sync += fill_row.eq(data)
            with m.Else():
                write_row(sp_addr_reg + row, data)

        def finish():
            """After the last row: write the tile's columns (TRANSPOSE), the
            broadcast rows (BROADCAST), or complete."""
            with m.If(transpose):
                m.next = "TURN"
            with m.Elif(broadcast):
                m.next = "REPEAT"
            with m.Else():
                m.next = "DONE"

        op = self.cmd_opcode
        cmd_to_acc = LocalAddr.is_acc(self.cmd_sp_addr)
        is_transpose = op == DmaOpcode.TRANSPOSE
        is_broadcast = op == DmaOpcode.BROADCAST
        is_clear = op == DmaOpcode.CLEAR
        accepted = (op == DmaOpcode.MEMCPY) | (is_transpose & ~cmd_to_acc) | is_broadcast | is_clear
        dst_bytes = Mux(cmd_to_acc, acc_bytes, sp_bytes)
        cmd_row_bytes = Mux(
            (self.cmd_row_bytes == 0) | (self.cmd_row_bytes > dst_bytes),
            dst_bytes,
            self.cmd_row_bytes,
        )

        with m.FSM(init="IDLE") as fsm:
            with m.State("IDLE"):
                m.d.comb += self.cmd_ready.eq(1)
                with m.If(self.cmd_valid & accepted):
                    cmd_len = Mux(is_transpose & (self.cmd_len > lanes), lanes, self.cmd_len)
                    m.d.sync += [
                        transpose.eq(is_transpose),
                        broadcast.eq(is_broadcast),
                        to_acc.eq(cmd_to_acc),
                        fill_row.eq(0),
                        col.eq(0),
                        cols.eq(
                            Mux(
                                is_transpose,
                                (cmd_row_bytes * 8 + eb - 1) // eb,
                                self.cmd_len,  # BROADCAST, CLEAR: every row
                            )
                        ),
                        row_addr.eq(self.cmd_dram_addr),
                        sp_addr_reg.eq(self.cmd_sp_addr),
                        len_reg.eq(Mux(is_broadcast, 1, cmd_len)),  # BROADCAST reads one row
                        stride_reg.eq(Mux(self.cmd_stride == 0, beat, self.cmd_stride)),
                        row_bytes.eq(cmd_row_bytes),
                        cmd_id_reg.eq(self.cmd_id),
                        row.eq(0),
                    ]
                    with m.If(self.cmd_len == 0):
                        m.next = "DONE"
                    with m.Elif(is_clear):
                        m.next = "REPEAT"
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
                # One beat per row, put down as it arrives (held off while the
                # accumulator's write port is the Streamer's)
                m.d.comb += self.dma_resp_ready.eq(can_put)
                with m.If(self.dma_resp_valid & can_put):
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

            with m.State("WRITE_SP"), m.If(can_put):
                put_row(row_data)
                m.d.sync += [row.eq(row + 1), row_addr.eq(row_addr + stride_reg)]
                with m.If(row + 1 >= len_reg):
                    finish()
                with m.Else():
                    m.next = "ISSUE_DMA"

            # TRANSPOSE: scratchpad row sp_addr + col is element 0 of every buffered
            # row; the buffer then shifts down one element
            with m.State("TURN"):
                write_row(sp_addr_reg + col, Cat(*[t[:eb] for t in turn]))
                m.d.sync += [t.eq(t >> eb) for t in turn]
                m.d.sync += col.eq(col + 1)
                with m.If(col + 1 >= cols):
                    m.next = "DONE"

            # BROADCAST and CLEAR: the same row (zero for CLEAR) into every row
            with m.State("REPEAT"), m.If(~to_acc | self.acc_write_ready):
                write_row(sp_addr_reg + col, fill_row)
                m.d.sync += col.eq(col + 1)
                with m.If(col + 1 >= cols):
                    m.next = "DONE"

            with m.State("DONE"):
                m.d.comb += [self.completed.eq(1), self.completed_id.eq(cmd_id_reg)]
                m.next = "IDLE"

        m.d.comb += self.busy.eq(~fsm.ongoing("IDLE"))
        return m
