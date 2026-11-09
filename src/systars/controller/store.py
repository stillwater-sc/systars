"""
StoreController - moves a result tile from the accumulator to L3 (the BlockMover's
L2 -> L3 leg).

In the KPU's CSP model (kpu-sim) the tile's AXI port is its link to L3. A STORE
command moves one tile: ``len`` accumulator rows, row ``n`` written to L3 byte
address ``dram_addr + n * stride``.

- Any byte address, with byte strobes: only ``row_bytes`` bytes of each row are
  written (0 = a whole accumulator row), so an edge N tile leaves the bytes past
  it untouched and C can be dense. A row that spills into the next beat is a
  2-beat burst.
- ``stride`` 0 means rows one bus beat apart.
- No burst crosses a 4 KB boundary (AXI forbids it): a streamed burst ends at the
  boundary and the next one starts there, and a row that spills from the last beat
  of a 4 KB page into the next page is written as two 1-beat bursts.
- The activation (``cmd_activation``) is applied to each element.

``TRANSPOSE`` is a MEMCPY with a corner turn, for results the array produces as
columns (input-stationary, #21). The tile is first read into a corner-turn
buffer of ``lanes`` accumulator rows (``lanes = acc_width // acc_bits``): one
row per element of an L3 row, ``ceil(row_bytes * 8 / acc_bits)`` rows from
``acc_addr``, read back to back. L3 row ``i`` is then element ``i`` of every
buffered row: accumulator row ``n`` becomes element ``n`` of each L3 row. The
rows go out exactly as a MEMCPY's, and the activation applies to each element. A
TRANSPOSE moves at most one tile: ``len`` is clamped to ``lanes``.

State Machine:
    IDLE [-> GATHER (TRANSPOSE: read the tile into the corner-turn buffer)]
         -> NEXT -> ISSUE_BURST -> (READ_ACC -> WAIT_VALID -> SEND_STREAM)* -> WAIT_DONE
                    (consecutive beats from a beat boundary: one burst of up to 256 rows,
                    within one 4 KB page)
                 -> READ_ACC -> WAIT_VALID -> ISSUE_ROW -> SEND_LO [-> SEND_HI] -> WAIT_DONE
                    (any other row; when its second beat starts the next 4 KB page,
                    SEND_LO -> WAIT_LO -> ISSUE_HI -> SEND_HI: two 1-beat bursts)
         -> (NEXT | DONE)

Data Flow:
    Accumulator --> StoreController --> StreamWriter --> L3
"""

from amaranth import Cat, Const, Module, Mux, Signal, signed
from amaranth.lib.wiring import Component, In, Out

from ..config import SystolicConfig
from ..dma.axi import AXI_4KB
from ..memory.activation import activate_row
from ..util.commands import DmaOpcode


class StoreController(Component):
    """
    Store controller for accumulator -> DRAM transfers.

    Coordinates data movement from the accumulator through the StreamWriter
    DMA engine to external memory.

    Ports:
        Command Interface:
            cmd_valid: Command available
            cmd_ready: Controller ready to accept command
            cmd_opcode: Operation code (MEMCPY, or TRANSPOSE: with a corner turn)
            cmd_acc_addr: Source address in accumulator (32-bit local format)
            cmd_dram_addr: Destination byte address of row 0 in L3 (64-bit, any alignment)
            cmd_len: Rows to move (0 completes at once)
            cmd_stride: Bytes between destination rows (0 = one bus beat)
            cmd_row_bytes: Bytes written per row (0 = a whole accumulator row)
            cmd_id: Command ID for tracking
            cmd_activation: Activation function (0=none, 1=relu)

        Accumulator Interface:
            acc_read_req: Read request
            acc_read_addr: Read address (local format)
            acc_read_data: Read data (acc_width bits)
            acc_read_valid: Data valid

        DMA Interface (to StreamWriter):
            dma_req_valid: Request valid
            dma_req_ready: DMA ready to accept
            dma_req_addr: DRAM address (64-bit)
            dma_req_len: Burst length (AXI4 style: 0=1 beat)

            dma_data_valid: Write data valid
            dma_data_ready: DMA ready for data
            dma_data: Write data (dma_buswidth bits)
            dma_data_strb: Byte strobes of the beat
            dma_data_last: Last beat of burst

            dma_done: DMA write complete

        Status:
            busy: Controller is processing a command
            completed: Command completed this cycle
            completed_id: ID of completed command
    """

    def __init__(self, config: SystolicConfig):
        self.config = config
        buswidth = config.dma_buswidth
        acc_width = config.acc_width
        if acc_width > buswidth:
            # One accumulator row is sent per DMA beat; a wider row would be
            # silently truncated to its low dma_buswidth bits
            raise ValueError(
                f"StoreController: accumulator row ({acc_width} bits = {config.grid_cols * config.tile_cols}"
                f" x {config.acc_bits}) is wider than dma_buswidth ({buswidth}); "
                f"set dma_buswidth >= {acc_width}"
            )

        self.lanes = acc_width // config.acc_bits  # elements per accumulator row
        beat = buswidth // 8
        if beat < 2 or beat & (beat - 1):
            raise ValueError(
                f"StoreController: dma_buswidth must be a power-of-two bytes, got {buswidth}"
            )
        if beat > AXI_4KB:
            raise ValueError(f"StoreController: a {beat}-byte beat is wider than a 4 KB page")

        super().__init__(
            {
                # Command interface
                "cmd_valid": In(1),
                "cmd_ready": Out(1),
                "cmd_opcode": In(8),
                "cmd_acc_addr": In(32),
                "cmd_dram_addr": In(64),
                "cmd_len": In(16),  # Rows to move
                "cmd_stride": In(32),  # Bytes between destination rows (0 = one beat)
                "cmd_row_bytes": In(16),  # Bytes written per row (0 = whole row)
                "cmd_id": In(8),
                "cmd_activation": In(4),  # Activation function
                # Accumulator read interface
                "acc_read_req": Out(1),
                "acc_read_addr": Out(32),
                "acc_read_data": In(signed(acc_width)),
                "acc_read_valid": In(1),
                # DMA interface (to StreamWriter)
                "dma_req_valid": Out(1),
                "dma_req_ready": In(1),
                "dma_req_addr": Out(64),
                "dma_req_len": Out(8),  # AXI4 style: 0=1 beat
                "dma_data_valid": Out(1),
                "dma_data_ready": In(1),
                "dma_data": Out(buswidth),
                "dma_data_strb": Out(buswidth // 8),
                "dma_data_last": Out(1),
                "dma_done": In(1),
                # Status
                "busy": Out(1),
                "completed": Out(1),
                "completed_id": Out(8),
            }
        )

    def elaborate(self, _platform):
        m = Module()
        cfg = self.config

        W = cfg.dma_buswidth
        beat = W // 8
        acc_bytes = cfg.acc_width // 8
        off_bits = (beat - 1).bit_length()

        # Registered command parameters
        acc_addr_reg = Signal(32)
        row_addr = Signal(64)  # byte address of the current row
        len_reg = Signal(16)
        stride_reg = Signal(32)
        row_bytes = Signal(16)
        cmd_id_reg = Signal(8)
        activation_reg = Signal(4)
        row = Signal(16)

        # TRANSPOSE: the corner-turn buffer (one tile), and its reads from the accumulator.
        # It is never cleared: rows past the ones a TRANSPOSE reads only fill elements
        # past row_bytes, which the byte strobes leave unwritten
        eb = cfg.acc_bits
        lanes = self.lanes
        transpose = Signal()
        turn = [Signal(cfg.acc_width, name=f"turn_{n}") for n in range(lanes)]
        gather = Signal(16)  # accumulator rows to read
        issued = Signal(16)
        received = Signal(16)
        turned_row = Cat(*[t[:eb] for t in turn])  # element 0 of every buffered row

        offset = Signal(off_bits)
        two_beats = Signal()
        m.d.comb += [
            offset.eq(row_addr[:off_bits]),
            two_beats.eq(offset + row_bytes > beat),
        ]

        # The row and its strobes, shifted up to its byte offset across two beats
        data_buf = Signal(2 * W)
        strb_buf = Signal(2 * beat)
        keep = Cat(*[i < row_bytes for i in range(acc_bytes)])

        m.d.comb += [
            self.cmd_ready.eq(0),
            self.acc_read_req.eq(0),
            self.acc_read_addr.eq(0),
            self.dma_req_valid.eq(0),
            self.dma_req_len.eq(0),
            self.dma_data_valid.eq(0),
            self.dma_data.eq(0),
            self.dma_data_strb.eq(0),
            self.dma_data_last.eq(0),
            self.completed.eq(0),
            self.completed_id.eq(0),
        ]

        # Rows that are consecutive beats from a beat boundary stream as bursts of up
        # to 256 beats, ending at the next 4 KB boundary; any other row is its own 1-
        # or 2-beat burst, or two 1-beat bursts when its beats are in different pages
        contiguous = (offset == 0) & (stride_reg == beat)
        streaming = Signal()
        burst_left = Signal(9)
        remaining = Signal(16)
        m.d.comb += remaining.eq(len_reg - row)
        page_beats = Signal(range(AXI_4KB // beat + 1))  # beats left in row_addr's page
        m.d.comb += page_beats.eq((AXI_4KB >> off_bits) - row_addr[off_bits:12])
        capped = Mux(remaining > 256, 256, remaining)  # AXI4: at most 256 beats
        burst_rows = Signal(9)
        m.d.comb += burst_rows.eq(Mux(page_beats < capped, page_beats, capped))
        split = Signal()  # a 2-beat row whose second beat starts a new page
        m.d.comb += split.eq(two_beats & (page_beats == 1))
        aligned = Cat(Const(0, off_bits), row_addr[off_bits:])
        m.d.comb += self.dma_req_addr.eq(aligned)

        def activated_row():
            # The latched activation applied to each acc_bits element (the
            # accumulator's own read activation is disabled by SystolicTop, so this
            # is the only one). A TRANSPOSE's row comes from the corner-turn buffer.
            source = Mux(transpose, turned_row, self.acc_read_data)
            return activate_row(source, activation_reg, cfg.acc_bits, lanes)

        accepted = (self.cmd_opcode == DmaOpcode.MEMCPY) | (self.cmd_opcode == DmaOpcode.TRANSPOSE)
        is_transpose = self.cmd_opcode == DmaOpcode.TRANSPOSE
        cmd_row_bytes = Mux(
            (self.cmd_row_bytes == 0) | (self.cmd_row_bytes > acc_bytes),
            acc_bytes,
            self.cmd_row_bytes,
        )

        with m.FSM(init="IDLE") as fsm:
            with m.State("IDLE"):
                m.d.comb += self.cmd_ready.eq(1)
                with m.If(self.cmd_valid & accepted):
                    cmd_len = Mux(is_transpose & (self.cmd_len > lanes), lanes, self.cmd_len)
                    m.d.sync += [
                        acc_addr_reg.eq(self.cmd_acc_addr),
                        row_addr.eq(self.cmd_dram_addr),
                        len_reg.eq(cmd_len),
                        stride_reg.eq(Mux(self.cmd_stride == 0, beat, self.cmd_stride)),
                        row_bytes.eq(cmd_row_bytes),
                        cmd_id_reg.eq(self.cmd_id),
                        activation_reg.eq(self.cmd_activation),
                        row.eq(0),
                        transpose.eq(is_transpose),
                        gather.eq((cmd_row_bytes * 8 + eb - 1) // eb),
                        issued.eq(0),
                        received.eq(0),
                    ]
                    with m.If(self.cmd_len == 0):
                        m.next = "DONE"
                    with m.Elif(is_transpose):
                        m.next = "GATHER"
                    with m.Else():
                        m.next = "NEXT"

            # TRANSPOSE: read the tile's accumulator rows into the corner-turn buffer,
            # one request per cycle
            with m.State("GATHER"):
                with m.If(issued < gather):
                    m.d.comb += [
                        self.acc_read_req.eq(1),
                        self.acc_read_addr.eq(acc_addr_reg + issued),
                    ]
                    m.d.sync += issued.eq(issued + 1)
                with m.If(self.acc_read_valid):
                    for n in range(lanes):
                        with m.If(received == n):
                            m.d.sync += turn[n].eq(self.acc_read_data)
                    m.d.sync += received.eq(received + 1)
                    with m.If(received + 1 >= gather):
                        m.next = "NEXT"

            with m.State("NEXT"):
                m.d.sync += streaming.eq(contiguous)
                with m.If(contiguous):
                    m.d.sync += burst_left.eq(burst_rows)
                    m.next = "ISSUE_BURST"
                with m.Else():
                    m.next = "READ_ACC"

            # --- Streaming: one burst, one beat per row ---
            with m.State("ISSUE_BURST"):
                m.d.comb += [self.dma_req_valid.eq(1), self.dma_req_len.eq(burst_left - 1)]
                with m.If(self.dma_req_ready):
                    m.next = "READ_ACC"

            # --- Read the row from the accumulator ---
            with m.State("READ_ACC"):
                with m.If(~transpose):
                    m.d.comb += [
                        self.acc_read_req.eq(1),
                        self.acc_read_addr.eq(acc_addr_reg + row),
                    ]
                m.next = "WAIT_VALID"

            # A TRANSPOSE's row is in the corner-turn buffer: no accumulator read
            with m.State("WAIT_VALID"), m.If(self.acc_read_valid | transpose):
                # Place the row at its byte offset (0 when streaming)
                m.d.sync += [
                    data_buf.eq(activated_row() << (offset * 8)),
                    strb_buf.eq(keep << offset),
                ]
                with m.If(transpose):
                    m.d.sync += [t.eq(t >> eb) for t in turn]
                with m.If(streaming):
                    m.next = "SEND_STREAM"
                with m.Else():
                    m.next = "ISSUE_ROW"

            with m.State("SEND_STREAM"):
                last = burst_left == 1
                m.d.comb += [
                    self.dma_data_valid.eq(1),
                    self.dma_data.eq(data_buf[:W]),
                    self.dma_data_strb.eq(strb_buf[:beat]),
                    self.dma_data_last.eq(last),
                ]
                with m.If(self.dma_data_ready):
                    m.d.sync += [
                        row.eq(row + 1),
                        row_addr.eq(row_addr + beat),
                        burst_left.eq(burst_left - 1),
                    ]
                    with m.If(last):
                        m.next = "WAIT_DONE"
                    with m.Else():
                        m.next = "READ_ACC"

            # --- One row: a 1- or 2-beat burst at its byte offset ---
            with m.State("ISSUE_ROW"):
                m.d.comb += [self.dma_req_valid.eq(1), self.dma_req_len.eq(two_beats & ~split)]
                with m.If(self.dma_req_ready):
                    m.next = "SEND_LO"

            with m.State("SEND_LO"):
                m.d.comb += [
                    self.dma_data_valid.eq(1),
                    self.dma_data.eq(data_buf[:W]),
                    self.dma_data_strb.eq(strb_buf[:beat]),
                    self.dma_data_last.eq(~two_beats | split),
                ]
                with m.If(self.dma_data_ready):
                    with m.If(split):
                        m.next = "WAIT_LO"
                    with m.Elif(two_beats):
                        m.next = "SEND_HI"
                    with m.Else():
                        m.next = "WAIT_DONE"

            # The row's second beat starts the next 4 KB page: once the first burst is
            # acknowledged, the second beat goes as a burst of its own
            with m.State("WAIT_LO"), m.If(self.dma_done):
                m.next = "ISSUE_HI"

            with m.State("ISSUE_HI"):
                m.d.comb += [
                    self.dma_req_valid.eq(1),
                    self.dma_req_addr.eq(aligned + beat),
                    self.dma_req_len.eq(0),
                ]
                with m.If(self.dma_req_ready):
                    m.next = "SEND_HI"

            with m.State("SEND_HI"):
                m.d.comb += [
                    self.dma_data_valid.eq(1),
                    self.dma_data.eq(data_buf[W:]),
                    self.dma_data_strb.eq(strb_buf[beat:]),
                    self.dma_data_last.eq(1),
                ]
                with m.If(self.dma_data_ready):
                    m.next = "WAIT_DONE"

            with m.State("WAIT_DONE"), m.If(self.dma_done):
                done_rows = Mux(streaming, row, row + 1)
                with m.If(~streaming):
                    m.d.sync += [row.eq(row + 1), row_addr.eq(row_addr + stride_reg)]
                with m.If(done_rows >= len_reg):
                    m.next = "DONE"
                with m.Else():
                    m.next = "NEXT"

            with m.State("DONE"):
                m.d.comb += [self.completed.eq(1), self.completed_id.eq(cmd_id_reg)]
                m.next = "IDLE"

        m.d.comb += self.busy.eq(~fsm.ongoing("IDLE"))
        return m
