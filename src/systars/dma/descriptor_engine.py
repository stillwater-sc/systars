"""
DescriptorEngine - Hardware engine for executing descriptor chains.

The DescriptorEngine fetches DMA descriptors from memory, parses them,
and executes the specified operations. It supports chained descriptors
for complex multi-operation transfers.

Descriptor Format (64 bytes):
    0x00: opcode (8 bits)
    0x01: flags (8 bits)
    0x02: reserved (16 bits)
    0x04: length (32 bits)
    0x08: src_addr (64 bits)
    0x10: dst_addr (64 bits)
    0x18: next_desc (64 bits)
    0x20: completion (64 bits)
    0x28: user_data (64 bits)
    0x30-0x3F: reserved (16 bytes)

Supported Operations:
    MEMCPY (0x00): Copy src_addr -> dst_addr for length bytes
    FILL (0x01): Fill dst_addr with pattern in src_addr field
    FENCE (0x20): Wait for all prior operations to complete
    NOP (0x21): No operation (skip)

State Machine:
    IDLE -> FETCH_DESC -> PARSE_DESC -> EXECUTE -> CHECK_CHAIN -> IDLE/FETCH_DESC
"""

from amaranth import Cat, Module, Signal
from amaranth.lib.wiring import Component, In, Out

from ..config import SystolicConfig


class DescriptorEngine(Component):
    """
    Hardware engine for executing DMA descriptor chains.

    Coordinates with StreamReader/StreamWriter to perform memory operations
    as specified by descriptors fetched from memory.

    Command Interface:
        start: Pulse to begin processing descriptor chain
        desc_addr: Address of first descriptor in chain (64-bit)
        busy: Engine is processing
        done: Chain processing complete (pulse)
        error: Error occurred (pulse)

    Descriptor Fetch Interface (to StreamReader):
        fetch_req_valid: Request to fetch descriptor
        fetch_req_ready: StreamReader ready
        fetch_req_addr: Descriptor address
        fetch_resp_valid: Descriptor data valid
        fetch_resp_data: Descriptor data (dma_buswidth bits)
        fetch_resp_last: Last beat of descriptor

    Data Read Interface (to StreamReader for MEMCPY source):
        read_req_valid: Request to read data
        read_req_ready: StreamReader ready
        read_req_addr: Source address
        read_req_len: Transfer length
        read_resp_valid: Read data valid
        read_resp_data: Read data
        read_resp_last: Last beat

    Data Write Interface (to StreamWriter for MEMCPY/FILL dest):
        write_req_valid: Request to write data
        write_req_ready: StreamWriter ready
        write_req_addr: Destination address
        write_req_len: Transfer length
        write_data_valid: Write data valid
        write_data_ready: StreamWriter ready for data
        write_data: Write data
        write_data_last: Last beat
        write_done: Write complete

    Status Writeback Interface:
        status_write_valid: Status writeback request
        status_write_addr: Completion address
        status_write_data: Status value

    Interrupt Interface:
        interrupt: Interrupt request (when INTERRUPT_ON_COMPLETE flag set)
    """

    def __init__(self, config: SystolicConfig):
        self.config = config
        buswidth = config.dma_buswidth

        super().__init__(
            {
                # Command interface
                "start": In(1),
                "desc_addr": In(64),
                "busy": Out(1),
                "done": Out(1),
                "error": Out(1),
                # Descriptor fetch interface (to StreamReader)
                "fetch_req_valid": Out(1),
                "fetch_req_ready": In(1),
                "fetch_req_addr": Out(64),
                "fetch_req_len": Out(8),
                "fetch_resp_valid": In(1),
                "fetch_resp_ready": Out(1),
                "fetch_resp_data": In(buswidth),
                "fetch_resp_last": In(1),
                # Data read interface (to StreamReader for MEMCPY)
                "read_req_valid": Out(1),
                "read_req_ready": In(1),
                "read_req_addr": Out(64),
                "read_req_len": Out(8),
                "read_resp_valid": In(1),
                "read_resp_ready": Out(1),
                "read_resp_data": In(buswidth),
                "read_resp_last": In(1),
                # Data write interface (to StreamWriter)
                "write_req_valid": Out(1),
                "write_req_ready": In(1),
                "write_req_addr": Out(64),
                "write_req_len": Out(8),
                "write_data_valid": Out(1),
                "write_data_ready": In(1),
                "write_data": Out(buswidth),
                "write_data_last": Out(1),
                "write_done": In(1),
                # Status writeback
                "status_write_valid": Out(1),
                "status_write_addr": Out(64),
                "status_write_data": Out(32),
                # Interrupt
                "interrupt": Out(1),
            }
        )

    def elaborate(self, _platform):
        m = Module()
        cfg = self.config
        buswidth = cfg.dma_buswidth
        bytes_per_beat = buswidth // 8

        # Number of beats to fetch a 64-byte descriptor
        desc_beats = 64 // bytes_per_beat

        # Opcode constants
        OP_MEMCPY = 0x00
        OP_FILL = 0x01
        OP_FENCE = 0x20
        OP_NOP = 0x21

        # Flag constants
        FLAG_CHAIN = 0x01
        FLAG_INTERRUPT = 0x02
        FLAG_WRITEBACK = 0x04
        # FLAG_SRC_PATTERN = 0x08  # Reserved for future use

        # State machine states
        STATE_IDLE = 0
        STATE_FETCH_DESC = 1
        STATE_WAIT_DESC = 2
        STATE_PARSE_DESC = 3
        STATE_EXEC_MEMCPY_REQ = 4
        STATE_EXEC_MEMCPY_DATA = 5
        STATE_EXEC_FILL_REQ = 6
        STATE_EXEC_FILL_DATA = 7
        STATE_WAIT_WRITE = 8
        STATE_CHECK_CHAIN = 9
        STATE_WRITEBACK = 10
        STATE_DONE = 11
        STATE_ERROR = 12

        state = Signal(4, init=STATE_IDLE)

        # Current descriptor address
        current_desc_addr = Signal(64)

        # Descriptor buffer (64 bytes = 512 bits)
        desc_buffer = Signal(512)
        desc_beat_count = Signal(range(desc_beats + 1))

        # Parsed descriptor fields
        desc_opcode = Signal(8)
        desc_flags = Signal(8)
        desc_length = Signal(32)
        desc_src_addr = Signal(64)
        desc_dst_addr = Signal(64)
        desc_next_desc = Signal(64)
        desc_completion = Signal(64)
        desc_user_data = Signal(64)

        # Transfer state
        transfer_beats_remaining = Signal(32)
        transfer_beat_count = Signal(32)

        # Fill pattern register
        fill_pattern = Signal(buswidth)

        # Default outputs
        m.d.comb += [
            self.busy.eq(state != STATE_IDLE),
            self.done.eq(0),
            self.error.eq(0),
            self.fetch_req_valid.eq(0),
            self.fetch_req_addr.eq(0),
            self.fetch_req_len.eq(0),
            self.fetch_resp_ready.eq(0),
            self.read_req_valid.eq(0),
            self.read_req_addr.eq(0),
            self.read_req_len.eq(0),
            self.read_resp_ready.eq(0),
            self.write_req_valid.eq(0),
            self.write_req_addr.eq(0),
            self.write_req_len.eq(0),
            self.write_data_valid.eq(0),
            self.write_data.eq(0),
            self.write_data_last.eq(0),
            self.status_write_valid.eq(0),
            self.status_write_addr.eq(0),
            self.status_write_data.eq(0),
            self.interrupt.eq(0),
        ]

        with m.Switch(state):
            with m.Case(STATE_IDLE), m.If(self.start):
                m.d.sync += [
                    current_desc_addr.eq(self.desc_addr),
                    desc_beat_count.eq(0),
                ]
                m.d.sync += state.eq(STATE_FETCH_DESC)

            with m.Case(STATE_FETCH_DESC):
                # Issue request to fetch descriptor
                m.d.comb += [
                    self.fetch_req_valid.eq(1),
                    self.fetch_req_addr.eq(current_desc_addr),
                    # AXI4 len: N-1 for N beats
                    self.fetch_req_len.eq(desc_beats - 1),
                ]

                with m.If(self.fetch_req_ready):
                    m.d.sync += state.eq(STATE_WAIT_DESC)

            with m.Case(STATE_WAIT_DESC):
                # Receive descriptor beats
                m.d.comb += self.fetch_resp_ready.eq(1)

                with m.If(self.fetch_resp_valid):
                    # Shift in descriptor data (LSB first)
                    shift_amount = desc_beat_count * buswidth
                    m.d.sync += desc_buffer.bit_select(shift_amount, buswidth).eq(
                        self.fetch_resp_data
                    )
                    m.d.sync += desc_beat_count.eq(desc_beat_count + 1)

                    with m.If(self.fetch_resp_last):
                        m.d.sync += state.eq(STATE_PARSE_DESC)

            with m.Case(STATE_PARSE_DESC):
                # Parse descriptor fields from buffer
                # Layout: opcode(8), flags(8), reserved(16), length(32),
                #         src_addr(64), dst_addr(64), next_desc(64),
                #         completion(64), user_data(64), reserved(128)
                m.d.sync += [
                    desc_opcode.eq(desc_buffer[0:8]),
                    desc_flags.eq(desc_buffer[8:16]),
                    # reserved: desc_buffer[16:32]
                    desc_length.eq(desc_buffer[32:64]),
                    desc_src_addr.eq(desc_buffer[64:128]),
                    desc_dst_addr.eq(desc_buffer[128:192]),
                    desc_next_desc.eq(desc_buffer[192:256]),
                    desc_completion.eq(desc_buffer[256:320]),
                    desc_user_data.eq(desc_buffer[320:384]),
                ]

                # Calculate beats for transfer
                beats = (desc_buffer[32:64] + bytes_per_beat - 1) // bytes_per_beat
                m.d.sync += [
                    transfer_beats_remaining.eq(beats),
                    transfer_beat_count.eq(0),
                ]

                # Dispatch based on opcode
                with m.Switch(desc_buffer[0:8]):
                    with m.Case(OP_MEMCPY):
                        m.d.sync += state.eq(STATE_EXEC_MEMCPY_REQ)
                    with m.Case(OP_FILL):
                        # Extract fill pattern (truncate or replicate to buswidth)
                        pattern_64 = desc_buffer[64:128]
                        if buswidth <= 64:
                            m.d.sync += fill_pattern.eq(pattern_64[:buswidth])
                        else:
                            # Replicate 64-bit pattern to fill buswidth
                            replications = buswidth // 64
                            m.d.sync += fill_pattern.eq(Cat(*[pattern_64] * replications))
                        m.d.sync += state.eq(STATE_EXEC_FILL_REQ)
                    with m.Case(OP_FENCE):
                        # FENCE: just proceed to check chain
                        m.d.sync += state.eq(STATE_CHECK_CHAIN)
                    with m.Case(OP_NOP):
                        # NOP: skip to check chain
                        m.d.sync += state.eq(STATE_CHECK_CHAIN)
                    with m.Default():
                        # Unknown opcode
                        m.d.sync += state.eq(STATE_ERROR)

            with m.Case(STATE_EXEC_MEMCPY_REQ):
                # Issue read and write requests
                # For simplicity, limit burst to 256 beats max (AXI4 limit)
                max_burst_beats = 256

                # Compute burst length: min(remaining, max) - 1 for AXI4
                capped_beats = Signal(32)
                with m.If(transfer_beats_remaining > max_burst_beats):
                    m.d.comb += capped_beats.eq(max_burst_beats)
                with m.Else():
                    m.d.comb += capped_beats.eq(transfer_beats_remaining)

                m.d.comb += [
                    self.read_req_valid.eq(1),
                    self.read_req_addr.eq(desc_src_addr + (transfer_beat_count * bytes_per_beat)),
                    self.read_req_len.eq(capped_beats[:8] - 1),
                    self.write_req_valid.eq(1),
                    self.write_req_addr.eq(desc_dst_addr + (transfer_beat_count * bytes_per_beat)),
                    self.write_req_len.eq(capped_beats[:8] - 1),
                ]

                with m.If(self.read_req_ready & self.write_req_ready):
                    m.d.sync += state.eq(STATE_EXEC_MEMCPY_DATA)

            with m.Case(STATE_EXEC_MEMCPY_DATA):
                # Stream data from read to write
                is_last_beat = transfer_beat_count >= transfer_beats_remaining - 1

                m.d.comb += [
                    self.read_resp_ready.eq(self.write_data_ready),
                    self.write_data_valid.eq(self.read_resp_valid),
                    self.write_data.eq(self.read_resp_data),
                    self.write_data_last.eq(self.read_resp_last | is_last_beat),
                ]

                with m.If(self.read_resp_valid & self.write_data_ready):
                    m.d.sync += transfer_beat_count.eq(transfer_beat_count + 1)
                    m.d.sync += transfer_beats_remaining.eq(transfer_beats_remaining - 1)

                    with m.If(self.read_resp_last | is_last_beat):
                        m.d.sync += state.eq(STATE_WAIT_WRITE)

            with m.Case(STATE_EXEC_FILL_REQ):
                # Issue write request for fill
                max_burst_beats = 256

                # Compute burst length: min(remaining, max) - 1 for AXI4
                fill_capped_beats = Signal(32)
                with m.If(transfer_beats_remaining > max_burst_beats):
                    m.d.comb += fill_capped_beats.eq(max_burst_beats)
                with m.Else():
                    m.d.comb += fill_capped_beats.eq(transfer_beats_remaining)

                m.d.comb += [
                    self.write_req_valid.eq(1),
                    self.write_req_addr.eq(desc_dst_addr + (transfer_beat_count * bytes_per_beat)),
                    self.write_req_len.eq(fill_capped_beats[:8] - 1),
                ]

                with m.If(self.write_req_ready):
                    m.d.sync += state.eq(STATE_EXEC_FILL_DATA)

            with m.Case(STATE_EXEC_FILL_DATA):
                # Send fill pattern data
                is_last_beat = transfer_beat_count >= transfer_beats_remaining - 1

                m.d.comb += [
                    self.write_data_valid.eq(1),
                    self.write_data.eq(fill_pattern),
                    self.write_data_last.eq(is_last_beat),
                ]

                with m.If(self.write_data_ready):
                    m.d.sync += transfer_beat_count.eq(transfer_beat_count + 1)
                    m.d.sync += transfer_beats_remaining.eq(transfer_beats_remaining - 1)

                    with m.If(is_last_beat):
                        m.d.sync += state.eq(STATE_WAIT_WRITE)

            with m.Case(STATE_WAIT_WRITE):  # noqa: SIM117
                # Wait for write to complete
                with m.If(self.write_done):  # noqa: SIM117
                    m.d.sync += state.eq(STATE_CHECK_CHAIN)

            with m.Case(STATE_CHECK_CHAIN):
                # Check if we should continue to next descriptor
                with m.If(desc_flags & FLAG_CHAIN):
                    # More descriptors in chain
                    m.d.sync += [
                        current_desc_addr.eq(desc_next_desc),
                        desc_beat_count.eq(0),
                    ]
                    m.d.sync += state.eq(STATE_FETCH_DESC)
                with m.Elif(desc_flags & FLAG_WRITEBACK):
                    # Write completion status
                    m.d.sync += state.eq(STATE_WRITEBACK)
                with m.Else():
                    m.d.sync += state.eq(STATE_DONE)

            with m.Case(STATE_WRITEBACK):
                # Write completion status
                m.d.comb += [
                    self.status_write_valid.eq(1),
                    self.status_write_addr.eq(desc_completion),
                    self.status_write_data.eq(desc_user_data[:32]),
                ]
                m.d.sync += state.eq(STATE_DONE)

            with m.Case(STATE_DONE):
                m.d.comb += self.done.eq(1)

                # Generate interrupt if flag set
                with m.If(desc_flags & FLAG_INTERRUPT):
                    m.d.comb += self.interrupt.eq(1)

                m.d.sync += state.eq(STATE_IDLE)

            with m.Case(STATE_ERROR):
                m.d.comb += self.error.eq(1)
                m.d.sync += state.eq(STATE_IDLE)

        return m
