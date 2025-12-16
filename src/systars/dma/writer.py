"""
StreamWriter - DMA engine for local memory to external memory transfers.

The StreamWriter handles AXI-like burst write transactions to external memory,
receiving data from the StoreController (reading from Accumulator).

Features (initial implementation):
- Single outstanding burst request
- Physical addresses only (no TLB)
- Fixed INCR burst type
- No max-pooling (deferred to Phase 4)
"""

from amaranth import Module, Signal
from amaranth.lib.wiring import Component, In, Out

from ..config import SystolicConfig


class StreamWriter(Component):
    """
    DMA write engine for local memory -> external memory transfers.

    Accepts write requests from the StoreController and issues AXI-like
    burst write transactions to external memory.

    Request Interface (from StoreController):
        req_valid: Request valid (start of burst)
        req_ready: Ready to accept request
        req_addr: Physical address (64-bit)
        req_len: Burst length in beats (0 = 1 beat, 255 = 256 beats)

    Data Interface (from StoreController):
        data_valid: Write data valid
        data_ready: Ready to accept data
        data: Write data (dma_buswidth bits)
        data_last: Last beat of burst

    Status Interface:
        busy: Transaction in progress
        done: Transaction complete (pulse)

    AXI Write Address Channel:
        mem_awvalid: Address valid
        mem_awready: Slave ready
        mem_awaddr: Write address (64-bit)
        mem_awlen: Burst length (8-bit, AXI4 style)
        mem_awsize: Beat size (3-bit, log2 of bytes)
        mem_awburst: Burst type (2-bit, fixed to INCR=01)

    AXI Write Data Channel:
        mem_wvalid: Data valid
        mem_wready: Slave ready
        mem_wdata: Write data (dma_buswidth bits)
        mem_wstrb: Write strobes (dma_buswidth/8 bits)
        mem_wlast: Last beat of burst

    AXI Write Response Channel:
        mem_bvalid: Response valid
        mem_bready: Master ready
        mem_bresp: Response status (2-bit, 00=OKAY)

    Parameters:
        config: SystolicConfig with DMA parameters
    """

    def __init__(self, config: SystolicConfig):
        self.config = config
        buswidth = config.dma_buswidth
        strb_width = buswidth // 8

        super().__init__(
            {
                # Request interface (from StoreController)
                "req_valid": In(1),
                "req_ready": Out(1),
                "req_addr": In(64),
                "req_len": In(8),
                # Data interface (from StoreController)
                "data_valid": In(1),
                "data_ready": Out(1),
                "data": In(buswidth),
                "data_last": In(1),
                # Status
                "busy": Out(1),
                "done": Out(1),
                # AXI Write Address Channel
                "mem_awvalid": Out(1),
                "mem_awready": In(1),
                "mem_awaddr": Out(64),
                "mem_awlen": Out(8),
                "mem_awsize": Out(3),
                "mem_awburst": Out(2),
                # AXI Write Data Channel
                "mem_wvalid": Out(1),
                "mem_wready": In(1),
                "mem_wdata": Out(buswidth),
                "mem_wstrb": Out(strb_width),
                "mem_wlast": Out(1),
                # AXI Write Response Channel
                "mem_bvalid": In(1),
                "mem_bready": Out(1),
                "mem_bresp": In(2),
            }
        )

    def elaborate(self, _platform):
        m = Module()
        cfg = self.config

        # Calculate awsize from bus width (log2 of bytes per beat)
        bytes_per_beat = cfg.dma_buswidth // 8
        awsize_value = (bytes_per_beat - 1).bit_length()  # log2
        strb_all_ones = (1 << bytes_per_beat) - 1

        # State machine
        # IDLE: Wait for request
        # ADDR: Issue address to AXI AW channel
        # DATA: Send data beats on AXI W channel
        # RESP: Wait for write response on AXI B channel
        state_idle = 0
        state_addr = 1
        state_data = 2
        state_resp = 3

        state = Signal(2, init=state_idle)

        # Registered request parameters
        addr_reg = Signal(64)
        len_reg = Signal(8)

        # Beat counter
        beat_count = Signal(8)

        # Track if address has been sent (for AXI allows addr and data in parallel)
        addr_sent = Signal()

        # Default outputs
        m.d.comb += [
            self.req_ready.eq(0),
            self.data_ready.eq(0),
            self.busy.eq(state != state_idle),
            self.done.eq(0),
            self.mem_awvalid.eq(0),
            self.mem_awaddr.eq(0),
            self.mem_awlen.eq(0),
            self.mem_awsize.eq(awsize_value),
            self.mem_awburst.eq(0b01),  # INCR burst
            self.mem_wvalid.eq(0),
            self.mem_wdata.eq(0),
            self.mem_wstrb.eq(0),
            self.mem_wlast.eq(0),
            self.mem_bready.eq(0),
        ]

        with m.Switch(state):
            with m.Case(state_idle):
                # Ready to accept new request
                m.d.comb += self.req_ready.eq(1)

                with m.If(self.req_valid):
                    # Capture request parameters
                    m.d.sync += [
                        addr_reg.eq(self.req_addr),
                        len_reg.eq(self.req_len),
                        beat_count.eq(0),
                        addr_sent.eq(0),
                    ]
                    m.d.sync += state.eq(state_addr)

            with m.Case(state_addr):
                # Issue address on AXI AW channel
                m.d.comb += [
                    self.mem_awvalid.eq(1),
                    self.mem_awaddr.eq(addr_reg),
                    self.mem_awlen.eq(len_reg),
                ]

                with m.If(self.mem_awready):
                    # Address accepted, move to data phase
                    m.d.sync += addr_sent.eq(1)
                    m.d.sync += state.eq(state_data)

            with m.Case(state_data):
                # Forward data from StoreController to AXI W channel
                m.d.comb += [
                    self.data_ready.eq(self.mem_wready),
                    self.mem_wvalid.eq(self.data_valid),
                    self.mem_wdata.eq(self.data),
                    self.mem_wstrb.eq(strb_all_ones),
                    self.mem_wlast.eq(self.data_last),
                ]

                with m.If(self.data_valid & self.mem_wready):
                    # Beat transferred
                    m.d.sync += beat_count.eq(beat_count + 1)

                    with m.If(self.data_last):
                        # All data sent, wait for response
                        m.d.sync += state.eq(state_resp)

            with m.Case(state_resp):
                # Wait for write response
                m.d.comb += self.mem_bready.eq(1)

                with m.If(self.mem_bvalid):
                    # Response received, transaction complete
                    m.d.comb += self.done.eq(1)
                    m.d.sync += state.eq(state_idle)

        return m
