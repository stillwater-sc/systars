"""
StreamReader - DMA engine for external memory to local memory transfers.

The StreamReader handles AXI-like burst read transactions from external memory,
providing data to the LoadController for writing to the Scratchpad.

Features (initial implementation):
- Single outstanding burst request
- Physical addresses only (no TLB)
- Fixed INCR burst type
- Configurable bus width
"""

from amaranth import Module, Signal
from amaranth.lib.wiring import Component, In, Out

from ..config import SystolicConfig


class StreamReader(Component):
    """
    DMA read engine for external memory -> local memory transfers.

    Accepts read requests from the LoadController and issues AXI-like
    burst read transactions to external memory.

    Request Interface (from LoadController):
        req_valid: Request valid
        req_ready: Ready to accept request
        req_addr: Physical address (64-bit)
        req_len: Burst length in beats (0 = 1 beat, 255 = 256 beats)

    Response Interface (to LoadController):
        resp_valid: Response data valid
        resp_ready: LoadController ready to accept
        resp_data: Read data (dma_buswidth bits)
        resp_last: Last beat of burst

    AXI Read Address Channel:
        mem_arvalid: Address valid
        mem_arready: Slave ready
        mem_araddr: Read address (64-bit)
        mem_arlen: Burst length (8-bit, AXI4 style)
        mem_arsize: Beat size (3-bit, log2 of bytes)
        mem_arburst: Burst type (2-bit, fixed to INCR=01)

    AXI Read Data Channel:
        mem_rvalid: Data valid from slave
        mem_rready: Master ready to accept
        mem_rdata: Read data (dma_buswidth bits)
        mem_rlast: Last beat of burst
        mem_rresp: Response status (2-bit, 00=OKAY)

    Parameters:
        config: SystolicConfig with DMA parameters
    """

    def __init__(self, config: SystolicConfig):
        self.config = config
        buswidth = config.dma_buswidth

        super().__init__(
            {
                # Request interface (from LoadController)
                "req_valid": In(1),
                "req_ready": Out(1),
                "req_addr": In(64),
                "req_len": In(8),  # AXI4 burst length (0-255 = 1-256 beats)
                # Response interface (to LoadController)
                "resp_valid": Out(1),
                "resp_ready": In(1),
                "resp_data": Out(buswidth),
                "resp_last": Out(1),
                # AXI Read Address Channel
                "mem_arvalid": Out(1),
                "mem_arready": In(1),
                "mem_araddr": Out(64),
                "mem_arlen": Out(8),
                "mem_arsize": Out(3),
                "mem_arburst": Out(2),
                # AXI Read Data Channel
                "mem_rvalid": In(1),
                "mem_rready": Out(1),
                "mem_rdata": In(buswidth),
                "mem_rlast": In(1),
                "mem_rresp": In(2),
            }
        )

    def elaborate(self, _platform):
        m = Module()
        cfg = self.config

        # Calculate arsize from bus width (log2 of bytes per beat)
        bytes_per_beat = cfg.dma_buswidth // 8
        arsize_value = (bytes_per_beat - 1).bit_length()  # log2

        # State machine
        # IDLE: Wait for request
        # ADDR: Issue address to AXI
        # DATA: Receive data beats and forward to response
        state_idle = 0
        state_addr = 1
        state_data = 2

        state = Signal(2, init=state_idle)

        # Registered request parameters
        addr_reg = Signal(64)
        len_reg = Signal(8)

        # Beat counter for tracking burst progress
        beat_count = Signal(8)

        # Default outputs
        m.d.comb += [
            self.req_ready.eq(0),
            self.resp_valid.eq(0),
            self.resp_data.eq(0),
            self.resp_last.eq(0),
            self.mem_arvalid.eq(0),
            self.mem_araddr.eq(0),
            self.mem_arlen.eq(0),
            self.mem_arsize.eq(arsize_value),
            self.mem_arburst.eq(0b01),  # INCR burst
            self.mem_rready.eq(0),
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
                    ]
                    m.d.sync += state.eq(state_addr)

            with m.Case(state_addr):
                # Issue address on AXI AR channel
                m.d.comb += [
                    self.mem_arvalid.eq(1),
                    self.mem_araddr.eq(addr_reg),
                    self.mem_arlen.eq(len_reg),
                ]

                with m.If(self.mem_arready):
                    # Address accepted, move to data phase
                    m.d.sync += state.eq(state_data)

            with m.Case(state_data):
                # Forward data from AXI R channel to response interface
                m.d.comb += [
                    self.mem_rready.eq(self.resp_ready),
                    self.resp_valid.eq(self.mem_rvalid),
                    self.resp_data.eq(self.mem_rdata),
                    self.resp_last.eq(self.mem_rlast),
                ]

                with m.If(self.mem_rvalid & self.resp_ready):
                    # Beat transferred
                    m.d.sync += beat_count.eq(beat_count + 1)

                    with m.If(self.mem_rlast):
                        # Burst complete, return to idle
                        m.d.sync += state.eq(state_idle)

        return m
