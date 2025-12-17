"""
StridedStreamReader - DMA engine with strided access for transpose operations.

The StridedStreamReader extends basic DMA functionality to support:
- Contiguous burst reads (standard DMA)
- Strided reads for gathering non-contiguous data (e.g., matrix columns)

For matrix transpose during load:
- Source matrix A is M×K, stored row-major in DRAM
- To read column c: start at offset c, read M elements with stride K
- Each strided read gathers one column, written as one row in scratchpad

Memory Access Pattern for Column Read:
    DRAM Layout (M=4, K=3, row-major):
    ┌─────────────────────────────────┐
    │ A[0,0] A[0,1] A[0,2]           │  addresses 0, 1, 2
    │ A[1,0] A[1,1] A[1,2]           │  addresses 3, 4, 5
    │ A[2,0] A[2,1] A[2,2]           │  addresses 6, 7, 8
    │ A[3,0] A[3,1] A[3,2]           │  addresses 9, 10, 11
    └─────────────────────────────────┘

    To read column 1 (for transpose):
    - Base addr: 1 (element A[0,1])
    - Stride: 3 (K elements = row width)
    - Count: 4 (M elements)
    - Reads: addr 1, 4, 7, 10 → A[0,1], A[1,1], A[2,1], A[3,1]

Implementation Notes:
- For strided access, issues individual read transactions (not bursts)
- Accumulates elements into output buffer
- Presents completed row as single response when all elements gathered
- At 250MHz with 4-cycle SRAM, throughput is ~1 element per 5 cycles
- For M=16 column read: ~80 cycles per column
"""

from amaranth import Array, Module, Signal
from amaranth.lib.wiring import Component, In, Out

from ..config import SystolicConfig


class StridedStreamReader(Component):
    """
    DMA read engine with strided access support.

    Supports two modes:
    1. Contiguous: Standard burst read (stride=1, efficient)
    2. Strided: Gather read with configurable stride (for transpose)

    Request Interface:
        req_valid: Request valid
        req_ready: Ready to accept
        req_addr: Base address (64-bit)
        req_count: Number of elements to read
        req_stride: Stride between elements (0 or 1 = contiguous burst)
        req_elem_bytes: Bytes per element (for address calculation)

    Response Interface:
        resp_valid: Response valid (entire row ready)
        resp_ready: Consumer ready
        resp_data: Packed row data (sp_width bits)

    AXI Interface: Standard AR/R channels

    Parameters:
        config: SystolicConfig with DMA and array parameters
    """

    def __init__(self, config: SystolicConfig):
        self.config = config
        buswidth = config.dma_buswidth
        sp_width = config.sp_width

        # Max elements we can gather (one row of the array)
        max_elements = max(config.grid_rows * config.tile_rows, config.grid_cols * config.tile_cols)

        super().__init__(
            {
                # Request interface
                "req_valid": In(1),
                "req_ready": Out(1),
                "req_addr": In(64),
                "req_count": In(16),  # Number of elements
                "req_stride": In(16),  # Stride in elements (0/1 = contiguous)
                "req_elem_bytes": In(4),  # Bytes per element
                # Response interface (packed row)
                "resp_valid": Out(1),
                "resp_ready": In(1),
                "resp_data": Out(sp_width),
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
                # Status
                "busy": Out(1),
            }
        )

        self.max_elements = max_elements

    def elaborate(self, _platform):
        m = Module()
        cfg = self.config

        # Calculate arsize for single-element reads
        # We'll read one element at a time for strided access
        max_elem_bytes = max(cfg.input_bits, cfg.weight_bits, cfg.acc_bits) // 8
        max_elem_bytes = max(1, max_elem_bytes)

        # State machine
        STATE_IDLE = 0
        STATE_ISSUE_BURST = 1  # For contiguous reads
        STATE_RECV_BURST = 2
        STATE_ISSUE_STRIDED = 3  # For strided reads
        STATE_RECV_STRIDED = 4
        STATE_OUTPUT = 5

        state = Signal(3, init=STATE_IDLE)

        # Request parameters (registered)
        base_addr = Signal(64)
        elem_count = Signal(16)
        stride = Signal(16)
        elem_bytes = Signal(4)

        # Progress tracking
        elem_idx = Signal(16)  # Current element being read
        current_addr = Signal(64)  # Current read address

        # Accumulation buffer for gathered elements
        # Store as array of elements
        max_elem_bits = max(cfg.input_bits, cfg.weight_bits, cfg.acc_bits)
        accum_buffer = Array(
            [Signal(max_elem_bits, name=f"accum_{i}") for i in range(self.max_elements)]
        )

        # Determine if request is contiguous
        is_contiguous = Signal()

        # Default outputs
        m.d.comb += [
            self.req_ready.eq(0),
            self.resp_valid.eq(0),
            self.resp_data.eq(0),
            self.mem_arvalid.eq(0),
            self.mem_araddr.eq(0),
            self.mem_arlen.eq(0),
            self.mem_arsize.eq(0),
            self.mem_arburst.eq(0b01),  # INCR
            self.mem_rready.eq(0),
            self.busy.eq(state != STATE_IDLE),
        ]

        with m.FSM(init="IDLE"):
            with m.State("IDLE"):
                m.d.comb += [
                    self.req_ready.eq(1),
                    state.eq(STATE_IDLE),
                ]

                with m.If(self.req_valid):
                    # Latch request parameters
                    m.d.sync += [
                        base_addr.eq(self.req_addr),
                        elem_count.eq(self.req_count),
                        stride.eq(self.req_stride),
                        elem_bytes.eq(self.req_elem_bytes),
                        elem_idx.eq(0),
                        current_addr.eq(self.req_addr),
                    ]

                    # Check if contiguous (stride 0 or 1)
                    with m.If((self.req_stride == 0) | (self.req_stride == 1)):
                        m.d.sync += is_contiguous.eq(1)
                        m.next = "ISSUE_BURST"
                    with m.Else():
                        m.d.sync += is_contiguous.eq(0)
                        m.next = "ISSUE_STRIDED"

            # =========================================================
            # Contiguous burst path
            # =========================================================
            with m.State("ISSUE_BURST"):
                m.d.comb += state.eq(STATE_ISSUE_BURST)

                # Calculate burst parameters
                # arsize based on bus width
                arsize = (cfg.dma_buswidth // 8 - 1).bit_length()

                # For simplicity, read as single elements (could optimize with wider bursts)
                m.d.comb += [
                    self.mem_arvalid.eq(1),
                    self.mem_araddr.eq(base_addr),
                    # Read all elements as burst if bus supports it
                    # For now, simplified: one element per beat
                    self.mem_arlen.eq(elem_count - 1),  # AXI: len=N means N+1 beats
                    self.mem_arsize.eq(arsize),
                ]

                with m.If(self.mem_arready):
                    m.next = "RECV_BURST"

            with m.State("RECV_BURST"):
                m.d.comb += [
                    state.eq(STATE_RECV_BURST),
                    self.mem_rready.eq(1),
                ]

                with m.If(self.mem_rvalid):
                    # Store received element (extract from bus)
                    # Assuming element is in lower bits of rdata
                    m.d.sync += accum_buffer[elem_idx].eq(self.mem_rdata[:max_elem_bits])
                    m.d.sync += elem_idx.eq(elem_idx + 1)

                    with m.If(self.mem_rlast):
                        m.next = "OUTPUT"

            # =========================================================
            # Strided gather path
            # =========================================================
            with m.State("ISSUE_STRIDED"):
                m.d.comb += state.eq(STATE_ISSUE_STRIDED)

                # Issue single-element read at current address
                arsize = (elem_bytes - 1).bit_length() if elem_bytes > 0 else 0

                m.d.comb += [
                    self.mem_arvalid.eq(1),
                    self.mem_araddr.eq(current_addr),
                    self.mem_arlen.eq(0),  # Single beat
                    self.mem_arsize.eq(arsize),
                ]

                with m.If(self.mem_arready):
                    m.next = "RECV_STRIDED"

            with m.State("RECV_STRIDED"):
                m.d.comb += [
                    state.eq(STATE_RECV_STRIDED),
                    self.mem_rready.eq(1),
                ]

                with m.If(self.mem_rvalid):
                    # Store element
                    m.d.sync += accum_buffer[elem_idx].eq(self.mem_rdata[:max_elem_bits])
                    m.d.sync += elem_idx.eq(elem_idx + 1)

                    # Calculate next address
                    next_addr = current_addr + (stride * elem_bytes)
                    m.d.sync += current_addr.eq(next_addr)

                    # Check if done
                    with m.If(elem_idx >= elem_count - 1):
                        m.next = "OUTPUT"
                    with m.Else():
                        m.next = "ISSUE_STRIDED"

            # =========================================================
            # Output packed row
            # =========================================================
            with m.State("OUTPUT"):
                m.d.comb += state.eq(STATE_OUTPUT)

                # Pack accumulated elements into output
                # Elements are packed [elem_0, elem_1, ...] in resp_data
                packed_data = Signal(cfg.sp_width)

                for i in range(self.max_elements):
                    start_bit = i * max_elem_bits
                    end_bit = start_bit + max_elem_bits
                    if end_bit <= cfg.sp_width:
                        m.d.comb += packed_data[start_bit:end_bit].eq(accum_buffer[i])

                m.d.comb += [
                    self.resp_valid.eq(1),
                    self.resp_data.eq(packed_data),
                ]

                with m.If(self.resp_ready):
                    m.next = "IDLE"

        return m
