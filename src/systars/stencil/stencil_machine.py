"""
Stencil Machine Top-Level Integration.

The stencil machine is a dedicated hardware unit for energy-efficient 2D
convolution. It integrates:
- Line buffer unit for input row storage
- Window former for sliding window extraction
- Channel-parallel MAC array for parallel output channel computation

This design achieves 1× DRAM reads per input pixel (vs 9× for im2col with
3×3 kernels), providing significant energy savings for CNN inference.

Top-Level Architecture:
    ┌─────────────────────────────────────────────────────────────────────┐
    │                        STENCIL MACHINE                               │
    │                                                                      │
    │  ┌─────────────┐   ┌─────────────┐   ┌─────────────────────────┐   │
    │  │   Input     │   │   Line      │   │     Window              │   │
    │  │   Stream    │──►│   Buffer    │──►│     Former              │   │
    │  │             │   │   Unit      │   │                         │   │
    │  └─────────────┘   └─────────────┘   └───────────┬─────────────┘   │
    │                                                   │                  │
    │  ┌─────────────┐                                 │                  │
    │  │   Filter    │                                 │                  │
    │  │   Buffer    │─────────────────────────────────┼───┐              │
    │  │             │                                 │   │              │
    │  └─────────────┘                                 │   │              │
    │                                                   ▼   ▼              │
    │                                        ┌─────────────────────────┐  │
    │                                        │   Channel-Parallel      │  │
    │                                        │   MAC Array             │  │
    │                                        │                         │  │
    │                                        └───────────┬─────────────┘  │
    │                                                    │                 │
    │                                                    ▼                 │
    │                                        ┌─────────────────────────┐  │
    │                                        │   Output Processing     │  │
    │                                        │   (Activation, Quant)   │  │
    │                                        └───────────┬─────────────┘  │
    │                                                    │                 │
    │  ┌─────────────────────────────────────────────────┼─────────────┐  │
    │  │                    CONTROLLER FSM                │             │  │
    │  └─────────────────────────────────────────────────┼─────────────┘  │
    │                                                    │                 │
    │                                                    ▼                 │
    │                                             Output Stream            │
    └─────────────────────────────────────────────────────────────────────┘
"""

from enum import IntEnum

from amaranth import Module, Signal, signed, unsigned
from amaranth.lib.wiring import Component, In, Out

from .config import StencilActivation, StencilConfig
from .line_buffer import LineBufferUnit
from .mac_array import ChannelParallelMAC
from .window_former import WindowFormer


class StencilState(IntEnum):
    """FSM states for stencil machine controller."""

    IDLE = 0
    CONFIG = 1
    FILL_BUFFER = 2
    COMPUTE = 3
    NEXT_ROW = 4
    NEXT_OC_TILE = 5
    DONE = 6


class StencilMachine(Component):
    """
    Top-level stencil machine for Conv2D operations.

    The stencil machine processes Conv2D layers using a streaming dataflow:
    1. Input pixels stream into line buffers (one row at a time)
    2. Window former extracts K_h × K_w sliding windows
    3. MAC array computes P_c output channels in parallel
    4. Output streams to DRAM after optional activation

    Ports:
        # Input stream (from DRAM/memory controller)
        in_valid: Input pixel valid
        in_ready: Ready to accept input
        in_data: Input pixel data
        in_last_col: End of row marker
        in_last_row: End of frame marker
        in_channel: Current input channel

        # Filter stream (from DRAM)
        filter_valid: Filter data valid
        filter_ready: Ready for filter data
        filter_data: Filter coefficients (K_h × K_w)
        filter_bank: Target MAC bank for filter
        filter_channel: Input channel index

        # Output stream (to DRAM)
        out_valid: Output data valid
        out_ready: Downstream ready
        out_data: P_c output values packed
        out_last_col: End of output row
        out_last_row: End of output frame

        # Control
        start: Start processing
        done: Processing complete

        # Configuration (set before start)
        cfg_in_height: Input height
        cfg_in_width: Input width
        cfg_in_channels: Input channel count
        cfg_out_channels: Output channel count
        cfg_kernel_h: Kernel height
        cfg_kernel_w: Kernel width
        cfg_stride_h: Vertical stride
        cfg_stride_w: Horizontal stride
        cfg_padding: Padding mode
        cfg_activation: Activation function

        # Status
        state: Current FSM state
        rows_processed: Output rows completed
        channels_processed: Input channels completed
    """

    def __init__(self, config: StencilConfig):
        """
        Initialize the stencil machine.

        Args:
            config: Stencil machine configuration
        """
        self.config = config

        # Bit widths
        self.dim_bits = 16
        self.channel_bits = 16
        self.kernel_bits = 4
        self.state_bits = 4

        # Window and output sizes
        self.window_bits = config.input_bits * config.max_kernel_h * config.max_kernel_w
        self.filter_bits = config.weight_bits * config.max_kernel_h * config.max_kernel_w
        self.out_width = config.acc_bits * config.parallel_channels

        # Bank index width
        self.bank_bits = max(1, (config.parallel_channels - 1).bit_length())

        super().__init__(
            {
                # Input stream
                "in_valid": In(1),
                "in_ready": Out(1),
                "in_data": In(unsigned(config.input_bits)),
                "in_last_col": In(1),
                "in_last_row": In(1),
                "in_channel": In(unsigned(self.channel_bits)),
                # Filter stream
                "filter_valid": In(1),
                "filter_ready": Out(1),
                "filter_data": In(unsigned(self.filter_bits)),
                "filter_bank": In(unsigned(self.bank_bits)),
                "filter_channel": In(unsigned(self.channel_bits)),
                # Output stream
                "out_valid": Out(1),
                "out_ready": In(1),
                "out_data": Out(signed(self.out_width)),
                "out_last_col": Out(1),
                "out_last_row": Out(1),
                # Control
                "start": In(1),
                "done": Out(1),
                # Configuration
                "cfg_in_height": In(unsigned(self.dim_bits)),
                "cfg_in_width": In(unsigned(self.dim_bits)),
                "cfg_in_channels": In(unsigned(self.channel_bits)),
                "cfg_out_channels": In(unsigned(self.channel_bits)),
                "cfg_kernel_h": In(unsigned(self.kernel_bits)),
                "cfg_kernel_w": In(unsigned(self.kernel_bits)),
                "cfg_stride_h": In(unsigned(self.kernel_bits)),
                "cfg_stride_w": In(unsigned(self.kernel_bits)),
                "cfg_activation": In(unsigned(4)),
                # Status
                "state": Out(unsigned(self.state_bits)),
                "rows_processed": Out(unsigned(self.dim_bits)),
                "channels_processed": Out(unsigned(self.channel_bits)),
            }
        )

    def elaborate(self, _platform):
        m = Module()
        cfg = self.config

        # =====================================================================
        # Submodule Instantiation
        # =====================================================================

        # Line buffer unit
        line_buffer = LineBufferUnit(cfg)
        m.submodules.line_buffer = line_buffer

        # Window former
        window_former = WindowFormer(cfg)
        m.submodules.window_former = window_former

        # MAC array
        mac_array = ChannelParallelMAC(cfg)
        m.submodules.mac_array = mac_array

        # =====================================================================
        # Configuration Distribution
        # =====================================================================

        # Line buffer configuration
        m.d.comb += [
            line_buffer.cfg_kernel_h.eq(self.cfg_kernel_h),
            line_buffer.cfg_width.eq(self.cfg_in_width),
        ]

        # Window former configuration
        m.d.comb += [
            window_former.cfg_kernel_h.eq(self.cfg_kernel_h),
            window_former.cfg_kernel_w.eq(self.cfg_kernel_w),
            window_former.cfg_stride_w.eq(self.cfg_stride_w),
            window_former.cfg_width.eq(self.cfg_in_width),
        ]

        # MAC array configuration
        m.d.comb += [
            mac_array.cfg_kernel_h.eq(self.cfg_kernel_h),
            mac_array.cfg_kernel_w.eq(self.cfg_kernel_w),
        ]

        # =====================================================================
        # Controller FSM
        # =====================================================================

        state = Signal(unsigned(self.state_bits), reset=StencilState.IDLE, name="state")
        m.d.comb += self.state.eq(state)

        # Counters
        row_counter = Signal(unsigned(self.dim_bits), name="row_counter")
        col_counter = Signal(unsigned(self.dim_bits), name="col_counter")
        in_channel_counter = Signal(unsigned(self.channel_bits), name="in_channel_counter")
        oc_tile_counter = Signal(unsigned(self.channel_bits), name="oc_tile_counter")

        m.d.comb += [
            self.rows_processed.eq(row_counter),
            self.channels_processed.eq(in_channel_counter),
        ]

        # Computed dimensions
        out_height = Signal(unsigned(self.dim_bits), name="out_height")
        out_width = Signal(unsigned(self.dim_bits), name="out_width")
        num_oc_tiles = Signal(unsigned(self.channel_bits), name="num_oc_tiles")

        # Simple output dimension calculation (no padding for now)
        # out_h = (in_h - kernel_h) / stride_h + 1
        m.d.comb += [
            out_height.eq((self.cfg_in_height - self.cfg_kernel_h) // self.cfg_stride_h + 1),
            out_width.eq((self.cfg_in_width - self.cfg_kernel_w) // self.cfg_stride_w + 1),
            num_oc_tiles.eq(
                (self.cfg_out_channels + cfg.parallel_channels - 1) // cfg.parallel_channels
            ),
        ]

        # =====================================================================
        # FSM State Machine
        # =====================================================================

        with m.FSM(name="controller"):
            with m.State("IDLE"):
                m.d.comb += self.done.eq(0)
                with m.If(self.start):
                    m.d.sync += [
                        row_counter.eq(0),
                        col_counter.eq(0),
                        in_channel_counter.eq(0),
                        oc_tile_counter.eq(0),
                    ]
                    m.next = "FILL_BUFFER"
                    m.d.sync += state.eq(StencilState.FILL_BUFFER)

            with m.State("FILL_BUFFER"):  # noqa: SIM117
                # Wait for line buffer to have enough rows
                with m.If(line_buffer.ready_for_compute):
                    m.next = "COMPUTE"
                    m.d.sync += state.eq(StencilState.COMPUTE)

            with m.State("COMPUTE"):  # noqa: SIM117
                # Process windows until end of row
                with m.If(window_former.out_valid & self.out_ready):
                    with m.If(col_counter >= (out_width - 1)):
                        # End of row
                        m.d.sync += col_counter.eq(0)
                        m.next = "NEXT_ROW"
                        m.d.sync += state.eq(StencilState.NEXT_ROW)
                    with m.Else():
                        m.d.sync += col_counter.eq(col_counter + 1)

            with m.State("NEXT_ROW"):
                with m.If(row_counter >= (out_height - 1)):
                    # End of all rows, check if more output channel tiles
                    m.d.sync += row_counter.eq(0)
                    m.next = "NEXT_OC_TILE"
                    m.d.sync += state.eq(StencilState.NEXT_OC_TILE)
                with m.Else():
                    m.d.sync += row_counter.eq(row_counter + 1)
                    m.next = "COMPUTE"
                    m.d.sync += state.eq(StencilState.COMPUTE)

            with m.State("NEXT_OC_TILE"):
                with m.If(oc_tile_counter >= (num_oc_tiles - 1)):
                    # All output channel tiles done
                    m.next = "DONE"
                    m.d.sync += state.eq(StencilState.DONE)
                with m.Else():
                    m.d.sync += oc_tile_counter.eq(oc_tile_counter + 1)
                    m.next = "FILL_BUFFER"
                    m.d.sync += state.eq(StencilState.FILL_BUFFER)

            with m.State("DONE"):
                m.d.comb += self.done.eq(1)
                m.next = "IDLE"
                m.d.sync += state.eq(StencilState.IDLE)

        # =====================================================================
        # Datapath Connections
        # =====================================================================

        # Input stream → Line buffer
        m.d.comb += [
            line_buffer.in_valid.eq(self.in_valid),
            self.in_ready.eq(line_buffer.in_ready),
            line_buffer.in_data.eq(self.in_data),
            line_buffer.in_last_col.eq(self.in_last_col),
            line_buffer.in_last_row.eq(self.in_last_row),
        ]

        # Line buffer → Window former
        m.d.comb += [
            window_former.in_valid.eq(line_buffer.out_valid),
            line_buffer.out_ready.eq(window_former.in_ready),
            window_former.in_data.eq(line_buffer.out_data),
        ]

        # Window former → MAC array
        m.d.comb += [
            mac_array.in_window_valid.eq(window_former.out_valid),
            mac_array.in_window.eq(window_former.out_window),
            mac_array.in_last_channel.eq(in_channel_counter >= (self.cfg_in_channels - 1)),
        ]

        # Filter loading
        m.d.comb += [
            self.filter_ready.eq(1),  # Always ready for filter data
            mac_array.filter_load.eq(self.filter_valid),
            mac_array.filter_data.eq(self.filter_data),
            mac_array.filter_bank.eq(self.filter_bank),
        ]

        # Clear accumulator at start of each output position
        clear_accum = Signal(name="clear_accum")
        m.d.comb += clear_accum.eq(in_channel_counter == 0)
        m.d.comb += mac_array.clear_accum.eq(clear_accum & window_former.out_valid)

        # =====================================================================
        # Output Processing
        # =====================================================================

        # MAC array output → Output stream
        mac_out_valid = mac_array.out_valid
        mac_out_data = mac_array.out_data

        # Apply activation function
        activated_data = Signal(signed(self.out_width), name="activated_data")

        with m.Switch(self.cfg_activation):
            with m.Case(StencilActivation.NONE.value):
                m.d.comb += activated_data.eq(mac_out_data)

            with m.Case(StencilActivation.RELU.value):
                # ReLU: max(0, x) for each of P_c outputs
                for i in range(cfg.parallel_channels):
                    bit_start = i * cfg.acc_bits
                    bit_end = (i + 1) * cfg.acc_bits
                    val = mac_out_data[bit_start:bit_end].as_signed()
                    with m.If(val < 0):
                        m.d.comb += activated_data[bit_start:bit_end].eq(0)
                    with m.Else():
                        m.d.comb += activated_data[bit_start:bit_end].eq(val)

            with m.Case(StencilActivation.RELU6.value):
                # ReLU6: min(6, max(0, x)) for each of P_c outputs
                relu6_max = 6
                for i in range(cfg.parallel_channels):
                    bit_start = i * cfg.acc_bits
                    bit_end = (i + 1) * cfg.acc_bits
                    val = mac_out_data[bit_start:bit_end].as_signed()
                    with m.If(val < 0):
                        m.d.comb += activated_data[bit_start:bit_end].eq(0)
                    with m.Elif(val > relu6_max):
                        m.d.comb += activated_data[bit_start:bit_end].eq(relu6_max)
                    with m.Else():
                        m.d.comb += activated_data[bit_start:bit_end].eq(val)

            with m.Default():
                m.d.comb += activated_data.eq(mac_out_data)

        # Output stream signals
        m.d.comb += [
            self.out_valid.eq(mac_out_valid),
            self.out_data.eq(activated_data),
            self.out_last_col.eq(col_counter >= (out_width - 1)),
            self.out_last_row.eq(row_counter >= (out_height - 1)),
        ]

        return m
