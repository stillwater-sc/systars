"""
Window Former for Stencil Machine.

The window former extracts a K_h × K_w convolution window from the line buffer
outputs using shift registers. Each cycle, a new column of K_h pixels enters
and the window slides by one position.

Architecture:
    ┌─────────────────────────────────────────────────────────────────────┐
    │                          WINDOW FORMER                               │
    │                                                                      │
    │  From Line Buffers (K_h pixels per cycle)                           │
    │        │                                                             │
    │        ▼                                                             │
    │  ┌───────────────────────────────────────────────────────────────┐  │
    │  │                   SHIFT REGISTER ARRAY                         │  │
    │  │                                                                │  │
    │  │  Row 0:  [col_0] ← [col_1] ← [col_2] ← ... ← [col_K-1] ← IN   │  │
    │  │  Row 1:  [col_0] ← [col_1] ← [col_2] ← ... ← [col_K-1] ← IN   │  │
    │  │    ...                                                         │  │
    │  │  Row K-1:[col_0] ← [col_1] ← [col_2] ← ... ← [col_K-1] ← IN   │  │
    │  │                                                                │  │
    │  └─────────────────────────────────────────────────────────────── ┘  │
    │                            │                                         │
    │                            ▼                                         │
    │  ┌───────────────────────────────────────────────────────────────┐  │
    │  │                    WINDOW OUTPUT                               │  │
    │  │        K_h × K_w pixels (flattened for MAC array)              │  │
    │  └───────────────────────────────────────────────────────────────┘  │
    │                            │                                         │
    │                            ▼                                         │
    │                      To MAC Array                                    │
    │                                                                      │
    └─────────────────────────────────────────────────────────────────────┘
"""

from amaranth import Module, Signal, unsigned
from amaranth.lib.wiring import Component, In, Out

from .config import StencilConfig


class WindowFormer(Component):
    """
    Sliding window extraction using shift registers.

    The window former maintains a K_h × K_w array of shift registers. Each
    cycle, when a new column of K_h pixels arrives from the line buffer,
    all registers shift and the new column enters at position K_w-1.

    Shift operation (K_w = 3 example):
        Cycle 0: Window = [col0, ---,  --- ] (filling)
        Cycle 1: Window = [col1, col0, --- ] (filling)
        Cycle 2: Window = [col2, col1, col0] ← First valid window
        Cycle 3: Window = [col3, col2, col1] ← Shifted by 1

    Stride handling:
        Stride 1: Output every cycle after window valid
        Stride 2: Output every 2nd cycle (skip alternate windows)
        Stride 4: Output every 4th cycle

    Ports:
        # Input from line buffers
        in_valid: Input data valid
        in_ready: Ready to accept input
        in_data: K_h pixels packed (from line buffer unit)

        # Output window to MAC array
        out_valid: Window output valid
        out_ready: MAC array ready
        out_window: K_h × K_w pixels packed
        out_col: Current output column index

        # Configuration
        cfg_kernel_h: Kernel height (1-7)
        cfg_kernel_w: Kernel width (1-7)
        cfg_stride_w: Horizontal stride (1, 2, or 4)
        cfg_width: Input width

        # Status
        window_valid: Window has K_w columns filled
        filling: Currently filling initial window
    """

    def __init__(self, config: StencilConfig):
        """
        Initialize the window former.

        Args:
            config: Stencil machine configuration
        """
        self.config = config

        # Bit widths
        self.kernel_bits = 4  # Enough for kernel size 7
        # addr_bits must be wide enough to hold max_width value (not just max_width-1)
        self.addr_bits = max(1, config.max_width.bit_length())

        # Input: K_h pixels from line buffer
        self.in_width = config.input_bits * config.max_kernel_h

        # Output: K_h × K_w pixels (complete window)
        self.out_width = config.input_bits * config.max_kernel_h * config.max_kernel_w

        super().__init__(
            {
                # Input from line buffers
                "in_valid": In(1),
                "in_ready": Out(1),
                "in_data": In(unsigned(self.in_width)),
                # Output window to MAC array
                "out_valid": Out(1),
                "out_ready": In(1),
                "out_window": Out(unsigned(self.out_width)),
                "out_col": Out(unsigned(self.addr_bits)),
                # Configuration
                "cfg_kernel_h": In(unsigned(self.kernel_bits)),
                "cfg_kernel_w": In(unsigned(self.kernel_bits)),
                "cfg_stride_w": In(unsigned(self.kernel_bits)),
                "cfg_width": In(unsigned(self.addr_bits)),
                # Status
                "window_valid": Out(1),
                "filling": Out(1),
            }
        )

    def elaborate(self, _platform):
        m = Module()
        cfg = self.config

        # =====================================================================
        # Shift Register Array: K_h rows × K_w columns
        # =====================================================================

        # Create 2D array of shift registers
        # shift_regs[row][col] holds one pixel
        shift_regs = [
            [
                Signal(unsigned(cfg.input_bits), name=f"sr_r{r}_c{c}")
                for c in range(cfg.max_kernel_w)
            ]
            for r in range(cfg.max_kernel_h)
        ]

        # =====================================================================
        # Fill Counter: Track how many columns have been shifted in
        # =====================================================================

        fill_count = Signal(unsigned(self.kernel_bits), name="fill_count")
        col_counter = Signal(unsigned(self.addr_bits), name="col_counter")

        # Input handshake
        input_xfer = Signal(name="input_xfer")
        m.d.comb += input_xfer.eq(self.in_valid & self.in_ready)

        # Ready to accept when output is ready (or we're still filling)
        # Add backpressure when window is valid but output not ready
        window_is_valid = Signal(name="window_is_valid")
        m.d.comb += window_is_valid.eq(fill_count >= self.cfg_kernel_w)

        with m.If(window_is_valid):
            # If window valid, only accept new data when output is consumed
            m.d.comb += self.in_ready.eq(self.out_ready)
        with m.Else():
            # Still filling, always ready
            m.d.comb += self.in_ready.eq(1)

        # =====================================================================
        # Shift Register Logic
        # =====================================================================

        with m.If(input_xfer):
            # Shift all columns to the left (older data moves left)
            for row in range(cfg.max_kernel_h):
                # Shift: col[0] gets col[1], col[1] gets col[2], etc.
                for col in range(cfg.max_kernel_w - 1):
                    m.d.sync += shift_regs[row][col].eq(shift_regs[row][col + 1])

                # New data enters at rightmost column (K_w - 1)
                # Extract row'th pixel from input data
                pixel_start = row * cfg.input_bits
                pixel_end = (row + 1) * cfg.input_bits
                m.d.sync += shift_regs[row][cfg.max_kernel_w - 1].eq(
                    self.in_data[pixel_start:pixel_end]
                )

            # Update fill counter
            with m.If(fill_count < cfg.max_kernel_w):
                m.d.sync += fill_count.eq(fill_count + 1)

            # Update column counter for output position
            m.d.sync += col_counter.eq(col_counter + 1)

            # Reset at end of row (when we reach cfg_width)
            with m.If(col_counter >= (self.cfg_width - 1)):
                m.d.sync += [
                    col_counter.eq(0),
                    fill_count.eq(0),  # Reset for next row
                ]

        # =====================================================================
        # Stride Counter: Skip windows based on stride
        # =====================================================================

        stride_counter = Signal(unsigned(self.kernel_bits), name="stride_counter")
        stride_match = Signal(name="stride_match")

        # Output on stride boundaries (when stride_counter == 0)
        m.d.comb += stride_match.eq(stride_counter == 0)

        with m.If(input_xfer & window_is_valid):
            with m.If(stride_counter >= (self.cfg_stride_w - 1)):
                m.d.sync += stride_counter.eq(0)
            with m.Else():
                m.d.sync += stride_counter.eq(stride_counter + 1)

        # Reset stride counter at end of row
        with m.If(input_xfer & (col_counter >= (self.cfg_width - 1))):
            m.d.sync += stride_counter.eq(0)

        # =====================================================================
        # Output Logic
        # =====================================================================

        # Output valid when window is full and on stride boundary
        m.d.comb += [
            self.out_valid.eq(window_is_valid & stride_match & self.in_valid),
            self.window_valid.eq(window_is_valid),
            self.filling.eq(~window_is_valid),
        ]

        # Pack shift register contents into output
        # Layout: [row0_col0, row0_col1, ..., row0_colK, row1_col0, ...]
        # That is: window[row][col] at bit position (row * K_w + col) * input_bits
        for row in range(cfg.max_kernel_h):
            for col in range(cfg.max_kernel_w):
                flat_idx = row * cfg.max_kernel_w + col
                bit_start = flat_idx * cfg.input_bits
                bit_end = (flat_idx + 1) * cfg.input_bits
                m.d.comb += self.out_window[bit_start:bit_end].eq(shift_regs[row][col])

        # Output column position
        # Adjust for fill count to get actual output position
        # The first valid output is at column (kernel_w - 1)
        out_col_raw = Signal(unsigned(self.addr_bits + 1), name="out_col_raw")
        m.d.comb += out_col_raw.eq(col_counter)

        # Clamp to valid range
        with m.If(out_col_raw >= self.cfg_width):
            m.d.comb += self.out_col.eq(self.cfg_width - 1)
        with m.Else():
            m.d.comb += self.out_col.eq(out_col_raw)

        return m
