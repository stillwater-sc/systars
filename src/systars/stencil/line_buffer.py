"""
Line Buffer Unit for Stencil Machine.

The line buffer stores K_h consecutive rows of the input feature map to enable
sliding window extraction without repeated DRAM reads. This is the key to
achieving 1× DRAM reads per input pixel (vs 9× for im2col with 3×3 kernels).

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                    LINE BUFFER UNIT                         │
    │                                                             │
    │   Input Stream    ┌─────────────────────────────────────┐   │
    │        │          │        Input Demux / Router         │   │
    │        ▼          └───────┬─────────┬─────────┬─────────┘   │
    │                           │         │         │             │
    │                           ▼         ▼         ▼             │
    │                   ┌───────────┐ ┌───────────┐ ┌───────────┐ │
    │                   │Line Buf 0 │ │Line Buf 1 │ │Line Buf K │ │
    │                   │  (SRAM)   │ │  (SRAM)   │ │  (SRAM)   │ │
    │                   └─────┬─────┘ └─────┬─────┘ └─────┬─────┘ │
    │                         │             │             │       │
    │                         └─────────────┼─────────────┘       │
    │                                       ▼                     │
    │                         ┌─────────────────────────────┐     │
    │                         │  Output Mux (K_h outputs)   │     │
    │                         └─────────────────────────────┘     │
    │                                       │                     │
    │                                       ▼                     │
    │                              To Window Former               │
    └─────────────────────────────────────────────────────────────┘
"""

from amaranth import Module, Signal, unsigned
from amaranth.lib.memory import Memory
from amaranth.lib.wiring import Component, In, Out

from .config import StencilConfig


class LineBufferBank(Component):
    """
    Single line buffer SRAM bank storing one row of input pixels.

    Each bank stores W_max pixels (for one input channel at a time in
    channel-serial mode). The bank has:
    - One write port for streaming in new row data
    - One read port for reading pixels during window extraction

    Ports:
        write_addr: Column address for writing
        write_en: Write enable
        write_data: Pixel data to write (input_bits wide)

        read_addr: Column address for reading
        read_en: Read enable
        read_data: Pixel data read (input_bits wide)
        read_valid: High when read_data is valid (1-cycle latency)
    """

    def __init__(self, config: StencilConfig, bank_id: int = 0):
        """
        Initialize a line buffer bank.

        Args:
            config: Stencil machine configuration
            bank_id: Identifier for this bank (for debugging)
        """
        self.config = config
        self.bank_id = bank_id

        # Address width for max_width entries (must hold max_width value)
        self.addr_bits = max(1, config.max_width.bit_length())

        super().__init__(
            {
                # Write port
                "write_addr": In(unsigned(self.addr_bits)),
                "write_en": In(1),
                "write_data": In(unsigned(config.input_bits)),
                # Read port
                "read_addr": In(unsigned(self.addr_bits)),
                "read_en": In(1),
                "read_data": Out(unsigned(config.input_bits)),
                "read_valid": Out(1),
            }
        )

    def elaborate(self, _platform):
        m = Module()
        cfg = self.config

        # Create memory for one row of pixels
        # Depth = max_width (one pixel per column)
        # Width = input_bits (typically 8 for INT8)
        mem = Memory(shape=unsigned(cfg.input_bits), depth=cfg.max_width, init=[])
        m.submodules.mem = mem

        # Read port with 1-cycle latency
        rd_port = mem.read_port()
        m.d.comb += [
            rd_port.addr.eq(self.read_addr),
            rd_port.en.eq(self.read_en),
            self.read_data.eq(rd_port.data),
        ]

        # Track read valid (1-cycle latency)
        m.d.sync += self.read_valid.eq(self.read_en)

        # Write port
        wr_port = mem.write_port()
        m.d.comb += [
            wr_port.addr.eq(self.write_addr),
            wr_port.data.eq(self.write_data),
            wr_port.en.eq(self.write_en),
        ]

        return m


class LineBufferUnit(Component):
    """
    Collection of K_h line buffers with circular buffer management.

    The unit stores K_h rows needed for window extraction. As new rows stream
    in, they overwrite the oldest row in circular fashion. The unit outputs
    K_h pixel values per cycle (one from each row) for the window former.

    Circular buffer operation (K_h = 3 example):
        Input row 0 → Buffer 0    Window uses: [0, -, -]
        Input row 1 → Buffer 1    Window uses: [0, 1, -]
        Input row 2 → Buffer 2    Window uses: [0, 1, 2] ← First valid
        Input row 3 → Buffer 0    Window uses: [1, 2, 0] (overwrite oldest)

    Ports:
        # Input stream (one pixel at a time)
        in_valid: Input pixel valid
        in_ready: Ready to accept input
        in_data: Input pixel data
        in_last_col: End of row marker
        in_last_row: End of frame marker

        # Output to window former (K_h parallel outputs)
        out_valid: Output data valid
        out_ready: Window former ready
        out_data: K_h pixels packed (K_h × input_bits)
        out_col: Current column index

        # Configuration
        cfg_kernel_h: Active kernel height (1-7)
        cfg_width: Input width

        # Status
        row_count: Number of rows received
        ready_for_compute: K_h rows are buffered and ready
    """

    def __init__(self, config: StencilConfig):
        """
        Initialize the line buffer unit.

        Args:
            config: Stencil machine configuration
        """
        self.config = config

        # Address/index bit widths (must hold max_width value)
        self.addr_bits = max(1, config.max_width.bit_length())
        self.row_bits = max(1, config.max_height.bit_length())
        self.kernel_bits = 4  # Enough for max kernel size 7

        # Output width: K_h pixels packed
        self.out_width = config.input_bits * config.max_kernel_h

        super().__init__(
            {
                # Input stream
                "in_valid": In(1),
                "in_ready": Out(1),
                "in_data": In(unsigned(config.input_bits)),
                "in_last_col": In(1),
                "in_last_row": In(1),
                # Output to window former
                "out_valid": Out(1),
                "out_ready": In(1),
                "out_data": Out(unsigned(self.out_width)),
                "out_col": Out(unsigned(self.addr_bits)),
                # Configuration
                "cfg_kernel_h": In(unsigned(self.kernel_bits)),
                "cfg_width": In(unsigned(self.addr_bits)),
                # Status
                "row_count": Out(unsigned(self.row_bits)),
                "ready_for_compute": Out(1),
            }
        )

    def elaborate(self, _platform):
        m = Module()
        cfg = self.config

        # Instantiate K_h line buffer banks
        banks = [LineBufferBank(cfg, i) for i in range(cfg.max_kernel_h)]
        for i, bank in enumerate(banks):
            m.submodules[f"bank_{i}"] = bank

        # =====================================================================
        # Write Path: Circular buffer management
        # =====================================================================

        # Current write position
        write_col = Signal(unsigned(self.addr_bits), name="write_col")
        write_row = Signal(unsigned(self.row_bits), name="write_row")

        # Which bank to write to (circular index)
        write_bank_idx = Signal(unsigned(self.kernel_bits), name="write_bank_idx")

        # Track rows filled for readiness
        rows_filled = Signal(unsigned(self.row_bits), name="rows_filled")

        # Input handshake
        input_xfer = Signal(name="input_xfer")
        m.d.comb += input_xfer.eq(self.in_valid & self.in_ready)

        # Always ready to accept input (could add backpressure later)
        m.d.comb += self.in_ready.eq(1)

        # Route write to appropriate bank
        for i, bank in enumerate(banks):
            m.d.comb += [
                bank.write_addr.eq(write_col),
                bank.write_data.eq(self.in_data),
                bank.write_en.eq(input_xfer & (write_bank_idx == i)),
            ]

        # Update write position on input transfer
        with m.If(input_xfer):
            with m.If(self.in_last_col):
                # End of row: advance to next row/bank
                m.d.sync += [
                    write_col.eq(0),
                    write_row.eq(write_row + 1),
                ]

                # Update circular bank index
                with m.If(write_bank_idx >= (self.cfg_kernel_h - 1)):
                    m.d.sync += write_bank_idx.eq(0)
                with m.Else():
                    m.d.sync += write_bank_idx.eq(write_bank_idx + 1)

                # Track total rows filled (saturate at max)
                with m.If(rows_filled < cfg.max_height):
                    m.d.sync += rows_filled.eq(rows_filled + 1)

                # Handle end of frame
                with m.If(self.in_last_row):
                    m.d.sync += [
                        write_row.eq(0),
                        rows_filled.eq(0),
                        write_bank_idx.eq(0),
                    ]
            with m.Else():
                # Advance column
                m.d.sync += write_col.eq(write_col + 1)

        # =====================================================================
        # Read Path: Output K_h pixels per cycle
        # =====================================================================

        # Read position for output
        read_col = Signal(unsigned(self.addr_bits), name="read_col")

        # The "base" bank for reading (oldest row in circular buffer)
        # This rotates as new rows are written
        read_base_bank = Signal(unsigned(self.kernel_bits), name="read_base_bank")

        # Output handshake
        output_xfer = Signal(name="output_xfer")
        m.d.comb += output_xfer.eq(self.out_valid & self.out_ready)

        # Read all banks at the same column (we'll reorder in output mux)
        for bank in banks:
            m.d.comb += [
                bank.read_addr.eq(read_col),
                bank.read_en.eq(self.ready_for_compute),
            ]

        # Output valid when we have enough rows and all banks have valid data
        # (1-cycle latency from read enable)
        banks_valid = Signal(name="banks_valid")
        m.d.sync += banks_valid.eq(self.ready_for_compute)
        m.d.comb += self.out_valid.eq(banks_valid)

        # Pack output data from K_h banks in correct order
        # The read_base_bank indicates which physical bank contains the oldest row
        # We need to output in logical order: [oldest_row, ..., newest_row]
        out_data_parts = []
        for logical_row in range(cfg.max_kernel_h):
            # Physical bank = (read_base_bank + logical_row) % cfg_kernel_h
            # For now, use a mux chain for each logical position
            physical_bank = Signal(unsigned(self.kernel_bits), name=f"phys_bank_{logical_row}")

            # Calculate physical bank index with modular arithmetic
            raw_idx = Signal(unsigned(self.kernel_bits + 1), name=f"raw_idx_{logical_row}")
            m.d.comb += raw_idx.eq(read_base_bank + logical_row)
            with m.If(raw_idx >= self.cfg_kernel_h):
                m.d.comb += physical_bank.eq(raw_idx - self.cfg_kernel_h)
            with m.Else():
                m.d.comb += physical_bank.eq(raw_idx)

            # Mux to select from physical banks
            row_data = Signal(unsigned(cfg.input_bits), name=f"row_data_{logical_row}")

            # Build mux tree for this logical row
            # Default to bank 0
            m.d.comb += row_data.eq(banks[0].read_data)
            for phys_idx in range(1, cfg.max_kernel_h):
                with m.If(physical_bank == phys_idx):
                    m.d.comb += row_data.eq(banks[phys_idx].read_data)

            out_data_parts.append(row_data)

        # Pack all K_h rows into output
        # out_data layout: [row_0 | row_1 | ... | row_(K_h-1)]
        # Each row is input_bits wide
        for i, row_data in enumerate(out_data_parts):
            m.d.comb += self.out_data[i * cfg.input_bits : (i + 1) * cfg.input_bits].eq(row_data)

        # Output column index
        # Need to delay by 1 cycle to match read latency
        read_col_delayed = Signal(unsigned(self.addr_bits), name="read_col_delayed")
        m.d.sync += read_col_delayed.eq(read_col)
        m.d.comb += self.out_col.eq(read_col_delayed)

        # Advance read position on output transfer
        with m.If(output_xfer):
            with m.If(read_col >= (self.cfg_width - 1)):
                # End of row: reset column, update base bank
                m.d.sync += read_col.eq(0)

                # Advance base bank (oldest row moves forward)
                with m.If(read_base_bank >= (self.cfg_kernel_h - 1)):
                    m.d.sync += read_base_bank.eq(0)
                with m.Else():
                    m.d.sync += read_base_bank.eq(read_base_bank + 1)
            with m.Else():
                m.d.sync += read_col.eq(read_col + 1)

        # =====================================================================
        # Status Signals
        # =====================================================================

        # Row count output
        m.d.comb += self.row_count.eq(rows_filled)

        # Ready for compute when we have K_h rows buffered
        m.d.comb += self.ready_for_compute.eq(rows_filled >= self.cfg_kernel_h)

        return m
