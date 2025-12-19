"""
MAC Array for Stencil Machine.

The MAC array computes dot products between the input window and multiple
filter kernels in parallel. It consists of P_c MAC banks, each processing
one output channel.

Architecture:
    ┌─────────────────────────────────────────────────────────────────────┐
    │                     CHANNEL-PARALLEL MAC ARRAY                       │
    │                                                                      │
    │  Window Input (K_h × K_w pixels)                                    │
    │        │                                                             │
    │        ▼                                                             │
    │  ┌───────────────────────────────────────────────────────────────┐  │
    │  │                    BROADCAST NETWORK                           │  │
    │  │    Window values broadcast to all P_c MAC banks                │  │
    │  └───────────┬─────────────┬─────────────┬───────────────────────┘  │
    │              │             │             │                           │
    │              ▼             ▼             ▼                           │
    │  ┌───────────────┐ ┌───────────────┐ ┌───────────────┐              │
    │  │  MAC Bank 0   │ │  MAC Bank 1   │ │  MAC Bank P-1 │              │
    │  │  ┌─────────┐  │ │  ┌─────────┐  │ │  ┌─────────┐  │              │
    │  │  │Filters  │  │ │  │Filters  │  │ │  │Filters  │  │              │
    │  │  │K×K coef │  │ │  │K×K coef │  │ │  │K×K coef │  │              │
    │  │  └────┬────┘  │ │  └────┬────┘  │ │  └────┬────┘  │              │
    │  │       │       │ │       │       │ │       │       │              │
    │  │  ┌────┴────┐  │ │  ┌────┴────┐  │ │  ┌────┴────┐  │              │
    │  │  │K×K MACs │  │ │  │K×K MACs │  │ │  │K×K MACs │  │              │
    │  │  └────┬────┘  │ │  └────┬────┘  │ │  └────┬────┘  │              │
    │  │       │       │ │       │       │ │       │       │              │
    │  │  ┌────┴────┐  │ │  ┌────┴────┐  │ │  ┌────┴────┐  │              │
    │  │  │Adder    │  │ │  │Adder    │  │ │  │Adder    │  │              │
    │  │  │Tree     │  │ │  │Tree     │  │ │  │Tree     │  │              │
    │  │  └────┬────┘  │ │  └────┬────┘  │ │  └────┬────┘  │              │
    │  │       │       │ │       │       │ │       │       │              │
    │  │  ┌────┴────┐  │ │  ┌────┴────┐  │ │  ┌────┴────┐  │              │
    │  │  │Accum    │  │ │  │Accum    │  │ │  │Accum    │  │              │
    │  │  │(32-bit) │  │ │  │(32-bit) │  │ │  │(32-bit) │  │              │
    │  │  └────┬────┘  │ │  └────┬────┘  │ │  └────┬────┘  │              │
    │  └───────┼───────┘ └───────┼───────┘ └───────┼───────┘              │
    │          │                 │                 │                       │
    │          └─────────────────┼─────────────────┘                       │
    │                            │                                         │
    │                            ▼                                         │
    │  ┌───────────────────────────────────────────────────────────────┐  │
    │  │                    OUTPUT COLLECTOR                            │  │
    │  │            P_c partial sums per output position                │  │
    │  └───────────────────────────────────────────────────────────────┘  │
    │                                                                      │
    └─────────────────────────────────────────────────────────────────────┘
"""

import math

from amaranth import Module, Signal, signed, unsigned
from amaranth.lib.wiring import Component, In, Out

from .config import StencilConfig


class MACBank(Component):
    """
    Single MAC bank computing one output channel.

    A MAC bank contains:
    - K_h × K_w filter coefficient registers
    - K_h × K_w multipliers (window × filter)
    - Adder tree to sum all products
    - Accumulator for summing across input channels

    The computation for one output pixel (one input channel) is:
        partial_sum = Σ(window[i,j] × filter[i,j]) for all i,j in kernel

    Ports:
        # Window input (broadcast from array)
        in_window: K_h × K_w pixels packed
        in_valid: Window data valid

        # Filter coefficients (loaded once per output tile)
        filter_load: Load new filter coefficients
        filter_data: K_h × K_w coefficients packed
        filter_channel: Input channel index for loaded filter

        # Accumulator control
        clear_accum: Reset accumulator to zero
        accum_en: Enable accumulator update

        # Output
        out_sum: Accumulated sum (acc_bits wide)
        out_valid: Output valid (after accumulation complete)
    """

    def __init__(self, config: StencilConfig, bank_id: int = 0):
        """
        Initialize a MAC bank.

        Args:
            config: Stencil machine configuration
            bank_id: Identifier for this bank
        """
        self.config = config
        self.bank_id = bank_id

        # Window and filter sizes
        self.window_size = config.max_kernel_h * config.max_kernel_w
        self.window_bits = config.input_bits * self.window_size
        self.filter_bits = config.weight_bits * self.window_size

        # Multiplier output width
        self.mult_width = config.input_bits + config.weight_bits

        # Adder tree depth
        self.adder_tree_depth = (
            math.ceil(math.log2(self.window_size)) if self.window_size > 1 else 0
        )

        super().__init__(
            {
                # Window input
                "in_window": In(unsigned(self.window_bits)),
                "in_valid": In(1),
                # Filter coefficients
                "filter_load": In(1),
                "filter_data": In(unsigned(self.filter_bits)),
                # Accumulator control
                "clear_accum": In(1),
                "accum_en": In(1),
                # Output
                "out_sum": Out(signed(config.acc_bits)),
                "out_valid": Out(1),
            }
        )

    def elaborate(self, _platform):
        m = Module()
        cfg = self.config
        K = self.window_size

        # =====================================================================
        # Filter Coefficient Registers
        # =====================================================================

        # Store K_h × K_w filter coefficients
        filter_regs = [Signal(signed(cfg.weight_bits), name=f"filt_{i}") for i in range(K)]

        # Load filter coefficients
        with m.If(self.filter_load):
            for i in range(K):
                bit_start = i * cfg.weight_bits
                bit_end = (i + 1) * cfg.weight_bits
                m.d.sync += filter_regs[i].eq(self.filter_data[bit_start:bit_end].as_signed())

        # =====================================================================
        # Multiplier Array
        # =====================================================================

        # K multipliers: window[i] × filter[i]
        products = [Signal(signed(self.mult_width), name=f"prod_{i}") for i in range(K)]

        for i in range(K):
            # Extract window pixel (unsigned input)
            pixel_start = i * cfg.input_bits
            pixel_end = (i + 1) * cfg.input_bits
            window_pixel = self.in_window[pixel_start:pixel_end]

            # Signed multiplication
            m.d.comb += products[i].eq(window_pixel.as_signed() * filter_regs[i])

        # =====================================================================
        # Adder Tree
        # =====================================================================

        # Build adder tree to sum all products
        # Level 0: products (K values)
        # Level 1: (K+1)//2 values
        # ...until 1 value

        # Pad products to next power of 2 for simpler tree
        tree_size = 1 << self.adder_tree_depth if K > 1 else 1

        # Current level of adder tree
        current_level = list(products)

        # Pad with zeros if needed
        while len(current_level) < tree_size:
            zero_sig = Signal(signed(self.mult_width), name=f"zero_{len(current_level)}")
            m.d.comb += zero_sig.eq(0)
            current_level.append(zero_sig)

        # Build tree levels
        level_num = 0
        # Determine maximum bit width needed (products + log2(K) extra bits for accumulation)
        sum_width = self.mult_width + self.adder_tree_depth

        while len(current_level) > 1:
            next_level = []
            for i in range(0, len(current_level), 2):
                if i + 1 < len(current_level):
                    sum_sig = Signal(signed(sum_width), name=f"sum_L{level_num}_{i // 2}")
                    m.d.comb += sum_sig.eq(
                        current_level[i].as_signed() + current_level[i + 1].as_signed()
                    )
                    next_level.append(sum_sig)
                else:
                    # Odd element, pass through
                    next_level.append(current_level[i])
            current_level = next_level
            level_num += 1

        # Final sum from adder tree
        tree_sum = current_level[0] if current_level else Signal(signed(sum_width))

        # =====================================================================
        # Accumulator
        # =====================================================================

        accumulator = Signal(signed(cfg.acc_bits), name="accumulator")

        with m.If(self.clear_accum):
            m.d.sync += accumulator.eq(0)
        with m.Elif(self.accum_en & self.in_valid):
            # Add tree sum to accumulator
            m.d.sync += accumulator.eq(accumulator + tree_sum)

        # Output
        m.d.comb += self.out_sum.eq(accumulator)

        # Valid output (pipeline delay would go here for timing)
        # For now, combinational output
        m.d.sync += self.out_valid.eq(self.in_valid & self.accum_en)

        return m


class ChannelParallelMAC(Component):
    """
    Channel-parallel MAC array with P_c parallel MAC banks.

    The array broadcasts the input window to all P_c banks, each of which
    computes the dot product with its own filter. This enables computing
    P_c output channels simultaneously.

    Ports:
        # Window input (broadcast to all banks)
        in_window_valid: Window data valid
        in_window: K_h × K_w pixels packed
        in_last_channel: Last input channel flag

        # Filter coefficients (loaded per bank)
        filter_load: Load filter to specified bank
        filter_data: K_h × K_w coefficients packed
        filter_bank: Target bank index

        # Partial sum output
        out_valid: All banks have valid output
        out_ready: Downstream ready
        out_data: P_c accumulated sums packed

        # Control
        clear_accum: Reset all accumulators
        cfg_kernel_h: Kernel height
        cfg_kernel_w: Kernel width
        cfg_active_banks: Number of active banks (for smaller C_out)
    """

    def __init__(self, config: StencilConfig):
        """
        Initialize the channel-parallel MAC array.

        Args:
            config: Stencil machine configuration
        """
        self.config = config

        # Window and filter sizes
        self.window_size = config.max_kernel_h * config.max_kernel_w
        self.window_bits = config.input_bits * self.window_size
        self.filter_bits = config.weight_bits * self.window_size

        # Output width: P_c accumulators
        self.out_width = config.acc_bits * config.parallel_channels

        # Bank index width
        self.bank_bits = max(1, (config.parallel_channels - 1).bit_length())

        super().__init__(
            {
                # Window input
                "in_window_valid": In(1),
                "in_window": In(unsigned(self.window_bits)),
                "in_last_channel": In(1),
                # Filter loading
                "filter_load": In(1),
                "filter_data": In(unsigned(self.filter_bits)),
                "filter_bank": In(unsigned(self.bank_bits)),
                # Output
                "out_valid": Out(1),
                "out_ready": In(1),
                "out_data": Out(signed(self.out_width)),
                # Control
                "clear_accum": In(1),
                "cfg_kernel_h": In(4),
                "cfg_kernel_w": In(4),
                "cfg_active_banks": In(unsigned(self.bank_bits + 1)),
            }
        )

    def elaborate(self, _platform):
        m = Module()
        cfg = self.config

        # =====================================================================
        # Instantiate P_c MAC Banks
        # =====================================================================

        banks = [MACBank(cfg, i) for i in range(cfg.parallel_channels)]
        for i, bank in enumerate(banks):
            m.submodules[f"mac_bank_{i}"] = bank

        # =====================================================================
        # Broadcast Window to All Banks
        # =====================================================================

        for bank in banks:
            m.d.comb += [
                bank.in_window.eq(self.in_window),
                bank.in_valid.eq(self.in_window_valid),
            ]

        # =====================================================================
        # Filter Loading: Route to Specific Bank
        # =====================================================================

        for i, bank in enumerate(banks):
            m.d.comb += [
                bank.filter_data.eq(self.filter_data),
                bank.filter_load.eq(self.filter_load & (self.filter_bank == i)),
            ]

        # =====================================================================
        # Accumulator Control
        # =====================================================================

        for bank in banks:
            m.d.comb += [
                bank.clear_accum.eq(self.clear_accum),
                bank.accum_en.eq(self.in_window_valid),
            ]

        # =====================================================================
        # Output Collection
        # =====================================================================

        # All banks output valid together (they process in lockstep)
        all_valid = Signal(name="all_valid")
        m.d.comb += all_valid.eq(banks[0].out_valid)

        # Pack outputs from all banks
        for i, bank in enumerate(banks):
            bit_start = i * cfg.acc_bits
            bit_end = (i + 1) * cfg.acc_bits
            m.d.comb += self.out_data[bit_start:bit_end].eq(bank.out_sum)

        # Output valid when all banks valid and on last channel
        # (accumulation complete for this output position)
        last_channel_delayed = Signal(name="last_channel_delayed")
        m.d.sync += last_channel_delayed.eq(self.in_last_channel)

        m.d.comb += self.out_valid.eq(all_valid & last_channel_delayed)

        return m
