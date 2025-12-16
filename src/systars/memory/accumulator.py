"""
Accumulator Memory.

The accumulator is a multi-bank memory for storing computation results.
It provides:
- Multiple banks for parallel access
- Accumulate mode (add to existing value)
- Activation functions (RELU, etc.)
- Output scaling

Architecture:
    ┌─────────────────────────────────────────────┐
    │             Accumulator Controller          │
    │  ┌─────────┐ ┌─────────┐     ┌─────────┐   │
    │  │ Bank 0  │ │ Bank 1  │ ... │ Bank N  │   │
    │  │ + Scale │ │ + Scale │     │ + Scale │   │
    │  │ + Activ │ │ + Activ │     │ + Activ │   │
    │  └─────────┘ └─────────┘     └─────────┘   │
    └─────────────────────────────────────────────┘
"""

from amaranth import Module, Mux, Signal, signed, unsigned
from amaranth.lib.memory import Memory
from amaranth.lib.wiring import Component, In, Out

from ..config import Activation, SystolicConfig
from .local_addr import LocalAddr


class AccumulatorBank(Component):
    """
    Single accumulator bank with read/write ports and activation.

    Features:
    - Read port with optional activation function
    - Write port with accumulate mode (add to existing)
    - Configurable read latency

    Ports:
        read_addr: Address to read from
        read_en: Read enable
        read_data: Data read (with activation applied)
        read_valid: High when read_data is valid
        read_activation: Activation function select (0=NONE, 1=RELU)

        write_addr: Address to write to
        write_en: Write enable
        write_data: Data to write
        accumulate: 1 = add to existing, 0 = overwrite
    """

    def __init__(self, config: SystolicConfig, bank_id: int):
        self.config = config
        self.bank_id = bank_id

        width = config.acc_width
        row_bits = max(1, (config.acc_bank_entries - 1).bit_length())

        super().__init__(
            {
                # Read port with activation
                "read_addr": In(unsigned(row_bits)),
                "read_en": In(1),
                "read_data": Out(signed(width)),
                "read_valid": Out(1),
                "read_activation": In(3),  # Activation function select
                # Write port with accumulate mode
                "write_addr": In(unsigned(row_bits)),
                "write_en": In(1),
                "write_data": In(signed(width)),
                "accumulate": In(1),
            }
        )

    def elaborate(self, _platform):
        m = Module()
        cfg = self.config
        width = cfg.acc_width

        # Create memory (signed values stored as unsigned)
        mem = Memory(shape=unsigned(width), depth=cfg.acc_bank_entries, init=[])
        m.submodules.mem = mem

        # Read port
        rd_port = mem.read_port()
        m.d.comb += [
            rd_port.addr.eq(self.read_addr),
            rd_port.en.eq(self.read_en),
        ]

        # Raw data from memory (interpret as signed)
        raw_data = Signal(signed(width))
        m.d.comb += raw_data.eq(rd_port.data.as_signed())

        # Apply activation function
        activated = Signal(signed(width))

        with m.Switch(self.read_activation):
            with m.Case(Activation.NONE.value):
                m.d.comb += activated.eq(raw_data)
            with m.Case(Activation.RELU.value):
                # ReLU: max(0, x)
                m.d.comb += activated.eq(Mux(raw_data < 0, 0, raw_data))
            with m.Case(Activation.RELU6.value):
                # ReLU6: min(6, max(0, x)) - simplified, 6 should be scaled
                relu_out = Mux(raw_data < 0, 0, raw_data)
                # For integer, use a threshold (e.g., 6 << some_shift)
                # Simplified: just use 6 for now
                m.d.comb += activated.eq(Mux(relu_out > 6, 6, relu_out))
            with m.Default():
                m.d.comb += activated.eq(raw_data)

        # Pipeline through read latency
        # Memory read port has inherent 1-cycle latency (synchronous read)
        # acc_latency is TOTAL latency from read_en to valid output
        # So we need (acc_latency - 1) additional pipeline stages
        latency = max(1, cfg.acc_latency)
        extra_stages = latency - 1  # Memory provides 1 cycle

        if extra_stages == 0:
            # Memory latency only - output directly (with 1 cycle delay for valid)
            m.d.sync += self.read_valid.eq(self.read_en)
            m.d.comb += self.read_data.eq(activated)
        else:
            # Pipeline stages for data and valid (on top of memory's 1 cycle)
            valid_pipe = [Signal(name=f"v_{i}") for i in range(extra_stages + 1)]
            data_pipe = [Signal(signed(width), name=f"d_{i}") for i in range(extra_stages)]

            # First valid stage tracks read_en
            m.d.sync += valid_pipe[0].eq(self.read_en)

            # Data pipeline starts from memory output (activated is already 1 cycle delayed)
            if extra_stages >= 1:
                m.d.sync += data_pipe[0].eq(activated)

            # Additional pipeline stages
            for i in range(1, extra_stages + 1):
                m.d.sync += valid_pipe[i].eq(valid_pipe[i - 1])
            for i in range(1, extra_stages):
                m.d.sync += data_pipe[i].eq(data_pipe[i - 1])

            m.d.comb += self.read_valid.eq(valid_pipe[-1])
            if extra_stages >= 1:
                m.d.comb += self.read_data.eq(data_pipe[-1])
            else:
                m.d.comb += self.read_data.eq(activated)

        # Write port - need to handle accumulate mode
        # For accumulate, we need read-modify-write
        # This requires reading the current value first

        # Create a second read port for accumulate read-modify-write
        acc_rd_port = mem.read_port()
        existing_value = Signal(signed(width))
        m.d.comb += [
            acc_rd_port.addr.eq(self.write_addr),
            acc_rd_port.en.eq(self.write_en & self.accumulate),
            existing_value.eq(acc_rd_port.data.as_signed()),
        ]

        # Write port
        wr_port = mem.write_port()

        # Compute new value
        new_value = Signal(signed(width))
        with m.If(self.accumulate):
            # Add to existing value
            # Note: This has timing issues - existing_value is from previous cycle
            # For proper accumulate, need to pipeline the write
            m.d.comb += new_value.eq(existing_value + self.write_data)
        with m.Else():
            m.d.comb += new_value.eq(self.write_data)

        # Pipeline the write for accumulate mode
        # Write happens one cycle after request for accumulate
        write_en_d = Signal()
        write_addr_d = Signal(unsigned(max(1, (cfg.acc_bank_entries - 1).bit_length())))
        write_data_d = Signal(signed(width))
        accumulate_d = Signal()

        m.d.sync += [
            write_en_d.eq(self.write_en),
            write_addr_d.eq(self.write_addr),
            write_data_d.eq(self.write_data),
            accumulate_d.eq(self.accumulate),
        ]

        # Actual write value
        actual_write_value = Signal(signed(width))
        with m.If(accumulate_d):
            m.d.comb += actual_write_value.eq(existing_value + write_data_d)
        with m.Else():
            m.d.comb += actual_write_value.eq(write_data_d)

        m.d.comb += [
            wr_port.addr.eq(Mux(self.accumulate, write_addr_d, self.write_addr)),
            wr_port.data.eq(
                Mux(self.accumulate, actual_write_value, self.write_data).as_unsigned()
            ),
            wr_port.en.eq(Mux(self.accumulate, write_en_d, self.write_en)),
        ]

        return m


class Accumulator(Component):
    """
    Multi-bank accumulator memory with address decoding.

    The accumulator contains multiple banks for storing computation results.
    It supports accumulate mode (add to existing) and activation functions.

    Ports:
        read_req: Request to read
        read_addr: Full 32-bit local address
        read_data: Data read from selected bank (with activation)
        read_valid: High when read_data is valid
        read_activation: Activation function to apply

        write_req: Request to write
        write_addr: Full 32-bit local address
        write_data: Data to write
        accumulate: 1 = add to existing, 0 = overwrite
    """

    def __init__(self, config: SystolicConfig):
        self.config = config

        width = config.acc_width

        super().__init__(
            {
                # Read interface
                "read_req": In(1),
                "read_addr": In(unsigned(32)),
                "read_data": Out(signed(width)),
                "read_valid": Out(1),
                "read_activation": In(3),
                # Write interface
                "write_req": In(1),
                "write_addr": In(unsigned(32)),
                "write_data": In(signed(width)),
                "accumulate": In(1),
            }
        )

    def elaborate(self, _platform):
        m = Module()
        cfg = self.config

        # Instantiate banks
        banks = [AccumulatorBank(cfg, i) for i in range(cfg.acc_banks)]
        for i, bank in enumerate(banks):
            m.submodules[f"bank_{i}"] = bank

        # Address decoding utilities
        addr_util = LocalAddr(cfg)

        # Read path
        read_bank_sel = Signal(unsigned(addr_util.acc_bank_bits))
        read_row_addr = Signal(unsigned(addr_util.acc_row_bits))

        m.d.comb += [
            read_bank_sel.eq(addr_util.acc_bank(self.read_addr)),
            read_row_addr.eq(addr_util.acc_row(self.read_addr)),
        ]

        # Pipeline the bank selection to match read latency
        latency = max(1, cfg.acc_latency)
        bank_sel_pipe = [
            Signal(unsigned(addr_util.acc_bank_bits), name=f"abs_{i}") for i in range(latency)
        ]
        m.d.sync += bank_sel_pipe[0].eq(read_bank_sel)
        for i in range(1, latency):
            m.d.sync += bank_sel_pipe[i].eq(bank_sel_pipe[i - 1])

        # Route read request to banks
        for i, bank in enumerate(banks):
            m.d.comb += [
                bank.read_addr.eq(read_row_addr),
                bank.read_en.eq(self.read_req & (read_bank_sel == i)),
                bank.read_activation.eq(self.read_activation),
            ]

        # Mux read data from selected bank
        for i, bank in enumerate(banks):
            with m.If(bank_sel_pipe[-1] == i):
                m.d.comb += [
                    self.read_data.eq(bank.read_data),
                    self.read_valid.eq(bank.read_valid),
                ]

        # Write path
        write_bank_sel = Signal(unsigned(addr_util.acc_bank_bits))
        write_row_addr = Signal(unsigned(addr_util.acc_row_bits))

        m.d.comb += [
            write_bank_sel.eq(addr_util.acc_bank(self.write_addr)),
            write_row_addr.eq(addr_util.acc_row(self.write_addr)),
        ]

        # Route write request to appropriate bank
        for i, bank in enumerate(banks):
            m.d.comb += [
                bank.write_addr.eq(write_row_addr),
                bank.write_data.eq(self.write_data),
                bank.accumulate.eq(self.accumulate),
                bank.write_en.eq(self.write_req & (write_bank_sel == i)),
            ]

        return m
