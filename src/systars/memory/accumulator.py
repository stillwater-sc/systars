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

from amaranth import Cat, Module, Mux, Signal, signed, unsigned
from amaranth.lib.memory import Memory
from amaranth.lib.wiring import Component, In, Out

from ..config import SystolicConfig
from .activation import activate_row
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

        # Apply the activation function to each acc_bits element of the row
        activated = Signal(signed(width))
        n_elems = width // cfg.acc_bits
        m.d.comb += activated.eq(
            activate_row(raw_data, self.read_activation, cfg.acc_bits, n_elems)
        )

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

        # Write port. Every write is pipelined by one cycle so that an accumulate
        # write can read the current value first (read-modify-write):
        #   cycle N:   request (write_en, write_addr, write_data, accumulate) is
        #              registered; for accumulate, the current value is read
        #   cycle N+1: the registered request is committed; for accumulate, the
        #              value read in cycle N is added to the registered data
        # The commit is selected by the *registered* accumulate flag, so callers
        # may drop `accumulate` together with `write_en` after the request cycle.
        wr_port = mem.write_port()

        # Read port for read-modify-write, transparent to the write port so that
        # back-to-back accumulates to the same row see each other's result.
        acc_rd_port = mem.read_port(transparent_for=(wr_port,))
        existing_value = Signal(signed(width))
        m.d.comb += [
            acc_rd_port.addr.eq(self.write_addr),
            acc_rd_port.en.eq(self.write_en & self.accumulate),
            existing_value.eq(acc_rd_port.data.as_signed()),
        ]

        write_en_d = Signal()
        write_addr_d = Signal(unsigned(max(1, (cfg.acc_bank_entries - 1).bit_length())))
        write_data_d = Signal(signed(width))
        accumulate_d = Signal()

        m.d.sync += [
            write_en_d.eq(self.write_en),
            write_addr_d.eq(self.write_addr),
            write_data_d.eq(self.write_data),
            accumulate_d.eq(self.write_en & self.accumulate),
        ]

        # Element-wise add for accumulate (no carries between packed elements)
        accumulated = Signal(signed(width))
        n_acc = width // cfg.acc_bits
        m.d.comb += accumulated.eq(
            Cat(
                *[
                    (
                        existing_value[i * cfg.acc_bits : (i + 1) * cfg.acc_bits].as_signed()
                        + write_data_d[i * cfg.acc_bits : (i + 1) * cfg.acc_bits].as_signed()
                    )[: cfg.acc_bits]
                    for i in range(n_acc)
                ]
            )
        )

        m.d.comb += [
            wr_port.addr.eq(write_addr_d),
            wr_port.data.eq(Mux(accumulate_d, accumulated, write_data_d).as_unsigned()),
            wr_port.en.eq(write_en_d),
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
