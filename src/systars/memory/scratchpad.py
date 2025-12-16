"""
Scratchpad Memory.

The scratchpad is a multi-bank local memory for storing input activations
and weights. It provides:
- Multiple banks for parallel access
- Byte-level write masking for partial updates
- Configurable read latency

Architecture:
    ┌─────────────────────────────────────────────┐
    │              Scratchpad Controller          │
    │  ┌─────────┐ ┌─────────┐     ┌─────────┐   │
    │  │ Bank 0  │ │ Bank 1  │ ... │ Bank N  │   │
    │  └─────────┘ └─────────┘     └─────────┘   │
    └─────────────────────────────────────────────┘
"""

from amaranth import Module, Signal, unsigned
from amaranth.lib.memory import Memory
from amaranth.lib.wiring import Component, In, Out

from ..config import SystolicConfig
from .local_addr import LocalAddr


class ScratchpadBank(Component):
    """
    Single scratchpad bank with read and write ports.

    Features:
    - Single read port with configurable latency
    - Single write port with byte-level masking
    - Valid signal indicating read data availability

    Ports:
        read_addr: Address to read from
        read_en: Read enable
        read_data: Data read from memory (valid after latency)
        read_valid: High when read_data is valid

        write_addr: Address to write to
        write_en: Write enable
        write_data: Data to write
        write_mask: Byte-level write mask (1 = write that byte)
    """

    def __init__(self, config: SystolicConfig, bank_id: int):
        self.config = config
        self.bank_id = bank_id

        width = config.sp_width
        row_bits = max(1, (config.sp_bank_entries - 1).bit_length())

        super().__init__(
            {
                # Read port
                "read_addr": In(unsigned(row_bits)),
                "read_en": In(1),
                "read_data": Out(unsigned(width)),
                "read_valid": Out(1),
                # Write port
                "write_addr": In(unsigned(row_bits)),
                "write_en": In(1),
                "write_data": In(unsigned(width)),
                "write_mask": In(unsigned(width // 8)),
            }
        )

    def elaborate(self, _platform):
        m = Module()
        cfg = self.config

        # Create memory
        mem = Memory(shape=unsigned(cfg.sp_width), depth=cfg.sp_bank_entries, init=[])
        m.submodules.mem = mem

        # Read port
        rd_port = mem.read_port()
        m.d.comb += [
            rd_port.addr.eq(self.read_addr),
            rd_port.en.eq(self.read_en),
        ]

        # Pipeline read valid through latency stages
        # Memory has 1-cycle latency, add additional pipeline stages
        latency = max(1, cfg.spad_read_delay)
        if latency == 1:
            # Just the memory's inherent 1-cycle latency
            m.d.sync += self.read_valid.eq(self.read_en)
            m.d.comb += self.read_data.eq(rd_port.data)
        else:
            # Additional pipeline stages for read valid
            read_valid_pipe = [Signal(name=f"rv_{i}") for i in range(latency)]
            m.d.sync += read_valid_pipe[0].eq(self.read_en)
            for i in range(1, latency):
                m.d.sync += read_valid_pipe[i].eq(read_valid_pipe[i - 1])
            m.d.comb += self.read_valid.eq(read_valid_pipe[-1])

            # Pipeline the data as well
            data_pipe = [Signal(unsigned(cfg.sp_width), name=f"rd_{i}") for i in range(latency - 1)]
            if data_pipe:
                m.d.sync += data_pipe[0].eq(rd_port.data)
                for i in range(1, len(data_pipe)):
                    m.d.sync += data_pipe[i].eq(data_pipe[i - 1])
                m.d.comb += self.read_data.eq(data_pipe[-1])
            else:
                m.d.comb += self.read_data.eq(rd_port.data)

        # Write port with byte masking
        wr_port = mem.write_port(granularity=8)
        m.d.comb += [
            wr_port.addr.eq(self.write_addr),
            wr_port.data.eq(self.write_data),
        ]

        # Apply write mask: replicate write_en for each byte position
        num_bytes = cfg.sp_width // 8
        for byte_idx in range(num_bytes):
            m.d.comb += wr_port.en[byte_idx].eq(self.write_en & self.write_mask[byte_idx])

        return m


class Scratchpad(Component):
    """
    Multi-bank scratchpad memory with address decoding.

    The scratchpad contains multiple banks that can be accessed in parallel.
    Address decoding routes requests to the appropriate bank based on
    the bank selection bits in the address.

    Ports:
        read_req: Request to read
        read_addr: Full 32-bit local address
        read_data: Data read from selected bank
        read_valid: High when read_data is valid

        write_req: Request to write
        write_addr: Full 32-bit local address
        write_data: Data to write
        write_mask: Byte-level write mask
    """

    def __init__(self, config: SystolicConfig):
        self.config = config

        width = config.sp_width

        super().__init__(
            {
                # Read interface
                "read_req": In(1),
                "read_addr": In(unsigned(32)),
                "read_data": Out(unsigned(width)),
                "read_valid": Out(1),
                # Write interface
                "write_req": In(1),
                "write_addr": In(unsigned(32)),
                "write_data": In(unsigned(width)),
                "write_mask": In(unsigned(width // 8)),
            }
        )

    def elaborate(self, _platform):
        m = Module()
        cfg = self.config

        # Instantiate banks
        banks = [ScratchpadBank(cfg, i) for i in range(cfg.sp_banks)]
        for i, bank in enumerate(banks):
            m.submodules[f"bank_{i}"] = bank

        # Address decoding utilities
        addr_util = LocalAddr(cfg)

        # Read path: decode address and route to bank
        read_bank_sel = Signal(unsigned(addr_util.sp_bank_bits))
        read_row_addr = Signal(unsigned(addr_util.sp_row_bits))

        m.d.comb += [
            read_bank_sel.eq(addr_util.sp_bank(self.read_addr)),
            read_row_addr.eq(addr_util.sp_row(self.read_addr)),
        ]

        # Pipeline the bank selection to match read latency
        latency = max(1, cfg.spad_read_delay)
        bank_sel_pipe = [
            Signal(unsigned(addr_util.sp_bank_bits), name=f"bs_{i}") for i in range(latency)
        ]
        m.d.sync += bank_sel_pipe[0].eq(read_bank_sel)
        for i in range(1, latency):
            m.d.sync += bank_sel_pipe[i].eq(bank_sel_pipe[i - 1])

        # Route read request to all banks (they ignore if not selected)
        # but only one bank will be read based on address
        for i, bank in enumerate(banks):
            m.d.comb += [
                bank.read_addr.eq(read_row_addr),
                bank.read_en.eq(self.read_req & (read_bank_sel == i)),
            ]

        # Mux read data from selected bank (using pipelined selection)
        for i, bank in enumerate(banks):
            with m.If(bank_sel_pipe[-1] == i):
                m.d.comb += [
                    self.read_data.eq(bank.read_data),
                    self.read_valid.eq(bank.read_valid),
                ]

        # Write path: decode address and route to bank
        write_bank_sel = Signal(unsigned(addr_util.sp_bank_bits))
        write_row_addr = Signal(unsigned(addr_util.sp_row_bits))

        m.d.comb += [
            write_bank_sel.eq(addr_util.sp_bank(self.write_addr)),
            write_row_addr.eq(addr_util.sp_row(self.write_addr)),
        ]

        # Route write request to appropriate bank
        for i, bank in enumerate(banks):
            m.d.comb += [
                bank.write_addr.eq(write_row_addr),
                bank.write_data.eq(self.write_data),
                bank.write_mask.eq(self.write_mask),
                bank.write_en.eq(self.write_req & (write_bank_sel == i)),
            ]

        return m
