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
    - Two read ports with the same configurable latency
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
                # Read port A
                "read_addr": In(unsigned(row_bits)),
                "read_en": In(1),
                "read_data": Out(unsigned(width)),
                "read_valid": Out(1),
                # Read port B (same latency as A)
                "read_b_addr": In(unsigned(row_bits)),
                "read_b_en": In(1),
                "read_b_data": Out(unsigned(width)),
                "read_b_valid": Out(1),
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

        # Two read ports with the same latency, so the execute controller can
        # read an A row and a B row in the same cycle from any banks
        latency = max(1, cfg.spad_read_delay)
        for addr, en, data, valid, tag in (
            (self.read_addr, self.read_en, self.read_data, self.read_valid, "a"),
            (self.read_b_addr, self.read_b_en, self.read_b_data, self.read_b_valid, "b"),
        ):
            rd_port = mem.read_port()
            m.d.comb += [rd_port.addr.eq(addr), rd_port.en.eq(en)]
            # The memory provides the first cycle of latency; the rest is pipeline
            valid_pipe = [Signal(name=f"rv_{tag}_{i}") for i in range(latency)]
            m.d.sync += valid_pipe[0].eq(en)
            for i in range(1, latency):
                m.d.sync += valid_pipe[i].eq(valid_pipe[i - 1])
            m.d.comb += valid.eq(valid_pipe[-1])
            value = rd_port.data
            for i in range(latency - 1):
                stage = Signal(unsigned(cfg.sp_width), name=f"rd_{tag}_{i}")
                m.d.sync += stage.eq(value)
                value = stage
            m.d.comb += data.eq(value)

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
        read_b_*: A second read port, identical to the first (same latency).
            Every bank has both ports, so the two never conflict.

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
                # Second read interface
                "read_b_req": In(1),
                "read_b_addr": In(unsigned(32)),
                "read_b_data": Out(unsigned(width)),
                "read_b_valid": Out(1),
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

        # Read paths: decode the address, route the request to its bank, and
        # mux the data back with the bank selection pipelined to the latency
        latency = max(1, cfg.spad_read_delay)
        for req, addr, data, valid, prefix in (
            (self.read_req, self.read_addr, self.read_data, self.read_valid, "read"),
            (self.read_b_req, self.read_b_addr, self.read_b_data, self.read_b_valid, "read_b"),
        ):
            bank_sel = Signal(unsigned(addr_util.sp_bank_bits), name=f"{prefix}_bank_sel")
            row_addr = Signal(unsigned(addr_util.sp_row_bits), name=f"{prefix}_row_addr")
            m.d.comb += [
                bank_sel.eq(addr_util.sp_bank(addr)),
                row_addr.eq(addr_util.sp_row(addr)),
            ]
            sel_pipe = [
                Signal(unsigned(addr_util.sp_bank_bits), name=f"{prefix}_bs_{i}")
                for i in range(latency)
            ]
            m.d.sync += sel_pipe[0].eq(bank_sel)
            for i in range(1, latency):
                m.d.sync += sel_pipe[i].eq(sel_pipe[i - 1])
            for i, bank in enumerate(banks):
                m.d.comb += [
                    getattr(bank, f"{prefix}_addr").eq(row_addr),
                    getattr(bank, f"{prefix}_en").eq(req & (bank_sel == i)),
                ]
                with m.If(sel_pipe[-1] == i):
                    m.d.comb += [
                        data.eq(getattr(bank, f"{prefix}_data")),
                        valid.eq(getattr(bank, f"{prefix}_valid")),
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
