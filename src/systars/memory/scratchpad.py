"""
L2, the systolic tile's scratchpad: two bank groups of 1R1W macros (#24).

The Streamer reads two L2 rows per clock on two read ports (``read_*`` and
``read_b_*``): an A row and a B row in OS, the streamed tile on port A and PRELOAD's
weight column on port B in the stationary mode. The BlockMover writes on the write
port. Real SRAM macros have one read port and one write port, so L2 is built so that
the two reads of a clock never need the same macro:

- **Two bank groups.** The lower half of L2's rows is the A group, the upper half the
  B group (``L2Layout``). The ISA places A tiles in the A group and B tiles in the B
  group (``isa.command.TileBuffers``), so the two reads of a clock are always in
  different groups, in every dataflow: in IS, PRELOAD reads the A slot on port B while
  COMPUTE streams the B slot on port A, still one read per group.
- **Banks in a group.** A group's rows are interleaved over its ``N`` banks (row ``r``
  of the group is in bank ``r mod N``, at bank row ``r div N``), so the Streamer's
  consecutive rows walk the banks. ``N`` is at least the macro's occupancy (the clocks
  it is busy per access, ``l2_bank_occupancy``) and enough that no bank is deeper than
  ``l2_macro_rows``. A pipelined macro (occupancy 1) needs one bank per group for
  bandwidth; more banks come only from its depth.
- **A crossbar** gives each group's read port to whichever request addresses it: port
  A first. Each bank has its own read and write port, so the BlockMover's writes never
  wait for the Streamer's reads.

``error`` (sticky) is an access the macros can't take: both reads of a clock in one
group (port A is served, port B's data is wrong), an access to a bank still busy
from its last one (``l2_bank_occupancy > 1``), or a row past L2. A schedule whose A and
B tiles sit in their own groups never raises it.

``IdealScratchpad`` is the model L2 replaced: one memory with two read ports, so any
two rows can be read in the same clock. It is kept (``SystolicConfig.l2_ideal``) to
compare results and cycles against.
"""

from __future__ import annotations

from amaranth import Module, Mux, Signal, unsigned
from amaranth.lib.memory import Memory
from amaranth.lib.wiring import Component, In, Out

from ..config import SystolicConfig
from .local_addr import LocalAddr


class L2Layout:
    """Where an L2 row lives: its group (0 = A, 1 = B), bank in the group and bank row.

    ``place`` is the model (Python ints), ``decode`` the same mapping in hardware.
    Rows past L2 (``row >= sp_total_rows``) are out of range.
    """

    def __init__(self, config: SystolicConfig):
        self.rows = config.sp_total_rows
        self.group_rows = config.l2_group_rows
        self.banks = config.l2_group_banks
        self.bank_rows = config.sp_bank_entries
        self.bank_bits = max(1, (self.banks - 1).bit_length())
        self.bank_row_bits = max(1, (self.bank_rows - 1).bit_length())

    def place(self, row: int) -> tuple[int, int, int]:
        """``(group, bank, bank_row)`` of L2 row ``row``."""
        if not 0 <= row < self.rows:
            raise ValueError(f"L2 row {row} is past L2 ({self.rows} rows)")
        group, r = divmod(row, self.group_rows)
        return group, r % self.banks, r // self.banks

    def decode(self, m: Module, addr: Signal, prefix: str):
        """``(group, bank, bank_row, out_of_range)`` signals of a local address."""
        row = Signal(LocalAddr.DATA_BITS, name=f"{prefix}_row")
        m.d.comb += row.eq(LocalAddr.data(addr))
        group = Signal(name=f"{prefix}_group")
        r = Signal(range(self.group_rows), name=f"{prefix}_group_row")
        m.d.comb += [
            group.eq(row >= self.group_rows),
            r.eq(Mux(group, row - self.group_rows, row)),
        ]
        bank = Signal(self.bank_bits, name=f"{prefix}_bank")
        bank_row = Signal(self.bank_row_bits, name=f"{prefix}_bank_row")
        n = self.banks
        if n & (n - 1) == 0:  # a power of two: the low bits
            shift = (n - 1).bit_length()
            m.d.comb += [bank.eq(r[:shift] if shift else 0), bank_row.eq(r >> shift)]
        else:  # e.g. 3 banks: a constant divider
            m.d.comb += [bank.eq(r % n), bank_row.eq(r // n)]
        out = Signal(name=f"{prefix}_out_of_range")
        m.d.comb += out.eq(row >= self.rows)
        return group, bank, bank_row, out


class ScratchpadBank(Component):
    """
    One L2 bank: a 1R1W macro, pipelined, with a byte-masked write port.

    Ports:
        read_addr, read_en: a read (bank row); one per clock
        read_data, read_valid: its data, ``spad_read_delay`` clocks later
        write_addr, write_en, write_data, write_mask: a write (byte mask, 1 = write)
    """

    def __init__(self, config: SystolicConfig, bank_id: int):
        self.config = config
        self.bank_id = bank_id
        width = config.sp_width
        row_bits = max(1, (config.sp_bank_entries - 1).bit_length())
        super().__init__(
            {
                "read_addr": In(unsigned(row_bits)),
                "read_en": In(1),
                "read_data": Out(unsigned(width)),
                "read_valid": Out(1),
                "write_addr": In(unsigned(row_bits)),
                "write_en": In(1),
                "write_data": In(unsigned(width)),
                "write_mask": In(unsigned(width // 8)),
            }
        )

    def elaborate(self, _platform):
        m = Module()
        cfg = self.config

        mem = Memory(shape=unsigned(cfg.sp_width), depth=cfg.sp_bank_entries, init=[])
        m.submodules.mem = mem
        self.mem = mem  # for testbenches (no hardware)

        # The memory provides the first clock of latency; the rest is pipeline
        latency = max(1, cfg.spad_read_delay)
        rd_port = mem.read_port()
        m.d.comb += [rd_port.addr.eq(self.read_addr), rd_port.en.eq(self.read_en)]
        valid_pipe = [Signal(name=f"rv_{i}") for i in range(latency)]
        m.d.sync += valid_pipe[0].eq(self.read_en)
        for i in range(1, latency):
            m.d.sync += valid_pipe[i].eq(valid_pipe[i - 1])
        m.d.comb += self.read_valid.eq(valid_pipe[-1])
        value = rd_port.data
        for i in range(latency - 1):
            stage = Signal(unsigned(cfg.sp_width), name=f"rd_{i}")
            m.d.sync += stage.eq(value)
            value = stage
        m.d.comb += self.read_data.eq(value)

        wr_port = mem.write_port(granularity=8)
        m.d.comb += [wr_port.addr.eq(self.write_addr), wr_port.data.eq(self.write_data)]
        for byte_idx in range(cfg.sp_width // 8):
            m.d.comb += wr_port.en[byte_idx].eq(self.write_en & self.write_mask[byte_idx])

        return m


def _ports(width: int) -> dict:
    return {
        "read_req": In(1),
        "read_addr": In(unsigned(32)),
        "read_data": Out(unsigned(width)),
        "read_valid": Out(1),
        "read_b_req": In(1),
        "read_b_addr": In(unsigned(32)),
        "read_b_data": Out(unsigned(width)),
        "read_b_valid": Out(1),
        "write_req": In(1),
        "write_addr": In(unsigned(32)),
        "write_data": In(unsigned(width)),
        "write_mask": In(unsigned(width // 8)),
        "error": Out(1),
    }


class Scratchpad(Component):
    """
    L2: two bank groups of ``l2_group_banks`` 1R1W banks (see the module docstring).

    Ports:
        read_req, read_addr / read_data, read_valid: read port A (a local address, a
            row of L2); the data ``spad_read_delay`` clocks later
        read_b_*: read port B, the same
        write_req, write_addr, write_data, write_mask: the write port
        error: sticky, an access the macros can't take
    """

    def __init__(self, config: SystolicConfig):
        self.config = config
        self.layout = L2Layout(config)
        super().__init__(_ports(config.sp_width))

    def elaborate(self, _platform):
        m = Module()
        cfg, lay = self.config, self.layout
        n = lay.banks
        latency = max(1, cfg.spad_read_delay)

        groups = [[ScratchpadBank(cfg, g * n + b) for b in range(n)] for g in range(2)]
        self.banks = [bank for group in groups for bank in group]  # for testbenches
        for g, group in enumerate(groups):
            for b, bank in enumerate(group):
                m.submodules[f"g{g}_bank_{b}"] = bank

        error_now = Signal()
        error = Signal()
        m.d.sync += error.eq(error | error_now)
        m.d.comb += self.error.eq(error)
        faults = []

        # Read ports A and B, decoded
        reads = []
        for tag, req, addr in (("a", self.read_req, self.read_addr), ("b", self.read_b_req, self.read_b_addr)):  # fmt: skip
            group, bank, bank_row, out = lay.decode(m, addr, f"rd_{tag}")
            reads.append((tag, req, group, bank, bank_row))
            faults.append(req & out)
        (_, a_req, a_group, a_bank, a_row), (_, b_req, b_group, b_bank, b_row) = reads
        # Both reads in one group: port A keeps it, port B's data is wrong
        faults.append(a_req & b_req & (a_group == b_group))

        # The crossbar: each group's read port serves whichever port addresses it
        for g, group in enumerate(groups):
            use_a = Signal(name=f"g{g}_use_a")
            use_b = Signal(name=f"g{g}_use_b")
            m.d.comb += [
                use_a.eq(a_req & (a_group == g)),
                use_b.eq(b_req & (b_group == g) & ~use_a),
            ]
            for b, bank in enumerate(group):
                m.d.comb += [
                    bank.read_en.eq((use_a & (a_bank == b)) | (use_b & (b_bank == b))),
                    bank.read_addr.eq(Mux(use_a, a_row, b_row)),
                ]

        # Return path: each port's (group, bank) follows its read down the pipeline
        for tag, req, group, bank, _ in reads:
            sel = Signal(1 + lay.bank_bits, name=f"rd_{tag}_sel")
            m.d.comb += sel.eq(group * n + bank)
            pipe = [Signal(1 + lay.bank_bits, name=f"rd_{tag}_sel_{i}") for i in range(latency)]
            valid = [Signal(name=f"rd_{tag}_v_{i}") for i in range(latency)]
            m.d.sync += [pipe[0].eq(sel), valid[0].eq(req)]
            for i in range(1, latency):
                m.d.sync += [pipe[i].eq(pipe[i - 1]), valid[i].eq(valid[i - 1])]
            data = self.read_data if tag == "a" else self.read_b_data
            m.d.comb += (self.read_valid if tag == "a" else self.read_b_valid).eq(valid[-1])
            with m.Switch(pipe[-1]):
                for i, bank_c in enumerate(self.banks):
                    with m.Case(i):
                        m.d.comb += data.eq(bank_c.read_data)

        # The write port: route to its bank
        w_group, w_bank, w_row, w_out = lay.decode(m, self.write_addr, "wr")
        faults.append(self.write_req & w_out)
        for g, group in enumerate(groups):
            for b, bank in enumerate(group):
                m.d.comb += [
                    bank.write_addr.eq(w_row),
                    bank.write_data.eq(self.write_data),
                    bank.write_mask.eq(self.write_mask),
                    bank.write_en.eq(self.write_req & (w_group == g) & (w_bank == b)),
                ]

        # Occupancy: a macro busy for more than one clock per access refuses another
        # access on that port until it is free
        occ = cfg.l2_bank_occupancy
        if occ > 1:
            for i, bank in enumerate(self.banks):
                for kind, en in (("rd", bank.read_en), ("wr", bank.write_en)):
                    busy = Signal(range(occ), name=f"bank_{i}_{kind}_busy")
                    with m.If(en):
                        m.d.sync += busy.eq(occ - 1)
                    with m.Elif(busy != 0):
                        m.d.sync += busy.eq(busy - 1)
                    faults.append(en & (busy != 0))

        for f in faults:
            with m.If(f):
                m.d.comb += error_now.eq(1)
        return m


class IdealScratchpad(Component):
    """
    The idealized L2 that #24 replaced: one memory, two read ports, one write port.

    Any two rows can be read in the same clock, which real macros can't do. It has
    the same ports as ``Scratchpad`` (``error`` is always 0) and the same rows, so a
    program gives bit-identical results and cycles on either.
    """

    def __init__(self, config: SystolicConfig):
        self.config = config
        super().__init__(_ports(config.sp_width))

    def elaborate(self, _platform):
        m = Module()
        cfg = self.config
        rows = cfg.sp_total_rows
        mem = Memory(shape=unsigned(cfg.sp_width), depth=rows, init=[])
        m.submodules.mem = mem
        self.mem = mem  # for testbenches (no hardware)
        self.banks = [self]  # one bank, for testbenches that walk L2's banks
        latency = max(1, cfg.spad_read_delay)
        addr_bits = max(1, (rows - 1).bit_length())
        for tag, req, addr, data, valid in (
            ("a", self.read_req, self.read_addr, self.read_data, self.read_valid),
            ("b", self.read_b_req, self.read_b_addr, self.read_b_data, self.read_b_valid),
        ):
            port = mem.read_port()
            m.d.comb += [port.addr.eq(addr[:addr_bits]), port.en.eq(req)]
            vpipe = [Signal(name=f"rv_{tag}_{i}") for i in range(latency)]
            m.d.sync += vpipe[0].eq(req)
            for i in range(1, latency):
                m.d.sync += vpipe[i].eq(vpipe[i - 1])
            m.d.comb += valid.eq(vpipe[-1])
            value = port.data
            for i in range(latency - 1):
                stage = Signal(unsigned(cfg.sp_width), name=f"rd_{tag}_{i}")
                m.d.sync += stage.eq(value)
                value = stage
            m.d.comb += data.eq(value)
        wr = mem.write_port(granularity=8)
        m.d.comb += [wr.addr.eq(self.write_addr[:addr_bits]), wr.data.eq(self.write_data)]
        for byte_idx in range(cfg.sp_width // 8):
            m.d.comb += wr.en[byte_idx].eq(self.write_req & self.write_mask[byte_idx])
        return m


def make_l2(config: SystolicConfig) -> Scratchpad | IdealScratchpad:
    """The tile's L2: banked, or the idealized reference when ``config.l2_ideal``."""
    return IdealScratchpad(config) if config.l2_ideal else Scratchpad(config)
