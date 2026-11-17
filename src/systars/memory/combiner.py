"""
The Combiner: the unit on the result path that merges partials into L2's result group.

In the KPU model the Combiner sits between the L1 result buffers and L2. It merges
the partials of a reduction (the WS/IS sum over K blocks, and ``COMPUTE_ACCUMULATE``
onto an existing C) and fuses bias, scaling, activation and block-format conversion as
results are written to L2 (#68, ``docs/architecture/kpu-tile.md``). This is its first
part (#74): the merge, on L2's result group.

**The result group (C).** Result rows live in their own group of L2 banks beside the A
and B groups (``memory/scratchpad.py``), as the L2 model gives one group per concurrent
access stream (``docs/architecture/l2.md``). Its rows are ``combiner_width`` wide (``C``
results of ``acc_bits``), and a local address with the Combiner bit (``LocalAddr``) names
one; the bank is the address's low bits. Its banks are 1R1W macros, pipelined, with read
latency ``combiner_latency``, like the A and B groups'. There is no second read port:
the merge and STORE share the one.

**The merge.** A request (``write_*``: the Streamer's result rows, which never wait,
and in the clocks they leave free a LOAD's rows into the group) is a row to write, with
``accumulate`` to add it to the row's current value. Every request runs down a
``combiner_latency``-clock pipeline and commits at its end:

- an ``accumulate`` request reads its row in its first clock, on the bank's read port,
  and commits the element-wise sum (each ``acc_bits`` lane wraps on its own);
- an overwrite commits its row as it is.

So requests commit in order, one per clock, and the Streamer never waits. A row a
request reads may still be on its way to the bank: the read can't see the commits of
the ``combiner_latency`` clocks before it lands, so the pipeline keeps those commits
and forwards the latest one for the same row. Back-to-back K blocks onto the same rows
see each other's sums.

**The epilogue** (#75, ``memory/epilogue.py``): a request whose ``epilogue`` code has
``EPILOGUE_FINAL`` (the command that finishes a reduction) commits its merged row after
the epilogue: requantization and the SFU function, in the configured order. Forwarding
carries the finished row like any other.

**STORE** reads the group on the same read port, in clocks the merge isn't reading
(``read_ready``): the BlockMover waits, the Streamer never does. It is also held off a
row with a request still in the pipeline, so it reads only committed rows whatever the
order of the tokens and completions around it.
"""

from amaranth import Array, Cat, Module, Mux, Signal, signed, unsigned
from amaranth.lib.memory import Memory
from amaranth.lib.wiring import Component, In, Out

from ..config import SystolicConfig
from .epilogue import EPILOGUE_FINAL, PROFILES, SFU_SHIFT, apply_epilogue
from .local_addr import LocalAddr


class CombinerBank(Component):
    """
    One bank of L2's result group: a 1R1W macro, pipelined.

    Ports:
        read_addr, read_en: a read (bank row); one per clock
        read_data, read_valid: its data, ``combiner_latency`` clocks later
        write_addr, write_en, write_data: a write
    """

    def __init__(self, config: SystolicConfig, bank_id: int):
        self.config = config
        self.bank_id = bank_id
        width = config.combiner_width
        row_bits = max(1, (config.combiner_bank_entries - 1).bit_length())
        super().__init__(
            {
                "read_addr": In(unsigned(row_bits)),
                "read_en": In(1),
                "read_data": Out(signed(width)),
                "read_valid": Out(1),
                "write_addr": In(unsigned(row_bits)),
                "write_en": In(1),
                "write_data": In(signed(width)),
            }
        )

    def elaborate(self, _platform):
        m = Module()
        cfg = self.config
        width = cfg.combiner_width

        mem = Memory(shape=unsigned(width), depth=cfg.combiner_bank_entries, init=[])
        m.submodules.mem = mem
        self.mem = mem  # for testbenches (no hardware)

        # The memory provides the first clock of latency; the rest is pipeline
        latency = max(1, cfg.combiner_latency)
        rd = mem.read_port()
        m.d.comb += [rd.addr.eq(self.read_addr), rd.en.eq(self.read_en)]
        valid = [Signal(name=f"rv_{i}") for i in range(latency)]
        m.d.sync += valid[0].eq(self.read_en)
        for i in range(1, latency):
            m.d.sync += valid[i].eq(valid[i - 1])
        m.d.comb += self.read_valid.eq(valid[-1])
        value = rd.data.as_signed()
        for i in range(latency - 1):
            stage = Signal(signed(width), name=f"rd_{i}")
            m.d.sync += stage.eq(value)
            value = stage
        m.d.comb += self.read_data.eq(value)

        wr = mem.write_port()
        m.d.comb += [
            wr.addr.eq(self.write_addr),
            wr.data.eq(self.write_data.as_unsigned()),
            wr.en.eq(self.write_en),
        ]
        return m


class Combiner(Component):
    """
    The Combiner's merge, on L2's result group (see the module docstring).

    Ports:
        write_req, write_addr, write_data, accumulate: a result row to commit (a local
            address with the Combiner bit), added to the row's value with ``accumulate``.
            One per clock, never refused.
        write_epilogue: the request's epilogue code (``memory/epilogue.py``): FINAL and
            the profile its finished rows get
        scale_<n>, control_<n>, clip_<n>: profile ``n``'s configuration (the tile's
            ``CMB_SCALE``; ``CMB_CONTROL``: shift[5:0], requant[8], act_first[9],
            mxint8[10], sfu[14:12]; ``CMB_CLIP``: lo[15:0], hi[31:16])
        read_req, read_addr: STORE's read; it is taken in a clock with ``read_ready``
            (hold ``read_req`` until then)
        read_data, read_valid: its row, ``combiner_latency`` clocks after it was taken
    """

    def __init__(self, config: SystolicConfig):
        self.config = config
        width = config.combiner_width
        super().__init__(
            {
                "read_req": In(1),
                "read_addr": In(unsigned(32)),
                "read_ready": Out(1),
                "read_data": Out(signed(width)),
                "read_valid": Out(1),
                "write_req": In(1),
                "write_addr": In(unsigned(32)),
                "write_data": In(signed(width)),
                "accumulate": In(1),
                "write_epilogue": In(4),
                # A request is in the pipeline: reconfigure the epilogue only when
                # this (and the tile's busy) is clear
                "busy": Out(1),
                **{f"scale_{n}": In(32, init=1) for n in range(PROFILES)},
                **{f"control_{n}": In(32) for n in range(PROFILES)},
                **{f"clip_{n}": In(32) for n in range(PROFILES)},
            }
        )

    def elaborate(self, _platform):
        m = Module()
        cfg = self.config
        width = cfg.combiner_width
        eb = cfg.acc_bits
        lanes = width // eb
        latency = max(1, cfg.combiner_latency)
        la = LocalAddr(cfg)
        key_bits = la.combiner_bank_bits + la.combiner_row_bits

        banks = [CombinerBank(cfg, i) for i in range(cfg.combiner_banks)]
        self.banks = banks  # for testbenches (no hardware)
        for i, bank in enumerate(banks):
            m.submodules[f"bank_{i}"] = bank

        def key(addr):
            """A row's bank and bank row: the address bits that name it."""
            return addr[:key_bits] if key_bits else Signal(1)

        # --- Request stage: an accumulate reads its row now ---
        w_bank = Signal(max(1, la.combiner_bank_bits))
        w_row = Signal(la.combiner_row_bits)
        m.d.comb += [w_bank.eq(la.combiner_bank(self.write_addr)), w_row.eq(la.combiner_row(self.write_addr))]  # fmt: skip
        merge_reads = Signal()
        m.d.comb += merge_reads.eq(self.write_req & self.accumulate)

        # The pipeline: stage i holds the request of i + 1 clocks ago; it commits at
        # the end (stage latency - 1), when its read's data arrives
        st_valid = [Signal(name=f"p{i}_valid") for i in range(latency)]
        st_acc = [Signal(name=f"p{i}_acc") for i in range(latency)]
        st_key = [Signal(max(1, key_bits), name=f"p{i}_key") for i in range(latency)]
        st_bank = [Signal(max(1, la.combiner_bank_bits), name=f"p{i}_bank") for i in range(latency)]
        st_row = [Signal(la.combiner_row_bits, name=f"p{i}_row") for i in range(latency)]
        st_data = [Signal(signed(width), name=f"p{i}_data") for i in range(latency)]
        st_ep = [Signal(4, name=f"p{i}_epilogue") for i in range(latency)]
        m.d.sync += st_ep[0].eq(Mux(self.write_req, self.write_epilogue, 0))
        for i in range(1, latency):
            m.d.sync += st_ep[i].eq(st_ep[i - 1])
        m.d.sync += [
            st_valid[0].eq(self.write_req),
            st_acc[0].eq(merge_reads),
            st_key[0].eq(key(self.write_addr)),
            st_bank[0].eq(w_bank),
            st_row[0].eq(w_row),
            st_data[0].eq(self.write_data),
        ]
        for i in range(1, latency):
            m.d.sync += [
                st_valid[i].eq(st_valid[i - 1]),
                st_acc[i].eq(st_acc[i - 1]),
                st_key[i].eq(st_key[i - 1]),
                st_bank[i].eq(st_bank[i - 1]),
                st_row[i].eq(st_row[i - 1]),
                st_data[i].eq(st_data[i - 1]),
            ]
        c_valid, c_acc, c_key = st_valid[-1], st_acc[-1], st_key[-1]
        m.d.comb += self.busy.eq(self.write_req | Cat(*st_valid).any())
        c_bank, c_row, c_data = st_bank[-1], st_row[-1], st_data[-1]

        # The commits of the last `latency` clocks: a read issued `latency` clocks ago
        # didn't see them (newest first)
        h_valid = [Signal(name=f"h{i}_valid") for i in range(latency)]
        h_key = [Signal(max(1, key_bits), name=f"h{i}_key") for i in range(latency)]
        h_data = [Signal(signed(width), name=f"h{i}_data") for i in range(latency)]

        # --- Commit stage ---
        from_bank = Signal(signed(width))
        with m.Switch(c_bank):
            for i, bank in enumerate(banks):
                with m.Case(i):
                    m.d.comb += from_bank.eq(bank.read_data)
        current = Signal(signed(width))
        m.d.comb += current.eq(from_bank)
        for i in reversed(range(latency)):  # the newest match wins: checked last
            with m.If(h_valid[i] & (h_key[i] == c_key)):
                m.d.comb += current.eq(h_data[i])
        summed = Signal(signed(width))
        m.d.comb += summed.eq(
            Cat(
                *[
                    (
                        current[i * eb : (i + 1) * eb].as_signed()
                        + c_data[i * eb : (i + 1) * eb].as_signed()
                    )[:eb]
                    for i in range(lanes)
                ]
            )  # fmt: skip
        )
        merged = Signal(signed(width))
        m.d.comb += merged.eq(Mux(c_acc, summed, c_data))
        # The profile the committing request names
        sel = st_ep[-1][:2]
        scale, control, clip = (Signal(32, name=f"profile_{k}") for k in ("scale", "control", "clip"))  # fmt: skip
        m.d.comb += [
            scale.eq(Array(getattr(self, f"scale_{n}") for n in range(PROFILES))[sel]),
            control.eq(Array(getattr(self, f"control_{n}") for n in range(PROFILES))[sel]),
            clip.eq(Array(getattr(self, f"clip_{n}") for n in range(PROFILES))[sel]),
        ]
        ep_cfg = {
            "mul": scale.as_signed(),
            "shift": control[0:6],
            "enable": control[8],
            "act_first": control[9],
            "mx": control[10],
            "sfu": control[SFU_SHIFT : SFU_SHIFT + 3],
            "clip_lo": clip[0:16].as_signed(),
            "clip_hi": clip[16:32].as_signed(),
        }
        finished = apply_epilogue(
            m, merged, ep_cfg, lanes=lanes, wide=eb, narrow=cfg.input_bits, width=width
        )
        commit = Signal(signed(width))
        m.d.comb += commit.eq(Mux(st_ep[-1] & EPILOGUE_FINAL, finished, merged))
        m.d.sync += [h_valid[0].eq(c_valid), h_key[0].eq(c_key), h_data[0].eq(commit)]
        for i in range(1, latency):
            m.d.sync += [
                h_valid[i].eq(h_valid[i - 1]),
                h_key[i].eq(h_key[i - 1]),
                h_data[i].eq(h_data[i - 1]),
            ]

        # --- STORE's read: a clock the merge leaves free, a row with nothing pending ---
        r_key = key(self.read_addr)
        pending = Signal()
        m.d.comb += pending.eq(self.write_req & (key(self.write_addr) == r_key))
        for i in range(latency):
            with m.If(st_valid[i] & (st_key[i] == r_key)):
                m.d.comb += pending.eq(1)
        grant = Signal()
        m.d.comb += [grant.eq(self.read_req & ~merge_reads & ~pending), self.read_ready.eq(grant)]
        r_bank = Signal(max(1, la.combiner_bank_bits))
        m.d.comb += r_bank.eq(la.combiner_bank(self.read_addr))

        # --- The banks: one read (the merge's or STORE's) and one write a clock ---
        for i, bank in enumerate(banks):
            m.d.comb += [
                bank.read_en.eq((merge_reads & (w_bank == i)) | (grant & (r_bank == i))),
                bank.read_addr.eq(Mux(merge_reads, w_row, la.combiner_row(self.read_addr))),
                bank.write_en.eq(c_valid & (c_bank == i)),
                bank.write_addr.eq(c_row),
                bank.write_data.eq(commit),
            ]

        # STORE's data: its read's bank, `latency` clocks later
        r_sel = [Signal(max(1, la.combiner_bank_bits), name=f"rs{i}_bank") for i in range(latency)]
        r_valid = [Signal(name=f"rs{i}_valid") for i in range(latency)]
        m.d.sync += [r_sel[0].eq(r_bank), r_valid[0].eq(grant)]
        for i in range(1, latency):
            m.d.sync += [r_sel[i].eq(r_sel[i - 1]), r_valid[i].eq(r_valid[i - 1])]
        m.d.comb += self.read_valid.eq(r_valid[-1])
        with m.Switch(r_sel[-1]):
            for i, bank in enumerate(banks):
                with m.Case(i):
                    m.d.comb += self.read_data.eq(bank.read_data)
        return m
