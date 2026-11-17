"""
Streamer - turns L2 tiles into streams through L1 into the fabric.

The Streamer is the only process that reads L2 for the fabric. It reads one L2 row
per clock on each of L2's two read ports (one per bank group, ``memory/scratchpad.py``)
and writes each row, in parallel, into one L1 stream buffer (``memory.stream_buffer``);
the buffer streams it into the fabric one element per clock. The buffers are loaded on
consecutive clocks, so buffer ``i`` starts one clock after buffer ``i - 1``: that
stagger is the skew. The fabric's results leave through the L1 result buffers, which
turn them back into row-major rows, and the Streamer writes those rows to the Combiner.

``R`` and ``C`` are the array's PE rows and columns, and ``KB`` the longest block a
stream buffer holds (``sp_width / input_bits`` A elements, ``sp_width /
weight_bits`` B elements: the smaller). An L2 row holds one block, element 0 in the
low bits. A Combiner row holds ``C`` results.

Modes (``CONFIG``, ``rs1``: 0 = OS, 1 = WS, 2 = IS; WS and IS are one fabric mode,
the stationary mode, and differ only in the operands the ISA lays out):

- **OS.** ``COMPUTE``: ``rs1`` is the L2 address of an A tile, row-major (``R`` rows,
  row ``i`` = ``A[i, k0:k0+K]``), ``rs2`` of a B tile, column-major (``C`` rows,
  row ``j`` = ``B[k0:k0+K, j]``), ``K = k_dim`` (at most ``KB``). A reduction may span
  several commands (``cmd_chain``): C stays in the PEs between them. The command
  that ends a reduction sends the C tile north and writes its ``R`` rows to the
  Combiner at ``rd``.
- **Stationary.** ``PRELOAD``: loads the stationary tile ``W`` (``R x C``) from
  ``rs1``, column-major (``C`` rows, row ``j`` = ``W[0:R, j]``, element 0 in the low
  bits); its ``k_dim`` rows (at most ``R``) are used and the rows below them get
  zero. ``COMPUTE``: ``rs1`` is a streamed tile, column-major (``R`` rows, row ``k`` =
  ``X[m0:m0+M, k]``), ``M = k_dim`` (at most ``KB``); the ``M`` rows of ``X @ W`` are
  written to the Combiner at ``rd``. The first ``COMPUTE`` after a ``PRELOAD``
  swaps the new weights in (``first`` on its stream, see ``PE``); later ones reuse
  them.

**PRELOAD runs beside the streams.** In the stationary mode only PRELOAD reads the
scratchpad's B port, so it has its own issue path there: it reads one L2 row (one
column of ``W``) per clock into the column's L1 weight buffer, and the buffers
stream their columns down the fabric's weight chains, bottom row first, staggered
one clock per column. The weights land in the PEs' shadow registers while the
previous stream still runs on the active ones.

``COMPUTE_ACCUMULATE`` adds the result rows to the Combiner instead of
overwriting them. ``cmd_chain`` (OS): bit 0 = this command continues the previous
command's reduction (no ``first`` marker), bit 1 = the reduction continues into the
next command (no ``last`` marker, no results). 0 is a whole reduction. A ``COMPUTE``
with ``k_dim = 0`` streams nothing and produces no results.

**Schedule.** A command starts only when the fabric can take it on schedule, and
then runs to the end without stopping:

- Streams: a command's reads are issued on consecutive clocks, and the next
  command's start ``I`` clocks later: ``I = max(K, R, C)`` (OS), ``max(M, R)``
  (stationary). With ``K >= R, C`` the next tile's blocks reach L1 exactly in the
  clock the previous blocks' last elements leave, so the streams have no bubbles
  between tiles.
- PRELOAD: its first read is issued in the clock it is accepted, and the next
  PRELOAD starts ``max(R, C)`` clocks later (the weight buffers stream ``R``
  elements). A stationary COMPUTE starts at least ``R - 1`` clocks after the last
  PRELOAD, so every PE's shadow holds its weight when the swap reaches it. A
  PRELOAD need not wait for the stream before it: its first weight reaches a PE
  after that stream's swap has passed it. So with ``M >= max(R, C)``, tiles with
  new weights (``PRELOAD``, ``COMPUTE``, ``PRELOAD``, ``COMPUTE``, ...) stream back
  to back.
- The OS drain: a command that ends a reduction starts only when its results
  will leave ``2R - 1`` clocks after the previous reduction's, so results never
  meet on a north link: with ``K >= 2R - 1`` that costs nothing.
- ``CONFIG`` waits until the Streamer is idle (every result written).

**Untracked commands** (``cmd_sync = 0``: a host issuing one command at a time)
run alone: one starts only when the Streamer is idle, nothing starts while it is
in flight, and a command with results completes once they are in the Combiner,
so a host can issue a STORE as soon as it sees the completion.

**Events**, at most one of each per clock, ``result`` in command order:

- ``completed`` (``completed_id``, ``completed_sync``): the command is done with
  L2: its last block is in L1. ``SystolicTop`` returns the command's L2 credit on it
  (by slot, so order does not matter). A PRELOAD's completion is a clock late,
  behind any other completion in the same clock, and may come before that of the
  COMPUTE issued ahead of it (the two paths run side by side).
- ``result`` (``result_id``, ``result_sync``): the command's results are in the
  Combiner. ``SystolicTop`` emits the Combiner tile's token on it.

``error`` (sticky) is a schedule or protocol violation: an L1 buffer loaded
without its credit or overflowing, a collision on a north link, a result
without a command, or a command with SIGNAL that writes no result (``k_dim = 0``,
an OS COMPUTE with ``CHAIN_MORE``, PRELOAD, CONFIG or an unknown opcode), whose
Combiner slot would never get its token. A valid schedule never raises it.

The command set (CONFIG, PRELOAD, COMPUTE, COMPUTE_ACCUMULATE) follows Gemmini's
execute commands (ucb-bar/gemmini; see NOTICE). The scheduling around it, L1 stream
buffers on per-buffer credits, follows the KPU's Streamer.
"""

from amaranth import Cat, Module, Mux, Signal, signed
from amaranth.lib.fifo import SyncFIFO
from amaranth.lib.wiring import Component, In, Out

from ..config import Dataflow, SystolicConfig
from ..memory.stream_buffer import ResultBuffers, StreamBuffers
from ..util.commands import ExecOpcode
from .sync import Sync

# Read kinds, carried with each read to its return
KIND_OS = 0
KIND_STATIONARY = 1

CHAIN_CONTINUES = 0b01  # cmd_chain bit 0: continues the previous reduction
CHAIN_MORE = 0b10  # cmd_chain bit 1: the reduction continues into the next command


class Streamer(Component):
    """
    The Streamer (see the module docstring).

    Ports:
        cmd_*: Command (``valid``/``ready``; fields read in the clock it is accepted)
        cfg_dataflow: Current mode (0 = OS, 1 = stationary)
        sp_a_read_*, sp_b_read_*: The scratchpad's two read ports
        combiner_write_*, combiner_accumulate: Combiner write port
        array_*: SystolicArray edges (see ``SystolicArray``)
        completed, completed_id, completed_sync: A command is done with L2
        result, result_id, result_sync: A command's results are in the Combiner
        busy, error, state_debug: Status
    """

    def __init__(self, config: SystolicConfig):
        """Build the ports for ``config``'s array and the L1 stream and result buffers."""
        self.config = config
        rows = config.grid_rows * config.tile_rows
        cols = config.grid_cols * config.tile_cols
        if (
            rows * config.input_bits > config.sp_width
            or cols * config.weight_bits > config.sp_width
        ):
            raise ValueError(
                f"a {config.sp_width}-bit scratchpad row cannot hold {rows} A elements "
                f"({config.input_bits}-bit) and {cols} B elements ({config.weight_bits}-bit)"
            )
        if (
            config.dataflow & (Dataflow.A_STATIONARY | Dataflow.B_STATIONARY)
            and rows * config.weight_bits > config.sp_width
        ):
            raise ValueError(
                f"a {config.sp_width}-bit scratchpad row cannot hold a {rows}-row column "
                f"of weights ({config.weight_bits}-bit)"
            )
        self.rows, self.cols = rows, cols
        self.block = min(
            config.sp_width // config.input_bits, config.sp_width // config.weight_bits
        )
        self.supports_os = bool(config.dataflow & Dataflow.OUTPUT_STATIONARY)
        self.supports_stationary = bool(
            config.dataflow & (Dataflow.A_STATIONARY | Dataflow.B_STATIONARY)
        )
        if not (self.supports_os or self.supports_stationary):
            raise ValueError(f"dataflow {config.dataflow!r} is not implemented")

        ports = {
            "cmd_valid": In(1),
            "cmd_ready": Out(1),
            "cmd_opcode": In(8),
            "cmd_rs1": In(32),
            "cmd_rs2": In(32),
            "cmd_rd": In(32),
            "cmd_k_dim": In(16),
            "cmd_chain": In(2),
            "cmd_sync": In(Sync.BITS),
            "cmd_id": In(8),
            "cfg_dataflow": Out(1),
            "sp_a_read_req": Out(1),
            "sp_a_read_addr": Out(32),
            "sp_a_read_data": In(config.sp_width),
            "sp_a_read_valid": In(1),
            "sp_b_read_req": Out(1),
            "sp_b_read_addr": Out(32),
            "sp_b_read_data": In(config.sp_width),
            "sp_b_read_valid": In(1),
            "combiner_write_req": Out(1),
            "combiner_write_addr": Out(32),
            "combiner_write_data": Out(signed(config.combiner_width)),
            "combiner_accumulate": Out(1),
            "completed": Out(1),
            "completed_id": Out(8),
            "completed_sync": Out(Sync.BITS),
            "result": Out(1),
            "result_id": Out(8),
            "result_sync": Out(Sync.BITS),
            "busy": Out(1),
            "error": Out(1),
            "state_debug": Out(8),
        }
        for i in range(rows):
            for name, shape in (
                ("a", signed(config.input_bits)), ("valid", 1), ("first", 1), ("last", 1)
            ):  # fmt: skip
                ports[f"array_in_{name}_{i}"] = Out(shape)
        for j in range(cols):
            ports[f"array_in_b_{j}"] = Out(signed(config.weight_bits))
            ports[f"array_in_w_{j}"] = Out(signed(config.weight_bits))
            ports[f"array_in_w_valid_{j}"] = Out(1)
            ports[f"array_out_c_{j}"] = In(signed(config.acc_bits))
            ports[f"array_out_c_valid_{j}"] = In(1)
            ports[f"array_out_psum_{j}"] = In(signed(config.acc_bits))
            ports[f"array_out_psum_valid_{j}"] = In(1)
        ports["array_in_dataflow"] = Out(1)
        ports["array_out_collision"] = In(1)
        super().__init__(ports)

    STATE_IDLE = 0
    STATE_BUSY = 1

    def elaborate(self, _platform):
        m = Module()
        cfg = self.config
        R, C, KB = self.rows, self.cols, self.block
        n_max = max(R, C)

        # ---------------------------------------------------------------------
        # L1
        # ---------------------------------------------------------------------
        m.submodules.a_buffers = a_buf = StreamBuffers(R, KB, cfg.input_bits)
        b_buf = None
        north = south = None
        if self.supports_os:
            m.submodules.b_buffers = b_buf = StreamBuffers(C, KB, cfg.weight_bits)
            m.submodules.north_results = north = ResultBuffers(C, C + 1, cfg.acc_bits)
        w_buf = None
        if self.supports_stationary:
            m.submodules.south_results = south = ResultBuffers(C, C + 1, cfg.acc_bits)
            # One per column: a column of W, streamed down the weight chain
            m.submodules.w_buffers = w_buf = StreamBuffers(C, R, cfg.weight_bits)
            for j in range(C):
                m.d.comb += [
                    getattr(self, f"array_in_w_{j}").eq(getattr(w_buf, f"out_{j}")),
                    getattr(self, f"array_in_w_valid_{j}").eq(getattr(w_buf, f"out_valid_{j}")),
                ]

        stationary = Signal(init=0 if self.supports_os else 1)
        m.d.comb += [self.cfg_dataflow.eq(stationary), self.array_in_dataflow.eq(stationary)]

        for i in range(R):
            for name in ("", "valid_", "first_", "last_"):
                port = "a" if name == "" else name[:-1]
                m.d.comb += getattr(self, f"array_in_{port}_{i}").eq(
                    getattr(a_buf, f"out_{name}{i}")
                )
        if b_buf is not None:
            for j in range(C):
                m.d.comb += getattr(self, f"array_in_b_{j}").eq(getattr(b_buf, f"out_{j}"))
        if north is not None:
            for j in range(C):
                m.d.comb += [
                    getattr(north, f"in_{j}").eq(getattr(self, f"array_out_c_{j}")),
                    getattr(north, f"in_valid_{j}").eq(
                        ~stationary & getattr(self, f"array_out_c_valid_{j}")
                    ),
                ]
        if south is not None:
            for j in range(C):
                m.d.comb += [
                    getattr(south, f"in_{j}").eq(getattr(self, f"array_out_psum_{j}")),
                    getattr(south, f"in_valid_{j}").eq(
                        stationary & getattr(self, f"array_out_psum_valid_{j}")
                    ),
                ]

        # ---------------------------------------------------------------------
        # Command decode and the start rule
        # ---------------------------------------------------------------------
        def clamp(value, limit):
            return Mux(value > limit, limit, value)

        opcode = self.cmd_opcode
        is_config = opcode == ExecOpcode.CONFIG
        is_preload = opcode == ExecOpcode.PRELOAD
        is_compute = (opcode == ExecOpcode.COMPUTE) | (opcode == ExecOpcode.COMPUTE_ACCUMULATE)
        k_len = Signal(range(KB + 1))
        m.d.comb += k_len.eq(clamp(self.cmd_k_dim, KB))
        first = ~self.cmd_chain[0]
        last = ~self.cmd_chain[1]

        os_stream = Signal()
        st_stream = Signal()
        preload = Signal()
        m.d.comb += [
            os_stream.eq(is_compute & ~stationary & (k_len != 0)),
            st_stream.eq(is_compute & stationary & (k_len != 0)),
            preload.eq(is_preload & stationary),
        ]
        produces = Signal()  # the command writes a result tile
        m.d.comb += produces.eq(st_stream | (os_stream & last))
        direct = Signal()  # the command reads nothing: it completes on its own
        m.d.comb += direct.eq(~(os_stream | st_stream | preload))

        # Schedule countdowns (clocks until the fabric can take the next ...)
        stream_wait = Signal(range(2 * (KB + n_max) + 1))  # ... stream
        drain_wait = Signal(range(KB + 2 * R + 1))  # ... OS reduction end (north drain)
        preload_wait = Signal(range(n_max + 1))  # ... PRELOAD (weight buffers free)
        swap_wait = Signal(range(R + 1))  # ... stationary COMPUTE (weights in the shadows)
        for reg in (stream_wait, drain_wait, preload_wait, swap_wait):
            with m.If(reg != 0):
                m.d.sync += reg.eq(reg - 1)

        # Reads in flight, one entry per issue clock: what to do when they return
        meta_layout = [
            ("a_req", 1), ("b_req", 1), ("kind", 2), ("index", range(n_max)),
            ("len", range(KB + 1)), ("first", 1), ("last", 1),
            ("final", 1), ("serial", 1), ("hold", 1), ("id", 8), ("sync", Sync.BITS),
        ]  # fmt: skip
        meta_width = sum(Signal(shape).shape().width for _, shape in meta_layout)
        m.submodules.reads = reads = SyncFIFO(
            width=meta_width, depth=n_max + cfg.spad_read_delay + 4
        )

        def unpack_meta(word):
            out, offset = {}, 0
            for name, shape in meta_layout:
                width = Signal(shape).shape().width
                out[name] = word[offset : offset + width]
                offset += width
            return out

        # Result tiles in flight: where their rows go
        res_layout = [("rd", 32), ("accumulate", 1), ("rows", range(KB + R + 1)),
                      ("hold", 1), ("id", 8), ("sync", Sync.BITS)]  # fmt: skip
        res_width = sum(Signal(shape).shape().width for _, shape in res_layout)
        # Deep enough for every tile in flight: a command's last row is written about
        # D + R + C + KB clocks after it starts, and streams start R or more apart
        res_depth = 2 + -(-(cfg.spad_read_delay + 6 + R + C + KB) // R)
        m.submodules.results = res_q = SyncFIFO(width=res_width, depth=max(4, res_depth))

        def unpack_res(word):
            out, offset = {}, 0
            for name, shape in res_layout:
                width = Signal(shape).shape().width
                out[name] = word[offset : offset + width]
                offset += width
            return out

        # The issue machine: one read slot per clock while ``left`` != 0
        left = Signal(range(n_max + 1))
        n = Signal(range(n_max + 1))
        kind = Signal(2)
        base_a = Signal(32)
        base_b = Signal(32)
        blk_len = Signal(range(KB + 1))
        blk_first = Signal()
        blk_last = Signal()
        cur_id = Signal(8)
        cur_sync = Signal(Sync.BITS)
        cur_serial = Signal()
        cur_hold = Signal()

        # An untracked command (cmd_sync = 0, a host issuing one command at a time)
        # runs alone: it starts when the Streamer is idle, nothing starts while it is
        # in flight, and it completes once its results are in the Combiner.
        untracked = Signal()
        m.d.comb += untracked.eq(self.cmd_sync == 0)
        serial = Signal()

        direct_pending = Signal()
        direct_id = Signal(8)
        direct_sync = Signal(Sync.BITS)

        # The PRELOAD path (stationary mode): its own issue counter and reads on port B
        p_left = Signal(range(C + 1))
        p_n = Signal(range(C + 1))
        p_base = Signal(32)
        p_rows = Signal(range(R + 1))
        p_id = Signal(8)
        p_sync = Signal(Sync.BITS)
        p_serial = Signal()
        p_layout = [("index", range(C)), ("rows", range(R + 1)), ("final", 1),
                    ("serial", 1), ("id", 8), ("sync", Sync.BITS)]  # fmt: skip
        p_width = sum(Signal(shape).shape().width for _, shape in p_layout)
        m.submodules.preload_reads = p_reads = SyncFIFO(
            width=p_width, depth=cfg.spad_read_delay + 4
        )

        def unpack_preload(word):
            """Split a ``preload_reads`` FIFO word back into its ``p_layout`` fields."""
            out, offset = {}, 0
            for name, shape in p_layout:
                width = Signal(shape).shape().width
                out[name] = word[offset : offset + width]
                offset += width
            return out

        # A PRELOAD's completion, waiting a clock behind any other in the same one
        p_done = Signal()
        p_done_id = Signal(8)
        p_done_sync = Signal(Sync.BITS)
        p_done_serial = Signal()
        swap_pending = Signal()  # a PRELOAD's weights wait for a COMPUTE to swap them in

        quiet = Signal()  # nothing issued and nothing in flight on the read side
        m.d.comb += quiet.eq(
            (left == 0) & ~reads.r_rdy & ~direct_pending & (p_left == 0) & ~p_reads.r_rdy & ~p_done
        )
        l1_busy = a_buf.busy
        for sb in (b_buf, w_buf):
            if sb is not None:
                l1_busy = l1_busy | sb.busy
        for rb in (north, south):
            if rb is not None:
                l1_busy = l1_busy | rb.busy
        idle = Signal()
        m.d.comb += idle.eq(quiet & (stream_wait == 0) & ~res_q.r_rdy & ~l1_busy)

        ok = Signal()
        with m.If(serial | (untracked & ~idle)):
            m.d.comb += ok.eq(0)
        with m.Elif(direct):
            m.d.comb += ok.eq(Mux(is_config, idle, quiet))
        with m.Elif(preload):
            m.d.comb += ok.eq((preload_wait == 0) & (p_left == 0) & ~direct_pending)
        with m.Else():
            m.d.comb += ok.eq(
                (stream_wait == 0)
                & ~direct_pending
                & (~produces | res_q.w_rdy)
                & (~(os_stream & last) | (k_len >= drain_wait))
                & (~st_stream | (swap_wait == 0))
            )
        m.d.comb += self.cmd_ready.eq(ok)
        accept = Signal()
        m.d.comb += accept.eq(self.cmd_valid & ok)

        tile_rows = Signal(range(KB + R + 1))  # rows of the command's result tile
        m.d.comb += tile_rows.eq(Mux(st_stream, k_len, R))

        # The Combiner token is emitted on the result event, so a tracked command
        # with SIGNAL that writes no result would leave its slot OWNED forever
        error = Signal()
        with m.If(accept & Sync.fields(self.cmd_sync).signal & ~produces):
            m.d.sync += error.eq(1)

        with m.If(accept):
            with m.If(direct):
                m.d.sync += [
                    direct_pending.eq(1),
                    direct_id.eq(self.cmd_id),
                    direct_sync.eq(self.cmd_sync),
                ]
                with m.If(is_config):
                    if self.supports_os and self.supports_stationary:
                        m.d.sync += stationary.eq(self.cmd_rs1[0] | self.cmd_rs1[1])
            with m.Elif(preload):
                # The first read is issued now (below); the rest follow, one per clock
                m.d.sync += [
                    p_n.eq(1),
                    p_left.eq(C - 1),
                    p_base.eq(self.cmd_rs1),
                    p_rows.eq(clamp(self.cmd_k_dim, R)),
                    p_id.eq(self.cmd_id),
                    p_sync.eq(self.cmd_sync),
                    p_serial.eq(untracked),
                    serial.eq(untracked),
                    preload_wait.eq(n_max - 1),
                    swap_wait.eq(max(R - 2, 0)),
                    swap_pending.eq(1),
                ]
            with m.Else():
                m.d.sync += [
                    n.eq(0),
                    base_a.eq(self.cmd_rs1),
                    base_b.eq(self.cmd_rs2),
                    blk_len.eq(k_len),
                    # Stationary: ``first`` is the swap marker, on the first COMPUTE
                    # after a PRELOAD
                    blk_first.eq(Mux(st_stream, swap_pending, first)),
                    blk_last.eq(last),
                    cur_id.eq(self.cmd_id),
                    cur_sync.eq(self.cmd_sync),
                    cur_serial.eq(untracked),
                    cur_hold.eq(untracked & produces),
                    serial.eq(untracked),
                ]
                with m.If(os_stream):
                    interval = Mux(k_len > n_max, k_len, n_max)
                    m.d.sync += [kind.eq(KIND_OS), left.eq(n_max), stream_wait.eq(interval - 1)]
                    with m.If(last):
                        m.d.sync += drain_wait.eq(k_len + 2 * R - 2)
                with m.Else():  # stationary
                    interval = Mux(k_len > R, k_len, R)
                    m.d.sync += [
                        kind.eq(KIND_STATIONARY),
                        left.eq(R),
                        stream_wait.eq(interval - 1),
                        swap_pending.eq(0),
                    ]
                with m.If(produces):
                    m.d.comb += [
                        res_q.w_en.eq(1),
                        res_q.w_data.eq(
                            Cat(
                                self.cmd_rd,
                                opcode == ExecOpcode.COMPUTE_ACCUMULATE,
                                tile_rows,
                                untracked,
                                self.cmd_id,
                                self.cmd_sync,
                            )
                        ),
                    ]

        # Issue one read slot
        with m.If(left != 0):
            a_req = Signal()
            b_req = Signal()
            with m.Switch(kind):
                with m.Case(KIND_OS):
                    m.d.comb += [a_req.eq(n < R), b_req.eq(n < C)]
                    m.d.comb += [
                        self.sp_a_read_addr.eq(base_a + n),
                        self.sp_b_read_addr.eq(base_b + n),
                    ]
                with m.Case(KIND_STATIONARY):
                    m.d.comb += [a_req.eq(1), self.sp_a_read_addr.eq(base_a + n)]
            if b_buf is not None:
                m.d.comb += self.sp_b_read_req.eq(b_req)
            m.d.comb += [
                self.sp_a_read_req.eq(a_req),
                reads.w_en.eq(1),
                reads.w_data.eq(
                    Cat(
                        a_req, b_req, kind, n[: Signal(range(n_max)).shape().width],
                        blk_len, blk_first, blk_last, left == 1, cur_serial, cur_hold,
                        cur_id, cur_sync,
                    )
                ),
            ]  # fmt: skip
            # A command accepted in this clock (the last slot of the previous one)
            # restarts the count instead
            with m.If(~(accept & ~direct & ~preload)):
                m.d.sync += [n.eq(n + 1), left.eq(left - 1)]

        # Issue one PRELOAD read (port B): the first in the clock it is accepted
        p_issue = Signal()
        p_index = Signal(range(C + 1))
        p_addr = Signal(32)
        p_issue_rows = Signal(range(R + 1))
        p_issue_id = Signal(8)
        p_issue_sync = Signal(Sync.BITS)
        p_issue_serial = Signal()
        with m.If(accept & preload):
            m.d.comb += [
                p_issue.eq(1),
                p_index.eq(0),
                p_addr.eq(self.cmd_rs1),
                p_issue_rows.eq(clamp(self.cmd_k_dim, R)),
                p_issue_id.eq(self.cmd_id),
                p_issue_sync.eq(self.cmd_sync),
                p_issue_serial.eq(untracked),
            ]
        with m.Elif(p_left != 0):
            m.d.comb += [
                p_issue.eq(1),
                p_index.eq(p_n),
                p_addr.eq(p_base + p_n),
                p_issue_rows.eq(p_rows),
                p_issue_id.eq(p_id),
                p_issue_sync.eq(p_sync),
                p_issue_serial.eq(p_serial),
            ]
            m.d.sync += [p_n.eq(p_n + 1), p_left.eq(p_left - 1)]
        if w_buf is not None:
            with m.If(p_issue):
                m.d.comb += [
                    self.sp_b_read_req.eq(1),
                    self.sp_b_read_addr.eq(p_addr),
                    p_reads.w_en.eq(1),
                    p_reads.w_data.eq(
                        Cat(
                            p_index[: Signal(range(C)).shape().width], p_issue_rows,
                            p_index == C - 1, p_issue_serial, p_issue_id, p_issue_sync,
                        )
                    ),
                ]  # fmt: skip

        # A read returns: load L1
        ret = unpack_meta(reads.r_data)
        returned = Signal()
        # (In the stationary mode port B returns PRELOAD reads, on their own queue)
        m.d.comb += returned.eq(
            reads.r_rdy & (self.sp_a_read_valid | (~stationary & self.sp_b_read_valid))
        )
        m.d.comb += reads.r_en.eq(returned)
        with m.If(returned):  # a stream block: into its L1 buffers
            m.d.comb += [
                a_buf.load_valid.eq(ret["a_req"]),
                a_buf.load_index.eq(ret["index"]),
                a_buf.load_data.eq(self.sp_a_read_data),
                a_buf.load_len.eq(ret["len"]),
                a_buf.load_first.eq(ret["first"]),
                a_buf.load_last.eq(ret["last"] & (ret["kind"] == KIND_OS)),
            ]
            if b_buf is not None:
                m.d.comb += [
                    b_buf.load_valid.eq(ret["b_req"]),
                    b_buf.load_index.eq(ret["index"]),
                    b_buf.load_data.eq(self.sp_b_read_data),
                    b_buf.load_len.eq(ret["len"]),
                ]

        # A PRELOAD read returns: one column of W into its weight buffer, bottom row
        # first (the chain fills from the bottom), rows from ``k_dim`` on zero
        p_ret = unpack_preload(p_reads.r_data)
        p_returned = Signal()
        m.d.comb += p_returned.eq(p_reads.r_rdy & stationary & self.sp_b_read_valid)
        m.d.comb += p_reads.r_en.eq(p_returned)
        if w_buf is not None:
            wb = cfg.weight_bits
            column = Cat(
                *[
                    Mux(
                        R - 1 - e < p_ret["rows"],
                        self.sp_b_read_data[(R - 1 - e) * wb : (R - e) * wb],
                        0,
                    )
                    for e in range(R)
                ]
            )
            m.d.comb += [
                w_buf.load_valid.eq(p_returned),
                w_buf.load_index.eq(p_ret["index"]),
                w_buf.load_data.eq(column),
                w_buf.load_len.eq(R),
            ]
        with m.If(p_returned & p_ret["final"]):
            m.d.sync += [
                p_done.eq(1),
                p_done_id.eq(p_ret["id"]),
                p_done_sync.eq(p_ret["sync"]),
                p_done_serial.eq(p_ret["serial"]),
            ]

        # Completion: a command's last read returned (an untracked command with
        # results completes with them instead), or a direct command
        tile = unpack_res(res_q.r_data)
        tile_done = Signal()  # the last row of the oldest result tile is written
        with m.If(returned & ret["final"]), m.If(~ret["hold"]):
            m.d.comb += [
                self.completed.eq(1),
                self.completed_id.eq(ret["id"]),
                self.completed_sync.eq(ret["sync"]),
            ]
            with m.If(ret["serial"]):
                m.d.sync += serial.eq(0)
        with m.Elif(tile_done & tile["hold"]):
            m.d.comb += [
                self.completed.eq(1),
                self.completed_id.eq(tile["id"]),
                self.completed_sync.eq(tile["sync"]),
            ]
            m.d.sync += serial.eq(0)
        with m.Elif(p_done):
            m.d.comb += [
                self.completed.eq(1),
                self.completed_id.eq(p_done_id),
                self.completed_sync.eq(p_done_sync),
            ]
            with m.If(~(p_returned & p_ret["final"])):
                m.d.sync += p_done.eq(0)
            with m.If(p_done_serial):
                m.d.sync += serial.eq(0)
        with m.Elif(direct_pending):
            m.d.comb += [
                self.completed.eq(1),
                self.completed_id.eq(direct_id),
                self.completed_sync.eq(direct_sync),
            ]
            m.d.sync += direct_pending.eq(0)

        # ---------------------------------------------------------------------
        # Results: L1 result buffers -> Combiner rows
        # ---------------------------------------------------------------------
        row_count = Signal(range(KB + R + 1))
        row_valid = Signal()
        row_data = Signal(cfg.combiner_width)
        if north is not None and south is not None:
            m.d.comb += [
                row_valid.eq(Mux(stationary, south.out_valid, north.out_valid)),
                row_data.eq(Mux(stationary, south.out_data, north.out_data)),
            ]
        else:
            rb = north if north is not None else south
            m.d.comb += [row_valid.eq(rb.out_valid), row_data.eq(rb.out_data)]

        with m.If(row_valid):
            with m.If(res_q.r_rdy):
                m.d.comb += [
                    self.combiner_write_req.eq(1),
                    self.combiner_write_addr.eq(tile["rd"] + row_count),
                    self.combiner_write_data.eq(row_data),
                    self.combiner_accumulate.eq(tile["accumulate"]),
                ]
                with m.If(row_count == tile["rows"] - 1):
                    m.d.sync += row_count.eq(0)
                    m.d.comb += [
                        res_q.r_en.eq(1),
                        tile_done.eq(1),
                        self.result.eq(1),
                        self.result_id.eq(tile["id"]),
                        self.result_sync.eq(tile["sync"]),
                    ]
                with m.Else():
                    m.d.sync += row_count.eq(row_count + 1)
            with m.Else():
                m.d.sync += error.eq(1)

        with m.If(self.array_out_collision):
            m.d.sync += error.eq(1)
        any_error = error | a_buf.error
        for unit in (b_buf, w_buf, north, south):
            if unit is not None:
                any_error = any_error | unit.error
        m.d.comb += [
            self.error.eq(any_error),
            self.busy.eq(~idle),
            self.state_debug.eq(Mux(idle, self.STATE_IDLE, self.STATE_BUSY)),
        ]
        return m
