"""
SystolicTop - Top-level integration of the systolic array accelerator.

This module wires together all major subsystems:
- SystolicArray: Compute fabric for matrix multiply
- Scratchpad: Input/weight storage
- Combiner: Output/partial sum storage
- StreamReader, StreamWriter: the BlockMover's engines, L3 → local memory and back
- BlockMoverIn: the BlockMover's inbound leg, L3 → L2 (or the Combiner), a LOAD
- BlockMoverOut: the BlockMover's outbound leg, the Combiner → L3, a STORE
- Streamer: L2 → L1 → fabric, and the results into the Combiner, an EXECUTE

The control processes keep their commands' types: ``cmd_type`` 0 (LOAD) goes to
BlockMoverIn, 1 (STORE) to BlockMoverOut and 2 (EXECUTE) to the Streamer.

External Interfaces:
- AXI memory interface (read and write channels): the tile's link to L3
- Command interface: commands are pushed into one in-order queue per process
  (BlockMover LOAD, BlockMover STORE, Streamer), and a queue's head starts when its
  credits and tile-ready tokens are there (``cmd_sync``, controller/sync.py)
- Status/interrupt signals, per-process start and completion events, and the tile
  slots' state

Data Flow for Matrix Multiply C = A × B + D:
1. Optionally load D (a bias, or an existing C) from L3 → Combiner via
   BlockMoverIn + StreamReader (a LOAD to a Combiner address)
2. Load A from DRAM → Scratchpad via BlockMoverIn + StreamReader
3. Load B from DRAM → Scratchpad via BlockMoverIn + StreamReader
4. Execute matmul via Streamer + SystolicArray (accumulating onto D)
5. Store C from Combiner → DRAM via BlockMoverOut + StreamWriter

The Combiner's write port is shared: the Streamer writes its results whenever
they arrive, and a LOAD into the Combiner writes in the clocks it leaves free.
"""

from amaranth import Cat, Module, Mux, Signal
from amaranth.lib.fifo import SyncFIFO
from amaranth.lib.wiring import Component, In, Out

from .blockmover.reader import StreamReader
from .blockmover.writer import StreamWriter
from .config import SystolicConfig
from .controller.blockmover_in import BlockMoverIn
from .controller.blockmover_out import BlockMoverOut
from .controller.streamer import Streamer
from .controller.sync import SlotPool, Sync
from .core.systolic_array import SystolicArray
from .errors import (
    CMD_ID,
    CMD_TYPE,
    CMD_VALID,
    KIND,
    KIND_BITS,
    MULTI,
    SOURCE,
    ErrorSource,
    L2Error,
    SlotError,
    SlotPoolId,
    axi_kind,
    sync_kind,
)  # fmt: skip
from .memory.combiner import Combiner
from .memory.epilogue import PROFILES
from .memory.local_addr import LocalAddr
from .memory.scratchpad import make_l2


class SystolicTop(Component):
    """
    Top-level systolic array accelerator.

    Integrates compute fabric, local memories, and BlockMover engines into
    a complete accelerator with external AXI memory interface.

    Ports:
        Command Interface (active controller select + parameters):
            cmd_valid: Command available
            cmd_ready: The target process's queue has room
            cmd_type: 0=load, 1=store, 2=execute
            cmd_opcode: Operation code
            cmd_* : Operation parameters
            cmd_sync: Credits and tile-ready tokens (controller/sync.py; 0 = untracked)

        AXI Read Interface (to external memory):
            axi_ar*: Read address channel
            axi_r*: Read data channel

        AXI Write Interface (to external memory):
            axi_aw*: Write address channel
            axi_w*: Write data channel
            axi_b*: Write response channel

        Status:
            busy: Any process is busy
            bm_in_busy, bm_out_busy, streamer_busy: A process runs a command or has commands queued
            completed: A command completed this cycle (one of possibly several)
            completed_type: Which process completed (its command type: 0=load, 1=store,
                2=exec)
            {bm_in,bm_out,streamer}_completed[_id]: Per-process completions (lossless).
                The Streamer's completion means it is done with L2 (its last block
                is in L1); streamer_result[_id] means its results are in the Combiner
            {load,store,exec}_started[_id]: A process started a command
            l2_free, l2_ready, combiner_free, combiner_ready: Slot states (OWNED = neither)
            sync_error: Sticky credit/token protocol error
            l1_error: Sticky L1/fabric schedule error
            l2_error: Sticky L2 access its macros can't take (memory/scratchpad.py)
            interrupt: Interrupt request

        First-error capture (errors.py, #84):
            cycle: Free-running clock counter (the CSR CYCLE)
            err_info: The first error event since reset or the last err_rearm: its source,
                kind and command, packed as the CSR ERR_INFO (``errors.FirstError``)
            err_cycle: The cycle it was seen (ERR_CYCLE)
            err_rearm: Strobe: forget the capture, so the next error event is recorded
    """

    # Controller type constants
    CTRL_LOAD = 0
    CTRL_STORE = 1
    CTRL_EXEC = 2

    def __init__(self, config: SystolicConfig):
        self.config = config

        buswidth = config.blockmover_buswidth
        strb_width = buswidth // 8

        ports = {
            # Command interface
            "cmd_valid": In(1),
            "cmd_ready": Out(1),
            "cmd_type": In(2),  # 0=load, 1=store, 2=execute
            "cmd_opcode": In(8),
            "cmd_dram_addr": In(64),  # For load/store: DRAM address
            "cmd_local_addr": In(32),  # For load: SP addr; store: CMB addr
            "cmd_len": In(16),  # Load/store: rows to move
            "cmd_stride": In(32),  # Load/store: bytes between L3 rows (0 = one beat)
            "cmd_row_bytes": In(16),  # Load/store: bytes per row (0 = whole row)
            "cmd_id": In(8),
            # Execute-specific command fields
            "cmd_rs1": In(32),  # A address or preload source
            "cmd_rs2": In(32),  # B address
            "cmd_rd": In(32),  # C destination
            "cmd_k_dim": In(16),  # Inner dimension
            # STORE: its activation (bit 3: narrow rows); COMPUTE: its Combiner
            # epilogue code (memory/epilogue.py)
            "cmd_activation": In(4),
            "cmd_sync": In(Sync.BITS),  # Credits and tokens (controller/sync.py); 0 = untracked
            "cmd_chain": In(2),  # Execute: OS reduction chaining (controller/streamer.py)
            # AXI Read Address Channel
            "axi_arvalid": Out(1),
            "axi_arready": In(1),
            "axi_araddr": Out(64),
            "axi_arlen": Out(8),
            "axi_arsize": Out(3),
            "axi_arburst": Out(2),
            # AXI Read Data Channel
            "axi_rvalid": In(1),
            "axi_rready": Out(1),
            "axi_rdata": In(buswidth),
            "axi_rlast": In(1),
            "axi_rresp": In(2),
            # AXI Write Address Channel
            "axi_awvalid": Out(1),
            "axi_awready": In(1),
            "axi_awaddr": Out(64),
            "axi_awlen": Out(8),
            "axi_awsize": Out(3),
            "axi_awburst": Out(2),
            # AXI Write Data Channel
            "axi_wvalid": Out(1),
            "axi_wready": In(1),
            "axi_wdata": Out(buswidth),
            "axi_wstrb": Out(strb_width),
            "axi_wlast": Out(1),
            # AXI Write Response Channel
            "axi_bvalid": In(1),
            "axi_bready": Out(1),
            "axi_bresp": In(2),
            # Status
            "busy": Out(1),
            "bm_in_busy": Out(1),
            "bm_out_busy": Out(1),
            "streamer_busy": Out(1),
            "completed": Out(1),
            "completed_type": Out(2),
            "completed_id": Out(8),
            # Per-controller completions. The controllers run concurrently and
            # can finish in the same cycle; `completed` above reports only one
            # of them (bm_in > bm_out > streamer), these report every completion.
            "bm_in_completed": Out(1),
            "bm_in_completed_id": Out(8),
            "bm_out_completed": Out(1),
            "bm_out_completed_id": Out(8),
            "streamer_completed": Out(1),
            "streamer_completed_id": Out(8),
            # The Streamer's results are in the Combiner (its result token)
            "streamer_result": Out(1),
            "streamer_result_id": Out(8),
            "interrupt": Out(1),
            # A process started a command (its queue's head was dispatched): the
            # CSP action's start, matching {bm_in,bm_out,streamer}_completed
            "bm_in_started": Out(1),
            "bm_in_started_id": Out(8),
            "bm_out_started": Out(1),
            "bm_out_started_id": Out(8),
            "streamer_started": Out(1),
            "streamer_started_id": Out(8),
            # Tile slots (controller/sync.py): FREE / READY bits per slot (OWNED =
            # neither), and a sticky protocol error
            "l2_free": Out(config.l2_slots),
            "l2_ready": Out(config.l2_slots),
            "combiner_free": Out(config.combiner_slots),
            "combiner_ready": Out(config.combiner_slots),
            "sync_error": Out(1),
            # Sticky L1/fabric schedule error (controller/streamer.py)
            "l1_error": Out(1),
            # The Combiner epilogue's configuration (memory/epilogue.py; the tile's
            # profile CSRs: CMB_SCALE, CMB_CONTROL and CMB_CLIP of each, #77)
            **{f"cmb_scale_{n}": In(32, init=1) for n in range(PROFILES)},
            **{f"cmb_control_{n}": In(32) for n in range(PROFILES)},
            **{f"cmb_clip_{n}": In(32) for n in range(PROFILES)},
            # Sticky L2 bank conflict (memory/scratchpad.py)
            "l2_error": Out(1),
            # The first-error capture (errors.py, #84)
            "cycle": Out(32),
            "err_info": Out(32),
            "err_cycle": Out(32),
            "err_rearm": In(1),
        }

        super().__init__(ports)

        # The submodules are built here, not in elaborate(), so a testbench or a trace's
        # probes can reach their signals before a simulation starts
        cfg = config
        self.systolic_array = SystolicArray(cfg)
        self.scratchpad = make_l2(cfg)  # L2
        self.combiner = Combiner(cfg)
        self.stream_reader = StreamReader(cfg)
        self.stream_writer = StreamWriter(cfg)
        self.bm_in = BlockMoverIn(cfg)
        self.bm_out = BlockMoverOut(cfg)
        self.streamer = Streamer(cfg)
        self.l2_slots = SlotPool(cfg.l2_slots)
        self.combiner_slots = SlotPool(cfg.combiner_slots, producers=2)

    def elaborate(self, _platform):
        m = Module()
        cfg = self.config

        total_rows = cfg.grid_rows * cfg.tile_rows
        total_cols = cfg.grid_cols * cfg.tile_cols

        # =====================================================================
        # Instantiate Submodules
        # =====================================================================

        # Compute fabric
        m.submodules.systolic_array = array = self.systolic_array

        # Local memories
        m.submodules.scratchpad = scratchpad = self.scratchpad
        m.submodules.combiner = combiner = self.combiner

        # BlockMover engines
        m.submodules.stream_reader = reader = self.stream_reader
        m.submodules.stream_writer = writer = self.stream_writer

        # Controllers
        m.submodules.bm_in = bm_in = self.bm_in
        m.submodules.bm_out = bm_out = self.bm_out
        m.submodules.streamer = streamer = self.streamer

        # =====================================================================
        # Command Queues, Credits and Tile-Ready Tokens (controller/sync.py)
        # =====================================================================

        # Slot pools: L2 (LOAD fills, PRELOAD/COMPUTE read) and the Combiner
        # (COMPUTE fills, STORE reads)
        m.submodules.l2_slots = l2 = self.l2_slots
        # The Combiner's second producer is the BlockMover's LOAD into it
        m.submodules.combiner_slots = acc = self.combiner_slots

        # One in-order queue per process: the host (or an ISA instruction) pushes
        # commands ahead, and a queue's head is dispatched when its credits and
        # tokens are there and its controller is idle. Each entry maps the
        # controller's command ports to SystolicTop's command fields.
        processes = {
            self.CTRL_LOAD: (
                "bm_in", bm_in, cfg.ld_queue_length,
                {"opcode": "opcode", "dram_addr": "dram_addr", "sp_addr": "local_addr",
                 "len": "len", "stride": "stride", "row_bytes": "row_bytes", "id": "id"},
            ),
            self.CTRL_STORE: (
                "bm_out", bm_out, cfg.st_queue_length,
                {"opcode": "opcode", "combiner_addr": "local_addr", "dram_addr": "dram_addr",
                 "len": "len", "stride": "stride", "row_bytes": "row_bytes", "id": "id",
                 "activation": "activation"},
            ),
            self.CTRL_EXEC: (
                "streamer", streamer, cfg.ex_queue_length,
                {"opcode": "opcode", "rs1": "rs1", "rs2": "rs2", "rd": "rd",
                 "k_dim": "k_dim", "chain": "chain", "sync": "sync", "id": "id",
                 "activation": "activation"},
            ),
        }  # fmt: skip

        queues = {}

        def gate(ctrl_type, ok_for):
            """Queue a process's commands; dispatch the head when ``ok_for(head)`` holds
            and the controller is idle. Returns the dispatch strobe, the head's sync
            fields and the in-flight command's (applied when the controller completes).
            """
            name, ctrl, depth, ports = processes[ctrl_type]
            sources = [getattr(self, f"cmd_{f}") for f in ports.values()] + [self.cmd_sync]
            queue = SyncFIFO(width=sum(len(v) for v in sources), depth=depth)
            m.submodules[f"{name}_queue"] = queues[ctrl_type] = queue
            m.d.comb += [
                queue.w_data.eq(Cat(*sources)),
                queue.w_en.eq(self.cmd_valid & (self.cmd_type == ctrl_type)),
            ]
            offset = 0
            for port, src in zip(ports, sources[:-1], strict=True):
                m.d.comb += getattr(ctrl, f"cmd_{port}").eq(queue.r_data[offset:][: len(src)])
                offset += len(src)
            head = Signal(Sync.BITS, name=f"{name}_head_sync")
            inflight = Signal(Sync.BITS, name=f"{name}_inflight_sync")
            dispatch = Signal(name=f"{name}_dispatch")
            m.d.comb += head.eq(queue.r_data[offset:][: Sync.BITS])
            ok = ok_for(Sync.fields(head))
            m.d.comb += [
                ctrl.cmd_valid.eq(queue.r_rdy & ok),
                dispatch.eq(queue.r_rdy & ok & ctrl.cmd_ready),
                queue.r_en.eq(dispatch),
            ]
            with m.If(dispatch):
                m.d.sync += inflight.eq(head)
            m.d.comb += [
                getattr(self, f"{name}_started").eq(dispatch),
                getattr(self, f"{name}_started_id").eq(ctrl.cmd_id),
            ]
            return dispatch, Sync.fields(head), Sync.fields(inflight)

        # BlockMover, L3 -> L2 (or the Combiner, for a LOAD to a Combiner
        # address): takes the destination slot's credit; emits its token when done
        ld_to_combiner = Signal()  # the head LOAD fills a Combiner slot
        ld_combiner_inflight = Signal()  # the running one does
        m.d.comb += ld_to_combiner.eq(LocalAddr.is_combiner(bm_in.cmd_sp_addr))
        ld_go, ld_head, ld_done = gate(
            self.CTRL_LOAD,
            lambda h: ~h.acquire | Mux(ld_to_combiner, acc.fill_credit, l2.prod_credit),
        )
        with m.If(ld_go):
            m.d.sync += ld_combiner_inflight.eq(ld_to_combiner)
        ld_signal = bm_in.completed & ld_done.signal
        m.d.comb += [
            l2.prod_slot.eq(ld_head.dst),
            l2.prod_acquire.eq(ld_head.acquire & ~ld_to_combiner),
            l2.prod_take.eq(ld_go & ld_head.acquire & ~ld_to_combiner),
            l2.prod_signal.eq(ld_signal & ~ld_combiner_inflight),
            l2.prod_signal_slot.eq(ld_done.dst),
            acc.fill_slot.eq(ld_head.dst),
            acc.fill_acquire.eq(ld_head.acquire & ld_to_combiner),
            acc.fill_take.eq(ld_go & ld_head.acquire & ld_to_combiner),
            acc.fill_signal.eq(ld_signal & ld_combiner_inflight),
            acc.fill_signal_slot.eq(ld_done.dst),
        ]

        # Streamer: starts when its L2 tile is ready, its Combiner slot is free
        # and the fabric can take it on schedule (and then runs to the end). Several
        # of its commands are in flight at once, so it carries each command's sync
        # fields to its own events: it returns the L2 credit once the tile's last
        # block is in L1, and emits the result token once the results are in the
        # Combiner.
        ex_go, ex_head, _ = gate(
            self.CTRL_EXEC,
            lambda h: (~h.wait | l2.cons_ready) & (~h.acquire | acc.prod_credit),
        )
        ex_released = Sync.fields(streamer.completed_sync)
        ex_resulted = Sync.fields(streamer.result_sync)
        m.d.comb += [
            l2.cons_slot.eq(ex_head.src),
            l2.cons_wait.eq(ex_head.wait),
            l2.cons_take.eq(ex_go & ex_head.wait & ex_head.release),
            l2.cons_release.eq(streamer.completed & ex_released.release),
            l2.cons_release_slot.eq(ex_released.src),
            acc.prod_slot.eq(ex_head.dst),
            acc.prod_acquire.eq(ex_head.acquire),
            acc.prod_take.eq(ex_go & ex_head.acquire),
            acc.prod_signal.eq(streamer.result & ex_resulted.signal),
            acc.prod_signal_slot.eq(ex_resulted.dst),
        ]

        # BlockMover, Combiner -> L3: fires on the result token; returns the
        # Combiner credit when done
        st_go, st_head, st_done = gate(self.CTRL_STORE, lambda h: ~h.wait | acc.cons_ready)
        m.d.comb += [
            acc.cons_slot.eq(st_head.src),
            acc.cons_wait.eq(st_head.wait),
            acc.cons_take.eq(st_go & st_head.wait & st_head.release),
            acc.cons_release.eq(bm_out.completed & st_done.release),
            acc.cons_release_slot.eq(st_done.src),
        ]

        # Command ready: the target queue has room
        with m.Switch(self.cmd_type):
            for ctrl_type, queue in queues.items():
                with m.Case(ctrl_type):
                    m.d.comb += self.cmd_ready.eq(queue.w_rdy)
            with m.Default():
                m.d.comb += self.cmd_ready.eq(0)

        m.d.comb += [
            self.l2_free.eq(l2.free),
            self.l2_ready.eq(l2.ready),
            self.combiner_free.eq(acc.free),
            self.combiner_ready.eq(acc.ready),
            self.sync_error.eq(l2.error | acc.error),
        ]

        # =====================================================================
        # BlockMoverIn <-> StreamReader <-> Scratchpad
        # =====================================================================

        # BlockMoverIn -> StreamReader
        m.d.comb += [
            reader.req_valid.eq(bm_in.engine_req_valid),
            bm_in.engine_req_ready.eq(reader.req_ready),
            reader.req_addr.eq(bm_in.engine_req_addr),
            reader.req_len.eq(bm_in.engine_req_len),
        ]

        # StreamReader -> BlockMoverIn
        m.d.comb += [
            bm_in.engine_resp_valid.eq(reader.resp_valid),
            reader.resp_ready.eq(bm_in.engine_resp_ready),
            bm_in.engine_resp_data.eq(reader.resp_data),
            bm_in.engine_resp_last.eq(reader.resp_last),
        ]

        # BlockMoverIn -> Scratchpad (write)
        m.d.comb += [
            scratchpad.write_req.eq(bm_in.sp_write_en),
            scratchpad.write_addr.eq(bm_in.sp_write_addr),
            scratchpad.write_data.eq(bm_in.sp_write_data),
            scratchpad.write_mask.eq(bm_in.sp_write_mask),
        ]

        # =====================================================================
        # BlockMoverOut <-> StreamWriter <-> Combiner
        # =====================================================================

        # BlockMoverOut -> StreamWriter
        m.d.comb += [
            writer.req_valid.eq(bm_out.engine_req_valid),
            bm_out.engine_req_ready.eq(writer.req_ready),
            writer.req_addr.eq(bm_out.engine_req_addr),
            writer.req_len.eq(bm_out.engine_req_len),
            writer.data_valid.eq(bm_out.engine_data_valid),
            bm_out.engine_data_ready.eq(writer.data_ready),
            writer.data.eq(bm_out.engine_data),
            writer.data_strb.eq(bm_out.engine_data_strb),
            writer.data_last.eq(bm_out.engine_data_last),
            bm_out.engine_done.eq(writer.done),
        ]

        # Combiner -> BlockMoverOut (read)
        m.d.comb += [
            combiner.read_req.eq(bm_out.combiner_read_req),
            combiner.read_addr.eq(bm_out.combiner_read_addr),
            bm_out.combiner_read_data.eq(combiner.read_data),
            bm_out.combiner_read_valid.eq(combiner.read_valid),
            # The merge has the read port first: STORE waits for read_ready
            bm_out.combiner_read_ready.eq(combiner.read_ready),
            *[
                getattr(combiner, f"{k}_{n}").eq(getattr(self, f"cmb_{k}_{n}"))
                for n in range(PROFILES)
                for k in ("scale", "control", "clip")
            ],  # fmt: skip
            # The activation is the BlockMoverOut's, per element
        ]

        # =====================================================================
        # Streamer <-> Scratchpad <-> SystolicArray <-> Combiner
        # =====================================================================

        # Streamer reads A and B on L2's two read ports (A and B bank groups)
        # (no other block reads the scratchpad)
        m.d.comb += [
            scratchpad.read_req.eq(streamer.sp_a_read_req),
            scratchpad.read_addr.eq(streamer.sp_a_read_addr),
            streamer.sp_a_read_data.eq(scratchpad.read_data),
            streamer.sp_a_read_valid.eq(scratchpad.read_valid),
            scratchpad.read_b_req.eq(streamer.sp_b_read_req),
            scratchpad.read_b_addr.eq(streamer.sp_b_read_addr),
            streamer.sp_b_read_data.eq(scratchpad.read_b_data),
            streamer.sp_b_read_valid.eq(scratchpad.read_b_valid),
        ]

        # The Combiner's write port: the Streamer's result rows, which
        # never wait, and in the clocks they leave free the BlockMoverIn's rows
        # (a LOAD to a Combiner address). The BlockMoverOut only reads it.
        m.d.comb += bm_in.combiner_write_ready.eq(~streamer.combiner_write_req)
        with m.If(streamer.combiner_write_req):
            m.d.comb += [
                combiner.write_req.eq(1),
                combiner.write_addr.eq(streamer.combiner_write_addr),
                combiner.write_data.eq(streamer.combiner_write_data),
                combiner.accumulate.eq(streamer.combiner_accumulate),
                combiner.write_epilogue.eq(streamer.combiner_write_epilogue),
            ]
        with m.Else():
            m.d.comb += [
                combiner.write_req.eq(bm_in.combiner_write_en),
                combiner.write_addr.eq(bm_in.combiner_write_addr),
                combiner.write_data.eq(bm_in.combiner_write_data),
                combiner.accumulate.eq(bm_in.combiner_write_accumulate),
            ]

        # =====================================================================
        # Streamer <-> SystolicArray
        # =====================================================================

        m.d.comb += [
            array.in_control_dataflow.eq(streamer.array_in_dataflow),
            streamer.array_out_collision.eq(array.out_collision),
        ]
        for i in range(total_rows):
            for name in ("a", "valid", "first", "last"):
                m.d.comb += getattr(array, f"in_{name}_{i}").eq(
                    getattr(streamer, f"array_in_{name}_{i}")
                )
        for j in range(total_cols):
            m.d.comb += [
                getattr(array, f"in_b_{j}").eq(getattr(streamer, f"array_in_b_{j}")),
                getattr(array, f"in_w_{j}").eq(getattr(streamer, f"array_in_w_{j}")),
                getattr(array, f"in_w_valid_{j}").eq(getattr(streamer, f"array_in_w_valid_{j}")),
            ]
            for name in ("c", "c_valid", "psum", "psum_valid"):
                m.d.comb += getattr(streamer, f"array_out_{name}_{j}").eq(
                    getattr(array, f"out_{name}_{j}")
                )

        # =====================================================================
        # AXI Interface Wiring
        # =====================================================================

        # StreamReader -> AXI Read
        m.d.comb += [
            self.axi_arvalid.eq(reader.mem_arvalid),
            reader.mem_arready.eq(self.axi_arready),
            self.axi_araddr.eq(reader.mem_araddr),
            self.axi_arlen.eq(reader.mem_arlen),
            self.axi_arsize.eq(reader.mem_arsize),
            self.axi_arburst.eq(reader.mem_arburst),
            reader.mem_rvalid.eq(self.axi_rvalid),
            self.axi_rready.eq(reader.mem_rready),
            reader.mem_rdata.eq(self.axi_rdata),
            reader.mem_rlast.eq(self.axi_rlast),
            reader.mem_rresp.eq(self.axi_rresp),
        ]

        # StreamWriter -> AXI Write
        m.d.comb += [
            self.axi_awvalid.eq(writer.mem_awvalid),
            writer.mem_awready.eq(self.axi_awready),
            self.axi_awaddr.eq(writer.mem_awaddr),
            self.axi_awlen.eq(writer.mem_awlen),
            self.axi_awsize.eq(writer.mem_awsize),
            self.axi_awburst.eq(writer.mem_awburst),
            self.axi_wvalid.eq(writer.mem_wvalid),
            writer.mem_wready.eq(self.axi_wready),
            self.axi_wdata.eq(writer.mem_wdata),
            self.axi_wstrb.eq(writer.mem_wstrb),
            self.axi_wlast.eq(writer.mem_wlast),
            writer.mem_bvalid.eq(self.axi_bvalid),
            self.axi_bready.eq(writer.mem_bready),
            writer.mem_bresp.eq(self.axi_bresp),
        ]

        # =====================================================================
        # Status Outputs
        # =====================================================================

        m.d.comb += [
            # A process is busy while it runs a command or has commands queued
            self.bm_in_busy.eq(bm_in.busy | (queues[self.CTRL_LOAD].w_level != 0)),
            self.bm_out_busy.eq(bm_out.busy | (queues[self.CTRL_STORE].w_level != 0)),
            self.streamer_busy.eq(streamer.busy | (queues[self.CTRL_EXEC].w_level != 0)),
            # The Combiner's pipeline too: a finished row commits after the Streamer is
            # done with it, and the epilogue's settings must not change under it
            self.busy.eq(self.bm_in_busy | self.bm_out_busy | self.streamer_busy | combiner.busy),
        ]

        # Per-controller completions (lossless)
        m.d.comb += [
            self.bm_in_completed.eq(bm_in.completed),
            self.bm_in_completed_id.eq(bm_in.completed_id),
            self.bm_out_completed.eq(bm_out.completed),
            self.bm_out_completed_id.eq(bm_out.completed_id),
            self.streamer_completed.eq(streamer.completed),
            self.streamer_completed_id.eq(streamer.completed_id),
            self.streamer_result.eq(streamer.result),
            self.streamer_result_id.eq(streamer.result_id),
            self.l1_error.eq(streamer.error),
            self.l2_error.eq(scratchpad.error),
        ]

        # Summary completion (priority: load > store > exec). Lossy when two
        # controllers complete in the same cycle; use the per-controller ports.
        with m.If(bm_in.completed):
            m.d.comb += [
                self.completed.eq(1),
                self.completed_type.eq(self.CTRL_LOAD),
                self.completed_id.eq(bm_in.completed_id),
            ]
        with m.Elif(bm_out.completed):
            m.d.comb += [
                self.completed.eq(1),
                self.completed_type.eq(self.CTRL_STORE),
                self.completed_id.eq(bm_out.completed_id),
            ]
        with m.Elif(streamer.completed):
            m.d.comb += [
                self.completed.eq(1),
                self.completed_type.eq(self.CTRL_EXEC),
                self.completed_id.eq(streamer.completed_id),
            ]
        with m.Else():
            m.d.comb += [
                self.completed.eq(0),
                self.completed_type.eq(0),
                self.completed_id.eq(0),
            ]

        # Interrupt (could be extended)
        m.d.comb += self.interrupt.eq(0)

        self._first_error(m, scratchpad, streamer, l2, acc)

        return m

    def _first_error(self, m, scratchpad, streamer, l2_pool, combiner_pool):
        """Latch the first error event: a sticky bit rising (L1, L2, SYNC) or an AXI error
        response, with its source, kind and the command responsible (errors.py)."""
        cycle = Signal(32)
        m.d.sync += cycle.eq(cycle + 1)
        m.d.comb += self.cycle.eq(cycle)

        # Per process: the command it started last (valid once it has started one), and
        # the command at its queue's head (``*_started_id`` carries the head's ID always)
        last, started, head = {}, {}, {}
        for name, ctrl_type in (("bm_in", self.CTRL_LOAD), ("bm_out", self.CTRL_STORE),
                                ("streamer", self.CTRL_EXEC)):  # fmt: skip
            last_id = Signal(8, name=f"{name}_last_id")
            has_started = Signal(name=f"{name}_has_started")
            with m.If(getattr(self, f"{name}_started")):
                m.d.sync += [last_id.eq(getattr(self, f"{name}_started_id")), has_started.eq(1)]
            last[ctrl_type], started[ctrl_type] = last_id, has_started
            head[ctrl_type] = getattr(self, f"{name}_started_id")

        # The command each completion event belongs to, as it fired: a pool's signal or
        # release at a command's completion names that command, though a later one may
        # have started since (or wait at the queue head)
        def last_of(pulse, ident, name):
            held = Signal(8, name=f"{name}_done_id")
            with m.If(pulse):
                m.d.sync += held.eq(ident)
            return held

        done = {
            "bm_in": last_of(self.bm_in_completed, self.bm_in_completed_id, "bm_in"),
            "bm_out": last_of(self.bm_out_completed, self.bm_out_completed_id, "bm_out"),
            "streamer": last_of(
                self.streamer_completed, self.streamer_completed_id, "streamer"
            ),  # fmt: skip
            "streamer_result": last_of(
                self.streamer_result, self.streamer_result_id, "streamer_result"
            ),  # fmt: skip
        }

        def risen(bit, name):
            prev = Signal(name=f"{name}_prev")
            m.d.sync += prev.eq(bit)
            return bit & ~prev

        def any_of(value, kinds):
            return Cat(*[value == k for k in kinds]).any()

        # Which process a kind names. SYNC: the port at fault, per pool. A slot number
        # past the pool is the queue head's (a command that never starts); every other
        # kind is the command the process started last
        S = SlotError
        prod = {S.TAKE_NOT_FREE, S.SIGNAL_NOT_OWNED, S.PROD_NO_SUCH_SLOT}
        fill = {S.FILL_TAKE_NOT_FREE, S.FILL_SIGNAL_NOT_OWNED, S.FILL_NO_SUCH_SLOT}
        cons = {S.WAIT_NOT_READY, S.RELEASE_NOT_HELD, S.CONS_NO_SUCH_SLOT, S.TAKE_AND_RELEASE}
        queued = {S.PROD_NO_SUCH_SLOT, S.FILL_NO_SUCH_SLOT, S.CONS_NO_SUCH_SLOT}
        pools = (
            (l2_pool, SlotPoolId.L2, {self.CTRL_LOAD: prod, self.CTRL_EXEC: cons}),
            (combiner_pool, SlotPoolId.COMBINER,
             {self.CTRL_EXEC: prod, self.CTRL_LOAD: fill, self.CTRL_STORE: cons}),
        )  # fmt: skip
        # Per pool and process: the completion event that drives its signal or release
        # (L2: a LOAD signals on its completion, the Streamer releases on its own; the
        # Combiner: the Streamer signals on its result, a LOAD fills and a STORE releases
        # on their completions)
        completion_ids = {
            SlotPoolId.L2: {self.CTRL_LOAD: done["bm_in"], self.CTRL_EXEC: done["streamer"]},
            SlotPoolId.COMBINER: {self.CTRL_EXEC: done["streamer_result"],
                                  self.CTRL_LOAD: done["bm_in"], self.CTRL_STORE: done["bm_out"]},
        }  # fmt: skip
        l2_writes = {L2Error.WRITE_PAST_L2, L2Error.WRITE_BANK_BUSY}  # the LOAD's writes

        # Events, in priority order for a clock with several: (fired, source, kind, names
        # a process, the process, its queue head's command, a completion's command and
        # its ID)
        events = []
        l2_kind = scratchpad.error_kind
        l2_proc = Mux(any_of(l2_kind, l2_writes), self.CTRL_LOAD, self.CTRL_EXEC)
        events.append((risen(scratchpad.error, "l2_error"), ErrorSource.L2, l2_kind, 1, l2_proc,
                       0, 0, 0))  # fmt: skip
        events.append((risen(streamer.error, "l1_error"), ErrorSource.L1,
                       streamer.error_kind, 1, self.CTRL_EXEC, 0, 0, 0))  # fmt: skip
        for pool, pool_id, owners in pools:
            kind = pool.error_kind
            tag = pool_id.name.lower()
            proc = Signal(2, name=f"{tag}_slot_err_proc")
            named = Signal(name=f"{tag}_slot_err_named")
            done_id = Signal(8, name=f"{tag}_slot_err_done_id")
            for ctrl_type, kinds in owners.items():
                with m.If(any_of(kind, kinds)):
                    m.d.comb += [proc.eq(ctrl_type), named.eq(1),
                                 done_id.eq(completion_ids[pool_id][ctrl_type])]  # fmt: skip
            at_done = pool.error_at_completion
            # A slot past the pool names the queue head only when a start or a head check
            # raised it; at a completion it is the completing command's
            events.append((risen(pool.error, f"{tag}_slot_error"), ErrorSource.SYNC,
                           sync_kind(pool_id, 0) | kind, named, proc,
                           any_of(kind, queued) & ~at_done, at_done, done_id))  # fmt: skip
        for write, valid, ready, resp, ctrl_type in (
            (False, self.axi_rvalid, self.axi_rready, self.axi_rresp, self.CTRL_LOAD),
            (True, self.axi_bvalid, self.axi_bready, self.axi_bresp, self.CTRL_STORE),
        ):
            events.append((valid & ready & (resp != 0), ErrorSource.AXI,
                           axi_kind(write, 0) | resp << 2, 1, ctrl_type, 0, 0, 0))  # fmt: skip

        # The highest-priority event of the clock, and whether there was another
        any_event = Signal()
        source = Signal(3)
        kind = Signal(KIND_BITS)
        named = Signal()
        proc = Signal(2)
        from_head = Signal()
        from_done = Signal()
        done_id = Signal(8)
        for fired, src, k, nm, p, hd, dn, dn_id in reversed(events):
            with m.If(fired):
                m.d.comb += [any_event.eq(1), source.eq(src), kind.eq(k), named.eq(nm),
                             proc.eq(p), from_head.eq(hd), from_done.eq(dn),
                             done_id.eq(dn_id)]  # fmt: skip
        multi = Signal()
        m.d.comb += multi.eq(sum(e[0].as_unsigned() for e in events) > 1)
        cmd_id = Signal(8)
        cmd_valid = Signal()
        with m.Switch(proc):
            for ctrl_type in last:
                with m.Case(ctrl_type):
                    m.d.comb += [
                        cmd_id.eq(Mux(from_done, done_id,
                                      Mux(from_head, head[ctrl_type], last[ctrl_type]))),
                        # A process that has started nothing has no last command to name
                        cmd_valid.eq(named & (from_done | from_head | started[ctrl_type])),
                    ]  # fmt: skip

        valid = Signal()
        info = Signal(32)
        at = Signal(32)
        with m.If(any_event & (~valid | self.err_rearm)):
            m.d.sync += [
                valid.eq(1),
                info.eq(1 | source << SOURCE[0] | kind << KIND[0]
                        | Mux(cmd_valid, 1 << CMD_VALID[0] | proc << CMD_TYPE[0]
                              | cmd_id << CMD_ID[0], 0)
                        | multi << MULTI[0]),
                at.eq(cycle),
            ]  # fmt: skip
        with m.Elif(self.err_rearm):
            m.d.sync += [valid.eq(0), info.eq(0), at.eq(0)]
        m.d.comb += [self.err_info.eq(info), self.err_cycle.eq(at)]
