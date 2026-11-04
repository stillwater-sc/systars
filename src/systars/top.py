"""
SystolicTop - Top-level integration of the systolic array accelerator.

This module wires together all major subsystems:
- SystolicArray: Compute fabric for matrix multiply
- Scratchpad: Input/weight storage
- Accumulator: Output/partial sum storage
- StreamReader: DMA for DRAM → local memory
- StreamWriter: DMA for local memory → DRAM
- LoadController: Orchestrates DRAM → Scratchpad
- StoreController: Orchestrates Accumulator → DRAM
- ExecuteController: Orchestrates Scratchpad → SystolicArray → Accumulator

External Interfaces:
- AXI memory interface (read and write channels): the tile's link to L3
- Command interface: commands are pushed into one in-order queue per process
  (BlockMover LOAD, BlockMover STORE, Streamer), and a queue's head starts when its
  credits and tile-ready tokens are there (``cmd_sync``, controller/sync.py)
- Status/interrupt signals, per-process start and completion events, and the tile
  slots' state

Data Flow for Matrix Multiply C = A × B + D:
1. Load A from DRAM → Scratchpad via LoadController + StreamReader
2. Load B from DRAM → Scratchpad via LoadController + StreamReader
3. Optionally preload D (bias) from Accumulator
4. Execute matmul via ExecuteController + SystolicArray
5. Store C from Accumulator → DRAM via StoreController + StreamWriter
"""

from amaranth import Cat, Module, Signal
from amaranth.lib.fifo import SyncFIFO
from amaranth.lib.wiring import Component, In, Out

from .config import SystolicConfig
from .controller.execute import ExecuteController
from .controller.load import LoadController
from .controller.store import StoreController
from .controller.sync import SlotPool, Sync
from .core.systolic_array import SystolicArray
from .dma.reader import StreamReader
from .dma.writer import StreamWriter
from .memory.accumulator import Accumulator
from .memory.scratchpad import Scratchpad


class SystolicTop(Component):
    """
    Top-level systolic array accelerator.

    Integrates compute fabric, local memories, and DMA engines into
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
            load_busy, store_busy, exec_busy: A process runs a command or has commands queued
            completed: A command completed this cycle (one of possibly several)
            completed_type: Which controller completed (0=load, 1=store, 2=exec)
            {load,store,exec}_completed[_id]: Per-controller completions (lossless).
                The Streamer's completion means it is done with L2 (its last block
                is in L1); exec_result[_id] means its results are in the accumulator
            {load,store,exec}_started[_id]: A process started a command
            l2_free, l2_ready, acc_free, acc_ready: Slot states (OWNED = neither)
            sync_error: Sticky credit/token protocol error
            l1_error: Sticky L1/fabric schedule error
            interrupt: Interrupt request
    """

    # Controller type constants
    CTRL_LOAD = 0
    CTRL_STORE = 1
    CTRL_EXEC = 2

    def __init__(self, config: SystolicConfig):
        self.config = config

        buswidth = config.dma_buswidth
        strb_width = buswidth // 8

        ports = {
            # Command interface
            "cmd_valid": In(1),
            "cmd_ready": Out(1),
            "cmd_type": In(2),  # 0=load, 1=store, 2=execute
            "cmd_opcode": In(8),
            "cmd_dram_addr": In(64),  # For load/store: DRAM address
            "cmd_local_addr": In(32),  # For load: SP addr; store: ACC addr
            "cmd_len": In(16),  # Load/store: rows to move
            "cmd_stride": In(32),  # Load/store: bytes between L3 rows (0 = one beat)
            "cmd_row_bytes": In(16),  # Load/store: bytes per row (0 = whole row)
            "cmd_id": In(8),
            # Execute-specific command fields
            "cmd_rs1": In(32),  # A address or preload source
            "cmd_rs2": In(32),  # B address
            "cmd_rd": In(32),  # C destination
            "cmd_k_dim": In(16),  # Inner dimension
            "cmd_activation": In(4),  # Activation function (for store)
            "cmd_sync": In(Sync.BITS),  # Credits and tokens (controller/sync.py); 0 = untracked
            "cmd_chain": In(2),  # Execute: OS reduction chaining (controller/execute.py)
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
            "load_busy": Out(1),
            "store_busy": Out(1),
            "exec_busy": Out(1),
            "completed": Out(1),
            "completed_type": Out(2),
            "completed_id": Out(8),
            # Per-controller completions. The controllers run concurrently and
            # can finish in the same cycle; `completed` above reports only one
            # of them (load > store > exec), these report every completion.
            "load_completed": Out(1),
            "load_completed_id": Out(8),
            "store_completed": Out(1),
            "store_completed_id": Out(8),
            "exec_completed": Out(1),
            "exec_completed_id": Out(8),
            # The Streamer's results are in the accumulator (its result token)
            "exec_result": Out(1),
            "exec_result_id": Out(8),
            "interrupt": Out(1),
            # A process started a command (its queue's head was dispatched): the
            # CSP action's start, matching {load,store,exec}_completed
            "load_started": Out(1),
            "load_started_id": Out(8),
            "store_started": Out(1),
            "store_started_id": Out(8),
            "exec_started": Out(1),
            "exec_started_id": Out(8),
            # Tile slots (controller/sync.py): FREE / READY bits per slot (OWNED =
            # neither), and a sticky protocol error
            "l2_free": Out(config.l2_slots),
            "l2_ready": Out(config.l2_slots),
            "acc_free": Out(config.acc_slots),
            "acc_ready": Out(config.acc_slots),
            "sync_error": Out(1),
            # Sticky L1/fabric schedule error (controller/execute.py)
            "l1_error": Out(1),
        }

        super().__init__(ports)

    def elaborate(self, _platform):
        m = Module()
        cfg = self.config

        total_rows = cfg.grid_rows * cfg.tile_rows
        total_cols = cfg.grid_cols * cfg.tile_cols

        # =====================================================================
        # Instantiate Submodules
        # =====================================================================

        # Compute fabric
        m.submodules.systolic_array = array = SystolicArray(cfg)

        # Local memories
        m.submodules.scratchpad = scratchpad = Scratchpad(cfg)
        m.submodules.accumulator = accumulator = Accumulator(cfg)

        # DMA engines
        m.submodules.stream_reader = reader = StreamReader(cfg)
        m.submodules.stream_writer = writer = StreamWriter(cfg)

        # Controllers
        m.submodules.load_ctrl = load_ctrl = LoadController(cfg)
        m.submodules.store_ctrl = store_ctrl = StoreController(cfg)
        m.submodules.exec_ctrl = exec_ctrl = ExecuteController(cfg)

        # =====================================================================
        # Command Queues, Credits and Tile-Ready Tokens (controller/sync.py)
        # =====================================================================

        # Slot pools: L2 (LOAD fills, PRELOAD/COMPUTE read) and the accumulator
        # (COMPUTE fills, STORE reads)
        m.submodules.l2_slots = l2 = SlotPool(cfg.l2_slots)
        m.submodules.acc_slots = acc = SlotPool(cfg.acc_slots)

        # One in-order queue per process: the host (or an ISA instruction) pushes
        # commands ahead, and a queue's head is dispatched when its credits and
        # tokens are there and its controller is idle. Each entry maps the
        # controller's command ports to SystolicTop's command fields.
        processes = {
            self.CTRL_LOAD: (
                "load", load_ctrl, cfg.ld_queue_length,
                {"opcode": "opcode", "dram_addr": "dram_addr", "sp_addr": "local_addr",
                 "len": "len", "stride": "stride", "row_bytes": "row_bytes", "id": "id"},
            ),
            self.CTRL_STORE: (
                "store", store_ctrl, cfg.st_queue_length,
                {"opcode": "opcode", "acc_addr": "local_addr", "dram_addr": "dram_addr",
                 "len": "len", "stride": "stride", "row_bytes": "row_bytes", "id": "id",
                 "activation": "activation"},
            ),
            self.CTRL_EXEC: (
                "exec", exec_ctrl, cfg.ex_queue_length,
                {"opcode": "opcode", "rs1": "rs1", "rs2": "rs2", "rd": "rd",
                 "k_dim": "k_dim", "chain": "chain", "sync": "sync", "id": "id"},
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

        # BlockMover, L3 -> L2: takes the L2 slot's credit; emits its token when done
        ld_go, ld_head, ld_done = gate(self.CTRL_LOAD, lambda h: ~h.acquire | l2.prod_credit)
        m.d.comb += [
            l2.prod_slot.eq(ld_head.dst),
            l2.prod_acquire.eq(ld_head.acquire),
            l2.prod_take.eq(ld_go & ld_head.acquire),
            l2.prod_signal.eq(load_ctrl.completed & ld_done.signal),
            l2.prod_signal_slot.eq(ld_done.dst),
        ]

        # Streamer: starts when its L2 tile is ready, its accumulator slot is free
        # and the fabric can take it on schedule (and then runs to the end). Several
        # of its commands are in flight at once, so it carries each command's sync
        # fields to its own events: it returns the L2 credit once the tile's last
        # block is in L1, and emits the result token once the results are in the
        # accumulator.
        ex_go, ex_head, _ = gate(
            self.CTRL_EXEC,
            lambda h: (~h.wait | l2.cons_ready) & (~h.acquire | acc.prod_credit),
        )
        ex_released = Sync.fields(exec_ctrl.completed_sync)
        ex_resulted = Sync.fields(exec_ctrl.result_sync)
        m.d.comb += [
            l2.cons_slot.eq(ex_head.src),
            l2.cons_wait.eq(ex_head.wait),
            l2.cons_take.eq(ex_go & ex_head.wait & ex_head.release),
            l2.cons_release.eq(exec_ctrl.completed & ex_released.release),
            l2.cons_release_slot.eq(ex_released.src),
            acc.prod_slot.eq(ex_head.dst),
            acc.prod_acquire.eq(ex_head.acquire),
            acc.prod_take.eq(ex_go & ex_head.acquire),
            acc.prod_signal.eq(exec_ctrl.result & ex_resulted.signal),
            acc.prod_signal_slot.eq(ex_resulted.dst),
        ]

        # BlockMover, accumulator -> L3: fires on the result token; returns the
        # accumulator credit when done
        st_go, st_head, st_done = gate(self.CTRL_STORE, lambda h: ~h.wait | acc.cons_ready)
        m.d.comb += [
            acc.cons_slot.eq(st_head.src),
            acc.cons_wait.eq(st_head.wait),
            acc.cons_take.eq(st_go & st_head.wait & st_head.release),
            acc.cons_release.eq(store_ctrl.completed & st_done.release),
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
            self.acc_free.eq(acc.free),
            self.acc_ready.eq(acc.ready),
            self.sync_error.eq(l2.error | acc.error),
        ]

        # =====================================================================
        # LoadController <-> StreamReader <-> Scratchpad
        # =====================================================================

        # LoadController -> StreamReader
        m.d.comb += [
            reader.req_valid.eq(load_ctrl.dma_req_valid),
            load_ctrl.dma_req_ready.eq(reader.req_ready),
            reader.req_addr.eq(load_ctrl.dma_req_addr),
            reader.req_len.eq(load_ctrl.dma_req_len),
        ]

        # StreamReader -> LoadController
        m.d.comb += [
            load_ctrl.dma_resp_valid.eq(reader.resp_valid),
            reader.resp_ready.eq(load_ctrl.dma_resp_ready),
            load_ctrl.dma_resp_data.eq(reader.resp_data),
            load_ctrl.dma_resp_last.eq(reader.resp_last),
        ]

        # LoadController -> Scratchpad (write)
        m.d.comb += [
            scratchpad.write_req.eq(load_ctrl.sp_write_en),
            scratchpad.write_addr.eq(load_ctrl.sp_write_addr),
            scratchpad.write_data.eq(load_ctrl.sp_write_data),
            scratchpad.write_mask.eq(load_ctrl.sp_write_mask),
        ]

        # =====================================================================
        # StoreController <-> StreamWriter <-> Accumulator
        # =====================================================================

        # StoreController -> StreamWriter
        m.d.comb += [
            writer.req_valid.eq(store_ctrl.dma_req_valid),
            store_ctrl.dma_req_ready.eq(writer.req_ready),
            writer.req_addr.eq(store_ctrl.dma_req_addr),
            writer.req_len.eq(store_ctrl.dma_req_len),
            writer.data_valid.eq(store_ctrl.dma_data_valid),
            store_ctrl.dma_data_ready.eq(writer.data_ready),
            writer.data.eq(store_ctrl.dma_data),
            writer.data_strb.eq(store_ctrl.dma_data_strb),
            writer.data_last.eq(store_ctrl.dma_data_last),
            store_ctrl.dma_done.eq(writer.done),
        ]

        # Accumulator -> StoreController (read)
        m.d.comb += [
            accumulator.read_req.eq(store_ctrl.acc_read_req),
            accumulator.read_addr.eq(store_ctrl.acc_read_addr),
            store_ctrl.acc_read_data.eq(accumulator.read_data),
            store_ctrl.acc_read_valid.eq(accumulator.read_valid),
            # Activation is applied during store
            # StoreController applies its latched activation per element
            accumulator.read_activation.eq(0),
        ]

        # =====================================================================
        # ExecuteController <-> Scratchpad <-> SystolicArray <-> Accumulator
        # =====================================================================

        # ExecuteController reads A and B on the scratchpad's two read ports
        # (no other block reads the scratchpad)
        m.d.comb += [
            scratchpad.read_req.eq(exec_ctrl.sp_a_read_req),
            scratchpad.read_addr.eq(exec_ctrl.sp_a_read_addr),
            exec_ctrl.sp_a_read_data.eq(scratchpad.read_data),
            exec_ctrl.sp_a_read_valid.eq(scratchpad.read_valid),
            scratchpad.read_b_req.eq(exec_ctrl.sp_b_read_req),
            scratchpad.read_b_addr.eq(exec_ctrl.sp_b_read_addr),
            exec_ctrl.sp_b_read_data.eq(scratchpad.read_b_data),
            exec_ctrl.sp_b_read_valid.eq(scratchpad.read_b_valid),
        ]

        # ExecuteController writes result rows to the accumulator (the
        # StoreController only reads it)
        m.d.comb += [
            accumulator.write_req.eq(exec_ctrl.acc_write_req),
            accumulator.write_addr.eq(exec_ctrl.acc_write_addr),
            accumulator.write_data.eq(exec_ctrl.acc_write_data),
            accumulator.accumulate.eq(exec_ctrl.acc_accumulate),
        ]

        # =====================================================================
        # ExecuteController <-> SystolicArray
        # =====================================================================

        m.d.comb += [
            array.in_shift.eq(exec_ctrl.array_in_shift),
            array.in_control_dataflow.eq(exec_ctrl.array_in_dataflow),
            exec_ctrl.array_out_collision.eq(array.out_collision),
        ]
        for i in range(total_rows):
            for name in ("a", "valid", "first", "last"):
                m.d.comb += getattr(array, f"in_{name}_{i}").eq(
                    getattr(exec_ctrl, f"array_in_{name}_{i}")
                )
        for j in range(total_cols):
            m.d.comb += [
                getattr(array, f"in_b_{j}").eq(getattr(exec_ctrl, f"array_in_b_{j}")),
                getattr(array, f"in_w_{j}").eq(getattr(exec_ctrl, f"array_in_w_{j}")),
            ]
            for name in ("c", "c_valid", "psum", "psum_valid"):
                m.d.comb += getattr(exec_ctrl, f"array_out_{name}_{j}").eq(
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
            self.load_busy.eq(load_ctrl.busy | (queues[self.CTRL_LOAD].w_level != 0)),
            self.store_busy.eq(store_ctrl.busy | (queues[self.CTRL_STORE].w_level != 0)),
            self.exec_busy.eq(exec_ctrl.busy | (queues[self.CTRL_EXEC].w_level != 0)),
            self.busy.eq(self.load_busy | self.store_busy | self.exec_busy),
        ]

        # Per-controller completions (lossless)
        m.d.comb += [
            self.load_completed.eq(load_ctrl.completed),
            self.load_completed_id.eq(load_ctrl.completed_id),
            self.store_completed.eq(store_ctrl.completed),
            self.store_completed_id.eq(store_ctrl.completed_id),
            self.exec_completed.eq(exec_ctrl.completed),
            self.exec_completed_id.eq(exec_ctrl.completed_id),
            self.exec_result.eq(exec_ctrl.result),
            self.exec_result_id.eq(exec_ctrl.result_id),
            self.l1_error.eq(exec_ctrl.error),
        ]

        # Summary completion (priority: load > store > exec). Lossy when two
        # controllers complete in the same cycle; use the per-controller ports.
        with m.If(load_ctrl.completed):
            m.d.comb += [
                self.completed.eq(1),
                self.completed_type.eq(self.CTRL_LOAD),
                self.completed_id.eq(load_ctrl.completed_id),
            ]
        with m.Elif(store_ctrl.completed):
            m.d.comb += [
                self.completed.eq(1),
                self.completed_type.eq(self.CTRL_STORE),
                self.completed_id.eq(store_ctrl.completed_id),
            ]
        with m.Elif(exec_ctrl.completed):
            m.d.comb += [
                self.completed.eq(1),
                self.completed_type.eq(self.CTRL_EXEC),
                self.completed_id.eq(exec_ctrl.completed_id),
            ]
        with m.Else():
            m.d.comb += [
                self.completed.eq(0),
                self.completed_type.eq(0),
                self.completed_id.eq(0),
            ]

        # Interrupt (could be extended)
        m.d.comb += self.interrupt.eq(0)

        return m
