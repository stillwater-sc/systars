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
- AXI memory interface (read and write channels)
- Command interface for operation dispatch
- Status/interrupt signals

Data Flow for Matrix Multiply C = A × B + D:
1. Load A from DRAM → Scratchpad via LoadController + StreamReader
2. Load B from DRAM → Scratchpad via LoadController + StreamReader
3. Optionally preload D (bias) from Accumulator
4. Execute matmul via ExecuteController + SystolicArray
5. Store C from Accumulator → DRAM via StoreController + StreamWriter
"""

from amaranth import Module, Mux, Signal
from amaranth.lib.wiring import Component, In, Out

from .config import SystolicConfig
from .controller.execute import ExecuteController
from .controller.load import LoadController
from .controller.store import StoreController
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
            cmd_ready: Ready to accept command
            cmd_type: 0=load, 1=store, 2=execute
            cmd_opcode: Operation code
            cmd_* : Operation parameters

        AXI Read Interface (to external memory):
            axi_ar*: Read address channel
            axi_r*: Read data channel

        AXI Write Interface (to external memory):
            axi_aw*: Write address channel
            axi_w*: Write data channel
            axi_b*: Write response channel

        Status:
            busy: Any controller is busy
            load_busy, store_busy, exec_busy: Individual status
            completed: A command completed this cycle
            completed_type: Which controller completed (0=load, 1=store, 2=exec)
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
            "cmd_len": In(16),  # Transfer length in beats
            "cmd_id": In(8),
            # Execute-specific command fields
            "cmd_rs1": In(32),  # A address or preload source
            "cmd_rs2": In(32),  # B address
            "cmd_rd": In(32),  # C destination
            "cmd_k_dim": In(16),  # Inner dimension
            "cmd_activation": In(4),  # Activation function (for store)
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
            "interrupt": Out(1),
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
        # Command Dispatch
        # =====================================================================

        # Route commands to appropriate controller based on cmd_type
        m.d.comb += [
            # Load controller command
            load_ctrl.cmd_valid.eq(self.cmd_valid & (self.cmd_type == self.CTRL_LOAD)),
            load_ctrl.cmd_opcode.eq(self.cmd_opcode),
            load_ctrl.cmd_dram_addr.eq(self.cmd_dram_addr),
            load_ctrl.cmd_sp_addr.eq(self.cmd_local_addr),
            load_ctrl.cmd_len.eq(self.cmd_len),
            load_ctrl.cmd_id.eq(self.cmd_id),
            # Store controller command
            store_ctrl.cmd_valid.eq(self.cmd_valid & (self.cmd_type == self.CTRL_STORE)),
            store_ctrl.cmd_opcode.eq(self.cmd_opcode),
            store_ctrl.cmd_acc_addr.eq(self.cmd_local_addr),
            store_ctrl.cmd_dram_addr.eq(self.cmd_dram_addr),
            store_ctrl.cmd_len.eq(self.cmd_len),
            store_ctrl.cmd_id.eq(self.cmd_id),
            store_ctrl.cmd_activation.eq(self.cmd_activation),
            # Execute controller command
            exec_ctrl.cmd_valid.eq(self.cmd_valid & (self.cmd_type == self.CTRL_EXEC)),
            exec_ctrl.cmd_opcode.eq(self.cmd_opcode),
            exec_ctrl.cmd_rs1.eq(self.cmd_rs1),
            exec_ctrl.cmd_rs2.eq(self.cmd_rs2),
            exec_ctrl.cmd_rd.eq(self.cmd_rd),
            exec_ctrl.cmd_k_dim.eq(self.cmd_k_dim),
            exec_ctrl.cmd_id.eq(self.cmd_id),
        ]

        # Command ready: ready if target controller is ready
        with m.Switch(self.cmd_type):
            with m.Case(self.CTRL_LOAD):
                m.d.comb += self.cmd_ready.eq(load_ctrl.cmd_ready)
            with m.Case(self.CTRL_STORE):
                m.d.comb += self.cmd_ready.eq(store_ctrl.cmd_ready)
            with m.Case(self.CTRL_EXEC):
                m.d.comb += self.cmd_ready.eq(exec_ctrl.cmd_ready)
            with m.Default():
                m.d.comb += self.cmd_ready.eq(0)

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
            accumulator.read_activation.eq(store_ctrl.cmd_activation),
        ]

        # =====================================================================
        # ExecuteController <-> Scratchpad <-> SystolicArray <-> Accumulator
        # =====================================================================

        # ExecuteController -> Scratchpad (read A)
        # Note: Scratchpad has single read port, need to arbitrate
        # For simplicity, exec_ctrl has priority when busy
        sp_read_req_exec = Signal()
        sp_read_addr_exec = Signal(32)

        m.d.comb += [
            sp_read_req_exec.eq(exec_ctrl.sp_a_read_req | exec_ctrl.sp_b_read_req),
            # Use A address when A is requested, else B address
            sp_read_addr_exec.eq(
                Mux(exec_ctrl.sp_a_read_req, exec_ctrl.sp_a_read_addr, exec_ctrl.sp_b_read_addr)
            ),
        ]

        # Scratchpad read arbitration: exec takes priority when active
        with m.If(exec_ctrl.busy):
            m.d.comb += [
                scratchpad.read_req.eq(sp_read_req_exec),
                scratchpad.read_addr.eq(sp_read_addr_exec),
            ]
        with m.Else():
            # No other readers in this design, but could add more
            m.d.comb += [
                scratchpad.read_req.eq(0),
                scratchpad.read_addr.eq(0),
            ]

        # Scratchpad -> ExecuteController
        # Route read data to both A and B (controller knows which it requested)
        m.d.comb += [
            exec_ctrl.sp_a_read_data.eq(scratchpad.read_data),
            exec_ctrl.sp_a_read_valid.eq(scratchpad.read_valid),
            exec_ctrl.sp_b_read_data.eq(scratchpad.read_data),
            exec_ctrl.sp_b_read_valid.eq(scratchpad.read_valid),
        ]

        # ExecuteController <-> Accumulator
        # Read (for preload)
        acc_read_by_exec = Signal()
        m.d.comb += acc_read_by_exec.eq(exec_ctrl.acc_read_req & exec_ctrl.busy)

        # Accumulator read arbitration: store_ctrl takes priority
        with m.If(store_ctrl.busy):
            pass  # Already connected above
        with m.Elif(acc_read_by_exec):
            m.d.comb += [
                accumulator.read_req.eq(exec_ctrl.acc_read_req),
                accumulator.read_addr.eq(exec_ctrl.acc_read_addr),
            ]

        m.d.comb += [
            exec_ctrl.acc_read_data.eq(accumulator.read_data),
            exec_ctrl.acc_read_valid.eq(accumulator.read_valid),
        ]

        # Accumulator write (from exec_ctrl)
        m.d.comb += [
            accumulator.write_req.eq(exec_ctrl.acc_write_req),
            accumulator.write_addr.eq(exec_ctrl.acc_write_addr),
            accumulator.write_data.eq(exec_ctrl.acc_write_data),
            accumulator.accumulate.eq(exec_ctrl.acc_accumulate),
        ]

        # =====================================================================
        # ExecuteController <-> SystolicArray
        # =====================================================================

        # Control signals
        m.d.comb += [
            array.in_control_dataflow.eq(exec_ctrl.array_in_control_dataflow),
            array.in_control_propagate.eq(exec_ctrl.array_in_control_propagate),
            array.in_control_shift.eq(exec_ctrl.array_in_control_shift),
            array.in_valid.eq(exec_ctrl.array_in_valid),
            array.in_id.eq(exec_ctrl.array_in_id),
            array.in_last.eq(exec_ctrl.array_in_last),
        ]

        # Input vectors A (one per row)
        for i in range(total_rows):
            m.d.comb += getattr(array, f"in_a_{i}").eq(getattr(exec_ctrl, f"array_in_a_{i}"))

        # Input vectors B and D (one per column)
        for j in range(total_cols):
            m.d.comb += [
                getattr(array, f"in_b_{j}").eq(getattr(exec_ctrl, f"array_in_b_{j}")),
                getattr(array, f"in_d_{j}").eq(getattr(exec_ctrl, f"array_in_d_{j}")),
            ]

        # Output vectors C (one per column)
        for j in range(total_cols):
            m.d.comb += getattr(exec_ctrl, f"array_out_c_{j}").eq(getattr(array, f"out_c_{j}"))

        # Output control
        m.d.comb += [
            exec_ctrl.array_out_valid.eq(array.out_valid),
            exec_ctrl.array_out_id.eq(array.out_id),
            exec_ctrl.array_out_last.eq(array.out_last),
        ]

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
            self.load_busy.eq(load_ctrl.busy),
            self.store_busy.eq(store_ctrl.busy),
            self.exec_busy.eq(exec_ctrl.busy),
            self.busy.eq(load_ctrl.busy | store_ctrl.busy | exec_ctrl.busy),
        ]

        # Completion signaling (priority: load > store > exec)
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
