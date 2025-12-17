"""
TransposeLoadController - 2D DMA with optional matrix transpose.

Handles loading matrices from DRAM to scratchpad with optional transpose:
- LOAD_CONTIGUOUS: Standard row-by-row copy (B matrix, no transpose needed)
- LOAD_TRANSPOSE: Column-by-column gather, write as rows (A matrix transpose)

For systolic array matmul C = A @ B:
- Matrix A (M×K) must be stored transposed (column-major) in scratchpad
- Matrix B (K×N) stored normally (row-major) in scratchpad

Memory Layout Transformation:

    DRAM (row-major):              Scratchpad A (column-major):
    ┌─────────────────┐            ┌─────────────────┐
    │ A[0,0] A[0,1] A[0,2] │            │ A[0,0] A[1,0] A[2,0] A[3,0] │  addr 0 (col 0)
    │ A[1,0] A[1,1] A[1,2] │   ──→     │ A[0,1] A[1,1] A[2,1] A[3,1] │  addr 1 (col 1)
    │ A[2,0] A[2,1] A[2,2] │            │ A[0,2] A[1,2] A[2,2] A[3,2] │  addr 2 (col 2)
    │ A[3,0] A[3,1] A[3,2] │            └─────────────────┘
    └─────────────────┘

Transfer sequence for transpose:
    1. Read column 0 of A (strided): A[0,0], A[1,0], A[2,0], A[3,0]
       Write as row 0 in scratchpad
    2. Read column 1 of A (strided): A[0,1], A[1,1], A[2,1], A[3,1]
       Write as row 1 in scratchpad
    3. ... repeat for all K columns

State Machine:
    IDLE → CALC_PARAMS → ISSUE_READ → RECV_DATA → WRITE_SP → NEXT_ROW → DONE

Performance:
    - Contiguous: ~1 cycle per element (burst efficiency)
    - Transpose: ~5 cycles per element (strided access overhead at 250MHz)
    - For 16×16 matrix: contiguous ~256 cycles, transpose ~1280 cycles
"""

from amaranth import Module, Signal
from amaranth.lib.wiring import Component, In, Out

from ..config import SystolicConfig


class TransposeOpcode:
    """Opcodes for TransposeLoadController."""

    LOAD_CONTIGUOUS = 0x10  # Standard 2D load, no transpose
    LOAD_TRANSPOSE = 0x11  # Load with transpose (columns → rows)


class TransposeLoadController(Component):
    """
    Load controller with 2D addressing and transpose support.

    Handles loading matrices from DRAM with optional transpose operation
    for preparing data in the correct layout for systolic array feeding.

    Ports:
        Command Interface:
            cmd_valid: Command available
            cmd_ready: Controller ready
            cmd_opcode: LOAD_CONTIGUOUS or LOAD_TRANSPOSE
            cmd_dram_addr: Base address in DRAM
            cmd_sp_addr: Base address in scratchpad
            cmd_src_rows: Number of rows in source matrix
            cmd_src_cols: Number of columns in source matrix
            cmd_elem_bytes: Bytes per element (1, 2, or 4)
            cmd_id: Command ID for tracking

        Strided DMA Interface:
            dma_req_valid/ready: Request handshake
            dma_req_addr: Read address
            dma_req_count: Number of elements
            dma_req_stride: Stride between elements
            dma_req_elem_bytes: Element size
            dma_resp_valid/ready: Response handshake
            dma_resp_data: Packed row data

        Scratchpad Interface:
            sp_write_en: Write enable
            sp_write_addr: Destination address
            sp_write_data: Data to write
            sp_write_mask: Byte mask

        Status:
            busy: Transfer in progress
            completed: Transfer done
            completed_id: ID of completed command
            progress_row: Current row being transferred
    """

    def __init__(self, config: SystolicConfig):
        self.config = config
        sp_width = config.sp_width
        mask_width = sp_width // 8

        super().__init__(
            {
                # Command interface
                "cmd_valid": In(1),
                "cmd_ready": Out(1),
                "cmd_opcode": In(8),
                "cmd_dram_addr": In(64),
                "cmd_sp_addr": In(32),
                "cmd_src_rows": In(16),  # M dimension
                "cmd_src_cols": In(16),  # K dimension
                "cmd_elem_bytes": In(4),
                "cmd_id": In(8),
                # Strided DMA interface
                "dma_req_valid": Out(1),
                "dma_req_ready": In(1),
                "dma_req_addr": Out(64),
                "dma_req_count": Out(16),
                "dma_req_stride": Out(16),
                "dma_req_elem_bytes": Out(4),
                "dma_resp_valid": In(1),
                "dma_resp_ready": Out(1),
                "dma_resp_data": In(sp_width),
                # Scratchpad write interface
                "sp_write_en": Out(1),
                "sp_write_addr": Out(32),
                "sp_write_data": Out(sp_width),
                "sp_write_mask": Out(mask_width),
                # Status
                "busy": Out(1),
                "completed": Out(1),
                "completed_id": Out(8),
                "progress_row": Out(16),  # Current row number
            }
        )

    def elaborate(self, _platform):
        m = Module()
        cfg = self.config

        mask_all_ones = (1 << (cfg.sp_width // 8)) - 1

        # State encoding
        STATE_IDLE = 0
        STATE_CALC = 1
        STATE_ISSUE = 2
        STATE_RECV = 3
        STATE_WRITE = 4
        STATE_NEXT = 5
        STATE_DONE = 6

        state = Signal(3, init=STATE_IDLE)

        # Command registers
        opcode_reg = Signal(8)
        dram_base = Signal(64)
        sp_base = Signal(32)
        src_rows = Signal(16)  # M
        src_cols = Signal(16)  # K
        elem_bytes = Signal(4)
        cmd_id_reg = Signal(8)

        # Transfer state
        current_row = Signal(16)  # For contiguous: row index; for transpose: column index
        total_rows = Signal(16)  # Number of rows to transfer
        is_transpose = Signal()

        # Calculated addresses
        dma_addr = Signal(64)
        dma_count = Signal(16)
        dma_stride = Signal(16)

        # Buffered response
        resp_data_buf = Signal(cfg.sp_width)
        resp_valid_buf = Signal()

        # Default outputs
        m.d.comb += [
            self.cmd_ready.eq(0),
            self.dma_req_valid.eq(0),
            self.dma_req_addr.eq(0),
            self.dma_req_count.eq(0),
            self.dma_req_stride.eq(0),
            self.dma_req_elem_bytes.eq(0),
            self.dma_resp_ready.eq(0),
            self.sp_write_en.eq(0),
            self.sp_write_addr.eq(0),
            self.sp_write_data.eq(0),
            self.sp_write_mask.eq(0),
            self.busy.eq(state != 0),  # 0 = IDLE
            self.completed.eq(0),
            self.completed_id.eq(0),
            self.progress_row.eq(current_row),
        ]

        with m.FSM(init="IDLE"):
            with m.State("IDLE"):
                m.d.comb += [
                    state.eq(STATE_IDLE),
                    self.cmd_ready.eq(1),
                ]

                with m.If(self.cmd_valid):
                    # Latch command
                    m.d.sync += [
                        opcode_reg.eq(self.cmd_opcode),
                        dram_base.eq(self.cmd_dram_addr),
                        sp_base.eq(self.cmd_sp_addr),
                        src_rows.eq(self.cmd_src_rows),
                        src_cols.eq(self.cmd_src_cols),
                        elem_bytes.eq(self.cmd_elem_bytes),
                        cmd_id_reg.eq(self.cmd_id),
                        current_row.eq(0),
                    ]

                    # Determine mode
                    with m.If(self.cmd_opcode == TransposeOpcode.LOAD_TRANSPOSE):
                        m.d.sync += [
                            is_transpose.eq(1),
                            # For transpose: we read K columns, each becomes a row
                            total_rows.eq(self.cmd_src_cols),  # K output rows
                        ]
                    with m.Else():
                        m.d.sync += [
                            is_transpose.eq(0),
                            # For contiguous: we read M rows
                            total_rows.eq(self.cmd_src_rows),  # M output rows
                        ]

                    m.next = "CALC"

            with m.State("CALC"):
                m.d.comb += state.eq(STATE_CALC)

                # Calculate DMA parameters for current row
                with m.If(is_transpose):
                    # Transpose mode: read column 'current_row' of source
                    # Base addr + column_offset, stride = src_cols
                    m.d.sync += [
                        dma_addr.eq(dram_base + (current_row * elem_bytes)),
                        dma_count.eq(src_rows),  # M elements per column
                        dma_stride.eq(src_cols),  # Stride = row width
                    ]
                with m.Else():
                    # Contiguous mode: read row 'current_row' of source
                    # Base addr + row_offset, stride = 1 (contiguous)
                    m.d.sync += [
                        dma_addr.eq(dram_base + (current_row * src_cols * elem_bytes)),
                        dma_count.eq(src_cols),  # K elements per row
                        dma_stride.eq(1),  # Contiguous
                    ]

                m.next = "ISSUE"

            with m.State("ISSUE"):
                m.d.comb += [
                    state.eq(STATE_ISSUE),
                    self.dma_req_valid.eq(1),
                    self.dma_req_addr.eq(dma_addr),
                    self.dma_req_count.eq(dma_count),
                    self.dma_req_stride.eq(dma_stride),
                    self.dma_req_elem_bytes.eq(elem_bytes),
                ]

                with m.If(self.dma_req_ready):
                    m.next = "RECV"

            with m.State("RECV"):
                m.d.comb += [
                    state.eq(STATE_RECV),
                    self.dma_resp_ready.eq(1),
                ]

                with m.If(self.dma_resp_valid):
                    # Buffer the response
                    m.d.sync += [
                        resp_data_buf.eq(self.dma_resp_data),
                        resp_valid_buf.eq(1),
                    ]
                    m.next = "WRITE"

            with m.State("WRITE"):
                m.d.comb += [
                    state.eq(STATE_WRITE),
                    self.sp_write_en.eq(1),
                    self.sp_write_addr.eq(sp_base + current_row),
                    self.sp_write_data.eq(resp_data_buf),
                    self.sp_write_mask.eq(mask_all_ones),
                ]

                m.d.sync += resp_valid_buf.eq(0)
                m.next = "NEXT"

            with m.State("NEXT"):
                m.d.comb += state.eq(STATE_NEXT)

                m.d.sync += current_row.eq(current_row + 1)

                with m.If(current_row >= total_rows - 1):
                    m.next = "DONE"
                with m.Else():
                    m.next = "CALC"

            with m.State("DONE"):
                m.d.comb += [
                    state.eq(STATE_DONE),
                    self.completed.eq(1),
                    self.completed_id.eq(cmd_id_reg),
                ]
                m.next = "IDLE"

        return m


class MatrixLoader(Component):
    """
    High-level matrix loader combining strided reader and transpose controller.

    Provides a simple interface for loading matrices A and B for systolic matmul:
    - load_a: Loads A with transpose (row-major → column-major)
    - load_b: Loads B directly (row-major → row-major)

    This component integrates:
    - TransposeLoadController for orchestration
    - StridedStreamReader for DMA access

    Ports:
        load_a_valid/ready: Load matrix A (with transpose)
        load_b_valid/ready: Load matrix B (no transpose)
        a_dram_addr: A base address in DRAM
        b_dram_addr: B base address in DRAM
        a_sp_addr: A destination in scratchpad
        b_sp_addr: B destination in scratchpad
        m_dim: M dimension (rows of A)
        k_dim: K dimension (cols of A, rows of B)
        n_dim: N dimension (cols of B)

        Scratchpad write interface (directly connected)
        AXI memory interface (directly connected)

        Status: busy, done
    """

    def __init__(self, config: SystolicConfig):
        self.config = config
        sp_width = config.sp_width
        mask_width = sp_width // 8
        buswidth = config.dma_buswidth

        super().__init__(
            {
                # High-level load commands
                "load_a_valid": In(1),
                "load_a_ready": Out(1),
                "load_b_valid": In(1),
                "load_b_ready": Out(1),
                # Matrix addresses
                "a_dram_addr": In(64),
                "b_dram_addr": In(64),
                "a_sp_addr": In(32),
                "b_sp_addr": In(32),
                # Dimensions
                "m_dim": In(16),
                "k_dim": In(16),
                "n_dim": In(16),
                # Scratchpad write interface
                "sp_write_en": Out(1),
                "sp_write_addr": Out(32),
                "sp_write_data": Out(sp_width),
                "sp_write_mask": Out(mask_width),
                # AXI Read Address Channel
                "mem_arvalid": Out(1),
                "mem_arready": In(1),
                "mem_araddr": Out(64),
                "mem_arlen": Out(8),
                "mem_arsize": Out(3),
                "mem_arburst": Out(2),
                # AXI Read Data Channel
                "mem_rvalid": In(1),
                "mem_rready": Out(1),
                "mem_rdata": In(buswidth),
                "mem_rlast": In(1),
                "mem_rresp": In(2),
                # Status
                "busy": Out(1),
                "done": Out(1),
            }
        )

    def elaborate(self, _platform):
        m = Module()
        cfg = self.config

        # Import here to avoid circular dependency
        from ..dma.strided_reader import StridedStreamReader

        # Instantiate components
        reader = StridedStreamReader(cfg)
        loader = TransposeLoadController(cfg)
        m.submodules.reader = reader
        m.submodules.loader = loader

        # Connect DMA reader to loader
        m.d.comb += [
            # Reader request <- Loader DMA request
            reader.req_valid.eq(loader.dma_req_valid),
            loader.dma_req_ready.eq(reader.req_ready),
            reader.req_addr.eq(loader.dma_req_addr),
            reader.req_count.eq(loader.dma_req_count),
            reader.req_stride.eq(loader.dma_req_stride),
            reader.req_elem_bytes.eq(loader.dma_req_elem_bytes),
            # Reader response -> Loader DMA response
            loader.dma_resp_valid.eq(reader.resp_valid),
            reader.resp_ready.eq(loader.dma_resp_ready),
            loader.dma_resp_data.eq(reader.resp_data),
        ]

        # Connect reader to AXI interface (pass through)
        m.d.comb += [
            self.mem_arvalid.eq(reader.mem_arvalid),
            reader.mem_arready.eq(self.mem_arready),
            self.mem_araddr.eq(reader.mem_araddr),
            self.mem_arlen.eq(reader.mem_arlen),
            self.mem_arsize.eq(reader.mem_arsize),
            self.mem_arburst.eq(reader.mem_arburst),
            reader.mem_rvalid.eq(self.mem_rvalid),
            self.mem_rready.eq(reader.mem_rready),
            reader.mem_rdata.eq(self.mem_rdata),
            reader.mem_rlast.eq(self.mem_rlast),
            reader.mem_rresp.eq(self.mem_rresp),
        ]

        # Connect loader to scratchpad interface (pass through)
        m.d.comb += [
            self.sp_write_en.eq(loader.sp_write_en),
            self.sp_write_addr.eq(loader.sp_write_addr),
            self.sp_write_data.eq(loader.sp_write_data),
            self.sp_write_mask.eq(loader.sp_write_mask),
        ]

        # State machine for handling load_a and load_b commands
        state = Signal(2, init=0)

        # Element size (assuming 1 byte = int8)
        elem_bytes = Signal(4, init=1)

        # Default outputs
        m.d.comb += [
            self.load_a_ready.eq(0),
            self.load_b_ready.eq(0),
            loader.cmd_valid.eq(0),
            loader.cmd_opcode.eq(0),
            loader.cmd_dram_addr.eq(0),
            loader.cmd_sp_addr.eq(0),
            loader.cmd_src_rows.eq(0),
            loader.cmd_src_cols.eq(0),
            loader.cmd_elem_bytes.eq(elem_bytes),
            loader.cmd_id.eq(0),
            self.busy.eq(state != 0),  # 0 = IDLE
            self.done.eq(0),
        ]

        with m.FSM(init="IDLE"):
            with m.State("IDLE"):
                m.d.comb += [
                    self.load_a_ready.eq(1),
                    self.load_b_ready.eq(1),
                ]

                # Priority: load_a before load_b
                with m.If(self.load_a_valid):
                    # Issue transpose load for A
                    m.d.comb += [
                        loader.cmd_valid.eq(1),
                        loader.cmd_opcode.eq(TransposeOpcode.LOAD_TRANSPOSE),
                        loader.cmd_dram_addr.eq(self.a_dram_addr),
                        loader.cmd_sp_addr.eq(self.a_sp_addr),
                        loader.cmd_src_rows.eq(self.m_dim),
                        loader.cmd_src_cols.eq(self.k_dim),
                    ]
                    m.next = "LOAD_A"

                with m.Elif(self.load_b_valid):
                    # Issue contiguous load for B
                    m.d.comb += [
                        loader.cmd_valid.eq(1),
                        loader.cmd_opcode.eq(TransposeOpcode.LOAD_CONTIGUOUS),
                        loader.cmd_dram_addr.eq(self.b_dram_addr),
                        loader.cmd_sp_addr.eq(self.b_sp_addr),
                        loader.cmd_src_rows.eq(self.k_dim),
                        loader.cmd_src_cols.eq(self.n_dim),
                    ]
                    m.next = "LOAD_B"

            with m.State("LOAD_A"), m.If(loader.completed):
                m.d.comb += self.done.eq(1)
                m.next = "IDLE"

            with m.State("LOAD_B"), m.If(loader.completed):
                m.d.comb += self.done.eq(1)
                m.next = "IDLE"

        return m
