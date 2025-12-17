"""
SkewBuffer - Input skew buffers for systolic array wavefront injection.

The SkewBuffer provides the staggered timing needed for systolic array operation.
For matrix multiply C = A @ B:
- A rows enter from the left, row i delayed by i cycles
- B columns enter from the top, column j delayed by j cycles

This creates the diagonal wavefront pattern where PE[i,j] receives its first
valid (a,b) pair at cycle (i + j).

Architecture:
                         SRAM Read Pipeline
                               │
                    ┌──────────┴──────────┐
                    │    Skew Buffer      │
                    │                     │
          Lane 0 ───┤→ [      ] ──────────┼──→ Array in_0
          Lane 1 ───┤→ [REG   ] ──────────┼──→ Array in_1
          Lane 2 ───┤→ [REG][REG] ────────┼──→ Array in_2
          Lane 3 ───┤→ [REG][REG][REG] ───┼──→ Array in_3
                    │     ...             │
                    └─────────────────────┘

Each lane i has i pipeline registers, creating the skew pattern.
Data entering lane 0 appears immediately; lane N-1 appears N-1 cycles later.

Memory Timing (250MHz, 4-cycle SRAM):
    Cycle 0: Issue read request
    Cycle 1: Address decode
    Cycle 2: Bitline sense
    Cycle 3: Output mux
    Cycle 4: Data valid at skew buffer input
    Cycle 4+i: Data valid at array input for lane i
"""

from amaranth import Module, Signal, signed, unsigned
from amaranth.lib.wiring import Component, In, Out

from ..config import SystolicConfig


class SkewBuffer(Component):
    """
    Skew buffer for one dimension of systolic array input.

    Implements N lanes where lane i has i stages of delay.
    Used for both A (row) and B (column) inputs.

    Ports:
        in_data_0..N: Input data for each lane (from SRAM read)
        in_valid: Input data is valid
        out_data_0..N: Skewed output data for each lane
        out_valid_0..N: Per-lane valid signals
        flush: Clear all pipeline registers

    Parameters:
        num_lanes: Number of parallel lanes (array rows or columns)
        data_width: Width of each data element in bits
        name_prefix: Prefix for signal names ("a" or "b")
    """

    def __init__(self, num_lanes: int, data_width: int, name_prefix: str = ""):
        self.num_lanes = num_lanes
        self.data_width = data_width
        self.name_prefix = name_prefix

        ports = {
            "in_valid": In(1),
            "flush": In(1),
        }

        # Per-lane input and output ports
        for i in range(num_lanes):
            ports[f"in_data_{i}"] = In(signed(data_width))
            ports[f"out_data_{i}"] = Out(signed(data_width))
            ports[f"out_valid_{i}"] = Out(1)

        super().__init__(ports)

    def elaborate(self, _platform):
        m = Module()

        n = self.num_lanes
        w = self.data_width

        # Create skew pipeline for each lane
        # Lane i has i stages of delay
        for lane in range(n):
            in_data = getattr(self, f"in_data_{lane}")
            out_data = getattr(self, f"out_data_{lane}")
            out_valid = getattr(self, f"out_valid_{lane}")

            if lane == 0:
                # Lane 0: no delay, direct pass-through
                m.d.comb += [
                    out_data.eq(in_data),
                    out_valid.eq(self.in_valid),
                ]
            else:
                # Lane i: i stages of pipeline registers
                data_pipe = [
                    Signal(signed(w), name=f"{self.name_prefix}_pipe_{lane}_{stage}")
                    for stage in range(lane)
                ]
                valid_pipe = [
                    Signal(name=f"{self.name_prefix}_valid_{lane}_{stage}") for stage in range(lane)
                ]

                # First stage from input
                with m.If(self.flush):
                    m.d.sync += [
                        data_pipe[0].eq(0),
                        valid_pipe[0].eq(0),
                    ]
                with m.Else():
                    m.d.sync += [
                        data_pipe[0].eq(in_data),
                        valid_pipe[0].eq(self.in_valid),
                    ]

                # Chain remaining stages
                for stage in range(1, lane):
                    with m.If(self.flush):
                        m.d.sync += [
                            data_pipe[stage].eq(0),
                            valid_pipe[stage].eq(0),
                        ]
                    with m.Else():
                        m.d.sync += [
                            data_pipe[stage].eq(data_pipe[stage - 1]),
                            valid_pipe[stage].eq(valid_pipe[stage - 1]),
                        ]

                # Output from last stage
                m.d.comb += [
                    out_data.eq(data_pipe[-1]),
                    out_valid.eq(valid_pipe[-1]),
                ]

        return m


class SRAMReadScheduler(Component):
    """
    Schedules staggered SRAM reads for systolic array feeding.

    For an NxN array computing over K iterations:
    - Issues read requests in a pattern that, combined with skew buffers,
      produces the correct wavefront timing
    - Handles SRAM latency by pipelining requests

    Two operating modes:
    1. Parallel banks: Read all rows/cols simultaneously from separate banks
    2. Sequential: Read one row/col per cycle, use deeper skew buffers

    This implementation uses parallel banks for maximum throughput.

    Timing (4-cycle SRAM, parallel read):
        Cycle 0: Issue read for k=0 (all rows of A, all cols of B)
        Cycle 4: Data arrives at skew buffer inputs
        Cycle 4: Lane 0 data enters array
        Cycle 5: Lane 1 data enters array
        ...
        Cycle 4+N-1: Lane N-1 data enters array
        Cycle 5: Issue read for k=1
        ...

    Ports:
        start: Begin a new matmul operation
        a_base_addr: Base address of A in scratchpad
        b_base_addr: Base address of B in scratchpad
        k_dim: Inner dimension (number of k iterations)

        sp_a_read_req: Scratchpad A read request
        sp_a_read_addr: Scratchpad A read address
        sp_a_read_data: Scratchpad A read data (packed rows)
        sp_a_read_valid: Scratchpad A data valid

        sp_b_read_req: Scratchpad B read request
        sp_b_read_addr: Scratchpad B read address
        sp_b_read_data: Scratchpad B read data (packed cols)
        sp_b_read_valid: Scratchpad B data valid

        out_valid: Output to skew buffers is valid
        busy: Operation in progress
        done: Operation complete
    """

    def __init__(self, config: SystolicConfig):
        self.config = config

        total_rows = config.grid_rows * config.tile_rows
        total_cols = config.grid_cols * config.tile_cols

        ports = {
            # Control
            "start": In(1),
            "a_base_addr": In(32),
            "b_base_addr": In(32),
            "k_dim": In(16),
            # Scratchpad A interface
            "sp_a_read_req": Out(1),
            "sp_a_read_addr": Out(32),
            "sp_a_read_data": In(unsigned(config.sp_width)),
            "sp_a_read_valid": In(1),
            # Scratchpad B interface
            "sp_b_read_req": Out(1),
            "sp_b_read_addr": Out(32),
            "sp_b_read_data": In(unsigned(config.sp_width)),
            "sp_b_read_valid": In(1),
            # Status
            "out_valid": Out(1),
            "busy": Out(1),
            "done": Out(1),
            # Current k index (for external use)
            "current_k": Out(16),
        }

        # Unpacked output data (one element per row/col)
        for i in range(total_rows):
            ports[f"out_a_{i}"] = Out(signed(config.input_bits))
        for j in range(total_cols):
            ports[f"out_b_{j}"] = Out(signed(config.weight_bits))

        super().__init__(ports)

    def elaborate(self, _platform):
        m = Module()
        cfg = self.config

        total_rows = cfg.grid_rows * cfg.tile_rows
        total_cols = cfg.grid_cols * cfg.tile_cols
        sram_latency = cfg.spad_read_delay

        # State
        STATE_IDLE = 0
        STATE_ISSUE = 1
        STATE_WAIT = 2
        STATE_OUTPUT = 3
        STATE_DONE = 4

        state = Signal(3, init=STATE_IDLE)

        # Counters
        k_count = Signal(16)
        k_limit = Signal(16)
        wait_count = Signal(4)

        # Latched addresses
        a_base = Signal(32)
        b_base = Signal(32)

        # Default outputs
        m.d.comb += [
            self.sp_a_read_req.eq(0),
            self.sp_a_read_addr.eq(0),
            self.sp_b_read_req.eq(0),
            self.sp_b_read_addr.eq(0),
            self.out_valid.eq(0),
            self.busy.eq(state != STATE_IDLE),
            self.done.eq(0),
            self.current_k.eq(k_count),
        ]

        # Default unpacked outputs to zero
        for i in range(total_rows):
            m.d.comb += getattr(self, f"out_a_{i}").eq(0)
        for j in range(total_cols):
            m.d.comb += getattr(self, f"out_b_{j}").eq(0)

        with m.FSM(init="IDLE"):
            with m.State("IDLE"):
                m.d.comb += state.eq(STATE_IDLE)

                with m.If(self.start):
                    m.d.sync += [
                        k_count.eq(0),
                        k_limit.eq(self.k_dim),
                        a_base.eq(self.a_base_addr),
                        b_base.eq(self.b_base_addr),
                    ]
                    m.next = "ISSUE"

            with m.State("ISSUE"):
                m.d.comb += state.eq(STATE_ISSUE)

                # Issue read requests for current k
                # A: read row k (all elements packed in one read)
                # B: read col k (all elements packed, already transposed in memory)
                m.d.comb += [
                    self.sp_a_read_req.eq(1),
                    self.sp_a_read_addr.eq(a_base + k_count),
                    self.sp_b_read_req.eq(1),
                    self.sp_b_read_addr.eq(b_base + k_count),
                ]

                m.d.sync += wait_count.eq(0)
                m.next = "WAIT"

            with m.State("WAIT"):
                m.d.comb += state.eq(STATE_WAIT)

                # Wait for SRAM latency
                m.d.sync += wait_count.eq(wait_count + 1)

                with m.If(wait_count >= sram_latency - 1):
                    m.next = "OUTPUT"

            with m.State("OUTPUT"):
                m.d.comb += state.eq(STATE_OUTPUT)

                # Data should be valid now
                with m.If(self.sp_a_read_valid & self.sp_b_read_valid):
                    m.d.comb += self.out_valid.eq(1)

                    # Unpack A data (one element per row)
                    for i in range(total_rows):
                        start_bit = i * cfg.input_bits
                        end_bit = start_bit + cfg.input_bits
                        m.d.comb += getattr(self, f"out_a_{i}").eq(
                            self.sp_a_read_data[start_bit:end_bit]
                        )

                    # Unpack B data (one element per column)
                    for j in range(total_cols):
                        start_bit = j * cfg.weight_bits
                        end_bit = start_bit + cfg.weight_bits
                        m.d.comb += getattr(self, f"out_b_{j}").eq(
                            self.sp_b_read_data[start_bit:end_bit]
                        )

                    # Check if done
                    with m.If(k_count >= k_limit - 1):
                        m.next = "DONE"
                    with m.Else():
                        m.d.sync += k_count.eq(k_count + 1)
                        m.next = "ISSUE"

            with m.State("DONE"):
                m.d.comb += [
                    state.eq(STATE_DONE),
                    self.done.eq(1),
                ]
                m.next = "IDLE"

        return m


class SkewedArrayFeeder(Component):
    """
    Complete skewed feeding system for systolic array.

    Combines:
    - SRAM read scheduler (handles memory latency)
    - A skew buffer (row delays)
    - B skew buffer (column delays)

    Data flow:
        Scratchpad ──→ Read Scheduler ──→ Skew Buffers ──→ Systolic Array

    Memory organization (after DMA transpose):
        Scratchpad A banks: Row k of A stored at address k
            Read produces: [A[0,k], A[1,k], ..., A[N-1,k]] (column k of A^T = row k of A)
            Actually for row-major A: [A[k,0], A[k,1], ..., A[k,N-1]]

        Scratchpad B banks: Column k of B stored at address k (transposed by DMA)
            Read produces: [B[0,k], B[1,k], ..., B[N-1,k]] = column k of B

    For C = A @ B, at iteration k:
        - Feed A[*,k] to left edge (row k of A, becomes column vector)
        - Feed B[k,*] to top edge (row k of B^T = column k of B)

    Wait, let me reconsider the data layout more carefully:

    For systolic array computing C[i,j] = sum_k A[i,k] * B[k,j]:
        - Row i of array needs A[i,0], A[i,1], ..., A[i,K-1] over time
        - Column j of array needs B[0,j], B[1,j], ..., B[K-1,j] over time

    Memory layout for feeding:
        - A stored row-major: A[i,*] is contiguous
          For k=0: feed A[0,0] to row 0, A[1,0] to row 1, etc.
          This requires reading column k of A, which means strided access
          OR store A transposed: A^T[k,*] = A[*,k] is contiguous

        - B stored column-major (transposed by DMA): B^T[j,*] = B[*,j] is contiguous
          For k=0: feed B[0,0] to col 0, B[0,1] to col 1, etc.
          This is row k of B, which is contiguous in original B

    Simplest approach: Both A and B stored with k as the row index
        - A bank: address k contains [A[0,k], A[1,k], ..., A[N-1,k]]
        - B bank: address k contains [B[k,0], B[k,1], ..., B[M-1,0]]

    This means:
        - A is stored transposed (column-major)
        - B is stored row-major (natural for feeding columns to array)

    Actually, let's match the typical convention:
        - A stored row-major in original DRAM, DMA loads without transpose
          Scratchpad A: address i contains row i of A = [A[i,0], A[i,1], ...]
        - B stored row-major in DRAM, DMA transposes during load
          Scratchpad B: address j contains column j of B = [B[0,j], B[1,j], ...]

    For matmul iteration k:
        - Need A[*,k]: this is column k of A, requires reading all rows
          With row-major storage, this is strided - not ideal
        - Need B[k,*]: this is row k of B, but we stored columns
          With column-major (transposed) storage, we read address k and get [B[k,0], B[k,1], ...]
          Wait, that's wrong. If address j has column j, then to get row k we need all addresses.

    Let me think again about the standard approach:

    For weight-stationary (B stationary):
        - Preload B into array (one column at a time, stays in place)
        - Stream A through (row at a time)

    For output-stationary (C stationary):
        - Stream both A and B
        - A[i,k] enters at row i, time offset by i
        - B[k,j] enters at col j, time offset by j

    The key insight: for each k iteration, we need:
        - One element per array row from A: specifically A[row, k]
        - One element per array column from B: specifically B[k, col]

    Memory organization for efficient access:
        - Store A in "k-major" order: address k has [A[0,k], A[1,k], ..., A[rows-1,k]]
          This means A is stored transposed!
        - Store B in "k-major" order: address k has [B[k,0], B[k,1], ..., B[k,cols-1]]
          This means B is stored normally (row-major)

    DMA responsibility:
        - When loading A from DRAM (row-major), transpose to column-major in scratchpad
        - When loading B from DRAM (row-major), store directly (already correct layout)

    Or alternatively, require software to prepare data in the right layout.

    For this implementation, I'll document the expected layout and assume DMA handles it.

    Ports:
        start: Begin feeding sequence
        a_base_addr: Base address of A data in scratchpad
        b_base_addr: Base address of B data in scratchpad
        k_dim: Number of k iterations

        Scratchpad interfaces (directly connected)

        Array outputs (directly connected to systolic array inputs)

        Status signals
    """

    def __init__(self, config: SystolicConfig):
        self.config = config

        total_rows = config.grid_rows * config.tile_rows
        total_cols = config.grid_cols * config.tile_cols

        ports = {
            # Control
            "start": In(1),
            "a_base_addr": In(32),
            "b_base_addr": In(32),
            "k_dim": In(16),
            # Scratchpad A interface
            "sp_a_read_req": Out(1),
            "sp_a_read_addr": Out(32),
            "sp_a_read_data": In(unsigned(config.sp_width)),
            "sp_a_read_valid": In(1),
            # Scratchpad B interface
            "sp_b_read_req": Out(1),
            "sp_b_read_addr": Out(32),
            "sp_b_read_data": In(unsigned(config.sp_width)),
            "sp_b_read_valid": In(1),
            # Status
            "busy": Out(1),
            "done": Out(1),
            "feeding": Out(1),  # Currently outputting valid data to array
        }

        # Skewed outputs to systolic array
        for i in range(total_rows):
            ports[f"array_in_a_{i}"] = Out(signed(config.input_bits))
            ports[f"array_in_a_valid_{i}"] = Out(1)
        for j in range(total_cols):
            ports[f"array_in_b_{j}"] = Out(signed(config.weight_bits))
            ports[f"array_in_b_valid_{j}"] = Out(1)

        super().__init__(ports)

    def elaborate(self, _platform):
        m = Module()
        cfg = self.config

        total_rows = cfg.grid_rows * cfg.tile_rows
        total_cols = cfg.grid_cols * cfg.tile_cols

        # Instantiate read scheduler
        scheduler = SRAMReadScheduler(cfg)
        m.submodules.scheduler = scheduler

        # Instantiate skew buffers
        a_skew = SkewBuffer(total_rows, cfg.input_bits, "a")
        b_skew = SkewBuffer(total_cols, cfg.weight_bits, "b")
        m.submodules.a_skew = a_skew
        m.submodules.b_skew = b_skew

        # Connect control signals
        m.d.comb += [
            scheduler.start.eq(self.start),
            scheduler.a_base_addr.eq(self.a_base_addr),
            scheduler.b_base_addr.eq(self.b_base_addr),
            scheduler.k_dim.eq(self.k_dim),
        ]

        # Connect scratchpad interfaces (pass through)
        m.d.comb += [
            self.sp_a_read_req.eq(scheduler.sp_a_read_req),
            self.sp_a_read_addr.eq(scheduler.sp_a_read_addr),
            scheduler.sp_a_read_data.eq(self.sp_a_read_data),
            scheduler.sp_a_read_valid.eq(self.sp_a_read_valid),
            self.sp_b_read_req.eq(scheduler.sp_b_read_req),
            self.sp_b_read_addr.eq(scheduler.sp_b_read_addr),
            scheduler.sp_b_read_data.eq(self.sp_b_read_data),
            scheduler.sp_b_read_valid.eq(self.sp_b_read_valid),
        ]

        # Connect scheduler outputs to skew buffer inputs
        m.d.comb += [
            a_skew.in_valid.eq(scheduler.out_valid),
            b_skew.in_valid.eq(scheduler.out_valid),
            a_skew.flush.eq(0),  # Could connect to a reset signal
            b_skew.flush.eq(0),
        ]

        for i in range(total_rows):
            m.d.comb += getattr(a_skew, f"in_data_{i}").eq(getattr(scheduler, f"out_a_{i}"))

        for j in range(total_cols):
            m.d.comb += getattr(b_skew, f"in_data_{j}").eq(getattr(scheduler, f"out_b_{j}"))

        # Connect skew buffer outputs to array interface
        for i in range(total_rows):
            m.d.comb += [
                getattr(self, f"array_in_a_{i}").eq(getattr(a_skew, f"out_data_{i}")),
                getattr(self, f"array_in_a_valid_{i}").eq(getattr(a_skew, f"out_valid_{i}")),
            ]

        for j in range(total_cols):
            m.d.comb += [
                getattr(self, f"array_in_b_{j}").eq(getattr(b_skew, f"out_data_{j}")),
                getattr(self, f"array_in_b_valid_{j}").eq(getattr(b_skew, f"out_valid_{j}")),
            ]

        # Status outputs
        # feeding is true when any skew buffer output is valid
        any_a_valid = Signal()
        any_b_valid = Signal()

        m.d.comb += any_a_valid.eq(0)
        m.d.comb += any_b_valid.eq(0)

        for i in range(total_rows):
            m.d.comb += any_a_valid.eq(any_a_valid | getattr(a_skew, f"out_valid_{i}"))
        for j in range(total_cols):
            m.d.comb += any_b_valid.eq(any_b_valid | getattr(b_skew, f"out_valid_{j}"))

        m.d.comb += [
            self.busy.eq(scheduler.busy | any_a_valid | any_b_valid),
            self.done.eq(scheduler.done & ~any_a_valid & ~any_b_valid),
            self.feeding.eq(any_a_valid & any_b_valid),
        ]

        return m
