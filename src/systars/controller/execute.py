"""
ExecuteController - runs matrix multiplies on the systolic array.

The controller reads operands from the scratchpad (two read ports, one row per
cycle on each), drives the ``SystolicArray`` with unskewed vectors (the array
skews them itself), and writes result rows to the accumulator. ``R`` and ``C``
are the array's rows and columns (in PEs). A scratchpad row holds ``R`` A
elements (``input_bits`` each) or ``C`` B elements (``weight_bits`` each), element
0 in the low bits. An accumulator row holds ``C`` results (``acc_bits`` each).

Commands (``cmd_opcode``, ``ExecOpcode``):

- ``CONFIG``: ``rs1[0]`` selects the dataflow, 0 = output-stationary (OS),
  1 = weight-stationary (WS). A configuration with only one dataflow ignores it.
- ``COMPUTE`` / ``COMPUTE_ACCUMULATE``: one matmul. ``COMPUTE`` overwrites the
  destination rows; ``COMPUTE_ACCUMULATE`` adds to them (a bias, or the
  partial result of an earlier K block).

  - OS: ``C[R x C] = A[R x K] @ B[K x C]`` with ``K = k_dim``. ``rs1`` is the
    scratchpad address of Aᵀ (K rows; row ``k`` holds column ``k`` of A),
    ``rs2`` of B (K rows), and ``rd`` the accumulator address of C (R rows).
  - WS: ``C[M x C] = A[M x R] @ W`` with ``M = k_dim``, where ``W`` is the
    ``R x C`` matrix in the PE weights (see ``PRELOAD``). ``rs1`` is the address
    of A (M rows), ``rd`` of C (M rows). ``rs2`` is unused.

- ``PRELOAD`` (WS): loads ``W`` from the B rows at scratchpad address ``rs1``.
  ``k_dim`` rows (at most ``R``) are read; the PE rows below them get zero
  weights, so a K < R product needs no padding. In OS, ``PRELOAD`` does nothing.

Every command, including an unknown opcode, pulses ``completed`` once, after its
last accumulator write has been issued.

Timing:

- OS: K cycles of streaming, the array latency, then R cycles of draining the
  results down the columns (bottom row first). Draining shifts zeros in, so the
  array is clear for the next ``COMPUTE``.
- WS: M cycles of streaming; row ``m`` of C is written ``latency`` cycles after
  row ``m`` of A enters the array.
"""

from amaranth import Cat, Module, Mux, Signal, signed
from amaranth.lib.wiring import Component, In, Out

from ..config import Dataflow, SystolicConfig
from ..util.commands import ExecOpcode


class ExecuteController(Component):
    """
    Execute controller for the systolic array.

    Ports:
        cmd_*: Command (``valid``/``ready`` handshake; fields latched on accept)
        cfg_dataflow: Current dataflow (0 = OS, 1 = WS)
        sp_a_read_*, sp_b_read_*: The scratchpad's two read ports
        acc_write_*, acc_accumulate: Accumulator write port
        array_*: SystolicArray interface (see ``SystolicArray``)
        busy, completed, completed_id, state_debug: Status
    """

    def __init__(self, config: SystolicConfig):
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
        self.supports_os = bool(config.dataflow & Dataflow.OUTPUT_STATIONARY)
        self.supports_ws = bool(config.dataflow & Dataflow.B_STATIONARY)
        if not (self.supports_os or self.supports_ws):
            raise ValueError(f"dataflow {config.dataflow!r} is not implemented (need OS or WS)")

        ports = {
            "cmd_valid": In(1),
            "cmd_ready": Out(1),
            "cmd_opcode": In(8),
            "cmd_rs1": In(32),
            "cmd_rs2": In(32),
            "cmd_rd": In(32),
            "cmd_k_dim": In(16),
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
            "acc_write_req": Out(1),
            "acc_write_addr": Out(32),
            "acc_write_data": Out(signed(config.acc_width)),
            "acc_accumulate": Out(1),
            "busy": Out(1),
            "completed": Out(1),
            "completed_id": Out(8),
            "state_debug": Out(8),
        }
        for i in range(rows):
            ports[f"array_in_a_{i}"] = Out(signed(config.input_bits))
        for j in range(cols):
            ports[f"array_in_b_{j}"] = Out(signed(config.weight_bits))
            ports[f"array_in_d_{j}"] = Out(signed(config.acc_bits))
            ports[f"array_out_psum_{j}"] = In(signed(config.acc_bits))
            ports[f"array_out_c_{j}"] = In(signed(config.acc_bits))
        ports["array_in_valid"] = Out(1)
        ports["array_in_last"] = Out(1)
        ports["array_in_shift"] = Out(1)
        ports["array_in_dataflow"] = Out(1)
        ports["array_out_valid"] = In(1)
        ports["array_out_last"] = In(1)
        super().__init__(ports)

    # State numbers reported on ``state_debug``
    STATE_IDLE = 0
    STATE_CONFIG = 1
    STATE_DONE = 2
    STATE_OS_COMPUTE = 3
    STATE_OS_DRAIN = 4
    STATE_PRELOAD_ZERO = 5
    STATE_PRELOAD_LOAD = 6
    STATE_WS_COMPUTE = 7

    def elaborate(self, _platform):
        m = Module()
        cfg = self.config
        rows = cfg.grid_rows * cfg.tile_rows
        cols = cfg.grid_cols * cfg.tile_cols

        dataflow = Signal(init=0 if self.supports_os else 1)
        cmd_id = Signal(8)
        rs1 = Signal(32)
        rs2 = Signal(32)
        rd = Signal(32)
        count = Signal(16)  # K (OS), M (WS compute), or weight rows (PRELOAD)
        accumulate = Signal()
        issued = Signal(16)  # scratchpad reads issued
        received = Signal(16)  # scratchpad reads returned
        written = Signal(16)  # accumulator rows written (WS compute)
        zeros = Signal(range(rows + 1))  # zero weight rows still to shift in
        drain = Signal(range(rows))  # OS drain step

        def a_lane(data, i):
            return data[i * cfg.input_bits : (i + 1) * cfg.input_bits].as_signed()

        def b_lane(data, j):
            return data[j * cfg.weight_bits : (j + 1) * cfg.weight_bits].as_signed()

        def acc_row(prefix):
            return Cat(*[getattr(self, f"{prefix}_{j}") for j in range(cols)])

        def write_row(addr, data):
            m.d.comb += [
                self.acc_write_req.eq(1),
                self.acc_write_addr.eq(addr),
                self.acc_write_data.eq(data),
                self.acc_accumulate.eq(accumulate),
            ]

        def stream(port, base, addr_offset):
            """Issue one read per cycle on ``port`` until ``count`` are issued."""
            with m.If(issued < count):
                m.d.comb += [
                    getattr(self, f"sp_{port}_read_req").eq(1),
                    getattr(self, f"sp_{port}_read_addr").eq(base + addr_offset),
                ]
                m.d.sync += issued.eq(issued + 1)

        m.d.comb += [
            self.cfg_dataflow.eq(dataflow),
            self.array_in_dataflow.eq(dataflow),
        ]

        with m.FSM(init="IDLE"):
            with m.State("IDLE"):
                m.d.comb += [self.state_debug.eq(self.STATE_IDLE), self.cmd_ready.eq(1)]
                with m.If(self.cmd_valid):
                    m.d.sync += [
                        cmd_id.eq(self.cmd_id),
                        rs1.eq(self.cmd_rs1),
                        rs2.eq(self.cmd_rs2),
                        rd.eq(self.cmd_rd),
                        count.eq(self.cmd_k_dim),
                        accumulate.eq(self.cmd_opcode == ExecOpcode.COMPUTE_ACCUMULATE),
                        issued.eq(0),
                        received.eq(0),
                        written.eq(0),
                        drain.eq(0),
                    ]
                    with m.Switch(self.cmd_opcode):
                        with m.Case(ExecOpcode.CONFIG):
                            m.next = "CONFIG"
                        with m.Case(ExecOpcode.PRELOAD):
                            with m.If(dataflow):
                                weight_rows = Mux(self.cmd_k_dim > rows, rows, self.cmd_k_dim)
                                m.d.sync += [count.eq(weight_rows), zeros.eq(rows - weight_rows)]
                                m.next = "PRELOAD_ZERO"
                            with m.Else():
                                m.next = "DONE"
                        with m.Case(ExecOpcode.COMPUTE, ExecOpcode.COMPUTE_ACCUMULATE):
                            with m.If(dataflow):
                                m.next = "WS_COMPUTE"
                            with m.Else():
                                m.next = "OS_COMPUTE"
                        with m.Default():
                            m.next = "DONE"

            with m.State("CONFIG"):
                m.d.comb += self.state_debug.eq(self.STATE_CONFIG)
                if self.supports_os and self.supports_ws:
                    m.d.sync += dataflow.eq(rs1[0])
                m.next = "DONE"

            with m.State("DONE"):
                m.d.comb += [
                    self.state_debug.eq(self.STATE_DONE),
                    self.completed.eq(1),
                    self.completed_id.eq(cmd_id),
                ]
                m.next = "IDLE"

            # OS: stream Aᵀ and B rows; the array accumulates C in place
            with m.State("OS_COMPUTE"):
                m.d.comb += self.state_debug.eq(self.STATE_OS_COMPUTE)
                stream("a", rs1, issued)
                with m.If(issued < count):
                    m.d.comb += [
                        self.sp_b_read_req.eq(1),
                        self.sp_b_read_addr.eq(rs2 + issued),
                    ]
                with m.If(self.sp_a_read_valid):
                    m.d.comb += [
                        self.array_in_valid.eq(1),
                        self.array_in_last.eq(received == count - 1),
                    ]
                    for i in range(rows):
                        m.d.comb += getattr(self, f"array_in_a_{i}").eq(
                            a_lane(self.sp_a_read_data, i)
                        )
                    for j in range(cols):
                        m.d.comb += getattr(self, f"array_in_b_{j}").eq(
                            b_lane(self.sp_b_read_data, j)
                        )
                    m.d.sync += received.eq(received + 1)
                # The last product is accumulated at the end of the cycle that
                # out_last is high; with K = 0 there is nothing to accumulate
                with m.If(self.array_out_last | (count == 0)):
                    m.next = "OS_DRAIN"

            # OS: shift the results down the columns, bottom row first
            with m.State("OS_DRAIN"):
                m.d.comb += [self.state_debug.eq(self.STATE_OS_DRAIN), self.array_in_shift.eq(1)]
                write_row(rd + (rows - 1 - drain), acc_row("array_out_c"))
                m.d.sync += drain.eq(drain + 1)
                with m.If(drain == rows - 1):
                    m.next = "DONE"

            # WS: shift in zero weights for the PE rows below the loaded ones
            with m.State("PRELOAD_ZERO"):
                m.d.comb += self.state_debug.eq(self.STATE_PRELOAD_ZERO)
                with m.If(zeros != 0):
                    m.d.comb += self.array_in_shift.eq(1)
                    m.d.sync += zeros.eq(zeros - 1)
                with m.Else():
                    m.next = "PRELOAD_LOAD"

            # WS: shift in the weight rows, last row first
            with m.State("PRELOAD_LOAD"):
                m.d.comb += self.state_debug.eq(self.STATE_PRELOAD_LOAD)
                stream("b", rs1, count - 1 - issued)
                with m.If(self.sp_b_read_valid):
                    m.d.comb += self.array_in_shift.eq(1)
                    for j in range(cols):
                        m.d.comb += getattr(self, f"array_in_d_{j}").eq(
                            b_lane(self.sp_b_read_data, j)
                        )
                    m.d.sync += received.eq(received + 1)
                    with m.If(received == count - 1):
                        m.next = "DONE"
                with m.If(count == 0):
                    m.next = "DONE"

            # WS: stream A rows; each leaves the array as a row of C
            with m.State("WS_COMPUTE"):
                m.d.comb += self.state_debug.eq(self.STATE_WS_COMPUTE)
                stream("a", rs1, issued)
                with m.If(self.sp_a_read_valid):
                    m.d.comb += [
                        self.array_in_valid.eq(1),
                        self.array_in_last.eq(received == count - 1),
                    ]
                    for i in range(rows):
                        m.d.comb += getattr(self, f"array_in_a_{i}").eq(
                            a_lane(self.sp_a_read_data, i)
                        )
                    m.d.sync += received.eq(received + 1)
                with m.If(self.array_out_valid):
                    write_row(rd + written, acc_row("array_out_psum"))
                    m.d.sync += written.eq(written + 1)
                    with m.If(self.array_out_last):
                        m.next = "DONE"
                with m.If(count == 0):
                    m.next = "DONE"

        m.d.comb += self.busy.eq(~self.cmd_ready)
        return m
