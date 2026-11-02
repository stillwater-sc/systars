"""
Matmul ISA instruction: C = A @ B

A hardware FSM that tiles a matrix multiply larger than the systolic array and
issues the LOAD / CONFIG / PRELOAD / COMPUTE / STORE commands for it, on the
command port ``SystolicTop`` takes (see ``isa.command``). ``connect_to_top``
wires it straight to a tile.

Dataflow: weight-stationary. Row-major tiles load one matrix row per scratchpad
row, which is the layout WS consumes: B tiles become the PE weights (PRELOAD)
and A tiles stream through them (COMPUTE).

For each output tile (i, j), and each K tile k:

- LOAD the A tile (``A[i*dim:, k*dim:]``) and the B tile (``B[k*dim:, j*dim:]``),
  one command per tile: the BlockMover moves its rows from any byte address,
  ``stride`` bytes apart, with ``row_bytes`` valid bytes (an edge tile's rows are
  zero-filled past them);
- PRELOAD the B tile as weights, and COMPUTE (the first K tile overwrites the
  accumulator rows, later ones accumulate);

then STORE the C tile, one command: its rows are written with byte strobes, so an
edge N tile writes only its own columns and C can be dense.

So A, B and C may be dense row-major at any address, with any M, N and K. Errors
(``error_code``): UNSUPPORTED_LAYOUT for a zero stride, or a C stride below N
results (rows would overlap). Bias (``CONFIG_D_ADDR``) and accumulating onto an
existing C (``CONFIG_OPTIONS`` bit 0) both need a load into the accumulator, which
the tile does not have yet: UNSUPPORTED_BIAS and UNSUPPORTED_ACCUMULATE.

Usage: configure dimensions, addresses and strides, then issue START.
"""

from enum import IntEnum

from amaranth import Module, Mux, Signal
from amaranth.lib.wiring import Component, In, Out

from ..config import Dataflow, SystolicConfig
from ..memory.local_addr import LocalAddr
from ..util.commands import DmaOpcode, ExecOpcode
from .command import CTRL_EXEC, CTRL_LOAD, CTRL_STORE, CommandIssuer, command_ports


class MatmulCmd(IntEnum):
    """Configuration commands for the Matmul instruction."""

    CONFIG_DIMS = 0x01  # data[15:0]=M, data[31:16]=N, data[47:32]=K
    CONFIG_A_ADDR = 0x10
    CONFIG_B_ADDR = 0x11
    CONFIG_C_ADDR = 0x12
    CONFIG_D_ADDR = 0x13  # Bias address, 0 = no bias (bias is not supported)
    CONFIG_A_STRIDE = 0x20  # Row strides in bytes
    CONFIG_B_STRIDE = 0x21
    CONFIG_C_STRIDE = 0x22
    CONFIG_D_STRIDE = 0x23
    CONFIG_OPTIONS = 0x30  # data[0]=accumulate, data[3:1]=activation
    START = 0xF0
    ABORT = 0xFF


class MatmulError(IntEnum):
    """Error codes for the Matmul instruction."""

    NONE = 0x00
    INVALID_DIMS = 0x01  # M, N or K is zero
    CONFIG_INCOMPLETE = 0x04  # START without dimensions and addresses
    UNSUPPORTED_LAYOUT = 0x06  # A zero stride, or C rows that would overlap
    UNSUPPORTED_BIAS = 0x07  # A bias (D) was configured
    UNSUPPORTED_ACCUMULATE = 0x08  # Accumulate onto C (C is not loaded into the accumulator)


WS = 1  # ExecuteController CONFIG rs1[0]


class Matmul(Component):
    """
    Matmul ISA instruction: C = A @ B.

    Ports:
        cfg_valid, cfg_ready, cfg_cmd, cfg_data: Configuration (``MatmulCmd``)
        cmd_*: Command port (``isa.command``)
        busy, done, error, error_code: Status
        progress_i, progress_j, progress_k: Current tile
        selected_dataflow: 1 (weight-stationary)
        param_M, param_N, param_K: Configured dimensions
        dbg_addr_A/B/C/D, dbg_tile_M/N/K: Current tile addresses and tile sizes
    """

    def __init__(self, config: SystolicConfig):
        self.config = config
        rows = config.grid_rows * config.tile_rows
        cols = config.grid_cols * config.tile_cols
        if rows != cols:
            raise ValueError(f"Matmul requires a square array, got {rows}x{cols}")
        self.dim = rows
        if self.dim & (self.dim - 1):
            # Tile offsets and counts are computed with shifts by log2(dim)
            raise ValueError(f"Matmul requires a power-of-two array dimension, got {self.dim}")
        if not config.dataflow & Dataflow.B_STATIONARY:
            raise ValueError("Matmul issues weight-stationary commands: the array must support WS")
        if config.sp_total_rows < 2 * self.dim or config.acc_total_rows < self.dim:
            raise ValueError("Matmul needs 2 * dim scratchpad rows and dim accumulator rows")

        addr = LocalAddr(config)
        # Scratchpad: A tile rows [0, dim), B tile rows [dim, 2*dim); accumulator: C rows [0, dim)
        self.sp_a = addr.make_sp_addr(0, 0)
        self.sp_b = addr.make_sp_addr(self.dim % config.sp_banks, self.dim // config.sp_banks)
        self.acc_c = addr.make_acc_addr(0, 0)

        ports = {
            "cfg_valid": In(1),
            "cfg_ready": Out(1),
            "cfg_cmd": In(8),
            "cfg_data": In(64),
            **command_ports(),
            "busy": Out(1),
            "done": Out(1),
            "error": Out(1),
            "error_code": Out(8),
            "progress_i": Out(16),
            "progress_j": Out(16),
            "progress_k": Out(16),
            "selected_dataflow": Out(2),
            "param_M": Out(32),
            "param_N": Out(32),
            "param_K": Out(32),
            "dbg_addr_A": Out(64),
            "dbg_addr_B": Out(64),
            "dbg_addr_C": Out(64),
            "dbg_addr_D": Out(64),
            "dbg_tile_M": Out(16),
            "dbg_tile_N": Out(16),
            "dbg_tile_K": Out(16),
        }
        super().__init__(ports)

    def elaborate(self, _platform):
        m = Module()
        cfg = self.config
        dim = self.dim
        dim_shift = (dim - 1).bit_length()
        in_bytes = cfg.input_bits // 8
        acc_bytes = cfg.acc_bits // 8
        issuer = CommandIssuer(m, self)

        reg_M, reg_N, reg_K = Signal(32), Signal(32), Signal(32)
        reg_A, reg_B, reg_C, reg_D = (Signal(64, name=f"reg_{n}_addr") for n in "ABCD")
        a_stride, b_stride, c_stride = (Signal(32, name=f"reg_{n}_stride") for n in "ABC")
        reg_accumulate = Signal()
        reg_activation = Signal(3)
        dims_valid = Signal()
        addrs_valid = Signal()

        tiles_M, tiles_N, tiles_K = Signal(16), Signal(16), Signal(16)
        tile_i, tile_j, tile_k = Signal(16), Signal(16), Signal(16)

        # Tile sizes (edge tiles are smaller)
        tile_M, tile_N, tile_K = Signal(16), Signal(16), Signal(16)
        for size, total, index in (
            (tile_M, reg_M, tile_i),
            (tile_N, reg_N, tile_j),
            (tile_K, reg_K, tile_k),
        ):
            remaining = Signal(32)
            m.d.comb += [
                remaining.eq(total - (index << dim_shift)),
                size.eq(Mux(remaining < dim, remaining[:16], dim)),
            ]

        # Tile addresses (row 0 of the tile)
        a_tile = Signal(64)
        b_tile = Signal(64)
        c_tile = Signal(64)
        m.d.comb += [
            a_tile.eq(reg_A + (tile_i << dim_shift) * a_stride + (tile_k << dim_shift) * in_bytes),
            b_tile.eq(reg_B + (tile_k << dim_shift) * b_stride + (tile_j << dim_shift) * in_bytes),
            c_tile.eq(reg_C + (tile_i << dim_shift) * c_stride + (tile_j << dim_shift) * acc_bytes),
        ]

        # Layout check: rows a stride apart (0 would mean one bus beat to the
        # BlockMover), and C rows that do not overlap
        layout_ok = Signal()
        m.d.comb += layout_ok.eq(
            (a_stride != 0) & (b_stride != 0) & (c_stride >= reg_N * acc_bytes)
        )

        m.d.comb += [
            self.param_M.eq(reg_M),
            self.param_N.eq(reg_N),
            self.param_K.eq(reg_K),
            self.selected_dataflow.eq(WS),
            self.progress_i.eq(tile_i),
            self.progress_j.eq(tile_j),
            self.progress_k.eq(tile_k),
            self.dbg_addr_A.eq(a_tile),
            self.dbg_addr_B.eq(b_tile),
            self.dbg_addr_C.eq(c_tile),
            self.dbg_addr_D.eq(reg_D),
            self.dbg_tile_M.eq(tile_M),
            self.dbg_tile_N.eq(tile_N),
            self.dbg_tile_K.eq(tile_K),
        ]

        with m.FSM(init="IDLE") as fsm:
            with m.State("IDLE"):
                m.d.comb += self.cfg_ready.eq(1)
                with m.If(self.cfg_valid), m.Switch(self.cfg_cmd):
                    with m.Case(MatmulCmd.CONFIG_DIMS):
                        m.d.sync += [
                            reg_M.eq(self.cfg_data[0:16]),
                            reg_N.eq(self.cfg_data[16:32]),
                            reg_K.eq(self.cfg_data[32:48]),
                            dims_valid.eq(1),
                        ]
                    with m.Case(MatmulCmd.CONFIG_A_ADDR):
                        m.d.sync += reg_A.eq(self.cfg_data)
                    with m.Case(MatmulCmd.CONFIG_B_ADDR):
                        m.d.sync += reg_B.eq(self.cfg_data)
                    with m.Case(MatmulCmd.CONFIG_C_ADDR):
                        m.d.sync += [reg_C.eq(self.cfg_data), addrs_valid.eq(1)]
                    with m.Case(MatmulCmd.CONFIG_D_ADDR):
                        m.d.sync += reg_D.eq(self.cfg_data)
                    with m.Case(MatmulCmd.CONFIG_A_STRIDE):
                        m.d.sync += a_stride.eq(self.cfg_data[0:32])
                    with m.Case(MatmulCmd.CONFIG_B_STRIDE):
                        m.d.sync += b_stride.eq(self.cfg_data[0:32])
                    with m.Case(MatmulCmd.CONFIG_C_STRIDE):
                        m.d.sync += c_stride.eq(self.cfg_data[0:32])
                    with m.Case(MatmulCmd.CONFIG_OPTIONS):
                        m.d.sync += [
                            reg_accumulate.eq(self.cfg_data[0]),
                            reg_activation.eq(self.cfg_data[1:4]),
                        ]
                    with m.Case(MatmulCmd.START):
                        with m.If(dims_valid & addrs_valid):
                            m.d.sync += [
                                tiles_M.eq((reg_M + dim - 1) >> dim_shift),
                                tiles_N.eq((reg_N + dim - 1) >> dim_shift),
                                tiles_K.eq((reg_K + dim - 1) >> dim_shift),
                                tile_i.eq(0),
                                tile_j.eq(0),
                                tile_k.eq(0),
                                *issuer.reset(),
                            ]
                            m.next = "VALIDATE"
                        with m.Else():
                            m.d.sync += self.error_code.eq(MatmulError.CONFIG_INCOMPLETE)
                            m.next = "ERROR"

            with m.State("VALIDATE"):
                with m.If((reg_M == 0) | (reg_N == 0) | (reg_K == 0)):
                    m.d.sync += self.error_code.eq(MatmulError.INVALID_DIMS)
                    m.next = "ERROR"
                with m.Elif(reg_D != 0):
                    m.d.sync += self.error_code.eq(MatmulError.UNSUPPORTED_BIAS)
                    m.next = "ERROR"
                with m.Elif(reg_accumulate):
                    m.d.sync += self.error_code.eq(MatmulError.UNSUPPORTED_ACCUMULATE)
                    m.next = "ERROR"
                with m.Elif(~layout_ok):
                    m.d.sync += self.error_code.eq(MatmulError.UNSUPPORTED_LAYOUT)
                    m.next = "ERROR"
                with m.Else():
                    m.d.sync += self.error_code.eq(MatmulError.NONE)
                    m.next = "CONFIG"

            issuer.state("CONFIG", "LOAD_A", CTRL_EXEC, ExecOpcode.CONFIG, rs1=WS)

            # One LOAD per tile: the A tile, then the B tile
            issuer.state(
                "LOAD_A", "LOAD_B", CTRL_LOAD, DmaOpcode.MEMCPY,
                dram_addr=a_tile, local_addr=self.sp_a, len=tile_M,
                stride=a_stride, row_bytes=tile_K * in_bytes,
            )  # fmt: skip
            issuer.state(
                "LOAD_B", "PRELOAD", CTRL_LOAD, DmaOpcode.MEMCPY,
                dram_addr=b_tile, local_addr=self.sp_b, len=tile_K,
                stride=b_stride, row_bytes=tile_N * in_bytes,
            )  # fmt: skip

            issuer.state(
                "PRELOAD", "COMPUTE", CTRL_EXEC, ExecOpcode.PRELOAD, rs1=self.sp_b, k_dim=tile_K
            )
            issuer.state(
                "COMPUTE", "NEXT_K", CTRL_EXEC,
                Mux(tile_k == 0, ExecOpcode.COMPUTE, ExecOpcode.COMPUTE_ACCUMULATE),
                rs1=self.sp_a, rd=self.acc_c, k_dim=tile_M,
            )  # fmt: skip

            with m.State("NEXT_K"):
                with m.If(tile_k + 1 < tiles_K):
                    m.d.sync += tile_k.eq(tile_k + 1)
                    m.next = "LOAD_A"
                with m.Else():
                    m.d.sync += tile_k.eq(0)
                    m.next = "STORE"

            # One STORE per C tile, only its own columns
            issuer.state(
                "STORE", "NEXT_IJ", CTRL_STORE, DmaOpcode.MEMCPY,
                dram_addr=c_tile, local_addr=self.acc_c, len=tile_M,
                stride=c_stride, row_bytes=tile_N * acc_bytes, activation=reg_activation,
            )  # fmt: skip

            with m.State("NEXT_IJ"):
                with m.If(tile_j + 1 < tiles_N):
                    m.d.sync += tile_j.eq(tile_j + 1)
                    m.next = "LOAD_A"
                with m.Elif(tile_i + 1 < tiles_M):
                    m.d.sync += [tile_i.eq(tile_i + 1), tile_j.eq(0)]
                    m.next = "LOAD_A"
                with m.Else():
                    m.next = "DONE"

            with m.State("DONE"):
                m.d.comb += self.done.eq(1)
                m.d.sync += [dims_valid.eq(0), addrs_valid.eq(0)]
                m.next = "IDLE"

            with m.State("ERROR"):
                m.d.comb += self.error.eq(1)
                m.next = "IDLE"

        m.d.comb += self.busy.eq(~fsm.ongoing("IDLE"))
        return m
