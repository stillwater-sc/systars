"""
Unit tests for the Conv2D ISA instruction (an implicit GEMM).

The command stream Conv2d issues is run through a Python model of SystolicTop
(``_isa.TopModel``) and the output compared with a numpy NHWC convolution: this
checks the address arithmetic (output-position decoding, kernel offsets, stride,
padding, dilation, channel chunks and N tiles) on dense NHWC tensors at any
address, the layouts the tile's BlockMover moves. Conv2d also drives a real
SystolicTop through a byte-addressed AXI memory model.
"""

import numpy as np
import pytest
from amaranth import Elaboratable, Module

from systars.config import Dataflow, SystolicConfig
from systars.controller.sync import Sync
from systars.isa.command import CTRL_EXEC, CTRL_LOAD, CTRL_STORE, connect_to_top
from systars.isa.conv2d import Conv2d, Conv2dCmd, Conv2dError
from systars.top import SystolicTop
from systars.util.commands import DmaOpcode, ExecOpcode

from ._axi import axi_memory
from ._isa import TopModel, from_bytes, run_isa, to_unsigned_bytes
from ._sim import skip_without_yosys

X_ADDR, F_ADDR, Y_ADDR, B_ADDR = 0x1000, 0x20000, 0x40000, 0xF0000
DIM = 4


def config(**overrides):
    fields = {"grid_rows": DIM, "grid_cols": DIM, "sp_capacity_kb": 4, "acc_capacity_kb": 4}
    fields.update(overrides)
    return SystolicConfig(**fields)


def x_and_f_loads(cmds):
    """The X row LOADs (one row each: MEMCPY, or CLEAR in the padding) and the filter
    LOADs (corner-turned). LOADs into the accumulator (bias, Y) are neither."""
    loads = [c for c in cmds if c.type == CTRL_LOAD and not c.local_addr >> 31]
    x = [c for c in loads if c.opcode in (DmaOpcode.MEMCPY, DmaOpcode.CLEAR)]
    return x, [c for c in loads if c.opcode == DmaOpcode.TRANSPOSE]


def out_size(size, kernel, stride, pad, dilation):
    return (size + 2 * pad - dilation * (kernel - 1) - 1) // stride + 1


def reference(X, F, stride, pad, dilation, bias=None, Y0=None):
    batch, in_h, in_w, _ = X.shape
    kh_n, kw_n, _, out_c = F.shape
    oh_n = out_size(in_h, kh_n, stride[0], pad[0], dilation[0])
    ow_n = out_size(in_w, kw_n, stride[1], pad[1], dilation[1])
    Y = np.zeros((batch, oh_n, ow_n, out_c), dtype=np.int64)
    for b in range(batch):
        for oh in range(oh_n):
            for ow in range(ow_n):
                for kh in range(kh_n):
                    for kw in range(kw_n):
                        ih = oh * stride[0] + kh * dilation[0] - pad[0]
                        iw = ow * stride[1] + kw * dilation[1] - pad[1]
                        if 0 <= ih < in_h and 0 <= iw < in_w:
                            Y[b, oh, ow] += X[b, ih, iw] @ F[kh, kw]
    if bias is not None:
        Y += bias
    if Y0 is not None:
        Y += Y0
    return Y


def setup(
    X_shape,
    F_shape,
    stride=(1, 1),
    pad=(0, 0),
    dilation=(1, 1),
    *,
    y_stride=0,
    bias=0,
    options=0,
):
    batch, in_h, in_w, in_c = X_shape
    kh, kw, _, out_c = F_shape
    oh = out_size(in_h, kh, stride[0], pad[0], dilation[0])
    ow = out_size(in_w, kw, stride[1], pad[1], dilation[1])
    cmds = [
        (Conv2dCmd.CONFIG_INPUT_DIMS, batch | in_h << 8 | in_w << 24 | in_c << 40),
        (Conv2dCmd.CONFIG_OUTPUT_DIMS, oh | ow << 16 | out_c << 32),
        (Conv2dCmd.CONFIG_KERNEL_DIMS, kh | kw << 8),
        (Conv2dCmd.CONFIG_STRIDE, stride[0] | stride[1] << 8),
        (Conv2dCmd.CONFIG_PADDING, pad[0] | pad[1] << 8),
        (Conv2dCmd.CONFIG_DILATION, dilation[0] | dilation[1] << 8),
        (Conv2dCmd.CONFIG_X_ADDR, X_ADDR),
        (Conv2dCmd.CONFIG_F_ADDR, F_ADDR),
        (Conv2dCmd.CONFIG_Y_ADDR, Y_ADDR),
        (Conv2dCmd.CONFIG_B_ADDR, bias),
        (Conv2dCmd.CONFIG_Y_STRIDE, y_stride),
        (Conv2dCmd.CONFIG_OPTIONS, options),
    ]
    return cmds


def operands(rng, X_shape, F_shape, stride, pad, dilation, *, bias=False, accumulate=False):
    """X, F, the bias (``out_c`` results), the existing Y (by position), and the expected Y."""
    X = rng.integers(-128, 128, X_shape)
    F = rng.integers(-128, 128, F_shape)
    out_c = F_shape[3]
    b = rng.integers(-5000, 5000, out_c) if bias else None
    shape = reference(X, F, stride, pad, dilation).shape
    Y0 = rng.integers(-5000, 5000, shape) if accumulate else None
    return X, F, b, Y0, reference(X, F, stride, pad, dilation, b, Y0)


def check(
    X_shape, F_shape, stride=(1, 1), pad=(0, 0), dilation=(1, 1), seed=0, y_stride=0,
    bias=False, accumulate=False,
):  # fmt: skip
    """Dense X, F and (with ``y_stride`` 0) dense Y; optionally a bias, and Y +=."""
    rng = np.random.default_rng(seed)
    X, F, b, Y0, expected = operands(
        rng, X_shape, F_shape, stride, pad, dilation, bias=bias, accumulate=accumulate
    )
    out_c = F_shape[3]
    row_stride = y_stride or out_c * 4
    positions = expected.shape[0] * expected.shape[1] * expected.shape[2]

    cfg = config()
    isa = Conv2d(cfg)
    cmds = setup(
        X_shape, F_shape, stride, pad, dilation, y_stride=y_stride,
        bias=B_ADDR if bias else 0, options=int(accumulate),
    )  # fmt: skip
    result = run_isa(isa, cmds, Conv2dCmd.START, latency=0)
    assert result.done and not result.error, f"error {result.error_code:#x}"

    model = TopModel(cfg)
    model.write(X_ADDR, to_unsigned_bytes(X, 8))
    model.write(F_ADDR, to_unsigned_bytes(F, 8))
    if b is not None:
        model.write(B_ADDR, to_unsigned_bytes(b, 32))
    if Y0 is not None:
        for p, row in enumerate(Y0.reshape(positions, out_c)):
            model.write(Y_ADDR + p * row_stride, to_unsigned_bytes(row, 32))
    model.run(result.commands)  # also checks the credits and tokens
    model.assert_all_free()
    got = np.array([model.read_elems(Y_ADDR + p * row_stride, 32, out_c) for p in range(positions)])
    np.testing.assert_array_equal(got, expected.reshape(positions, out_c))
    return result


class TestNumerical:
    def test_pointwise(self):
        check((1, 3, 3, 3), (1, 1, 3, 4))

    def test_3x3_same_padding_two_channel_chunks_two_n_tiles_two_batches(self):
        """M tiles cross output rows and batches; K walks 9 offsets x 2 channel chunks."""
        check((2, 5, 5, 6), (3, 3, 6, 6), pad=(1, 1), seed=1)

    def test_stride_dilation_rectangular_kernel(self):
        check((1, 9, 7, 2), (2, 3, 2, 3), stride=(2, 1), pad=(2, 1), dilation=(2, 2), seed=2)

    def test_stride_two_no_padding(self):
        check((1, 8, 8, 4), (3, 3, 4, 2), stride=(2, 2), seed=3)

    def test_kernel_larger_than_padding_reach(self):
        """Output positions whose whole window is padding read only zero rows."""
        check((1, 2, 2, 1), (3, 3, 1, 1), pad=(3, 3), seed=4)

    def test_padded_output_rows(self):
        """Y rows wider than out_c (y_stride): the bytes between them are not written."""
        check((1, 4, 4, 3), (3, 3, 3, 5), pad=(1, 1), seed=5, y_stride=24)

    def test_bias_padding_stride_dilation(self):
        """Y = conv2d(X, F) + bias: the bias row is broadcast to every output position."""
        check((2, 7, 6, 5), (3, 2, 5, 6), stride=(2, 1), pad=(2, 1), dilation=(2, 1), seed=6,
              bias=True)  # fmt: skip

    def test_bias_two_n_tiles(self):
        check((1, 4, 4, 3), (3, 3, 3, 7), pad=(1, 1), seed=7, bias=True)

    @pytest.mark.parametrize("bias", [False, True])
    def test_accumulate_onto_y(self, bias):
        """Y += conv2d(X, F) (+ bias): the existing Y tile is loaded too."""
        check((1, 4, 5, 3), (3, 3, 3, 5), pad=(1, 1), seed=8, y_stride=24, bias=bias,
              accumulate=True)  # fmt: skip


class TestCommandStream:
    def commands(self, **kw):
        X_shape, F_shape = (1, 4, 4, 2), (3, 3, 2, 3)
        cfg = config()
        cmds = setup(X_shape, F_shape, pad=(1, 1), **kw)
        return run_isa(Conv2d(cfg), cmds, Conv2dCmd.START, latency=0).commands

    def test_structure(self):
        cmds = self.commands()
        assert (cmds[0].type, cmds[0].opcode, cmds[0].rs1) == (CTRL_EXEC, ExecOpcode.CONFIG, 0)
        execs = [c for c in cmds[1:] if c.type == CTRL_EXEC]
        # 16 positions = 4 M tiles, 1 N tile, 9 K tiles (3x3 kernel, 1 channel chunk):
        # OS, one COMPUTE per K tile, the 9 chained into one reduction per output tile
        assert [c.opcode for c in execs] == [ExecOpcode.COMPUTE] * 9 * 4
        assert [c.chain for c in execs] == ([0b10] + [0b11] * 7 + [0b01]) * 4
        assert {c.k_dim for c in execs} == {2}  # the K tile: 2 input channels
        # One STORE per output tile (4 positions, 3 channels, dense)
        stores = [c for c in cmds if c.type == CTRL_STORE]
        assert [(c.dram_addr, c.len, c.stride, c.row_bytes) for c in stores] == [
            (Y_ADDR + m * 4 * 12, 4, 12, 12) for m in range(4)
        ]
        # Per K tile: one LOAD per output position (len 1, its 2 channels), one per filter tile
        # The filter tile is corner-turned into L2 (column-major, as OS streams it)
        x_loads, f_loads = x_and_f_loads(cmds)
        assert {(c.opcode, c.len, c.row_bytes) for c in x_loads} == {
            (DmaOpcode.MEMCPY, 1, 2),
            (DmaOpcode.CLEAR, 1, 2),  # CLEAR: in the padding
        }
        assert {(c.opcode, c.len, c.stride, c.row_bytes) for c in f_loads} == {
            (DmaOpcode.TRANSPOSE, 2, 3, 3)
        }
        assert len(f_loads) == 9 * 4

    def test_x_rows_and_filter_fill_one_slot(self):
        """A K tile's X rows and its filter tile share the pair's A slot (one COMPUTE
        reads both): the first X LOAD takes its credit, the filter LOAD emits its token;
        the slot pairs alternate between K tiles."""
        x_loads, f_loads = x_and_f_loads(self.commands())
        assert [(c.sync >> Sync.DST) & 0xF for c in f_loads[:4]] == [0, 1, 0, 1]
        assert all(c.sync & Sync.SIGNAL and not c.sync & Sync.ACQUIRE for c in f_loads)
        tile = x_loads[:4]  # the first K tile's 4 positions
        assert [(c.sync >> Sync.DST) & 0xF for c in tile] == [0] * 4
        assert [bool(c.sync & Sync.ACQUIRE) for c in tile] == [True, False, False, False]
        assert not any(c.sync & Sync.SIGNAL for c in tile)
        assert [c.local_addr for c in x_loads[4:8]] == [DIM, DIM + 1, DIM + 2, DIM + 3]

    def test_padding_rows_are_cleared(self):
        """A position in the padding is a CLEAR: a zero L2 row with no L3 read."""
        cmds = self.commands()
        x_loads, _ = x_and_f_loads(cmds)
        cleared = [c for c in x_loads if c.opcode == DmaOpcode.CLEAR]
        # Kernel offset (0, 0) of the first M tile (positions (0,0)..(0,3)) is all padding
        assert all(c.opcode == DmaOpcode.CLEAR for c in x_loads[:4])
        assert cleared and len(cleared) < len(x_loads)
        assert {(c.len, c.local_addr >> 31) for c in cleared} == {(1, 0)}

    def test_bias_and_y_loads(self):
        """Each output tile first broadcasts its bias channels into the accumulator slot
        (taking its credit), then adds the existing Y tile; every COMPUTE accumulates and
        none takes the accumulator credit."""
        cmds = self.commands(bias=B_ADDR, options=1)
        acc_loads = [c for c in cmds if c.type == CTRL_LOAD and c.local_addr >> 31]
        acc_flag, add = 1 << 31, 1 << 30
        assert [(c.opcode, c.dram_addr, c.local_addr, c.len, c.row_bytes, c.sync)
                for c in acc_loads[:4]] == [
            (DmaOpcode.BROADCAST, B_ADDR, acc_flag, 4, 12, Sync.ACQUIRE),
            (DmaOpcode.MEMCPY, Y_ADDR, acc_flag | add, 4, 12, 0),
            (DmaOpcode.BROADCAST, B_ADDR, acc_flag | DIM, 4, 12, 1 << Sync.DST | Sync.ACQUIRE),
            (DmaOpcode.MEMCPY, Y_ADDR + 48, acc_flag | add | DIM, 4, 12, 1 << Sync.DST),
        ]  # fmt: skip
        computes = [c for c in cmds if c.type == CTRL_EXEC][1:]
        assert {c.opcode for c in computes} == {ExecOpcode.COMPUTE_ACCUMULATE}
        assert not any(c.sync & Sync.ACQUIRE for c in computes)


class TestErrors:
    def run(self, cmds):
        return run_isa(Conv2d(config()), cmds, Conv2dCmd.START)

    def test_start_without_config(self):
        assert self.run([]).error_code == Conv2dError.CONFIG_INCOMPLETE

    def test_padding_needs_no_zero_row(self):
        """CONFIG_ZERO_ADDR is no longer needed (and is ignored)."""
        result = self.run(setup((1, 4, 4, 4), (3, 3, 4, 4), pad=(1, 1)))
        assert result.done and not result.error

    def test_zero_stride(self):
        cmds = setup((1, 4, 4, 4), (1, 1, 4, 4))
        cmds.append((Conv2dCmd.CONFIG_STRIDE, 0))
        assert self.run(cmds).error_code == Conv2dError.INVALID_DIMS

    def test_overlapping_output_rows(self):
        """y_stride below out_c results: output rows would overlap."""
        cmds = setup((1, 4, 4, 4), (1, 1, 4, 5), y_stride=16)
        result = self.run(cmds)
        assert result.error_code == Conv2dError.UNSUPPORTED_LAYOUT
        assert not result.commands

    def test_real_layouts_are_accepted(self):
        """Dense NHWC with odd channel counts: no alignment requirement any more."""
        cmds = setup((1, 5, 5, 3), (3, 3, 3, 5), pad=(1, 1))
        result = self.run(cmds)
        assert result.done and not result.error


class TestConstruction:
    @pytest.mark.parametrize(
        "dataflow", [Dataflow.B_STATIONARY, Dataflow.A_STATIONARY | Dataflow.B_STATIONARY]
    )
    def test_requires_output_stationary(self, dataflow):
        """Per-position LOADs write A row-major, which only OS streams."""
        with pytest.raises(ValueError, match="output-stationary"):
            Conv2d(config(dataflow=dataflow))

    def test_requires_square_power_of_two(self):
        with pytest.raises(ValueError, match="square"):
            Conv2d(config(grid_cols=2))
        with pytest.raises(ValueError, match="power-of-two"):
            Conv2d(config(grid_rows=3, grid_cols=3))

    def test_verilog(self):
        skip_without_yosys()
        from amaranth.back import verilog

        assert "module Conv2d" in verilog.convert(Conv2d(config()), name="Conv2d")


def test_m_tile_count_does_not_truncate():
    """batch 8 x 224 x 224 = 401,408 positions = 100,352 M tiles: wider than 16 bits."""
    from ._sim import run_sim

    isa = Conv2d(config())
    cmds = setup((8, 224, 224, 4), (1, 1, 4, 4))
    seen = {}

    async def testbench(ctx):
        for cmd, data in [*cmds, (Conv2dCmd.START, 0)]:
            ctx.set(isa.cfg_valid, 1)
            ctx.set(isa.cfg_cmd, cmd)
            ctx.set(isa.cfg_data, data)
            await ctx.tick()
        ctx.set(isa.cfg_valid, 0)
        # The first command (CONFIG) is issued after the tile counts are computed
        await ctx.tick().until(isa.cmd_valid)
        seen.update(
            M=ctx.get(isa.param_M), tiles=ctx.get(isa.dbg_tiles_M), error=ctx.get(isa.error)
        )

    run_sim(isa, testbench, max_cycles=200)
    assert seen == {"M": 8 * 224 * 224, "tiles": 8 * 224 * 224 // DIM, "error": 0}


class Conv2dOnTop(Elaboratable):
    def __init__(self, cfg):
        self.top = SystolicTop(cfg)
        self.isa = Conv2d(cfg)

    def elaborate(self, _platform):
        m = Module()
        m.submodules.top = self.top
        m.submodules.isa = self.isa
        connect_to_top(m, self.isa, self.top)
        return m


@pytest.mark.parametrize(
    "X_shape,F_shape,stride,pad,dilation,bias,accumulate",
    [
        ((1, 4, 4, 3), (3, 3, 3, 5), (1, 1), (1, 1), (1, 1), False, False),
        ((2, 5, 4, 6), (2, 3, 6, 2), (2, 1), (1, 0), (1, 1), False, False),
        ((1, 6, 5, 3), (2, 2, 3, 6), (1, 2), (1, 1), (2, 1), True, False),
        ((1, 4, 4, 5), (3, 3, 5, 3), (1, 1), (1, 1), (1, 1), True, True),
    ],
    ids=["pad", "stride", "bias-dilation", "bias-accumulate"],
)
def test_conv2d_drives_systolic_top(X_shape, F_shape, stride, pad, dilation, bias, accumulate):
    """Conv2d on a real SystolicTop: dense NHWC X, F and Y with odd channel counts,
    padding, stride, dilation and bias. A padded position reads nothing from L3."""
    from ._sim import run_sim

    cfg = config()
    beat = cfg.dma_buswidth // 8
    rng = np.random.default_rng(sum(X_shape) + sum(F_shape))
    X, F, b, Y0, expected = operands(
        rng, X_shape, F_shape, stride, pad, dilation, bias=bias, accumulate=accumulate
    )
    memory = bytearray(rng.integers(0, 256, 1 << 20, dtype=np.uint8).tobytes())
    memory[X_ADDR : X_ADDR + X.size] = to_unsigned_bytes(X, 8)
    memory[F_ADDR : F_ADDR + F.size] = to_unsigned_bytes(F, 8)
    if b is not None:
        memory[B_ADDR : B_ADDR + b.size * 4] = to_unsigned_bytes(b, 32)
    y_bytes = expected.size * 4
    if Y0 is not None:
        memory[Y_ADDR : Y_ADDR + y_bytes] = to_unsigned_bytes(Y0, 32)
    before = bytes(memory)

    dut = Conv2dOnTop(cfg)
    isa, top = dut.isa, dut.top
    status = {}
    cmds = setup(
        X_shape, F_shape, stride, pad, dilation, bias=B_ADDR if bias else 0,
        options=int(accumulate),
    )  # fmt: skip

    async def testbench(ctx):
        for cmd, data in [*cmds, (Conv2dCmd.START, 0)]:
            ctx.set(isa.cfg_valid, 1)
            ctx.set(isa.cfg_cmd, cmd)
            ctx.set(isa.cfg_data, data)
            await ctx.tick()
        ctx.set(isa.cfg_valid, 0)
        done, error = await ctx.tick().sample(isa.done, isa.error).until(isa.done | isa.error)
        # Both are sticky: a schedule or protocol violation anywhere in the run stays set
        status.update(
            done=done, error=error, l1=ctx.get(top.l1_error), sync=ctx.get(top.sync_error)
        )

    in_bounds = 0  # X row LOADs that read L3 (MEMCPY, not CLEAR), as the ISA issues them

    async def count_x_loads(ctx):
        nonlocal in_bounds
        ports = (isa.cmd_valid, isa.cmd_ready, isa.cmd_type, isa.cmd_opcode, isa.cmd_local_addr)
        while True:
            valid, ready, ctype, opcode, local = (await ctx.tick().sample(*ports))[2:]
            if valid and ready and ctype == CTRL_LOAD and not local >> 31:
                in_bounds += opcode == DmaOpcode.MEMCPY

    bursts: list = []
    run_sim(
        dut,
        testbench,
        processes=[axi_memory(top, memory, beat, log=bursts), count_x_loads],
        max_cycles=500_000,
    )
    assert status == {"done": 1, "error": 0, "l1": 0, "sync": 0}
    Y = np.array(from_bytes(memory[Y_ADDR : Y_ADDR + y_bytes], 32, expected.size))
    np.testing.assert_array_equal(Y.reshape(expected.shape), expected)
    assert memory[:Y_ADDR] == before[:Y_ADDR]
    assert memory[Y_ADDR + y_bytes :] == before[Y_ADDR + y_bytes :]

    # Every L3 read is of X, F, the bias or Y; X is read once per in-bounds position
    # and kernel offset (no row is padding), and the bias once per output tile
    reads = [(addr, beats) for kind, addr, beats in bursts if kind == "r"]
    regions = [(X_ADDR, X.size), (F_ADDR, F.size), (B_ADDR, F_shape[3] * 4), (Y_ADDR, y_bytes)]
    assert all(any(base - beat < a < base + size for base, size in regions) for a, _ in reads)
    x_reads = [a for a, _ in reads if X_ADDR - beat < a < X_ADDR + X.size]
    assert len(x_reads) == in_bounds
    b_reads = [a for a, _ in reads if B_ADDR - beat < a < B_ADDR + F_shape[3] * 4]
    tiles = -(-expected.size // F_shape[3] // DIM) * -(-F_shape[3] // DIM)
    assert len(b_reads) == (tiles if bias else 0)
