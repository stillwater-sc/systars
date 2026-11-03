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
from systars.util.commands import ExecOpcode

from ._axi import axi_memory
from ._isa import TopModel, from_bytes, run_isa, to_unsigned_bytes
from ._sim import skip_without_yosys

X_ADDR, F_ADDR, Y_ADDR, ZERO_ADDR = 0x1000, 0x20000, 0x40000, 0xF0000
DIM = 4


def config(**overrides):
    fields = {"grid_rows": DIM, "grid_cols": DIM, "sp_capacity_kb": 4, "acc_capacity_kb": 4}
    fields.update(overrides)
    return SystolicConfig(**fields)


def x_and_f_loads(cmds):
    """The X row LOADs (into the A slots, 0 and 1) and the filter LOADs (B slots, 2 and 3)."""
    loads = [c for c in cmds if c.type == CTRL_LOAD]
    x = [c for c in loads if (c.sync >> Sync.DST) & 0xF < 2]
    return x, [c for c in loads if (c.sync >> Sync.DST) & 0xF >= 2]


def out_size(size, kernel, stride, pad, dilation):
    return (size + 2 * pad - dilation * (kernel - 1) - 1) // stride + 1


def reference(X, F, stride, pad, dilation):
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
    return Y


def setup(
    X_shape,
    F_shape,
    stride=(1, 1),
    pad=(0, 0),
    dilation=(1, 1),
    *,
    y_stride=0,
    zero=True,
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
    if zero:
        cmds.append((Conv2dCmd.CONFIG_ZERO_ADDR, ZERO_ADDR))
    return cmds


def check(X_shape, F_shape, stride=(1, 1), pad=(0, 0), dilation=(1, 1), seed=0, y_stride=0):
    """Dense X, F and (with ``y_stride`` 0) dense Y."""
    rng = np.random.default_rng(seed)
    X = rng.integers(-128, 128, X_shape)
    F = rng.integers(-128, 128, F_shape)
    expected = reference(X, F, stride, pad, dilation)
    row_stride = y_stride or F_shape[3] * 4

    cfg = config()
    isa = Conv2d(cfg)
    result = run_isa(
        isa,
        setup(X_shape, F_shape, stride, pad, dilation, y_stride=y_stride),
        Conv2dCmd.START,
        latency=0,
    )
    assert result.done and not result.error, f"error {result.error_code:#x}"

    model = TopModel(cfg)
    model.write(X_ADDR, to_unsigned_bytes(X, 8))
    model.write(F_ADDR, to_unsigned_bytes(F, 8))
    model.run(result.commands)  # also checks the credits and tokens
    model.assert_all_free()
    positions = expected.shape[0] * expected.shape[1] * expected.shape[2]
    out_c = F_shape[3]
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


class TestCommandStream:
    def commands(self, **kw):
        X_shape, F_shape = (1, 4, 4, 2), (3, 3, 2, 3)
        cfg = config()
        cmds = setup(X_shape, F_shape, pad=(1, 1), **kw)
        return run_isa(Conv2d(cfg), cmds, Conv2dCmd.START, latency=0).commands

    def test_structure(self):
        cmds = self.commands()
        assert (cmds[0].type, cmds[0].opcode, cmds[0].rs1) == (CTRL_EXEC, ExecOpcode.CONFIG, 1)
        execs = [c.opcode for c in cmds[1:] if c.type == CTRL_EXEC]
        # 16 positions = 4 M tiles, 1 N tile, 9 K tiles (3x3 kernel, 1 channel chunk)
        per_m_tile = [ExecOpcode.PRELOAD, ExecOpcode.COMPUTE] + [
            ExecOpcode.PRELOAD,
            ExecOpcode.COMPUTE_ACCUMULATE,
        ] * 8
        assert execs == per_m_tile * 4
        # One STORE per output tile (4 positions, 3 channels, dense)
        stores = [c for c in cmds if c.type == CTRL_STORE]
        assert [(c.dram_addr, c.len, c.stride, c.row_bytes) for c in stores] == [
            (Y_ADDR + m * 4 * 12, 4, 12, 12) for m in range(4)
        ]
        # Per K tile: one LOAD per output position (len 1, its 2 channels), one per filter tile
        x_loads, f_loads = x_and_f_loads(cmds)
        assert {(c.len, c.row_bytes) for c in x_loads} == {(1, 2)}
        assert {(c.len, c.stride, c.row_bytes) for c in f_loads} == {(2, 3, 3)}
        assert len(f_loads) == 9 * 4

    def test_x_rows_fill_one_slot(self):
        """A K tile's X rows share an A slot: the first LOAD takes its credit, the last
        emits its token; the slot pairs alternate between K tiles."""
        x_loads, f_loads = x_and_f_loads(self.commands())
        first = [(c.sync >> Sync.DST) & 0xF for c in f_loads[:4]]
        assert first == [2, 3, 2, 3]  # B slots of the two pairs
        tile = x_loads[:4]  # the first K tile's 4 positions
        assert [(c.sync >> Sync.DST) & 0xF for c in tile] == [0] * 4
        assert [bool(c.sync & Sync.ACQUIRE) for c in tile] == [True, False, False, False]
        assert [bool(c.sync & Sync.SIGNAL) for c in tile] == [False, False, False, True]
        assert [c.local_addr for c in x_loads[4:8]] == [DIM, DIM + 1, DIM + 2, DIM + 3]

    def test_padding_rows_load_the_zero_row(self):
        cmds = self.commands()
        x_loads, _ = x_and_f_loads(cmds)
        zero = [c for c in x_loads if c.dram_addr == ZERO_ADDR]
        # Kernel offset (0, 0) of the first M tile (positions (0,0)..(0,3)) is all padding
        assert all(c.dram_addr == ZERO_ADDR for c in x_loads[:4])
        assert zero and len(zero) < len(x_loads)


class TestErrors:
    def run(self, cmds):
        return run_isa(Conv2d(config()), cmds, Conv2dCmd.START)

    def test_start_without_config(self):
        assert self.run([]).error_code == Conv2dError.CONFIG_INCOMPLETE

    def test_padding_needs_zero_row(self):
        cmds = setup((1, 4, 4, 4), (3, 3, 4, 4), pad=(1, 1), zero=False)
        assert self.run(cmds).error_code == Conv2dError.CONFIG_INCOMPLETE

    def test_zero_stride(self):
        cmds = setup((1, 4, 4, 4), (1, 1, 4, 4))
        cmds.append((Conv2dCmd.CONFIG_STRIDE, 0))
        assert self.run(cmds).error_code == Conv2dError.INVALID_DIMS

    def test_bias_is_unsupported(self):
        cmds = setup((1, 4, 4, 4), (1, 1, 4, 4), bias=0x8000)
        assert self.run(cmds).error_code == Conv2dError.UNSUPPORTED_BIAS

    def test_accumulate_is_unsupported(self):
        """Y is never loaded into the accumulator, so accumulating onto it would add stale rows."""
        cmds = setup((1, 4, 4, 4), (1, 1, 4, 4), options=1)
        result = self.run(cmds)
        assert result.error_code == Conv2dError.UNSUPPORTED_ACCUMULATE
        assert not result.commands

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
    def test_requires_weight_stationary(self):
        with pytest.raises(ValueError, match="weight-stationary"):
            Conv2d(config(dataflow=Dataflow.OUTPUT_STATIONARY))

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
    "X_shape,F_shape,stride,pad",
    [((1, 4, 4, 3), (3, 3, 3, 5), (1, 1), (1, 1)), ((2, 5, 4, 6), (2, 3, 6, 2), (2, 1), (1, 0))],
)
def test_conv2d_drives_systolic_top(X_shape, F_shape, stride, pad):
    """Conv2d on a real SystolicTop: dense NHWC X, F and Y with odd channel counts."""
    from ._sim import run_sim

    cfg = config()
    beat = cfg.dma_buswidth // 8
    rng = np.random.default_rng(sum(X_shape) + sum(F_shape))
    X = rng.integers(-128, 128, X_shape)
    F = rng.integers(-128, 128, F_shape)
    expected = reference(X, F, stride, pad, (1, 1))
    memory = bytearray(rng.integers(0, 256, 1 << 20, dtype=np.uint8).tobytes())
    memory[X_ADDR : X_ADDR + X.size] = to_unsigned_bytes(X, 8)
    memory[F_ADDR : F_ADDR + F.size] = to_unsigned_bytes(F, 8)
    memory[ZERO_ADDR : ZERO_ADDR + beat] = bytes(beat)
    y_bytes = expected.size * 4
    before = bytes(memory)

    dut = Conv2dOnTop(cfg)
    isa, top = dut.isa, dut.top
    status = {}

    async def testbench(ctx):
        for cmd, data in [*setup(X_shape, F_shape, stride, pad), (Conv2dCmd.START, 0)]:
            ctx.set(isa.cfg_valid, 1)
            ctx.set(isa.cfg_cmd, cmd)
            ctx.set(isa.cfg_data, data)
            await ctx.tick()
        ctx.set(isa.cfg_valid, 0)
        done, error = await ctx.tick().sample(isa.done, isa.error).until(isa.done | isa.error)
        status.update(done=done, error=error)

    run_sim(dut, testbench, processes=[axi_memory(top, memory, beat)], max_cycles=500_000)
    assert status == {"done": 1, "error": 0}
    Y = np.array(from_bytes(memory[Y_ADDR : Y_ADDR + y_bytes], 32, expected.size))
    np.testing.assert_array_equal(Y.reshape(expected.shape), expected)
    assert memory[:Y_ADDR] == before[:Y_ADDR]
    assert memory[Y_ADDR + y_bytes :] == before[Y_ADDR + y_bytes :]
