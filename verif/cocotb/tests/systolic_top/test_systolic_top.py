"""
Cocotb tests for ``SystolicTop`` (the ``systolic_top`` target: 4x4, OS, WS and IS).

The tile's AXI master is served by a Python model of L3 (``tb.axi.dram_model``),
which asserts that no burst crosses a 4 KB boundary, and commands go straight onto
the tile's command port. A LOAD and a STORE across 4 KB boundaries, a WS GEMM whose
transfers cross them, and an OS and a WS GEMM issued command by command check the
BlockMover and the Streamer on the Verilated RTL.

A Matmul tiled in M, N and K (with and without a bias and accumulate) in each
dataflow, and an output-stationary Conv2d with padding, stride, dilation, odd channel
counts and a bias, run as a host would issue them: the ISA instruction's command
stream (``tb.isa``) is pushed with its credits and tokens, without waiting for
completions, and the results are checked against a reference, with every other byte
of L3 untouched.

Port of kpugen's ``tests/systolic_tile`` (kpugen #19). kpugen drives its tile through
CSRs and builds one tile per dataflow; ``SystolicTop`` has a command port and one
build runs every dataflow (CONFIG, ``CONFIG_DATAFLOW``). kpugen's CSR, NoC and
memory-port tests have no ``SystolicTop`` counterpart.
"""

import random

from tb.axi import dram_model
from tb.gemm import pack, unpack
from tb.isa import FIELDS, idle_commands, isa_commands, push_commands

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge
from systars.isa.conv2d import Conv2d, Conv2dCmd
from systars.isa.matmul import Matmul, MatmulCmd
from systars.targets import systolic_top_config
from systars.top import SystolicTop
from systars.util.commands import DmaOpcode, ExecOpcode

ROWS = COLS = 4
BEAT = systolic_top_config().dma_buswidth // 8  # 16 bytes: one accumulator row
L3 = 1 << 16  # bytes of L3 the model serves
DATAFLOWS = {"os": 0, "ws": 1, "is": 2}  # CONFIG rs1, CONFIG_DATAFLOW
LOAD, STORE, EXEC = SystolicTop.CTRL_LOAD, SystolicTop.CTRL_STORE, SystolicTop.CTRL_EXEC


async def setup(dut, memory: bytearray):
    """Start the clock, drive every input idle, reset, and serve L3 from ``memory``.
    Returns the burst log."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    idle_commands(dut)
    dut.rst.value = 1
    for _ in range(2):
        await RisingEdge(dut.clk)
    dut.rst.value = 0
    bursts: list = []
    cocotb.start_soon(dram_model(dut, memory, BEAT, bursts))
    await RisingEdge(dut.clk)
    return bursts


async def run_command(dut, cmd_type, opcode, cmd_id, **fields):
    """Push one command and wait for it to complete (every other field 0)."""
    cmd = dict.fromkeys(FIELDS, 0) | {"type": cmd_type, "opcode": opcode, "id": cmd_id} | fields
    await push_commands(dut, [cmd], timeout=5000)


def matmul(A, B):
    """Reference C = A @ B on lists of rows."""
    return [
        [sum(A[i][k] * B[k][j] for k in range(len(B))) for j in range(len(B[0]))]
        for i in range(len(A))
    ]


def put(memory, addr, value, nbytes):
    """Write the integer ``value`` as ``nbytes`` little-endian bytes at ``addr``."""
    memory[addr : addr + nbytes] = value.to_bytes(nbytes, "little")


def get(memory, addr, nbytes):
    """The ``nbytes`` little-endian bytes at ``addr``, as an integer."""
    return int.from_bytes(memory[addr : addr + nbytes], "little")


@cocotb.test()
async def test_top_bursts_split_at_4kb(dut):
    """LOADs and STOREs across 4 KB boundaries: a streamed tile ends a burst at the
    boundary, and a row spilling over it is two 1-beat bursts. A STORE writes only
    the bytes of its rows."""
    memory = bytearray(random.Random(18).randbytes(L3))
    bursts = await setup(dut, memory)

    # Streamed: 12 consecutive beats from 4 beats before 0x1000
    page = 0x1000
    await run_command(dut, LOAD, DmaOpcode.MEMCPY, 1, dram_addr=page - 64, local_addr=0, len=12)
    assert 1 << int(dut.axi_arsize.value) == BEAT
    assert bursts == [("r", page - 64, 64 // BEAT), ("r", page, 12 - 64 // BEAT)]

    # Four accumulator rows out, streamed across 0x2000
    bursts.clear()
    page = 0x2000
    before = bytes(memory)
    await run_command(dut, STORE, DmaOpcode.MEMCPY, 2, dram_addr=page - 2 * BEAT, len=4)
    assert bursts == [("w", page - 2 * BEAT, 2), ("w", page, 2)]
    assert memory[: page - 2 * BEAT] == before[: page - 2 * BEAT]
    assert memory[page + 2 * BEAT :] == before[page + 2 * BEAT :]

    # One row spilling over 0x3000: two 1-beat bursts, read and written
    bursts.clear()
    page = 0x3000
    await run_command(dut, LOAD, DmaOpcode.MEMCPY, 3, dram_addr=page - 2, local_addr=0, len=1)
    assert bursts == [("r", page - BEAT, 1), ("r", page, 1)]
    bursts.clear()
    before = bytes(memory)
    await run_command(dut, STORE, DmaOpcode.MEMCPY, 4, dram_addr=page - 4, len=1)
    assert bursts == [("w", page - BEAT, 1), ("w", page, 1)]
    assert memory[: page - 4] == before[: page - 4]
    assert memory[page - 4 + BEAT :] == before[page - 4 + BEAT :]


@cocotb.test()
async def test_top_gemm_across_4kb(dut):
    """A WS GEMM whose transfers cross 4 KB boundaries, checked by its values: A streams
    across 0x1000, a row of B spills over 0x2000, and C goes out twice, streamed across
    0x3000 and with a row spilling over 0x4000."""
    memory = bytearray(L3)
    bursts = await setup(dut, memory)
    rng = random.Random(46)
    A = [[rng.randint(-128, 127) for _ in range(ROWS)] for _ in range(ROWS)]
    B = [[rng.randint(-128, 127) for _ in range(COLS)] for _ in range(ROWS)]
    sp_a, sp_b = 0, 64

    # A column-major (L2 row k = A[:, k]), one beat per row, streamed across 0x1000
    a_addr = 0x1000 - 2 * BEAT
    for k in range(ROWS):
        put(memory, a_addr + k * BEAT, pack([A[i][k] for i in range(ROWS)], 8), BEAT)
    await run_command(dut, LOAD, DmaOpcode.MEMCPY, 1, dram_addr=a_addr, local_addr=sp_a, len=ROWS)
    assert bursts == [("r", a_addr, 2), ("r", 0x1000, 2)]

    # B row-major, rows 4 bytes at byte 14 of a beat: row 1 spills over 0x2000
    bursts.clear()
    b_addr = 0x2000 - 2 - BEAT
    for k in range(ROWS):
        put(memory, b_addr + k * BEAT, pack(B[k], 8), COLS)
    await run_command(dut, LOAD, DmaOpcode.MEMCPY, 2, dram_addr=b_addr, local_addr=sp_b,
                      len=ROWS, stride=BEAT, row_bytes=COLS)  # fmt: skip
    p = 0x2000
    assert bursts == [("r", p - 2 * BEAT, 2), ("r", p - BEAT, 1), ("r", p, 1), ("r", p, 2),
                      ("r", p + BEAT, 2)]  # fmt: skip

    await run_command(dut, EXEC, ExecOpcode.CONFIG, 3, rs1=DATAFLOWS["ws"])
    await run_command(dut, EXEC, ExecOpcode.PRELOAD, 4, rs1=sp_b, k_dim=ROWS)
    await run_command(dut, EXEC, ExecOpcode.COMPUTE, 5, rs1=sp_a, rd=0, k_dim=ROWS)
    expected = matmul(A, B)

    # C streamed across 0x3000: one 16-byte row per beat
    bursts.clear()
    c_addr = 0x3000 - 2 * BEAT
    await run_command(dut, STORE, DmaOpcode.MEMCPY, 6, dram_addr=c_addr, len=ROWS)
    assert bursts == [("w", c_addr, 2), ("w", 0x3000, 2)]
    got = [unpack(get(memory, c_addr + r * BEAT, BEAT), 32, COLS) for r in range(ROWS)]
    assert got == expected

    # C again, rows at byte 12 of a beat: row 1 spills over 0x4000
    bursts.clear()
    c_addr = 0x4000 - 4 - BEAT
    await run_command(dut, STORE, DmaOpcode.MEMCPY, 7, dram_addr=c_addr, len=ROWS, stride=BEAT)
    p = 0x4000
    assert bursts == [("w", p - 2 * BEAT, 2), ("w", p - BEAT, 1), ("w", p, 1), ("w", p, 2),
                      ("w", p + BEAT, 2)]  # fmt: skip
    got = [unpack(get(memory, c_addr + r * BEAT, BEAT), 32, COLS) for r in range(ROWS)]
    assert got == expected


@cocotb.test()
@cocotb.parametrize(dataflow=["os", "ws"])
async def test_top_gemm(dut, dataflow):
    """C = A @ B, command by command: every operand through L3."""
    memory = bytearray(L3)
    await setup(dut, memory)
    rng = random.Random(7)
    ws = dataflow == "ws"
    # WS: K is the PE rows and M one stream block; OS: K = 6, two K tiles chained
    inner = ROWS if ws else 6
    outer = ROWS
    A = [[rng.randint(-128, 127) for _ in range(inner)] for _ in range(outer)]
    B = [[rng.randint(-128, 127) for _ in range(COLS)] for _ in range(inner)]

    # L2 rows as the Streamer streams them (one row per stream-buffer block)
    a_addr, b_addr, c_addr = 16 * BEAT, 64 * BEAT, 128 * BEAT
    if ws:  # A column-major (row k = A[:, k]), B row-major (the stationary tile)
        a_rows = [list(col) for col in zip(*A, strict=True)]
        b_rows = B
        k_tiles = []
    else:  # per K tile: A row-major (row i = A[i, k0:]), B column-major (row j = B[k0:, j])
        k_tiles = [(k0, min(ROWS, inner - k0)) for k0 in range(0, inner, ROWS)]
        a_rows = [A[i][k0 : k0 + kn] for k0, kn in k_tiles for i in range(ROWS)]
        b_rows = [[B[k][j] for k in range(k0, k0 + kn)] for k0, kn in k_tiles for j in range(COLS)]
    for n, row in enumerate(a_rows):
        put(memory, a_addr + n * BEAT, pack(row, 8), BEAT)
    for n, row in enumerate(b_rows):
        put(memory, b_addr + n * BEAT, pack(row, 8), BEAT)
    sp_a, sp_b = 0, 64

    await run_command(dut, LOAD, DmaOpcode.MEMCPY, 1, dram_addr=a_addr, local_addr=sp_a,
                      len=len(a_rows))  # fmt: skip
    await run_command(dut, LOAD, DmaOpcode.MEMCPY, 2, dram_addr=b_addr, local_addr=sp_b,
                      len=len(b_rows))  # fmt: skip
    await run_command(dut, EXEC, ExecOpcode.CONFIG, 3, rs1=DATAFLOWS[dataflow])
    if ws:
        await run_command(dut, EXEC, ExecOpcode.PRELOAD, 4, rs1=sp_b, k_dim=len(B))
        await run_command(dut, EXEC, ExecOpcode.COMPUTE, 5, rs1=sp_a, rd=0, k_dim=outer)
    else:  # one reduction across the K tiles (cmd_chain), held in the PEs
        for t, (_, kn) in enumerate(k_tiles):
            chain = (1 if t else 0) | (2 if t < len(k_tiles) - 1 else 0)
            await run_command(dut, EXEC, ExecOpcode.COMPUTE, 5 + t, rs1=sp_a + t * ROWS,
                              rs2=sp_b + t * COLS, rd=0, k_dim=kn, chain=chain)  # fmt: skip
    before = bytes(memory)
    await run_command(dut, STORE, DmaOpcode.MEMCPY, 9, dram_addr=c_addr, len=outer)

    C = [unpack(get(memory, c_addr + n * BEAT, BEAT), 32, COLS) for n in range(outer)]
    assert matmul(A, B) == C
    assert memory[:c_addr] == before[:c_addr]
    assert memory[c_addr + outer * BEAT :] == before[c_addr + outer * BEAT :]


# ---------------------------------------------------------------------------
# Matmul and Conv2d as a host issues them (#12, kpugen #19)
# ---------------------------------------------------------------------------

CELL = 4096  # bytes of L3 an instruction's operands and results live in


def to_bytes(values, bits):
    """Pack ``values`` as ``bits``-wide little-endian two's-complement elements."""
    return b"".join((int(v) & ((1 << bits) - 1)).to_bytes(bits // 8, "little") for v in values)


def from_bytes(data, bits):
    """Unpack ``data`` into signed ``bits``-wide little-endian elements."""
    n = bits // 8
    return [int.from_bytes(data[i : i + n], "little", signed=True) for i in range(0, len(data), n)]


async def run_isa(dut, image, isa, config_cmds, start_cmd):
    """Push ``isa``'s command stream onto the tile's command port with ``image`` at the
    bottom of L3; returns L3's bytes afterwards."""
    memory = bytearray(image) + bytearray(L3 - len(image))
    bursts = await setup(dut, memory)
    commands = isa_commands(isa, config_cmds, start_cmd)
    await push_commands(dut, commands)
    assert bursts, "no AXI bursts seen"
    assert not any(memory[len(image) :]), "a write outside the image"
    return bytes(memory[: len(image)])


@cocotb.test()
@cocotb.parametrize(
    dataflow=list(DATAFLOWS),
    variant=["plain", "bias_acc"],
)
async def test_top_matmul(dut, dataflow, variant):
    """C (+)= A @ B (+ D), Matmul 9x7x10 tiled in M, N and K: dense row-major operands
    at odd addresses (cmd_stride, cmd_row_bytes), edge tiles in every dimension. With
    a bias and accumulate, D (one row) and the existing C are loaded into the
    accumulator before each output tile's first K step."""
    bias = accumulate = variant == "bias_acc"
    seed = 20 if bias else 19
    M, N, K = 9, 7, 10
    rng = random.Random(seed)
    image = bytearray(rng.randbytes(CELL))
    A = [[rng.randint(-128, 127) for _ in range(K)] for _ in range(M)]
    B = [[rng.randint(-128, 127) for _ in range(N)] for _ in range(K)]
    D = [rng.randint(-5000, 5000) for _ in range(N)]  # one row, added to every row of C
    a_addr, b_addr, c_addr, d_addr = 0x003, 0x111, 0x205, 0x3A1
    c_bytes = M * N * 4
    image[a_addr : a_addr + M * K] = to_bytes([x for row in A for x in row], 8)
    image[b_addr : b_addr + K * N] = to_bytes([x for row in B for x in row], 8)
    image[d_addr : d_addr + N * 4] = to_bytes(D, 32)
    C0 = [from_bytes(image[c_addr + r * N * 4 : c_addr + (r + 1) * N * 4], 32) for r in range(M)]
    expected = [
        [v + (D[j] if bias else 0) + (C0[i][j] if accumulate else 0) for j, v in enumerate(row)]
        for i, row in enumerate(matmul(A, B))
    ]
    expected = [[(v + (1 << 31)) % (1 << 32) - (1 << 31) for v in row] for row in expected]
    config_cmds = [
        (MatmulCmd.CONFIG_DATAFLOW, DATAFLOWS[dataflow]),
        (MatmulCmd.CONFIG_DIMS, M | N << 16 | K << 32),
        (MatmulCmd.CONFIG_A_ADDR, a_addr),
        (MatmulCmd.CONFIG_B_ADDR, b_addr),
        (MatmulCmd.CONFIG_C_ADDR, c_addr),
        (MatmulCmd.CONFIG_D_ADDR, d_addr if bias else 0),
        (MatmulCmd.CONFIG_D_STRIDE, 0),
        (MatmulCmd.CONFIG_A_STRIDE, K),
        (MatmulCmd.CONFIG_B_STRIDE, N),
        (MatmulCmd.CONFIG_C_STRIDE, N * 4),
        (MatmulCmd.CONFIG_OPTIONS, int(accumulate)),
    ]
    before = bytes(image)
    isa = Matmul(systolic_top_config())
    after = await run_isa(dut, image, isa, config_cmds, MatmulCmd.START)
    got = [from_bytes(after[c_addr + r * N * 4 : c_addr + (r + 1) * N * 4], 32) for r in range(M)]
    assert got == expected
    assert after[:c_addr] == before[:c_addr]
    assert after[c_addr + c_bytes :] == before[c_addr + c_bytes :]


def conv2d_reference(X, F, shape, kernel, stride, pad, dilation, bias):
    """Y[b, oh, ow, oc] (NHWC lists) of int8 X and F, plus the bias."""
    batch, in_h, in_w, in_c = shape
    kh_n, kw_n, out_c = kernel
    out_h = (in_h + 2 * pad[0] - dilation[0] * (kh_n - 1) - 1) // stride[0] + 1
    out_w = (in_w + 2 * pad[1] - dilation[1] * (kw_n - 1) - 1) // stride[1] + 1

    def x(b, h, w, c):
        """X[b, h, w, c] (NHWC)."""
        return X[((b * in_h + h) * in_w + w) * in_c + c]

    def f(kh, kw, c, oc):
        """F[kh, kw, c, oc]."""
        return F[((kh * kw_n + kw) * in_c + c) * out_c + oc]

    Y = []
    for b in range(batch):
        for oh in range(out_h):
            for ow in range(out_w):
                for oc in range(out_c):
                    acc = bias[oc]
                    for kh in range(kh_n):
                        for kw in range(kw_n):
                            ih = oh * stride[0] + kh * dilation[0] - pad[0]
                            iw = ow * stride[1] + kw * dilation[1] - pad[1]
                            if 0 <= ih < in_h and 0 <= iw < in_w:
                                acc += sum(x(b, ih, iw, c) * f(kh, kw, c, oc) for c in range(in_c))
                    Y.append(acc)
    return (out_h, out_w), Y


@cocotb.test()
async def test_top_conv2d(dut):
    """Conv2d (output-stationary): NHWC X 1x6x5x3 with a 3x2 kernel to 5 channels,
    padding (1, 1), stride (1, 2), dilation (2, 1) and a bias. Padded positions are
    CLEARs, the bias is broadcast under every output position, and Y is dense with
    odd channel counts."""
    rng = random.Random(21)
    shape, kernel = (1, 6, 5, 3), (3, 2, 5)
    stride, pad, dilation = (1, 2), (1, 1), (2, 1)
    image = bytearray(rng.randbytes(CELL))
    X = [rng.randint(-128, 127) for _ in range(6 * 5 * 3)]
    F = [rng.randint(-128, 127) for _ in range(3 * 2 * 3 * 5)]
    bias = [rng.randint(-5000, 5000) for _ in range(5)]
    x_addr, f_addr, b_addr, y_addr = 0x011, 0x0F3, 0x1C9, 0x207
    image[x_addr : x_addr + len(X)] = to_bytes(X, 8)
    image[f_addr : f_addr + len(F)] = to_bytes(F, 8)
    image[b_addr : b_addr + 20] = to_bytes(bias, 32)
    (out_h, out_w), expected = conv2d_reference(X, F, shape, kernel, stride, pad, dilation, bias)
    y_bytes = len(expected) * 4
    batch, in_h, in_w, in_c = shape
    config_cmds = [
        (Conv2dCmd.CONFIG_INPUT_DIMS, batch | in_h << 8 | in_w << 24 | in_c << 40),
        (Conv2dCmd.CONFIG_OUTPUT_DIMS, out_h | out_w << 16 | kernel[2] << 32),
        (Conv2dCmd.CONFIG_KERNEL_DIMS, kernel[0] | kernel[1] << 8),
        (Conv2dCmd.CONFIG_STRIDE, stride[0] | stride[1] << 8),
        (Conv2dCmd.CONFIG_PADDING, pad[0] | pad[1] << 8),
        (Conv2dCmd.CONFIG_DILATION, dilation[0] | dilation[1] << 8),
        (Conv2dCmd.CONFIG_X_ADDR, x_addr),
        (Conv2dCmd.CONFIG_F_ADDR, f_addr),
        (Conv2dCmd.CONFIG_Y_ADDR, y_addr),
        (Conv2dCmd.CONFIG_B_ADDR, b_addr),
    ]
    before = bytes(image)
    after = await run_isa(dut, image, Conv2d(systolic_top_config()), config_cmds, Conv2dCmd.START)
    assert from_bytes(after[y_addr : y_addr + y_bytes], 32) == expected
    assert after[:y_addr] == before[:y_addr]
    assert after[y_addr + y_bytes :] == before[y_addr + y_bytes :]
