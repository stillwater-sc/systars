"""Cocotb tests for a 2x2 PEArray: OS block product and WS combinational partial sums."""

import random

from tb.gemm import cycle, signed, start, tick

import cocotb

ROWS = COLS = 2
OS, WS = 0, 1


def matmul(A, B):
    return [
        [sum(A[i][k] * B[k][j] for k in range(len(B))) for j in range(len(B[0]))]
        for i in range(len(A))
    ]


@cocotb.test()
async def test_pe_array_output_stationary(dut):
    """A and B in the same cycle meet at every PE; draining gives A @ B, bottom row first."""
    await start(dut)
    rng = random.Random(3)
    k = 5
    A = [[rng.randint(-128, 127) for _ in range(k)] for _ in range(ROWS)]
    B = [[rng.randint(-128, 127) for _ in range(COLS)] for _ in range(k)]
    for step in range(k):
        inputs = {f"in_a_{i}": A[i][step] for i in range(ROWS)}
        inputs.update({f"in_b_{j}": B[step][j] for j in range(COLS)})
        await cycle(dut, in_control_dataflow=OS, in_valid=1, in_shift=0, **inputs)
        await tick(dut)
    drained = []
    for _ in range(ROWS):
        await cycle(dut, in_valid=0, in_shift=1, **{f"in_d_{j}": 0 for j in range(COLS)})
        drained.append([signed(dut, f"out_c_{j}") for j in range(COLS)])
        await tick(dut)
    assert drained[::-1] == matmul(A, B)


@cocotb.test()
async def test_pe_array_weight_stationary(dut):
    """Weights shift in last row first; out_psum = in_psum + a @ W, combinationally."""
    await start(dut)
    rng = random.Random(4)
    W = [[rng.randint(-128, 127) for _ in range(COLS)] for _ in range(ROWS)]
    for r in reversed(range(ROWS)):
        await cycle(
            dut, in_control_dataflow=WS, in_shift=1, **{f"in_d_{j}": W[r][j] for j in range(COLS)}
        )
        await tick(dut)
    for _ in range(4):
        a = [rng.randint(-128, 127) for _ in range(ROWS)]
        psum = [rng.randint(-1000, 1000) for _ in range(COLS)]
        inputs = {f"in_a_{i}": a[i] for i in range(ROWS)}
        inputs.update({f"in_psum_{j}": psum[j] for j in range(COLS)})
        await cycle(dut, in_shift=0, **inputs)
        expected = [psum[j] + sum(a[i] * W[i][j] for i in range(ROWS)) for j in range(COLS)]
        assert [signed(dut, f"out_psum_{j}") for j in range(COLS)] == expected
        await tick(dut)
