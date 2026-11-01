"""
Cocotb tests for the ``systolic_array`` target: a 2x2 grid of 2x2 PEArrays
(4x4 PEs), so every signal crosses PEArray boundaries in both directions.
Unskewed A and B vectors in, A @ B out, for output- and weight-stationary.
"""

import random

import cocotb
from tb.gemm import cycle, signed, start, tick

ROWS = COLS = 4
LATENCY = (2 - 1) + (2 - 1)  # (grid_rows - 1) + (grid_cols - 1)
OS, WS = 0, 1


def matmul(A, B):
    return [
        [sum(A[i][k] * B[k][j] for k in range(len(B))) for j in range(len(B[0]))]
        for i in range(len(A))
    ]


def rand_matrix(rng, rows, cols):
    return [[rng.randint(-128, 127) for _ in range(cols)] for _ in range(rows)]


@cocotb.test()
async def test_systolic_array_output_stationary(dut):
    await start(dut)
    rng = random.Random(5)
    k = 6
    A, B = rand_matrix(rng, ROWS, k), rand_matrix(rng, k, COLS)
    out_last_at = None
    for t in range(k + LATENCY + 1):
        inputs = {
            "in_control_dataflow": OS,
            "in_shift": 0,
            "in_valid": t < k,
            "in_last": t == k - 1,
        }
        for i in range(ROWS):
            inputs[f"in_a_{i}"] = A[i][t] if t < k else 0
        for j in range(COLS):
            inputs[f"in_b_{j}"] = B[t][j] if t < k else 0
        await cycle(dut, **inputs)
        if int(dut.out_last.value):
            out_last_at = t
        await tick(dut)
    assert out_last_at == k - 1 + LATENCY
    drained = []
    for _ in range(ROWS):
        await cycle(dut, in_valid=0, in_last=0, in_shift=1, **{f"in_d_{j}": 0 for j in range(COLS)})
        drained.append([signed(dut, f"out_c_{j}") for j in range(COLS)])
        await tick(dut)
    assert drained[::-1] == matmul(A, B)


@cocotb.test()
async def test_systolic_array_weight_stationary(dut):
    await start(dut)
    rng = random.Random(6)
    m_rows = 7
    W, A = rand_matrix(rng, ROWS, COLS), rand_matrix(rng, m_rows, ROWS)
    for r in reversed(range(ROWS)):
        await cycle(
            dut,
            in_control_dataflow=WS,
            in_valid=0,
            in_shift=1,
            **{f"in_d_{j}": W[r][j] for j in range(COLS)},
        )
        await tick(dut)
    out = []
    for t in range(m_rows + LATENCY + 1):
        inputs = {"in_shift": 0, "in_valid": t < m_rows, "in_last": t == m_rows - 1}
        for i in range(ROWS):
            inputs[f"in_a_{i}"] = A[t][i] if t < m_rows else 0
        await cycle(dut, **inputs)
        if int(dut.out_valid.value):
            out.append([signed(dut, f"out_psum_{j}") for j in range(COLS)])
        await tick(dut)
    assert out == matmul(A, W)
