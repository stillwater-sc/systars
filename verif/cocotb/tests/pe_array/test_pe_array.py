"""Cocotb tests for a 2x2 PEArray: staggered streams meet at every PE, OS results
drain north, and stationary partial sums leave the south edge."""

import random

from tb.gemm import os_schedule, play, rows_of, start, stationary_schedule

import cocotb

ROWS = COLS = 2
OS, STATIONARY = 0, 1


def matmul(A, B):
    return [
        [sum(A[i][k] * B[k][j] for k in range(len(B))) for j in range(len(B[0]))]
        for i in range(len(A))
    ]


@cocotb.test()
async def test_pe_array_output_stationary(dut):
    await start(dut)
    rng = random.Random(3)
    k = 5
    A = [[rng.randint(-128, 127) for _ in range(k)] for _ in range(ROWS)]
    B = [[rng.randint(-128, 127) for _ in range(COLS)] for _ in range(k)]
    north, _ = await play(dut, COLS, os_schedule(A, B), k + 3 * ROWS + COLS, dataflow=OS)
    assert rows_of(north) == matmul(A, B)


@cocotb.test()
async def test_pe_array_stationary(dut):
    await start(dut)
    rng = random.Random(4)
    W = [[rng.randint(-128, 127) for _ in range(COLS)] for _ in range(ROWS)]
    A = [[rng.randint(-128, 127) for _ in range(ROWS)] for _ in range(4)]
    _, south = await play(dut, COLS, stationary_schedule(A, W), 12, dataflow=STATIONARY)
    assert rows_of(south) == matmul(A, W)
