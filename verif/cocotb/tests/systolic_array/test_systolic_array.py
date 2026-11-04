"""
Cocotb tests for the ``systolic_array`` target: a 2x2 grid of 2x2 PEArrays
(4x4 PEs), so every link crosses PEArray boundaries in both directions. Streams
are staggered as the L1 stream buffers stagger them; OS results drain north, and
back-to-back OS tiles 2R - 1 clocks apart never collide.
"""

import random

import cocotb
from tb.gemm import os_schedule, play, rows_of, start, stationary_schedule

ROWS = COLS = 4
OS, STATIONARY = 0, 1


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
    north, _ = await play(dut, COLS, os_schedule(A, B), k + 3 * ROWS + COLS, dataflow=OS)
    assert rows_of(north) == matmul(A, B)


@cocotb.test()
async def test_systolic_array_back_to_back_tiles(dut):
    """Short reductions 2R - 1 clocks apart: every result lands, none collide."""
    await start(dut)
    rng = random.Random(7)
    sched, expected = {}, []
    for n in range(3):
        A, B = rand_matrix(rng, ROWS, 2), rand_matrix(rng, 2, COLS)
        for t, ports in os_schedule(A, B, start=n * (2 * ROWS - 1)).items():
            sched.setdefault(t, {}).update(ports)
        expected += matmul(A, B)
    north, _ = await play(dut, COLS, sched, 3 * (2 * ROWS - 1) + 3 * ROWS + COLS, dataflow=OS)
    assert rows_of(north) == expected


@cocotb.test()
async def test_systolic_array_stationary(dut):
    await start(dut)
    rng = random.Random(6)
    W, A = rand_matrix(rng, ROWS, COLS), rand_matrix(rng, 7, ROWS)
    _, south = await play(dut, COLS, stationary_schedule(A, W), 4 + 7 + ROWS + COLS + 2,
                          dataflow=STATIONARY)  # fmt: skip
    assert rows_of(south) == matmul(A, W)
