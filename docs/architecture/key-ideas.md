# Key Ideas

SYSTARS draws on two sources: the original SYSTARS work on systolic/wavefront arrays, which
supplies the fabric and its wavefront timing, and the KPU, which supplies the data movement
around it. The KPU's innovations (whole-tile pushes between L3, L2 and L1, the BlockMover and
the Streamer as CSP processes, credits and tile-ready tokens) are what let one `SystolicTop`
run output-stationary, weight-stationary and input-stationary scheduled matmuls on the same
fabric. Each idea below names the code that implements it.

## 1. The skew comes from L1, not from register triangles

A textbook array needs a triangle of skew registers on its inputs and another of deskew
registers on its outputs, about R(R - 1)/2 registers each. SYSTARS has neither. The Streamer
loads the L1 stream buffers on consecutive clocks, so buffer i starts one clock after
buffer i - 1: that stagger is the skew. One result FIFO per column turns the columns back into
rows. (`memory/stream_buffer.py`, `controller/execute.py`)

## 2. A fully registered fabric

Every PE link has an input register, so a value moves exactly one PE per clock in every
direction. PEArray boundaries make no difference to the timing, the critical path is one MAC
and a register, and the same RTL scales from 8x8 to 64x64 PEs. (`core/pe.py`)

## 3. Schedule at issue, never stall

The Streamer starts a command only when the fabric can take it on schedule, then runs it to the
end. There is no backpressure inside the fabric. Instead of stall logic, a violation raises a
sticky protocol flag (`l1_error`, `out_collision`, `sync_error`) that a valid schedule never
sets, and the tests assert that the flags stay low.

## 4. Bubble-free streaming

A stream buffer takes its next block in the clock its last element leaves, so with
K >= R, C the next tile's blocks reach the fabric with no gap. In OS, back-to-back reductions
are also bound by the drain rule: their results must leave at least 2R - 1 clocks apart, so
they stream without bubbles only when K >= 2R - 1 (chained K tiles count). On a 4x4 array with
K = 4, consecutive single-command reductions start 7 clocks apart, not 4. When both conditions
hold, the fabric stays fully busy across tile boundaries: the 16x16 animation (K = 32) on the
[wavefront page](../wavefront-animation.md) plots the busy fraction.

## 5. CSP flow control from the KPU

L2 and the accumulator hold whole tiles in slots, one credit each, and a slot is always in
exactly one of four states: FREE, OWNED, READY or READING. Every command carries `cmd_sync`
flags (ACQUIRE, SIGNAL, WAIT, RELEASE). Commands are pushed ahead with no completion round
trip. The streams the ISA instructions issue cannot deadlock, since every credit a command
waits for is returned by a command issued before it. The controller does not check that
order: a hand-written stream that queues a WAIT ahead of its producer on the same in-order
queue stalls that queue. With two buffers, an 8x8x16 Matmul on a 4x4 array
takes 744 cycles instead of 907, with bit-identical results. (`controller/sync.py`,
`isa/command.py`, `tests/unit/test_credits.py`)

## 6. Layout is the DMA's job

The BlockMover does the corner turns (`TRANSPOSE`), so the Streamer always reads one L2 row
per L1 buffer. It also handles any byte address, strides, partial rows, byte strobes and the
AXI 4 KB rule. The fabric and the Streamer never see a layout problem.

## 7. The accumulator has two producers

A LOAD into the accumulator presets a bias or an existing C (`BROADCAST`, `TRANSPOSE` or a
plain MEMCPY), or writes zero rows with no L3 read (`CLEAR`), in the write-port clocks the
Streamer leaves free. That gives `C = A @ B + D`, `C += A @ B` and Conv2d's padding without
special compute modes. (`SlotPool(producers=2)` in `controller/sync.py`)

## 8. One fabric, three dataflows, chosen at run time

OS, WS and IS share the RTL; a `CONFIG` command picks one. What makes this work is the data
movement, not the PE: the BlockMover's corner turns lay each operand out as the chosen schedule
streams it, the Streamer feeds L1 on that schedule, and credits and tokens order the tiles the
same way in every dataflow. Choosing a dataflow is a software decision per instruction, which
is what a compiler targeting the tile needs.

## 9. Visualization that is also verification

`systars.wavefront` records the real RTL in simulation, checks every PE in every clock against
the wavefront equation, and renders the animated SVGs the docs embed. The committed
animations fail the tests when they go stale, so the pictures cannot drift from the hardware.

## 10. Hardware as a Python program

Amaranth makes the array a parameterized Python object, and `systars.targets` is the single
list of RTL targets that `just gen`, `just synth` and CI read.
