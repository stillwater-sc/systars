# ISA Instructions

`Matmul` (`C = A @ B`) and `Conv2d` (`Y = conv2d(X, F)`, NHWC, as an implicit GEMM) tile an
operation larger than the array and issue the same commands a host would, on a command port
that `isa.command.connect_to_top` wires to `SystolicTop`:

- every command names its controller (`cmd_type`) and that controller's opcode, and the next
  is issued only after `cmd_completed`, because the accelerator has no dependency tracking yet
  (#9: credits and tile-ready tokens);
- both run weight-stationary: per K tile, LOAD the operand tiles, PRELOAD the weights,
  COMPUTE (the first K tile overwrites, the rest accumulate); then STORE the output tile;
- Conv2d walks one K tile per kernel offset (kh, kw) and chunk of input channels. Each output
  position loads the input pixel that offset reads, with stride, dilation and padding applied,
  or the zero row at `CONFIG_ZERO_ADDR` in the padding.

## Data movement (the CSP roles)

Systars follows the KPU's CSP model ([kpu-sim](https://github.com/stillwater-sc/kpu-sim),
`docs/01-architecture/kpu-execution-model.md`): `SystolicTop` is a systolic KPU compute tile.
Its AXI port is its link to **L3** (a memory tile in a SoC; the AXI memory model in tests), the
scratchpad is **L2**, and the array's edge registers are **L1**. LOAD and STORE are the
**BlockMover**'s two legs (L3 → L2, and accumulator → L3); the execute controller is the
**Streamer** (L2 → L1).

A LOAD or STORE moves a whole tile: `len` rows, row `n` at L3 byte address
`dram_addr + n * stride` (`cmd_stride`; 0 = one bus beat), `row_bytes` bytes each
(`cmd_row_bytes`; 0 = a whole row). Rows may start at any byte: a row that starts mid-beat, or
spills into the next beat, is a 1- or 2-beat burst from the beat boundary, shifted into place.
A loaded row is zero-filled past `row_bytes`; a stored row writes only its own bytes (AXI byte
strobes, `StreamWriter.data_strb`). Rows that are consecutive beats from a beat boundary stream
as bursts of up to 256 beats. *Limitation:* bursts are not yet split at 4 KB boundaries, which
AXI forbids crossing; a long streamed burst, or a 2-beat row burst at the end of a 4 KB region,
can cross one (#11).

## Layouts

A, B and C (Matmul) and X, F and Y (Conv2d) are dense, at any address, with any M, N and K:
START rejects (`UNSUPPORTED_LAYOUT`) only a zero stride, or output rows that would overlap.
Bias, and accumulating onto an existing C or Y (`CONFIG_OPTIONS` bit 0), need a load into the
accumulator, which the accelerator does not have yet (`UNSUPPORTED_BIAS`,
`UNSUPPORTED_ACCUMULATE`; #10).

## Examples

```bash
python examples/gemm/07_isa_matmul.py            # Matmul's command stream
python examples/conv/01_isa_conv2d.py            # Conv2d's command stream
```

Both run the instruction in simulation against a stand-in controller that accepts and
completes every command, and check the count of each command against the tiling. They do
not run `SystolicTop` or check numerical results; the unit tests below do.

## Verification

`tests/unit/test_block_mover.py`: both BlockMover legs over a byte-addressed AXI memory
(`tests/unit/_axi.py`): every byte offset, spilling rows, partial rows, strides, streamed
bursts past 256 beats, and a GEMM with unaligned, strided operands.

`tests/unit/test_matmul.py` and `tests/unit/test_conv2d.py`: the issued commands run through a
Python model of `SystolicTop` (`tests/unit/_isa.py`) and match numpy (Matmul tiled in M, N and
K; Conv2d with padding, stride, dilation, several batches, channel chunks and N tiles), and
both instructions drive a real `SystolicTop` to a correct dense result, leaving every other
byte untouched.

[The original implementation plan](plan/isa-instructions.md) is superseded by this page.
