# ISA Instructions

`Matmul` (`C = A @ B`) and `Conv2d` (`Y = conv2d(X, F)`, NHWC, as an implicit GEMM) tile an
operation larger than the array and issue the same commands a host would, on a command port
that `isa.command.connect_to_top` wires to `SystolicTop`:

- every command names its controller (`cmd_type`) and that controller's opcode, and the next
  is issued only after `cmd_completed`, because the accelerator has no dependency tracking;
- both run weight-stationary: per K tile, LOAD the operand rows (one command per row, at its
  exact address), PRELOAD the weights, COMPUTE (the first K tile overwrites, the rest
  accumulate); then STORE the output rows;
- Conv2d walks one K tile per kernel offset (kh, kw) and chunk of input channels. Each row
  loads the input pixel that offset reads, with stride, dilation and padding applied, or the
  zero row at `CONFIG_ZERO_ADDR` in the padding.

## Layouts

The DMA moves whole bus beats, so START rejects (`UNSUPPORTED_LAYOUT`) a layout with a row
that does not start on a beat boundary. A beat is at least as wide as an accumulator row, so
today Matmul needs N and K to fit one tile (M can be anything, with rows a whole number of
beats apart), and real convolutions are rejected. `require_beat_aligned=False` skips the
check, for a DMA that can address any byte. Bias needs a DMA into the accumulator, which the
accelerator lacks (`UNSUPPORTED_BIAS`).

## Examples

```bash
python examples/gemm/07_isa_matmul.py            # Matmul on a real SystolicTop
python examples/gemm/07_isa_matmul.py --any-layout
python examples/conv/01_isa_conv2d.py            # Conv2d command stream, checked against numpy
```

## Verification

`tests/unit/test_matmul.py` and `tests/unit/test_conv2d.py`: the issued commands run through a
Python model of `SystolicTop` (`tests/unit/_isa.py`) and match numpy (Matmul tiled in M, N and
K; Conv2d with padding, stride, dilation, several batches, channel chunks and N tiles), and
Matmul drives a real `SystolicTop` to a correct C.

[The original implementation plan](plan/isa-instructions.md) is superseded by this page.
