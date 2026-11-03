# ISA Instructions

`Matmul` (`C = A @ B`) and `Conv2d` (`Y = conv2d(X, F)`, NHWC, as an implicit GEMM) tile an
operation larger than the array and issue the same commands a host would, on a command port
that `isa.command.connect_to_top` wires to `SystolicTop`:

- every command names its process (`cmd_type`) and that process's opcode, and is pushed as
  soon as the process's queue has room, never waiting for a completion: credits and tile-ready
  tokens order the commands ([below](#credits-and-tile-ready-tokens)). An instruction is done
  when its last STORE completes (`cmd_store_completed`);
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

## Credits and tile-ready tokens

`SystolicTop` has an in-order command queue per process (`ld_queue_length`,
`ex_queue_length`, `st_queue_length`), and a queue's head is dispatched only when its credits
and tokens are there (`controller/sync.py`, after kpu-sim's `credit_pool` and `tag_cam`). L2
and the accumulator hold whole tiles in slots (`l2_slots`, default 4; `acc_slots`, default 2),
one credit per slot: there are no row or burst credits, since the BlockMover sequences its own
bursts inside a tile. A slot is FREE (its credit is available), OWNED (a producer is filling
it) or READY (its tile-ready token waits for the consumer). Each command's `cmd_sync` names the
slot it reads (`src`) and the slot it fills (`dst`), with four flags:

| Flag | Effect |
|---|---|
| `ACQUIRE` | start only when `dst` is FREE, and take its credit |
| `SIGNAL` | on completion, emit `dst`'s token |
| `WAIT` | start only when `src` is READY |
| `RELEASE` | on completion, return `src`'s credit |

A LOAD acquires and signals its L2 slot (Conv2d: the first of a tile's per-position LOADs
acquires, the last signals); PRELOAD and COMPUTE wait for and release their L2 tile; COMPUTE
acquires the accumulator slot on a C tile's first K step and signals it on the last; STORE
waits for and releases it. `cmd_sync = 0` is an untracked command, which starts when its
process is idle, as before. New ports: `cmd_sync`, `{load,store,exec}_started[_id]` (the CSP
action starts, matching `*_completed`), the slot state bits (`l2_free`, `l2_ready`,
`acc_free`, `acc_ready`) and a sticky `sync_error`.

Matmul and Conv2d use the slots in ring order (`isa.command.TileBuffers`): with two A/B slot
pairs and two accumulator slots (`buffers=2`, the default) the BlockMover fills the next
tile's slot while the Streamer works on the current one, and STOREs overlap the next C tile.
Issuing in program order cannot deadlock, since every credit a command waits for is returned
by a command issued earlier. Pipelining tiles through the fabric itself, with L1 stream
buffers, is #15.

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

`tests/unit/test_credits.py` replays every command's push, start and completion through an
independent slot model each cycle (credit conservation, no tile consumed before its token, no
credit taken twice, every slot free at the end) and measures the overlap: an 8x8x16 Matmul on
a 4x4 array takes 686 cycles with two buffers against 1091 with one, bit-identical, and L3
backpressure changes no value.

[The original implementation plan](plan/isa-instructions.md) is superseded by this page.
