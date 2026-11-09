# ISA Instructions

`Matmul` (`C = A @ B`, plus a bias D or onto the existing C) and `Conv2d`
(`Y = conv2d(X, F)`, NHWC, as an implicit GEMM, plus a bias row or onto the existing Y) tile an
operation larger than the array and issue the same commands a host would, on a command port
that `isa.command.connect_to_top` wires to `SystolicTop`:

- every command names its process (`cmd_type`) and that process's opcode, and is pushed as
  soon as the process's queue has room, never waiting for a completion: credits and tile-ready
  tokens order the commands ([below](#credits-and-tile-ready-tokens)). An instruction is done
  when its last STORE completes (`cmd_store_completed`);
- Matmul runs OS, WS or IS (`CONFIG_DATAFLOW`; OS by default when the array has it, else WS,
  else IS), and the LOADs' corner turns lay each tile out as the Streamer streams it. OS: per
  K tile, LOAD A (row-major) and B (TRANSPOSE: column-major), one COMPUTE, the K tiles chained
  into one reduction in the PEs. WS: LOAD A (TRANSPOSE) and B, PRELOAD B, COMPUTE A (later K
  tiles accumulate). IS: the same loads, PRELOAD A, COMPUTE B, and the STORE's corner turn
  writes C from the Cᵀ the accumulator holds. Then STORE the C tile;
- Conv2d runs OS: its per-position LOADs write the A tile row-major, which only OS streams
  (WS and IS stream A column-major), and the filter tile is corner-turned. An array without
  OS cannot run Conv2d yet (a column-write LOAD is the follow-up,
  branes-ai/embodied-ai-soc-generator#25);
- Conv2d walks one K tile per kernel offset (kh, kw) and chunk of input channels. Each output
  position loads the input pixel that offset reads, with stride, dilation and padding applied;
  a position in the padding is a `CLEAR` of its L2 row, with no L3 read;
- a bias or accumulating onto the existing output starts each output tile from loaded values
  ([below](#bias-and-accumulate)).

## Data movement (the CSP roles)

Systars follows the KPU's CSP model ([kpu-sim](https://github.com/stillwater-sc/kpu-sim),
`docs/01-architecture/kpu-execution-model.md`): `SystolicTop` is a systolic KPU compute tile.
Its AXI port is its link to **L3** (a memory tile in a SoC; the AXI memory model in tests), the
scratchpad is **L2**, and the stream and result buffers at the array's edges are **L1**. LOAD and STORE are the
**BlockMover**'s two legs (L3 → L2, and accumulator → L3); the execute controller is the
**Streamer** (L2 → L1).

A LOAD or STORE moves a whole tile: `len` rows, row `n` at L3 byte address
`dram_addr + n * stride` (`cmd_stride`; 0 = one bus beat), `row_bytes` bytes each
(`cmd_row_bytes`; 0 = a whole row). Rows may start at any byte: a row that starts mid-beat, or
spills into the next beat, is a 1- or 2-beat burst from the beat boundary, shifted into place.
A loaded row is zero-filled past `row_bytes`; a stored row writes only its own bytes (AXI byte
strobes, `StreamWriter.data_strb`). Rows that are consecutive beats from a beat boundary stream
as bursts of up to 256 beats.

No burst crosses a 4 KB boundary, which AXI forbids (#11):

- a streamed burst ends at the boundary, and the next one starts there;
- a row whose two beats are in different 4 KB pages is two 1-beat bursts. For a STORE, the
  second burst is issued once the first is acknowledged.

The unit AXI memory model (`tests/unit/_axi.py`) asserts this for every burst a test issues.

The LOAD leg also writes the accumulator: a `local_addr` with the accumulator flag
(`LocalAddr`, bit 31) writes accumulator rows (`row_bytes` 0 is then a whole accumulator row),
the starting values of a result tile. The accumulator has one write port and the Streamer has
it first, since the fabric never stalls: an accumulator row waits for a clock in which the
Streamer writes none. Two more LOAD opcodes (`DmaOpcode`) fill rows without moving a tile:

| Opcode | Effect |
|---|---|
| `BROADCAST` (0x05) | reads one L3 row and writes it to all `len` rows (kpu-sim's `STR_BROADCAST_ROW`): a bias row for every output row |
| `CLEAR` (0x06) | writes `len` zero rows, with no L3 read (a state operation, kpu-sim ADR 0002): a padded position's row |

The corner turn writes only the scratchpad: a `TRANSPOSE` into the accumulator is not accepted.

## Corner turns

A LOAD or STORE with opcode `TRANSPOSE` (`DmaOpcode`, 0x04) is a MEMCPY that turns one tile's
rows into its columns, for the layouts the Streamer reads: each L2 row fills one L1 stream
buffer. The LOAD leg collects the tile's L3 rows in a corner-turn buffer (`lanes` rows of
`lanes` elements of `input_bits`, `lanes = sp_width // input_bits`), then writes scratchpad
row `sp_addr + j` with element `j` of every L3 row: one row per element of an L3 row, zero
past the `len`-th element. The STORE leg reads the tile's accumulator rows into its own
buffer, back to back, and writes L3 row `i` with element `i` of each: the results an
input-stationary schedule produces as columns of C reach L3 as its rows. The rows still move
as a MEMCPY's do (any address, stride, partial rows, bursts), and STORE's activation applies
per element. A TRANSPOSE moves at most one tile: `len` is clamped to `lanes`. The buffers are
never cleared, since each turn leaves nothing that a later one could write.

## The fabric, L1 and the Streamer

`R` and `C` are the array's rows and columns in PEs (`grid_rows * tile_rows`,
`grid_cols * tile_cols`), and `KB` the longest stream block (an L2 row: `sp_width /
input_bits` elements). An accumulator row holds `C` results.

**The fabric.** Every PE has an input register on each of its links, so a value moves one PE
per clock in every direction and a PEArray is only a structural grouping (it scales to 8x8
up to 64x64 PEs). A moves east with three markers (`valid`, `first`, `last`); B and partial
sums move south; OS results move north. Two modes, set by `CONFIG`:

- **Output-stationary (OS).** A PE accumulates `c += a * b`; `first` overwrites, so a
  reduction needs no clear and may span several commands. On `last` it writes `c + a * b`
  into the `c_up` register of the PE to its north, and the results move north one PE per
  clock into the column's L1 result buffer: PE(i, j)'s result leaves the north edge in clock
  `s + K + j + 2i`, so a column receives one every other clock.
- **Stationary (WS and IS).** One stationary register per PE, loaded through a registered
  chain down each column; the streamed operand moves east and `psum + a * w` moves south. WS
  holds B and streams A; IS holds Aᵀ and streams B, producing Cᵀ. They are one fabric mode:
  only the operand layouts differ.

**L1** (`memory/stream_buffer.py`). One stream buffer per array row (A) and per column (B),
each one block (an L2 row) with one credit. The Streamer writes a block in parallel; the
buffer streams it out one element per clock and takes the next block in the clock its last
element leaves. The buffers are loaded one per clock, so buffer `i` starts one clock after
buffer `i - 1`: that stagger is the skew, and the array has no skew or deskew registers. At
the edge the results leave (north for OS, south for stationary), one result buffer per
column turns the columns back into row-major rows for the accumulator.

**The Streamer** (`controller/execute.py`) reads one L2 row per clock on each of the
scratchpad's two read ports into L1:

| Command (`ExecOpcode`) | OS | Stationary (WS, IS) |
|---|---|---|
| `CONFIG` | `rs1` = 0 selects OS; 1 (WS) or 2 (IS) selects the stationary mode | |
| `PRELOAD` | does nothing | the stationary tile, row-major, from `rs1`: `k_dim` rows (at most `R`), zero below |
| `COMPUTE` | an A tile row-major at `rs1` (`R` rows), a B tile column-major at `rs2` (`C` rows), `K = k_dim <= KB`; `cmd_chain` chains K tiles into one reduction | a streamed tile column-major at `rs1` (`R` rows), `M = k_dim <= KB`; `M` result rows |
| `COMPUTE_ACCUMULATE` | as `COMPUTE`, adding the result rows to the accumulator | |

`cmd_chain` (a `SystolicTop` port): bit 0 = continues the previous command's reduction, bit 1
= continues into the next (no results yet); 0 is a whole reduction. A command starts only
when the fabric can take it on schedule and never stalls:

- a command's reads take `I = max(K, R, C)` clocks (stationary: `max(M, R)`), and the next
  command starts `I` clocks later. With `K >= R, C` the next tile's blocks reach L1 in the
  clock the previous ones finish: no bubbles between tiles;
- **the OS drain**: a command that ends a reduction starts so that its results leave `2R - 1`
  clocks after the previous reduction's, so results never meet on a north link (a collision
  flag checks it). With `K >= 2R - 1` (chained K tiles count) this is free;
- `PRELOAD` waits until the previous stationary stream has left the array.

Two in-order events carry each command's `cmd_sync`: **completed** when its last block is in
L1 (`SystolicTop` returns the L2 credit, so the BlockMover can refill the slot early) and
**result** (`exec_result`) when its results are in the accumulator (the result token). An
untracked command (`cmd_sync` 0) runs alone and completes with its results, so a host can
issue one command at a time. `l1_error` (sticky) reports a schedule violation, which a valid
schedule never raises.

Verified by `tests/unit/test_systolic_array.py` and `test_pe*.py` (the result clocks, back to
back tiles, the `2R - 2` collision), `test_stream_buffer.py`, `test_execute_controller.py`
(no bubbles within or between tiles, the drain spacing, the events) and
`test_gemm_systolic.py` (LOAD → COMPUTE → STORE through `SystolicTop`: OS with chained K
tiles, WS, and IS with a STORE corner turn). The L2 behind the Streamer is still the
idealized two-read-port scratchpad; the banked L2 that sustains two vectors per clock is #17.

## Credits and tile-ready tokens

`SystolicTop` has an in-order command queue per process (`ld_queue_length`,
`ex_queue_length`, `st_queue_length`), and a queue's head is dispatched only when its credits
and tokens are there (`controller/sync.py`, after kpu-sim's `credit_pool` and `tag_cam`). L2
and the accumulator hold whole tiles in slots (`l2_slots`, default 4; `acc_slots`, default 2),
one credit per slot: there are no row or burst credits, since the BlockMover sequences its own
bursts inside a tile. A slot is FREE (its credit is available), OWNED (a producer is filling
it), READY (its tile-ready token waits for the consumer) or READING (a consumer took the token
at its start and is reading the tile, so the next consumer of the slot waits for the next
tile's token). An accumulator slot can also be LOADED: an accumulator LOAD filled it with a
tile's starting values and handed it to the Streamer. Each command's `cmd_sync` names the
slot it reads (`src`) and the slot it fills (`dst`), with five flags:

| Flag | Effect |
|---|---|
| `ACQUIRE` | start only when `dst` is FREE, and take its credit |
| `SIGNAL` | on completion, emit `dst`'s token (the Streamer: on its result event, so a Streamer command with `SIGNAL` must write a result; one that writes none sets `l1_error`) |
| `WAIT` | start only when `src` is READY (with `RELEASE`: take the token, READY → READING) |
| `RESUME` | with `ACQUIRE`: start only when `dst` is LOADED, and take its hand-off token (LOADED → OWNED) instead of a credit |
| `RELEASE` | on completion, return `src`'s credit (the Streamer: once the tile's last block is in L1) |

A LOAD acquires and signals its L2 slot (a tile filled by several LOADs: the first acquires,
the last signals; an OS COMPUTE reads both tiles of its pair, so an OS pair is one slot,
acquired by the A tile's LOAD and signalled by the B tile's); PRELOAD and COMPUTE wait for and
release their L2 tile; COMPUTE acquires the accumulator slot on a C tile's first K step and
signals it (on its result event) on the last; STORE
waits for and releases it. `cmd_sync = 0` is an untracked command, which starts when its
process is idle, as before. New ports: `cmd_sync`, `{load,store,exec}_started[_id]` (the CSP
action starts, matching `*_completed`), the slot state bits (`l2_free`, `l2_ready`,
`acc_free`, `acc_ready`) and a sticky `sync_error`.

A tile that starts from loaded values has two producers in turn. The accumulator LOAD
acquires the accumulator slot (FREE → OWNED) and its `SIGNAL` hands the tile on (OWNED →
LOADED, the hand-off token); the Streamer's first COMPUTE of the tile acquires it with
`RESUME` (LOADED → OWNED), and its last signals the result token as before. A STORE waits for
READY, so it never mistakes the hand-off token for the result.

Matmul and Conv2d use the slots in ring order (`isa.command.TileBuffers`): with two A/B slot
pairs and two accumulator slots (`buffers=2`, the default) the BlockMover fills the next
tile's slot while the Streamer works on the current one, and STOREs overlap the next C tile.
Issuing in program order cannot deadlock, since every credit a command waits for is returned
by a command issued earlier.

## Layouts

A, B and C (Matmul) and X, F and Y (Conv2d) are dense, at any address, with any M, N and K:
START rejects (`UNSUPPORTED_LAYOUT`) only a zero stride, or output rows that would overlap.

## Bias and accumulate

A bias, or accumulating onto the existing output (`CONFIG_OPTIONS` bit 0), starts each output
tile from loaded values. On the tile's first K tile, after that K tile's LOADs, one accumulator
LOAD writes them into the tile's accumulator slot and hands it to the Streamer, and the K tiles'
results are added onto them (`COMPUTE_ACCUMULATE`):

| Instruction | Setting | Loaded into the tile |
|---|---|---|
| Matmul | `CONFIG_D_ADDR` != 0, `CONFIG_D_STRIDE` > 0 | the D tile: int32, N per row, rows `D_STRIDE` bytes apart (`C = A @ B + D`) |
| Matmul | `CONFIG_D_ADDR` != 0, `CONFIG_D_STRIDE` = 0 | one bias row of N int32 values, in every row (`BROADCAST`) |
| Matmul | `CONFIG_OPTIONS` bit 0 | the C tile (`C = C + A @ B`) |
| Conv2d | `CONFIG_B_ADDR` != 0 | one bias row of `out_c` int32 values, in every output position (`BROADCAST`) |
| Conv2d | `CONFIG_OPTIONS` bit 0 | the Y tile (`Y = Y + conv2d(X, F)`) |

OS adds the reduction held in the PEs onto the loaded tile when it writes its results; WS
accumulates from the first K tile. IS holds Cᵀ in the accumulator, so its loaded values would
need a corner turn into it: Matmul rejects a bias in IS (`UNSUPPORTED_BIAS`) and accumulating
in IS (`UNSUPPORTED_ACCUMULATE`). A bias and accumulating together is `UNSUPPORTED_ACCUMULATE`
for both instructions (add the bias into the output first). Conv2d's padding no longer needs a
zero row in L3: `CONFIG_ZERO_ADDR` is gone, and a padded position is a `CLEAR`.

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
bursts past 256 beats, a GEMM with unaligned, strided operands, and rows and streamed tiles
that cross a 4 KB boundary, for both legs and a GEMM (`TestFourKBoundaries`).

`tests/unit/test_load_controller.py`: accumulator LOADs (streamed and unaligned rows),
`BROADCAST` (one L3 read) and `CLEAR` (none), with the Streamer holding the accumulator's write
port on random clocks.

`tests/unit/test_matmul.py` and `tests/unit/test_conv2d.py`: the issued commands run through a
Python model of `SystolicTop` (`tests/unit/_isa.py`) and match numpy (Matmul tiled in M, N and
K, with a D matrix, a bias row, or onto C; Conv2d with padding, stride, dilation, several
batches, channel chunks and N tiles, with a bias row or onto Y), and both instructions drive a
real `SystolicTop` to a correct dense result, leaving every other byte untouched. On the real
tile, Conv2d reads L3 only inside its operands, and once per X row that is not padding.

`tests/unit/test_credits.py` replays every command's push, start and completion through an
independent slot model each cycle (credit conservation, no tile consumed before its token, no
credit taken twice, every slot free at the end; the Streamer's events matched to their
commands by ID) and measures the overlap: an 8x8x16 Matmul on a 4x4 array takes 744 cycles
with two buffers against 907 with one (OS; WS and IS measure the same), bit-identical, and L3
backpressure changes no value. The Streamer returns a tile's L2 credit a few clocks after it
starts, so the run is bound by the BlockMover's L3 traffic. With a bias or accumulation the
monitor follows each accumulator slot through LOADED, and no STORE starts on a hand-off
token.

[The original implementation plan](plan/isa-instructions.md) is superseded by this page.
