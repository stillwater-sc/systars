# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Accumulator loads: Matmul and Conv2d with a bias or accumulating onto the output, and
  padding without a zero row** (#10, 2026-11-08; the port of
  branes-ai/embodied-ai-soc-generator#17 goes the other way, systars first). The BlockMover's
  LOAD writes the accumulator when its local address has the accumulator flag, sharing the
  accumulator's write port with the Streamer, which has it first. Two LOAD opcodes:
  `DmaOpcode.BROADCAST` writes one L3 row to every row (kpu-sim's `STR_BROADCAST_ROW`), and
  `DmaOpcode.CLEAR` writes zero rows with no L3 read. An accumulator slot gains a LOADED
  state and `cmd_sync` a `RESUME` flag (`controller/sync.py`): an accumulator LOAD hands its
  slot to the Streamer, whose first COMPUTE takes the hand-off token, so a STORE never takes
  it for the result. Matmul: `CONFIG_D_ADDR` / `CONFIG_D_STRIDE` (`C = A @ B + D`, a stride
  of 0 being one bias row for every output row) and `CONFIG_OPTIONS` bit 0 (`C = C + A @ B`)
  in OS and WS; IS still rejects them (`UNSUPPORTED_BIAS`, `UNSUPPORTED_ACCUMULATE`), since
  it holds Cᵀ in the accumulator. Conv2d: a bias row (`CONFIG_B_ADDR`, broadcast) and
  accumulating onto Y; a padded position is a `CLEAR`, so `CONFIG_ZERO_ADDR` is removed and
  padding reads nothing from L3. A bias with accumulate is `UNSUPPORTED_ACCUMULATE`.
  `tests/unit/test_load_controller.py`, `test_credits.py`, `test_matmul.py`,
  `test_conv2d.py`; `docs/isa-instructions.md`, "Bias and accumulate".

- **The PolarFire flow drops memory initial values too, and a test runs GEMMs from random
  memory** (#27, 2026-11-06; kpugen parity, branes-ai/embodied-ai-soc-generator#38). The
  `polarfire` prep in `just synth` is now `proc; delete t:$meminit_v2; attrmap -remove
  init;`, the same as kpugen's, whose tile accumulator otherwise lands in initialized
  flip-flops. On hardware, memories start unspecified, not at zero: the design never reads a
  scratchpad or accumulator word before writing it, and the host must write before it reads
  (`docs/development-workflow.md`, "Synthesis"). `test_no_read_before_write`
  (`tests/unit/test_gemm_systolic.py`) fills every scratchpad and accumulator word with
  full-width random data and checks OS, WS and IS GEMMs, including `COMPUTE_ACCUMULATE`,
  stay exact; `tests/unit/test_targets.py` checks that no target carries memory contents.
  The scratchpad and accumulator banks (and `SystolicTop`) keep references to their
  memories for testbenches; the hardware is unchanged.

- **`just synth <target>` for every module, and a PolarFire target** (#25, 2026-11-06).
  `systars.targets` is the one list of RTL targets (name, top module, factory) that
  `just gen`, `just synth` and CI read (`python3 -m systars.targets list`), ported from
  kpugen. `just gen` now also writes the Scratchpad and the Accumulator. `just synth
  generic|ice40|ecp5|polarfire` maps every target onto the device's cells and prints the
  counts, and `just synth-report` writes the resource report. The PolarFire flow is Yosys's
  `synth_microchip -family polarfire`, after dropping the registers' `init` values, which
  PolarFire flip-flops don't have. CI's synthesis jobs now call these recipes, with
  `polarfire` added to the matrix, instead of synthesizing only the PE inline. The
  [PolarFire plan](docs/plan/polarfire-deployment.md) has Yosys's counts for a 16x16 array.

- **Wavefront animations: a 2x2 -> 4x4 -> 16x16 progression from the RTL** (#23,
  2026-11-05). `systars.wavefront` records one tile or several back to back, started on the
  Streamer's schedule (`streamer_starts`, checked against the Streamer in
  `tests/unit/test_execute_controller.py`), and checks every cell by value, so any array
  size works, with any operands that fit the PE's input widths (`label_operands` and its
  int8 limit are gone; `A @ B` is checked in int64, so int8 operand arrays don't wrap). `render_svg` draws
  a `value` view (operands, running sums, C filling in, `C == A @ B` on screen), an `index`
  view (`kN` / `mN`, with the L1 stream buffers feeding the edges) and a `front` view
  (cells coloured by `k` / `m`, results moving north, PEs busy per clock), with captions
  that name the clock and phase and that dwell on the first product, the first result and
  each next tile. Each cell is one SMIL animation, and cells that step through the same
  colours share one, so the 16x16 three-tile OS animation is about 320 KB. Theme-aware
  colours (`prefers-color-scheme`) and a viridis palette. `STAGES` defines the docs'
  progression. `scripts/gen_wavefronts.py` / `just wavefronts` write the committed SVGs in
  `docs/wavefront/`, and `just wavefronts-check` and `tests/unit/test_wavefront.py` fail when
  they are stale. `examples/gemm/08_rtl_wavefront.py` gains `--stage`, `--view`, `--tiles`
  and `--small`. The wavefront page is rewritten around the progression: the reduction
  (2x2), the stagger from the L1 stream buffers (4x4) and the schedule (16x16, back-to-back
  tiles and the north drain), OS beside WS; the textbook model (examples 02, 03) is now a
  labelled aside. `docs/rtl-wavefront-{os,ws}.svg` are replaced.

- **A registered fabric, L1 stream buffers and a bubble-free Streamer** (#15, 2026-11-03;
  ported from branes-ai/embodied-ai-soc-generator#21). Every PE has an input register on each
  link, so values move one PE per clock and a PEArray is only a structural grouping (8x8 up
  to 64x64 PEs). OS results move north on the `last` marker into L1 result buffers; `first`
  overwrites, so a reduction spans commands (`cmd_chain`) with C held in the PEs. WS and IS
  share one stationary mode. L1 (`memory/stream_buffer.py`): one-block stream buffers per row
  and column with one credit each, whose staggered starts are the skew, and result buffers
  that turn columns back into rows. The Streamer (`controller/execute.py`) starts a command
  only when the fabric can take it on schedule and never stalls: with `K >= R, C` tiles
  stream back to back, and a reduction's end waits for the north drain (`2R - 1` clocks
  between reductions, free with `K >= 2R - 1`). It reports each command's L2 release
  (`exec_completed`, once its last block is in L1) and its result (`exec_result`)
  separately; `l1_error` is sticky. Matmul runs OS, WS or IS (`CONFIG_DATAFLOW`, OS by
  default), laying tiles out with the corner turns; Conv2d runs OS.
  `tests/unit/test_execute_controller.py`, `test_stream_buffer.py`,
  `test_systolic_array.py`, `test_gemm_systolic.py`

- **Corner turns in the BlockMover** (#15; ported from
  branes-ai/embodied-ai-soc-generator#21). LOAD and STORE take the new
  `DmaOpcode.TRANSPOSE` (0x04), a MEMCPY that turns one tile's rows into its columns through
  a corner-turn buffer of one tile: LOAD writes the tile column-major into L2 (one L2 row per
  L1 stream buffer, for OS's B, WS's A and IS's stationary A), and STORE turns an
  input-stationary result (columns of C) back into row-major rows in L3. Any address,
  stride and partial rows, as MEMCPY; `len` is clamped to one tile.
  `tests/unit/test_block_mover.py`

- **Credits and tile-ready tokens** (#9, 2026-11-02; ported from
  branes-ai/embodied-ai-soc-generator#16). `SystolicTop` queues each process's commands
  (BlockMover LOAD and STORE, Streamer) and dispatches a queue's head only when its credits and
  tokens are there (`controller/sync.py`: `SlotPool`, `Sync`). L2 and the accumulator hold
  whole tiles in slots (`l2_slots`, `acc_slots`), one credit each; a command's `cmd_sync`
  acquires and signals the slot it fills and waits for and releases the slot it reads. Matmul
  and Conv2d push their commands ahead with no completion round trip, double-buffered
  (`buffers=2`): an 8x8x16 Matmul on a 4x4 array drops from 1091 to 686 cycles with
  bit-identical results. New ports: `cmd_sync`, `{load,store,exec}_started[_id]`, the slot
  state bits and `sync_error`. `tests/unit/test_credits.py`

- **The BlockMover moves a tile from or to any byte address** (#8, 2026-11-02; ported from
  branes-ai/embodied-ai-soc-generator#14). Following the KPU's CSP model (kpu-sim), the AXI
  port is `SystolicTop`'s link to L3 and LOAD/STORE are the BlockMover's two legs. A LOAD or
  STORE now moves a whole tile: `len` rows, `stride` bytes apart (`cmd_stride`; 0 = one beat),
  `row_bytes` bytes each (`cmd_row_bytes`; 0 = a whole row). A row may start at any byte (a 1-
  or 2-beat burst, shifted into place); a loaded row is zero-filled past `row_bytes`, and a
  stored row writes only its own bytes, with AXI byte strobes (`StreamWriter.data_strb`).
  Consecutive beats from a beat boundary still stream as one burst (up to 256 beats).
  `tests/unit/test_block_mover.py`. Bursts were not yet split at 4 KB boundaries; that is
  fixed by #11 (see Fixed)
- **Matmul and Conv2d on dense layouts** (#8): Matmul computes any M, N and K with dense
  row-major A, B and C at any address, and Conv2d dense NHWC X, F and Y, both verified end to
  end through a real `SystolicTop`. They issue one LOAD per operand tile (Conv2d: one per
  output position, and one per filter tile) and one STORE per output tile

- **RTL wavefront animation** (`src/systars/wavefront.py`, `examples/gemm/08_rtl_wavefront.py`,
  2026-11-01; ported from branes-ai/embodied-ai-soc-generator#12). Runs the real
  `SystolicArray` in simulation, records what every PE sees in every cycle, and checks every PE
  in every stream cycle (and `out_last`, and the result) against the predicted wavefront. Renders text frames and an animated SVG, each cell showing which k or
  m is there (read back from the RTL's values), with the PEArray boundaries marked.
  `tests/unit/test_wavefront.py` runs the check on 7 shapes, OS and WS, and shows that a skew
  bug is caught. Two SVGs are in [the wavefront page](docs/wavefront-animation.md#how-the-animations-are-made-and-checked) (since replaced by the progression, #23).
  The existing examples 02–05 animate a Python model, not the RTL

### Fixed

- **The tile's bursts never cross a 4 KB boundary** (2026-11-09, #11; port of
  branes-ai/embodied-ai-soc-generator#18). AXI forbids a burst that crosses one, and the
  BlockMover's LOAD and STORE legs could issue them. A streamed burst now ends at the
  boundary. A row that spills from the last beat of a 4 KB page into the next is now two
  1-beat bursts; a STORE issues the second once the first is acknowledged.
  - The unit AXI memory model (`tests/unit/_axi.py`) asserts that no read or write burst
    crosses 4 KB, and that `WLAST` marks a write burst's last beat. It can also log every
    burst. `AXI_4KB` is in the new `systars.dma.axi`.
  - New tests: rows spilling across a boundary and streamed tiles crossing one, for both
    LOAD and STORE, and a GEMM with each operand straddling a boundary
    (`TestFourKBoundaries` in `tests/unit/test_block_mover.py`). kpugen's cocotb
    `systolic_tile` tests have no systars counterpart yet (#12).

#### The Matmul and Conv2d ISA instructions issue commands the tile executes (2026-11-01)

Ported from branes-ai/embodied-ai-soc-generator#11, items 4–6 of #2. They emitted private
opcodes (`EXEC_PRELOAD` 0x01 and `EXEC_COMPUTE` 0x02, which the execute controller does not
decode, while their `LOAD_A` 0x10 is its `PRELOAD`), a CONFIG dataflow of 0/1/2 that the
controller reads as 0/1, and scratchpad and accumulator addresses with the bank in
`bank << 16`, where the memories decode it from the low bits. Now:

- both issue on a shared command port (`isa/command.py`): the target controller (`cmd_type`)
  and that controller's own opcode (`DmaOpcode`, `ExecOpcode`), with `SystolicTop`'s fields.
  `connect_to_top` wires an instruction to a `SystolicTop`
- they issued one command at a time and waited for its completion (`cmd_completed`); since #9
  they push commands ahead, ordered by credits and tokens
- both run weight-stationary, loading one matrix row per scratchpad row at its exact address,
  with tile buffers at consecutive `LocalAddr` rows
- *Conv2d addressing*: the output position (batch, row, column) of each M-tile row was never
  decoded, the kernel offsets were dropped, and stride, padding and dilation were never
  applied. Conv2d is now an implicit GEMM: one K tile per kernel offset and chunk of input
  channels, each row loading the input pixel that offset reads, or a zero row
  (`CONFIG_ZERO_ADDR`) in the padding. Its M-tile counters are 40 bits wide
  (`batch * out_h * out_w` overflowed 16)

`tests/unit/test_matmul.py` and `tests/unit/test_conv2d.py` run the issued commands through a
Python model of `SystolicTop` (`tests/unit/_isa.py`) and match numpy; Matmul also drives a real
`SystolicTop` to a correct C. See [ISA instructions](docs/isa-instructions.md).

#### SystolicTop computes a GEMM end to end (2026-11-01)

Ported from branes-ai/embodied-ai-soc-generator#9, items 1–3 and 7 of #2. `SystolicTop` now
runs LOAD A and B, CONFIG, PRELOAD, COMPUTE and STORE C to a verified `A @ B`, output-stationary
and weight-stationary. The previous execute path could not:

- *B was never read*: both operands shared the scratchpad's one read port, and the address mux
  preferred A. The scratchpad has a second read port (in every bank), and A and B stream at one
  row per cycle
- *COMPUTE never completed*: it waited for an `out_last` pulse that had already passed, and
  dropped the outputs produced meanwhile
- *A, B and control were skewed inconsistently*: A was registered in every PE and at PEArray
  boundaries, B only at boundaries, and `valid` not at all across PEArray columns. Now every
  signal crossing a PEArray boundary passes through one register, and `SystolicArray` skews its
  inputs and deskews its outputs itself
- *B-stationary held no weight*: the PE multiplied the incoming partial sum. The PE now has a
  weight register, loaded by PRELOAD, and partial sums flow down
- *OS results could not be read out*: the PE ignored `in_d` in OS. Results now drain down each
  column through a shift chain, bottom row first

### Changed

- **The GitHub Pages site is built with Astro Starlight** (#20, 2026-11-04), the same
  setup as stillwater-sc/universal and stillwater-sc/mtl5, replacing MkDocs. Pages are
  still written in `docs/`; `docs-site/sync-content.mjs` generates the site's content
  from them and from this changelog, and fails on a relative link that resolves to
  nothing or a top-level page with no place in the sidebar. `starlight-links-validator`
  fails the build on a broken internal link or anchor. `just docs` / `just docs-serve`
  build and serve the site (Node >= 22.23.4). CI builds it on docs pull requests and
  deploys it from `main` with SHA-pinned actions. Removed: `mkdocs.yml` and the `docs`
  extra in `pyproject.toml`. Fixed: the README's link to the implementation plan.
- **Streamer commands** (#15): OS COMPUTE takes an A tile row-major and a B tile
  column-major, `K <= KB` per command (chain longer reductions); stationary COMPUTE streams a
  column-major tile. The array's latency contract is replaced by per-edge result clocks
  (`SystolicArray.os_result_clock`, `stationary_result_clock`); the shift-chain ports
  `in_d`/`out_c` become `in_w`/`out_w` and the north results `out_c`/`out_c_valid`.
- **Conv2d runs output-stationary** (#15): an array without OS (`LEAN_CONFIG`,
  `CHIP_CONFIG`) cannot run Conv2d until a column-write LOAD exists
  (branes-ai/embodied-ai-soc-generator#25).
- **A consumer takes a slot's token at its start** (#15): `SlotState.READING`, so with
  several Streamer commands in flight the next consumer of a slot waits for the next
  tile's token. Measured, 8x8x16 Matmul on a 4x4 array: 907 -> 744 cycles with one and two
  buffers (was 1091 -> 686): the Streamer returns L2 early, and the run is BlockMover-bound.

- **ISA command port** (#9): `cmd_completed` is replaced by `cmd_store_completed` (an
  instruction only needs to know its STOREs are done) and the port gains `cmd_sync`.
  `SystolicTop.cmd_ready` now means the target process's queue has room, not that its
  controller is idle, and `{load,store,exec}_busy` include queued commands. Matmul and Conv2d
  place tiles in ring-ordered slots (`isa.command.TileBuffers`) instead of fixed scratchpad
  rows, and take `buffers` (default 2)

- **ISA layout check** (#8): `require_beat_aligned` is gone from `Matmul` and `Conv2d`, and
  START reports `UNSUPPORTED_LAYOUT` only for a zero stride or output rows that would overlap.
  C and Y rows no longer need to hold whole N tiles. The ISA examples use dense operands
  (`07_isa_matmul.py` loses `--any-layout`)
- **Bank counts** (#8): `sp_banks` and `acc_banks` must be powers of two, and one bank takes
  no address bits. With one bank, every other row selected a bank that did not exist; with a
  count that is not a power of two, some rows did
- **ISA instructions**: `InternalOpcode` and the `dbg_sp_bank_*`/`dbg_acc_bank` ports are gone;
  `cmd_rs1`/`cmd_rs2`/`cmd_rd` are 32-bit and carry execute operands only. START reports
  `UNSUPPORTED_LAYOUT` when the DMA could not move a row (today N and K must each fit one tile
  for Matmul, and real convolutions do not fit; `require_beat_aligned=False` skips the check),
  and `UNSUPPORTED_BIAS` for a bias. `UNSUPPORTED_ACCUMULATE` rejects the accumulate option
  (`CONFIG_OPTIONS` bit 0): C or Y is never loaded into the accumulator, so it would have added
  stale rows from an earlier tile or instruction. Conv2d gains `CONFIG_ZERO_ADDR`, and `CONFIG_Y_STRIDE` is
  the byte distance between output positions. The ISA examples (`examples/gemm/07_isa_matmul.py`,
  `examples/conv/01_isa_conv2d.py`) are rewritten against the new port, with a cycle limit
- **Execute semantics**: `COMPUTE` overwrites its destination rows and `COMPUTE_ACCUMULATE` adds
  to them. OS computes `C[R x C] = A[R x K] @ B[K x C]` from Aᵀ and B rows; WS computes
  `C[M x C] = A[M x R] @ W` with `W` loaded by `PRELOAD` (which no longer reads the accumulator,
  and does nothing in OS). Details in `controller/execute.py`
- **PE ports**: `in_psum`/`out_psum` (WS partial sums) and `in_shift` (shift chain); the
  `propagate` double buffer, the rounding `shift`, `id` and the PE's pass-through registers are
  gone, and so is `PEWithShift` (unused)
- **Generated RTL**: the cocotb PEArray is 2x2, and `gen/systolic_array.v` is a 2x2 grid of 2x2
  PEArrays (4x4 PEs)
- **Unit-test simulations** (`tests/unit/_sim.py`) end when the testbench returns, with a cycle
  limit as a watchdog

#### Bugs found reviewing the kpugen port (2026-10-31)

These were found in code review of branes-ai/embodied-ai-soc-generator#1, which ports this
repository's systolic stack. Each RTL fix has a test that fails on the previous code.

- **Accumulator**: accumulate writes were lost when `accumulate` dropped after the request cycle
  (as `ExecuteController` drives it). Writes are now always pipelined, with a transparent
  read-modify-write read. Accumulate adds each element separately (no carries between packed
  elements), and activations apply per element (`memory/activation.py`)
- **Store/top**: activation was applied twice (store ReLU + accumulator read activation driven
  from the live command port) and to the whole packed row. It is now applied once, per element,
  from the latched command. Load/Store reject rows wider than `dma_buswidth` (were silently
  truncated)
- **SystolicTop**: per-controller completion ports; the summary `completed` drops simultaneous
  completions
- **ExecuteController**: CONFIG decodes the latched `rs1`, not the live port
- **DescriptorEngine**: transfers ended at half their length; multi-burst (> 256 beats) and
  zero-length transfers
- **StridedStreamReader / MatrixLoader**: could not elaborate (Python `if` on a `Signal`), packed
  elements at a 32-bit stride, and `MatrixLoader.busy` was never driven
- **SRAMReadScheduler**: missed the one-cycle `read_valid` pulse and hung
- **LocalAddr**: `make_sp_addr`/`make_acc_addr` used the bank value's bit length as the row shift;
  now instance methods using the configured bank bits
- **SystolicConfig** validates row/bank geometry; **Matmul/Conv2d** reject non-power-of-two dims;
  Conv2d no longer reports IDLE in `CALC_TILES`; `CONFIG_INPUT_DIMS` comment corrected
- **Examples**: int8 wraparound (03-06), scratchpad overlap in 01 for size > 16, `--step` crash
  in 02, workload-scaled cycle limit in conv 01, `run_all_gemm_demos.sh` runs from any directory
- **Scripts/CI**: OSS CAD Suite installs through a private temp dir, with archive verification and
  a staged replace (`bin/lib/oss-cad-suite.sh`); `$OSS_CAD_SUITE` honoured and on PATH for just
  recipes (after the venv); non-root `just` install; docs deploy concurrency and permissions;
  authenticated `setup-oss-cad-suite`; the Verilator pre-commit hook fails on errors; pytest
  discovery no longer collects cocotb tests
- **Tests**: PE tests now assert their results; new tests for accumulator row semantics,
  descriptor-engine transfers, the read scheduler, LocalAddr layout, and reader elaboration

### Added

#### Repository Velocity Report Script (2025-12-23)

- **Velocity Report Generator** (`scripts/repo_velocity.py`): Measures development velocity across multiple repositories
  - Tracks source code files only: C/C++, Python, Go, Shell, CMake, Verilog/VHDL
  - Excludes large files (>100k lines) that are likely generated content
  - Reports: total LOC, commits, lines added/deleted, net change, contribution percentage
  - Multiple output formats: markdown (default), CSV, JSON
  - Configurable date range via `--since` and `--until` arguments
  - Supports both branes-ai and stillwater-sc organization repos

### Fixed

#### OSS-CAD-SUITE PATH in Virtual Environment (2025-12-23)

- **Venv Activation Script**: Fixed missing Yosys/Verilator in PATH after activating .venv
  - `.venv/bin/activate` now automatically adds `/opt/oss-cad-suite/bin` to PATH
  - RTL generation (`just gen`) and synthesis checks now work without manual environment sourcing

- **Setup Script Enhancement** (`bin/setup-dev-env.sh`): Auto-patches venv activation
  - New step patches `.venv/bin/activate` to include OSS-CAD-SUITE path
  - Idempotent: skips if already patched
  - Ensures `just ci` works immediately after `source .venv/bin/activate`

#### MasPar MP-2 SIMD Implementation (2025-12-21)

- **SIMD Package** (`src/systars/simd/`): New package for classic SIMD architectures
  - Distinct from SIMT (no divergence handling, pure lock-step execution)
  - MasPar MP-1/MP-2 as first implementation

- **MasPar Simulator** (`src/systars/simd/maspar/`): Complete MP-2 SIMD array processor
  - `MasParConfig`: Hardware configuration (64x64 array, 64 registers/PE, 64KB local memory)
  - `PE`: Processing Element with 32-bit ALU, registers, local memory
  - `PEArray`: 2D mesh with 8-neighbor XNET connectivity (toroidal wrap-around)
  - `ACU`: Array Control Unit for instruction fetch/decode/broadcast
  - `MasParSim`: Top-level simulator with GEMM and Conv2D support

- **Instruction Set** (`src/systars/simd/maspar/instruction.py`): 40+ opcodes
  - Integer ALU: IADD, ISUB, IMUL, IAND, IOR, IXOR, ISHL, ISHR, IMOV
  - Floating-point: FADD, FSUB, FMUL, FFMA
  - Memory: LD, ST, LDI (load immediate)
  - XNET: 8 directions (N, S, E, W, NE, NW, SE, SW)
  - Router: ROUTE_SEND, ROUTE_RECV
  - Control: SETMASK, CLRMASK (PE masking)
  - Reduction: REDUCE_SUM, REDUCE_MAX, REDUCE_MIN
  - Comparison: ICMP_EQ/NE/LT/LE/GT/GE

- **Matrix Multiplication (GEMM)**: Cannon's algorithm implementation
  - Pre-skewed data distribution for parallel execution
  - XNET shifts: A west, B north each iteration
  - Performance: 4x4 GEMM in 29 cycles (16 instructions)

- **2D Convolution (Conv2D)**: 3x3 kernel support
  - Leverages XNET 8-neighbor mesh for efficient gather
  - 36 instructions: 9 LDI + 8 XNET + 1 init + 18 MAC
  - Kernels: identity, box, edge, sharpen, sobel_x, sobel_y

- **Animation Examples** (`examples/simd/maspar/`):
  - `01_animated_maspar.py`: GEMM visualization with Cannon's algorithm
  - `02_animated_conv2d.py`: Conv2D visualization with kernel selection

- **Pre-defined Configurations**:
  - `DEFAULT_MASPAR_CONFIG`: 64x64 MP-2
  - `SMALL_MASPAR_CONFIG`: 4x4 for testing
  - `MEDIUM_MASPAR_CONFIG`: 16x16
  - `LARGE_MASPAR_CONFIG`: 128x128
  - `MP1_CONFIG`: 64x64 with 4-bit ALU (8 cycles for 32-bit ops)

#### SM-Level LSU with MSHR Tracking (2025-12-20)

- **SM-Level Load/Store Unit** (`src/systars/simt/sm_lsu.py`): Restructured from per-partition LSUs to single SM-level LSU
  - Single MIO queue (16 entries) shared by all 4 partitions
  - 64 MSHRs tracking cache lines at 128B granularity
  - Primary/secondary miss optimization: multiple warps share same MSHR for same cache line
  - 1 IPC processing bandwidth to LSU pipe
  - Shared memory bypass (no MSHR tracking needed)
  - Statistics: `total_primary_misses`, `total_secondary_misses`, `total_mio_queue_full_stalls`

- **MSHR Data Structures**:
  - `MSHREntry`: Tracks cache line address, state, waiters, cycles remaining
  - `MSHRWaiter`: Partition ID, warp ID, thread mask, destination register, thread offsets
  - `MIOQueueEntry`: Queued memory request with partition/warp info, addresses, data

- **Config Parameters** (`src/systars/simt/config.py`):
  - `num_mshrs: int = 64` - Number of Miss Status Holding Registers per SM
  - `mio_queue_depth: int = 16` - Depth of MIO queue
  - `lsu_issue_bandwidth: int = 1` - Requests processed per cycle

- **Architecture Benefits**:
  - 64 MSHRs vs previous 8 outstanding requests (4 partitions × 2 pending)
  - Secondary miss optimization reduces redundant memory traffic
  - Cross-partition visibility for better load balancing
  - Matches NVIDIA's proven SM architecture

#### SIMT Visualization Enhancements (2025-12-20)

- **Timeline Logging** (`examples/simt/01_animated_simt.py`): Pipeline event tracing
  - `TimelineLogger` class for CSV-format event capture
  - Logs ISSUE, COMPLETE, and STALL events with cycle-accurate timing
  - `--timeline FILE` command line option to enable logging
  - Summary statistics printed at end of simulation

- **Store Completion Tracking**: Distinguish computed vs stored elements
  - `GEMMTracker` now tracks pending stores per warp
  - Matrix header shows status: `[computing]`, `[ST pending: N]`, or `[COMPLETE]`
  - Summary shows both "Computed: X/Y" and "Stored: X/Y" counts

- **Updated Matrix Visualization**: Clear visual state progression
  - `W#` (dim) = not started
  - `░░` / `▓░` = partial progress (some FFMAs done)
  - `▓▓` (yellow) = computed but ST pending
  - `██` (green) = truly complete (stored to memory)

### Fixed

#### Multi-Partition Warp Distribution (2025-12-20)

- **Warp Distribution Bug**: Fixed tiled GEMM loading all warps into partition 0 only
  - Now distributes warps across all 4 partitions (P0-P3)
  - 32×32 GEMM now uses all 128 ALU cores instead of just 32
  - Added global warp ID calculation for proper GEMM tracking across partitions

#### LSU Queue Overflow Bug (2025-12-20)

- **Memory Instruction Drops**: Fixed critical bug where LD/ST instructions were silently dropped
  - When LSU queue was full (max_pending=2), `issue()` returned False and instruction was lost
  - Warps continued executing with garbage register data, causing incorrect results
  - Fix: Added `pending_memory_requests` buffer to retry rejected instructions on subsequent cycles
  - Warps now properly stall until their memory request is accepted by LSU
  - Added `total_lsu_rejections` statistic to track queue-full events

#### ISA-Level Matmul Instruction (2025-12-18)

- **Matmul ISA Instruction** (`src/systars/isa/matmul.py`): High-level tiled matrix multiply
  - Configuration interface: dimensions (M, N, K), addresses, strides, options
  - Automatic dataflow selection: Output-Stationary, A-Stationary, B-Stationary
  - Tiled computation with edge tile handling for arbitrary matrix sizes
  - Internal command generation: LOAD_A, LOAD_B, LOAD_D, EXEC_CONFIG, EXEC_PRELOAD, EXEC_COMPUTE, STORE_C
  - Backpressure handling via valid/ready handshake on command interface
  - Double buffering with bank toggling for latency hiding

- **ISA Module** (`src/systars/isa/__init__.py`): New ISA instructions package

- **Matmul Example** (`examples/gemm/07_isa_matmul.py`): Interactive demonstration
  - Shows configuration, tiling, command generation, and dataflow selection
  - Supports configurable M, N, K dimensions and array size
  - Validates command counts against expected values

- **Matmul Unit Tests** (`tests/unit/test_matmul.py`): 13 comprehensive tests
  - Configuration interface tests
  - Address calculation for tiled operations
  - Dataflow selection heuristics
  - Command emission verification
  - Backpressure handling
  - Double buffering bank toggling
  - Multi-tile execution
  - Bias loading (LOAD_D)
  - Error handling

- **Implementation Plan** (`docs/plan/isa-instructions.md`): Design documentation for Matmul and Conv2d

### Changed

#### SIMT Package Reorganization (2025-12-20)

- **Multi-Architecture Structure**: Reorganized `src/systars/simt/` to support multiple architectures
  - `simt/nv/`: NVIDIA Streaming Multiprocessor (current implementation)
  - `simt/nv_v1/`: Legacy NVIDIA SM with per-partition LSU bug
  - `simt/amd/`: AMD RDNA/CDNA placeholder (NotImplementedError)
  - `simt/maspar/`: MasPar MP-1/MP-2 placeholder (NotImplementedError)

- **Backward Compatibility Layer** (`src/systars/simt/__init__.py`):
  - Re-exports all NVIDIA symbols for existing code: `from systars.simt import SMSim`
  - Explicit architecture selection now possible: `from systars.simt.nv import SMSim`

- **Common Protocols** (`src/systars/simt/base.py`):
  - `ProcessorSim`: Protocol for top-level processor simulation
  - `MemorySubsystem`: Protocol for memory read/write operations
  - `SIMTConfigBase`: Abstract base for architecture configurations

- **Round-Robin Warp Distribution Fix**:
  - Changed from sequential fill (P0=8, P1=8, P2=2, P3=0 for 18 warps)
  - To balanced round-robin (P0=5, P1=5, P2=4, P3=4 for 18 warps)
  - Fixed in both `sm_controller.py` and `examples/simt/01_animated_simt.py`

#### CI Improvements (2025-12-18)

- **Unit Tests CI**: Added OSS CAD Suite (Yosys) to unit tests job for full Verilog generation test coverage
- **Dependencies**: Added `numpy>=1.24` to dev dependencies for GEMM demo/e2e tests

### Fixed

#### Verilog Generation Tests (2025-12-18)

- Added skip-if-no-Yosys logic to 3 tests in `test_gemm_demo.py` and `test_gemm_e2e.py`
- All 23 Verilog generation tests now properly skip when Yosys unavailable (safety net)

#### Skew Buffer Timing Visualization (2025-12-17)

- **Interactive Step Mode**: `--step` flag for manual single-stepping through animation
- **Dynamic Sizing**: `--k` now defaults to `--size` for intuitive square matrix behavior
- **Timeline Logging**: Cycle-by-cycle event log with throughput metrics after validation

### Fixed

#### Skew Buffer Display Fixes (2025-12-17)

- **SRAM Timing Display**: Fixed k-value calculation showing k=-1 when data became valid
- **Column Alignment**: Fixed Skew A (4-char) and Skew B (5-char) column width consistency
- **Dot-Value Alignment**: Changed dot position to align with value ones digit
- **Dynamic Left Margin**: Margin now scales with array size (`7 + max_depth * 4`)
- **FIFO Register Mapping**: Fixed slot assignment so data enters at deepest register, not R0
  - Formula: `R[d]` displays `fifo[d - depth + fifo_len]`
- **Display Timing**: All state now captured and displayed consistently after `step()`

#### Examples and Wavefront Visualization (2025-12-16)

- **Examples Framework** (`examples/`): User-facing demonstration applications
  - `examples/common/dram_model.py`: Simulated DRAM with matrix load/store operations
  - `examples/common/tensor_utils.py`: Matrix packing, unpacking, and tiling utilities
  - `examples/gemm/01_simple_matmul.py`: Complete matrix multiply demo with command generation
  - `examples/gemm/02_animated_wavefront.py`: Terminal animation of systolic wavefront
  - `examples/gemm/03_wavefront_gif.py`: GIF generator for shareable animations

- **Wavefront Documentation** (`docs/wavefront-animation.md`): Comprehensive guide
  - Theory of wavefront data flow and skewing patterns
  - Cycle-by-cycle breakdown of systolic array operation
  - Visualization tool usage and color coding guide
  - GIF generation for Slack/documentation sharing

- **End-to-End Integration Tests** (`tests/unit/test_gemm_e2e.py`)
  - `SimulatedAXIDRAM` class for testing with inline AXI handling
  - Tests for load, store, execute, and full GEMM sequence
  - Verification against NumPy reference computations

- **Top-Level Integration** (`src/systars/top.py`): SystolicTop module
  - Wires all accelerator components together
  - Command dispatch to load/store/execute controllers
  - AXI interface connection to StreamReader/StreamWriter

#### Phase 2: Memory System (2025-12-15)

- **LocalAddr** (`src/systars/memory/local_addr.py`): Unified address encoding utilities
  - Static methods for bit field extraction: `is_acc()`, `accumulate()`, `read_full_row()`, `is_garbage()`
  - Instance methods for bank/row decoding: `sp_bank()`, `sp_row()`, `acc_bank()`, `acc_row()`
  - Address construction helpers: `make_sp_addr()`, `make_acc_addr()`, `make_garbage_addr()`

- **ScratchpadBank** (`src/systars/memory/scratchpad.py`): Single bank local memory
  - Configurable read latency pipeline
  - Byte-level write masking with granularity=8
  - Read/write ports with valid signaling

- **Scratchpad** (`src/systars/memory/scratchpad.py`): Multi-bank scratchpad controller
  - Address-based bank selection routing
  - Parallel bank instantiation
  - Pipelined bank selection to match read latency

- **AccumulatorBank** (`src/systars/memory/accumulator.py`): Single bank accumulator memory
  - Activation functions: NONE, RELU, RELU6
  - Accumulate mode (read-modify-write) for partial sum accumulation
  - Signed data storage with configurable width

- **Accumulator** (`src/systars/memory/accumulator.py`): Multi-bank accumulator controller
  - Bank selection via address decoding
  - Activation function passthrough to banks

- **Unit Tests**: 34 new tests for memory subsystem
  - `tests/unit/test_local_addr.py`: 11 tests for address encoding
  - `tests/unit/test_scratchpad.py`: 12 tests for scratchpad memory
  - `tests/unit/test_accumulator.py`: 11 tests for accumulator memory

#### Phase 1: Core Systolic Array (Previous)

- **ProcessingElement (PE)**: MAC unit with input/weight propagation
- **PEArray**: Combinational grid of PEs with systolic data flow
- **SystolicArray**: Top-level pipelined grid of PEArrays
- **SystolicConfig**: Centralized configuration dataclass
