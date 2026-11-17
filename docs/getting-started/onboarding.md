# Your First Week

Plan on one day to get a working environment and see the array run, and one week to make a
first real change. Read the [introduction](introduction.md) and
[the systolic tile](../architecture/systolic-tile.md) first.

## Day 1: the environment

You need Python 3.10+, [just](https://github.com/casey/just), and
[OSS-CAD-SUITE](https://github.com/YosysHQ/oss-cad-suite-build) for Verilator, Yosys and
GTKWave. Set `OSS_CAD_SUITE` if it is not in `/opt/oss-cad-suite`.
[The development environment](../initial-development-environment.md) has the details.

```bash
git clone https://github.com/stillwater-sc/systars.git
cd systars
./bin/setup-dev-env.sh        # or: just setup, if OSS-CAD-SUITE is installed
source .venv/bin/activate
just check-tools
```

To start on the Python side only, a plain virtual environment is enough:
`pip install amaranth numpy pytest pytest-subtests` and `export PYTHONPATH=src`. Without
Yosys, the unit tests that need synthesis are skipped.

## Day 1: see it run

1. `just test-unit` runs the unit tests. For a quick first run:
   `pytest tests/unit/test_pe.py tests/unit/test_stream_buffer.py tests/unit/test_systolic_array.py`.
2. `python examples/gemm/08_rtl_wavefront.py --stage 1 --svg s1.svg` records a 2x2 array from
   the RTL, checks it against the predicted wavefront and writes an animation; open it in a
   browser. Then try `--stage 2`, `--stage 3` and `--dataflow ws`.
3. Read [the wavefront animation](../wavefront-animation.md) page: its three animations are
   the quickest way into the timing.
4. `python examples/gemm/07_isa_matmul.py` prints the command stream of a Matmul,
   C[10x4] = A[10x4] @ B[4x4] on a 4x4 array (below).
5. `just gen` writes every RTL target into `gen/`, and `python3 -m systars.targets list` lists
   them. Look at what Amaranth emits for the PE.
6. `just test-cocotb` runs the RTL tests on Verilator.

### Reading a command stream

Example 07 prints three output tiles of the same shape:

```text
CONFIG   rs1=0x0                       # select OS
LOAD     dram=0x10000000 local=0x00000000 rows=4 stride=4 row_bytes=4   # A rows 0-3
LOAD     dram=0x20000000 local=0x00000008 rows=4 stride=4 row_bytes=4   # B, corner-turned
COMPUTE  rs1=0x0 rd=0x80000000 k_dim=4 # bit 31 of rd: the accumulator
STORE    dram=0x30000000 local=0x80000000 rows=4 stride=16 row_bytes=16 # int32 rows of C
... the same for A rows 4-7 (L2 slot 0x4, accumulator 0x80000004)
... and for rows 8-9 (rows=2), back in slot 0x0
```

Ask three questions of it: which slot each command fills or reads, which `cmd_sync` flags make
the third tile wait until the first has released slot 0x0, and why the second tile does not
wait at all. The printout shows neither the LOAD opcodes (B's is a `TRANSPOSE`) nor
`cmd_sync`, so read `src/systars/isa/matmul.py` beside it. The example runs the instruction
against a stand-in controller and checks the command counts only; the unit and cocotb tests
check the numbers.

## Week 1: exercises

Each one ends in something you can check. Do them in order.

- [ ] **Predict a result clock.** PE(i, j)'s OS result leaves the north edge in clock
  `s + K + j + 2i` (`core/pe.py`). Predict the clocks for a 4x4 array with K = 4, then
  confirm with `08_rtl_wavefront.py --grid 2 2 --tile 2 2 --n 4`.
- [ ] **Break the skew and watch it fail.** On a scratch branch, change the order in which the
  Streamer loads the L1 buffers (`controller/execute.py`). Run `tests/unit/test_wavefront.py`
  and `tests/unit/test_execute_controller.py`, read the failures, then revert.
- [ ] **Measure double buffering.** Find where `tests/unit/test_credits.py` compares
  `buffers=1` with `buffers=2` (907 against 744 cycles for 8x8x16 on 4x4). Change the shape
  and explain the new numbers.
- [ ] **Trace one slot.** Pick an L2 slot in the command stream above and write down its state
  (FREE, OWNED, READY, READING) after each command. Check yourself against
  `controller/sync.py`.
- [ ] **Run a Matmul with a bias in WS.** Copy a case from `tests/unit/test_matmul.py`, set
  `CONFIG_DATAFLOW` to WS and a bias row, and check the result against numpy.
- [ ] **Synthesize.** Run `just synth ecp5`, then `just synth polarfire`, and compare the PE's
  resource counts with the [PolarFire plan](../plan/polarfire-deployment.md).

## How correctness is established

Five layers, each catching what the one before cannot. A change is done when the layers it
touches pass.

| Layer | Where | Tool | What it proves |
|---|---|---|---|
| Unit and integration | `tests/unit/` | pytest, Amaranth simulator | Component behaviour and cycle timing (result clocks, no bubbles, the `2R - 1` drain); GEMMs end to end through `SystolicTop`; Matmul and Conv2d against numpy in every dataflow |
| Independent models | `tests/unit/_axi.py`, `_isa.py`, `test_credits.py` | Python reference models | The AXI rules (no burst across 4 KB); a Python model of `SystolicTop` for the ISA; a cycle-by-cycle slot monitor (credits conserved, no tile used before its token) |
| Wavefront check | `wavefront.py`, `test_wavefront.py` | Amaranth simulator | Every PE in every clock matches the wavefront equation; the committed SVGs are current |
| RTL simulation | `verif/cocotb/tests/` | cocotb, Verilator | The generated Verilog, including Matmul and Conv2d on the 4x4 `systolic_top` target in OS, WS and IS |
| Synthesis | `just synth <target>` | Yosys (generic, iCE40, ECP5, PolarFire) | The targets map onto device cells; resource counts |

When you add tests:

- **Check every other byte.** End-to-end tests verify the result and that every byte of L3
  outside it is untouched.
- **Start from random memory.** `test_no_read_before_write` fills the scratchpad and the
  accumulator with random data, since memories on hardware have no reset value.
- **Assert the error flags.** A valid schedule never raises `l1_error` or `sync_error`.

A new cocotb suite goes in both `just test-cocotb` and the `rtl-tests` matrix in
`.github/workflows/test.yml`. A change to the fabric, the Streamer's schedule or the renderer
needs `just wavefronts`, with the regenerated SVGs committed.

## Codebase map

Not everything under `src/systars/` is on the path to `SystolicTop`. Some modules predate the
registered fabric and the CSP rework, and three packages model other architectures.

| Path | Status | What it is |
|---|---|---|
| `config.py`, `targets.py`, `top.py` | Current | Configuration, the RTL target list, the compute tile |
| `core/` | Current | `PE`, `PEArray`, `SystolicArray` |
| `memory/scratchpad.py`, `accumulator.py`, `stream_buffer.py`, `local_addr.py` | Current | L2, the accumulator, L1, local address encoding |
| `controller/execute.py`, `load.py`, `store.py`, `sync.py` | Current | The Streamer, the BlockMover's legs, credits and tokens |
| `dma/reader.py`, `writer.py`, `axi.py` | Current | The AXI burst engines behind the BlockMover |
| `isa/` | Current | `Matmul`, `Conv2d`, the command port |
| `wavefront.py` | Current | RTL tracing, checking and SVG rendering |
| `util/commands.py` | Current | `DmaOpcode`, `ExecOpcode` and command helpers |
| `memory/skew_buffer.py` | Legacy | The skew-register approach; used only by `test_sram_read_scheduler.py` and example 05 |
| `controller/load_transpose.py`, `dma/strided_reader.py` | Legacy | An earlier transposing load; `SystolicTop` turns corners in the BlockMover |
| `dma/descriptor_engine.py` | Not integrated | DMA descriptor chains ([design](../architecture/dma-descriptors.md)); tested, not wired into `SystolicTop` |
| `loop/`, `arithmetic/` | Empty | Placeholders for planned components |
| `stencil/` | Side package | A line-buffer stencil machine for 2D convolution |
| `simt/`, `simd/maspar/` | Side package | Models of an NVIDIA-style SM and a MasPar SIMD array, with animations and energy estimates |

In `examples/gemm/`, 02 and 03 animate the textbook model, 05 uses the legacy skew buffer and
06 the legacy transposing load. Example 07 (an ISA command stream) and 08 (the current `SystolicArray` RTL) reflect the
current design, so start there; the integrated tile runs in `tests/unit/test_gemm_systolic.py`,
`tests/unit/test_matmul.py` and the cocotb `systolic_top` suite.
`docs/sessions/` holds dated development logs: useful history, not specifications.

## House rules

- Use KPU and CSP terms (credits, push, tile-ready tokens, L1/L2/L3), never cache terms.
- New components are `lib.wiring.Component` subclasses whose docstring states their ports and
  timing: the docstrings are the specification.
- Pages are plain Markdown in `docs/`; the site in `docs-site/` is generated, and a broken link
  fails its build. See [the development workflow](../development-workflow.md).
- Run `just check` before every push; CI runs the same recipes.
- Look terms up in the [glossary](../guides/glossary.md).
