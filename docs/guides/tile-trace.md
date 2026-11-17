# Tracing a tile run

`systars.trace` records a `SystolicTop` simulation as a **trace bundle**: every command the
tile ran, its first error, and its internal lanes. It is kpu-gen's trace package
(branes-ai/embodied-ai-soc-generator, `kpu_gen.trace`) for its standalone-tile path (#18).
systars has no SoC, so the bundle's NoC, DMA and memory-cell tables stay empty.

## Record a run

```python
from amaranth.sim import Simulator

from systars.trace import TileRecorder

rec = TileRecorder(top)          # top: the SystolicTop in the design being simulated
sim = Simulator(design)
sim.add_process(rec.process())   # samples every probe once per clock
...                              # run the program
rec.trace(program="...").save("build/run.json")
```

`TileRecorder(top, fine=False)` skips the fine lanes, the per-column L1 and result-buffer
detail.

## What a bundle holds

The format is kpu-gen's (`format` `"kpu-gen-trace"`, version 3), so kpu-gen's tools read a
systars bundle. A systars bundle has `scope` `"tile"` and `producer` `systars <sha>`.

| Table | Rows |
|---|---|
| `commands` | Every command, from its accept into its process's queue (`accept`) and its start (`open`) to its completion (`close`). A COMPUTE also has the clock its results were in the Combiner (`result`). Each row lists its AXI transactions. |
| `intervals` | Resource occupancy, run-length encoded (`trace/lanes.py`). The coarse lanes are L2's read and write ports by bank group (with conflicts), each L2 and Combiner credit slot's state, the Combiner's write and read ports, each process's queue, and the Streamer's drain. The fine lanes are each L1 stream buffer, each result column, and the north collision. |
| `errors` | The tile's first error ([ISA instructions](../isa-instructions.md)), on the trace's clock, and the command it names. |
| `values` | Result blocks beside their expected values (`TileRecorder.add_values`). |

## Report, check and draw

```bash
python -m systars.trace build/run.json --check --svg build/svg   # report, checks, SVGs
python -m systars.trace build/run.json --perf                    # performance summary (JSON)
```

`--check` runs the trace's invariant checks (`trace/check.py`): commands open and close in
order, and the slots go OWNED → READY → READING. It also checks that the L1 streams are well
formed, and that no lane runs overlap. It exits 1 on a problem. `--svg` writes the timeline.

## The differential harness

`tests/fuzz` (kpu-gen's harness, #83 there) generates random `Matmul` programs and checks
each against the reference models. A failing seed is shrunk to a smaller program that still
fails, and the failure prints the command that replays it. Its trace is written only when
`KPU_FUZZ_RUN` names an existing directory, which `just fuzz` sets up for each run.

```bash
just fuzz 200          # 200 random programs, then the coverage report (build/fuzz/)
just tile-trace 3      # seed 3 with its trace, reported and drawn in build/tile_trace/
python -m tests.fuzz.run --seed 3 --trace build/t.json   # the same, by hand
```

`pytest tests/fuzz` runs a smaller default set of seeds as part of the unit tests. Run that
way, a failure reports its shrunk program but writes no trace, unless `KPU_FUZZ_RUN` is set.

## Not yet

The `.dflow` export for kpu-sim's tile-flow viewer (stillwater-sc/kpu-sim#391) is not here
yet. kpu-gen derives that record's CSP actions from its runtime's annotations, which say the
tile each command moves. systars has no runtime, so the export needs an issuer that
annotates its commands. That is the rest of #18.
