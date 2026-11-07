# Session: Synthesis Targets and PolarFire

**Date:** 2026-11-06
**Focus:** Wavefront progression merged; `just synth <target>` ported from kpugen; a
PolarFire target in both repos

## Summary

The wavefront progression (#23) was reviewed and merged. kpugen's `just synth <target>` was
ported to systars (#25), with a target registry as the one source of what is generated and
synthesized, and a new PolarFire target through Yosys's `synth_microchip`. The PolarFire
target was then ported to kpugen (branes-ai/embodied-ai-soc-generator#37), where the tile's
accumulator memory needed one more prep step; that step was back-ported here so the two
recipes stay identical. Review on the kpugen side tightened the claim that dropping initial
values is safe into a host requirement and a test.

| PR | Repo | Merged | What |
|---|---|---|---|
| #24 | systars | `a8ca48c` | 2x2 -> 4x4 -> 16x16 wavefront animations from the RTL (#23) |
| #26 | systars | `279b047` | `systars.targets`, `just synth <target>`, `just synth-report`, PolarFire (#25) |
| esg#38 | kpugen | `72f064d` | PolarFire target, memory init dropped, host requirement documented (esg#37) |
| #27 | systars | `bfaaabf` | Back-port of the memory-init step and the random-memory test |

## Wavefront progression (#24)

Review fixes, then a squash merge. See the [changelog](../../CHANGELOG.md) entry for #23 and
[the wavefront page](../wavefront-animation.md).

## `just synth <target>` (#26)

Before, `just synth-check` only elaborated `gen/*.v`, and CI mapped only the PE with inline
Yosys. kpugen had fixed this with a registry (`kpugen list`) that the justfile and CI read.

- **`src/systars/targets.py`**: `Target(name, top, factory, description)` and `TARGETS` (PE,
  PEArray 1x1 and 2x2, SystolicArray 2x2-of-2x2 and 4x4, Scratchpad, Accumulator).
  `python3 -m systars.targets list [--plain]` and `gen [NAME...] [--out-dir gen]`. `just gen`,
  `scripts/gen_*.py`, the cocotb Makefiles and CI all generate through it.
- **`just synth generic|ice40|ecp5|polarfire`**: every target, hierarchically (`-noflatten`),
  with `hierarchy -top` first. **`just synth-report`** writes the Markdown resource report.
- **CI** (`synthesis.yml`): a `[ice40, ecp5, polarfire, generic]` matrix that calls the
  recipes.
- The cocotb `pe_array` test used to overwrite `gen/pe_array.v` with a 2x2 design; it now
  uses the `pe_array_2x2` target.

### PolarFire

Yosys 0.60 maps PolarFire with `synth_microchip -family polarfire -noiopad`. PolarFire
flip-flops have no power-up value, and Amaranth gives every register an `init`, so the flow
fails with `initialized D flip-flops are not supported` until `attrmap -remove init` drops
them. That is safe for registers because every one also has a synchronous reset (nothing in
`src/` is `reset_less`), so the design must be reset before use.

The default 16x16 SystolicArray maps to 256 `MACC_PA`, 28,245 `CFG`, 16,384 `ARI1` and 31,744
`SLE`, about 45K logic elements (about 18% of an MPFS250T). It takes 160 s and 1.4 GB, so CI
synthesizes the smaller targets. The counts are in the
[PolarFire plan](../plan/polarfire-deployment.md); they are before place and route, and
Libero stays the real flow because nextpnr does not support PolarFire.

## The kpugen port and the back-port (esg#38, #27)

kpugen's `systolic_tile`, `systolic_tile_os` and `kpu_soc_2x2` still failed after
`attrmap -remove init`. The tile accumulator's `Memory(..., init=[])` is written as an
`initial` block, which Yosys keeps as `$meminit_v2` cells; that memory lands in flip-flops,
which inherit the value. `delete t:$meminit_v2` fixes it, so the prep is now:

```
proc; delete t:$meminit_v2; attrmap -remove init;
```

`kpu_soc_2x2` then maps in about 120 s: 64 `MACC_PA`, 72 `RAM1K20`, 280 `RAM64x12`, 30,849
`CFG`, 4,472 `ARI1`, 26,914 `SLE`. systars' targets don't need the step today, but
`SystolicTop` would, so #27 adds it here too.

### Is dropping memory contents safe?

CodeRabbit challenged the first wording, "nothing reads a memory word before writing it".
That holds for the design, not for the host: a DMA of words nobody wrote, a
`COMPUTE_ACCUMULATE` onto a row no `COMPUTE` wrote, or the bytes a masked write leaves out
read back unspecified on hardware, where simulation reads zero. The docs now state it as a
host requirement: write before you read. Matmul uses `COMPUTE` on its first k-slice and Conv2d
rejects accumulating onto Y, so the ISA instructions meet it.

The test that backs this, `test_no_read_before_write`, fills every scratchpad and accumulator
word with random data across the full word width (a review caught that the first version
covered only 62 of 128 bits), runs OS, WS and IS GEMMs twice into the same C, and checks
`2 * (A @ B)`. As a negative control, making the first pass `COMPUTE_ACCUMULATE` passes with
zeroed memory and fails with random memory. A second test checks that no synthesis target
carries memory contents; its parser reads only `initial begin` blocks, because a plain
`x[n] = v` pattern also matched always-block bit assignments.

## Lessons

- CodeRabbit skips draft PRs; an `@coderabbitai review` comment starts a review.
- A claim that a transformation is safe needs a test that would fail if it weren't; the
  negative control is what shows the poisoned-memory test can fail.

## Open items

- `lint.yml` and `test.yml` still generate `gen/pe.v` inline instead of through
  `systars.targets`.
- The GIF/MP4 question from #23; the block-diagram generator (#21, kpugen #29, kpu-sim #392).
