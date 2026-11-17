# The Combiner

The **Combiner** is the KPU tile's unit on the result path, between L1 and L2. It is a new
concept, added on 2026-11-14 for the KPU RTL generator
(branes-ai/embodied-ai-soc-generator#68), and ported to systars in #38. In kpu-sim it is
the Vector Engine, renamed and given the merge of partials (stillwater-sc/kpu-sim#427,
stillwater-sc/kpu-sim#429). It replaced systars's accumulator.

```text
fabric → L1 result buffers → Combiner → L2 (scratchpad) → BlockMover (STORE) → L3
```

## What it does

The Combiner is a **fusing, scaling, compressing and accumulating unit**. It is not a
result memory. For each result vector that leaves the array through the L1 result
buffers, it can:

- **Merge intermediate partials.** The weight- and input-stationary schedules reduce across
  K blocks. The Combiner reads the partial from L2, merges the new vector into it, and
  writes it back.
- **Fuse the operators applied as results are written to memory:**
  - bias;
  - activation functions;
  - scaling up or down (requantization, with rounding and saturation);
  - block-format conversion (compression). The first block formats are the OCP Microscaling
    (MX) formats, an open specification: blocks of 32 elements sharing one 8-bit
    power-of-two scale (MXFP8, MXFP6, MXFP4, MXINT8). MXINT8 comes first, then MXFP8.

  The activation functions run on the Combiner's SFU banks (special-function units), as in
  kpu-sim.
- **Write the result into L2.** STORE (the BlockMover) moves it on to L3. In a KPU SoC, a
  result that must spill continues to the memory controllers' DMA and DRAM.

Its pipeline keeps up with the array's result rate, one vector per array clock.

The KPU tile has **no separate accumulator**: partials and results live in L2 slots,
under the slot credits. It has **no DMA** either. The tile's data movement is its CSPs, the
BlockMover and the Streamer, and DMA belongs to the SoC's memory control layer.

## What systars has today

The port of the Combiner (#38, after kpugen esg#74, #75, #76 and #77) builds it on the
result path. It is no longer Gemmini's accumulator ([lineage](lineage.md#gemmini)).

| Part | `memory/` | What it does |
|---|---|---|
| **The merge** (esg#74) | `combiner.py` | Result rows go into L2's **result group** of 1R1W banks, pipelined (`combiner_latency`), addressed with the Combiner bit of `LocalAddr`. A request is an overwrite or an accumulate (read the row, commit the lane-wise sum). Commits are in order, one per clock, and the Streamer never waits; the last `combiner_latency` commits are forwarded, so back-to-back K blocks onto the same rows see each other's sums. STORE reads in the clocks the merge leaves free (`combiner_read_ready`). |
| **The epilogue** (esg#75) | `epilogue.py` | On a reduction's final commit: **requantization** (each result times a signed 16-bit scale, shifted right with round-to-nearest-even, saturated to `input_bits`) and the **SFU bank** (NONE, RELU, RELU6, CLIP), in either order (`act_first`). |
| **MXINT8** (esg#76) | `mx.py` | A finished row as OCP Microscaling MXINT8: blocks of `min(32, C)` along the row, one E8M0 scale per block, packed in place (`C` elements, then the scales). |
| **Profiles** (esg#77) | `epilogue.py` | Four complete epilogues: format (wide, int8 or MXINT8), order, scale and shift, clip bounds, SFU function. |

**Naming a profile.** A reduction's last COMPUTE carries an epilogue code in its
`activation` field: `EPILOGUE_FINAL` (bit 3) and the profile index (bits 1:0,
`epilogue_code(n)`). Earlier K blocks commit plain sums.

**The profiles' ports.** kpugen's tile sets the profiles through its CSRs. systars's
standalone tile takes them on `SystolicTop`'s ports `cmb_scale_<n>`, `cmb_control_<n>` and
`cmb_clip_<n>`, in the same layout (`Profile.control` in `epilogue.py` builds the word):

- `CMB_CONTROL` bits 5:0 are the shift (0..63);
- bit 8 is `requant`, bit 9 `act_first`, bit 10 `mxint8`;
- bits 14:12 are the SFU function.

**STORE and the ISA.**

- **Narrow rows.** A requantized row keeps its row address with its results packed in its
  low bytes. STORE it with `row_bytes = C * input_bits / 8` and `STORE_NARROW` (bit 3 of
  STORE's activation).
- **Choosing a profile.** `Matmul` and `Conv2d` name their profile in `CONFIG_OPTIONS[2:1]`.
  `Matmul` reads that profile's format at START (`combiner_formats`, wired by
  `connect_to_top`), so C's element size and STORE row bytes follow from it. `Conv2d`
  writes int32 rows, and refuses a narrow or MX profile.
- **Changing a profile.** Only while the tile is idle (`SystolicTop.busy` clear, which
  covers rows still in the Combiner's pipeline). An ISA instruction's owner changes them
  only between instructions.

A bias is still a LOAD into the result group that the reduction accumulates onto, so the
order is always bias, then the epilogue. **Not yet:** the Combiner writing into L2's A/B
groups for the next layer, and MX decode (with MXFP8).
