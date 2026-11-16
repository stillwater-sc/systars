# The Combiner

The **Combiner** is the KPU tile's unit on the result path, between L1 and L2. It is a new
concept, added on 2026-11-14 for the KPU RTL generator
(branes-ai/embodied-ai-soc-generator#68). systars does not have it yet (#38). In kpu-sim it
is the Vector Engine, renamed and given the merge of partials (stillwater-sc/kpu-sim#427,
stillwater-sc/kpu-sim#429). It replaces systars's accumulator.

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

The accumulator (`memory/accumulator.py`) comes from Gemmini, systars's origin. It is a
separate result memory:

| The accumulator today | The Combiner |
|---|---|
| WS/IS partial sums accumulate in it (`COMPUTE_ACCUMULATE`). OS results drain into it. | Merges partials in L2 |
| A bias is a LOAD into it (the accumulator pool's second producer, `SlotPool(producers=2)`) | Applies the bias as results pass through. No LOAD into a result memory. |
| ReLU and ReLU6 as rows are read out (`memory/activation.py`) | Activation on the way into L2 |
| No scaling. The module's header diagram draws "+ Scale", but the RTL has none. | Scaling up or down |
| No block-format conversion | Block-format conversion: MXINT8, then MXFP8 |
| Results stay in it, and STORE moves them from it | Results land in L2, and STORE moves them from there |

The port is #38, after the rename (#37) that calls the accumulator the Combiner's storage
and `dma/` the BlockMover. kpugen keeps the same code, so both repos change together.
