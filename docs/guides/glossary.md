# Glossary

| Term | Meaning in SYSTARS |
|---|---|
| Combiner | The unit on the result path between L1 and L2. Today it is the tile's result memory (`memory/combiner.py`), addressed with `LocalAddr` bit 31 (bit 30: add on write); #38 gives it the KPU model's function |
| Block | One L2 row: what one L1 stream buffer holds and streams, up to KB elements |
| BlockMover | The process that moves whole tiles between L3 and the tile: LOAD and STORE |
| `cmd_chain` | OS flag bits that join several COMPUTEs into one reduction, with C held in the PEs |
| `cmd_sync` | A command's slot tags and its ACQUIRE, SIGNAL, WAIT and RELEASE flags |
| Corner turn | A transpose done by the BlockMover (`TRANSPOSE`), so each L2 row feeds one L1 buffer |
| Credit | Permission to fill a free slot; one per whole-tile slot |
| CSP | Communicating sequential processes: the KPU's model of processes that push tiles on credits and tokens |
| Drain rule | OS reductions end at least `2R - 1` clocks apart, so results never collide moving north |
| `first`, `last` | Markers that travel east with A and start and end an OS reduction |
| IS | Input-stationary: Aᵀ held in the PEs, B streamed, Cᵀ produced |
| KB | The longest stream block: `sp_width / input_bits` elements |
| KPU | Stillwater's Knowledge Processing Unit, a Domain Flow Architecture; `SystolicTop` follows its data movement model |
| L1, L2, L3 | The stream and result buffers at the array's edges; the scratchpad; external memory over AXI |
| OS | Output-stationary: C held in the PEs, A streamed east, B south |
| PE, PEArray | One MAC with registered links; a structural block of PEs |
| R, C | The array's rows and columns in PEs: `grid_rows * tile_rows`, `grid_cols * tile_cols` |
| Result path | The process (`controller/result_path.py`) that takes the fabric's results from L1's result buffers into the Combiner |
| Skew | The one-clock stagger between neighbouring streams that makes operands meet; made by L1, not registers |
| Slot | A whole-tile region of L2 or the Combiner: FREE, OWNED, READY or READING |
| Streamer | The process (`controller/streamer.py`) that streams L2 tiles through L1 into the fabric |
| Tile-ready token | The signal that a slot's tile is complete and may be consumed |
| Wavefront | The anti-diagonal of PEs working on the same k (or m), moving one PE per clock |
| WS | Weight-stationary: B held in the PEs, A streamed, partial sums south |
