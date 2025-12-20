# Session Log: SIMT Visualization Enhancements

**Date:** 2025-12-20
**Focus:** Enhanced SIMT streaming multiprocessor animation with timeline logging and store completion tracking

## Summary

This session enhanced the SIMT visualization (`examples/simt/01_animated_simt.py`) with timeline logging for pipeline analysis and proper tracking of store completion status. The matrix visualization now correctly distinguishes between "computed" (FFMAs done) and "stored" (truly complete) elements.

## Work Completed

### 1. Timeline Logging for Pipeline Analysis

Added `TimelineLogger` class to capture issue/complete events in CSV format for analyzing pipeline behavior and finding bubbles:

```python
class TimelineLogger:
    def enable(self, filename: str = "timeline.log")
    def log_issue(self, cycle, partition, warp, opcode, dst, src1, src2, latency, info)
    def log_complete(self, cycle, partition, warp, opcode, dst, info)
    def log_stall(self, cycle, partition, warp, reason)
    def print_summary(self)
```

**Features:**

- CSV output format: `cycle,event,partition,warp,opcode,dst,src1,src2,latency,info`
- Events: ISSUE, COMPLETE, STALL
- Summary statistics at end of simulation
- Enabled via `--timeline FILE` command line option

**Example output:**

```csv
cycle,event,partition,warp,opcode,dst,src1,src2,latency,info
0,ISSUE,0,0,MOV,0,0,None,1,""
1,ISSUE,0,0,LD,1,3,None,4,""
3,COMPLETE,0,0,LD,1,-1,-1,0,""
24,ISSUE,0,0,ST,5,0,None,4,""
27,COMPLETE,0,0,ST,-1,-1,-1,0,""
```

### 2. Store Completion Tracking in GEMMTracker

Extended `GEMMTracker` class to distinguish between computed and stored elements:

```python
class GEMMTracker:
    def record_store_issued(self, warp_id: int)
    def record_store_complete(self, warp_id: int)
    def is_element_stored(self, row: int, col: int) -> bool
    def get_pending_store_count(self) -> int
    def is_truly_complete(self) -> bool
```

**State Tracking:**

- `pending_stores: set[int]` - Warps with pending ST operations
- `stored: list[list[bool]]` - Per-element store completion status

### 3. Updated Matrix Visualization

Modified `render_gemm_matrix()` to show distinct element states:

| Symbol | Color | Meaning |
|--------|-------|---------|
| `W#` | Dim | Not started |
| `░░`, `▓░` | Warp color | Partial progress (some FFMAs) |
| `▓▓` | Yellow | Computed but ST pending |
| `██` | Green | Truly complete (stored) |
| `W#` | Green BG | Updating this cycle |

**Header Status:**

- `[computing]` - FFMAs in progress
- `[ST pending: N]` - N warps have pending stores
- `[COMPLETE]` - All elements stored

**Summary Line:**

```
FMAs: 64/64 (100.0%)  Computed: 32/32  Stored: 32/32
```

### 4. Animation Loop Integration

Updated main animation loop to:

1. Track ST instruction issues via `gemm_tracker.record_store_issued(warp_id)`
2. Track ST completions (dst_reg == -1) via `gemm_tracker.record_store_complete(warp_id)`
3. Log all events to timeline when enabled

### 5. Linting Fixes

Fixed all ruff and mypy issues:

- Renamed unused loop variables (`p_vis` → `_p_vis`, `results` → `_results`, etc.)
- Used enumerate() instead of manual index incrementing
- Removed unused variables (`end_idx`, `num_banks`)
- Fixed type annotations (`conflicting_banks: list[int]`, `instructions: list[Instruction]`)
- Fixed return type mismatch in `drain()` method

## Files Modified

| File | Changes |
|------|---------|
| `examples/simt/01_animated_simt.py` | Added TimelineLogger, store tracking, updated matrix viz, multi-partition warp distribution |
| `src/systars/simt/partition.py` | Fixed opcode shadowing, type annotations, **LSU queue overflow fix** (pending_memory_requests buffer) |
| `src/systars/simt/register_file.py` | Removed unused variables, added type annotations |
| `src/systars/simt/execution_unit.py` | Fixed drain() return type |

## Testing

- All 56 SIMT unit tests pass
- Ruff linter: All checks passed
- Mypy: Success, no issues found
- Manual verification with tiled GEMM animation
- LSU fix verified: 16×16 GEMM with K=2 completes all 64 instructions across all 8 warps

## Usage Examples

```bash
# Run with timeline logging
python examples/simt/01_animated_simt.py --tiled --m 8 --n 4 --k 2 \
    --fast --fast-mem --timeline timeline.csv

# Analyze timeline for bubbles
cat timeline.csv

# View matrix completion tracking
python examples/simt/01_animated_simt.py --tiled --m 8 --n 8 --k 4 \
    --fast --max-cycles 500
```

### 6. Multi-Partition Warp Distribution

Fixed a bug where all warps were loaded into partition 0 only, leaving partitions 1-3 idle.

**Before (bug):**

```python
# All warps loaded into P0 only
sm.partitions[0].load_program(warp_id, program)
sm.partitions[0].activate_warps(num_warps)
```

**After (fixed):**

```python
# Distribute warps across all partitions
# warp_id 0-7 → P0, 8-15 → P1, 16-23 → P2, 24-31 → P3
partition_id = global_warp_id // warps_per_partition
local_warp_id = global_warp_id % warps_per_partition
sm.partitions[partition_id].load_program(local_warp_id, program)
```

**Impact:**

- 16×16 GEMM: 8 warps → 1 partition, 32 cores
- 32×16 GEMM: 16 warps → 2 partitions, 64 cores
- 32×32 GEMM: 32 warps → 4 partitions, 128 cores (full utilization)

Also fixed global warp ID calculation in animation loop for proper GEMM tracking across partitions.

### 7. LSU Queue Overflow Bug Fix (Critical)

Fixed a critical bug where LD/ST instructions were silently dropped when the LSU queue was full.

**Bug Analysis:**

Timeline analysis revealed that warps W2-W7 appeared to "complete" their LDs faster than W0/W1, despite issuing later. Root cause investigation showed:

1. LSU has `max_pending = 2` queue depth
2. When `load_store_unit.issue()` returns False (queue full), the instruction was silently dropped
3. Warp continued executing with garbage register data (no LD completion event)
4. Timeline showed W2, W5, W6, W7 had NO LD COMPLETE events

**Before (bug in partition.py):**

```python
# Issue to LSU
if self.load_store_unit.issue(warp_id, instruction, addresses, store_data):
    # Mark warp as stalled on memory
    self.scheduler.warps[warp_id].state = WarpState.STALLED_MEM
# BUG: If issue() returns False, instruction is silently dropped!
```

**After (fixed):**

```python
# Issue to LSU
if self.load_store_unit.issue(warp_id, instruction, addresses, store_data):
    self.scheduler.warps[warp_id].state = WarpState.STALLED_MEM
    self.total_instructions += 1
else:
    # LSU queue full - buffer request and stall warp
    self.pending_memory_requests.append(
        (warp_id, instruction, addresses, store_data)
    )
    self.scheduler.warps[warp_id].state = WarpState.STALLED_MEM
    self.total_lsu_rejections += 1
```

**Fix Implementation:**

- Added `pending_memory_requests` list to buffer rejected memory instructions
- Each cycle, try to issue pending requests before handling new fires
- Properly stall warp until memory request is accepted
- Added `total_lsu_rejections` statistic to track queue-full events

**Verification:**

- 16×16 GEMM with K=2 now completes all 64 instructions (32 LDs, 16 FFMAs, 8 MOVs, 8 STs)
- All 8 warps have exactly 4 LD completions each
- Works with both --fast-mem (4 cycles) and default (200 cycles) latency

## Key Insights

1. **Store Completion Matters**: Users were confused when the matrix appeared "complete" but stores were still pending. The new visualization clearly shows computed vs stored status.

2. **Timeline Analysis**: The CSV timeline enables detailed post-mortem analysis of pipeline behavior, useful for identifying serialization and bubble-causing patterns.

3. **Visual Feedback**: The yellow `▓▓` (computed but not stored) provides immediate feedback that the system is waiting on memory operations.

4. **Partition Utilization**: Distributing warps across all partitions enables true parallel execution using all 128 ALU cores instead of just 32.

5. **Silent Failures are Dangerous**: The LSU queue overflow bug was subtle - warps appeared to complete but with garbage data. Timeline logging was essential for root cause analysis. Always check return values from issue() functions!
