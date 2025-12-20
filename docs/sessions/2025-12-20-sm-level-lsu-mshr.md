# Session Log: SM-Level LSU with MSHR Tracking

**Date:** 2025-12-20
**Focus:** Restructure SIMT memory system from per-partition LSUs to SM-level LSU with MSHR tracking

## Summary

This session restructured the SIMT memory architecture from 4 per-partition Load/Store Units (each with max_pending=2) to a single SM-level LSU with MSHR (Miss Status Holding Register) tracking. This matches NVIDIA's architecture where all 4 warp schedulers issue to a single MIO queue per SM.

## Background and Motivation

### Problems with Per-Partition LSU

- Per-partition LSU with `max_pending=2` = only 8 outstanding requests SM-wide
- Frequent queue-full rejections causing warp stalls
- No cross-partition request visibility or load balancing
- Tracks instructions, not cache lines (no secondary miss optimization)

### NVIDIA Architecture Research

Web research revealed NVIDIA's actual architecture:

- Single MIO queue per SM - all 4 schedulers issue to one queue
- ~64 MSHRs per SM tracking cache lines, not instructions
- Secondary misses piggyback on primary (no new MSHR needed)
- 1 IPC to LSU pipe despite 4 schedulers issuing in parallel

## Implementation

### Phase 1: Configuration Parameters

Added to `config.py`:

```python
num_mshrs: int = 64
"""Number of Miss Status Holding Registers per SM."""

mio_queue_depth: int = 16
"""Depth of MIO queue (memory requests from all partitions)."""

lsu_issue_bandwidth: int = 1
"""Memory requests processed per cycle (1 IPC to LSU pipe)."""
```

### Phase 2: SM-Level LSU Module

Created `sm_lsu.py` with:

**Data Structures:**

```python
class MSHRState(IntEnum):
    INVALID = 0
    PENDING = auto()
    WAITING_DATA = auto()

@dataclass
class MSHRWaiter:
    partition_id: int
    warp_id: int
    thread_mask: int
    dst_reg: int
    thread_offsets: list[int]  # Per-thread offset within 128B line

@dataclass
class MSHREntry:
    cache_line_addr: int = 0
    state: MSHRState = MSHRState.INVALID
    waiters: list[MSHRWaiter] = field(default_factory=list)
    cycles_remaining: int = 0
    is_store: bool = False
    store_data: dict[int, int] = field(default_factory=dict)

@dataclass
class MIOQueueEntry:
    partition_id: int
    warp_id: int
    instruction: Instruction
    thread_mask: int
    addresses: list[int]
    store_data: list[int]
    is_load: bool
    address_space: AddressSpace
    issue_cycle: int = 0
```

**SMLevelLSUSim Key Methods:**

- `submit(partition_id, warp_id, instruction, addresses, store_data)` - Submit to MIO queue
- `tick()` - Process one request from MIO queue per cycle
- `get_completions(partition_id)` - Get completed requests for a partition
- `_find_mshr(cache_line_addr)` - Find existing MSHR for cache line
- `_allocate_mshr(cache_line_addr)` - Allocate new MSHR (primary miss)
- `_add_waiter(mshr, entry, segment_addr)` - Add waiter for secondary miss
- `_handle_shared_memory(entry)` - Bypass MSHR for shared memory
- `_handle_global_memory(entry)` - Full MSHR tracking for global memory
- `_deliver_mshr_data(mshr)` - Distribute data to all waiters

**MSHR Tracking Logic:**

```
For each memory request:
1. Coalesce addresses into 128B segments
2. For each segment:
   a. Check if MSHR exists (secondary miss)
      - Yes: Add waiter to existing MSHR
      - No: Allocate new MSHR (primary miss)
3. On completion: Deliver data to all waiters
```

### Phase 3: Partition Integration

Modified `partition.py`:

- Removed `load_store_unit` and `shared_memory` fields
- Added `sm_lsu: SMLevelLSUSim | None` reference
- Renamed `total_lsu_rejections` to `total_mio_queue_full`
- Updated `step()` to use `sm_lsu.submit()` and `sm_lsu.get_completions()`

**Interface Change:**

```python
# OLD (per-partition LSU)
if self.load_store_unit.issue(warp_id, instruction, addresses, store_data):
    self.scheduler.warps[warp_id].state = WarpState.STALLED_MEM

# NEW (SM-level LSU)
if self.sm_lsu.submit(self.partition_id, warp_id, instruction, addresses, store_data):
    self.scheduler.warps[warp_id].state = WarpState.STALLED_MEM
```

### Phase 4: SM Controller Integration

Modified `sm_controller.py`:

- Added `sm_lsu: SMLevelLSUSim` field
- Created SM-level LSU in `__post_init__` and connected to all partitions
- Call `sm_lsu.tick()` once per cycle before partition stepping
- Updated done check: `if all_done and not self.sm_lsu.is_busy()`

**Architecture Diagram:**

```
                          SM CONTROLLER
    ┌─────────────────────────────────────────────────────────────────────┐
    │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐              │
    │  │PARTITION │ │PARTITION │ │PARTITION │ │PARTITION │              │
    │  │    0     │ │    1     │ │    2     │ │    3     │              │
    │  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘              │
    │       └────────────┴────────────┴────────────┘                     │
    │                         │                                          │
    │                         ▼                                          │
    │       ┌─────────────────────────────────────────┐                 │
    │       │          MIO QUEUE (16 entries)          │                 │
    │       │   (1 IPC processed to LSU pipe)          │                 │
    │       └─────────────────────┬───────────────────┘                 │
    │                             ▼                                      │
    │       ┌─────────────────────────────────────────┐                 │
    │       │       SM-LEVEL LOAD/STORE UNIT          │                 │
    │       │  ┌────────────────────────────────────┐ │                 │
    │       │  │      MSHR TABLE (64 entries)       │ │                 │
    │       │  │  Tag │ Cache Line │ Waiters        │ │                 │
    │       │  └────────────────────────────────────┘ │                 │
    │       └─────────────────────────────────────────┘                 │
    │                     │               │                              │
    │              ┌──────┴───────┐ ┌─────┴──────┐                       │
    │              │SHARED MEMORY │ │GLOBAL MEM  │                       │
    │              └──────────────┘ └────────────┘                       │
    └────────────────────────────────────────────────────────────────────┘
```

### Phase 5: Statistics and Exports

Added statistics tracking:

- `total_primary_misses` - New MSHR allocations
- `total_secondary_misses` - Requests added to existing MSHRs
- `total_mio_queue_full_stalls` - Queue rejection count
- `total_shared_mem_requests` - Shared memory bypass count

Updated `__init__.py` exports:

- Added: `SMLevelLSUSim`, `MSHREntry`, `MSHRState`, `MSHRWaiter`, `MIOQueueEntry`
- Removed: `LoadStoreUnitSim` from exports (kept `AddressSpace`)

## Files Modified

| File | Changes |
|------|---------|
| `src/systars/simt/config.py` | Added `num_mshrs`, `mio_queue_depth`, `lsu_issue_bandwidth` |
| `src/systars/simt/sm_lsu.py` | **NEW** - SM-level LSU with MSHR tracking |
| `src/systars/simt/partition.py` | Removed per-partition LSU, added SM-level LSU reference |
| `src/systars/simt/sm_controller.py` | Instantiate and tick SM-level LSU |
| `src/systars/simt/__init__.py` | Updated exports for new classes |
| `tests/unit/test_simt_memory.py` | Updated tests for SM-level LSU |

## Testing

- All 57 SIMT unit tests pass
- All 328 total unit tests pass
- New tests added:
  - `TestSMLevelLSU::test_address_space_decode`
  - `TestSMLevelLSU::test_shared_memory_access`
  - `TestSMLevelLSU::test_mio_queue_capacity`
  - `TestSMSimMemory::test_partitions_share_sm_lsu`

## Key Design Decisions

1. **Single MIO Queue**: All 4 partitions submit to one queue (matches NVIDIA)
2. **MSHR Tracks Cache Lines**: 128B granularity, not per-instruction
3. **Secondary Miss Optimization**: Multiple warps waiting on same cache line share MSHR
4. **Shared Memory Bypass**: Shared memory requests bypass MSHR entirely (no cache)
5. **1 IPC Processing**: One request from MIO queue processed per cycle
6. **Partition Isolation**: Completions delivered to specific partition via `get_completions(partition_id)`

## Benefits

1. **Better Resource Utilization**: 64 MSHRs vs 8 outstanding requests
2. **Secondary Miss Optimization**: Reduces redundant memory traffic
3. **Cross-Partition Visibility**: SM-level can see all memory patterns
4. **Matches Real Hardware**: Follows NVIDIA's proven architecture
5. **Simpler Partition Logic**: Partitions just submit/receive, no LSU management
