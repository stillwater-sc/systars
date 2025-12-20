"""
SM-Level Load/Store Unit with MSHR Tracking.

This module implements an SM-wide LSU based on NVIDIA's architecture:
- Single MIO (Memory I/O) queue receives requests from all 4 partitions
- 1 request processed per cycle (1 IPC to LSU pipe)
- MSHR (Miss Status Holding Register) tracks cache lines, not instructions
- Secondary misses piggyback on primary (no new MSHR allocation needed)
- Completions routed back to correct partition/warp

Architecture:
    ┌─────────────────────────────────────────────────────────────────────┐
    │                    SM-LEVEL LOAD/STORE UNIT                          │
    │                                                                      │
    │  Partitions 0-3 ──► MIO QUEUE (16 entries, 1 IPC processing)        │
    │                            │                                         │
    │                            ▼                                         │
    │                 ┌─────────────────────┐                             │
    │                 │   MSHR TABLE (64)   │                             │
    │                 │  cache_line → waiters                             │
    │                 └──────────┬──────────┘                             │
    │                            │                                         │
    │              ┌─────────────┴─────────────┐                          │
    │              ▼                           ▼                          │
    │       SHARED MEMORY                GLOBAL MEMORY                    │
    │       (bypass MSHR)                (use MSHR)                       │
    │                                                                      │
    └─────────────────────────────────────────────────────────────────────┘
"""

from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Any

from .config import SIMTConfig
from .load_store_unit import AddressSpace
from .memory_coalescer import MemoryCoalescerSim
from .shared_memory import SharedMemorySim
from .warp_scheduler import Instruction


class MSHRState(IntEnum):
    """State of an MSHR entry."""

    INVALID = 0
    PENDING = auto()  # Request issued, waiting for memory
    WAITING_DATA = auto()  # Data returning, distributing to waiters


@dataclass
class MSHRWaiter:
    """
    A warp waiting for an MSHR to complete.

    When multiple warps access the same cache line (secondary miss),
    each gets added as a waiter. When data returns, all waiters are
    notified and receive their portion of the cache line.
    """

    partition_id: int
    warp_id: int
    thread_mask: int
    dst_reg: int
    # Per-thread offset within 128B cache line (-1 for inactive threads)
    thread_offsets: list[int]


@dataclass
class MSHREntry:
    """
    Miss Status Holding Register entry - tracks one cache line request.

    An MSHR tracks a pending memory request at cache line granularity (128B).
    Multiple warps can wait on the same cache line (secondary misses),
    avoiding redundant memory requests.
    """

    # Cache line identification (128-byte aligned address)
    cache_line_addr: int = 0

    # State
    state: MSHRState = MSHRState.INVALID

    # Waiters: multiple warps can wait on same cache line
    waiters: list[MSHRWaiter] = field(default_factory=list)

    # Timing
    cycles_remaining: int = 0
    issue_cycle: int = 0

    # Store handling
    is_store: bool = False
    store_data: dict[int, int] = field(default_factory=dict)  # offset -> data


@dataclass
class MIOQueueEntry:
    """
    Entry in the SM-level Memory I/O queue.

    All partitions submit memory requests to this queue.
    Requests are processed at 1 IPC (1 per cycle).
    """

    partition_id: int
    warp_id: int
    instruction: Instruction
    thread_mask: int
    addresses: list[int]  # 32 per-thread addresses
    store_data: list[int]  # 32 per-thread store values (for ST)
    is_load: bool
    address_space: AddressSpace
    issue_cycle: int = 0


@dataclass
class SMLevelLSUSim:
    """
    SM-wide Load/Store Unit with MSHR tracking.

    Key behaviors:
    1. Single MIO queue receives requests from all 4 partitions
    2. 1 request processed per cycle (1 IPC to LSU pipe)
    3. MSHR tracks cache lines, not instructions
    4. Secondary misses piggyback on primary (no new MSHR needed)
    5. Completions routed back to correct partition/warp
    """

    config: SIMTConfig

    # MIO Queue (pending requests from all partitions)
    mio_queue: deque[MIOQueueEntry] = field(default_factory=deque)

    # MSHR Table (tracks cache lines)
    mshrs: list[MSHREntry] = field(init=False)

    # Per-partition completion queues
    completion_queues: dict[int, deque[tuple[int, int, list[int]]]] = field(default_factory=dict)

    # Memory references (set by SM controller)
    shared_memory: SharedMemorySim | None = None
    global_memory: Any = None  # GlobalMemorySim

    # Coalescer for 128B segment analysis
    coalescer: MemoryCoalescerSim = field(init=False)

    # Current cycle (for timing)
    cycle: int = 0

    # Statistics
    total_primary_misses: int = 0
    total_secondary_misses: int = 0
    total_mshr_full_stalls: int = 0
    total_shared_accesses: int = 0
    total_global_accesses: int = 0
    total_mio_queue_full: int = 0
    total_energy_pj: float = 0.0

    def __post_init__(self) -> None:
        """Initialize MSHR table and per-partition queues."""
        self.mshrs = [
            MSHREntry(cache_line_addr=0, state=MSHRState.INVALID)
            for _ in range(self.config.num_mshrs)
        ]
        for i in range(self.config.num_partitions):
            self.completion_queues[i] = deque()
        self.coalescer = MemoryCoalescerSim(self.config)

    def can_accept(self) -> bool:
        """Check if MIO queue can accept new request."""
        return len(self.mio_queue) < self.config.mio_queue_depth

    def submit(
        self,
        partition_id: int,
        warp_id: int,
        instruction: Instruction,
        addresses: list[int],
        store_data: list[int] | None = None,
        thread_mask: int = 0xFFFF_FFFF,
    ) -> bool:
        """
        Submit memory request from partition to MIO queue.

        Args:
            partition_id: Source partition (0-3)
            warp_id: Warp within partition
            instruction: The LD/ST instruction
            addresses: Per-thread addresses (32 values)
            store_data: Per-thread store data (for ST)
            thread_mask: Active thread mask

        Returns:
            True if accepted, False if MIO queue full.
        """
        if not self.can_accept():
            self.total_mio_queue_full += 1
            return False

        if store_data is None:
            store_data = [0] * 32

        # Decode address space from first active thread
        first_addr = None
        for i in range(32):
            if (thread_mask >> i) & 1 and i < len(addresses):
                first_addr = addresses[i]
                break

        if first_addr is None:
            return True  # No active threads, nothing to do

        addr_space = self._decode_address_space(first_addr)
        is_load = instruction.opcode in ("LD", "LDG", "LDS")

        entry = MIOQueueEntry(
            partition_id=partition_id,
            warp_id=warp_id,
            instruction=instruction,
            thread_mask=thread_mask,
            addresses=addresses,
            store_data=store_data,
            is_load=is_load,
            address_space=addr_space,
            issue_cycle=self.cycle,
        )
        self.mio_queue.append(entry)
        return True

    def tick(self) -> None:
        """
        Process one cycle of LSU operation.

        1. Process MSHR completions (data returning from memory)
        2. Process one request from MIO queue (1 IPC)
        """
        # 1. Process MSHR completions
        self._process_mshr_completions()

        # 2. Process requests from MIO queue (up to lsu_issue_bandwidth per cycle)
        for _ in range(self.config.lsu_issue_bandwidth):
            if self.mio_queue:
                entry = self.mio_queue.popleft()
                self._process_mio_entry(entry)

        self.cycle += 1

    def get_completions(self, partition_id: int) -> list[tuple[int, int, list[int]]]:
        """
        Get completed memory operations for a partition.

        Returns:
            List of (warp_id, dst_reg, per_thread_data) tuples.
            For stores, dst_reg is -1 and data is empty.
        """
        completions: list[tuple[int, int, list[int]]] = []
        queue = self.completion_queues.get(partition_id, deque())
        while queue:
            completions.append(queue.popleft())
        return completions

    def is_busy(self) -> bool:
        """Check if LSU has pending work."""
        # Check MIO queue
        if self.mio_queue:
            return True
        # Check MSHRs
        return any(mshr.state != MSHRState.INVALID for mshr in self.mshrs)

    # =========================================================================
    # Internal Methods
    # =========================================================================

    def _decode_address_space(self, addr: int) -> AddressSpace:
        """Decode address to determine memory space."""
        space_bits = (addr >> 30) & 0x3
        return AddressSpace(space_bits)

    def _strip_space_bits(self, addr: int) -> int:
        """Remove address space bits to get actual address."""
        return addr & 0x3FFF_FFFF  # Lower 30 bits

    def _process_mio_entry(self, entry: MIOQueueEntry) -> None:
        """Process a single MIO queue entry."""
        if entry.address_space == AddressSpace.SHARED:
            # Shared memory: direct access, no MSHR needed
            self._handle_shared_memory(entry)
        else:
            # Global memory: use coalescing and MSHR
            self._handle_global_memory(entry)

    def _handle_shared_memory(self, entry: MIOQueueEntry) -> None:
        """
        Handle shared memory request (bypass MSHR).

        Shared memory accesses go directly to the banked SRAM
        without MSHR tracking since they're not cached.
        """
        if self.shared_memory is None:
            # No shared memory - queue dummy completion
            if entry.is_load:
                self.completion_queues[entry.partition_id].append(
                    (entry.warp_id, entry.instruction.dst or 0, [0] * 32)
                )
            else:
                self.completion_queues[entry.partition_id].append((entry.warp_id, -1, []))
            return

        # Strip address space bits
        stripped_addrs = [self._strip_space_bits(a) for a in entry.addresses]

        if entry.is_load:
            read_data, conflicts, _ = self.shared_memory.access(
                stripped_addrs,
                is_write=False,
                thread_mask=entry.thread_mask,
            )
            # Queue completion (shared memory is fast, complete immediately)
            self.completion_queues[entry.partition_id].append(
                (entry.warp_id, entry.instruction.dst or 0, read_data)
            )
            # Track energy
            self.total_energy_pj += self.config.shared_mem_access_energy_pj
            if conflicts > 0:
                self.total_energy_pj += conflicts * self.config.shared_mem_conflict_energy_pj
        else:
            # Store
            self.shared_memory.access(
                stripped_addrs,
                is_write=True,
                write_data=entry.store_data,
                thread_mask=entry.thread_mask,
            )
            # Queue store completion
            self.completion_queues[entry.partition_id].append((entry.warp_id, -1, []))
            self.total_energy_pj += self.config.shared_mem_access_energy_pj

        self.total_shared_accesses += 1

    def _handle_global_memory(self, entry: MIOQueueEntry) -> None:
        """
        Handle global memory request with MSHR tracking.

        Uses coalescing to group addresses into 128B segments,
        then allocates MSHRs for each segment.
        """
        # Strip address space bits and coalesce
        stripped: list[int | None] = [
            self._strip_space_bits(entry.addresses[i])
            if (entry.thread_mask >> i) & 1 and i < len(entry.addresses)
            else None
            for i in range(32)
        ]
        result = self.coalescer.analyze(stripped, entry.thread_mask)

        # For each 128B segment, check/allocate MSHR
        for segment_addr in result.segments:
            cache_line_addr = segment_addr  # Already 128B aligned

            # Check for existing MSHR (secondary miss)
            mshr = self._find_mshr(cache_line_addr)

            if mshr is not None:
                # Secondary miss - add waiter to existing MSHR
                self._add_waiter(mshr, entry, segment_addr)
                self.total_secondary_misses += 1
            else:
                # Primary miss - allocate new MSHR
                mshr = self._allocate_mshr(cache_line_addr)
                if mshr is None:
                    # MSHR full - re-queue entry and stop processing
                    self.mio_queue.appendleft(entry)
                    self.total_mshr_full_stalls += 1
                    return

                self._add_waiter(mshr, entry, segment_addr)
                self._issue_memory_request(mshr, entry)
                self.total_primary_misses += 1

        self.total_global_accesses += 1

    def _find_mshr(self, cache_line_addr: int) -> MSHREntry | None:
        """Find MSHR for cache line address."""
        for mshr in self.mshrs:
            if mshr.state != MSHRState.INVALID and mshr.cache_line_addr == cache_line_addr:
                return mshr
        return None

    def _allocate_mshr(self, cache_line_addr: int) -> MSHREntry | None:
        """Allocate MSHR for new cache line."""
        for mshr in self.mshrs:
            if mshr.state == MSHRState.INVALID:
                mshr.cache_line_addr = cache_line_addr
                mshr.state = MSHRState.PENDING
                mshr.waiters = []
                mshr.is_store = False
                mshr.store_data = {}
                return mshr
        return None

    def _add_waiter(
        self,
        mshr: MSHREntry,
        entry: MIOQueueEntry,
        segment_addr: int,
    ) -> None:
        """Add waiter to MSHR."""
        # Compute per-thread offsets within cache line
        thread_offsets: list[int] = []
        thread_mask_for_segment = 0

        for i in range(32):
            if (entry.thread_mask >> i) & 1 and i < len(entry.addresses):
                addr = self._strip_space_bits(entry.addresses[i])
                # Check if this thread's address falls in this segment
                if (addr // 128) * 128 == segment_addr:
                    offset = addr - segment_addr
                    thread_offsets.append(offset)
                    thread_mask_for_segment |= 1 << i
                else:
                    thread_offsets.append(-1)  # Not in this segment
            else:
                thread_offsets.append(-1)  # Inactive thread

        waiter = MSHRWaiter(
            partition_id=entry.partition_id,
            warp_id=entry.warp_id,
            thread_mask=thread_mask_for_segment,
            dst_reg=entry.instruction.dst if entry.instruction.dst is not None else 0,
            thread_offsets=thread_offsets,
        )
        mshr.waiters.append(waiter)

        # Handle stores
        if not entry.is_load:
            mshr.is_store = True
            for i in range(32):
                if (thread_mask_for_segment >> i) & 1 and thread_offsets[i] >= 0:
                    offset = thread_offsets[i]
                    mshr.store_data[offset] = entry.store_data[i]

    def _issue_memory_request(self, mshr: MSHREntry, _entry: MIOQueueEntry) -> None:
        """Issue memory request for MSHR (set latency)."""
        mshr.cycles_remaining = self.config.l1_miss_latency
        mshr.issue_cycle = self.cycle
        # Energy for DRAM access
        self.total_energy_pj += self.config.dram_access_energy_pj

    def _process_mshr_completions(self) -> None:
        """Process MSHRs with completed data."""
        for mshr in self.mshrs:
            if mshr.state == MSHRState.PENDING:
                mshr.cycles_remaining -= 1
                if mshr.cycles_remaining <= 0:
                    mshr.state = MSHRState.WAITING_DATA
                    self._deliver_mshr_data(mshr)

    def _deliver_mshr_data(self, mshr: MSHREntry) -> None:
        """Deliver data from completed MSHR to all waiters."""
        if mshr.is_store:
            # For stores, write data to global memory
            if self.global_memory is not None:
                for offset, data in mshr.store_data.items():
                    addr = mshr.cache_line_addr + offset
                    self.global_memory.write(addr, data)

            # Queue store completions for all waiters
            for waiter in mshr.waiters:
                self.completion_queues[waiter.partition_id].append((waiter.warp_id, -1, []))
        else:
            # For loads, read cache line from memory
            cache_line_data = self._read_cache_line(mshr.cache_line_addr)

            for waiter in mshr.waiters:
                # Build per-thread result data
                result_data: list[int] = []
                for i in range(32):
                    if waiter.thread_offsets[i] >= 0:
                        offset = waiter.thread_offsets[i]
                        # Read 4-byte word at offset (offset is byte address)
                        word_offset = offset // 4
                        value = (
                            cache_line_data[word_offset]
                            if word_offset < len(cache_line_data)
                            else 0
                        )
                        result_data.append(value)
                    else:
                        result_data.append(0)

                # Queue completion for partition
                self.completion_queues[waiter.partition_id].append(
                    (waiter.warp_id, waiter.dst_reg, result_data)
                )

        # Free MSHR
        mshr.state = MSHRState.INVALID
        mshr.waiters = []
        mshr.store_data = {}

    def _read_cache_line(self, cache_line_addr: int) -> list[int]:
        """Read 128B cache line from global memory as 32 words."""
        if self.global_memory is None:
            return [0] * 32

        words: list[int] = []
        for word_idx in range(32):  # 32 words × 4 bytes = 128 bytes
            addr = cache_line_addr + word_idx * 4
            words.append(self.global_memory.read(addr))
        return words

    # =========================================================================
    # Statistics and Visualization
    # =========================================================================

    def get_statistics(self) -> dict[str, Any]:
        """Get LSU statistics."""
        active_mshrs = sum(1 for m in self.mshrs if m.state != MSHRState.INVALID)
        return {
            "total_primary_misses": self.total_primary_misses,
            "total_secondary_misses": self.total_secondary_misses,
            "secondary_miss_rate": (
                self.total_secondary_misses
                / max(1, self.total_primary_misses + self.total_secondary_misses)
            ),
            "total_mshr_full_stalls": self.total_mshr_full_stalls,
            "total_shared_accesses": self.total_shared_accesses,
            "total_global_accesses": self.total_global_accesses,
            "total_mio_queue_full": self.total_mio_queue_full,
            "total_energy_pj": self.total_energy_pj,
            "active_mshrs": active_mshrs,
            "mshr_utilization": active_mshrs / self.config.num_mshrs,
            "mio_queue_depth": len(self.mio_queue),
            "coalescer_stats": self.coalescer.get_statistics(),
        }

    def get_visualization(self) -> dict[str, Any]:
        """Get visualization data."""
        active_mshrs = [
            {
                "idx": i,
                "addr": hex(m.cache_line_addr),
                "state": m.state.name,
                "waiters": len(m.waiters),
                "cycles_remaining": m.cycles_remaining,
            }
            for i, m in enumerate(self.mshrs)
            if m.state != MSHRState.INVALID
        ]
        return {
            "mio_queue_depth": len(self.mio_queue),
            "mio_queue_max": self.config.mio_queue_depth,
            "active_mshrs": len(active_mshrs),
            "num_mshrs": self.config.num_mshrs,
            "mshr_entries": active_mshrs[:8],  # Show first 8 for display
        }
