"""
Load/Store Unit (LSU) for SIMT Streaming Multiprocessor.

Executes LD/ST instructions, decodes address space, and routes
memory requests to shared memory or global memory path.

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                   LOAD/STORE UNIT (LSU)                      │
    │                                                              │
    │  ┌──────────────────────────────────────────────────────┐   │
    │  │                   ADDRESS DECODE                      │   │
    │  │   [31:30] = 00: Global Memory → Coalescer → L1/DRAM  │   │
    │  │   [31:30] = 01: Shared Memory → Direct SRAM access   │   │
    │  │   [31:30] = 10: Constant Memory (cached, read-only)  │   │
    │  └──────────────────────────────────────────────────────┘   │
    │                            │                                 │
    │          ┌─────────────────┴─────────────────┐              │
    │          ▼                                   ▼              │
    │  ┌───────────────┐                   ┌───────────────┐      │
    │  │ SHARED MEMORY │                   │ GLOBAL MEMORY │      │
    │  │ (4 cycles)    │                   │ (100+ cycles) │      │
    │  └───────────────┘                   └───────────────┘      │
    │                                                              │
    └─────────────────────────────────────────────────────────────┘
"""

from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Any

from amaranth import Module, Signal, unsigned
from amaranth.lib.wiring import Component, In, Out

from .config import SIMTConfig
from .memory_coalescer import MemoryCoalescerSim
from .shared_memory import SharedMemorySim
from .warp_scheduler import Instruction


class AddressSpace(IntEnum):
    """Memory address space type."""

    GLOBAL = 0  # [31:30] = 00
    SHARED = 1  # [31:30] = 01
    CONSTANT = 2  # [31:30] = 10
    RESERVED = 3  # [31:30] = 11


class MemoryRequestState(IntEnum):
    """State of a memory request in the pipeline."""

    IDLE = 0
    PENDING = auto()  # Request issued, waiting for data
    READY = auto()  # Data ready for writeback


@dataclass
class MemoryRequest:
    """A pending memory request."""

    warp_id: int
    thread_mask: int  # Which threads are active
    is_load: bool  # True for LD, False for ST
    address_space: AddressSpace
    addresses: list[int]  # Per-thread addresses (32 threads)
    dst_reg: int  # Destination register for loads
    store_data: list[int]  # Data for stores (32 threads)
    cycles_remaining: int  # Cycles until completion
    read_data: list[int] = field(default_factory=list)  # Result data


class LoadStoreUnit(Component):
    """
    RTL Load/Store Unit.

    Decodes addresses and routes to shared or global memory.
    """

    def __init__(self, config: SIMTConfig, partition_id: int = 0):
        """
        Initialize LSU.

        Args:
            config: SIMT configuration
            partition_id: Identifier for this partition
        """
        self.config = config
        self.partition_id = partition_id

        warp_bits = max(1, (config.max_warps_per_partition - 1).bit_length())
        reg_bits = max(1, (config.registers_per_partition - 1).bit_length())

        super().__init__(
            {
                # Request from execution unit
                "req_valid": In(1),
                "req_ready": Out(1),
                "req_is_load": In(1),
                "req_warp_id": In(unsigned(warp_bits)),
                "req_addr": In(unsigned(32)),
                "req_dst_reg": In(unsigned(reg_bits)),
                "req_store_data": In(unsigned(32)),
                # Writeback to register file
                "wb_valid": Out(1),
                "wb_warp_id": Out(unsigned(warp_bits)),
                "wb_dst_reg": Out(unsigned(reg_bits)),
                "wb_data": Out(unsigned(32)),
                # Stall signals to scheduler
                "stall_warp_id": Out(unsigned(warp_bits)),
                "stall_valid": Out(1),
                "unstall_warp_id": Out(unsigned(warp_bits)),
                "unstall_valid": Out(1),
                # Status
                "busy": Out(1),
            }
        )

    def elaborate(self, _platform):
        m = Module()

        # Address space decode
        addr_space = Signal(2)
        m.d.comb += addr_space.eq(self.req_addr[30:32])

        # Placeholder implementation
        m.d.comb += [
            self.req_ready.eq(1),
            self.wb_valid.eq(0),
            self.stall_valid.eq(0),
            self.unstall_valid.eq(0),
            self.busy.eq(0),
        ]

        return m


# =============================================================================
# Simulation Model
# =============================================================================


@dataclass
class LoadStoreUnitSim:
    """
    Behavioral simulation model for Load/Store Unit.

    Handles address decoding, latency tracking, and routing to
    shared or global memory.
    """

    config: SIMTConfig
    partition_id: int = 0

    # Shared memory reference (set by partition)
    shared_memory: SharedMemorySim | None = None

    # Global memory reference (set by SM controller) - optional for now
    global_memory: Any = None  # Will be GlobalMemorySim

    # Memory coalescer for global memory access analysis
    coalescer: MemoryCoalescerSim = field(init=False)  # type: ignore[assignment]

    # Pending requests (max 2 for latency hiding)
    pending_requests: list[MemoryRequest] = field(default_factory=list)
    max_pending: int = 2

    # Completed requests ready for writeback
    completed_requests: list[MemoryRequest] = field(default_factory=list)

    # Statistics
    total_loads: int = 0
    total_stores: int = 0
    total_shared_accesses: int = 0
    total_global_accesses: int = 0
    total_coalesced_transactions: int = 0
    total_energy_pj: float = 0.0

    def __post_init__(self) -> None:
        """Initialize coalescer."""
        self.coalescer = MemoryCoalescerSim(self.config)

    def _decode_address_space(self, addr: int) -> AddressSpace:
        """Decode address to determine memory space."""
        space_bits = (addr >> 30) & 0x3
        return AddressSpace(space_bits)

    def _strip_space_bits(self, addr: int) -> int:
        """Remove address space bits to get actual address."""
        return addr & 0x3FFF_FFFF  # Lower 30 bits

    def can_accept(self) -> bool:
        """Check if LSU can accept a new request."""
        return len(self.pending_requests) < self.max_pending

    def issue(
        self,
        warp_id: int,
        instruction: Instruction,
        addresses: list[int],
        store_data: list[int] | None = None,
        thread_mask: int = 0xFFFF_FFFF,
    ) -> bool:
        """
        Issue a load/store instruction.

        Args:
            warp_id: Warp issuing the instruction
            instruction: The LD/ST instruction
            addresses: Per-thread addresses (32 values)
            store_data: Per-thread store data (for ST)
            thread_mask: Active thread mask

        Returns:
            True if accepted, False if queue full
        """
        if not self.can_accept():
            return False

        if store_data is None:
            store_data = [0] * 32

        # Determine address space from first active thread's address
        first_addr = None
        for i in range(32):
            if (thread_mask >> i) & 1 and i < len(addresses):
                first_addr = addresses[i]
                break

        if first_addr is None:
            return True  # No active threads, nothing to do

        addr_space = self._decode_address_space(first_addr)
        is_load = instruction.opcode in ("LD", "LDG", "LDS")

        # Determine latency based on address space
        num_transactions = 1
        if addr_space == AddressSpace.SHARED:
            latency = self.config.shared_mem_latency
        elif addr_space == AddressSpace.GLOBAL:
            # Analyze coalescing for global memory accesses
            stripped_addresses: list[int | None] = [
                self._strip_space_bits(a) if (thread_mask >> i) & 1 else None
                for i, a in enumerate(addresses)
            ]
            coalesce_result = self.coalescer.analyze(stripped_addresses, thread_mask)
            num_transactions = coalesce_result.num_transactions
            self.total_coalesced_transactions += num_transactions

            # Latency scales with number of transactions (simplified)
            # Each transaction adds to the total access time
            latency = self.config.l1_miss_latency * num_transactions
        else:
            latency = self.config.shared_mem_latency

        # Create request
        request = MemoryRequest(
            warp_id=warp_id,
            thread_mask=thread_mask,
            is_load=is_load,
            address_space=addr_space,
            addresses=[self._strip_space_bits(a) for a in addresses],
            dst_reg=instruction.dst if instruction.dst is not None else 0,
            store_data=store_data,
            cycles_remaining=latency,
        )

        self.pending_requests.append(request)

        # Track statistics
        if is_load:
            self.total_loads += 1
        else:
            self.total_stores += 1

        if addr_space == AddressSpace.SHARED:
            self.total_shared_accesses += 1
        else:
            self.total_global_accesses += 1

        return True

    def tick(self) -> list[tuple[int, int, list[int]]]:
        """
        Advance LSU by one cycle.

        Returns:
            List of (warp_id, dst_reg, data) tuples for completed loads
        """
        completed: list[tuple[int, int, list[int]]] = []

        # Process pending requests
        still_pending: list[MemoryRequest] = []

        for req in self.pending_requests:
            req.cycles_remaining -= 1

            if req.cycles_remaining <= 0:
                # Request complete - execute the memory access
                if req.address_space == AddressSpace.SHARED:
                    if self.shared_memory is not None:
                        if req.is_load:
                            read_data, conflicts, _ = self.shared_memory.access(
                                req.addresses,
                                is_write=False,
                                thread_mask=req.thread_mask,
                            )
                            req.read_data = read_data
                            # Add extra cycles for conflicts
                            if conflicts > 0:
                                req.cycles_remaining = conflicts
                                still_pending.append(req)
                                continue
                        else:
                            self.shared_memory.access(
                                req.addresses,
                                is_write=True,
                                write_data=req.store_data,
                                thread_mask=req.thread_mask,
                            )

                elif req.address_space == AddressSpace.GLOBAL:
                    if self.global_memory is not None:
                        if req.is_load:
                            # Read from global memory
                            req.read_data = [
                                self.global_memory.read(addr) if (req.thread_mask >> i) & 1 else 0
                                for i, addr in enumerate(req.addresses)
                            ]
                        else:
                            # Write to global memory
                            for i, addr in enumerate(req.addresses):
                                if (req.thread_mask >> i) & 1:
                                    data = req.store_data[i] if i < len(req.store_data) else 0
                                    self.global_memory.write(addr, data)
                    else:
                        # No global memory - return zeros for loads
                        req.read_data = [0] * 32

                # Request complete
                if req.is_load:
                    completed.append((req.warp_id, req.dst_reg, req.read_data))

                # Energy accounting
                if req.address_space == AddressSpace.SHARED:
                    self.total_energy_pj += self.config.shared_mem_access_energy_pj
                else:
                    self.total_energy_pj += self.config.dram_access_energy_pj
            else:
                still_pending.append(req)

        self.pending_requests = still_pending
        return completed

    def is_busy(self) -> bool:
        """Check if LSU has pending requests."""
        return len(self.pending_requests) > 0

    def get_stalled_warps(self) -> list[int]:
        """Get list of warp IDs waiting for memory."""
        return [req.warp_id for req in self.pending_requests]

    def get_statistics(self) -> dict[str, Any]:
        """Get LSU statistics."""
        return {
            "total_loads": self.total_loads,
            "total_stores": self.total_stores,
            "total_shared_accesses": self.total_shared_accesses,
            "total_global_accesses": self.total_global_accesses,
            "total_coalesced_transactions": self.total_coalesced_transactions,
            "total_energy_pj": self.total_energy_pj,
            "pending_requests": len(self.pending_requests),
            "coalescer_stats": self.coalescer.get_statistics(),
        }

    def get_visualization(self) -> dict[str, Any]:
        """Get visualization data."""
        return {
            "pending": len(self.pending_requests),
            "requests": [
                {
                    "warp_id": req.warp_id,
                    "is_load": req.is_load,
                    "space": req.address_space.name,
                    "cycles_remaining": req.cycles_remaining,
                }
                for req in self.pending_requests
            ],
        }
