"""
Shared Memory for SIMT Streaming Multiprocessor.

Implements banked shared memory with conflict detection, following
NVIDIA's unified L1/shared memory architecture (Volta onwards).

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │              SHARED MEMORY (48KB default, 32 banks)          │
    │                                                              │
    │  ┌────┬────┬────┬────┬────┬────┬────┬────┬─────┬────┐       │
    │  │ B0 │ B1 │ B2 │ B3 │ B4 │ B5 │ B6 │ B7 │ ... │B31 │       │
    │  └────┴────┴────┴────┴────┴────┴────┴────┴─────┴────┘       │
    │                                                              │
    │  Bank selection: bank_id = (addr >> 2) % 32                  │
    │  Conflict: Multiple threads accessing same bank              │
    │  Serialization: N conflicts = N extra cycles                 │
    └─────────────────────────────────────────────────────────────┘
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Any

from amaranth import Module, unsigned
from amaranth.lib.memory import Memory
from amaranth.lib.wiring import Component, In, Out

from .config import SIMTConfig


class SharedMemoryBankState(IntEnum):
    """State of a shared memory bank."""

    IDLE = 0
    READ = auto()
    WRITE = auto()
    CONFLICT = auto()


class SharedMemoryBank(Component):
    """
    Single shared memory bank (SRAM).

    Each bank is independently addressable with single-cycle read latency.
    """

    def __init__(self, config: SIMTConfig, bank_id: int = 0):
        """
        Initialize shared memory bank.

        Args:
            config: SIMT configuration
            bank_id: Identifier for this bank
        """
        self.config = config
        self.bank_id = bank_id

        # Bank size in 32-bit words
        self.bank_words = config.shared_memory_words_per_bank
        self.addr_bits = max(1, (self.bank_words - 1).bit_length())

        super().__init__(
            {
                # Read port
                "read_addr": In(unsigned(self.addr_bits)),
                "read_en": In(1),
                "read_data": Out(unsigned(32)),
                "read_valid": Out(1),
                # Write port
                "write_addr": In(unsigned(self.addr_bits)),
                "write_en": In(1),
                "write_data": In(unsigned(32)),
            }
        )

    def elaborate(self, _platform):
        m = Module()

        # Memory storage
        m.submodules.mem = mem = Memory(
            shape=unsigned(32),
            depth=self.bank_words,
            init=[],
        )

        # Read port
        rd_port = mem.read_port()
        m.d.comb += [
            rd_port.addr.eq(self.read_addr),
            rd_port.en.eq(self.read_en),
            self.read_data.eq(rd_port.data),
        ]

        # One-cycle read latency
        m.d.sync += self.read_valid.eq(self.read_en)

        # Write port
        wr_port = mem.write_port()
        m.d.comb += [
            wr_port.addr.eq(self.write_addr),
            wr_port.en.eq(self.write_en),
            wr_port.data.eq(self.write_data),
        ]

        return m


class SharedMemory(Component):
    """
    Multi-bank shared memory with conflict detection.

    Supports parallel access from 32 threads (one warp), with
    bank conflict detection and serialization.
    """

    def __init__(self, config: SIMTConfig):
        """
        Initialize shared memory.

        Args:
            config: SIMT configuration
        """
        self.config = config
        self.num_banks = config.shared_memory_banks
        self.bank_words = config.shared_memory_words_per_bank
        self.addr_bits = max(1, (config.shared_memory_words - 1).bit_length())

        warp_bits = max(1, (config.max_warps_per_sm - 1).bit_length())

        super().__init__(
            {
                # Request interface (from LSU)
                "req_valid": In(1),
                "req_ready": Out(1),
                "req_warp_id": In(unsigned(warp_bits)),
                "req_is_write": In(1),
                "req_addr": In(unsigned(self.addr_bits)),
                "req_data": In(unsigned(32)),
                "req_thread_id": In(unsigned(5)),  # 0-31
                # Response interface
                "resp_valid": Out(1),
                "resp_data": Out(unsigned(32)),
                "resp_warp_id": Out(unsigned(warp_bits)),
                # Status
                "conflict": Out(1),
            }
        )

    def elaborate(self, _platform):
        m = Module()

        # Instantiate banks
        for i in range(self.num_banks):
            bank = SharedMemoryBank(self.config, bank_id=i)
            m.submodules[f"bank{i}"] = bank

        # Bank selection and routing would go here
        # For now, placeholder implementation
        m.d.comb += [
            self.req_ready.eq(1),
            self.resp_valid.eq(0),
            self.resp_data.eq(0),
            self.conflict.eq(0),
        ]

        return m


# =============================================================================
# Simulation Model
# =============================================================================


@dataclass
class SharedMemorySim:
    """
    Behavioral simulation model for shared memory.

    Provides cycle-accurate simulation with bank conflict detection,
    following the RegisterFileSim pattern.
    """

    config: SIMTConfig

    # Storage: list of banks, each bank is a dict[addr] -> value
    banks: list[dict[int, int]] = field(default_factory=list)

    # Bank status for visualization
    bank_status: list[SharedMemoryBankState] = field(default_factory=list)

    # Statistics
    total_reads: int = 0
    total_writes: int = 0
    total_conflicts: int = 0
    total_energy_pj: float = 0.0

    def __post_init__(self) -> None:
        """Initialize bank storage and status."""
        self.banks = [{} for _ in range(self.config.shared_memory_banks)]
        self.bank_status = [
            SharedMemoryBankState.IDLE for _ in range(self.config.shared_memory_banks)
        ]

    def _get_bank_id(self, addr: int) -> int:
        """
        Get bank ID for address.

        Bank selection: bank_id = (addr >> 2) % num_banks
        This maps consecutive 4-byte words to different banks.
        """
        return (addr >> 2) % self.config.shared_memory_banks

    def _get_bank_offset(self, addr: int) -> int:
        """
        Get offset within bank for address.

        After bank selection, the remaining address bits give the offset.
        """
        return (addr >> 2) // self.config.shared_memory_banks

    def reset_cycle(self) -> None:
        """Reset bank status for new cycle."""
        self.bank_status = [
            SharedMemoryBankState.IDLE for _ in range(self.config.shared_memory_banks)
        ]

    def access(
        self,
        addresses: Sequence[int | None],
        is_write: bool,
        write_data: Sequence[int] | None = None,
        thread_mask: int = 0xFFFF_FFFF,
    ) -> tuple[list[int], int, list[int]]:
        """
        Access shared memory for 32 threads in a warp.

        Args:
            addresses: List of 32 addresses (one per thread), None if inactive
            is_write: True for store, False for load
            write_data: List of 32 data values for writes
            thread_mask: Bitmask of active threads

        Returns:
            Tuple of:
                - read_data: List of 32 read values (0 for writes/inactive)
                - conflict_cycles: Number of extra cycles due to conflicts
                - conflicting_banks: List of bank IDs that had conflicts
        """
        if write_data is None:
            write_data = [0] * 32

        # Track which banks are accessed by which threads
        bank_accesses: dict[int, list[int]] = {}  # bank_id -> [thread_ids]

        for thread_id in range(32):
            if not ((thread_mask >> thread_id) & 1):
                continue
            addr = addresses[thread_id] if thread_id < len(addresses) else None
            if addr is None:
                continue

            bank_id = self._get_bank_id(addr)
            if bank_id not in bank_accesses:
                bank_accesses[bank_id] = []
            bank_accesses[bank_id].append(thread_id)

        # Detect conflicts (multiple threads accessing same bank)
        conflict_cycles = 0
        conflicting_banks: list[int] = []

        for bank_id, thread_ids in bank_accesses.items():
            if len(thread_ids) > 1:
                # Each additional thread after the first causes 1 extra cycle
                conflict_cycles += len(thread_ids) - 1
                conflicting_banks.append(bank_id)
                self.bank_status[bank_id] = SharedMemoryBankState.CONFLICT
            else:
                self.bank_status[bank_id] = (
                    SharedMemoryBankState.WRITE if is_write else SharedMemoryBankState.READ
                )

        # Perform the actual memory access
        read_data: list[int] = []

        for thread_id in range(32):
            if not ((thread_mask >> thread_id) & 1):
                read_data.append(0)
                continue

            addr = addresses[thread_id] if thread_id < len(addresses) else None
            if addr is None:
                read_data.append(0)
                continue

            bank_id = self._get_bank_id(addr)
            offset = self._get_bank_offset(addr)

            if is_write:
                # Store
                data = write_data[thread_id] if thread_id < len(write_data) else 0
                self.banks[bank_id][offset] = data
                self.total_writes += 1
                read_data.append(0)
            else:
                # Load
                value = self.banks[bank_id].get(offset, 0)
                self.total_reads += 1
                read_data.append(value)

        # Energy accounting
        num_accesses = sum(
            1 for i in range(32) if ((thread_mask >> i) & 1) and addresses[i] is not None
        )
        self.total_energy_pj += num_accesses * self.config.shared_mem_access_energy_pj
        self.total_conflicts += len(conflicting_banks)
        self.total_energy_pj += conflict_cycles * self.config.shared_mem_conflict_energy_pj

        return read_data, conflict_cycles, conflicting_banks

    def read(self, addr: int) -> int:
        """
        Single-thread read for testing.

        Args:
            addr: Byte address

        Returns:
            32-bit value at address
        """
        bank_id = self._get_bank_id(addr)
        offset = self._get_bank_offset(addr)
        return self.banks[bank_id].get(offset, 0)

    def write(self, addr: int, value: int) -> None:
        """
        Single-thread write for testing.

        Args:
            addr: Byte address
            value: 32-bit value to write
        """
        bank_id = self._get_bank_id(addr)
        offset = self._get_bank_offset(addr)
        self.banks[bank_id][offset] = value & 0xFFFF_FFFF

    def get_statistics(self) -> dict[str, Any]:
        """Get shared memory statistics."""
        return {
            "total_reads": self.total_reads,
            "total_writes": self.total_writes,
            "total_conflicts": self.total_conflicts,
            "total_energy_pj": self.total_energy_pj,
        }

    def get_visualization(self) -> list[str]:
        """
        Get visualization of bank states.

        Returns:
            List of 2-character state strings for each bank.
        """
        state_map = {
            SharedMemoryBankState.IDLE: "░░",
            SharedMemoryBankState.READ: "██",
            SharedMemoryBankState.WRITE: "▓▓",
            SharedMemoryBankState.CONFLICT: "XX",
        }
        return [state_map[s] for s in self.bank_status]
