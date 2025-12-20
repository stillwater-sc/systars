"""
Tests for SIMT memory architecture components.

Tests shared memory, global memory, load/store unit, memory coalescing,
and functional GEMM execution.
"""

import numpy as np
import pytest

from systars.simt import (
    AddressSpace,
    GlobalMemorySim,
    Instruction,
    LoadStoreUnitSim,
    MemoryCoalescerSim,
    SharedMemorySim,
    SIMTConfig,
    SMSim,
)


class TestSharedMemory:
    """Tests for SharedMemorySim."""

    def test_basic_read_write(self):
        """Test basic read/write operations."""
        config = SIMTConfig()
        smem = SharedMemorySim(config)

        # Write and read back
        smem.write(0, 0xDEADBEEF)
        assert smem.read(0) == 0xDEADBEEF

        smem.write(128, 0x12345678)
        assert smem.read(128) == 0x12345678

    def test_bank_id_calculation(self):
        """Test bank ID calculation."""
        config = SIMTConfig()
        smem = SharedMemorySim(config)

        # Address 0 -> bank 0
        assert smem._get_bank_id(0) == 0
        # Address 4 -> bank 1
        assert smem._get_bank_id(4) == 1
        # Address 128 -> bank 0 (wraps around)
        assert smem._get_bank_id(128) == 0

    def test_bank_conflict_detection(self):
        """Test detection of bank conflicts."""
        config = SIMTConfig()
        smem = SharedMemorySim(config)

        # All threads access same bank (stride = 128 bytes = 32 words)
        # This causes bank conflicts
        addresses = [i * 128 for i in range(32)]  # All hit bank 0

        _, conflict_cycles, conflicting_banks = smem.access(addresses, is_write=False)

        # 32 threads hitting same bank = 31 extra cycles
        assert conflict_cycles == 31
        assert 0 in conflicting_banks

    def test_no_bank_conflict_coalesced(self):
        """Test coalesced access pattern has no conflicts."""
        config = SIMTConfig()
        smem = SharedMemorySim(config)

        # Write initial values
        for i in range(32):
            smem.write(i * 4, i)

        # All threads access consecutive addresses (stride = 4 bytes)
        # Each thread hits a different bank
        addresses = [i * 4 for i in range(32)]

        data, conflict_cycles, _ = smem.access(addresses, is_write=False)

        assert conflict_cycles == 0
        assert data == list(range(32))


class TestGlobalMemory:
    """Tests for GlobalMemorySim."""

    def test_basic_read_write(self):
        """Test basic read/write operations."""
        config = SIMTConfig()
        gmem = GlobalMemorySim(config)

        gmem.write(0, 42)
        assert gmem.read(0) == 42

        gmem.write(1000, 0xABCD)
        assert gmem.read(1000) == 0xABCD

    def test_matrix_load_read(self):
        """Test matrix load and read operations."""
        config = SIMTConfig()
        gmem = GlobalMemorySim(config)

        # Create a 4x4 matrix
        matrix = np.array(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
            dtype=np.int32,
        )

        # Load matrix to global memory at address 0
        gmem.load_matrix(0, matrix)

        # Read it back
        result = gmem.read_matrix(0, 4, 4)

        np.testing.assert_array_equal(result, matrix)

    def test_bulk_operations(self):
        """Test bulk read/write operations."""
        config = SIMTConfig()
        gmem = GlobalMemorySim(config)

        values = list(range(100))
        gmem.write_bulk(0, values)

        result = gmem.read_bulk(0, 100)
        assert result == values


class TestMemoryCoalescer:
    """Tests for MemoryCoalescerSim."""

    def test_fully_coalesced_access(self):
        """Test fully coalesced access pattern."""
        config = SIMTConfig()
        coalescer = MemoryCoalescerSim(config)

        # Consecutive 4-byte accesses = 1 transaction
        addresses = [i * 4 for i in range(32)]
        result = coalescer.analyze(addresses)

        # 32 threads * 4 bytes = 128 bytes = 1 transaction
        assert result.num_transactions == 1
        assert result.is_fully_coalesced
        assert result.efficiency == 1.0

    def test_strided_access(self):
        """Test strided access pattern (worst case)."""
        config = SIMTConfig()
        coalescer = MemoryCoalescerSim(config)

        # Each thread hits different 128-byte segment
        addresses = [i * 128 for i in range(32)]
        result = coalescer.analyze(addresses)

        # 32 transactions (each thread hits different segment)
        assert result.num_transactions == 32
        assert not result.is_fully_coalesced
        # 128 bytes accessed, 32 * 128 transferred
        assert result.efficiency < 0.1

    def test_broadcast_access(self):
        """Test broadcast access (all threads same address)."""
        config = SIMTConfig()
        coalescer = MemoryCoalescerSim(config)

        # All threads access same address
        addresses = [0] * 32
        result = coalescer.analyze(addresses)

        # 1 transaction (broadcast is handled specially)
        assert result.num_transactions == 1
        assert result.is_fully_coalesced


class TestLoadStoreUnit:
    """Tests for LoadStoreUnitSim."""

    def test_address_space_decode(self):
        """Test address space decoding."""
        config = SIMTConfig()
        lsu = LoadStoreUnitSim(config)

        # Global memory: [31:30] = 00
        assert lsu._decode_address_space(0x0000_0000) == AddressSpace.GLOBAL
        assert lsu._decode_address_space(0x1000_0000) == AddressSpace.GLOBAL

        # Shared memory: [31:30] = 01
        assert lsu._decode_address_space(0x4000_0000) == AddressSpace.SHARED
        assert lsu._decode_address_space(0x4000_1000) == AddressSpace.SHARED

        # Constant memory: [31:30] = 10
        assert lsu._decode_address_space(0x8000_0000) == AddressSpace.CONSTANT

    def test_shared_memory_access(self):
        """Test shared memory access through LSU."""
        config = SIMTConfig()
        lsu = LoadStoreUnitSim(config)
        smem = SharedMemorySim(config)
        lsu.shared_memory = smem

        # Write data to shared memory
        smem.write(0, 42)
        smem.write(4, 100)

        # Create LD instruction for shared memory
        instr = Instruction(opcode="LD", dst=1, src1=0)

        # Issue load from shared memory base (0x4000_0000)
        addresses = [0x4000_0000 + i * 4 for i in range(32)]
        assert lsu.issue(0, instr, addresses)

        # Advance cycles until completion
        for _ in range(config.shared_mem_latency + 1):
            completed = lsu.tick()
            if completed:
                warp_id, dst_reg, data = completed[0]
                assert warp_id == 0
                assert dst_reg == 1
                assert data[0] == 42
                assert data[1] == 100
                break


class TestSMSimMemory:
    """Tests for SMSim with memory architecture."""

    def test_sm_has_global_memory(self):
        """Test that SM has global memory attached."""
        sm = SMSim()
        assert sm.global_memory is not None
        assert isinstance(sm.global_memory, GlobalMemorySim)

    def test_sm_has_shared_memory(self):
        """Test that SM has SM-wide shared memory."""
        sm = SMSim()
        assert sm.shared_memory is not None
        assert isinstance(sm.shared_memory, SharedMemorySim)

    def test_partitions_share_memory(self):
        """Test that all partitions share the same memory."""
        sm = SMSim()

        # All partitions should reference the same shared memory
        for partition in sm.partitions:
            assert partition.shared_memory is sm.shared_memory
            assert partition.load_store_unit.shared_memory is sm.shared_memory
            assert partition.load_store_unit.global_memory is sm.global_memory

    def test_global_memory_persistence(self):
        """Test that global memory data persists across partition accesses."""
        sm = SMSim()

        # Write data from "partition 0's perspective"
        sm.global_memory.write(0, 0xDEADBEEF)

        # Read from "partition 1's perspective"
        value = sm.global_memory.read(0)
        assert value == 0xDEADBEEF


class TestFunctionalGEMM:
    """Functional tests for GEMM execution with data verification."""

    def test_simple_matrix_multiply_manual(self):
        """Test manual matrix multiplication through shared memory."""
        config = SIMTConfig()
        sm = SMSim(config)

        # Set up 2x2 matrices in shared memory
        # A = [[1, 2], [3, 4]]
        # B = [[5, 6], [7, 8]]
        # C = A @ B = [[19, 22], [43, 50]]

        # Write A to shared memory at offset 0
        sm.shared_memory.write(0, 1)  # A[0,0]
        sm.shared_memory.write(4, 2)  # A[0,1]
        sm.shared_memory.write(8, 3)  # A[1,0]
        sm.shared_memory.write(12, 4)  # A[1,1]

        # Write B to shared memory at offset 16
        sm.shared_memory.write(16, 5)  # B[0,0]
        sm.shared_memory.write(20, 6)  # B[0,1]
        sm.shared_memory.write(24, 7)  # B[1,0]
        sm.shared_memory.write(28, 8)  # B[1,1]

        # Verify reads
        assert sm.shared_memory.read(0) == 1
        assert sm.shared_memory.read(4) == 2
        assert sm.shared_memory.read(16) == 5
        assert sm.shared_memory.read(28) == 8

        # Compute C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] = 1*5 + 2*7 = 19
        c00 = sm.shared_memory.read(0) * sm.shared_memory.read(16) + sm.shared_memory.read(
            4
        ) * sm.shared_memory.read(24)
        assert c00 == 19

        # Compute C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] = 1*6 + 2*8 = 22
        c01 = sm.shared_memory.read(0) * sm.shared_memory.read(20) + sm.shared_memory.read(
            4
        ) * sm.shared_memory.read(28)
        assert c01 == 22

        # Compute C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] = 3*5 + 4*7 = 43
        c10 = sm.shared_memory.read(8) * sm.shared_memory.read(16) + sm.shared_memory.read(
            12
        ) * sm.shared_memory.read(24)
        assert c10 == 43

        # Compute C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1] = 3*6 + 4*8 = 50
        c11 = sm.shared_memory.read(8) * sm.shared_memory.read(20) + sm.shared_memory.read(
            12
        ) * sm.shared_memory.read(28)
        assert c11 == 50

    def test_global_memory_matrix_roundtrip(self):
        """Test loading matrices to global memory and reading back."""
        config = SIMTConfig()
        sm = SMSim(config)

        # Create test matrices
        A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int32)
        B = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]], dtype=np.int32)

        # Load to global memory
        base_a = 0x0000_0000  # Global memory region
        base_b = 0x0000_0100

        sm.global_memory.load_matrix(base_a, A)
        sm.global_memory.load_matrix(base_b, B)

        # Read back and verify
        A_read = sm.global_memory.read_matrix(base_a, 3, 3)
        B_read = sm.global_memory.read_matrix(base_b, 3, 3)

        np.testing.assert_array_equal(A_read, A)
        np.testing.assert_array_equal(B_read, B)

    def test_gemm_reference_computation(self):
        """Test GEMM computation using NumPy as reference."""
        config = SIMTConfig()
        sm = SMSim(config)

        # Create random matrices
        np.random.seed(42)
        M, N, K = 4, 4, 4
        A = np.random.randint(0, 10, (M, K), dtype=np.int32)
        B = np.random.randint(0, 10, (K, N), dtype=np.int32)

        # Expected result
        C_expected = A @ B

        # Load matrices to global memory
        base_a = 0
        base_b = M * K * 4
        base_c = base_b + K * N * 4

        sm.global_memory.load_matrix(base_a, A)
        sm.global_memory.load_matrix(base_b, B)

        # Manually compute C element by element (simulating kernel)
        C_result = np.zeros((M, N), dtype=np.int32)
        for i in range(M):
            for j in range(N):
                acc = 0
                for k in range(K):
                    a_val = sm.global_memory.read(base_a + (i * K + k) * 4)
                    b_val = sm.global_memory.read(base_b + (k * N + j) * 4)
                    acc += a_val * b_val
                C_result[i, j] = acc
                sm.global_memory.write(base_c + (i * N + j) * 4, acc)

        # Verify computation
        np.testing.assert_array_equal(C_result, C_expected)

        # Read back from global memory and verify
        C_read = sm.global_memory.read_matrix(base_c, M, N)
        np.testing.assert_array_equal(C_read, C_expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
