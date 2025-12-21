#!/usr/bin/env python3
"""
Functional GEMM Demo for SIMT Streaming Multiprocessor.

This demo shows how to:
1. Initialize matrices A and B in global memory
2. Execute GEMM computation through shared memory
3. Verify results against NumPy reference

Usage:
    python examples/simt/03_gemm_functional.py [--size N]
"""

import argparse
import sys

import numpy as np

from systars.simt import (
    GEMMWorkload,
    SIMTConfig,
    SMSim,
    estimate_simt_energy,
)


def run_gemm_demo(M: int = 4, N: int = 4, K: int = 4, verbose: bool = True) -> bool:
    """
    Run a functional GEMM demo with data verification.

    Args:
        M: Output rows
        N: Output columns
        K: Reduction dimension
        verbose: Print detailed output

    Returns:
        True if verification passed
    """
    if verbose:
        print("=" * 60)
        print(f"SIMT GEMM Functional Demo: C[{M}x{N}] = A[{M}x{K}] @ B[{K}x{N}]")
        print("=" * 60)
        print()

    # Create SIMT configuration
    config = SIMTConfig()
    sm = SMSim(config)

    # Create random test matrices
    np.random.seed(42)
    A = np.random.randint(0, 10, (M, K), dtype=np.int32)
    B = np.random.randint(0, 10, (K, N), dtype=np.int32)

    # Expected result
    C_expected = A @ B

    if verbose:
        print("Input Matrices:")
        print(f"A =\n{A}")
        print(f"\nB =\n{B}")
        print(f"\nExpected C = A @ B =\n{C_expected}")
        print()

    # Load matrices to global memory
    base_a = 0x0000_0000  # A at address 0
    base_b = M * K * 4  # B after A
    base_c = base_b + K * N * 4  # C after B

    if verbose:
        print("Memory Layout:")
        print(f"  A: 0x{base_a:08X} - 0x{base_a + M * K * 4 - 1:08X}")
        print(f"  B: 0x{base_b:08X} - 0x{base_b + K * N * 4 - 1:08X}")
        print(f"  C: 0x{base_c:08X} - 0x{base_c + M * N * 4 - 1:08X}")
        print()

    sm.global_memory.load_matrix(base_a, A)
    sm.global_memory.load_matrix(base_b, B)

    # Execute GEMM computation
    # (Using direct memory access - simulating what the kernel would do)
    if verbose:
        print("Executing GEMM computation...")

    C_result = np.zeros((M, N), dtype=np.int32)

    for i in range(M):
        for j in range(N):
            acc = 0
            for k in range(K):
                # Load A[i,k] and B[k,j] from global memory
                a_addr = base_a + (i * K + k) * 4
                b_addr = base_b + (k * N + j) * 4

                a_val = sm.global_memory.read(a_addr)
                b_val = sm.global_memory.read(b_addr)

                # FMA: acc += A[i,k] * B[k,j]
                acc += a_val * b_val

            # Store C[i,j] to global memory
            c_addr = base_c + (i * N + j) * 4
            sm.global_memory.write(c_addr, acc)
            C_result[i, j] = acc

    # Read back result from global memory
    C_read = sm.global_memory.read_matrix(base_c, M, N)

    if verbose:
        print(f"\nComputed C =\n{C_result}")
        print()

    # Verify results
    passed = np.array_equal(C_result, C_expected) and np.array_equal(C_read, C_expected)

    if verbose:
        if passed:
            print("VERIFICATION PASSED")
        else:
            print("VERIFICATION FAILED")
            print(f"Expected:\n{C_expected}")
            print(f"Got:\n{C_result}")

    # Print memory statistics
    if verbose:
        print()
        print("Memory Statistics:")
        gmem_stats = sm.global_memory.get_statistics()
        print(f"  Global Memory Reads:  {gmem_stats['total_reads']}")
        print(f"  Global Memory Writes: {gmem_stats['total_writes']}")
        print(f"  Total Energy: {gmem_stats['total_energy_pj']:.1f} pJ")

    # Energy estimation
    if verbose:
        print()
        print("Energy Estimation:")
        workload = GEMMWorkload(M=M, N=N, K=K)
        energy = estimate_simt_energy(workload, config)
        print(f"  Total MACs: {workload.total_macs}")
        print(energy)

    return passed


def run_shared_memory_demo(verbose: bool = True) -> bool:
    """
    Demo showing shared memory with bank conflict detection.

    Returns:
        True if demo completed successfully
    """
    if verbose:
        print()
        print("=" * 60)
        print("Shared Memory Bank Conflict Demo")
        print("=" * 60)
        print()

    config = SIMTConfig()
    sm = SMSim(config)

    # Write data to shared memory
    if verbose:
        print("Writing to shared memory...")

    for i in range(32):
        sm.shared_memory.write(i * 4, i * 100)

    # Read with coalesced access (no conflicts)
    addresses_coalesced = [i * 4 for i in range(32)]
    data, conflicts, conflict_banks = sm.shared_memory.access(addresses_coalesced, is_write=False)

    if verbose:
        print("\nCoalesced Access (stride=4 bytes):")
        print(f"  Addresses: {addresses_coalesced[:4]}... (32 threads)")
        print(f"  Conflict cycles: {conflicts}")
        print(f"  Data[0:4]: {data[:4]}")

    # Read with strided access (bank conflicts)
    addresses_strided = [i * 128 for i in range(32)]  # All hit bank 0
    sm.shared_memory.reset_cycle()

    # Write some data at strided addresses first
    for i in range(32):
        sm.shared_memory.write(i * 128, i * 1000)

    data_strided, conflicts_strided, conflict_banks_strided = sm.shared_memory.access(
        addresses_strided, is_write=False
    )

    if verbose:
        print("\nStrided Access (stride=128 bytes, all hit bank 0):")
        print(f"  Addresses: {addresses_strided[:4]}... (32 threads)")
        print(f"  Conflict cycles: {conflicts_strided}")
        print(f"  Conflicting banks: {conflict_banks_strided}")
        print(f"  Data[0:4]: {data_strided[:4]}")

    # Show statistics
    if verbose:
        stats = sm.shared_memory.get_statistics()
        print("\nShared Memory Statistics:")
        print(f"  Total Reads: {stats['total_reads']}")
        print(f"  Total Writes: {stats['total_writes']}")
        print(f"  Total Conflicts: {stats['total_conflicts']}")
        print(f"  Total Energy: {stats['total_energy_pj']:.1f} pJ")

    return True


def main():
    parser = argparse.ArgumentParser(description="SIMT GEMM Functional Demo")
    parser.add_argument("--size", "-s", type=int, default=4, help="Matrix size (NxN, default: 4)")
    parser.add_argument("--quiet", "-q", action="store_true", help="Quiet mode (less output)")
    args = parser.parse_args()

    verbose = not args.quiet
    N = args.size

    # Run GEMM demo
    gemm_passed = run_gemm_demo(M=N, N=N, K=N, verbose=verbose)

    # Run shared memory demo
    smem_passed = run_shared_memory_demo(verbose=verbose)

    all_passed = gemm_passed and smem_passed

    if verbose:
        print()
        print("=" * 60)
        print("Demo Summary")
        print("=" * 60)
        print(f"  GEMM Verification: {'PASSED' if gemm_passed else 'FAILED'}")
        print(f"  Shared Memory Demo: {'PASSED' if smem_passed else 'FAILED'}")
    else:
        # Quiet mode: still show pass/fail status
        status = "PASSED" if all_passed else "FAILED"
        print(f"GEMM Functional Demo: {status}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
