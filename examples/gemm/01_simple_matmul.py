#!/usr/bin/env python3
"""
Simple Matrix Multiply Demo.

This example demonstrates a basic matrix multiply operation C = A @ B
using the systolic array accelerator. It shows:

1. Problem Setup
   - Define input matrices A and B using NumPy
   - Configure the systolic array hardware parameters

2. Memory Layout
   - Plan DRAM addresses for inputs and outputs
   - Load matrices into simulated DRAM

3. Command Sequence Generation
   - Generate the sequence of commands for the accelerator:
     * LOAD A from DRAM to Scratchpad
     * LOAD B from DRAM to Scratchpad
     * EXECUTE matrix multiply
     * STORE C from Accumulator to DRAM

4. Execution (RTL Simulation)
   - Run the command sequence through the hardware model
   - Monitor progress and completion

5. Verification
   - Read results from DRAM
   - Compare against NumPy reference computation

Usage:
    python 01_simple_matmul.py [--size N] [--simulate]

    --size N      Matrix size (default: 4, creates NxN matrices)
    --simulate    Run RTL simulation (requires Amaranth)
"""

import argparse
import sys
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from examples.common import SimulatedDRAM

# =============================================================================
# Hardware Configuration
# =============================================================================


@dataclass
class AcceleratorConfig:
    """
    Systolic array accelerator configuration.

    These parameters define the hardware capabilities and must match
    the RTL configuration used during synthesis.
    """

    # Array dimensions
    array_rows: int = 4  # Number of PE rows
    array_cols: int = 4  # Number of PE columns

    # Data widths
    input_bits: int = 8  # Input/weight element width
    acc_bits: int = 32  # Accumulator width

    # Memory sizes
    scratchpad_kb: int = 64  # Scratchpad capacity
    accumulator_kb: int = 16  # Accumulator capacity

    # Bus parameters
    dma_buswidth: int = 128  # DMA bus width in bits

    @property
    def bytes_per_beat(self) -> int:
        return self.dma_buswidth // 8

    @property
    def elements_per_beat_int8(self) -> int:
        return self.dma_buswidth // 8

    @property
    def elements_per_beat_int32(self) -> int:
        return self.dma_buswidth // 32


# =============================================================================
# Command Definitions
# =============================================================================


class CommandType(IntEnum):
    """Command types for the accelerator."""

    LOAD = 0  # Load data from DRAM to Scratchpad
    STORE = 1  # Store data from Accumulator to DRAM
    EXECUTE = 2  # Execute compute operation


class LoadOpcode(IntEnum):
    """Load controller opcodes."""

    MEMCPY = 0  # Simple memory copy


class StoreOpcode(IntEnum):
    """Store controller opcodes."""

    MEMCPY = 0  # Simple memory copy
    MEMCPY_RELU = 1  # Memory copy with ReLU activation


class ExecOpcode(IntEnum):
    """Execute controller opcodes."""

    CONFIG = 0  # Configure execution parameters
    PRELOAD = 1  # Preload bias/initial values
    COMPUTE = 2  # Execute matrix multiply


@dataclass
class Command:
    """A command for the accelerator."""

    cmd_type: CommandType
    opcode: int
    dram_addr: int = 0
    local_addr: int = 0
    length: int = 0
    cmd_id: int = 0
    extra: dict = None

    def __post_init__(self):
        if self.extra is None:
            self.extra = {}

    def __str__(self) -> str:
        type_name = self.cmd_type.name
        if self.cmd_type == CommandType.LOAD:
            return (
                f"LOAD[{self.cmd_id}]: DRAM[0x{self.dram_addr:X}] -> "
                f"SP[0x{self.local_addr:X}], len={self.length}"
            )
        elif self.cmd_type == CommandType.STORE:
            return (
                f"STORE[{self.cmd_id}]: ACC[0x{self.local_addr:X}] -> "
                f"DRAM[0x{self.dram_addr:X}], len={self.length}"
            )
        elif self.cmd_type == CommandType.EXECUTE:
            op_name = ExecOpcode(self.opcode).name
            return f"EXEC[{self.cmd_id}]: {op_name} {self.extra}"
        return f"{type_name}[{self.cmd_id}]"


# =============================================================================
# Command Sequence Generator
# =============================================================================


class CommandGenerator:
    """
    Generates command sequences for matrix operations.

    This class takes high-level matrix operations and produces the
    sequence of load/execute/store commands needed to run them on
    the systolic array.
    """

    def __init__(self, config: AcceleratorConfig):
        self.config = config
        self.cmd_id = 0

    def next_id(self) -> int:
        """Get next command ID."""
        self.cmd_id += 1
        return self.cmd_id

    def generate_matmul(
        self,
        a_dram_addr: int,
        b_dram_addr: int,
        c_dram_addr: int,
        a_sp_addr: int,
        b_sp_addr: int,
        c_acc_addr: int,
        M: int,
        K: int,
        N: int,
    ) -> list[Command]:
        """
        Generate command sequence for C = A @ B.

        Args:
            a_dram_addr: DRAM address of matrix A
            b_dram_addr: DRAM address of matrix B
            c_dram_addr: DRAM address for result C
            a_sp_addr: Scratchpad address for A
            b_sp_addr: Scratchpad address for B
            c_acc_addr: Accumulator address for C
            M: Rows in A and C
            K: Cols in A, rows in B
            N: Cols in B and C

        Returns:
            List of commands to execute
        """
        commands = []

        # Calculate transfer lengths (in bus beats)
        a_bytes = M * K  # int8 elements
        b_bytes = K * N
        c_bytes = M * N * 4  # int32 results

        a_beats = (a_bytes + self.config.bytes_per_beat - 1) // self.config.bytes_per_beat
        b_beats = (b_bytes + self.config.bytes_per_beat - 1) // self.config.bytes_per_beat
        c_beats = (c_bytes + self.config.bytes_per_beat - 1) // self.config.bytes_per_beat

        # 1. Configure execute controller (output-stationary mode)
        commands.append(
            Command(
                cmd_type=CommandType.EXECUTE,
                opcode=ExecOpcode.CONFIG,
                cmd_id=self.next_id(),
                extra={"dataflow": "OS", "shift": 0},
            )
        )

        # 2. Load matrix A: DRAM -> Scratchpad
        commands.append(
            Command(
                cmd_type=CommandType.LOAD,
                opcode=LoadOpcode.MEMCPY,
                dram_addr=a_dram_addr,
                local_addr=a_sp_addr,
                length=a_beats,
                cmd_id=self.next_id(),
            )
        )

        # 3. Load matrix B: DRAM -> Scratchpad
        commands.append(
            Command(
                cmd_type=CommandType.LOAD,
                opcode=LoadOpcode.MEMCPY,
                dram_addr=b_dram_addr,
                local_addr=b_sp_addr,
                length=b_beats,
                cmd_id=self.next_id(),
            )
        )

        # 4. Execute matrix multiply
        commands.append(
            Command(
                cmd_type=CommandType.EXECUTE,
                opcode=ExecOpcode.COMPUTE,
                local_addr=c_acc_addr,
                cmd_id=self.next_id(),
                extra={
                    "a_sp_addr": a_sp_addr,
                    "b_sp_addr": b_sp_addr,
                    "M": M,
                    "K": K,
                    "N": N,
                },
            )
        )

        # 5. Store result C: Accumulator -> DRAM
        commands.append(
            Command(
                cmd_type=CommandType.STORE,
                opcode=StoreOpcode.MEMCPY,
                dram_addr=c_dram_addr,
                local_addr=c_acc_addr,
                length=c_beats,
                cmd_id=self.next_id(),
            )
        )

        return commands


# =============================================================================
# Software Simulation (No RTL)
# =============================================================================


def simulate_matmul_software(
    dram: SimulatedDRAM,
    commands: list[Command],
    config: AcceleratorConfig,
) -> None:
    """
    Simulate matrix multiply in software (no RTL).

    This provides a functional model of what the hardware does,
    useful for verifying the command sequence before RTL simulation.
    """
    # Simulated local memories
    scratchpad = {}  # addr -> data
    accumulator = {}  # addr -> data

    for cmd in commands:
        if cmd.cmd_type == CommandType.LOAD:
            # Copy from DRAM to scratchpad
            print(f"  Executing: {cmd}")
            for i in range(cmd.length):
                beat_addr = cmd.dram_addr + i * config.bytes_per_beat
                data = dram.read_beat(beat_addr)
                sp_addr = cmd.local_addr + i
                scratchpad[sp_addr] = data

        elif cmd.cmd_type == CommandType.STORE:
            # Copy from accumulator to DRAM
            print(f"  Executing: {cmd}")
            for i in range(cmd.length):
                acc_addr = cmd.local_addr + i
                data = accumulator.get(acc_addr, 0)
                beat_addr = cmd.dram_addr + i * config.bytes_per_beat
                dram.write_beat(beat_addr, data)

        elif cmd.cmd_type == CommandType.EXECUTE:
            print(f"  Executing: {cmd}")
            if cmd.opcode == ExecOpcode.CONFIG:
                pass  # Configuration is handled implicitly

            elif cmd.opcode == ExecOpcode.COMPUTE:
                # Extract matrix dimensions and addresses
                extra = cmd.extra
                M, K, N = extra["M"], extra["K"], extra["N"]
                a_sp_addr = extra["a_sp_addr"]
                b_sp_addr = extra["b_sp_addr"]
                c_acc_addr = cmd.local_addr

                # Reconstruct matrices from scratchpad
                # (simplified - assumes data fits in loaded beats)
                a_bytes = bytearray()
                b_bytes = bytearray()

                a_beats = (M * K + config.bytes_per_beat - 1) // config.bytes_per_beat
                b_beats = (K * N + config.bytes_per_beat - 1) // config.bytes_per_beat

                for i in range(a_beats):
                    beat = scratchpad.get(a_sp_addr + i, 0)
                    for b in range(config.bytes_per_beat):
                        a_bytes.append((beat >> (b * 8)) & 0xFF)

                for i in range(b_beats):
                    beat = scratchpad.get(b_sp_addr + i, 0)
                    for b in range(config.bytes_per_beat):
                        b_bytes.append((beat >> (b * 8)) & 0xFF)

                # Convert to numpy arrays
                A = np.array(
                    [(x if x < 128 else x - 256) for x in a_bytes[: M * K]],
                    dtype=np.int8,
                ).reshape(M, K)

                B = np.array(
                    [(x if x < 128 else x - 256) for x in b_bytes[: K * N]],
                    dtype=np.int8,
                ).reshape(K, N)

                # Compute result
                C = A.astype(np.int32) @ B.astype(np.int32)
                print(f"    Computed C[{M}x{N}] = A[{M}x{K}] @ B[{K}x{N}]")

                # Store result in accumulator
                c_flat = C.flatten()
                elements_per_beat = config.dma_buswidth // 32

                for i in range(0, len(c_flat), elements_per_beat):
                    beat = 0
                    for j in range(elements_per_beat):
                        if i + j < len(c_flat):
                            val = int(c_flat[i + j])
                            if val < 0:
                                val += 1 << 32
                            beat |= (val & 0xFFFFFFFF) << (j * 32)
                    accumulator[c_acc_addr + i // elements_per_beat] = beat


# =============================================================================
# Main Demo
# =============================================================================


def run_demo(matrix_size: int = 4, run_rtl: bool = False):
    """
    Run the simple matrix multiply demonstration.

    Args:
        matrix_size: Size of square matrices (NxN)
        run_rtl: If True, run RTL simulation (requires Amaranth)
    """
    print("=" * 70)
    print("Simple Matrix Multiply Demo")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # 1. Problem Setup
    # -------------------------------------------------------------------------
    print("\n1. Problem Setup")
    print("-" * 40)

    M, K, N = matrix_size, matrix_size, matrix_size
    print(f"   Computing C[{M}x{N}] = A[{M}x{K}] @ B[{K}x{N}]")

    # Create random input matrices
    np.random.seed(42)  # Reproducible results
    A = np.random.randint(-10, 10, size=(M, K), dtype=np.int8)
    B = np.random.randint(-10, 10, size=(K, N), dtype=np.int8)

    print(f"\n   Matrix A ({M}x{K}):")
    print(A)
    print(f"\n   Matrix B ({K}x{N}):")
    print(B)

    # Compute expected result
    C_expected = A.astype(np.int32) @ B.astype(np.int32)
    print(f"\n   Expected C ({M}x{N}):")
    print(C_expected)

    # -------------------------------------------------------------------------
    # 2. Hardware Configuration
    # -------------------------------------------------------------------------
    print("\n2. Hardware Configuration")
    print("-" * 40)

    config = AcceleratorConfig(
        array_rows=matrix_size,
        array_cols=matrix_size,
        input_bits=8,
        acc_bits=32,
        dma_buswidth=128,
    )

    print(f"   Array size: {config.array_rows} x {config.array_cols}")
    print(f"   Input precision: {config.input_bits}-bit")
    print(f"   Accumulator precision: {config.acc_bits}-bit")
    print(f"   DMA bus width: {config.dma_buswidth}-bit")
    print(f"   Bytes per beat: {config.bytes_per_beat}")

    # -------------------------------------------------------------------------
    # 3. Memory Layout
    # -------------------------------------------------------------------------
    print("\n3. Memory Layout")
    print("-" * 40)

    # DRAM addresses (could be any physical addresses)
    A_DRAM_ADDR = 0x0000_1000
    B_DRAM_ADDR = 0x0000_2000
    C_DRAM_ADDR = 0x0000_3000

    # Scratchpad addresses (local to accelerator)
    A_SP_ADDR = 0x00
    B_SP_ADDR = 0x10

    # Accumulator address
    C_ACC_ADDR = 0x00

    print("   DRAM Layout:")
    print(f"     A: 0x{A_DRAM_ADDR:08X} ({M * K} bytes)")
    print(f"     B: 0x{B_DRAM_ADDR:08X} ({K * N} bytes)")
    print(f"     C: 0x{C_DRAM_ADDR:08X} ({M * N * 4} bytes)")
    print("\n   Scratchpad Layout:")
    print(f"     A: 0x{A_SP_ADDR:02X}")
    print(f"     B: 0x{B_SP_ADDR:02X}")
    print("\n   Accumulator Layout:")
    print(f"     C: 0x{C_ACC_ADDR:02X}")

    # Initialize DRAM and load input matrices
    dram = SimulatedDRAM(size_bytes=64 * 1024, buswidth=config.dma_buswidth)
    dram.store_matrix("A", A_DRAM_ADDR, A)
    dram.store_matrix("B", B_DRAM_ADDR, B)

    dram.print_memory_map()

    # -------------------------------------------------------------------------
    # 4. Command Sequence Generation
    # -------------------------------------------------------------------------
    print("\n4. Command Sequence")
    print("-" * 40)

    generator = CommandGenerator(config)
    commands = generator.generate_matmul(
        a_dram_addr=A_DRAM_ADDR,
        b_dram_addr=B_DRAM_ADDR,
        c_dram_addr=C_DRAM_ADDR,
        a_sp_addr=A_SP_ADDR,
        b_sp_addr=B_SP_ADDR,
        c_acc_addr=C_ACC_ADDR,
        M=M,
        K=K,
        N=N,
    )

    print(f"   Generated {len(commands)} commands:")
    for cmd in commands:
        print(f"     {cmd}")

    # -------------------------------------------------------------------------
    # 5. Execution
    # -------------------------------------------------------------------------
    print("\n5. Execution")
    print("-" * 40)

    if run_rtl:
        print("   Running RTL simulation...")
        # TODO: Integrate with Amaranth RTL simulation
        print("   [RTL simulation not yet implemented]")
        print("   Falling back to software simulation...")

    print("   Running software simulation:")
    simulate_matmul_software(dram, commands, config)

    # -------------------------------------------------------------------------
    # 6. Verification
    # -------------------------------------------------------------------------
    print("\n6. Verification")
    print("-" * 40)

    # Read result from DRAM
    C_result = dram.load_matrix("C", C_DRAM_ADDR, (M, N), np.int32)

    print(f"   Result C ({M}x{N}):")
    print(C_result)

    # Compare with expected
    if np.array_equal(C_result, C_expected):
        print("\n   PASS: Result matches expected!")
    else:
        print("\n   FAIL: Result does not match expected!")
        print(f"   Expected:\n{C_expected}")
        print(f"   Got:\n{C_result}")
        print(f"   Difference:\n{C_result - C_expected}")
        return False

    print("\n" + "=" * 70)
    print("Demo completed successfully!")
    print("=" * 70)
    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Simple Matrix Multiply Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--size",
        type=int,
        default=4,
        help="Matrix size N (creates NxN matrices, default: 4)",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Run RTL simulation (requires Amaranth)",
    )

    args = parser.parse_args()

    success = run_demo(matrix_size=args.size, run_rtl=args.simulate)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
