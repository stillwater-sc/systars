"""
MasPar SIMD Array Processor Simulator.

This is the top-level simulator that orchestrates the ACU and PE array
to execute SIMD programs. It provides:
- Program loading and execution
- Cycle-accurate simulation
- Statistics collection
- Matrix operation helpers

Example usage:
    config = MasParConfig(array_rows=4, array_cols=4)
    sim = MasParSim(config)
    sim.load_gemm_data(A, B)
    cycles = sim.run_gemm(k_dim=4)
    C = sim.extract_result()
"""

from dataclasses import dataclass, field

import numpy as np

from .acu import ACU
from .config import MasParConfig
from .instruction import Instruction, Opcode
from .pe_array import PEArray


@dataclass
class MasParSim:
    """
    Top-level MasPar SIMD simulator.

    Coordinates the ACU (instruction dispatch) and PE array (execution)
    to simulate MasPar program execution.
    """

    config: MasParConfig

    # Components
    pe_array: PEArray = field(init=False)
    acu: ACU = field(init=False)

    # Simulation state
    cycle: int = 0
    done: bool = False

    # Statistics
    total_instructions: int = 0
    total_xnet_ops: int = 0
    total_alu_ops: int = 0
    total_memory_ops: int = 0

    # Register assignments for GEMM
    REG_A: int = 0  # A matrix element
    REG_B: int = 1  # B matrix element
    REG_C: int = 2  # C accumulator
    REG_TMP: int = 3  # Temporary for multiply result

    def __post_init__(self) -> None:
        """Initialize simulator components."""
        self.pe_array = PEArray(self.config)
        self.acu = ACU()

    def reset(self) -> None:
        """Reset simulator to initial state."""
        self.pe_array.reset()
        self.acu.reset()
        self.cycle = 0
        self.done = False
        self.total_instructions = 0
        self.total_xnet_ops = 0
        self.total_alu_ops = 0
        self.total_memory_ops = 0

    def load_program(self, program: list[Instruction]) -> None:
        """Load a program into the ACU."""
        self.acu.load_program(program)
        self.done = False

    def step(self) -> dict:
        """
        Execute one simulation cycle.

        Returns:
            Dictionary with cycle info:
            - cycle: Current cycle number
            - instruction: Instruction executed (or None)
            - done: Whether program is complete
        """
        # Fetch next instruction
        instr = self.acu.fetch()

        result = {
            "cycle": self.cycle,
            "instruction": instr,
            "done": False,
        }

        if instr is None:
            if self.acu.is_done():
                self.done = True
                result["done"] = True
            # else: stalled on multi-cycle instruction
        else:
            # Dispatch to PE array
            self.pe_array.broadcast_instruction(instr)
            self._update_statistics(instr)

        self.cycle += 1
        return result

    def run_to_completion(self, max_cycles: int = 100000) -> int:
        """
        Run program until completion.

        Args:
            max_cycles: Maximum cycles before forced stop

        Returns:
            Total cycles executed
        """
        while not self.done and self.cycle < max_cycles:
            self.step()
        return self.cycle

    def _update_statistics(self, instr: Instruction) -> None:
        """Update execution statistics based on instruction type."""
        self.total_instructions += 1

        if instr.is_xnet():
            self.total_xnet_ops += 1
        elif instr.is_alu() or instr.is_fpu():
            self.total_alu_ops += 1
        elif instr.is_memory():
            self.total_memory_ops += 1

    def get_statistics(self) -> dict:
        """
        Get execution statistics.

        Returns:
            Dictionary with execution statistics
        """
        return {
            "cycles": self.cycle,
            "total_instructions": self.total_instructions,
            "alu_ops": self.total_alu_ops,
            "xnet_ops": self.total_xnet_ops,
            "memory_ops": self.total_memory_ops,
            "active_pes": self.pe_array.get_active_count(),
            "total_pes": self.config.total_pes,
        }

    # ========== Matrix Multiplication (GEMM) Support ==========

    def load_gemm_data(self, A: np.ndarray, B: np.ndarray, use_cannon_skew: bool = True) -> None:
        """
        Load matrices for GEMM computation.

        For Cannon's algorithm, data is pre-skewed:
        - A[i,j] is loaded into PE[i, (j-i) mod cols]
        - B[i,j] is loaded into PE[(i-j) mod rows, j]

        Args:
            A: Left matrix (MxK)
            B: Right matrix (KxN)
            use_cannon_skew: Apply Cannon's algorithm pre-skew
        """
        rows = self.config.array_rows
        cols = self.config.array_cols

        # Initialize C accumulator to zero
        for r in range(rows):
            for c in range(cols):
                self.pe_array.pes[r][c].set_register(self.REG_C, 0)

        if use_cannon_skew:
            # Cannon's algorithm: pre-skew A and B
            # PE(i,j) gets A[i, (i+j) mod K] and B[(i+j) mod K, j]
            K = min(A.shape[1], B.shape[0], rows, cols)
            for r in range(rows):
                for c in range(cols):
                    # A is skewed left by row index
                    a_col = (r + c) % K
                    a_val = int(A[r, a_col]) if r < A.shape[0] and a_col < A.shape[1] else 0

                    # B is skewed up by column index
                    b_row = (r + c) % K
                    b_val = int(B[b_row, c]) if b_row < B.shape[0] and c < B.shape[1] else 0

                    self.pe_array.pes[r][c].set_register(self.REG_A, a_val)
                    self.pe_array.pes[r][c].set_register(self.REG_B, b_val)
        else:
            # Simple load without skew
            self.pe_array.load_matrix_block(A, self.REG_A)
            self.pe_array.load_matrix_block(B, self.REG_B)

    def create_gemm_program(self, k_dim: int) -> list[Instruction]:
        """
        Create GEMM program using Cannon's algorithm.

        The algorithm performs K iterations:
        1. C += A * B (local multiply-accumulate)
        2. Shift A west (XNET)
        3. Shift B north (XNET)

        Args:
            k_dim: K dimension (number of iterations)

        Returns:
            List of instructions for GEMM
        """
        program = []

        for _ in range(k_dim):
            # R_TMP = A * B
            program.append(
                Instruction(opcode=Opcode.IMUL, dst=self.REG_TMP, src1=self.REG_A, src2=self.REG_B)
            )

            # C += R_TMP
            program.append(
                Instruction(opcode=Opcode.IADD, dst=self.REG_C, src1=self.REG_C, src2=self.REG_TMP)
            )

            # Shift A west (each PE gets A from eastern neighbor)
            program.append(
                Instruction(
                    opcode=Opcode.XNET_E,  # Read from east to shift west
                    dst=self.REG_A,
                    src1=self.REG_A,
                )
            )

            # Shift B north (each PE gets B from southern neighbor)
            program.append(
                Instruction(
                    opcode=Opcode.XNET_S,  # Read from south to shift north
                    dst=self.REG_B,
                    src1=self.REG_B,
                )
            )

        return program

    def run_gemm(self, A: np.ndarray, B: np.ndarray, k_dim: int | None = None) -> int:
        """
        Execute complete GEMM operation.

        Loads data with Cannon pre-skew, generates program, and runs
        to completion.

        Args:
            A: Left matrix (MxK)
            B: Right matrix (KxN)
            k_dim: K dimension (default: min of array dims and matrix dims)

        Returns:
            Total cycles for execution
        """
        self.reset()

        # Determine K dimension
        if k_dim is None:
            k_dim = min(A.shape[1], B.shape[0], self.config.array_rows, self.config.array_cols)

        # Load data with Cannon's pre-skew
        self.load_gemm_data(A, B, use_cannon_skew=True)

        # Generate and load program
        program = self.create_gemm_program(k_dim)
        self.load_program(program)

        # Execute
        return self.run_to_completion()

    def extract_result(self, rows: int | None = None, cols: int | None = None) -> np.ndarray:
        """
        Extract GEMM result matrix from PE registers.

        Args:
            rows: Number of rows to extract
            cols: Number of columns to extract

        Returns:
            Result matrix C
        """
        return self.pe_array.extract_matrix_block(self.REG_C, rows, cols)

    # ========== 2D Convolution Support ==========

    # Register assignments for Conv2D
    # R0: Center pixel (input)
    # R1-R8: Neighbor pixels (N, S, E, W, NE, NW, SE, SW)
    # R10-R18: Kernel weights (9 values for 3x3)
    # R20: Output accumulator
    # R21: Temporary for multiply

    REG_CENTER: int = 0
    REG_N: int = 1
    REG_S: int = 2
    REG_E: int = 3
    REG_W: int = 4
    REG_NE: int = 5
    REG_NW: int = 6
    REG_SE: int = 7
    REG_SW: int = 8
    REG_K_CENTER: int = 10
    REG_K_N: int = 11
    REG_K_S: int = 12
    REG_K_E: int = 13
    REG_K_W: int = 14
    REG_K_NE: int = 15
    REG_K_NW: int = 16
    REG_K_SE: int = 17
    REG_K_SW: int = 18
    REG_CONV_OUT: int = 20
    REG_CONV_TMP: int = 21

    def load_conv2d_data(self, image: np.ndarray, kernel: np.ndarray) -> None:
        """
        Load image and kernel data for 2D convolution.

        Args:
            image: 2D input image (must fit in PE array)
            kernel: 3x3 convolution kernel
        """
        rows = self.config.array_rows
        cols = self.config.array_cols

        if kernel.shape != (3, 3):
            raise ValueError("Kernel must be 3x3")

        # Load image into center register (R0) for each PE
        for r in range(rows):
            for c in range(cols):
                # Zero-pad if outside image bounds
                val = int(image[r, c]) if r < image.shape[0] and c < image.shape[1] else 0
                self.pe_array.pes[r][c].set_register(self.REG_CENTER, val)
                # Initialize output to zero
                self.pe_array.pes[r][c].set_register(self.REG_CONV_OUT, 0)

        # Store kernel weights for use by create_conv2d_program
        # Kernel layout (standard convolution order):
        #   K[0,0] K[0,1] K[0,2]   ->  NW  N  NE
        #   K[1,0] K[1,1] K[1,2]   ->  W   C  E
        #   K[2,0] K[2,1] K[2,2]   ->  SW  S  SE
        self._kernel = kernel.copy()

    def create_conv2d_program(self) -> list[Instruction]:
        """
        Create 3x3 convolution program.

        The algorithm:
        1. Load kernel weights into registers (via LDI)
        2. Gather all 8 neighbors via XNET
        3. Compute 9 multiply-adds: output = sum(pixel[i] * kernel[i])

        Returns:
            List of instructions for conv2d
        """
        program = []
        kernel = self._kernel

        # Step 1: Load kernel weights into registers via LDI
        # Kernel mapping:
        #   [0,0]=NW  [0,1]=N   [0,2]=NE
        #   [1,0]=W   [1,1]=C   [1,2]=E
        #   [2,0]=SW  [2,1]=S   [2,2]=SE
        kernel_map = [
            (self.REG_K_NW, int(kernel[0, 0])),
            (self.REG_K_N, int(kernel[0, 1])),
            (self.REG_K_NE, int(kernel[0, 2])),
            (self.REG_K_W, int(kernel[1, 0])),
            (self.REG_K_CENTER, int(kernel[1, 1])),
            (self.REG_K_E, int(kernel[1, 2])),
            (self.REG_K_SW, int(kernel[2, 0])),
            (self.REG_K_S, int(kernel[2, 1])),
            (self.REG_K_SE, int(kernel[2, 2])),
        ]

        for reg, val in kernel_map:
            program.append(Instruction(opcode=Opcode.LDI, dst=reg, immediate=val))

        # Step 2: Gather neighbors via XNET
        # Each XNET instruction reads from the specified neighbor
        xnet_gather = [
            (Opcode.XNET_N, self.REG_N),  # Get north neighbor's value
            (Opcode.XNET_S, self.REG_S),  # Get south neighbor's value
            (Opcode.XNET_E, self.REG_E),  # Get east neighbor's value
            (Opcode.XNET_W, self.REG_W),  # Get west neighbor's value
            (Opcode.XNET_NE, self.REG_NE),  # Get northeast neighbor's value
            (Opcode.XNET_NW, self.REG_NW),  # Get northwest neighbor's value
            (Opcode.XNET_SE, self.REG_SE),  # Get southeast neighbor's value
            (Opcode.XNET_SW, self.REG_SW),  # Get southwest neighbor's value
        ]

        for opcode, dst_reg in xnet_gather:
            program.append(
                Instruction(
                    opcode=opcode,
                    dst=dst_reg,
                    src1=self.REG_CENTER,  # Read neighbor's center value
                )
            )

        # Step 3: Compute convolution (9 multiply-adds)
        # output = sum(pixel[i] * kernel[i]) for all 9 positions

        # Initialize output to zero (already done in load, but be explicit)
        program.append(Instruction(opcode=Opcode.LDI, dst=self.REG_CONV_OUT, immediate=0))

        # Multiply-accumulate for each position
        mac_pairs = [
            (self.REG_CENTER, self.REG_K_CENTER),  # Center
            (self.REG_N, self.REG_K_N),  # North
            (self.REG_S, self.REG_K_S),  # South
            (self.REG_E, self.REG_K_E),  # East
            (self.REG_W, self.REG_K_W),  # West
            (self.REG_NE, self.REG_K_NE),  # Northeast
            (self.REG_NW, self.REG_K_NW),  # Northwest
            (self.REG_SE, self.REG_K_SE),  # Southeast
            (self.REG_SW, self.REG_K_SW),  # Southwest
        ]

        for pixel_reg, kernel_reg in mac_pairs:
            # tmp = pixel * kernel_weight
            program.append(
                Instruction(
                    opcode=Opcode.IMUL, dst=self.REG_CONV_TMP, src1=pixel_reg, src2=kernel_reg
                )
            )
            # output += tmp
            program.append(
                Instruction(
                    opcode=Opcode.IADD,
                    dst=self.REG_CONV_OUT,
                    src1=self.REG_CONV_OUT,
                    src2=self.REG_CONV_TMP,
                )
            )

        return program

    def run_conv2d(self, image: np.ndarray, kernel: np.ndarray) -> int:
        """
        Execute complete 2D convolution.

        Args:
            image: 2D input image
            kernel: 3x3 convolution kernel

        Returns:
            Total cycles for execution
        """
        self.reset()

        # Load data
        self.load_conv2d_data(image, kernel)

        # Generate and load program
        program = self.create_conv2d_program()
        self.load_program(program)

        # Execute
        return self.run_to_completion()

    def extract_conv2d_result(self, rows: int | None = None, cols: int | None = None) -> np.ndarray:
        """
        Extract convolution result from PE registers.

        Args:
            rows: Number of rows to extract
            cols: Number of columns to extract

        Returns:
            Convolution output
        """
        return self.pe_array.extract_matrix_block(self.REG_CONV_OUT, rows, cols)

    def __repr__(self) -> str:
        """String representation."""
        status = "done" if self.done else "running"
        return f"MasParSim({self.config.array_rows}x{self.config.array_cols}, cycle={self.cycle}, {status})"
