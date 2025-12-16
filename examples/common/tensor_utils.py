"""
Tensor Utilities for Systars Examples.

This module provides utilities for preparing tensors for systolic array
execution, including:
- Packing/unpacking between NumPy arrays and hardware formats
- Tiling large matrices to fit systolic array dimensions
- Data layout transformations for different dataflows

The systolic array has fixed dimensions (e.g., 16x16). Larger matrices
must be tiled and processed in multiple passes.
"""

import numpy as np


def pack_matrix_int8(matrix: np.ndarray, buswidth: int = 128) -> list[int]:
    """
    Pack an int8 matrix into bus-width aligned words.

    The systolic array DMA transfers data in bus-width beats.
    This function packs matrix elements into these beat-sized words.

    Args:
        matrix: 2D numpy array of int8 values
        buswidth: Bus width in bits

    Returns:
        List of integers, each representing one bus beat

    Example:
        >>> A = np.array([[1, 2, 3, 4]], dtype=np.int8)
        >>> beats = pack_matrix_int8(A, buswidth=32)
        >>> hex(beats[0])
        '0x4030201'
    """
    bytes_per_beat = buswidth // 8
    flat = matrix.flatten().astype(np.int8)

    # Pad to bus width boundary
    pad_len = (bytes_per_beat - (len(flat) % bytes_per_beat)) % bytes_per_beat
    if pad_len > 0:
        flat = np.concatenate([flat, np.zeros(pad_len, dtype=np.int8)])

    beats = []
    for i in range(0, len(flat), bytes_per_beat):
        word = 0
        for j in range(bytes_per_beat):
            byte_val = int(flat[i + j]) & 0xFF
            word |= byte_val << (j * 8)
        beats.append(word)

    return beats


def unpack_matrix_int8(
    beats: list[int],
    shape: tuple[int, int],
    buswidth: int = 128,
) -> np.ndarray:
    """
    Unpack bus-width beats back into an int8 matrix.

    Args:
        beats: List of bus-width integers
        shape: Target matrix shape (rows, cols)
        buswidth: Bus width in bits

    Returns:
        2D numpy array of int8 values
    """
    bytes_per_beat = buswidth // 8
    total_elements = shape[0] * shape[1]

    flat = []
    for beat in beats:
        for j in range(bytes_per_beat):
            byte_val = (beat >> (j * 8)) & 0xFF
            # Sign extend
            if byte_val >= 128:
                byte_val -= 256
            flat.append(byte_val)

    return np.array(flat[:total_elements], dtype=np.int8).reshape(shape)


def pack_matrix_int32(matrix: np.ndarray, buswidth: int = 128) -> list[int]:
    """
    Pack an int32 matrix into bus-width aligned words.

    Used for accumulator results which are typically 32-bit.

    Args:
        matrix: 2D numpy array of int32 values
        buswidth: Bus width in bits

    Returns:
        List of integers, each representing one bus beat
    """
    elements_per_beat = buswidth // 32
    flat = matrix.flatten().astype(np.int32)

    # Pad to beat boundary
    pad_len = (elements_per_beat - (len(flat) % elements_per_beat)) % elements_per_beat
    if pad_len > 0:
        flat = np.concatenate([flat, np.zeros(pad_len, dtype=np.int32)])

    beats = []
    for i in range(0, len(flat), elements_per_beat):
        word = 0
        for j in range(elements_per_beat):
            val = int(flat[i + j])
            if val < 0:
                val = val + (1 << 32)
            word |= (val & 0xFFFFFFFF) << (j * 32)
        beats.append(word)

    return beats


def unpack_matrix_int32(
    beats: list[int],
    shape: tuple[int, int],
    buswidth: int = 128,
) -> np.ndarray:
    """
    Unpack bus-width beats back into an int32 matrix.

    Args:
        beats: List of bus-width integers
        shape: Target matrix shape (rows, cols)
        buswidth: Bus width in bits

    Returns:
        2D numpy array of int32 values
    """
    elements_per_beat = buswidth // 32
    total_elements = shape[0] * shape[1]

    flat = []
    for beat in beats:
        for j in range(elements_per_beat):
            val = (beat >> (j * 32)) & 0xFFFFFFFF
            # Sign extend
            if val >= (1 << 31):
                val -= 1 << 32
            flat.append(val)

    return np.array(flat[:total_elements], dtype=np.int32).reshape(shape)


def tile_matrix(
    matrix: np.ndarray,
    tile_rows: int,
    tile_cols: int,
) -> list[tuple[int, int, np.ndarray]]:
    """
    Tile a matrix into smaller chunks that fit the systolic array.

    For a matrix larger than the array dimensions, we must process it
    in tiles. This function breaks a matrix into tiles and returns
    their positions for reassembly.

    Args:
        matrix: Input 2D numpy array
        tile_rows: Maximum rows per tile (array height)
        tile_cols: Maximum cols per tile (array width)

    Returns:
        List of (row_offset, col_offset, tile_data) tuples

    Example:
        >>> A = np.arange(16).reshape(4, 4)
        >>> tiles = tile_matrix(A, tile_rows=2, tile_cols=2)
        >>> len(tiles)
        4
        >>> tiles[0]  # Top-left tile
        (0, 0, array([[0, 1], [4, 5]]))
    """
    rows, cols = matrix.shape
    tiles = []

    for r in range(0, rows, tile_rows):
        for c in range(0, cols, tile_cols):
            r_end = min(r + tile_rows, rows)
            c_end = min(c + tile_cols, cols)
            tile = matrix[r:r_end, c:c_end]

            # Pad tile to full size if at edge
            if tile.shape != (tile_rows, tile_cols):
                padded = np.zeros((tile_rows, tile_cols), dtype=matrix.dtype)
                padded[: tile.shape[0], : tile.shape[1]] = tile
                tile = padded

            tiles.append((r, c, tile))

    return tiles


def untile_matrix(
    tiles: list[tuple[int, int, np.ndarray]],
    shape: tuple[int, int],
) -> np.ndarray:
    """
    Reassemble tiles back into a full matrix.

    Args:
        tiles: List of (row_offset, col_offset, tile_data) tuples
        shape: Target matrix shape (rows, cols)

    Returns:
        Reassembled 2D numpy array
    """
    result = np.zeros(shape, dtype=tiles[0][2].dtype)

    for r_off, c_off, tile in tiles:
        r_end = min(r_off + tile.shape[0], shape[0])
        c_end = min(c_off + tile.shape[1], shape[1])
        actual_rows = r_end - r_off
        actual_cols = c_end - c_off
        result[r_off:r_end, c_off:c_end] = tile[:actual_rows, :actual_cols]

    return result


def compute_tiled_gemm_schedule(
    M: int,
    K: int,
    N: int,
    array_rows: int,
    array_cols: int,
) -> list[dict]:
    """
    Compute the tiling schedule for a GEMM operation C[M,N] = A[M,K] @ B[K,N].

    This generates the sequence of operations needed to compute a large
    matrix multiply using a smaller systolic array.

    Args:
        M: Rows in A and C
        K: Cols in A, rows in B
        N: Cols in B and C
        array_rows: Systolic array height
        array_cols: Systolic array width

    Returns:
        List of operation descriptors, each containing:
        - 'a_tile': (row_start, row_end, col_start, col_end) for A
        - 'b_tile': (row_start, row_end, col_start, col_end) for B
        - 'c_tile': (row_start, row_end, col_start, col_end) for C
        - 'accumulate': Whether to accumulate into existing C values
    """
    schedule = []

    # Tile over output dimensions first (M and N)
    for m_start in range(0, M, array_rows):
        m_end = min(m_start + array_rows, M)

        for n_start in range(0, N, array_cols):
            n_end = min(n_start + array_cols, N)

            # For each output tile, iterate over K dimension
            first_k = True
            for k_start in range(0, K, array_cols):  # K tiles based on array width
                k_end = min(k_start + array_cols, K)

                op = {
                    "a_tile": (m_start, m_end, k_start, k_end),
                    "b_tile": (k_start, k_end, n_start, n_end),
                    "c_tile": (m_start, m_end, n_start, n_end),
                    "accumulate": not first_k,
                }
                schedule.append(op)
                first_k = False

    return schedule


def print_gemm_schedule(schedule: list[dict]):
    """Pretty-print a GEMM tiling schedule."""
    print("\nGEMM Tiling Schedule:")
    print("-" * 70)
    for i, op in enumerate(schedule):
        acc_str = "accumulate" if op["accumulate"] else "overwrite"
        print(
            f"  Op {i}: A[{op['a_tile'][0]}:{op['a_tile'][1]}, "
            f"{op['a_tile'][2]}:{op['a_tile'][3]}] @ "
            f"B[{op['b_tile'][0]}:{op['b_tile'][1]}, "
            f"{op['b_tile'][2]}:{op['b_tile'][3]}] -> "
            f"C[{op['c_tile'][0]}:{op['c_tile'][1]}, "
            f"{op['c_tile'][2]}:{op['c_tile'][3]}] ({acc_str})"
        )
    print("-" * 70)
    print(f"Total operations: {len(schedule)}")
