# Session Log: Examples and Wavefront Animation

**Date:** 2025-12-16
**Focus:** User-facing examples and systolic array visualization

## Summary

This session created a comprehensive examples framework for demonstrating the systolic array accelerator, including animated wavefront visualizations for understanding and sharing the data flow patterns.

## Work Completed

### 1. Examples Directory Structure

Created `examples/` directory with organized structure:

```
examples/
├── README.md                    # Overview and getting started
├── common/                      # Shared utilities
│   ├── __init__.py
│   ├── dram_model.py           # SimulatedDRAM class
│   └── tensor_utils.py         # Matrix packing/tiling
├── gemm/                        # Matrix multiply demos
│   ├── 01_simple_matmul.py     # Basic C = A × B
│   ├── 02_animated_wavefront.py # Terminal animation
│   └── 03_wavefront_gif.py     # GIF generator
├── nn/                          # Neural network (planned)
├── dsp/                         # DSP (planned)
└── scientific/                  # Scientific computing (planned)
```

### 2. Common Utilities

**SimulatedDRAM** (`examples/common/dram_model.py`):

- Models external memory with bus-width aligned transfers
- `store_matrix()` / `load_matrix()` for numpy arrays
- `read_beat()` / `write_beat()` for DMA-level access
- Memory region tracking and hex dump for debugging

**Tensor Utilities** (`examples/common/tensor_utils.py`):

- `pack_matrix_int8/int32` - Pack matrices for DMA transfers
- `unpack_matrix_int8/int32` - Extract results from bus beats
- `tile_matrix/untile_matrix` - Handle matrices larger than array
- `compute_tiled_gemm_schedule` - Generate tiling schedule for large GEMM

### 3. Simple Matrix Multiply Demo

**`examples/gemm/01_simple_matmul.py`**:

- Complete workflow: setup → config → load → execute → store → verify
- Command sequence generation (CONFIG, LOAD, COMPUTE, STORE)
- Software simulation of accelerator behavior
- Verification against NumPy reference

Usage:

```bash
python examples/gemm/01_simple_matmul.py --size 4
```

### 4. Animated Wavefront Visualization

**Terminal Animation** (`examples/gemm/02_animated_wavefront.py`):

- Cycle-accurate systolic array simulator
- Color-coded PE states (green=computing, blue=has data)
- Shows skewed input pattern and wavefront propagation
- Step-by-step mode for detailed study

Usage:

```bash
python examples/gemm/02_animated_wavefront.py --size 3 --step
```

**GIF Generator** (`examples/gemm/03_wavefront_gif.py`):

- Three-panel visualization (inputs, array, accumulator)
- Matplotlib-based animation export
- Shareable on Slack, docs, presentations

Usage:

```bash
pip install matplotlib pillow
python examples/gemm/03_wavefront_gif.py --output wavefront.gif
```

### 5. Wavefront Documentation

Created `docs/wavefront-animation.md` (353 lines) covering:

- Why wavefront data flow (O(N²) vs O(N³) memory access)
- The skewing pattern (row i delayed by i cycles)
- Cycle-by-cycle breakdown with diagrams
- Visualization tool usage guide
- Color coding and output interpretation
- GIF generation settings and file sizes

### 6. End-to-End Integration Tests

**`tests/unit/test_gemm_e2e.py`**:

- `SimulatedAXIDRAM` with int8/int32 matrix support
- Inline AXI handling within testbenches
- Tests: load_matrix, store_matrix, execute_config, full_gemm_sequence
- All 10 tests passing

**`tests/unit/test_gemm_demo.py`**:

- SystolicTop instantiation and port verification
- Command dispatch tests (load, store, execute)
- Verilog generation tests
- All 13 tests passing

### 7. Top-Level Integration

**`src/systars/top.py`** - SystolicTop module:

- Wires all components: SystolicArray, Scratchpad, Accumulator, DMA engines, Controllers
- Command dispatch based on cmd_type (LOAD/STORE/EXEC)
- AXI interface to StreamReader/StreamWriter
- Status signals: busy, completed, completed_type

## Test Results

```
201 passed in 14.15s
```

All unit tests passing including:

- 10 E2E GEMM tests
- 13 GEMM demo tests
- Existing Phase 1-3 tests

## Files Created/Modified

### New Files

- `examples/README.md`
- `examples/common/__init__.py`
- `examples/common/dram_model.py`
- `examples/common/tensor_utils.py`
- `examples/gemm/__init__.py`
- `examples/gemm/01_simple_matmul.py`
- `examples/gemm/02_animated_wavefront.py`
- `examples/gemm/03_wavefront_gif.py`
- `docs/wavefront-animation.md`
- `src/systars/top.py`
- `tests/unit/test_gemm_e2e.py`
- `tests/unit/test_gemm_demo.py`

### Modified Files

- `CHANGELOG.md` - Added examples and visualization section
- `src/systars/__init__.py` - Added SystolicTop export

## Key Insights

### Wavefront Pattern

The systolic array processes data in diagonal wavefronts:

- PE[i,j] first computes at cycle i+j
- Skewing ensures A[i,k] meets B[k,j] at the right time
- Total cycles for N×N: 3N-2 (fill + compute + drain)

### Visualization Value

Terminal animation helps understand:

- How data flows through the array
- Why skewing is necessary
- How partial products accumulate

GIF export enables:

- Sharing on Slack/Teams
- Embedding in documentation
- Use in presentations

## Next Steps

Planned examples to add:

1. `examples/nn/01_fc_layer.py` - Fully connected layer with batching
2. `examples/nn/02_conv2d.py` - Convolution via im2col transformation
3. `examples/dsp/01_fir_filter.py` - FIR filter as Toeplitz matmul

Future vision: Python decorators for automatic compilation:

```python
@systars.accelerate(array_size=(16, 16), dataflow="OS")
def matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return A @ B
```

## Notes

- Large GIF files (>1MB) blocked by pre-commit hook
- Recommend adding `*.gif` to `.gitignore`
- Users can generate GIFs locally with the tool
- matplotlib/pillow needed for GIF generation (not in base requirements)
