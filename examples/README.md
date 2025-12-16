# Systars Examples

This directory contains demonstration applications for the systolic array accelerator. Each example shows how to use the hardware for specific workloads.

## Getting Started

```bash
# Ensure you're in the project virtual environment
source .venv/bin/activate

# Run the simple matrix multiply demo
python examples/gemm/01_simple_matmul.py

# Run with a larger matrix size
python examples/gemm/01_simple_matmul.py --size 8
```

## Directory Structure

```
examples/
├── common/                      # Shared utilities
│   ├── dram_model.py           # Simulated DRAM memory model
│   └── tensor_utils.py         # Matrix packing/tiling utilities
│
├── gemm/                        # Matrix multiply demonstrations
│   ├── 01_simple_matmul.py     # Basic C = A × B
│   ├── 02_animated_wavefront.py # Terminal wavefront animation
│   └── 03_wavefront_gif.py     # Generate shareable GIF
│
├── nn/                          # Neural network operators (planned)
│   ├── 01_fc_layer.py          # Fully connected layer
│   ├── 02_conv2d.py            # 2D convolution via im2col
│   └── 03_conv3d.py            # 3D convolution
│
├── dsp/                         # Digital signal processing (planned)
│   └── 01_fir_filter.py        # FIR filter implementation
│
└── scientific/                  # Scientific computing (planned)
    └── 01_iterative_solver.py  # Linear system solver
```

## Example Workflow

Each demo follows a consistent structure:

1. **Problem Setup** - Define input tensors using NumPy
2. **Hardware Configuration** - Set array dimensions, precision, bus widths
3. **Memory Layout** - Plan DRAM and local memory addresses
4. **Command Generation** - Create the sequence of accelerator commands
5. **Execution** - Run via software model or RTL simulation
6. **Verification** - Compare results against NumPy reference

## Common Utilities

### SimulatedDRAM

Models external memory with bus-width aligned transfers:

```python
from examples.common import SimulatedDRAM

dram = SimulatedDRAM(size_bytes=1024*1024, buswidth=128)

# Store a matrix
A = np.array([[1, 2], [3, 4]], dtype=np.int8)
dram.store_matrix("A", base_addr=0x1000, matrix=A)

# Load a matrix
A_back = dram.load_matrix("A", base_addr=0x1000, shape=(2, 2), dtype=np.int8)
```

### Tensor Utilities

Pack matrices for DMA transfers and tile for large workloads:

```python
from examples.common import tile_matrix, pack_matrix_int8

# Tile a large matrix for a 4x4 array
tiles = tile_matrix(large_matrix, tile_rows=4, tile_cols=4)

# Pack for bus transfer
beats = pack_matrix_int8(matrix, buswidth=128)
```

## Wavefront Visualization

The systolic array processes data in a wavefront pattern. Run the animated demo to see this:

```bash
# Animated visualization (press Enter to start, Ctrl+C to skip)
python examples/gemm/02_animated_wavefront.py

# Step-by-step mode (press Enter for each cycle)
python examples/gemm/02_animated_wavefront.py --step

# Larger array, faster animation
python examples/gemm/02_animated_wavefront.py --size 6 --delay 200
```

The visualization shows:

- **Skewed inputs**: Row i of A is delayed by i cycles, Column j of B by j cycles
- **Wavefront propagation**: Diagonal bands of activity moving through the array
- **PE states**: Current a, b values and ongoing `a×b` computations
- **Accumulator buildup**: C values accumulating as partial products arrive

```
Wavefront Pattern (cycle when each PE activates):
       Col0   Col1   Col2   Col3
Row0 [   0     1     2     3  ]
Row1 [   1     2     3     4  ]
Row2 [   2     3     4     5  ]
Row3 [   3     4     5     6  ]
```

### Generating Shareable GIFs

For sharing on Slack, docs, or presentations:

```bash
# Install dependencies (if not already installed)
pip install matplotlib pillow

# Generate a GIF
python examples/gemm/03_wavefront_gif.py --output wavefront.gif

# Customize size, speed, and resolution
python examples/gemm/03_wavefront_gif.py --size 4 --fps 2 --dpi 150 --output systolic.gif

# Preview in a window instead of saving
python examples/gemm/03_wavefront_gif.py --show
```

The GIF shows three panels:

1. **Input matrices** with highlighted elements being fed
2. **PE array** with color-coded states (green=computing, blue=has data)
3. **Accumulator** with values building up

## Dataflow Modes

The systolic array supports two dataflow modes:

### Output-Stationary (OS)

- Accumulator stays in place, weights flow through
- Best for: Large batch sizes, high output reuse
- Use case: Fully connected layers with many samples

### Weight-Stationary (WS)

- Weights stay in place, partial sums flow through
- Best for: Weight reuse across spatial dimensions
- Use case: Convolutions with large filter reuse

## Application Mapping Guide

### Fully Connected Layer: Y = XW + b

```
X: [batch, in_features]   -> Load as input activations
W: [in_features, out_features] -> Load as weights
b: [out_features]         -> Preload to accumulator
Y: [batch, out_features]  -> Store from accumulator
```

### Conv2D via im2col

Transform convolution to matrix multiply:

```
Input: [N, C_in, H, W]
Kernel: [C_out, C_in, Kh, Kw]

im2col(Input) -> [N * H_out * W_out, C_in * Kh * Kw]
reshape(Kernel) -> [C_in * Kh * Kw, C_out]
Output = im2col @ Kernel -> [N * H_out * W_out, C_out]
col2im(Output) -> [N, C_out, H_out, W_out]
```

### FIR Filter: y[n] = sum(h[k] * x[n-k])

Map to matrix multiply using Toeplitz structure:

```
X_toeplitz: [output_samples, filter_length]
H: [filter_length, 1]
Y = X_toeplitz @ H
```

## Future: Python API with Annotations

The long-term vision is to enable automatic compilation from NumPy:

```python
@systars.accelerate(array_size=(16, 16), dataflow="OS")
def matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return A @ B

# Automatically generates commands and runs on accelerator
C = matmul(A, B)
```

This will be implemented through:

1. Tensor shape analysis for tiling decisions
2. Memory layout optimization
3. Command sequence generation
4. DMA scheduling for data movement
5. Result retrieval and unpacking
