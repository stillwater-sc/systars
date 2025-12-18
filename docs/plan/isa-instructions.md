# ISA Instruction Implementation Plan

This document details the implementation plan for the `Matmul` and `Conv2d` ISA instructions, the final Phase 6 components needed to complete the SYSTARS RTL.

---

## Overview

These are high-level ISA instructions that automatically handle tiling and scheduling for operations that exceed the systolic array dimensions. The hardware:

- **Selects the optimal dataflow** (output-stationary, A-stationary, B-stationary) based on tensor shapes
- **Tiles the computation** to fit the array dimensions
- **Manages double buffering** to hide memory latency
- **Generates command sequences** internally (LOAD, PRELOAD, COMPUTE, STORE)

The user simply issues `Matmul` or `Conv2d` with dimensions and addresses.

### Current State

| Component | Status |
|-----------|--------|
| ExecuteController | Done - handles CONFIG, PRELOAD, COMPUTE |
| LoadController | Done - handles DRAM → Scratchpad |
| StoreController | Done - handles Accumulator → DRAM |
| DMA Engines | Done - StreamReader, StreamWriter, Descriptors |
| ISA Instructions | **Not Started** |

### Target Operations

| Instruction | Operation | Description |
|-------------|-----------|-------------|
| `Matmul` | `C = A @ B + D` | Tiled matrix multiply with optional bias |
| `Conv2d` | `Y = conv(X, F) + B` | 2D convolution with optional bias |

---

## Part 1: Matmul Instruction

### 1.1 Algorithm: Tiled Matrix Multiply

For matrices larger than the systolic array dimension (DIM × DIM), we tile the computation:

```
C[M,N] = A[M,K] @ B[K,N] + D[M,N]

For each output tile C[i*DIM : (i+1)*DIM, j*DIM : (j+1)*DIM]:
    Initialize accumulator with D tile (or zeros)
    For each K tile:
        Load A[i*DIM:(i+1)*DIM, k*DIM:(k+1)*DIM] → scratchpad bank 0
        Load B[k*DIM:(k+1)*DIM, j*DIM:(j+1)*DIM] → scratchpad bank 1
        Compute: accumulator += A_tile @ B_tile
    Store accumulator → C[i*DIM:(i+1)*DIM, j*DIM:(j+1)*DIM]
```

### 1.2 Loop Order Options

The loop order affects memory access patterns and reuse:

| Loop Order | A Reuse | B Reuse | Best For |
|------------|---------|---------|----------|
| IJK | Low | High | B fits in cache, streaming A |
| IKJ | High | Low | A fits in cache, streaming B |
| JIK | Low | High | Column-major C |
| JKI | High | Low | Row-major C |
| KIJ | Medium | Medium | Balanced |
| KJI | Medium | Medium | Balanced |

**Recommendation**: Start with **KIJ** (K outer, I middle, J inner) for balanced behavior, then add loop order configuration.

### 1.3 Double Buffering

To hide memory latency, use double buffering:

```
Scratchpad Layout:
  Bank 0: A tile (current)
  Bank 1: B tile (current)
  Bank 2: A tile (next)  -- prefetch
  Bank 3: B tile (next)  -- prefetch

Pipeline:
  Cycle N:   Compute(A0, B0),  Load(A1, B1)
  Cycle N+1: Compute(A1, B1),  Load(A0, B0)
  ...
```

### 1.4 Interface Definition

```python
# src/systars/isa/matmul.py

class Matmul(Component):
    """
    Matmul ISA instruction: C = A @ B + D

    The hardware automatically:
    - Selects optimal dataflow (OS/AS/BS) based on M, N, K dimensions
    - Tiles the computation to fit array dimensions
    - Double-buffers to hide memory latency
    - Generates internal command sequences

    Configuration (via CONFIG commands):
        M, N, K:        Matrix dimensions
        A_addr:         DRAM base address for A
        B_addr:         DRAM base address for B
        C_addr:         DRAM base address for C
        D_addr:         DRAM base address for D (bias), or 0 for zeros
        A_stride:       Row stride for A in bytes
        B_stride:       Row stride for B in bytes
        C_stride:       Row stride for C in bytes
        D_stride:       Row stride for D in bytes
        accumulate:     Add to existing C (vs overwrite)
        activation:     NONE, RELU, RELU6

    Internal Commands Generated:
        LOAD:    DRAM → Scratchpad (for A, B tiles)
        LOAD:    DRAM → Accumulator (for D bias tiles)
        PRELOAD: Initialize PE registers from accumulator
        COMPUTE: Execute systolic matmul
        STORE:   Accumulator → DRAM (for C tiles)
    """

    def __init__(self, config: SystolicConfig):
        self.config = config

        ports = {
            # Configuration interface (from CPU/command decoder)
            "cfg_valid": In(1),
            "cfg_ready": Out(1),
            "cfg_cmd": In(8),       # CONFIG_DIMS, CONFIG_ADDRS, CONFIG_STRIDES, START
            "cfg_data": In(64),     # Configuration data

            # Dimension parameters (set via cfg interface)
            "param_M": Out(32),     # Rows of A and C
            "param_N": Out(32),     # Cols of B and C
            "param_K": Out(32),     # Cols of A, Rows of B

            # Command output (to reservation station)
            "cmd_valid": Out(1),
            "cmd_ready": In(1),
            "cmd_opcode": Out(8),
            "cmd_rs1": Out(64),
            "cmd_rs2": Out(64),
            "cmd_rd": Out(64),

            # Status
            "busy": Out(1),
            "done": Out(1),
            "error": Out(1),
            "error_code": Out(8),

            # Progress (for debugging/monitoring)
            "progress_i": Out(16),
            "progress_j": Out(16),
            "progress_k": Out(16),
        }

        super().__init__(ports)
```

### 1.5 State Machine

```
┌──────────────────────────────────────────────────────────────┐
│                       Matmul FSM                             │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  IDLE ──[start]──→ INIT                                      │
│    │                 │                                       │
│    │            ┌────┴────┐                                  │
│    │            ▼         │                                  │
│    │         LOAD_D ──────┤  (optional: load bias tile)      │
│    │            │         │                                  │
│    │            ▼         │                                  │
│    │    ┌→ LOAD_A ────────┤                                  │
│    │    │      │          │                                  │
│    │    │      ▼          │                                  │
│    │    │   LOAD_B ───────┤                                  │
│    │    │      │          │                                  │
│    │    │      ▼          │                                  │
│    │    │   PRELOAD ──────┤  (first K iteration only)        │
│    │    │      │          │                                  │
│    │    │      ▼          │                                  │
│    │    │   COMPUTE ──────┤                                  │
│    │    │      │          │                                  │
│    │    │      ▼          │                                  │
│    │    │   NEXT_K        │                                  │
│    │    │      │          │                                  │
│    │    │      ├──[k<K]───┘                                  │
│    │    │      │                                             │
│    │    │      ▼                                             │
│    │    │   STORE_C                                          │
│    │    │      │                                             │
│    │    │      ▼                                             │
│    │    │   NEXT_IJ                                          │
│    │    │      │                                             │
│    │    └──────┴──[more tiles]                               │
│    │            │                                            │
│    │            ▼                                            │
│    └────────  DONE                                           │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### 1.6 Configuration Commands

```python
class MatmulCmd(IntEnum):
    """Configuration commands for Matmul instruction."""

    # Matrix dimensions
    CONFIG_DIMS = 0x01      # data = (K << 32) | (N << 16) | M

    # Base addresses
    CONFIG_A_ADDR = 0x10    # data = A base address (64-bit)
    CONFIG_B_ADDR = 0x11    # data = B base address
    CONFIG_C_ADDR = 0x12    # data = C base address
    CONFIG_D_ADDR = 0x13    # data = D base address (0 = no bias)

    # Strides (in bytes)
    CONFIG_A_STRIDE = 0x20  # data = A row stride
    CONFIG_B_STRIDE = 0x21  # data = B row stride
    CONFIG_C_STRIDE = 0x22  # data = C row stride
    CONFIG_D_STRIDE = 0x23  # data = D row stride

    # Options
    CONFIG_OPTIONS = 0x30   # data = activation | accumulate | transpose_a | transpose_b

    # Control
    START = 0xF0            # Start execution
    ABORT = 0xFF            # Abort current operation
```

### 1.7 Address Calculation

For tile (i, j, k):

```python
# A tile: rows [i*DIM : (i+1)*DIM], cols [k*DIM : (k+1)*DIM]
a_tile_addr = A_addr + (i * DIM * A_stride) + (k * DIM * input_bytes)

# B tile: rows [k*DIM : (k+1)*DIM], cols [j*DIM : (j+1)*DIM]
b_tile_addr = B_addr + (k * DIM * B_stride) + (j * DIM * input_bytes)

# C tile: rows [i*DIM : (i+1)*DIM], cols [j*DIM : (j+1)*DIM]
c_tile_addr = C_addr + (i * DIM * C_stride) + (j * DIM * acc_bytes)

# D tile (same as C)
d_tile_addr = D_addr + (i * DIM * D_stride) + (j * DIM * acc_bytes)
```

### 1.8 Implementation Steps

1. **Define interface** (ports, signals)
2. **Implement configuration registers** (latch cfg_data on cfg_valid)
3. **Implement tile counters** (i, j, k with proper bounds)
4. **Implement address generators** (compute tile addresses)
5. **Implement command generation** (emit LOAD, PRELOAD, COMPUTE, STORE)
6. **Implement state machine** (sequence states correctly)
7. **Add double buffering** (toggle bank selection)
8. **Add error handling** (bounds checking, invalid config)

### 1.9 Test Cases

| Test | Description |
|------|-------------|
| `test_small_matmul` | 16x16 @ 16x16 = single tile |
| `test_tiled_matmul` | 32x32 @ 32x32 = 2x2 tiles |
| `test_rectangular` | 16x32 @ 32x48 = mixed tile counts |
| `test_with_bias` | Verify D is loaded and preloaded |
| `test_accumulate` | Verify accumulate mode adds to existing C |
| `test_large_k` | 16x256 @ 256x16 = many K iterations |

---

## Part 2: Conv2d Instruction

### 2.1 Algorithm: im2col + Matmul

Convolution is mapped to matrix multiply using im2col transformation:

```
Input:  X[N, C_in, H, W]      - Batch of images
Filter: F[C_out, C_in, Kh, Kw] - Convolution kernels
Output: Y[N, C_out, H', W']    - Output feature maps

im2col transforms input patches into columns:
  A[H'*W', C_in*Kh*Kw] = im2col(X)
  B[C_in*Kh*Kw, C_out] = reshape(F)
  Y = (A @ B).reshape(N, C_out, H', W')
```

### 2.2 Convolution Parameters

```python
class ConvParams:
    """Convolution parameters."""

    # Input dimensions
    batch: int          # N
    in_channels: int    # C_in
    in_height: int      # H
    in_width: int       # W

    # Kernel dimensions
    out_channels: int   # C_out
    kernel_h: int       # Kh
    kernel_w: int       # Kw

    # Convolution parameters
    stride_h: int = 1
    stride_w: int = 1
    pad_h: int = 0
    pad_w: int = 0
    dilation_h: int = 1
    dilation_w: int = 1

    # Output dimensions (computed)
    @property
    def out_height(self) -> int:
        return (self.in_height + 2*self.pad_h - self.dilation_h*(self.kernel_h-1) - 1) // self.stride_h + 1

    @property
    def out_width(self) -> int:
        return (self.in_width + 2*self.pad_w - self.dilation_w*(self.kernel_w-1) - 1) // self.stride_w + 1
```

### 2.3 Convolution Strategies

| Strategy | Use Case | Memory | Compute |
|----------|----------|--------|---------|
| **im2col + GEMM** | Standard conv | High (expanded patches) | Efficient |
| **Direct conv** | Small kernels | Low | May be slower |
| **Winograd** | 3x3 kernels | Medium | Fast for specific sizes |
| **FFT** | Large kernels | High | Fast for large K |

**Recommendation**: Start with **im2col + Matmul** as it maps directly to the Matmul instruction.

### 2.4 Interface Definition

```python
# src/systars/isa/conv2d.py

class Conv2d(Component):
    """
    Conv2d ISA instruction: Y = conv(X, F) + B

    The hardware automatically:
    - Performs im2col transformation on input patches
    - Maps convolution to Matmul operations
    - Handles padding, stride, and dilation
    - Supports depthwise and transposed modes

    Supports:
    - Standard 2D convolution
    - Depthwise separable convolution
    - Transposed convolution (deconv)
    - Pooling during output (max, avg)

    Strategy: im2col + Matmul
    - Transform input patches to matrix A
    - Reshape filters to matrix B
    - Compute Y = A @ B
    - Reshape output to tensor
    """

    def __init__(self, config: SystolicConfig):
        self.config = config

        ports = {
            # Configuration interface
            "cfg_valid": In(1),
            "cfg_ready": Out(1),
            "cfg_cmd": In(8),
            "cfg_data": In(64),

            # Command output
            "cmd_valid": Out(1),
            "cmd_ready": In(1),
            "cmd_opcode": Out(8),
            "cmd_rs1": Out(64),
            "cmd_rs2": Out(64),
            "cmd_rd": Out(64),

            # Status
            "busy": Out(1),
            "done": Out(1),
            "error": Out(1),

            # Convolution-specific outputs
            "im2col_addr": Out(64),   # Current im2col patch address
            "filter_addr": Out(64),   # Current filter address
        }

        super().__init__(ports)
```

### 2.5 State Machine

```
┌──────────────────────────────────────────────────────────────┐
│                       Conv2d FSM                             │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  IDLE ──[start]──→ INIT                                      │
│                       │                                      │
│                       ▼                                      │
│                 ┌→ LOAD_FILTERS  (once per output channel    │
│                 │       │         group, can be reused)      │
│                 │       ▼                                    │
│                 │  OUTER_LOOP    (batch, output tile)        │
│                 │       │                                    │
│                 │       ▼                                    │
│                 │  ┌→ IM2COL     (transform input patch)     │
│                 │  │      │                                  │
│                 │  │      ▼                                  │
│                 │  │   COMPUTE   (matmul: patch @ filters)   │
│                 │  │      │                                  │
│                 │  │      ▼                                  │
│                 │  │   NEXT_PATCH                            │
│                 │  │      │                                  │
│                 │  └──────┴──[more patches in tile]          │
│                 │         │                                  │
│                 │         ▼                                  │
│                 │      STORE                                 │
│                 │         │                                  │
│                 │         ▼                                  │
│                 │      NEXT_TILE                             │
│                 │         │                                  │
│                 └─────────┴──[more tiles]                    │
│                           │                                  │
│                           ▼                                  │
│                         DONE                                 │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### 2.6 im2col Address Calculation

For output position (oh, ow) with kernel position (kh, kw):

```python
def im2col_input_addr(
    input_base: int,
    n: int,           # batch index
    c: int,           # channel index
    oh: int,          # output height index
    ow: int,          # output width index
    kh: int,          # kernel height index
    kw: int,          # kernel width index
    params: ConvParams,
    input_stride: int,
) -> int:
    """Calculate input address for im2col transformation."""

    # Calculate input position
    ih = oh * params.stride_h - params.pad_h + kh * params.dilation_h
    iw = ow * params.stride_w - params.pad_w + kw * params.dilation_w

    # Check bounds (return 0 address for padding)
    if ih < 0 or ih >= params.in_height or iw < 0 or iw >= params.in_width:
        return 0  # Will be zero-padded

    # Calculate linear address
    # Assuming NCHW layout
    addr = input_base
    addr += n * params.in_channels * params.in_height * params.in_width
    addr += c * params.in_height * params.in_width
    addr += ih * params.in_width
    addr += iw
    addr *= input_stride  # Scale by element size

    return addr
```

### 2.7 Configuration Commands

```python
class Conv2dCmd(IntEnum):
    """Configuration commands for Conv2d instruction."""

    # Input dimensions
    CONFIG_INPUT_DIMS = 0x01    # data = (W << 48) | (H << 32) | (C << 16) | N
    CONFIG_INPUT_ADDR = 0x02    # data = input base address

    # Kernel dimensions
    CONFIG_KERNEL_DIMS = 0x10   # data = (Kw << 48) | (Kh << 32) | C_out
    CONFIG_KERNEL_ADDR = 0x11   # data = filter base address

    # Output
    CONFIG_OUTPUT_ADDR = 0x20   # data = output base address
    CONFIG_BIAS_ADDR = 0x21     # data = bias address (0 = no bias)

    # Convolution parameters
    CONFIG_STRIDE = 0x30        # data = (stride_w << 16) | stride_h
    CONFIG_PAD = 0x31           # data = (pad_w << 16) | pad_h
    CONFIG_DILATION = 0x32      # data = (dilation_w << 16) | dilation_h

    # Options
    CONFIG_OPTIONS = 0x40       # data = flags (depthwise, transposed, pool mode, activation)

    # Control
    START = 0xF0
    ABORT = 0xFF
```

### 2.8 Depthwise Convolution

For depthwise separable convolution (groups = in_channels):

```
Each input channel has its own filter (no cross-channel mixing)
Output: Y[N, C, H', W'] where each output channel depends only on
        the corresponding input channel.

Modified loop:
  For each channel c:
      A = im2col(X[:, c, :, :])  # Single channel patches
      B = F[c, :, :]             # Single filter
      Y[:, c, :, :] = A @ B
```

### 2.9 Implementation Steps

1. **Start with im2col + Matmul wrapper** (reuse Matmul instruction)
2. **Add im2col address generator** (complex, handles padding/stride/dilation)
3. **Add filter reshape logic** (OIHW → [I*Kh*Kw, O])
4. **Implement output tile loop** (batch, height, width)
5. **Add depthwise mode** (modify loop to process channels independently)
6. **Add pooling option** (max/avg pooling in store controller)
7. **Add transposed conv** (swap input/output roles, adjust strides)

### 2.10 Test Cases

| Test | Description |
|------|-------------|
| `test_conv_3x3` | Basic 3x3 convolution, stride=1, pad=1 |
| `test_conv_5x5` | Larger kernel |
| `test_conv_stride2` | Stride=2 convolution |
| `test_conv_padding` | Various padding modes |
| `test_conv_dilation` | Dilated convolution |
| `test_depthwise` | Depthwise separable conv |
| `test_transposed` | Transposed convolution |
| `test_conv_relu` | Convolution with ReLU activation |
| `test_conv_maxpool` | Convolution with 2x2 max pooling |

---

## Part 3: Common Infrastructure

### 3.1 Shared Components

Both ISA instructions need:

1. **Address Generators**: Compute tile/patch addresses
2. **Bound Checkers**: Handle edge tiles with partial data
3. **Double Buffer Manager**: Toggle buffer selection
4. **Command Sequencer**: Emit commands in correct order
5. **Wait Logic**: Handle command backpressure

### 3.2 Command Queue Interface

```python
class CommandQueue(Component):
    """
    Interface to reservation station / command dispatcher.

    Handles backpressure and command ordering.
    """

    def __init__(self):
        super().__init__({
            # From ISA instruction unit
            "cmd_valid": In(1),
            "cmd_opcode": In(8),
            "cmd_data": In(128),

            # To ISA instruction unit
            "cmd_ready": Out(1),

            # To controllers
            "load_valid": Out(1),
            "load_ready": In(1),
            "load_cmd": Out(128),

            "exec_valid": Out(1),
            "exec_ready": In(1),
            "exec_cmd": Out(128),

            "store_valid": Out(1),
            "store_ready": In(1),
            "store_cmd": Out(128),
        })
```

### 3.3 Error Handling

```python
class IsaErrorCode(IntEnum):
    """Error codes for ISA instructions."""

    NONE = 0x00
    INVALID_DIMS = 0x01       # M, N, K out of range
    INVALID_ADDR = 0x02       # Address alignment error
    INVALID_STRIDE = 0x03     # Stride too small
    OVERFLOW = 0x04           # Address overflow
    TIMEOUT = 0x05            # Command not accepted
    CONFIG_INCOMPLETE = 0x06  # Started without full config
```

---

## Part 4: Implementation Timeline

### Phase 6a: Matmul Instruction (First Priority)

| Step | Task | Complexity |
|------|------|------------|
| 1 | Define Matmul interface and ports | Low |
| 2 | Implement configuration register bank | Low |
| 3 | Implement tile counter (i, j, k) | Low |
| 4 | Implement address generator | Medium |
| 5 | Implement basic state machine (no double buffer) | Medium |
| 6 | Write unit tests for single-tile matmul | Low |
| 7 | Add double buffering | Medium |
| 8 | Write integration tests for multi-tile matmul | Medium |
| 9 | Add error handling | Low |
| 10 | Documentation and examples | Low |

### Phase 6b: Conv2d Instruction (Second Priority)

| Step | Task | Complexity |
|------|------|------------|
| 1 | Define Conv2d interface | Low |
| 2 | Implement convolution parameter registers | Low |
| 3 | Implement im2col address generator | High |
| 4 | Implement basic conv FSM (im2col + Matmul) | Medium |
| 5 | Write unit tests for basic convolution | Medium |
| 6 | Add depthwise mode | Medium |
| 7 | Add pooling integration | Medium |
| 8 | Add transposed convolution | High |
| 9 | Write integration tests | Medium |
| 10 | Documentation and examples | Low |

---

## Part 5: File Structure

```
src/systars/isa/
├── __init__.py
├── matmul.py           # Matmul instruction FSM
├── conv2d.py           # Conv2d instruction FSM
├── scheduler.py        # Dataflow schedule selection
├── address_gen.py      # Shared address generation utilities
├── buffer_manager.py   # Double buffer management
└── commands.py         # ISA command definitions

tests/unit/
├── test_matmul.py
├── test_conv2d.py
├── test_scheduler.py
└── test_address_gen.py

tests/integration/
├── test_tiled_matmul.py    # End-to-end tiled matmul
└── test_conv_layer.py      # End-to-end convolution layer
```

---

## Part 6: Dependencies

```
Matmul depends on:
├── LoadController (emit LOAD commands)
├── ExecuteController (emit PRELOAD, COMPUTE commands)
├── StoreController (emit STORE commands)
├── Scratchpad (bank allocation)
└── Accumulator (double buffering)

Conv2d depends on:
├── Matmul (reuse for im2col matrix multiply)
├── LoadController (im2col transformation)
├── StoreController (pooling during store)
└── All Matmul dependencies
```

---

## Part 7: Success Criteria

| Criterion | Metric |
|-----------|--------|
| Functional correctness | All unit tests pass |
| Tiled operation | 64x64 matmul works correctly |
| Performance | No stalls between tiles (double buffering works) |
| Integration | Works with existing controllers |
| Convolution | Basic 3x3 conv produces correct output |
| Documentation | Examples demonstrate usage |

---

## Next Steps

1. **Start with Matmul** - simpler, provides foundation for Conv2d
2. **Test incrementally** - single tile → 2x2 tiles → larger
3. **Add Conv2d as im2col wrapper** - reuse Matmul core
4. **Verify with examples** - update examples to use ISA instructions

Ready to begin implementation.
