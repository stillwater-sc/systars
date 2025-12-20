# Conv2D Operator: Architecture Options and Tradeoffs

This document analyzes different hardware architecture approaches for implementing the 2D convolution operator, a fundamental building block of convolutional neural networks.

## Table of Contents

1. [Background](#background)
2. [Architecture Options](#architecture-options)
   - [SIMD (CPU-style)](#1-simd-single-instruction-multiple-data)
   - [SIMT (GPU-style)](#2-simt-single-instruction-multiple-threads)
   - [im2col with Systolic Array](#3-im2col-with-systolic-array)
   - [Stencil Machine](#4-stencil-machine)
3. [Comparative Analysis](#comparative-analysis)
4. [Energy Analysis](#energy-analysis)
5. [Recommendations for SYSTARS](#recommendations-for-systars)
6. [References](#references)

---

## Background

### The Conv2D Operation

2D convolution slides a kernel (filter) over an input feature map, computing dot products at each position:

```
Y[n, oh, ow, oc] = Σ Σ Σ X[n, oh*s+kh, ow*s+kw, ic] × F[kh, kw, ic, oc] + B[oc]
                  kh kw ic
```

Where:

- **X**: Input tensor [batch, height, width, channels_in]
- **F**: Filter tensor [kernel_h, kernel_w, channels_in, channels_out]
- **Y**: Output tensor [batch, out_h, out_w, channels_out]
- **s**: Stride

### Key Characteristics

| Property | Typical Values | Impact |
|----------|---------------|--------|
| Kernel size | 1×1, 3×3, 5×5, 7×7 | Data reuse pattern |
| Channels | 64 - 2048 | Parallelism opportunity |
| Spatial size | 7×7 - 224×224 | Memory footprint |
| Operations | 10⁹ - 10¹² MACs | Compute requirement |

### Data Reuse Opportunities

Conv2D offers multiple dimensions of data reuse:

1. **Input reuse**: Each input pixel participates in K_h × K_w output calculations
2. **Filter reuse**: Each filter is applied to all spatial positions
3. **Convolutional reuse**: Overlapping windows share input data
4. **Batch reuse**: Same filters applied across batch dimension

Exploiting these reuse patterns is key to energy-efficient implementations.

---

## Architecture Options

### 1. SIMD (Single Instruction, Multiple Data)

#### SIMD Overview

SIMD extends traditional CPU architectures with vector processing units that apply the same operation to multiple data elements simultaneously.

```
┌─────────────────────────────────────────────────────┐
│                       CPU Core                      │
│  ┌─────────┐  ┌─────────────────────────────────┐   │
│  │         │  │         SIMD Vector Unit        │   │
│  │ Scalar  │  │  ┌────┬────┬────┬────┬────┬────┐│   │
│  │         │  │  │ALU │ALU │ALU │ALU │... │ALU ││   │
│  │  Unit   │  │  └────┴────┴────┴────┴────┴────┘│   │
│  │         │  │           (8-64 lanes)          │   │
│  └─────────┘  └─────────────────────────────────┘   │
│                          │                          │
│                    ┌─────┴─────┐                    │
│                    │  L1 Cache │                    │
│                    └───────────┘                    │
└─────────────────────────────────────────────────────┘
```

#### SIMD Implementation Approach

```c
// AVX-512 example: 16 × float32 per instruction
for (int oc = 0; oc < out_channels; oc += 16) {
    __m512 acc = _mm512_setzero_ps();
    for (int kh = 0; kh < kernel_h; kh++) {
        for (int kw = 0; kw < kernel_w; kw++) {
            for (int ic = 0; ic < in_channels; ic++) {
                __m512 x = _mm512_set1_ps(input[kh][kw][ic]);
                __m512 f = _mm512_load_ps(&filter[kh][kw][ic][oc]);
                acc = _mm512_fmadd_ps(x, f, acc);
            }
        }
    }
    _mm512_store_ps(&output[oh][ow][oc], acc);
}
```

#### SIMD Characteristics

| Aspect | Assessment |
|--------|------------|
| **Peak Performance** | 1-4 TFLOPS (modern CPUs) |
| **Vector Width** | 128-512 bits (4-16 FP32) |
| **Memory Hierarchy** | L1/L2/L3 caches + DRAM |
| **Flexibility** | Very high (general purpose) |
| **Power Efficiency** | 10-50 GFLOPS/W |
| **Utilization** | 30-70% typical for CNNs |

#### SIMD Pros

- **Ubiquitous**: Available on all modern CPUs
- **Flexible**: Same hardware runs any algorithm
- **Mature toolchain**: Compilers, libraries (MKL, oneDNN)
- **Low latency**: Good for small batch inference
- **Cache hierarchy**: Automatic data management

#### SIMD Cons

- **Limited parallelism**: 4-16 lanes vs. hundreds in accelerators
- **Memory bound**: Often limited by cache bandwidth
- **Power inefficient**: General-purpose overhead
- **Control overhead**: Loop control, address calculation
- **Suboptimal for large models**: Cannot match accelerator throughput

#### SIMD Real-World Examples

- Intel Xeon with AVX-512 (VNNI for int8)
- AMD EPYC with AVX2
- ARM Neon / SVE
- Apple M-series AMX (matrix extension)

---

### 2. SIMT (Single Instruction, Multiple Threads)

#### SIMT Overview

SIMT (GPU-style) executes thousands of threads in lockstep, with each thread handling a small portion of the computation. Threads are grouped into warps/wavefronts that share an instruction stream.

```text
┌─────────────────────────────────────────────────────────────┐
│                             GPU                             │
│  ┌──────────────────┐  ┌──────────────────┐                 │
│  │ Streaming Multi- │  │ Streaming Multi- │  ...  (N SMs)   │
│  │ processor (SM)   │  │ processor (SM)   │                 │
│  │ ┌──────────────┐ │  │ ┌──────────────┐ │                 │
│  │ │ Warp 0 (32T) │ │  │ │ Warp 0 (32T) │ │                 │
│  │ │ Warp 1 (32T) │ │  │ │ Warp 1 (32T) │ │                 │
│  │ │    ...       │ │  │ │    ...       │ │                 │
│  │ │ Warp N (32T) │ │  │ │ Warp N (32T) │ │                 │
│  │ └──────────────┘ │  │ └──────────────┘ │                 │
│  │ ┌──────────────┐ │  │ ┌──────────────┐ │                 │
│  │ │ Shared Mem   │ │  │ │ Shared Mem   │ │                 │
│  │ │ (48-164 KB)  │ │  │ │ (48-164 KB)  │ │                 │
│  │ └──────────────┘ │  │ └──────────────┘ │                 │
│  └──────────────────┘  └──────────────────┘                 │
│                              │                              │
│                    ┌─────────┴─────────┐                    │
│                    │   GDDR6 / HBM     │                    │
│                    │  (256-2048 GB/s)  │                    │
│                    └───────────────────┘                    │
└─────────────────────────────────────────────────────────────┘
```

#### SIMT Implementation Approach

```cuda
// CUDA kernel: each thread computes one output element
__global__ void conv2d_naive(float* X, float* F, float* Y, ...) {
    int oc = blockIdx.x * blockDim.x + threadIdx.x;
    int ow = blockIdx.y * blockDim.y + threadIdx.y;
    int oh = blockIdx.z;

    float acc = 0.0f;
    for (int kh = 0; kh < KERNEL_H; kh++) {
        for (int kw = 0; kw < KERNEL_W; kw++) {
            for (int ic = 0; ic < IN_CHANNELS; ic++) {
                acc += X[...] * F[...];
            }
        }
    }
    Y[oh][ow][oc] = acc;
}

// Optimized: im2col + cuBLAS GEMM, or cuDNN
```

#### SIMT Characteristics

| Aspect | Assessment |
|--------|------------|
| **Peak Performance** | 10-300 TFLOPS (FP32), 1000+ TFLOPS (INT8) |
| **Thread Count** | 1000s - 100,000s concurrent |
| **Memory Bandwidth** | 256-3000 GB/s (HBM) |
| **Flexibility** | High (programmable) |
| **Power Efficiency** | 50-200 GFLOPS/W |
| **Utilization** | 50-85% with optimized libraries |

#### SIMT Pros

- **Massive parallelism**: Thousands of concurrent threads
- **High memory bandwidth**: HBM provides 1-3 TB/s
- **Mature ecosystem**: cuDNN, TensorRT, highly optimized
- **Tensor Cores**: Dedicated matrix units (NVIDIA)
- **Flexibility**: Same hardware for training and inference

#### SIMT Cons

- **Power hungry**: 150-700W typical
- **Memory capacity**: Limited by GPU memory (24-80 GB)
- **Latency**: Kernel launch overhead, batch requirements
- **Occupancy challenges**: Register/shared memory pressure
- **PCIe bottleneck**: Data transfer to/from CPU

#### SIMT Real-World Examples

- NVIDIA A100/H100 (Tensor Cores)
- AMD MI300X (Matrix Cores)
- Intel Ponte Vecchio (XMX)

---

### 3. im2col with Systolic Array

#### im2col Overview

This approach transforms convolution into matrix multiplication (GEMM), then executes on a 2D array of processing elements with data flowing systolically (rhythmically) through the array.

```
                    im2col Transformation

Input [N,H,W,C]     Patches [N×OH×OW, KH×KW×C]      Output [N×OH×OW, OC]
┌─────────────┐     ┌─────────────────────┐         ┌─────────────────┐
│  ░░░▓▓▓     │     │  patch_0: [K×K×C]   │         │ out_pos_0       │
│  ░░░▓▓▓     │ ──► │  patch_1: [K×K×C]   │  ──►    │ out_pos_1       │
│  ░░░▓▓▓     │     │    ...              │  GEMM   │ ...             │
│             │     │  patch_N×OH×OW      │         │ out_pos_N×OH×OW │
└─────────────┘     └─────────────────────┘         └─────────────────┘
                              ×
                    Filter [KH×KW×C, OC]
                    ┌─────────────────────┐
                    │  filter_0: [K×K×C]  │
                    │  filter_1: [K×K×C]  │
                    │    ...              │
                    │  filter_OC          │
                    └─────────────────────┘
```

```
                    Systolic Array Execution

        ──► A flows horizontally ──►
        ┌────┬────┬────┬────┐
    │   │ PE │ PE │ PE │ PE │
    B   ├────┼────┼────┼────┤
    │   │ PE │ PE │ PE │ PE │
 flows  ├────┼────┼────┼────┤
    │   │ PE │ PE │ PE │ PE │
  down  ├────┼────┼────┼────┤
    ▼   │ PE │ PE │ PE │ PE │
        └────┴────┴────┴────┘
              C accumulates in place (output-stationary)
              or flows right (weight-stationary)
```

#### im2col Implementation Approach (Current SYSTARS)

```python
# Mapping Conv2D to GEMM
M = batch * out_h * out_w      # Output spatial positions
N = channels_out               # Output channels (filters)
K = kernel_h * kernel_w * channels_in  # Flattened kernel

# Tile computation across systolic array
for tile_m in range(tiles_M):
    for tile_n in range(tiles_N):
        for tile_k in range(tiles_K):
            # Load input patch tile (im2col on-the-fly or pre-materialized)
            load_X_tile(tile_m, tile_k)
            # Load filter tile
            load_F_tile(tile_k, tile_n)
            # Systolic array MAC accumulation
            compute_tile()
        store_Y_tile(tile_m, tile_n)
```

#### im2col Characteristics

| Aspect | Assessment |
|--------|------------|
| **Peak Performance** | 10-400 TOPS (INT8) |
| **Array Size** | 64×64 to 256×256 PEs |
| **Data Reuse** | High within array, limited across tiles |
| **Flexibility** | Moderate (GEMM-centric) |
| **Power Efficiency** | 1-5 TOPS/W |
| **Utilization** | 70-95% for large matrices |

#### im2col Pros

- **Hardware reuse**: Same array for Conv2D and GEMM (FC layers)
- **High utilization**: Dense matrix operations map well
- **Simple PE design**: Just MAC + registers
- **Proven architecture**: Google TPU, many commercial designs
- **Dataflow flexibility**: OS, WS, IS variants

#### im2col Cons

- **Data duplication**: im2col amplifies memory reads by K_h × K_w
- **Memory bandwidth**: High pressure from data expansion
- **Edge inefficiency**: Small/odd dimensions waste PEs
- **Tiling overhead**: Setup cost for each tile
- **Not optimal for depthwise**: Poor utilization for Cin=1

#### im2col Data Duplication Analysis

For a 3×3 convolution:

```
Original input reads:    N × H × W × C
After im2col expansion:  N × OH × OW × (9 × C)
                         ─────────────────────
Amplification factor:    ~9× (for 3×3 kernel)
```

This means each input element is read from memory up to 9 times instead of once.

#### im2col Real-World Examples

- Google TPU v1-v4
- NVIDIA Deep Learning Accelerator (NVDLA)
- Intel Nervana NNP
- Current SYSTARS implementation

---

### 4. Stencil Machine

#### Stencil Machine Overview

A stencil machine is specifically designed for spatial operations like convolution. It uses line buffers to cache rows of the input and shift registers to slide the convolution window, achieving optimal input data reuse.

```
┌─────────────────────────────────────────────────────────────────┐
│                       Stencil Machine                           │
│                                                                 │
│  Input Stream          Line Buffers              Window Buffer  │
│  ┌──────────┐         ┌─────────────┐           ┌───────────┐   │
│  │ Row N+2  │────────►│ Line Buf 0  │──────────►│ ■ ■ ■     │   │
│  │ Row N+1  │────────►│ Line Buf 1  │──────────►│ ■ ■ ■     │   │
│  │ Row N    │────────►│ Line Buf 2  │──────────►│ ■ ■ ■     │   │
│  └──────────┘         └─────────────┘           └───────────┘   │
│       │                      │                        │         │
│       │               Width x Channels                │         │
│       │                      │                     3×3 Window   │
│       ▼                      ▼                        │         │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    Compute Array                         │   │
│  │  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐         ┌─────┐         │   │
│  │  │ MAC │ │ MAC │ │ MAC │ │ MAC │  ...    │ MAC │         │   │
│  │  │ OC0 │ │ OC1 │ │ OC2 │ │ OC3 │         │ OCn │         │   │
│  │  └─────┘ └─────┘ └─────┘ └─────┘         └─────┘         │   │
│  │     │       │       │       │               │            │   │
│  │     └───────┴───────┴───────┴───────────────┘            │   │
│  │                         │                                │   │
│  │                    ┌────┴────┐                           │   │
│  │                    │ Output  │                           │   │
│  │                    │ Stream  │                           │   │
│  │                    └─────────┘                           │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

#### Stencil Machine Line Buffer Operation

```
Cycle 0: Window at position (0,0)
┌───┬───┬───┬───┬───┬───┬───┐
│ A │ B │ C │ D │ E │ F │ G │ ◄── Line Buffer 0
├───┼───┼───┼───┼───┼───┼───┤
│ H │ I │ J │ K │ L │ M │ N │ ◄── Line Buffer 1
├───┼───┼───┼───┼───┼───┼───┤
│ O │ P │ Q │ R │ S │ T │ U │ ◄── Line Buffer 2
└───┴───┴───┴───┴───┴───┴───┘
  ▲
  └── Window: [A,B,C; H,I,J; O,P,Q]

Cycle 1: Window at position (0,1)
┌───┬───┬───┬───┬───┬───┬───┐
│ A │ B │ C │ D │ E │ F │ G │
├───┼───┼───┼───┼───┼───┼───┤
│ H │ I │ J │ K │ L │ M │ N │
├───┼───┼───┼───┼───┼───┼───┤
│ O │ P │ Q │ R │ S │ T │ U │
└───┴───┴───┴───┴───┴───┴───┘
      ▲
      └── Window: [B,C,D; I,J,K; P,Q,R]

      Shift by 1 element (just register shift, no memory read!)
```

#### Stencil Machine Implementation Approach

```python
class StencilConvEngine:
    def __init__(self, kernel_h, width, channels):
        # Line buffers: kernel_h rows × width × channels
        self.line_buffers = [SRAM(width * channels) for _ in range(kernel_h)]
        # Window registers: kernel_h × kernel_w × channels
        self.window_regs = Registers(kernel_h * kernel_w * channels)

    def process_pixel(self, new_pixel):
        # 1. Shift window registers (no memory access)
        self.window_regs.shift()

        # 2. Load new column from line buffers into window
        for row in range(kernel_h - 1):
            self.window_regs[row, -1] = self.line_buffers[row].read()
        self.window_regs[-1, -1] = new_pixel

        # 3. Update line buffers (shift rows)
        for row in range(kernel_h - 1):
            self.line_buffers[row].write(self.line_buffers[row + 1].read())
        self.line_buffers[-1].write(new_pixel)

        # 4. Compute: window × filter (all output channels in parallel)
        return self.compute_all_filters(self.window_regs)
```

#### Stencil Machine Characteristics

| Aspect | Assessment |
|--------|------------|
| **Peak Performance** | Depends on # parallel output channels |
| **Memory Reads** | **1× per input pixel** (optimal) |
| **Line Buffer Size** | K_h × W × C × element_size |
| **Flexibility** | Lower (conv-specific) |
| **Power Efficiency** | **2-10 TOPS/W** (best for conv) |
| **Utilization** | 90%+ for regular convolutions |

#### Stencil Machine Pros

- **Optimal data reuse**: Each input read exactly once from DRAM
- **Minimal memory bandwidth**: No im2col amplification
- **Energy efficient**: Most data movement is register shifts
- **Streaming friendly**: Processes data as it arrives
- **Low latency**: No need to buffer entire feature map
- **Natural for edge/padding**: Explicit boundary handling

#### Stencil Machine Cons

- **Less flexible**: Optimized specifically for convolution
- **Separate from GEMM**: Cannot reuse for fully-connected layers
- **Line buffer overhead**: Requires K_h × W × C on-chip SRAM
- **Complex for large strides**: Skip logic needed for stride > 1
- **Channel parallelism tradeoff**: More output channels = more MACs

#### Stencil Machine Line Buffer Sizing

| Input Size | Kernel | Channels | Line Buffer Size |
|------------|--------|----------|------------------|
| 224×224 | 3×3 | 64 | 3 × 224 × 64 = 43 KB |
| 56×56 | 3×3 | 256 | 3 × 56 × 256 = 43 KB |
| 14×14 | 3×3 | 512 | 3 × 14 × 512 = 21 KB |
| 7×7 | 3×3 | 512 | 3 × 7 × 512 = 10.5 KB |

Line buffer requirements scale with width × channels, decreasing through the network.

#### Stencil Machine Real-World Examples

- Eyeriss (MIT) - Row-stationary with local reuse
- ShiDianNao - Specialized CNN accelerator
- Many edge AI ASICs
- FPGA CNN implementations

---

## Comparative Analysis

### Performance Comparison

| Architecture | Peak TOPS | Typical Util. | Effective TOPS |
|--------------|-----------|---------------|----------------|
| SIMD (CPU) | 0.5-4 | 30-70% | 0.2-2.5 |
| SIMT (GPU) | 100-500 | 50-85% | 50-400 |
| Systolic (im2col) | 50-400 | 70-95% | 35-380 |
| Stencil | 10-100 | 90-99% | 9-99 |

### Memory Bandwidth Comparison

For 3×3 convolution on 224×224×64 input, 128 output channels:

| Architecture | Input Reads | Amplification | Bandwidth Need |
|--------------|-------------|---------------|----------------|
| SIMD | 9× (cache helps) | ~3× effective | Medium |
| SIMT | 9× (shared mem helps) | ~2-3× effective | High |
| Systolic (im2col) | 9× per tile | 9× worst case | Very High |
| Stencil | 1× | **1×** | **Minimal** |

### Energy Efficiency

| Architecture | TOPS/W | Memory Energy | Total Efficiency |
|--------------|--------|---------------|------------------|
| SIMD | 0.05-0.2 | High | Poor |
| SIMT | 0.3-1.0 | Medium-High | Moderate |
| Systolic | 1-5 | Medium-High | Good |
| Stencil | **2-10** | **Low** | **Excellent** |

### Flexibility vs Efficiency Tradeoff

```
High ┌─────────────────────────────────────────┐
     │                                         │
     │   SIMD (CPU)                            │
     │      ●                                  │
     │                                         │
F    │         SIMT (GPU)                      │
l    │            ●                            │
e    │                                         │
x    │              Systolic                   │
i    │                 ●                       │
b    │                                         │
i    │                    Stencil              │
l    │                       ●                 │
i    │                                         │
t    │                          Fixed-function │
y    │                             ●           │
     │                                         │
Low  └─────────────────────────────────────────┘
    Low ◄──────── Energy Efficiency ────────► High
```

### Operation Support

| Operation | SIMD | SIMT | Systolic | Stencil |
|-----------|------|------|----------|---------|
| Conv 1×1 | ✓ | ✓ | ✓✓ | ✓ |
| Conv 3×3 | ✓ | ✓ | ✓ | ✓✓ |
| Conv 5×5+ | ✓ | ✓ | ✓ | ✓✓ |
| Depthwise | ✓ | ✓ | △ | ✓✓ |
| GEMM/FC | ✓ | ✓ | ✓✓ | △ |
| Pooling | ✓ | ✓ | △ | ✓✓ |
| Elementwise | ✓ | ✓ | △ | ✓ |
| Arbitrary ops | ✓✓ | ✓✓ | △ | ✗ |

Legend: ✓✓ = Excellent, ✓ = Good, △ = Possible but inefficient, ✗ = Not supported

---

## Energy Analysis

### Energy Cost Hierarchy

Understanding relative energy costs is crucial for efficiency:

```
Operation                    Relative Energy (pJ)
─────────────────────────────────────────────────
8-bit MAC                           0.2
32-bit MAC                          3.0
Register read/write                 0.5
SRAM read (32-bit)                  5.0
SRAM write (32-bit)                 5.0
DRAM read (32-bit)                200-500
DRAM write (32-bit)               200-500
─────────────────────────────────────────────────
```

**Key insight**: DRAM access costs 100-1000× more than computation!

### Energy Breakdown by Architecture

#### SIMD (CPU)

```
Energy per MAC operation:
├── Instruction fetch/decode:  30%
├── Address calculation:       20%
├── Cache access:              25%
├── Actual MAC computation:    10%
└── Data movement overhead:    15%
```

#### SIMT (GPU)

```
Energy per MAC operation:
├── Thread scheduling:         15%
├── Shared memory access:      20%
├── Register file access:      15%
├── Actual MAC computation:    20%
└── Memory controller:         30%
```

#### Systolic (im2col)

```
Energy per MAC operation:
├── Data loading (im2col):     40%  ← im2col amplification!
├── Systolic data flow:        15%
├── Actual MAC computation:    30%
└── Result writeback:          15%
```

#### Stencil

```
Energy per MAC operation:
├── Line buffer access:        15%
├── Register shifts:           10%
├── Actual MAC computation:    50%  ← Most energy goes to compute!
└── Result writeback:          25%
```

### Quantitative Comparison

For a layer: 3×3 conv, 56×56×256 input, 256 output channels

| Architecture | MACs | DRAM Reads | Energy (relative) |
|--------------|------|------------|-------------------|
| Baseline | 1.85G | - | - |
| Systolic (im2col) | 1.85G | 9× input | **1.0×** |
| Stencil | 1.85G | 1× input | **0.3-0.5×** |

The stencil approach can be **2-3× more energy efficient** for convolution due to reduced memory traffic.

---

## Recommendations for SYSTARS

### Current State

SYSTARS currently uses the **im2col + Systolic Array** approach, which:

- Maximizes hardware reuse (same array for Conv2D and GEMM)
- Has proven high utilization for large matrices
- Suffers from memory bandwidth amplification for convolution

### Strategic Direction: Dual-Path Architecture

After careful analysis, the chosen strategy is a **Dual-Path Architecture** that maintains clean separation between:

1. **Pure Systolic Array**: Optimal for GEMM/matmul operations
2. **Dedicated Stencil Machine**: Optimal for Conv2D operations

This approach prioritizes **design clarity** and **per-workload optimization** over hardware reuse.

```
┌─────────────────────────────────────────────────────────────────┐
│                      SYSTARS Accelerator                        │
│                                                                 │
│  ┌───────────────────────┐      ┌───────────────────────┐       │
│  │    Systolic Array     │      │    Stencil Machine    │       │
│  │    (Pure GEMM)        │      │    (Pure Conv2D)      │       │
│  │                       │      │                       │       │
│  │  ┌─────────────────┐  │      │  ┌─────────────────┐  │       │
│  │  │ PE  PE  PE  PE  │  │      │  │  Line Buffers   │  │       │
│  │  │ PE  PE  PE  PE  │  │      │  ├─────────────────┤  │       │
│  │  │ PE  PE  PE  PE  │  │      │  │ Window Former   │  │       │
│  │  │ PE  PE  PE  PE  │  │      │  ├─────────────────┤  │       │
│  │  └─────────────────┘  │      │  │ MAC Array       │  │       │
│  │                       │      │  └─────────────────┘  │       │
│  │  Workloads:           │      │  Workloads:           │       │
│  │  - Dense matmul       │      │  - Conv2D (all sizes) │       │
│  │  - FC layers          │      │  - Depthwise conv     │       │
│  │  - Attention (QKV)    │      │  - Pooling            │       │
│  │  - Batched GEMM       │      │  - Image processing   │       │
│  └───────────────────────┘      └───────────────────────┘       │
│              │                            │                     │
│              └────────────┬───────────────┘                     │
│                           │                                     │
│                  ┌────────┴────────┐                            │
│                  │ Unified Memory  │                            │
│                  │   Controller    │                            │
│                  └─────────────────┘                            │
│                           │                                     │
│                  ┌────────┴────────┐                            │
│                  │  Scratchpad /   │                            │
│                  │  Accumulator    │                            │
│                  └─────────────────┘                            │
└─────────────────────────────────────────────────────────────────┘
```

### Rationale for Dual-Path

| Benefit | Description |
|---------|-------------|
| **Clean abstractions** | Systolic array remains a pure, well-understood GEMM engine |
| **Optimal per-workload** | Each unit tuned for its domain without compromise |
| **Independent verification** | Test each unit in isolation with appropriate workloads |
| **Composable IP** | Mix and match for different products/applications |
| **Future-proof** | New operators don't pollute existing designs |
| **Benchmark baseline** | Compare designs with real performance data |

### Architecture Options Detail

#### Option A: Hybrid Systolic Array (Future Phase)

A *secondary* systolic array variant that adds line buffer front-end:

```
┌─────────────────────────────────────────────────────────┐
│              Hybrid Systolic Array (Variant)            │
│                                                         │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────┐  │
│  │ Line Buffer  │───►│ Patch Former │───►│ Systolic  │  │
│  │ Unit         │    │              │    │ Array     │  │
│  └──────────────┘    └──────────────┘    └───────────┘  │
│         ▲                   │                   │       │
│         │                   ▼                   ▼       │
│    ┌────┴────┐        ┌─────────┐        ┌──────────┐   │
│    │ DRAM    │        │ Filter  │        │  Output  │   │
│    │ (Input) │        │ Buffer  │        │  Buffer  │   │
│    └─────────┘        └─────────┘        └──────────┘   │
└─────────────────────────────────────────────────────────┘
```

**Purpose**: Area-constrained edge deployments where two separate units is prohibitive.
**Timing**: After dual-path is proven, as an alternative design option.
**Note**: The pure systolic array is always maintained as the primary GEMM unit.

#### Option B: Dual-Path Architecture (Primary Strategy)

Maintain two separate, optimized datapaths:

```
                    ┌─────────────────┐
                    │   Scheduler /   │
                    │   Dispatcher    │
                    └────────┬────────┘
                             │
              ┌──────────────┴─────────────┐
              ▼                            ▼
    ┌──────────────────┐          ┌──────────────────┐
    │   Systolic Array │          │  Stencil Machine │
    │   (Pure GEMM)    │          │  (Pure Conv2D)   │
    │                  │          │                  │
    │  - No im2col     │          │  - Line buffers  │
    │    overhead for  │          │  - 1× input read │
    │    actual GEMM   │          │  - Optimal for   │
    │  - Full array    │          │    spatial ops   │
    │    utilization   │          │                  │
    └──────────────────┘          └──────────────────┘
```

**Implementation complexity**: Higher initial investment
**Energy improvement**: Optimal for each operation type
**Design quality**: Clean, verifiable, maintainable

#### Option C: Enhanced im2col with Caching (Incremental)

Keep current architecture but add smart caching for overlap regions:

```python
# Tile-aware im2col with overlap caching
class SmartImCol:
    def __init__(self):
        self.overlap_cache = SRAM(size=overlap_region)

    def load_tile(self, tile_m, tile_k):
        # Check if overlap region is cached
        if self.is_cached(tile_m, tile_k):
            # Load only non-overlapping portion from DRAM
            new_data = self.load_new_portion()
            cached = self.overlap_cache.read()
            return combine(cached, new_data)
        else:
            # Full load
            return self.load_full_tile()
```

**Purpose**: Quick improvement to existing Conv2D instruction.
**Timing**: Can be done in parallel with stencil machine development.

### Implementation Roadmap

```text
Implementation Phases

Phase 1: Stencil Machine Design
════════════════════════════════
  ├── Define micro-architecture specification
  ├── Implement LineBufferUnit
  ├── Implement WindowFormer (sliding window)
  ├── Implement ChannelParallelMAC array
  ├── Implement StencilController (FSM)
  ├── Unit tests + cocotb verification
  └── Standalone Conv2D demo

Phase 2: Integration
════════════════════
  ├── Unified memory controller (arbitration)
  ├── Operation scheduler / dispatcher
  ├── Shared scratchpad access protocol
  ├── ISA extension for stencil operations
  └── End-to-end CNN demo (conv→stencil, FC→systolic)

Phase 3: Hybrid Exploration (Future)
════════════════════════════════════
  ├── Design hybrid systolic array variant
  ├── Benchmark: dual-path vs hybrid (area/power/perf)
  ├── Document tradeoffs for different deployment targets
  └── Recommend configurations per use case
```

### Workload-to-Unit Mapping

| Workload | Unit | Rationale |
|----------|------|-----------|
| Dense matmul (BLAS) | Systolic Array | Native GEMM operation |
| Fully-connected layers | Systolic Array | Weight × activation GEMM |
| Attention (QK^T, softmax·V) | Systolic Array | Batched matrix multiply |
| Batched GEMM | Systolic Array | High utilization |
| Conv2D (1×1) | Either | 1×1 maps well to GEMM |
| Conv2D (3×3, 5×5, 7×7) | Stencil Machine | Optimal input reuse |
| Depthwise separable conv | Stencil Machine | Poor GEMM utilization |
| Pooling (max, avg) | Stencil Machine | Spatial sliding window |
| Image preprocessing | Stencil Machine | Streaming friendly |

### Design Portfolio

The dual-path strategy creates a portfolio of IP blocks:

| IP Block | Target Use Case | Power Profile |
|----------|-----------------|---------------|
| **Pure Systolic Array** | HPC, scientific computing, Transformers | Performance-optimized |
| **Stencil Machine** | CNN inference, edge vision | Energy-optimized |
| **Hybrid Array** (future) | Area-constrained edge SoCs | Balanced |

This provides flexibility to configure SYSTARS for different market segments without compromising design quality.

---

## References

### Academic Papers

1. Chen, Y., et al. "Eyeriss: A Spatial Architecture for Energy-Efficient Dataflow for Convolutional Neural Networks." ISCA 2016.
2. Jouppi, N., et al. "In-Datacenter Performance Analysis of a Tensor Processing Unit." ISCA 2017.
3. Chen, Y., et al. "Eyeriss v2: A Flexible Accelerator for Emerging Deep Neural Networks." JSSC 2019.
4. Sze, V., et al. "Efficient Processing of Deep Neural Networks: A Tutorial and Survey." Proc. IEEE 2017.

### Industry Implementations

- Google TPU: Systolic array with bfloat16
- NVIDIA Tensor Cores: Matrix multiply-accumulate units
- Apple Neural Engine: Specialized CNN accelerator
- Qualcomm Hexagon: DSP with NN extensions

### Relevant SYSTARS Documentation

- [Implementation Plan](../plan/implementation.md)
- [ISA Instructions](../plan/isa-instructions.md)
- [DMA Descriptors](../architecture/dma-descriptors.md)

---

*Document version: 1.0*
*Last updated: 2025-12-19*
