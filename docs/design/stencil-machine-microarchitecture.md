# Stencil Machine Micro-Architecture Specification

This document specifies the micro-architecture of the SYSTARS Stencil Machine, a dedicated hardware unit optimized for 2D convolution and other spatial operations.

## Table of Contents

1. [Overview](#overview)
2. [Top-Level Architecture](#top-level-architecture)
3. [Line Buffer Unit](#line-buffer-unit)
4. [Window Former](#window-former)
5. [Channel-Parallel MAC Array](#channel-parallel-mac-array)
6. [Filter Buffer](#filter-buffer)
7. [Output Accumulator](#output-accumulator)
8. [Stencil Controller](#stencil-controller)
9. [Memory Interfaces](#memory-interfaces)
10. [Configuration Registers](#configuration-registers)
11. [Timing Analysis](#timing-analysis)
12. [Area and Power Estimates](#area-and-power-estimates)
13. [Implementation Notes](#implementation-notes)

---

## Overview

### Design Goals

| Goal | Target | Rationale |
|------|--------|-----------|
| **Energy efficiency** | 2-5× vs im2col | Minimize memory traffic |
| **Input reuse** | 1× DRAM read per pixel | Line buffer architecture |
| **Throughput** | Match systolic array | Parallel output channels |
| **Flexibility** | 1×1 to 7×7 kernels | Configurable window size |
| **Utilization** | >90% for typical CNNs | Streaming dataflow |

### Supported Operations

- **Conv2D**: Standard 2D convolution (any kernel size 1-7)
- **Depthwise Conv**: Per-channel convolution (Cin = groups)
- **Pooling**: Max pooling, average pooling
- **Strided Conv**: Stride 1, 2, or 4
- **Dilated Conv**: Dilation 1 or 2
- **Padding**: Zero padding, reflection padding

### Key Parameters

| Parameter | Symbol | Typical Range | Description |
|-----------|--------|---------------|-------------|
| Input width | W_in | 7 - 224 | Feature map width |
| Input channels | C_in | 3 - 2048 | Input channel count |
| Output channels | C_out | 16 - 2048 | Number of filters |
| Kernel height | K_h | 1 - 7 | Convolution kernel height |
| Kernel width | K_w | 1 - 7 | Convolution kernel width |
| Stride | S | 1, 2, 4 | Convolution stride |
| Parallel channels | P_c | 16 - 64 | MAC array parallelism |

---

## Top-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           STENCIL MACHINE                                   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                           INPUT DATAPATH                            │    │
│  │                                                                     │    │
│  │    DRAM        Line Buffer Unit          Window Former              │    │
│  │   ┌─────┐     ┌─────────────────┐      ┌─────────────────┐          │    │
│  │   │     │     │ ┌─────────────┐ │      │ ┌─────────────┐ │          │    │
│  │   │  I  │────►│ │ Line Buf 0  │ │─────►│ │ Shift Regs  │ │          │    │
│  │   │  N  │     │ ├─────────────┤ │      │ │ (K_h × K_w  │ │          │    │
│  │   │  P  │     │ │ Line Buf 1  │ │      │ │  × C_in)    │ │          │    │
│  │   │  U  │     │ ├─────────────┤ │      │ └─────────────┘ │          │    │
│  │   │  T  │     │ │ Line Buf 2  │ │      │       │         │          │    │
│  │   │     │     │ ├─────────────┤ │      │       ▼         │          │    │
│  │   │  S  │     │ │    ...      │ │      │ ┌───────────┐   │          │    │
│  │   │  T  │     │ ├─────────────┤ │      │ │  Window   │   │          │    │
│  │   │  R  │     │ │ Line Buf K-1│ │      │ │  Output   │   │          │    │
│  │   │  E  │     │ └─────────────┘ │      │ └───────────┘   │          │    │
│  │   │  A  │     │                 │      │                 │          │    │
│  │   │  M  │     │ (K_h buffers,   │      │ (Sliding window │          │    │
│  │   │     │     │  W×C_in each)   │      │  extraction)    │          │    │
│  │   └─────┘     └─────────────────┘      └────────┬────────┘          │    │
│  │                                                 │                   │    │
│  └─────────────────────────────────────────────────┼───────────────────┘    │
│                                                    │                        │
│                                                    ▼                        │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                        COMPUTE DATAPATH                             │    │
│  │                                                                     │    │
│  │   Filter Buffer            MAC Array              Output Accum      │    │
│  │  ┌─────────────┐     ┌─────────────────┐      ┌─────────────────┐   │    │
│  │  │ ┌─────────┐ │     │ ┌─────────────┐ │      │ ┌─────────────┐ │   │    │
│  │  │ │Filter 0 │ │────►│ │ MAC Bank 0  │ │─────►│ │  Acc Bank 0 │ │   │    │
│  │  │ ├─────────┤ │     │ ├─────────────┤ │      │ ├─────────────┤ │   │    │
│  │  │ │Filter 1 │ │────►│ │ MAC Bank 1  │ │─────►│ │  Acc Bank 1 │ │   │    │
│  │  │ ├─────────┤ │     │ ├─────────────┤ │      │ ├─────────────┤ │   │    │
│  │  │ │  ...    │ │     │ │    ...      │ │      │ │     ...     │ │   │    │
│  │  │ ├─────────┤ │────►│ ├─────────────┤ │─────►│ ├─────────────┤ │   │    │
│  │  │ │Filter P │ │     │ │ MAC Bank P  │ │      │ │  Acc Bank P │ │   │    │
│  │  │ └─────────┘ │     │ └─────────────┘ │      │ └─────────────┘ │   │    │
│  │  └─────────────┘     └─────────────────┘      └────────┬────────┘   │    │
│  │                                                        │            │    │
│  │   (P_c filters         (P_c parallel             (P_c accumulators, │    │
│  │    cached locally)      MAC units)                post-processing)  │    │
│  │                                                        │            │    │
│  └────────────────────────────────────────────────────────┼────────────┘    │
│                                                           │                 │
│                                                           ▼                 │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                        OUTPUT DATAPATH                              │    │
│  │                                                                     │    │
│  │   ┌───────────────┐    ┌───────────────┐    ┌───────────────┐       │    │
│  │   │  Activation   │───►│  Quantization │───►│ Output Stream │       │    │
│  │   │  (ReLU, etc.) │    │  (optional)   │    │  to DRAM      │       │    │
│  │   └───────────────┘    └───────────────┘    └───────────────┘       │    │
│  │                                                                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                         CONTROLLER                                   │   │
│  │  ┌────────────┐  ┌────────────┐      ┌────────────┐  ┌────────────┐  │   │
│  │  │  Config    │  │   Main     │      │  Address   │  │  Sync &    │  │   │
│  │  │  Registers │  │   FSM      │      │  Generator │  │  Handshake │  │   │
│  │  └────────────┘  └────────────┘      └────────────┘  └────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow Summary

1. **Input pixels** stream from DRAM one row at a time
2. **Line buffers** store K_h rows of the input feature map
3. **Window former** extracts K_h × K_w × C_in window as it slides
4. **MAC array** computes dot products with P_c filters in parallel
5. **Accumulators** accumulate partial sums across input channels
6. **Output** streams results back to DRAM after activation

---

## Line Buffer Unit

### Purpose

Store K_h consecutive rows of the input feature map to enable sliding window extraction without repeated DRAM reads.

### Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                       LINE BUFFER UNIT                               │
│                                                                      │
│   Input Stream                                                       │
│   (W_in × C_in pixels/row)                                          │
│        │                                                             │
│        ▼                                                             │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │                    Input Demux / Router                      │   │
│   └───────────┬─────────────┬─────────────┬─────────────────────┘   │
│               │             │             │                          │
│               ▼             ▼             ▼                          │
│   ┌───────────────┐ ┌───────────────┐ ┌───────────────┐             │
│   │  Line Buf 0   │ │  Line Buf 1   │ │  Line Buf K-1 │             │
│   │  ┌─────────┐  │ │  ┌─────────┐  │ │  ┌─────────┐  │             │
│   │  │  SRAM   │  │ │  │  SRAM   │  │ │  │  SRAM   │  │             │
│   │  │ W × C_in│  │ │  │ W × C_in│  │ │  │ W × C_in│  │             │
│   │  │ × 8bits │  │ │  │ × 8bits │  │ │  │ × 8bits │  │             │
│   │  └─────────┘  │ │  └─────────┘  │ │  └─────────┘  │             │
│   │       │       │ │       │       │ │       │       │             │
│   │  ┌────┴────┐  │ │  ┌────┴────┐  │ │  ┌────┴────┐  │             │
│   │  │ Read Ptr│  │ │  │ Read Ptr│  │ │  │ Read Ptr│  │             │
│   │  └─────────┘  │ │  └─────────┘  │ │  └─────────┘  │             │
│   └───────┬───────┘ └───────┬───────┘ └───────┬───────┘             │
│           │                 │                 │                      │
│           └─────────────────┼─────────────────┘                      │
│                             │                                        │
│                             ▼                                        │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │                    Output Mux (to Window Former)             │   │
│   │                    Outputs K_h × C_in values/cycle           │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Specifications

| Parameter | Value | Notes |
|-----------|-------|-------|
| Number of line buffers | K_h (configurable, max 7) | One per kernel row |
| Buffer depth | W_in × C_in | Full row storage |
| Data width | 8 bits | INT8 activations |
| Read ports | K_h parallel | One per buffer |
| Write ports | 1 | Sequential row fill |
| Total SRAM | K_h × W_in × C_in × 8 bits | See sizing table |

### SRAM Sizing Examples

| Input Size | Kernel | Channels | Line Buffer Total |
|------------|--------|----------|-------------------|
| 224×224 | 3×3 | 64 | 3 × 224 × 64 = 43 KB |
| 112×112 | 3×3 | 128 | 3 × 112 × 128 = 43 KB |
| 56×56 | 3×3 | 256 | 3 × 56 × 256 = 43 KB |
| 28×28 | 3×3 | 512 | 3 × 28 × 512 = 43 KB |
| 14×14 | 3×3 | 512 | 3 × 14 × 512 = 21 KB |
| 7×7 | 3×3 | 512 | 3 × 7 × 512 = 10.5 KB |

**Observation**: Line buffer size naturally decreases through CNN layers.

### Operation

```
Row Fill Phase:
  1. New row arrives from DRAM
  2. Write to line buffer [row_idx % K_h]
  3. Increment row counter

Sliding Window Phase:
  1. Read column col from all K_h line buffers (K_h × C_in values)
  2. Output to window former
  3. Increment column pointer
  4. When col reaches W_out, advance to next output row
```

### Circular Buffer Management

```
Line buffer assignment (for K_h = 3):

Input row 0 → Line Buf 0    Output window uses: [0, -, -]
Input row 1 → Line Buf 1    Output window uses: [0, 1, -]
Input row 2 → Line Buf 2    Output window uses: [0, 1, 2] ← First valid window
Input row 3 → Line Buf 0    Output window uses: [1, 2, 0] (overwrite oldest)
Input row 4 → Line Buf 1    Output window uses: [2, 0, 1]
...

The "oldest" buffer is always overwritten, and read order rotates.
```

### Interface Signals

```python
class LineBufferUnit(Component):
    """Line buffer unit for stencil machine."""

    def __init__(self, config: StencilConfig):
        ports = {
            # Input stream (from DRAM/memory controller)
            "in_valid": In(1),
            "in_ready": Out(1),
            "in_data": In(config.input_bits),  # One pixel at a time
            "in_last_col": In(1),  # End of row marker
            "in_last_row": In(1),  # End of frame marker

            # Output to window former (K_h parallel outputs)
            "out_valid": Out(1),
            "out_ready": In(1),
            "out_data": Out(config.input_bits * config.max_kernel_h),  # K_h pixels
            "out_col": Out(16),  # Current column index

            # Configuration
            "cfg_kernel_h": In(4),  # Active kernel height (1-7)
            "cfg_width": In(16),    # Input width
            "cfg_channels": In(16), # Input channels

            # Status
            "row_count": Out(16),   # Rows received
            "ready_for_compute": Out(1),  # K_h rows buffered
        }
        super().__init__(ports)
```

---

## Window Former

### Window Former Purpose

Extract the K_h × K_w × C_in convolution window from line buffer outputs using shift registers, providing the complete window to the MAC array each cycle.

### Window Former Micro-Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          WINDOW FORMER                                  │
│                                                                         │
│  From Line Buffers (K_h × C_in values per cycle)                        │
│        │                                                                │
│        ▼                                                                │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                   SHIFT REGISTER ARRAY                          │    │
│  │                                                                 │    │
│  │  Row 0:  [C0..Cn] → [C0..Cn] → [C0..Cn] → ... → [C0..Cn]        │    │
│  │          col K-1    col K-2    col K-3          col 0           │    │
│  │             │          │          │              │              │    │
│  │  Row 1:  [C0..Cn] → [C0..Cn] → [C0..Cn] → ... → [C0..Cn]        │    │
│  │             │          │          │              │              │    │
│  │    ...      │          │          │              │              │    │
│  │             │          │          │              │              │    │
│  │  Row K-1:[C0..Cn] → [C0..Cn] → [C0..Cn] → ... → [C0..Cn]        │    │
│  │             │          │          │              │              │    │
│  └─────────────┼──────────┼──────────┼──────────────┼──────────────┘    │
│                │          │          │              │                   │
│                ▼          ▼          ▼              ▼                   │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    WINDOW OUTPUT MUX                            │    │
│  │                                                                 │    │
│  │   Window[0,0,:]  Window[0,1,:]  ...  Window[K-1,K-1,:]          │    │
│  │        │              │                    │                    │    │
│  │        └──────────────┴────────────────────┘                    │    │
│  │                         │                                       │    │
│  │              K_h × K_w × C_in values                            │    │
│  │                         │                                       │    │
│  └─────────────────────────┼───────────────────────────────────────┘    │
│                            │                                            │
│                            ▼                                            │
│                    To MAC Array                                         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Shift Register Operation

Each cycle:

1. New column of K_h × C_in values enters from line buffers
2. All values shift right by one position
3. Oldest column (position K_w-1) is discarded
4. Window output is all K_h × K_w × C_in values

```
Cycle 0: Window = [col0, ---, ---]  (filling)
Cycle 1: Window = [col1, col0, ---] (filling)
Cycle 2: Window = [col2, col1, col0] ← First valid 3×3 window
Cycle 3: Window = [col3, col2, col1] ← Shifted by 1
...
```

### Handling Channels

For C_in input channels, we have two options:

#### Option A: Channel-Serial (simpler, lower area)

```
Process one input channel at a time:
- Window size: K_h × K_w × 1
- Cycles per output: C_in
- Accumulate across channels in MAC array
```

#### Option B: Channel-Parallel (higher throughput)

```
Process all input channels together:
- Window size: K_h × K_w × C_in
- Cycles per output: 1 (for K_h × K_w × C_in ≤ MAC capacity)
- More registers, but maximum throughput
```

**Recommended**: Channel-serial for area efficiency, with configurable parallelism.

### Stride and Dilation Support

**Stride handling**:

```
Stride = 1: Output every cycle after window is valid
Stride = 2: Output every 2nd cycle (skip alternate windows)
Stride = 4: Output every 4th cycle
```

**Dilation handling**:

```
Dilation = 1: Read consecutive columns from line buffers
Dilation = 2: Read every 2nd column (requires wider line buffer read port or multi-cycle)
```

#### Stride/Dilation Interface Signals

```python
class WindowFormer(Component):
    """Sliding window extraction for stencil machine."""

    def __init__(self, config: StencilConfig):
        ports = {
            # Input from line buffers
            "in_valid": In(1),
            "in_ready": Out(1),
            "in_data": In(config.input_bits * config.max_kernel_h),
            "in_channel": In(16),  # Current input channel

            # Output window to MAC array
            "out_valid": Out(1),
            "out_ready": In(1),
            "out_window": Out(config.input_bits * config.max_kernel_h * config.max_kernel_w),
            "out_position": Out(32),  # (batch, oh, ow) encoded

            # Configuration
            "cfg_kernel_h": In(4),
            "cfg_kernel_w": In(4),
            "cfg_stride_h": In(4),
            "cfg_stride_w": In(4),
            "cfg_dilation_h": In(4),
            "cfg_dilation_w": In(4),

            # Status
            "window_valid": Out(1),  # K_w columns accumulated
        }
        super().__init__(ports)
```

---

## Channel-Parallel MAC Array

### Channel-Parallel MAC Array Purpose

Compute dot products between the input window and multiple filters in parallel, accumulating results across input channels.

### MAC Array Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     CHANNEL-PARALLEL MAC ARRAY                          │
│                                                                         │
│  Window Input (K_h × K_w values for current channel)                    │
│        │                                                                │
│        ▼                                                                │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    BROADCAST NETWORK                            │    │
│  │    Window values broadcast to all P_c MAC banks                 │    │
│  └───────────┬─────────────┬─────────────┬─────────────────────────┘    │
│              │             │             │                              │
│              ▼             ▼             ▼                              │
│  ┌───────────────┐ ┌───────────────┐ ┌───────────────┐                  │
│  │  MAC Bank 0   │ │  MAC Bank 1   │ │  MAC Bank P-1 │                  │
│  │               │ │               │ │               │                  │
│  │  Filter 0     │ │  Filter 1     │ │  Filter P-1   │                  │
│  │  ┌─────────┐  │ │  ┌─────────┐  │ │  ┌─────────┐  │                  │
│  │  │K×K coef │  │ │  │K×K coef │  │ │  │K×K coef │  │                  │
│  │  └────┬────┘  │ │  └────┬────┘  │ │  └────┬────┘  │                  │
│  │       │       │ │       │       │ │       │       │                  │
│  │       ▼       │ │       ▼       │ │       ▼       │                  │
│  │  ┌─────────┐  │ │  ┌─────────┐  │ │  ┌─────────┐  │                  │
│  │  │ Σ(w×x)  │  │ │  │ Σ(w×x)  │  │ │  │ Σ(w×x)  │  │                  │
│  │  │ K×K MACs│  │ │  │ K×K MACs│  │ │  │ K×K MACs│  │                  │
│  │  └────┬────┘  │ │  └────┬────┘  │ │  └────┬────┘  │                  │
│  │       │       │ │       │       │ │       │       │                  │
│  │       ▼       │ │       ▼       │ │       ▼       │                  │
│  │  ┌─────────┐  │ │  ┌─────────┐  │ │  ┌─────────┐  │                  │
│  │  │ Accum   │  │ │  │ Accum   │  │ │  │ Accum   │  │                  │
│  │  │ (32-bit)│  │ │  │ (32-bit)│  │ │  │ (32-bit)│  │                  │
│  │  └────┬────┘  │ │  └────┬────┘  │ │  └────┬────┘  │                  │
│  └───────┼───────┘ └───────┼───────┘ └───────┼───────┘                  │
│          │                 │                 │                          │
│          └─────────────────┼─────────────────┘                          │
│                            │                                            │
│                            ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    OUTPUT COLLECTOR                             │    │
│  │            P_c partial sums per output position                 │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### MAC Bank Detail

Each MAC bank processes one output channel:

```
┌──────────────────────────────────────────────────────────────┐
│                      MAC BANK (one output channel)           │
│                                                              │
│  Window Input: [w00, w01, w02, w10, w11, w12, w20, w21, w22] │
│  Filter Coeff: [f00, f01, f02, f10, f11, f12, f20, f21, f22] │
│                                                              │
│  ┌───────────────────────────────────────────────────────┐   │
│  │              MULTIPLIER ARRAY (K×K = 9 for 3×3)       │   │
│  │                                                       │   │
│  │   w00×f00  w01×f01  w02×f02                           │   │
│  │   w10×f10  w11×f11  w12×f12                           │   │
│  │   w20×f20  w21×f21  w22×f22                           │   │
│  │      │        │        │                              │   │
│  │      └────────┼────────┘                              │   │
│  │               │                                       │   │
│  │               ▼                                       │   │
│  │   ┌────────────────────────────────────────────────┐  │   │
│  │   │              ADDER TREE                        │  │   │
│  │   │                                                │  │   │
│  │   │    Level 1: 5 adders (9→5)                     │  │   │
│  │   │    Level 2: 3 adders (5→3)                     │  │   │
│  │   │    Level 3: 2 adders (3→2)                     │  │   │
│  │   │    Level 4: 1 adder  (2→1)                     │  │   │
│  │   │                                                │  │   │
│  │   └──────────────────────┬─────────────────────────┘  │   │
│  │                          │                            │   │
│  │                          ▼                            │   │
│  │   ┌────────────────────────────────────────────────┐  │   │
│  │   │           ACCUMULATOR (32-bit)                 │  │   │
│  │   │                                                │  │   │
│  │   │   acc_new = acc_old + adder_tree_out           │  │   │
│  │   │   (accumulate across input channels)           │  │   │
│  │   │                                                │  │   │
│  │   └──────────────────────┬─────────────────────────┘  │   │
│  │                          │                            │   │
│  └──────────────────────────┼────────────────────────────┘   │
│                             │                                │
│                             ▼                                │
│                      Partial Sum Output                      │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### MAC Array Specifications

| Parameter | Value | Notes |
|-----------|-------|-------|
| Parallel output channels | P_c (16-64) | Configurable |
| Multipliers per bank | K_h × K_w (max 49) | 7×7 worst case |
| Multiplier width | 8×8 → 16 bit | INT8 × INT8 |
| Adder tree depth | ceil(log2(K×K)) | 4 levels for 3×3 |
| Accumulator width | 32 bits | Prevent overflow |
| Total multipliers | P_c × K_h × K_w | 16 × 9 = 144 for 3×3 |

### Resource Scaling

| Config | Multipliers | Adders | Accumulators |
|--------|-------------|--------|--------------|
| P_c=16, 3×3 | 144 | ~128 | 16 |
| P_c=32, 3×3 | 288 | ~256 | 32 |
| P_c=64, 3×3 | 576 | ~512 | 64 |
| P_c=16, 5×5 | 400 | ~384 | 16 |
| P_c=16, 7×7 | 784 | ~768 | 16 |

### MAC Array Interface Signals

```python
class ChannelParallelMAC(Component):
    """Channel-parallel MAC array for stencil machine."""

    def __init__(self, config: StencilConfig):
        ports = {
            # Window input (broadcast to all banks)
            "in_window_valid": In(1),
            "in_window": In(config.input_bits * config.max_kernel_h * config.max_kernel_w),
            "in_channel": In(16),      # Current input channel index
            "in_last_channel": In(1),  # Last channel flag

            # Filter coefficients (loaded once per output tile)
            "filter_load": In(1),
            "filter_data": In(config.weight_bits * config.max_kernel_h * config.max_kernel_w),
            "filter_bank": In(8),      # Which MAC bank to load

            # Partial sum output
            "out_valid": Out(1),
            "out_ready": In(1),
            "out_data": Out(config.acc_bits * config.parallel_channels),
            "out_position": Out(32),   # Output position (batch, oh, ow)

            # Control
            "clear_accum": In(1),      # Reset accumulators
            "cfg_kernel_h": In(4),
            "cfg_kernel_w": In(4),
            "cfg_parallel_channels": In(8),
        }
        super().__init__(ports)
```

---

## Filter Buffer

### Filter Buffer Purpose

Cache filter coefficients locally to avoid repeated DRAM access during convolution.

### Filter Buffer Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      FILTER BUFFER                          │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                   FILTER SRAM                       │    │
│  │                                                     │    │
│  │   Capacity: P_c × K_h × K_w × C_in × 8 bits         │    │
│  │                                                     │    │
│  │   Organization:                                     │    │
│  │   ┌─────────────────────────────────────────────┐   │    │
│  │   │ Filter 0: [K_h × K_w × C_in coefficients]   │   │    │
│  │   ├─────────────────────────────────────────────┤   │    │
│  │   │ Filter 1: [K_h × K_w × C_in coefficients]   │   │    │
│  │   ├─────────────────────────────────────────────┤   │    │
│  │   │ ...                                         │   │    │
│  │   ├─────────────────────────────────────────────┤   │    │
│  │   │ Filter P-1: [K_h × K_w × C_in coefficients] │   │    │
│  │   └─────────────────────────────────────────────┘   │    │
│  │                                                     │    │
│  └──────────────────────────┬──────────────────────────┘    │
│                             │                               │
│                             ▼                               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              READ PORT ARRAY (P_c ports)            │    │
│  │                                                     │    │
│  │   Each cycle: Read K_h × K_w coefficients for       │    │
│  │   current input channel from all P_c filters        │    │
│  │                                                     │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Filter Buffer Sizing

| Config | Filter Buffer Size |
|--------|-------------------|
| P_c=16, 3×3, C_in=64 | 16 × 9 × 64 = 9 KB |
| P_c=32, 3×3, C_in=128 | 32 × 9 × 128 = 36 KB |
| P_c=64, 3×3, C_in=256 | 64 × 9 × 256 = 144 KB |

### Double Buffering

For large C_out, filters are loaded in tiles of P_c. Double buffering hides load latency:

```
Bank A: Computing with filters [0, P_c)
Bank B: Loading filters [P_c, 2×P_c)
        ↓ (swap when done)
Bank A: Loading filters [2×P_c, 3×P_c)
Bank B: Computing with filters [P_c, 2×P_c)
```

---

## Output Accumulator

### Output Accumulator Purpose

Accumulate partial sums across input channels and apply post-processing (activation, quantization).

### Output Accumulator Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                       OUTPUT ACCUMULATOR                            │
│                                                                     │
│  From MAC Array (P_c partial sums)                                  │
│        │                                                            │
│        ▼                                                            │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                   ACCUMULATOR BANKS                         │    │
│  │                                                             │    │
│  │   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │    │
│  │   │ Acc[0]   │ │ Acc[1]   │ │   ...    │ │ Acc[P-1] │       │    │
│  │   │ 32-bit   │ │ 32-bit   │ │          │ │ 32-bit   │       │    │
│  │   └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘       │    │
│  │        │            │            │            │             │    │
│  └────────┼────────────┼────────────┼────────────┼─────────────┘    │
│           │            │            │            │                  │
│           ▼            ▼            ▼            ▼                  │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    BIAS ADDITION                            │    │
│  │        acc[i] + bias[oc_base + i]                           │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                            │                                        │
│                            ▼                                        │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    ACTIVATION FUNCTION                      │    │
│  │                                                             │    │
│  │   ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐        │    │
│  │   │  None   │  │  ReLU   │  │ ReLU6   │  │  Clip   │        │    │
│  │   │ (pass)  │  │ max(0,x)│  │min(6,..)│  │ [a,b]   │        │    │
│  │   └─────────┘  └─────────┘  └─────────┘  └─────────┘        │    │
│  │                                                             │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                            │                                        │
│                            ▼                                        │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    QUANTIZATION (optional)                  │    │
│  │                                                             │    │
│  │   output = (acc × scale + zero_point) >> shift              │    │
│  │                                                             │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                            │                                        │
│                            ▼                                        │
│                    To Output Stream (DRAM)                          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Stencil Controller

### State Machine

```
┌─────────────────────────────────────────────────────────────────────┐
│                      STENCIL CONTROLLER FSM                          │
│                                                                      │
│                          ┌──────────┐                               │
│                          │   IDLE   │◄─────────────────┐            │
│                          └────┬─────┘                  │            │
│                               │ START                  │            │
│                               ▼                        │            │
│                          ┌──────────┐                  │            │
│                          │  CONFIG  │                  │            │
│                          └────┬─────┘                  │            │
│                               │                        │            │
│                               ▼                        │            │
│                          ┌──────────┐                  │            │
│                          │ LOAD_F   │ Load filter tile │            │
│                          └────┬─────┘                  │            │
│                               │                        │            │
│                               ▼                        │            │
│          ┌──────────────────────────────────────┐     │            │
│          │          ROW PROCESSING LOOP          │     │            │
│          │                                       │     │            │
│          │    ┌──────────┐                      │     │            │
│          │    │ FILL_BUF │ Fill line buffers    │     │            │
│          │    └────┬─────┘ (first K_h-1 rows)   │     │            │
│          │         │                             │     │            │
│          │         ▼                             │     │            │
│          │    ┌──────────┐                      │     │            │
│          │    │ COMPUTE  │◄────────┐            │     │            │
│          │    └────┬─────┘         │            │     │            │
│          │         │               │            │     │            │
│          │         ▼               │            │     │            │
│          │    ┌──────────┐         │            │     │            │
│          │    │ NEXT_COL │─────────┘            │     │            │
│          │    └────┬─────┘  more cols           │     │            │
│          │         │                             │     │            │
│          │         │ row done                    │     │            │
│          │         ▼                             │     │            │
│          │    ┌──────────┐                      │     │            │
│          │    │ NEXT_ROW │──────────────────────┼─────┘            │
│          │    └────┬─────┘  more rows           │     all rows     │
│          │         │                             │                  │
│          └─────────┼─────────────────────────────┘                  │
│                    │                                                │
│                    │ all output channels done                       │
│                    ▼                                                │
│               ┌──────────┐                                          │
│               │NEXT_OC_TL│ Next output channel tile                │
│               └────┬─────┘                                          │
│                    │                                                │
│                    │ all tiles done                                 │
│                    ▼                                                │
│               ┌──────────┐                                          │
│               │   DONE   │──────────────────────────────────────────┘
│               └──────────┘                                          │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Nested Loop Structure

```python
# Pseudo-code for stencil controller operation
for oc_tile in range(0, C_out, P_c):           # Output channel tiles
    load_filters(oc_tile, oc_tile + P_c)       # Load P_c filters

    for batch in range(N):                      # Batch dimension
        for row in range(H_out):                # Output rows
            # Ensure line buffers have data
            while not line_buffers_ready():
                fill_line_buffer_row()

            for col in range(W_out):            # Output columns
                window = extract_window(row, col)

                for ic in range(C_in):          # Input channels
                    filter_slice = get_filter_slice(ic)
                    partial_sum = mac_compute(window[ic], filter_slice)
                    accumulate(partial_sum)

                output = apply_activation(accumulator)
                write_output(batch, row, col, oc_tile:oc_tile+P_c, output)
                clear_accumulator()
```

---

## Memory Interfaces

### Input Stream Interface

```
┌───────────────────────────────────────────────────────────────┐
│                    INPUT STREAM INTERFACE                      │
│                                                                │
│  AXI4-Stream compatible:                                       │
│                                                                │
│  Signal        Width    Direction    Description               │
│  ─────────────────────────────────────────────────────────────│
│  s_axis_tvalid   1       in         Data valid                 │
│  s_axis_tready   1       out        Ready to accept            │
│  s_axis_tdata   128      in         Data (configurable width)  │
│  s_axis_tlast    1       in         End of row/frame           │
│  s_axis_tuser    8       in         Metadata (row/frame flags) │
│                                                                │
└───────────────────────────────────────────────────────────────┘
```

### Output Stream Interface

```
┌───────────────────────────────────────────────────────────────┐
│                   OUTPUT STREAM INTERFACE                      │
│                                                                │
│  AXI4-Stream compatible:                                       │
│                                                                │
│  Signal        Width    Direction    Description               │
│  ─────────────────────────────────────────────────────────────│
│  m_axis_tvalid   1       out        Data valid                 │
│  m_axis_tready   1       in         Downstream ready           │
│  m_axis_tdata   128      out        Data (P_c outputs packed)  │
│  m_axis_tlast    1       out        End of row/frame           │
│  m_axis_tuser    8       out        Metadata                   │
│                                                                │
└───────────────────────────────────────────────────────────────┘
```

### Filter Load Interface

```
┌───────────────────────────────────────────────────────────────┐
│                   FILTER LOAD INTERFACE                        │
│                                                                │
│  DMA-style burst interface:                                    │
│                                                                │
│  Signal        Width    Direction    Description               │
│  ─────────────────────────────────────────────────────────────│
│  flt_req        1       out        Request filter load         │
│  flt_addr      64       out        DRAM address                │
│  flt_len       16       out        Burst length                │
│  flt_valid      1       in         Data valid                  │
│  flt_ready      1       out        Ready for data              │
│  flt_data     256       in         Filter data burst           │
│  flt_done       1       in         Transfer complete           │
│                                                                │
└───────────────────────────────────────────────────────────────┘
```

---

## Configuration Registers

### Register Map

| Address | Name | Width | Description |
|---------|------|-------|-------------|
| 0x00 | CTRL | 32 | Control register (start, stop, reset) |
| 0x04 | STATUS | 32 | Status register (busy, done, error) |
| 0x08 | IN_DIMS | 32 | Input dimensions: [H:16, W:16] |
| 0x0C | IN_CHANNELS | 16 | Input channel count |
| 0x10 | OUT_DIMS | 32 | Output dimensions: [H:16, W:16] |
| 0x14 | OUT_CHANNELS | 16 | Output channel count |
| 0x18 | KERNEL_SIZE | 16 | Kernel: [H:8, W:8] |
| 0x1C | STRIDE | 16 | Stride: [H:8, W:8] |
| 0x20 | PADDING | 16 | Padding: [H:8, W:8] |
| 0x24 | DILATION | 16 | Dilation: [H:8, W:8] |
| 0x28 | BATCH_SIZE | 8 | Batch count |
| 0x30 | IN_ADDR | 64 | Input tensor DRAM address |
| 0x38 | FILTER_ADDR | 64 | Filter tensor DRAM address |
| 0x40 | BIAS_ADDR | 64 | Bias vector DRAM address |
| 0x48 | OUT_ADDR | 64 | Output tensor DRAM address |
| 0x50 | ACTIVATION | 8 | Activation function select |
| 0x54 | QUANT_SCALE | 32 | Quantization scale factor |
| 0x58 | QUANT_ZERO | 32 | Quantization zero point |

---

## Timing Analysis

### Throughput Model

For a Conv2D layer with parameters:

- Input: [N, H_in, W_in, C_in]
- Output: [N, H_out, W_out, C_out]
- Kernel: [K_h, K_w]

**Cycles per output pixel** (channel-serial mode):

```
cycles_per_pixel = C_in × (K_h × K_w adder tree latency)
                 ≈ C_in × ceil(log2(K_h × K_w))
                 ≈ C_in × 4  (for 3×3)
```

**Total cycles** (excluding filter load):

```
total_cycles = N × H_out × W_out × C_out / P_c × cycles_per_pixel
```

**Example**: Conv2D 56×56×256 → 56×56×256, 3×3 kernel, P_c=32

```
cycles_per_pixel = 256 × 4 = 1024
total_cycles = 1 × 56 × 56 × 256 / 32 × 1024 = 25.7M cycles
At 500 MHz: 51.4 ms
```

### Comparison with im2col + Systolic

| Metric | Stencil Machine | im2col + Systolic |
|--------|-----------------|-------------------|
| Input DRAM reads | 1× | 9× (for 3×3) |
| Filter DRAM reads | 1× | 1× |
| Output DRAM writes | 1× | 1× |
| Memory bandwidth | **Low** | High |
| Compute cycles | Similar | Similar |
| Energy (memory) | **~0.3×** | 1× |

---

## Area and Power Estimates

### Component Area Breakdown

| Component | Area (mm² @ 7nm) | Notes |
|-----------|------------------|-------|
| Line Buffers (43 KB) | 0.05 | SRAM macros |
| Window Former | 0.01 | Shift registers |
| MAC Array (P_c=32, 3×3) | 0.15 | 288 multipliers, adder trees |
| Filter Buffer (36 KB) | 0.04 | SRAM macros |
| Output Accumulators | 0.02 | 32 × 32-bit registers |
| Controller | 0.01 | FSM, address gen |
| **Total** | **~0.28** | |

### Power Breakdown

| Component | Power (mW @ 500MHz) | Notes |
|-----------|---------------------|-------|
| Line Buffers | 15 | SRAM read/write |
| MAC Array | 80 | 288 MACs active |
| Filter Buffer | 10 | SRAM read |
| Controller | 5 | Logic |
| **Total** | **~110 mW** | |

### Efficiency

```
Peak throughput: P_c × (K_h × K_w) MACs/cycle
               = 32 × 9 = 288 MACs/cycle
               = 288 × 500M = 144 GOPS (INT8)

Power efficiency: 144 GOPS / 110 mW = 1.3 TOPS/W

With typical 70% utilization: ~0.9 TOPS/W effective
```

---

## Implementation Notes

### Amaranth HDL Structure

```
src/systars/stencil/
├── __init__.py
├── config.py              # StencilConfig dataclass
├── line_buffer.py         # LineBufferUnit component
├── window_former.py       # WindowFormer component
├── mac_array.py           # ChannelParallelMAC component
├── filter_buffer.py       # FilterBuffer component
├── output_accum.py        # OutputAccumulator component
├── controller.py          # StencilController FSM
└── stencil_machine.py     # Top-level integration
```

### Key Design Decisions

1. **Channel-serial processing**: Reduces register count, trades latency for area
2. **Configurable P_c**: Allow area/throughput tradeoff per deployment
3. **AXI-Stream interfaces**: Standard, easy integration
4. **Double-buffered filters**: Hide DRAM latency for large C_out
5. **Streaming datapath**: Minimize intermediate storage

### Future Enhancements

1. **Channel-parallel mode**: Process multiple C_in simultaneously for higher throughput
2. **Depthwise optimization**: Special path for groups=C_in
3. **Winograd support**: Reduced multiplications for 3×3 kernels
4. **Sparsity support**: Skip zero weights/activations

---

*Document version: 1.0*
*Last updated: 2025-12-19*
