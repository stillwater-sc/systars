# PolarFire SoC (Icicle Kit) Deployment Plan

This document outlines the steps required to deploy SYSTARS on a Microchip PolarFire SoC FPGA, specifically targeting the Icicle Kit development board.

---

## Current State Assessment

### What's Ready

| Component | Status | Notes |
|-----------|--------|-------|
| Core Systolic Array | Done | PE, PEArray, SystolicArray fully implemented |
| Memory System | Done | Scratchpad, Accumulator, SkewBuffer |
| Controllers | Done | Load, Execute, Store controllers |
| DMA Engines | Done | StreamReader, StreamWriter, DescriptorEngine |
| Top-Level Integration | Done | SystolicTop with AXI-like interfaces |
| Synthesizable Verilog | Done | Yosys synthesis passes for ICE40/ECP5 |

### What's Missing for FPGA Deployment

| Gap | Priority | Effort | Description |
|-----|----------|--------|-------------|
| Loop Unrollers | High | Medium | LoopMatmul and LoopConv FSMs (see loop-unrollers.md) |
| Timing Constraints | High | Medium | SDC files for PolarFire |
| Pin Assignments | High | Medium | PDC files for Icicle Kit |
| Libero Project | Medium | Low | Project setup and configuration |
| AXI4 Adapter | Medium | Medium | Bridge to PolarFire fabric interconnect |
| MSS Integration | Medium | High | Connection to RISC-V subsystem |
| Driver Software | Medium | Medium | Bare-metal driver for RISC-V |
| Clock Domain Crossing | Low | Medium | If using multiple clock domains |

---

## Target Platform: PolarFire SoC Icicle Kit

### Hardware Specifications

- **FPGA**: MPFS250T PolarFire SoC
- **Logic Elements**: ~254K
- **Math Blocks**: 784 (18x18 multipliers)
- **Block RAM**: 16Mb
- **LSRAM**: 7Mb
- **Microprocessor Subsystem (MSS)**:
  - 4x U54 RISC-V cores (RV64GC)
  - 1x E51 monitor core
  - L2 cache
  - DDR4 controller
- **Memory**: 2GB DDR4

### Fabric Interface Controllers (FIC)

PolarFire SoC provides 4 FIC interfaces to connect FPGA fabric to MSS:

| FIC | Bus Width | Use Case |
|-----|-----------|----------|
| FIC0 | 64-bit AXI4 | High-bandwidth DMA (recommended for SYSTARS) |
| FIC1 | 64-bit AXI4 | Secondary DMA or control |
| FIC2 | 32-bit APB | Low-bandwidth configuration registers |
| FIC3 | 64-bit AXI4 | Additional bandwidth if needed |

**Recommendation**: Use FIC0 for DMA data path, FIC2 for control registers.

---

## Phase 1: Standalone FPGA Validation

**Goal**: Verify synthesis and basic functionality on PolarFire fabric without MSS integration.

### 1.1 Create Libero Project

```
systars-polarfire/
├── constraints/
│   ├── timing.sdc           # Timing constraints
│   └── pinout.pdc           # Pin assignments
├── rtl/
│   └── (generated Verilog)
├── sim/
│   └── tb_systolic_array.v  # Testbench for simulation
├── scripts/
│   └── build.tcl            # Libero build script
└── systars_polarfire.prjx   # Libero project file
```

### 1.2 Timing Constraints (SDC)

```tcl
# constraints/timing.sdc

# Main clock from MSS (example: 100 MHz)
create_clock -name clk_mss -period 10.0 [get_ports clk]

# Internal clocks if any PLL is used
# create_generated_clock -name clk_sys -source [get_ports clk] \
#     -divide_by 1 [get_pins pll/clk_out]

# Input delays (adjust based on actual I/O timing)
set_input_delay -clock clk_mss -max 2.0 [get_ports {axi_*}]
set_input_delay -clock clk_mss -min 0.5 [get_ports {axi_*}]

# Output delays
set_output_delay -clock clk_mss -max 2.0 [get_ports {axi_*}]
set_output_delay -clock clk_mss -min 0.5 [get_ports {axi_*}]

# False paths for asynchronous resets
set_false_path -from [get_ports rst_n]

# Multicycle paths if needed for accumulator operations
# set_multicycle_path 2 -setup -from [get_registers acc_*] -to [get_registers acc_*]
```

### 1.3 Pin Assignments (PDC)

```tcl
# constraints/pinout.pdc
# Note: Actual pins depend on Icicle Kit schematic

# Clock and Reset (directly from MSS via fabric interface)
# These are typically internal signals, not external pins

# Debug LEDs (optional, for status indication)
set_io -port_name led[0] -pin_name H13 -fixed true -io_std LVCMOS33
set_io -port_name led[1] -pin_name J13 -fixed true -io_std LVCMOS33
set_io -port_name led[2] -pin_name K14 -fixed true -io_std LVCMOS33
set_io -port_name led[3] -pin_name L14 -fixed true -io_std LVCMOS33

# Debug UART (optional)
# set_io -port_name uart_tx -pin_name ... -fixed true -io_std LVCMOS33
# set_io -port_name uart_rx -pin_name ... -fixed true -io_std LVCMOS33
```

### 1.4 Resource Estimation

Based on a 16x16 systolic array with 8-bit inputs and 32-bit accumulators:

| Resource | Estimated Usage | MPFS250T Available | Utilization |
|----------|-----------------|-------------------|-------------|
| Logic Elements | ~50K | 254K | ~20% |
| Math Blocks | 256 (16x16 MACs) | 784 | ~33% |
| Block RAM | ~2Mb (scratchpad+acc) | 16Mb | ~12% |
| LSRAM | 0 | 7Mb | 0% |

**Note**: These are rough estimates. Actual utilization depends on:

- Tile configuration (grid_rows, grid_cols, tile_rows, tile_cols)
- Memory bank count and depth
- DMA engine configuration

### 1.5 Synthesis Checklist

- [ ] Generate Verilog with `just gen`
- [ ] Create Libero project
- [ ] Import RTL files
- [ ] Add timing constraints
- [ ] Run synthesis
- [ ] Check resource utilization report
- [ ] Run place-and-route
- [ ] Check timing report (all paths meeting timing?)
- [ ] Generate bitstream

---

## Phase 2: MSS Integration

**Goal**: Connect SYSTARS fabric logic to PolarFire MSS for RISC-V control.

### 2.1 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        PolarFire SoC                            │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                  MSS (RISC-V Cores)                       │  │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐       │  │
│  │  │   U54   │  │   U54   │  │   U54   │  │   U54   │       │  │
│  │  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘       │  │
│  │       └──────────┬─┴──────────┬─┴────────────┘            │  │
│  │                  │            │                            │  │
│  │              ┌───┴───┐    ┌───┴───┐                        │  │
│  │              │  L2   │    │  DDR  │                        │  │
│  │              │ Cache │    │ Ctrl  │                        │  │
│  │              └───┬───┘    └───┬───┘                        │  │
│  │                  │            │                            │  │
│  └──────────────────┼────────────┼────────────────────────────┘  │
│                     │            │                               │
│                 ┌───┴───┐    ┌───┴───┐                          │
│                 │ FIC0  │    │ FIC2  │                          │
│                 │ AXI4  │    │ APB   │                          │
│                 └───┬───┘    └───┬───┘                          │
│                     │            │                               │
│  ┌──────────────────┼────────────┼────────────────────────────┐  │
│  │               FPGA Fabric     │                            │  │
│  │                  │            │                            │  │
│  │              ┌───┴───┐    ┌───┴───┐                        │  │
│  │              │ AXI4  │    │ APB   │                        │  │
│  │              │Adapter│    │Regs   │                        │  │
│  │              └───┬───┘    └───┬───┘                        │  │
│  │                  │            │                            │  │
│  │              ┌───┴────────────┴───┐                        │  │
│  │              │    SystolicTop     │                        │  │
│  │              │  ┌──────────────┐  │                        │  │
│  │              │  │ SystolicArray│  │                        │  │
│  │              │  └──────────────┘  │                        │  │
│  │              │  ┌──────┐┌──────┐  │                        │  │
│  │              │  │ SPAD ││ ACC  │  │                        │  │
│  │              │  └──────┘└──────┘  │                        │  │
│  │              └────────────────────┘                        │  │
│  └────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 AXI4 Adapter

The current SystolicTop uses AXI-like signals. A thin adapter is needed:

```python
# Pseudo-code for AXI4 adapter requirements

class AXI4Adapter(Component):
    """
    Bridge between PolarFire FIC (AXI4) and SystolicTop interface.

    FIC0 AXI4 Interface:
    - 64-bit data width
    - Supports AXI4 burst transactions
    - ID width configurable

    SystolicTop Interface:
    - Custom streaming interface
    - Needs conversion to/from AXI4
    """

    # AXI4 Slave interface (from MSS)
    # - Write address channel (AW)
    # - Write data channel (W)
    # - Write response channel (B)
    # - Read address channel (AR)
    # - Read data channel (R)

    # SystolicTop interface
    # - DMA read/write streams
    # - Command interface
```

### 2.3 Memory Map

```
0x6000_0000 - 0x6000_FFFF : Control Registers (via FIC2/APB)
    0x6000_0000 : STATUS      (RO)  - Accelerator status
    0x6000_0004 : CONTROL     (RW)  - Start/stop/reset
    0x6000_0008 : CMD_QUEUE   (WO)  - Command submission
    0x6000_000C : IRQ_STATUS  (RW1C) - Interrupt status
    0x6000_0010 : IRQ_ENABLE  (RW)  - Interrupt enable
    0x6000_0100 : DESC_BASE   (RW)  - Descriptor base address
    0x6000_0104 : DESC_COUNT  (RW)  - Number of descriptors

0x8000_0000 - 0x8FFF_FFFF : DMA Window (via FIC0/AXI4)
    Maps to DDR4 physical memory for DMA operations
```

### 2.4 Integration Steps

1. **MSS Configurator**
   - Enable FIC0 (AXI4) for DMA
   - Enable FIC2 (APB) for control registers
   - Configure interrupts (fabric to MSS)
   - Set memory protection units

2. **SmartDesign**
   - Import SystolicTop Verilog
   - Add AXI4 adapter
   - Add APB register block
   - Connect to FIC0/FIC2
   - Add clock/reset conditioning

3. **Verification**
   - RTL simulation with AXI4 BFM
   - Check memory transactions
   - Verify interrupt generation

---

## Phase 3: Software Stack

**Goal**: Enable RISC-V cores to program and control the accelerator.

### 3.1 Bare-Metal Driver

```c
// systars_driver.h

#ifndef SYSTARS_DRIVER_H
#define SYSTARS_DRIVER_H

#include <stdint.h>

// Register offsets
#define SYSTARS_STATUS      0x00
#define SYSTARS_CONTROL     0x04
#define SYSTARS_CMD_QUEUE   0x08
#define SYSTARS_IRQ_STATUS  0x0C
#define SYSTARS_IRQ_ENABLE  0x10
#define SYSTARS_DESC_BASE   0x100
#define SYSTARS_DESC_COUNT  0x104

// Status bits
#define SYSTARS_STATUS_BUSY     (1 << 0)
#define SYSTARS_STATUS_DONE     (1 << 1)
#define SYSTARS_STATUS_ERROR    (1 << 2)

// Control bits
#define SYSTARS_CTRL_START      (1 << 0)
#define SYSTARS_CTRL_RESET      (1 << 1)

// Initialize accelerator
int systars_init(uintptr_t base_addr);

// Submit descriptor chain for execution
int systars_submit(uint64_t desc_phys_addr, uint32_t desc_count);

// Wait for completion
int systars_wait(uint32_t timeout_ms);

// Perform matrix multiply: C = A @ B + D
int systars_matmul(
    const int8_t* A, uint32_t A_rows, uint32_t A_cols,
    const int8_t* B, uint32_t B_cols,
    const int32_t* D,  // bias, may be NULL
    int32_t* C
);

#endif
```

### 3.2 Header File Generation

Generate hardware parameters for software:

```c
// systars_params.h (auto-generated)

#pragma once

#define SYSTARS_DIM           16
#define SYSTARS_INPUT_BITS    8
#define SYSTARS_ACC_BITS      32
#define SYSTARS_SP_KB         256
#define SYSTARS_ACC_KB        64
#define SYSTARS_SP_BANKS      4
#define SYSTARS_ACC_BANKS     2
#define SYSTARS_DMA_WIDTH     128

typedef int8_t  systars_input_t;
typedef int32_t systars_acc_t;
```

### 3.3 Test Application

```c
// test_matmul.c

#include "systars_driver.h"
#include "systars_params.h"

int main() {
    // Initialize accelerator
    systars_init(0x60000000);

    // Allocate matrices (aligned for DMA)
    int8_t A[16][16] __attribute__((aligned(64)));
    int8_t B[16][16] __attribute__((aligned(64)));
    int32_t C[16][16] __attribute__((aligned(64)));

    // Fill with test data
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j++) {
            A[i][j] = i + 1;
            B[i][j] = j + 1;
        }
    }

    // Execute matmul
    systars_matmul(&A[0][0], 16, 16, &B[0][0], 16, NULL, &C[0][0]);

    // Verify results
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j++) {
            int32_t expected = 0;
            for (int k = 0; k < 16; k++) {
                expected += A[i][k] * B[k][j];
            }
            if (C[i][j] != expected) {
                printf("Mismatch at [%d][%d]: got %d, expected %d\n",
                       i, j, C[i][j], expected);
                return 1;
            }
        }
    }

    printf("Test passed!\n");
    return 0;
}
```

---

## Phase 4: Full System Validation

### 4.1 Performance Benchmarks

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| Clock Frequency | 100+ MHz | Timing report |
| Throughput | 25.6 GOPS @ 100MHz | Matrix multiply benchmark |
| DMA Bandwidth | 800 MB/s | DMA throughput test |
| Latency (16x16 matmul) | <10us | Cycle counter |

### 4.2 Test Suite

1. **Unit Tests**
   - Single PE operation
   - Row/column data flow
   - Memory bank access

2. **Integration Tests**
   - Small matmul (4x4)
   - Medium matmul (16x16)
   - Large tiled matmul (64x64)

3. **Stress Tests**
   - Continuous operation
   - DMA stress
   - Interrupt handling

### 4.3 Known Limitations

- No TLB/virtual memory support (use physical addresses)
- Single-threaded command submission (no concurrent queues)
- Fixed precision (INT8 input, INT32 accumulator)

---

## Timeline Dependencies

```
Loop Unrollers ────────────────┐
                               ├──→ Full RTL Complete
RTL Verification ──────────────┘
                                        │
                                        ▼
                               Libero Project Setup
                                        │
                                        ▼
                               Synthesis & P&R
                                        │
                                        ▼
                               Resource/Timing Analysis
                                        │
                                        ▼
                               MSS Integration
                                        │
                                        ▼
                               Driver Development
                                        │
                                        ▼
                               Hardware Validation
```

---

## References

- [PolarFire SoC Icicle Kit User Guide](https://www.microsemi.com/product-directory/soc-fpgas/5498-polarfire-soc-icicle-kit)
- [Libero SoC Design Suite](https://www.microchip.com/en-us/products/fpgas-and-plds/fpga-and-soc-design-tools/fpga/libero-software-later-versions)
- [PolarFire SoC MSS Technical Reference Manual](https://www.microsemi.com/document-portal/doc_download/1244570-polarfire-soc-fpga-mss-technical-reference-manual)
