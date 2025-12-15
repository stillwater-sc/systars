# SYSTARS Python RTL Rewrite Plan

This document provides a comprehensive plan for rewriting the SYSTARS systolic array layout generator from C to a modern Python-based RTL framework.

---

## Executive Summary

**Goal:** Create a Python-based systolic array generator that replicates SYSTARS's functionality using modern Python RTL tools.

**Recommended Framework:** **Amaranth HDL** (formerly nMigen)

- Pros: Active development, good documentation, Python 3.8+, native simulation, formal verification support
- Alternative: PyRTL (simpler but less feature-rich)

**Target Compatibility:** Generate Verilog that is functionally equivalent to SYSTARS's output, adding a system integration workflow.

---

## Architecture Overview (What We're Building)

### Core Components to Implement

```
┌────────────────────────────────────────────────────────────────┐
│                         Top Level                              │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                 Accelerator Interface                    │  │
│  │      (Custom instruction decode & command dispatch)      │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              │                                 │
│              ┌───────────────┼───────────────┐                 │
│              ▼               ▼               ▼                 │
│  ┌────────────────┐ ┌─────────────────┐ ┌───────────────┐      │
│  │ LoadController │ │ExecuteController│ │StoreController│      │
│  │    (DMA In)    │ │    (Compute)    │ │   (DMA Out)   │      │
│  └───────┬────────┘ └───────┬─────────┘ └───────┬───────┘      │
│          │                  │                   │              │
│          ▼                  ▼                   ▼              │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    Scratchpad Memory                    │   │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐        │   │
│  │  │ Bank 0  │ │ Bank 1  │ │ Bank 2  │ │ Bank 3  │        │   │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘        │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                 │
│                              ▼                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    Systolic Array                       │   │
│  │  ┌─────────────────────────────────────────────────┐    │   │
│  │  │              MeshWithDelays                     │    │   │
│  │  │  ┌───────────────────────────────────────────┐  │    │   │
│  │  │  │                 Mesh                      │  │    │   │
│  │  │  │  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐          │  │    │   │
│  │  │  │  │Tile │→│Tile │→│Tile │→│Tile │ ...      │  │    │   │
│  │  │  │  └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘          │  │    │   │
│  │  │  │     ↓       ↓       ↓       ↓             │  │    │   │
│  │  │  │  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐          │  │    │   │
│  │  │  │  │Tile │→│Tile │→│Tile │→│Tile │ ...      │  │    │   │
│  │  │  │  └─────┘ └─────┘ └─────┘ └─────┘          │  │    │   │
│  │  │  └───────────────────────────────────────────┘  │    │   │
│  │  │  + Transposer + Input Buffers + Tag Tracking    │    │   │
│  │  └─────────────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                 │
│                              ▼                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   Accumulator Memory                    │   │
│  │  ┌──────────────┐ ┌──────────────┐                      │   │
│  │  │   Bank 0     │ │   Bank 1     │                      │   │
│  │  │ + Adders     │ │ + Adders     │                      │   │
│  │  │ + Scale      │ │ + Scale      │                      │   │
│  │  │ + Activation │ │ + Activation │                      │   │
│  │  └──────────────┘ └──────────────┘                      │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                 │
│                              ▼                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                      DMA Engine                         │   │
│  │  ┌─────────────────┐    ┌─────────────────┐             │   │
│  │  │  StreamReader   │    │  StreamWriter   │             │   │
│  │  │  (DRAM→Local)   │    │  (Local→DRAM)   │             │   │
│  │  └─────────────────┘    └─────────────────┘             │   │
│  │              │                    │                     │   │
│  │              └────────┬───────────┘                     │   │
│  │                       ▼                                 │   │
│  │                 ┌──────────┐                            │   │
│  │                 │   TLB    │                            │   │
│  │                 └──────────┘                            │   │
│  └─────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────┘
```

### Key Parameters (Configuration System)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mesh_rows` | 16 | Systolic array height |
| `mesh_cols` | 16 | Systolic array width |
| `tile_rows` | 1 | PEs per tile (height) |
| `tile_cols` | 1 | PEs per tile (width) |
| `input_bits` | 8 | Input/weight element width |
| `acc_bits` | 32 | Accumulator element width |
| `dataflow` | BOTH | WS, OS, or BOTH |
| `sp_capacity_kb` | 256 | Scratchpad size |
| `sp_banks` | 4 | Scratchpad banks |
| `acc_capacity_kb` | 64 | Accumulator size |
| `acc_banks` | 2 | Accumulator banks |
| `dma_buswidth` | 128 | DMA bus width (bits) |

---

## Phase 1: Foundation & PE Array (Weeks 1-2)

### 1.1 Repository Setup

```
pygemmini/
├── pygemmini/
│   ├── __init__.py
│   ├── config.py              # Configuration dataclass
│   ├── arithmetic/
│   │   ├── __init__.py
│   │   ├── base.py            # Arithmetic protocol/ABC
│   │   ├── integer.py         # SInt/UInt operations
│   │   └── floating.py        # Float operations (optional)
│   ├── core/
│   │   ├── __init__.py
│   │   ├── pe.py              # Processing Element
│   │   ├── tile.py            # Tile (grid of PEs)
│   │   ├── mesh.py            # Mesh (grid of Tiles)
│   │   └── mesh_with_delays.py # Full mesh with I/O buffering
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── local_addr.py      # Address encoding/decoding
│   │   ├── scratchpad.py      # Scratchpad banks
│   │   └── accumulator.py     # Accumulator with scale/activation
│   ├── controller/
│   │   ├── __init__.py
│   │   ├── load.py            # LoadController
│   │   ├── execute.py         # ExecuteController
│   │   └── store.py           # StoreController
│   ├── dma/
│   │   ├── __init__.py
│   │   ├── reader.py          # StreamReader
│   │   ├── writer.py          # StreamWriter
│   │   └── tracker.py         # DMACommandTracker
│   ├── util/
│   │   ├── __init__.py
│   │   ├── transposer.py      # Matrix transposer
│   │   ├── reservation.py     # ReservationStation/ROB
│   │   └── pipeline.py        # Pipeline utilities
│   ├── loop/
│   │   ├── __init__.py
│   │   ├── matmul.py          # LoopMatmul FSM
│   │   └── conv.py            # LoopConv FSM
│   └── top.py                 # Top-level SYSTARS module
├── tests/
│   ├── test_pe.py
│   ├── test_tile.py
│   ├── test_mesh.py
│   ├── test_scratchpad.py
│   ├── test_accumulator.py
│   ├── test_transposer.py
│   └── integration/
│       ├── test_matmul.py
│       └── test_conv.py
├── examples/
│   ├── simple_matmul.py
│   └── generate_verilog.py
├── scripts/
│   ├── generate_header.py     # Generate gemmini_params.h
│   └── compare_with_chisel.py # Validation against reference
├── pyproject.toml
├── README.md
└── docs/
    ├── architecture.md
    ├── isa.md
    └── configuration.md
```

### 1.2 Configuration System

```python
# pygemmini/config.py
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Callable

class Dataflow(Enum):
    OS = auto()   # Output-Stationary
    WS = auto()   # Weight-Stationary
    BOTH = auto() # Runtime selectable

@dataclass
class SYSTARSConfig:
    # Systolic array dimensions
    mesh_rows: int = 16
    mesh_cols: int = 16
    tile_rows: int = 1
    tile_cols: int = 1

    # Data types (bit widths)
    input_bits: int = 8
    weight_bits: int = 8
    acc_bits: int = 32
    output_bits: int = 20  # PE output before accumulation

    # Dataflow
    dataflow: Dataflow = Dataflow.BOTH

    # Scratchpad configuration
    sp_capacity_kb: int = 256
    sp_banks: int = 4
    sp_singleported: bool = True
    spad_read_delay: int = 4

    # Accumulator configuration
    acc_capacity_kb: int = 64
    acc_banks: int = 2
    acc_singleported: bool = False
    acc_latency: int = 2

    # DMA configuration
    dma_maxbytes: int = 64
    dma_buswidth: int = 128
    max_in_flight_reqs: int = 16

    # Queue depths
    ld_queue_length: int = 8
    st_queue_length: int = 2
    ex_queue_length: int = 8

    # Reservation station entries
    rs_entries_ld: int = 8
    rs_entries_st: int = 4
    rs_entries_ex: int = 16

    # Features
    has_training_convs: bool = True
    has_max_pool: bool = True
    has_nonlinear_activations: bool = True
    has_normalizations: bool = False

    # Pipeline tuning
    tile_latency: int = 0
    mesh_output_delay: int = 1

    # Computed properties
    @property
    def dim(self) -> int:
        """Systolic array dimension (assumes square)"""
        return self.mesh_rows * self.tile_rows

    @property
    def sp_width(self) -> int:
        """Scratchpad row width in bits"""
        return self.mesh_cols * self.tile_cols * self.input_bits

    @property
    def sp_bank_entries(self) -> int:
        """Rows per scratchpad bank"""
        return (self.sp_capacity_kb * 1024 * 8) // (self.sp_banks * self.sp_width)

    @property
    def acc_bank_entries(self) -> int:
        """Rows per accumulator bank"""
        acc_row_bits = self.mesh_cols * self.tile_cols * self.acc_bits
        return (self.acc_capacity_kb * 1024 * 8) // (self.acc_banks * acc_row_bits)
```

### 1.3 Processing Element (PE)

```python
# pygemmini/core/pe.py
from amaranth import *
from amaranth.lib.data import StructLayout
from ..config import SYSTARSConfig, Dataflow

class PEControl(StructLayout):
    """Control signals for PE operation"""
    dataflow: unsigned(1)    # 0=OS, 1=WS
    propagate: unsigned(1)   # Which register to output
    shift: unsigned(5)       # Rounding shift amount

class PE(Elaboratable):
    """
    Processing Element - the fundamental compute unit.

    Performs: out = in_c + (in_a * in_b) >> shift

    Supports both output-stationary and weight-stationary dataflows:
    - OS: accumulator stays, weights flow through
    - WS: weights stay, partial sums flow through
    """

    def __init__(self, config: SYSTARSConfig):
        self.config = config

        # Inputs
        self.in_a = Signal(signed(config.input_bits))
        self.in_b = Signal(signed(config.weight_bits))
        self.in_d = Signal(signed(config.acc_bits))  # Partial sum (OS) or bias
        self.in_control = Signal(PEControl)
        self.in_valid = Signal()
        self.in_id = Signal(8)  # Tag for tracking
        self.in_last = Signal()

        # Outputs
        self.out_a = Signal(signed(config.input_bits))
        self.out_b = Signal(signed(config.output_bits))
        self.out_c = Signal(signed(config.acc_bits))
        self.out_control = Signal(PEControl)
        self.out_valid = Signal()
        self.out_id = Signal(8)
        self.out_last = Signal()

    def elaborate(self, platform):
        m = Module()
        cfg = self.config

        # Internal registers
        c1 = Signal(signed(cfg.acc_bits))  # Accumulator register 1
        c2 = Signal(signed(cfg.acc_bits))  # Accumulator register 2

        # MAC computation
        product = Signal(signed(cfg.input_bits + cfg.weight_bits))
        m.d.comb += product.eq(self.in_a * self.in_b)

        # Extend product to accumulator width
        product_ext = Signal(signed(cfg.acc_bits))
        m.d.comb += product_ext.eq(product.as_signed())

        # Accumulation with rounding shift
        shift_amt = self.in_control.shift
        accumulated = Signal(signed(cfg.acc_bits))

        # Select accumulator input based on dataflow
        acc_input = Signal(signed(cfg.acc_bits))
        with m.If(self.in_control.dataflow):  # WS
            m.d.comb += acc_input.eq(self.in_d)
        with m.Else():  # OS
            with m.If(self.in_control.propagate):
                m.d.comb += acc_input.eq(c2)
            with m.Else():
                m.d.comb += acc_input.eq(c1)

        # MAC operation
        m.d.comb += accumulated.eq(acc_input + product_ext)

        # Update registers on valid
        with m.If(self.in_valid):
            with m.If(self.in_control.propagate):
                m.d.sync += c2.eq(accumulated)
            with m.Else():
                m.d.sync += c1.eq(accumulated)

        # Output selection
        with m.If(self.in_control.propagate):
            m.d.comb += self.out_c.eq(c1)
        with m.Else():
            m.d.comb += self.out_c.eq(c2)

        # Pass-through signals
        m.d.sync += [
            self.out_a.eq(self.in_a),
            self.out_b.eq(self.in_b[:cfg.output_bits]),
            self.out_control.eq(self.in_control),
            self.out_valid.eq(self.in_valid),
            self.out_id.eq(self.in_id),
            self.out_last.eq(self.in_last),
        ]

        return m
```

### 1.4 Tile and Mesh

```python
# pygemmini/core/tile.py
from amaranth import *
from .pe import PE

class Tile(Elaboratable):
    """
    A tile is a combinational grid of PEs.
    Data flows: A rightward, B/D downward.
    """

    def __init__(self, config: SYSTARSConfig):
        self.config = config
        rows = config.tile_rows
        cols = config.tile_cols

        # Inputs (from left and top)
        self.in_a = [Signal(signed(config.input_bits), name=f"in_a_{i}")
                     for i in range(rows)]
        self.in_b = [Signal(signed(config.weight_bits), name=f"in_b_{i}")
                     for i in range(cols)]
        self.in_d = [Signal(signed(config.acc_bits), name=f"in_d_{i}")
                     for i in range(cols)]
        # ... control signals per column

        # Outputs (to right and bottom)
        self.out_a = [Signal(signed(config.input_bits), name=f"out_a_{i}")
                      for i in range(rows)]
        self.out_b = [Signal(signed(config.output_bits), name=f"out_b_{i}")
                      for i in range(cols)]
        self.out_c = [Signal(signed(config.acc_bits), name=f"out_c_{i}")
                      for i in range(cols)]

    def elaborate(self, platform):
        m = Module()
        cfg = self.config

        # Create PE grid
        pes = [[PE(cfg) for _ in range(cfg.tile_cols)]
               for _ in range(cfg.tile_rows)]

        for r in range(cfg.tile_rows):
            for c in range(cfg.tile_cols):
                m.submodules[f"pe_{r}_{c}"] = pes[r][c]

        # Wire up horizontal (A) connections
        for r in range(cfg.tile_rows):
            m.d.comb += pes[r][0].in_a.eq(self.in_a[r])
            for c in range(1, cfg.tile_cols):
                m.d.comb += pes[r][c].in_a.eq(pes[r][c-1].out_a)
            m.d.comb += self.out_a[r].eq(pes[r][-1].out_a)

        # Wire up vertical (B, D) connections
        for c in range(cfg.tile_cols):
            m.d.comb += pes[0][c].in_b.eq(self.in_b[c])
            m.d.comb += pes[0][c].in_d.eq(self.in_d[c])
            for r in range(1, cfg.tile_rows):
                m.d.comb += pes[r][c].in_b.eq(pes[r-1][c].out_b)
                m.d.comb += pes[r][c].in_d.eq(pes[r-1][c].out_c)
            m.d.comb += self.out_b[c].eq(pes[-1][c].out_b)
            m.d.comb += self.out_c[c].eq(pes[-1][c].out_c)

        return m
```

```python
# pygemmini/core/mesh.py
from amaranth import *
from .tile import Tile

class Mesh(Elaboratable):
    """
    Mesh of tiles with pipeline registers between tiles.
    """

    def __init__(self, config: SYSTARSConfig):
        self.config = config
        dim = config.mesh_rows  # Assume square for simplicity

        # Vector inputs
        self.in_a = [Signal(signed(config.input_bits), name=f"mesh_in_a_{i}")
                     for i in range(dim)]
        self.in_b = [Signal(signed(config.weight_bits), name=f"mesh_in_b_{i}")
                     for i in range(dim)]
        self.in_d = [Signal(signed(config.acc_bits), name=f"mesh_in_d_{i}")
                     for i in range(dim)]

        # Vector outputs
        self.out_b = [Signal(signed(config.output_bits), name=f"mesh_out_b_{i}")
                      for i in range(dim)]
        self.out_c = [Signal(signed(config.acc_bits), name=f"mesh_out_c_{i}")
                      for i in range(dim)]

    def elaborate(self, platform):
        m = Module()
        cfg = self.config
        rows, cols = cfg.mesh_rows, cfg.mesh_cols

        # Create tile grid
        tiles = [[Tile(cfg) for _ in range(cols)] for _ in range(rows)]
        for r in range(rows):
            for c in range(cols):
                m.submodules[f"tile_{r}_{c}"] = tiles[r][c]

        # Wire tiles with pipeline registers
        # A flows rightward with latency between tiles
        # B/D flow downward with latency between tiles

        # ... (similar wiring pattern with ShiftRegister equivalent)

        return m
```

### 1.5 Deliverables for Phase 1

1. **Working PE** with both WS and OS dataflows
2. **Tile** as combinational PE grid
3. **Mesh** with configurable dimensions and pipeline registers
4. **Unit tests** for each component using Amaranth's built-in simulator
5. **Verilog generation** script

---

## Phase 2: Memory System (Weeks 3-4)

### 2.1 Local Address Encoding

```python
# pygemmini/memory/local_addr.py
from amaranth import *
from ..config import SYSTARSConfig

class LocalAddr(StructLayout):
    """
    Unified address format for scratchpad and accumulator.

    Bit layout (32 bits):
    [31]    is_acc_addr    - 1 = accumulator, 0 = scratchpad
    [30]    accumulate     - 1 = accumulate mode
    [29]    read_full_row  - 1 = full width read
    [28:0]  address        - Bank + row address
    """

    def __init__(self, config: SYSTARSConfig):
        sp_addr_bits = (config.sp_banks * config.sp_bank_entries - 1).bit_length()
        acc_addr_bits = (config.acc_banks * config.acc_bank_entries - 1).bit_length()
        self.addr_bits = max(sp_addr_bits, acc_addr_bits)

        super().__init__({
            "data": unsigned(self.addr_bits),
            "garbage": unsigned(1),
            "norm_cmd": unsigned(4),
            "read_full_row": unsigned(1),
            "accumulate": unsigned(1),
            "is_acc": unsigned(1),
        })

    def sp_bank(self, addr):
        """Extract scratchpad bank from address"""
        # Implementation based on config
        pass

    def sp_row(self, addr):
        """Extract scratchpad row within bank"""
        pass

    def is_garbage(self, addr):
        """Check if address is garbage (invalid)"""
        return addr.garbage & addr.is_acc & addr.accumulate & addr.read_full_row
```

### 2.2 Scratchpad Memory

```python
# pygemmini/memory/scratchpad.py
from amaranth import *
from amaranth.lib.memory import Memory

class ScratchpadBank(Elaboratable):
    """Single scratchpad bank with read/write ports"""

    def __init__(self, config: SYSTARSConfig, bank_id: int):
        self.config = config
        self.bank_id = bank_id

        width = config.sp_width
        depth = config.sp_bank_entries

        # Read port
        self.read_addr = Signal(range(depth))
        self.read_data = Signal(width)
        self.read_en = Signal()

        # Write port
        self.write_addr = Signal(range(depth))
        self.write_data = Signal(width)
        self.write_en = Signal()
        self.write_mask = Signal(width // 8)  # Byte mask

    def elaborate(self, platform):
        m = Module()

        mem = Memory(
            width=self.config.sp_width,
            depth=self.config.sp_bank_entries,
            name=f"sp_bank_{self.bank_id}"
        )
        m.submodules.mem = mem

        # Read port with registered output
        rd_port = mem.read_port()
        m.d.comb += rd_port.addr.eq(self.read_addr)
        m.d.comb += rd_port.en.eq(self.read_en)
        m.d.sync += self.read_data.eq(rd_port.data)

        # Write port with byte masking
        wr_port = mem.write_port(granularity=8)
        m.d.comb += wr_port.addr.eq(self.write_addr)
        m.d.comb += wr_port.data.eq(self.write_data)
        m.d.comb += wr_port.en.eq(self.write_mask & self.write_en.replicate(len(self.write_mask)))

        return m

class Scratchpad(Elaboratable):
    """
    Multi-bank scratchpad memory with arbitration.
    """

    def __init__(self, config: SYSTARSConfig):
        self.config = config
        self.banks = [ScratchpadBank(config, i) for i in range(config.sp_banks)]

        # DMA interface
        self.dma_read_req = Signal()  # ... full interface
        self.dma_write_req = Signal()

        # Execute controller interface
        self.exe_read_req = Signal()
        self.exe_write_req = Signal()

    def elaborate(self, platform):
        m = Module()

        for i, bank in enumerate(self.banks):
            m.submodules[f"bank_{i}"] = bank

        # Bank selection and arbitration logic
        # ...

        return m
```

### 2.3 Accumulator with Scaling

```python
# pygemmini/memory/accumulator.py
from amaranth import *

class AccumulatorBank(Elaboratable):
    """
    Accumulator bank with:
    - In-place accumulation (read-modify-write)
    - Output scaling
    - Activation functions
    """

    def __init__(self, config: SYSTARSConfig, bank_id: int):
        self.config = config
        self.bank_id = bank_id

        # Read port with scaling
        self.read_addr = Signal(range(config.acc_bank_entries))
        self.read_data = Signal(config.acc_bits * config.dim)
        self.read_scale = Signal(32)  # Float32 scale factor
        self.read_activation = Signal(2)  # NONE, RELU, etc.

        # Write port with accumulate mode
        self.write_addr = Signal(range(config.acc_bank_entries))
        self.write_data = Signal(config.acc_bits * config.dim)
        self.write_en = Signal()
        self.accumulate = Signal()  # Add to existing vs overwrite
```

### 2.4 Deliverables for Phase 2

1. **LocalAddr** encoding/decoding utilities
2. **ScratchpadBank** with masked writes
3. **Scratchpad** multi-bank with arbitration
4. **AccumulatorBank** with scale/activation pipeline
5. **Accumulator** multi-bank module
6. **Integration tests** for memory system

---

## Phase 3: Controllers & DMA (Weeks 5-7)

### 3.1 Reservation Station (ROB)

```python
# pygemmini/util/reservation.py
from amaranth import *
from amaranth.lib.fifo import SyncFIFOBuffered

class ReservationStation(Elaboratable):
    """
    Command queue with dependency tracking.
    Separate queues for load, store, execute.
    """

    def __init__(self, config: SYSTARSConfig):
        self.config = config

        # Command input (from CPU)
        self.cmd_valid = Signal()
        self.cmd_ready = Signal()
        self.cmd = Signal(128)  # RoCC command

        # Issue ports (to controllers)
        self.ld_issue_valid = Signal()
        self.ld_issue_ready = Signal()
        self.ld_issue_cmd = Signal(128)

        self.st_issue_valid = Signal()
        self.st_issue_ready = Signal()
        self.st_issue_cmd = Signal(128)

        self.ex_issue_valid = Signal()
        self.ex_issue_ready = Signal()
        self.ex_issue_cmd = Signal(128)

        # Completion signals
        self.ld_complete = Signal()
        self.st_complete = Signal()
        self.ex_complete = Signal()
```

### 3.2 Execute Controller

```python
# pygemmini/controller/execute.py
from amaranth import *
from ..core.mesh_with_delays import MeshWithDelays

class ExecuteController(Elaboratable):
    """
    Orchestrates matrix multiply operations.

    Commands:
    - CONFIG: Set dataflow, shift, strides
    - PRELOAD: Load bias/initial values
    - COMPUTE: Execute matmul
    """

    def __init__(self, config: SYSTARSConfig):
        self.config = config

        # Command interface
        self.cmd_valid = Signal()
        self.cmd_ready = Signal()
        self.cmd = Signal(128)

        # Scratchpad read interface
        self.sp_read_req = Signal()
        self.sp_read_addr = Signal(32)
        self.sp_read_data = Signal(config.sp_width)
        self.sp_read_valid = Signal()

        # Accumulator interface
        self.acc_read_req = Signal()
        self.acc_write_req = Signal()
        # ...

        # Completion
        self.completed = Signal()
        self.completed_id = Signal(8)

    def elaborate(self, platform):
        m = Module()

        # Instantiate mesh
        m.submodules.mesh = mesh = MeshWithDelays(self.config)

        # State machine
        # States: IDLE, CONFIG, PRELOAD, COMPUTE, FLUSH

        # ...

        return m
```

### 3.3 Load/Store Controllers

```python
# pygemmini/controller/load.py
class LoadController(Elaboratable):
    """
    DMA load orchestration.
    Handles MVIN, MVIN2, MVIN3 commands with different configurations.
    """

    def __init__(self, config: SYSTARSConfig):
        # 3 load states for overlapping configurations
        self.load_states = [LoadState() for _ in range(3)]
        # ...

# pygemmini/controller/store.py
class StoreController(Elaboratable):
    """
    DMA store with optional max-pooling.
    """
    pass
```

### 3.4 DMA Engine

```python
# pygemmini/dma/reader.py
class StreamReader(Elaboratable):
    """
    DMA reader: Main memory → Local memory

    Features:
    - Virtual address translation via TLB
    - Configurable burst sizes
    - Scaling during transfer
    """
    pass

# pygemmini/dma/writer.py
class StreamWriter(Elaboratable):
    """
    DMA writer: Local memory → Main memory

    Features:
    - Max-pooling during write
    - Activation function application
    """
    pass
```

### 3.5 Deliverables for Phase 3

1. **ReservationStation** with dependency tracking
2. **ExecuteController** FSM with mesh integration
3. **LoadController** with 3 configuration states
4. **StoreController** with pooling support
5. **StreamReader/Writer** DMA engines
6. **DMACommandTracker** for in-flight request management
7. **Integration tests** for command flow

---

## Phase 4: Loop Unrollers & Top-Level (Weeks 8-9)

### 4.1 Loop Matmul FSM

```python
# pygemmini/loop/matmul.py
from amaranth import *

class LoopMatmul(Elaboratable):
    """
    Hardware loop unroller for large matrix multiplications.

    Generates sequences of LOAD, PRELOAD, COMPUTE commands
    for tiled matrix multiply: C = A @ B + D

    Parameters configured via LOOP_WS_CONFIG_* commands.
    """

    def __init__(self, config: SYSTARSConfig):
        self.config = config

        # Configuration registers
        self.max_i = Signal(16)
        self.max_j = Signal(16)
        self.max_k = Signal(16)
        self.a_addr = Signal(64)  # DRAM base
        self.b_addr = Signal(64)
        self.c_addr = Signal(64)
        self.d_addr = Signal(64)
        self.a_stride = Signal(32)
        self.b_stride = Signal(32)
        # ... more config

        # Command output (to reservation station)
        self.cmd_valid = Signal()
        self.cmd_ready = Signal()
        self.cmd = Signal(128)

        # Status
        self.busy = Signal()
        self.done = Signal()

    def elaborate(self, platform):
        m = Module()

        # Loop counters
        i = Signal(16)
        j = Signal(16)
        k = Signal(16)

        # State machine
        with m.FSM():
            with m.State("IDLE"):
                # Wait for start
                pass

            with m.State("LOAD_A"):
                # Generate MVIN for A tile
                pass

            with m.State("LOAD_B"):
                # Generate MVIN for B tile
                pass

            with m.State("LOAD_D"):
                # Generate MVIN for D tile (bias)
                pass

            with m.State("COMPUTE"):
                # Generate PRELOAD + COMPUTE
                pass

            with m.State("STORE_C"):
                # Generate MVOUT for C tile
                pass

            with m.State("NEXT"):
                # Increment counters, check bounds
                pass

        return m
```

### 4.2 Loop Conv FSM

```python
# pygemmini/loop/conv.py
class LoopConv(Elaboratable):
    """
    Hardware loop unroller for convolutions.

    Supports:
    - Standard convolutions
    - Depthwise convolutions
    - Transposed convolutions
    - Max pooling during output
    """
    pass
```

### 4.3 Top-Level Integration

```python
# pygemmini/top.py
from amaranth import *

class SYSTARS(Elaboratable):
    """
    Top-level SYSTARS accelerator.

    Interfaces:
    - RoCC command input (from CPU)
    - Memory bus (TileLink or AXI)
    - Interrupt/completion output
    """

    def __init__(self, config: SYSTARSConfig):
        self.config = config

        # RoCC interface
        self.rocc_cmd_valid = Signal()
        self.rocc_cmd_ready = Signal()
        self.rocc_cmd = Signal(128)
        self.rocc_resp_valid = Signal()
        self.rocc_resp_data = Signal(64)

        # Memory interface (simplified AXI-like)
        self.mem_read_valid = Signal()
        self.mem_read_addr = Signal(64)
        self.mem_read_data = Signal(config.dma_buswidth)
        self.mem_read_ready = Signal()

        self.mem_write_valid = Signal()
        self.mem_write_addr = Signal(64)
        self.mem_write_data = Signal(config.dma_buswidth)
        self.mem_write_ready = Signal()

    def elaborate(self, platform):
        m = Module()

        # Instantiate all submodules
        m.submodules.rob = rob = ReservationStation(self.config)
        m.submodules.scratchpad = sp = Scratchpad(self.config)
        m.submodules.accumulator = acc = Accumulator(self.config)
        m.submodules.exe_ctrl = exe = ExecuteController(self.config)
        m.submodules.load_ctrl = ld = LoadController(self.config)
        m.submodules.store_ctrl = st = StoreController(self.config)
        m.submodules.loop_matmul = lm = LoopMatmul(self.config)
        m.submodules.loop_conv = lc = LoopConv(self.config)
        m.submodules.dma_reader = dr = StreamReader(self.config)
        m.submodules.dma_writer = dw = StreamWriter(self.config)

        # Wire everything together
        # ...

        return m
```

### 4.4 Deliverables for Phase 4

1. **LoopMatmul** FSM for tiled GEMM
2. **LoopConv** FSM for convolutions
3. **SYSTARS** top-level module
4. **Full integration tests**
5. **Verilog generation** with all components

---

## Phase 5: Validation & Tooling (Weeks 10-11)

### 5.1 Header File Generator

```python
# scripts/generate_header.py
from pysystars.config import SystarsConfig

def generate_header(config: SystarsConfig, output_path: str):
    """
    Generate gemmini_params.h for software compatibility.

    Must match the format expected by gemmini-rocc-tests.
    """

    header = f"""
#pragma once

#include <stdint.h>
#include <limits.h>

// Auto-generated by PySystars

#define DIM {config.dim}
#define ADDR_LEN 32
#define BANK_NUM {config.sp_banks}
#define BANK_ROWS {config.sp_bank_entries}
#define ACC_ROWS {config.acc_banks * config.acc_bank_entries}
#define MAX_BYTES {config.dma_maxbytes}
#define MAX_BLOCK_LEN (MAX_BYTES / (DIM * sizeof(elem_t)))
#define MAX_BLOCK_LEN_ACC (MAX_BYTES / (DIM * sizeof(acc_t)))

typedef int8_t elem_t;
typedef int16_t full_t;
typedef int32_t acc_t;
typedef float acc_scale_t;

#define ELEM_T_IS_FLOAT 0
#define ACC_T_IS_FLOAT 0

#define row_align(blocks) __attribute__((aligned(blocks * DIM * sizeof(elem_t))))
#define row_align_acc(blocks) __attribute__((aligned(blocks * DIM * sizeof(acc_t))))

#define HAS_MVIN_SCALE
#define MVIN_SCALE_IDENTITY 1.0

#define HAS_FIRST_LAYER_OPTIMIZATIONS 1

// Activation functions
#define ACTIVATION_NONE 0
#define ACTIVATION_RELU 1
#define ACTIVATION_RELU6 2

"""

    with open(output_path, 'w') as f:
        f.write(header)
```

### 5.2 Comparison Testing

```python
# scripts/compare_with_chisel.py
"""
Compare PySystars output against reference fsim implementation.

1. Generate Verilog from both implementations
2. Run identical test vectors
3. Compare outputs cycle-by-cycle
"""

def compare_pe_output():
    """Compare single PE behavior"""
    pass

def compare_mesh_output():
    """Compare full mesh behavior"""
    pass

def compare_matmul_output():
    """Compare complete matmul operation"""
    pass
```

### 5.3 Deliverables for Phase 5

1. **Header generator** compatible with gemmini-rocc-tests
2. **Verilog comparison** scripts
3. **Simulation testbench** with reference vectors
4. **Documentation** for all modules
5. **Example configurations** (default, lean, FP)

---

## Implementation Priority

| Priority | Component | Complexity | Dependencies |
|----------|-----------|------------|--------------|
| 1 | PE | Low | None |
| 2 | Tile | Low | PE |
| 3 | Mesh | Medium | Tile |
| 4 | LocalAddr | Low | None |
| 5 | ScratchpadBank | Medium | None |
| 6 | Scratchpad | Medium | ScratchpadBank |
| 7 | AccumulatorBank | High | None |
| 8 | Transposer | Medium | None |
| 9 | MeshWithDelays | High | Mesh, Transposer |
| 10 | ReservationStation | High | None |
| 11 | ExecuteController | High | MeshWithDelays, Memory |
| 12 | LoadController | Medium | DMA |
| 13 | StoreController | Medium | DMA |
| 14 | StreamReader | High | TLB (optional) |
| 15 | StreamWriter | High | TLB (optional) |
| 16 | LoopMatmul | High | All controllers |
| 17 | LoopConv | High | All controllers |
| 18 | Top-level | Medium | All |

---

## Success Criteria

1. **Functional**: PySystars generates Verilog that produces functional systolic arrays
2. **Software Compatibility**: Generated header file works with Stillwater Domain Flow Architecture compiler
3. **Configuration Parity**: All major configuration parameters are supported
4. **Performance**: Generated Verilog meets timing at target frequencies
5. **Maintainability**: Clean Python code with comprehensive tests

---

## Recommended First Steps

1. **Set up repository** with Amaranth HDL, pytest, and development dependencies
2. **Implement PE** with comprehensive unit tests
3. **Build Tile and Mesh** incrementally
4. **Create simple testbench** that exercises basic matmul
5. **Compare against fsim** output for validation

---

## Notes for Separate Repository

This plan assumes a **clean-slate repository** with:

- Modern dependencies (Python, Just, Yosys, Verilator/GHDL, RTL linting, synthesis checks, etc.)
- Python 3.10+ with Amaranth HDL
- pytest for Python script testing
- cocotb for RTL testbenches and co-simulation with Verilator

The new repository should be independent and self-contained.
