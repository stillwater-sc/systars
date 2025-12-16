# DMA Descriptor Architecture

This document describes the descriptor-based DMA architecture for SYSTARS, including the descriptor format, memory map, hardware components, and forward refinement path toward virtual memory support.

## Overview

The SYSTARS DMA subsystem uses a **descriptor-based architecture** inspired by modern DMA engines like Intel IOAT (I/O Acceleration Technology). Instead of issuing individual commands for each transfer, software constructs chains of descriptors in memory that the hardware fetches and executes autonomously.

### Key Benefits

1. **Reduced CPU overhead**: Software builds descriptor chains once; hardware executes them without further intervention
2. **Complex transfer patterns**: Chain multiple operations (copy, fill, fence) in a single submission
3. **Asynchronous completion**: Hardware signals completion via interrupts or status writeback
4. **Scalability**: Natural path to scatter-gather, virtual addressing, and multi-queue support

## Descriptor Format

Each descriptor is **64 bytes**, 8-byte aligned, with the following layout:

```
Offset  Size   Field         Description
──────────────────────────────────────────────────────────────────
0x00    1      opcode        Operation code (MEMCPY, FILL, etc.)
0x01    1      flags         Operation flags (CHAIN, INTERRUPT, etc.)
0x02    2      reserved      Reserved for future use
0x04    4      length        Transfer length in bytes
0x08    8      src_addr      Source address (or fill pattern for FILL)
0x10    8      dst_addr      Destination address
0x18    8      next_desc     Pointer to next descriptor (0 = end of chain)
0x20    8      completion    Status writeback address
0x28    8      user_data     User-defined data (passed to completion)
0x30    8      reserved2     Reserved for future use
0x38    8      reserved3     Reserved for future use
──────────────────────────────────────────────────────────────────
Total: 64 bytes
```

### Binary Layout (Little-Endian)

```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
├─┴─┴─┴─┴─┴─┴─┴─┼─┴─┴─┴─┴─┴─┴─┴─┼─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┤
│    opcode     │     flags     │           reserved            │
├───────────────┴───────────────┴───────────────────────────────┤
│                            length                             │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│                           src_addr                            │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│                           dst_addr                            │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│                          next_desc                            │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│                          completion                           │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│                          user_data                            │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│                          reserved2                            │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│                          reserved3                            │
└───────────────────────────────────────────────────────────────┘
```

## Opcodes

The DMA engine supports the following operation codes:

| Opcode | Value | Description |
|--------|-------|-------------|
| `MEMCPY` | 0x00 | Copy `length` bytes from `src_addr` to `dst_addr` |
| `FILL` | 0x01 | Fill `dst_addr` with 64-bit pattern in `src_addr` field |
| `GATHER` | 0x02 | Gather scattered elements to contiguous destination (future) |
| `SCATTER` | 0x03 | Scatter contiguous source to multiple destinations (future) |
| `CRC32` | 0x10 | Compute CRC32 over source region, write to `dst_addr` |
| `CHECKSUM` | 0x11 | Compute simple checksum (future) |
| `FENCE` | 0x20 | Memory barrier - wait for all prior operations |
| `NOP` | 0x21 | No operation (useful for padding/alignment) |
| `SIGNAL` | 0x30 | Write completion value to address (future) |
| `INTERRUPT` | 0x31 | Generate interrupt (future) |

## Flags

| Flag | Value | Description |
|------|-------|-------------|
| `CHAIN` | 0x01 | Continue to `next_desc` after completion |
| `INTERRUPT_ON_COMPLETE` | 0x02 | Generate interrupt when descriptor completes |
| `WRITEBACK_STATUS` | 0x04 | Write status to `completion` address |
| `SRC_IS_PATTERN` | 0x08 | `src_addr` contains fill pattern, not address |
| `DST_IS_LOCAL` | 0x10 | Destination is local memory (scratchpad/accumulator) |
| `SRC_IS_LOCAL` | 0x20 | Source is local memory |

## Memory Map

The DMA engine uses a **unified physical address space** where the address determines the memory type:

```
Address Range                    Memory Type
─────────────────────────────────────────────────────────
0x0000_0000_0000 - 0x0000_FFFF_FFFF    External DRAM (via AXI)
0x0001_0000_0000 - 0x0001_0000_FFFF    Scratchpad (local SRAM)
0x0002_0000_0000 - 0x0002_0000_FFFF    Accumulator (local SRAM)
0x0003_0000_0000 - ...                  Reserved for future use
```

### Address-Based Routing

The hardware examines address bits to route transfers:

```python
def get_memory_region(addr: int) -> str:
    if addr < 0x0001_0000_0000:
        return "DRAM"        # External memory via AXI
    elif addr < 0x0002_0000_0000:
        return "SCRATCHPAD"  # Local memory bank
    elif addr < 0x0003_0000_0000:
        return "ACCUMULATOR" # Accumulator memory
    else:
        return "RESERVED"
```

This design allows a single `MEMCPY` operation to handle:

- **DRAM → Scratchpad** (load): `src_addr=0x1000`, `dst_addr=0x0001_0000_0100`
- **Accumulator → DRAM** (store): `src_addr=0x0002_0000_0000`, `dst_addr=0x2000`
- **Scratchpad → Accumulator** (local copy): Both addresses in local range

## Hardware Architecture

### Component Hierarchy

```
┌─────────────────────────────────────────────────────────────────┐
│                        DescriptorEngine                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │   Fetch     │  │   Parse     │  │       Execute           │ │
│  │   State     │──│   State     │──│  MEMCPY/FILL/FENCE/...  │ │
│  │   Machine   │  │   Machine   │  │                         │ │
│  └──────┬──────┘  └─────────────┘  └───────────┬─────────────┘ │
│         │                                      │               │
│         │ Fetch Interface                      │ Data Interface│
└─────────┼──────────────────────────────────────┼───────────────┘
          │                                      │
          ▼                                      ▼
┌─────────────────┐                    ┌─────────────────┐
│  StreamReader   │                    │  StreamWriter   │
│  (AXI Read)     │                    │  (AXI Write)    │
└────────┬────────┘                    └────────┬────────┘
         │                                      │
         └──────────────┬───────────────────────┘
                        │
                        ▼
              ┌─────────────────┐
              │   AXI Fabric    │
              │   (to DRAM)     │
              └─────────────────┘
```

### DescriptorEngine State Machine

```
                    ┌──────────┐
                    │   IDLE   │◄──────────────────────────┐
                    └────┬─────┘                           │
                         │ start                           │
                         ▼                                 │
                ┌────────────────┐                         │
                │  FETCH_DESC    │                         │
                │  (Issue read)  │                         │
                └───────┬────────┘                         │
                        │ req_ready                        │
                        ▼                                  │
                ┌────────────────┐                         │
                │   WAIT_DESC    │                         │
                │  (Recv beats)  │                         │
                └───────┬────────┘                         │
                        │ last beat                        │
                        ▼                                  │
                ┌────────────────┐                         │
                │  PARSE_DESC    │                         │
                │  (Decode)      │                         │
                └───────┬────────┘                         │
                        │                                  │
           ┌────────────┼────────────┬──────────┐          │
           ▼            ▼            ▼          ▼          │
    ┌───────────┐ ┌───────────┐ ┌────────┐ ┌────────┐     │
    │  MEMCPY   │ │   FILL    │ │ FENCE  │ │  NOP   │     │
    │  States   │ │  States   │ │        │ │        │     │
    └─────┬─────┘ └─────┬─────┘ └───┬────┘ └───┬────┘     │
          │             │           │          │          │
          └─────────────┴───────────┴──────────┘          │
                        │                                  │
                        ▼                                  │
                ┌────────────────┐                         │
                │  CHECK_CHAIN   │                         │
                └───────┬────────┘                         │
                        │                                  │
              ┌─────────┴─────────┐                        │
              │                   │                        │
         CHAIN=1             CHAIN=0                       │
              │                   │                        │
              ▼                   ▼                        │
       (back to            ┌───────────┐                   │
        FETCH_DESC)        │   DONE    │───────────────────┘
                           │(interrupt)│
                           └───────────┘
```

### Interfaces

#### Command Interface (from Host/Driver)

```
start      : in  - Pulse to begin processing
desc_addr  : in  - Address of first descriptor (64-bit)
busy       : out - Engine is processing
done       : out - Chain complete (pulse)
error      : out - Error occurred (pulse)
```

#### Descriptor Fetch Interface (to StreamReader)

```
fetch_req_valid  : out - Request to fetch descriptor
fetch_req_ready  : in  - Ready to accept
fetch_req_addr   : out - Descriptor address
fetch_resp_valid : in  - Data valid
fetch_resp_data  : in  - Descriptor data (bus width)
fetch_resp_last  : in  - Last beat
```

#### Data Interfaces (to StreamReader/StreamWriter)

```
read_req_valid   : out - Read request
read_req_addr    : out - Source address
read_resp_data   : in  - Read data

write_req_valid  : out - Write request
write_req_addr   : out - Destination address
write_data       : out - Write data
write_done       : in  - Write complete
```

## Software API

### Python Descriptor API

```python
from systars.util.commands import (
    DmaDescriptor, DmaOpcode, DmaFlags, MemoryRegion,
    make_memcpy, make_fill, make_fence,
    DescriptorChain,
)

# Create individual descriptors
desc = make_memcpy(
    src_addr=0x1000,                          # DRAM source
    dst_addr=MemoryRegion.SCRATCHPAD_BASE,    # Scratchpad dest
    length=4096,
    interrupt=True,
)

# Convert to hardware format
hw_bytes = desc.to_bytes()  # 64 bytes

# Build descriptor chains
chain = (
    DescriptorChain(base_addr=0x10000)  # Where to store descriptors
    .fill(dst=MemoryRegion.SCRATCHPAD_BASE, length=4096, pattern=0)
    .memcpy(src=0x0, dst=MemoryRegion.SCRATCHPAD_BASE, length=4096)
    .memcpy(src=0x1000, dst=MemoryRegion.SCRATCHPAD_BASE + 4096, length=4096)
    .fence(interrupt=True)
)
descriptors = chain.build()
chain_bytes = chain.to_bytes()  # All descriptors concatenated
```

### Descriptor Chain Example

A typical matrix load sequence:

```python
# Load matrix A (16KB) and matrix B (16KB) to scratchpad
chain = (
    DescriptorChain(base_addr=DESC_BUFFER_ADDR)
    # Zero scratchpad first
    .fill(
        dst=MemoryRegion.SCRATCHPAD_BASE,
        length=32768,
        pattern=0
    )
    # Load matrix A
    .memcpy(
        src=MATRIX_A_DRAM_ADDR,
        dst=MemoryRegion.SCRATCHPAD_BASE,
        length=16384
    )
    # Load matrix B
    .memcpy(
        src=MATRIX_B_DRAM_ADDR,
        dst=MemoryRegion.SCRATCHPAD_BASE + 16384,
        length=16384
    )
    # Memory barrier before compute
    .fence(interrupt=True)
)

# Write descriptors to memory and start engine
write_to_memory(DESC_BUFFER_ADDR, chain.to_bytes())
start_descriptor_engine(DESC_BUFFER_ADDR)
```

## Forward Refinement Path

### Phase 1: Current Implementation (Physical Addresses)

- Direct physical addresses in descriptors
- Single descriptor chain at a time
- Basic operations: MEMCPY, FILL, FENCE, NOP
- Polling or interrupt-based completion

### Phase 2: Address Translation (TLB)

Add a Translation Lookaside Buffer (TLB) to support virtual addresses:

```
┌─────────────────────────────────────────────────────────────┐
│                     DescriptorEngine                        │
│                            │                                │
│                    ┌───────▼───────┐                        │
│                    │  TLB Lookup   │◄─── Page Table Walker  │
│                    │  (VA → PA)    │                        │
│                    └───────┬───────┘                        │
│                            │                                │
└────────────────────────────┼────────────────────────────────┘
                             ▼
                      Physical Address
```

**New Descriptor Fields** (using reserved space):

```
Offset  Field         Description
0x30    page_table    Pointer to page table (VA mode)
0x38    asid          Address Space ID for TLB tagging
```

**New Flags**:

- `USE_VIRTUAL_ADDR` (0x40): Interpret addresses as virtual
- `PRIVILEGED` (0x80): Privileged mode access

### Phase 3: Multi-Queue Support

Support multiple concurrent descriptor queues for:

- Different priority levels
- Multiple contexts/processes
- Out-of-order completion

```
┌─────────────────────────────────────────────────┐
│               Queue Arbiter                     │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐           │
│  │ Queue 0 │ │ Queue 1 │ │ Queue 2 │  ...      │
│  │ (High)  │ │ (Med)   │ │ (Low)   │           │
│  └────┬────┘ └────┬────┘ └────┬────┘           │
│       └──────────┬┴──────────┘                 │
│                  ▼                              │
│          DescriptorEngine                       │
└─────────────────────────────────────────────────┘
```

**Queue Descriptor** (for queue management):

```
Offset  Field         Description
0x00    head_ptr      Head pointer (next to process)
0x08    tail_ptr      Tail pointer (last submitted)
0x10    base_addr     Queue base address
0x18    size_mask     Queue size (power of 2) - 1
0x20    doorbell      Doorbell register offset
```

### Phase 4: Scatter-Gather

Enable complex memory patterns for:

- Strided access (matrix column extraction)
- Non-contiguous buffers
- Multi-dimensional data movement

**Scatter-Gather List Entry**:

```
Offset  Field         Description
0x00    addr          Segment address
0x08    length        Segment length
0x10    stride        Stride for repeated segments
0x18    count         Number of repetitions
```

**New Opcodes**:

- `GATHER` (0x02): Multiple sources → contiguous destination
- `SCATTER` (0x03): Contiguous source → multiple destinations

### Phase 5: Hardware Acceleration Extensions

Add specialized operations for ML workloads:

| Opcode | Description |
|--------|-------------|
| `TRANSPOSE` | In-flight matrix transpose |
| `PACK` | Data type conversion (FP32→INT8) |
| `UNPACK` | Data type conversion (INT8→FP32) |
| `IM2COL` | Image to column transform |
| `REDUCE` | Partial sum reduction |

## Performance Considerations

### Descriptor Prefetching

The engine can prefetch the next descriptor while executing the current one:

```
Time ─────────────────────────────────────────────────►

         ┌─────────┐     ┌─────────┐     ┌─────────┐
Fetch:   │  Desc0  │     │  Desc1  │     │  Desc2  │
         └────┬────┘     └────┬────┘     └────┬────┘
              │               │               │
              ▼               ▼               ▼
         ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
Execute: │   Op0       │ │   Op1       │ │   Op2       │
         └─────────────┘ └─────────────┘ └─────────────┘

With prefetch:
         ┌─────────┐
Fetch:   │  Desc0  │─┐
         └────┬────┘ │
              │      │ ┌─────────┐
              ▼      └─│  Desc1  │─┐
         ┌─────────────┐└────┬────┘ │
Execute: │   Op0       │     │      │ ┌─────────┐
         └─────────────┘     ▼      └─│  Desc2  │
                        ┌─────────────┐└────────┘
                        │   Op1       │
                        └─────────────┘
```

### Coalescing

For small transfers, batch multiple descriptors into a single large DMA operation:

```python
# Instead of many small transfers:
for i in range(100):
    chain.memcpy(src=base + i*64, dst=sp + i*64, length=64)

# Coalesce into one:
chain.memcpy(src=base, dst=sp, length=6400)
```

### Alignment Requirements

- Descriptor addresses: 64-byte aligned
- Transfer addresses: Bus-width aligned (e.g., 16-byte for 128-bit bus)
- Length: Multiple of bus width for optimal throughput

## Error Handling

### Error Conditions

| Error | Cause | Recovery |
|-------|-------|----------|
| `INVALID_OPCODE` | Unknown opcode in descriptor | Skip descriptor, set error flag |
| `ADDR_MISALIGN` | Address not properly aligned | Set error flag, report address |
| `LENGTH_ZERO` | Zero-length transfer | Skip (NOP behavior) |
| `TLB_MISS` | Virtual address not mapped (Phase 2+) | Invoke page fault handler |
| `AXI_ERROR` | Bus error during transfer | Retry or abort chain |

### Error Reporting

The `completion` field can receive error status:

```
Completion Status Word (32-bit):
┌───────────────────────────────────────────────────────┐
│ 31    24 │ 23    16 │ 15     8 │ 7      0            │
├──────────┼──────────┼──────────┼─────────────────────┤
│  Error   │ Reserved │ Desc Idx │     Status          │
│  Code    │          │ (chain)  │  0=OK, 1=Err        │
└───────────────────────────────────────────────────────┘
```

## References

- Intel I/O Acceleration Technology (IOAT) Architecture
- ARM SMMU (System Memory Management Unit) Specification
- PCI Express DMA Controller specifications
- NVIDIA GPUDirect RDMA documentation
