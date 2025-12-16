# Session Log: Phase 2 Memory System Implementation

**Date**: 2025-12-15
**Duration**: ~1 hour
**Model**: Claude Opus 4.5

## Summary

Implemented the Phase 2 Memory System for the systars systolic array RTL generator. This includes local address encoding, multi-bank scratchpad memory, and accumulator memory with activation functions.

## Work Completed

### Files Created

| File | Lines | Description |
|------|-------|-------------|
| `src/systars/memory/__init__.py` | 8 | Module exports |
| `src/systars/memory/local_addr.py` | ~120 | Address encoding utilities |
| `src/systars/memory/scratchpad.py` | ~230 | Scratchpad bank and controller |
| `src/systars/memory/accumulator.py` | ~300 | Accumulator bank and controller |
| `tests/unit/test_local_addr.py` | ~280 | LocalAddr unit tests |
| `tests/unit/test_scratchpad.py` | ~320 | Scratchpad unit tests |
| `tests/unit/test_accumulator.py` | ~390 | Accumulator unit tests |
| `CHANGELOG.md` | ~60 | Project changelog |

### Components Implemented

#### 1. LocalAddr - Address Encoding Utilities

Unified 32-bit address format for both scratchpad and accumulator:

```
[31]    is_acc        - 1 = accumulator, 0 = scratchpad
[30]    accumulate    - 1 = accumulate mode (add to existing)
[29]    read_full_row - 1 = read entire row width
[28:0]  data          - Bank + row address
```

Features:

- Static methods for bit field extraction
- Instance methods for bank/row decoding (config-dependent)
- Address construction helpers
- Garbage address detection

#### 2. ScratchpadBank / Scratchpad

Multi-bank local memory for storing input activations and weights:

- **ScratchpadBank**: Single bank with read/write ports
  - Configurable read latency pipeline
  - Byte-level write masking (granularity=8)
  - Valid signal tracking through pipeline

- **Scratchpad**: Multi-bank controller
  - Address-based bank selection
  - Pipelined bank selection to match read latency
  - Parallel bank instantiation

#### 3. AccumulatorBank / Accumulator

Result memory with activation functions:

- **AccumulatorBank**: Single bank with activation
  - Activation functions: NONE, RELU, RELU6
  - Accumulate mode (read-modify-write)
  - Signed data storage
  - Configurable latency pipeline

- **Accumulator**: Multi-bank controller
  - Bank selection routing
  - Activation function passthrough

### Test Results

```
60 passed, 13 skipped (Yosys not found)
```

All functional tests pass. Verilog generation tests skipped due to Yosys not being installed in the environment.

## Issues Encountered and Resolved

### 1. Elaboratable Inheritance

**Issue**: Inner `TestModule` classes in `test_local_addr.py` failed with "not an Elaboratable" error.

**Solution**: Added explicit `Elaboratable` inheritance and a dummy sync signal to create the sync domain required by `sim.add_clock()`.

### 2. Memory Read Pipeline Timing

**Issue**: Accumulator and scratchpad tests failed with `assert 0 == 1` (valid signal not going high) and `assert 0 == 100` (data not being read correctly).

**Root Cause**: The read pipeline was adding `acc_latency` stages on top of the memory's inherent 1-cycle latency, resulting in `1 + latency` total cycles instead of `latency` cycles.

**Solution**: Restructured the pipeline to account for memory's inherent latency:

- `extra_stages = latency - 1` (memory provides 1 cycle)
- For `latency=1`: Direct output with 1-cycle valid delay
- For `latency>1`: `(latency-1)` additional pipeline stages

### 3. Test Timing Adjustments

**Issue**: Tests were waiting `latency` additional cycles after the initial tick, causing the valid signal to already be back to 0 by the time it was checked.

**Solution**: Adjusted test timing to wait `(latency - 1)` ticks after `read_en` goes low, so total wait from `read_en` going high equals `latency` cycles.

## Architecture Decisions

1. **Memory Primitive**: Used Amaranth's `Memory` class for synthesis to SRAM/BRAM
2. **Read-Modify-Write**: Accumulator uses a second read port for existing value retrieval during accumulate mode
3. **Byte Masking**: Scratchpad uses `write_port(granularity=8)` for DMA partial writes
4. **Activation Functions**: Implemented as combinatorial logic on memory output

## Next Steps (Phase 3)

Per the project roadmap:

- Controller FSM for orchestrating data flow
- DMA engine for external memory interface
- Instruction decoder for operation sequencing
