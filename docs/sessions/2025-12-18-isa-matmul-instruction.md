# Session Log: ISA-Level Matmul Instruction

**Date:** 2025-12-18
**Focus:** Implementing a high-level Matmul ISA instruction with automatic tiling and command generation

## Summary

This session implemented a complete ISA-level `matmul` instruction that automatically handles tiled matrix multiplication, dataflow selection, backpressure handling, and double buffering. The instruction generates low-level commands (LOAD_A, LOAD_B, LOAD_D, EXEC_CONFIG, EXEC_PRELOAD, EXEC_COMPUTE, STORE_C) from a high-level configuration interface.

## Work Completed

### 1. Fixed Address Calculation Test

The initial `test_load_addresses_correct` test was failing, capturing address `0xaaaa0004` instead of expected `0xaaaa0000`:

- **Root Cause:** Test loop was capturing the second LOAD_A command (after `tile_k` incremented) instead of the first
- **Fix:** Modified test to capture the first LOAD_A occurrence and break immediately

### 2. Implemented Backpressure Handling

Added valid/ready handshake support to all command-emitting states:

```python
with m.State("CONFIG"):
    m.d.comb += [
        self.cmd_valid.eq(1),
        self.cmd_opcode.eq(InternalOpcode.EXEC_CONFIG),
        self.cmd_rs1.eq(dataflow_mode),
    ]
    with m.If(self.cmd_ready):  # Wait for consumer ready
        with m.If(reg_D_addr != 0):
            m.next = "LOAD_D"
        with m.Else():
            m.next = "LOAD_A"
```

States with backpressure: CONFIG, LOAD_D, LOAD_A, LOAD_B, PRELOAD, COMPUTE, STORE

### 3. Added Double Buffering with Bank Toggling

Implemented bank toggling for latency hiding during K-dimension iteration:

**Scratchpad Banks (NEXT_K state):**

```python
m.d.sync += [
    sp_bank_A.eq(sp_bank_A ^ 2),  # 0↔2
    sp_bank_B.eq(sp_bank_B ^ 2),  # 1↔3
]
```

**Accumulator Banks (NEXT_IJ state):**

```python
m.d.sync += [
    acc_bank.eq(acc_bank ^ 1),    # 0↔1
    sp_bank_A.eq(0),              # Reset for new tile
    sp_bank_B.eq(1),
]
```

Added debug ports (`dbg_sp_bank_A`, `dbg_sp_bank_B`, `dbg_acc_bank`) for test visibility.

### 4. Comprehensive Unit Tests

Created 13 tests in `tests/unit/test_matmul.py`:

| Test | Description |
|------|-------------|
| `test_config_dims` | Dimension configuration interface |
| `test_config_addresses` | Address/stride configuration |
| `test_config_options` | Options (accumulate, activation) |
| `test_load_addresses_correct` | Address calculation for tiled operations |
| `test_dataflow_selection_*` | Output/A/B-stationary selection heuristics |
| `test_command_emission_sequence` | Correct command ordering |
| `test_stall_on_backpressure` | Backpressure handling |
| `test_bank_toggling` | Double buffering bank toggle |
| `test_multi_tile_execution` | Multi-tile K-dimension iteration |
| `test_with_bias` | Bias loading (LOAD_D) |
| `test_error_on_zero_dimension` | Error handling for invalid config |

### 5. Application Example

Created `examples/gemm/07_isa_matmul.py` demonstrating:

- Configuration interface with configurable M, N, K dimensions
- Automatic tiling for arbitrary matrix sizes
- Dataflow selection heuristics explained
- Command sequence generation and verification
- Expected command count validation

```bash
python examples/gemm/07_isa_matmul.py           # Default 32x32 matrices
python examples/gemm/07_isa_matmul.py -m 64 -n 64 -k 128 --array-size 16
```

### 6. CI Fixes

**NumPy Dependency:**
Added `numpy>=1.24` to dev dependencies in `pyproject.toml` for GEMM demo/e2e tests.

**Yosys Skip Logic:**
Added skip-if-no-Yosys logic to 3 tests in `test_gemm_demo.py` and `test_gemm_e2e.py`:

```python
def test_verilog_generation(self):
    from amaranth._toolchain.yosys import find_yosys
    try:
        find_yosys(lambda ver: ver >= (0, 40))
    except Exception:
        pytest.skip("Yosys not found")
```

**CI Workflow Update:**
Added OSS CAD Suite to unit-tests job for full Verilog generation test coverage:

```yaml
- name: Install OSS CAD Suite (Yosys)
  uses: YosysHQ/setup-oss-cad-suite@v3
  with:
    version: "2025-12-12"
```

## Technical Details

### State Machine Design

The Matmul instruction uses a 12-state FSM:

```
IDLE → CONFIG → [LOAD_D] → LOAD_A → LOAD_B → PRELOAD → COMPUTE → STORE
                    ↑                                              |
                    +--- NEXT_K (tile K loop) ---←-----------------+
                    |                                              |
                    +--- NEXT_IJ (tile M/N loop) ---←--------------+
                    |
                    +--- DONE (completion) ---→ IDLE
```

### Dataflow Selection Heuristics

```python
if (K > M) and (K > N):
    dataflow = B_STATIONARY  # Weight-stationary for large K
elif M >= N:
    dataflow = OUTPUT_STATIONARY  # Output stays for wide M
else:
    dataflow = A_STATIONARY  # A stays for tall N
```

### Address Calculation

For tile (tile_i, tile_j, tile_k) with array size S:

```python
# Matrix A: [M, K] stored row-major
A_addr = A_base + (tile_i * S * K + tile_k * S) * elem_bytes

# Matrix B: [K, N] stored row-major
B_addr = B_base + (tile_k * S * N + tile_j * S) * elem_bytes

# Matrix C: [M, N] stored row-major
C_addr = C_base + (tile_i * S * N + tile_j * S) * elem_bytes
```

### Internal Command Opcodes

| Opcode | Description |
|--------|-------------|
| `LOAD_A` | Load A tile to scratchpad bank A |
| `LOAD_B` | Load B tile to scratchpad bank B |
| `LOAD_D` | Load D (bias) to accumulator |
| `EXEC_CONFIG` | Configure dataflow mode |
| `EXEC_PRELOAD` | Preload weights for computation |
| `EXEC_COMPUTE` | Execute MAC operations |
| `STORE_C` | Store result from accumulator |

## Files Modified

- `src/systars/isa/__init__.py` - New ISA module
- `src/systars/isa/matmul.py` - Main Matmul instruction (~800 lines)
- `tests/unit/test_matmul.py` - 13 comprehensive tests
- `examples/gemm/07_isa_matmul.py` - Interactive demonstration
- `pyproject.toml` - Added numpy dependency
- `tests/unit/test_gemm_demo.py` - Added Yosys skip logic
- `tests/unit/test_gemm_e2e.py` - Added Yosys skip logic
- `.github/workflows/test.yml` - Added Yosys to unit tests CI
- `CHANGELOG.md` - Documented all changes

## Test Results

All 214 unit tests passing:

```
tests/unit/test_matmul.py: 13 passed
tests/unit/test_gemm_demo.py: 12 passed
tests/unit/test_gemm_e2e.py: 11 passed
(plus 178 other unit tests)
```

## Key Insights

### Pytest Fixture Dependencies

When using pytest fixtures that depend on other fixtures, don't include the parent fixture as a test method parameter:

```python
# Wrong - pytest can't find '_config' fixture
def test_example(self, _config, matmul): ...

# Correct - matmul already depends on config fixture
def test_example(self, matmul): ...
```

### Amaranth Simulation Timing

When testing state machines with valid/ready handshake:

- Set `cmd_ready=1` before the loop starts
- Check state immediately after asserting ready, not after another `Tick()`
- The state machine transitions on the clock edge when both valid and ready are high

### CI Tool Dependencies

Verilog generation tests require Yosys for elaboration. Options:

1. Skip tests when Yosys unavailable (safety net for local dev)
2. Add Yosys to CI for full coverage (preferred for catching elaboration errors)

## Next Steps

Potential future enhancements:

1. Implement Conv2d ISA instruction following same pattern
2. Add fused operations (matmul + activation)
3. Support strided access patterns
4. Implement software loop unrolling for outer tiles
