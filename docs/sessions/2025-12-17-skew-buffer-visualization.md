# Session Log: Skew Buffer Timing Visualization

**Date:** 2025-12-17
**Focus:** Fixing and enhancing the 2D skew buffer timing animation

## Summary

This session comprehensively improved the skew buffer timing visualization (`examples/gemm/05_skew_buffer_timing.py`), fixing numerous alignment issues, timing bugs, and adding new features for interactive debugging.

## Work Completed

### 1. Added Manual Step Mode

Added `--step` argument for single-stepping through the animation:

```bash
python examples/gemm/05_skew_buffer_timing.py --animate --step
```

Press Enter to advance each cycle, enabling detailed study of data flow.

### 2. Fixed SRAM Timing Display

Corrected k-value calculation that was showing k=-1 when data became valid:

- Old formula: `k_val = feeder.cycle - sram_latency - 1` (wrong)
- Fixed to: `k_val = cycle_num - sram_latency + 1`

Also fixed `current_k` calculation in `step()` method for proper bounds checking.

### 3. Fixed Display Timing Consistency

Restructured the animation loop to display all state consistently:

- Previously: SRAM state shown BEFORE `step()`, skew buffer output AFTER
- Now: All state captured and displayed AFTER `step()` for consistency

### 4. Fixed Column Alignment

Fixed multiple alignment issues between headers and data:

**Skew Buffer B (5-char columns):**

- Header: `" b{col}  "` = 5 chars
- Values: `fmt_val() + " "` = 4 + 1 = 5 chars
- Empty dots: `"  · " + " "` = 4 + 1 = 5 chars
- Unused: 5 spaces

**Skew A (4-char columns):**

- Header: `" R{d} "` = 4 chars
- Values: `fmt_val()` = 4 chars (no trailing space)
- Empty dots: `"  · "` = 4 chars
- Unused: 4 spaces

### 5. Fixed Dot-Value Alignment

Changed dot position to align with value ones digit:

- Old: `" ·  "` (dot at position 2)
- New: `"  · "` (dot at position 3, matching `"  4S"` value format)

### 6. Made Left Margin Dynamic

Left margin now scales with array size to keep B columns aligned with array:

```python
left_margin = 7 + max_depth * 4
```

- Size 4 (max_depth=3): left_margin = 19
- Size 8 (max_depth=7): left_margin = 35

### 7. Fixed Skew Buffer Slot Assignment

Corrected FIFO-to-register mapping for proper wavefront visualization:

When FIFO isn't full, data enters at the deepest register, not R0:

```python
# Old (wrong): fifo[d] → R[d]
# New (correct): fifo[d - depth + len] → R[d]
fifo_idx = d - depth + fifo_len
```

This ensures new data appears in the correct register position (e.g., first data for lane 7 shows in R6, not R0).

### 8. Made K Dimension Default to Size

Changed `--k` argument to default to `--size` for intuitive square matrix behavior:

```python
# --size 8 now gives 8x8 matrices (was 8x4 × 4x8)
if args.k is None:
    args.k = args.size
```

### 9. Fixed Ruff Linting Errors

Cleaned up unused variables and style issues:

- Removed unused `W` and `CELL_W` variables
- Renamed `col` to `_col` in arrow row loop
- Converted `a_data` if-else to ternary expression

## Technical Details

### FIFO Register Mapping Formula

For a lane with `depth` delay stages and `fifo_len` items:

- FIFO fills from input side (deepest register)
- `fifo[i]` is at `R[depth - fifo_len + i]`
- For display `R[d]`, use `fifo[d - depth + fifo_len]`

Example for lane 7 (depth=7) with 1 item:

- `fifo[0]` → `R[7 - 1 + 0]` = `R[6]` (correct: newest at deepest)

### Column Width Standards

| Section | Cell Width | Format |
|---------|------------|--------|
| Skew B header | 5 chars | `" b{col}  "` |
| Skew B data | 5 chars | `fmt_val() + " "` |
| Skew A header | 4 chars | `" R{d} "` |
| Skew A data | 4 chars | `fmt_val()` |
| Array cells | 5 chars | `fmt_pe()` |

### fmt_val Output Format

```
Valid:   "  4S" (space, space, digit, mark) - 4 chars
Invalid: "  · " (space, space, dot, space)  - 4 chars
```

Both formats align the significant content at position 3.

## Files Modified

- `examples/gemm/05_skew_buffer_timing.py` - All visualization fixes

## Test Results

Animation now correctly displays:

```
                              b0   b1   b2   b3   b4   b5   b6   b7
                      R6                                          3S
                      R5                                     3S   ·
                      R4                                1S   ·    ·
                      R3                           2S   ·    ·    ·
                      R2                      4S   ·    ·    ·    ·
                      R1                 1S   ·    ·    ·    ·    ·
                      R0            3S   ·    ·    ·    ·    ·    ·
                     Out →     3S   ·    ·    ·    ·    ·    ·    ·
```

Each column properly shows data entering at the deepest register and propagating toward output.

## Key Insights

### Shift Register vs FIFO Semantics

The SkewBuffer model uses a dynamic FIFO for efficiency, but visualization requires mapping to physical shift register positions. The key insight:

- FIFO[0] is always oldest (next to output when full)
- But when not full, items are at the "deep end" of the shift register
- Display must account for this with the offset formula

### Consistent Column Widths

All cells in a column must have exactly the same visible character width, regardless of content (value, dot, or empty). ANSI escape codes don't count toward width.

## Next Steps

Potential future enhancements:

1. Add cycle-by-cycle log export to file
2. Support non-square arrays (M×N where M≠N)
3. Add throughput metrics to timeline summary
