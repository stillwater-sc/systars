# Systolic Array Wavefront Visualization

This document explains the wavefront data flow pattern in systolic arrays and how to use the visualization tools to understand and share these concepts.

## Table of Contents

1. [Why Wavefront Data Flow?](#why-wavefront-data-flow)
2. [The Skewing Pattern](#the-skewing-pattern)
3. [Cycle-by-Cycle Breakdown](#cycle-by-cycle-breakdown)
4. [Using the Visualization Tools](#using-the-visualization-tools)
5. [Interpreting the Output](#interpreting-the-output)
6. [Generating Shareable GIFs](#generating-shareable-gifs)

---

## Why Wavefront Data Flow?

Systolic arrays are designed to maximize data reuse and minimize memory bandwidth. The key insight is that matrix multiplication has inherent data reuse:

```
C[i,j] = Σ A[i,k] × B[k,j]  for k = 0 to K-1
```

- Each element of **A** is used in computing an entire row of C
- Each element of **B** is used in computing an entire column of C
- Each element of **C** accumulates K partial products

### The Problem with Naive Approaches

A naive implementation would:

1. Fetch A[i,k] and B[k,j] from memory
2. Multiply and accumulate into C[i,j]
3. Repeat for all i, j, k

This requires O(N³) memory accesses for an N×N multiplication.

### The Systolic Solution

A systolic array solves this by:

1. **Flowing data through the array** - Each value is read once and passes through multiple PEs
2. **Stationary accumulators** - Results stay in place while inputs flow past
3. **Wavefront pattern** - Careful timing ensures the right values meet at the right PE

The result: O(N²) memory accesses instead of O(N³).

---

## The Skewing Pattern

The magic of systolic arrays lies in the **skewed input pattern**. For matrix multiplication C = A × B:

### Input Delays

- **Row i of A** is delayed by **i cycles** before entering the left edge
- **Column j of B** is delayed by **j cycles** before entering the top edge

### Visual Example (3×3)

Consider matrices:

```
A = [1 2 3]    B = [1 4 7]
    [4 5 6]        [2 5 8]
    [7 8 9]        [3 6 9]
```

The skewed input pattern over time:

```
Time →    t=0   t=1   t=2   t=3   t=4   t=5   t=6

A inputs (entering from left):
  Row 0:   1     2     3     ·     ·     ·     ·
  Row 1:   ·     4     5     6     ·     ·     ·
  Row 2:   ·     ·     7     8     9     ·     ·

B inputs (entering from top):
  Col 0:   1     2     3     ·     ·     ·     ·
  Col 1:   ·     4     5     6     ·     ·     ·
  Col 2:   ·     ·     7     8     9     ·     ·
```

### Why This Works

At cycle t, PE[i,j] receives:

- A value that entered the array at cycle (t - j) from row i
- B value that entered the array at cycle (t - i) from column j

For the values to "meet" at PE[i,j] at time t:

- A[i,k] enters at cycle k, arrives at column j at cycle k + j
- B[k,j] enters at cycle k, arrives at row i at cycle k + i

Both arrive at PE[i,j] at cycle k + i + j. Since this happens for all k values (0 to N-1), PE[i,j] correctly computes:

```
C[i,j] = A[i,0]×B[0,j] + A[i,1]×B[1,j] + A[i,2]×B[2,j] + ...
```

---

## Cycle-by-Cycle Breakdown

Let's trace through a 3×3 multiplication step by step.

### Initial State (Cycle 0)

```
         B inputs: [1] [ ] [ ]
                    ↓
A inputs: [1] →  [PE00] [PE01] [PE02]
          [ ]    [PE10] [PE11] [PE12]
          [ ]    [PE20] [PE21] [PE22]

PE states: All idle, C = 0 everywhere
```

### Cycle 1

```
         B inputs: [2] [4] [ ]
                    ↓   ↓
A inputs: [2] →  [1×1] [a=1] [    ]
          [4] →  [    ] [    ] [    ]
          [ ]    [    ] [    ] [    ]

PE00: Computing 1×1=1, c=1
PE01: Has a=1 (from PE00), waiting for b
PE10: Has b=1 (from PE00), waiting for a
```

### Cycle 2

```
         B inputs: [3] [5] [7]
                    ↓   ↓   ↓
A inputs: [3] →  [2×2] [1×4] [a=1]
          [5] →  [4×1] [    ] [    ]
          [7] →  [    ] [    ] [    ]

PE00: c = 1 + 2×2 = 5
PE01: c = 0 + 1×4 = 4
PE10: c = 0 + 4×1 = 4
PE02: Has a=1, waiting for b=7
PE11: Has a=4, b=2, will compute next
PE20: Has b=1, waiting for a=7
```

### Cycle 3

```
PE00: c = 5 + 3×3 = 14  ✓ (Done: C[0,0] = 1×1 + 2×2 + 3×3 = 14)
PE01: c = 4 + 2×5 = 14
PE02: c = 0 + 1×7 = 7
PE10: c = 4 + 5×2 = 14
PE11: c = 0 + 4×4 = 16
PE20: c = 0 + 7×1 = 7
...
```

### Final Result (Cycle 7)

All PEs have accumulated their final values:

```
C = [14  32  50]    (matches A @ B)
    [32  77 122]
    [50 122 194]
```

### Wavefront Pattern

Notice how computation moves diagonally through the array:

```
Cycle when each PE first computes:

       Col0  Col1  Col2
Row0 [  0     1     2  ]
Row1 [  1     2     3  ]
Row2 [  2     3     4  ]
```

This diagonal pattern is the **wavefront** - it sweeps from top-left to bottom-right.

---

## Using the Visualization Tools

### Terminal Animation

The terminal-based animation shows the wavefront in real-time:

```bash
# Basic animation (4×4 array)
python examples/gemm/02_animated_wavefront.py

# Smaller array, easier to follow
python examples/gemm/02_animated_wavefront.py --size 3

# Step-by-step mode (press Enter for each cycle)
python examples/gemm/02_animated_wavefront.py --step --size 3

# Faster animation
python examples/gemm/02_animated_wavefront.py --delay 200

# Without colors (for basic terminals)
python examples/gemm/02_animated_wavefront.py --no-color
```

### What You'll See

```
Cycle 2
============================================================

  B inputs (flowing ↓):
       [3]   [5]   [7]

  2 → [2×2   1×4   a=1 ]
  5 → [4×1    ·     ·  ]
  7 → [ ·     ·     ·  ]

  Accumulator C:
       [  5     4     0  ]
       [  4     0     0  ]
       [  0     0     0  ]
```

### GIF Generator

For sharing on Slack, in documentation, or presentations:

```bash
# Install dependencies
pip install matplotlib pillow

# Generate GIF
python examples/gemm/03_wavefront_gif.py --output wavefront.gif

# Customize
python examples/gemm/03_wavefront_gif.py \
    --size 4 \
    --fps 2 \
    --dpi 150 \
    --output my_animation.gif

# Preview in window first
python examples/gemm/03_wavefront_gif.py --show
```

---

## Interpreting the Output

### Terminal Animation Colors

| Color | Meaning |
|-------|---------|
| **Green background** | PE is actively computing (a×b) |
| **Blue background** | PE has partial data (a or b, not both) |
| **No background** | PE is idle |
| **Yellow text** | A values (flowing right) |
| **Cyan text** | B values (flowing down) |
| **Green text** | Accumulated C values |

### PE State Display

```
[2×3]  - PE is computing: a=2, b=3
[a=5]  - PE has a value, waiting for b
[b=7]  - PE has b value, waiting for a
[ · ]  - PE is idle
```

### GIF Panels

The generated GIF shows three panels:

1. **Left Panel - Input Matrices**
   - Shows A and B matrices
   - Highlighted cells indicate values currently being fed
   - Yellow = A values, Blue = B values

2. **Middle Panel - PE Array**
   - Shows current state of each PE
   - Green = computing, Blue = has data, Gray = idle
   - Displays current operation (e.g., "2×3")

3. **Right Panel - Accumulator**
   - Shows C values building up
   - Intensity indicates magnitude
   - Final values match expected result

---

## Generating Shareable GIFs

### Quick Start

```bash
# Default 4×4 array, 2 FPS
python examples/gemm/03_wavefront_gif.py --output wavefront.gif
```

### Recommended Settings

| Use Case | Command |
|----------|---------|
| Slack/quick share | `--size 3 --fps 2 --dpi 100` |
| Documentation | `--size 4 --fps 1 --dpi 150` |
| Presentation | `--size 4 --fps 1 --dpi 200` |
| Detailed study | `--size 3 --fps 0.5 --dpi 100` |

### File Size Guide

| Size | Frames | Typical File Size |
|------|--------|-------------------|
| 3×3 | 10 | ~200 KB |
| 4×4 | 13 | ~300 KB |
| 5×5 | 17 | ~450 KB |
| 6×6 | 21 | ~600 KB |

### Alternative: Record Terminal

For a more "authentic" look, record the terminal animation:

```bash
# Using asciinema + agg
asciinema rec demo.cast -c "python examples/gemm/02_animated_wavefront.py --size 3"
agg demo.cast demo.gif

# Using VHS (Charmbracelet)
# Create a tape file first, then:
vhs < demo.tape

# Using termtosvg (creates SVG, not GIF)
termtosvg -c "python examples/gemm/02_animated_wavefront.py --size 3"
```

---

## Key Takeaways

1. **Wavefront = diagonal sweep** - Computation moves from top-left to bottom-right
2. **Skewing creates synchronization** - Input delays ensure correct values meet
3. **Data flows, results stay** - A flows right, B flows down, C accumulates in place
4. **Total cycles = 2N-1 + N** - For an N×N array: N cycles to fill, N-1 to drain, N for computation overlap
5. **Maximum parallelism at center** - All N² PEs active when wavefront fills the array

## Further Reading

- Original systolic array paper: H.T. Kung, "Why Systolic Architectures?" (1982)
- Google TPU architecture: Uses systolic arrays for ML inference
- Gemmini project: Open-source systolic array generator (Berkeley)
