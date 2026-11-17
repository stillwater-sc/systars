# Introduction to SYSTARS

SYSTARS is a Python generator, written in [Amaranth HDL](https://amaranth-lang.org/), that
emits synthesizable Verilog for a configurable systolic array and the memory and control
around it. Its top level, `SystolicTop`, is a systolic **KPU compute tile**: it moves data the
way the KPU does, pushing whole tiles between memory levels on credits and tile-ready tokens.
Working on SYSTARS is therefore a practical way to learn the KPU's data movement architecture.

These pages are for engineers joining the project who know digital design and Python, but not
necessarily systolic arrays, Amaranth or the KPU. Read them in this order:

1. This page: why systolic arrays exist and what a wavefront is.
2. [The systolic tile](../architecture/systolic-tile.md): what the generator builds, and the
   three dataflows.
3. [Key ideas](../architecture/key-ideas.md): the design decisions that make SYSTARS different
   from a textbook array.
4. [ISA instructions](../isa-instructions.md): how work reaches the tile.
5. [Your first week](onboarding.md): from clone to a passing test and a first change.

The module docstrings and the [ISA page](../isa-instructions.md) are the specification. Where
these introductory pages and the code disagree, the code wins and the page has a bug.

## Systolic arrays

A systolic array wins by reading each operand from memory once and reusing it many times as it
moves from processing element (PE) to PE. Matrix multiply is the canonical case:

```
C[i,j] = Σ A[i,k] × B[k,j]   for k = 0 .. K-1
```

Each element of A is used for a whole row of C and each element of B for a whole column. In an
output-stationary array, A flows east along the rows, B flows south down the columns, and
PE(i, j) multiplies what passes through it and keeps the running sum of C[i, j]. Memory
bandwidth is paid once per operand at the array edge, while compute grows with the number of
PEs.

## The wavefront

For A[i, k] and B[k, j] to meet in PE(i, j), row i's stream must start i clocks after row 0's,
and column j's j clocks after column 0's. Then both reach PE(i, j) in clock `1 + i + j + k`.
For a fixed k, the PEs working on it form an anti-diagonal that moves one PE further each
clock. That moving front is the **wavefront**, and the staggered start is the **skew**. Every
timing rule in SYSTARS follows from this one equation.

The [wavefront animation](../wavefront-animation.md) page shows it on 2x2, 4x4 and 16x16
arrays, recorded from the generated RTL.

## Lineage

The name comes from SYSTARS, a CAD tool for the synthesis and analysis of VLSI
systolic/wavefront arrays built at Delft University of Technology: E.T.L. Omtzigt,
"SYSTARS: A CAD tool for the synthesis and analysis of VLSI systolic/wavefront arrays",
*Proc. International Conference on Systolic Arrays*, 1988
([doi:10.1109/ARRAYS.1988.18078](https://ieeexplore.ieee.org/document/18078)). This
repository is a modern reimplementation of that idea: describe the array in a high-level
language, then generate, analyse and visualise the hardware.

SYSTARS synthesised arrays from recurrence equations by space-time mapping, the method of
Karp, Miller and Winograd, Moldovan, Quinton and S. Y. Kung. The three dataflows systars
builds are three projections in that method. [Lineage](../architecture/lineage.md) explains
it, credits Gemmini for the parts systars took from it, and shows the schedules beyond a
systolic array that the KPU executes.

## The link to the KPU

The Stillwater KPU is a Domain Flow Architecture: it executes systems of recurrence equations
whose indices have a spatial meaning, so data moves through a fabric of PEs as wavefronts. A
systolic array is the special case where the recurrence is fixed (GEMM) and the schedule is
fixed in hardware.

SYSTARS keeps the fabric simple and puts the KPU's ideas in the data movement around it: the
L3/L2/L1 memory hierarchy, the BlockMover and Streamer processes, and credit-based flow control
after the CSP model of [kpu-sim](https://github.com/stillwater-sc/kpu-sim).

## Beyond the systolic tile

The repository also holds models of other parallel architectures, used to compare against the
systolic approach: a line-buffer stencil machine for 2D convolution
([microarchitecture](../design/stencil-machine-microarchitecture.md)), an NVIDIA-style SIMT
streaming multiprocessor ([architecture](../design/simt-architecture.md)) and a MasPar SIMD
array. They are not part of `SystolicTop`; the [codebase map](onboarding.md) says what is what.
