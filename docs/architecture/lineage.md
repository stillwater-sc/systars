# Lineage: from recurrence equations to SYSTARS

SYSTARS generates systolic arrays, but systolic arrays are not where its ideas start. They
start with recurrence equations, and with a method for turning a recurrence into hardware by
choosing *when* each operation runs and *where*. This page sets out that lineage, shows that
the three dataflows systars implements today are three choices in that method, and credits
the work systars builds on, Gemmini included.

## Systems of uniform recurrence equations

Karp, Miller and Winograd defined the **system of uniform recurrence equations** (SURE) in
1967. A SURE computes variables over the integer points of an index domain, and every
variable reads its operands at a **constant displacement**, a dependence vector, from its own
point. Matrix multiply over the domain `0 <= i < M, 0 <= j < N, 0 <= k < K` is:

```text
a(i,j,k) = a(i,j-1,k)                        # A[i,k] moves along +j    d_a = (0,1,0)
b(i,j,k) = b(i-1,j,k)                        # B[k,j] moves along +i    d_b = (1,0,0)
c(i,j,k) = c(i,j,k-1) + a(i,j-1,k)*b(i-1,j,k) # C[i,j] reduces along +k  d_c = (0,0,1)
```

Because every dependence is a constant vector, every value travels a fixed distance in a
fixed direction. That is what makes a SURE map onto a regular grid of PEs with
nearest-neighbour wires. The
[domain_flow matmul page](https://branes-ai.github.io/domain_flow/theory/matmul/) develops
this system, its batched and higher-order tensor forms, and an executable version.

## Space-time mapping: the schedule τ and the projection P

Lamport's hyperplane method (1974) scheduled a loop nest with a linear function of its
indices. Moldovan (1983), Quinton (1984) and S. Y. Kung developed the same idea into a
synthesis method for systolic arrays, the **space-time transformation**:

- A **schedule vector** τ sets the time of point `p` to `t(p) = τ·p`. It is legal when every
  dependence advances time, `τ·d >= 1` for each dependence vector `d`.
- A **projection** along a direction `u` assigns a PE to each point: the points on one line
  parallel to `u` share a PE, and a projection matrix `P` with `P·u = 0` gives the PE
  coordinates. It is legal when `τ·u != 0`, so the points that share a PE fire at different
  times.
- Each dependence becomes a link of the array: `P·d` is the direction it points in, and `τ·d`
  is the number of registers on it. A dependence that projects to zero is a value that stays
  in its PE, a **stationary** operand.

Every legal pair (τ, P) is an array. For matrix multiply with `τ = (1, 1, 1)`, the projection
alone decides what the array is:

| Projection `u` | PE grid | Stationary | Moving | The array |
|---|---|---|---|---|
| `(0, 0, 1)`, along k | (i, j) | C | A east, B south | **Output-stationary**: systars's OS mode |
| `(1, 0, 0)`, along i | (j, k) | B | A east, partial sums south | **B-stationary**: systars's WS mode |
| `(0, 1, 0)`, along j | (i, k) | A | B and partial sums | **A-stationary**: systars's IS mode |
| `(1, 1, 1)`, diagonal | hexagonal | none | A, B and C on three axes | Kung and Leiserson's **hexagonal array** (1978) |

With projection along k, PE(i, j) fires point `(i, j, k)` at clock `i + j + k`. That is the
wavefront in the [introduction](../getting-started/introduction.md#the-wavefront), and the
skew the L1 stream buffers produce. The diagonal projection has `τ·u = 3`, so each PE of the
hexagonal array works one clock in three and three problems can be interleaved. Projections
for which `P·d` is not a unit vector give links that skip over PEs, and other schedules give
other register counts on the same links. The design space is the set of all legal (τ, P).

Mesh arrays, including systars's today, the TPU and Gemmini, use three points of this space.

## SYSTARS, 1988

E. T. L. Omtzigt built SYSTARS at Delft University of Technology as a CAD tool for the
synthesis and analysis of systolic and wavefront arrays by this method. It took a recurrence
and a choice of space-time mapping, and produced the array and its analysis. The work
continued in Omtzigt's Yale dissertation on domain flow and streaming architectures, which
leads to the KPU.

This repository is the modern reimplementation. It generates synthesizable RTL with Amaranth,
verifies it in simulation and on Verilator, and records the wavefront from the RTL.

## Beyond linear schedules: the KPU

A systolic array executes exactly one linear schedule. Every link is a register and a clock,
so a value that travels `n` links arrives `n` clocks later, and `t = τ·p` is built into the
wiring. Schedules that are not linear need a fabric that fires each operation when its
operands arrive. That fabric is the KPU's domain flow architecture, and it is the KPU's main
difference from the TPU and other systolic-array fabrics.

The **free schedule** is the earliest time at which each operation can fire, given only its
dependences. For the matrix multiply above it is

```text
t(c(i,j,k)) = max(i, j) + k
```

a maximum of two linear forms. No τ produces it. For a 15 x 15 x 15 domain it finishes in 29
steps, where the linear schedule `τ = (1, 1, 1)` takes 43. The free schedule pays for its lower
latency with wider wavefronts and more live values. The domain_flow
[matmul page](https://branes-ai.github.io/domain_flow/theory/matmul/#watch-the-schedule)
animates both schedules side by side and colours each dependence by its slack. The domain_flow
simulator also checks piecewise-linear schedules, which suit piecewise-uniform systems such as
LU and the regularised iterative solvers on its
[systolic SURE transforms](https://branes-ai.github.io/domain_flow/theory/systolic-transforms/)
page.

systars implements the linear corner: one τ and three projections, in a fabric with no flow
control. Synthesis from an arbitrary (τ, P), and the schedules only a domain flow fabric can
execute, are the direction that this lineage points in.

## Gemmini

systars began as `pygemmini`, a Python plan modelled on Gemmini, the open-source systolic
generator from UC Berkeley (Genc et al., DAC 2021). Gemmini made a full-stack DNN accelerator,
from the array to the software, something anyone can generate, run and measure. No Gemmini
code was translated: systars is written from scratch in Amaranth. Its scaffolding, though,
follows Gemmini's, and systars owes it the following:

- the hierarchy PE, `PEArray`, `SystolicArray`, after Gemmini's PE, Tile and Mesh
- the parameter set of `SystolicConfig` (scratchpad and accumulator capacity and banks, DMA
  and queue sizes), after Gemmini's array configuration
- the execute command set: `CONFIG`, `PRELOAD`, `COMPUTE` and `COMPUTE_ACCUMULATE`
  (`controller/execute.py`)
- the local address encoding, where bit 31 selects the accumulator, bit 30 accumulates and
  bit 29 reads a full row (`memory/local_addr.py`)
- the accumulator, a separate result memory with activation on read-out
  (`memory/accumulator.py`)
- the split into load, execute and store controllers, the DMA engines `StreamReader` and
  `StreamWriter`, and the plan for `LoopMatmul` and `LoopConv` unrollers
  ([implementation plan](../plan/implementation.md))

Since then systars has followed the KPU rather than Gemmini, and the interior is new. Every
PE link is registered and the array has no skew or deskew registers. Whole tiles move between
memory levels on credits and tile-ready tokens (the CSP model of
[kpu-sim](https://github.com/stillwater-sc/kpu-sim)) instead of through a reservation station,
and the Streamer stages operands in L1 stream buffers. The BlockMover's opcodes (strided and
transposing moves, `CLEAR`, `BROADCAST`) are not Gemmini's. The [Combiner](combiner.md) will replace the accumulator. Gemmini's copyright
notice and license are in the repository's `NOTICE` file.

## References

1. R. M. Karp, R. E. Miller and S. Winograd, "The Organization of Computations for Uniform
   Recurrence Equations," *Journal of the ACM*, 14(3):563–590, 1967.
2. L. Lamport, "The Parallel Execution of DO Loops," *Communications of the ACM*,
   17(2):83–93, 1974.
3. H. T. Kung and C. E. Leiserson, "Systolic Arrays (for VLSI)," in I. S. Duff and
   G. W. Stewart (eds.), *Sparse Matrix Proceedings 1978*, SIAM, 1979, pp. 256–282.
4. D. I. Moldovan, "On the Design of Algorithms for VLSI Systolic Arrays," *Proceedings of
   the IEEE*, 71(1):113–120, 1983.
5. P. Quinton, "Automatic Synthesis of Systolic Arrays from Uniform Recurrent Equations,"
   *Proc. 11th International Symposium on Computer Architecture*, 1984, pp. 208–214.
6. D. I. Moldovan and J. A. B. Fortes, "Partitioning and Mapping Algorithms into Fixed Size
   Systolic Arrays," *IEEE Transactions on Computers*, C-35(1):1–12, 1986.
7. S. Y. Kung, *VLSI Array Processors*, Prentice Hall, 1988.
8. E. T. L. Omtzigt, "SYSTARS: A CAD Tool for the Synthesis and Analysis of VLSI
   Systolic/Wavefront Arrays," *Proc. International Conference on Systolic Arrays*, 1988
   ([doi:10.1109/ARRAYS.1988.18078](https://ieeexplore.ieee.org/document/18078)).
9. E. T. L. Omtzigt, *Domain Flow and Streaming Architectures: A Paradigm for Efficient
   Parallel Computation*, PhD dissertation, Yale University.
10. H. Genc et al., "Gemmini: Enabling Systematic Deep-Learning Architecture Evaluation via
    Full-Stack Integration," *Proc. 58th Design Automation Conference (DAC)*, 2021.
11. branes-ai/domain_flow: [documentation](https://branes-ai.github.io/domain_flow/) and
    [source](https://github.com/branes-ai/domain_flow). SURE specifications, schedule
    legality, free and piecewise-linear schedules, and their visualisations.
