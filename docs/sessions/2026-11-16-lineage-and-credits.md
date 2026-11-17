# Session: Lineage and Credits

**Date:** 2026-11-16
**Focus:** Record where systars's method comes from, credit Gemmini precisely, and plan the
removal of the Gemmini scaffolding that remains

## Summary

systars reimplements SYSTARS (1988), which synthesised systolic and wavefront arrays from
recurrence equations by space-time mapping. The docs had called Gemmini "systars's origin".
PR #41 adds a lineage page and replaces that framing:

- **Foundations:** SUREs (Karp, Miller, Winograd), the schedule vector τ and the projection P
  (Lamport, Moldovan, Quinton, S. Y. Kung), and SYSTARS.
- **Engineering:** Gemmini, credited for exactly what systars took from it.
- **The KPU:** the schedules beyond a systolic array that it executes, after
  branes-ai/domain_flow.

An audit then found how much Gemmini structure remains. No Gemmini code was translated, but
the scaffolding was Gemmini's. Three cleanup issues were opened in each repo.

| PR / issue | Repo | Result | What |
|---|---|---|---|
| #41 | systars | merged `265c16b` | Lineage page, `NOTICE`, `CITATION.cff`, README "Lineage and credits", docstring credits |
| #42 / esg#96 | both | open | Delete the 16 `SystolicConfig` fields from Gemmini that no RTL reads |
| #43 / esg#97 | both | open | Rename Load/Execute/StoreController to the BlockMover legs and the Streamer |
| #44 / esg#98 | both | open | Organise the tile's control as CSP processes per tile-flow hop |

## The lineage page (`docs/architecture/lineage.md`)

- **Dataflows as projections.** With τ = (1, 1, 1), projecting matmul's (i, j, k) space along
  k, i or j gives systars's OS, WS (B-stationary) and IS (A-stationary) modes. Projecting
  along (1, 1, 1) gives Kung and Leiserson's hexagonal array, with τ·u = 3, so each PE works
  one clock in three. PE(i, j) firing at `i + j + k` is the wavefront of the introduction.
- **Beyond linear schedules.** matmul's free schedule is `max(i, j) + k`: 29 steps for 15³,
  against 43 for the linear schedule. It is not of the form τ·p, so a systolic array, whose
  wiring fixes one linear schedule, cannot execute it, and a domain flow fabric can. The page
  links domain_flow's animations and its piecewise-linear schedules.
- **What it does not claim.** systars today is one τ and three projections. Synthesis from an
  arbitrary (τ, P) is stated as heritage and direction.

## What systars keeps from Gemmini

There is no translated code: the first commit (`880c29b`) wrote the PE and config fresh in
Amaranth, from a plan that was a Python port spec of Gemmini (`docs/plan/implementation.md`,
now marked historical). What carried over:

- the PE / PEArray / SystolicArray hierarchy (Gemmini's PE, Tile and Mesh);
- the `SystolicConfig` field set;
- the controller split and the `StreamReader` / `StreamWriter` names;
- `LocalAddr`'s bits 31, 30 and 29;
- the accumulator and its activation;
- the execute command names.

The KPU work replaced the inside: registered PE links with no skew registers, credits and
tokens instead of a reservation station, the Streamer and L1 buffers, and the BlockMover's
opcodes. #37 and #38 already retire the accumulator and `LocalAddr`. #42–#44 cover the rest,
each kpugen-first (esg#96 waits for kpugen PR #95, which adds the `NOT_READ` list).

## Review

CodeRabbit found that `NOTICE` used generic BSD-3 wording. Gemmini's license has a
Regents-specific disclaimer and a no-maintenance clause. `NOTICE` now carries it verbatim,
from ucb-bar/gemmini at `b82d49c`, checked with `diff`.

## Open

- **Citations.** Check the page numbers on the lineage page, the year of the Yale
  dissertation (left out), and what "jump grids" should mean.
- **`CITATION.cff`** was not validated against the CFF schema locally. GitHub checks it on
  the repository page.
