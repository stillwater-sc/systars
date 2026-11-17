# Session: The kpugen Realignment

**Date:** 2026-11-17
**Focus:** Port kpugen's renames and the features systars had missed since 2026-11-10, so
the control rewrite (#44) could be ported 1:1

## Summary

The day started with docs. #46 replaced the `pygemmini/` layout in the plan with the
current repository and gave the Yale dissertation's year.

Then came kpugen's renames: the Combiner and BlockMover names (#37), the deleted Gemmini
config fields (#42) and the CSP process names (#43). They were ported against a base that
had drifted from kpugen since 2026-11-10: `streamer.py` differed by 376 lines and `top.py`
by 352. #44, kpugen's control rewrite, changes exactly those two files.

So the missing features were ported first: weight loading (#34), the banked L2 (#17), the
Combiner (#38), the first-error capture (#51), the trace hooks (#18) and the queued ports
(#52). After them `top.py` differed by 75 lines, all of them systars-only by design, and
the #44 port applied almost cleanly. `blockmover.py`, `result_path.py` and their tests
are identical to kpugen's.

| PR | Issue | kpugen | What |
|---|---|---|---|
| #46 | — | — | The plan's current layout; 1993 for the dissertation |
| #47 | #37 | esg#67 | The Combiner's storage, `blockmover/`, `BlockMoverOpcode` (rename) |
| #48 | #42 | esg#96 | 16 unread Gemmini config fields deleted; `test_config.py` field-read guard |
| #49 | #43 | esg#97 | `BlockMoverIn`, `BlockMoverOut`, `Streamer` (rename) |
| #50 | #34 | esg#55 | Nearest-neighbour weight loading, PRELOAD beside the stream |
| #53 | #17 | esg#24 | A banked L2: two groups of 1R1W banks |
| #54 | #38 (part) | esg#74–#77 | The Combiner: merge, epilogue, MXINT8, four profiles |
| #55 | #51 | esg#84 | The tile's first error, behind the sticky bits |
| #56 | #18 (part) | esg#27 | The trace hooks: `SlotPool.reading`, submodules built in `__init__` |
| #57 | #52 | esg#85 | `*_queued` ports, `Combiner.forward_hit` |
| #58 | #44 | esg#98 | The control as CSP processes: one BlockMover, the result path |

## How the ports were done

- **Path-rewritten diffs.** Most ports applied kpugen's diff with its paths rewritten
  (`kpu_gen.tiles.systolic` → `systars`, and the renamed modules). A small script then
  applied rejected hunks wherever their removed lines appeared verbatim.
- **Through the renames.** #34's kpugen PR predates the renames. It was applied in a worktree
  at the matching pre-rename commit (`fa9888f`), then carried through #37, #42 and #43 by a
  rename-aware cherry-pick. The conflicts were its new lines in the old names.
- **The check.** Every port compared each file's difference from kpugen before and after:
  the differences must stay the same. Renames were also checked on the generated Verilog,
  with identifiers normalized and lines sorted, since a rename reorders ports.
- **systars-only code.** `wavefront.py`'s progression (#34, ported by hand), the cocotb
  `systolic_top` bench (B tiles at the B group, column-major PRELOAD, the per-process
  completion ports) and `blockmover/axi.py`.

## Review fixes ahead of kpugen

CodeRabbit found real bugs in code identical to kpugen's. They are fixed here, with
regression tests, and each has a kpugen issue for the back-port:

- **esg#105:** a CLIP bound outside the narrow range wrapped instead of saturating, and the
  post-SFU result was a fixed 16 bits wide.
- **esg#106:** a SYNC fault at a command's completion named the command started last, or
  the one at the queue head, instead of the one that completed (`SlotPool.error_at_completion`).
- **esg#104**, the other direction: kpugen's `test_machine.py` counts any attribute load as a
  config read; systars's `test_config.py` counts only reads on a config receiver.

## Open

- **#38** stays open, as esg#68 does, for MXFP8 and bias fusion.
- **#18** stays open for kpugen's trace package (lanes, recorder, rendering). It builds on
  the tile-flow recorder (#14).
- **kpugen back-ports:** esg#104, #105 and #106.
- **Not ported:** kpugen's chore PR #80 (cocotb build-speed flags).
