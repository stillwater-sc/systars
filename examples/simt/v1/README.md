# Pedagogical Example of a SIMT design with poor latency hiding

Changes to examples/simt/v1/01_animated_simt.py:

  1. Updated TimelineLogger with Chrome Trace format support:
    - Same dual-format capability (csv and chrome)
    - Added log_lsu_drop() method specific to v1's bug (shows DROPPED events in trace)
    - Partition names show "(LSU max_pending=2)" as a reminder of the v1 limitation
    - Metadata notes the LSU queue overflow bug
  2. Added --timeline-format argument:

```bash
  # Chrome Trace format (default)
  python examples/simt/v1/01_animated_simt.py --tiled -m 8 -n 8 --k 2 \
      --fast --fast-mem --timeline v1_trace.json

  # CSV format
  python examples/simt/v1/01_animated_simt.py --timeline events.csv --timeline-format csv
```

Pedagogical Value:

  The v1 Chrome Trace is especially useful for demonstrating:

- Missing LD completions (instructions that never complete due to queue overflow)
- Gaps in the timeline where warps stall waiting for dropped instructions
- The LD_DROPPED events (if you add calls to log_lsu_drop() in the partition code)

Standalone:

- Uses systars.simt_v1 module (separate from main systars.simt)
- All 57 SIMT tests still pass (v1 is isolated)

## LSU Enhancement

```bash
> python examples/simt/v1/01_animated_simt.py --tiled --m 4 --n 4 --k 2 --fast --max-cycles 100 --lsu-queue-depth 4 --mem-latency 20 2>&1 |  timeout:

  ⎿  LSU queue depth: 4 (default buggy value is 2)
     Global memory latency: 20 cycles
     Cycle    1 │ State: EXECUTE  │ Instructions:    0 │ Bank Conflicts:  16 │ Energy: 82.0 pJ
     … +347 lines (ctrl+o to expand)
```

The new CLI options are working. Here's a summary:

New CLI options for v1 example:

```bash
  # Default buggy behavior (queue depth = 2)
  python examples/simt/v1/01_animated_simt.py --tiled --m 8 --n 8 --k 2 --fast

  # Increase LSU queue depth to fix the bug
  python examples/simt/v1/01_animated_simt.py --tiled --m 8 --n 8 --k 2 --fast \
      --lsu-queue-depth 8

  # Custom memory latency (e.g., 50 cycles instead of 200)
  python examples/simt/v1/01_animated_simt.py --tiled --m 8 --n 8 --k 2 --fast \
      --mem-latency 50

  # Compare traces: buggy vs fixed
  python examples/simt/v1/01_animated_simt.py --tiled --m 8 --n 8 --k 4 --fast \
      --lsu-queue-depth 2 --timeline v1_bug_latency_hiding.json

  python examples/simt/v1/01_animated_simt.py --tiled --m 8 --n 8 --k 4 --fast \
      --lsu-queue-depth 8 --timeline v1_fixed.json
```

Changes made:

  1. src/systars/simt_v1/config.py: Added lsu_max_pending config parameter with documentation about why 2 is inadequate
  2. src/systars/simt_v1/partition.py: Updated to use config.lsu_max_pending when creating the LSU
  3. examples/simt/v1/01_animated_simt.py: Added CLI options:
    - --lsu-queue-depth N - Set LSU max pending (default: 2)
    - --mem-latency CYCLES - Set global memory latency (default: 200)

  This makes the v1 example a great pedagogical tool for demonstrating how queue depth affects latency hiding in data-parallel architectures.
