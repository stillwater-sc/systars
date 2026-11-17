"""
The differential harness's coverage report: which bins a fuzz run hit, and which none did.

``python -m tests.fuzz.coverage RUN_DIR`` reads the per-seed records (``seed-<n>.json``)
of one ``just fuzz`` run and prints the hit counts and the bins left uncovered. It only
reads: other files in the directory (the run's ``manifest.json``, a failed seed's
``seed-<n>.trace.json``) are ignored.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from .program import all_bins


def report(directory: Path) -> tuple[Counter, set[str], int, int]:
    hits: Counter = Counter()
    runs = failed = 0
    records = (p for p in directory.glob("seed-*.json") if not p.name.endswith(".trace.json"))
    for f in sorted(records):
        d = json.loads(f.read_text())
        runs += 1
        failed += not d["ok"]
        hits.update(d["bins"])
    return hits, all_bins() - set(hits), runs, failed


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("usage: python -m tests.fuzz.coverage RUN_DIR", file=sys.stderr)
        return 2
    directory = Path(argv[0])
    hits, missing, runs, failed = report(directory)
    print(f"{runs} programs, {failed} failed, {len(hits)} bins hit, {len(missing)} missed")
    for name in sorted(hits):
        print(f"  {hits[name]:6d}  {name}")
    if missing:
        print("missed:")
        for name in sorted(missing):
            print(f"          {name}")
    return 1 if failed or missing else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
