"""
The trace bundle: one run's manifest and tables, written as columnar JSON.

A bundle is one JSON object::

    {"format": "kpu-gen-trace", "version": 3,
     "manifest": {"level": "RTL", "producer": "systars <sha>", "cycles": ..., "geometry": {...}},
     "tables": {"packets": {"id": [...], "net": [...], ...}, "intervals": {...}, ...}}

Each table is stored by column (a list per field, every list the same length),
as kpu-sim's records are. Rows are plain ``dict``\\ s in memory. A table's
columns are the union of its rows' fields, so a row without a field (an
annotation of another kind, say) reads back with that field as ``None``.

Tables
    ``packets``      every NoC packet: network, type, physical source and
                     destination, address, words, tag, error flag, the cycles
                     it entered and left the network, and its hops (link names)
    ``hops``         one row per packet per link: when its head crossed and
                     when its last flit did
    ``transfers``    every SoC DMA transfer, with the L3 words it touched
    ``commands``     every tile command, from accept to completion, with the
                     AXI transactions the tile issued for it
    ``intervals``    resource occupancy: ``resource`` busy from ``open`` to
                     ``close`` (cycles) for ``client``, and the event behind it
    ``residencies``  where each annotated tensor block was, at which address,
                     from when to when (``close`` -1: until the end of the run)
    ``annotations``  what the issuer asked for (its observer events)
    ``values``       result blocks: actual values beside the expected ones
    ``errors``       each tile's first error (#84): the cycle it happened, the site, its
                     source and kind, the command named (its ``commands`` row too)

The vocabulary (events with IDs, processes, layers, residencies at addresses,
cycles on a timed level) follows kpu-sim's shared tile-flow record
(stillwater-sc/kpu-sim#386). That format is not specified yet, so this bundle
is kpu-gen's own, versioned here, and an exporter to it can be added beside
this module once it is.

systars writes the same bundle (``FORMAT`` and ``VERSION`` are kpu-gen's, #18), so the two
tools read each other's: a systars trace is one tile's (``scope`` ``"tile"``), with the SoC
tables (``packets``, ``hops``, ``transfers``) empty and ``producer`` ``systars <sha>``.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

FORMAT = "kpu-gen-trace"
VERSION = 3
"""2 (#67): the Combiner's residency layer is ``CMB`` (was ``ACC``). 3 (#84): the ``errors``
table. A version 2 bundle still reads, with no errors."""
READS = (2, 3)


def _regenerate(fmt: object) -> str:
    """A hint for a file the tool wrote under its old name, kpugen (#70)."""
    if isinstance(fmt, str) and fmt.startswith("kpugen-"):
        return ": written before kpugen was renamed kpu-gen; regenerate it"
    return ""


TABLES = (
    "packets",
    "hops",
    "transfers",
    "commands",
    "intervals",
    "residencies",
    "annotations",
    "values",
    "errors",
)


def producer() -> str:
    """``systars <sha>`` (``systars <version>`` outside a git checkout)."""
    from systars import __version__

    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).parent,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        sha = __version__
    return f"systars {sha}"


@dataclass
class Trace:
    """One run: ``manifest`` and a list of row dicts per table."""

    manifest: dict[str, Any]
    tables: dict[str, list[dict[str, Any]]] = field(
        default_factory=lambda: {name: [] for name in TABLES}
    )

    def __getitem__(self, table: str) -> list[dict[str, Any]]:
        """The rows of ``table``."""
        return self.tables[table]

    def to_json(self) -> dict[str, Any]:
        """The bundle as a JSON-able object, tables by column."""
        columns = {}
        for name, rows in self.tables.items():
            keys: list[str] = []
            for row in rows:
                keys.extend(k for k in row if k not in keys)
            columns[name] = {k: [row.get(k) for row in rows] for k in keys}
        return {"format": FORMAT, "version": VERSION, "manifest": self.manifest, "tables": columns}

    @classmethod
    def from_json(cls, obj: dict[str, Any]) -> Trace:
        """Read a bundle written by ``to_json`` (``ValueError`` if it is not one)."""
        if not isinstance(obj, dict):
            raise ValueError(f"not a {FORMAT} bundle (a JSON {type(obj).__name__})")
        if obj.get("format") != FORMAT:
            raise ValueError(
                f"not a {FORMAT} bundle (format {obj.get('format')!r}){_regenerate(obj.get('format'))}"
            )
        if obj.get("version") not in READS:
            raise ValueError(
                f"{FORMAT} version {obj.get('version')}: this reader reads "
                + " and ".join(map(str, READS))
            )
        if not isinstance(obj.get("manifest"), dict) or not isinstance(obj.get("tables"), dict):
            raise ValueError(f"{FORMAT} bundle: the manifest and the tables must be objects")
        tables: dict[str, list[dict[str, Any]]] = {name: [] for name in TABLES}
        for name, cols in obj["tables"].items():
            if not isinstance(cols, dict) or not all(isinstance(v, list) for v in cols.values()):
                raise ValueError(f"table {name}: not an object of column lists")
            lengths = {len(v) for v in cols.values()}
            if len(lengths) > 1:
                raise ValueError(f"table {name}: columns of different lengths {sorted(lengths)}")
            n = lengths.pop() if lengths else 0
            tables[name] = [{k: v[i] for k, v in cols.items()} for i in range(n)]
        return cls(obj["manifest"], tables)

    def save(self, path: str | Path) -> Path:
        """Write the bundle to ``path`` (JSON)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), separators=(",", ":")), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> Trace:
        """Read a bundle written by ``save``."""
        return cls.from_json(json.loads(Path(path).read_text(encoding="utf-8")))
