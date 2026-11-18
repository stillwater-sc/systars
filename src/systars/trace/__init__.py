"""Run recording for a systars tile: tile commands, the tile's internal lanes and its first
error, written as one trace bundle (kpu-gen's format, #18).

``TileRecorder`` (``recorder.py``) taps a ``SystolicTop`` in the Amaranth simulator through
the probe map (``probes.py``) and builds a ``Trace`` (``record.py``). ``check_trace``
(``check.py``) checks a trace's invariants, ``render`` draws it as text and SVG, and
``perf.summary`` summarises its performance. ``python -m systars.trace BUNDLE`` reports on a
saved one.
"""

from .record import FORMAT, VERSION, Trace
from .recorder import TileRecorder

__all__ = ["FORMAT", "VERSION", "TileRecorder", "Trace"]
