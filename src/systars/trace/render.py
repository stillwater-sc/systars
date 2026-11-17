"""
Renderings of a ``Trace``: a text report and three SVG views.

``text_report``   counts, per-link traffic, the packet list, the DMA transfers
                  and the residencies, as plain text (also the table view of
                  every SVG below)
``svg_links``     the two networks as router grids, each link drawn with its
                  flit count (darker = more flits)
``svg_timeline``  resource occupancy over cycles: one lane per resource (the
                  DMA, each tile controller, each memory-cell port, each busy
                  NoC link), a bar per interval, and each tile's first error
                  (#84) marked at its cycle
``svg_heatmap``   payload words moved over the NoC, source site by destination
                  site

The SVGs are standalone files: they follow the viewer's light or dark scheme
(``prefers-color-scheme``) and every mark has a ``<title>`` tooltip. Colours
are the dataviz reference palette: categorical slots 1-4 in fixed order for
the timeline's resource groups, the blue sequential ramp for magnitudes, and the
reserved status-critical red for an error marker, always with its icon and label.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from html import escape
from pathlib import Path
from typing import Any

from .record import Trace

_STYLE = """
<style>
  svg { --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
        --grid: #e1e0d9; --axis: #c3c2b7;
        --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; --s4: #eda100;
        --s5: #e87ba4; --s6: #008300; --s7: #6250d6; --s8: #e34948;
        --critical: #d03b3b;
        --q0: #f0efec; --q1: #cde2fb; --q2: #9ec5f4; --q3: #6da7ec; --q4: #3987e5;
        --q5: #256abf; --q6: #184f95; --q7: #0d366b; }
  @media (prefers-color-scheme: dark) {
    svg { --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7; --muted: #898781;
          --grid: #2c2c2a; --axis: #383835;
          --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500;
          --s5: #d55181; --s6: #0ca30c; --s7: #9085e9; --s8: #e66767;
          --q0: #2c2c2a; --q1: #0d366b; --q2: #104281; --q3: #184f95; --q4: #1c5cab;
          --q5: #2a78d6; --q6: #5598e7; --q7: #9ec5f4; }
  }
  text { font-family: system-ui, -apple-system, "Segoe UI", sans-serif; fill: var(--ink); }
  .t2 { fill: var(--ink-2); } .mut { fill: var(--muted); }
  .num { font-variant-numeric: tabular-nums; }
  .on-dark { fill: #ffffff; }
  @media (prefers-color-scheme: dark) { .on-dark { fill: #0b0b0b; } }
</style>
"""

# Sequential steps (light -> dark), and the ink that reads on each
_RAMP = 8
_ON_DARK_FROM = 5  # steps q5..q7 take the inverse ink (.on-dark)


def _svg(width: int, height: int, body: list[str], title: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">'
        f"<title>{escape(title)}</title>{_STYLE}"
        f'<rect width="{width}" height="{height}" fill="var(--surface)"/>'
        + "".join(body)
        + "</svg>"
    )


def _step(value: float, peak: float) -> int:
    """Sequential step 1..7 for ``value`` of ``peak`` (0 for nothing)."""
    if value <= 0 or peak <= 0:
        return 0
    return min(_RAMP - 1, 1 + int((value / peak) * (_RAMP - 2) + 0.5))


def _site(trace: Trace, row: int, col: int) -> int:
    return row * trace.manifest["geometry"]["cols"] + col


# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------


def link_traffic(trace: Trace) -> dict[str, tuple[int, int]]:
    """Per link: ``(packets, flits)`` it carried."""
    packets: Counter[str] = Counter()
    flits: Counter[str] = Counter()
    by_id = {p["id"]: p for p in trace["packets"]}
    for h in trace["hops"]:
        packets[h["link"]] += 1
        flits[h["link"]] += by_id.get(h["packet"], {}).get("flits", 0)
    return {link: (packets[link], flits[link]) for link in packets}


def text_report(trace: Trace, *, max_rows: int = 40) -> str:
    """The trace as plain text."""
    m, g = trace.manifest, trace.manifest["geometry"]
    if "tile" in g:  # one tile (TileRecorder)
        t = g["tile"]
        shape = (f"one tile: {t['array'][0]}x{t['array'][1]} array  L2 {t['l2_rows']} rows "
                 f"(B group from row {t['l2_group_rows']})  {t['l2_slots']} L2 and "
                 f"{t['combiner_slots']} Combiner slots")  # fmt: skip
    else:
        shape = (f"grid {g['rows']}x{g['cols']}  {g['word_bytes']}-byte words  "
                 f"L3 {g['l3_words']} words per site")  # fmt: skip
    out = [
        f"{m['producer']}  level {m['level']}  {m['cycles']} cycles",
        shape,
        "",
        "  ".join(f"{name} {len(rows)}" for name, rows in trace.tables.items()),
    ]
    if m.get("problems"):
        out += ["", "recorder problems:", *(f"  {p}" for p in m["problems"])]
    errors = sorted(trace["errors"], key=lambda e: (e["cycle"], e["site"]))
    if errors:
        e = errors[0]
        out += ["", f"first error: site {e['site']}: {e['text']}"]

    if errors:
        out += ["", "tile errors", f"  {'cycle':>7}{'site':>5}  {'source':<7}{'kind':<28}command"]
        for e in errors:
            cmd = "-" if e["cmd_type"] is None else f"{e['cmd_type']} {e['cmd_id']}"
            if e["command"] is not None:
                cmd += f" (commands row {e['command']})"
            more = "  +another source" if e["multi"] else ""
            out.append(
                f"  {e['cycle']:>7}{e['site']:>5}  {e['source']:<7}{e['kind']:<28}{cmd}{more}"
            )

    if "tile" in g:  # one tile: no NoC, DMA or residencies
        return "\n".join(out) + "\n"

    out += ["", "NoC links (busiest first)", f"  {'link':<22}{'packets':>8}{'flits':>7}"]
    traffic = sorted(link_traffic(trace).items(), key=lambda kv: (-kv[1][1], kv[0]))
    out += [f"  {link:<22}{n:>8}{f:>7}" for link, (n, f) in traffic]

    out += ["", "packets", f"  {'id':>4} {'net':<4}{'type':<7}{'src':<10}{'dst':<10}"
            f"{'addr':>6}{'words':>6}{'in':>7}{'out':>7}  hops"]  # fmt: skip
    for p in trace["packets"][:max_rows]:
        src, dst = "{},{}.{}".format(*p["src"]), "{},{}.{}".format(*p["dst"])
        err = " ERROR" if p["error"] else ""
        out.append(
            f"  {p['id']:>4} {p['net']:<4}{p['type']:<7}{src:<10}{dst:<10}{p['addr']:>6}"
            f"{p['words']:>6}{p['inject']:>7}{p['eject']:>7}  {len(p['path']) - 2}{err}"
        )
    if len(trace["packets"]) > max_rows:
        out.append(f"  ... {len(trace['packets']) - max_rows} more")

    out += ["", "DMA transfers", f"  {'id':>4} {'dir':<12}{'site':>5}{'l3':>6}{'len':>5}"
            f"{'dram':>10}{'open':>7}{'close':>7}"]  # fmt: skip
    for t in trace["transfers"]:
        what = "L3 -> DRAM" if t["store"] else "DRAM -> L3"
        out.append(
            f"  {t['id']:>4} {what:<12}{t['site']:>5}{t['l3_addr']:>6}{t['len']:>5}"
            f"{t['dram_addr']:>#10x}{t['open']:>7}{t['close']:>7}"
            + (" ERROR" if t["error"] else "")
        )

    out += ["", "residencies", f"  {'block':<9}{'layer':<6}{'site':>5}{'addr':>6}{'words':>6}"
            f"{'open':>7}{'close':>7}  event"]  # fmt: skip
    for r in sorted(trace["residencies"], key=lambda r: (r["open"], r["layer"])):
        tile = "{operand}[{i},{j}]".format(**r["tile"])
        close = "end" if r["close"] == -1 else r["close"]
        out.append(
            f"  {tile:<9}{r['layer']:<6}{r['site']:>5}{r['addr']:>6}{r['words']:>6}"
            f"{r['open']:>7}{close!s:>7}  {r['event']}"
        )
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# Link map
# ---------------------------------------------------------------------------


def svg_links(trace: Trace) -> str:
    """Both networks as router grids, every link labelled with the flits it carried."""
    g = trace.manifest["geometry"]
    rows, cols = g["rows"], g["cols"]
    traffic = link_traffic(trace)
    peak = max((f for _, f in traffic.values()), default=0)
    pitch, node, pad, top = 150, 46, 40, 96
    panel_w = cols * pitch + pad
    width, height = 2 * panel_w + pad, top + (rows - 1) * pitch + node + 40
    body = [
        f'<text x="{pad}" y="28" font-size="16" font-weight="600">NoC link traffic</text>',
        f'<text x="{pad}" y="48" font-size="12" class="t2">flits per link over '
        f"{trace.manifest['cycles']} cycles; darker links carried more</text>",
    ]

    def centre(px: int, r: int, c: int) -> tuple[float, float]:
        return px + pad + c * pitch + node / 2, top + r * pitch + node / 2

    for n, net in enumerate(("req", "rsp")):
        px = n * panel_w
        name = "requests" if net == "req" else "responses"
        body.append(f'<text x="{px + pad}" y="{top - 22}" font-size="13" font-weight="600">'
                    f"{name}</text>")  # fmt: skip
        for link in g["links"]:
            if link["net"] != net or link["kind"] != "hop":
                continue
            (r0, c0), (r1, c1) = link["router"], link["to"]
            x0, y0 = centre(px, r0, c0)
            x1, y1 = centre(px, r1, c1)
            # the two directions of a neighbour pair sit side by side
            dx, dy = (0, 7) if r0 == r1 else (7, 0)
            sign = 1 if (c1 > c0 or r1 > r0) else -1
            ox, oy = dx * sign, dy * sign
            sx, sy = (x1 - x0), (y1 - y0)
            length = (sx * sx + sy * sy) ** 0.5
            ux, uy = sx / length, sy / length
            a = (x0 + ux * node * 0.6 + ox, y0 + uy * node * 0.6 + oy)
            b = (x1 - ux * node * 0.6 + ox, y1 - uy * node * 0.6 + oy)
            packets, flits = traffic.get(link["name"], (0, 0))
            step = _step(flits, peak)
            colour = f"var(--q{step})" if step else "var(--grid)"
            tip = f"{link['name']}: {packets} packets, {flits} flits"
            body.append(
                f"<g><title>{escape(tip)}</title>"
                f'<line x1="{a[0]:.1f}" y1="{a[1]:.1f}" x2="{b[0]:.1f}" y2="{b[1]:.1f}" '
                f'stroke="{colour}" stroke-width="{4 if flits else 2}" stroke-linecap="round"/>'
                f'<polygon points="{b[0]:.1f},{b[1]:.1f} {b[0] - ux * 9 - uy * 5:.1f},'
                f"{b[1] - uy * 9 + ux * 5:.1f} {b[0] - ux * 9 + uy * 5:.1f},"
                f'{b[1] - uy * 9 - ux * 5:.1f}" fill="{colour}"/>'
                f'<line x1="{a[0]:.1f}" y1="{a[1]:.1f}" x2="{b[0]:.1f}" y2="{b[1]:.1f}" '
                f'stroke="transparent" stroke-width="14"/></g>'
            )
            if flits:
                mx, my = (a[0] + b[0]) / 2 + ox * 2.2, (a[1] + b[1]) / 2 + oy * 2.2
                anchor = "middle" if r0 == r1 else ("start" if sign > 0 else "end")
                body.append(f'<text x="{mx:.1f}" y="{my + 4:.1f}" font-size="11" '
                            f'text-anchor="{anchor}" class="t2 num">{flits}</text>')  # fmt: skip
        for r in range(rows):
            for c in range(cols):
                x, y = px + pad + c * pitch, top + r * pitch
                site = _site(trace, r, c)
                dash = "" if g["present"][site] else ' stroke-dasharray="4 3"'
                local = {
                    k: traffic.get(f"{net}:{r},{c}:{k}", (0, 0))[1]
                    for k in ("from_tile", "to_tile", "from_cell", "to_cell")
                }
                tip = f"site {site} ({r},{c}): " + ", ".join(f"{k} {v}" for k, v in local.items())
                body.append(
                    f"<g><title>{escape(tip)}</title>"
                    f'<rect x="{x}" y="{y}" width="{node}" height="{node}" rx="6" '
                    f'fill="var(--surface)" stroke="var(--axis)" stroke-width="1.5"{dash}/>'
                    f'<text x="{x + node / 2}" y="{y + 19}" font-size="12" text-anchor="middle" '
                    f'font-weight="600">{site}</text>'
                    f'<text x="{x + node / 2}" y="{y + 34}" font-size="10" text-anchor="middle" '
                    f'class="mut">{r},{c}</text></g>'
                )
                inj = local["from_tile"] + local["from_cell"]
                ej = local["to_tile"] + local["to_cell"]
                body.append(f'<text x="{x + node / 2}" y="{y + node + 15}" font-size="10" '
                            f'text-anchor="middle" class="t2 num">in {inj} / out {ej}</text>')  # fmt: skip
    return _svg(width, height, body, "NoC link traffic")


# ---------------------------------------------------------------------------
# Timeline
# ---------------------------------------------------------------------------

# Categorical slots in the palette's fixed order, all eight. Slot 8's red sits near the
# status-critical red of a tile error, which is never a bar: a dashed line with a ⚠ icon
# and a label, so the two never rest on colour alone
_GROUPS = (("dma", "DMA", "--s1"), ("cmd", "tile command", "--s2"), ("l3", "L3 port", "--s3"),
           ("link", "NoC link", "--s4"), ("l2", "L2 port", "--s5"),
           ("slot", "credit slot", "--s6"), ("cmb", "Combiner", "--s7"),
           ("fabric", "L1 / fabric", "--s8"))  # fmt: skip


def _group(resource: str) -> str:
    if resource == "dma":
        return "dma"
    if resource.endswith(":l3"):
        return "l3"
    if resource.startswith(("req:", "rsp:")):
        return "link"
    kind = resource.partition(":")[2]
    if " slot " in kind:
        return "slot"
    if kind.startswith("l2 "):
        return "l2"
    if kind.startswith("cmb "):
        return "cmb"
    if kind.startswith(("l1 ", "res ")) or kind == "collision":
        return "fabric"
    return "cmd"


_KIND_ORDER = {"load": 0, "load wait": 1, "exec": 2, "exec wait": 3, "exec drain": 4,
               "store": 5, "store wait": 6, "l3": 7, "l2 A rd": 8, "l2 B rd": 9,
               "l2 A wr": 10, "l2 B wr": 11, "cmb wr": 12, "cmb fwd": 13, "cmb rd": 14}  # fmt: skip


def _lane_key(resource: str) -> tuple:
    """Lanes of one site in pipeline order (load, exec, its drain, store, then the
    memories' ports and the slots in number order); links by name."""
    if resource.startswith("site "):
        site, _, kind = resource[5:].partition(":")
        pool, _, n = kind.partition(" slot ")
        if n:
            return (int(site), 20 + (pool == "cmb"), int(n), resource)
        if kind.startswith(("l1 ", "res ")):  # A, B, W streams, then N, S columns, by number
            name = kind.split(" ", 1)[1]
            if name[1:].isdigit():
                return (int(site), 40 + "ABWNS".index(name[0]), int(name[1:]), resource)
            return (int(site), 46, 0, resource)  # res rows
        return (int(site), _KIND_ORDER.get(kind, 30), 0, resource)
    return (0, 0, 0, resource)


def _tick_step(cycles: int) -> int:
    """A 1-2-5 step giving at most ten ticks over ``cycles``."""
    base = 1
    while True:
        for m in (1, 2, 5):
            if cycles <= 10 * base * m:
                return base * m
        base *= 10


def svg_timeline(trace: Trace, *, max_lanes: int = 120) -> str:
    """Resource occupancy over cycles: a lane per resource, a bar per interval."""
    intervals = trace["intervals"]
    cycles = max(1, trace.manifest["cycles"])
    lanes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for iv in intervals:
        lanes[iv["resource"]].append(iv)
    order = {g: n for n, (g, _, _) in enumerate(_GROUPS)}
    names = sorted(lanes, key=lambda r: (order[_group(r)], *_lane_key(r)))[:max_lanes]
    lane_h, left, right = 16, 150, 24
    plot_w = 900

    def x(cycle: int) -> float:
        return left + plot_w * cycle / cycles

    # The tile errors' markers: one per cycle (it names every site with an error then),
    # each label on the first row above the plot where it clears the labels before it
    markers = _error_markers(trace["errors"], x, left + plot_w)
    rows = 1 + max((m["row"] for m in markers), default=-1)
    top = 84 + 14 * rows
    width, height = left + plot_w + right, top + len(names) * lane_h + 46

    body = [
        '<text x="16" y="28" font-size="16" font-weight="600">Resource occupancy</text>',
        f'<text x="16" y="48" font-size="12" class="t2">{len(intervals)} intervals on '
        f"{len(lanes)} resources over {cycles} cycles"
        + (f" (the first {len(names)} drawn)" if len(names) < len(lanes) else "")
        + "</text>",
    ]
    lx = 16
    present = {_group(r) for r in names}
    for group, label, var in _GROUPS:  # legend: the groups on the plot
        if group not in present:
            continue
        body.append(f'<rect x="{lx}" y="60" width="10" height="10" rx="2" fill="var({var})"/>'
                    f'<text x="{lx + 15}" y="69" font-size="11" class="t2">{label}</text>')  # fmt: skip
        lx += 34 + 6 * len(label)
    step = _tick_step(cycles)
    for c in range(0, cycles + 1, step):
        body.append(f'<line x1="{x(c):.1f}" y1="{top - 4}" x2="{x(c):.1f}" '
                    f'y2="{top + len(names) * lane_h}" stroke="var(--grid)" stroke-width="1"/>'
                    f'<text x="{x(c):.1f}" y="{top + len(names) * lane_h + 16}" font-size="10" '
                    f'text-anchor="middle" class="mut num">{c}</text>')  # fmt: skip
    body.append(f'<text x="{left + plot_w / 2}" y="{height - 8}" font-size="11" '
                f'text-anchor="middle" class="t2">cycle</text>')  # fmt: skip
    errors = trace["errors"]
    if errors:  # the legend's last entry, with the marker's icon
        body.append(f'<text x="{lx}" y="69" font-size="11" fill="var(--critical)">\u26a0</text>'
                    f'<text x="{lx + 15}" y="69" font-size="11" class="t2">tile error</text>')  # fmt: skip
    for n, name in enumerate(names):
        y = top + n * lane_h
        var = {g: v for g, _, v in _GROUPS}[_group(name)]
        body.append(f'<text x="{left - 8}" y="{y + lane_h - 4}" font-size="10" text-anchor="end" '
                    f'class="t2">{escape(name)}</text>')  # fmt: skip
        for iv in lanes[name]:
            close = iv["close"] if iv["close"] != -1 else cycles
            x0, x1 = x(iv["open"]), x(close + 1)
            tip = (f"{name}: {iv['what']} for {iv['client']}, cycles {iv['open']}-{close} "
                   f"({iv['event']})")  # fmt: skip
            body.append(
                f"<g><title>{escape(tip)}</title>"
                f'<rect x="{x0:.1f}" y="{y + 2}" width="{max(1.5, x1 - x0 - 1):.1f}" '
                f'height="{lane_h - 4}" rx="2" fill="var({var})"/></g>'
            )
    # Each marker: a line across the lanes at its cycle, and its icon and label above the
    # plot, starting at that cycle (the line stops at the plot, so it never crosses a
    # label on a row below its own)
    bottom = top + len(names) * lane_h
    for mk in markers:
        ex, dx = mk["x"], (-4 if mk["anchor"] == "end" else 4)
        y = top - 8 - 14 * mk["row"]
        body.append(
            f"<g><title>{escape(mk['tip'])}</title>"
            f'<line x1="{ex:.1f}" y1="{top - 4}" x2="{ex:.1f}" y2="{bottom}" '
            f'stroke="var(--critical)" stroke-width="2" stroke-dasharray="4 3"/>'
            f'<rect x="{ex - 6:.1f}" y="{y - 10}" width="12" height="{bottom - y + 10}" '
            f'fill="transparent"/>'
            f'<text x="{ex + dx:.1f}" y="{y}" font-size="11" text-anchor="{mk["anchor"]}">'
            f'<tspan fill="var(--critical)">\u26a0</tspan> {escape(mk["label"])}</text></g>'
        )
    return _svg(width, height, body, "Resource occupancy")


def _error_markers(errors: list[dict[str, Any]], x: Any, right: float) -> list[dict[str, Any]]:
    """The timeline's error markers: the errors grouped by cycle, each group's label
    (``site 0: AXI READ_SLVERR``; ``sites 0, 2: ...`` when they agree; else ``N tile
    errors``), its tooltip (every error), and the label row where it clears the others."""
    by_cycle: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for e in errors:
        by_cycle[e["cycle"]].append(e)
    markers: list[dict[str, Any]] = []
    row_ends: list[float] = []  # per row, where its last label ends
    row_starts: list[float] = []
    for cycle in sorted(by_cycle):
        group = sorted(by_cycle[cycle], key=lambda e: e["site"])
        kinds = {(e["source"], e["kind"]) for e in group}
        sites = ", ".join(str(e["site"]) for e in group)
        if len(kinds) == 1:
            src, kind = kinds.pop()
            label = f"site{'s' if len(group) > 1 else ''} {sites}: {src} {kind}"
        else:
            label = f"{len(group)} tile errors (sites {sites})"
        ex = x(max(cycle, 0))
        width = 16 + 6.5 * len(label)  # the icon and an estimate of the text
        anchor = "end" if ex + width > right else "start"
        lo, hi = (ex - width, ex) if anchor == "end" else (ex, ex + width)
        row = next((r for r in range(len(row_ends))
                    if lo > row_ends[r] + 6 or hi < row_starts[r] - 6), len(row_ends))  # fmt: skip
        if row == len(row_ends):
            row_ends.append(hi)
            row_starts.append(lo)
        else:
            row_ends[row], row_starts[row] = max(row_ends[row], hi), min(row_starts[row], lo)
        tip = "\n".join(f"site {e['site']}: {e['text']}" for e in group)
        markers.append({"x": ex, "anchor": anchor, "row": row, "label": label, "tip": tip})
    return markers


# ---------------------------------------------------------------------------
# Heat map
# ---------------------------------------------------------------------------


def noc_words(trace: Trace) -> dict[tuple[int, int], int]:
    """Payload words carried over the NoC, by ``(source site, destination site)``
    (write data goes requester -> cell, read data cell -> requester)."""
    out: Counter[tuple[int, int]] = Counter()
    for p in trace["packets"]:
        if p["type"] in ("WR", "RD_RSP") and not p["error"]:
            out[(_site(trace, *p["src"][:2]), _site(trace, *p["dst"][:2]))] += p["words"]
    return dict(out)


def svg_heatmap(trace: Trace) -> str:
    """Payload words over the NoC, source site (rows) by destination site (columns)."""
    g = trace.manifest["geometry"]
    n = g["rows"] * g["cols"]
    words = noc_words(trace)
    peak = max(words.values(), default=0)
    cell, gap, left, top = 54, 2, 110, 96
    width = max(560, left + n * (cell + gap) + 30)
    height = top + n * (cell + gap) + 30
    body = [
        '<text x="16" y="28" font-size="16" font-weight="600">NoC payload by site pair</text>',
        '<text x="16" y="48" font-size="12" class="t2">words moved from a source site (row) '
        "to a destination site (column)</text>",
        f'<text x="{left + n * (cell + gap) / 2}" y="{top - 26}" font-size="11" '
        'text-anchor="middle" class="t2">destination site</text>',
        f'<text x="{left - 60}" y="{top + n * (cell + gap) / 2}" font-size="11" '
        f'text-anchor="middle" class="t2" transform="rotate(-90 {left - 60} '
        f'{top + n * (cell + gap) / 2})">source site</text>',
    ]
    for d in range(n):
        body.append(f'<text x="{left + d * (cell + gap) + cell / 2}" y="{top - 8}" font-size="11" '
                    f'text-anchor="middle" class="t2 num">{d}</text>')  # fmt: skip
    for s in range(n):
        y = top + s * (cell + gap)
        body.append(f'<text x="{left - 10}" y="{y + cell / 2 + 4}" font-size="11" '
                    f'text-anchor="end" class="t2 num">{s}</text>')  # fmt: skip
        for d in range(n):
            x = left + d * (cell + gap)
            value = words.get((s, d), 0)
            step = _step(value, peak)
            ink = "num on-dark" if step >= _ON_DARK_FROM else "num"
            tip = f"site {s} -> site {d}: {value} words"
            body.append(
                f'<g><title>{escape(tip)}</title>'
                f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" rx="4" '
                f'fill="var(--q{step})"/>'
                + (f'<text x="{x + cell / 2}" y="{y + cell / 2 + 4}" font-size="12" '
                   f'text-anchor="middle" class="{ink}">{value}</text>'
                   if value else "")
                + "</g>"
            )  # fmt: skip
    return _svg(width, height, body, "NoC payload by site pair")


def write_all(trace: Trace, out_dir: str | Path, stem: str = "trace") -> list[Path]:
    """Write the text report and the SVGs into ``out_dir`` (a tile's trace: the timeline
    alone); returns the paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = {f"{stem}.txt": text_report(trace), f"{stem}_timeline.svg": svg_timeline(trace)}
    if "tile" not in trace.manifest["geometry"]:  # the NoC's views: a SoC's trace only
        files[f"{stem}_links.svg"] = svg_links(trace)
        files[f"{stem}_heatmap.svg"] = svg_heatmap(trace)
    paths = []
    for name, text in files.items():
        path = out / name
        path.write_text(text, encoding="utf-8")
        paths.append(path)
    return paths
