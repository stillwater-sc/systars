"""Every SystolicConfig field is read by the RTL (#42, after kpugen esg#96).

SystolicConfig kept Gemmini's array options, carried over by SYSTARS, that no generated
RTL read. A field the generator does not read is an option it does not have. These tests
keep a new unread field out: a field must be read, as ``config.<field>`` or through a
``SystolicConfig`` property, by code that builds RTL. ``config.py``'s own validation, and
a name in a comment, docstring or string, do not count.
"""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

from systars.config import SystolicConfig

SRC = Path(__file__).resolve().parents[2] / "src" / "systars"
CONFIG_PY = SRC / "config.py"

# Side packages with configurations of their own; they build no SystolicTop RTL
SIDE_PACKAGES = ("simt", "simd", "stencil")


def rtl_sources() -> list[Path]:
    return sorted(
        p
        for p in SRC.rglob("*.py")
        if p != CONFIG_PY and p.relative_to(SRC).parts[0] not in SIDE_PACKAGES
    )


# The names the RTL holds a SystolicConfig under: ``config.x``, ``cfg.x``, and any
# ``<expr>.config.x`` / ``<expr>.cfg.x`` (``self.config``, ``self.t.config``)
CONFIG_NAMES = {"config", "cfg"}


def is_config(node: ast.expr) -> bool:
    return (isinstance(node, ast.Name) and node.id in CONFIG_NAMES) or (
        isinstance(node, ast.Attribute) and node.attr in CONFIG_NAMES
    )


def attribute_reads(sources: list[str]) -> set[str]:
    """The attribute names the sources read from a config (``config.name``), from their
    syntax trees: a name in a comment, a docstring or another string is no read, and
    neither is the same name on another object (``packet.input_bits``)."""
    names: set[str] = set()
    for text in sources:
        for node in ast.walk(ast.parse(text)):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.ctx, ast.Load)
                and is_config(node.value)
            ):
                names.add(node.attr)
    return names


def property_reads(config_source: str) -> dict[str, set[str]]:
    """Per ``SystolicConfig`` property, the ``self.<name>`` it reads."""
    tree = ast.parse(config_source)
    (cls,) = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SystolicConfig"]
    out = {}
    for fn in cls.body:
        if isinstance(fn, ast.FunctionDef) and any(
            isinstance(d, ast.Name) and d.id == "property" for d in fn.decorator_list
        ):
            out[fn.name] = {
                n.attr
                for n in ast.walk(fn)
                if isinstance(n, ast.Attribute)
                and isinstance(n.ctx, ast.Load)
                and isinstance(n.value, ast.Name)
                and n.value.id == "self"
            }
    return out


def config_reads(rtl: list[str], config_source: str) -> set[str]:
    """The SystolicConfig names the RTL reads: directly (``config.<name>``), or through a
    property it reads (``config.sp_total_rows`` reads ``sp_banks``), transitively.
    ``__post_init__``'s checks are no read."""
    props = property_reads(config_source)
    reads = attribute_reads(rtl)
    todo = [name for name in reads if name in props]
    while todo:
        for name in props[todo.pop()]:
            if name not in reads:
                reads.add(name)
                if name in props:
                    todo.append(name)
    return reads


def test_a_field_named_only_in_a_comment_or_docstring_is_not_read():
    src = (
        '"""cfg.l2_slots in a docstring."""\n'
        "# cfg.combiner_slots in a comment\n"
        "x = 'cfg.ld_queue_length in a string'\n"
        "depth = cfg.st_queue_length\n"
    )
    assert attribute_reads([src]) == {"st_queue_length"}


def test_a_field_name_on_another_object_is_not_read():
    src = (
        "n = packet.input_bits\n"
        "rows = self.dim\n"
        "w = self.config.acc_bits\n"
        "k = self.t.cfg.l2_slots\n"
    )
    assert attribute_reads([src]) == {"acc_bits", "l2_slots"}


def test_a_field_read_through_a_property_is_read():
    config = (
        "class SystolicConfig:\n"
        "    @property\n"
        "    def banks(self):\n"
        "        return self.depth // self.macro_rows\n"
        "    def __post_init__(self):\n"
        "        assert self.unused > 0\n"
    )
    assert config_reads(["n = cfg.banks"], config) >= {"banks", "depth", "macro_rows"}
    assert "unused" not in config_reads(["n = cfg.banks"], config)
    assert "macro_rows" not in config_reads(["# cfg.banks"], config)


def test_an_attribute_a_property_only_writes_is_not_read():
    config = (
        "class SystolicConfig:\n"
        "    @property\n"
        "    def banks(self):\n"
        "        self.depth = 32\n"
        "        return 2\n"
    )
    assert "depth" not in config_reads(["n = cfg.banks"], config)


def test_every_config_field_is_read_by_the_rtl():
    reads = config_reads([p.read_text() for p in rtl_sources()], CONFIG_PY.read_text())
    unread = [f.name for f in dataclasses.fields(SystolicConfig) if f.name not in reads]
    assert not unread, f"no RTL reads {unread}"
