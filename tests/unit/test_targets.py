"""
Tests for the RTL generation targets (``systars.targets``): the list ``just gen``,
``just synth`` and CI read, and the generation command.
"""

import re

import pytest

from systars import targets
from systars.targets import TARGETS, emit_target


def test_list_plain_is_name_and_top(capsys):
    assert targets.main(["list", "--plain"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert [line.split() for line in lines] == [[t.name, t.top] for t in TARGETS.values()]


@pytest.mark.parametrize("name", ["pe", "pe_array_2x2", "scratchpad"])
def test_emit_target_writes_the_top_module(tmp_path, name):
    target = TARGETS[name]
    path = emit_target(target, tmp_path)
    assert path == tmp_path / f"{name}.v"
    assert f"module {target.top}(" in path.read_text(encoding="utf-8")


def test_gen_rejects_an_unknown_target(tmp_path, capsys):
    assert targets.main(["gen", "nope", "--out-dir", str(tmp_path)]) == 2
    assert "unknown target" in capsys.readouterr().err
    assert not list(tmp_path.iterdir())


# Amaranth writes a memory's contents, and only those, in an ``initial`` block
# (``mem[3] = 32'd42;``); register initial values are written inline instead.
INITIAL_BLOCK = re.compile(r"^\s*initial begin\n(.*?)^\s*end$", re.MULTILINE | re.DOTALL)
WORD = re.compile(r"^\s*\w+\[\d+\] = \d+'([bdh])([0-9a-fA-F_xz]+);$", re.MULTILINE)
RADIX = {"b": 2, "d": 10, "h": 16}


def memory_words(verilog: str) -> list[int]:
    """Every memory word's initial value in ``verilog``."""
    words = []
    for block in INITIAL_BLOCK.findall(verilog):
        for radix, digits in WORD.findall(block):
            words.append(int(digits.replace("_", ""), RADIX[radix]))
    return words


@pytest.mark.parametrize("name", list(TARGETS))
def test_targets_have_no_memory_contents(tmp_path, name):
    """``just synth polarfire`` drops memory initial values (``delete t:$meminit_v2``):
    safe only while every target's memories start at zero."""
    verilog = emit_target(TARGETS[name], tmp_path).read_text(encoding="utf-8")
    assert not [w for w in memory_words(verilog) if w]


def test_memory_words_sees_the_memories(tmp_path):
    """The check is live: the scratchpad's words are found."""
    verilog = emit_target(TARGETS["scratchpad"], tmp_path).read_text(encoding="utf-8")
    assert len(memory_words(verilog)) > 0
