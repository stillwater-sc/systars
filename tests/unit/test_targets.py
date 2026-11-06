"""
Tests for the RTL generation targets (``systars.targets``): the list ``just gen``,
``just synth`` and CI read, and the generation command.
"""

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
