"""
The differential harness (#83), level 1: random Matmul programs on the systolic tile,
against the reference models.

The unit run takes a small smoke set of seeds (``SMOKE``). ``just fuzz N`` runs
``KPU_FUZZ_SEEDS=N`` seeds from ``KPU_FUZZ_START`` across every core. It gives the run a
new directory of its own (``KPU_FUZZ_RUN``), where each program's coverage bins go as
``seed-<n>.json``; ``python -m tests.fuzz.coverage <dir>`` reports the bins no run hit.
Without ``KPU_FUZZ_RUN`` nothing is written. A failing seed is shrunk to the smallest program that still fails,
and the failure prints the command that replays it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from .program import generate
from .run import repro, run, shrink

SMOKE = 64
START = int(os.environ.get("KPU_FUZZ_START", "0"))
SEEDS = int(os.environ.get("KPU_FUZZ_SEEDS", str(SMOKE)))
RUN_DIR = os.environ.get("KPU_FUZZ_RUN")  # this run's own directory, made by `just fuzz`


@pytest.mark.parametrize("seed", [0, 1, 7, 12345])
def test_a_seed_always_draws_the_same_program(seed):
    """The replay contract: a seed names one program, decoys included."""
    first, again = generate(seed).to_json(), generate(seed).to_json()
    assert first == again
    assert first["decoys"] == again["decoys"] and len(first["decoys"]) == 4


@pytest.mark.parametrize("seed", range(START, START + SEEDS))
def test_random_matmul_matches_the_models(seed):
    prog = generate(seed)
    result = run(prog)
    if RUN_DIR:
        (Path(RUN_DIR) / f"seed-{seed}.json").write_text(
            json.dumps({"bins": sorted(prog.bins()), "ok": result.ok, "cycles": result.cycles})
        )
    if not result.ok:
        small, small_result = shrink(prog)
        traced = ""
        if RUN_DIR:  # the shrunk program's trace, beside the run's records
            bundle = Path(RUN_DIR) / f"seed-{seed}.trace.json"
            run(small, trace=bundle)
            traced = f"\ntrace: {bundle} (python -m systars.trace {bundle} --svg <dir>)"
        pytest.fail(
            f"seed {seed}: {'; '.join(result.problems)}\n"
            f"shrunk to: {'; '.join(small_result.problems)}\nreplay: {repro(small)}{traced}",
            pytrace=False,
        )


@pytest.mark.parametrize("kind", ["mx_in_is", "mx_edge_n", "mx_accumulate", "int8_accumulate"])
def test_a_violation_breaks_only_its_own_rule(kind):
    """Each drawn violation breaks exactly the rule it names, so its refusal tests that rule."""
    from .program import IS, violate

    for seed in range(200):
        prog = violate(generate(seed), kind)
        broken = {
            "mx_in_is": prog.fmt == "mxint8" and prog.dataflow == IS,
            "mx_edge_n": prog.fmt == "mxint8" and prog.N % prog.dim != 0,
            "mx_accumulate": prog.fmt == "mxint8" and prog.accumulate,
            "int8_accumulate": prog.fmt == "int8" and prog.accumulate,
        }
        assert {k for k, v in broken.items() if v} == {kind}, (seed, prog)


def test_the_coverage_report_reads_only_its_runs_records(tmp_path):
    """A run directory holds a manifest beside the records: the report reads only the records."""
    from .coverage import report

    (tmp_path / "manifest.json").write_text('{"seeds": 1}')
    (tmp_path / "seed-0.json").write_text('{"bins": ["dim=2"], "ok": true, "cycles": 1}')
    hits, _missing, runs, failed = report(tmp_path)
    assert (runs, failed, hits["dim=2"]) == (1, 0, 1)
