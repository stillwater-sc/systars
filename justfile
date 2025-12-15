# Systars - Systolic Array RTL Generator
# Modern task runner using just (https://github.com/casey/just)

# Default recipe - show help
default:
    @just --list

# Project paths
project_root := justfile_directory()
src_dir := project_root / "src"
gen_dir := project_root / "gen"
verif_dir := project_root / "verif"
tests_dir := project_root / "tests"
docs_dir := project_root / "docs"

# Simulator selection (verilator or icarus)
sim := env_var_or_default("SIM", "verilator")

# =============================================================================
# Environment Setup
# =============================================================================

# Install all Python dependencies
install:
    python3 -m pip install -e ".[all]"

# Install development dependencies and pre-commit hooks
install-dev:
    python3 -m pip install -e ".[dev]"
    pre-commit install

# Install simulation dependencies (cocotb)
install-sim:
    python3 -m pip install -e ".[sim]"

# Create a virtual environment
venv:
    python3 -m venv .venv
    @echo "Run 'source .venv/bin/activate' to activate"

# =============================================================================
# Linting & Formatting
# =============================================================================

# Run all linters
lint: lint-python lint-rtl

# Lint Python files with ruff
lint-python:
    python3 -m ruff check {{src_dir}} {{verif_dir}} {{tests_dir}}

# Lint generated RTL files with Verilator
lint-rtl:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Linting RTL with Verilator..."
    verilog_files=$(find {{gen_dir}} -name "*.v" -o -name "*.sv" 2>/dev/null || true)
    if [ -n "$verilog_files" ]; then
        verilator --lint-only -Wall --timing \
            -Wno-MULTITOP \
            -Wno-UNUSEDSIGNAL \
            -Wno-UNUSEDPARAM \
            -Wno-UNDRIVEN \
            $verilog_files
        echo "RTL lint passed"
    else
        echo "No Verilog files found in gen/"
    fi

# Format Python code
format:
    python3 -m ruff format {{src_dir}} {{verif_dir}} {{tests_dir}}
    python3 -m ruff check --fix {{src_dir}} {{verif_dir}} {{tests_dir}}

# Run type checking with mypy
typecheck:
    python3 -m mypy {{src_dir}}/systars

# =============================================================================
# RTL Generation
# =============================================================================

# Generate all RTL
gen: gen-pe gen-mesh

# Generate PE Verilog
gen-pe:
    #!/usr/bin/env bash
    set -euo pipefail
    mkdir -p {{gen_dir}}
    python3 -c "
from systars.config import SystolicConfig
from systars.core.pe import PE
from amaranth.back import verilog
config = SystolicConfig()
pe = PE(config)
with open('{{gen_dir}}/pe.v', 'w') as f:
    f.write(verilog.convert(pe, name='PE'))
print('Generated {{gen_dir}}/pe.v')
"

# Generate Mesh Verilog (when implemented)
gen-mesh:
    @echo "Mesh generation not yet implemented"

# =============================================================================
# Testing
# =============================================================================

# Run all tests (unit + cocotb)
test: test-unit test-cocotb

# Run unit tests only
test-unit:
    python3 -m pytest {{tests_dir}} -v

# Run cocotb simulation tests
test-cocotb:
    #!/usr/bin/env bash
    set -euo pipefail
    if [ -d "{{verif_dir}}/cocotb/tests" ]; then
        cd {{verif_dir}}/cocotb && SIM={{sim}} python3 -m pytest tests/ -v
    else
        echo "No cocotb tests found yet"
    fi

# Run tests with Verilator
test-verilator:
    cd {{verif_dir}}/cocotb && SIM=verilator python3 -m pytest tests/ -v

# Run tests with coverage
test-cov:
    python3 -m pytest {{tests_dir}} -v --cov=src/systars --cov-report=html --cov-report=term-missing

# Run a specific test file
test-file file:
    python3 -m pytest {{file}} -v

# Run tests matching a pattern
test-k pattern:
    python3 -m pytest -v -k "{{pattern}}"

# =============================================================================
# Simulation
# =============================================================================

# Open waveform viewer
waves:
    #!/usr/bin/env bash
    vcd_file=$(find {{verif_dir}} -name "*.vcd" | head -1)
    if [ -n "$vcd_file" ]; then
        gtkwave "$vcd_file" &
    else
        echo "No waveform file found. Run tests first."
        exit 1
    fi

# =============================================================================
# Synthesis (Smoke Tests)
# =============================================================================

# Run Yosys synthesis check
synth-check:
    #!/usr/bin/env bash
    set -euo pipefail
    verilog_files=$(find {{gen_dir}} -name "*.v" -o -name "*.sv" 2>/dev/null || true)
    if [ -n "$verilog_files" ]; then
        read_cmds=""
        for f in $verilog_files; do
            read_cmds="$read_cmds read_verilog -sv $f;"
        done
        yosys -p "$read_cmds hierarchy -check; proc; opt; stat"
    else
        echo "No Verilog files found. Run 'just gen' first."
    fi

# =============================================================================
# Documentation
# =============================================================================

# Build documentation
docs:
    mkdocs build

# Serve documentation locally
docs-serve:
    mkdocs serve

# =============================================================================
# Cleanup
# =============================================================================

# Clean build artifacts
clean:
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name "sim_build" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.vcd" -delete 2>/dev/null || true
    find . -type f -name "*.fst" -delete 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true

# Clean generated files
clean-gen:
    rm -rf {{gen_dir}}/*.v {{gen_dir}}/*.sv 2>/dev/null || true

# Clean everything
clean-all: clean clean-gen
    rm -rf {{verif_dir}}/coverage/* 2>/dev/null || true

# =============================================================================
# CI Helpers
# =============================================================================

# Run CI lint checks
ci-lint: lint-python typecheck

# Run CI tests
ci-test: test-unit test-cov

# Run full CI pipeline
ci: ci-lint ci-test

# =============================================================================
# Utility
# =============================================================================

# Check tool versions
check-tools:
    @echo "Checking tool versions..."
    @echo "========================="
    @which python3 && python3 --version || echo "python3: not found"
    @which verilator && verilator --version || echo "verilator: not found"
    @which yosys && yosys --version || echo "yosys: not found"
    @which cocotb-config && cocotb-config --version || echo "cocotb: not found"
    @which ruff && ruff --version || echo "ruff: not found"
    @which just && just --version || echo "just: not found"

# Show project statistics
stats:
    #!/usr/bin/env bash
    echo "Project Statistics:"
    echo "==================="
    py_count=$(find {{src_dir}} -name "*.py" | wc -l)
    test_count=$(find {{tests_dir}} {{verif_dir}} -name "test_*.py" 2>/dev/null | wc -l)
    v_count=$(find {{gen_dir}} -name "*.v" 2>/dev/null | wc -l)
    echo "Python source files: $py_count"
    echo "Test files:          $test_count"
    echo "Generated Verilog:   $v_count"
    echo ""
    if [ $py_count -gt 0 ]; then
        echo "Lines of Python:"
        find {{src_dir}} -name "*.py" -exec cat {} + | wc -l
    fi
