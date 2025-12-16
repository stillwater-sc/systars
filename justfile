# Systars - Systolic Array RTL Generator
# Modern task runner using just (https://github.com/casey/just)

# Default recipe - show help
default:
    @just --list

# Project paths
project_root := justfile_directory()
src_dir := project_root / "src"
rtl_dir := project_root / "rtl"
gen_dir := project_root / "gen"
verif_dir := project_root / "verif"
tests_dir := project_root / "tests"
docs_dir := project_root / "docs"

# Simulator selection (verilator or icarus)
sim := env_var_or_default("SIM", "verilator")

# OSS CAD Suite path (override with OSS_CAD_SUITE env var)
oss_cad_suite := env_var_or_default("OSS_CAD_SUITE", "/opt/oss-cad-suite")

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

# Create virtual environment
venv:
    python3 -m venv .venv
    @echo "Virtual environment created. To activate:"
    @echo "  source .venv/bin/activate"

# Full setup: create venv, install deps
setup:
    #!/usr/bin/env bash
    set -euo pipefail

    # Check for OSS-CAD-SUITE tools
    if [[ -d "{{oss_cad_suite}}/bin" ]]; then
        echo "Found OSS-CAD-SUITE at {{oss_cad_suite}}"
    else
        echo "WARNING: OSS-CAD-SUITE not found at {{oss_cad_suite}}"
        echo "Tools like yosys/verilator may not be available."
        echo "Set OSS_CAD_SUITE env var or install OSS-CAD-SUITE."
    fi

    # Create venv with system Python
    if [[ ! -d .venv ]]; then
        echo "Creating virtual environment..."
        python3 -m venv .venv
    fi

    # Activate and install
    source .venv/bin/activate
    echo "Using Python: $(python3 --version) from $(which python3)"
    echo ""
    echo "Installing dependencies..."
    pip install --upgrade pip
    pip install -e ".[dev,sim]"
    pre-commit install
    echo ""
    echo "Setup complete! To activate in a new shell:"
    echo "  source .venv/bin/activate"

# =============================================================================
# Linting & Formatting
# =============================================================================

# Run all linters
lint: lint-python lint-rtl

# Lint Python files with ruff
lint-python:
    python3 -m ruff check {{src_dir}} {{verif_dir}} {{tests_dir}}

# Lint all RTL files with Verilator
lint-rtl: lint-rtl-gen lint-rtl-hand

# Lint generated RTL (Amaranth output - relaxed checks for generator artifacts)
lint-rtl-gen:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Linting generated RTL (gen/)..."
    verilog_files=$(find {{gen_dir}} -name "*.v" -o -name "*.sv" 2>/dev/null || true)
    if [ -n "$verilog_files" ]; then
        # Amaranth/Yosys generates code with these known artifacts:
        # - WIDTHEXPAND/WIDTHTRUNC: Yosys optimizes sign extension
        # - UNUSEDSIGNAL: Yosys leaves dead wire declarations
        # - DECLFILENAME: Module name doesn't match filename
        # - PROCASSINIT: Amaranth uses both initial values and procedural assigns
        verilator --lint-only -Wall --timing \
            -Wno-MULTITOP \
            -Wno-WIDTHEXPAND \
            -Wno-WIDTHTRUNC \
            -Wno-UNUSEDSIGNAL \
            -Wno-DECLFILENAME \
            -Wno-PROCASSINIT \
            $verilog_files
        echo "Generated RTL lint passed"
    else
        echo "No generated Verilog files found in gen/"
    fi

# Lint hand-written RTL (strict - all WIDTH checks enabled)
lint-rtl-hand:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Linting hand-written RTL (rtl/)..."
    verilog_files=$(find {{rtl_dir}} -name "*.v" -o -name "*.sv" 2>/dev/null || true)
    if [ -n "$verilog_files" ]; then
        # Strict checking for hand-written code
        verilator --lint-only -Wall --timing \
            -Wno-MULTITOP \
            $verilog_files
        echo "Hand-written RTL lint passed"
    else
        echo "No hand-written Verilog files found in rtl/ (this is OK)"
    fi

# Format Python code
format:
    python3 -m ruff format {{src_dir}} {{verif_dir}} {{tests_dir}}
    python3 -m ruff check --fix {{src_dir}} {{verif_dir}} {{tests_dir}}

# Alias for format (easy to remember)
ruff: format

# Check code formatting (for CI)
format-check:
    python3 -m ruff format --check {{src_dir}} {{verif_dir}} {{tests_dir}}

# Run type checking with mypy
typecheck:
    python3 -m mypy {{src_dir}}/systars

# =============================================================================
# RTL Generation
# =============================================================================

# Generate all RTL
gen: gen-pe gen-pe-array gen-systolic-array

# Generate PE Verilog
gen-pe:
    python3 {{project_root}}/scripts/gen_pe.py

# Generate PEArray Verilog
gen-pe-array:
    python3 {{project_root}}/scripts/gen_pe_array.py

# Generate SystolicArray Verilog
gen-systolic-array:
    python3 {{project_root}}/scripts/gen_systolic_array.py

# =============================================================================
# Testing
# =============================================================================

# Run all tests (unit + cocotb)
test: test-unit test-cocotb

# Run unit tests only
test-unit:
    python3 -m pytest {{tests_dir}} -v

# Run cocotb simulation tests for all components
test-cocotb:
    #!/usr/bin/env bash
    set -euo pipefail
    failed=0
    for component in pe pe_array systolic_array scratchpad accumulator; do
        testdir="{{verif_dir}}/cocotb/tests/$component"
        if [ -d "$testdir" ]; then
            echo "========================================"
            echo "Running cocotb tests for: $component"
            echo "========================================"
            cd "$testdir" && SIM={{sim}} make test-all || failed=1
        fi
    done
    if [ $failed -ne 0 ]; then
        echo "Some cocotb tests failed!"
        exit 1
    fi
    echo "========================================"
    echo "All cocotb tests passed!"
    echo "========================================"

# Run cocotb tests for a specific component
test-cocotb-component component:
    cd {{verif_dir}}/cocotb/tests/{{component}} && SIM={{sim}} make test-all

# Run tests with Verilator (all components)
test-verilator:
    just test-cocotb

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

# Quick check (no RTL tools needed)
check: format-check lint-python typecheck test-unit

# Run full CI pipeline (matches GitHub Actions)
ci: format-check lint-python typecheck gen lint-rtl test-unit test-cocotb synth-check

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

# Check ruff version consistency across config files
check-ruff-version:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Checking ruff version consistency..."
    echo "====================================="

    # Extract versions from each config file
    precommit_ver=$(grep -A1 'astral-sh/ruff-pre-commit' .pre-commit-config.yaml | grep 'rev:' | sed 's/.*v//')
    ci_ver=$(grep 'ruff==' .github/workflows/lint.yml | sed 's/.*ruff==//' | sed 's/ .*//')
    pyproject_ver=$(grep 'ruff==' pyproject.toml | sed 's/.*ruff==//' | sed 's/[",].*//')
    installed_ver=$(ruff --version 2>/dev/null | awk '{print $2}' || echo "not installed")

    echo "  .pre-commit-config.yaml: $precommit_ver"
    echo "  .github/workflows/lint.yml: $ci_ver"
    echo "  pyproject.toml: $pyproject_ver"
    echo "  installed: $installed_ver"
    echo ""

    if [[ "$precommit_ver" == "$ci_ver" && "$ci_ver" == "$pyproject_ver" ]]; then
        echo "✓ All config files have matching ruff versions"
        if [[ "$installed_ver" != "$pyproject_ver" ]]; then
            echo "⚠ Installed version differs - run: pip install ruff==$pyproject_ver"
        fi
    else
        echo "✗ Version mismatch detected!"
        echo "  Update all files to use the same version."
        exit 1
    fi

# Update ruff to latest version across all configs
update-ruff:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Updating ruff to latest version..."

    # Get latest version
    latest=$(pip index versions ruff 2>/dev/null | head -1 | sed 's/.*(\(.*\))/\1/')
    echo "Latest ruff version: $latest"

    # Update pre-commit config
    pre-commit autoupdate --repo https://github.com/astral-sh/ruff-pre-commit

    # Update CI workflow
    sed -i "s/ruff==[0-9.]*/ruff==$latest/" .github/workflows/lint.yml

    # Update pyproject.toml
    sed -i "s/ruff==[0-9.]*/ruff==$latest/" pyproject.toml

    # Reinstall
    pip install ruff==$latest

    echo ""
    echo "Updated to ruff $latest. Run 'just check-ruff-version' to verify."

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
