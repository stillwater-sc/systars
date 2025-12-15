# New Development Infrastructure

  systars/
  ├── bin/
  │   ├── setup-dev-env.sh         # Full environment setup (OSS CAD Suite, venv, etc.)
  │   └── install-oss-cad-suite.sh # Standalone CAD tools installer
  ├── justfile                     # Task runner (just --list for commands)
  ├── .pre-commit-config.yaml      # Linting hooks (ruff, mypy, shellcheck, etc.)
  ├── .editorconfig                # Consistent editor settings
  ├── pyproject.toml               # Updated with cocotb, coverage, etc.
  ├── verif/
  │   └── cocotb/
  │       ├── conftest.py          # Pytest fixtures for simulation
  │       ├── tests/pe/
  │       │   ├── Makefile         # Cocotb simulation Makefile
  │       │   └── test_pe.py       # PE simulation tests
  │       ├── tb/                  # Testbench modules (placeholder)
  │       └── models/              # Reference models (placeholder)
  └── gen/                         # Generated RTL output directory

## Key Commands

```bash
  # cd into your hw development environment
  cd dev/stillwater/clones
  # clone the repo
  git clone git@github.com:stillwater-sc/systars
  cd systars

  # One-time setup (installs everything)
  ./bin/setup-dev-env.sh

  # Or manually:
  source .venv/bin/activate
  pip install -e ".[all]"
  pre-commit install
```

## Daily usage

```bash
  just                    # Show all available tasks
  just lint               # Run all linters
  just format             # Format code
  just test-unit          # Run unit tests
  just gen-pe             # Generate PE Verilog
  just test-cocotb        # Run cocotb simulation tests
  just check-tools        # Verify tool installation
```

## Cocotb Testing Flow

```bash
  # Generate RTL first
  just gen-pe

  # Run cocotb tests
  cd verif/cocotb/tests/pe
  make test-all           # Generates RTL and runs simulation
```

The setup mirrors Stillwater Supercomuting's custom hardware design approach with Just for task running, pre-commit for code quality, and cocotb for RTL simulation testing.
