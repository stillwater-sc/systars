# SYSTARS Development Workflow

This guide explains how to develop, test, and verify RTL components in the SYSTARS project using Amaranth HDL and cocotb.

## Overview

SYSTARS uses a modern Python-based RTL development flow:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Amaranth HDL   │────>│  Verilog Gen    │────>│  Cocotb Tests   │
│  (Python)       │     │  (gen/*.v)      │     │  (Verilator)    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
         │                                               │
         ▼                                               ▼
┌─────────────────┐                             ┌─────────────────┐
│  Unit Tests     │                             │  Waveforms      │
│  (pytest)       │                             │  (*.vcd)        │
└─────────────────┘                             └─────────────────┘
```

| Layer | Tool | Purpose |
|-------|------|---------|
| RTL Description | Amaranth HDL | Python-based hardware description |
| Unit Tests | pytest + Amaranth simulator | Fast Python-level verification |
| RTL Generation | `amaranth.back.verilog` | Convert to synthesizable Verilog |
| RTL Simulation | cocotb + Verilator | Cycle-accurate Verilog simulation |
| Waveform Debug | GTKWave | Visual signal inspection |

---

## Getting Started

### Prerequisites

1. **Python 3.10+** with virtual environment
2. **OSS-CAD-SUITE** (provides Verilator, Yosys, GTKWave)
3. **just** command runner

### Initial Setup

```bash
# Clone the repository
git clone https://github.com/stillwater-sc/systars.git
cd systars

# Run automated setup (creates venv, installs deps, downloads tools if needed)
./bin/setup-dev-env.sh

# Or if OSS-CAD-SUITE is already installed:
just setup

# Activate the virtual environment
source .venv/bin/activate

# Verify tools are available
just check-tools
```

### Project Structure

```
systars/
├── src/systars/           # Amaranth RTL source code
│   ├── config.py          # Central configuration
│   ├── core/              # PE, PEArray, SystolicArray components
│   └── memory/            # Scratchpad, Accumulator
├── tests/unit/            # Python unit tests (pytest)
├── verif/cocotb/tests/    # RTL simulation tests (cocotb)
│   ├── pe/
│   ├── pe_array/
│   ├── systolic_array/
│   ├── scratchpad/
│   └── accumulator/
├── gen/                   # Generated Verilog (gitignored)
├── scripts/               # RTL generation scripts
├── docs/                  # Documentation pages (Markdown)
└── docs-site/             # Starlight site built from docs/
```

---

## Development Workflow

### 1. Write Amaranth RTL

Create or modify components in `src/systars/`. Amaranth uses Python to describe hardware:

```python
# src/systars/core/example.py
from amaranth import Module, Signal, signed, unsigned
from amaranth.lib.wiring import Component, In, Out

from ..config import SystolicConfig


class MyComponent(Component):
    """
    A new hardware component.

    Ports:
        input_data: Input data signal
        output_data: Processed output
        valid: Data valid flag
    """

    def __init__(self, config: SystolicConfig):
        self.config = config

        # Define the component's interface
        super().__init__({
            "input_data": In(signed(config.input_bits)),
            "output_data": Out(signed(config.acc_bits)),
            "valid": Out(1),
        })

    def elaborate(self, _platform):
        m = Module()

        # Combinational logic
        m.d.comb += self.output_data.eq(self.input_data << 2)

        # Sequential logic (registered)
        m.d.sync += self.valid.eq(1)

        return m
```

**Key Amaranth Patterns:**

| Pattern | Code | Description |
|---------|------|-------------|
| Combinational | `m.d.comb += sig.eq(value)` | Wire assignment |
| Sequential | `m.d.sync += sig.eq(value)` | Registered (clocked) |
| Conditional | `with m.If(cond):` | Mux/conditional logic |
| Submodule | `m.submodules.name = Module()` | Instantiate child |
| Memory | `Memory(shape=..., depth=...)` | RAM/ROM |

### 2. Write Unit Tests

Create pytest tests in `tests/unit/` to verify Python-level behavior:

```python
# tests/unit/test_example.py
import pytest
from amaranth.sim import Simulator

from systars.config import SystolicConfig
from systars.core.example import MyComponent


class TestMyComponent:
    @pytest.fixture
    def config(self):
        return SystolicConfig()

    @pytest.fixture
    def dut(self, config):
        return MyComponent(config)

    def test_instantiation(self, dut):
        """Test that the component can be instantiated."""
        assert dut is not None

    def test_output_scaling(self, dut):
        """Test that output is input << 2."""

        async def testbench(ctx):  # Amaranth 0.5 async testbench API
            ctx.set(dut.input_data, 5)
            await ctx.tick()  # Advance one clock
            result = ctx.get(dut.output_data)
            assert result == 20, f"Expected 20, got {result}"

        sim = Simulator(dut)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()
```

**Run unit tests:**

```bash
just test-unit                             # Run all unit tests
just test-file tests/unit/test_example.py  # Run specific file
just test-k "test_output"                  # Run tests matching pattern
```

### 3. Generate Verilog

Once unit tests pass, generate synthesizable Verilog:

```bash
just gen                    # Generate every target into gen/
just gen-pe                 # Generate PE Verilog
just gen-pe-array           # Generate PEArray Verilog
just gen-systolic-array     # Generate SystolicArray Verilog
```

The targets (`python3 -m systars.targets list`) are in `src/systars/targets.py`, the one
list that `just gen`, `just synth` and CI read. To generate and synthesize a new module, add
a `Target` there.

Or manually:

```python
from amaranth.back import verilog
from systars.config import SystolicConfig
from systars.core.example import MyComponent

config = SystolicConfig()
component = MyComponent(config)
rtl = verilog.convert(component, name="MyComponent")

with open("gen/mycomponent.v", "w") as f:
    f.write(rtl)
```

### 4. Write Cocotb Tests

Create RTL simulation tests in `verif/cocotb/tests/<component>/`:

**Directory structure:**

```
verif/cocotb/tests/mycomponent/
├── Makefile
├── __init__.py
└── test_mycomponent.py
```

**Makefile** (recipe lines must start with a tab):

<!-- markdownlint-disable MD010 -->
```makefile
# Makefile for MyComponent cocotb tests

SIM ?= verilator
TOPLEVEL_LANG ?= verilog

PROJECT_ROOT := $(shell cd ../../../.. && pwd)
GEN_DIR := $(PROJECT_ROOT)/gen

VERILOG_SOURCES = $(GEN_DIR)/mycomponent.v
TOPLEVEL = MyComponent
COCOTB_TEST_MODULES = test_mycomponent

# Disable pytest assertion rewriting conflicts
export COCOTB_REWRITE_ASSERTION_FILES :=

# Verilator lint waivers for Amaranth-generated code
ifeq ($(SIM),verilator)
EXTRA_ARGS += --timing
EXTRA_ARGS += -Wno-WIDTHEXPAND
EXTRA_ARGS += -Wno-WIDTHTRUNC
EXTRA_ARGS += -Wno-UNUSEDSIGNAL
EXTRA_ARGS += -Wno-DECLFILENAME
EXTRA_ARGS += -Wno-PROCASSINIT
endif

include $(shell cocotb-config --makefiles)/Makefile.sim

.PHONY: gen-rtl
gen-rtl:
	@echo "Generating MyComponent Verilog..."
	@cd $(PROJECT_ROOT) && python3 -c "\
from systars.config import SystolicConfig; \
from systars.core.example import MyComponent; \
from amaranth.back import verilog; \
config = SystolicConfig(); \
comp = MyComponent(config); \
import os; os.makedirs('gen', exist_ok=True); \
open('gen/mycomponent.v', 'w').write(verilog.convert(comp, name='MyComponent')); \
print('Generated gen/mycomponent.v')"

.PHONY: test-all
test-all: gen-rtl
	$(MAKE) sim
```
<!-- markdownlint-enable MD010 -->

**Test file:**

```python
# test_mycomponent.py
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge


async def reset_dut(dut):
    """Reset the DUT."""
    dut.rst.value = 1
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst.value = 0
    await RisingEdge(dut.clk)


@cocotb.test()
async def test_reset(dut):
    """Test initial state after reset."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    # Verify initial state
    assert dut.valid.value == 0, "valid should be 0 after reset"
    dut._log.info("Reset test passed")


@cocotb.test()
async def test_scaling(dut):
    """Test output = input << 2."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    # Apply input
    dut.input_data.value = 5
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)

    # Check output
    result = dut.output_data.value.to_signed()
    assert result == 20, f"Expected 20, got {result}"

    dut._log.info(f"Scaling test passed: 5 << 2 = {result}")
```

### 5. Run Cocotb Tests

```bash
just test-cocotb                      # Run all cocotb tests
just test-cocotb-component pe         # Run specific component
just test-cocotb-component scratchpad # Run memory tests
```

---

## Investigating Failures

### Unit Test Failures

```bash
# Run with verbose output
pytest tests/unit/test_example.py -v -s

# Run with debugger on failure
pytest tests/unit/test_example.py --pdb

# Run specific test
pytest tests/unit/test_example.py::TestMyComponent::test_scaling -v
```

### Cocotb Test Failures

1. **Check the log output** - cocotb prints detailed timing information
2. **Generate waveforms** - Add to your test:

```python
@cocotb.test()
async def test_with_waves(dut):
    # Waveforms are automatically generated as dump.vcd
    ...
```

3. **View waveforms:**

```bash
just waves  # Opens GTKWave with the most recent .vcd file
# Or manually:
gtkwave verif/cocotb/tests/mycomponent/sim_build/dump.vcd
```

4. **Add debug logging:**

```python
dut._log.info(f"Signal value: {dut.some_signal.value}")
dut._log.info(f"As signed: {dut.some_signal.value.to_signed()}")
dut._log.info(f"As hex: {hex(dut.some_signal.value.to_unsigned())}")
```

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| `AttributeError: 'NoneType'` | Signal doesn't exist | Check signal name matches Verilog |
| `to_signed()` returns large positive | Value is unsigned | Use two's complement: `(1 << bits) - value` |
| Test hangs | Missing clock or `await` | Ensure `cocotb.start_soon(clock.start())` |
| `UnusedElaboratable` warning | Fixture creates unused DUT | Filtered in `pyproject.toml` |

---

## Running the Full Test Suite

```bash
# Quick check (no RTL tools needed)
just check

# Full CI pipeline
just ci

# Individual steps
just lint           # Lint Python and RTL
just test-unit      # Python unit tests
just test-cocotb    # RTL simulation tests
just synth-check    # Yosys synthesis check
```

### Synthesis

`just synth <target>` maps every generated module onto a device's cells with Yosys and
prints the cell counts. CI runs it for each target, and `just synth-report` writes the
Markdown report that CI uploads.

| Target | Yosys command | Device cells |
|--------|---------------|--------------|
| `generic` | `synth` | Yosys's internal gates |
| `ice40` | `synth_ice40 -noflatten` | Lattice iCE40: `SB_LUT4`, `SB_CARRY`, `SB_DFF*`, `SB_RAM40_4K` |
| `ecp5` | `synth_ecp5 -noflatten` | Lattice ECP5: `LUT4`, `CCU2C`, `TRELLIS_FF`, `DP16KD`, `MULT18X18D` |
| `polarfire` | `synth_microchip -family polarfire -noflatten -noiopad` | Microchip PolarFire: `CFG1`-`CFG4`, `ARI1`, `SLE`, `MACC_PA`, `RAM1K20` |

This is a check, not an FPGA build. It shows that the RTL maps onto real cells and gives a
resource estimate, with no place and route, timing or bitstream. iCE40 and ECP5 have open
place and route (nextpnr); PolarFire's is Libero. The vendor flows run hierarchically
(`-noflatten`), so the counts are per module and Yosys can't delete logic that no top-level
port observes.

PolarFire flip-flops have no power-up value, so the `polarfire` flow first drops the initial
values Amaranth writes:

- **Registers** (`attrmap -remove init`). Every register also has a synchronous reset, so
  this is safe as long as the design is reset before use.
- **Memories** (`delete t:$meminit_v2`). The targets' memories (scratchpad, accumulator)
  start at zero only in simulation, and nothing reads an entry before writing it: Matmul
  overwrites an accumulator row with `COMPUTE` on its first k-slice and accumulates only
  after that. `tests/unit/test_targets.py` checks that no target has non-zero memory
  contents. Today's targets map their memories to block RAM without this step, but
  `SystolicTop` would need it, as kpugen's tiles do.

The numbers for a 16x16 array are in the
[PolarFire deployment plan](plan/polarfire-deployment.md#14-resource-estimation).

---

## Documentation

The site at <https://stillwater-sc.github.io/systars/> is built with
[Astro Starlight](https://starlight.astro.build/), the same setup as
stillwater-sc/universal and stillwater-sc/mtl5. It needs Node.js 22.23.4 or later.

```bash
just docs           # Build the site into docs-site/dist (npm ci, npm run build)
just docs-serve     # Serve it locally with live reload
```

While `just docs-serve` runs, an edit, a new page or a removed page in `docs/` (or
`CHANGELOG.md`) re-runs the sync and the browser reloads; no restart is needed.

- **Write pages in `docs/`** as plain Markdown, starting with an H1 (it becomes the
  page title). `docs-site/sync-content.mjs` generates
  `docs-site/src/content/docs/` from `docs/` and `CHANGELOG.md`. That directory is
  git-ignored, so never edit it.
- A page in a subdirectory of `docs/` (`design/`, `architecture/`, `plan/`,
  `sessions/`) appears in that sidebar group automatically. A page at the top of
  `docs/` needs an entry in `ROOT_PAGES` in `sync-content.mjs`, or the sync fails.
- Link pages with relative paths (`[ISA](isa-instructions.md#corner-turns)`), so the
  links work on GitHub too; the sync rewrites them for the site. Images and other
  files under `docs/` are served from `/systars/assets/`. A link to another file in
  the repository becomes a GitHub link.
- A link that resolves to nothing fails the sync, and a broken internal link or
  anchor fails the build (`starlight-links-validator`).
- Starlight-native pages (MDX with Starlight components, such as the landing page)
  live in `docs/site/` and are copied as they are.
- CI (`.github/workflows/docs.yml`) builds the site on every pull request that
  touches the docs, and deploys it from `main`.

---

## Best Practices

### Amaranth RTL

1. **Use `Component` base class** for clean interfaces
2. **Document ports** in docstrings
3. **Keep `elaborate()` focused** - extract complex logic to methods
4. **Use `SystolicConfig`** for all parameterization
5. **Prefer `signed()` for arithmetic** signals

### Unit Tests

1. **Test instantiation first** - catches import/syntax errors
2. **Test port existence** - verify interface matches spec
3. **Test edge cases** - zero, max, negative values
4. **Use fixtures** for common setup

### Cocotb Tests

1. **Always reset the DUT** before testing
2. **Use `unit="ns"`** (not deprecated `units`)
3. **Use `to_signed()`** (not deprecated `signed_integer`)
4. **Add meaningful log messages** for debugging
5. **Test timing** - verify latencies match design

### Adding a New Component

1. Create Amaranth module in `src/systars/<category>/`
2. Add to `__init__.py` exports
3. Create unit tests in `tests/unit/test_<component>.py`
4. Create cocotb test directory `verif/cocotb/tests/<component>/`
5. Add Makefile and test file
6. Add to `justfile` component list in `test-cocotb`
7. Add generation script if needed

---

## Tool Reference

| Command | Description |
|---------|-------------|
| `just` | Show all available commands |
| `just setup` | Initial project setup |
| `just test` | Run all tests (unit + cocotb) |
| `just test-unit` | Run Python unit tests |
| `just test-cocotb` | Run all cocotb RTL tests |
| `just test-cocotb-component <name>` | Run single component tests |
| `just gen` | Generate all Verilog |
| `just synth <target>` | Map every module onto generic, iCE40, ECP5 or PolarFire cells |
| `just synth-report` | Write the resource utilization report |
| `just lint` | Run all linters |
| `just format` | Format Python code |
| `just waves` | Open waveform viewer |
| `just docs` | Build the documentation site |
| `just docs-serve` | Serve the documentation site locally |
| `just check-tools` | Verify tool installation |
| `just check-ruff-version` | Verify ruff version consistency |

---

## Troubleshooting

### "No module named 'systars'"

```bash
pip install -e ".[dev]"
```

### Verilator not found

```bash
# Check OSS-CAD-SUITE is in PATH
export PATH="/opt/oss-cad-suite/bin:$PATH"
# Or set environment variable
export OSS_CAD_SUITE=/path/to/oss-cad-suite
```

### Cocotb pytest traceback

The `COCOTB_REWRITE_ASSERTION_FILES` env var should be set to empty in the Makefile. This is already configured in all test Makefiles.

### Ruff formatting conflicts

```bash
just check-ruff-version  # Verify versions match
just update-ruff         # Update all configs to latest
```
