# Makefile for SystolicTop cocotb tests

# Default simulator
SIM ?= verilator

# Language
TOPLEVEL_LANG ?= verilog

# Project paths
PROJECT_ROOT := $(shell cd ../../../.. && pwd)
GEN_DIR := $(PROJECT_ROOT)/gen

# Verilog sources
VERILOG_SOURCES = $(GEN_DIR)/systolic_top.v

# Top-level module
TOPLEVEL = SystolicTop

# Python test module
COCOTB_TEST_MODULES = test_systolic_top

# Shared testbench helpers (verif/cocotb/tb) and systars itself (src): the helpers
# simulate the ISA instructions and read the target's config, even when cocotb runs
# on a Python where the project is not installed
export PYTHONPATH := $(PROJECT_ROOT)/verif/cocotb:$(PROJECT_ROOT)/src:$(PYTHONPATH)

# Disable cocotb's pytest assertion rewriting (avoids conftest.py conflicts)
export COCOTB_REWRITE_ASSERTION_FILES :=

# Verilator specific options
ifeq ($(SIM),verilator)
EXTRA_ARGS += --timing
EXTRA_ARGS += -Wno-WIDTHEXPAND
EXTRA_ARGS += -Wno-WIDTHTRUNC
EXTRA_ARGS += -Wno-UNUSEDSIGNAL
EXTRA_ARGS += -Wno-DECLFILENAME
EXTRA_ARGS += -Wno-PROCASSINIT
EXTRA_ARGS += -Wno-CASEINCOMPLETE
EXTRA_ARGS += -Wno-SYMRSVDWORD
endif

# Include cocotb makefile
include $(shell cocotb-config --makefiles)/Makefile.sim

# Generate RTL before running tests (the systars.targets registry: `just gen` writes the same file)
.PHONY: gen-rtl
gen-rtl:
	@cd $(PROJECT_ROOT) && python3 -m systars.targets gen systolic_top

# Run tests (generate RTL first)
.PHONY: test-all
test-all: gen-rtl
	$(MAKE) sim
