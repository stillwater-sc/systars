# Makefile for Accumulator cocotb tests

# Default simulator
SIM ?= verilator

# Language
TOPLEVEL_LANG ?= verilog

# Project paths
PROJECT_ROOT := $(shell cd ../../../.. && pwd)
GEN_DIR := $(PROJECT_ROOT)/gen

# Verilog sources
VERILOG_SOURCES = $(GEN_DIR)/accumulator.v

# Top-level module
TOPLEVEL = Accumulator

# Python test module
COCOTB_TEST_MODULES = test_accumulator

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
endif

# Include cocotb makefile
include $(shell cocotb-config --makefiles)/Makefile.sim

# Generate RTL before running tests
.PHONY: gen-rtl
gen-rtl:
	@echo "Generating Accumulator Verilog..."
	@cd $(PROJECT_ROOT) && python3 -c "\
from systars.config import SystolicConfig; \
from systars.memory.accumulator import Accumulator; \
from amaranth.back import verilog; \
config = SystolicConfig(acc_banks=2, acc_capacity_kb=4, acc_latency=2); \
acc = Accumulator(config); \
import os; os.makedirs('gen', exist_ok=True); \
open('gen/accumulator.v', 'w').write(verilog.convert(acc, name='Accumulator')); \
print('Generated gen/accumulator.v')"

# Run tests (generate RTL first)
.PHONY: test-all
test-all: gen-rtl
	$(MAKE) sim
