# Makefile for Scratchpad cocotb tests

# Default simulator
SIM ?= verilator

# Language
TOPLEVEL_LANG ?= verilog

# Project paths
PROJECT_ROOT := $(shell cd ../../../.. && pwd)
GEN_DIR := $(PROJECT_ROOT)/gen

# Verilog sources
VERILOG_SOURCES = $(GEN_DIR)/scratchpad.v

# Top-level module
TOPLEVEL = Scratchpad

# Python test module
COCOTB_TEST_MODULES = test_scratchpad

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
	@echo "Generating Scratchpad Verilog..."
	@cd $(PROJECT_ROOT) && python3 -c "\
from systars.config import SystolicConfig; \
from systars.memory.scratchpad import Scratchpad; \
from amaranth.back import verilog; \
config = SystolicConfig(sp_banks=2, sp_capacity_kb=4, spad_read_delay=2); \
sp = Scratchpad(config); \
import os; os.makedirs('gen', exist_ok=True); \
open('gen/scratchpad.v', 'w').write(verilog.convert(sp, name='Scratchpad')); \
print('Generated gen/scratchpad.v')"

# Run tests (generate RTL first)
.PHONY: test-all
test-all: gen-rtl
	$(MAKE) sim
