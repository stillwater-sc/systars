# Makefile for Tile cocotb tests

# Default simulator
SIM ?= verilator

# Language
TOPLEVEL_LANG ?= verilog

# Project paths
PROJECT_ROOT := $(shell cd ../../../.. && pwd)
GEN_DIR := $(PROJECT_ROOT)/gen

# Verilog sources
VERILOG_SOURCES = $(GEN_DIR)/tile.v

# Top-level module
TOPLEVEL = Tile

# Python test module
MODULE = test_tile

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
	@echo "Generating Tile Verilog..."
	@cd $(PROJECT_ROOT) && python3 -c "\
from systars.config import SystolicConfig; \
from systars.core.tile import Tile; \
from amaranth.back import verilog; \
config = SystolicConfig(tile_rows=1, tile_cols=1); \
tile = Tile(config); \
import os; os.makedirs('gen', exist_ok=True); \
open('gen/tile.v', 'w').write(verilog.convert(tile, name='Tile')); \
print('Generated gen/tile.v')"

# Run tests (generate RTL first)
.PHONY: test-all
test-all: gen-rtl
	$(MAKE) sim
