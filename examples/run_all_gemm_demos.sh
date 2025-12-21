#!/bin/bash
# Run 16x16x16 GEMM demos across all animation scripts
#
# Usage:
#   ./run_all_gemm_demos.sh           # Run all demos with default delay (500ms)
#   ./run_all_gemm_demos.sh --fast    # Run all demos without delay
#   ./run_all_gemm_demos.sh --step    # Run all demos in step mode
#   ./run_all_gemm_demos.sh --select  # Interactive menu to select which demo

set -e

# Parse arguments
EXTRA_ARGS=""
SELECT_MODE=false
for arg in "$@"; do
    case $arg in
        --select)
            SELECT_MODE=true
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $arg"
            ;;
    esac
done

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Demo definitions (all use unified --m, --n, --k flags)
declare -A DEMOS
DEMOS["wavefront"]="examples/gemm/02_animated_wavefront.py --m 16 --n 16 --k 16"
DEMOS["skew_buffer"]="examples/gemm/05_skew_buffer_timing.py --m 16 --n 16 --k 16 --animate"
DEMOS["simt_nv"]="examples/simt/nv/01_animated_simt.py --tiled --m 16 --n 16 --k 16 --fast-mem"
DEMOS["simt_nv_v1"]="examples/simt/nv_v1/01_animated_simt.py --tiled --m 16 --n 16 --k 16 --fast-mem"

declare -A DESCRIPTIONS
DESCRIPTIONS["wavefront"]="Systolic Array Wavefront (16x16 array)"
DESCRIPTIONS["skew_buffer"]="Skew Buffer Timing (16x16x16 GEMM)"
DESCRIPTIONS["simt_nv"]="NVIDIA SIMT SM (16x16x16 tiled GEMM)"
DESCRIPTIONS["simt_nv_v1"]="NVIDIA SIMT SM v1 (16x16x16 tiled GEMM)"

# Order for running demos
DEMO_ORDER=("wavefront" "skew_buffer" "simt_nv" "simt_nv_v1")

print_header() {
    echo ""
    echo -e "${BOLD}${CYAN}════════════════════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}${CYAN}  $1${NC}"
    echo -e "${BOLD}${CYAN}════════════════════════════════════════════════════════════════${NC}"
    echo ""
}

print_demo_info() {
    local name=$1
    local desc=${DESCRIPTIONS[$name]}
    local cmd=${DEMOS[$name]}
    echo -e "${YELLOW}Demo:${NC} $desc"
    echo -e "${BLUE}Command:${NC} python $cmd$EXTRA_ARGS"
    echo ""
}

run_demo() {
    local name=$1
    local cmd=${DEMOS[$name]}

    print_header "${DESCRIPTIONS[$name]}"
    print_demo_info "$name"

    # Run the demo
    python $cmd $EXTRA_ARGS

    echo ""
    echo -e "${GREEN}✓ $name completed${NC}"
    echo ""
}

select_menu() {
    echo ""
    echo -e "${BOLD}${CYAN}16x16x16 GEMM Demo Selection${NC}"
    echo ""
    echo "Available demos:"
    echo ""

    local i=1
    for name in "${DEMO_ORDER[@]}"; do
        echo -e "  ${YELLOW}$i)${NC} ${DESCRIPTIONS[$name]}"
        i=$((i + 1))
    done
    echo -e "  ${YELLOW}$i)${NC} Run all demos"
    echo -e "  ${YELLOW}0)${NC} Exit"
    echo ""

    read -p "Select demo (0-$i): " choice

    case $choice in
        0)
            echo "Exiting."
            exit 0
            ;;
        1)
            run_demo "wavefront"
            ;;
        2)
            run_demo "skew_buffer"
            ;;
        3)
            run_demo "simt_nv"
            ;;
        4)
            run_demo "simt_nv_v1"
            ;;
        5)
            run_all
            ;;
        *)
            echo -e "${RED}Invalid selection${NC}"
            exit 1
            ;;
    esac
}

run_all() {
    print_header "Running All 16x16x16 GEMM Demos"

    echo "This will run the following demos:"
    echo ""
    for name in "${DEMO_ORDER[@]}"; do
        echo -e "  • ${DESCRIPTIONS[$name]}"
    done
    echo ""
    echo -e "Extra arguments:${EXTRA_ARGS:-" (none)"}"
    echo ""

    read -p "Press Enter to start (or Ctrl+C to cancel)..."

    for name in "${DEMO_ORDER[@]}"; do
        run_demo "$name"

        # Pause between demos unless in fast mode
        if [[ ! "$EXTRA_ARGS" =~ "--fast" ]]; then
            echo ""
            read -p "Press Enter for next demo (or Ctrl+C to stop)..."
        fi
    done

    print_header "All Demos Complete"
}

# Main
if $SELECT_MODE; then
    select_menu
else
    run_all
fi
