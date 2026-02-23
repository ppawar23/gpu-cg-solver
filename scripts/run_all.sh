#!/bin/bash
# =============================================================================
# run_all.sh — Orchestrate Full Experiment Sweep
# =============================================================================
#
# Usage:
#   ./scripts/run_all.sh                    # auto-detect GPU
#   ./scripts/run_all.sh --gpu rtx4090      # force GPU label
#   ./scripts/run_all.sh --skip-correctness # skip tests (if already passed)
#
# This script:
#   1. Runs correctness gate (CPU + GPU tests)
#   2. Runs full CG solver sweep → results/<gpu>/cg_full.csv
#   3. Runs SpMV microbenchmark sweep → results/<gpu>/spmv_bench.csv
#   4. Validates output CSVs
#
# Prerequisites:
#   - Build cg_solver and spmv_bench first (see CMakeLists.txt)
#   - Executables should be in build/ directory
#
# =============================================================================

set -e  # Exit on any error

# ── Parse arguments ──
GPU_LABEL=""
SKIP_CORRECTNESS=false

for arg in "$@"; do
    case $arg in
        --gpu)       shift; GPU_LABEL="$1"; shift ;;
        --skip-correctness) SKIP_CORRECTNESS=true; shift ;;
        *)           ;;
    esac
done

# ── Auto-detect GPU name if not specified ──
if [ -z "$GPU_LABEL" ]; then
    if command -v nvidia-smi &> /dev/null; then
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | xargs)
        case "$GPU_NAME" in
            *4090*) GPU_LABEL="rtx4090" ;;
            *3080*) GPU_LABEL="rtx3080" ;;
            *MX250*|*MX150*) GPU_LABEL="mx250" ;;
            *) GPU_LABEL=$(echo "$GPU_NAME" | tr ' ' '_' | tr '[:upper:]' '[:lower:]') ;;
        esac
    else
        echo "ERROR: nvidia-smi not found and --gpu not specified"
        exit 1
    fi
fi

echo "=================================="
echo "  GPU CG Solver — Full Sweep"
echo "  GPU: $GPU_LABEL"
echo "=================================="

# ── Paths ──
BUILD_DIR="build"
RESULTS_DIR="results/$GPU_LABEL"
mkdir -p "$RESULTS_DIR"

CG_SOLVER="$BUILD_DIR/cg_solver"
SPMV_BENCH="$BUILD_DIR/spmv_bench"

# Check executables exist
if [ ! -f "$CG_SOLVER" ]; then
    echo "ERROR: $CG_SOLVER not found. Build first:"
    echo "  mkdir build && cd build && cmake .. && cmake --build . --config Release"
    exit 1
fi

# ── Step 1: Correctness Gate ──
if [ "$SKIP_CORRECTNESS" = false ]; then
    echo ""
    echo "─── Step 1: Correctness Gate ───"
    $CG_SOLVER --correctness
    echo "Correctness gate PASSED."
else
    echo ""
    echo "─── Step 1: SKIPPED (--skip-correctness) ───"
fi

# ── Step 2: CG Solver Sweep ──
echo ""
echo "─── Step 2: CG Solver Sweep ───"
echo "  Output: $RESULTS_DIR/cg_full.csv"
$CG_SOLVER --sweep > "$RESULTS_DIR/cg_full.csv"
echo "  CG sweep complete."

# ── Step 3: SpMV Microbenchmark Sweep ──
if [ -f "$SPMV_BENCH" ]; then
    echo ""
    echo "─── Step 3: SpMV Microbenchmark Sweep ───"
    echo "  Output: $RESULTS_DIR/spmv_bench.csv"
    $SPMV_BENCH --sweep > "$RESULTS_DIR/spmv_bench.csv"
    echo "  SpMV bench complete."
else
    echo ""
    echo "─── Step 3: SKIPPED ($SPMV_BENCH not found) ───"
fi

# ── Step 4: Validate CSVs ──
echo ""
echo "─── Step 4: Validate Output ───"
python3 scripts/validate.py "$RESULTS_DIR/"

echo ""
echo "=================================="
echo "  All done! Results in: $RESULTS_DIR/"
echo "  Next: run plotting with:"
echo "    python3 scripts/plot.py --dir results/ --out report/figures/"
echo "=================================="
