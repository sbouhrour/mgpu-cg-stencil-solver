#!/bin/bash
# Benchmark script for AmgX multi-GPU CG/PCG solver
# Tests: CG, PCG (with JACOBI, BLOCK_JACOBI)
# Ranks: 1, 2, 4, 8
# Outputs: JSON and CSV files per config

set -e

# ============================================================
# CONFIGURATION - EDIT THIS
# ============================================================
MATRIX="matrix/3000"  # ← CHANGE THIS TO YOUR MATRIX
RUNS=10
TOLERANCE="1e-6"
MAX_ITERS=5000

# ============================================================
# Auto-detect configuration
# ============================================================
# Get GPU architecture (first GPU)
GPU_NAME=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader -i 0 | head -1 | tr -d ' ')

# Extract matrix size from filename
MATRIX_BASENAME=$(basename "$MATRIX")
MATRIX_SIZE="${MATRIX_BASENAME}"

# Date for filename
DATE=$(date +%Y%m%d_%H%M%S)

# Results directory
RESULTS_DIR="results_amgx_${GPU_NAME}_${MATRIX_SIZE}_${DATE}"
mkdir -p "$RESULTS_DIR"

echo "============================================================"
echo "Configuration:"
echo "  GPU: $GPU_NAME"
echo "  Matrix: $MATRIX (size: $MATRIX_SIZE)"
echo "  Tolerance: $TOLERANCE"
echo "  Max iterations: $MAX_ITERS"
echo "  Runs per config: $RUNS"
echo "  Results dir: $RESULTS_DIR"
echo "============================================================"

# ============================================================
# Solver configurations to test
# ============================================================
# Format: "PRECONDITIONER:LABEL"
# Note: Code uses PCG for all configs; "none" = no preconditioner (CG equivalent)
CONFIGS=(
    "none:pcg-none"
    "jacobi:pcg-jacobi"
    "amg:pcg-amg"
)

# Rank configurations to test
RANKS=(1 2 4 8)

# Summary file
SUMMARY_FILE="$RESULTS_DIR/summary.txt"

# ============================================================
# Helper functions
# ============================================================
print_header() {
    echo ""
    echo "============================================================"
    echo "$1"
    echo "============================================================"
}

print_section() {
    echo ""
    echo "------------------------------------------------------------"
    echo "$1"
    echo "------------------------------------------------------------"
}

# ============================================================
# Main benchmark loop
# ============================================================
print_header "AmgX Multi-GPU Solver Benchmark"
echo "Start time: $(date)"

# Get git information for reproducibility
GIT_HASH=$(git rev-parse HEAD 2>/dev/null || echo "unknown")
GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
GIT_DIRTY=$(git diff-index --quiet HEAD -- 2>/dev/null || echo " (dirty)")

# Initialize summary file
cat > "$SUMMARY_FILE" <<EOF
AmgX Multi-GPU Solver Benchmark Results
========================================
GPU: $GPU_NAME
Matrix: $MATRIX (size: $MATRIX_SIZE)
Tolerance: $TOLERANCE
Max iterations: $MAX_ITERS
Runs per config: $RUNS
Date: $(date)

Git Info (Reproducibility):
  Commit: $GIT_HASH$GIT_DIRTY
  Branch: $GIT_BRANCH

EOF

# Executable path
EXECUTABLE="./external/benchmarks/amgx/amgx_cg_solver_mgpu"

# Check executable exists
if [ ! -f "$EXECUTABLE" ]; then
    echo "Error: Executable not found at $EXECUTABLE"
    echo "Please compile first: cd external/benchmarks/amgx && nvcc ..."
    exit 1
fi

# Loop over solver configs
for CONFIG in "${CONFIGS[@]}"; do
    # Parse config string
    IFS=':' read -r PRECOND LABEL <<< "$CONFIG"

    print_header "CONFIG: PCG with precond=$PRECOND"

    # Run benchmarks for each rank count
    for NP in "${RANKS[@]}"; do
        print_section "Testing with $NP rank(s)"

        # Generate output filenames
        BASE_NAME="${GPU_NAME}_${MATRIX_SIZE}_${LABEL}_np${NP}"
        JSON_FILE="$RESULTS_DIR/${BASE_NAME}.json"
        CSV_FILE="$RESULTS_DIR/${BASE_NAME}.csv"

        # Write header to summary
        echo "" >> "$SUMMARY_FILE"
        echo "========================================" >> "$SUMMARY_FILE"
        echo "Solver: PCG | Precond: $PRECOND | Ranks: $NP" >> "$SUMMARY_FILE"
        echo "Files: ${BASE_NAME}.{json,csv}" >> "$SUMMARY_FILE"
        echo "========================================" >> "$SUMMARY_FILE"

        # Build command with proper flags
        CMD="mpirun --allow-run-as-root -np $NP $EXECUTABLE $MATRIX"
        CMD="$CMD --precond=$PRECOND"
        CMD="$CMD --tol=$TOLERANCE --max-iters=$MAX_ITERS --runs=$RUNS"
        CMD="$CMD --json=$JSON_FILE --csv=$CSV_FILE --timers"

        # Run benchmark
        echo "Running: $CMD"

        if $CMD 2>&1 | tee -a "$SUMMARY_FILE"; then
            echo "✓ Test completed successfully"
            echo "  JSON: $JSON_FILE"
            echo "  CSV:  $CSV_FILE"
        else
            echo "✗ Test failed (exit code: $?)"
            echo "FAILED: Precond=$PRECOND, Ranks=$NP" >> "$SUMMARY_FILE"
        fi

        # Small delay between tests
        sleep 2
    done

    echo "✓ Config $LABEL complete"
done

# Summary
print_header "Benchmark Complete"

echo ""
echo "============================================================"
echo "Summary:"
echo "  GPU: $GPU_NAME"
echo "  Matrix: $MATRIX_SIZE"
echo "  Configs tested: ${#CONFIGS[@]}"
echo "  Rank configs: ${RANKS[*]}"
echo "  Total tests: $((${#CONFIGS[@]} * ${#RANKS[@]}))"
echo ""
echo "Results directory: $RESULTS_DIR"
echo "  - Summary: $SUMMARY_FILE"
echo "  - JSON files: $((${#CONFIGS[@]} * ${#RANKS[@]})) files"
echo "  - CSV files:  $((${#CONFIGS[@]} * ${#RANKS[@]})) files"
echo "============================================================"
echo "End time: $(date)"
