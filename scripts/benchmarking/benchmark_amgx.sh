#!/bin/bash
# Benchmark the AmgX CG reference (unpreconditioned, as in the published comparison)
# on 1, 2, 4 and 8 ranks, keeping the rank counts that fit the GPUs of the node.
#
# Usage:
#   ./scripts/benchmarking/benchmark_amgx.sh matrix/stencil_10000x10000.mtx
#
# Output: results_amgx_<GPU>_<matrix>_<date>/ with one JSON and one CSV per rank count,
# and summary.txt. The Custom-vs-AmgX ratios are printed by ./scripts/run_all.sh.

set -e

MATRIX="$1"
RUNS=10
TOLERANCE="1e-6"
MAX_ITERS=5000

if [ -z "$MATRIX" ] || [ ! -f "$MATRIX" ]; then
    echo "Usage: $0 <matrix.mtx>   (e.g. matrix/stencil_10000x10000.mtx, see ./bin/generate_matrix)"
    exit 1
fi

EXECUTABLE="./external/benchmarks/amgx/amgx_cg_solver_mgpu"
if [ ! -f "$EXECUTABLE" ]; then
    echo "Error: $EXECUTABLE not found. Install AmgX first: ./scripts/setup/full_setup.sh --amgx"
    exit 1
fi

GPU_NAME=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader -i 0 | head -1 | tr -d ' ')
# GPUs this process may use: the entries of CUDA_VISIBLE_DEVICES when it is set, else every GPU
# nvidia-smi lists (nvidia-smi ignores CUDA_VISIBLE_DEVICES)
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
else
    NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
fi
MATRIX_SIZE=$(basename "$MATRIX" .mtx)
DATE=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="results_amgx_${GPU_NAME}_${MATRIX_SIZE}_${DATE}"
SUMMARY_FILE="$RESULTS_DIR/summary.txt"
mkdir -p "$RESULTS_DIR"

GIT_HASH=$(git rev-parse HEAD 2>/dev/null || echo "unknown")
GIT_DIRTY=$(git diff-index --quiet HEAD -- 2>/dev/null || echo " (dirty)")

cat > "$SUMMARY_FILE" <<EOF
AmgX CG (unpreconditioned) benchmark
GPU: $GPU_NAME ($NUM_GPUS detected)
Matrix: $MATRIX
Tolerance: $TOLERANCE, max iterations: $MAX_ITERS, runs: $RUNS
Date: $(date)
Commit: $GIT_HASH$GIT_DIRTY
EOF

echo "AmgX CG benchmark: $MATRIX on $NUM_GPUS x $GPU_NAME, results in $RESULTS_DIR"

for NP in 1 2 4 8; do
    if [ "$NP" -gt "$NUM_GPUS" ]; then
        echo "Skipping $NP ranks (only $NUM_GPUS GPUs)" | tee -a "$SUMMARY_FILE"
        continue
    fi

    BASE_NAME="${GPU_NAME}_${MATRIX_SIZE}_cg_np${NP}"
    echo "" | tee -a "$SUMMARY_FILE"
    echo "=== $NP rank(s) ===" | tee -a "$SUMMARY_FILE"

    if mpirun --allow-run-as-root -np "$NP" "$EXECUTABLE" "$MATRIX" \
        --tol="$TOLERANCE" --max-iters="$MAX_ITERS" --runs="$RUNS" \
        --json="$RESULTS_DIR/${BASE_NAME}.json" --csv="$RESULTS_DIR/${BASE_NAME}.csv" --timers \
        2>&1 | tee -a "$SUMMARY_FILE"; then
        echo "Done: $RESULTS_DIR/${BASE_NAME}.{json,csv}"
    else
        echo "FAILED: $NP ranks" | tee -a "$SUMMARY_FILE"
    fi
done

echo ""
echo "Summary: $SUMMARY_FILE"
