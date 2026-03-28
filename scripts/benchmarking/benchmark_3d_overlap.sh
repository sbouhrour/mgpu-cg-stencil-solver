#!/bin/bash
# =============================================================================
# benchmark_3d_overlap.sh - 3D stencil compute-communication overlap benchmark
#
# Usage:
#   ./scripts/benchmarking/benchmark_3d_overlap.sh [OPTIONS]
#
# Options:
#   --gpus=1,2,4,8      GPU counts to benchmark (default: auto-detect 1 through max)
#   --grids=128,256,512 Grid sizes (default: 128,256)
#   --stencil=7,27      Stencil types (default: 7,27)
#   --runs=N            Benchmark runs per config (default: 10)
#   --output-dir=DIR    Output root (default: results/3d/)
#   --quick             Alias: --grids=128 --runs=3
#   --profile           Also run nsys for the largest overlap config
#   --help
#
# Must be run from repo root.
# =============================================================================

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# =============================================================================
# Defaults
# =============================================================================
GPU_LIST=""
GRID_LIST="128,256"
STENCIL_LIST="7,27"
NUM_RUNS=10
OUTPUT_DIR="results/3d"
PROFILE=0

# =============================================================================
# Parse arguments
# =============================================================================
for arg in "$@"; do
    case "$arg" in
        --gpus=*)       GPU_LIST="${arg#*=}" ;;
        --grids=*)      GRID_LIST="${arg#*=}" ;;
        --stencil=*)    STENCIL_LIST="${arg#*=}" ;;
        --runs=*)       NUM_RUNS="${arg#*=}" ;;
        --output-dir=*) OUTPUT_DIR="${arg#*=}" ;;
        --quick)
            GRID_LIST="128"
            NUM_RUNS=3
            ;;
        --profile)      PROFILE=1 ;;
        --help|-h)
            head -25 "$0" | tail -20
            exit 0
            ;;
        *)
            echo "Unknown option: $arg" >&2
            exit 1
            ;;
    esac
done

# =============================================================================
# Validation
# =============================================================================
if ! command -v mpirun &>/dev/null; then
    echo "ERROR: mpirun not found in PATH" >&2
    exit 1
fi
if ! command -v nvcc &>/dev/null; then
    echo "WARNING: nvcc not found — build may fail"
fi

# Detect available GPUs
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l || echo 1)
echo "Detected GPUs: $NUM_GPUS"

# Locate binaries — support both bin/release/ (default) and bin/ (legacy builds)
find_bin() {
    local name="$1"
    for candidate in "bin/release/${name}" "bin/${name}"; do
        [ -f "$candidate" ] && { echo "$candidate"; return; }
    done
    echo ""
}

BIN=$(find_bin cg_solver_mgpu_stencil_3d)
GEN7=$(find_bin generate_matrix_3d)

if [ -z "$BIN" ] || [ -z "$GEN7" ]; then
    echo "Building 3D binaries..."
    make cg_solver_mgpu_stencil_3d generate_matrix_3d generate_matrix_3d_27pt 2>&1 | tail -5
    BIN=$(find_bin cg_solver_mgpu_stencil_3d)
    GEN7=$(find_bin generate_matrix_3d)
fi

if [ -z "$BIN" ] || [ ! -f "$BIN" ]; then
    echo "ERROR: cg_solver_mgpu_stencil_3d not found after build" >&2
    exit 1
fi

# Build GPU list
if [ -z "$GPU_LIST" ]; then
    GPU_LIST="1"
    N=2
    while [ "$N" -le "$NUM_GPUS" ]; do
        GPU_LIST="${GPU_LIST},${N}"
        N=$((N * 2))
    done
fi

# Parse comma-separated lists into arrays
IFS=',' read -ra GPU_COUNTS   <<< "$GPU_LIST"
IFS=',' read -ra GRID_SIZES   <<< "$GRID_LIST"
IFS=',' read -ra STENCIL_TYPES <<< "$STENCIL_LIST"

# =============================================================================
# Output directories
# =============================================================================
mkdir -p "${OUTPUT_DIR}/json" "${OUTPUT_DIR}/raw"

echo "================================================================"
echo "3D Stencil Overlap Benchmark"
echo "================================================================"
echo "Date:      $(date)"
echo "Host:      $(hostname)"
echo "GPUs:      ${GPU_LIST}"
echo "Grids:     ${GRID_LIST}"
echo "Stencils:  ${STENCIL_LIST}"
echo "Runs:      ${NUM_RUNS}"
echo "Output:    ${OUTPUT_DIR}"
echo "================================================================"
echo ""

# =============================================================================
# Helper: run one config with timeout and failure capture
# =============================================================================
run_config() {
    local label="$1"; shift
    local json="$1"; shift
    local logfile="$1"; shift
    # remaining args: the full mpirun command

    echo "  Running: $label"
    if timeout 600 "$@" --json="$json" > "$logfile" 2>&1; then
        echo "  [OK] $label"
        return 0
    else
        echo "  [FAILED] $label (see $logfile)"
        return 1
    fi
}

# =============================================================================
# Helper: parse timing.median_ms from a JSON file
# =============================================================================
parse_median() {
    local f="$1"
    [ -f "$f" ] && python3 -c "
import json, sys
try:
    d = json.load(open('$f'))
    print(f\"{d['timing']['median_ms']:.1f}\")
except:
    print('n/a')
" 2>/dev/null || echo "n/a"
}

# =============================================================================
# Main loop
# =============================================================================
PROFILE_DONE=0
PROFILE_LABEL=""

for S in "${STENCIL_TYPES[@]}"; do
    for GRID in "${GRID_SIZES[@]}"; do

        # Skip 27pt 512³ — requires ~54 GB host RAM
        if [ "$S" = "27" ] && [ "$GRID" = "512" ]; then
            echo "[SKIP] 27pt 512³: requires ~54 GB host RAM"
            continue
        fi

        # Determine matrix file
        if [ "$S" = "7" ]; then
            MATRIX="matrix/stencil3d_${GRID}.mtx"
            if [ ! -f "$MATRIX" ]; then
                echo "Generating 7pt ${GRID}³ matrix..."
                mkdir -p matrix
                ./bin/release/generate_matrix_3d "$GRID" "$MATRIX"
            fi
        else
            MATRIX="matrix/stencil3d_27pt_${GRID}.mtx"
            if [ "$GRID" = "256" ]; then
                echo "  Note: 27pt 256³ matrix file is ~2 GB"
            fi
            if [ ! -f "$MATRIX" ]; then
                echo "Generating 27pt ${GRID}³ matrix (in-memory format)..."
                mkdir -p matrix
                # The binary generates in-memory from the filename's grid size
                # Create a placeholder so the filename pattern is recognizable
                touch "$MATRIX"
            fi
        fi

        for N in "${GPU_COUNTS[@]}"; do
            if [ "$N" -gt "$NUM_GPUS" ]; then
                echo "[SKIP] ${S}pt ${GRID}³ ${N}GPU: only ${NUM_GPUS} GPU(s) available"
                continue
            fi

            OUTDIR="${OUTPUT_DIR}"
            LABEL="${S}pt ${GRID}³ ${N}GPU"
            MPIRUN_CMD="mpirun --allow-run-as-root -np ${N}"
            BASE_CMD="./bin/release/cg_solver_mgpu_stencil_3d ${MATRIX} --stencil=${S}"

            echo ""
            echo "--- ${LABEL} ---"

            # Sync run
            SYNC_JSON="${OUTDIR}/json/3d_${S}pt_${GRID}_${N}gpu_sync.json"
            SYNC_LOG="${OUTDIR}/raw/3d_${S}pt_${GRID}_${N}gpu_sync.txt"
            run_config "sync" "$SYNC_JSON" "$SYNC_LOG" \
                $MPIRUN_CMD $BASE_CMD \
                || true

            # Overlap run (only if N > 1)
            if [ "$N" -gt 1 ]; then
                OVL_JSON="${OUTDIR}/json/3d_${S}pt_${GRID}_${N}gpu_overlap.json"
                OVL_LOG="${OUTDIR}/raw/3d_${S}pt_${GRID}_${N}gpu_overlap.txt"
                run_config "overlap" "$OVL_JSON" "$OVL_LOG" \
                    $MPIRUN_CMD $BASE_CMD --overlap \
                    || true

                # Track largest config for profiling
                PROFILE_LABEL="${S}pt_${GRID}_${N}gpu"
                PROFILE_STENCIL="$S"
                PROFILE_GRID="$GRID"
                PROFILE_N="$N"
                PROFILE_MATRIX="$MATRIX"
            fi

        done  # GPU_COUNTS
    done  # GRID_SIZES
done  # STENCIL_TYPES

# =============================================================================
# Optional nsys profiling of the largest overlap config
# =============================================================================
if [ "$PROFILE" = "1" ] && [ -n "$PROFILE_LABEL" ]; then
    mkdir -p "${OUTPUT_DIR}/nsys"
    echo ""
    echo "--- nsys profiling: ${PROFILE_LABEL} overlap ---"
    MPIRUN_CMD="mpirun --allow-run-as-root -np ${PROFILE_N}"
    BASE_CMD="./bin/release/cg_solver_mgpu_stencil_3d ${PROFILE_MATRIX} --stencil=${PROFILE_STENCIL}"
    nsys profile \
        --trace=cuda,nvtx,mpi \
        --trace-fork-before-exec=true \
        --output="${OUTPUT_DIR}/nsys/3d_${PROFILE_LABEL}_overlap" \
        $MPIRUN_CMD $BASE_CMD --overlap || echo "  [WARN] nsys profile failed"
fi

# =============================================================================
# Summary table
# =============================================================================
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SUMMARY_FILE="${OUTPUT_DIR}/summary_${TIMESTAMP}.txt"

{
    echo "================================================================"
    echo "3D Stencil Overlap Benchmark Summary"
    echo "================================================================"
    printf "%-30s | %9s | %12s | %6s\n" "Config" "Sync (ms)" "Overlap (ms)" "Gain"
    printf "%-30s-|-%9s-|-%12s-|-%6s\n" "------------------------------" "---------" "------------" "------"

    for S in "${STENCIL_TYPES[@]}"; do
        for GRID in "${GRID_SIZES[@]}"; do
            [ "$S" = "27" ] && [ "$GRID" = "512" ] && continue
            for N in "${GPU_COUNTS[@]}"; do
                [ "$N" -gt "$NUM_GPUS" ] && continue

                LABEL=$(printf "%dpt %d³ %d GPU" "$S" "$GRID" "$N")
                SYNC_JSON="${OUTPUT_DIR}/json/3d_${S}pt_${GRID}_${N}gpu_sync.json"
                OVL_JSON="${OUTPUT_DIR}/json/3d_${S}pt_${GRID}_${N}gpu_overlap.json"

                SYNC_T=$(parse_median "$SYNC_JSON")
                if [ "$N" -gt 1 ]; then
                    OVL_T=$(parse_median "$OVL_JSON")
                else
                    OVL_T="n/a"
                fi

                GAIN="n/a"
                if [ "$OVL_T" != "n/a" ] && [ "$SYNC_T" != "n/a" ]; then
                    GAIN=$(python3 -c "print(f'{float('$SYNC_T')/float('$OVL_T'):.2f}×')" 2>/dev/null || echo "n/a")
                fi

                printf "%-30s | %9s | %12s | %6s\n" "$LABEL" "$SYNC_T" "$OVL_T" "$GAIN"
            done
        done
    done

    echo "================================================================"
    echo "Generated: $(date)"
} | tee "$SUMMARY_FILE"

echo ""
echo "Summary saved to: $SUMMARY_FILE"
