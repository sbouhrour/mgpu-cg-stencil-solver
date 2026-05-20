#!/usr/bin/env bash
#
# bench_amgx_3d_27pt.sh — AmgX-CG vs custom CG (Sync / Overlap), 3D 27-point.
#
# Compares NVIDIA AmgX unpreconditioned CG against the custom multi-GPU CG
# solver (Sync and Overlap modes) on a 3D 27-point stencil matrix, across a
# user-supplied list of MPI ranks.
#
# All three solve the SAME matrix with the SAME algorithm: unpreconditioned CG,
# b = ones, x0 = 0, ||r||/||b|| < tol (L2, relative to initial residual).
#
#   - AmgX np=1   : binaire amgx_cg_solver       (AMGX_matrix_upload_all)
#   - AmgX np>=2  : binaire amgx_cg_solver_mgpu  (AMGX_matrix_upload_all_global)
#   - Custom Sync : cg_solver_mgpu_stencil_3d --stencil=27
#   - Custom Over : cg_solver_mgpu_stencil_3d --stencil=27 --overlap
#
# Usage:
#   scripts/bench_amgx_3d_27pt.sh <grid_dim> <np1> [np2 ...]
#
# Examples:
#   scripts/bench_amgx_3d_27pt.sh 192 1              # local RTX validation
#   scripts/bench_amgx_3d_27pt.sh 256 1 2 4 8        # A100 campaign (12 cells)
#
# Note: the AmgX file-based harness stores nnz/row_ptr as int, so the global
# nnz = (3N-2)^3 must stay under INT_MAX (~2.15e9) => grid ceiling ~420^3.
# 256^3 (nnz 449,455,096) is well within range; 512^3 (3.6e9) overflows.
#
# Env overrides:
#   TOL=1e-6   RUNS=10   MPIRUN_FLAGS="--allow-run-as-root"
#
set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <grid_dim> <np1> [np2 ...]" >&2
    echo "Example: $0 192 1 2 4 8" >&2
    exit 1
fi

GRID="$1"; shift
NP_LIST=("$@")

TOL="${TOL:-1e-6}"
RUNS="${RUNS:-10}"
MPIRUN_FLAGS="${MPIRUN_FLAGS:---allow-run-as-root}"

# Resolve repo root from this script's location (scripts/ is at repo root).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

MATRIX="matrix/stencil3d_27pt_${GRID}.mtx"
AMGX_SINGLE="external/benchmarks/amgx/amgx_cg_solver"
AMGX_MGPU="external/benchmarks/amgx/amgx_cg_solver_mgpu"
CUSTOM="bin/cg_solver_mgpu_stencil_3d"
GEN="bin/generate_matrix_3d_27pt"

OUTDIR="exploration_amgx_3d/data"
LOGDIR="exploration_amgx_3d/logs"
mkdir -p "$OUTDIR" "$LOGDIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS="$OUTDIR/campaign_${GRID}_${STAMP}.md"

# --- sanity checks -----------------------------------------------------------
for bin in "$AMGX_SINGLE" "$AMGX_MGPU" "$CUSTOM"; do
    [ -x "$bin" ] || { echo "ERROR: missing binary $bin (build it first)" >&2; exit 1; }
done

if [ ! -f "$MATRIX" ]; then
    echo "Matrix $MATRIX not found — generating (N=$GRID)..."
    [ -x "$GEN" ] || { echo "ERROR: missing $GEN" >&2; exit 1; }
    "$GEN" "$GRID" "$MATRIX"
fi

# --- parse helpers -----------------------------------------------------------
# Both AmgX and custom binaries print:
#   "Converged: YES in <N> iterations"
#   "Time (median): <X> ms"
get_iters()  { grep -oE "Converged: (YES|NO) in [0-9]+ iterations" "$1" | grep -oE "[0-9]+" | head -1; }
get_conv()   { grep -oE "Converged: (YES|NO)" "$1" | awk '{print $2}' | head -1; }
get_median() { grep -oE "Time \(median\): [0-9.]+ ms" "$1" | grep -oE "[0-9.]+" | head -1; }

# --- run ---------------------------------------------------------------------
echo "=============================================================="
echo " AmgX 3D 27-point CG campaign"
echo " grid=${GRID}^3   np=[${NP_LIST[*]}]   tol=${TOL}   runs=${RUNS}"
echo " matrix: $MATRIX"
echo "=============================================================="

{
    echo "# AmgX-CG vs custom CG — 3D 27-point, grid ${GRID}^3"
    echo
    echo "- Date: $(date -Is)"
    echo "- Matrix: \`$MATRIX\`"
    echo "- Algorithm: unpreconditioned CG, b=ones, x0=0, ||r||/||b|| < ${TOL} (L2)"
    echo "- Runs: ${RUNS} (median reported)"
    echo "- GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
    echo
    echo "| np | Method | Iterations | Median (ms) | Converged |"
    echo "|---:|--------|-----------:|------------:|:---------:|"
} > "$RESULTS"

for NP in "${NP_LIST[@]}"; do
    echo
    echo "--- np=$NP -------------------------------------------------"

    # AmgX: single-GPU binary at np=1, distributed binary at np>=2.
    AMGX_LOG="$LOGDIR/amgx_np${NP}_${GRID}.log"
    if [ "$NP" -eq 1 ]; then
        echo "[np=$NP] AmgX (single-GPU binary)..."
        "$AMGX_SINGLE" "$MATRIX" --tol="$TOL" --runs="$RUNS" > "$AMGX_LOG" 2>&1 || true
    else
        echo "[np=$NP] AmgX (mgpu binary)..."
        mpirun $MPIRUN_FLAGS -np "$NP" "$AMGX_MGPU" "$MATRIX" \
            --tol="$TOL" --runs="$RUNS" > "$AMGX_LOG" 2>&1 || true
    fi
    printf "| %d | AmgX-CG | %s | %s | %s |\n" \
        "$NP" "$(get_iters "$AMGX_LOG")" "$(get_median "$AMGX_LOG")" "$(get_conv "$AMGX_LOG")" \
        >> "$RESULTS"

    # Custom Sync
    SYNC_LOG="$LOGDIR/custom_sync_np${NP}_${GRID}.log"
    echo "[np=$NP] Custom Sync..."
    mpirun $MPIRUN_FLAGS -np "$NP" "$CUSTOM" "$MATRIX" --stencil=27 \
        > "$SYNC_LOG" 2>&1 || true
    printf "| %d | Custom Sync | %s | %s | %s |\n" \
        "$NP" "$(get_iters "$SYNC_LOG")" "$(get_median "$SYNC_LOG")" "$(get_conv "$SYNC_LOG")" \
        >> "$RESULTS"

    # Custom Overlap
    OVL_LOG="$LOGDIR/custom_overlap_np${NP}_${GRID}.log"
    echo "[np=$NP] Custom Overlap..."
    mpirun $MPIRUN_FLAGS -np "$NP" "$CUSTOM" "$MATRIX" --stencil=27 --overlap \
        > "$OVL_LOG" 2>&1 || true
    printf "| %d | Custom Overlap | %s | %s | %s |\n" \
        "$NP" "$(get_iters "$OVL_LOG")" "$(get_median "$OVL_LOG")" "$(get_conv "$OVL_LOG")" \
        >> "$RESULTS"
done

echo
echo "=============================================================="
echo " Results written to: $RESULTS"
echo "=============================================================="
cat "$RESULTS"
