#!/bin/bash
# =============================================================================
# verify_reproduction.sh - Check that the build computes what the docs report
# =============================================================================
#
# Runs small cases and compares counts, never times: iteration counts against
# docs/results.md and docs/methodology.md, --verify, Custom CG against AmgX, one
# rank against two, and a matrix read from a file against the same matrix
# generated in memory. Counts do not depend on the GPU model.
#
# Usage:
#   ./scripts/verify_reproduction.sh        # after make (and the AmgX setup, if wanted)
#
# Exit status: 0 when every check passes, 1 otherwise. Checks that need AmgX or
# two GPUs are skipped when those are missing.
# =============================================================================

set -u
export LC_ALL=C

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

MPIRUN="mpirun --allow-run-as-root"
AMGX1="./external/benchmarks/amgx/amgx_cg_solver"
AMGXN="./external/benchmarks/amgx/amgx_cg_solver_mgpu"
DIR="matrix/verify"
LOG="results/verify"
mkdir -p "$DIR" "$LOG"

# GPUs this process may use: the entries of CUDA_VISIBLE_DEVICES when it is set, else every GPU
# nvidia-smi lists (nvidia-smi ignores CUDA_VISIBLE_DEVICES)
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
else
    NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
fi
PASS=0
FAIL=0
SKIP=0

for bin in bin/spmv_bench bin/cg_solver_mgpu_stencil bin/cg_solver_mgpu_stencil_3d bin/generate_matrix bin/generate_matrix_3d; do
    if [ ! -x "$bin" ]; then
        echo "Missing $bin: run make first"
        exit 1
    fi
done

# --- helpers ------------------------------------------------------------------
iters() { grep -m1 -o 'Converged: YES in [0-9]*' "$1" | awk '{print $4}'; }
sumx() { grep -m1 'Sum(x)' "$1" | awk '{print $2}'; }

report() {  # name, status, detail
    printf "%-44s %-5s %s\n" "$1" "$2" "$3"
    case "$2" in
        PASS) PASS=$((PASS + 1)) ;;
        FAIL) FAIL=$((FAIL + 1)) ;;
        *) SKIP=$((SKIP + 1)) ;;
    esac
}

check_eq() {  # name, got, expected
    if [ -n "$2" ] && [ "$2" = "$3" ]; then
        report "$1" PASS "$2"
    else
        report "$1" FAIL "got '${2}', expected '${3}'"
    fi
}

check_rel() {  # name, a, b, tolerance: |a - b| <= tol * |b|
    if [ -n "$2" ] && [ -n "$3" ] &&
        awk -v a="$2" -v b="$3" -v t="$4" 'BEGIN { d = a - b; if (d < 0) d = -d; m = (b < 0) ? -b : b; exit !(d <= t * m) }'; then
        report "$1" PASS "$2 vs $3"
    else
        report "$1" FAIL "'${2}' vs '${3}' (tolerance $4)"
    fi
}

stub() {  # grid size, path
    echo "% STENCIL_GRID_SIZE $1" > "$2"
}

run() {  # log name, command...
    local log="$LOG/$1.log"
    shift
    "$@" > "$log" 2>&1
    echo "$log"
}

echo "=== verify_reproduction: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1), ${NUM_GPUS} GPU(s) ==="

# --- 2D, 5-point ----------------------------------------------------------------
stub 512 "$DIR/s512.mtx"
stub 1000 "$DIR/s1000.mtx"
[ -f "$DIR/f512.mtx" ] || ./bin/generate_matrix 512 "$DIR/f512.mtx" > /dev/null

L_S512=$(run c2d_s512 $MPIRUN -np 1 ./bin/cg_solver_mgpu_stencil "$DIR/s512.mtx")
L_F512=$(run c2d_f512 $MPIRUN -np 1 ./bin/cg_solver_mgpu_stencil "$DIR/f512.mtx")
L_S1000=$(run c2d_s1000 $MPIRUN -np 1 ./bin/cg_solver_mgpu_stencil "$DIR/s1000.mtx")
check_eq "2D 512^2 Custom CG iterations" "$(iters "$L_S512")" 17
check_eq "2D 1000^2 Custom CG iterations" "$(iters "$L_S1000")" 16
check_eq "2D 512^2 in memory = file (Sum(x))" "$(sumx "$L_S512")" "$(sumx "$L_F512")"

L_SPMV=$(run spmv_s512 ./bin/spmv_bench "$DIR/s512.mtx" --mode=cusparse-csr,stencil5-csr)
if grep -q "Execution time" "$L_SPMV" && ! grep -q -i "error" "$L_SPMV"; then
    report "2D 512^2 spmv_bench (both modes)" PASS "ran"
else
    report "2D 512^2 spmv_bench (both modes)" FAIL "see $L_SPMV"
fi

if [ "$NUM_GPUS" -ge 2 ]; then
    L_S512N2=$(run c2d_s512_np2 $MPIRUN -np 2 ./bin/cg_solver_mgpu_stencil "$DIR/s512.mtx")
    check_eq "2D 512^2 Custom CG iterations, 2 ranks" "$(iters "$L_S512N2")" 17
    check_rel "2D 512^2 Sum(x), 2 ranks vs 1" "$(sumx "$L_S512N2")" "$(sumx "$L_S512")" 1e-10
else
    report "2D 512^2 Custom CG, 2 ranks" SKIP "needs 2 GPUs"
fi

# --- AmgX --------------------------------------------------------------------------
if [ -x "$AMGX1" ]; then
    L_A512=$(run amgx_s512 "$AMGX1" "$DIR/s512.mtx" --runs=3)
    L_AF512=$(run amgx_f512 "$AMGX1" "$DIR/f512.mtx" --runs=3)
    check_eq "2D 512^2 AmgX iterations = Custom CG" "$(iters "$L_A512")" "$(iters "$L_S512")"
    check_eq "2D 512^2 AmgX in memory = file (Sum(x))" "$(sumx "$L_A512")" "$(sumx "$L_AF512")"
    check_rel "2D 512^2 Sum(x), AmgX vs Custom CG" "$(sumx "$L_A512")" "$(sumx "$L_S512")" 1e-10
    if [ -x "$AMGXN" ] && [ "$NUM_GPUS" -ge 2 ]; then
        L_AN2=$(run amgx_s512_np2 $MPIRUN -np 2 "$AMGXN" "$DIR/s512.mtx" --runs=3)
        check_eq "2D 512^2 AmgX iterations, 2 ranks" "$(iters "$L_AN2")" 17
        check_rel "2D 512^2 AmgX Sum(x), 2 ranks vs 1" "$(sumx "$L_AN2")" "$(sumx "$L_A512")" 1e-10
    else
        report "2D 512^2 AmgX, 2 ranks" SKIP "needs amgx_cg_solver_mgpu and 2 GPUs"
    fi
else
    report "AmgX checks" SKIP "AmgX drivers not built (./scripts/setup/full_setup.sh --amgx)"
fi

# --- 3D, 7-point and 27-point -------------------------------------------------------
stub 64 "$DIR/s3d7_64.mtx"
stub 128 "$DIR/s3d7_128.mtx"
stub 128 "$DIR/s3d27_128.mtx"
[ -f "$DIR/f3d7_64.mtx" ] || ./bin/generate_matrix_3d 64 "$DIR/f3d7_64.mtx" > /dev/null

L_7F=$(run c3d7_f64 $MPIRUN -np 1 ./bin/cg_solver_mgpu_stencil_3d "$DIR/f3d7_64.mtx")
L_7S=$(run c3d7_s64 $MPIRUN -np 1 ./bin/cg_solver_mgpu_stencil_3d "$DIR/s3d7_64.mtx")
check_eq "3D 7pt 64^3 in memory = file (Sum(x))" "$(sumx "$L_7S")" "$(sumx "$L_7F")"

L_7=$(run c3d7_s128 $MPIRUN -np 1 ./bin/cg_solver_mgpu_stencil_3d "$DIR/s3d7_128.mtx")
L_7V=$(run c3d7_s128_verify $MPIRUN -np 1 ./bin/cg_solver_mgpu_stencil_3d "$DIR/s3d7_128.mtx" --overlap --verify)
check_eq "3D 7pt 128^3 iterations" "$(iters "$L_7")" 261
check_eq "3D 7pt 128^3 --overlap --verify" "$(grep -m1 -o 'VERIFY: [A-Z]*' "$L_7V")" "VERIFY: PASS"

L_27=$(run c3d27_s128 $MPIRUN -np 1 ./bin/cg_solver_mgpu_stencil_3d "$DIR/s3d27_128.mtx" --stencil=27)
L_27V=$(run c3d27_s128_verify $MPIRUN -np 1 ./bin/cg_solver_mgpu_stencil_3d "$DIR/s3d27_128.mtx" --stencil=27 --overlap --verify)
check_eq "3D 27pt 128^3 iterations" "$(iters "$L_27")" 151
check_eq "3D 27pt 128^3 --overlap --verify" "$(grep -m1 -o 'VERIFY: [A-Z]*' "$L_27V")" "VERIFY: PASS"

# AmgX on the same 3D operator (built in memory from the header-only file), same tolerance
if [ -x "$AMGXN" ]; then
    L_A27=$(run amgx_s3d27_128 $MPIRUN -np 1 "$AMGXN" "$DIR/s3d27_128.mtx" --stencil=27 --runs=1)
    check_eq "3D 27pt 128^3 AmgX iterations = Custom CG" "$(iters "$L_A27")" "$(iters "$L_27")"
    check_rel "3D 27pt 128^3 Sum(x), AmgX vs Custom CG" "$(sumx "$L_A27")" "$(sumx "$L_27")" 1e-10
else
    report "3D 27pt 128^3 AmgX" SKIP "AmgX driver not built (./scripts/setup/full_setup.sh --amgx)"
fi

if [ "$NUM_GPUS" -ge 2 ]; then
    L_27N2=$(run c3d27_s128_np2 $MPIRUN -np 2 ./bin/cg_solver_mgpu_stencil_3d "$DIR/s3d27_128.mtx" --stencil=27 --overlap)
    check_eq "3D 27pt 128^3 iterations, 2 ranks overlap" "$(iters "$L_27N2")" 151
else
    report "3D 27pt 128^3, 2 ranks" SKIP "needs 2 GPUs"
fi

echo ""
echo "RESULT: ${PASS} passed, ${FAIL} failed, ${SKIP} skipped (logs in ${LOG}/)"
[ "$FAIL" -eq 0 ]
