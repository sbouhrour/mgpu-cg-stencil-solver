#!/usr/bin/env bash
# Times the multi-GPU CG solver across communication backends, dot-product modes, stencils, grid
# sizes and GPU counts. Run only after comm_preflight.sh has passed on the same node.
#
# Every run uses a fixed iteration count: all backends reach convergence in the same number of
# iterations (checked by the preflight), so the time per iteration is the quantity compared, and a
# fixed count keeps the 512^3 single-GPU runs affordable. Each run is the solver's own protocol:
# 3 warmup solves, 1 profiled solve, median of 10 timed solves. One JSON file per configuration.
#
# Configurations: for each stencil and size, the single-GPU runs once per dot mode (no halo, so the
# backend is irrelevant); then for each GPU count >= 2, every backend x dot mode, plus NCCL with
# NCCL_P2P_DISABLE=1 (forces the host path) to show what NVLink itself is worth.
# AmgX (AMGX=1, default when its driver is built) runs the same operator and partition with both of
# its communicators: MPI (host-staged, its default) and MPI_DIRECT (CUDA-aware). Without that pair,
# a faster transport on our side would be compared with a host path on AmgX's.
#
# Usage:  ./scripts/benchmarking/comm_matrix.sh
#         DRY_RUN=1 ./scripts/benchmarking/comm_matrix.sh       list the configurations only
#         STENCILS="27" SIZES="256" RANKS="2 8" ITERS=300 CHECK_EVERY=10 ...
#         SHARED_GPU=1 ...   several ranks per GPU: checks the script runs, times mean nothing
# Uses the binary as built by comm_preflight.sh (same toolchain), and does not rebuild it.
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT="${OUT_DIR:-out}/comm_matrix"
mkdir -p "$OUT" matrix
STENCILS="${STENCILS:-7 27}"
SIZES="${SIZES:-128 256 512}"
ITERS="${ITERS:-300}"
CHECK_EVERY="${CHECK_EVERY:-10}"
BACKENDS="${BACKENDS:-staged gpuaware nccl}"
BIN=./bin/cg_solver_mgpu_stencil_3d
AMGX_BIN=./external/benchmarks/amgx/amgx_cg_solver_mgpu
AMGX="${AMGX:-$([ -x "$AMGX_BIN" ] && echo 1 || echo 0)}"
MPIRUN="${MPIRUN:-mpirun}"
NGPU=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
if [ -z "${RANKS:-}" ]; then
    RANKS=""
    for r in 1 2 4 8; do [ "$r" -le "$NGPU" ] && RANKS="$RANKS $r"; done
fi
ROOT=()
[ "$(id -u)" = 0 ] && ROOT=(--allow-run-as-root)
[ "${SHARED_GPU:-0}" = 1 ] && ROOT+=(--oversubscribe -x NCCL_MULTI_RANK_GPU_ENABLE=1)

# Header-only files: both loaders build this rank's rows in memory
stub() {  # $1 stencil, $2 N -> path
    local f="matrix/stencil3d_${1}pt_${2}_stub.mtx" rows=$(($2 * $2 * $2))
    { printf '%%%%MatrixMarket matrix coordinate real general\n'
      printf '%% STENCIL_GRID_SIZE %d\n' "$2"
      printf '%d %d 0\n' "$rows" "$rows"; } > "$f"
    echo "$f"
}

CONFIGS=()
for st in $STENCILS; do
    for n in $SIZES; do
        for np in $RANKS; do
            if [ "$np" = 1 ]; then
                CONFIGS+=("$st $n 1 staged host" "$st $n 1 staged device")
                continue
            fi
            for be in $BACKENDS; do
                for dots in host device; do CONFIGS+=("$st $n $np $be $dots"); done
            done
            [[ " $BACKENDS " == *" nccl "* ]] && CONFIGS+=("$st $n $np nccl device p2poff")
        done
        if [ "$AMGX" = 1 ]; then
            for np in $RANKS; do
                # AmgX dDDI indexes local entries with int: skip partitions above 2^31 entries
                [ $((st * n * n * n / np)) -lt 2147483647 ] || continue
                CONFIGS+=("$st $n $np amgx host mpi" "$st $n $np amgx host mpidirect")
            done
        fi
    done
done
echo "${#CONFIGS[@]} configurations, ${ITERS} iterations each, output in $OUT"

for cfg in "${CONFIGS[@]}"; do
    read -r st n np be dots tag <<< "$cfg"
    name="${st}pt_N${n}_np${np}_${be}_${dots}${tag:+_$tag}"
    if [ "${DRY_RUN:-0}" = 1 ]; then echo "  $name"; continue; fi
    [ -s "$OUT/$name.json" ] && { echo "  $name: done, skipped"; continue; }
    envx=()
    [ "$tag" = p2poff ] && envx=(-x NCCL_P2P_DISABLE=1)
    if [ "$be" = amgx ]; then
        # Same fixed iteration count: a tolerance no run can reach
        exe="$AMGX_BIN"
        comm=MPI; [ "$tag" = mpidirect ] && comm=MPI_DIRECT
        args=(--stencil="$st" --communicator="$comm" --max-iters="$ITERS" --tol=1e-300 --runs=10
              --json="$OUT/$name.json")
    else
        exe="$BIN"
        args=(--stencil="$st" --comm="$be" --dots="$dots" --max-iters="$ITERS" --json="$OUT/$name.json")
        [ "$dots" = device ] && args+=(--check-every="$CHECK_EVERY")
    fi
    start=$(date +%s)
    if "$MPIRUN" "${ROOT[@]}" "${envx[@]}" -np "$np" "$exe" "$(stub "$st" "$n")" "${args[@]}" \
         > "$OUT/$name.log" 2>&1; then
        printf '  %-40s %4ss  %s\n' "$name" $(($(date +%s) - start)) \
            "$(grep -a -m1 -iE 'Time \(median\)|median' "$OUT/$name.log" | tr -s ' ')"
    else
        printf '  %-40s FAILED, see %s\n' "$name" "$OUT/$name.log"
    fi
done

[ "${DRY_RUN:-0}" = 1 ] || python3 scripts/benchmarking/comm_table.py "$OUT" | tee "$OUT/table.txt"
