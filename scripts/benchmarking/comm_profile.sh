#!/usr/bin/env bash
# Profiles that explain the communication study's table, after comm_preflight.sh has passed:
#
#   1. nsys timelines of one CG iteration at 8 GPUs, one per communication path: staged
#      (synchronous and overlap), NCCL with device dots and a CUDA graph, NVSHMEM with the fused
#      halo. They show where the host waits and where the GPU idles.
#   2. Overlap contention: the interior SpMV's duration next to a staged halo (copy engines and
#      host) and next to an NCCL halo (a kernel sharing the SMs). ncu serializes kernels and
#      cannot see this; the nsys kernel trace can.
#   3. AmgX with MPI and with MPI_DIRECT: device<->host copies per iteration, to check that
#      MPI_DIRECT really keeps the halo on the device.
#   4. ncu, if the host allows counters: the axpby kernels with alpha/beta read from device memory
#      (a full FP64 division per thread) against the ones taking them by value.
#
# Each capture is summarised by comm_nsys_summary.py (per iteration: CUDA API calls, host
# synchronizations, kernels, copies by direction, mean kernel durations).
#
# Usage:  ./scripts/benchmarking/comm_profile.sh           (27-point, 256^3, 8 GPUs)
#         N=512 NP=8 ITERS=20 ./scripts/benchmarking/comm_profile.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT="${OUT_DIR:-out}/comm_profile"
mkdir -p "$OUT/nsys" "$OUT/ncu" matrix
N="${N:-256}"
NP="${NP:-8}"
ITERS="${ITERS:-20}"
BIN=./bin/cg_solver_mgpu_stencil_3d
AMGX_BIN=./external/benchmarks/amgx/amgx_cg_solver_mgpu
MPIRUN="${MPIRUN:-mpirun}"
# AmgX MPI_DIRECT needs a CUDA-aware MPI: the separate build of comm_setup.sh when present
CA_AMGX_BIN="${CUDA_AWARE_AMGX_BIN:-./external/benchmarks/amgx/amgx_cg_solver_mgpu_cuda_aware}"
CA_MPIRUN="${CUDA_AWARE_MPIRUN:-}"
if [ -z "$CA_MPIRUN" ] || [ ! -x "$CA_AMGX_BIN" ]; then CA_AMGX_BIN=$AMGX_BIN; CA_MPIRUN=$MPIRUN; fi
ROOT=()
[ "$(id -u)" = 0 ] && ROOT=(--allow-run-as-root)
for v in NVSHMEM_REMOTE_TRANSPORT NVSHMEM_SYMMETRIC_SIZE; do
    [ -n "${!v:-}" ] && ROOT+=(-x "$v")
done
MTX="matrix/stencil3d_27pt_${N}_stub.mtx"
{ printf '%%%%MatrixMarket matrix coordinate real general\n'; printf '%% STENCIL_GRID_SIZE %d\n' "$N"
  printf '%d %d 0\n' $((N * N * N)) $((N * N * N)); } > "$MTX"

hr() { printf '\n===== %s =====\n' "$1"; }
pin_args() {  # $1 mpirun -> options binding rank i to the cores listed for it in RANKFILE
    [ -n "${RANKFILE:-}" ] || return 0
    if "$1" --version 2>/dev/null | grep -qE 'Open MPI\) [5-9]'; then echo "--map-by rankfile:file=$RANKFILE"
    else echo "--rankfile $RANKFILE"; fi
}


# nsys on every rank (the output name carries the rank); the solver runs its usual protocol with
# few solves, so the capture stays small. A CUDA graph is traced node by node: by default nsys records
# a replayed graph as one opaque range, and the per-iteration kernel counts would read zero.
profile() {  # $1 name, $2 executable, rest: arguments (RUN: the mpirun, default MPIRUN)
    local name=$1 exe=$2 run="${RUN:-$MPIRUN}"; shift 2
    # shellcheck disable=SC2046
    "$run" "${ROOT[@]}" $(pin_args "$run") -np "$NP" nsys profile --trace=cuda,nvtx,osrt --cuda-graph-trace=node \
        --force-overwrite=true \
        -o "$OUT/nsys/${name}_r%q{OMPI_COMM_WORLD_RANK}" "$exe" "$MTX" "$@" \
        > "$OUT/nsys/$name.log" 2>&1 || { echo "  $name: FAILED, see $OUT/nsys/$name.log"; return; }
    nsys export --type sqlite --force-overwrite=true -o "$OUT/nsys/${name}_r0.sqlite" \
        "$OUT/nsys/${name}_r0.nsys-rep" > /dev/null 2>&1
    echo "  $name: done"
}

hr "1-2. nsys: one timeline per communication path (27-point, ${N}^3, ${NP} GPUs)"
common=(--stencil=27 --max-iters="$ITERS" --runs=3)
profile staged_sync "$BIN" "${common[@]}" --comm=staged
profile staged_overlap "$BIN" "${common[@]}" --comm=staged --overlap
profile nccl_overlap "$BIN" "${common[@]}" --comm=nccl --overlap
profile nccl_device_graph "$BIN" "${common[@]}" --comm=nccl --dots=device --check-every=10 --graph
profile nvshmem_fused "$BIN" "${common[@]}" --comm=nvshmem --dots=device --check-every=10 --fused-halo

hr "3. AmgX: where does the halo go?"
profile amgx_mpi "$AMGX_BIN" --stencil=27 --communicator=MPI --max-iters="$ITERS" --tol=1e-300 --runs=3
RUN="$CA_MPIRUN" profile amgx_mpidirect "$CA_AMGX_BIN" --stencil=27 --communicator=MPI_DIRECT --max-iters="$ITERS" \
    --tol=1e-300 --runs=3

hr "Summary per iteration (rank 0)"
python3 scripts/benchmarking/comm_nsys_summary.py --iters "$ITERS" "$OUT"/nsys/*_r0.sqlite | tee "$OUT/nsys/summary.txt"

hr "4. ncu: axpby with scalars by value vs read from device memory (1 GPU)"
if command -v ncu > /dev/null; then
    for mode in host device; do
        "$MPIRUN" "${ROOT[@]}" -np 1 ncu --kernel-name regex:'axpy|axpby' --launch-skip 30 \
            --launch-count 12 --set basic --metrics sm__inst_executed_pipe_fp64.sum \
            --csv "$BIN" "$MTX" --stencil=27 --dots="$mode" --max-iters=10 --runs=3 \
            > "$OUT/ncu/axpby_dots_$mode.csv" 2> "$OUT/ncu/axpby_dots_$mode.log" \
            && echo "  dots=$mode: done" \
            || echo "  dots=$mode: FAILED (counters denied?), see $OUT/ncu/axpby_dots_$mode.log"
    done
else
    echo "  ncu not found"
fi
