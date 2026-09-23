#!/usr/bin/env bash
# Calibrates the node's communication costs with nccl-tests, so that the solver's communication time
# can be predicted before it is measured (comm_model.py).
#
#   - sendrecv_perf and all_reduce_perf, doubles, 8 B to 1 GiB, on 2 GPUs and on all GPUs
#   - all_reduce at small sizes (8 B to 1 MiB) for every NCCL_ALGO x NCCL_PROTO pair: the solver's
#     dot products are 8-byte all-reduces, where the algorithm and protocol choice is the whole story
#   - the same 8-byte all-reduce launched from a CUDA graph (-G), for comparison with stream launches
#
# Run after comm_preflight.sh has passed. nccl-tests is fetched and built outside the repository.
#
# Usage:  ./scripts/benchmarking/comm_calibrate.sh
#         NCCL_TESTS=/path/to/nccl-tests NCCL_HOME=/path/to/nccl MPI_HOME=/path/to/mpi ...
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT="${OUT_DIR:-out}/comm_calibrate"
mkdir -p "$OUT"
NT="${NCCL_TESTS:-$HOME/nccl-tests}"
MPIRUN="${MPIRUN:-mpirun}"
NGPU=$(nvidia-smi --list-gpus | wc -l)
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' .')
ROOT=()
[ "$(id -u)" = 0 ] && ROOT=(--allow-run-as-root)

hr() { printf '\n===== %s =====\n' "$1"; }

hr "1. nccl-tests"
if [ ! -x "$NT/build/all_reduce_perf" ]; then
    [ -d "$NT" ] || git clone -q --depth 1 https://github.com/NVIDIA/nccl-tests.git "$NT"
    MPI_HOME="${MPI_HOME:-$(dirname "$(dirname "$(command -v "$MPIRUN")")")}"
    make -C "$NT" -j"$(nproc)" MPI=1 MPI_HOME="$MPI_HOME" ${NCCL_HOME:+NCCL_HOME="$NCCL_HOME"} \
        NVCC_GENCODE="-gencode=arch=compute_${CC},code=sm_${CC}" > "$OUT/build.log" 2>&1 \
        || { echo "  build FAILED, see $OUT/build.log"; exit 1; }
fi
(cd "$NT" && git log --oneline -1 2>/dev/null) | sed 's/^/  nccl-tests /'

# One process per GPU, as in the solver
nt() {  # $1 ranks, $2 binary, rest: arguments -> stdout
    local np=$1 bin=$2; shift 2
    "$MPIRUN" "${ROOT[@]}" -np "$np" "${ENVX[@]}" "$NT/build/$bin" -g 1 -d double -w 20 -n 100 "$@"
}
ENVX=()

hr "2. Bandwidth and latency, 8 B to 1 GiB"
for np in 2 "$NGPU"; do
    [ "$np" -le "$NGPU" ] || continue
    for op in sendrecv all_reduce; do
        f="$OUT/${op}_np${np}.txt"
        nt "$np" "${op}_perf" -b 8 -e 1G -f 2 > "$f" 2>&1
        printf '  %-11s np=%-2s %s\n' "$op" "$np" "$(grep 'Avg bus bandwidth' "$f" | tr -s ' ')"
    done
done

hr "3. Small all-reduce: algorithm x protocol (np=$NGPU)"
for algo in Ring Tree NVLS; do
    for proto in LL LL128 Simple; do
        f="$OUT/all_reduce_small_${algo}_${proto}.txt"
        ENVX=(-x NCCL_ALGO="$algo" -x NCCL_PROTO="$proto")
        nt "$NGPU" all_reduce_perf -b 8 -e 1M -f 4 > "$f" 2>&1
        t8=$(awk '!/^#/ && $1 == 8 {print $6; exit}' "$f")
        printf '  %-5s %-6s 8 B: %s us\n' "$algo" "$proto" "${t8:-n/a (unsupported here?)}"
    done
done
ENVX=()

hr "4. 8-byte all-reduce from a CUDA graph (np=$NGPU)"
nt "$NGPU" all_reduce_perf -b 8 -e 8 -G 100 > "$OUT/all_reduce_graph_np${NGPU}.txt" 2>&1
awk '!/^#/ && $1 == 8 {print "  graph launch, 8 B: " $6 " us"; exit}' "$OUT/all_reduce_graph_np${NGPU}.txt"

hr "5. Model"
python3 scripts/benchmarking/comm_model.py "$OUT/sendrecv_np${NGPU}.txt" "$OUT/all_reduce_np${NGPU}.txt" \
    | tee "$OUT/model.txt"
