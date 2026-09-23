#!/usr/bin/env bash
# Checks, before any timing, that every communication backend of the multi-GPU solver is correct
# on this node and that the node is the one the measurements assume.
#
#   1. topology: is every GPU pair connected by NVLink, or does traffic go through PCIe/host?
#   2. toolchain: NCCL present, MPI CUDA-aware (the gpuaware backend needs it), UCX CUDA transports
#   3. placement: does each rank drive a distinct GPU?
#   4. correctness: at a fixed iteration count, every backend and both dot-product modes must
#      reproduce the residual history of the staged/host reference, bit for bit, except where the
#      summation order legitimately changes (NCCL all-reduce over 3+ ranks): there, relative
#      deviation of r.r must stay below TOL.
#
# Nothing here is timed. A backend that fails section 4 must not be measured.
#
# Usage:  ./scripts/benchmarking/comm_preflight.sh            (all GPUs of the node)
#         RANKS="1 2 4" N=128 ./scripts/benchmarking/comm_preflight.sh
#         SHARED_GPU=1 RANKS="1 2 4" ...   several ranks per GPU (correctness only; NCCL >= 2.31)
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT="${OUT_DIR:-out}/comm_preflight"
mkdir -p "$OUT" matrix
N="${N:-128}"
ITERS="${ITERS:-60}"
TOL="${TOL:-1e-10}"
BIN=./bin/cg_solver_mgpu_stencil_3d
MPIRUN="${MPIRUN:-mpirun}"

hr() { printf '\n===== %s =====\n' "$1"; }

NGPU=$(nvidia-smi --list-gpus | wc -l)
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' .')
if [ -z "${RANKS:-}" ]; then
    RANKS=""
    for r in 1 2 4 8; do [ "$r" -le "$NGPU" ] && RANKS="$RANKS $r"; done
fi

hr "1. Topology ($NGPU GPUs, sm_$CC)"
nvidia-smi topo -m | tee "$OUT/topo.txt"
# An HGX node shows NV<k> between every GPU pair; a PCIe box shows PIX/PXB/PHB/NODE/SYS. Only the
# first NGPU columns are GPU-to-GPU links: NIC columns legitimately show PCIe paths on any node.
if [ "$NGPU" -gt 1 ]; then
    if awk -v n="$NGPU" '/^GPU[0-9]/{for(i=2;i<=n+1;i++) if($i ~ /^(PIX|PXB|PHB|NODE|SYS)$/) bad=1}
                         END{exit !bad}' "$OUT/topo.txt"; then
        echo "  WARNING: at least one GPU pair is not NVLink-connected. NCCL/gpuaware numbers will"
        echo "           measure PCIe or host paths, not NVLink. Check this is the node you meant to rent."
    else
        echo "  all GPU pairs NVLink-connected"
    fi
fi

hr "2. Toolchain"
for t in nvcc "$MPIRUN" nsys; do printf '  %-8s %s\n' "$t" "$(command -v "$t" || echo ABSENT)"; done
NCCL_H=$(ls "${NCCL_HOME:-/usr}"/include/nccl.h 2>/dev/null | head -1)
if [ -n "$NCCL_H" ]; then
    printf '  NCCL     %s\n' "$(grep -E '#define NCCL_(MAJOR|MINOR|PATCH) ' "$NCCL_H" | awk '{printf "%s.", $3}' | sed 's/\.$//')"
else
    echo "  NCCL     ABSENT (set NCCL_HOME) -- the nccl backend will be skipped"
fi
if command -v ompi_info >/dev/null; then
    printf '  MPI CUDA support (build): %s\n' \
        "$(ompi_info --parsable --all 2>/dev/null | grep -m1 'mpi_built_with_cuda_support:value' | cut -d: -f7)"
fi
command -v ucx_info >/dev/null && ucx_info -d 2>/dev/null | grep -E 'Transport: (cuda|gdr)' | sort -u | sed 's/^#/ /'

hr "3. Build"
# Always rebuilt: a binary left from another toolchain (non-CUDA-aware MPI, a NCCL the driver cannot
# run) would make this check test the wrong thing. What the binary links is read from the binary.
make -B -j"$(nproc)" ARCH="$CC" cg_solver_mgpu_stencil_3d > "$OUT/build.log" 2>&1 \
    || { echo "  build FAILED, see $OUT/build.log"; exit 1; }
ldd "$BIN" | grep -E 'libmpi\.so|libnccl' | sed 's/^\s*/  links /'
ldd "$BIN" | grep -q libnccl || echo "  built WITHOUT NCCL"

# 27-point operator built in memory from a header-only file
ROWS=$((N * N * N)); K=$((3 * N - 2))
MTX="matrix/stencil3d_27pt_${N}.mtx"
{ printf '%%%%MatrixMarket matrix coordinate real general\n'; printf '%% STENCIL_GRID_SIZE %d\n' "$N"
  printf '%d %d %d\n' "$ROWS" "$ROWS" $((K * K * K)); } > "$MTX"

hr "4. Placement and correctness (27-point, ${N}^3, ${ITERS} iterations)"
ENVV=()
[ "$(id -u)" = 0 ] && ENVV+=(--allow-run-as-root)   # rented containers usually run as root
[ "${SHARED_GPU:-0}" = 1 ] && ENVV+=(-x NCCL_MULTI_RANK_GPU_ENABLE=1)

run() {  # $1 ranks, $2 backend, $3 dots -> writes $OUT/r$1_$2_$3.log, returns the exit status
    "$MPIRUN" --oversubscribe "${ENVV[@]}" -np "$1" "$BIN" "$MTX" --stencil=27 --comm="$2" \
        --dots="$3" --max-iters="$ITERS" --verbose=3 > "$OUT/r$1_$2_$3.log" 2>&1
}

# Compares two hex traces: prints IDENTICAL, or the max relative deviation of r.r
compare() {
    python3 - "$1" "$2" <<'EOF'
import re, sys
def load(f):
    return [float.fromhex(v) for v in re.findall(r'\[Trace +\d+\] rs=(\S+)', open(f).read())]
a, b = load(sys.argv[1]), load(sys.argv[2])
if not a or len(a) != len(b):
    print("MISSING"); sys.exit()
print("IDENTICAL" if a == b else "%.2e" % max(abs(x - y) / abs(x) for x, y in zip(a, b)))
EOF
}

FAIL=0
printf '  %-6s %-9s %-7s %s\n' ranks backend dots result
for np in $RANKS; do
    run "$np" staged host || { echo "  $np ranks: reference run failed, see $OUT/r${np}_staged_host.log"; FAIL=1; continue; }
    devs=$(grep -aoE '^\[Rank [0-9]+\] GPU [0-9]+' "$OUT/r${np}_staged_host.log" | awk '{print $4}' | sort -u | wc -l)
    if [ "${SHARED_GPU:-0}" != 1 ] && [ "$devs" -ne "$np" ]; then
        echo "  $np ranks: only $devs distinct GPUs used -- rank placement is wrong, stop here"; FAIL=1
    fi
    for be in staged gpuaware nccl; do
        for dots in host device; do
            [ "$be/$dots" = staged/host ] && continue
            log="$OUT/r${np}_${be}_${dots}.log"
            if ! run "$np" "$be" "$dots"; then
                # -a: progress lines end in carriage returns, which make grep call the log binary
                if grep -aqE 'built without NCCL|needs a CUDA-aware MPI' "$log"; then
                    printf '  %-6s %-9s %-7s SKIP (%s)\n' "$np" "$be" "$dots" "$(grep -a -m1 -oE 'built without NCCL|needs a CUDA-aware MPI' "$log")"
                else
                    printf '  %-6s %-9s %-7s FAIL (run error, see %s)\n' "$np" "$be" "$dots" "$log"; FAIL=1
                fi
                continue
            fi
            res=$(compare "$OUT/r${np}_staged_host.log" "$log")
            verdict=PASS
            case "$res" in
                IDENTICAL) ;;
                MISSING) verdict=FAIL ;;
                *)  # only the device all-reduce of NCCL over 3+ ranks may change summation order
                    if [ "$be/$dots" = nccl/device ] && [ "$np" -ge 3 ] \
                       && python3 -c "import sys; sys.exit(not float('$res') < float('$TOL'))"; then
                        verdict="PASS (reordered sum)"
                    else
                        verdict=FAIL
                    fi ;;
            esac
            [ "${verdict%% *}" = FAIL ] && FAIL=1
            printf '  %-6s %-9s %-7s %-10s %s\n' "$np" "$be" "$dots" "$res" "$verdict"
        done
    done
done

hr "Verdict"
if [ "$FAIL" = 0 ]; then
    echo "  every backend that ran is correct -- measurements may start"
else
    echo "  FAILURES above -- do not measure the failing backends"
fi
exit "$FAIL"
