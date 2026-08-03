#!/usr/bin/env bash
# Runs the full benchmark session on a rented instance, unattended, and collects everything.
#
# Three questions, in decreasing order of how much they need this machine:
#
#   B7  Does narrowing the stencil coefficients pay on datacenter hardware?
#       Measured locally on an RTX 4060 it does not, but that part has a ridge point of about
#       1.2 FLOP/B against 4.8 on an A100 -- the local card is the one least able to show a
#       memory-side gain, so the negative result does not transfer. Single GPU.
#
#   B6  Is the coefficient-major (SoA) layout still ahead at solver level, multi-GPU?
#       The SoA notes carry an explicit placeholder pending a datacenter run. Needs several GPUs.
#
#   B1  How much do the published numbers lose to being built at -O2 while AmgX is built at -O3?
#       Re-runs the whole suite at -O3 for comparison.
#
# Usage:  ./scripts/benchmarking/rental_session.sh [max_gpus]
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT="${OUT_DIR:-out}"
mkdir -p "$OUT"
[ -s "$OUT/hw_info.txt" ] || { echo "Run rental_preflight.sh first."; exit 1; }

NGPU_AVAIL=$(nvidia-smi --list-gpus | wc -l)
MAXG="${1:-$NGPU_AVAIL}"
SIZES="${SIZES:-128 192 256}"
REPS="${REPS:-10}"
log() { printf '\n[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

run_suite() {         # $1 = build tag (o2 | o3)
    local tag="$1"
    log "B7 - coefficient precision sweep, single GPU, build=$tag"
    for N in $SIZES; do
        local f="matrix/stencil3d_27pt_${N}.mtx"
        [ -s "$f" ] || { echo "  missing $f, skipping N=$N"; continue; }
        log "  N=$N"
        ./bin/bench_27pt_precision "$f" --reps="$REPS" \
            --csv="$OUT/b7_${tag}_N${N}.csv" > "$OUT/b7_${tag}_N${N}.log" 2>&1 \
            && tail -14 "$OUT/b7_${tag}_N${N}.log" \
            || echo "  FAILED (see $OUT/b7_${tag}_N${N}.log)"
    done

    log "B6 - SoA vs CSR at solver level, build=$tag"
    for np in 1 2 4 8; do
        [ "$np" -le "$MAXG" ] || continue
        for mode in csr soa; do
            log "  np=$np spmv=$mode"
            mpirun --allow-run-as-root -np "$np" ./bin/cg_solver_mgpu_stencil_3d \
                matrix/stencil3d_27pt_192.mtx --stencil=27 --spmv="$mode" \
                > "$OUT/b6_${tag}_np${np}_${mode}.log" 2>&1 \
                && grep -iE "total|iteration|time|residual" "$OUT/b6_${tag}_np${np}_${mode}.log" | tail -6 \
                || echo "  FAILED (see $OUT/b6_${tag}_np${np}_${mode}.log)"
        done
    done
}

log "Session start -- $NGPU_AVAIL GPU(s) present, using up to $MAXG"
nvidia-smi --query-gpu=index,clocks.sm,clocks.mem,temperature.gpu --format=csv > "$OUT/clocks_start.csv"

log "=== Build at -O2 (the configuration the published numbers use) ==="
make clean >/dev/null 2>&1
make -j"$(nproc)" bench_27pt_precision cg_solver_mgpu_stencil_3d >> "$OUT/build.log" 2>&1 \
    || { echo "build -O2 failed"; exit 1; }
run_suite o2

log "=== B1: rebuild at -O3 and repeat ==="
make clean >/dev/null 2>&1
make -j"$(nproc)" NVCCFLAGS='-O3 --ptxas-options=-O3 --ptxas-options=-allow-expensive-optimizations=true -std=c++11' \
    bench_27pt_precision cg_solver_mgpu_stencil_3d >> "$OUT/build.log" 2>&1 \
    && run_suite o3 \
    || echo "  -O3 build failed, skipping B1 (see $OUT/build.log)"

nvidia-smi --query-gpu=index,clocks.sm,clocks.mem,temperature.gpu --format=csv > "$OUT/clocks_end.csv"

log "Done. Collecting."
tar czf results.tgz "$OUT"
echo
echo "  Bring back: results.tgz  ($(du -h results.tgz | cut -f1))"
echo "  Everything is in $OUT/ : CSVs for B7, logs for B6, build log, clocks before and after."
