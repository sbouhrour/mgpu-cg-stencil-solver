#!/usr/bin/env bash
# One-GPU session for the 27-point SpMV benchmark: build, timings, then Nsight Compute bytes.
#
#   1. constant coefficients (the solver's matrix), N = 128, 256, 384
#   2. variable symmetric coefficients, same sizes
#   3. L2 fetch granularity 32 / 64 / 128 bytes, N = 256, 384
#   4. ncu: DRAM bytes per row of every variant at N = 256
#
# Usage: [ARCH=80] [OUT=dir] [SIZES=128,256,384] bench/spmv_27pt/run_session.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

ARCH=${ARCH:-80}
SIZES=${SIZES:-128,256,384}
OUT=${OUT:-results_spmv27_$(date +%Y%m%d_%H%M)}
mkdir -p "$OUT"

nvidia-smi -q > "$OUT/nvidia-smi.txt" 2>&1 || true
nvcc --version > "$OUT/nvcc.txt"
git rev-parse HEAD > "$OUT/commit.txt"
make bench_spmv_27pt SPMV27_ARCH="$ARCH" 2>&1 | tee "$OUT/build.log"
BIN=$( (ls bin/bench_spmv_27pt bin/*/bench_spmv_27pt 2>/dev/null || true) | head -n 1)
echo "binary: $BIN"

"$BIN" --sizes="$SIZES" --coeffs=const --csv="$OUT/timings.csv" | tee "$OUT/1_const.txt"
"$BIN" --sizes="$SIZES" --coeffs=var --csv="$OUT/timings.csv" | tee "$OUT/2_var.txt"
for g in 32 64 128; do
    "$BIN" --sizes=256,384 --l2fetch="$g" --csv="$OUT/timings.csv" | tee "$OUT/3_l2fetch_$g.txt"
done

if command -v "${NCU:-ncu}" > /dev/null; then
    BIN="$BIN" N=256 OUT="$OUT/ncu" bench/spmv_27pt/ncu_bytes.sh | tee "$OUT/4_ncu.txt"
else
    echo "ncu not found: step 4 skipped" | tee "$OUT/4_ncu_skipped.txt"
fi
echo "Results in $OUT"
