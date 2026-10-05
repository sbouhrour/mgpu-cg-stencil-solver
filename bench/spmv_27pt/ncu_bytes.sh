#!/usr/bin/env bash
# DRAM bytes per row and kernel time of each SpMV variant, measured by Nsight Compute.
#
# Profiles only the timed launches of bench_spmv_27pt (cudaProfilerStart/Stop), at the clocks the
# GPU actually runs (--clock-control none). Bytes = dram__bytes_read + dram__bytes_write, which do
# not depend on the clock; time = gpu__time_duration summed over the kernels of one SpMV call.
#
# Usage: N=256 [COEFFS=const] [L2FETCH=32] [REPS=5] [VARIANTS="rowmajor staged"] \
#        [BIN=bin/bench_spmv_27pt] [OUT=dir] bench/spmv_27pt/ncu_bytes.sh
set -euo pipefail

BIN=${BIN:-bin/bench_spmv_27pt}
N=${N:-256}
COEFFS=${COEFFS:-const}
REPS=${REPS:-5}
L2FETCH=${L2FETCH:-}
VARIANTS=${VARIANTS:-"cusparse-alg1 cusparse-alg2 rowmajor staged"}
OUT=${OUT:-ncu_spmv27_N${N}_${COEFFS}${L2FETCH:+_l2f$L2FETCH}}
NCU=${NCU:-ncu}

METRICS=dram__bytes_read.sum,dram__bytes_write.sum,gpu__time_duration.sum
METRICS+=,lts__t_sectors_srcunit_tex_op_read.sum,sm__warps_active.avg.pct_of_peak_sustained_active
METRICS+=,launch__registers_per_thread

mkdir -p "$OUT"
for v in $VARIANTS; do
    echo "== $v"
    "$NCU" --profile-from-start off --clock-control none --csv --page raw --print-units base \
        --metrics "$METRICS" \
        "$BIN" --sizes="$N" --coeffs="$COEFFS" --reps="$REPS" --only="$v" \
        ${L2FETCH:+--l2fetch=$L2FETCH} > "$OUT/$v.csv" 2> "$OUT/$v.err" || {
        echo "ncu failed for $v, see $OUT/$v.err"
        continue
    }
done

python3 - "$OUT" "$N" "$REPS" $VARIANTS <<'EOF'
import csv, io, sys, statistics
out, N, reps, variants = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4:]
n = N ** 3
print(f"\nN={N}, {n} rows, metrics per SpMV call (median over {reps} calls)")
print(f"{'variant':14s} {'kernels':>7s} {'time ms':>9s} {'DRAM B/row':>11s} {'read':>7s} {'write':>6s}"
      f" {'GB/s':>7s} {'L2 rd B/row':>11s} {'occ %':>6s}")
for v in variants:
    try:
        text = open(f"{out}/{v}.csv").read()
    except OSError:
        continue
    lines = [l for l in text.splitlines() if l.startswith('"')]
    if len(lines) < 3:
        print(f"{v:14s} no data")
        continue
    rows = list(csv.DictReader(io.StringIO("\n".join([lines[0]] + lines[2:]))))
    per_call = max(1, len(rows) // reps)
    calls = [rows[i:i + per_call] for i in range(0, per_call * reps, per_call)]
    def tot(call, key):
        return sum(float(r[key].replace(",", "")) for r in call)
    t = statistics.median(tot(c, "gpu__time_duration.sum") for c in calls)  # ns
    rd = statistics.median(tot(c, "dram__bytes_read.sum") for c in calls)
    wr = statistics.median(tot(c, "dram__bytes_write.sum") for c in calls)
    l2 = statistics.median(tot(c, "lts__t_sectors_srcunit_tex_op_read.sum") for c in calls) * 32
    occ = statistics.median(float(c[-1]["sm__warps_active.avg.pct_of_peak_sustained_active"])
                            for c in calls)
    print(f"{v:14s} {per_call:7d} {t / 1e6:9.4f} {(rd + wr) / n:11.1f} {rd / n:7.1f} {wr / n:6.1f}"
          f" {(rd + wr) / t:7.0f} {l2 / n:11.1f} {occ:6.1f}")
EOF
