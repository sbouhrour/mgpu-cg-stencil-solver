#!/usr/bin/env bash
# Remote validation runner — 27-point SpMV variants on datacenter GPUs
# (A100 / H100), where clocks can be locked and wall-clock times are
# meaningful. Designed to run WITHOUT Nsight Compute (GPU performance
# counters are usually blocked on rented instances); if ncu is present
# and permitted, an optional profiling section runs as a bonus.
#
# Usage (from the repo root, on the rented box):
#   ./exploration/run_remote.sh
#
# Env overrides:
#   SIZES="128 192 256 320 384"   grid sizes to attempt (memory-gated)
#   REPS=3                        timing repetitions per size
#   NCU=auto                      auto | 0 (skip ncu even if present)
#
# Outputs: exploration/data/remote/<gpu>_<stamp>/ with run.log, hw_info,
# clock traces, medians.csv, SUMMARY.md, and a .tar.gz of the lot.

set -u
cd "$(dirname "$0")/.."

SIZES=${SIZES:-"128 192 256 320 384"}
REPS=${REPS:-3}
NCU=${NCU:-auto}

STAMP=$(date +%Y%m%d_%H%M)
GPU_RAW=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
GPU_TAG=$(echo "$GPU_RAW" | tr ' /' '__' | tr -cd '[:alnum:]_-')
OUT="exploration/data/remote/${GPU_TAG}_${STAMP}"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/run.log") 2>&1

echo "==================================================================="
echo "Remote 27pt SpMV validation — $GPU_RAW — $STAMP"
echo "Sizes: $SIZES   Reps: $REPS"
echo "==================================================================="

# ---------------------------------------------------------------- HW info
{
    nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total,memory.free --format=csv
    echo "--- clocks (max) ---"
    nvidia-smi --query-gpu=clocks.max.sm,clocks.max.memory --format=csv
    echo "--- host ---"
    free -g | head -2
    nvcc --version | tail -1
} | tee "$OUT/hw_info.txt"

VRAM_FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
HOST_AVAIL_GB=$(awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo)

# ------------------------------------------------------------------ build
./exploration/build.sh || { echo "BUILD FAILED"; exit 1; }

# ------------------------------------------- clock locking (best effort)
MAX_SM=$(nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader,nounits | head -1)
MAX_MEM=$(nvidia-smi --query-gpu=clocks.max.memory --format=csv,noheader,nounits | head -1)
CLOCK_LOCK="none"
SMI="nvidia-smi"
command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null && SMI="sudo nvidia-smi"
$SMI -pm 1 >/dev/null 2>&1
if $SMI -lgc "$MAX_SM" >/dev/null 2>&1; then
    CLOCK_LOCK="lgc=$MAX_SM"
    $SMI -lmc "$MAX_MEM" >/dev/null 2>&1 && CLOCK_LOCK="$CLOCK_LOCK,lmc=$MAX_MEM"
fi
echo "Clock lock: $CLOCK_LOCK (verify against the per-run clock traces)"

# ------------------------------------- header-only matrix files (loader
# reads only '% STENCIL_GRID_SIZE N'; entries are generated in memory)
gen_matrix_header() {
    local N=$1 f="matrix/stencil3d_27pt_$1.mtx"
    [ -s "$f" ] && return 0
    local ROWS=$((N * N * N))
    local T=$((3 * N - 2))
    local NNZ=$((T * T * T))
    mkdir -p matrix
    printf '%%%%MatrixMarket matrix coordinate real general\n%% STENCIL_GRID_SIZE %d\n%d %d %d\n' \
        "$N" "$ROWS" "$ROWS" "$NNZ" > "$f"
    echo "Generated header-only $f (rows=$ROWS nnz=$NNZ)"
}

# Memory needs per size (GB): device = CSR + SoA + 3 vectors; host = peak
# of (COO+CSR) during load. Gated against free VRAM / MemAvailable.
mem_need() {
    case $1 in
        64)  echo "1 1" ;;
        128) echo "2 2" ;;
        192) echo "5 6" ;;
        256) echo "10 13" ;;
        320) echo "19 25" ;;
        384) echo "33 43" ;;
        *)   echo "999 999" ;;
    esac
}

# ------------------------------------------------------------- main loop
CSV="$OUT/medians.csv"
echo "gpu,N,rep,variant,median_ms" > "$CSV"
RAN_SIZES=""

for N in $SIZES; do
    read -r DEV_GB HOST_GB <<< "$(mem_need "$N")"
    DEV_MB=$((DEV_GB * 1024))
    if [ "$DEV_MB" -gt $((VRAM_FREE_MB * 90 / 100)) ]; then
        echo "SKIP N=$N: needs ~${DEV_GB} GB device, free ${VRAM_FREE_MB} MiB"
        continue
    fi
    if [ "$HOST_GB" -gt $((HOST_AVAIL_GB * 85 / 100)) ]; then
        echo "SKIP N=$N: needs ~${HOST_GB} GB host RAM, available ${HOST_AVAIL_GB} GiB"
        continue
    fi
    gen_matrix_header "$N"

    for rep in $(seq 1 "$REPS"); do
        echo "----- N=$N rep $rep/$REPS -----"
        TRACE="$OUT/clock_trace_N${N}_rep${rep}.csv"
        nvidia-smi --query-gpu=clocks.sm,clocks.memory,temperature.gpu,power.draw \
            --format=csv,noheader -l 1 > "$TRACE" 2>/dev/null &
        TRACE_PID=$!

        RUN_OUT="$OUT/bench_N${N}_rep${rep}.log"
        ./bin/bench_27pt_variants "matrix/stencil3d_27pt_${N}.mtx" \
            2>&1 | grep -vE "Generating entries|➤" > "$RUN_OUT"
        RC=${PIPESTATUS[0]}
        kill "$TRACE_PID" 2>/dev/null
        wait "$TRACE_PID" 2>/dev/null

        PASS=$(grep -c "CHECKSUM PASSED" "$RUN_OUT")
        if [ "$RC" -ne 0 ] || [ "$PASS" -ne 4 ]; then
            echo "ABORT: N=$N rep=$rep rc=$RC checksum_passed=$PASS/4 — see $RUN_OUT"
            tail -20 "$RUN_OUT"
            exit 1
        fi
        # harness line: "[i] name ...  median X.XXXX ms"
        awk -v gpu="$GPU_TAG" -v n="$N" -v rep="$rep" '
            $1 ~ /^\[[0-9]+\]$/ && / median / {
                idx = substr($1, 2, length($1) - 2)
                for (i = 1; i <= NF; i++) if ($i == "median") m = $(i + 1)
                name = (idx == 0) ? "baseline" : (idx == 1) ? "soa" : \
                       (idx == 2) ? "soa_b128" : (idx == 3) ? "soa_b512" : "soa_ldcs"
                printf "%s,%s,%s,%s,%s\n", gpu, n, rep, name, m
            }' "$RUN_OUT" >> "$CSV"
        grep "median" "$RUN_OUT"
    done
    RAN_SIZES="$RAN_SIZES $N"
done

# ------------------------------------------------- optional NCU profiles
NCU_STATUS="not present"
[ "$NCU" = "0" ] && NCU_STATUS="skipped (NCU=0)"
if [ "$NCU" != "0" ] && command -v ncu >/dev/null 2>&1; then
    gen_matrix_header 64
    echo "Probing ncu counter permissions on 64^3..."
    if LC_ALL=C ncu --metrics gpu__time_duration.sum -k regex:stencil27_soa_k \
        --launch-count 1 ./bin/bench_27pt_variants matrix/stencil3d_27pt_64.mtx \
        --profile > "$OUT/ncu_probe.log" 2>&1 \
        && ! grep -q "ERR_NVGPUCTRPERM" "$OUT/ncu_probe.log"; then
        NCU_STATUS="available — full profiles collected at 192^3"
        for KSPEC in "stencil27_csr:baseline" "stencil27_soa_k:soa"; do
            KREG="${KSPEC%%:*}"; TAG="${KSPEC##*:}"
            LC_ALL=C ncu --set full -k "regex:$KREG" --launch-count 1 \
                -o "$OUT/ncu_${TAG}_192_${GPU_TAG}" -f \
                ./bin/bench_27pt_variants matrix/stencil3d_27pt_192.mtx --profile \
                > /dev/null 2>&1 || NCU_STATUS="probe ok but full profile failed"
        done
    else
        NCU_STATUS="present but counters blocked (expected on rented instances)"
    fi
fi
echo "NCU: $NCU_STATUS"

# ---------------------------------------------------------------- summary
python3 - "$CSV" "$OUT/SUMMARY.md" "$GPU_RAW" "$CLOCK_LOCK" "$NCU_STATUS" <<'EOF'
import csv, sys
from collections import defaultdict
rows = list(csv.DictReader(open(sys.argv[1])))
med = defaultdict(list)
for r in rows:
    med[(int(r["N"]), r["variant"])].append(float(r["median_ms"]))
def m(vals):
    s = sorted(vals)
    return s[len(s)//2] if len(s) % 2 else 0.5*(s[len(s)//2-1]+s[len(s)//2])
sizes = sorted({int(r["N"]) for r in rows})
variants = ["baseline", "soa", "soa_b128", "soa_b512", "soa_ldcs"]
with open(sys.argv[2], "w") as f:
    f.write(f"# Remote validation summary — {sys.argv[3]}\n\n")
    f.write(f"Clock lock: {sys.argv[4]}  |  NCU: {sys.argv[5]}\n")
    f.write("Median of per-rep medians (ms); ratio = variant / baseline.\n\n")
    f.write("| N | " + " | ".join(variants) + " | soa/baseline |\n")
    f.write("|" + "---|" * (len(variants) + 2) + "\n")
    for n in sizes:
        vals = [m(med[(n, v)]) if med[(n, v)] else None for v in variants]
        cells = [f"{v:.3f}" if v else "-" for v in vals]
        ratio = f"**{vals[1]/vals[0]:.3f}**" if vals[0] and vals[1] else "-"
        f.write(f"| {n}³ | " + " | ".join(cells) + f" | {ratio} |\n")
    f.write("\nPer-rep spread is in medians.csv; clock traces per run alongside.\n")
print(open(sys.argv[2]).read())
EOF

# unlock clocks (best effort)
$SMI -rgc >/dev/null 2>&1
$SMI -rmc >/dev/null 2>&1

tar -czf "${OUT}.tar.gz" "$OUT"
echo "==================================================================="
echo "DONE. Bring back: ${OUT}.tar.gz (or commit $OUT on a results branch)"
echo "==================================================================="
