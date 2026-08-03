#!/usr/bin/env bash
# Prepares a rented GPU instance for a benchmark session and reports what it can measure.
#
# Answers three questions before any paid time is spent on measurement:
#   1. will Nsight Compute work here (hardware counters), or only timings and nsys?
#   2. what is the machine — GPU count, model, clocks, theoretical bandwidth?
#   3. does everything build and run?
#
# Matrices are not shipped or generated on disk: the 3D 27-point loader reads only the header and
# builds the operator in memory, so a three-line stub is enough for any grid size.
#
# Usage:  ./scripts/benchmarking/rental_preflight.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT="${OUT_DIR:-out}"
mkdir -p "$OUT" matrix

hr() { printf '\n===== %s =====\n' "$1"; }

hr "1. Nsight Compute verdict"
# A remapped UID namespace means the NVIDIA driver will refuse counter access whatever the
# container capabilities report, because it checks the initial namespace.
if head -1 /proc/self/uid_map 2>/dev/null | grep -qE '^\s*0\s+0\s'; then
    echo "  uid_map: initial namespace -- ncu may work"
    NCU_LIKELY=1
else
    echo "  uid_map: REMAPPED ($(head -1 /proc/self/uid_map 2>/dev/null | tr -s ' ')) -- ncu will be denied"
    NCU_LIKELY=0
fi
grep -i RestrictProfiling /proc/driver/nvidia/params 2>/dev/null || echo "  (host module params not exposed, normal in a container)"

hr "2. Hardware"
nvidia-smi --query-gpu=index,name,compute_cap,memory.total,driver_version --format=csv | tee "$OUT/hw_gpus.csv"
NGPU=$(nvidia-smi --list-gpus | wc -l)
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' .')
echo "  GPUs: $NGPU   compute capability: sm_${CC}"
nvidia-smi -q -d CLOCK | grep -A3 "Max Clocks" | head -4 | tee "$OUT/hw_clocks.txt"
{ echo "gpus=$NGPU"; echo "cc=$CC"; echo "date=$(date -Is)"; uname -a; } > "$OUT/hw_info.txt"

hr "3. Toolchain"
for t in nvcc mpirun ncu nsys; do printf '  %-8s %s\n' "$t" "$(command -v $t || echo ABSENT)"; done
if ! command -v nvcc >/dev/null; then
    cat <<'EOS'
  nvcc missing. On a bare Ubuntu 24.04 image:
    cd /tmp && wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
    dpkg -i cuda-keyring_1.1-1_all.deb && apt-get update -qq && apt-get install -y cuda-toolkit
    export PATH=/usr/local/cuda/bin:$PATH
EOS
    exit 1
fi

hr "4. Matrix headers (operator is built in memory, nnz = (3N-2)^3)"
for N in 128 192 256 320; do
    ROWS=$((N*N*N)); K=$((3*N-2)); NNZ=$((K*K*K))
    F="matrix/stencil3d_27pt_${N}.mtx"
    if [ ! -s "$F" ]; then
        printf '%%%%MatrixMarket matrix coordinate real general\n%%%% STENCIL_GRID_SIZE %d\n%d %d %d\n' \
            "$N" "$ROWS" "$ROWS" "$NNZ" > "$F"
    fi
    printf '  N=%-4s rows=%-12s nnz=%s\n' "$N" "$ROWS" "$NNZ"
done

hr "5. Build"
make clean >/dev/null 2>&1
if make -j"$(nproc)" bench_27pt_precision cg_solver_mgpu_stencil_3d > "$OUT/build.log" 2>&1; then
    echo "  build OK  (log: $OUT/build.log)"
else
    echo "  BUILD FAILED -- see $OUT/build.log"; tail -20 "$OUT/build.log"; exit 1
fi

hr "6. Smoke test"
./bin/bench_27pt_precision matrix/stencil3d_27pt_128.mtx --reps=2 2>&1 | tail -12 | tee "$OUT/smoke.txt"

hr "Verdict"
if [ "$NCU_LIKELY" = 1 ] && command -v ncu >/dev/null; then
    if ncu --metrics dram__bytes.sum ./bin/bench_27pt_precision matrix/stencil3d_27pt_128.mtx --reps=1 2>&1 \
         | grep -qi ERR_NVGPUCTRPERM; then
        echo "  ncu: DENIED by host -- timings and nsys only"
    else
        echo "  ncu: WORKS -- capture counters too, and note the provider id"
    fi
else
    echo "  ncu: unavailable -- timings and nsys only (expected on container marketplaces)"
fi
echo "  Ready. Next: ./scripts/benchmarking/rental_session.sh"
