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
# The host driver decides. RmProfilingAdminOnly=0 opens the counters to every user, whatever the
# container's user namespace. Only when it is 1 does a remapped uid_map matter: the driver then wants
# root of the initial namespace, which a container never is. Seen on 2026-09-24: uid_map remapped,
# RmProfilingAdminOnly=0, ncu working. The final verdict below runs ncu for real either way.
ADMIN_ONLY=$(awk '/RmProfilingAdminOnly/{print $2}' /proc/driver/nvidia/params 2>/dev/null)
if [ "$ADMIN_ONLY" = 0 ]; then
    echo "  RmProfilingAdminOnly: 0 -- counters open to all users, ncu should work"
    NCU_LIKELY=1
elif head -1 /proc/self/uid_map 2>/dev/null | grep -qE '^\s*0\s+0\s'; then
    echo "  RmProfilingAdminOnly: ${ADMIN_ONLY:-unknown}, uid_map initial -- ncu may work as root"
    NCU_LIKELY=1
else
    echo "  RmProfilingAdminOnly: ${ADMIN_ONLY:-unknown}, uid_map REMAPPED -- ncu will be denied"
    NCU_LIKELY=0
fi

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
# One header line carries the grid size, an optional second one the coefficient contrast. A single
# leading '%' marks a Matrix Market comment; two would not be recognised by the loader.
write_header() {                      # $1 = path, $2 = N, $3 = contrast ("" for constant)
    local rows=$(( $2 * $2 * $2 )) k=$(( 3 * $2 - 2 ))
    {
        printf '%%%%MatrixMarket matrix coordinate real general\n'
        printf '%% STENCIL_GRID_SIZE %d\n' "$2"
        [ -n "$3" ] && printf '%% STENCIL_CONTRAST %s\n' "$3"
        printf '%d %d %d\n' "$rows" "$rows" $(( k * k * k ))
    } > "$1"
}
for N in 128 192 256 320; do
    ROWS=$((N*N*N)); K=$((3*N-2))
    write_header "matrix/stencil3d_27pt_${N}.mtx" "$N" ""
    printf '  N=%-4s rows=%-12s nnz=%-12s constant\n' "$N" "$ROWS" "$((K*K*K))"
done
# Variable-coefficient variants: the only ones that can measure what reduced precision costs, since the
# constant operator's coefficients are exact in every format down to eight bits.
for N in 128 192; do
    for C in 0.1 0.7 3.0; do
        write_header "matrix/stencil3d_27pt_${N}_var${C}.mtx" "$N" "$C"
        printf '  N=%-4s contrast=%-5s variable\n' "$N" "$C"
    done
done

hr "5. Build"
# Built separately, and the single-GPU one first. The multi-GPU solver needs MPI; the precision
# benchmark does not include mpi.h anywhere in its five sources. Building them together made a
# missing mpirun look like a failure of the whole session, on an instance that bills by the hour
# and where the single-GPU question was the only one on the programme.
#
# Compiled offline for the architecture actually present, rather than left to JIT the embedded PTX
# of whatever target nvcc defaults to.
make clean >/dev/null 2>&1
BUILD_OK=0
if make -j"$(nproc)" ARCH="$CC" bench_27pt_precision > "$OUT/build.log" 2>&1; then
    echo "  bench_27pt_precision  OK  (sm_${CC})   -- single-GPU work can proceed"
    BUILD_OK=1
else
    echo "  bench_27pt_precision  FAILED"
    grep -iE 'error|No rule' "$OUT/build.log" | head -10
    echo "  full log: $OUT/build.log"
fi
if command -v mpirun >/dev/null; then
    if make -j"$(nproc)" ARCH="$CC" cg_solver_mgpu_stencil_3d >> "$OUT/build.log" 2>&1; then
        echo "  cg_solver_mgpu_stencil_3d  OK       -- multi-GPU work can proceed"
    else
        echo "  cg_solver_mgpu_stencil_3d  FAILED   -- multi-GPU work only; see $OUT/build.log"
    fi
else
    echo "  cg_solver_mgpu_stencil_3d  SKIPPED -- no mpirun. Single-GPU work is unaffected;"
    echo "                                        install MPI only if this session covers B6."
fi
[ "$BUILD_OK" = 1 ] || exit 1

hr "6. Smoke test"
./bin/bench_27pt_precision matrix/stencil3d_27pt_128.mtx --reps=2 2>&1 | tail -12 | tee "$OUT/smoke.txt"

hr "7. MPI between two ranks on this node (512 KB, host buffers)"
# The multi-GPU solvers stage every halo through host memory and hand it to MPI, so a node whose MPI
# moves data slowly between local processes is unusable for scaling runs, however good its GPUs are.
# Seen on 2026-09-24: an 8x A100 node reproduced single-GPU results within 1% and ran 8-GPU CG 2.2x
# slower than published; nsys put the loss on MPI_Waitall, 512 KB messages at ~0.8 GB/s. 512 KB is one
# halo plane of a 256^3 grid in double precision. The 3 GB/s threshold is a judgement: well above that
# node, well below what shared memory usually sustains. Below it, measure single-GPU work only.
MPI_VERDICT="not tested (no mpicc/mpirun)"
if command -v mpicc >/dev/null && command -v mpirun >/dev/null; then
    cat > "$OUT/mpi_pingpong.c" <<'EOC'
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
int main(int argc, char** argv) {
    const int bytes = 512 * 1024, warmup = 20, iters = 200;
    int rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    char* buf = malloc(bytes);
    for (int i = 0; i < bytes; i++) buf[i] = (char)i;
    double t0 = 0.0;
    for (int i = 0; i < warmup + iters; i++) {
        if (i == warmup) { MPI_Barrier(MPI_COMM_WORLD); t0 = MPI_Wtime(); }
        if (rank == 0) {
            MPI_Send(buf, bytes, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
            MPI_Recv(buf, bytes, MPI_CHAR, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } else if (rank == 1) {
            MPI_Recv(buf, bytes, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(buf, bytes, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
        }
    }
    double dt = MPI_Wtime() - t0;
    if (rank == 0)  /* two messages per round trip */
        printf("%.2f %.1f\n", 2.0 * bytes * iters / dt / 1e9, dt / iters * 1e6);
    free(buf);
    MPI_Finalize();
    return 0;
}
EOC
    if mpicc -O2 -o "$OUT/mpi_pingpong" "$OUT/mpi_pingpong.c" 2>"$OUT/mpi_pingpong.build.log"; then
        MPI_OUT=$(timeout 120 mpirun --allow-run-as-root -np 2 "$OUT/mpi_pingpong" 2>"$OUT/mpi_pingpong.err" | tail -1)
        MPI_GBS=${MPI_OUT%% *}
        if [ -z "$MPI_GBS" ]; then
            MPI_VERDICT="FAILED or hung (see $OUT/mpi_pingpong.err) -- no multi-GPU runs"
        elif awk -v b="$MPI_GBS" 'BEGIN{exit !(b >= 3.0)}'; then
            MPI_VERDICT="OK, ${MPI_GBS} GB/s (round trip ${MPI_OUT##* } us) -- multi-GPU runs can proceed"
        else
            MPI_VERDICT="SLOW, ${MPI_GBS} GB/s (round trip ${MPI_OUT##* } us) -- single-GPU work only"
        fi
    else
        MPI_VERDICT="mpicc failed (see $OUT/mpi_pingpong.build.log)"
    fi
fi
echo "  $MPI_VERDICT"
printf 'mpi_512k=%s\n' "$MPI_VERDICT" >> "$OUT/hw_info.txt"

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
echo "  mpi: $MPI_VERDICT"
echo "  Ready. Next: ./scripts/benchmarking/rental_session.sh"
