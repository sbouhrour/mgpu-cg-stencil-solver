#!/usr/bin/env bash
# Prepares a rented multi-GPU instance for the communication study and reports what it can measure.
#
# Answers four questions before any paid time is spent on measurement:
#   1. will Nsight Compute work here (hardware counters), or only timings and nsys?
#   2. what is the machine: GPU count, model, clocks?
#   3. does the 3D solver build and converge on one GPU?
#   4. can host-staged multi-GPU runs be trusted: MPI between local ranks, and device-to-host copies
#      with all GPUs copying at once, unpinned and pinned to each GPU's cores?
#
# Matrices are not shipped or generated on disk: the 3D loaders read only the header and build the
# operator in memory, so a three-line stub is enough for any grid size.
#
# Usage:  ./scripts/benchmarking/rental_preflight.sh       (writes out/rankfile when pinning works)
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT="${OUT_DIR:-out}"
mkdir -p "$OUT" matrix
# On some container images a web proxy listens on port 6006; hwloc's GL plugin, loaded by mpirun,
# takes it for X display :6 and waits forever for an answer. Ranks inherit this setting from mpirun.
export HWLOC_COMPONENTS="${HWLOC_COMPONENTS:--gl}"

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
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i 0 | tr -d ' .')
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


hr "4. Matrix headers (operator built in memory)"
write_header() {                      # $1 = path, $2 = N
    local rows=$(( $2 * $2 * $2 )) k=$(( 3 * $2 - 2 ))
    { printf '%%%%MatrixMarket matrix coordinate real general\n'
      printf '%% STENCIL_GRID_SIZE %d\n' "$2"
      printf '%d %d %d\n' "$rows" "$rows" $(( k * k * k )); } > "$1"
}
for N in 128 256 512; do
    write_header "matrix/stencil3d_27pt_${N}.mtx" "$N"
    printf '  N=%-4s rows=%-12s 27-point nnz=%s\n' "$N" $((N*N*N)) $(( (3*N-2)*(3*N-2)*(3*N-2) ))
done

hr "5. Build"
# Compiled offline for the architecture actually present, rather than left to JIT the embedded PTX
# of whatever target nvcc defaults to. comm_setup.sh builds the full toolchain; this build only
# checks that the solver compiles with what the node has.
if ! command -v mpirun >/dev/null; then
    echo "  no mpirun: install MPI (comm_setup.sh does) before multi-GPU work"; exit 1
fi
if make -j"$(nproc)" ARCH="$CC" cg_solver_mgpu_stencil_3d > "$OUT/build.log" 2>&1; then
    echo "  cg_solver_mgpu_stencil_3d  OK  (sm_${CC})"
else
    echo "  cg_solver_mgpu_stencil_3d  FAILED"
    grep -iE 'error|No rule' "$OUT/build.log" | head -10
    exit 1
fi

hr "6. Smoke test (27-point 128^3, 1 GPU, expect 151 iterations)"
ROOTOPT=(); [ "$(id -u)" = 0 ] && ROOTOPT=(--allow-run-as-root)
timeout 300 mpirun "${ROOTOPT[@]}" -np 1 ./bin/cg_solver_mgpu_stencil_3d matrix/stencil3d_27pt_128.mtx \
    --stencil=27 --runs=3 > "$OUT/smoke.txt" 2>&1
grep -aE 'Converged|Time \(median\)' "$OUT/smoke.txt" | sed 's/^/  /'

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

hr "8. Device-to-host copies, all GPUs at once (128 KB, pinned host memory, processes unpinned)"
# Section 7 moves host buffers only; the staged halo first copies each plane from the GPU. Seen on
# 2026-09-25: an 8x A100 node passed section 7 at 15 GB/s, copied 128 KB from one GPU in 18.5 us, and
# took 162 us (0.8 GB/s) when the eight GPUs copied together. Staged 8-GPU CG ran 1.9x slower than
# published, while the same solve with NCCL and no host copy ran faster than published. Part of it was
# NUMA placement the VM did not expose (copies to the other socket's memory ran 3x slower); pinning each
# rank to its GPU's socket removed that part, not the rest. One process per GPU, as in the solver; each
# copies for a fixed time, so the launch skew between them does not matter. The threshold (half of one
# GPU copying alone) is a judgement: that node scored about 0.15, and a healthy node should lose little
# more than the share of a PCIe switch.
D2H_VERDICT="not tested"
NGPU=$(nvidia-smi -L | wc -l)
cat > "$OUT/d2h_probe.cu" <<'EOC'
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cuda_runtime.h>
int main(int argc, char** argv) {
    const size_t bytes = 128 * 1024;
    const double seconds = atof(argv[2]);
    cudaSetDevice(atoi(argv[1]));
    void *d, *h;
    cudaMalloc(&d, bytes);
    cudaMallocHost(&h, bytes);
    cudaStream_t s;
    cudaStreamCreate(&s);
    for (int i = 0; i < 100; i++) cudaMemcpyAsync(h, d, bytes, cudaMemcpyDeviceToHost, s);
    cudaStreamSynchronize(s);
    auto t0 = std::chrono::steady_clock::now();
    long n = 0;
    double dt = 0.0;
    while (dt < seconds) {  /* synchronous per copy, like one staged halo */
        cudaMemcpyAsync(h, d, bytes, cudaMemcpyDeviceToHost, s);
        cudaStreamSynchronize(s);
        n++;
        dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    }
    printf("%.2f\n", (double)bytes * n / dt / 1e9);
    return 0;
}
EOC
if [ "$NGPU" -lt 2 ]; then
    D2H_VERDICT="one GPU, not relevant"
elif nvcc -O2 -arch=sm_"$CC" -o "$OUT/d2h_probe" "$OUT/d2h_probe.cu" 2>"$OUT/d2h_probe.build.log"; then
    ALONE=$("$OUT/d2h_probe" 0 1)
    for g in $(seq 0 $((NGPU - 1))); do "$OUT/d2h_probe" "$g" 3 > "$OUT/d2h_probe.$g" & done
    wait
    WORST=$(sort -g "$OUT"/d2h_probe.[0-9]* | head -1)
    rm -f "$OUT"/d2h_probe.[0-9]*
    if awk -v w="$WORST" -v a="$ALONE" -v r=0.5 'BEGIN{exit !(w >= r * a)}'; then
        D2H_VERDICT="OK, ${WORST} GB/s per GPU with ${NGPU} copying (${ALONE} alone) -- staged runs can proceed"
    else
        D2H_VERDICT="SLOW, ${WORST} GB/s per GPU with ${NGPU} copying (${ALONE} alone) -- no staged or AmgX MPI runs"
    fi
else
    D2H_VERDICT="nvcc failed (see $OUT/d2h_probe.build.log)"
fi
echo "  $D2H_VERDICT"
printf 'd2h_concurrent=%s\n' "$D2H_VERDICT" >> "$OUT/hw_info.txt"

hr "8b. Same copies, each process pinned to its GPU's local cores"
# Section 8 lets the scheduler place each process. On 2026-09-30 and 2026-10-01 that placement decided
# the verdict: one node scored 0.07 to 0.18 unpinned and 0.81 pinned, and on another the single-GPU
# reference moved between 3.7 and 9.5 GB/s depending on the socket it landed on. The solver's ranks
# have the same exposure, so the runs are pinned too: make_rankfile.sh writes the rankfile they use.
# This verdict is the one to gate on. A node that hides NUMA (empty local_cpulist, typical of VMs)
# cannot be pinned and is reported as such.
D2H_PIN_VERDICT="not tested"
CPUS=()
for b in $(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader); do
    CPUS+=("$(cat /sys/bus/pci/devices/"$(echo "${b: -12}" | tr 'A-Z' 'a-z')"/local_cpulist 2>/dev/null)")
done
if [ "$NGPU" -ge 2 ] && [ -x "$OUT/d2h_probe" ] && [ -n "${CPUS[0]:-}" ]; then
    ./scripts/benchmarking/make_rankfile.sh > "$OUT/rankfile" && sed 's/^/  /' "$OUT/rankfile"
    ALONE=$(taskset -c "${CPUS[0]}" "$OUT/d2h_probe" 0 1)
    for g in $(seq 0 $((NGPU - 1))); do taskset -c "${CPUS[$g]}" "$OUT/d2h_probe" "$g" 3 > "$OUT/d2h_pin.$g" & done
    wait
    WORST=$(sort -g "$OUT"/d2h_pin.[0-9]* | head -1)
    rm -f "$OUT"/d2h_pin.[0-9]*
    if awk -v w="$WORST" -v a="$ALONE" -v r=0.5 'BEGIN{exit !(w >= r * a)}'; then
        D2H_PIN_VERDICT="OK, ${WORST} GB/s per GPU with ${NGPU} copying (${ALONE} alone) -- run pinned: RANKFILE=$PWD/$OUT/rankfile"
    else
        D2H_PIN_VERDICT="SLOW, ${WORST} GB/s per GPU with ${NGPU} copying (${ALONE} alone) -- no staged or AmgX MPI runs, even pinned"
    fi
elif [ -z "${CPUS[0]:-}" ]; then
    D2H_PIN_VERDICT="not tested (no local_cpulist: NUMA hidden, likely a VM)"
fi
echo "  $D2H_PIN_VERDICT"
printf 'd2h_concurrent_pinned=%s\n' "$D2H_PIN_VERDICT" >> "$OUT/hw_info.txt"

hr "Verdict"
if [ "$NCU_LIKELY" = 1 ] && command -v ncu >/dev/null; then
    # Decided from ncu's own exit status and output, written to a file first: "ncu ... | grep -q"
    # under pipefail reports ncu's SIGPIPE, not grep's match, and read a refusal as success
    ncu --metrics dram__bytes.sum ./bin/cg_solver_mgpu_stencil_3d matrix/stencil3d_27pt_128.mtx \
        --stencil=27 --max-iters=2 --runs=3 > "$OUT/ncu_check.txt" 2>&1
    NCU_RC=$?
    if grep -qi ERR_NVGPUCTRPERM "$OUT/ncu_check.txt"; then
        echo "  ncu:        DENIED by host -- timings and nsys only"
    elif [ "$NCU_RC" = 0 ] && grep -q dram__bytes.sum "$OUT/ncu_check.txt"; then
        echo "  ncu:        WORKS -- counters can be captured"
    else
        echo "  ncu:        FAILED (exit $NCU_RC, see $OUT/ncu_check.txt)"
    fi
else
    echo "  ncu:        unavailable -- timings and nsys only"
fi
echo "  mpi:        $MPI_VERDICT"
echo "  d2h:        $D2H_VERDICT"
echo "  d2h pinned: $D2H_PIN_VERDICT"
echo "  Next: comm_setup.sh (if not done), then comm_preflight.sh"

