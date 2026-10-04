#!/usr/bin/env bash
# Installs, on a rented multi-GPU node, everything the communication study needs: CUDA 12.8 (the
# toolkit, cuBLAS and cuSPARSE the published results were measured with), NCCL 2.31.2, NVSHMEM
# 3.7.2, the system Open MPI, UCX 1.18.1 + Open MPI 5.0.8 built with CUDA, nccl-tests, and AmgX
# (commit cc1cebd). Then builds the solver and the AmgX driver twice and writes two environment
# files:
#
#   default build      system Open MPI     bin/cg_solver_mgpu_stencil_3d, amgx_cg_solver_mgpu
#                      env: $PREFIX/env.sh  backends staged, nccl, nvshmem; AmgX MPI
#   CUDA-aware build   Open MPI 5 + UCX    bin/cuda-aware/cg_solver_mgpu_stencil_3d,
#                      env: $PREFIX/env-cuda-aware.sh   amgx_cg_solver_mgpu_cuda_aware
#                                          backend gpuaware; AmgX MPI_DIRECT
#
# Why two builds: on an 8x A100-SXM4-80GB node, the staged path (host buffers handed to MPI) ran
# 323.9 us per CG iteration under the system Open MPI 4.1.6 and 461.9 us under this CUDA-aware
# Open MPI 5.0.8 (27-point, 128^3, 8 GPUs), with identical kernels; NCCL with device dots does not
# call MPI in the loop and ran 232 us under both. comm_matrix.sh and comm_preflight.sh run each
# configuration with the build it needs.
#
# NCCL 2.31.2 is only packaged for CUDA 12.9 and later; it carries its own CUDA runtime and runs
# next to a CUDA 12.8 application when the driver supports CUDA 12.9. On an older driver the
# script falls back to NCCL 2.26.2, the newest build for CUDA 12.8, and says so.
#
# Each stage is skipped when its result is already there, so the script can be rerun after an
# interruption, or on a node where PREFIX was restored from an archive of a previous run:
#     tar czf comm-toolchain.tgz -C / opt/comm          (on the first node)
#     tar xzf comm-toolchain.tgz -C /                   (on the next one, then rerun this script)
#
# Usage (as root, from the repository root):  ./scripts/benchmarking/comm_setup.sh
# Then:  source /opt/comm/env.sh
set -eo pipefail
cd "$(dirname "$0")/../.."
PREFIX="${PREFIX:-/opt/comm}"
CUDA_VER="${CUDA_VER:-12.8}"
CUDA="${CUDA:-/usr/local/cuda-$CUDA_VER}"
J=$(nproc)
# -i 0 rather than "| head -1": under pipefail, head closing the pipe early can kill the script
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i 0 | tr -d ' .')
mkdir -p "$PREFIX/src" "$PREFIX/logs"
stage() { echo "=== $(date +%T) $*"; }
export DEBIAN_FRONTEND=noninteractive

# Highest CUDA version the driver runs, e.g. 12.9 -> 1209
drv=$(nvidia-smi | sed -n 's/.*CUDA Version: *\([0-9]*\)\.\([0-9]*\).*/\1 \2/p' | awk '{printf "%d", $1 * 100 + $2}')
if [ "${drv:-0}" -ge 1209 ]; then NCCL_PKG=2.31.2-1+cuda12.9; else NCCL_PKG=2.26.2-1+cuda12.8; fi
echo "  driver runs CUDA up to $((drv / 100)).$((drv % 100)); NCCL package: $NCCL_PKG"

stage "packages (CUDA $CUDA_VER, NCCL, NVSHMEM, system Open MPI)"
pkg_ok() { [ "$(dpkg-query -W -f='${Version}' "$1" 2>/dev/null)" = "$2" ]; }
PKGS=()
[ -x "$CUDA/bin/nvcc" ] || PKGS+=("cuda-toolkit-${CUDA_VER/./-}")
pkg_ok libnccl2 "$NCCL_PKG" || PKGS+=("libnccl2=$NCCL_PKG" "libnccl-dev=$NCCL_PKG")
pkg_ok libnvshmem3-static-cuda-12 3.7.2-1 ||
    PKGS+=(libnvshmem3-cuda-12=3.7.2-1 libnvshmem3-dev-cuda-12=3.7.2-1 libnvshmem3-static-cuda-12=3.7.2-1)
command -v mpicxx >/dev/null || PKGS+=(openmpi-bin libopenmpi-dev)
if [ "${#PKGS[@]}" -gt 0 ]; then
    # Images often declare NVIDIA's CUDA repository already; a second declaration with another key
    # makes apt refuse every source, so the keyring is only installed when no source names it
    if ! grep -rqs 'developer.download.nvidia.com/compute/cuda/repos' /etc/apt/sources.list /etc/apt/sources.list.d/; then
        distro=$(. /etc/os-release && echo "${ID}${VERSION_ID//./}")
        wget -q "https://developer.download.nvidia.com/compute/cuda/repos/$distro/x86_64/cuda-keyring_1.1-1_all.deb" \
            -O /tmp/cuda-keyring.deb
        dpkg -i /tmp/cuda-keyring.deb > /dev/null
    fi
    apt-get update -qq
    # The toolkit is only requested when nvcc is missing: images that pin some CUDA libraries
    # (apt-mark hold) cannot take a newer toolkit metapackage, but already have the toolkit
    apt-get install -y -qq --allow-downgrades "${PKGS[@]}" build-essential pkg-config git wget \
        > "$PREFIX/logs/apt.log" 2>&1
fi
"$CUDA/bin/nvcc" --version | tail -1
# The solver's Makefile links CUDA through /usr/local/cuda: point it at the toolchain in use
[ "$(readlink -f /usr/local/cuda)" = "$(readlink -f "$CUDA")" ] || ln -sfn "$CUDA" /usr/local/cuda
SYS_PATH="$CUDA/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
echo "  system MPI: $(PATH=$SYS_PATH mpirun --version 2>/dev/null | sed -n 1p)"

stage "UCX + Open MPI with CUDA"
if [ ! -x "$PREFIX/bin/mpirun" ]; then
  (  # subshell: the builds change directory
    cd "$PREFIX/src"
    wget -q https://github.com/openucx/ucx/releases/download/v1.18.1/ucx-1.18.1.tar.gz
    wget -q https://download.open-mpi.org/release/open-mpi/v5.0/openmpi-5.0.8.tar.bz2
    tar xzf ucx-1.18.1.tar.gz && cd ucx-1.18.1
    ./contrib/configure-release --prefix="$PREFIX" --with-cuda="$CUDA" --without-java \
        --disable-doxygen-doc > "$PREFIX/logs/ucx.log" 2>&1
    make -j"$J" >> "$PREFIX/logs/ucx.log" 2>&1 && make install >> "$PREFIX/logs/ucx.log" 2>&1
    cd "$PREFIX/src" && tar xjf openmpi-5.0.8.tar.bz2 && cd openmpi-5.0.8
    # Internal PMIx and PRRTE: an external PMIx can override Open MPI's component path, and the
    # CUDA component is then silently never loaded
    ./configure --prefix="$PREFIX" --with-cuda="$CUDA" --with-cuda-libdir="$CUDA/lib64/stubs" \
        --with-ucx="$PREFIX" --with-pmix=internal --with-prrte=internal --with-hwloc=internal \
        --with-libevent=internal --disable-mpi-fortran --disable-oshmem > "$PREFIX/logs/ompi.log" 2>&1
    make -j"$J" >> "$PREFIX/logs/ompi.log" 2>&1 && make install >> "$PREFIX/logs/ompi.log" 2>&1
  )
fi
CA_PATH="$PREFIX/bin:$SYS_PATH"
PATH=$CA_PATH LD_LIBRARY_PATH="$PREFIX/lib:$CUDA/lib64" ompi_info --parsable --all |
    grep -m1 mpi_built_with_cuda_support:value || true

stage "nccl-tests"
# Pinned: nccl-tests 2.21.0 (a3d4589) uses NCCL_WIN_GIN_ONLY, which NCCL 2.31.2 does not declare
NCCL_TESTS_COMMIT=b4d5bee   # nccl-tests 2.20.0
NT="$PREFIX/src/nccl-tests"
if [ ! -x "$NT/build/all_reduce_perf" ]; then
    [ -d "$NT/.git" ] || git clone -q https://github.com/NVIDIA/nccl-tests.git "$NT"
    if [ -f "$NT/.git/shallow" ]; then git -C "$NT" fetch -q --unshallow origin; else git -C "$NT" fetch -q origin; fi
    git -C "$NT" checkout -q "$NCCL_TESTS_COMMIT"
    make -C "$NT" clean > /dev/null 2>&1 || true
    PATH=$CA_PATH make -C "$NT" -j"$J" MPI=1 MPI_HOME="$PREFIX" CUDA_HOME="$CUDA" \
        NVCC_GENCODE="-gencode=arch=compute_${CC},code=sm_${CC}" > "$PREFIX/logs/nccl-tests.log" 2>&1
fi

# AmgX links MPI statically into its library: one build per MPI, from the same sources
AMGX="$PREFIX/src/amgx"
[ -d "$AMGX" ] || { git clone -q https://github.com/NVIDIA/AMGX.git "$AMGX"; git -C "$AMGX" checkout -q cc1cebd
                    git -C "$AMGX" submodule update --init --recursive -q || true; }
build_amgx() {  # $1 build directory, $2 PATH whose mpicc/mpicxx to use
    if [ ! -f "$1/libamgx.a" ]; then
        PATH=$2 cmake -S "$AMGX" -B "$1" -DCMAKE_BUILD_TYPE=Release -DCMAKE_NO_MPI=0 \
            -DCMAKE_CUDA_ARCHITECTURES="$CC" -DCMAKE_CUDA_COMPILER="$CUDA/bin/nvcc" \
            -DMPI_C_COMPILER="$(PATH=$2 command -v mpicc)" -DMPI_CXX_COMPILER="$(PATH=$2 command -v mpicxx)" \
            > "$PREFIX/logs/$(basename "$1").log" 2>&1
        PATH=$2 make -C "$1" -j"$J" amgx >> "$PREFIX/logs/$(basename "$1").log" 2>&1
    fi
    grep -q DAMGX_WITH_MPI "$1/CMakeFiles/amgx.dir/flags.make" || { echo "AmgX in $1 has no MPI"; exit 1; }
}
stage "AmgX, system MPI"
build_amgx "$AMGX/build-sysmpi" "$SYS_PATH"
stage "AmgX, CUDA-aware MPI"
build_amgx "$AMGX/build-cuda-aware" "$CA_PATH"

stage "environment files"
cat > "$PREFIX/env.sh" <<EOF
# Default build: system Open MPI (backends staged, nccl, nvshmem; AmgX MPI)
export PATH=$SYS_PATH:\$PATH
export LD_LIBRARY_PATH=$CUDA/lib64
export NCCL_TESTS=$PREFIX/src/nccl-tests
export NCCL_HOME=/usr NVSHMEM_HOME=/usr
# single node: no network transport for NVSHMEM (avoids probing InfiniBand)
export NVSHMEM_REMOTE_TRANSPORT=none
# The CUDA-aware build, used by the scripts for gpuaware and AmgX MPI_DIRECT
export CUDA_AWARE_MPIRUN=$PREFIX/bin/mpirun
EOF
cat > "$PREFIX/env-cuda-aware.sh" <<EOF
# CUDA-aware build: Open MPI 5 + UCX (backend gpuaware; AmgX MPI_DIRECT; nccl-tests)
export PATH=$CA_PATH:\$PATH
export LD_LIBRARY_PATH=$PREFIX/lib:$CUDA/lib64
export MPI_HOME=$PREFIX NCCL_TESTS=$PREFIX/src/nccl-tests
export NCCL_HOME=/usr NVSHMEM_HOME=/usr
export NVSHMEM_REMOTE_TRANSPORT=none
EOF

stage "solver and AmgX driver, default build (system MPI)"
PATH=$SYS_PATH make -B -j"$J" ARCH="$CC" NCCL_HOME=/usr NVSHMEM_HOME=/usr cg_solver_mgpu_stencil_3d \
    > "$PREFIX/logs/solver.log" 2>&1
PATH=$SYS_PATH make -B -C external/benchmarks/amgx amgx_cg_solver_mgpu AMGX_INCLUDE="-I$AMGX/include" \
    AMGX_LIBDIR="$AMGX/build-sysmpi" > "$PREFIX/logs/amgx-driver.log" 2>&1

stage "solver and AmgX driver, CUDA-aware build (Open MPI 5)"
PATH=$CA_PATH make -B -j"$J" ARCH="$CC" BUILD_TYPE=cuda-aware NCCL_HOME=/usr NVSHMEM_HOME=/usr \
    cg_solver_mgpu_stencil_3d > "$PREFIX/logs/solver-cuda-aware.log" 2>&1
cp external/benchmarks/amgx/amgx_cg_solver_mgpu "$PREFIX/amgx_cg_solver_mgpu.sysmpi"
PATH=$CA_PATH make -B -C external/benchmarks/amgx amgx_cg_solver_mgpu AMGX_INCLUDE="-I$AMGX/include" \
    AMGX_LIBDIR="$AMGX/build-cuda-aware" \
    MPI_CXXFLAGS="-std=c++17 -O3 -I$AMGX/include -I$PREFIX/include" \
    MPI_LDFLAGS="-L$AMGX/build-cuda-aware -lamgx -L$CUDA/lib64 -lcudart -lcusparse -lcublas -lcusolver -lcuda \
-L$PREFIX/lib -lmpi -Xlinker -rpath -Xlinker $PREFIX/lib" > "$PREFIX/logs/amgx-driver-cuda-aware.log" 2>&1
mv external/benchmarks/amgx/amgx_cg_solver_mgpu external/benchmarks/amgx/amgx_cg_solver_mgpu_cuda_aware
mv "$PREFIX/amgx_cg_solver_mgpu.sysmpi" external/benchmarks/amgx/amgx_cg_solver_mgpu

for b in bin/cg_solver_mgpu_stencil_3d bin/cuda-aware/cg_solver_mgpu_stencil_3d \
         external/benchmarks/amgx/amgx_cg_solver_mgpu external/benchmarks/amgx/amgx_cg_solver_mgpu_cuda_aware; do
    echo "  $b"
    ldd "$b" | grep -E 'libmpi\.so|libnccl|libnvshmem|libcudart' | sed 's/^\s*/    /; s/ (0x.*//' || true
done
stage "DONE -- next: source $PREFIX/env.sh, then rental_preflight.sh and comm_preflight.sh"
