#!/usr/bin/env bash
# Installs, on a rented multi-GPU node, everything the communication study needs: CUDA 12.8 (the
# toolkit, cuBLAS and cuSPARSE the published results were measured with), NCCL 2.31.2, NVSHMEM
# 3.7.2, UCX 1.18.1 + Open MPI 5.0.8 built with CUDA, nccl-tests, and AmgX (commit cc1cebd) built
# with MPI. Then builds the solver and the AmgX driver against them and writes an environment file.
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
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' .')
mkdir -p "$PREFIX/src" "$PREFIX/logs"
stage() { echo "=== $(date +%T) $*"; }
export DEBIAN_FRONTEND=noninteractive

# Highest CUDA version the driver runs, e.g. 12.9 -> 1209
drv=$(nvidia-smi | sed -n 's/.*CUDA Version: *\([0-9]*\)\.\([0-9]*\).*/\1 \2/p' | awk '{printf "%d", $1 * 100 + $2}')
if [ "${drv:-0}" -ge 1209 ]; then NCCL_PKG=2.31.2-1+cuda12.9; else NCCL_PKG=2.26.2-1+cuda12.8; fi
echo "  driver runs CUDA up to $((drv / 100)).$((drv % 100)); NCCL package: $NCCL_PKG"

stage "packages (CUDA $CUDA_VER, NCCL, NVSHMEM)"
if [ ! -x "$CUDA/bin/nvcc" ] || ! dpkg -s libnvshmem3-static-cuda-12 >/dev/null 2>&1; then
    apt-get update -qq
    if ! apt-cache policy "cuda-toolkit-${CUDA_VER/./-}" | grep -q 'Candidate: [0-9]'; then
        # NVIDIA's CUDA repository is not configured on this image: add it
        distro=$(. /etc/os-release && echo "${ID}${VERSION_ID//./}")
        wget -q "https://developer.download.nvidia.com/compute/cuda/repos/$distro/x86_64/cuda-keyring_1.1-1_all.deb" \
            -O /tmp/cuda-keyring.deb
        dpkg -i /tmp/cuda-keyring.deb > /dev/null && apt-get update -qq
    fi
    apt-get install -y -qq --allow-downgrades "cuda-toolkit-${CUDA_VER/./-}" \
        "libnccl2=$NCCL_PKG" "libnccl-dev=$NCCL_PKG" \
        libnvshmem3-cuda-12=3.7.2-1 libnvshmem3-dev-cuda-12=3.7.2-1 libnvshmem3-static-cuda-12=3.7.2-1 \
        build-essential pkg-config git wget > "$PREFIX/logs/apt.log" 2>&1
fi
"$CUDA/bin/nvcc" --version | tail -1
# The solver's Makefile links CUDA through /usr/local/cuda: point it at the toolchain in use
[ "$(readlink -f /usr/local/cuda)" = "$(readlink -f "$CUDA")" ] || ln -sfn "$CUDA" /usr/local/cuda
export PATH="$CUDA/bin:$PATH"

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
export PATH="$PREFIX/bin:$PATH" LD_LIBRARY_PATH="$PREFIX/lib:$CUDA/lib64"
ompi_info --parsable --all | grep -m1 mpi_built_with_cuda_support:value

stage "nccl-tests"
if [ ! -x "$PREFIX/src/nccl-tests/build/all_reduce_perf" ]; then
    git clone -q --depth 1 https://github.com/NVIDIA/nccl-tests.git "$PREFIX/src/nccl-tests"
    make -C "$PREFIX/src/nccl-tests" -j"$J" MPI=1 MPI_HOME="$PREFIX" CUDA_HOME="$CUDA" \
        NVCC_GENCODE="-gencode=arch=compute_${CC},code=sm_${CC}" > "$PREFIX/logs/nccl-tests.log" 2>&1
fi

stage "AmgX with MPI"
AMGX="$PREFIX/src/amgx"
if [ ! -f "$AMGX/build/libamgx.a" ]; then
    git clone -q https://github.com/NVIDIA/AMGX.git "$AMGX"
    git -C "$AMGX" checkout -q cc1cebd
    git -C "$AMGX" submodule update --init --recursive -q || true
    cmake -S "$AMGX" -B "$AMGX/build" -DCMAKE_BUILD_TYPE=Release -DCMAKE_NO_MPI=0 \
        -DCMAKE_CUDA_ARCHITECTURES="$CC" -DCMAKE_CUDA_COMPILER="$CUDA/bin/nvcc" \
        -DMPI_C_COMPILER="$PREFIX/bin/mpicc" -DMPI_CXX_COMPILER="$PREFIX/bin/mpicxx" \
        > "$PREFIX/logs/amgx.log" 2>&1
    make -C "$AMGX/build" -j"$J" amgx >> "$PREFIX/logs/amgx.log" 2>&1
fi
grep -q DAMGX_WITH_MPI "$AMGX/build/CMakeFiles/amgx.dir/flags.make" \
    || { echo "AmgX was configured without MPI"; exit 1; }

stage "environment file"
cat > "$PREFIX/env.sh" <<EOF
export PATH=$PREFIX/bin:$CUDA/bin:\$PATH
export LD_LIBRARY_PATH=$PREFIX/lib:$CUDA/lib64
export MPI_HOME=$PREFIX NCCL_TESTS=$PREFIX/src/nccl-tests
export NCCL_HOME=/usr NVSHMEM_HOME=/usr
# single node: no network transport for NVSHMEM (avoids probing InfiniBand)
export NVSHMEM_REMOTE_TRANSPORT=none
EOF
# shellcheck disable=SC1091
source "$PREFIX/env.sh"

stage "solver and AmgX driver"
make -B -j"$J" ARCH="$CC" cg_solver_mgpu_stencil_3d > "$PREFIX/logs/solver.log" 2>&1
make -B -C external/benchmarks/amgx amgx_cg_solver_mgpu AMGX_INCLUDE="-I$AMGX/include" \
    AMGX_LIBDIR="$AMGX/build" \
    MPI_CXXFLAGS="-std=c++17 -O3 -I$AMGX/include -I$PREFIX/include" \
    MPI_LDFLAGS="-L$AMGX/build -lamgx -L$CUDA/lib64 -lcudart -lcusparse -lcublas -lcusolver -lcuda \
-L$PREFIX/lib -lmpi -Xlinker -rpath -Xlinker $PREFIX/lib" > "$PREFIX/logs/amgx-driver.log" 2>&1
ldd bin/cg_solver_mgpu_stencil_3d | grep -E 'libmpi\.so|libnccl|libnvshmem|libcudart' | sed 's/^\s*/  /'
stage "DONE -- next: source $PREFIX/env.sh, then rental_preflight.sh and comm_preflight.sh"
