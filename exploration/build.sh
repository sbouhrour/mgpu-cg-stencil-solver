#!/usr/bin/env bash
# Build the 27-point variants bench for NCU profiling / remote validation.
#
# Flags = project release flags, plus:
#   -arch=<native>: native SASS for the local GPU (the project Makefile
#                   ships sm_52 PTX and JITs at runtime; native SASS gives
#                   NCU a stable code object). Auto-detected from
#                   nvidia-smi compute_cap; override with ARCH=sm_XX.
#                   Guard run vs the canonical JIT binary: see
#                   data/ncu_baseline_jit_guard_192.ncu-rep (equivalent).
#   -lineinfo     : source<->SASS correlation for the NCU Source page
#                   (no effect on optimization level).
set -e
cd "$(dirname "$0")/.."
mkdir -p bin

if [ -z "${ARCH:-}" ]; then
    CC_DIGITS=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
                | head -1 | tr -d '. ')
    if [ -n "$CC_DIGITS" ]; then
        ARCH="sm_${CC_DIGITS}"
    else
        ARCH="sm_89"
        echo "WARNING: GPU compute capability not detected, defaulting to $ARCH"
    fi
fi
echo "Building for ARCH=$ARCH"

nvcc -O2 --ptxas-options=-O2 --ptxas-options=-allow-expensive-optimizations=true \
     -std=c++11 -arch=$ARCH -lineinfo \
     -Iinclude -Iinclude/solvers \
     exploration/bench_27pt_variants.cu \
     exploration/stencil27_soa_kernel.cu \
     src/spmv/spmv_stencil_3d_27pt_partitioned_halo_kernel.cu \
     src/io/io.cu src/spmv/spmv_cusparse_csr.cu \
     -o bin/bench_27pt_variants -lcusparse
echo "Built bin/bench_27pt_variants"
