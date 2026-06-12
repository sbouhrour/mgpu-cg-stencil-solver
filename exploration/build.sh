#!/usr/bin/env bash
# Build the 27-point variants bench for NCU profiling.
#
# Flags = project release flags, plus:
#   -arch=sm_89  : native SASS for the local RTX 4060 (the project Makefile
#                  ships sm_52 PTX and JITs at runtime; native SASS gives NCU
#                  a stable code object). A one-shot guard run compares
#                  headline metrics against the canonical JIT binary.
#   -lineinfo    : source<->SASS correlation for the NCU Source page
#                  (no effect on optimization level).
set -e
cd "$(dirname "$0")/.."
mkdir -p bin
nvcc -O2 --ptxas-options=-O2 --ptxas-options=-allow-expensive-optimizations=true \
     -std=c++11 -arch=sm_89 -lineinfo \
     -Iinclude -Iinclude/solvers \
     exploration/bench_27pt_variants.cu \
     exploration/stencil27_soa_kernel.cu \
     src/spmv/spmv_stencil_3d_27pt_partitioned_halo_kernel.cu \
     src/io/io.cu src/spmv/spmv_cusparse_csr.cu \
     -o bin/bench_27pt_variants -lcusparse
echo "Built bin/bench_27pt_variants"
