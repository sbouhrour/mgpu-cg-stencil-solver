/**
 * @file spmv_stencil_3d_27pt.h
 * @brief Declarations for the 3D 27-point stencil SpMV kernels
 *
 * @details
 * The production kernel keeps its double-precision signature. The templated variant exposes the
 * same arithmetic with the matrix coefficients held at a chosen storage precision, accumulated in
 * double; it exists to measure the bandwidth cost of coefficient width, which is 90% of this
 * kernel's DRAM traffic.
 */

#ifndef SPMV_STENCIL_3D_27PT_H
#define SPMV_STENCIL_3D_27PT_H

#include <cuda_runtime.h>

/** @brief Production 27-point kernel: coefficients stored and read in double */
__global__ void stencil27_csr_partitioned_halo_kernel_3d(
    const long long* __restrict__ row_ptr, const int* __restrict__ col_idx,
    const double* __restrict__ values, const double* __restrict__ x_local,
    const double* __restrict__ x_halo_prev, const double* __restrict__ x_halo_next,
    double* __restrict__ y, int n_local, int row_offset, int N_total, int grid_size);

/**
 * @brief 27-point kernel with coefficients at storage precision ValueT, accumulated in double
 * @tparam ValueT double or float
 */
template <typename ValueT>
__global__ void stencil27_mixed_precision_kernel_3d(
    const long long* __restrict__ row_ptr, const int* __restrict__ col_idx,
    const ValueT* __restrict__ values, const double* __restrict__ x_local,
    const double* __restrict__ x_halo_prev, const double* __restrict__ x_halo_next,
    double* __restrict__ y, int n_local, int row_offset, int N_total, int grid_size);

#endif  // SPMV_STENCIL_3D_27PT_H
