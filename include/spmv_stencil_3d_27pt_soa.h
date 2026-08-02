/**
 * @file spmv_stencil_3d_27pt_soa.h
 * @brief Declarations for the coefficient-major (SoA) 3D 27-point stencil SpMV
 *
 * @details
 * The production kernel keeps its double-precision signature. The templated form holds the
 * coefficients at a chosen storage width and accumulates in double, so that the only variable is
 * how many bytes and cache sectors a coefficient load costs.
 */

#ifndef SPMV_STENCIL_3D_27PT_SOA_H
#define SPMV_STENCIL_3D_27PT_SOA_H

#include <cuda_runtime.h>

/** @brief Production SoA 27-point kernel: coefficients stored and read in double */
__global__ void stencil27_soa_halo_kernel_3d(const double* __restrict__ values_soa,
                                             const double* __restrict__ x_ext,
                                             double* __restrict__ y, int n_local, int grid_size);

/**
 * @brief SoA 27-point kernel with coefficients at storage width ValueT, accumulated in double
 * @tparam ValueT double, float, __half or __nv_bfloat16
 */
template <typename ValueT>
__global__ void stencil27_soa_halo_kernel_3d_t(const ValueT* __restrict__ values_soa,
                                               const double* __restrict__ x_ext,
                                               double* __restrict__ y, int n_local, int grid_size);

/** @brief Host transform: local CSR slice -> coefficient-major (SoA) values */
void build_values_soa_27pt_3d(const long long* row_ptr, const int* col_idx, const double* values,
                              int n_local, long long row_offset, int grid_size, double* values_soa);

#endif  // SPMV_STENCIL_3D_27PT_SOA_H
