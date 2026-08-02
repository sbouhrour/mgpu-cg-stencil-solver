/**
 * @file spmv_stencil_3d_27pt_soa_halo_kernel.cu
 * @brief Coefficient-major (SoA) 27-point stencil SpMV with ghost-layer halos.
 *
 * @details
 * Storage: values_soa[c * n_local + row], c = 0..26 in ascending neighbor
 * offset order (di, dj, dk in {-1,0,1}, di major) — the same ascending
 * column order as the CSR fast path of stencil27_csr_partitioned_halo_kernel_3d.
 * Rows missing a neighbor (global domain boundary) hold 0.0 at that slot.
 *
 * The input vector uses a ghost-layer ("extended") layout:
 *   x_ext = [ prev halo plane (N²) | local rows (n_local) | next halo plane (N²) ]
 * so every neighbor access is x_ext[N² + row + delta_c], uniform for all rows.
 * Indices are clamped to the extended buffer; clamped (out-of-range) accesses
 * only occur where the coefficient is 0.0 — the stencil reaches at most N+1
 * elements past a halo plane, and exactly at global-boundary slots — so
 * results are exact (finite x assumed, which CG guarantees). Both halo
 * regions must be allocated on every rank (zero-filled where unused).
 *
 * Requires plane-aligned Z-slab partitions (n_local multiple of N²), the
 * same assumption as the CSR kernel's one-plane halo routing.
 *
 * Why this layout: each warp reads 32 consecutive doubles per coefficient
 * stream (coalesced: 8 sectors per request instead of 32 with row-major CSR
 * values), row_ptr/col_idx are never read, and all rows share one
 * branch-free path (no intra-warp divergence). Measured at 192³ on RTX 4060
 * (NCU, fixed clocks): DRAM throughput 71% -> 96% of peak, elapsed cycles
 * -28%. Wall-clock on A100-SXM4-80GB: 1.43-1.51x at 128³-384³, output
 * bitwise-identical to the CSR kernel.
 *
 * Author: Bouhrour Stephane
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "spmv_stencil_3d_27pt_soa.h"

__device__ __forceinline__ int clamp_ext(int v, int ext_n) {
    v = v < 0 ? 0 : v;
    return v >= ext_n ? ext_n - 1 : v;
}

/**
 * @brief Reads one coefficient at its storage width and widens it to double
 *
 * @details Accumulation is double for every width, so the only thing the storage type changes is
 * the number of bytes and cache sectors the load costs. Narrow types are converted here, on the
 *          CUDA cores — no tensor-core pipeline is involved, and none could be: a stencil has no
 * matmul shape to feed one.
 */
__device__ __forceinline__ double ld_val(double v) {
    return v;
}
__device__ __forceinline__ double ld_val(float v) {
    return (double)v;
}
__device__ __forceinline__ double ld_val(__half v) {
    return (double)__half2float(v);
}
__device__ __forceinline__ double ld_val(__nv_bfloat16 v) {
    return (double)__bfloat162float(v);
}

/**
 * @brief SoA 27-point stencil SpMV over a Z-slab partition with ghost layers
 *
 * @param[in] values_soa Coefficient-major values, 27 * n_local doubles
 * @param[in] x_ext Extended input vector: [N² halo | n_local rows | N² halo]
 * @param[out] y Output vector partition (n_local rows)
 * @param[in] n_local Number of local rows (multiple of N²)
 * @param[in] grid_size N (grid is N×N×N)
 */
template <typename ValueT>
__device__ __forceinline__ double stencil27_soa_row(const ValueT* __restrict__ values_soa,
                                                    const double* __restrict__ x_ext, int row,
                                                    int n_local, int grid_size) {
    const int N = grid_size;
    const int NN = N * N;
    const int ext_n = n_local + 2 * NN;
    const int base = NN + row;  // position of this row in x_ext
    const long long ln = n_local;

    double sum = 0.0;
    // Z-plane i-1
    sum = ld_val(values_soa[0 * ln + row]) * x_ext[clamp_ext(base - NN - N - 1, ext_n)];
    sum += ld_val(values_soa[1 * ln + row]) * x_ext[clamp_ext(base - NN - N, ext_n)];
    sum += ld_val(values_soa[2 * ln + row]) * x_ext[clamp_ext(base - NN - N + 1, ext_n)];
    sum += ld_val(values_soa[3 * ln + row]) * x_ext[clamp_ext(base - NN - 1, ext_n)];
    sum += ld_val(values_soa[4 * ln + row]) * x_ext[clamp_ext(base - NN, ext_n)];
    sum += ld_val(values_soa[5 * ln + row]) * x_ext[clamp_ext(base - NN + 1, ext_n)];
    sum += ld_val(values_soa[6 * ln + row]) * x_ext[clamp_ext(base - NN + N - 1, ext_n)];
    sum += ld_val(values_soa[7 * ln + row]) * x_ext[clamp_ext(base - NN + N, ext_n)];
    sum += ld_val(values_soa[8 * ln + row]) * x_ext[clamp_ext(base - NN + N + 1, ext_n)];
    // Z-plane i
    sum += ld_val(values_soa[9 * ln + row]) * x_ext[clamp_ext(base - N - 1, ext_n)];
    sum += ld_val(values_soa[10 * ln + row]) * x_ext[clamp_ext(base - N, ext_n)];
    sum += ld_val(values_soa[11 * ln + row]) * x_ext[clamp_ext(base - N + 1, ext_n)];
    sum += ld_val(values_soa[12 * ln + row]) * x_ext[clamp_ext(base - 1, ext_n)];
    sum += ld_val(values_soa[13 * ln + row]) * x_ext[base];  // center, always in range
    sum += ld_val(values_soa[14 * ln + row]) * x_ext[clamp_ext(base + 1, ext_n)];
    sum += ld_val(values_soa[15 * ln + row]) * x_ext[clamp_ext(base + N - 1, ext_n)];
    sum += ld_val(values_soa[16 * ln + row]) * x_ext[clamp_ext(base + N, ext_n)];
    sum += ld_val(values_soa[17 * ln + row]) * x_ext[clamp_ext(base + N + 1, ext_n)];
    // Z-plane i+1
    sum += ld_val(values_soa[18 * ln + row]) * x_ext[clamp_ext(base + NN - N - 1, ext_n)];
    sum += ld_val(values_soa[19 * ln + row]) * x_ext[clamp_ext(base + NN - N, ext_n)];
    sum += ld_val(values_soa[20 * ln + row]) * x_ext[clamp_ext(base + NN - N + 1, ext_n)];
    sum += ld_val(values_soa[21 * ln + row]) * x_ext[clamp_ext(base + NN - 1, ext_n)];
    sum += ld_val(values_soa[22 * ln + row]) * x_ext[clamp_ext(base + NN, ext_n)];
    sum += ld_val(values_soa[23 * ln + row]) * x_ext[clamp_ext(base + NN + 1, ext_n)];
    sum += ld_val(values_soa[24 * ln + row]) * x_ext[clamp_ext(base + NN + N - 1, ext_n)];
    sum += ld_val(values_soa[25 * ln + row]) * x_ext[clamp_ext(base + NN + N, ext_n)];
    sum += ld_val(values_soa[26 * ln + row]) * x_ext[clamp_ext(base + NN + N + 1, ext_n)];

    return sum;
}

/** @brief Production entry point: unchanged signature, double coefficients */
__global__ void stencil27_soa_halo_kernel_3d(const double* __restrict__ values_soa,
                                             const double* __restrict__ x_ext,
                                             double* __restrict__ y, int n_local, int grid_size) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_local)
        return;
    y[row] = stencil27_soa_row<double>(values_soa, x_ext, row, n_local, grid_size);
}

/** @brief Same operator with the coefficients held at storage width ValueT */
template <typename ValueT>
__global__ void stencil27_soa_halo_kernel_3d_t(const ValueT* __restrict__ values_soa,
                                               const double* __restrict__ x_ext,
                                               double* __restrict__ y, int n_local, int grid_size) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_local)
        return;
    y[row] = stencil27_soa_row<ValueT>(values_soa, x_ext, row, n_local, grid_size);
}

template __global__ void stencil27_soa_halo_kernel_3d_t<double>(const double* __restrict__,
                                                                const double* __restrict__,
                                                                double* __restrict__, int, int);
template __global__ void stencil27_soa_halo_kernel_3d_t<float>(const float* __restrict__,
                                                               const double* __restrict__,
                                                               double* __restrict__, int, int);
template __global__ void stencil27_soa_halo_kernel_3d_t<__half>(const __half* __restrict__,
                                                                const double* __restrict__,
                                                                double* __restrict__, int, int);
template __global__ void stencil27_soa_halo_kernel_3d_t<__nv_bfloat16>(
    const __nv_bfloat16* __restrict__, const double* __restrict__, double* __restrict__, int, int);

/**
 * @brief Host transform: local CSR slice -> coefficient-major (SoA) values
 *
 * @param[in] row_ptr Local row pointers (row_ptr[0] == 0, n_local+1 entries)
 * @param[in] col_idx Global column indices of the local nnz slice
 * @param[in] values Values of the local nnz slice
 * @param[in] n_local Number of local rows
 * @param[in] row_offset Global index of local row 0
 * @param[in] grid_size N (grid is N×N×N)
 * @param[out] values_soa Zero-initialized output, 27 * n_local doubles
 *
 * Validates that per-row columns are sorted ascending and that every entry
 * maps to one of the 27 stencil offsets (both guaranteed by build_csr_struct
 * for matrices from the 27-point generator); aborts otherwise.
 */
void build_values_soa_27pt_3d(const long long* row_ptr, const int* col_idx, const double* values,
                              int n_local, long long row_offset, int grid_size,
                              double* values_soa) {
    const int N = grid_size;
    long long offs[27];
    int c = 0;
    for (int di = -1; di <= 1; di++)
        for (int dj = -1; dj <= 1; dj++)
            for (int dk = -1; dk <= 1; dk++)
                offs[c++] = (long long)di * N * N + (long long)dj * N + dk;

    for (int r = 0; r < n_local; r++) {
        const long long global_row = row_offset + r;
        int ci = 0;
        for (long long e = row_ptr[r]; e < row_ptr[r + 1]; e++) {
            const long long delta = (long long)col_idx[e] - global_row;
            while (ci < 27 && offs[ci] < delta)
                ci++;
            if (ci >= 27 || offs[ci] != delta) {
                fprintf(stderr,
                        "build_values_soa_27pt_3d: row %lld has a non-stencil or "
                        "unsorted column (delta %lld)\n",
                        global_row, delta);
                exit(EXIT_FAILURE);
            }
            values_soa[(size_t)ci * n_local + r] = values[e];
            ci++;
        }
    }
}
