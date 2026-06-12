/**
 * @file stencil27_soa_kernel.cu
 * @brief Coefficient-major (SoA) variant of the 27-point stencil SpMV.
 *
 * Layout: values_soa[c * n + row], c = 0..26 in ascending neighbor-offset
 * order — identical to the ascending-column order the production kernel's
 * fast path assumes in CSR. All rows take the same fully-unrolled path:
 * boundary rows carry 0.0 padding coefficients, and neighbor indices are
 * clamped to [0, n-1] (the 0.0 coefficient makes the clamped product
 * exactly 0.0, so the result is exact; finite x assumed).
 *
 * Differences vs the CSR baseline, by construction:
 *   - every values load is stride-1 across threads (coalesced);
 *   - row_ptr and col_idx are not read at all;
 *   - no boundary branch, hence no intra-warp divergence;
 *   - FP accumulation order matches the baseline fast path (ascending c):
 *     interior rows are bitwise identical, boundary rows only insert +0.0
 *     terms, which leaves IEEE-754 sums unchanged.
 */

#include <cuda_runtime.h>

__device__ __forceinline__ int clamp_row(int v, int n) {
    v = v < 0 ? 0 : v;
    return v >= n ? n - 1 : v;
}

__global__ void stencil27_soa_kernel_3d(const double* __restrict__ values_soa,
                                        const double* __restrict__ x, double* __restrict__ y, int n,
                                        int N) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n)
        return;

    const int NN = N * N;
    const long long ln = n;

    double sum = 0.0;
    // Z-plane i-1
    sum = values_soa[0 * ln + row] * x[clamp_row(row - NN - N - 1, n)];
    sum += values_soa[1 * ln + row] * x[clamp_row(row - NN - N, n)];
    sum += values_soa[2 * ln + row] * x[clamp_row(row - NN - N + 1, n)];
    sum += values_soa[3 * ln + row] * x[clamp_row(row - NN - 1, n)];
    sum += values_soa[4 * ln + row] * x[clamp_row(row - NN, n)];
    sum += values_soa[5 * ln + row] * x[clamp_row(row - NN + 1, n)];
    sum += values_soa[6 * ln + row] * x[clamp_row(row - NN + N - 1, n)];
    sum += values_soa[7 * ln + row] * x[clamp_row(row - NN + N, n)];
    sum += values_soa[8 * ln + row] * x[clamp_row(row - NN + N + 1, n)];
    // Z-plane i
    sum += values_soa[9 * ln + row] * x[clamp_row(row - N - 1, n)];
    sum += values_soa[10 * ln + row] * x[clamp_row(row - N, n)];
    sum += values_soa[11 * ln + row] * x[clamp_row(row - N + 1, n)];
    sum += values_soa[12 * ln + row] * x[clamp_row(row - 1, n)];
    sum += values_soa[13 * ln + row] * x[row];  // center
    sum += values_soa[14 * ln + row] * x[clamp_row(row + 1, n)];
    sum += values_soa[15 * ln + row] * x[clamp_row(row + N - 1, n)];
    sum += values_soa[16 * ln + row] * x[clamp_row(row + N, n)];
    sum += values_soa[17 * ln + row] * x[clamp_row(row + N + 1, n)];
    // Z-plane i+1
    sum += values_soa[18 * ln + row] * x[clamp_row(row + NN - N - 1, n)];
    sum += values_soa[19 * ln + row] * x[clamp_row(row + NN - N, n)];
    sum += values_soa[20 * ln + row] * x[clamp_row(row + NN - N + 1, n)];
    sum += values_soa[21 * ln + row] * x[clamp_row(row + NN - 1, n)];
    sum += values_soa[22 * ln + row] * x[clamp_row(row + NN, n)];
    sum += values_soa[23 * ln + row] * x[clamp_row(row + NN + 1, n)];
    sum += values_soa[24 * ln + row] * x[clamp_row(row + NN + N - 1, n)];
    sum += values_soa[25 * ln + row] * x[clamp_row(row + NN + N, n)];
    sum += values_soa[26 * ln + row] * x[clamp_row(row + NN + N + 1, n)];

    y[row] = sum;
}

void launch_stencil27_soa(int blocks, int threads, const double* values_soa, const double* x,
                          double* y, int n, int N) {
    stencil27_soa_kernel_3d<<<blocks, threads>>>(values_soa, x, y, n, N);
}

/**
 * __ldcs variant: identical to stencil27_soa_kernel_3d except the 27
 * coefficient loads use the cache-streaming policy (evict-first hint).
 * The coefficients are stream-once data with no reuse; the hint asks L1/L2
 * not to retain their lines, protecting the x reuse window from eviction
 * pressure. x loads keep the default (caching) policy.
 */
__global__ void stencil27_soa_ldcs_kernel_3d(const double* __restrict__ values_soa,
                                             const double* __restrict__ x, double* __restrict__ y,
                                             int n, int N) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n)
        return;

    const int NN = N * N;
    const long long ln = n;

    double sum = 0.0;
    // Z-plane i-1
    sum = __ldcs(&values_soa[0 * ln + row]) * x[clamp_row(row - NN - N - 1, n)];
    sum += __ldcs(&values_soa[1 * ln + row]) * x[clamp_row(row - NN - N, n)];
    sum += __ldcs(&values_soa[2 * ln + row]) * x[clamp_row(row - NN - N + 1, n)];
    sum += __ldcs(&values_soa[3 * ln + row]) * x[clamp_row(row - NN - 1, n)];
    sum += __ldcs(&values_soa[4 * ln + row]) * x[clamp_row(row - NN, n)];
    sum += __ldcs(&values_soa[5 * ln + row]) * x[clamp_row(row - NN + 1, n)];
    sum += __ldcs(&values_soa[6 * ln + row]) * x[clamp_row(row - NN + N - 1, n)];
    sum += __ldcs(&values_soa[7 * ln + row]) * x[clamp_row(row - NN + N, n)];
    sum += __ldcs(&values_soa[8 * ln + row]) * x[clamp_row(row - NN + N + 1, n)];
    // Z-plane i
    sum += __ldcs(&values_soa[9 * ln + row]) * x[clamp_row(row - N - 1, n)];
    sum += __ldcs(&values_soa[10 * ln + row]) * x[clamp_row(row - N, n)];
    sum += __ldcs(&values_soa[11 * ln + row]) * x[clamp_row(row - N + 1, n)];
    sum += __ldcs(&values_soa[12 * ln + row]) * x[clamp_row(row - 1, n)];
    sum += __ldcs(&values_soa[13 * ln + row]) * x[row];  // center
    sum += __ldcs(&values_soa[14 * ln + row]) * x[clamp_row(row + 1, n)];
    sum += __ldcs(&values_soa[15 * ln + row]) * x[clamp_row(row + N - 1, n)];
    sum += __ldcs(&values_soa[16 * ln + row]) * x[clamp_row(row + N, n)];
    sum += __ldcs(&values_soa[17 * ln + row]) * x[clamp_row(row + N + 1, n)];
    // Z-plane i+1
    sum += __ldcs(&values_soa[18 * ln + row]) * x[clamp_row(row + NN - N - 1, n)];
    sum += __ldcs(&values_soa[19 * ln + row]) * x[clamp_row(row + NN - N, n)];
    sum += __ldcs(&values_soa[20 * ln + row]) * x[clamp_row(row + NN - N + 1, n)];
    sum += __ldcs(&values_soa[21 * ln + row]) * x[clamp_row(row + NN - 1, n)];
    sum += __ldcs(&values_soa[22 * ln + row]) * x[clamp_row(row + NN, n)];
    sum += __ldcs(&values_soa[23 * ln + row]) * x[clamp_row(row + NN + 1, n)];
    sum += __ldcs(&values_soa[24 * ln + row]) * x[clamp_row(row + NN + N - 1, n)];
    sum += __ldcs(&values_soa[25 * ln + row]) * x[clamp_row(row + NN + N, n)];
    sum += __ldcs(&values_soa[26 * ln + row]) * x[clamp_row(row + NN + N + 1, n)];

    y[row] = sum;
}

void launch_stencil27_soa_ldcs(int blocks, int threads, const double* values_soa, const double* x,
                               double* y, int n, int N) {
    stencil27_soa_ldcs_kernel_3d<<<blocks, threads>>>(values_soa, x, y, n, N);
}
