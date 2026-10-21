/**
 * @file spmv_stencil27_fast.cuh
 * @brief 3D 27-point stencil SpMV that reads a standard CSR matrix with coalesced loads
 *
 * @details
 * The kernel takes the CSR arrays as they are (row_ptr, col_idx, values, columns sorted in each
 * row). Nothing is converted, copied or reordered: what changes is how values is read.
 *
 * Grid convention: row r = (i * N + j) * N + k, with i the slowest (z-plane) index. A row is
 * interior when 0 < i, j, k < N - 1; it then holds exactly 27 entries, in ascending column order,
 * so entry p (0..26) couples r to the neighbour at offset (p / 9 - 1, (p / 3) % 3 - 1, p % 3 - 1).
 * Rows on the domain faces keep the truncated pattern and go through a plain CSR loop.
 *
 * csr_staged_kernel: one thread per row, as the row-major kernel of the solver, but each warp
 * first copies the contiguous value span of its 32 rows into shared memory with 16-byte cp.async,
 * so every global load of values is coalesced. Same arithmetic, same summation order as the
 * row-major kernel: the result is bitwise identical.
 *
 * Requirement checked once by check_pattern_kernel: every row holds the stencil pattern (27
 * entries inside, truncated on faces). values must be 16-byte aligned (cudaMalloc: 256).
 *
 * Author: Bouhrour Stephane
 */

#ifndef SPMV_STENCIL27_FAST_CUH
#define SPMV_STENCIL27_FAST_CUH

#include <cuda_runtime.h>

namespace stencil27 {

constexpr unsigned kFull = 0xffffffffu;

/** @brief Doubles of shared memory per warp for the staged kernel: 32 rows x 27, rounded to 16 B */
constexpr int kStage1 = 896;

/** @brief Asynchronous 16-byte global to shared copy that bypasses L1 (sm_80+) */
__device__ __forceinline__ void cp_async16(double* smem, const double* gmem) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem) : "memory");
#else
    smem[0] = gmem[0];
    smem[1] = gmem[1];
#endif
}

/** @brief Waits for every cp.async issued by this thread (visibility to the warp needs a sync) */
__device__ __forceinline__ void cp_async_wait_all() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;\n" ::: "memory");
#endif
}

/** @brief Linear offset of stencil entry p in an interior row */
__host__ __device__ __forceinline__ int entry_offset(int p, int N) {
    return (p / 9 - 1) * N * N + ((p / 3) % 3 - 1) * N + (p % 3 - 1);
}

/**
 * @brief Entries of row r in CSR order: calls f(position, column), returns the row length
 *
 * @details The valid stencil neighbours in stencil order p, which is ascending column order.
 */
template <typename F> __host__ __device__ int stencil_row(int r, int N, F f) {
    const int NN = N * N;
    const int i = r / NN, j = (r / N) % N, k = r % N;
    int q = 0;
    for (int p = 0; p < 27; p++) {
        const int a = i + p / 9 - 1, b = j + (p / 3) % 3 - 1, c = k + p % 3 - 1;
        if (a >= 0 && a < N && b >= 0 && b < N && c >= 0 && c < N)
            f(q++, r + entry_offset(p, N));
    }
    return q;
}

/** @brief Interior row: 27 terms in ascending column order, the row-major kernel's exact order */
__device__ __forceinline__ double row27(const double* v, const double* __restrict__ x, int row,
                                        int N) {
    double sum = v[0] * x[row + entry_offset(0, N)];
#pragma unroll
    for (int p = 1; p < 27; p++)
        sum += v[p] * x[row + entry_offset(p, N)];
    return sum;
}

/** @brief Face row: plain CSR loop over its own entries */
__device__ __forceinline__ double row_generic(const double* v, const int* __restrict__ cols,
                                              int len, const double* __restrict__ x) {
    double sum = 0.0;
    for (int q = 0; q < len; q++)
        sum += v[q] * x[cols[q]];
    return sum;
}

/**
 * @brief Row-major CSR SpMV with warp-coalesced staging of the coefficients
 *
 * @details One warp = 32 consecutive rows, whose values form one contiguous span of the CSR
 * values array. The warp copies that span to shared memory with 16-byte cp.async (each lane
 * copies consecutive chunks, so each instruction moves 512 contiguous bytes), then every thread
 * reads its own 27 coefficients from shared memory. The stride-27 shared-memory reads are free
 * of bank conflicts (27 is odd). x is read through L1, which the coefficients no longer evict.
 *
 * Launch: blockDim = WARPS * 32, gridDim = ceil(N^3 / (WARPS * 32)).
 */
template <int WARPS>
__global__ void __launch_bounds__(WARPS * 32)
    csr_staged_kernel(const long long* __restrict__ row_ptr, const int* __restrict__ col_idx,
                      const double* __restrict__ values, const double* __restrict__ x,
                      double* __restrict__ y, int N) {
    __shared__ __align__(16) double stage[WARPS][kStage1];
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int n = N * N * N;
    const int row0 = (blockIdx.x * WARPS + w) * 32;
    if (row0 >= n)
        return;  // whole warp
    const int row = row0 + lane;
    const bool valid = row < n;
    const int last = min(row0 + 31, n - 1);
    const long long rp = valid ? row_ptr[row] : 0;
    const long long base = __shfl_sync(kFull, rp, 0);
    const long long end = row_ptr[last + 1];
    const long long abase = base & ~1LL;
    const int nchunks = (int)((end - abase + 1) >> 1);
    const bool staged = nchunks <= kStage1 / 2;
    double* st = stage[w];
    if (staged) {
#pragma unroll
        for (int c = 0; c < kStage1 / 64; c++) {
            const int g = c * 32 + lane;
            if (abase + 2 * g + 1 < end)
                cp_async16(st + 2 * g, values + abase + 2 * g);
            else if (abase + 2 * g < end)
                st[2 * g] = values[abase + 2 * g];  // last double of the array, no 16-byte read
        }
        cp_async_wait_all();
    }
    __syncwarp();
    if (!valid)
        return;

    const int NN = N * N;
    const int i = row / NN, j = (row / N) % N, k = row % N;
    const bool interior = i > 0 && i < N - 1 && j > 0 && j < N - 1 && k > 0 && k < N - 1;
    double sum;
    if (staged) {
        const double* v = st + (rp - abase);
        sum = interior ? row27(v, x, row, N)
                       : row_generic(v, col_idx + rp, (int)(row_ptr[row + 1] - rp), x);
    } else {
        const double* v = values + rp;
        sum = interior ? row27(v, x, row, N)
                       : row_generic(v, col_idx + rp, (int)(row_ptr[row + 1] - rp), x);
    }
    y[row] = sum;
}

/**
 * @brief One-time check of the kernel's assumption, one thread per row
 *
 * @details Counts rows whose length or columns differ from stencil_row (bad_pattern).
 * csr_staged_kernel needs bad_pattern == 0.
 */
__global__ void check_pattern_kernel(const long long* __restrict__ row_ptr,
                                     const int* __restrict__ col_idx, int N,
                                     unsigned long long* bad_pattern) {
    const int n = N * N * N;
    for (int r = blockIdx.x * blockDim.x + threadIdx.x; r < n; r += gridDim.x * blockDim.x) {
        const long long rp = row_ptr[r], len = row_ptr[r + 1] - rp;
        bool ok = true;
        const int q = stencil_row(
            r, N, [&](int pos, int col) { ok = ok && pos < len && col_idx[rp + pos] == col; });
        if (!ok || q != len)
            atomicAdd(bad_pattern, 1ULL);
    }
}

}  // namespace stencil27

#endif  // SPMV_STENCIL27_FAST_CUH
