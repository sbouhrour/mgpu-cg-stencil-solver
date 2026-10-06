/**
 * @file spmv_stencil_3d_27pt_partitioned_halo_kernel.cu
 * @brief Optimized 3D 27-point stencil SpMV kernel with Z-slab partitioning
 *
 * @details
 * Z-slab partitioning: each GPU owns contiguous Z-planes.
 * Halo contains one full XY-plane (N² elements) from neighbors.
 * 27-point stencil: center + 6 face + 12 edge + 8 corner neighbors.
 *
 * Two kernels with the same arithmetic and summation order, hence bitwise identical results:
 * - stencil27_csr_partitioned_halo_kernel_3d: one thread per row, coefficients read straight
 *   from global memory (consecutive lanes are 27 doubles apart).
 * - stencil27_staged_subrange_kernel_3d: each warp first copies the coefficients of its 32 rows
 *   to shared memory with coalesced 16-byte loads (see spmv_stencil27_fast.cuh).
 *
 * Author: Bouhrour Stephane
 */

#include "spmv_stencil27_fast.cuh"

/**
 * @brief Optimized 3D 27-point stencil SpMV kernel for partitioned CSR with Z-slab halo
 *
 * @param[in] row_ptr CSR row pointers
 * @param[in] col_idx CSR column indices
 * @param[in] values CSR values
 * @param[in] x_local Local vector partition
 * @param[in] x_halo_prev Previous Z-plane halo (NULL if rank==0)
 * @param[in] x_halo_next Next Z-plane halo (NULL if rank==world_size-1)
 * @param[out] y Output vector partition
 * @param[in] n_local Number of local rows
 * @param[in] row_offset Global row offset for this partition
 * @param[in] N_total Total grid dimension (NxNxN grid)
 * @param[in] grid_size N (used for stencil pattern)
 */
__global__ void stencil27_csr_partitioned_halo_kernel_3d(
    const long long* __restrict__ row_ptr, const int* __restrict__ col_idx,
    const double* __restrict__ values, const double* __restrict__ x_local,
    const double* __restrict__ x_halo_prev, const double* __restrict__ x_halo_next,
    double* __restrict__ y, int n_local, int row_offset, int N_total, int grid_size) {

    int local_row = blockIdx.x * blockDim.x + threadIdx.x;
    if (local_row >= n_local)
        return;

    int global_row = row_offset + local_row;
    int N = grid_size;

    // Decompose global row to 3D coordinates: (i, j, k)
    int i = global_row / (N * N);
    int j = (global_row / N) % N;
    int k = global_row % N;

    // Decompose local row to Z-plane information
    int local_nz = n_local / (N * N);
    int local_z = local_row / (N * N);

    double sum = 0.0;

    // Geometric interior check — no row_ptr reads needed
    bool is_interior = (i > 0 && i < N - 1 && j > 0 && j < N - 1 && k > 0 && k < N - 1 &&
                        local_z > 0 && local_z < local_nz - 1);

    if (is_interior) {
        long long csr_offset = row_ptr[local_row];

        // 27 coefficients from CSR values (sorted by ascending global column index)
        // Z-plane i-1
        sum = values[csr_offset + 0] * x_local[local_row - N * N - N - 1];   // (i-1,j-1,k-1)
        sum += values[csr_offset + 1] * x_local[local_row - N * N - N];      // (i-1,j-1,k)
        sum += values[csr_offset + 2] * x_local[local_row - N * N - N + 1];  // (i-1,j-1,k+1)
        sum += values[csr_offset + 3] * x_local[local_row - N * N - 1];      // (i-1,j,k-1)
        sum += values[csr_offset + 4] * x_local[local_row - N * N];          // (i-1,j,k)
        sum += values[csr_offset + 5] * x_local[local_row - N * N + 1];      // (i-1,j,k+1)
        sum += values[csr_offset + 6] * x_local[local_row - N * N + N - 1];  // (i-1,j+1,k-1)
        sum += values[csr_offset + 7] * x_local[local_row - N * N + N];      // (i-1,j+1,k)
        sum += values[csr_offset + 8] * x_local[local_row - N * N + N + 1];  // (i-1,j+1,k+1)
        // Z-plane i
        sum += values[csr_offset + 9] * x_local[local_row - N - 1];   // (i,j-1,k-1)
        sum += values[csr_offset + 10] * x_local[local_row - N];      // (i,j-1,k)
        sum += values[csr_offset + 11] * x_local[local_row - N + 1];  // (i,j-1,k+1)
        sum += values[csr_offset + 12] * x_local[local_row - 1];      // (i,j,k-1)
        sum += values[csr_offset + 13] * x_local[local_row];          // (i,j,k) center
        sum += values[csr_offset + 14] * x_local[local_row + 1];      // (i,j,k+1)
        sum += values[csr_offset + 15] * x_local[local_row + N - 1];  // (i,j+1,k-1)
        sum += values[csr_offset + 16] * x_local[local_row + N];      // (i,j+1,k)
        sum += values[csr_offset + 17] * x_local[local_row + N + 1];  // (i,j+1,k+1)
        // Z-plane i+1
        sum += values[csr_offset + 18] * x_local[local_row + N * N - N - 1];  // (i+1,j-1,k-1)
        sum += values[csr_offset + 19] * x_local[local_row + N * N - N];      // (i+1,j-1,k)
        sum += values[csr_offset + 20] * x_local[local_row + N * N - N + 1];  // (i+1,j-1,k+1)
        sum += values[csr_offset + 21] * x_local[local_row + N * N - 1];      // (i+1,j,k-1)
        sum += values[csr_offset + 22] * x_local[local_row + N * N];          // (i+1,j,k)
        sum += values[csr_offset + 23] * x_local[local_row + N * N + 1];      // (i+1,j,k+1)
        sum += values[csr_offset + 24] * x_local[local_row + N * N + N - 1];  // (i+1,j+1,k-1)
        sum += values[csr_offset + 25] * x_local[local_row + N * N + N];      // (i+1,j+1,k)
        sum += values[csr_offset + 26] * x_local[local_row + N * N + N + 1];  // (i+1,j+1,k+1)
    }
    // Boundary/corner: CSR traversal with halo mapping
    else {
        long long row_start = row_ptr[local_row];
        long long row_end = row_ptr[local_row + 1];
        for (long long jj = row_start; jj < row_end; jj++) {
            int global_col = col_idx[jj];
            double val;

            // Check if column is in local partition
            if (global_col >= row_offset && global_col < row_offset + n_local) {
                val = x_local[global_col - row_offset];
            }
            // Check if column is in previous Z-plane halo
            else if (x_halo_prev != NULL && global_col >= row_offset - (N * N) &&
                     global_col < row_offset) {
                int halo_offset = global_col - (row_offset - (N * N));
                val = x_halo_prev[halo_offset];
            }
            // Check if column is in next Z-plane halo
            else if (x_halo_next != NULL && global_col >= row_offset + n_local &&
                     global_col < row_offset + n_local + (N * N)) {
                int halo_offset = global_col - (row_offset + n_local);
                val = x_halo_next[halo_offset];
            }
            // Column is outside known regions (boundary of domain)
            else {
                val = 0.0;
            }

            sum += values[jj] * val;
        }
    }

    y[local_row] = sum;
}

/** @brief Warps per block of stencil27_staged_subrange_kernel_3d */
static constexpr int kStencil27StagedWarps = 4;

/**
 * @brief Boundary row of the partition: CSR traversal with halo mapping
 *
 * @details v and cols point at the row's first entry. Same loop as the boundary branch of
 * stencil27_csr_partitioned_halo_kernel_3d.
 */
__device__ __forceinline__ double stencil27_row_halo(const double* v, const int* __restrict__ cols,
                                                     int len, const double* __restrict__ x_local,
                                                     const double* __restrict__ x_halo_prev,
                                                     const double* __restrict__ x_halo_next,
                                                     int n_local, int row_offset, int N) {
    double sum = 0.0;
    for (int q = 0; q < len; q++) {
        const int global_col = cols[q];
        double val;
        if (global_col >= row_offset && global_col < row_offset + n_local)
            val = x_local[global_col - row_offset];
        else if (x_halo_prev != NULL && global_col >= row_offset - (N * N) &&
                 global_col < row_offset)
            val = x_halo_prev[global_col - (row_offset - (N * N))];
        else if (x_halo_next != NULL && global_col >= row_offset + n_local &&
                 global_col < row_offset + n_local + (N * N))
            val = x_halo_next[global_col - (row_offset + n_local)];
        else
            val = 0.0;
        sum += v[q] * val;
    }
    return sum;
}

/**
 * @brief Row on a domain face: plain CSR loop when every column lies in the local partition,
 * halo mapping otherwise
 *
 * @details The plain loop has no branch per entry, so the compiler batches its loads; a warp
 * holding a face row (k = 0 or N - 1 every N rows) waits for that row's loop.
 */
__device__ __forceinline__ double
stencil27_row_face(const double* v, const int* __restrict__ cols, int len, int local_row,
                   const double* __restrict__ x_local, const double* __restrict__ x_halo_prev,
                   const double* __restrict__ x_halo_next, int n_local, int row_offset, int N) {
    const int reach = N * N + N + 1;  // largest |column - row| of the stencil
    if (local_row >= reach && local_row + reach < n_local) {
        double sum = 0.0;
        for (int q = 0; q < len; q++)
            sum += v[q] * x_local[cols[q] - row_offset];
        return sum;
    }
    return stencil27_row_halo(v, cols, len, x_local, x_halo_prev, x_halo_next, n_local, row_offset,
                              N);
}

/**
 * @brief 27-point SpMV over local rows [subrange_start, subrange_start + subrange_count), with
 * warp-coalesced staging of the coefficients
 *
 * @details One thread per row, as stencil27_csr_partitioned_halo_kernel_3d. The 32 rows of a
 * warp own one contiguous span of values: the warp copies it to shared memory with 16-byte
 * cp.async, then each thread reads its coefficients from there (below sm_80, where cp.async does
 * not exist, the coefficients are read from global memory). A span larger than the stage
 * buffer (not a stencil pattern) is read from global memory instead.
 * Interior rows read x_local at the 27 stencil offsets; rows on a domain face or on the first or
 * last plane of the partition go through the CSR loop with halo mapping.
 *
 * @param[in] n_local Number of rows of the whole partition (subrange or not)
 *
 * Launched by stencil27_staged_spmv_3d. values must be 16-byte aligned (cudaMalloc: 256).
 */
__global__ void __launch_bounds__(kStencil27StagedWarps * 32) stencil27_staged_subrange_kernel_3d(
    const long long* __restrict__ row_ptr, const int* __restrict__ col_idx,
    const double* __restrict__ values, const double* __restrict__ x_local,
    const double* __restrict__ x_halo_prev, const double* __restrict__ x_halo_next,
    double* __restrict__ y, int n_local, int row_offset, int N_total, int grid_size,
    int subrange_start, int subrange_count) {
    using namespace stencil27;
    __shared__ __align__(16) double stage[kStencil27StagedWarps][kStage1];
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int t0 = (blockIdx.x * kStencil27StagedWarps + w) * 32;
    if (t0 >= subrange_count)
        return;  // whole warp
    const int local_row = subrange_start + t0 + lane;
    const bool valid = t0 + lane < subrange_count;
    const int last = subrange_start + min(t0 + 31, subrange_count - 1);
    const long long rp = valid ? row_ptr[local_row] : 0;
    const long long base = __shfl_sync(kFull, rp, 0);
    const long long end = row_ptr[last + 1];
    const long long abase = base & ~1LL;
    const int nchunks = (int)((end - abase + 1) >> 1);
    // Below sm_80 (no cp.async), staging through registers is slower than direct reads
#if __CUDA_ARCH__ >= 800
    const bool staged = nchunks <= kStage1 / 2;
#else
    const bool staged = false;
#endif
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

    const int N = grid_size;
    const int global_row = row_offset + local_row;
    const int NN = N * N;
    const int j = (global_row / N) % N, k = global_row % N;
    // 0 < i < N - 1 and 0 < local_z < local_nz - 1, written as row bounds (no per-row division)
    const int z_end = (n_local / NN - 1) * NN;
    const bool interior = global_row >= NN && global_row < (N - 1) * NN && j > 0 && j < N - 1 &&
                          k > 0 && k < N - 1 && local_row >= NN && local_row < z_end;
    double sum;
    // Two branches so that each reads v from a known address space (shared or global)
    if (staged) {
        const double* v = st + (rp - abase);
        sum = interior ? row27(v, x_local, local_row, N)
                       : stencil27_row_face(v, col_idx + rp, (int)(row_ptr[local_row + 1] - rp),
                                            local_row, x_local, x_halo_prev, x_halo_next, n_local,
                                            row_offset, N);
    } else {
        const double* v = values + rp;
        sum = interior ? row27(v, x_local, local_row, N)
                       : stencil27_row_face(v, col_idx + rp, (int)(row_ptr[local_row + 1] - rp),
                                            local_row, x_local, x_halo_prev, x_halo_next, n_local,
                                            row_offset, N);
    }
    y[local_row] = sum;
}

/**
 * @brief Launches stencil27_staged_subrange_kernel_3d over local rows [start, start + count)
 */
void stencil27_staged_spmv_3d(const long long* row_ptr, const int* col_idx, const double* values,
                              const double* x_local, const double* x_halo_prev,
                              const double* x_halo_next, double* y, int n_local, int row_offset,
                              int N_total, int grid_size, int start, int count,
                              cudaStream_t stream) {
    if (count <= 0)
        return;
    const int threads = kStencil27StagedWarps * 32;
    const int blocks = (count + threads - 1) / threads;
    stencil27_staged_subrange_kernel_3d<<<blocks, threads, 0, stream>>>(
        row_ptr, col_idx, values, x_local, x_halo_prev, x_halo_next, y, n_local, row_offset,
        N_total, grid_size, start, count);
}
