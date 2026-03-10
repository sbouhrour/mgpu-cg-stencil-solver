/**
 * @file spmv_stencil_csr_direct.cu
 * @brief SpMV for 5-point stencil using CSR with geometric structure exploitation
 *
 * @details
 * CG-friendly stencil SpMV operating directly on CSR without format conversion.
 * Exploits geometric structure to calculate CSR offsets and column indices directly,
 * eliminating memory indirection for interior points.
 *
 * Key optimizations:
 * - Direct CSR offset calculation from grid coordinates (zero row_ptr indirection)
 * - Direct column index calculation (zero col_idx indirection)
 * - Optimized memory access: W-C-E (stride-1) then N-S (stride-grid_size)
 * - Eliminates indirection for 90%+ of points on large grids
 * - Maintains CSR format compatibility for CG iterations
 * - Requires sorted CSR (by column index within each row)
 *
 * Author: Bouhrour Stephane
 * Date: 2025-10-14
 */

#include <stdio.h>
#include "spmv.h"
#include "io.h"

// Device memory
static int* d_row_ptr = nullptr;
static int* d_col_idx = nullptr;
static double* d_values = nullptr;
static double* dX = nullptr;
static double* dY = nullptr;

static const double alpha = 1.0;
static const double beta = 0.0;

// Store grid_size from init
static int grid_size_stored = 0;

/**
 * @brief Calculate CSR offset for interior stencil point
 * @details Exploits regular 5-point stencil structure:
 * - Row 0: corner(3) + edges(4 each) + corner(3)
 * - Rows 1..(grid-2): edge(4) + interior(5 each) + edge(4)
 * - Row (grid-1): corner(3) + edges(4 each) + corner(3)
 *
 * @param row Global row index (1D)
 * @param grid_size Grid dimension
 * @return CSR offset for this row's data
 */
__device__ inline int calculate_interior_csr_offset(int row, int grid_size) {
    int i = row / grid_size;  // Grid row
    int j = row % grid_size;  // Grid column

    // Row 0: corner + edges + corner
    int row0_nnz = 3 + (grid_size - 2) * 4 + 3;

    // Interior rows [1..(i-1)]: edge + interiors + edge
    int interior_row_nnz = 4 + (grid_size - 2) * 5 + 4;

    // Offset before row i
    int offset = row0_nnz + (i - 1) * interior_row_nnz;

    // Within row i: left edge (4) + interior points before j
    offset += 4 + (j - 1) * 5;

    return offset;
}

/**
 * @brief CUDA kernel: CSR-direct stencil SpMV with calculated offsets (single-GPU)
 *
 * @details Interior points: calculate CSR offset and column indices directly
 *          Boundary points: standard CSR traversal
 *          Memory access pattern optimized: West-Center-East (contiguous), then North-South
 */
__global__ void stencil5_csr_direct_kernel(const int* __restrict__ row_ptr,
                                           const int* __restrict__ col_idx,
                                           const double* __restrict__ values,
                                           const double* __restrict__ x, double* __restrict__ y,
                                           int N, int grid_size, double alpha) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N)
        return;

    int i = row / grid_size;
    int j = row % grid_size;

    double sum = 0.0;

    // Interior: direct offset and column calculation (zero indirection)
    if (i > 0 && i < grid_size - 1 && j > 0 && j < grid_size - 1) {
        // Calculate CSR offset directly from grid coordinates (no row_ptr lookup)
        int csr_offset = calculate_interior_csr_offset(row, grid_size);

        // Column indices known from stencil structure (no col_idx lookup)
        // CSR order after sorting: [North, West, Center, East, South] at offsets [0,1,2,3,4]
        int idx_west = row - 1;
        int idx_center = row;
        int idx_east = row + 1;
        int idx_north = row - grid_size;
        int idx_south = row + grid_size;

        // Optimized memory access order: group spatially adjacent vec[] accesses
        // West-Center-East (stride 1, contiguous) first, then North-South (stride grid_size)
        sum = values[csr_offset + 1] * x[idx_west]      // West (CSR offset 1)
              + values[csr_offset + 2] * x[idx_center]  // Center (CSR offset 2)
              + values[csr_offset + 3] * x[idx_east]    // East (CSR offset 3)
              + values[csr_offset + 0] * x[idx_north]   // North (CSR offset 0)
              + values[csr_offset + 4] * x[idx_south];  // South (CSR offset 4)
    }
    // Boundary/corner: standard CSR traversal
    else {
        int row_start = row_ptr[row];
        int row_end = row_ptr[row + 1];

#pragma unroll 8
        for (int k = row_start; k < row_end; k++) {
            sum += values[k] * x[col_idx[k]];
        }
    }

    y[row] = alpha * sum;
}

/**
 * @brief CUDA kernel: CSR-direct stencil SpMV for multi-GPU with row partitioning
 *
 * @details Multi-GPU version with row-band decomposition:
 *  - Each GPU processes rows [row_offset : row_offset + local_rows)
 *  - Full input vector x replicated on all GPUs (required for stencil neighbors)
 *  - Output y_local contains only local partition results
 *  - Zero communication during kernel execution (NCCL used only for CG dot products)
 *
 * @param row_ptr CSR row pointers (full matrix, but only local rows accessed)
 * @param col_idx CSR column indices (full matrix)
 * @param values CSR values (full matrix)
 * @param x Input vector (full, replicated on all GPUs)
 * @param y_local Output vector (local partition only, size = local_rows)
 * @param row_offset Starting row for this GPU's partition
 * @param local_rows Number of rows to process on this GPU
 * @param grid_size 2D grid dimension (sqrt of matrix size)
 * @param alpha Scalar multiplier
 */
__global__ void stencil5_csr_direct_mgpu_kernel(const int* __restrict__ row_ptr,
                                                const int* __restrict__ col_idx,
                                                const double* __restrict__ values,
                                                const double* __restrict__ x,
                                                double* __restrict__ y_local, int row_offset,
                                                int local_rows, int grid_size, double alpha) {
    int local_row = blockIdx.x * blockDim.x + threadIdx.x;
    if (local_row >= local_rows)
        return;

    // Global row index in full matrix
    int row = row_offset + local_row;

    int i = row / grid_size;
    int j = row % grid_size;

    double sum = 0.0;

    // Interior: direct offset and column calculation (requires full CSR)
    if (i > 0 && i < grid_size - 1 && j > 0 && j < grid_size - 1) {
        int csr_offset = calculate_interior_csr_offset(row, grid_size);

        int idx_west = row - 1;
        int idx_center = row;
        int idx_east = row + 1;
        int idx_north = row - grid_size;
        int idx_south = row + grid_size;

        sum = values[csr_offset + 1] * x[idx_west] + values[csr_offset + 2] * x[idx_center] +
              values[csr_offset + 3] * x[idx_east] + values[csr_offset + 0] * x[idx_north] +
              values[csr_offset + 4] * x[idx_south];
    }
    // Boundary/corner: standard CSR traversal
    else {
        int row_start = row_ptr[row];
        int row_end = row_ptr[row + 1];

#pragma unroll 8
        for (int k = row_start; k < row_end; k++) {
            sum += values[k] * x[col_idx[k]];
        }
    }

    // Write to local output (indexed by local_row, not global row)
    y_local[local_row] = alpha * sum;
}

/**
 * @brief Initialize CSR-direct stencil operator
 */
int stencil_csr_direct_init(MatrixData* mat) {
    printf("[STENCIL-CSR-DIRECT] Initializing (CG-friendly, calculated offsets)\n");

    // Build CSR if needed (function handles reuse check)
    build_csr_struct(mat);

    // Allocate device CSR arrays
    CUDA_CHECK(cudaMalloc(&d_row_ptr, (csr_mat.nb_rows + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_col_idx, csr_mat.nb_nonzeros * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_values, csr_mat.nb_nonzeros * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dX, csr_mat.nb_cols * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dY, csr_mat.nb_rows * sizeof(double)));

    // Transfer CSR to device
    CUDA_CHECK(cudaMemcpy(d_row_ptr, csr_mat.row_ptr, (csr_mat.nb_rows + 1) * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_idx, csr_mat.col_indices, csr_mat.nb_nonzeros * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_values, csr_mat.values, csr_mat.nb_nonzeros * sizeof(double),
                          cudaMemcpyHostToDevice));

    // Store grid_size for kernel
    grid_size_stored = mat->grid_size;

    printf("CSR-direct initialized: %d rows, %lld nnz, grid %dx%d\n", csr_mat.nb_rows,
           csr_mat.nb_nonzeros, mat->grid_size, mat->grid_size);

    return 0;
}

/**
 * @brief Execute CSR-direct SpMV
 */
int stencil_csr_direct_run_timed(const double* x, double* y, double* kernel_time_ms) {
    CUDA_CHECK(cudaMemcpy(dX, x, csr_mat.nb_cols * sizeof(double), cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (csr_mat.nb_rows + threads - 1) / threads;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    stencil5_csr_direct_kernel<<<blocks, threads>>>(d_row_ptr, d_col_idx, d_values, dX, dY,
                                                    csr_mat.nb_rows, grid_size_stored, alpha);

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    *kernel_time_ms = (double)milliseconds;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaMemcpy(y, dY, csr_mat.nb_rows * sizeof(double), cudaMemcpyDeviceToHost));

    return 0;
}

/**
 * @brief Execute CSR-direct SpMV with device pointers (GPU-native, zero-copy)
 * @details
 *   Device-native interface for CG solver - no host transfers.
 *   Directly launches kernel with provided device pointers.
 * @param d_x Device input vector pointer
 * @param d_y Device output vector pointer
 * @return 0 on success
 */
int stencil_csr_direct_run_device(const double* d_x, double* d_y) {
    int threads = 256;
    int blocks = (csr_mat.nb_rows + threads - 1) / threads;

    stencil5_csr_direct_kernel<<<blocks, threads>>>(d_row_ptr, d_col_idx, d_values, d_x, d_y,
                                                    csr_mat.nb_rows, grid_size_stored, alpha);

    return 0;
}

/**
 * @brief Cleanup
 */
void stencil_csr_direct_free() {
    printf("[STENCIL-CSR-DIRECT] Cleaning up\n");

    if (d_row_ptr) {
        cudaFree(d_row_ptr);
        d_row_ptr = nullptr;
    }
    if (d_col_idx) {
        cudaFree(d_col_idx);
        d_col_idx = nullptr;
    }
    if (d_values) {
        cudaFree(d_values);
        d_values = nullptr;
    }
    if (dX) {
        cudaFree(dX);
        dX = nullptr;
    }
    if (dY) {
        cudaFree(dY);
        dY = nullptr;
    }
}

// Operator structure
SpmvOperator SPMV_STENCIL5_CSR = {"stencil5-csr", stencil_csr_direct_init,
                                  stencil_csr_direct_run_timed, stencil_csr_direct_run_device,
                                  stencil_csr_direct_free};
