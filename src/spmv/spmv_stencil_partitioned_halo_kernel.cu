/**
 * @file spmv_stencil_partitioned_halo_kernel.cu
 * @brief Shared optimized stencil SpMV kernel with halo zones
 *
 * @details
 * Used by both CG solver and standalone SpMV benchmark.
 * Single source of truth for the kernel implementation.
 *
 * Author: Bouhrour Stephane
 * Date: 2025-11-20
 */

/**
 * @brief Optimized stencil SpMV kernel for partitioned CSR with halo zones
 * @details Uses direct column indices (no indirection) for interior stencil points
 */
__global__ void stencil5_csr_partitioned_halo_kernel(
    const long long* __restrict__ row_ptr, const int* __restrict__ col_idx,
    const double* __restrict__ values, const double* __restrict__ x_local,
    const double* __restrict__ x_halo_prev, const double* __restrict__ x_halo_next,
    double* __restrict__ y, int n_local, int row_offset, int N, int grid_size) {
    int local_row = blockIdx.x * blockDim.x + threadIdx.x;
    if (local_row >= n_local)
        return;

    int row = row_offset + local_row;
    int i = row / grid_size;
    int j = row % grid_size;

    long long row_start = row_ptr[local_row];
    long long row_end = row_ptr[local_row + 1];

    double sum = 0.0;

    // Interior points: direct column calculation (no col_idx lookup)
    if (i > 0 && i < grid_size - 1 && j > 0 && j < grid_size - 1 && (row_end - row_start) == 5) {
        int idx_north = row - grid_size;
        int idx_west = row - 1;
        int idx_center = row;
        int idx_east = row + 1;
        int idx_south = row + grid_size;

        // North
        double val_north;
        if (idx_north >= row_offset && idx_north < row_offset + n_local) {
            val_north = x_local[idx_north - row_offset];
        } else if (idx_north >= row_offset - grid_size && idx_north < row_offset) {
            val_north = x_halo_prev[idx_north - (row_offset - grid_size)];
        } else {
            val_north = 0.0;
        }

        // Optimized memory access order: group spatially adjacent accesses first
        // West, Center, East: stride 1 (cache-friendly, always local for interior)
        double val_west = x_local[idx_west - row_offset];
        double val_center = x_local[idx_center - row_offset];
        double val_east = x_local[idx_east - row_offset];

        // South: stride grid_size (separate from contiguous accesses)
        double val_south;
        if (idx_south >= row_offset && idx_south < row_offset + n_local) {
            val_south = x_local[idx_south - row_offset];
        } else if (idx_south >= row_offset + n_local &&
                   idx_south < row_offset + n_local + grid_size) {
            val_south = x_halo_next[idx_south - (row_offset + n_local)];
        } else {
            val_south = 0.0;
        }

        // Reorder operations to group contiguous memory accesses (W-C-E) before large-stride (N-S)
        sum = values[row_start + 1] * val_west + values[row_start + 2] * val_center +
              values[row_start + 3] * val_east + values[row_start + 0] * val_north +
              values[row_start + 4] * val_south;
    }
    // Boundary: CSR traversal with halo mapping
    else {
        for (long long k = row_start; k < row_end; k++) {
            int global_col = col_idx[k];
            double val;

            if (global_col >= row_offset && global_col < row_offset + n_local) {
                val = x_local[global_col - row_offset];
            } else if (x_halo_prev != NULL && global_col >= row_offset - grid_size &&
                       global_col < row_offset) {
                val = x_halo_prev[global_col - (row_offset - grid_size)];
            } else if (x_halo_next != NULL && global_col >= row_offset + n_local &&
                       global_col < row_offset + n_local + grid_size) {
                val = x_halo_next[global_col - (row_offset + n_local)];
            } else {
                val = 0.0;
            }

            sum += values[k] * val;
        }
    }

    y[local_row] = sum;
}
