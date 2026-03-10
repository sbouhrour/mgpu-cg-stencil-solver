/**
 * @file cg_solver_mgpu_overlap.cu
 * @brief Multi-GPU CG solver with compute-communication overlap
 *
 * Architecture:
 * - Two CUDA streams: stream_compute (interior SpMV), stream_comm (halo D2H/H2D)
 * - Per CG iteration, the SpMV phase overlaps MPI halo exchange with interior SpMV
 * - Boundary SpMV runs after halo data arrives
 * - All other CG operations (dot, axpy, etc.) are identical to the synchronous solver
 *
 * Overlap sequence per iteration:
 *   1. D2H boundary p on stream_comm (async, returns immediately)
 *   2. Interior SpMV on stream_compute (overlaps with D2H + MPI)
 *   3. cudaStreamSynchronize(stream_comm) - D2H complete, host buffers ready
 *   4. MPI_Isend/Irecv (non-blocking)
 *   5. MPI_Waitall (interior kernel continues on GPU during MPI)
 *   6. H2D halo on stream_comm
 *   7. cudaStreamSynchronize(stream_comm) - halo data on device
 *   8. cudaStreamSynchronize(stream_compute) - interior done
 *   9. Boundary SpMV on stream_compute
 *
 * Interior and boundary write to disjoint ranges of d_Ap.
 *
 * Author: Bouhrour Stephane
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <nvtx3/nvToolsExt.h>

#include "spmv.h"
#include "io.h"
#include "solvers/cg_solver_mgpu_partitioned.h"

/* ================================================================
 * OverlapPartition: defines interior vs boundary row ranges
 * ================================================================ */

typedef struct {
    int interior_start;       ///< First interior row (local index)
    int interior_count;       ///< Number of interior rows
    int boundary_prev_start;  ///< Always 0
    int boundary_prev_count;  ///< boundary_prev rows needing halo_prev (0 if rank==0)
    int boundary_next_start;  ///< n_local - boundary_next_count
    int boundary_next_count;  ///< boundary_next rows needing halo_next (0 if last rank)
    int stencil_dim;          ///< 2 or 3 (determines halo_elems)
    int halo_elems;  ///< halo elements per direction (grid_size for 2D, grid_size^2 for 3D)
} OverlapPartition;

static void compute_overlap_partition(OverlapPartition* part, int n_local, int grid_size, int rank,
                                      int world_size, int stencil_dim) {
    part->stencil_dim = stencil_dim;

    // Compute halo_elems based on stencil dimension
    if (stencil_dim == 2) {
        part->halo_elems = grid_size;
    } else if (stencil_dim == 3) {
        part->halo_elems = grid_size * grid_size;
    } else {
        // Default to 2D behavior
        part->halo_elems = grid_size;
        part->stencil_dim = 2;
    }

    // Boundary rows = rows that depend on halo data from neighbors
    // For 2D: one grid row = grid_size rows
    // For 3D: one XY-plane = grid_size * grid_size rows
    int boundary_rows = (stencil_dim == 3) ? (grid_size * grid_size) : grid_size;

    int boundary_prev_rows = (rank > 0) ? boundary_rows : 0;
    int boundary_next_rows = 0;

    part->boundary_prev_start = 0;
    part->boundary_prev_count = boundary_prev_rows;

    int remaining = n_local - part->boundary_prev_count;
    if (remaining < 0)
        remaining = 0;

    boundary_next_rows =
        (rank < world_size - 1) ? (boundary_rows < remaining ? boundary_rows : remaining) : 0;
    part->boundary_next_count = boundary_next_rows;
    part->boundary_next_start = n_local - part->boundary_next_count;

    part->interior_start = part->boundary_prev_count;
    part->interior_count = part->boundary_next_start - part->interior_start;
    if (part->interior_count < 0)
        part->interior_count = 0;
}

/* ================================================================
 * Subrange SpMV kernel for overlap
 * ================================================================ */

/**
 * @brief Stencil SpMV kernel operating on a sub-range of local rows
 *
 * Same logic as stencil5_csr_partitioned_halo_kernel but processes only
 * rows [subrange_start, subrange_start + subrange_count) within the
 * local partition. Uses n_local_full for x_local bounds checking.
 */
__global__ void stencil5_overlap_subrange_kernel(
    const long long* __restrict__ row_ptr, const int* __restrict__ col_idx,
    const double* __restrict__ values, const double* __restrict__ x_local,
    const double* __restrict__ x_halo_prev, const double* __restrict__ x_halo_next,
    double* __restrict__ y, int n_local_full, int row_offset, int N, int grid_size,
    int subrange_start, int subrange_count) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= subrange_count)
        return;

    int local_row = subrange_start + tid;
    int row = row_offset + local_row;
    int i = row / grid_size;
    int j = row % grid_size;

    long long row_start = row_ptr[local_row];
    long long row_end = row_ptr[local_row + 1];

    double sum = 0.0;

    // Interior stencil points: direct column calculation
    if (i > 0 && i < grid_size - 1 && j > 0 && j < grid_size - 1 && (row_end - row_start) == 5) {
        int idx_north = row - grid_size;
        int idx_west = row - 1;
        int idx_center = row;
        int idx_east = row + 1;
        int idx_south = row + grid_size;

        // North
        double val_north;
        if (idx_north >= row_offset && idx_north < row_offset + n_local_full) {
            val_north = x_local[idx_north - row_offset];
        } else if (idx_north >= row_offset - grid_size && idx_north < row_offset) {
            val_north = x_halo_prev[idx_north - (row_offset - grid_size)];
        } else {
            val_north = 0.0;
        }

        double val_west = x_local[idx_west - row_offset];
        double val_center = x_local[idx_center - row_offset];
        double val_east = x_local[idx_east - row_offset];

        // South
        double val_south;
        if (idx_south >= row_offset && idx_south < row_offset + n_local_full) {
            val_south = x_local[idx_south - row_offset];
        } else if (idx_south >= row_offset + n_local_full &&
                   idx_south < row_offset + n_local_full + grid_size) {
            val_south = x_halo_next[idx_south - (row_offset + n_local_full)];
        } else {
            val_south = 0.0;
        }

        sum = values[row_start + 1] * val_west + values[row_start + 2] * val_center +
              values[row_start + 3] * val_east + values[row_start + 0] * val_north +
              values[row_start + 4] * val_south;
    }
    // Boundary/corner: CSR traversal with halo mapping
    else {
        for (long long k = row_start; k < row_end; k++) {
            int global_col = col_idx[k];
            double val;

            if (global_col >= row_offset && global_col < row_offset + n_local_full) {
                val = x_local[global_col - row_offset];
            } else if (x_halo_prev != NULL && global_col >= row_offset - grid_size &&
                       global_col < row_offset) {
                val = x_halo_prev[global_col - (row_offset - grid_size)];
            } else if (x_halo_next != NULL && global_col >= row_offset + n_local_full &&
                       global_col < row_offset + n_local_full + grid_size) {
                val = x_halo_next[global_col - (row_offset + n_local_full)];
            } else {
                val = 0.0;
            }

            sum += values[k] * val;
        }
    }

    y[local_row] = sum;
}

/**
 * @brief 3D 7-point stencil SpMV kernel operating on a sub-range of local rows
 *
 * Same logic as stencil7_csr_partitioned_halo_kernel_3d but processes only
 * rows [subrange_start, subrange_start + subrange_count) within the
 * local partition. Halo is one full XY-plane (N² elements) per direction.
 */
__global__ void stencil7_overlap_subrange_kernel_3d(
    const long long* __restrict__ row_ptr, const int* __restrict__ col_idx,
    const double* __restrict__ values, const double* __restrict__ x_local,
    const double* __restrict__ x_halo_prev, const double* __restrict__ x_halo_next,
    double* __restrict__ y, int n_local_full, int row_offset, int N_total, int grid_size,
    int subrange_start, int subrange_count) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= subrange_count)
        return;

    int local_row = subrange_start + tid;
    int global_row = row_offset + local_row;
    int N = grid_size;

    // 3D coordinates
    int i = global_row / (N * N);
    int j = (global_row / N) % N;
    int k = global_row % N;

    // Local Z-plane info
    int local_nz = n_local_full / (N * N);
    int local_z = local_row / (N * N);

    double sum = 0.0;

    // Geometric interior check — no row_ptr reads needed
    bool is_interior = (i > 0 && i < N - 1 && j > 0 && j < N - 1 && k > 0 && k < N - 1 &&
                        local_z > 0 && local_z < local_nz - 1);

    if (is_interior) {
        long long csr_offset = row_ptr[local_row];

        // 7 coefficients from CSR values (sorted by ascending global column index)
        sum = values[csr_offset + 0] * x_local[local_row - N * N];   // (i-1,j,k)
        sum += values[csr_offset + 1] * x_local[local_row - N];      // (i,j-1,k)
        sum += values[csr_offset + 2] * x_local[local_row - 1];      // (i,j,k-1)
        sum += values[csr_offset + 3] * x_local[local_row];          // (i,j,k) center
        sum += values[csr_offset + 4] * x_local[local_row + 1];      // (i,j,k+1)
        sum += values[csr_offset + 5] * x_local[local_row + N];      // (i,j+1,k)
        sum += values[csr_offset + 6] * x_local[local_row + N * N];  // (i+1,j,k)
    }
    // Boundary: CSR traversal with Z-plane halo mapping
    else {
        long long row_start = row_ptr[local_row];
        long long row_end = row_ptr[local_row + 1];
        for (long long jj = row_start; jj < row_end; jj++) {
            int global_col = col_idx[jj];
            double val;

            if (global_col >= row_offset && global_col < row_offset + n_local_full) {
                val = x_local[global_col - row_offset];
            } else if (x_halo_prev != NULL && global_col >= row_offset - (N * N) &&
                       global_col < row_offset) {
                val = x_halo_prev[global_col - (row_offset - (N * N))];
            } else if (x_halo_next != NULL && global_col >= row_offset + n_local_full &&
                       global_col < row_offset + n_local_full + (N * N)) {
                val = x_halo_next[global_col - (row_offset + n_local_full)];
            } else {
                val = 0.0;
            }

            sum += values[jj] * val;
        }
    }

    y[local_row] = sum;
}

/**
 * @brief 3D 27-point stencil SpMV kernel operating on a sub-range of local rows
 *
 * Same logic as stencil27_csr_partitioned_halo_kernel_3d but processes only
 * rows [subrange_start, subrange_start + subrange_count) within the
 * local partition. Halo is one full XY-plane (N² elements) per direction.
 */
__global__ void stencil27_overlap_subrange_kernel_3d(
    const long long* __restrict__ row_ptr, const int* __restrict__ col_idx,
    const double* __restrict__ values, const double* __restrict__ x_local,
    const double* __restrict__ x_halo_prev, const double* __restrict__ x_halo_next,
    double* __restrict__ y, int n_local_full, int row_offset, int N_total, int grid_size,
    int subrange_start, int subrange_count) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= subrange_count)
        return;

    int local_row = subrange_start + tid;
    int global_row = row_offset + local_row;
    int N = grid_size;

    // 3D coordinates
    int i = global_row / (N * N);
    int j = (global_row / N) % N;
    int k = global_row % N;

    // Local Z-plane info
    int local_nz = n_local_full / (N * N);
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
    // Boundary: CSR traversal with Z-plane halo mapping
    else {
        long long row_start = row_ptr[local_row];
        long long row_end = row_ptr[local_row + 1];
        for (long long jj = row_start; jj < row_end; jj++) {
            int global_col = col_idx[jj];
            double val;

            if (global_col >= row_offset && global_col < row_offset + n_local_full) {
                val = x_local[global_col - row_offset];
            } else if (x_halo_prev != NULL && global_col >= row_offset - (N * N) &&
                       global_col < row_offset) {
                val = x_halo_prev[global_col - (row_offset - (N * N))];
            } else if (x_halo_next != NULL && global_col >= row_offset + n_local_full &&
                       global_col < row_offset + n_local_full + (N * N)) {
                val = x_halo_next[global_col - (row_offset + n_local_full)];
            } else {
                val = 0.0;
            }

            sum += values[jj] * val;
        }
    }

    y[local_row] = sum;
}

/* ================================================================
 * External kernels (defined in cg_solver_mgpu_partitioned.cu and
 * spmv_stencil_partitioned_halo_kernel.cu)
 * ================================================================ */

extern __global__ void axpy_kernel(double alpha, const double* x, double* y, int n);
extern __global__ void axpby_kernel(double alpha, const double* x, double beta, double* y, int n);
extern __global__ void stencil5_csr_partitioned_halo_kernel(
    const long long* __restrict__ row_ptr, const int* __restrict__ col_idx,
    const double* __restrict__ values, const double* __restrict__ x_local,
    const double* __restrict__ x_halo_prev, const double* __restrict__ x_halo_next,
    double* __restrict__ y, int n_local, int row_offset, int N, int grid_size);
extern __global__ void stencil7_csr_partitioned_halo_kernel_3d(
    const long long* __restrict__ row_ptr, const int* __restrict__ col_idx,
    const double* __restrict__ values, const double* __restrict__ x_local,
    const double* __restrict__ x_halo_prev, const double* __restrict__ x_halo_next,
    double* __restrict__ y, int n_local, int row_offset, int N_total, int grid_size);
extern __global__ void stencil27_csr_partitioned_halo_kernel_3d(
    const long long* __restrict__ row_ptr, const int* __restrict__ col_idx,
    const double* __restrict__ values, const double* __restrict__ x_local,
    const double* __restrict__ x_halo_prev, const double* __restrict__ x_halo_next,
    double* __restrict__ y, int n_local, int row_offset, int N_total, int grid_size);

/* ================================================================
 * Local helpers
 * ================================================================ */

static double overlap_compute_local_dot(cublasHandle_t cublas_handle, const double* d_x,
                                        const double* d_y, int n) {
    double result;
    cublasStatus_t status = cublasDdot(cublas_handle, n, d_x, 1, d_y, 1, &result);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS ddot failed\n");
        exit(EXIT_FAILURE);
    }
    return result;
}

/**
 * @brief Synchronous halo exchange (for initial setup only)
 */
static void exchange_halo_sync(const double* d_send_prev, const double* d_send_next,
                               double* d_recv_prev, double* d_recv_next, double* h_send_prev,
                               double* h_send_next, double* h_recv_prev, double* h_recv_next,
                               int halo_size, int rank, int world_size, cudaStream_t stream) {
    MPI_Request requests[4];
    int req_count = 0;

    // D2H
    if (rank > 0 && d_send_prev != NULL) {
        CUDA_CHECK(cudaMemcpyAsync(h_send_prev, d_send_prev, halo_size * sizeof(double),
                                   cudaMemcpyDeviceToHost, stream));
    }
    if (rank < world_size - 1 && d_send_next != NULL) {
        CUDA_CHECK(cudaMemcpyAsync(h_send_next, d_send_next, halo_size * sizeof(double),
                                   cudaMemcpyDeviceToHost, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // MPI
    if (rank > 0) {
        MPI_Isend(h_send_prev, halo_size, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD,
                  &requests[req_count++]);
        MPI_Irecv(h_recv_prev, halo_size, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD,
                  &requests[req_count++]);
    }
    if (rank < world_size - 1) {
        MPI_Isend(h_send_next, halo_size, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD,
                  &requests[req_count++]);
        MPI_Irecv(h_recv_next, halo_size, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD,
                  &requests[req_count++]);
    }
    if (req_count > 0) {
        MPI_Waitall(req_count, requests, MPI_STATUSES_IGNORE);
    }

    // H2D
    if (rank > 0 && d_recv_prev != NULL) {
        CUDA_CHECK(cudaMemcpyAsync(d_recv_prev, h_recv_prev, halo_size * sizeof(double),
                                   cudaMemcpyHostToDevice, stream));
    }
    if (rank < world_size - 1 && d_recv_next != NULL) {
        CUDA_CHECK(cudaMemcpyAsync(d_recv_next, h_recv_next, halo_size * sizeof(double),
                                   cudaMemcpyHostToDevice, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

/**
 * @brief Launch async D2H copies for halo boundary data
 *
 * Returns immediately. The D2H transfers run on the given stream.
 * Caller must sync the stream before reading h_send buffers.
 */
static void exchange_halo_d2h_start(const double* d_local, int n_local, double* h_send_prev,
                                    double* h_send_next, int halo_elems, int rank, int world_size,
                                    cudaStream_t stream) {
    if (rank > 0) {
        CUDA_CHECK(cudaMemcpyAsync(h_send_prev, d_local, halo_elems * sizeof(double),
                                   cudaMemcpyDeviceToHost, stream));
    }
    if (rank < world_size - 1) {
        CUDA_CHECK(cudaMemcpyAsync(h_send_next, d_local + (n_local - halo_elems),
                                   halo_elems * sizeof(double), cudaMemcpyDeviceToHost, stream));
    }
}

/**
 * @brief Sync D2H stream and start MPI non-blocking sends/receives
 *
 * Must be called after exchange_halo_d2h_start. Synchronizes the stream
 * to ensure D2H is complete, then issues MPI_Isend/Irecv.
 */
static void exchange_halo_mpi_start(double* h_send_prev, double* h_send_next, double* h_recv_prev,
                                    double* h_recv_next, int halo_elems, int rank, int world_size,
                                    cudaStream_t stream, MPI_Request* requests, int* req_count) {
    *req_count = 0;
    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (rank > 0) {
        MPI_Isend(h_send_prev, halo_elems, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD,
                  &requests[(*req_count)++]);
        MPI_Irecv(h_recv_prev, halo_elems, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD,
                  &requests[(*req_count)++]);
    }
    if (rank < world_size - 1) {
        MPI_Isend(h_send_next, halo_elems, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD,
                  &requests[(*req_count)++]);
        MPI_Irecv(h_recv_next, halo_elems, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD,
                  &requests[(*req_count)++]);
    }
}

/**
 * @brief Finish async halo exchange: MPI_Waitall + H2D
 *
 * @param[out] d_recv_prev Device buffer to receive prev halo
 * @param[out] d_recv_next Device buffer to receive next halo
 * @param[in] h_recv_prev Pinned host buffer with received data from prev
 * @param[in] h_recv_next Pinned host buffer with received data from next
 * @param[in] halo_elems Number of elements in halo
 * @param[in] rank MPI rank
 * @param[in] world_size MPI world size
 * @param[in] stream CUDA stream for H2D operations
 * @param[in] requests MPI request array
 * @param[in] req_count Number of active MPI requests
 */
static void exchange_halo_async_finish(double* d_recv_prev, double* d_recv_next,
                                       const double* h_recv_prev, const double* h_recv_next,
                                       int halo_elems, int rank, int world_size,
                                       cudaStream_t stream, MPI_Request* requests, int req_count) {
    // MPI_Waitall
    if (req_count > 0) {
        MPI_Waitall(req_count, requests, MPI_STATUSES_IGNORE);
    }

    // H2D on stream
    if (rank > 0 && d_recv_prev != NULL) {
        CUDA_CHECK(cudaMemcpyAsync(d_recv_prev, h_recv_prev, halo_elems * sizeof(double),
                                   cudaMemcpyHostToDevice, stream));
    }
    if (rank < world_size - 1 && d_recv_next != NULL) {
        CUDA_CHECK(cudaMemcpyAsync(d_recv_next, h_recv_next, halo_elems * sizeof(double),
                                   cudaMemcpyHostToDevice, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

/* ================================================================
 * Multi-GPU CG solver with compute-communication overlap
 * ================================================================ */

int cg_solve_mgpu_partitioned_overlap(SpmvOperator* spmv_op, MatrixData* mat, const double* b,
                                      double* x, CGConfigMultiGPU config, CGStatsMultiGPU* stats) {

    // MPI initialization
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int n = mat->rows;
    int grid_size = mat->grid_size;

    if (rank == 0 && config.verbose >= 1) {
        printf("\n========================================\n");
        printf("Multi-GPU CG Solver (OVERLAP)\n");
        printf("========================================\n");
        printf("MPI ranks: %d\n", world_size);
        printf("Problem size: %d unknowns\n", n);
        printf("Max iterations: %d\n", config.max_iters);
        printf("Tolerance: %.1e\n", config.tolerance);
        printf("========================================\n\n");
    }

    // Set GPU device
    CUDA_CHECK(cudaSetDevice(rank));

    // Partition: 1D row-band decomposition
    int n_local = n / world_size;
    int row_offset = rank * n_local;
    if (rank == world_size - 1) {
        n_local = n - row_offset;
    }

    if (config.verbose >= 1) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, rank));
        printf("[Rank %d] GPU %d: %s (CC %d.%d)\n", rank, rank, prop.name, prop.major, prop.minor);
        printf("[Rank %d] Rows: [%d:%d) (%d rows)\n", rank, row_offset, row_offset + n_local,
               n_local);
    }

    // Compute overlap partition (2D stencil)
    OverlapPartition partition;
    compute_overlap_partition(&partition, n_local, grid_size, rank, world_size, 2);

    if (config.verbose >= 1) {
        printf("[Rank %d] Overlap partition: interior=[%d:%d) (%d rows), "
               "boundary_prev=%d rows, boundary_next=%d rows\n",
               rank, partition.interior_start, partition.interior_start + partition.interior_count,
               partition.interior_count, partition.boundary_prev_count,
               partition.boundary_next_count);
    }

    // Create two CUDA streams (non-blocking for concurrent execution)
    cudaStream_t stream_compute, stream_comm;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_compute, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_comm, cudaStreamNonBlocking));

    // Event to synchronize p updates (stream_compute) before D2H reads (stream_comm)
    cudaEvent_t p_updated_event;
    CUDA_CHECK(cudaEventCreate(&p_updated_event));

    cublasHandle_t cublas_handle;
    cublasStatus_t cublas_status = cublasCreate(&cublas_handle);
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS initialization failed\n");
        exit(EXIT_FAILURE);
    }
    cublasSetStream(cublas_handle, stream_compute);

    // Build local CSR partition (same as synchronous solver)
    if (rank == 0 && config.verbose >= 1) {
        printf("Building local CSR partitions...\n");
    }

    build_csr_struct(mat);

    long long local_nnz = csr_mat.row_ptr[row_offset + n_local] - csr_mat.row_ptr[row_offset];

    long long* d_row_ptr;
    int* d_col_idx;
    double* d_values;

    CUDA_CHECK(cudaMalloc(&d_row_ptr, (n_local + 1) * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_col_idx, (size_t)local_nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_values, (size_t)local_nnz * sizeof(double)));

    long long* local_row_ptr = (long long*)malloc((n_local + 1) * sizeof(long long));
    long long offset = csr_mat.row_ptr[row_offset];
    for (int i = 0; i <= n_local; i++) {
        local_row_ptr[i] = csr_mat.row_ptr[row_offset + i] - offset;
    }

    // Use stream_compute for ALL device operations to avoid race conditions
    // with non-blocking streams (cudaMemset/cudaMemcpy on the default stream
    // do not synchronize with non-blocking streams)
    CUDA_CHECK(cudaMemcpyAsync(d_row_ptr, local_row_ptr, (n_local + 1) * sizeof(long long),
                               cudaMemcpyHostToDevice, stream_compute));
    CUDA_CHECK(cudaMemcpyAsync(d_col_idx, &csr_mat.col_indices[offset],
                               (size_t)local_nnz * sizeof(int), cudaMemcpyHostToDevice,
                               stream_compute));
    CUDA_CHECK(cudaMemcpyAsync(d_values, &csr_mat.values[offset],
                               (size_t)local_nnz * sizeof(double), cudaMemcpyHostToDevice,
                               stream_compute));

    free(local_row_ptr);

    if (config.verbose >= 1) {
        printf("[Rank %d] Local CSR: %d rows, %lld nnz (%.2f MB)\n", rank, n_local, local_nnz,
               (n_local * sizeof(long long) + (double)local_nnz * (sizeof(int) + sizeof(double))) /
                   1e6);
    }

    // Allocate vectors (local partition + halo for p only)
    double *d_x_local, *d_r_local, *d_p_local, *d_Ap, *d_b;
    double *d_p_halo_prev = NULL, *d_p_halo_next = NULL;

    CUDA_CHECK(cudaMalloc(&d_x_local, n_local * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_r_local, n_local * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_p_local, n_local * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Ap, n_local * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_b, n_local * sizeof(double)));

    if (rank > 0) {
        CUDA_CHECK(cudaMalloc(&d_p_halo_prev, grid_size * sizeof(double)));
    }
    if (rank < world_size - 1) {
        CUDA_CHECK(cudaMalloc(&d_p_halo_next, grid_size * sizeof(double)));
    }

    // Pinned host buffers for MPI staging
    double *h_send_prev = NULL, *h_send_next = NULL;
    double *h_recv_prev = NULL, *h_recv_next = NULL;
    if (rank > 0) {
        CUDA_CHECK(cudaMallocHost(&h_send_prev, grid_size * sizeof(double)));
        CUDA_CHECK(cudaMallocHost(&h_recv_prev, grid_size * sizeof(double)));
    }
    if (rank < world_size - 1) {
        CUDA_CHECK(cudaMallocHost(&h_send_next, grid_size * sizeof(double)));
        CUDA_CHECK(cudaMallocHost(&h_recv_next, grid_size * sizeof(double)));
    }

    // Initialize vectors on stream_compute (non-blocking streams require explicit stream usage)
    CUDA_CHECK(cudaMemcpyAsync(d_b, &b[row_offset], n_local * sizeof(double),
                               cudaMemcpyHostToDevice, stream_compute));
    CUDA_CHECK(cudaMemcpyAsync(d_x_local, &x[row_offset], n_local * sizeof(double),
                               cudaMemcpyHostToDevice, stream_compute));
    CUDA_CHECK(cudaMemsetAsync(d_r_local, 0, n_local * sizeof(double), stream_compute));
    CUDA_CHECK(cudaMemsetAsync(d_p_local, 0, n_local * sizeof(double), stream_compute));
    CUDA_CHECK(cudaStreamSynchronize(stream_compute));

    MPI_Barrier(MPI_COMM_WORLD);

    // Timing events
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    cudaEvent_t timer_start, timer_stop;
    cudaEvent_t timer_comm_start, timer_comm_stop;
    cudaEvent_t timer_phase_start, timer_phase_stop;
    if (config.enable_detailed_timers) {
        CUDA_CHECK(cudaEventCreate(&timer_start));
        CUDA_CHECK(cudaEventCreate(&timer_stop));
        CUDA_CHECK(cudaEventCreate(&timer_comm_start));
        CUDA_CHECK(cudaEventCreate(&timer_comm_stop));
        CUDA_CHECK(cudaEventCreate(&timer_phase_start));
        CUDA_CHECK(cudaEventCreate(&timer_phase_stop));
    }

    CUDA_CHECK(cudaEventRecord(start, stream_compute));

    // Initialize CG statistics
    stats->time_spmv_ms = 0.0;
    stats->time_blas1_ms = 0.0;
    stats->time_reductions_ms = 0.0;
    stats->time_allreduce_ms = 0.0;
    stats->time_allgather_ms = 0.0;
    stats->time_spmv_interior_ms = 0.0;
    stats->time_spmv_boundary_ms = 0.0;
    stats->time_comm_total_ms = 0.0;
    stats->time_comm_hidden_ms = 0.0;
    stats->time_comm_exposed_ms = 0.0;
    stats->overlap_efficiency = 0.0;

    if (rank == 0 && config.verbose >= 1) {
        printf("\nStarting CG iterations (overlap mode)...\n");
    }

    int threads = 256;
    int blocks_local = (n_local + threads - 1) / threads;

    // === Initial residual: r = b - A*x0 ===

    // Exchange x halo (synchronous, one-time)
    double *d_x_halo_prev = NULL, *d_x_halo_next = NULL;
    if (rank > 0) {
        CUDA_CHECK(cudaMalloc(&d_x_halo_prev, grid_size * sizeof(double)));
    }
    if (rank < world_size - 1) {
        CUDA_CHECK(cudaMalloc(&d_x_halo_next, grid_size * sizeof(double)));
    }

    exchange_halo_sync(d_x_local, d_x_local + (n_local - grid_size), d_x_halo_prev, d_x_halo_next,
                       h_send_prev, h_send_next, h_recv_prev, h_recv_next, grid_size, rank,
                       world_size, stream_compute);

    // Ap = A*x0 (full kernel, synchronous)
    stencil5_csr_partitioned_halo_kernel<<<blocks_local, threads, 0, stream_compute>>>(
        d_row_ptr, d_col_idx, d_values, d_x_local, d_x_halo_prev, d_x_halo_next, d_Ap, n_local,
        row_offset, n, grid_size);

    // r = b - Ap (reuse d_b as temporary: d_b = d_b - Ap, then copy to r)
    axpy_kernel<<<blocks_local, threads, 0, stream_compute>>>(-1.0, d_Ap, d_b, n_local);
    CUDA_CHECK(cudaMemcpyAsync(d_r_local, d_b, n_local * sizeof(double), cudaMemcpyDeviceToDevice,
                               stream_compute));

    // p = r (local only; halo will be exchanged at start of first iteration)
    CUDA_CHECK(cudaMemcpyAsync(d_p_local, d_r_local, n_local * sizeof(double),
                               cudaMemcpyDeviceToDevice, stream_compute));
    // Signal initial p is ready for first iteration's D2H on stream_comm
    CUDA_CHECK(cudaEventRecord(p_updated_event, stream_compute));
    CUDA_CHECK(cudaStreamSynchronize(stream_compute));

    // rs_old = dot(r, r) + AllReduce
    double rs_local_old = overlap_compute_local_dot(cublas_handle, d_r_local, d_r_local, n_local);
    double rs_old;
    MPI_Allreduce(&rs_local_old, &rs_old, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    double b_norm = sqrt(rs_old);

    if (rank == 0 && config.verbose >= 2) {
        printf("[Iter   0] Residual: %.6e\n", sqrt(rs_old));
    }

    // Free x halo (no longer needed)
    if (d_x_halo_prev)
        cudaFree(d_x_halo_prev);
    if (d_x_halo_next)
        cudaFree(d_x_halo_next);

    // Overlap timing accumulators
    double cum_interior_ms = 0.0;
    double cum_boundary_ms = 0.0;
    double cum_comm_ms = 0.0;
    double cum_overlap_phase_ms = 0.0;

    // === CG iteration loop ===
    nvtxRangePush("CG_Solver_Overlap");
    int iter;
    for (iter = 0; iter < config.max_iters; iter++) {
        nvtxRangePush("CG_Iteration_Overlap");

        // ============================================================
        // Overlapped SpMV: Ap = A * p
        // ============================================================
        nvtxRangePush("SpMV_Overlap");

        if (config.enable_detailed_timers) {
            CUDA_CHECK(cudaEventRecord(timer_phase_start, stream_comm));
        }

        // Ensure p update (axpby on stream_compute) is done before D2H reads
        CUDA_CHECK(cudaStreamWaitEvent(stream_comm, p_updated_event, 0));

        // D2H halo copies (async on stream_comm, returns immediately)
        exchange_halo_d2h_start(d_p_local, n_local, h_send_prev, h_send_next, partition.halo_elems,
                                rank, world_size, stream_comm);

        // Interior SpMV on stream_compute (overlaps with D2H + MPI)
        if (config.enable_detailed_timers) {
            CUDA_CHECK(cudaEventRecord(timer_start, stream_compute));
        }
        if (partition.interior_count > 0) {
            int blocks_interior = (partition.interior_count + threads - 1) / threads;
            stencil5_overlap_subrange_kernel<<<blocks_interior, threads, 0, stream_compute>>>(
                d_row_ptr, d_col_idx, d_values, d_p_local, d_p_halo_prev, d_p_halo_next, d_Ap,
                n_local, row_offset, n, grid_size, partition.interior_start,
                partition.interior_count);
        }
        if (config.enable_detailed_timers) {
            CUDA_CHECK(cudaEventRecord(timer_stop, stream_compute));
        }

        // Sync D2H + MPI non-blocking sends/receives
        MPI_Request requests[4];
        int req_count = 0;
        exchange_halo_mpi_start(h_send_prev, h_send_next, h_recv_prev, h_recv_next,
                                partition.halo_elems, rank, world_size, stream_comm, requests,
                                &req_count);

        // Finish halo exchange: MPI_Waitall + H2D
        exchange_halo_async_finish(d_p_halo_prev, d_p_halo_next, h_recv_prev, h_recv_next,
                                   partition.halo_elems, rank, world_size, stream_comm, requests,
                                   req_count);

        // Sync stream_compute (interior SpMV done)
        CUDA_CHECK(cudaStreamSynchronize(stream_compute));

        // Measure interior SpMV time
        if (config.enable_detailed_timers) {
            float interior_ms;
            CUDA_CHECK(cudaEventElapsedTime(&interior_ms, timer_start, timer_stop));
            cum_interior_ms += interior_ms;
        }

        // Step i: Boundary SpMV on stream_compute
        if (config.enable_detailed_timers) {
            CUDA_CHECK(cudaEventRecord(timer_start, stream_compute));
        }
        if (partition.boundary_prev_count > 0) {
            int blocks_prev = (partition.boundary_prev_count + threads - 1) / threads;
            stencil5_overlap_subrange_kernel<<<blocks_prev, threads, 0, stream_compute>>>(
                d_row_ptr, d_col_idx, d_values, d_p_local, d_p_halo_prev, d_p_halo_next, d_Ap,
                n_local, row_offset, n, grid_size, partition.boundary_prev_start,
                partition.boundary_prev_count);
        }
        if (partition.boundary_next_count > 0) {
            int blocks_next = (partition.boundary_next_count + threads - 1) / threads;
            stencil5_overlap_subrange_kernel<<<blocks_next, threads, 0, stream_compute>>>(
                d_row_ptr, d_col_idx, d_values, d_p_local, d_p_halo_prev, d_p_halo_next, d_Ap,
                n_local, row_offset, n, grid_size, partition.boundary_next_start,
                partition.boundary_next_count);
        }
        if (config.enable_detailed_timers) {
            CUDA_CHECK(cudaEventRecord(timer_stop, stream_compute));
            CUDA_CHECK(cudaEventRecord(timer_phase_stop, stream_compute));
            CUDA_CHECK(cudaStreamSynchronize(stream_compute));

            float boundary_ms;
            CUDA_CHECK(cudaEventElapsedTime(&boundary_ms, timer_start, timer_stop));
            cum_boundary_ms += boundary_ms;

            float phase_ms;
            CUDA_CHECK(cudaEventElapsedTime(&phase_ms, timer_phase_start, timer_phase_stop));
            cum_overlap_phase_ms += phase_ms;
        }

        nvtxRangePop();  // SpMV_Overlap

        // ============================================================
        // Standard CG operations (on stream_compute)
        // ============================================================

        // alpha = rs_old / (p^T * Ap)
        nvtxRangePush("Dot_Product");
        if (config.enable_detailed_timers) {
            CUDA_CHECK(cudaEventRecord(timer_start, stream_compute));
        }
        double pAp_local = overlap_compute_local_dot(cublas_handle, d_p_local, d_Ap, n_local);
        if (config.enable_detailed_timers) {
            CUDA_CHECK(cudaEventRecord(timer_stop, stream_compute));
            CUDA_CHECK(cudaEventSynchronize(timer_stop));
            float elapsed_ms;
            CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, timer_start, timer_stop));
            stats->time_reductions_ms += elapsed_ms;
            stats->time_dot_pAp_ms += elapsed_ms;
        }
        nvtxRangePop();

        if (config.enable_detailed_timers) {
            CUDA_CHECK(cudaEventRecord(timer_start, stream_compute));
        }
        double pAp;
        MPI_Allreduce(&pAp_local, &pAp, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        if (config.enable_detailed_timers) {
            CUDA_CHECK(cudaEventRecord(timer_stop, stream_compute));
            CUDA_CHECK(cudaEventSynchronize(timer_stop));
            float elapsed_ms;
            CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, timer_start, timer_stop));
            stats->time_allreduce_ms += elapsed_ms;
        }

        double alpha = rs_old / pAp;

        // x = x + alpha * p
        nvtxRangePush("BLAS_AXPY");
        if (config.enable_detailed_timers) {
            CUDA_CHECK(cudaEventRecord(timer_start, stream_compute));
        }
        axpy_kernel<<<blocks_local, threads, 0, stream_compute>>>(alpha, d_p_local, d_x_local,
                                                                  n_local);
        if (config.enable_detailed_timers) {
            CUDA_CHECK(cudaEventRecord(timer_stop, stream_compute));
            CUDA_CHECK(cudaEventSynchronize(timer_stop));
            float elapsed_ms;
            CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, timer_start, timer_stop));
            stats->time_blas1_ms += elapsed_ms;
            stats->time_axpy_update_x_ms += elapsed_ms;
        }

        // r = r - alpha * Ap
        if (config.enable_detailed_timers) {
            CUDA_CHECK(cudaEventRecord(timer_start, stream_compute));
        }
        axpy_kernel<<<blocks_local, threads, 0, stream_compute>>>(-alpha, d_Ap, d_r_local, n_local);
        if (config.enable_detailed_timers) {
            CUDA_CHECK(cudaEventRecord(timer_stop, stream_compute));
            CUDA_CHECK(cudaEventSynchronize(timer_stop));
            float elapsed_ms;
            CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, timer_start, timer_stop));
            stats->time_blas1_ms += elapsed_ms;
            stats->time_axpy_update_r_ms += elapsed_ms;
        }
        nvtxRangePop();

        // rs_new = r^T * r
        nvtxRangePush("Dot_Product");
        if (config.enable_detailed_timers) {
            CUDA_CHECK(cudaEventRecord(timer_start, stream_compute));
        }
        double rs_local_new =
            overlap_compute_local_dot(cublas_handle, d_r_local, d_r_local, n_local);
        if (config.enable_detailed_timers) {
            CUDA_CHECK(cudaEventRecord(timer_stop, stream_compute));
            CUDA_CHECK(cudaEventSynchronize(timer_stop));
            float elapsed_ms;
            CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, timer_start, timer_stop));
            stats->time_reductions_ms += elapsed_ms;
            stats->time_dot_rs_new_ms += elapsed_ms;
        }
        nvtxRangePop();

        if (config.enable_detailed_timers) {
            CUDA_CHECK(cudaEventRecord(timer_start, stream_compute));
        }
        double rs_new;
        MPI_Allreduce(&rs_local_new, &rs_new, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        if (config.enable_detailed_timers) {
            CUDA_CHECK(cudaEventRecord(timer_stop, stream_compute));
            CUDA_CHECK(cudaEventSynchronize(timer_stop));
            float elapsed_ms;
            CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, timer_start, timer_stop));
            stats->time_allreduce_ms += elapsed_ms;
        }

        double residual_norm = sqrt(rs_new);
        double rel_residual = residual_norm / b_norm;

        if (rank == 0 && config.verbose >= 2) {
            printf("[Iter %3d] Residual: %.6e (rel: %.6e, alpha: %.4e)\n", iter + 1, residual_norm,
                   rel_residual, alpha);
        }

        // Check convergence
        if (rel_residual < config.tolerance) {
            iter++;
            stats->converged = 1;
            stats->iterations = iter;
            stats->residual_norm = residual_norm;
            nvtxRangePop();  // CG_Iteration_Overlap
            break;
        }

        // beta = rs_new / rs_old
        double beta = rs_new / rs_old;

        // p = r + beta * p
        nvtxRangePush("BLAS_AXPBY");
        if (config.enable_detailed_timers) {
            CUDA_CHECK(cudaEventRecord(timer_start, stream_compute));
        }
        axpby_kernel<<<blocks_local, threads, 0, stream_compute>>>(1.0, d_r_local, beta, d_p_local,
                                                                   n_local);
        if (config.enable_detailed_timers) {
            CUDA_CHECK(cudaEventRecord(timer_stop, stream_compute));
            CUDA_CHECK(cudaEventSynchronize(timer_stop));
            float elapsed_ms;
            CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, timer_start, timer_stop));
            stats->time_blas1_ms += elapsed_ms;
            stats->time_axpby_update_p_ms += elapsed_ms;
        }
        nvtxRangePop();

        // Signal that p update is done (stream_comm must wait before D2H reads)
        CUDA_CHECK(cudaEventRecord(p_updated_event, stream_compute));

        rs_old = rs_new;
        nvtxRangePop();  // CG_Iteration_Overlap
    }
    nvtxRangePop();  // CG_Solver_Overlap

    // Check max iterations
    if (iter == config.max_iters && rank == 0) {
        printf("\nMax iterations reached without convergence\n");
        stats->converged = 0;
        stats->iterations = iter;
        stats->residual_norm = sqrt(rs_old);
    }

    // Stop timing
    CUDA_CHECK(cudaEventRecord(stop, stream_compute));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float time_ms;
    CUDA_CHECK(cudaEventElapsedTime(&time_ms, start, stop));

    // Compute overlap metrics
    if (config.enable_detailed_timers && stats->iterations > 0) {
        stats->time_spmv_interior_ms = cum_interior_ms;
        stats->time_spmv_boundary_ms = cum_boundary_ms;
        stats->time_spmv_ms = cum_interior_ms + cum_boundary_ms;

        double sequential_est = cum_interior_ms + cum_boundary_ms + cum_comm_ms;
        stats->time_comm_total_ms = cum_overlap_phase_ms - cum_interior_ms - cum_boundary_ms;
        if (stats->time_comm_total_ms < 0.0)
            stats->time_comm_total_ms = 0.0;

        stats->time_comm_hidden_ms = sequential_est - cum_overlap_phase_ms;
        if (stats->time_comm_hidden_ms < 0.0)
            stats->time_comm_hidden_ms = 0.0;

        stats->time_comm_exposed_ms = stats->time_comm_total_ms - stats->time_comm_hidden_ms;
        if (stats->time_comm_exposed_ms < 0.0)
            stats->time_comm_exposed_ms = 0.0;

        stats->overlap_efficiency = (stats->time_comm_total_ms > 0.0)
                                        ? stats->time_comm_hidden_ms / stats->time_comm_total_ms
                                        : 0.0;
        if (stats->overlap_efficiency > 1.0)
            stats->overlap_efficiency = 1.0;

        // Normalize per-iteration granular timers
        stats->time_dot_pAp_ms /= stats->iterations;
        stats->time_dot_rs_new_ms /= stats->iterations;
        stats->time_axpy_update_x_ms /= stats->iterations;
        stats->time_axpy_update_r_ms /= stats->iterations;
        stats->time_axpby_update_p_ms /= stats->iterations;
    }

    stats->time_total_ms = time_ms;

    // MPI stats aggregation
    if (world_size > 1) {
        double local_times[6], max_times[6], min_times[6];

        local_times[0] = stats->time_total_ms;
        local_times[1] = stats->time_spmv_ms;
        local_times[2] = stats->time_blas1_ms;
        local_times[3] = stats->time_reductions_ms;
        local_times[4] = stats->time_allreduce_ms;
        local_times[5] = stats->time_allgather_ms;

        MPI_Reduce(local_times, max_times, 6, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(local_times, min_times, 6, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            stats->time_total_ms = max_times[0];
            stats->time_spmv_ms = max_times[1];
            stats->time_blas1_ms = max_times[2];
            stats->time_reductions_ms = max_times[3];
            stats->time_allreduce_ms = max_times[4];
            stats->time_allgather_ms = max_times[5];

            double imbalance_pct = 100.0 * (max_times[0] - min_times[0]) / max_times[0];
            printf("Total time: %.2f ms (max), %.2f ms (min) - Load imbalance: %.1f%%\n",
                   max_times[0], min_times[0], imbalance_pct);
        }
    }

    // Print results
    if (rank == 0 && config.verbose >= 1) {
        printf("Total time: %.2f ms\n", stats->time_total_ms);
        if (config.enable_detailed_timers) {
            printf("\nOverlap Metrics (%d iterations):\n", stats->iterations);
            printf("  Interior SpMV: %.2f ms (%.3f ms/iter)\n", stats->time_spmv_interior_ms,
                   stats->time_spmv_interior_ms / stats->iterations);
            printf("  Boundary SpMV: %.2f ms (%.3f ms/iter)\n", stats->time_spmv_boundary_ms,
                   stats->time_spmv_boundary_ms / stats->iterations);
            printf("  Comm total:    %.2f ms (%.3f ms/iter)\n", stats->time_comm_total_ms,
                   stats->time_comm_total_ms / stats->iterations);
            printf("  Comm hidden:   %.2f ms\n", stats->time_comm_hidden_ms);
            printf("  Comm exposed:  %.2f ms\n", stats->time_comm_exposed_ms);
            printf("  Overlap eff:   %.1f%%\n", stats->overlap_efficiency * 100.0);
        }
        printf("========================================\n");
    }

    // Copy result back (sync stream_compute first to ensure all work is done)
    CUDA_CHECK(cudaStreamSynchronize(stream_compute));
    CUDA_CHECK(cudaMemcpyAsync(&x[row_offset], d_x_local, n_local * sizeof(double),
                               cudaMemcpyDeviceToHost, stream_compute));

    // Gather full solution to rank 0
    int* recvcounts = NULL;
    int* displs = NULL;
    if (rank == 0) {
        recvcounts = (int*)malloc(world_size * sizeof(int));
        displs = (int*)malloc(world_size * sizeof(int));
        int base_size = n / world_size;
        for (int i = 0; i < world_size; i++) {
            displs[i] = i * base_size;
            recvcounts[i] = (i == world_size - 1) ? (n - displs[i]) : base_size;
        }
    }
    MPI_Gatherv(&x[row_offset], n_local, MPI_DOUBLE, x, recvcounts, displs, MPI_DOUBLE, 0,
                MPI_COMM_WORLD);
    if (rank == 0) {
        free(recvcounts);
        free(displs);
    }

    // Solution validation checksums
    if (rank == 0) {
        double sol_sum = 0.0, sol_norm_sq = 0.0;
        for (int i = 0; i < n; i++) {
            sol_sum += x[i];
            sol_norm_sq += x[i] * x[i];
        }
        stats->solution_sum = sol_sum;
        stats->solution_norm = sqrt(sol_norm_sq);
    }

    // Cleanup
    cudaFree(d_x_local);
    cudaFree(d_r_local);
    cudaFree(d_p_local);
    cudaFree(d_Ap);
    cudaFree(d_b);
    cudaFree(d_row_ptr);
    cudaFree(d_col_idx);
    cudaFree(d_values);

    if (d_p_halo_prev)
        cudaFree(d_p_halo_prev);
    if (d_p_halo_next)
        cudaFree(d_p_halo_next);

    if (h_send_prev)
        cudaFreeHost(h_send_prev);
    if (h_send_next)
        cudaFreeHost(h_send_next);
    if (h_recv_prev)
        cudaFreeHost(h_recv_prev);
    if (h_recv_next)
        cudaFreeHost(h_recv_next);

    cublasDestroy(cublas_handle);
    cudaEventDestroy(p_updated_event);
    cudaStreamDestroy(stream_compute);
    cudaStreamDestroy(stream_comm);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    if (config.enable_detailed_timers) {
        cudaEventDestroy(timer_start);
        cudaEventDestroy(timer_stop);
        cudaEventDestroy(timer_comm_start);
        cudaEventDestroy(timer_comm_stop);
        cudaEventDestroy(timer_phase_start);
        cudaEventDestroy(timer_phase_stop);
    }

    return 0;
}

/* ================================================================
 * 3D Multi-GPU CG solver with compute-communication overlap
 * ================================================================ */

int cg_solve_mgpu_partitioned_overlap_3d(SpmvOperator* spmv_op, MatrixData* mat, const double* b,
                                         double* x, CGConfigMultiGPU config,
                                         CGStatsMultiGPU* stats) {

    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int n = mat->rows;
    int grid_size = mat->grid_size;

    if (rank == 0 && config.verbose >= 1) {
        printf("\n========================================\n");
        printf("Multi-GPU CG Solver (3D OVERLAP)\n");
        printf("========================================\n");
        printf("MPI ranks: %d\n", world_size);
        printf("Problem size: %d unknowns (%d³ grid)\n", n, grid_size);
        printf("Max iterations: %d\n", config.max_iters);
        printf("Tolerance: %.1e\n", config.tolerance);
        printf("========================================\n\n");
    }

    CUDA_CHECK(cudaSetDevice(rank));

    int n_local = n / world_size;
    int row_offset = rank * n_local;
    if (rank == world_size - 1) {
        n_local = n - row_offset;
    }

    if (config.verbose >= 1) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, rank));
        printf("[Rank %d] GPU %d: %s (CC %d.%d)\n", rank, rank, prop.name, prop.major, prop.minor);
        printf("[Rank %d] Rows: [%d:%d) (%d rows, %d Z-planes)\n", rank, row_offset,
               row_offset + n_local, n_local, n_local / (grid_size * grid_size));
    }

    // 3D overlap partition
    OverlapPartition partition;
    compute_overlap_partition(&partition, n_local, grid_size, rank, world_size, 3);

    if (config.verbose >= 1) {
        printf("[Rank %d] Overlap partition: interior=[%d:%d) (%d rows), "
               "boundary_prev=%d rows, boundary_next=%d rows, halo_elems=%d\n",
               rank, partition.interior_start, partition.interior_start + partition.interior_count,
               partition.interior_count, partition.boundary_prev_count,
               partition.boundary_next_count, partition.halo_elems);
    }

    // Two CUDA streams for overlap
    cudaStream_t stream_compute, stream_comm;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_compute, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_comm, cudaStreamNonBlocking));

    cudaEvent_t p_updated_event;
    CUDA_CHECK(cudaEventCreate(&p_updated_event));

    cublasHandle_t cublas_handle;
    cublasStatus_t cublas_status = cublasCreate(&cublas_handle);
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS initialization failed\n");
        exit(EXIT_FAILURE);
    }
    cublasSetStream(cublas_handle, stream_compute);

    // Build local CSR partition
    if (rank == 0 && config.verbose >= 1) {
        printf("Building local CSR partitions...\n");
    }

    build_csr_struct(mat);

    long long local_nnz = csr_mat.row_ptr[row_offset + n_local] - csr_mat.row_ptr[row_offset];

    long long* d_row_ptr;
    int* d_col_idx;
    double* d_values;

    CUDA_CHECK(cudaMalloc(&d_row_ptr, (n_local + 1) * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_col_idx, (size_t)local_nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_values, (size_t)local_nnz * sizeof(double)));

    long long* local_row_ptr = (long long*)malloc((n_local + 1) * sizeof(long long));
    long long offset = csr_mat.row_ptr[row_offset];
    for (int i = 0; i <= n_local; i++) {
        local_row_ptr[i] = csr_mat.row_ptr[row_offset + i] - offset;
    }

    CUDA_CHECK(cudaMemcpyAsync(d_row_ptr, local_row_ptr, (n_local + 1) * sizeof(long long),
                               cudaMemcpyHostToDevice, stream_compute));
    CUDA_CHECK(cudaMemcpyAsync(d_col_idx, &csr_mat.col_indices[offset],
                               (size_t)local_nnz * sizeof(int), cudaMemcpyHostToDevice,
                               stream_compute));
    CUDA_CHECK(cudaMemcpyAsync(d_values, &csr_mat.values[offset],
                               (size_t)local_nnz * sizeof(double), cudaMemcpyHostToDevice,
                               stream_compute));

    free(local_row_ptr);

    if (config.verbose >= 1) {
        printf("[Rank %d] Local CSR: %d rows, %lld nnz (%.2f MB)\n", rank, n_local, local_nnz,
               (n_local * sizeof(long long) + (double)local_nnz * (sizeof(int) + sizeof(double))) /
                   1e6);
    }

    // Allocate vectors
    double *d_x_local, *d_r_local, *d_p_local, *d_Ap, *d_b;
    double *d_p_halo_prev = NULL, *d_p_halo_next = NULL;

    CUDA_CHECK(cudaMalloc(&d_x_local, n_local * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_r_local, n_local * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_p_local, n_local * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Ap, n_local * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_b, n_local * sizeof(double)));

    // Halo buffers: N² per direction
    if (rank > 0) {
        CUDA_CHECK(cudaMalloc(&d_p_halo_prev, partition.halo_elems * sizeof(double)));
    }
    if (rank < world_size - 1) {
        CUDA_CHECK(cudaMalloc(&d_p_halo_next, partition.halo_elems * sizeof(double)));
    }

    // Pinned host buffers
    double *h_send_prev = NULL, *h_send_next = NULL;
    double *h_recv_prev = NULL, *h_recv_next = NULL;
    if (rank > 0) {
        CUDA_CHECK(cudaMallocHost(&h_send_prev, partition.halo_elems * sizeof(double)));
        CUDA_CHECK(cudaMallocHost(&h_recv_prev, partition.halo_elems * sizeof(double)));
    }
    if (rank < world_size - 1) {
        CUDA_CHECK(cudaMallocHost(&h_send_next, partition.halo_elems * sizeof(double)));
        CUDA_CHECK(cudaMallocHost(&h_recv_next, partition.halo_elems * sizeof(double)));
    }

    // Initialize vectors
    CUDA_CHECK(cudaMemcpyAsync(d_b, &b[row_offset], n_local * sizeof(double),
                               cudaMemcpyHostToDevice, stream_compute));
    CUDA_CHECK(cudaMemcpyAsync(d_x_local, &x[row_offset], n_local * sizeof(double),
                               cudaMemcpyHostToDevice, stream_compute));
    CUDA_CHECK(cudaMemsetAsync(d_r_local, 0, n_local * sizeof(double), stream_compute));
    CUDA_CHECK(cudaMemsetAsync(d_p_local, 0, n_local * sizeof(double), stream_compute));
    CUDA_CHECK(cudaStreamSynchronize(stream_compute));

    MPI_Barrier(MPI_COMM_WORLD);

    // Timing events
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    cudaEvent_t timer_start, timer_stop;
    cudaEvent_t timer_comm_start, timer_comm_stop;
    cudaEvent_t timer_phase_start, timer_phase_stop;
    if (config.enable_detailed_timers) {
        CUDA_CHECK(cudaEventCreate(&timer_start));
        CUDA_CHECK(cudaEventCreate(&timer_stop));
        CUDA_CHECK(cudaEventCreate(&timer_comm_start));
        CUDA_CHECK(cudaEventCreate(&timer_comm_stop));
        CUDA_CHECK(cudaEventCreate(&timer_phase_start));
        CUDA_CHECK(cudaEventCreate(&timer_phase_stop));
    }

    CUDA_CHECK(cudaEventRecord(start, stream_compute));

    // Initialize stats
    stats->time_spmv_ms = 0.0;
    stats->time_blas1_ms = 0.0;
    stats->time_reductions_ms = 0.0;
    stats->time_allreduce_ms = 0.0;
    stats->time_allgather_ms = 0.0;
    stats->time_spmv_interior_ms = 0.0;
    stats->time_spmv_boundary_ms = 0.0;
    stats->time_comm_total_ms = 0.0;
    stats->time_comm_hidden_ms = 0.0;
    stats->time_comm_exposed_ms = 0.0;
    stats->overlap_efficiency = 0.0;

    if (rank == 0 && config.verbose >= 1) {
        printf("\nStarting CG iterations (3D overlap mode)...\n");
    }

    int threads = 256;
    int blocks_local = (n_local + threads - 1) / threads;

    // Initial residual: r = b - A*x0
    double *d_x_halo_prev = NULL, *d_x_halo_next = NULL;
    if (rank > 0) {
        CUDA_CHECK(cudaMalloc(&d_x_halo_prev, partition.halo_elems * sizeof(double)));
    }
    if (rank < world_size - 1) {
        CUDA_CHECK(cudaMalloc(&d_x_halo_next, partition.halo_elems * sizeof(double)));
    }

    exchange_halo_sync(d_x_local, d_x_local + (n_local - partition.halo_elems), d_x_halo_prev,
                       d_x_halo_next, h_send_prev, h_send_next, h_recv_prev, h_recv_next,
                       partition.halo_elems, rank, world_size, stream_compute);

    // Ap = A*x0 (full 3D kernel)
    stencil7_csr_partitioned_halo_kernel_3d<<<blocks_local, threads, 0, stream_compute>>>(
        d_row_ptr, d_col_idx, d_values, d_x_local, d_x_halo_prev, d_x_halo_next, d_Ap, n_local,
        row_offset, n, grid_size);

    // r = b - Ap
    axpy_kernel<<<blocks_local, threads, 0, stream_compute>>>(-1.0, d_Ap, d_b, n_local);
    CUDA_CHECK(cudaMemcpyAsync(d_r_local, d_b, n_local * sizeof(double), cudaMemcpyDeviceToDevice,
                               stream_compute));

    // p = r
    CUDA_CHECK(cudaMemcpyAsync(d_p_local, d_r_local, n_local * sizeof(double),
                               cudaMemcpyDeviceToDevice, stream_compute));
    CUDA_CHECK(cudaEventRecord(p_updated_event, stream_compute));
    CUDA_CHECK(cudaStreamSynchronize(stream_compute));

    // rs_old
    double rs_local_old = overlap_compute_local_dot(cublas_handle, d_r_local, d_r_local, n_local);
    double rs_old;
    MPI_Allreduce(&rs_local_old, &rs_old, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    double b_norm = sqrt(rs_old);

    if (rank == 0 && config.verbose >= 2) {
        printf("[Iter   0] Residual: %.6e\n", sqrt(rs_old));
    }

    if (d_x_halo_prev)
        cudaFree(d_x_halo_prev);
    if (d_x_halo_next)
        cudaFree(d_x_halo_next);

    // Overlap timing accumulators
    double cum_interior_ms = 0.0;
    double cum_boundary_ms = 0.0;
    double cum_comm_ms = 0.0;
    double cum_overlap_phase_ms = 0.0;

    // CG iteration loop
    nvtxRangePush("CG_Solver_3D_Overlap");
    int iter;
    for (iter = 0; iter < config.max_iters; iter++) {
        nvtxRangePush("CG_Iteration_3D_Overlap");

        // Overlapped SpMV: Ap = A * p
        nvtxRangePush("SpMV_3D_Overlap");

        if (config.enable_detailed_timers) {
            CUDA_CHECK(cudaEventRecord(timer_phase_start, stream_comm));
        }

        CUDA_CHECK(cudaStreamWaitEvent(stream_comm, p_updated_event, 0));

        // D2H halo copies (async on stream_comm, returns immediately)
        exchange_halo_d2h_start(d_p_local, n_local, h_send_prev, h_send_next, partition.halo_elems,
                                rank, world_size, stream_comm);

        // Interior SpMV on stream_compute (overlaps with D2H + MPI)
        if (config.enable_detailed_timers) {
            CUDA_CHECK(cudaEventRecord(timer_start, stream_compute));
        }
        if (partition.interior_count > 0) {
            int blocks_interior = (partition.interior_count + threads - 1) / threads;
            stencil7_overlap_subrange_kernel_3d<<<blocks_interior, threads, 0, stream_compute>>>(
                d_row_ptr, d_col_idx, d_values, d_p_local, d_p_halo_prev, d_p_halo_next, d_Ap,
                n_local, row_offset, n, grid_size, partition.interior_start,
                partition.interior_count);
        }
        if (config.enable_detailed_timers) {
            CUDA_CHECK(cudaEventRecord(timer_stop, stream_compute));
        }

        // Sync D2H + MPI non-blocking sends/receives
        MPI_Request requests[4];
        int req_count = 0;
        exchange_halo_mpi_start(h_send_prev, h_send_next, h_recv_prev, h_recv_next,
                                partition.halo_elems, rank, world_size, stream_comm, requests,
                                &req_count);

        // Finish halo exchange: MPI_Waitall + H2D
        exchange_halo_async_finish(d_p_halo_prev, d_p_halo_next, h_recv_prev, h_recv_next,
                                   partition.halo_elems, rank, world_size, stream_comm, requests,
                                   req_count);

        CUDA_CHECK(cudaStreamSynchronize(stream_compute));

        if (config.enable_detailed_timers) {
            float interior_ms;
            CUDA_CHECK(cudaEventElapsedTime(&interior_ms, timer_start, timer_stop));
            cum_interior_ms += interior_ms;
        }

        // Boundary SpMV
        if (config.enable_detailed_timers) {
            CUDA_CHECK(cudaEventRecord(timer_start, stream_compute));
        }
        if (partition.boundary_prev_count > 0) {
            int blocks_prev = (partition.boundary_prev_count + threads - 1) / threads;
            stencil7_overlap_subrange_kernel_3d<<<blocks_prev, threads, 0, stream_compute>>>(
                d_row_ptr, d_col_idx, d_values, d_p_local, d_p_halo_prev, d_p_halo_next, d_Ap,
                n_local, row_offset, n, grid_size, partition.boundary_prev_start,
                partition.boundary_prev_count);
        }
        if (partition.boundary_next_count > 0) {
            int blocks_next = (partition.boundary_next_count + threads - 1) / threads;
            stencil7_overlap_subrange_kernel_3d<<<blocks_next, threads, 0, stream_compute>>>(
                d_row_ptr, d_col_idx, d_values, d_p_local, d_p_halo_prev, d_p_halo_next, d_Ap,
                n_local, row_offset, n, grid_size, partition.boundary_next_start,
                partition.boundary_next_count);
        }
        if (config.enable_detailed_timers) {
            CUDA_CHECK(cudaEventRecord(timer_stop, stream_compute));
            CUDA_CHECK(cudaEventRecord(timer_phase_stop, stream_compute));
            CUDA_CHECK(cudaStreamSynchronize(stream_compute));

            float boundary_ms;
            CUDA_CHECK(cudaEventElapsedTime(&boundary_ms, timer_start, timer_stop));
            cum_boundary_ms += boundary_ms;

            float phase_ms;
            CUDA_CHECK(cudaEventElapsedTime(&phase_ms, timer_phase_start, timer_phase_stop));
            cum_overlap_phase_ms += phase_ms;
        }

        nvtxRangePop();  // SpMV_3D_Overlap

        // Standard CG operations
        nvtxRangePush("Dot_Product");
        if (config.enable_detailed_timers) {
            CUDA_CHECK(cudaEventRecord(timer_start, stream_compute));
        }
        double pAp_local = overlap_compute_local_dot(cublas_handle, d_p_local, d_Ap, n_local);
        if (config.enable_detailed_timers) {
            CUDA_CHECK(cudaEventRecord(timer_stop, stream_compute));
            CUDA_CHECK(cudaEventSynchronize(timer_stop));
            float elapsed_ms;
            CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, timer_start, timer_stop));
            stats->time_reductions_ms += elapsed_ms;
            stats->time_dot_pAp_ms += elapsed_ms;
        }
        nvtxRangePop();

        if (config.enable_detailed_timers) {
            CUDA_CHECK(cudaEventRecord(timer_start, stream_compute));
        }
        double pAp;
        MPI_Allreduce(&pAp_local, &pAp, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        if (config.enable_detailed_timers) {
            CUDA_CHECK(cudaEventRecord(timer_stop, stream_compute));
            CUDA_CHECK(cudaEventSynchronize(timer_stop));
            float elapsed_ms;
            CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, timer_start, timer_stop));
            stats->time_allreduce_ms += elapsed_ms;
        }

        double alpha = rs_old / pAp;

        // x = x + alpha * p
        nvtxRangePush("BLAS_AXPY");
        if (config.enable_detailed_timers) {
            CUDA_CHECK(cudaEventRecord(timer_start, stream_compute));
        }
        axpy_kernel<<<blocks_local, threads, 0, stream_compute>>>(alpha, d_p_local, d_x_local,
                                                                  n_local);
        if (config.enable_detailed_timers) {
            CUDA_CHECK(cudaEventRecord(timer_stop, stream_compute));
            CUDA_CHECK(cudaEventSynchronize(timer_stop));
            float elapsed_ms;
            CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, timer_start, timer_stop));
            stats->time_blas1_ms += elapsed_ms;
            stats->time_axpy_update_x_ms += elapsed_ms;
        }

        // r = r - alpha * Ap
        if (config.enable_detailed_timers) {
            CUDA_CHECK(cudaEventRecord(timer_start, stream_compute));
        }
        axpy_kernel<<<blocks_local, threads, 0, stream_compute>>>(-alpha, d_Ap, d_r_local, n_local);
        if (config.enable_detailed_timers) {
            CUDA_CHECK(cudaEventRecord(timer_stop, stream_compute));
            CUDA_CHECK(cudaEventSynchronize(timer_stop));
            float elapsed_ms;
            CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, timer_start, timer_stop));
            stats->time_blas1_ms += elapsed_ms;
            stats->time_axpy_update_r_ms += elapsed_ms;
        }
        nvtxRangePop();

        // rs_new
        nvtxRangePush("Dot_Product");
        if (config.enable_detailed_timers) {
            CUDA_CHECK(cudaEventRecord(timer_start, stream_compute));
        }
        double rs_local_new =
            overlap_compute_local_dot(cublas_handle, d_r_local, d_r_local, n_local);
        if (config.enable_detailed_timers) {
            CUDA_CHECK(cudaEventRecord(timer_stop, stream_compute));
            CUDA_CHECK(cudaEventSynchronize(timer_stop));
            float elapsed_ms;
            CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, timer_start, timer_stop));
            stats->time_reductions_ms += elapsed_ms;
            stats->time_dot_rs_new_ms += elapsed_ms;
        }
        nvtxRangePop();

        if (config.enable_detailed_timers) {
            CUDA_CHECK(cudaEventRecord(timer_start, stream_compute));
        }
        double rs_new;
        MPI_Allreduce(&rs_local_new, &rs_new, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        if (config.enable_detailed_timers) {
            CUDA_CHECK(cudaEventRecord(timer_stop, stream_compute));
            CUDA_CHECK(cudaEventSynchronize(timer_stop));
            float elapsed_ms;
            CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, timer_start, timer_stop));
            stats->time_allreduce_ms += elapsed_ms;
        }

        double residual_norm = sqrt(rs_new);
        double rel_residual = residual_norm / b_norm;

        if (rank == 0 && config.verbose >= 2) {
            printf("[Iter %3d] Residual: %.6e (rel: %.6e, alpha: %.4e)\n", iter + 1, residual_norm,
                   rel_residual, alpha);
        }

        if (rel_residual < config.tolerance) {
            iter++;
            stats->converged = 1;
            stats->iterations = iter;
            stats->residual_norm = residual_norm;
            nvtxRangePop();
            break;
        }

        double beta = rs_new / rs_old;

        // p = r + beta * p
        nvtxRangePush("BLAS_AXPBY");
        if (config.enable_detailed_timers) {
            CUDA_CHECK(cudaEventRecord(timer_start, stream_compute));
        }
        axpby_kernel<<<blocks_local, threads, 0, stream_compute>>>(1.0, d_r_local, beta, d_p_local,
                                                                   n_local);
        if (config.enable_detailed_timers) {
            CUDA_CHECK(cudaEventRecord(timer_stop, stream_compute));
            CUDA_CHECK(cudaEventSynchronize(timer_stop));
            float elapsed_ms;
            CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, timer_start, timer_stop));
            stats->time_blas1_ms += elapsed_ms;
            stats->time_axpby_update_p_ms += elapsed_ms;
        }
        nvtxRangePop();

        CUDA_CHECK(cudaEventRecord(p_updated_event, stream_compute));

        rs_old = rs_new;
        nvtxRangePop();
    }
    nvtxRangePop();

    if (iter == config.max_iters && rank == 0) {
        printf("\nMax iterations reached without convergence\n");
        stats->converged = 0;
        stats->iterations = iter;
        stats->residual_norm = sqrt(rs_old);
    }

    // Stop timing
    CUDA_CHECK(cudaEventRecord(stop, stream_compute));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float time_ms;
    CUDA_CHECK(cudaEventElapsedTime(&time_ms, start, stop));

    // Overlap metrics
    if (config.enable_detailed_timers && stats->iterations > 0) {
        stats->time_spmv_interior_ms = cum_interior_ms;
        stats->time_spmv_boundary_ms = cum_boundary_ms;
        stats->time_spmv_ms = cum_interior_ms + cum_boundary_ms;

        double sequential_est = cum_interior_ms + cum_boundary_ms + cum_comm_ms;
        stats->time_comm_total_ms = cum_overlap_phase_ms - cum_interior_ms - cum_boundary_ms;
        if (stats->time_comm_total_ms < 0.0)
            stats->time_comm_total_ms = 0.0;

        stats->time_comm_hidden_ms = sequential_est - cum_overlap_phase_ms;
        if (stats->time_comm_hidden_ms < 0.0)
            stats->time_comm_hidden_ms = 0.0;

        stats->time_comm_exposed_ms = stats->time_comm_total_ms - stats->time_comm_hidden_ms;
        if (stats->time_comm_exposed_ms < 0.0)
            stats->time_comm_exposed_ms = 0.0;

        stats->overlap_efficiency = (stats->time_comm_total_ms > 0.0)
                                        ? stats->time_comm_hidden_ms / stats->time_comm_total_ms
                                        : 0.0;
        if (stats->overlap_efficiency > 1.0)
            stats->overlap_efficiency = 1.0;

        stats->time_dot_pAp_ms /= stats->iterations;
        stats->time_dot_rs_new_ms /= stats->iterations;
        stats->time_axpy_update_x_ms /= stats->iterations;
        stats->time_axpy_update_r_ms /= stats->iterations;
        stats->time_axpby_update_p_ms /= stats->iterations;
    }

    stats->time_total_ms = time_ms;

    // MPI stats aggregation
    if (world_size > 1) {
        double local_times[6], max_times[6], min_times[6];

        local_times[0] = stats->time_total_ms;
        local_times[1] = stats->time_spmv_ms;
        local_times[2] = stats->time_blas1_ms;
        local_times[3] = stats->time_reductions_ms;
        local_times[4] = stats->time_allreduce_ms;
        local_times[5] = stats->time_allgather_ms;

        MPI_Reduce(local_times, max_times, 6, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(local_times, min_times, 6, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            stats->time_total_ms = max_times[0];
            stats->time_spmv_ms = max_times[1];
            stats->time_blas1_ms = max_times[2];
            stats->time_reductions_ms = max_times[3];
            stats->time_allreduce_ms = max_times[4];
            stats->time_allgather_ms = max_times[5];

            double imbalance_pct = 100.0 * (max_times[0] - min_times[0]) / max_times[0];
            printf("Total time: %.2f ms (max), %.2f ms (min) - Load imbalance: %.1f%%\n",
                   max_times[0], min_times[0], imbalance_pct);
        }
    }

    if (rank == 0 && config.verbose >= 1) {
        printf("Total time: %.2f ms\n", stats->time_total_ms);
        if (config.enable_detailed_timers) {
            printf("\nOverlap Metrics (%d iterations):\n", stats->iterations);
            printf("  Interior SpMV: %.2f ms (%.3f ms/iter)\n", stats->time_spmv_interior_ms,
                   stats->time_spmv_interior_ms / stats->iterations);
            printf("  Boundary SpMV: %.2f ms (%.3f ms/iter)\n", stats->time_spmv_boundary_ms,
                   stats->time_spmv_boundary_ms / stats->iterations);
            printf("  Comm total:    %.2f ms (%.3f ms/iter)\n", stats->time_comm_total_ms,
                   stats->time_comm_total_ms / stats->iterations);
            printf("  Comm hidden:   %.2f ms\n", stats->time_comm_hidden_ms);
            printf("  Comm exposed:  %.2f ms\n", stats->time_comm_exposed_ms);
            printf("  Overlap eff:   %.1f%%\n", stats->overlap_efficiency * 100.0);
        }
        printf("========================================\n");
    }

    // Copy result back
    CUDA_CHECK(cudaStreamSynchronize(stream_compute));
    CUDA_CHECK(cudaMemcpyAsync(&x[row_offset], d_x_local, n_local * sizeof(double),
                               cudaMemcpyDeviceToHost, stream_compute));

    // Gather full solution to rank 0
    int* recvcounts = NULL;
    int* displs = NULL;
    if (rank == 0) {
        recvcounts = (int*)malloc(world_size * sizeof(int));
        displs = (int*)malloc(world_size * sizeof(int));
        int base_size = n / world_size;
        for (int i = 0; i < world_size; i++) {
            displs[i] = i * base_size;
            recvcounts[i] = (i == world_size - 1) ? (n - displs[i]) : base_size;
        }
    }
    MPI_Gatherv(&x[row_offset], n_local, MPI_DOUBLE, x, recvcounts, displs, MPI_DOUBLE, 0,
                MPI_COMM_WORLD);
    if (rank == 0) {
        free(recvcounts);
        free(displs);
    }

    // Solution validation
    if (rank == 0) {
        double sol_sum = 0.0, sol_norm_sq = 0.0;
        for (int i = 0; i < n; i++) {
            sol_sum += x[i];
            sol_norm_sq += x[i] * x[i];
        }
        stats->solution_sum = sol_sum;
        stats->solution_norm = sqrt(sol_norm_sq);
    }

    // Cleanup
    cudaFree(d_x_local);
    cudaFree(d_r_local);
    cudaFree(d_p_local);
    cudaFree(d_Ap);
    cudaFree(d_b);
    cudaFree(d_row_ptr);
    cudaFree(d_col_idx);
    cudaFree(d_values);

    if (d_p_halo_prev)
        cudaFree(d_p_halo_prev);
    if (d_p_halo_next)
        cudaFree(d_p_halo_next);

    if (h_send_prev)
        cudaFreeHost(h_send_prev);
    if (h_send_next)
        cudaFreeHost(h_send_next);
    if (h_recv_prev)
        cudaFreeHost(h_recv_prev);
    if (h_recv_next)
        cudaFreeHost(h_recv_next);

    cublasDestroy(cublas_handle);
    cudaEventDestroy(p_updated_event);
    cudaStreamDestroy(stream_compute);
    cudaStreamDestroy(stream_comm);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    if (config.enable_detailed_timers) {
        cudaEventDestroy(timer_start);
        cudaEventDestroy(timer_stop);
        cudaEventDestroy(timer_comm_start);
        cudaEventDestroy(timer_comm_stop);
        cudaEventDestroy(timer_phase_start);
        cudaEventDestroy(timer_phase_stop);
    }

    return 0;
}

/* ================================================================
 * 3D 27-Point Multi-GPU CG solver with compute-communication overlap
 * ================================================================ */

int cg_solve_mgpu_partitioned_overlap_27pt_3d(SpmvOperator* spmv_op, MatrixData* mat,
                                              const double* b, double* x, CGConfigMultiGPU config,
                                              CGStatsMultiGPU* stats) {

    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int n = mat->rows;
    int grid_size = mat->grid_size;

    if (rank == 0 && config.verbose >= 1) {
        printf("\n========================================\n");
        printf("Multi-GPU CG Solver (3D 27-POINT OVERLAP)\n");
        printf("========================================\n");
        printf("MPI ranks: %d\n", world_size);
        printf("Problem size: %d unknowns (%d³ grid)\n", n, grid_size);
        printf("Max iterations: %d\n", config.max_iters);
        printf("Tolerance: %.1e\n", config.tolerance);
        printf("========================================\n\n");
    }

    CUDA_CHECK(cudaSetDevice(rank));

    int n_local = n / world_size;
    int row_offset = rank * n_local;
    if (rank == world_size - 1) {
        n_local = n - row_offset;
    }

    if (config.verbose >= 1) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, rank));
        printf("[Rank %d] GPU %d: %s (CC %d.%d)\n", rank, rank, prop.name, prop.major, prop.minor);
        printf("[Rank %d] Rows: [%d:%d) (%d rows, %d Z-planes)\n", rank, row_offset,
               row_offset + n_local, n_local, n_local / (grid_size * grid_size));
    }

    OverlapPartition partition;
    compute_overlap_partition(&partition, n_local, grid_size, rank, world_size, 3);

    if (config.verbose >= 1) {
        printf("[Rank %d] Overlap partition: interior=[%d:%d) (%d rows), "
               "boundary_prev=%d rows, boundary_next=%d rows, halo_elems=%d\n",
               rank, partition.interior_start, partition.interior_start + partition.interior_count,
               partition.interior_count, partition.boundary_prev_count,
               partition.boundary_next_count, partition.halo_elems);
    }

    cudaStream_t stream_compute, stream_comm;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_compute, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_comm, cudaStreamNonBlocking));

    cudaEvent_t p_updated_event;
    CUDA_CHECK(cudaEventCreate(&p_updated_event));

    cublasHandle_t cublas_handle;
    cublasStatus_t cublas_status = cublasCreate(&cublas_handle);
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS initialization failed\n");
        exit(EXIT_FAILURE);
    }
    cublasSetStream(cublas_handle, stream_compute);

    if (rank == 0 && config.verbose >= 1) {
        printf("Building local CSR partitions...\n");
    }

    build_csr_struct(mat);

    long long local_nnz = csr_mat.row_ptr[row_offset + n_local] - csr_mat.row_ptr[row_offset];

    long long* d_row_ptr;
    int* d_col_idx;
    double* d_values;

    CUDA_CHECK(cudaMalloc(&d_row_ptr, (n_local + 1) * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_col_idx, (size_t)local_nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_values, (size_t)local_nnz * sizeof(double)));

    long long* local_row_ptr = (long long*)malloc((n_local + 1) * sizeof(long long));
    long long offset = csr_mat.row_ptr[row_offset];
    for (int i = 0; i <= n_local; i++) {
        local_row_ptr[i] = csr_mat.row_ptr[row_offset + i] - offset;
    }

    CUDA_CHECK(cudaMemcpyAsync(d_row_ptr, local_row_ptr, (n_local + 1) * sizeof(long long),
                               cudaMemcpyHostToDevice, stream_compute));
    CUDA_CHECK(cudaMemcpyAsync(d_col_idx, &csr_mat.col_indices[offset],
                               (size_t)local_nnz * sizeof(int), cudaMemcpyHostToDevice,
                               stream_compute));
    CUDA_CHECK(cudaMemcpyAsync(d_values, &csr_mat.values[offset],
                               (size_t)local_nnz * sizeof(double), cudaMemcpyHostToDevice,
                               stream_compute));

    free(local_row_ptr);

    if (config.verbose >= 1) {
        printf("[Rank %d] Local CSR: %d rows, %lld nnz (%.2f MB)\n", rank, n_local, local_nnz,
               (n_local * sizeof(long long) + (double)local_nnz * (sizeof(int) + sizeof(double))) /
                   1e6);
    }

    double *d_x_local, *d_r_local, *d_p_local, *d_Ap, *d_b;
    double *d_p_halo_prev = NULL, *d_p_halo_next = NULL;

    CUDA_CHECK(cudaMalloc(&d_x_local, n_local * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_r_local, n_local * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_p_local, n_local * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Ap, n_local * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_b, n_local * sizeof(double)));

    if (rank > 0) {
        CUDA_CHECK(cudaMalloc(&d_p_halo_prev, partition.halo_elems * sizeof(double)));
    }
    if (rank < world_size - 1) {
        CUDA_CHECK(cudaMalloc(&d_p_halo_next, partition.halo_elems * sizeof(double)));
    }

    double *h_send_prev = NULL, *h_send_next = NULL;
    double *h_recv_prev = NULL, *h_recv_next = NULL;
    if (rank > 0) {
        CUDA_CHECK(cudaMallocHost(&h_send_prev, partition.halo_elems * sizeof(double)));
        CUDA_CHECK(cudaMallocHost(&h_recv_prev, partition.halo_elems * sizeof(double)));
    }
    if (rank < world_size - 1) {
        CUDA_CHECK(cudaMallocHost(&h_send_next, partition.halo_elems * sizeof(double)));
        CUDA_CHECK(cudaMallocHost(&h_recv_next, partition.halo_elems * sizeof(double)));
    }

    CUDA_CHECK(cudaMemcpyAsync(d_b, &b[row_offset], n_local * sizeof(double),
                               cudaMemcpyHostToDevice, stream_compute));
    CUDA_CHECK(cudaMemcpyAsync(d_x_local, &x[row_offset], n_local * sizeof(double),
                               cudaMemcpyHostToDevice, stream_compute));
    CUDA_CHECK(cudaMemsetAsync(d_r_local, 0, n_local * sizeof(double), stream_compute));
    CUDA_CHECK(cudaMemsetAsync(d_p_local, 0, n_local * sizeof(double), stream_compute));
    CUDA_CHECK(cudaStreamSynchronize(stream_compute));

    MPI_Barrier(MPI_COMM_WORLD);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    cudaEvent_t timer_start, timer_stop;
    cudaEvent_t timer_comm_start, timer_comm_stop;
    cudaEvent_t timer_phase_start, timer_phase_stop;
    if (config.enable_detailed_timers) {
        CUDA_CHECK(cudaEventCreate(&timer_start));
        CUDA_CHECK(cudaEventCreate(&timer_stop));
        CUDA_CHECK(cudaEventCreate(&timer_comm_start));
        CUDA_CHECK(cudaEventCreate(&timer_comm_stop));
        CUDA_CHECK(cudaEventCreate(&timer_phase_start));
        CUDA_CHECK(cudaEventCreate(&timer_phase_stop));
    }

    CUDA_CHECK(cudaEventRecord(start, stream_compute));

    stats->time_spmv_ms = 0.0;
    stats->time_blas1_ms = 0.0;
    stats->time_reductions_ms = 0.0;
    stats->time_allreduce_ms = 0.0;
    stats->time_allgather_ms = 0.0;
    stats->time_spmv_interior_ms = 0.0;
    stats->time_spmv_boundary_ms = 0.0;
    stats->time_comm_total_ms = 0.0;
    stats->time_comm_hidden_ms = 0.0;
    stats->time_comm_exposed_ms = 0.0;
    stats->overlap_efficiency = 0.0;

    if (rank == 0 && config.verbose >= 1) {
        printf("\nStarting CG iterations (3D 27-point overlap mode)...\n");
    }

    int threads = 256;
    int blocks_local = (n_local + threads - 1) / threads;

    // Initial residual: r = b - A*x0
    double *d_x_halo_prev = NULL, *d_x_halo_next = NULL;
    if (rank > 0) {
        CUDA_CHECK(cudaMalloc(&d_x_halo_prev, partition.halo_elems * sizeof(double)));
    }
    if (rank < world_size - 1) {
        CUDA_CHECK(cudaMalloc(&d_x_halo_next, partition.halo_elems * sizeof(double)));
    }

    exchange_halo_sync(d_x_local, d_x_local + (n_local - partition.halo_elems), d_x_halo_prev,
                       d_x_halo_next, h_send_prev, h_send_next, h_recv_prev, h_recv_next,
                       partition.halo_elems, rank, world_size, stream_compute);

    // Ap = A*x0 (full 27-point kernel)
    stencil27_csr_partitioned_halo_kernel_3d<<<blocks_local, threads, 0, stream_compute>>>(
        d_row_ptr, d_col_idx, d_values, d_x_local, d_x_halo_prev, d_x_halo_next, d_Ap, n_local,
        row_offset, n, grid_size);

    axpy_kernel<<<blocks_local, threads, 0, stream_compute>>>(-1.0, d_Ap, d_b, n_local);
    CUDA_CHECK(cudaMemcpyAsync(d_r_local, d_b, n_local * sizeof(double), cudaMemcpyDeviceToDevice,
                               stream_compute));

    CUDA_CHECK(cudaMemcpyAsync(d_p_local, d_r_local, n_local * sizeof(double),
                               cudaMemcpyDeviceToDevice, stream_compute));
    CUDA_CHECK(cudaEventRecord(p_updated_event, stream_compute));
    CUDA_CHECK(cudaStreamSynchronize(stream_compute));

    double rs_local_old = overlap_compute_local_dot(cublas_handle, d_r_local, d_r_local, n_local);
    double rs_old;
    MPI_Allreduce(&rs_local_old, &rs_old, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    double b_norm = sqrt(rs_old);

    if (rank == 0 && config.verbose >= 2) {
        printf("[Iter   0] Residual: %.6e\n", sqrt(rs_old));
    }

    if (d_x_halo_prev)
        cudaFree(d_x_halo_prev);
    if (d_x_halo_next)
        cudaFree(d_x_halo_next);

    double cum_interior_ms = 0.0;
    double cum_boundary_ms = 0.0;
    double cum_comm_ms = 0.0;
    double cum_overlap_phase_ms = 0.0;

    // CG iteration loop
    nvtxRangePush("CG_Solver_27PT_3D_Overlap");
    int iter;
    for (iter = 0; iter < config.max_iters; iter++) {
        nvtxRangePush("CG_Iteration_27PT_3D_Overlap");

        nvtxRangePush("SpMV_27PT_3D_Overlap");

        if (config.enable_detailed_timers) {
            CUDA_CHECK(cudaEventRecord(timer_phase_start, stream_comm));
        }

        CUDA_CHECK(cudaStreamWaitEvent(stream_comm, p_updated_event, 0));

        exchange_halo_d2h_start(d_p_local, n_local, h_send_prev, h_send_next, partition.halo_elems,
                                rank, world_size, stream_comm);

        if (config.enable_detailed_timers) {
            CUDA_CHECK(cudaEventRecord(timer_start, stream_compute));
        }
        if (partition.interior_count > 0) {
            int blocks_interior = (partition.interior_count + threads - 1) / threads;
            stencil27_overlap_subrange_kernel_3d<<<blocks_interior, threads, 0, stream_compute>>>(
                d_row_ptr, d_col_idx, d_values, d_p_local, d_p_halo_prev, d_p_halo_next, d_Ap,
                n_local, row_offset, n, grid_size, partition.interior_start,
                partition.interior_count);
        }
        if (config.enable_detailed_timers) {
            CUDA_CHECK(cudaEventRecord(timer_stop, stream_compute));
        }

        MPI_Request requests[4];
        int req_count = 0;
        exchange_halo_mpi_start(h_send_prev, h_send_next, h_recv_prev, h_recv_next,
                                partition.halo_elems, rank, world_size, stream_comm, requests,
                                &req_count);

        exchange_halo_async_finish(d_p_halo_prev, d_p_halo_next, h_recv_prev, h_recv_next,
                                   partition.halo_elems, rank, world_size, stream_comm, requests,
                                   req_count);

        CUDA_CHECK(cudaStreamSynchronize(stream_compute));

        if (config.enable_detailed_timers) {
            float interior_ms;
            CUDA_CHECK(cudaEventElapsedTime(&interior_ms, timer_start, timer_stop));
            cum_interior_ms += interior_ms;
        }

        if (config.enable_detailed_timers) {
            CUDA_CHECK(cudaEventRecord(timer_start, stream_compute));
        }
        if (partition.boundary_prev_count > 0) {
            int blocks_prev = (partition.boundary_prev_count + threads - 1) / threads;
            stencil27_overlap_subrange_kernel_3d<<<blocks_prev, threads, 0, stream_compute>>>(
                d_row_ptr, d_col_idx, d_values, d_p_local, d_p_halo_prev, d_p_halo_next, d_Ap,
                n_local, row_offset, n, grid_size, partition.boundary_prev_start,
                partition.boundary_prev_count);
        }
        if (partition.boundary_next_count > 0) {
            int blocks_next = (partition.boundary_next_count + threads - 1) / threads;
            stencil27_overlap_subrange_kernel_3d<<<blocks_next, threads, 0, stream_compute>>>(
                d_row_ptr, d_col_idx, d_values, d_p_local, d_p_halo_prev, d_p_halo_next, d_Ap,
                n_local, row_offset, n, grid_size, partition.boundary_next_start,
                partition.boundary_next_count);
        }
        if (config.enable_detailed_timers) {
            CUDA_CHECK(cudaEventRecord(timer_stop, stream_compute));
            CUDA_CHECK(cudaEventRecord(timer_phase_stop, stream_compute));
            CUDA_CHECK(cudaStreamSynchronize(stream_compute));

            float boundary_ms;
            CUDA_CHECK(cudaEventElapsedTime(&boundary_ms, timer_start, timer_stop));
            cum_boundary_ms += boundary_ms;

            float phase_ms;
            CUDA_CHECK(cudaEventElapsedTime(&phase_ms, timer_phase_start, timer_phase_stop));
            cum_overlap_phase_ms += phase_ms;
        }

        nvtxRangePop();  // SpMV_27PT_3D_Overlap

        // Standard CG operations (identical to 7-point)
        nvtxRangePush("Dot_Product");
        if (config.enable_detailed_timers) {
            CUDA_CHECK(cudaEventRecord(timer_start, stream_compute));
        }
        double pAp_local = overlap_compute_local_dot(cublas_handle, d_p_local, d_Ap, n_local);
        if (config.enable_detailed_timers) {
            CUDA_CHECK(cudaEventRecord(timer_stop, stream_compute));
            CUDA_CHECK(cudaEventSynchronize(timer_stop));
            float elapsed_ms;
            CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, timer_start, timer_stop));
            stats->time_reductions_ms += elapsed_ms;
            stats->time_dot_pAp_ms += elapsed_ms;
        }
        nvtxRangePop();

        if (config.enable_detailed_timers) {
            CUDA_CHECK(cudaEventRecord(timer_start, stream_compute));
        }
        double pAp;
        MPI_Allreduce(&pAp_local, &pAp, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        if (config.enable_detailed_timers) {
            CUDA_CHECK(cudaEventRecord(timer_stop, stream_compute));
            CUDA_CHECK(cudaEventSynchronize(timer_stop));
            float elapsed_ms;
            CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, timer_start, timer_stop));
            stats->time_allreduce_ms += elapsed_ms;
        }

        double alpha = rs_old / pAp;

        nvtxRangePush("BLAS_AXPY");
        if (config.enable_detailed_timers) {
            CUDA_CHECK(cudaEventRecord(timer_start, stream_compute));
        }
        axpy_kernel<<<blocks_local, threads, 0, stream_compute>>>(alpha, d_p_local, d_x_local,
                                                                  n_local);
        if (config.enable_detailed_timers) {
            CUDA_CHECK(cudaEventRecord(timer_stop, stream_compute));
            CUDA_CHECK(cudaEventSynchronize(timer_stop));
            float elapsed_ms;
            CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, timer_start, timer_stop));
            stats->time_blas1_ms += elapsed_ms;
            stats->time_axpy_update_x_ms += elapsed_ms;
        }

        if (config.enable_detailed_timers) {
            CUDA_CHECK(cudaEventRecord(timer_start, stream_compute));
        }
        axpy_kernel<<<blocks_local, threads, 0, stream_compute>>>(-alpha, d_Ap, d_r_local, n_local);
        if (config.enable_detailed_timers) {
            CUDA_CHECK(cudaEventRecord(timer_stop, stream_compute));
            CUDA_CHECK(cudaEventSynchronize(timer_stop));
            float elapsed_ms;
            CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, timer_start, timer_stop));
            stats->time_blas1_ms += elapsed_ms;
            stats->time_axpy_update_r_ms += elapsed_ms;
        }
        nvtxRangePop();

        nvtxRangePush("Dot_Product");
        if (config.enable_detailed_timers) {
            CUDA_CHECK(cudaEventRecord(timer_start, stream_compute));
        }
        double rs_local_new =
            overlap_compute_local_dot(cublas_handle, d_r_local, d_r_local, n_local);
        if (config.enable_detailed_timers) {
            CUDA_CHECK(cudaEventRecord(timer_stop, stream_compute));
            CUDA_CHECK(cudaEventSynchronize(timer_stop));
            float elapsed_ms;
            CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, timer_start, timer_stop));
            stats->time_reductions_ms += elapsed_ms;
            stats->time_dot_rs_new_ms += elapsed_ms;
        }
        nvtxRangePop();

        if (config.enable_detailed_timers) {
            CUDA_CHECK(cudaEventRecord(timer_start, stream_compute));
        }
        double rs_new;
        MPI_Allreduce(&rs_local_new, &rs_new, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        if (config.enable_detailed_timers) {
            CUDA_CHECK(cudaEventRecord(timer_stop, stream_compute));
            CUDA_CHECK(cudaEventSynchronize(timer_stop));
            float elapsed_ms;
            CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, timer_start, timer_stop));
            stats->time_allreduce_ms += elapsed_ms;
        }

        double residual_norm = sqrt(rs_new);
        double rel_residual = residual_norm / b_norm;

        if (rank == 0 && config.verbose >= 2) {
            printf("[Iter %3d] Residual: %.6e (rel: %.6e, alpha: %.4e)\n", iter + 1, residual_norm,
                   rel_residual, alpha);
        }

        if (rel_residual < config.tolerance) {
            iter++;
            stats->converged = 1;
            stats->iterations = iter;
            stats->residual_norm = residual_norm;
            nvtxRangePop();
            break;
        }

        double beta = rs_new / rs_old;

        nvtxRangePush("BLAS_AXPBY");
        if (config.enable_detailed_timers) {
            CUDA_CHECK(cudaEventRecord(timer_start, stream_compute));
        }
        axpby_kernel<<<blocks_local, threads, 0, stream_compute>>>(1.0, d_r_local, beta, d_p_local,
                                                                   n_local);
        if (config.enable_detailed_timers) {
            CUDA_CHECK(cudaEventRecord(timer_stop, stream_compute));
            CUDA_CHECK(cudaEventSynchronize(timer_stop));
            float elapsed_ms;
            CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, timer_start, timer_stop));
            stats->time_blas1_ms += elapsed_ms;
            stats->time_axpby_update_p_ms += elapsed_ms;
        }
        nvtxRangePop();

        CUDA_CHECK(cudaEventRecord(p_updated_event, stream_compute));

        rs_old = rs_new;
        nvtxRangePop();
    }
    nvtxRangePop();

    if (iter == config.max_iters && rank == 0) {
        printf("\nMax iterations reached without convergence\n");
        stats->converged = 0;
        stats->iterations = iter;
        stats->residual_norm = sqrt(rs_old);
    }

    CUDA_CHECK(cudaEventRecord(stop, stream_compute));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float time_ms;
    CUDA_CHECK(cudaEventElapsedTime(&time_ms, start, stop));

    if (config.enable_detailed_timers && stats->iterations > 0) {
        stats->time_spmv_interior_ms = cum_interior_ms;
        stats->time_spmv_boundary_ms = cum_boundary_ms;
        stats->time_spmv_ms = cum_interior_ms + cum_boundary_ms;

        double sequential_est = cum_interior_ms + cum_boundary_ms + cum_comm_ms;
        stats->time_comm_total_ms = cum_overlap_phase_ms - cum_interior_ms - cum_boundary_ms;
        if (stats->time_comm_total_ms < 0.0)
            stats->time_comm_total_ms = 0.0;

        stats->time_comm_hidden_ms = sequential_est - cum_overlap_phase_ms;
        if (stats->time_comm_hidden_ms < 0.0)
            stats->time_comm_hidden_ms = 0.0;

        stats->time_comm_exposed_ms = stats->time_comm_total_ms - stats->time_comm_hidden_ms;
        if (stats->time_comm_exposed_ms < 0.0)
            stats->time_comm_exposed_ms = 0.0;

        stats->overlap_efficiency = (stats->time_comm_total_ms > 0.0)
                                        ? stats->time_comm_hidden_ms / stats->time_comm_total_ms
                                        : 0.0;
        if (stats->overlap_efficiency > 1.0)
            stats->overlap_efficiency = 1.0;

        stats->time_dot_pAp_ms /= stats->iterations;
        stats->time_dot_rs_new_ms /= stats->iterations;
        stats->time_axpy_update_x_ms /= stats->iterations;
        stats->time_axpy_update_r_ms /= stats->iterations;
        stats->time_axpby_update_p_ms /= stats->iterations;
    }

    stats->time_total_ms = time_ms;

    if (world_size > 1) {
        double local_times[6], max_times[6], min_times[6];

        local_times[0] = stats->time_total_ms;
        local_times[1] = stats->time_spmv_ms;
        local_times[2] = stats->time_blas1_ms;
        local_times[3] = stats->time_reductions_ms;
        local_times[4] = stats->time_allreduce_ms;
        local_times[5] = stats->time_allgather_ms;

        MPI_Reduce(local_times, max_times, 6, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(local_times, min_times, 6, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            stats->time_total_ms = max_times[0];
            stats->time_spmv_ms = max_times[1];
            stats->time_blas1_ms = max_times[2];
            stats->time_reductions_ms = max_times[3];
            stats->time_allreduce_ms = max_times[4];
            stats->time_allgather_ms = max_times[5];

            double imbalance_pct = 100.0 * (max_times[0] - min_times[0]) / max_times[0];
            printf("Total time: %.2f ms (max), %.2f ms (min) - Load imbalance: %.1f%%\n",
                   max_times[0], min_times[0], imbalance_pct);
        }
    }

    if (rank == 0 && config.verbose >= 1) {
        printf("Total time: %.2f ms\n", stats->time_total_ms);
        if (config.enable_detailed_timers) {
            printf("\nOverlap Metrics (%d iterations):\n", stats->iterations);
            printf("  Interior SpMV: %.2f ms (%.3f ms/iter)\n", stats->time_spmv_interior_ms,
                   stats->time_spmv_interior_ms / stats->iterations);
            printf("  Boundary SpMV: %.2f ms (%.3f ms/iter)\n", stats->time_spmv_boundary_ms,
                   stats->time_spmv_boundary_ms / stats->iterations);
            printf("  Comm total:    %.2f ms (%.3f ms/iter)\n", stats->time_comm_total_ms,
                   stats->time_comm_total_ms / stats->iterations);
            printf("  Comm hidden:   %.2f ms\n", stats->time_comm_hidden_ms);
            printf("  Comm exposed:  %.2f ms\n", stats->time_comm_exposed_ms);
            printf("  Overlap eff:   %.1f%%\n", stats->overlap_efficiency * 100.0);
        }
        printf("========================================\n");
    }

    CUDA_CHECK(cudaStreamSynchronize(stream_compute));
    CUDA_CHECK(cudaMemcpyAsync(&x[row_offset], d_x_local, n_local * sizeof(double),
                               cudaMemcpyDeviceToHost, stream_compute));

    int* recvcounts = NULL;
    int* displs = NULL;
    if (rank == 0) {
        recvcounts = (int*)malloc(world_size * sizeof(int));
        displs = (int*)malloc(world_size * sizeof(int));
        int base_size = n / world_size;
        for (int i = 0; i < world_size; i++) {
            displs[i] = i * base_size;
            recvcounts[i] = (i == world_size - 1) ? (n - displs[i]) : base_size;
        }
    }
    MPI_Gatherv(&x[row_offset], n_local, MPI_DOUBLE, x, recvcounts, displs, MPI_DOUBLE, 0,
                MPI_COMM_WORLD);
    if (rank == 0) {
        free(recvcounts);
        free(displs);
    }

    if (rank == 0) {
        double sol_sum = 0.0, sol_norm_sq = 0.0;
        for (int i = 0; i < n; i++) {
            sol_sum += x[i];
            sol_norm_sq += x[i] * x[i];
        }
        stats->solution_sum = sol_sum;
        stats->solution_norm = sqrt(sol_norm_sq);
    }

    cudaFree(d_x_local);
    cudaFree(d_r_local);
    cudaFree(d_p_local);
    cudaFree(d_Ap);
    cudaFree(d_b);
    cudaFree(d_row_ptr);
    cudaFree(d_col_idx);
    cudaFree(d_values);

    if (d_p_halo_prev)
        cudaFree(d_p_halo_prev);
    if (d_p_halo_next)
        cudaFree(d_p_halo_next);

    if (h_send_prev)
        cudaFreeHost(h_send_prev);
    if (h_send_next)
        cudaFreeHost(h_send_next);
    if (h_recv_prev)
        cudaFreeHost(h_recv_prev);
    if (h_recv_next)
        cudaFreeHost(h_recv_next);

    cublasDestroy(cublas_handle);
    cudaEventDestroy(p_updated_event);
    cudaStreamDestroy(stream_compute);
    cudaStreamDestroy(stream_comm);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    if (config.enable_detailed_timers) {
        cudaEventDestroy(timer_start);
        cudaEventDestroy(timer_stop);
        cudaEventDestroy(timer_comm_start);
        cudaEventDestroy(timer_comm_stop);
        cudaEventDestroy(timer_phase_start);
        cudaEventDestroy(timer_phase_stop);
    }

    return 0;
}
