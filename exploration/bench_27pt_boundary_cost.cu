/**
 * @file bench_27pt_boundary_cost.cu
 * @brief Exploration micro-benchmark: cost of the boundary branch in the
 *        3D 27-point partitioned-halo SpMV kernel.
 *
 * Single-GPU configuration: 1 rank, x_halo_prev = x_halo_next = NULL,
 * row_offset = 0. Three kernels, all derived from the production kernel
 * stencil27_csr_partitioned_halo_kernel_3d:
 *
 *   - stencil27_full                  : EXACT copy of the production kernel.
 *   - stencil27_boundary_neutralized  : else block kept entirely (row_start,
 *                                       row_end, loop, registers) but loop
 *                                       body reduced to `sum += values[jj];`.
 *   - stencil27_interior_pure         : else block physically removed.
 *
 * Only stencil27_full produces a correct y. The other two produce a wrong y
 * on boundary rows by design — only timing is measured, y is not reused.
 *
 * Matrix loaded via load_matrix_stencil27_3d_from_grid (Path B, rank 0 /
 * world_size 1) then build_csr_struct, both reused as-is.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <algorithm>
#include <cuda_runtime.h>
#include "io.h"
#include "spmv_csr.h"
#include "spmv.h"

extern struct CSRMatrix csr_mat;

/* ===========================================================================
 * Kernel 1 — stencil27_full : EXACT copy of the production kernel
 *            stencil27_csr_partitioned_halo_kernel_3d (unchanged).
 * =========================================================================*/
__global__ void stencil27_full(
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

/* ===========================================================================
 * Kernel 2 — stencil27_boundary_neutralized (variant A)
 *            else block KEPT entirely: row_start, row_end, loop variable and
 *            the loop itself remain (all associated registers stay allocated).
 *            Only the expensive work (col_idx indirection + halo routing) is
 *            removed; loop body reduced to `sum += values[jj];`.
 * =========================================================================*/
__global__ void stencil27_boundary_neutralized(
    const long long* __restrict__ row_ptr, const int* __restrict__ col_idx,
    const double* __restrict__ values, const double* __restrict__ x_local,
    const double* __restrict__ x_halo_prev, const double* __restrict__ x_halo_next,
    double* __restrict__ y, int n_local, int row_offset, int N_total, int grid_size) {

    int local_row = blockIdx.x * blockDim.x + threadIdx.x;
    if (local_row >= n_local)
        return;

    int global_row = row_offset + local_row;
    int N = grid_size;

    int i = global_row / (N * N);
    int j = (global_row / N) % N;
    int k = global_row % N;

    int local_nz = n_local / (N * N);
    int local_z = local_row / (N * N);

    double sum = 0.0;

    bool is_interior = (i > 0 && i < N - 1 && j > 0 && j < N - 1 && k > 0 && k < N - 1 &&
                        local_z > 0 && local_z < local_nz - 1);

    if (is_interior) {
        long long csr_offset = row_ptr[local_row];

        sum = values[csr_offset + 0] * x_local[local_row - N * N - N - 1];
        sum += values[csr_offset + 1] * x_local[local_row - N * N - N];
        sum += values[csr_offset + 2] * x_local[local_row - N * N - N + 1];
        sum += values[csr_offset + 3] * x_local[local_row - N * N - 1];
        sum += values[csr_offset + 4] * x_local[local_row - N * N];
        sum += values[csr_offset + 5] * x_local[local_row - N * N + 1];
        sum += values[csr_offset + 6] * x_local[local_row - N * N + N - 1];
        sum += values[csr_offset + 7] * x_local[local_row - N * N + N];
        sum += values[csr_offset + 8] * x_local[local_row - N * N + N + 1];
        sum += values[csr_offset + 9] * x_local[local_row - N - 1];
        sum += values[csr_offset + 10] * x_local[local_row - N];
        sum += values[csr_offset + 11] * x_local[local_row - N + 1];
        sum += values[csr_offset + 12] * x_local[local_row - 1];
        sum += values[csr_offset + 13] * x_local[local_row];
        sum += values[csr_offset + 14] * x_local[local_row + 1];
        sum += values[csr_offset + 15] * x_local[local_row + N - 1];
        sum += values[csr_offset + 16] * x_local[local_row + N];
        sum += values[csr_offset + 17] * x_local[local_row + N + 1];
        sum += values[csr_offset + 18] * x_local[local_row + N * N - N - 1];
        sum += values[csr_offset + 19] * x_local[local_row + N * N - N];
        sum += values[csr_offset + 20] * x_local[local_row + N * N - N + 1];
        sum += values[csr_offset + 21] * x_local[local_row + N * N - 1];
        sum += values[csr_offset + 22] * x_local[local_row + N * N];
        sum += values[csr_offset + 23] * x_local[local_row + N * N + 1];
        sum += values[csr_offset + 24] * x_local[local_row + N * N + N - 1];
        sum += values[csr_offset + 25] * x_local[local_row + N * N + N];
        sum += values[csr_offset + 26] * x_local[local_row + N * N + N + 1];
    }
    // else block kept (registers allocated, loop runs); expensive work removed.
    else {
        long long row_start = row_ptr[local_row];
        long long row_end = row_ptr[local_row + 1];
        for (long long jj = row_start; jj < row_end; jj++) {
            sum += values[jj];
        }
    }

    y[local_row] = sum;
}

/* ===========================================================================
 * Kernel 3 — stencil27_interior_pure (variant B)
 *            else block PHYSICALLY REMOVED. A non-interior row falls directly
 *            to y[local_row] = sum with sum == 0.0. The compiler never sees
 *            the CSR boundary code.
 * =========================================================================*/
__global__ void stencil27_interior_pure(
    const long long* __restrict__ row_ptr, const int* __restrict__ col_idx,
    const double* __restrict__ values, const double* __restrict__ x_local,
    const double* __restrict__ x_halo_prev, const double* __restrict__ x_halo_next,
    double* __restrict__ y, int n_local, int row_offset, int N_total, int grid_size) {

    int local_row = blockIdx.x * blockDim.x + threadIdx.x;
    if (local_row >= n_local)
        return;

    int global_row = row_offset + local_row;
    int N = grid_size;

    int i = global_row / (N * N);
    int j = (global_row / N) % N;
    int k = global_row % N;

    int local_nz = n_local / (N * N);
    int local_z = local_row / (N * N);

    double sum = 0.0;

    bool is_interior = (i > 0 && i < N - 1 && j > 0 && j < N - 1 && k > 0 && k < N - 1 &&
                        local_z > 0 && local_z < local_nz - 1);

    if (is_interior) {
        long long csr_offset = row_ptr[local_row];

        sum = values[csr_offset + 0] * x_local[local_row - N * N - N - 1];
        sum += values[csr_offset + 1] * x_local[local_row - N * N - N];
        sum += values[csr_offset + 2] * x_local[local_row - N * N - N + 1];
        sum += values[csr_offset + 3] * x_local[local_row - N * N - 1];
        sum += values[csr_offset + 4] * x_local[local_row - N * N];
        sum += values[csr_offset + 5] * x_local[local_row - N * N + 1];
        sum += values[csr_offset + 6] * x_local[local_row - N * N + N - 1];
        sum += values[csr_offset + 7] * x_local[local_row - N * N + N];
        sum += values[csr_offset + 8] * x_local[local_row - N * N + N + 1];
        sum += values[csr_offset + 9] * x_local[local_row - N - 1];
        sum += values[csr_offset + 10] * x_local[local_row - N];
        sum += values[csr_offset + 11] * x_local[local_row - N + 1];
        sum += values[csr_offset + 12] * x_local[local_row - 1];
        sum += values[csr_offset + 13] * x_local[local_row];
        sum += values[csr_offset + 14] * x_local[local_row + 1];
        sum += values[csr_offset + 15] * x_local[local_row + N - 1];
        sum += values[csr_offset + 16] * x_local[local_row + N];
        sum += values[csr_offset + 17] * x_local[local_row + N + 1];
        sum += values[csr_offset + 18] * x_local[local_row + N * N - N - 1];
        sum += values[csr_offset + 19] * x_local[local_row + N * N - N];
        sum += values[csr_offset + 20] * x_local[local_row + N * N - N + 1];
        sum += values[csr_offset + 21] * x_local[local_row + N * N - 1];
        sum += values[csr_offset + 22] * x_local[local_row + N * N];
        sum += values[csr_offset + 23] * x_local[local_row + N * N + 1];
        sum += values[csr_offset + 24] * x_local[local_row + N * N + N - 1];
        sum += values[csr_offset + 25] * x_local[local_row + N * N + N];
        sum += values[csr_offset + 26] * x_local[local_row + N * N + N + 1];
    }
    // No else block.

    y[local_row] = sum;
}

/* ===========================================================================
 * Harness
 * =========================================================================*/

enum KernelId { K_FULL = 0, K_NEUTRALIZED = 1, K_INTERIOR = 2 };
static const char* kernel_names[3] = {
    "stencil27_full", "stencil27_boundary_neutralized", "stencil27_interior_pure"};

static void launch_kernel(int which, int blocks, int threads,
                          const long long* d_row_ptr, const int* d_col_idx,
                          const double* d_values, const double* d_x, double* d_y,
                          int n_local, int grid_size) {
    switch (which) {
        case K_FULL:
            stencil27_full<<<blocks, threads>>>(d_row_ptr, d_col_idx, d_values, d_x,
                                                NULL, NULL, d_y, n_local, 0,
                                                n_local, grid_size);
            break;
        case K_NEUTRALIZED:
            stencil27_boundary_neutralized<<<blocks, threads>>>(
                d_row_ptr, d_col_idx, d_values, d_x, NULL, NULL, d_y, n_local, 0,
                n_local, grid_size);
            break;
        case K_INTERIOR:
            stencil27_interior_pure<<<blocks, threads>>>(d_row_ptr, d_col_idx,
                                                         d_values, d_x, NULL, NULL,
                                                         d_y, n_local, 0, n_local,
                                                         grid_size);
            break;
    }
}

int main(int argc, char** argv) {
    const char* matrix_file = "matrix/stencil3d_27pt_192.mtx";
    int profile_mode = 0;
    for (int a = 1; a < argc; a++) {
        if (strcmp(argv[a], "--profile") == 0)
            profile_mode = 1;
        else if (argv[a][0] != '-')
            matrix_file = argv[a];
    }

    const int N_WARMUP = 20;   // per-kernel warmup launches (untimed)
    const int N_RUNS = 100;    // per-kernel timed samples
    const int N_TRIM = 10;     // drop this many lowest AND highest (outliers)

    // ---- Load matrix via Path B (rank 0, world_size 1), reused as-is ----
    printf("Loading matrix (Path B): %s\n", matrix_file);
    MatrixData mat;
    memset(&mat, 0, sizeof(mat));
    if (load_matrix_stencil27_3d_from_grid(matrix_file, &mat, 0, 1) != 0) {
        fprintf(stderr, "Error loading matrix\n");
        return 1;
    }
    printf("Matrix: %d x %d, %lld nnz, grid_size=%d\n", mat.rows, mat.cols, mat.nnz,
           mat.grid_size);

    // ---- Build CSR (reused as-is; performs the per-row column sort) ----
    if (build_csr_struct(&mat) != EXIT_SUCCESS) {
        fprintf(stderr, "build_csr_struct failed\n");
        return 1;
    }
    int n = csr_mat.nb_rows;
    long long nnz = csr_mat.nb_nonzeros;
    int grid_size = mat.grid_size;

    // entries no longer needed once CSR is built — free to relieve host RAM
    if (mat.entries) {
        free(mat.entries);
        mat.entries = NULL;
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s (CC %d.%d, %d SMs)\n", prop.name, prop.major, prop.minor,
           prop.multiProcessorCount);

    // ---- Device allocations ----
    long long* d_row_ptr;
    int* d_col_idx;
    double *d_values, *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_row_ptr, (size_t)(n + 1) * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_col_idx, (size_t)nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_values, (size_t)nnz * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_x, (size_t)n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_y, (size_t)n * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_row_ptr, csr_mat.row_ptr,
                          (size_t)(n + 1) * sizeof(long long), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_idx, csr_mat.col_indices, (size_t)nnz * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_values, csr_mat.values, (size_t)nnz * sizeof(double),
                          cudaMemcpyHostToDevice));

    double* h_x = (double*)malloc((size_t)n * sizeof(double));
    for (int idx = 0; idx < n; idx++)
        h_x[idx] = 1.0;
    CUDA_CHECK(cudaMemcpy(d_x, h_x, (size_t)n * sizeof(double), cudaMemcpyHostToDevice));
    free(h_x);

    // ---- Launch configuration (identical for the three kernels) ----
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    printf("Launch config: blocks=%d, threads=%d (n=%d, nnz=%lld)\n\n", blocks,
           threads, n, nnz);

    if (profile_mode) {
        // Exactly one launch per kernel: NCU sees 3 clean launches.
        for (int kk = 0; kk < 3; kk++) {
            launch_kernel(kk, blocks, threads, d_row_ptr, d_col_idx, d_values, d_x,
                          d_y, n, grid_size);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        printf("Profile mode: one launch per kernel issued.\n");
    } else {
        // Separated per-kernel measurement. GPU clocks are locked externally
        // (nvidia-smi -lgc/-lmc), so the DVFS confound is removed at the
        // source: each kernel is measured in its own block, large sample,
        // after warmup, with explicit outlier trimming. Median and
        // trimmed-mean (drop N_TRIM lowest + N_TRIM highest) are robust to
        // the residual isolated spikes (OS scheduling / launch hiccups).

        double samples[3][256];
        double med[3], tmean[3], lo[3], hi[3], sd[3];
        double checksum[3];

        for (int kk = 0; kk < 3; kk++) {
            for (int w = 0; w < N_WARMUP; w++)
                launch_kernel(kk, blocks, threads, d_row_ptr, d_col_idx,
                              d_values, d_x, d_y, n, grid_size);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            for (int r = 0; r < N_RUNS; r++) {
                cudaEvent_t s, e;
                CUDA_CHECK(cudaEventCreate(&s));
                CUDA_CHECK(cudaEventCreate(&e));
                CUDA_CHECK(cudaEventRecord(s));
                launch_kernel(kk, blocks, threads, d_row_ptr, d_col_idx,
                              d_values, d_x, d_y, n, grid_size);
                CUDA_CHECK(cudaEventRecord(e));
                CUDA_CHECK(cudaEventSynchronize(e));
                float ms = 0.0f;
                CUDA_CHECK(cudaEventElapsedTime(&ms, s, e));
                samples[kk][r] = (double)ms;
                CUDA_CHECK(cudaEventDestroy(s));
                CUDA_CHECK(cudaEventDestroy(e));
            }

            // Sanity checksum (prevents dead-code elimination).
            double* h_y = (double*)malloc((size_t)n * sizeof(double));
            CUDA_CHECK(cudaMemcpy(h_y, d_y, (size_t)n * sizeof(double),
                                  cudaMemcpyDeviceToHost));
            double cs = 0.0;
            for (int idx = 0; idx < n; idx++)
                cs += h_y[idx];
            checksum[kk] = cs;
            free(h_y);

            std::sort(samples[kk], samples[kk] + N_RUNS);
            med[kk] = (samples[kk][N_RUNS / 2 - 1] + samples[kk][N_RUNS / 2]) / 2.0;
            lo[kk] = samples[kk][0];
            hi[kk] = samples[kk][N_RUNS - 1];
            // Trimmed mean: drop N_TRIM lowest and N_TRIM highest.
            double acc = 0.0;
            int cnt = 0;
            for (int r = N_TRIM; r < N_RUNS - N_TRIM; r++) {
                acc += samples[kk][r];
                cnt++;
            }
            tmean[kk] = acc / cnt;
            double var = 0.0;
            for (int r = 0; r < N_RUNS; r++)
                var += (samples[kk][r] - (med[kk])) * (samples[kk][r] - (med[kk]));
            sd[kk] = sqrt(var / N_RUNS);
        }

        for (int kk = 0; kk < 3; kk++) {
            printf("=== %s ===\n", kernel_names[kk]);
            printf("  N=%d  median %.4f  trimmed-mean(%d/side) %.4f  "
                   "min %.4f  max %.4f  sd %.4f\n",
                   N_RUNS, med[kk], N_TRIM, tmean[kk], lo[kk], hi[kk], sd[kk]);
            printf("  y checksum: %.6e\n\n", checksum[kk]);
        }

        printf("===== SUMMARY (separated per-kernel, locked clock, "
               "N=%d, trim %d/side) =====\n",
               N_RUNS, N_TRIM);
        printf("  %-32s median %.4f   trimmed-mean %.4f ms\n",
               kernel_names[K_FULL], med[K_FULL], tmean[K_FULL]);
        printf("  %-32s median %.4f   trimmed-mean %.4f ms\n",
               kernel_names[K_NEUTRALIZED], med[K_NEUTRALIZED],
               tmean[K_NEUTRALIZED]);
        printf("  %-32s median %.4f   trimmed-mean %.4f ms\n",
               kernel_names[K_INTERIOR], med[K_INTERIOR], tmean[K_INTERIOR]);
        printf("-----------------------------------------------------------\n");
        printf("  deltas on median       : full-A %+.4f  A-B %+.4f  "
               "full-B %+.4f ms\n",
               med[K_FULL] - med[K_NEUTRALIZED],
               med[K_NEUTRALIZED] - med[K_INTERIOR],
               med[K_FULL] - med[K_INTERIOR]);
        printf("  deltas on trimmed-mean : full-A %+.4f  A-B %+.4f  "
               "full-B %+.4f ms\n",
               tmean[K_FULL] - tmean[K_NEUTRALIZED],
               tmean[K_NEUTRALIZED] - tmean[K_INTERIOR],
               tmean[K_FULL] - tmean[K_INTERIOR]);
        printf("===========================================================\n");
    }

    CUDA_CHECK(cudaFree(d_row_ptr));
    CUDA_CHECK(cudaFree(d_col_idx));
    CUDA_CHECK(cudaFree(d_values));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    return 0;
}
