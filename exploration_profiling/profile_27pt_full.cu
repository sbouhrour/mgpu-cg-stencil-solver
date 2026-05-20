/**
 * @file profile_27pt_full.cu
 * @brief Minimal NCU launch harness for the production 27-point partitioned-halo
 *        SpMV kernel.
 *
 * Calls the UNMODIFIED production kernel
 * stencil27_csr_partitioned_halo_kernel_3d (linked from
 * src/spmv/spmv_stencil_3d_27pt_partitioned_halo_kernel.cu) exactly once,
 * under single-GPU conditions identical to Exploration 1:
 *   1 rank, x_halo_prev = x_halo_next = NULL, row_offset = 0.
 *
 * Matrix loaded via load_matrix_stencil27_3d_from_grid (Path B, rank 0 /
 * world_size 1) then build_csr_struct, both reused as-is.
 *
 * No internal loop, no warmup: NCU handles replay/warmup itself.
 * Launch config: 256 threads/block, blocks = ceil(n_local / 256).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include "io.h"
#include "spmv_csr.h"
#include "spmv.h"

extern struct CSRMatrix csr_mat;

// Production kernel — declared extern, defined in
// src/spmv/spmv_stencil_3d_27pt_partitioned_halo_kernel.cu (linked, unmodified).
__global__ void stencil27_csr_partitioned_halo_kernel_3d(
    const long long* __restrict__ row_ptr, const int* __restrict__ col_idx,
    const double* __restrict__ values, const double* __restrict__ x_local,
    const double* __restrict__ x_halo_prev, const double* __restrict__ x_halo_next,
    double* __restrict__ y, int n_local, int row_offset, int N_total, int grid_size);

int main(int argc, char** argv) {
    const char* matrix_file = "matrix/stencil3d_27pt_192.mtx";
    for (int a = 1; a < argc; a++) {
        if (argv[a][0] != '-')
            matrix_file = argv[a];
    }

    // ---- Load matrix via Path B (rank 0, world_size 1), reused as-is ----
    printf("Loading matrix (Path B): %s\n", matrix_file);
    MatrixData mat;
    memset(&mat, 0, sizeof(mat));
    if (load_matrix_stencil27_3d_from_grid(matrix_file, &mat, 0, 1) != 0) {
        fprintf(stderr, "Error loading matrix\n");
        return 1;
    }
    printf("Matrix: %d x %d, %lld nnz, grid_size=%d\n", mat.rows, mat.cols, mat.nnz, mat.grid_size);

    // ---- Build CSR (reused as-is; performs the per-row column sort) ----
    if (build_csr_struct(&mat) != EXIT_SUCCESS) {
        fprintf(stderr, "build_csr_struct failed\n");
        return 1;
    }
    int n = csr_mat.nb_rows;
    long long nnz = csr_mat.nb_nonzeros;
    int grid_size = mat.grid_size;

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

    CUDA_CHECK(cudaMemcpy(d_row_ptr, csr_mat.row_ptr, (size_t)(n + 1) * sizeof(long long),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_idx, csr_mat.col_indices, (size_t)nnz * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(d_values, csr_mat.values, (size_t)nnz * sizeof(double), cudaMemcpyHostToDevice));

    double* h_x = (double*)malloc((size_t)n * sizeof(double));
    for (int idx = 0; idx < n; idx++)
        h_x[idx] = 1.0;
    CUDA_CHECK(cudaMemcpy(d_x, h_x, (size_t)n * sizeof(double), cudaMemcpyHostToDevice));
    free(h_x);

    // ---- Launch configuration ----
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    printf("Launch config: blocks=%d, threads=%d (n=%d, nnz=%lld)\n", blocks, threads, n, nnz);

    // ---- Single launch of the unmodified production kernel ----
    stencil27_csr_partitioned_halo_kernel_3d<<<blocks, threads>>>(
        d_row_ptr, d_col_idx, d_values, d_x, NULL, NULL, d_y, n, 0, n, grid_size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaFree(d_row_ptr));
    CUDA_CHECK(cudaFree(d_col_idx));
    CUDA_CHECK(cudaFree(d_values));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));

    printf("kernel launched OK\n");
    return 0;
}
