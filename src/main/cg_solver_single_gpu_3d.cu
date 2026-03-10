/**
 * @file cg_solver_single_gpu_3d.cu
 * @brief Single-GPU CG solver for 3D 7-point stencil validation
 *
 * Minimal single-GPU test for 3D stencil correctness.
 * Usage: ./bin/cg_solver_single_gpu_3d matrix/stencil3d_64.mtx [--verify] [--max-iters=N]
 *
 * Author: Bouhrour Stephane
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "spmv.h"
#include "io.h"
#include "spmv_csr.h"

// Extern kernel declarations
extern __global__ void stencil7_csr_partitioned_halo_kernel_3d(
    const long long* __restrict__ row_ptr, const int* __restrict__ col_idx,
    const double* __restrict__ values, const double* __restrict__ x_local,
    const double* __restrict__ x_halo_prev, const double* __restrict__ x_halo_next,
    double* __restrict__ y, int n_local, int row_offset, int N_total, int grid_size);

extern __global__ void axpy_kernel(double alpha, const double* x, double* y, int n);
extern __global__ void axpby_kernel(double alpha, const double* x, double beta, double* y, int n);

// Local helper: cuBLAS ddot
double compute_dot(cublasHandle_t h, const double* x, const double* y, int n) {
    double result;
    cublasStatus_t status = cublasDdot(h, n, x, 1, y, 1, &result);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS ddot failed\n");
        exit(EXIT_FAILURE);
    }
    return result;
}

// Extern declarations
extern struct CSRMatrix csr_mat;

// BLAS-like kernels
__global__ void axpy_kernel_impl(double alpha, const double* x, double* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = alpha * x[i] + y[i];
    }
}

__global__ void axpby_kernel_impl(double alpha, const double* x, double beta, double* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = alpha * x[i] + beta * y[i];
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s <matrix.mtx> [--verify] [--max-iters=N]\n", argv[0]);
        printf("Example: %s matrix/stencil3d_64.mtx --verify\n", argv[0]);
        printf("Options:\n");
        printf("  --verify         Use known solution (x=1) to verify correctness\n");
        printf("  --max-iters=N    Set maximum CG iterations (default: 1000)\n");
        return 1;
    }

    const char* matrix_file = argv[1];
    int verify_mode = 0;
    int max_iters = 1000;

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--verify") == 0) {
            verify_mode = 1;
        } else if (strncmp(argv[i], "--max-iters=", 12) == 0) {
            max_iters = atoi(argv[i] + 12);
        }
    }

    printf("Loading matrix: %s\n", matrix_file);
    MatrixData mat;
    if (load_matrix_market(matrix_file, &mat) != 0) {
        fprintf(stderr, "Error loading matrix\n");
        return 1;
    }

    printf("Matrix loaded: %d × %d, %lld nonzeros (grid_size=%d)\n", mat.rows, mat.cols, mat.nnz,
           mat.grid_size);

    // Setup GPU
    CUDA_CHECK(cudaSetDevice(0));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s (CC %d.%d)\n\n", prop.name, prop.major, prop.minor);

    // Create cuBLAS handle
    cublasHandle_t cublas_handle;
    if (cublasCreate(&cublas_handle) != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS initialization failed\n");
        return 1;
    }

    // Build CSR
    build_csr_struct(&mat);
    long long nnz = csr_mat.nb_nonzeros;

    // Allocate device memory
    long long* d_row_ptr;
    int* d_col_idx;
    double* d_values;
    double *d_x, *d_b, *d_r, *d_p, *d_Ap;

    CUDA_CHECK(cudaMalloc(&d_row_ptr, (mat.rows + 1) * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_col_idx, (size_t)nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_values, (size_t)nnz * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_x, mat.rows * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_b, mat.rows * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_r, mat.rows * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_p, mat.rows * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Ap, mat.rows * sizeof(double)));

    // Copy CSR to device
    CUDA_CHECK(cudaMemcpy(d_row_ptr, csr_mat.row_ptr, (mat.rows + 1) * sizeof(long long),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(d_col_idx, csr_mat.col_indices, nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_values, csr_mat.values, nnz * sizeof(double), cudaMemcpyHostToDevice));

    // Initialize vectors
    double* b = (double*)malloc(mat.rows * sizeof(double));
    double* x = (double*)calloc(mat.rows, sizeof(double));

    int threads = 256, blocks = (mat.rows + threads - 1) / threads;

    if (verify_mode) {
        // Verify mode: x_exact = ones, compute b = A * x_exact
        printf("Verify mode: x_exact = ones, computing b = A * x_exact...\n");
        double* x_exact = (double*)malloc(mat.rows * sizeof(double));
        for (int i = 0; i < mat.rows; i++) {
            x_exact[i] = 1.0;
        }
        CUDA_CHECK(cudaMemcpy(d_x, x_exact, mat.rows * sizeof(double), cudaMemcpyHostToDevice));

        // b = A * x_exact
        stencil7_csr_partitioned_halo_kernel_3d<<<blocks, threads>>>(d_row_ptr, d_col_idx, d_values,
                                                                     d_x, NULL, NULL, d_b, mat.rows,
                                                                     0, mat.rows, mat.grid_size);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy b back to host for reference
        CUDA_CHECK(cudaMemcpy(b, d_b, mat.rows * sizeof(double), cudaMemcpyDeviceToHost));
        free(x_exact);

        // Reset x to zero for the solve
        CUDA_CHECK(cudaMemset(d_x, 0, mat.rows * sizeof(double)));
    } else {
        // Standard mode: b = ones
        for (int i = 0; i < mat.rows; i++) {
            b[i] = 1.0;
        }
        CUDA_CHECK(cudaMemcpy(d_b, b, mat.rows * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_x, x, mat.rows * sizeof(double), cudaMemcpyHostToDevice));
    }

    // Initial residual: r = b - A*x0 (x0=0 so r=b)
    CUDA_CHECK(cudaMemcpy(d_r, d_b, mat.rows * sizeof(double), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_p, d_r, mat.rows * sizeof(double), cudaMemcpyDeviceToDevice));

    double rs_old = compute_dot(cublas_handle, d_r, d_r, mat.rows);
    double b_norm = sqrt(rs_old);

    double tolerance = 1e-6;

    printf("Initial residual: %.6e\n", sqrt(rs_old));
    printf("Max iterations: %d, tolerance: %.1e\n", max_iters, tolerance);
    printf("\nStarting CG iterations (single-GPU, 3D stencil)...\n");
    int iter;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    for (iter = 0; iter < max_iters; iter++) {
        // Ap = A*p (using 3D kernel, no halo needed for single-GPU)
        stencil7_csr_partitioned_halo_kernel_3d<<<blocks, threads>>>(
            d_row_ptr, d_col_idx, d_values, d_p, NULL, NULL, d_Ap, mat.rows, 0, mat.rows,
            mat.grid_size);

        // alpha = rs_old / (p^T*Ap)
        double pAp = compute_dot(cublas_handle, d_p, d_Ap, mat.rows);
        double alpha = rs_old / pAp;

        // x = x + alpha*p
        axpy_kernel_impl<<<blocks, threads>>>(alpha, d_p, d_x, mat.rows);

        // r = r - alpha*Ap
        axpy_kernel_impl<<<blocks, threads>>>(-alpha, d_Ap, d_r, mat.rows);

        // rs_new = r^T*r
        double rs_new = compute_dot(cublas_handle, d_r, d_r, mat.rows);
        double residual = sqrt(rs_new);
        double rel_residual = residual / b_norm;

        if ((iter + 1) % 10 == 0 || rel_residual < tolerance) {
            printf("[Iter %4d] Residual: %.6e (rel: %.6e)\n", iter + 1, residual, rel_residual);
        }

        if (rel_residual < tolerance) {
            iter++;
            break;
        }

        // beta = rs_new / rs_old
        double beta = rs_new / rs_old;

        // p = r + beta*p
        axpby_kernel_impl<<<blocks, threads>>>(1.0, d_r, beta, d_p, mat.rows);

        rs_old = rs_new;
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_ms;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    // Copy result back
    CUDA_CHECK(cudaMemcpy(x, d_x, mat.rows * sizeof(double), cudaMemcpyDeviceToHost));

    printf("\n========================================\n");
    printf("Converged: %s in %d iterations\n", (iter < max_iters) ? "YES" : "NO", iter);
    printf("Total time: %.2f ms\n", elapsed_ms);

    // Checksum
    double sum_x = 0.0, norm2_x = 0.0;
    for (int i = 0; i < mat.rows; i++) {
        sum_x += x[i];
        norm2_x += x[i] * x[i];
    }
    printf("Solution checksum:\n");
    printf("  Sum(x):   %.6e\n", sum_x);
    printf("  Norm2(x): %.6e\n", sqrt(norm2_x));

    // Verify mode: compare with x_exact = ones
    if (verify_mode) {
        double err_sq = 0.0, exact_sq = 0.0;
        for (int i = 0; i < mat.rows; i++) {
            double diff = x[i] - 1.0;
            err_sq += diff * diff;
            exact_sq += 1.0;
        }
        double rel_error = sqrt(err_sq) / sqrt(exact_sq);
        printf("\n=== Verification ===\n");
        printf("x_exact:        ones (%d elements)\n", mat.rows);
        printf("||x - x_exact||:     %.6e\n", sqrt(err_sq));
        printf("||x_exact||:         %.6e\n", sqrt(exact_sq));
        printf("Relative error:      %.6e\n", rel_error);
        printf("Tolerance:           %.1e\n", tolerance);
        printf("VERIFY: %s\n", rel_error < tolerance * 10.0 ? "PASS" : "FAIL");
    }

    printf("========================================\n");

    // Cleanup
    CUDA_CHECK(cudaFree(d_row_ptr));
    CUDA_CHECK(cudaFree(d_col_idx));
    CUDA_CHECK(cudaFree(d_values));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_r));
    CUDA_CHECK(cudaFree(d_p));
    CUDA_CHECK(cudaFree(d_Ap));
    cublasDestroy(cublas_handle);
    free(b);
    free(x);
    if (mat.entries)
        free(mat.entries);

    return 0;
}
