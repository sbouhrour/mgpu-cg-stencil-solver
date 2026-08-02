/**
 * @file bench_27pt_precision.cu
 * @brief Bandwidth and accuracy of the 27-point stencil SpMV at reduced coefficient precision
 *
 * @details
 * Coefficients are 90% of this kernel's DRAM traffic (216 B of 240 B per row, measured with Nsight
 * Compute on an RTX 4060 Laptop). Storing them in float halves that term while accumulation stays
 * in double. This benchmark runs both storage widths on the same matrix and reports the time, the
 * modelled traffic, and the difference between the two results.
 *
 * The kernel reads every coefficient from memory in both cases: precision here is a storage choice,
 * not knowledge of the operator's values.
 *
 * Accuracy is reported as a relative Frobenius norm, ||y_low - y_ref||_2 / ||y_ref||_2, computed in
 * double on the host. A normwise measure is used rather than a maximum element-wise ratio, which
 * reports the cancellation of small reference entries instead of the error of the kernel.
 *
 * Single GPU, one rank: x_halo_prev = x_halo_next = NULL, row_offset = 0.
 *
 * Usage: ./bin/bench_27pt_precision [matrix.mtx] [--reps=N]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include "io.h"
#include "spmv_csr.h"
#include "spmv.h"
#include "spmv_stencil_3d_27pt.h"

extern struct CSRMatrix csr_mat;

/** @brief Median of a small array of timings, sorted in place */
static double median_ms(double* v, int n) {
    for (int i = 1; i < n; i++) {
        double key = v[i];
        int j = i - 1;
        while (j >= 0 && v[j] > key) {
            v[j + 1] = v[j];
            j--;
        }
        v[j + 1] = key;
    }
    return (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

/**
 * @brief Times one launch of the kernel at storage precision ValueT
 *
 * @details The two precisions are timed alternately rather than in two blocks. Running all
 * repetitions of one variant and then all of the other lets the die heat up under the first and
 * clock down under the second, which on a laptop part is worth more than the effect being measured.
 */
template <typename ValueT>
static double time_one_launch(cudaEvent_t start, cudaEvent_t stop, const long long* d_row_ptr,
                              const int* d_col_idx, const ValueT* d_values, const double* d_x,
                              double* d_y, int n, int grid_size, int blocks, int threads) {
    cudaEventRecord(start);
    stencil27_mixed_precision_kernel_3d<ValueT><<<blocks, threads>>>(
        d_row_ptr, d_col_idx, d_values, d_x, NULL, NULL, d_y, n, 0, n, grid_size);
    cudaEventRecord(stop);
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    return (double)ms;
}

int main(int argc, char** argv) {
    const char* matrix_file = "matrix/stencil3d_27pt_192.mtx";
    int reps = 10;
    for (int a = 1; a < argc; a++) {
        if (strncmp(argv[a], "--reps=", 7) == 0)
            reps = atoi(argv[a] + 7);
        else if (argv[a][0] != '-')
            matrix_file = argv[a];
    }
    if (reps < 1)
        reps = 1;

    printf("Loading matrix: %s\n", matrix_file);
    MatrixData mat;
    memset(&mat, 0, sizeof(mat));
    if (load_matrix_stencil27_3d_from_grid(matrix_file, &mat, 0, 1) != 0) {
        fprintf(stderr, "Error loading matrix\n");
        return 1;
    }
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
    double peak_gbs = 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6;
    printf("GPU: %s (CC %d.%d, %d SMs, peak DRAM %.1f GB/s)\n", prop.name, prop.major, prop.minor,
           prop.multiProcessorCount, peak_gbs);
    printf("Matrix: %d rows, %lld nnz, grid_size=%d\n", n, nnz, grid_size);

    // ---- Device allocations: one coefficient array per storage width ----
    long long* d_row_ptr;
    int* d_col_idx;
    double *d_values64, *d_x, *d_y64, *d_y32;
    float* d_values32;
    CUDA_CHECK(cudaMalloc(&d_row_ptr, (size_t)(n + 1) * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_col_idx, (size_t)nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_values64, (size_t)nnz * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_values32, (size_t)nnz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_x, (size_t)n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_y64, (size_t)n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_y32, (size_t)n * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_row_ptr, csr_mat.row_ptr, (size_t)(n + 1) * sizeof(long long),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_idx, csr_mat.col_indices, (size_t)nnz * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_values64, csr_mat.values, (size_t)nnz * sizeof(double),
                          cudaMemcpyHostToDevice));

    float* h_values32 = (float*)malloc((size_t)nnz * sizeof(float));
    long long n_inexact = 0;
    for (long long e = 0; e < nnz; e++) {
        h_values32[e] = (float)csr_mat.values[e];
        if ((double)h_values32[e] != csr_mat.values[e])
            n_inexact++;
    }
    CUDA_CHECK(
        cudaMemcpy(d_values32, h_values32, (size_t)nnz * sizeof(float), cudaMemcpyHostToDevice));
    free(h_values32);

    // A right-hand side that is not uniform, so that a wrong column index shows up in the result
    double* h_x = (double*)malloc((size_t)n * sizeof(double));
    for (int idx = 0; idx < n; idx++)
        h_x[idx] = 1.0 + 0.5 * sin(0.001 * (double)idx);
    CUDA_CHECK(cudaMemcpy(d_x, h_x, (size_t)n * sizeof(double), cudaMemcpyHostToDevice));
    free(h_x);

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);

    for (int r = 0; r < 3; r++) {
        stencil27_mixed_precision_kernel_3d<double><<<blocks, threads>>>(
            d_row_ptr, d_col_idx, d_values64, d_x, NULL, NULL, d_y64, n, 0, n, grid_size);
        stencil27_mixed_precision_kernel_3d<float><<<blocks, threads>>>(
            d_row_ptr, d_col_idx, d_values32, d_x, NULL, NULL, d_y32, n, 0, n, grid_size);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    double* s64 = (double*)malloc((size_t)reps * sizeof(double));
    double* s32 = (double*)malloc((size_t)reps * sizeof(double));
    for (int r = 0; r < reps; r++) {
        s64[r] = time_one_launch<double>(ev_start, ev_stop, d_row_ptr, d_col_idx, d_values64, d_x,
                                         d_y64, n, grid_size, blocks, threads);
        s32[r] = time_one_launch<float>(ev_start, ev_stop, d_row_ptr, d_col_idx, d_values32, d_x,
                                        d_y32, n, grid_size, blocks, threads);
    }
    double t64 = median_ms(s64, reps);
    double t32 = median_ms(s32, reps);
    free(s64);
    free(s32);
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);

    // ---- Accuracy: relative Frobenius norm against the double-storage result ----
    double* h_y64 = (double*)malloc((size_t)n * sizeof(double));
    double* h_y32 = (double*)malloc((size_t)n * sizeof(double));
    CUDA_CHECK(cudaMemcpy(h_y64, d_y64, (size_t)n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_y32, d_y32, (size_t)n * sizeof(double), cudaMemcpyDeviceToHost));

    double num = 0.0, den = 0.0;
    long long n_bitdiff = 0;
    for (int idx = 0; idx < n; idx++) {
        double d = h_y32[idx] - h_y64[idx];
        num += d * d;
        den += h_y64[idx] * h_y64[idx];
        if (h_y32[idx] != h_y64[idx])
            n_bitdiff++;
    }
    double rel_fro = (den > 0.0) ? sqrt(num) / sqrt(den) : 0.0;

    // ---- Modelled traffic: coefficients + one amortised x read + one y write, per row ----
    double bytes64 =
        (double)nnz * sizeof(double) + (double)n * (sizeof(long long) + 2 * sizeof(double));
    double bytes32 =
        (double)nnz * sizeof(float) + (double)n * (sizeof(long long) + 2 * sizeof(double));

    printf("\n--- Coefficient storage ---\n");
    printf("Coefficients not representable in float: %lld / %lld\n", n_inexact, nnz);
    printf("\n--- Performance (alternating A/B, median of %d, 3 warmup pairs discarded) ---\n",
           reps);
    printf("%-22s %10s %12s %12s %12s\n", "values storage", "time(ms)", "B/row", "GB moved",
           "GB/s");
    printf("%-22s %10.3f %12.1f %12.3f %12.1f\n", "double", t64, bytes64 / n, bytes64 / 1e9,
           bytes64 / (t64 / 1e3) / 1e9);
    printf("%-22s %10.3f %12.1f %12.3f %12.1f\n", "float", t32, bytes32 / n, bytes32 / 1e9,
           bytes32 / (t32 / 1e3) / 1e9);
    printf("%-22s %10.3fx\n", "speedup", t64 / t32);
    printf("%-22s %10.3fx\n", "traffic ratio (model)", bytes64 / bytes32);

    printf("\n--- Accuracy vs double-storage reference ---\n");
    printf("Relative Frobenius norm: %.6e\n", rel_fro);
    printf("Entries differing at all: %lld / %d\n", n_bitdiff, n);
    if (n_inexact == 0) {
        printf(
            "NOTE: every coefficient of this matrix is exactly representable in float, so float\n");
        printf(
            "      storage is lossless here and a zero error measures nothing about precision.\n");
        printf(
            "      Measuring the numerical cost needs coefficients that float cannot represent.\n");
    }

    free(h_y64);
    free(h_y32);
    CUDA_CHECK(cudaFree(d_row_ptr));
    CUDA_CHECK(cudaFree(d_col_idx));
    CUDA_CHECK(cudaFree(d_values64));
    CUDA_CHECK(cudaFree(d_values32));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y64));
    CUDA_CHECK(cudaFree(d_y32));
    return 0;
}
