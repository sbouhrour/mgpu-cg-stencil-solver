/**
 * @file bench_27pt_precision.cu
 * @brief Layout and coefficient-precision sweep for the 3D 27-point stencil SpMV
 *
 * @details
 * Runs the same operator through six variants — row-major CSR and coefficient-major SoA, each with
 * the coefficients held in double, float, and (SoA only) half and bfloat16 — and reports time,
 * modelled traffic, and accuracy against the double-storage result of the same layout.
 *
 * Accumulation is double everywhere. Only the storage width of the coefficients changes, so the
 * measurement isolates what a narrower element costs and buys. The kernel reads every coefficient
 * from memory in all variants: precision is a storage choice, not knowledge of the operator's
 * values.
 *
 * **Timing method.** All variants are timed round-robin, one launch each per repetition, rather
 * than one variant to completion and then the next. Timing them in blocks lets the die heat up
 * under the early variants and clock down under the late ones; on a laptop part that ordering was
 * worth 27%, more than the effect being measured. The median over repetitions is reported.
 *
 * Accuracy is a relative Frobenius norm, ||y_low - y_ref||_2 / ||y_ref||_2, computed in double on
 * the host. A normwise measure is used rather than a maximum element-wise ratio, which reports the
 * cancellation of small reference entries instead of the error of the kernel.
 *
 * Single GPU, one rank: no halo, row_offset = 0.
 *
 * Usage: ./bin/bench_27pt_precision [matrix.mtx] [--reps=N] [--csv=file]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "io.h"
#include "spmv_csr.h"
#include "spmv.h"
#include "spmv_stencil_3d_27pt.h"
#include "spmv_stencil_3d_27pt_soa.h"

extern struct CSRMatrix csr_mat;

/** @brief The six layout/precision combinations under test */
enum Variant { CSR_F64 = 0, CSR_F32, SOA_F64, SOA_F32, SOA_F16, SOA_BF16, N_VARIANTS };

static const char* variant_name(int v) {
    static const char* names[N_VARIANTS] = {"CSR double", "CSR float", "SoA double",
                                            "SoA float",  "SoA half",  "SoA bfloat16"};
    return names[v];
}

/** @brief Modelled DRAM bytes per row: coefficients, one amortised x read, one y write */
static double variant_bytes_per_row(int v) {
    const double vec = 2.0 * sizeof(double);  // x amortised + y
    switch (v) {
        case CSR_F64:
            return 27.0 * sizeof(double) + sizeof(long long) + vec;
        case CSR_F32:
            return 27.0 * sizeof(float) + sizeof(long long) + vec;
        case SOA_F64:
            return 27.0 * sizeof(double) + vec;  // no row_ptr, no col_idx
        case SOA_F32:
            return 27.0 * sizeof(float) + vec;
        case SOA_F16:
            return 27.0 * sizeof(__half) + vec;
        case SOA_BF16:
            return 27.0 * sizeof(__nv_bfloat16) + vec;
        default:
            return 0.0;
    }
}

/** @brief Device-side inputs shared by the variants */
struct Buffers {
    long long* row_ptr;
    int* col_idx;
    double* csr_v64;
    float* csr_v32;
    double* x;      // n entries, CSR layout
    double* x_ext;  // n + 2*N^2 entries, SoA ghost-layer layout
    double* soa_v64;
    float* soa_v32;
    __half* soa_v16;
    __nv_bfloat16* soa_vbf;
};

/** @brief Launches one variant into d_y */
static void launch_variant(int v, const Buffers& b, double* d_y, int n, int grid_size, int blocks,
                           int threads) {
    switch (v) {
        case CSR_F64:
            stencil27_mixed_precision_kernel_3d<double><<<blocks, threads>>>(
                b.row_ptr, b.col_idx, b.csr_v64, b.x, NULL, NULL, d_y, n, 0, n, grid_size);
            break;
        case CSR_F32:
            stencil27_mixed_precision_kernel_3d<float><<<blocks, threads>>>(
                b.row_ptr, b.col_idx, b.csr_v32, b.x, NULL, NULL, d_y, n, 0, n, grid_size);
            break;
        case SOA_F64:
            stencil27_soa_halo_kernel_3d_t<double>
                <<<blocks, threads>>>(b.soa_v64, b.x_ext, d_y, n, grid_size);
            break;
        case SOA_F32:
            stencil27_soa_halo_kernel_3d_t<float>
                <<<blocks, threads>>>(b.soa_v32, b.x_ext, d_y, n, grid_size);
            break;
        case SOA_F16:
            stencil27_soa_halo_kernel_3d_t<__half>
                <<<blocks, threads>>>(b.soa_v16, b.x_ext, d_y, n, grid_size);
            break;
        case SOA_BF16:
            stencil27_soa_halo_kernel_3d_t<__nv_bfloat16>
                <<<blocks, threads>>>(b.soa_vbf, b.x_ext, d_y, n, grid_size);
            break;
        default:
            break;
    }
}

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

/** @brief ||a - ref||_2 / ||ref||_2, accumulated in double */
static double rel_frobenius(const double* a, const double* ref, int n, long long* n_diff) {
    double num = 0.0, den = 0.0;
    long long d = 0;
    for (int i = 0; i < n; i++) {
        double e = a[i] - ref[i];
        num += e * e;
        den += ref[i] * ref[i];
        if (a[i] != ref[i])
            d++;
    }
    if (n_diff)
        *n_diff = d;
    return (den > 0.0) ? sqrt(num) / sqrt(den) : 0.0;
}

int main(int argc, char** argv) {
    const char* matrix_file = "matrix/stencil3d_27pt_192.mtx";
    const char* csv_path = NULL;
    int reps = 10;
    for (int a = 1; a < argc; a++) {
        if (strncmp(argv[a], "--reps=", 7) == 0)
            reps = atoi(argv[a] + 7);
        else if (strncmp(argv[a], "--csv=", 6) == 0)
            csv_path = argv[a] + 6;
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
    const int n = csr_mat.nb_rows;
    const long long nnz = csr_mat.nb_nonzeros;
    const int grid_size = mat.grid_size;
    if (mat.entries) {
        free(mat.entries);
        mat.entries = NULL;
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    const double peak_gbs = 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6;
    printf("GPU: %s (CC %d.%d, %d SMs, peak DRAM %.1f GB/s)\n", prop.name, prop.major, prop.minor,
           prop.multiProcessorCount, peak_gbs);
    printf("Matrix: %d rows, %lld nnz, grid_size=%d\n", n, nnz, grid_size);

    const long long NN = (long long)grid_size * grid_size;
    const long long ext_n = (long long)n + 2 * NN;
    const long long n_soa = 27LL * n;

    // ---- Host-side coefficient arrays ----
    double* h_soa64 = (double*)calloc((size_t)n_soa, sizeof(double));
    if (!h_soa64) {
        fprintf(stderr, "host allocation failed for SoA coefficients\n");
        return 1;
    }
    build_values_soa_27pt_3d(csr_mat.row_ptr, csr_mat.col_indices, csr_mat.values, n, 0, grid_size,
                             h_soa64);

    long long n_inexact = 0;
    for (long long e = 0; e < nnz; e++)
        if ((double)(float)csr_mat.values[e] != csr_mat.values[e])
            n_inexact++;

    // ---- Device allocations ----
    Buffers b;
    memset(&b, 0, sizeof(b));
    double* d_y;
    CUDA_CHECK(cudaMalloc(&b.row_ptr, (size_t)(n + 1) * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&b.col_idx, (size_t)nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&b.csr_v64, (size_t)nnz * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&b.csr_v32, (size_t)nnz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b.x, (size_t)n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&b.x_ext, (size_t)ext_n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&b.soa_v64, (size_t)n_soa * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&b.soa_v32, (size_t)n_soa * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b.soa_v16, (size_t)n_soa * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&b.soa_vbf, (size_t)n_soa * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_y, (size_t)n * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(b.row_ptr, csr_mat.row_ptr, (size_t)(n + 1) * sizeof(long long),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b.col_idx, csr_mat.col_indices, (size_t)nnz * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b.csr_v64, csr_mat.values, (size_t)nnz * sizeof(double),
                          cudaMemcpyHostToDevice));
    {
        float* t = (float*)malloc((size_t)nnz * sizeof(float));
        for (long long e = 0; e < nnz; e++)
            t[e] = (float)csr_mat.values[e];
        CUDA_CHECK(cudaMemcpy(b.csr_v32, t, (size_t)nnz * sizeof(float), cudaMemcpyHostToDevice));
        free(t);
    }
    CUDA_CHECK(
        cudaMemcpy(b.soa_v64, h_soa64, (size_t)n_soa * sizeof(double), cudaMemcpyHostToDevice));
    {
        float* t32 = (float*)malloc((size_t)n_soa * sizeof(float));
        __half* t16 = (__half*)malloc((size_t)n_soa * sizeof(__half));
        __nv_bfloat16* tbf = (__nv_bfloat16*)malloc((size_t)n_soa * sizeof(__nv_bfloat16));
        for (long long e = 0; e < n_soa; e++) {
            t32[e] = (float)h_soa64[e];
            t16[e] = __double2half(h_soa64[e]);
            tbf[e] = __double2bfloat16(h_soa64[e]);
        }
        CUDA_CHECK(
            cudaMemcpy(b.soa_v32, t32, (size_t)n_soa * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(
            cudaMemcpy(b.soa_v16, t16, (size_t)n_soa * sizeof(__half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(b.soa_vbf, tbf, (size_t)n_soa * sizeof(__nv_bfloat16),
                              cudaMemcpyHostToDevice));
        free(t32);
        free(t16);
        free(tbf);
    }
    free(h_soa64);

    // A non-uniform right-hand side, so that a wrong column index shows up in the result
    double* h_x = (double*)malloc((size_t)n * sizeof(double));
    for (int i = 0; i < n; i++)
        h_x[i] = 1.0 + 0.5 * sin(0.001 * (double)i);
    CUDA_CHECK(cudaMemcpy(b.x, h_x, (size_t)n * sizeof(double), cudaMemcpyHostToDevice));
    free(h_x);
    CUDA_CHECK(cudaMemset(b.x_ext, 0, (size_t)ext_n * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(b.x_ext + NN, b.x, (size_t)n * sizeof(double), cudaMemcpyDeviceToDevice));

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    // ---- Correctness pass, before timing ----
    double* h_ref[N_VARIANTS];
    for (int v = 0; v < N_VARIANTS; v++) {
        h_ref[v] = (double*)malloc((size_t)n * sizeof(double));
        launch_variant(v, b, d_y, n, grid_size, blocks, threads);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_ref[v], d_y, (size_t)n * sizeof(double), cudaMemcpyDeviceToHost));
    }

    // ---- Round-robin timing ----
    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);
    for (int w = 0; w < 3; w++)
        for (int v = 0; v < N_VARIANTS; v++)
            launch_variant(v, b, d_y, n, grid_size, blocks, threads);
    CUDA_CHECK(cudaDeviceSynchronize());

    double* samples = (double*)malloc((size_t)N_VARIANTS * reps * sizeof(double));
    for (int r = 0; r < reps; r++) {
        for (int v = 0; v < N_VARIANTS; v++) {
            cudaEventRecord(ev0);
            launch_variant(v, b, d_y, n, grid_size, blocks, threads);
            cudaEventRecord(ev1);
            CUDA_CHECK(cudaEventSynchronize(ev1));
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, ev0, ev1);
            samples[(size_t)v * reps + r] = (double)ms;
        }
    }
    double t[N_VARIANTS];
    for (int v = 0; v < N_VARIANTS; v++)
        t[v] = median_ms(samples + (size_t)v * reps, reps);
    free(samples);
    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);

    // ---- Report ----
    printf("\n--- Layout and coefficient precision, round-robin timing, median of %d ---\n", reps);
    printf("%-14s %9s %9s %9s %9s %10s %14s\n", "variant", "time(ms)", "B/row", "GB/s", "%peak",
           "vs CSR f64", "rel Frobenius");
    long long n_diff_soa_csr = 0;
    for (int v = 0; v < N_VARIANTS; v++) {
        const double bpr = variant_bytes_per_row(v);
        const double gb = bpr * (double)n / 1e9;
        const double gbs = gb / (t[v] / 1e3);
        const int base = (v <= CSR_F32) ? CSR_F64 : SOA_F64;
        long long nd = 0;
        const double err = rel_frobenius(h_ref[v], h_ref[base], n, &nd);
        printf("%-14s %9.3f %9.1f %9.1f %8.1f%% %9.3fx %14.3e\n", variant_name(v), t[v], bpr, gbs,
               100.0 * gbs / peak_gbs, t[CSR_F64] / t[v], err);
    }
    rel_frobenius(h_ref[SOA_F64], h_ref[CSR_F64], n, &n_diff_soa_csr);
    printf("\nSoA double vs CSR double: %lld / %d entries differ%s\n", n_diff_soa_csr, n,
           n_diff_soa_csr == 0 ? " (bitwise identical)" : " -- LAYOUTS DISAGREE");

    printf("\nCoefficients not representable in float: %lld / %lld\n", n_inexact, nnz);
    if (n_inexact == 0)
        printf(
            "NOTE: every coefficient of this matrix is exact in float, so narrowing is lossless\n"
            "      here and the zero errors above measure nothing about precision. Quantifying\n"
            "      that cost needs a variable-coefficient operator.\n");

    if (csv_path) {
        FILE* f = fopen(csv_path, "w");
        if (f) {
            fprintf(f, "gpu,grid,rows,nnz,variant,time_ms,bytes_per_row,gbs,pct_peak,rel_fro\n");
            for (int v = 0; v < N_VARIANTS; v++) {
                const double bpr = variant_bytes_per_row(v);
                const double gbs = (bpr * (double)n / 1e9) / (t[v] / 1e3);
                const int base = (v <= CSR_F32) ? CSR_F64 : SOA_F64;
                fprintf(f, "\"%s\",%d,%d,%lld,\"%s\",%.4f,%.1f,%.2f,%.2f,%.6e\n", prop.name,
                        grid_size, n, nnz, variant_name(v), t[v], bpr, gbs, 100.0 * gbs / peak_gbs,
                        rel_frobenius(h_ref[v], h_ref[base], n, NULL));
            }
            fclose(f);
            printf("\nCSV written to %s\n", csv_path);
        } else {
            fprintf(stderr, "could not open %s for writing\n", csv_path);
        }
    }

    for (int v = 0; v < N_VARIANTS; v++)
        free(h_ref[v]);
    CUDA_CHECK(cudaFree(b.row_ptr));
    CUDA_CHECK(cudaFree(b.col_idx));
    CUDA_CHECK(cudaFree(b.csr_v64));
    CUDA_CHECK(cudaFree(b.csr_v32));
    CUDA_CHECK(cudaFree(b.x));
    CUDA_CHECK(cudaFree(b.x_ext));
    CUDA_CHECK(cudaFree(b.soa_v64));
    CUDA_CHECK(cudaFree(b.soa_v32));
    CUDA_CHECK(cudaFree(b.soa_v16));
    CUDA_CHECK(cudaFree(b.soa_vbf));
    CUDA_CHECK(cudaFree(d_y));
    return 0;
}
