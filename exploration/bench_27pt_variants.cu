/**
 * @file bench_27pt_variants.cu
 * @brief Correctness gate + NCU launch harness for 27-point SpMV variants.
 *
 * Baseline: production kernel stencil27_csr_partitioned_halo_kernel_3d, linked
 * UNMODIFIED from src/spmv/spmv_stencil_3d_27pt_partitioned_halo_kernel.cu.
 * Optimization variants are registered in VARIANTS[] as the exploration
 * progresses; each must match the baseline output (rel diff < 1e-12 per row,
 * blocking) before any measurement is taken.
 *
 * Single-GPU config: 1 rank, x_halo_prev = x_halo_next = NULL, row_offset = 0.
 * Matrix loaded via load_matrix_stencil27_3d_from_grid (header-only read,
 * entries generated in memory), then build_csr_struct — both reused as-is.
 * Input x: x[i] = sin(i * 0.001) (deterministic, non-trivial; same as the
 * May 2026 profiling runs for comparability).
 *
 * Modes:
 *   --profile : one launch per registered kernel (for ncu -k targeting), exit.
 *   (default) : correctness gate vs baseline, then median timing per kernel
 *               (CUDA events; medians/deltas only — absolute times are not
 *               trustworthy on this laptop GPU, see exploration notes).
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

// Production kernel (unmodified), linked from
// src/spmv/spmv_stencil_3d_27pt_partitioned_halo_kernel.cu
__global__ void stencil27_csr_partitioned_halo_kernel_3d(
    const long long* __restrict__ row_ptr, const int* __restrict__ col_idx,
    const double* __restrict__ values, const double* __restrict__ x_local,
    const double* __restrict__ x_halo_prev, const double* __restrict__ x_halo_next,
    double* __restrict__ y, int n_local, int row_offset, int N_total, int grid_size);

// SoA variant launchers, defined in exploration/stencil27_soa_kernel.cu
void launch_stencil27_soa(int blocks, int threads, const double* values_soa, const double* x,
                          double* y, int n, int N);
void launch_stencil27_soa_ldcs(int blocks, int threads, const double* values_soa, const double* x,
                               double* y, int n, int N);

// ---------------------------------------------------------------------------
// Variant registry
// ---------------------------------------------------------------------------

struct DeviceData {
    const long long* row_ptr;
    const int* col_idx;
    const double* values;
    const double* values_soa;  // [27][n] coefficient-major, 0.0-padded
    const double* x;
    double* y;
    int n;  // rows (= N^3)
    int N;  // grid dimension
};

typedef void (*LaunchFn)(const DeviceData&);

static void launch_baseline(const DeviceData& d) {
    const int threads = 256;
    const int blocks = (d.n + threads - 1) / threads;
    stencil27_csr_partitioned_halo_kernel_3d<<<blocks, threads>>>(
        d.row_ptr, d.col_idx, d.values, d.x, NULL, NULL, d.y, d.n, 0, d.n, d.N);
}

static void launch_soa(const DeviceData& d) {
    const int threads = 256;
    const int blocks = (d.n + threads - 1) / threads;
    launch_stencil27_soa(blocks, threads, d.values_soa, d.x, d.y, d.n, d.N);
}

static void launch_soa_b128(const DeviceData& d) {
    const int threads = 128;
    const int blocks = (d.n + threads - 1) / threads;
    launch_stencil27_soa(blocks, threads, d.values_soa, d.x, d.y, d.n, d.N);
}

static void launch_soa_b512(const DeviceData& d) {
    const int threads = 512;
    const int blocks = (d.n + threads - 1) / threads;
    launch_stencil27_soa(blocks, threads, d.values_soa, d.x, d.y, d.n, d.N);
}

static void launch_soa_ldcs(const DeviceData& d) {
    const int threads = 256;
    const int blocks = (d.n + threads - 1) / threads;
    launch_stencil27_soa_ldcs(blocks, threads, d.values_soa, d.x, d.y, d.n, d.N);
}

struct Variant {
    const char* name;
    LaunchFn launch;
};

// Index 0 is the reference; later entries are validated against it.
// NCU --profile launch order for -k regex:stencil27_soa filtering:
//   launch 0 = soa@256, 1 = soa@128, 2 = soa@512, 3 = soa_ldcs@256.
static const Variant VARIANTS[] = {
    {"baseline (production kernel)", launch_baseline},
    {"soa (coefficient-major values)", launch_soa},
    {"soa, block 128", launch_soa_b128},
    {"soa, block 512", launch_soa_b512},
    {"soa + __ldcs on values", launch_soa_ldcs},
};
static const int N_VARIANTS = (int)(sizeof(VARIANTS) / sizeof(VARIANTS[0]));

// ---------------------------------------------------------------------------
// CSR -> coefficient-major (SoA) transform, 0.0-padded
// ---------------------------------------------------------------------------
// values_soa[c * n + row] = coefficient of neighbor offset offs[c] for `row`,
// with offs[] the 27 stencil offsets in ascending order (same order as the
// ascending-column CSR rows). Missing neighbors (boundary rows) stay 0.0.
// Also re-validates that every CSR entry maps to a stencil offset and that
// per-row columns are sorted ascending (both assumed by the production
// kernel's fast path).
static double* build_values_soa_host(int n, int N) {
    long long offs[27];
    int c = 0;
    for (int di = -1; di <= 1; di++)
        for (int dj = -1; dj <= 1; dj++)
            for (int dk = -1; dk <= 1; dk++)
                offs[c++] = (long long)di * N * N + (long long)dj * N + dk;

    double* soa = (double*)calloc((size_t)27 * n, sizeof(double));
    if (!soa) {
        fprintf(stderr, "calloc of values_soa (%zu bytes) failed\n",
                (size_t)27 * n * sizeof(double));
        exit(EXIT_FAILURE);
    }
    for (int r = 0; r < n; r++) {
        int ci = 0;
        for (long long e = csr_mat.row_ptr[r]; e < csr_mat.row_ptr[r + 1]; e++) {
            long long delta = (long long)csr_mat.col_indices[e] - r;
            while (ci < 27 && offs[ci] < delta)
                ci++;
            if (ci >= 27 || offs[ci] != delta) {
                fprintf(stderr,
                        "SoA transform: row %d entry has non-stencil/unsorted "
                        "column (delta %lld)\n",
                        r, delta);
                exit(EXIT_FAILURE);
            }
            soa[(size_t)ci * n + r] = csr_mat.values[e];
            ci++;
        }
    }
    return soa;
}

// ---------------------------------------------------------------------------

static double median_of(double* a, int n) {
    std::sort(a, a + n);
    if (n % 2)
        return a[n / 2];
    return 0.5 * (a[n / 2 - 1] + a[n / 2]);
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

    printf("Loading matrix: %s\n", matrix_file);
    MatrixData mat;
    memset(&mat, 0, sizeof(mat));
    if (load_matrix_stencil27_3d_from_grid(matrix_file, &mat, 0, 1) != 0) {
        fprintf(stderr, "Error loading matrix\n");
        return 1;
    }
    printf("Matrix: %d x %d, %lld nnz, grid_size=%d\n", mat.rows, mat.cols, mat.nnz, mat.grid_size);

    if (build_csr_struct(&mat) != EXIT_SUCCESS) {
        fprintf(stderr, "build_csr_struct failed\n");
        return 1;
    }
    const int n = csr_mat.nb_rows;
    const long long nnz = csr_mat.nb_nonzeros;
    const int N = mat.grid_size;
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
    double *d_values, *d_x, *d_y_ref, *d_y_cur;
    CUDA_CHECK(cudaMalloc(&d_row_ptr, (size_t)(n + 1) * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_col_idx, (size_t)nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_values, (size_t)nnz * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_x, (size_t)n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_y_ref, (size_t)n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_y_cur, (size_t)n * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_row_ptr, csr_mat.row_ptr, (size_t)(n + 1) * sizeof(long long),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_idx, csr_mat.col_indices, (size_t)nnz * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(d_values, csr_mat.values, (size_t)nnz * sizeof(double), cudaMemcpyHostToDevice));

    double* h_x = (double*)malloc((size_t)n * sizeof(double));
    for (int idx = 0; idx < n; idx++)
        h_x[idx] = sin(idx * 0.001);
    CUDA_CHECK(cudaMemcpy(d_x, h_x, (size_t)n * sizeof(double), cudaMemcpyHostToDevice));
    free(h_x);

    // SoA values (variant input), built from the validated host CSR
    printf("Building coefficient-major (SoA) values, 27 x %d doubles...\n", n);
    double* h_soa = build_values_soa_host(n, N);
    double* d_values_soa;
    CUDA_CHECK(cudaMalloc(&d_values_soa, (size_t)27 * n * sizeof(double)));
    CUDA_CHECK(
        cudaMemcpy(d_values_soa, h_soa, (size_t)27 * n * sizeof(double), cudaMemcpyHostToDevice));
    free(h_soa);

    DeviceData dd_ref = {d_row_ptr, d_col_idx, d_values, d_values_soa, d_x, d_y_ref, n, N};
    DeviceData dd_cur = {d_row_ptr, d_col_idx, d_values, d_values_soa, d_x, d_y_cur, n, N};

    printf("Registered variants: %d\n", N_VARIANTS);
    for (int v = 0; v < N_VARIANTS; v++)
        printf("  [%d] %s\n", v, VARIANTS[v].name);

    if (profile_mode) {
        // One launch each: ncu -k <kernel name regex> profiles exactly one.
        for (int v = 0; v < N_VARIANTS; v++) {
            VARIANTS[v].launch(v == 0 ? dd_ref : dd_cur);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        printf("Profile mode: one launch per variant issued.\n");
        goto cleanup;
    }

    // ===================== Correctness gate (BLOCKING) =======================
    VARIANTS[0].launch(dd_ref);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    {
        double* h_ref = (double*)malloc((size_t)n * sizeof(double));
        double* h_cur = (double*)malloc((size_t)n * sizeof(double));
        CUDA_CHECK(cudaMemcpy(h_ref, d_y_ref, (size_t)n * sizeof(double), cudaMemcpyDeviceToHost));

        int all_pass = 1;
        for (int v = 1; v < N_VARIANTS; v++) {
            CUDA_CHECK(cudaMemset(d_y_cur, 0, (size_t)n * sizeof(double)));
            VARIANTS[v].launch(dd_cur);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(
                cudaMemcpy(h_cur, d_y_cur, (size_t)n * sizeof(double), cudaMemcpyDeviceToHost));

            double max_rel = 0.0, max_abs = 0.0;
            long long worst_row = -1, n_disagree = 0;
            for (int idx = 0; idx < n; idx++) {
                double a = h_ref[idx], b = h_cur[idx];
                double absd = fabs(a - b);
                double denom = fabs(a) > 1e-300 ? fabs(a) : 1.0;
                double rel = absd / denom;
                if (rel > 1e-12)
                    n_disagree++;
                if (rel > max_rel) {
                    max_rel = rel;
                    worst_row = idx;
                }
                if (absd > max_abs)
                    max_abs = absd;
            }
            printf("\n===== CORRECTNESS [%d] %s =====\n", v, VARIANTS[v].name);
            printf("rows disagree : %lld (rel diff > 1e-12)\n", n_disagree);
            printf("max abs diff  : %.3e\n", max_abs);
            printf("max rel diff  : %.3e (row %lld)\n", max_rel, worst_row);
            if (max_rel >= 1e-12 || n_disagree > 0) {
                printf("CHECKSUM FAILED for variant %d.\n", v);
                all_pass = 0;
            } else {
                printf("CHECKSUM PASSED.\n");
            }
        }
        free(h_ref);
        free(h_cur);
        if (!all_pass) {
            printf("\nAt least one variant FAILED — skipping perf measurement.\n");
            goto cleanup;
        }
        if (N_VARIANTS == 1)
            printf("\nNo variants beyond baseline registered; timing baseline only.\n");
    }

    // ===================== Median timing per variant =========================
    {
        const int N_WARMUP = 3, N_RUNS = 10;
        printf("\n===== TIMING (median of %d, %d warmups dropped) =====\n", N_RUNS, N_WARMUP);
        printf("Reminder: medians/deltas only — DVFS makes absolute times unreliable "
               "on this GPU.\n");
        for (int v = 0; v < N_VARIANTS; v++) {
            const DeviceData& dd = (v == 0) ? dd_ref : dd_cur;
            double t[64];
            for (int w = 0; w < N_WARMUP; w++)
                VARIANTS[v].launch(dd);
            CUDA_CHECK(cudaDeviceSynchronize());
            for (int r = 0; r < N_RUNS; r++) {
                cudaEvent_t s, e;
                CUDA_CHECK(cudaEventCreate(&s));
                CUDA_CHECK(cudaEventCreate(&e));
                CUDA_CHECK(cudaEventRecord(s));
                VARIANTS[v].launch(dd);
                CUDA_CHECK(cudaEventRecord(e));
                CUDA_CHECK(cudaEventSynchronize(e));
                float ms = 0.f;
                CUDA_CHECK(cudaEventElapsedTime(&ms, s, e));
                t[r] = ms;
                CUDA_CHECK(cudaEventDestroy(s));
                CUDA_CHECK(cudaEventDestroy(e));
            }
            printf("[%d] %-36s median %.4f ms\n", v, VARIANTS[v].name, median_of(t, N_RUNS));
        }
    }

cleanup:
    CUDA_CHECK(cudaFree(d_row_ptr));
    CUDA_CHECK(cudaFree(d_col_idx));
    CUDA_CHECK(cudaFree(d_values));
    CUDA_CHECK(cudaFree(d_values_soa));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y_ref));
    CUDA_CHECK(cudaFree(d_y_cur));
    return 0;
}
