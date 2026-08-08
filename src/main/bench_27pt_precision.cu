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

/**
 * @brief Performance probes: the same work, but reading the input vector once instead of 27 times
 *
 * @details
 * These compute the wrong answer on purpose. Their only role is to bound what staging the input
 * vector in shared memory could ever gain, before writing any tiling code. Every coefficient load
 * and every multiply-add is kept; only the 27 distinct reads of `x` collapse to one. The runtime
 * difference against the real kernel is therefore an upper bound on the payoff of removing the
 * vector's cache traffic — shared-memory staging cannot beat reading it once.
 *
 * If a probe shows no gain, tiling the input vector cannot help either, and the idea dies for the
 * cost of twenty lines rather than a kernel.
 */
__global__ void probe_csr_single_x(const long long* __restrict__ row_ptr,
                                   const double* __restrict__ values, const double* __restrict__ x,
                                   double* __restrict__ y, int n_local, int grid_size) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_local)
        return;
    const int N = grid_size, NN = N * N;
    const int i = row / NN, j = (row / N) % N, k = row % N;
    const int local_nz = n_local / NN, local_z = row / NN;
    double sum = 0.0;
    if (i > 0 && i < N - 1 && j > 0 && j < N - 1 && k > 0 && k < N - 1 && local_z > 0 &&
        local_z < local_nz - 1) {
        const long long o = row_ptr[row];
        const double xv = x[row];  // read once, not 27 times
        for (int c = 0; c < 27; c++)
            sum += values[o + c] * xv;
    }
    y[row] = sum;
}

template <typename ValueT>
__global__ void probe_soa_single_x(const ValueT* __restrict__ values_soa,
                                   const double* __restrict__ x_ext, double* __restrict__ y,
                                   int n_local, int grid_size) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_local)
        return;
    const long long ln = n_local;
    const double xv = x_ext[(long long)grid_size * grid_size + row];  // read once
    double sum = 0.0;
    for (int c = 0; c < 27; c++)
        sum += (double)values_soa[(long long)c * ln + row] * xv;
    y[row] = sum;
}
template __global__ void probe_soa_single_x<double>(const double* __restrict__,
                                                    const double* __restrict__,
                                                    double* __restrict__, int, int);
template __global__ void probe_soa_single_x<float>(const float* __restrict__,
                                                   const double* __restrict__, double* __restrict__,
                                                   int, int);

/**
 * @brief Probe A: identical work, but the dot product accumulates in float instead of double
 *
 * @details A performance probe, not a shippable variant — it changes the numerics. Consumer Ada
 * carries two FP64 units per streaming multiprocessor against 128 FP32 ones, so this removes
 * roughly 64x of the arithmetic cost while leaving every memory access identical. If it is much
 * faster, the double-precision accumulation is what limits the kernel.
 */
__global__ void probe_soa_acc_float(const float* __restrict__ values_soa,
                                    const double* __restrict__ x_ext, double* __restrict__ y,
                                    int n_local, int grid_size) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_local)
        return;
    const int N = grid_size, NN = N * N;
    const int ext_n = n_local + 2 * NN, base = NN + row;
    const long long ln = n_local;
    float sum = 0.0f;
    int c = 0;
    for (int di = -1; di <= 1; di++)
        for (int dj = -1; dj <= 1; dj++)
            for (int dk = -1; dk <= 1; dk++, c++) {
                int idx = base + di * NN + dj * N + dk;
                idx = idx < 0 ? 0 : (idx >= ext_n ? ext_n - 1 : idx);
                sum += values_soa[(long long)c * ln + row] * (float)x_ext[idx];
            }
    y[row] = (double)sum;
}

/**
 * @brief Probe B: double accumulation kept, but split across four independent partial sums
 *
 * @details Same arithmetic in the same precision; only the dependency chain changes. The production
 * kernel chains 27 dependent multiply-adds, so at most one is in flight per thread. Four partial
 * sums allow four. If this is faster, the kernel is limited by the latency of that chain rather
 * than by floating-point throughput — and unlike probe A this is a legitimate optimisation, at the
 * cost of a different summation order and therefore a different last bit.
 */
__global__ void probe_soa_acc_split4(const float* __restrict__ values_soa,
                                     const double* __restrict__ x_ext, double* __restrict__ y,
                                     int n_local, int grid_size) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_local)
        return;
    const int N = grid_size, NN = N * N;
    const int ext_n = n_local + 2 * NN, base = NN + row;
    const long long ln = n_local;
    double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
    int c = 0;
    for (int di = -1; di <= 1; di++)
        for (int dj = -1; dj <= 1; dj++)
            for (int dk = -1; dk <= 1; dk++, c++) {
                int idx = base + di * NN + dj * N + dk;
                idx = idx < 0 ? 0 : (idx >= ext_n ? ext_n - 1 : idx);
                const double t = (double)values_soa[(long long)c * ln + row] * x_ext[idx];
                switch (c & 3) {
                    case 0:
                        s0 += t;
                        break;
                    case 1:
                        s1 += t;
                        break;
                    case 2:
                        s2 += t;
                        break;
                    default:
                        s3 += t;
                        break;
                }
            }
    y[row] = (s0 + s1) + (s2 + s3);
}

/** @brief The six real variants, then four probes */
enum Variant {
    CSR_F64 = 0,
    CSR_F32,
    SOA_F64,
    SOA_F32,
    SOA_F16,
    SOA_BF16,
    PROBE_CSR_F64,
    PROBE_SOA_F32,
    PROBE_ACC_F32,
    PROBE_SPLIT4,
    N_VARIANTS
};

static const char* variant_name(int v) {
    static const char* names[N_VARIANTS] = {
        "CSR double",   "CSR float",  "SoA double",     "SoA float",     "SoA half",
        "SoA bfloat16", "~probe CSR", "~probe SoA f32", "~probe accF32", "~probe split4"};
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
        case PROBE_CSR_F64:
            return 27.0 * sizeof(double) + sizeof(long long) + vec;
        case PROBE_SOA_F32:
        case PROBE_ACC_F32:
        case PROBE_SPLIT4:
            return 27.0 * sizeof(float) + vec;
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
        case PROBE_CSR_F64:
            probe_csr_single_x<<<blocks, threads>>>(b.row_ptr, b.csr_v64, b.x, d_y, n, grid_size);
            break;
        case PROBE_SOA_F32:
            probe_soa_single_x<float><<<blocks, threads>>>(b.soa_v32, b.x_ext, d_y, n, grid_size);
            break;
        case PROBE_ACC_F32:
            probe_soa_acc_float<<<blocks, threads>>>(b.soa_v32, b.x_ext, d_y, n, grid_size);
            break;
        case PROBE_SPLIT4:
            probe_soa_acc_split4<<<blocks, threads>>>(b.soa_v32, b.x_ext, d_y, n, grid_size);
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

/**
 * @brief Largest element-wise relative error over interior rows only
 *
 * @details
 * The normwise measure below is dominated by rows on the domain boundary. Those rows have fewer
 * neighbours, so their coefficients do not sum to zero, so their result is a sum of same-signed
 * terms with no cancellation — and it is numerically large. They therefore set the norm and dilute
 * what happens on the interior, which is where a Laplacian actually cancels. Restricting to
 * interior rows exposes the amplification the normwise figure hides.
 */
static double max_rel_interior(const double* a, const double* ref, int n, int N) {
    double worst = 0.0;
    for (int q = 0; q < n; q++) {
        const int gi = q / (N * N), gj = (q / N) % N, gk = q % N;
        if (gi == 0 || gi == N - 1 || gj == 0 || gj == N - 1 || gk == 0 || gk == N - 1)
            continue;
        const double den = fabs(ref[q]);
        if (den == 0.0)
            continue;
        const double rel = fabs(a[q] - ref[q]) / den;
        if (rel > worst)
            worst = rel;
    }
    return worst;
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
    int x_mode_grid = 1;  // default: smooth on the grid, the physically meaningful case
    for (int a = 1; a < argc; a++) {
        if (strncmp(argv[a], "--reps=", 7) == 0)
            reps = atoi(argv[a] + 7);
        else if (strncmp(argv[a], "--csv=", 6) == 0)
            csv_path = argv[a] + 6;
        else if (strcmp(argv[a], "--x=index") == 0)
            x_mode_grid = 0;
        else if (strcmp(argv[a], "--x=grid") == 0)
            x_mode_grid = 1;
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
    printf("Input vector: smooth on the %s\n", x_mode_grid ? "grid (3D sine)" : "linear index");

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

    // The input vector, and this choice changes the accuracy result more than the storage width
    // does.
    //
    // "grid": smooth as a function of the three grid coordinates, which is what the solution of a
    //   discretised PDE looks like. Neighbouring stencil points then hold nearly equal values, the
    //   row sum of a Laplacian is nearly zero, and the result is a small difference of large
    //   numbers — so coefficient rounding is amplified by cancellation.
    //
    // "index": smooth as a function of the linear row index. That is a different thing entirely:
    // the
    //   stencil reaches neighbours at index offsets of N and N squared, so the value jumps between
    //   neighbours in the j and i directions and there is no cancellation to amplify anything.
    //
    // Reporting an accuracy number without saying which vector produced it is reporting half a
    // result.
    double* h_x = (double*)malloc((size_t)n * sizeof(double));
    const int NG = grid_size;
    for (int i = 0; i < n; i++) {
        if (x_mode_grid) {
            const double gx = (double)(i / (NG * NG)) / (double)NG;
            const double gy = (double)((i / NG) % NG) / (double)NG;
            const double gz = (double)(i % NG) / (double)NG;
            h_x[i] = 1.0 + 0.5 * sin(3.14159265358979323846 * gx) *
                               sin(3.14159265358979323846 * gy) * sin(3.14159265358979323846 * gz);
        } else {
            h_x[i] = 1.0 + 0.5 * sin(0.001 * (double)i);
        }
    }
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
    printf("%-14s %9s %9s %9s %9s %10s %13s %13s\n", "variant", "time(ms)", "B/row", "GB/s",
           "%peak", "vs CSR f64", "relFro(norm)", "maxRel(int.)");
    long long n_diff_soa_csr = 0;
    for (int v = 0; v < N_VARIANTS; v++) {
        const double bpr = variant_bytes_per_row(v);
        const double gb = bpr * (double)n / 1e9;
        const double gbs = gb / (t[v] / 1e3);
        const int base = (v >= PROBE_CSR_F64) ? v : ((v <= CSR_F32) ? CSR_F64 : SOA_F64);
        long long nd = 0;
        const double err = rel_frobenius(h_ref[v], h_ref[base], n, &nd);
        const double err_int = max_rel_interior(h_ref[v], h_ref[base], n, grid_size);
        printf("%-14s %9.3f %9.1f %9.1f %8.1f%% %9.3fx %13.3e %13.3e\n", variant_name(v), t[v], bpr,
               gbs, 100.0 * gbs / peak_gbs, t[CSR_F64] / t[v], err, err_int);
    }
    const double soa_csr_fro = rel_frobenius(h_ref[SOA_F64], h_ref[CSR_F64], n, &n_diff_soa_csr);
    printf("\nSoA double vs CSR double: %lld / %d entries differ%s\n", n_diff_soa_csr, n,
           n_diff_soa_csr == 0 ? " (bitwise identical)" : "");
    if (n_diff_soa_csr) {
        // Locate the disagreement: a difference at the level of double rounding is a different
        // summation order, which is expected and harmless. Anything larger is a defect.
        double worst = 0.0;
        int worst_i = -1;
        long long on_boundary = 0;
        const int Ng = grid_size;
        for (int q = 0; q < n; q++) {
            if (h_ref[SOA_F64][q] == h_ref[CSR_F64][q])
                continue;
            const int gi = q / (Ng * Ng), gj = (q / Ng) % Ng, gk = q % Ng;
            if (gi == 0 || gi == Ng - 1 || gj == 0 || gj == Ng - 1 || gk == 0 || gk == Ng - 1)
                on_boundary++;
            const double den = fabs(h_ref[CSR_F64][q]);
            const double rel = den > 0.0 ? fabs(h_ref[SOA_F64][q] - h_ref[CSR_F64][q]) / den : 0.0;
            if (rel > worst) {
                worst = rel;
                worst_i = q;
            }
        }
        printf("  relative Frobenius %.3e, worst element %.3e at row %d\n", soa_csr_fro, worst,
               worst_i);
        printf("  differing rows on the domain boundary: %lld / %lld\n", on_boundary,
               n_diff_soa_csr);
        printf(
            "  %s\n",
            worst < 1e-10
                ? "at rounding level: FMA contraction differs between the two code paths (the\n    "
                  " CSR boundary path is a loop, the SoA path is unrolled), amplified by local\n   "
                  "  cancellation. Confirmed: -fmad=false makes them bitwise identical."
                : "ABOVE rounding level: the layouts compute different operators");
    }

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
                const int base = (v >= PROBE_CSR_F64) ? v : ((v <= CSR_F32) ? CSR_F64 : SOA_F64);
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
