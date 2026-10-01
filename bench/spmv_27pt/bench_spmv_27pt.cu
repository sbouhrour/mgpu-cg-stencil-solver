/**
 * @file bench_spmv_27pt.cu
 * @brief 3D 27-point SpMV on one GPU: cuSPARSE CSR against stencil-aware kernels, same CSR arrays
 *
 * @details
 * Builds the 27-point operator directly on the device (no host copy, no file), as a standard
 * CSR matrix with columns sorted in each row, then times on that same matrix:
 *   cusparse-alg1, cusparse-alg2  cusparseSpMV, 32-bit indices, preprocessed when available
 *   rowmajor                      the kernel of the published solver (one thread per row)
 *   staged                        rowmajor with warp-coalesced cp.async staging of the values
 *   sym-tj4/8/16                  symmetric half-read kernel, 2.5D tile of 30 x (TJ-2) columns
 *   sym-pad-tj8/16                same kernel on a padded copy of the operator (explicit zeros,
 *                                 32 entries per interior row): reads one aligned 128-byte line
 *                                 per interior row; cuSPARSE stays on the unpadded matrix
 *
 * Every variant is checked against a plain CSR reference (one thread per row, CSR order) before
 * its time is reported, and the one-time pattern and symmetry check is timed separately.
 *
 * Coefficients: "const" is the solver's matrix (26 on the diagonal, -1 off it). "var" draws each
 * off-diagonal coupling from a hash of the (min, max) row pair, so the matrix stays exactly
 * symmetric while no two rows are equal, and sets the diagonal to 1 + sum |off-diagonal|.
 *
 * Timing: variants are launched round-robin, one launch each per repetition, each launch between
 * its own pair of CUDA events, and the median is reported. Model bytes are written per variant;
 * DRAM bytes actually moved come from Nsight Compute (ncu_bytes.sh), which profiles the timed
 * launches only (they run between cudaProfilerStart and cudaProfilerStop).
 *
 * Usage: bench_spmv_27pt [--sizes=128,256,384] [--coeffs=const|var] [--reps=30] [--zc=12]
 *                        [--only=variant[,variant]] [--l2fetch=32|64|128] [--csv=file]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <vector>
#include <string>
#include <algorithm>
#include <cuda_runtime.h>
#if __has_include(<cuda_profiler_api.h>)
    #include <cuda_profiler_api.h>
#else
extern "C" cudaError_t cudaProfilerStart(void);
extern "C" cudaError_t cudaProfilerStop(void);
#endif
#include <cusparse.h>
#include <cub/device/device_scan.cuh>

#include "spmv_stencil27_fast.cuh"
// The row-major kernel of the published solver, compiled from its own source
#include "spmv/spmv_stencil_3d_27pt_partitioned_halo_kernel.cu"

#define CUDA_CHECK(call)                                                                  \
    do {                                                                                  \
        cudaError_t e_ = (call);                                                          \
        if (e_ != cudaSuccess) {                                                          \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, \
                    __LINE__);                                                            \
            exit(1);                                                                      \
        }                                                                                 \
    } while (0)
#define CUSPARSE_CHECK(call)                                                              \
    do {                                                                                  \
        cusparseStatus_t s_ = (call);                                                     \
        if (s_ != CUSPARSE_STATUS_SUCCESS) {                                              \
            fprintf(stderr, "cuSPARSE error %d at %s:%d\n", (int)s_, __FILE__, __LINE__); \
            exit(1);                                                                      \
        }                                                                                 \
    } while (0)

// ---------------------------------------------------------------------------------------------
// Matrix and vector generation on the device
// ---------------------------------------------------------------------------------------------

__host__ __device__ inline uint64_t splitmix64(uint64_t z) {
    z += 0x9e3779b97f4a7c15ULL;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

/** @brief Off-diagonal coupling of rows a and b, symmetric by construction, in [-1.5, -0.5) */
__device__ inline double coupling(int a, int b) {
    const uint64_t lo = (uint64_t)min(a, b), hi = (uint64_t)max(a, b);
    return -(0.5 + (double)(splitmix64((lo << 32) ^ hi) >> 11) * 0x1.0p-53);
}

template <bool PAD> __global__ void count_kernel(long long* counts, int N) {
    const long long n = (long long)N * N * N;
    for (long long r = blockIdx.x * (long long)blockDim.x + threadIdx.x; r <= n;
         r += (long long)gridDim.x * blockDim.x)
        counts[r] = r == n ? 0 : stencil27::stencil_row<PAD>((int)r, N, [](int, int, bool) {});
}

template <bool PAD>
__global__ void fill_kernel(const long long* __restrict__ row_ptr, int* __restrict__ col,
                            double* __restrict__ val, int N, int variable) {
    const int n = N * N * N, NN = N * N;
    for (int r = blockIdx.x * blockDim.x + threadIdx.x; r < n; r += gridDim.x * blockDim.x) {
        const int i = r / NN, j = (r / N) % N, k = r % N;
        double diag = variable ? 1.0 : 26.0;
        if (variable) {
            for (int p = 0; p < 27; p++) {
                const int a = i + p / 9 - 1, b = j + (p / 3) % 3 - 1, c = k + p % 3 - 1;
                if (p != 13 && a >= 0 && a < N && b >= 0 && b < N && c >= 0 && c < N)
                    diag -= coupling(r, a * NN + b * N + c);
            }
        }
        const long long q0 = row_ptr[r];
        stencil27::stencil_row<PAD>(r, N, [&](int q, int cc, bool zero) {
            col[q0 + q] = cc;
            val[q0 + q] = zero ? 0.0 : (cc == r ? diag : (variable ? coupling(r, cc) : -1.0));
        });
    }
}

__global__ void narrow_kernel(const long long* __restrict__ in, int* __restrict__ out,
                              long long n) {
    for (long long r = blockIdx.x * (long long)blockDim.x + threadIdx.x; r < n;
         r += (long long)gridDim.x * blockDim.x)
        out[r] = (int)in[r];
}

__global__ void init_x_kernel(double* x, int n) {
    for (int r = blockIdx.x * blockDim.x + threadIdx.x; r < n; r += gridDim.x * blockDim.x)
        x[r] = (double)(splitmix64((uint64_t)r * 7 + 1) >> 11) * 0x1.0p-52 - 1.0;
}

/** @brief Reference: one thread per row, entries in CSR order */
__global__ void reference_kernel(const long long* __restrict__ row_ptr, const int* __restrict__ col,
                                 const double* __restrict__ val, const double* __restrict__ x,
                                 double* __restrict__ y, int n) {
    for (int r = blockIdx.x * blockDim.x + threadIdx.x; r < n; r += gridDim.x * blockDim.x) {
        double s = 0.0;
        for (long long q = row_ptr[r]; q < row_ptr[r + 1]; q++)
            s += val[q] * x[col[q]];
        y[r] = s;
    }
}

/** @brief max |y - ref|, max |ref| (as ordered bit patterns), and bitwise mismatches against cmp */
__global__ void compare_kernel(const double* __restrict__ y, const double* __restrict__ ref,
                               const double* __restrict__ cmp, int n, unsigned long long* out) {
    unsigned long long md = 0, mr = 0, nb = 0;
    for (int r = blockIdx.x * blockDim.x + threadIdx.x; r < n; r += gridDim.x * blockDim.x) {
        const double d = fabs(y[r] - ref[r]);
        md = max(md, (unsigned long long)__double_as_longlong(isnan(d) ? INFINITY : d));
        mr = max(mr, (unsigned long long)__double_as_longlong(fabs(ref[r])));
        if (cmp && __double_as_longlong(y[r]) != __double_as_longlong(cmp[r]))
            nb++;
    }
    atomicMax(&out[0], md);
    atomicMax(&out[1], mr);
    atomicAdd(&out[2], nb);
}

// ---------------------------------------------------------------------------------------------
// Benchmark driver
// ---------------------------------------------------------------------------------------------

struct Matrix {
    int N = 0, n = 0;
    long long nnz = 0, n_interior = 0;
    long long* rp64 = nullptr;
    int* rp32 = nullptr;
    int* col = nullptr;
    double* val = nullptr;
    bool pad = false;
};

static void build_matrix(Matrix& A, int N, bool variable, bool pad) {
    A.N = N;
    A.pad = pad;
    A.n = N * N * N;
    const long long n = A.n;
    long long* counts;
    CUDA_CHECK(cudaMalloc(&counts, (n + 1) * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&A.rp64, (n + 1) * sizeof(long long)));
    if (pad)
        count_kernel<true><<<1024, 256>>>(counts, N);
    else
        count_kernel<false><<<1024, 256>>>(counts, N);
    size_t tmp_bytes = 0;
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(nullptr, tmp_bytes, counts, A.rp64, n + 1));
    void* tmp;
    CUDA_CHECK(cudaMalloc(&tmp, tmp_bytes));
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(tmp, tmp_bytes, counts, A.rp64, n + 1));
    CUDA_CHECK(cudaFree(tmp));
    CUDA_CHECK(cudaFree(counts));
    CUDA_CHECK(cudaMemcpy(&A.nnz, A.rp64 + n, sizeof(long long), cudaMemcpyDeviceToHost));
    if (A.nnz >= (1LL << 31)) {
        fprintf(stderr, "N=%d: nnz=%lld does not fit 32-bit cuSPARSE indices\n", N, A.nnz);
        exit(1);
    }
    CUDA_CHECK(cudaMalloc(&A.col, A.nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&A.val, A.nnz * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&A.rp32, (n + 1) * sizeof(int)));
    if (pad)
        fill_kernel<true><<<4096, 256>>>(A.rp64, A.col, A.val, N, variable ? 1 : 0);
    else
        fill_kernel<false><<<4096, 256>>>(A.rp64, A.col, A.val, N, variable ? 1 : 0);
    narrow_kernel<<<1024, 256>>>(A.rp64, A.rp32, n + 1);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    A.n_interior = (long long)(N - 2) * (N - 2) * (N - 2);
}

static void free_matrix(Matrix& A) {
    cudaFree(A.rp64);
    cudaFree(A.rp32);
    cudaFree(A.col);
    cudaFree(A.val);
}

struct Variant {
    std::string name;
    double model_bytes;  // per SpMV
    bool needs_sym;
};

struct Args {
    std::vector<int> sizes{128, 256, 384};
    bool variable = false;
    int reps = 30;
    int zc = 12;
    int l2fetch = 0;
    std::vector<std::string> only;
    const char* csv = nullptr;
};

static std::vector<std::string> split(const char* s) {
    std::vector<std::string> out;
    std::string cur;
    for (; *s; ++s) {
        if (*s == ',') {
            out.push_back(cur);
            cur.clear();
        } else {
            cur += *s;
        }
    }
    if (!cur.empty())
        out.push_back(cur);
    return out;
}

template <typename K> static void print_attrs(const char* name, K kernel, int block, size_t smem) {
    cudaFuncAttributes fa;
    CUDA_CHECK(cudaFuncGetAttributes(&fa, kernel));
    int blocks = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel, block, smem));
    printf("  %-14s regs %3d  static smem %6zu  dyn smem %6zu  local %zu B  blocks/SM %d (%d "
           "warps)\n",
           name, fa.numRegs, fa.sharedSizeBytes, smem, fa.localSizeBytes, blocks,
           blocks * block / 32);
}

int main(int argc, char** argv) {
    Args args;
    for (int a = 1; a < argc; a++) {
        if (!strncmp(argv[a], "--sizes=", 8)) {
            args.sizes.clear();
            for (auto& s : split(argv[a] + 8))
                args.sizes.push_back(atoi(s.c_str()));
        } else if (!strncmp(argv[a], "--coeffs=", 9)) {
            args.variable = !strcmp(argv[a] + 9, "var");
        } else if (!strncmp(argv[a], "--reps=", 7)) {
            args.reps = atoi(argv[a] + 7);
        } else if (!strncmp(argv[a], "--zc=", 5)) {
            args.zc = atoi(argv[a] + 5);
        } else if (!strncmp(argv[a], "--l2fetch=", 10)) {
            args.l2fetch = atoi(argv[a] + 10);
        } else if (!strncmp(argv[a], "--only=", 7)) {
            args.only = split(argv[a] + 7);
        } else if (!strncmp(argv[a], "--csv=", 6)) {
            args.csv = argv[a] + 6;
        } else {
            fprintf(stderr, "unknown argument %s\n", argv[a]);
            return 1;
        }
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    int rt = 0, drv = 0;
    cudaRuntimeGetVersion(&rt);
    cudaDriverGetVersion(&drv);
    if (args.l2fetch)
        CUDA_CHECK(cudaDeviceSetLimit(cudaLimitMaxL2FetchGranularity, args.l2fetch));
    size_t l2fetch = 0;
    CUDA_CHECK(cudaDeviceGetLimit(&l2fetch, cudaLimitMaxL2FetchGranularity));
    printf("GPU %s (sm_%d%d, %d SMs, L2 %d MB), runtime %d, driver %d, cuSPARSE %d\n", prop.name,
           prop.major, prop.minor, prop.multiProcessorCount, prop.l2CacheSize >> 20, rt, drv,
           CUSPARSE_VERSION);
    printf("L2 fetch granularity limit: %zu B%s | coefficients: %s | reps %d | zc %d\n", l2fetch,
           args.l2fetch ? " (set)" : " (default)", args.variable ? "variable" : "constant",
           args.reps, args.zc);

    constexpr int kStagedWarps = 4;
    CUDA_CHECK(cudaFuncSetAttribute(stencil27::sym25d_kernel<4>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    stencil27::sym25d_smem_bytes(4)));
    CUDA_CHECK(cudaFuncSetAttribute(stencil27::sym25d_kernel<8>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    stencil27::sym25d_smem_bytes(8)));
    CUDA_CHECK(cudaFuncSetAttribute(stencil27::sym25d_kernel<16>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    stencil27::sym25d_smem_bytes(16)));
    CUDA_CHECK(cudaFuncSetAttribute(stencil27::sym25d_kernel<8, true>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    stencil27::sym25d_smem_bytes(8)));
    CUDA_CHECK(cudaFuncSetAttribute(stencil27::sym25d_kernel<16, true>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    stencil27::sym25d_smem_bytes(16)));
    printf("Kernel resources:\n");
    print_attrs("rowmajor", stencil27_csr_partitioned_halo_kernel_3d, 256, 0);
    print_attrs("staged", stencil27::csr_staged_kernel<kStagedWarps>, kStagedWarps * 32, 0);
    print_attrs("sym-tj4", stencil27::sym25d_kernel<4>, 128, stencil27::sym25d_smem_bytes(4));
    print_attrs("sym-tj8", stencil27::sym25d_kernel<8>, 256, stencil27::sym25d_smem_bytes(8));
    print_attrs("sym-tj16", stencil27::sym25d_kernel<16>, 512, stencil27::sym25d_smem_bytes(16));
    print_attrs("sym-pad-tj8", stencil27::sym25d_kernel<8, true>, 256,
                stencil27::sym25d_smem_bytes(8));
    print_attrs("sym-pad-tj16", stencil27::sym25d_kernel<16, true>, 512,
                stencil27::sym25d_smem_bytes(16));

    FILE* csv = nullptr;
    if (args.csv) {
        csv = fopen(args.csv, "a");
        if (csv && ftell(csv) == 0)
            fprintf(csv, "gpu,N,coeffs,l2fetch,zc,variant,median_ms,min_ms,model_bytes,"
                         "model_gbps,speedup_vs_best_cusparse,rel_err\n");
    }

    for (int N : args.sizes) {
        Matrix A, P;
        build_matrix(A, N, args.variable, false);
        const int n = A.n;
        const long long nnz = A.nnz;
        printf("\nN=%d: %d rows, %lld nnz (%.2f per row), interior %.1f%%\n", N, n, nnz,
               (double)nnz / n, 100.0 * A.n_interior / n);

        // One-time check of the kernels' assumptions
        unsigned long long* d_bad;
        CUDA_CHECK(cudaMalloc(&d_bad, 3 * sizeof(unsigned long long)));
        CUDA_CHECK(cudaMemset(d_bad, 0, 3 * sizeof(unsigned long long)));
        cudaEvent_t c0, c1;
        cudaEventCreate(&c0);
        cudaEventCreate(&c1);
        cudaEventRecord(c0);
        stencil27::check_pattern_sym_kernel<false>
            <<<prop.multiProcessorCount * 8, 256>>>(A.rp64, A.col, A.val, N, d_bad, d_bad + 1);
        cudaEventRecord(c1);
        CUDA_CHECK(cudaEventSynchronize(c1));
        float check_ms = 0;
        cudaEventElapsedTime(&check_ms, c0, c1);
        unsigned long long bad[3];
        CUDA_CHECK(cudaMemcpy(bad, d_bad, sizeof(bad), cudaMemcpyDeviceToHost));
        const bool pattern_ok = bad[0] == 0, sym_ok = bad[0] == 0 && bad[1] == 0;
        printf("Pattern/symmetry check: %.3f ms, bad pattern rows %llu, asymmetric pairs %llu\n",
               check_ms, bad[0], bad[1]);

        // Padded copy of the same operator, for the sym-pad variants only
        const bool want_pad =  // the padded face rows need N - 2 > 14
            N >= 17 && (args.only.empty() ||
                        std::find_if(args.only.begin(), args.only.end(), [](const std::string& s) {
                            return s.rfind("sym-pad", 0) == 0;
                        }) != args.only.end());
        bool pad_ok = false;
        if (want_pad) {
            cudaEventRecord(c0);
            build_matrix(P, N, args.variable, true);
            cudaEventRecord(c1);
            CUDA_CHECK(cudaEventSynchronize(c1));
            float build_ms = 0;
            cudaEventElapsedTime(&build_ms, c0, c1);
            CUDA_CHECK(cudaMemset(d_bad, 0, 3 * sizeof(unsigned long long)));
            cudaEventRecord(c0);
            stencil27::check_pattern_sym_kernel<true>
                <<<prop.multiProcessorCount * 8, 256>>>(P.rp64, P.col, P.val, N, d_bad, d_bad + 1);
            cudaEventRecord(c1);
            CUDA_CHECK(cudaEventSynchronize(c1));
            float pcheck_ms = 0;
            cudaEventElapsedTime(&pcheck_ms, c0, c1);
            unsigned long long pbad[3];
            CUDA_CHECK(cudaMemcpy(pbad, d_bad, sizeof(pbad), cudaMemcpyDeviceToHost));
            pad_ok = pbad[0] == 0 && pbad[1] == 0;
            const double bytes_a = 12.0 * A.nnz + 8.0 * (n + 1),
                         bytes_p = 12.0 * P.nnz + 8.0 * (n + 1);
            printf("Padded matrix: %lld nnz (+%.1f%%), CSR bytes +%.1f%%, built on the GPU in %.3f "
                   "ms; "
                   "check %.3f ms, bad pattern rows %llu, asymmetric pairs %llu\n",
                   P.nnz, 100.0 * (P.nnz - A.nnz) / A.nnz, 100.0 * (bytes_p - bytes_a) / bytes_a,
                   build_ms, pcheck_ms, pbad[0], pbad[1]);
        }

        // Vectors
        double *x, *yref, *y, *yrow;
        CUDA_CHECK(cudaMalloc(&x, n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&yref, n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&y, n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&yrow, n * sizeof(double)));
        init_x_kernel<<<1024, 256>>>(x, n);
        reference_kernel<<<(n + 255) / 256, 256>>>(A.rp64, A.col, A.val, x, yref, n);
        CUDA_CHECK(cudaDeviceSynchronize());

        // cuSPARSE
        cusparseHandle_t handle;
        CUSPARSE_CHECK(cusparseCreate(&handle));
        cusparseSpMatDescr_t matA;
        cusparseDnVecDescr_t vecX, vecY;
        CUSPARSE_CHECK(cusparseCreateCsr(&matA, n, n, nnz, A.rp32, A.col, A.val, CUSPARSE_INDEX_32I,
                                         CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
        CUSPARSE_CHECK(cusparseCreateDnVec(&vecX, n, x, CUDA_R_64F));
        CUSPARSE_CHECK(cusparseCreateDnVec(&vecY, n, y, CUDA_R_64F));
        const double one = 1.0, zero = 0.0;
        const cusparseSpMVAlg_t algs[2] = {CUSPARSE_SPMV_CSR_ALG1, CUSPARSE_SPMV_CSR_ALG2};
        void* sp_buf[2] = {nullptr, nullptr};
        for (int a = 0; a < 2; a++) {
            size_t bytes = 0;
            CUSPARSE_CHECK(cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one,
                                                   matA, vecX, &zero, vecY, CUDA_R_64F, algs[a],
                                                   &bytes));
            CUDA_CHECK(cudaMalloc(&sp_buf[a], bytes ? bytes : 1));
        }
#if defined(CUDART_VERSION) && CUDART_VERSION >= 12080
        // ALG2 builds its row partition here; with one matrix per descriptor, only once
        CUSPARSE_CHECK(cusparseSpMV_preprocess(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matA,
                                               vecX, &zero, vecY, CUDA_R_64F, algs[1], sp_buf[1]));
#endif

        // Variants
        const double vec = 16.0 * n;  // x once, y once
        const double face_nnz = (double)(nnz - 27 * A.n_interior);
        std::vector<Variant> vars = {
            {"cusparse-alg1", 12.0 * nnz + 4.0 * (n + 1) + vec, false},
            {"cusparse-alg2", 12.0 * nnz + 4.0 * (n + 1) + vec, false},
            {"rowmajor", 8.0 * nnz + 8.0 * (n + 1) + vec, false},
            {"staged", 8.0 * nnz + 8.0 * (n + 1) + vec, false},
            {"sym-tj4", 136.0 * A.n_interior + 12.0 * face_nnz + vec, true},
            {"sym-tj8", 136.0 * A.n_interior + 12.0 * face_nnz + vec, true},
            {"sym-tj16", 136.0 * A.n_interior + 12.0 * face_nnz + vec, true},
            {"sym-pad-tj8", 128.0 * A.n_interior + 12.0 * face_nnz + vec, true},
            {"sym-pad-tj16", 128.0 * A.n_interior + 12.0 * face_nnz + vec, true},
        };
        std::vector<int> active;
        for (int v = 0; v < (int)vars.size(); v++) {
            if (!args.only.empty() &&
                std::find(args.only.begin(), args.only.end(), vars[v].name) == args.only.end())
                continue;
            if (v >= 3 && !pattern_ok)
                continue;
            if (vars[v].needs_sym && !sym_ok)
                continue;
            if (v >= 7 && !pad_ok)
                continue;
            active.push_back(v);
        }

        auto launch = [&](int v) {
            switch (v) {
                case 0:
                case 1:
                    CUSPARSE_CHECK(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one,
                                                matA, vecX, &zero, vecY, CUDA_R_64F, algs[v],
                                                sp_buf[v]));
                    break;
                case 2:
                    stencil27_csr_partitioned_halo_kernel_3d<<<(n + 255) / 256, 256>>>(
                        A.rp64, A.col, A.val, x, nullptr, nullptr, y, n, 0, n, N);
                    break;
                case 3: {
                    const int rows_per_block = kStagedWarps * 32;
                    stencil27::csr_staged_kernel<kStagedWarps>
                        <<<(n + rows_per_block - 1) / rows_per_block, rows_per_block>>>(
                            A.rp64, A.col, A.val, x, y, N);
                    break;
                }
                case 7:
                case 8: {
                    const int tj = v == 7 ? 8 : 16;
                    const dim3 grid((N + 29) / 30, (N + tj - 3) / (tj - 2),
                                    (N + args.zc - 1) / args.zc);
                    const size_t sm = stencil27::sym25d_smem_bytes(tj);
                    if (tj == 8)
                        stencil27::sym25d_kernel<8, true>
                            <<<grid, 256, sm>>>(P.rp64, P.col, P.val, x, y, N, args.zc);
                    else
                        stencil27::sym25d_kernel<16, true>
                            <<<grid, 512, sm>>>(P.rp64, P.col, P.val, x, y, N, args.zc);
                    break;
                }
                default: {
                    const int tj = v == 4 ? 4 : (v == 5 ? 8 : 16);
                    const dim3 grid((N + 29) / 30, (N + tj - 3) / (tj - 2),
                                    (N + args.zc - 1) / args.zc);
                    const size_t sm = stencil27::sym25d_smem_bytes(tj);
                    if (tj == 4)
                        stencil27::sym25d_kernel<4>
                            <<<grid, 128, sm>>>(A.rp64, A.col, A.val, x, y, N, args.zc);
                    else if (tj == 8)
                        stencil27::sym25d_kernel<8>
                            <<<grid, 256, sm>>>(A.rp64, A.col, A.val, x, y, N, args.zc);
                    else
                        stencil27::sym25d_kernel<16>
                            <<<grid, 512, sm>>>(A.rp64, A.col, A.val, x, y, N, args.zc);
                }
            }
        };

        // Correctness first: each variant against the CSR reference
        CUDA_CHECK(cudaMemset(yrow, 0, n * sizeof(double)));
        launch(2);
        CUDA_CHECK(cudaMemcpy(yrow, y, n * sizeof(double), cudaMemcpyDeviceToDevice));
        unsigned long long* d_cmp;
        CUDA_CHECK(cudaMalloc(&d_cmp, 3 * sizeof(unsigned long long)));
        std::vector<double> rel_err(vars.size(), NAN);
        std::vector<unsigned long long> diff_bits(vars.size(), 0);
        bool all_ok = true;
        for (int v : active) {
            CUDA_CHECK(cudaMemset(y, 0xff, n * sizeof(double)));  // NaN everywhere
            launch(v);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemset(d_cmp, 0, 3 * sizeof(unsigned long long)));
            compare_kernel<<<1024, 256>>>(y, yref, yrow, n, d_cmp);
            unsigned long long h[3];
            CUDA_CHECK(cudaMemcpy(h, d_cmp, sizeof(h), cudaMemcpyDeviceToHost));
            double md, mr;
            memcpy(&md, &h[0], 8);
            memcpy(&mr, &h[1], 8);
            rel_err[v] = md / mr;
            diff_bits[v] = h[2];
            if (!(rel_err[v] < 1e-12)) {
                printf("  FAIL %-14s max|y-ref|/max|ref| = %.3e\n", vars[v].name.c_str(),
                       rel_err[v]);
                all_ok = false;
            }
        }
        if (!all_ok) {
            fprintf(stderr, "Validation failed at N=%d, no timing reported\n", N);
            return 2;
        }

        // Timing, round-robin
        const int R = args.reps;
        std::vector<cudaEvent_t> ev(2 * R * vars.size());
        for (auto& e : ev)
            cudaEventCreate(&e);
        for (int v : active)
            for (int w = 0; w < 3; w++)
                launch(v);
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaProfilerStart();  // ncu --profile-from-start off sees the timed launches only
        for (int rep = 0; rep < R; rep++) {
            for (int v : active) {
                cudaEventRecord(ev[2 * (rep * vars.size() + v)]);
                launch(v);
                cudaEventRecord(ev[2 * (rep * vars.size() + v) + 1]);
            }
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        cudaProfilerStop();
        CUDA_CHECK(cudaGetLastError());
        std::vector<double> med(vars.size(), NAN), mn(vars.size(), NAN);
        for (int v : active) {
            std::vector<float> t(R);
            for (int rep = 0; rep < R; rep++)
                cudaEventElapsedTime(&t[rep], ev[2 * (rep * vars.size() + v)],
                                     ev[2 * (rep * vars.size() + v) + 1]);
            std::sort(t.begin(), t.end());
            med[v] = R % 2 ? t[R / 2] : 0.5 * (t[R / 2 - 1] + t[R / 2]);
            mn[v] = t[0];
        }
        double best_sp = INFINITY;
        for (int v : active)
            if (v < 2)
                best_sp = std::min(best_sp, med[v]);
        const double t_row = med[2];

        printf("%-14s %10s %10s %9s %9s %9s %9s %10s %s\n", "variant", "median ms", "min ms",
               "model B/r", "model GB/s", "vs cuSP", "vs rowmaj", "rel err", "bits != rowmajor");
        for (int v : active) {
            const double gbps = vars[v].model_bytes / (med[v] * 1e-3) / 1e9;
            printf("%-14s %10.4f %10.4f %9.1f %9.0f %8.2fx %8.2fx %10.2e %llu\n",
                   vars[v].name.c_str(), med[v], mn[v], vars[v].model_bytes / n, gbps,
                   best_sp / med[v], t_row / med[v], rel_err[v], diff_bits[v]);
            if (csv)
                fprintf(csv, "\"%s\",%d,%s,%zu,%d,%s,%.5f,%.5f,%.0f,%.1f,%.4f,%.3e\n", prop.name, N,
                        args.variable ? "var" : "const", l2fetch, args.zc, vars[v].name.c_str(),
                        med[v], mn[v], vars[v].model_bytes, gbps, best_sp / med[v], rel_err[v]);
        }
        double best = INFINITY;
        for (int v : active)
            best = std::min(best, med[v]);
        printf("One-time check = %.2f SpMV of the fastest variant\n", check_ms / best);

        for (auto& e : ev)
            cudaEventDestroy(e);
        cudaFree(sp_buf[0]);
        cudaFree(sp_buf[1]);
        cusparseDestroySpMat(matA);
        cusparseDestroyDnVec(vecX);
        cusparseDestroyDnVec(vecY);
        cusparseDestroy(handle);
        cudaFree(d_cmp);
        cudaFree(d_bad);
        cudaFree(x);
        cudaFree(y);
        cudaFree(yref);
        cudaFree(yrow);
        free_matrix(A);
        if (want_pad)
            free_matrix(P);
    }
    if (csv)
        fclose(csv);
    return 0;
}
