/**
 * @file spmv_stencil27_fast.cuh
 * @brief 3D 27-point stencil SpMV kernels that read a standard CSR matrix faster than row-major
 *
 * @details
 * Both kernels take the CSR arrays as they are (row_ptr, col_idx, values, columns sorted in each
 * row). Nothing is converted, copied or reordered: what changes is which bytes are read and how.
 *
 * Grid convention: row r = (i * N + j) * N + k, with i the slowest (z-plane) index. A row is
 * interior when 0 < i, j, k < N - 1; it then holds exactly 27 entries, in ascending column order,
 * so entry p (0..26) couples r to the neighbour at offset (p / 9 - 1, (p / 3) % 3 - 1, p % 3 - 1).
 * The order is antisymmetric: entry 26 - p points in the opposite direction to entry p. Rows on
 * the domain faces keep the truncated pattern and go through a plain CSR loop.
 *
 * 1. csr_staged_kernel: one thread per row, as the row-major kernel, but each warp first copies
 *    the contiguous value span of its 32 rows into shared memory with 16-byte cp.async, so every
 *    global load is coalesced. Same arithmetic, same summation order as the row-major kernel.
 *
 * 2. sym25d_kernel: for a symmetric matrix, A(r, r + d) = A(r + d, r). Each interior row stores
 *    its 13 "upper" couplings (positive offsets) plus the diagonal in entries 13..26, one
 *    contiguous run of 14 doubles; the 13 "lower" entries 0..12 duplicate couplings already held
 *    by the upper half of another row. The kernel reads entries 13..26 only and recovers each
 *    lower term from the row that stores it: a thread multiplies its upper coefficient by its own
 *    x and hands the product to the neighbour (warp shuffles along k, shared memory along j, and a
 *    register carried to the next plane along i). Blocks march along i over a 2D (j, k) tile, so
 *    the neighbour that needs a product is always in the same block, one plane later at most.
 *    Interior rows next to a face cannot receive products from the truncated face rows; they read
 *    the few lower entries concerned from their own row instead.
 *
 * Requirements checked once by check_pattern_sym_kernel: interior rows hold the 27-entry pattern,
 * and (for sym25d_kernel) values are bitwise symmetric between interior rows. CG requires a
 * symmetric matrix in any case. values must be 16-byte aligned (cudaMalloc guarantees 256).
 *
 * Author: Bouhrour Stephane
 */

#ifndef SPMV_STENCIL27_FAST_CUH
#define SPMV_STENCIL27_FAST_CUH

#include <cuda_runtime.h>

namespace stencil27 {

constexpr unsigned kFull = 0xffffffffu;

/** @brief Doubles of shared memory per warp for the staged kernel: 32 rows x 27, rounded to 16 B */
constexpr int kStage1 = 896;

/** @brief Asynchronous 16-byte global to shared copy that bypasses L1 (sm_80+) */
__device__ __forceinline__ void cp_async16(double* smem, const double* gmem) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem) : "memory");
#else
    smem[0] = gmem[0];
    smem[1] = gmem[1];
#endif
}

/** @brief Waits for every cp.async issued by this thread (visibility to the warp needs a sync) */
__device__ __forceinline__ void cp_async_wait_all() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;\n" ::: "memory");
#endif
}

/** @brief Linear offset of stencil entry p in an interior row */
__host__ __device__ __forceinline__ int entry_offset(int p, int N) {
    return (p / 9 - 1) * N * N + ((p / 3) % 3 - 1) * N + (p % 3 - 1);
}

/** @brief Interior row: 27 terms in ascending column order, the row-major kernel's exact order */
__device__ __forceinline__ double row27(const double* v, const double* __restrict__ x, int row,
                                        int N) {
    double sum = v[0] * x[row + entry_offset(0, N)];
#pragma unroll
    for (int p = 1; p < 27; p++)
        sum += v[p] * x[row + entry_offset(p, N)];
    return sum;
}

/** @brief Face row: plain CSR loop over its own entries */
__device__ __forceinline__ double row_generic(const double* v, const int* __restrict__ cols,
                                              int len, const double* __restrict__ x) {
    double sum = 0.0;
    for (int q = 0; q < len; q++)
        sum += v[q] * x[cols[q]];
    return sum;
}

/**
 * @brief Face row of the truncated stencil, columns from the geometry, same order as the CSR row
 *
 * @details Valid neighbours are stored in ascending column order, that is in stencil order p,
 * so entry q of the row is the q-th valid p. Columns need no col_idx load: values and x loads
 * are independent and issued in batches of UNROLL positions. The pattern of face rows is
 * checked once by check_pattern_sym_kernel.
 */
template <int UNROLL>
__device__ __forceinline__ double row_face(const double* __restrict__ v,
                                           const double* __restrict__ x, int r, int i, int j,
                                           int k, int N) {
    double sum = 0.0;
    int q = 0;
#pragma unroll UNROLL
    for (int p = 0; p < 27; p++) {
        const bool ok = (unsigned)(i + p / 9 - 1) < (unsigned)N &&
                        (unsigned)(j + (p / 3) % 3 - 1) < (unsigned)N &&
                        (unsigned)(k + p % 3 - 1) < (unsigned)N;
        if (ok) {
            sum += v[q] * x[r + entry_offset(p, N)];
            q++;
        }
    }
    return sum;
}

/**
 * @brief Row-major CSR SpMV with warp-coalesced staging of the coefficients
 *
 * @details One warp = 32 consecutive rows, whose values form one contiguous span of the CSR
 * values array. The warp copies that span to shared memory with 16-byte cp.async (each lane
 * copies consecutive chunks, so each instruction moves 512 contiguous bytes), then every thread
 * reads its own 27 coefficients from shared memory. The stride-27 shared-memory reads are free
 * of bank conflicts (27 is odd). x is read through L1, which the coefficients no longer evict.
 *
 * Launch: blockDim = WARPS * 32, gridDim = ceil(N^3 / (WARPS * 32)).
 */
template <int WARPS>
__global__ void __launch_bounds__(WARPS * 32)
    csr_staged_kernel(const long long* __restrict__ row_ptr, const int* __restrict__ col_idx,
                      const double* __restrict__ values, const double* __restrict__ x,
                      double* __restrict__ y, int N) {
    __shared__ __align__(16) double stage[WARPS][kStage1];
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int n = N * N * N;
    const int row0 = (blockIdx.x * WARPS + w) * 32;
    if (row0 >= n)
        return;  // whole warp
    const int row = row0 + lane;
    const bool valid = row < n;
    const int last = min(row0 + 31, n - 1);
    const long long rp = valid ? row_ptr[row] : 0;
    const long long base = __shfl_sync(kFull, rp, 0);
    const long long end = row_ptr[last + 1];
    const long long abase = base & ~1LL;
    const int nchunks = (int)((end - abase + 1) >> 1);
    const bool staged = nchunks <= kStage1 / 2;
    double* st = stage[w];
    if (staged) {
#pragma unroll
        for (int c = 0; c < kStage1 / 64; c++) {
            const int g = c * 32 + lane;
            if (abase + 2 * g + 1 < end)
                cp_async16(st + 2 * g, values + abase + 2 * g);
            else if (abase + 2 * g < end)
                st[2 * g] = values[abase + 2 * g];  // last double of the array, no 16-byte read
        }
        cp_async_wait_all();
    }
    __syncwarp();
    if (!valid)
        return;

    const int NN = N * N;
    const int i = row / NN, j = (row / N) % N, k = row % N;
    const bool interior = i > 0 && i < N - 1 && j > 0 && j < N - 1 && k > 0 && k < N - 1;
    double sum;
    if (staged) {
        const double* v = st + (rp - abase);
        sum = interior ? row27(v, x, row, N)
                       : row_generic(v, col_idx + rp, (int)(row_ptr[row + 1] - rp), x);
    } else {
        const double* v = values + rp;
        sum = interior ? row27(v, x, row, N)
                       : row_generic(v, col_idx + rp, (int)(row_ptr[row + 1] - rp), x);
    }
    y[row] = sum;
}

/** @brief Shared memory of sym25d_kernel, in bytes, for a tile of TJ warps */
__host__ __device__ constexpr int sym25d_smem_bytes(int TJ) {
    // stage 16 doubles per row, x for 3 planes, 3 exchange arrays double-buffered
    return TJ * 32 * (16 + 3 + 6) * (int)sizeof(double);
}

/**
 * @brief Symmetric 27-point SpMV reading half of the CSR values, 2.5D tile marching along i
 *
 * @details Block = TJ warps. Warp w holds line j = j0 + w - 1, lane L holds k = k0 + L - 1, so
 * the block covers a 32 x TJ tile of (k, j) columns whose outer ring is a halo: only lanes
 * 1..30 of warps 1..TJ-2 write y. The block walks planes i = i0 - 1 .. i0 + ZC - 1; the first
 * plane only produces the products that plane i0 needs from below.
 *
 * Per plane, an interior row r reads u[0..13] = values[row_ptr[r] + 13 .. 26] and:
 *  - gathers u[0] x_r + u[1] x(i, j, k+1) + u[2..4] x(i, j+1, k-1..k+1)
 *    + u[5..13] x(i+1, j-1..j+1, k-1..k+1), all x from shared memory;
 *  - publishes u[1] x_r for row (i, j, k+1) by a shuffle up one lane;
 *  - publishes u[2..4] x_r for rows (i, j+1, k-1..k+1), summed along k by shuffles, then
 *    through shared memory (B);
 *  - publishes u[5..13] x_r for rows (i+1, j-1..j+1, k-1..k+1), summed along k by shuffles,
 *    then through shared memory (A-, A+) and a register (A0) for the next plane.
 * Each thread thus publishes three doubles per plane, and x costs one coalesced load per row.
 *
 * Coefficients are staged per warp with 16-byte cp.async: 7 or 8 chunks per row depending on the
 * parity of row_ptr, which touch no 32-byte sector outside entries 13..26 (an extra double at
 * either end shares its 16 bytes with entry 13 or 26). Chunks are XOR-swizzled to spread banks.
 * The copies of plane i + 1 are issued during plane i, into the same buffer once u[] holds
 * plane i, and x and row_ptr are loaded two planes ahead: no DRAM latency is waited for inside
 * the plane that issued the load.
 *
 * Launch: blockDim = TJ * 32, gridDim = (ceil(N / 30), ceil(N / (TJ - 2)), ceil(N / ZC)),
 * dynamic shared memory sym25d_smem_bytes(TJ).
 */
template <int TJ>
__global__ void __launch_bounds__(TJ * 32)
    sym25d_kernel(const long long* __restrict__ row_ptr, const int* __restrict__ col_idx,
                  const double* __restrict__ values, const double* __restrict__ x,
                  double* __restrict__ y, int N, int ZC) {
    constexpr int T = TJ * 32;
    extern __shared__ __align__(16) double smem[];
    double* const stage = smem;        // [TJ][32][16]
    double* const Xs = smem + T * 16;  // [3][T]: x of planes, slot = plane % 3
    double* const Bs = Xs + 3 * T;     // [2][T]: in-plane products for line j + 1
    double* const Aps = Bs + 2 * T;    // [2][T]: next-plane products for line j + 1
    double* const Ams = Aps + 2 * T;   // [2][T]: next-plane products for line j - 1

    const int tid = threadIdx.x;
    const int lane = tid & 31, w = tid >> 5;
    const int k = blockIdx.x * 30 + lane - 1;
    const int j = blockIdx.y * (TJ - 2) + w - 1;
    const int i0 = blockIdx.z * ZC;
    const int i1 = min(i0 + ZC, N);
    const int NN = N * N;
    const bool col_in = k >= 0 && k < N && j >= 0 && j < N;
    const bool col_inner = k > 0 && k < N - 1 && j > 0 && j < N - 1;
    const bool out_col = col_in && lane >= 1 && lane <= 30 && w >= 1 && w <= TJ - 2;
    const bool near_face = j == 1 || j == N - 2 || k == 1 || k == N - 2;
    const int col_off = j * N + k;
    double* const st = stage + w * 32 * 16;

    // Interior lanes are consecutive rows of one grid line, each holding 27 entries, so the
    // row_ptr of the first one gives every other: one row_ptr load per warp and plane.
    const unsigned colmask = __ballot_sync(kFull, col_inner);
    const int a = colmask ? __ffs(colmask) - 1 : 0;
    auto load_rpa = [&](int ii) -> long long {
        return (lane == a && col_inner && ii > 0 && ii < N - 1) ? row_ptr[ii * NN + col_off] : 0;
    };
    auto plane_mask = [&](int ii) -> unsigned { return (ii > 0 && ii < N - 1) ? colmask : 0u; };
    // Coefficients 13..26 of the interior rows of plane ii, 16-byte copies into st
    auto issue_copies = [&](long long rpb, unsigned m) {
        if (!m)
            return;
#pragma unroll
        for (int c = 0; c < 8; c++) {
            const int e = c * 32 + lane, src = e >> 3, q = e & 7;
            if ((m >> src) & 1u) {
                const long long rps = rpb + 27LL * (src - a);
                const long long d = ((rps + 13) & ~1LL) + 2 * q;
                if (d < rps + 27)
                    cp_async16(st + src * 16 + ((q ^ (src & 7)) << 1), values + d);
            }
        }
    };

    long long rpa_cur = __shfl_sync(kFull, load_rpa(i0 - 1), a);
    issue_copies(rpa_cur, plane_mask(i0 - 1));
    long long rpa_n1 = load_rpa(i0);  // held by lane a until shuffled
    double xc = (col_in && i0 >= 1) ? x[(i0 - 1) * NN + col_off] : 0.0;
    double xn = (col_in && i0 < N) ? x[i0 * NN + col_off] : 0.0;
    double P = 0.0;  // products received from plane i - 1

    for (int i = i0 - 1; i < i1; ++i) {
        const int s_cur = (i + 3) % 3, s_next = (i + 4) % 3, buf = (i + 2) & 1;
        const bool plane_inner = i > 0 && i < N - 1;
        const bool interior = col_inner && plane_inner;
        const int r = i * NN + col_off;
        const unsigned imask = plane_mask(i);
        const long long rpa = rpa_cur;
        const long long rpa_n2 = load_rpa(i + 2);
        const double xnn = (col_in && i + 2 < N) ? x[r + 2 * NN] : 0.0;
        // Face rows: row_ptr issued now, used after the barrier
        const long long rpf = (i >= i0 && out_col && !interior) ? row_ptr[r] : 0;

        // Coefficients of this plane (copies issued one plane earlier)
        double u[14];
        long long rp = 0;
        cp_async_wait_all();
        __syncwarp();
        if (imask) {
            rp = rpa + 27LL * (lane - a);
            const int par = (int)((rp + 13) & 1);
#pragma unroll
            for (int p = 0; p < 14; p++) {
                const int d = p + par;
                u[p] = interior ? st[lane * 16 + (((d >> 1) ^ (lane & 7)) << 1) + (d & 1)] : 0.0;
            }
        } else {
#pragma unroll
            for (int p = 0; p < 14; p++)
                u[p] = 0.0;
        }
        __syncwarp();  // st is free: start the copies of plane i + 1
        rpa_cur = __shfl_sync(kFull, rpa_n1, a);
        if (i + 1 < i1)
            issue_copies(rpa_cur, plane_mask(i + 1));
        rpa_n1 = rpa_n2;
        Xs[s_next * T + tid] = xn;  // loaded one plane earlier

        // Products for the rows that hold the transposed couplings, pre-summed along k
        const double sk_in = __shfl_up_sync(kFull, u[1] * xc, 1);
        const double B =
            __shfl_down_sync(kFull, u[2] * xc, 1) + u[3] * xc + __shfl_up_sync(kFull, u[4] * xc, 1);
        const double Am =
            __shfl_down_sync(kFull, u[5] * xc, 1) + u[6] * xc + __shfl_up_sync(kFull, u[7] * xc, 1);
        const double A0 = __shfl_down_sync(kFull, u[8] * xc, 1) + u[9] * xc +
                          __shfl_up_sync(kFull, u[10] * xc, 1);
        const double Ap = __shfl_down_sync(kFull, u[11] * xc, 1) + u[12] * xc +
                          __shfl_up_sync(kFull, u[13] * xc, 1);
        Bs[buf * T + tid] = B;
        Aps[buf * T + tid] = Ap;
        Ams[buf * T + tid] = Am;
        __syncthreads();

        if (i >= i0 && out_col) {
            double s;
            if (interior) {
                const double* Xc = Xs + s_cur * T + tid;
                const double* Xn = Xs + s_next * T + tid;
                double y0 = u[0] * xc + u[1] * Xc[1] + P + sk_in + Bs[buf * T + tid - 32];
                const double y1 = u[2] * Xc[31] + u[3] * Xc[32] + u[4] * Xc[33];
                double y2 = 0.0;
#pragma unroll
                for (int q = 0; q < 9; q++)
                    y2 += u[5 + q] * Xn[(q / 3 - 1) * 32 + (q % 3 - 1)];
                if (i == 1 || near_face) {
                    // Lower neighbours on a face publish nothing: use this row's own entries
#pragma unroll
                    for (int p = 0; p < 13; p++) {
                        const int ni = i + p / 9 - 1, nj = j + (p / 3) % 3 - 1, nk = k + p % 3 - 1;
                        if (ni == 0 || nj == 0 || nj == N - 1 || nk == 0 || nk == N - 1)
                            y0 += values[rp + p] * x[r + entry_offset(p, N)];
                    }
                }
                s = (y0 + y1) + y2;
            } else {
                s = row_face<9>(values + rpf, x, r, i, j, k, N);
            }
            y[r] = s;
        }
        if (w >= 1 && w <= TJ - 2)
            P = Aps[buf * T + tid - 32] + A0 + Ams[buf * T + tid + 32];
        xc = xn;
        xn = xnn;
    }
}

/**
 * @brief One-time check of the assumptions of the kernels above, one thread per row
 *
 * @details Counts rows whose pattern differs from the stencil, 27 entries inside and the
 * truncated stencil on faces (bad_pattern),
 * and couplings between two interior rows whose two stored copies differ bitwise (bad_sym).
 * csr_staged_kernel needs bad_pattern == 0; sym25d_kernel needs both counts at 0.
 */
__global__ void check_pattern_sym_kernel(const long long* __restrict__ row_ptr,
                                         const int* __restrict__ col_idx,
                                         const double* __restrict__ values, int N,
                                         unsigned long long* bad_pattern,
                                         unsigned long long* bad_sym) {
    const int n = N * N * N, NN = N * N;
    for (int r = blockIdx.x * blockDim.x + threadIdx.x; r < n; r += gridDim.x * blockDim.x) {
        const int i = r / NN, j = (r / N) % N, k = r % N;
        const long long rp = row_ptr[r];
        if (!(i > 0 && i < N - 1 && j > 0 && j < N - 1 && k > 0 && k < N - 1)) {
            // Face row: the valid stencil neighbours, in stencil (= column) order
            long long q = rp;
            bool ok = true;
            for (int p = 0; ok && p < 27; p++) {
                const int ni = i + p / 9 - 1, nj = j + (p / 3) % 3 - 1, nk = k + p % 3 - 1;
                if (ni < 0 || ni >= N || nj < 0 || nj >= N || nk < 0 || nk >= N)
                    continue;
                ok = q < row_ptr[r + 1] && col_idx[q] == r + entry_offset(p, N);
                q++;
            }
            if (!ok || q != row_ptr[r + 1])
                atomicAdd(bad_pattern, 1ULL);
            continue;
        }
        bool ok = row_ptr[r + 1] - rp == 27;
        for (int p = 0; ok && p < 27; p++)
            ok = col_idx[rp + p] == r + entry_offset(p, N);
        if (!ok) {
            atomicAdd(bad_pattern, 1ULL);
            continue;
        }
        unsigned long long bad = 0;
        for (int p = 14; p < 27; p++) {
            const int nb = r + entry_offset(p, N);
            const int ni = nb / NN, nj = (nb / N) % N, nk = nb % N;
            if (ni > 0 && ni < N - 1 && nj > 0 && nj < N - 1 && nk > 0 && nk < N - 1 &&
                __double_as_longlong(values[rp + p]) !=
                    __double_as_longlong(values[row_ptr[nb] + 26 - p]))
                bad++;
        }
        if (bad)
            atomicAdd(bad_sym, bad);
    }
}

}  // namespace stencil27

#endif  // SPMV_STENCIL27_FAST_CUH
