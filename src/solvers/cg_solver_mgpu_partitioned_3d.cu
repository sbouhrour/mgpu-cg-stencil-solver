/**
 * @file cg_solver_mgpu_partitioned_3d.cu
 * @brief Multi-GPU CG solver with CSR partitioning for 3D 7-point stencil
 *
 * Architecture:
 * - Z-slab partitioning: each GPU owns contiguous Z-planes
 * - Halo zones: one full XY-plane (N² elements) per neighbor direction
 * - Communication: MPI explicit staging (D2H, MPI, H2D)
 *
 * For 7-point stencil on 64×64×64 grid with 2 GPUs:
 * - GPU0: rows [0:131072), needs plane from GPU1 (4096 doubles = 32 KB)
 * - GPU1: rows [131072:262144), needs plane from GPU0 (32 KB)
 * - Total communication: 64 KB per iteration
 *
 * Author: Bouhrour Stephane
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <mpi.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <nvtx3/nvToolsExt.h>

#include "spmv.h"
#include "io.h"
#include "solvers/cg_solver_mgpu_partitioned.h"
#include "solvers/comm_backend.h"

/* External kernels */
extern __global__ void axpy_kernel(double alpha, const double* x, double* y, int n);
extern __global__ void axpby_kernel(double alpha, const double* x, double beta, double* y, int n);
extern __global__ void stencil7_csr_partitioned_halo_kernel_3d(
    const long long* __restrict__ row_ptr, const int* __restrict__ col_idx,
    const double* __restrict__ values, const double* __restrict__ x_local,
    const double* __restrict__ x_halo_prev, const double* __restrict__ x_halo_next,
    double* __restrict__ y, int n_local, int row_offset, int N_total, int grid_size);

/**
 * @brief Compute local dot product using cuBLAS
 */
static double compute_local_dot_3d(cublasHandle_t cublas_handle, const double* d_x,
                                   const double* d_y, int n) {
    double result;
    cublasStatus_t status = cublasDdot(cublas_handle, n, d_x, 1, d_y, 1, &result);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS ddot failed\n");
        exit(EXIT_FAILURE);
    }
    return result;
}

/*
 * Global CG scalars (p.Ap, r.r) and the alpha/beta derived from them.
 *
 * Host mode: cublasDdot returns a host double (the call waits for the stream),
 * MPI sums it, alpha and beta are computed on the host and passed by value.
 * Device mode: the scalars never leave the GPU. cublasDdot writes to device
 * memory, the reduction is enqueued on the stream, and the BLAS1 kernels read
 * alpha and beta from device memory. The host only reads r.r back when it has
 * to test convergence.
 */
typedef struct {
    int on_device;
    cublasHandle_t cublas;
    CommContext* comm;
    cudaStream_t stream;
    double* d_buf;  // device mode: [p.Ap, r.r (old), r.r (new)]
    double* d_pAp;
    double* d_rs_old;
    double* d_rs_new;
    double* h_read;  // pinned, device mode: [p.Ap, r.r (new)] read back
    double pAp;      // host values: always valid in host mode, after
    double rs_old;   // cg_scalars_read in device mode
    double rs_new;
} CgScalars;

// y = (sign * num/den) * x + y: same expression as axpy_kernel, alpha read from the device
static __global__ void axpy_ratio_kernel(const double* num, const double* den, double sign,
                                         const double* x, double* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double alpha = sign * (*num / *den);
        y[i] = alpha * x[i] + y[i];
    }
}

// y = alpha * x + (num/den) * y: same expression as axpby_kernel, beta read from the device
static __global__ void axpby_ratio_kernel(double alpha, const double* x, const double* num,
                                          const double* den, double* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double beta = *num / *den;
        y[i] = alpha * x[i] + beta * y[i];
    }
}

static void cg_scalars_init(CgScalars* S, int on_device, cublasHandle_t cublas, CommContext* comm,
                            cudaStream_t stream) {
    memset(S, 0, sizeof(*S));
    S->on_device = on_device;
    S->cublas = cublas;
    S->comm = comm;
    S->stream = stream;
    if (on_device) {
        CUDA_CHECK(cudaMalloc(&S->d_buf, 3 * sizeof(double)));
        CUDA_CHECK(cudaMallocHost(&S->h_read, 2 * sizeof(double)));
        S->d_pAp = S->d_buf;
        S->d_rs_old = S->d_buf + 1;
        S->d_rs_new = S->d_buf + 2;
        cublasSetPointerMode(cublas, CUBLAS_POINTER_MODE_DEVICE);
    }
}

static void cg_scalars_free(CgScalars* S) {
    if (S->on_device) {
        cublasSetPointerMode(S->cublas, CUBLAS_POINTER_MODE_HOST);
        cudaFree(S->d_buf);
        cudaFreeHost(S->h_read);
    }
}

static void dot_device(CgScalars* S, const double* x, const double* y, int n, double* d_out) {
    if (cublasDdot(S->cublas, n, x, 1, y, 1, d_out) != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS ddot failed\n");
        exit(EXIT_FAILURE);
    }
    comm_allreduce_sum_device(S->comm, d_out, S->stream);
}

/** rs_old = r.r, summed over ranks; host copy always valid afterwards (read once per solve). */
static void cg_scalars_rs_init(CgScalars* S, const double* r, int n) {
    if (S->on_device) {
        dot_device(S, r, r, n, S->d_rs_old);
        CUDA_CHECK(cudaMemcpyAsync(S->h_read, S->d_rs_old, sizeof(double), cudaMemcpyDeviceToHost,
                                   S->stream));
        CUDA_CHECK(cudaStreamSynchronize(S->stream));
        S->rs_old = S->h_read[0];
    } else {
        S->rs_old = comm_allreduce_sum(S->comm, compute_local_dot_3d(S->cublas, r, r, n));
    }
}

static void cg_scalars_pAp(CgScalars* S, const double* p, const double* Ap, int n) {
    if (S->on_device)
        dot_device(S, p, Ap, n, S->d_pAp);
    else
        S->pAp = comm_allreduce_sum(S->comm, compute_local_dot_3d(S->cublas, p, Ap, n));
}

/** x += alpha p ; r -= alpha Ap, with alpha = rs_old / p.Ap */
static void cg_scalars_update_xr(CgScalars* S, const double* p, double* x, const double* Ap,
                                 double* r, int n, int blocks, int threads) {
    if (S->on_device) {
        axpy_ratio_kernel<<<blocks, threads, 0, S->stream>>>(S->d_rs_old, S->d_pAp, 1.0, p, x, n);
        axpy_ratio_kernel<<<blocks, threads, 0, S->stream>>>(S->d_rs_old, S->d_pAp, -1.0, Ap, r, n);
    } else {
        double alpha = S->rs_old / S->pAp;
        axpy_kernel<<<blocks, threads, 0, S->stream>>>(alpha, p, x, n);
        axpy_kernel<<<blocks, threads, 0, S->stream>>>(-alpha, Ap, r, n);
    }
}

static void cg_scalars_rs_new(CgScalars* S, const double* r, int n) {
    if (S->on_device)
        dot_device(S, r, r, n, S->d_rs_new);
    else
        S->rs_new = comm_allreduce_sum(S->comm, compute_local_dot_3d(S->cublas, r, r, n));
}

/** Device mode: bring p.Ap and r.r back to the host (one stream synchronization). */
static void cg_scalars_read(CgScalars* S) {
    if (!S->on_device)
        return;
    CUDA_CHECK(cudaMemcpyAsync(&S->h_read[0], S->d_pAp, sizeof(double), cudaMemcpyDeviceToHost,
                               S->stream));
    CUDA_CHECK(cudaMemcpyAsync(&S->h_read[1], S->d_rs_new, sizeof(double), cudaMemcpyDeviceToHost,
                               S->stream));
    CUDA_CHECK(cudaStreamSynchronize(S->stream));
    S->pAp = S->h_read[0];
    S->rs_new = S->h_read[1];
}

/** p = r + beta p, with beta = rs_new / rs_old */
static void cg_scalars_update_p(CgScalars* S, const double* r, double* p, int n, int blocks,
                                int threads) {
    if (S->on_device) {
        axpby_ratio_kernel<<<blocks, threads, 0, S->stream>>>(1.0, r, S->d_rs_new, S->d_rs_old, p,
                                                              n);
    } else {
        axpby_kernel<<<blocks, threads, 0, S->stream>>>(1.0, r, S->rs_new / S->rs_old, p, n);
    }
}

/** rs_old = rs_new (device mode swaps the two slots, the host copy follows what was read). */
static void cg_scalars_advance(CgScalars* S) {
    if (S->on_device) {
        double* t = S->d_rs_old;
        S->d_rs_old = S->d_rs_new;
        S->d_rs_new = t;
    }
    S->rs_old = S->rs_new;
}

/**
 * @brief Multi-GPU CG solver for 3D 7-point stencil with Z-slab partitioning
 */
int cg_solve_mgpu_partitioned_3d(SpmvOperator* spmv_op, MatrixData* mat, const double* b, double* x,
                                 CGConfigMultiGPU config, CGStatsMultiGPU* stats) {

    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int n = mat->rows;
    int grid_size = mat->grid_size;
    int halo_size = grid_size * grid_size;  // One full XY-plane

    if (rank == 0 && config.verbose >= 1) {
        printf("\n========================================\n");
        printf("Multi-GPU CG Solver (3D PARTITIONED CSR)\n");
        printf("========================================\n");
        printf("MPI ranks: %d\n", world_size);
        printf("Problem size: %d unknowns (%d³ grid)\n", n, grid_size);
        printf("Halo size: %d elements (N²=%d² per direction)\n", halo_size, grid_size);
        printf("Max iterations: %d\n", config.max_iters);
        printf("Tolerance: %.1e\n", config.tolerance);
        printf("========================================\n\n");
    }

    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    int device_id = rank % device_count;
    CUDA_CHECK(cudaSetDevice(device_id));

    // Z-slab partition: contiguous Z-planes per GPU
    int n_local = n / world_size;
    int row_offset = rank * n_local;
    if (rank == world_size - 1) {
        n_local = n - row_offset;
    }

    if (config.verbose >= 1) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
        printf("[Rank %d] GPU %d: %s (CC %d.%d)\n", rank, device_id, prop.name, prop.major,
               prop.minor);
        printf("[Rank %d] Rows: [%d:%d) (%d rows, %d Z-planes)\n", rank, row_offset,
               row_offset + n_local, n_local, n_local / (grid_size * grid_size));
    }

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    cublasHandle_t cublas_handle;
    cublasStatus_t cublas_status = cublasCreate(&cublas_handle);
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS initialization failed\n");
        exit(EXIT_FAILURE);
    }
    cublasSetStream(cublas_handle, stream);

    CommContext* comm = config.comm;
    int own_comm = 0;
    if (comm == NULL) {
        comm = comm_create(COMM_STAGED, MPI_COMM_WORLD, (size_t)halo_size);
        own_comm = 1;
    }

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

    CUDA_CHECK(cudaMemcpy(d_row_ptr, local_row_ptr, (n_local + 1) * sizeof(long long),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_idx, &csr_mat.col_indices[offset], (size_t)local_nnz * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_values, &csr_mat.values[offset], (size_t)local_nnz * sizeof(double),
                          cudaMemcpyHostToDevice));

    free(local_row_ptr);

    if (config.verbose >= 1) {
        printf("[Rank %d] Local CSR: %d rows, %lld nnz (%.2f MB)\n", rank, n_local, local_nnz,
               (n_local * sizeof(long long) + (double)local_nnz * (sizeof(int) + sizeof(double))) /
                   1e6);
    }

    // Allocate vectors
    double *d_x_local, *d_r_local, *d_p_local, *d_Ap, *d_b;
    double *d_p_halo_prev = NULL, *d_p_halo_next = NULL;
    double *d_r_halo_prev = NULL, *d_r_halo_next = NULL;

    CUDA_CHECK(cudaMalloc(&d_x_local, n_local * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_r_local, n_local * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_p_local, n_local * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Ap, n_local * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_b, n_local * sizeof(double)));

    // Halo buffers: one XY-plane per direction
    if (rank > 0) {
        CUDA_CHECK(cudaMalloc(&d_p_halo_prev, halo_size * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_r_halo_prev, halo_size * sizeof(double)));
    }
    if (rank < world_size - 1) {
        CUDA_CHECK(cudaMalloc(&d_p_halo_next, halo_size * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_r_halo_next, halo_size * sizeof(double)));
    }

    if (config.verbose >= 1) {
        size_t local_mem = n_local * 5 * sizeof(double);
        size_t halo_mem = 0;
        if (rank > 0)
            halo_mem += halo_size * 2 * sizeof(double);
        if (rank < world_size - 1)
            halo_mem += halo_size * 2 * sizeof(double);
        printf("[Rank %d] Vector memory: %.2f MB (local) + %.2f KB (halo)\n", rank, local_mem / 1e6,
               halo_mem / 1e3);
    }

    // Initialize vectors
    CUDA_CHECK(cudaMemcpy(d_b, &b[row_offset], n_local * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(d_x_local, &x[row_offset], n_local * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_r_local, 0, n_local * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_p_local, 0, n_local * sizeof(double)));

    double *d_x_halo_prev = NULL, *d_x_halo_next = NULL;
    if (rank > 0) {
        CUDA_CHECK(cudaMalloc(&d_x_halo_prev, halo_size * sizeof(double)));
    }
    if (rank < world_size - 1) {
        CUDA_CHECK(cudaMalloc(&d_x_halo_next, halo_size * sizeof(double)));
    }

    // Exercise every send/receive buffer pair once before the clock starts, so
    // that lazy connection setup or buffer registration by the communication
    // library is never timed. The timed region rewrites every halo it reads.
    comm_halo_exchange(comm, d_x_local, d_x_local + (n_local - halo_size), d_x_halo_prev,
                       d_x_halo_next, halo_size, stream);
    comm_halo_exchange(comm, d_r_local, d_r_local + (n_local - halo_size), d_r_halo_prev,
                       d_r_halo_next, halo_size, stream);
    comm_halo_exchange(comm, d_p_local, d_p_local + (n_local - halo_size), d_p_halo_prev,
                       d_p_halo_next, halo_size, stream);

    CgScalars S;
    cg_scalars_init(&S, config.dots_device, cublas_handle, comm, stream);

    MPI_Barrier(MPI_COMM_WORLD);

    // Timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, stream));

    if (rank == 0 && config.verbose >= 1) {
        printf("\nStarting CG iterations...\n");
    }

    int threads = 256;
    int blocks_local = (n_local + threads - 1) / threads;

    // Initial x halo exchange
    comm_halo_exchange(comm, d_x_local, d_x_local + (n_local - halo_size), d_x_halo_prev,
                       d_x_halo_next, halo_size, stream);

    // Initial SpMV: Ap = A*x
    stencil7_csr_partitioned_halo_kernel_3d<<<blocks_local, threads, 0, stream>>>(
        d_row_ptr, d_col_idx, d_values, d_x_local, d_x_halo_prev, d_x_halo_next, d_Ap, n_local,
        row_offset, n, grid_size);

    // r = b - Ap
    axpy_kernel<<<blocks_local, threads, 0, stream>>>(-1.0, d_Ap, d_b, n_local);
    CUDA_CHECK(cudaMemcpy(d_r_local, d_b, n_local * sizeof(double), cudaMemcpyDeviceToDevice));

    // Exchange r halo
    comm_halo_exchange(comm, d_r_local, d_r_local + (n_local - halo_size), d_r_halo_prev,
                       d_r_halo_next, halo_size, stream);

    // p = r
    CUDA_CHECK(
        cudaMemcpy(d_p_local, d_r_local, n_local * sizeof(double), cudaMemcpyDeviceToDevice));

    // Copy r halo to p halo
    if (rank > 0) {
        CUDA_CHECK(cudaMemcpy(d_p_halo_prev, d_r_halo_prev, halo_size * sizeof(double),
                              cudaMemcpyDeviceToDevice));
    }
    if (rank < world_size - 1) {
        CUDA_CHECK(cudaMemcpy(d_p_halo_next, d_r_halo_next, halo_size * sizeof(double),
                              cudaMemcpyDeviceToDevice));
    }

    // rs_old = r.r, summed over ranks
    cg_scalars_rs_init(&S, d_r_local, n_local);
    double b_norm = sqrt(S.rs_old);

    if (rank == 0 && config.verbose >= 2) {
        printf("[Iter   0] Residual: %.6e\n", sqrt(S.rs_old));
    }

    // Device mode reads r.r back only to test convergence: every check_every
    // iterations, at the last one, and at every iteration when tracing.
    const int check_every = config.check_every > 0 ? config.check_every : 1;

    // CG iteration loop
    nvtxRangePush("CG_Solver_3D");
    int iter;
    for (iter = 0; iter < config.max_iters; iter++) {
        nvtxRangePush("CG_Iteration_3D");

        // Ap = A * p
        nvtxRangePush("SpMV_3D");
        stencil7_csr_partitioned_halo_kernel_3d<<<blocks_local, threads, 0, stream>>>(
            d_row_ptr, d_col_idx, d_values, d_p_local, d_p_halo_prev, d_p_halo_next, d_Ap, n_local,
            row_offset, n, grid_size);
        nvtxRangePop();

        // alpha = rs_old / (p^T * Ap)
        nvtxRangePush("Dot_Product");
        cg_scalars_pAp(&S, d_p_local, d_Ap, n_local);
        nvtxRangePop();

        // x = x + alpha * p ; r = r - alpha * Ap
        nvtxRangePush("BLAS_AXPY");
        cg_scalars_update_xr(&S, d_p_local, d_x_local, d_Ap, d_r_local, n_local, blocks_local,
                             threads);
        nvtxRangePop();

        // rs_new = dot(r, r)
        nvtxRangePush("Dot_Product");
        cg_scalars_rs_new(&S, d_r_local, n_local);
        nvtxRangePop();

        int check = !S.on_device || config.verbose >= 2 || (iter + 1) % check_every == 0 ||
                    iter + 1 == config.max_iters;
        if (check) {
            cg_scalars_read(&S);
            double residual_norm = sqrt(S.rs_new);
            double rel_residual = residual_norm / b_norm;
            double alpha = S.rs_old / S.pAp;

            if (rank == 0 && config.verbose >= 2) {
                printf("[Iter %3d] Residual: %.6e (rel: %.6e, alpha: %.4e)\n", iter + 1,
                       residual_norm, rel_residual, alpha);
            }
            if (rank == 0 && config.verbose >= 3) {
                printf("[Trace %3d] rs=%a alpha=%a\n", iter + 1, S.rs_new, alpha);
            }

            if (rel_residual < config.tolerance) {
                iter++;
                stats->converged = 1;
                stats->iterations = iter;
                stats->residual_norm = residual_norm;
                nvtxRangePop();
                break;
            }
        }

        // p = r + beta * p, beta = rs_new / rs_old
        nvtxRangePush("BLAS_AXPBY");
        cg_scalars_update_p(&S, d_r_local, d_p_local, n_local, blocks_local, threads);
        nvtxRangePop();

        // Halo exchange for p (N² elements per direction)
        comm_halo_exchange(comm, d_p_local, d_p_local + (n_local - halo_size), d_p_halo_prev,
                           d_p_halo_next, halo_size, stream);

        cg_scalars_advance(&S);
        nvtxRangePop();
    }
    nvtxRangePop();

    if (iter == config.max_iters && rank == 0) {
        printf("\nMax iterations reached without convergence\n");
        stats->converged = 0;
        stats->iterations = iter;
        stats->residual_norm = sqrt(S.rs_old);
    }

    // Stop timing
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float time_ms;
    CUDA_CHECK(cudaEventElapsedTime(&time_ms, start, stop));

    stats->time_total_ms = time_ms;

    // MPI stats aggregation
    if (world_size > 1) {
        double local_time = stats->time_total_ms;
        double max_time, min_time;
        MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_time, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            stats->time_total_ms = max_time;
            double imbalance_pct = 100.0 * (max_time - min_time) / max_time;
            printf("Total time: %.2f ms (max), %.2f ms (min) - Load imbalance: %.1f%%\n", max_time,
                   min_time, imbalance_pct);
            printf("========================================\n");
        }
    } else if (rank == 0 && config.verbose >= 1) {
        printf("Total time: %.2f ms\n", stats->time_total_ms);
        printf("========================================\n");
    }

    // Copy result back
    CUDA_CHECK(
        cudaMemcpy(&x[row_offset], d_x_local, n_local * sizeof(double), cudaMemcpyDeviceToHost));

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

    if (d_x_halo_prev)
        cudaFree(d_x_halo_prev);
    if (d_x_halo_next)
        cudaFree(d_x_halo_next);
    if (d_p_halo_prev)
        cudaFree(d_p_halo_prev);
    if (d_p_halo_next)
        cudaFree(d_p_halo_next);
    if (d_r_halo_prev)
        cudaFree(d_r_halo_prev);
    if (d_r_halo_next)
        cudaFree(d_r_halo_next);

    if (own_comm)
        comm_destroy(comm);

    cg_scalars_free(&S);
    cublasDestroy(cublas_handle);
    cudaStreamDestroy(stream);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}

/* External 27-point kernel */
extern __global__ void stencil27_csr_partitioned_halo_kernel_3d(
    const long long* __restrict__ row_ptr, const int* __restrict__ col_idx,
    const double* __restrict__ values, const double* __restrict__ x_local,
    const double* __restrict__ x_halo_prev, const double* __restrict__ x_halo_next,
    double* __restrict__ y, int n_local, int row_offset, int N_total, int grid_size);

// SoA variant (coefficient-major values, ghost-layer x), defined in
// src/spmv/spmv_stencil_3d_27pt_soa_halo_kernel.cu
extern __global__ void stencil27_soa_halo_kernel_3d(const double* __restrict__ values_soa,
                                                    const double* __restrict__ x_ext,
                                                    double* __restrict__ y, int n_local,
                                                    int grid_size);
extern void build_values_soa_27pt_3d(const long long* row_ptr, const int* col_idx,
                                     const double* values, int n_local, long long row_offset,
                                     int grid_size, double* values_soa);

/**
 * @brief Multi-GPU CG solver for 3D 27-point stencil with Z-slab partitioning (synchronous)
 */
int cg_solve_mgpu_partitioned_27pt_3d(SpmvOperator* spmv_op, MatrixData* mat, const double* b,
                                      double* x, CGConfigMultiGPU config, CGStatsMultiGPU* stats) {

    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int n = mat->rows;
    int grid_size = mat->grid_size;
    int halo_size = grid_size * grid_size;

    if (rank == 0 && config.verbose >= 1) {
        printf("\n========================================\n");
        printf("Multi-GPU CG Solver (3D 27-POINT PARTITIONED CSR)\n");
        printf("========================================\n");
        printf("MPI ranks: %d\n", world_size);
        printf("Problem size: %d unknowns (%d³ grid)\n", n, grid_size);
        printf("Halo size: %d elements (N²=%d² per direction)\n", halo_size, grid_size);
        printf("Max iterations: %d\n", config.max_iters);
        printf("Tolerance: %.1e\n", config.tolerance);
        printf("========================================\n\n");
    }

    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    int device_id = rank % device_count;
    CUDA_CHECK(cudaSetDevice(device_id));

    int n_local = n / world_size;
    int row_offset = rank * n_local;
    if (rank == world_size - 1) {
        n_local = n - row_offset;
    }

    if (config.verbose >= 1) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
        printf("[Rank %d] GPU %d: %s (CC %d.%d)\n", rank, device_id, prop.name, prop.major,
               prop.minor);
        printf("[Rank %d] Rows: [%d:%d) (%d rows, %d Z-planes)\n", rank, row_offset,
               row_offset + n_local, n_local, n_local / (grid_size * grid_size));
    }

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    cublasHandle_t cublas_handle;
    cublasStatus_t cublas_status = cublasCreate(&cublas_handle);
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS initialization failed\n");
        exit(EXIT_FAILURE);
    }
    cublasSetStream(cublas_handle, stream);

    CommContext* comm = config.comm;
    int own_comm = 0;
    if (comm == NULL) {
        comm = comm_create(COMM_STAGED, MPI_COMM_WORLD, (size_t)halo_size);
        own_comm = 1;
    }

    if (rank == 0 && config.verbose >= 1) {
        printf("Building local CSR partitions...\n");
    }

    build_csr_struct(mat);

    long long local_nnz = csr_mat.row_ptr[row_offset + n_local] - csr_mat.row_ptr[row_offset];

    const int use_soa = config.spmv_soa;

    long long* d_row_ptr = NULL;
    int* d_col_idx = NULL;
    double* d_values = NULL;
    double* d_values_soa = NULL;

    long long* local_row_ptr = (long long*)malloc((n_local + 1) * sizeof(long long));
    long long offset = csr_mat.row_ptr[row_offset];
    for (int i = 0; i <= n_local; i++) {
        local_row_ptr[i] = csr_mat.row_ptr[row_offset + i] - offset;
    }

    if (use_soa) {
        // Coefficient-major transform; the device never sees row_ptr/col_idx
        double* h_values_soa = (double*)calloc((size_t)27 * n_local, sizeof(double));
        if (!h_values_soa) {
            fprintf(stderr, "[Rank %d] values_soa host allocation failed\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        build_values_soa_27pt_3d(local_row_ptr, &csr_mat.col_indices[offset],
                                 &csr_mat.values[offset], n_local, row_offset, grid_size,
                                 h_values_soa);
        CUDA_CHECK(cudaMalloc(&d_values_soa, (size_t)27 * n_local * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_values_soa, h_values_soa, (size_t)27 * n_local * sizeof(double),
                              cudaMemcpyHostToDevice));
        free(h_values_soa);
    } else {
        CUDA_CHECK(cudaMalloc(&d_row_ptr, (n_local + 1) * sizeof(long long)));
        CUDA_CHECK(cudaMalloc(&d_col_idx, (size_t)local_nnz * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_values, (size_t)local_nnz * sizeof(double)));

        CUDA_CHECK(cudaMemcpy(d_row_ptr, local_row_ptr, (n_local + 1) * sizeof(long long),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_col_idx, &csr_mat.col_indices[offset],
                              (size_t)local_nnz * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_values, &csr_mat.values[offset], (size_t)local_nnz * sizeof(double),
                              cudaMemcpyHostToDevice));
    }

    free(local_row_ptr);

    if (config.verbose >= 1) {
        if (use_soa) {
            printf("[Rank %d] Local SoA values: %d rows, 27 coefficient streams (%.2f MB)\n", rank,
                   n_local, (double)27 * n_local * sizeof(double) / 1e6);
        } else {
            printf(
                "[Rank %d] Local CSR: %d rows, %lld nnz (%.2f MB)\n", rank, n_local, local_nnz,
                (n_local * sizeof(long long) + (double)local_nnz * (sizeof(int) + sizeof(double))) /
                    1e6);
        }
    }

    double *d_x_local, *d_r_local, *d_p_local, *d_Ap, *d_b;
    double *d_p_halo_prev = NULL, *d_p_halo_next = NULL;
    double *d_r_halo_prev = NULL, *d_r_halo_next = NULL;
    // SoA mode: x and p live in ghost-layer buffers [N2 | n_local | N2] so the
    // SpMV reads one contiguous vector; the halo pointers alias the ghost
    // planes and the existing exchange/copy logic works unchanged.
    double *d_x_ext = NULL, *d_p_ext = NULL;
    const size_t ext_elems = (size_t)n_local + 2 * (size_t)halo_size;

    if (use_soa) {
        CUDA_CHECK(cudaMalloc(&d_x_ext, ext_elems * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_p_ext, ext_elems * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_x_ext, 0, ext_elems * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_p_ext, 0, ext_elems * sizeof(double)));
        d_x_local = d_x_ext + halo_size;
        d_p_local = d_p_ext + halo_size;
    } else {
        CUDA_CHECK(cudaMalloc(&d_x_local, n_local * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_p_local, n_local * sizeof(double)));
    }
    CUDA_CHECK(cudaMalloc(&d_r_local, n_local * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Ap, n_local * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_b, n_local * sizeof(double)));

    if (rank > 0) {
        if (use_soa)
            d_p_halo_prev = d_p_ext;  // alias: prev ghost plane
        else
            CUDA_CHECK(cudaMalloc(&d_p_halo_prev, halo_size * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_r_halo_prev, halo_size * sizeof(double)));
    }
    if (rank < world_size - 1) {
        if (use_soa)
            d_p_halo_next = d_p_ext + halo_size + n_local;  // alias: next ghost plane
        else
            CUDA_CHECK(cudaMalloc(&d_p_halo_next, halo_size * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_r_halo_next, halo_size * sizeof(double)));
    }

    CUDA_CHECK(cudaMemcpy(d_b, &b[row_offset], n_local * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(d_x_local, &x[row_offset], n_local * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_r_local, 0, n_local * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_p_local, 0, n_local * sizeof(double)));

    double *d_x_halo_prev = NULL, *d_x_halo_next = NULL;
    if (rank > 0) {
        if (use_soa)
            d_x_halo_prev = d_x_ext;  // alias: prev ghost plane
        else
            CUDA_CHECK(cudaMalloc(&d_x_halo_prev, halo_size * sizeof(double)));
    }
    if (rank < world_size - 1) {
        if (use_soa)
            d_x_halo_next = d_x_ext + halo_size + n_local;  // alias
        else
            CUDA_CHECK(cudaMalloc(&d_x_halo_next, halo_size * sizeof(double)));
    }

    // Exercise every send/receive buffer pair once before the clock starts, so
    // that lazy connection setup or buffer registration by the communication
    // library is never timed. The timed region rewrites every halo it reads.
    comm_halo_exchange(comm, d_x_local, d_x_local + (n_local - halo_size), d_x_halo_prev,
                       d_x_halo_next, halo_size, stream);
    comm_halo_exchange(comm, d_r_local, d_r_local + (n_local - halo_size), d_r_halo_prev,
                       d_r_halo_next, halo_size, stream);
    comm_halo_exchange(comm, d_p_local, d_p_local + (n_local - halo_size), d_p_halo_prev,
                       d_p_halo_next, halo_size, stream);

    CgScalars S;
    cg_scalars_init(&S, config.dots_device, cublas_handle, comm, stream);

    MPI_Barrier(MPI_COMM_WORLD);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, stream));

    if (rank == 0 && config.verbose >= 1) {
        printf("\nStarting CG iterations (27-point sync)...\n");
    }

    int threads = 256;
    int blocks_local = (n_local + threads - 1) / threads;

    // Initial x halo exchange
    comm_halo_exchange(comm, d_x_local, d_x_local + (n_local - halo_size), d_x_halo_prev,
                       d_x_halo_next, halo_size, stream);

    // Initial SpMV: Ap = A*x (27-point kernel, CSR or SoA)
    if (use_soa) {
        stencil27_soa_halo_kernel_3d<<<blocks_local, threads, 0, stream>>>(
            d_values_soa, d_x_ext, d_Ap, n_local, grid_size);
    } else {
        stencil27_csr_partitioned_halo_kernel_3d<<<blocks_local, threads, 0, stream>>>(
            d_row_ptr, d_col_idx, d_values, d_x_local, d_x_halo_prev, d_x_halo_next, d_Ap, n_local,
            row_offset, n, grid_size);
    }

    // r = b - Ap
    axpy_kernel<<<blocks_local, threads, 0, stream>>>(-1.0, d_Ap, d_b, n_local);
    CUDA_CHECK(cudaMemcpy(d_r_local, d_b, n_local * sizeof(double), cudaMemcpyDeviceToDevice));

    // Exchange r halo
    comm_halo_exchange(comm, d_r_local, d_r_local + (n_local - halo_size), d_r_halo_prev,
                       d_r_halo_next, halo_size, stream);

    // p = r
    CUDA_CHECK(
        cudaMemcpy(d_p_local, d_r_local, n_local * sizeof(double), cudaMemcpyDeviceToDevice));

    if (rank > 0) {
        CUDA_CHECK(cudaMemcpy(d_p_halo_prev, d_r_halo_prev, halo_size * sizeof(double),
                              cudaMemcpyDeviceToDevice));
    }
    if (rank < world_size - 1) {
        CUDA_CHECK(cudaMemcpy(d_p_halo_next, d_r_halo_next, halo_size * sizeof(double),
                              cudaMemcpyDeviceToDevice));
    }

    // rs_old = r.r, summed over ranks
    cg_scalars_rs_init(&S, d_r_local, n_local);
    double b_norm = sqrt(S.rs_old);

    if (rank == 0 && config.verbose >= 2) {
        printf("[Iter   0] Residual: %.6e\n", sqrt(S.rs_old));
    }

    // Device mode reads r.r back only to test convergence: every check_every
    // iterations, at the last one, and at every iteration when tracing.
    const int check_every = config.check_every > 0 ? config.check_every : 1;

    // CG iteration loop
    nvtxRangePush("CG_Solver_27PT_3D");
    int iter;
    for (iter = 0; iter < config.max_iters; iter++) {
        nvtxRangePush("CG_Iteration_27PT_3D");

        // Ap = A * p (27-point kernel, CSR or SoA)
        nvtxRangePush("SpMV_27PT_3D");
        if (use_soa) {
            stencil27_soa_halo_kernel_3d<<<blocks_local, threads, 0, stream>>>(
                d_values_soa, d_p_ext, d_Ap, n_local, grid_size);
        } else {
            stencil27_csr_partitioned_halo_kernel_3d<<<blocks_local, threads, 0, stream>>>(
                d_row_ptr, d_col_idx, d_values, d_p_local, d_p_halo_prev, d_p_halo_next, d_Ap,
                n_local, row_offset, n, grid_size);
        }
        nvtxRangePop();

        // alpha = rs_old / (p^T * Ap)
        nvtxRangePush("Dot_Product");
        cg_scalars_pAp(&S, d_p_local, d_Ap, n_local);
        nvtxRangePop();

        // x = x + alpha * p ; r = r - alpha * Ap
        nvtxRangePush("BLAS_AXPY");
        cg_scalars_update_xr(&S, d_p_local, d_x_local, d_Ap, d_r_local, n_local, blocks_local,
                             threads);
        nvtxRangePop();

        // rs_new = dot(r, r)
        nvtxRangePush("Dot_Product");
        cg_scalars_rs_new(&S, d_r_local, n_local);
        nvtxRangePop();

        int check = !S.on_device || config.verbose >= 2 || (iter + 1) % check_every == 0 ||
                    iter + 1 == config.max_iters;
        if (check) {
            cg_scalars_read(&S);
            double residual_norm = sqrt(S.rs_new);
            double rel_residual = residual_norm / b_norm;
            double alpha = S.rs_old / S.pAp;

            if (rank == 0 && config.verbose >= 2) {
                printf("[Iter %3d] Residual: %.6e (rel: %.6e, alpha: %.4e)\n", iter + 1,
                       residual_norm, rel_residual, alpha);
            }
            if (rank == 0 && config.verbose >= 3) {
                printf("[Trace %3d] rs=%a alpha=%a\n", iter + 1, S.rs_new, alpha);
            }

            if (rel_residual < config.tolerance) {
                iter++;
                stats->converged = 1;
                stats->iterations = iter;
                stats->residual_norm = residual_norm;
                nvtxRangePop();
                break;
            }
        }

        // p = r + beta * p, beta = rs_new / rs_old
        nvtxRangePush("BLAS_AXPBY");
        cg_scalars_update_p(&S, d_r_local, d_p_local, n_local, blocks_local, threads);
        nvtxRangePop();

        // Halo exchange for p (N² elements per direction)
        comm_halo_exchange(comm, d_p_local, d_p_local + (n_local - halo_size), d_p_halo_prev,
                           d_p_halo_next, halo_size, stream);

        cg_scalars_advance(&S);
        nvtxRangePop();
    }
    nvtxRangePop();

    if (iter == config.max_iters && rank == 0) {
        printf("\nMax iterations reached without convergence\n");
        stats->converged = 0;
        stats->iterations = iter;
        stats->residual_norm = sqrt(S.rs_old);
    }

    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float time_ms;
    CUDA_CHECK(cudaEventElapsedTime(&time_ms, start, stop));

    stats->time_total_ms = time_ms;

    if (world_size > 1) {
        double local_time = stats->time_total_ms;
        double max_time, min_time;
        MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_time, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            stats->time_total_ms = max_time;
            double imbalance_pct = 100.0 * (max_time - min_time) / max_time;
            printf("Total time: %.2f ms (max), %.2f ms (min) - Load imbalance: %.1f%%\n", max_time,
                   min_time, imbalance_pct);
            printf("========================================\n");
        }
    } else if (rank == 0 && config.verbose >= 1) {
        printf("Total time: %.2f ms\n", stats->time_total_ms);
        printf("========================================\n");
    }

    CUDA_CHECK(
        cudaMemcpy(&x[row_offset], d_x_local, n_local * sizeof(double), cudaMemcpyDeviceToHost));

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

    if (use_soa) {
        // x/p halo pointers alias the ghost layers: free the buffers once
        cudaFree(d_x_ext);
        cudaFree(d_p_ext);
        cudaFree(d_values_soa);
    } else {
        cudaFree(d_x_local);
        cudaFree(d_p_local);
        cudaFree(d_row_ptr);
        cudaFree(d_col_idx);
        cudaFree(d_values);
        if (d_x_halo_prev)
            cudaFree(d_x_halo_prev);
        if (d_x_halo_next)
            cudaFree(d_x_halo_next);
        if (d_p_halo_prev)
            cudaFree(d_p_halo_prev);
        if (d_p_halo_next)
            cudaFree(d_p_halo_next);
    }
    cudaFree(d_r_local);
    cudaFree(d_Ap);
    cudaFree(d_b);
    if (d_r_halo_prev)
        cudaFree(d_r_halo_prev);
    if (d_r_halo_next)
        cudaFree(d_r_halo_next);

    if (own_comm)
        comm_destroy(comm);

    cg_scalars_free(&S);
    cublasDestroy(cublas_handle);
    cudaStreamDestroy(stream);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
