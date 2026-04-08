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
#include <mpi.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <nvtx3/nvToolsExt.h>

#include "spmv.h"
#include "io.h"
#include "solvers/cg_solver_mgpu_partitioned.h"

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

/**
 * @brief Exchange halo zones with neighbors using MPI with explicit staging (3D version)
 *
 * For 3D 7-point stencil with Z-slab partitioning:
 * - Each halo is one full XY-plane = grid_size² elements
 * - Send first plane to prev, last plane to next
 */
static void exchange_halo_mpi_3d(const double* d_local_send_prev, const double* d_local_send_next,
                                 double* d_halo_recv_prev, double* d_halo_recv_next,
                                 double* h_send_prev, double* h_send_next, double* h_recv_prev,
                                 double* h_recv_next, int halo_size, int rank, int world_size,
                                 cudaStream_t stream) {
    MPI_Request requests[4];
    int req_count = 0;

    // D2H
    if (rank > 0 && d_local_send_prev != NULL) {
        CUDA_CHECK(cudaMemcpyAsync(h_send_prev, d_local_send_prev, halo_size * sizeof(double),
                                   cudaMemcpyDeviceToHost, stream));
    }
    if (rank < world_size - 1 && d_local_send_next != NULL) {
        CUDA_CHECK(cudaMemcpyAsync(h_send_next, d_local_send_next, halo_size * sizeof(double),
                                   cudaMemcpyDeviceToHost, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // MPI non-blocking
    if (rank > 0) {
        MPI_Isend(h_send_prev, halo_size, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD,
                  &requests[req_count++]);
        MPI_Irecv(h_recv_prev, halo_size, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD,
                  &requests[req_count++]);
    }
    if (rank < world_size - 1) {
        MPI_Isend(h_send_next, halo_size, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD,
                  &requests[req_count++]);
        MPI_Irecv(h_recv_next, halo_size, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD,
                  &requests[req_count++]);
    }
    if (req_count > 0) {
        MPI_Waitall(req_count, requests, MPI_STATUSES_IGNORE);
    }

    // H2D
    if (rank > 0 && d_halo_recv_prev != NULL) {
        CUDA_CHECK(cudaMemcpyAsync(d_halo_recv_prev, h_recv_prev, halo_size * sizeof(double),
                                   cudaMemcpyHostToDevice, stream));
    }
    if (rank < world_size - 1 && d_halo_recv_next != NULL) {
        CUDA_CHECK(cudaMemcpyAsync(d_halo_recv_next, h_recv_next, halo_size * sizeof(double),
                                   cudaMemcpyHostToDevice, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
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

    CUDA_CHECK(cudaSetDevice(rank));

    // Z-slab partition: contiguous Z-planes per GPU
    int n_local = n / world_size;
    int row_offset = rank * n_local;
    if (rank == world_size - 1) {
        n_local = n - row_offset;
    }

    if (config.verbose >= 1) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, rank));
        printf("[Rank %d] GPU %d: %s (CC %d.%d)\n", rank, rank, prop.name, prop.major, prop.minor);
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

    // Pinned host buffers for MPI staging
    double *h_send_prev = NULL, *h_send_next = NULL;
    double *h_recv_prev = NULL, *h_recv_next = NULL;
    if (rank > 0) {
        CUDA_CHECK(cudaMallocHost(&h_send_prev, halo_size * sizeof(double)));
        CUDA_CHECK(cudaMallocHost(&h_recv_prev, halo_size * sizeof(double)));
    }
    if (rank < world_size - 1) {
        CUDA_CHECK(cudaMallocHost(&h_send_next, halo_size * sizeof(double)));
        CUDA_CHECK(cudaMallocHost(&h_recv_next, halo_size * sizeof(double)));
    }

    // Initialize vectors
    CUDA_CHECK(cudaMemcpy(d_b, &b[row_offset], n_local * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(d_x_local, &x[row_offset], n_local * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_r_local, 0, n_local * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_p_local, 0, n_local * sizeof(double)));

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
    double *d_x_halo_prev = NULL, *d_x_halo_next = NULL;
    if (rank > 0) {
        CUDA_CHECK(cudaMalloc(&d_x_halo_prev, halo_size * sizeof(double)));
    }
    if (rank < world_size - 1) {
        CUDA_CHECK(cudaMalloc(&d_x_halo_next, halo_size * sizeof(double)));
    }

    exchange_halo_mpi_3d(d_x_local,                          // First plane to prev
                         d_x_local + (n_local - halo_size),  // Last plane to next
                         d_x_halo_prev, d_x_halo_next, h_send_prev, h_send_next, h_recv_prev,
                         h_recv_next, halo_size, rank, world_size, stream);

    // Initial SpMV: Ap = A*x
    stencil7_csr_partitioned_halo_kernel_3d<<<blocks_local, threads, 0, stream>>>(
        d_row_ptr, d_col_idx, d_values, d_x_local, d_x_halo_prev, d_x_halo_next, d_Ap, n_local,
        row_offset, n, grid_size);

    // r = b - Ap
    axpy_kernel<<<blocks_local, threads, 0, stream>>>(-1.0, d_Ap, d_b, n_local);
    CUDA_CHECK(cudaMemcpy(d_r_local, d_b, n_local * sizeof(double), cudaMemcpyDeviceToDevice));

    // Exchange r halo
    exchange_halo_mpi_3d(d_r_local, d_r_local + (n_local - halo_size), d_r_halo_prev, d_r_halo_next,
                         h_send_prev, h_send_next, h_recv_prev, h_recv_next, halo_size, rank,
                         world_size, stream);

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

    // rs_old = dot(r, r) + AllReduce
    double rs_local_old = compute_local_dot_3d(cublas_handle, d_r_local, d_r_local, n_local);
    double rs_old;
    MPI_Allreduce(&rs_local_old, &rs_old, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    double b_norm = sqrt(rs_old);

    if (rank == 0 && config.verbose >= 2) {
        printf("[Iter   0] Residual: %.6e\n", sqrt(rs_old));
    }

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
        double pAp_local = compute_local_dot_3d(cublas_handle, d_p_local, d_Ap, n_local);
        nvtxRangePop();

        double pAp;
        MPI_Allreduce(&pAp_local, &pAp, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        double alpha = rs_old / pAp;

        // x = x + alpha * p
        nvtxRangePush("BLAS_AXPY");
        axpy_kernel<<<blocks_local, threads, 0, stream>>>(alpha, d_p_local, d_x_local, n_local);

        // r = r - alpha * Ap
        axpy_kernel<<<blocks_local, threads, 0, stream>>>(-alpha, d_Ap, d_r_local, n_local);
        nvtxRangePop();

        // rs_new = dot(r, r)
        nvtxRangePush("Dot_Product");
        double rs_local_new = compute_local_dot_3d(cublas_handle, d_r_local, d_r_local, n_local);
        nvtxRangePop();

        double rs_new;
        MPI_Allreduce(&rs_local_new, &rs_new, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        double residual_norm = sqrt(rs_new);
        double rel_residual = residual_norm / b_norm;

        if (rank == 0 && config.verbose >= 2) {
            printf("[Iter %3d] Residual: %.6e (rel: %.6e, alpha: %.4e)\n", iter + 1, residual_norm,
                   rel_residual, alpha);
        }

        if (rel_residual < config.tolerance) {
            iter++;
            stats->converged = 1;
            stats->iterations = iter;
            stats->residual_norm = residual_norm;
            nvtxRangePop();
            break;
        }

        double beta = rs_new / rs_old;

        // p = r + beta * p
        nvtxRangePush("BLAS_AXPBY");
        axpby_kernel<<<blocks_local, threads, 0, stream>>>(1.0, d_r_local, beta, d_p_local,
                                                           n_local);
        nvtxRangePop();

        // Halo exchange for p (N² elements per direction)
        nvtxRangePush("Halo_Exchange_MPI_3D");
        exchange_halo_mpi_3d(d_p_local, d_p_local + (n_local - halo_size), d_p_halo_prev,
                             d_p_halo_next, h_send_prev, h_send_next, h_recv_prev, h_recv_next,
                             halo_size, rank, world_size, stream);
        nvtxRangePop();

        rs_old = rs_new;
        nvtxRangePop();
    }
    nvtxRangePop();

    if (iter == config.max_iters && rank == 0) {
        printf("\nMax iterations reached without convergence\n");
        stats->converged = 0;
        stats->iterations = iter;
        stats->residual_norm = sqrt(rs_old);
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

    if (h_send_prev)
        cudaFreeHost(h_send_prev);
    if (h_send_next)
        cudaFreeHost(h_send_next);
    if (h_recv_prev)
        cudaFreeHost(h_recv_prev);
    if (h_recv_next)
        cudaFreeHost(h_recv_next);

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

    CUDA_CHECK(cudaSetDevice(rank));

    int n_local = n / world_size;
    int row_offset = rank * n_local;
    if (rank == world_size - 1) {
        n_local = n - row_offset;
    }

    if (config.verbose >= 1) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, rank));
        printf("[Rank %d] GPU %d: %s (CC %d.%d)\n", rank, rank, prop.name, prop.major, prop.minor);
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

    double *d_x_local, *d_r_local, *d_p_local, *d_Ap, *d_b;
    double *d_p_halo_prev = NULL, *d_p_halo_next = NULL;
    double *d_r_halo_prev = NULL, *d_r_halo_next = NULL;

    CUDA_CHECK(cudaMalloc(&d_x_local, n_local * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_r_local, n_local * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_p_local, n_local * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Ap, n_local * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_b, n_local * sizeof(double)));

    if (rank > 0) {
        CUDA_CHECK(cudaMalloc(&d_p_halo_prev, halo_size * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_r_halo_prev, halo_size * sizeof(double)));
    }
    if (rank < world_size - 1) {
        CUDA_CHECK(cudaMalloc(&d_p_halo_next, halo_size * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_r_halo_next, halo_size * sizeof(double)));
    }

    double *h_send_prev = NULL, *h_send_next = NULL;
    double *h_recv_prev = NULL, *h_recv_next = NULL;
    if (rank > 0) {
        CUDA_CHECK(cudaMallocHost(&h_send_prev, halo_size * sizeof(double)));
        CUDA_CHECK(cudaMallocHost(&h_recv_prev, halo_size * sizeof(double)));
    }
    if (rank < world_size - 1) {
        CUDA_CHECK(cudaMallocHost(&h_send_next, halo_size * sizeof(double)));
        CUDA_CHECK(cudaMallocHost(&h_recv_next, halo_size * sizeof(double)));
    }

    CUDA_CHECK(cudaMemcpy(d_b, &b[row_offset], n_local * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(d_x_local, &x[row_offset], n_local * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_r_local, 0, n_local * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_p_local, 0, n_local * sizeof(double)));

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
    double *d_x_halo_prev = NULL, *d_x_halo_next = NULL;
    if (rank > 0) {
        CUDA_CHECK(cudaMalloc(&d_x_halo_prev, halo_size * sizeof(double)));
    }
    if (rank < world_size - 1) {
        CUDA_CHECK(cudaMalloc(&d_x_halo_next, halo_size * sizeof(double)));
    }

    exchange_halo_mpi_3d(d_x_local, d_x_local + (n_local - halo_size), d_x_halo_prev, d_x_halo_next,
                         h_send_prev, h_send_next, h_recv_prev, h_recv_next, halo_size, rank,
                         world_size, stream);

    // Initial SpMV: Ap = A*x (27-point kernel)
    stencil27_csr_partitioned_halo_kernel_3d<<<blocks_local, threads, 0, stream>>>(
        d_row_ptr, d_col_idx, d_values, d_x_local, d_x_halo_prev, d_x_halo_next, d_Ap, n_local,
        row_offset, n, grid_size);

    // r = b - Ap
    axpy_kernel<<<blocks_local, threads, 0, stream>>>(-1.0, d_Ap, d_b, n_local);
    CUDA_CHECK(cudaMemcpy(d_r_local, d_b, n_local * sizeof(double), cudaMemcpyDeviceToDevice));

    // Exchange r halo
    exchange_halo_mpi_3d(d_r_local, d_r_local + (n_local - halo_size), d_r_halo_prev, d_r_halo_next,
                         h_send_prev, h_send_next, h_recv_prev, h_recv_next, halo_size, rank,
                         world_size, stream);

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

    // rs_old
    double rs_local_old = compute_local_dot_3d(cublas_handle, d_r_local, d_r_local, n_local);
    double rs_old;
    MPI_Allreduce(&rs_local_old, &rs_old, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    double b_norm = sqrt(rs_old);

    if (rank == 0 && config.verbose >= 2) {
        printf("[Iter   0] Residual: %.6e\n", sqrt(rs_old));
    }

    // CG iteration loop
    nvtxRangePush("CG_Solver_27PT_3D");
    int iter;
    for (iter = 0; iter < config.max_iters; iter++) {
        nvtxRangePush("CG_Iteration_27PT_3D");

        // Ap = A * p (27-point kernel)
        nvtxRangePush("SpMV_27PT_3D");
        stencil27_csr_partitioned_halo_kernel_3d<<<blocks_local, threads, 0, stream>>>(
            d_row_ptr, d_col_idx, d_values, d_p_local, d_p_halo_prev, d_p_halo_next, d_Ap, n_local,
            row_offset, n, grid_size);
        nvtxRangePop();

        // alpha = rs_old / (p^T * Ap)
        nvtxRangePush("Dot_Product");
        double pAp_local = compute_local_dot_3d(cublas_handle, d_p_local, d_Ap, n_local);
        nvtxRangePop();

        double pAp;
        MPI_Allreduce(&pAp_local, &pAp, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        double alpha = rs_old / pAp;

        nvtxRangePush("BLAS_AXPY");
        axpy_kernel<<<blocks_local, threads, 0, stream>>>(alpha, d_p_local, d_x_local, n_local);

        axpy_kernel<<<blocks_local, threads, 0, stream>>>(-alpha, d_Ap, d_r_local, n_local);
        nvtxRangePop();

        nvtxRangePush("Dot_Product");
        double rs_local_new = compute_local_dot_3d(cublas_handle, d_r_local, d_r_local, n_local);
        nvtxRangePop();

        double rs_new;
        MPI_Allreduce(&rs_local_new, &rs_new, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        double residual_norm = sqrt(rs_new);
        double rel_residual = residual_norm / b_norm;

        if (rank == 0 && config.verbose >= 2) {
            printf("[Iter %3d] Residual: %.6e (rel: %.6e, alpha: %.4e)\n", iter + 1, residual_norm,
                   rel_residual, alpha);
        }

        if (rel_residual < config.tolerance) {
            iter++;
            stats->converged = 1;
            stats->iterations = iter;
            stats->residual_norm = residual_norm;
            nvtxRangePop();
            break;
        }

        double beta = rs_new / rs_old;

        nvtxRangePush("BLAS_AXPBY");
        axpby_kernel<<<blocks_local, threads, 0, stream>>>(1.0, d_r_local, beta, d_p_local,
                                                           n_local);
        nvtxRangePop();

        // Halo exchange for p
        nvtxRangePush("Halo_Exchange_MPI_27PT_3D");
        exchange_halo_mpi_3d(d_p_local, d_p_local + (n_local - halo_size), d_p_halo_prev,
                             d_p_halo_next, h_send_prev, h_send_next, h_recv_prev, h_recv_next,
                             halo_size, rank, world_size, stream);
        nvtxRangePop();

        rs_old = rs_new;
        nvtxRangePop();
    }
    nvtxRangePop();

    if (iter == config.max_iters && rank == 0) {
        printf("\nMax iterations reached without convergence\n");
        stats->converged = 0;
        stats->iterations = iter;
        stats->residual_norm = sqrt(rs_old);
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

    if (h_send_prev)
        cudaFreeHost(h_send_prev);
    if (h_send_next)
        cudaFreeHost(h_send_next);
    if (h_recv_prev)
        cudaFreeHost(h_recv_prev);
    if (h_recv_next)
        cudaFreeHost(h_recv_next);

    cublasDestroy(cublas_handle);
    cudaStreamDestroy(stream);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
