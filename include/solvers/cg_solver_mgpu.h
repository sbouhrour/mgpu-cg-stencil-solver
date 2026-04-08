/**
 * @file cg_solver_mgpu.h
 * @brief Multi-GPU Conjugate Gradient solver with MPI+NCCL
 *
 * @details
 * Architecture: 1 MPI rank per GPU (standard HPC pattern)
 * - MPI: Process management and bootstrapping
 * - NCCL: GPU-GPU communication (AllReduce, AllGather)
 * - CUDA: Local computation on each GPU
 *
 * Phase 1: Full vector replication
 * - Each rank/GPU maintains complete vectors (x, r, p, Ap)
 * - Matrix rows partitioned across ranks (1D row-band decomposition)
 * - NCCL AllReduce for dot products
 * - NCCL AllGather for vector synchronization after updates
 *
 * Communication pattern:
 *   Local SpMV: y_local = A_local * x_full (no communication)
 *   Local BLAS1: operates on local row segments
 *   Dot products: local sum + NCCL AllReduce
 *   Vector sync: NCCL AllGather (200 MB for stencil 5000×5000)
 *
 * Launch: mpirun -np <num_gpus> ./cg_mgpu matrix.mtx
 *
 * Author: Bouhrour Stephane
 * Date: 2025-11-06
 */

#ifndef CG_SOLVER_MGPU_H
#define CG_SOLVER_MGPU_H

#include "spmv.h"
#include "solvers/cg_solver.h"

/**
 * @brief Multi-GPU CG configuration (per MPI rank)
 */
typedef struct {
    int max_iters;               ///< Maximum iterations
    double tolerance;            ///< Convergence tolerance
    int verbose;                 ///< Verbosity level (0=silent, 1=summary, 2=per-iter)
    int enable_detailed_timers;  ///< Enable timing breakdown
    int enable_overlap;          ///< Enable compute-communication overlap
} CGConfigMultiGPU;

/**
 * @brief Multi-GPU CG statistics
 */
typedef struct {
    int iterations;             ///< Actual iterations performed
    double residual_norm;       ///< Final residual norm
    double time_total_ms;       ///< Total solve time
    double time_spmv_ms;        ///< SpMV time
    double time_blas1_ms;       ///< BLAS1 operations time (total)
    double time_reductions_ms;  ///< Dot products time (total)
    double time_allreduce_ms;   ///< NCCL AllReduce time
    double time_allgather_ms;   ///< NCCL AllGather time (or halo exchange)
    int converged;              ///< 1 if converged

    // Granular BLAS1 timings (per-iteration averages)
    double time_dot_rs_initial_ms;  ///< Initial dot(r,r) before loop
    double time_dot_pAp_ms;         ///< dot(p, Ap) average per iteration
    double time_dot_rs_new_ms;      ///< dot(r, r) convergence check average
    double time_axpy_update_x_ms;   ///< x = x + alpha*p average
    double time_axpy_update_r_ms;   ///< r = r - alpha*Ap average
    double time_axpby_update_p_ms;  ///< p = r + beta*p average
    double time_initial_r_ms;       ///< Initial r = b - A*x0

    // Solution validation
    double solution_sum;   ///< Sum of solution vector elements
    double solution_norm;  ///< L2 norm of solution vector

    // Overlap metrics (populated when enable_overlap=1 and enable_detailed_timers=1)
    double time_spmv_interior_ms;  ///< Interior SpMV time (halo-independent rows)
    double time_spmv_boundary_ms;  ///< Boundary SpMV time (halo-dependent rows)
    double time_comm_total_ms;     ///< Total communication time (D2H + MPI + H2D)
    double time_comm_hidden_ms;    ///< Communication hidden behind interior compute
    double time_comm_exposed_ms;   ///< Exposed communication (not hidden)
    double overlap_efficiency;     ///< Fraction of comm hidden (0.0 to 1.0)
} CGStatsMultiGPU;

/**
 * @brief Multi-GPU CG solver with full vector replication (Phase 1)
 *
 * Configuration: 2-4 GPUs, NCCL communication
 * Memory: Full vectors replicated on each GPU
 * Communication: AllReduce (dot products) + AllGather (vector sync)
 *
 * @param spmv_op SpMV operator (must support multi-GPU)
 * @param mat Matrix data (for partitioning)
 * @param b Right-hand side vector
 * @param x Solution vector (output)
 * @param config Multi-GPU CG configuration
 * @param stats Output statistics
 * @return 0 on success
 */
int cg_solve_mgpu(SpmvOperator* spmv_op, MatrixData* mat, const double* b, double* x,
                  CGConfigMultiGPU config, CGStatsMultiGPU* stats);

#endif  // CG_SOLVER_MGPU_H
