/**
 * @file cg_solver_mgpu.h
 * @brief Shared configuration and timing types for the multi-GPU CG solvers
 *
 * @details
 * Architecture: 1 MPI rank per GPU.
 * - MPI: process management, dot-product reductions, halo exchange
 * - CUDA: local computation on each GPU
 *
 * The solver built on these types is in `cg_solver_mgpu_partitioned.h`: rows are partitioned across
 * ranks in a 1D band decomposition, each rank owns its slice of the vectors, and neighbours
 * exchange one boundary row through explicit D2H -> MPI_Isend/Irecv -> H2D staging.
 *
 * Communication pattern:
 *   Local SpMV:   y_local = A_local * [x_local | halo]
 *   Local BLAS1:  operates on local row segments
 *   Dot products: local sum + MPI_Allreduce
 *   Halo:         one boundary row per neighbour, staged through pinned host buffers
 *
 * An earlier design replicated the full vector on every rank and synchronised it with NCCL
 * AllGather, which is what `time_allgather_ms` below was originally named for; that field now
 * records halo exchange time. The NCCL implementations and the measurements that led to MPI staging
 * are on their branches — see "Approaches tried and set aside" in `docs/development.md`.
 *
 * Launch: mpirun -np <num_gpus> ./cg_solver_mgpu_stencil matrix.mtx
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
    int spmv_soa;                ///< Use coefficient-major (SoA) SpMV (27-point 3D sync solver)
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
    double time_allreduce_ms;   ///< MPI_Allreduce time (dot-product reductions)
    double time_allgather_ms;   ///< Halo exchange time (name predates the switch from AllGather)
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

#endif  // CG_SOLVER_MGPU_H
