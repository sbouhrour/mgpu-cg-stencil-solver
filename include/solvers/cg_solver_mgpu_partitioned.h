/**
 * @file cg_solver_mgpu_partitioned.h
 * @brief Multi-GPU CG solver with true CSR partitioning
 *
 * @details
 * Architecture: Local CSR partition + halo zone exchange
 * - Each GPU stores only its CSR partition (not full matrix)
 * - Halo zones: minimal ghost cells for stencil neighbors
 * - Communication: P2P exchange of boundary rows only
 *
 * Memory footprint (10000×10000 stencil, 2 GPUs):
 * - Full replication: 6.4 GB CSR + 3.2 GB vectors = 9.6 GB per GPU
 * - Partitioned: 3.2 GB CSR + 1.6 GB local vectors + 80 KB halo = 4.8 GB per GPU
 *
 * Communication pattern:
 * - SpMV: Halo exchange before multiplication (~80 KB, 1 row for 5-point stencil)
 * - Dot products: NCCL AllReduce (8 bytes scalar)
 * - BLAS1: Local operations only (no communication)
 *
 * vs Full-replication:
 * - Communication: 80 KB vs 400 MB per iteration (5000× reduction)
 * - Memory: 4.8 GB vs 9.6 GB per GPU (2× reduction)
 *
 * Author: Bouhrour Stephane
 * Date: 2025-11-11
 */

#ifndef CG_SOLVER_MGPU_PARTITIONED_H
#define CG_SOLVER_MGPU_PARTITIONED_H

#include "spmv.h"
#include "solvers/cg_solver.h"
#include "solvers/cg_solver_mgpu.h"

/**
 * @brief Multi-GPU CG solver with CSR partitioning (scalable version)
 *
 * Configuration: 2+ GPUs, NCCL P2P communication
 * Memory: Local CSR partition + minimal halo zones
 * Communication: Halo exchange (KB) instead of full AllGather (MB)
 *
 * @param spmv_op SpMV operator (unused, we use simple CSR kernel)
 * @param mat Matrix data (for partitioning)
 * @param b Right-hand side vector
 * @param x Solution vector (output)
 * @param config Multi-GPU CG configuration
 * @param stats Output statistics
 * @return 0 on success
 */
int cg_solve_mgpu_partitioned(SpmvOperator* spmv_op, MatrixData* mat, const double* b, double* x,
                              CGConfigMultiGPU config, CGStatsMultiGPU* stats);

/**
 * @brief Multi-GPU CG solver with compute-communication overlap
 *
 * Overlaps interior SpMV with MPI halo exchange using two CUDA streams.
 * Boundary SpMV executes after halo data arrives.
 * Results are identical to the synchronous partitioned solver.
 *
 * @param spmv_op SpMV operator (unused, uses internal stencil kernel)
 * @param mat Matrix data (for partitioning)
 * @param b Right-hand side vector
 * @param x Solution vector (output)
 * @param config Multi-GPU CG configuration (enable_overlap ignored here)
 * @param stats Output statistics (includes overlap metrics when detailed timers enabled)
 * @return 0 on success
 */
int cg_solve_mgpu_partitioned_overlap(SpmvOperator* spmv_op, MatrixData* mat, const double* b,
                                      double* x, CGConfigMultiGPU config, CGStatsMultiGPU* stats);

/**
 * @brief Multi-GPU CG solver for 3D 7-point stencil with Z-slab partitioning
 *
 * Synchronous version: halo exchange completes before SpMV.
 * Halo = one XY-plane (grid_size² elements) per neighbor direction.
 */
int cg_solve_mgpu_partitioned_3d(SpmvOperator* spmv_op, MatrixData* mat, const double* b, double* x,
                                 CGConfigMultiGPU config, CGStatsMultiGPU* stats);

/**
 * @brief Multi-GPU CG solver for 3D stencil with compute-communication overlap
 *
 * Overlaps interior SpMV with MPI halo exchange.
 * Results are identical to the synchronous 3D solver.
 */
int cg_solve_mgpu_partitioned_overlap_3d(SpmvOperator* spmv_op, MatrixData* mat, const double* b,
                                         double* x, CGConfigMultiGPU config,
                                         CGStatsMultiGPU* stats);

/**
 * @brief Multi-GPU CG solver for 3D 27-point stencil with Z-slab partitioning
 *
 * Synchronous version: halo exchange completes before SpMV.
 * Halo = one XY-plane (grid_size² elements) per neighbor direction.
 */
int cg_solve_mgpu_partitioned_27pt_3d(SpmvOperator* spmv_op, MatrixData* mat, const double* b,
                                      double* x, CGConfigMultiGPU config, CGStatsMultiGPU* stats);

/**
 * @brief Multi-GPU CG solver for 3D 27-point stencil with compute-communication overlap
 *
 * Overlaps interior SpMV with MPI halo exchange.
 * Results are identical to the synchronous 27-point 3D solver.
 */
int cg_solve_mgpu_partitioned_overlap_27pt_3d(SpmvOperator* spmv_op, MatrixData* mat,
                                              const double* b, double* x, CGConfigMultiGPU config,
                                              CGStatsMultiGPU* stats);

#endif  // CG_SOLVER_MGPU_PARTITIONED_H
