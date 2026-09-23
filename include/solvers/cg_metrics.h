/**
 * @file cg_metrics.h
 * @brief CG solver metrics export (JSON/CSV)
 *
 * Author: Bouhrour Stephane
 * Date: 2025-11-24
 */

#ifndef CG_METRICS_H
#define CG_METRICS_H

#include "solvers/cg_solver.h"
#include "solvers/cg_solver_mgpu.h"
#include "benchmark_stats.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Export CG benchmark results to JSON format
 */
void export_cg_json(const char* filename, const char* mode, const MatrixData* mat,
                    const BenchmarkStats* bench_stats, const CGStats* cg_stats);

/**
 * @brief Export CG multi-GPU benchmark results to JSON
 * @param comm Halo communication backend ("staged", "gpuaware", "nccl")
 */
void export_cg_mgpu_json(const char* filename, const char* mode, const char* comm,
                         const MatrixData* mat, const BenchmarkStats* bench_stats,
                         const CGStatsMultiGPU* cg_stats, int num_gpus);

/**
 * @brief Export CG benchmark results to CSV format
 */
void export_cg_csv(const char* filename, const char* mode, const MatrixData* mat,
                   const BenchmarkStats* bench_stats, const CGStats* cg_stats, bool write_header);

#ifdef __cplusplus
}
#endif

#endif  // CG_METRICS_H
