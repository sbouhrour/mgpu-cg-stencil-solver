/**
 * @file test_mgpu_cg_partitioned.cu
 * @brief Test program for partitioned multi-GPU CG solver
 *
 * Launch: mpirun -np 2 ./bin/test_mgpu_cg_partitioned matrix/stencil_512x512.mtx
 *
 * Author: Bouhrour Stephane
 * Date: 2025-11-11
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <cuda_profiler_api.h>
#include "io.h"
#include "spmv.h"
#include "solvers/cg_solver_mgpu_partitioned.h"
#include "solvers/cg_metrics.h"
#include "benchmark_stats_mgpu.h"

int main(int argc, char** argv) {
    // Initialize MPI
    MPI_Init(&argc, &argv);

    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (argc < 2) {
        if (rank == 0) {
            printf("Usage: mpirun -np <N> %s <matrix.mtx> [--timers] [--overlap] [--json=<file>] "
                   "[--csv=<file>]\n",
                   argv[0]);
            printf("Example: mpirun -np 2 %s matrix/stencil_512x512.mtx --json=results.json\n",
                   argv[0]);
            printf("Options:\n");
            printf("  --timers      Enable detailed timing breakdown (adds GPU sync overhead)\n");
            printf("  --overlap     Use compute-communication overlap solver\n");
            printf("  --json=<file> Export results to JSON file\n");
            printf("  --csv=<file>  Export results to CSV file\n");
        }
        MPI_Finalize();
        return 1;
    }

    const char* matrix_file = argv[1];
    const char* json_file = NULL;
    const char* csv_file = NULL;

    // Each rank loads matrix independently (avoids MPI_Bcast size limit)
    MatrixData mat;

    if (rank == 0) {
        printf("Loading matrix: %s\n", matrix_file);
    }

    if (load_matrix_market(matrix_file, &mat) != 0) {
        fprintf(stderr, "[Rank %d] Error loading matrix: %s\n", rank, matrix_file);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (rank == 0) {
        printf("Matrix loaded: %d × %d, %lld nonzeros\n", mat.rows, mat.cols, mat.nnz);
        printf("\nCalling partitioned multi-GPU CG solver...\n");
    }

    // Create deterministic RHS: b = ones (all ranks)
    double* b = (double*)malloc(mat.rows * sizeof(double));
    double* x = (double*)calloc(mat.rows, sizeof(double));

    for (int i = 0; i < mat.rows; i++) {
        b[i] = 1.0;
        x[i] = 0.0;
    }

    // CG configuration
    CGConfigMultiGPU config;
    config.max_iters = 1000;
    config.tolerance = 1e-6;
    config.verbose = 1;

    // Detailed timers: disabled by default (no sync overhead), enable with --timers flag
    config.enable_detailed_timers = 0;
    config.enable_overlap = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--timers") == 0) {
            config.enable_detailed_timers = 1;
            if (rank == 0) {
                printf("Detailed timers enabled (adds sync overhead)\n");
            }
        } else if (strcmp(argv[i], "--overlap") == 0) {
            config.enable_overlap = 1;
            if (rank == 0) {
                printf("Overlap solver enabled (compute-communication overlap)\n");
            }
        } else if (strncmp(argv[i], "--json=", 7) == 0) {
            json_file = argv[i] + 7;
        } else if (strncmp(argv[i], "--csv=", 6) == 0) {
            csv_file = argv[i] + 6;
        }
    }

    // Select solver function
    int (*solver_fn)(SpmvOperator*, MatrixData*, const double*, double*, CGConfigMultiGPU,
                     CGStatsMultiGPU*) =
        config.enable_overlap ? cg_solve_mgpu_partitioned_overlap : cg_solve_mgpu_partitioned;

    // Warmup: 3 full CG runs
    if (rank == 0)
        printf("Warmup (3 runs)...\n");
    CGConfigMultiGPU warmup_config = config;
    warmup_config.verbose = 0;
    CGStatsMultiGPU warmup_stats;
    for (int w = 0; w < 3; w++) {
        memset(x, 0, mat.rows * sizeof(double));
        solver_fn(NULL, &mat, b, x, warmup_config, &warmup_stats);
    }

    // Reset x to zero before profiled run (warmup leaves x = solution)
    memset(x, 0, mat.rows * sizeof(double));

    // Single profiled run (captured by nsys, excluded from statistics)
    if (rank == 0)
        printf("Running profiled iteration (for nsys)...\n");
    CGStatsMultiGPU profiled_stats;
    cudaProfilerStart();
    solver_fn(NULL, &mat, b, x, config, &profiled_stats);
    cudaProfilerStop();
    if (rank == 0) {
        printf("Profiled run: %s in %d iterations\n",
               profiled_stats.converged ? "converged" : "failed", profiled_stats.iterations);
    }

    // Reset x again before benchmark
    memset(x, 0, mat.rows * sizeof(double));

    BenchmarkStats bench_stats;
    CGStatsMultiGPU stats;

    if (!config.enable_overlap) {
        // Benchmark: 10 runs with statistical analysis (match AmgX default)
        if (rank == 0)
            printf("Running benchmark (10 runs)...\n");
        cg_benchmark_with_stats_mgpu_partitioned(NULL, &mat, b, x, config, 10, &bench_stats,
                                                 &stats);
        if (rank == 0) {
            printf("Completed: %d valid runs, %d outliers removed\n", bench_stats.valid_runs,
                   bench_stats.outliers_removed);
        }
    } else {
        // Overlap: single timed run (use profiled run stats)
        solver_fn(NULL, &mat, b, x, config, &stats);
        bench_stats.median_ms = stats.time_total_ms;
        bench_stats.mean_ms = stats.time_total_ms;
        bench_stats.min_ms = stats.time_total_ms;
        bench_stats.max_ms = stats.time_total_ms;
        bench_stats.std_dev_ms = 0.0;
        bench_stats.valid_runs = 1;
        bench_stats.outliers_removed = 0;
    }

    // Display results for verification (rank 0 only)
    if (rank == 0) {
        printf("\n========================================\n");
        printf("Results:\n");
        printf("========================================\n");
        printf("Converged: %s in %d iterations\n", stats.converged ? "YES" : "NO",
               stats.iterations);
        printf("Solution norm: %.15e\n", stats.residual_norm);

        // Timing summary with statistics
        if (config.enable_detailed_timers) {
            printf("\nTime (median): %.3f ms (SpMV: %.1f%%, BLAS1: %.1f%%, Reductions: %.1f%%, "
                   "Halo: %.1f%%)\n",
                   bench_stats.median_ms, 100.0 * stats.time_spmv_ms / stats.time_total_ms,
                   100.0 * stats.time_blas1_ms / stats.time_total_ms,
                   100.0 * stats.time_reductions_ms / stats.time_total_ms,
                   100.0 * stats.time_allgather_ms / stats.time_total_ms);
        } else {
            printf("\nTime (median): %.3f ms\n", bench_stats.median_ms);
        }
        if (bench_stats.valid_runs > 1) {
            printf("Stats: min=%.3f ms, max=%.3f ms, std=%.3f ms\n", bench_stats.min_ms,
                   bench_stats.max_ms, bench_stats.std_dev_ms);
        }

        // Solution checksum for verification
        double sum_x = 0.0, norm2_x = 0.0;
        for (int i = 0; i < mat.rows; i++) {
            sum_x += x[i];
            norm2_x += x[i] * x[i];
        }
        printf("\n=== Output Checksum ===\n");
        printf("Sum(x):    %.16e\n", sum_x);
        printf("Norm2(x):  %.16e\n", sqrt(norm2_x));
        printf("========================================\n");

        // Export results if requested
        if (json_file || csv_file) {
            if (json_file) {
                export_cg_mgpu_json(json_file, "partitioned-halo", &mat, &bench_stats, &stats,
                                    world_size);
                printf("\nResults exported to JSON: %s\n", json_file);
            }
            if (csv_file) {
                // TODO: Implement CSV export for multi-GPU CG
                printf("\nWarning: CSV export not yet implemented for multi-GPU CG\n");
            }
        }
    }

    // Cleanup
    free(b);
    free(x);
    if (mat.entries) {
        free(mat.entries);
    }

    MPI_Finalize();
    return 0;
}
