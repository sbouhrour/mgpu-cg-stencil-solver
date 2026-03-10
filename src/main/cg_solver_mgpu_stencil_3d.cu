/**
 * @file cg_solver_mgpu_stencil_3d.cu
 * @brief Multi-GPU CG solver entry point for 3D 7-point stencils
 *
 * Usage: mpirun -np N ./bin/cg_solver_mgpu_stencil_3d matrix/stencil3d_64.mtx [--timers]
 * [--overlap]
 *
 * Author: Bouhrour Stephane
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <cuda_profiler_api.h>
#include "io.h"
#include "spmv.h"
#include "solvers/cg_solver_mgpu_partitioned.h"
#include "solvers/cg_metrics.h"
#include "benchmark_stats_mgpu.h"

static int compare_doubles(const void* a, const void* b) {
    double da = *(const double*)a;
    double db = *(const double*)b;
    return (da > db) - (da < db);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (argc < 2) {
        if (rank == 0) {
            printf("Usage: mpirun -np <N> %s <matrix.mtx> [options]\n", argv[0]);
            printf("Example: mpirun -np 2 %s matrix/stencil3d_64.mtx --overlap\n", argv[0]);
            printf("Options:\n");
            printf("  --timers        Enable detailed timing breakdown\n");
            printf("  --overlap       Use compute-communication overlap solver\n");
            printf("  --verify        Use known solution (x=1) to verify correctness\n");
            printf("  --max-iters=N   Set maximum CG iterations (default: 1000)\n");
            printf("  --json=<file>   Export results to JSON file\n");
            printf("  --stencil=N     Stencil type: 7 (default) or 27\n");
        }
        MPI_Finalize();
        return 1;
    }

    const char* matrix_file = argv[1];
    const char* json_file = NULL;
    int verify_mode = 0;
    int custom_max_iters = 0;
    int max_iters_value = 1000;
    int stencil_points = 7;  // default: 7-point stencil

    // Configuration
    CGConfigMultiGPU config;
    config.max_iters = 1000;
    config.tolerance = 1e-6;
    config.verbose = 1;
    config.enable_detailed_timers = 0;
    config.enable_overlap = 0;

    // Parse arguments before using them
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--timers") == 0) {
            config.enable_detailed_timers = 1;
            if (rank == 0)
                printf("Detailed timers enabled\n");
        } else if (strcmp(argv[i], "--overlap") == 0) {
            config.enable_overlap = 1;
            if (rank == 0)
                printf("Overlap solver enabled\n");
        } else if (strcmp(argv[i], "--verify") == 0) {
            verify_mode = 1;
            if (rank == 0)
                printf("Verify mode: using known solution x=1\n");
        } else if (strncmp(argv[i], "--max-iters=", 12) == 0) {
            custom_max_iters = 1;
            max_iters_value = atoi(argv[i] + 12);
        } else if (strncmp(argv[i], "--json=", 7) == 0) {
            json_file = argv[i] + 7;
        } else if (strncmp(argv[i], "--stencil=", 10) == 0) {
            stencil_points = atoi(argv[i] + 10);
            if (stencil_points != 7 && stencil_points != 27) {
                if (rank == 0)
                    fprintf(stderr, "Error: --stencil must be 7 or 27\n");
                MPI_Finalize();
                return 1;
            }
            if (rank == 0)
                printf("Stencil: %d-point\n", stencil_points);
        }
    }

    if (custom_max_iters) {
        config.max_iters = max_iters_value;
        if (rank == 0)
            printf("Max iterations: %d\n", config.max_iters);
    }

    MatrixData mat;

    if (rank == 0) {
        printf("Loading 3D matrix: %s\n", matrix_file);
    }

    // For 27pt stencil, generate entries in memory instead of reading the large text file
    // (e.g. N=384 produces a 34 GB file — fscanf on 1.5B entries takes 20-40 minutes)
    int load_err;
    if (stencil_points == 27) {
        load_err = load_matrix_stencil27_3d_from_grid(matrix_file, &mat, rank, world_size);
    } else {
        load_err = load_matrix_market(matrix_file, &mat);
    }
    if (load_err != 0) {
        fprintf(stderr, "[Rank %d] Error loading matrix: %s\n", rank, matrix_file);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (rank == 0) {
        printf("Matrix loaded: %d x %d, %lld nonzeros (grid_size=%d, 3D)\n", mat.rows, mat.cols,
               mat.nnz, mat.grid_size);
    }

    // RHS
    double* b = (double*)calloc(mat.rows, sizeof(double));
    double* x = (double*)calloc(mat.rows, sizeof(double));

    if (verify_mode) {
        // b = A * x_exact where x_exact = ones (use COO entries)
        for (long long k = 0; k < mat.nnz; k++) {
            b[mat.entries[k].row] += mat.entries[k].value;
        }
        if (rank == 0) {
            double b_sum = 0.0;
            for (int i = 0; i < mat.rows; i++)
                b_sum += b[i];
            printf("Verify: b = A * ones, sum(b) = %.6e\n", b_sum);
        }
    } else {
        for (int i = 0; i < mat.rows; i++) {
            b[i] = 1.0;
        }
    }

    // Select solver based on stencil type and overlap mode
    int (*solver_fn)(SpmvOperator*, MatrixData*, const double*, double*, CGConfigMultiGPU,
                     CGStatsMultiGPU*);
    int (*warmup_fn)(SpmvOperator*, MatrixData*, const double*, double*, CGConfigMultiGPU,
                     CGStatsMultiGPU*);

    if (stencil_points == 27) {
        solver_fn = config.enable_overlap ? cg_solve_mgpu_partitioned_overlap_27pt_3d
                                          : cg_solve_mgpu_partitioned_27pt_3d;
        warmup_fn = cg_solve_mgpu_partitioned_27pt_3d;
    } else {
        solver_fn = config.enable_overlap ? cg_solve_mgpu_partitioned_overlap_3d
                                          : cg_solve_mgpu_partitioned_3d;
        warmup_fn = cg_solve_mgpu_partitioned_3d;
    }

    // Warmup: 3 runs with synchronous solver
    if (rank == 0)
        printf("Warmup (3 runs, synchronous)...\n");
    CGConfigMultiGPU warmup_config = config;
    warmup_config.verbose = 0;
    CGStatsMultiGPU warmup_stats;
    for (int w = 0; w < 3; w++) {
        memset(x, 0, mat.rows * sizeof(double));
        warmup_fn(NULL, &mat, b, x, warmup_config, &warmup_stats);
    }

    // Profiled run
    memset(x, 0, mat.rows * sizeof(double));
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

    // Benchmark: 10 runs with inline median
    memset(x, 0, mat.rows * sizeof(double));
    if (rank == 0)
        printf("Running benchmark (10 runs)...\n");

    const int num_runs = 10;
    double times[10];
    CGStatsMultiGPU all_stats[10];
    CGConfigMultiGPU bench_config = config;
    bench_config.verbose = 0;

    int valid_count = 0;
    for (int r = 0; r < num_runs; r++) {
        memset(x, 0, mat.rows * sizeof(double));
        if (solver_fn(NULL, &mat, b, x, bench_config, &all_stats[valid_count]) == 0) {
            times[valid_count] = all_stats[valid_count].time_total_ms;
            valid_count++;
        }
    }

    // Compute median
    BenchmarkStats bench_stats;
    CGStatsMultiGPU stats;

    if (valid_count >= 3) {
        // Sort times for median
        double sorted_times[10];
        memcpy(sorted_times, times, valid_count * sizeof(double));
        qsort(sorted_times, valid_count, sizeof(double), compare_doubles);

        double median;
        if (valid_count % 2 == 0) {
            median = (sorted_times[valid_count / 2 - 1] + sorted_times[valid_count / 2]) / 2.0;
        } else {
            median = sorted_times[valid_count / 2];
        }

        // Find which run produced the median time
        int median_idx = 0;
        double min_diff = 1e30;
        for (int r = 0; r < valid_count; r++) {
            double diff = fabs(times[r] - median);
            if (diff < min_diff) {
                min_diff = diff;
                median_idx = r;
            }
        }
        stats = all_stats[median_idx];

        // Stats
        double sum = 0.0;
        for (int r = 0; r < valid_count; r++)
            sum += times[r];
        double mean = sum / valid_count;

        double sum_sq = 0.0;
        for (int r = 0; r < valid_count; r++) {
            double d = times[r] - mean;
            sum_sq += d * d;
        }

        bench_stats.median_ms = median;
        bench_stats.mean_ms = mean;
        bench_stats.std_dev_ms = sqrt(sum_sq / valid_count);
        bench_stats.min_ms = sorted_times[0];
        bench_stats.max_ms = sorted_times[valid_count - 1];
        bench_stats.valid_runs = valid_count;
        bench_stats.outliers_removed = num_runs - valid_count;
    } else {
        // Fallback: use profiled run
        stats = profiled_stats;
        bench_stats.median_ms = stats.time_total_ms;
        bench_stats.mean_ms = stats.time_total_ms;
        bench_stats.min_ms = stats.time_total_ms;
        bench_stats.max_ms = stats.time_total_ms;
        bench_stats.std_dev_ms = 0.0;
        bench_stats.valid_runs = 1;
        bench_stats.outliers_removed = 0;
    }

    if (rank == 0) {
        printf("Completed: %d valid runs\n", bench_stats.valid_runs);
    }

    // Display results
    if (rank == 0) {
        printf("\n========================================\n");
        printf("3D CG Solver Results (%d-point stencil):\n", stencil_points);
        printf("========================================\n");
        printf("Converged: %s in %d iterations\n", stats.converged ? "YES" : "NO",
               stats.iterations);
        printf("Residual norm: %.15e\n", stats.residual_norm);
        printf("Time (median): %.3f ms\n", bench_stats.median_ms);

        if (bench_stats.valid_runs > 1) {
            printf("Stats: min=%.3f ms, max=%.3f ms, std=%.3f ms\n", bench_stats.min_ms,
                   bench_stats.max_ms, bench_stats.std_dev_ms);
        }

        // Solution checksum
        double sum_x = 0.0, norm2_x = 0.0;
        for (int i = 0; i < mat.rows; i++) {
            sum_x += x[i];
            norm2_x += x[i] * x[i];
        }
        printf("\n=== Output Checksum ===\n");
        printf("Sum(x):    %.16e\n", sum_x);
        printf("Norm2(x):  %.16e\n", sqrt(norm2_x));

        if (verify_mode) {
            double err_sq = 0.0, exact_sq = 0.0;
            for (int i = 0; i < mat.rows; i++) {
                double diff = x[i] - 1.0;
                err_sq += diff * diff;
                exact_sq += 1.0;
            }
            double rel_error = sqrt(err_sq) / sqrt(exact_sq);
            printf("\n=== Verification ===\n");
            printf("x_exact:         ones (%d elements)\n", mat.rows);
            printf("||x - x_exact||: %.6e\n", sqrt(err_sq));
            printf("||x_exact||:     %.6e\n", sqrt(exact_sq));
            printf("Relative error:  %.6e\n", rel_error);
            printf("VERIFY: %s\n", rel_error < config.tolerance * 10.0 ? "PASS" : "FAIL");
        }

        printf("========================================\n");

        if (json_file) {
            const char* mode_str = (stencil_points == 27) ? "3d-stencil-27pt" : "3d-stencil";
            export_cg_mgpu_json(json_file, mode_str, &mat, &bench_stats, &stats, world_size);
            printf("\nResults exported to JSON: %s\n", json_file);
        }
    }

    free(b);
    free(x);
    if (mat.entries)
        free(mat.entries);

    MPI_Finalize();
    return 0;
}
