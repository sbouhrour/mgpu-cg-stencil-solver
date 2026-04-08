/**
 * @file cg_test.cu
 * @brief CG solver benchmark with multi-mode support
 *
 * @details
 * Solves a Laplacian system using CG with different SpMV operators.
 * Supports comma-separated modes for comparison without CSR rebuild.
 *
 * Author: Bouhrour Stephane
 * Date: 2025-10-14
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "io.h"
#include "spmv.h"
#include "solvers/cg_solver.h"
#include "solvers/cg_metrics.h"
#include "benchmark_stats.h"

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s <matrix.mtx> [--mode=<modes>] [--host] [--tol=<tol>] [--maxiter=<n>] "
               "[--timers] [--json=<file>] [--csv=<file>]\n",
               argv[0]);
        printf("Example: %s matrix/stencil_512x512.mtx --mode=cusparse-csr,stencil5-csr "
               "--json=results/cg.json\n",
               argv[0]);
        printf("\nOptions:\n");
        printf(
            "  --mode=<modes>         SpMV operators, comma-separated (default: stencil5-csr)\n");
        printf("  --host                 Use host interface (default: device-native GPU)\n");
        printf("  --tol=<tol>            Convergence tolerance (default: 1e-6)\n");
        printf("  --maxiter=<n>          Maximum iterations (default: 1000)\n");
        printf(
            "  --timers               Enable detailed per-category timing (adds sync overhead)\n");
        printf("  --json=<file>          Export results to JSON (one file per mode: "
               "<file>_<mode>.json)\n");
        printf(
            "  --csv=<file>           Export results to CSV (append all modes to single file)\n");
        return 1;
    }

    const char* matrix_file = argv[1];
    const char* modes_string = "stencil5-csr";  // Default
    bool use_device = true;
    bool enable_detailed_timers = false;  // Opt-in with --timers (avoids sync overhead)
    double tolerance = 1e-6;
    int max_iters = 5000;
    const char* json_file = NULL;
    const char* csv_file = NULL;

    // Parse arguments
    for (int i = 2; i < argc; i++) {
        if (strncmp(argv[i], "--mode=", 7) == 0) {
            modes_string = argv[i] + 7;
        } else if (strcmp(argv[i], "--host") == 0) {
            use_device = false;
        } else if (strcmp(argv[i], "--device") == 0) {
            use_device = true;
        } else if (strcmp(argv[i], "--timers") == 0) {
            enable_detailed_timers = true;
        } else if (strncmp(argv[i], "--tol=", 6) == 0) {
            tolerance = atof(argv[i] + 6);
        } else if (strncmp(argv[i], "--maxiter=", 10) == 0) {
            max_iters = atoi(argv[i] + 10);
        } else if (strncmp(argv[i], "--json=", 7) == 0) {
            json_file = argv[i] + 7;
        } else if (strncmp(argv[i], "--csv=", 6) == 0) {
            csv_file = argv[i] + 6;
        }
    }

    // Parse modes (split by comma)
    char modes_buffer[256];
    strncpy(modes_buffer, modes_string, sizeof(modes_buffer) - 1);
    modes_buffer[sizeof(modes_buffer) - 1] = '\0';

    const char* mode_tokens[10];
    int num_modes = 0;

    char* token = strtok(modes_buffer, ",");
    while (token != NULL && num_modes < 10) {
        mode_tokens[num_modes++] = token;
        token = strtok(NULL, ",");
    }

    // Validate all modes before loading matrix
    printf("Validating %d mode(s): ", num_modes);
    for (int i = 0; i < num_modes; i++) {
        printf("%s%s", mode_tokens[i], (i < num_modes - 1) ? ", " : "\n");
        SpmvOperator* op = get_operator(mode_tokens[i]);
        if (!op) {
            fprintf(stderr, "Error: unknown SpMV mode '%s'\n", mode_tokens[i]);
            fprintf(stderr, "Available: cusparse-csr, stencil5-csr\n");
            return 1;
        }
        if (use_device && !op->run_device) {
            fprintf(stderr, "Error: mode '%s' does not support device-native interface\n",
                    mode_tokens[i]);
            return 1;
        }
    }

    // Load matrix
    printf("\nLoading matrix: %s\n", matrix_file);
    MatrixData mat;
    if (load_matrix_market(matrix_file, &mat) != 0) {
        fprintf(stderr, "Error loading matrix\n");
        return 1;
    }
    printf("Matrix loaded: %d x %d, %d nonzeros\n", mat.rows, mat.cols, mat.nnz);

    // Multi-mode warning
    if (num_modes > 1) {
        printf(
            "\nNOTE: Multi-mode benchmark - performance may vary with order due to GPU state.\n");
        printf("      For accurate comparison, run each mode separately.\n");
    }

    // Create RHS and solution vectors (shared across modes)
    double* b = (double*)malloc(mat.rows * sizeof(double));
    double* x = (double*)calloc(mat.rows, sizeof(double));
    for (int i = 0; i < mat.rows; i++) {
        b[i] = 1.0;
    }

    // CG configuration
    CGConfig config;
    config.max_iters = max_iters;
    config.tolerance = tolerance;
    config.verbose = 2;
    config.enable_detailed_timers = enable_detailed_timers;

    // Loop through all modes
    for (int mode_idx = 0; mode_idx < num_modes; mode_idx++) {
        const char* current_mode = mode_tokens[mode_idx];

        printf("\n========================================\n");
        printf("CG Solver - Mode: %s\n", current_mode);
        printf("========================================\n");
        printf("Interface: %s\n", use_device ? "Device-native (GPU)" : "Host");
        printf("Tolerance: %.1e, Max iterations: %d\n", tolerance, max_iters);

        // Get and initialize operator
        SpmvOperator* spmv_op = get_operator(current_mode);
        spmv_op->init(&mat);

        // Reset solution vector
        memset(x, 0, mat.rows * sizeof(double));

        // Warmup: 3 CG iterations (HPC best practice)
        printf("Warmup (3 runs)...\n");
        CGConfig warmup_config = {max_iters, tolerance, 0, false};
        CGStats warmup_stats;
        for (int w = 0; w < 3; w++) {
            memset(x, 0, mat.rows * sizeof(double));
            if (use_device) {
                cg_solve_device(spmv_op, &mat, b, x, warmup_config, &warmup_stats);
            } else {
                cg_solve(spmv_op, &mat, b, x, warmup_config, &warmup_stats);
            }
        }

        // Benchmark: 10 runs with statistical analysis
        printf("Running benchmark (10 runs)...\n");
        BenchmarkStats bench_stats;
        CGStats stats;

        if (use_device) {
            cg_benchmark_with_stats_device(spmv_op, &mat, b, x, config, 10, &bench_stats, &stats);
        } else {
            // For host interface, run single iteration (host is not performance-critical)
            memset(x, 0, mat.rows * sizeof(double));
            cg_solve(spmv_op, &mat, b, x, config, &stats);
            bench_stats.median_ms = stats.time_total_ms;
            bench_stats.valid_runs = 1;
            bench_stats.outliers_removed = 0;
        }

        printf("Completed: %d valid runs, %d outliers removed\n", bench_stats.valid_runs,
               bench_stats.outliers_removed);

        // Verify solution
        double error = 0.0;
        for (int i = 0; i < mat.rows; i++) {
            double diff = x[i] - 1.0;
            error += diff * diff;
        }
        error = sqrt(error / mat.rows);

        // Summary for this mode
        printf("\n--- Results for %s ---\n", current_mode);
        printf("Converged: %s in %d iterations\n", stats.converged ? "YES" : "NO",
               stats.iterations);
        // Only show breakdown if --timers flag was used
        if (config.enable_detailed_timers) {
            printf("Time (median): %.3f ms (SpMV: %.1f%%, BLAS1: %.1f%%, Reductions: %.1f%%)\n",
                   bench_stats.median_ms, 100.0 * stats.time_spmv_ms / stats.time_total_ms,
                   100.0 * stats.time_blas1_ms / stats.time_total_ms,
                   100.0 * stats.time_reductions_ms / stats.time_total_ms);
        } else {
            printf("Time (median): %.3f ms\n", bench_stats.median_ms);
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

        // Export results if requested
        if (json_file) {
            char json_filename[512];
            snprintf(json_filename, sizeof(json_filename), "%s_%s.json", json_file, current_mode);
            export_cg_json(json_filename, current_mode, &mat, &bench_stats, &stats);
        }
        if (csv_file) {
            bool write_header = (mode_idx == 0);  // Header only for first mode
            export_cg_csv(csv_file, current_mode, &mat, &bench_stats, &stats, write_header);
        }

        // Cleanup operator (GPU memory only, CSR host preserved)
        spmv_op->free();
    }

    // Final cleanup
    printf("\n========================================\n");
    free(b);
    free(x);
    free(mat.entries);

    return 0;
}
