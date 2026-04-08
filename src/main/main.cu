/**
 * @file main.cu
 * @brief Main entry point for benchmarking different SpMV implementations.
 *
 * @details
 * This file serves as the primary driver for the SpMV benchmark suite:
 * - Parses command-line arguments to determine benchmark mode and matrix file
 * - Loads sparse matrices from Matrix Market (.mtx) format files
 * - Selects and initializes the appropriate SpMV operator (CSR, ELLPACK, or STENCIL)
 * - Allocates and initializes input/output vectors on host memory
 * - Executes the SpMV benchmark using the selected GPU implementation
 * - Manages memory cleanup and resource deallocation
 *
 * The program supports three SpMV implementations:
 * - CSR (Compressed Sparse Row) format
 * - ELLPACK format for regular sparsity patterns
 * - STENCIL format for structured grid operations
 *
 * Author: Bouhrour Stephane
 * Date: 2025-07-15
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include "spmv.h"
#include "io.h"
#include "benchmark_stats.h"

/**
 * @brief Main function - Entry point for SpMV benchmark program.
 * @details Orchestrates the entire benchmark workflow including argument parsing,
 * matrix loading, operator selection, memory management, and benchmark execution.
 * The function expects command-line arguments specifying the matrix file and
 * the desired SpMV implementation mode.
 *
 * Expected usage: ./program <matrix_file.mtx> --mode=<csr-cusparse|stencil5-csr>
 *
 * @param argc Number of command-line arguments
 * @param argv Array of command-line argument strings
 * @return EXIT_SUCCESS (0) on successful completion, EXIT_FAILURE (1) on error
 */
int main(int argc, char* argv[]) {
    // Check for correct number of command-line arguments
    if (argc < 3) {
        fprintf(stderr,
                "Usage: %s <matrix_file.mtx> --mode=<mode1[,mode2,...]> [--json=<file>] "
                "[--csv=<file>]\n",
                argv[0]);
        fprintf(stderr, "Available modes: cusparse-csr, stencil5-csr\n");
        fprintf(stderr,
                "Example: %s matrix.mtx --mode=cusparse-csr,stencil5-csr --json=results.json\n",
                argv[0]);
        return EXIT_FAILURE;
    }

    const char* matrix_file = argv[1];  ///< Path to Matrix Market file containing sparse matrix
    const char* modes_string = NULL;    ///< SpMV implementation modes (comma-separated)
    const char* json_file = NULL;       ///< JSON output file
    const char* csv_file = NULL;        ///< CSV output file

    // Parse command-line arguments
    for (int i = 2; i < argc; ++i) {
        if (strncmp(argv[i], "--mode=", 7) == 0) {
            modes_string = argv[i] + 7;
        } else if (strncmp(argv[i], "--json=", 7) == 0) {
            json_file = argv[i] + 7;
        } else if (strncmp(argv[i], "--csv=", 6) == 0) {
            csv_file = argv[i] + 6;
        }
    }

    // Validate that modes were specified
    if (modes_string == NULL) {
        fprintf(stderr, "Error: mode not specified. Use --mode=<mode1[,mode2,...]>\n");
        return EXIT_FAILURE;
    }

    // Parse modes (split by comma) BEFORE loading matrix
    char modes_buffer[256];
    strncpy(modes_buffer, modes_string, sizeof(modes_buffer) - 1);
    modes_buffer[sizeof(modes_buffer) - 1] = '\0';

    const char* mode_tokens[10];  // Support up to 10 modes
    int num_modes = 0;

    char* token = strtok(modes_buffer, ",");
    while (token != NULL && num_modes < 10) {
        mode_tokens[num_modes++] = token;
        token = strtok(NULL, ",");
    }

    // Validate all modes BEFORE loading matrix (saves time on invalid modes)
    printf("Validating %d mode(s): ", num_modes);
    for (int i = 0; i < num_modes; i++) {
        printf("%s%s", mode_tokens[i], (i < num_modes - 1) ? ", " : "\n");

        SpmvOperator* op = get_operator(mode_tokens[i]);
        if (op == NULL) {
            fprintf(stderr, "Error: Unknown mode '%s'\n", mode_tokens[i]);
            fprintf(stderr, "Available modes: cusparse-csr, stencil5-csr\n");
            return EXIT_FAILURE;
        }
    }

    // Load the matrix from Matrix Market file into a generic structure (AFTER mode validation)
    printf("\nLoading matrix: %s\n", matrix_file);
    MatrixData mat;  ///< Container for matrix data loaded from file
    if (load_matrix_market(matrix_file, &mat) != 0) {
        fprintf(stderr, "Failed to load matrix %s\n", matrix_file);
        return EXIT_FAILURE;
    }

    printf("Matrix loaded: %d rows, %d cols, %lld nonzeros\n", mat.rows, mat.cols, mat.nnz);
    printf("Testing %d mode(s): ", num_modes);
    for (int i = 0; i < num_modes; i++) {
        printf("%s%s", mode_tokens[i], (i < num_modes - 1) ? ", " : "\n");
    }
    if (num_modes > 1) {
        printf("NOTE: Multi-mode benchmark - performance may vary with order due to GPU state.\n");
        printf("      For accurate comparison, run each mode separately.\n");
    }

    // Allocate and initialize input/output vectors on the host (shared across modes)
    double* x = (double*)malloc(
        mat.cols * sizeof(double));  ///< Input vector for SpMV operation (x in y = A*x)
    double* y = (double*)malloc(
        mat.rows * sizeof(double));  ///< Output vector for SpMV operation (y in y = A*x)
    if (!x || !y) {
        fprintf(stderr, "Error allocating vectors\n");
        return EXIT_FAILURE;
    }

    // Initialize vectors with appropriate values
    for (int i = 0; i < mat.cols; i++)
        x[i] = 1.0;  // Fill input vector with 1.0

    // Loop through all requested modes
    for (int mode_idx = 0; mode_idx < num_modes; mode_idx++) {
        const char* current_mode = mode_tokens[mode_idx];

        printf("\n=== Testing mode: %s ===\n", current_mode);

        // Select the corresponding SpMV operator (already validated)
        SpmvOperator* op = get_operator(current_mode);

        // Initialize the SpMV operator (ELLPACK reused if already built)
        if (op->init(&mat) != 0) {
            fprintf(stderr, "Failed to initialize operator '%s'\n", op->name);
            continue;
        }

        // Reset output vector for this mode
        memset(y, 0, mat.rows * sizeof(double));

        // Warmup runs to stabilize GPU clock frequency
        printf("Warmup (5 runs)...\n");
        double dummy_time;
        for (int w = 0; w < 5; w++) {
            op->run_timed(x, y, &dummy_time);
        }

        // Statistical benchmark with outlier detection
        printf("Running statistical benchmark (10 iterations)...\n");
        BenchmarkStats bench_stats;
        if (benchmark_with_stats(op->run_timed, x, y, 10, &bench_stats) != 0) {
            fprintf(stderr, "Statistical benchmark failed for mode '%s'\n", op->name);
            op->free();
            continue;
        }

        printf("Completed: %d valid runs, %d outliers removed\n", bench_stats.valid_runs,
               bench_stats.outliers_removed);

        // Compute checksum for validation (before export)
        double sum = 0.0;
        double norm2 = 0.0;
        for (int i = 0; i < mat.rows; i++) {
            sum += y[i];
            norm2 += y[i] * y[i];
        }
        norm2 = sqrt(norm2);

        // Calculate performance metrics using median time
        BenchmarkMetrics metrics;
        calculate_spmv_metrics(bench_stats.median_ms, &mat, op->name, &metrics);
        metrics.sum_y = sum;
        metrics.norm2_y = norm2;

        // Add GPU specifications to metrics
        if (get_gpu_properties(&metrics) != 0) {
            fprintf(stderr, "Warning: Could not retrieve GPU properties\n");
        }

        // Always print human-readable metrics to stdout
        print_benchmark_metrics(&metrics, stdout);

        // Export to JSON if requested (one file per mode)
        if (json_file != NULL) {
            char mode_json_file[512];
            // Find extension position
            const char* ext = strrchr(json_file, '.');
            if (ext != NULL) {
                size_t base_len = ext - json_file;
                snprintf(mode_json_file, sizeof(mode_json_file), "%.*s_%s%s", (int)base_len,
                         json_file, op->name, ext);
            } else {
                snprintf(mode_json_file, sizeof(mode_json_file), "%s_%s.json", json_file, op->name);
            }
            FILE* fp = fopen(mode_json_file, "w");
            if (fp == NULL) {
                fprintf(stderr, "Error: Could not open JSON file '%s' for writing\n",
                        mode_json_file);
            } else {
                print_metrics_json(&metrics, fp);
                fclose(fp);
                printf("Metrics exported to JSON: %s\n", mode_json_file);
            }
        }

        // Export to CSV if requested (one file per mode)
        if (csv_file != NULL) {
            char mode_csv_file[512];
            const char* ext = strrchr(csv_file, '.');
            if (ext != NULL) {
                size_t base_len = ext - csv_file;
                snprintf(mode_csv_file, sizeof(mode_csv_file), "%.*s_%s%s", (int)base_len, csv_file,
                         op->name, ext);
            } else {
                snprintf(mode_csv_file, sizeof(mode_csv_file), "%s_%s.csv", csv_file, op->name);
            }
            FILE* fp = fopen(mode_csv_file, "w");
            if (fp == NULL) {
                fprintf(stderr, "Error: Could not open CSV file '%s' for writing\n", mode_csv_file);
            } else {
                print_metrics_csv(&metrics, fp);
                fclose(fp);
                printf("Metrics exported to CSV: %s\n", mode_csv_file);
            }
        }

        printf("SpMV completed successfully using mode: %s\n", op->name);

        printf("\n=== Output Checksum ===\n");
        printf("Sum(y):    %.16e\n", sum);
        printf("Norm2(y):  %.16e\n", norm2);
        printf("=======================\n\n");

        // Free GPU memory after each mode to prevent accumulation on large matrices
        op->free();
    }

    if (num_modes > 1) {
        printf("\n=== Multi-mode benchmark completed ===\n");
    }

    // Free host memory for vectors and matrix data
    free(x);
    free(y);

    // Free matrix entries allocated in load_matrix_market
    if (mat.entries) {
        free(mat.entries);
    }

    return EXIT_SUCCESS;
}
