/**
 * @file amgx_cg_solver.cpp
 * @brief CG solver using NVIDIA AMGX with benchmarking and metrics export
 *
 * Provides fair comparison with NVIDIA's optimized production solver.
 * Uses AMGX PCG solver without preconditioning.
 * Includes statistical benchmarking and JSON/CSV export.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <amgx_c.h>
#include <nvtx3/nvToolsExt.h>
#include <chrono>
#include <vector>
#include <algorithm>
#include "amgx_benchmark.h"

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

#define AMGX_CHECK(call)                                                                \
    do {                                                                                \
        AMGX_RC err = call;                                                             \
        if (err != AMGX_RC_OK) {                                                        \
            fprintf(stderr, "AMGX error at %s:%d: code %d\n", __FILE__, __LINE__, err); \
            exit(EXIT_FAILURE);                                                         \
        }                                                                               \
    } while (0)

struct MatrixMarket {
    int rows, cols, nnz;
    int* row_ptr;
    int* col_idx;
    double* values;
};

MatrixMarket read_matrix_market(const char* filename) {
    FILE* f = fopen(filename, "r");
    if (!f) {
        fprintf(stderr, "Failed to open %s\n", filename);
        exit(EXIT_FAILURE);
    }

    // Skip comments
    char line[1024];
    while (fgets(line, sizeof(line), f)) {
        if (line[0] != '%')
            break;
    }

    MatrixMarket mat;
    sscanf(line, "%d %d %d", &mat.rows, &mat.cols, &mat.nnz);

    // Temporary storage for COO format
    int* coo_rows = (int*)malloc(mat.nnz * sizeof(int));
    int* coo_cols = (int*)malloc(mat.nnz * sizeof(int));
    double* coo_vals = (double*)malloc(mat.nnz * sizeof(double));

    for (int i = 0; i < mat.nnz; i++) {
        int ret = fscanf(f, "%d %d %lf", &coo_rows[i], &coo_cols[i], &coo_vals[i]);
        if (ret != 3) {
            fprintf(stderr, "Error reading matrix line %d\n", i);
            exit(EXIT_FAILURE);
        }
        coo_rows[i]--;  // 1-based to 0-based
        coo_cols[i]--;
    }
    fclose(f);

    // Convert COO to CSR
    mat.row_ptr = (int*)calloc(mat.rows + 1, sizeof(int));
    mat.col_idx = (int*)malloc(mat.nnz * sizeof(int));
    mat.values = (double*)malloc(mat.nnz * sizeof(double));

    // Count non-zeros per row
    for (int i = 0; i < mat.nnz; i++) {
        mat.row_ptr[coo_rows[i] + 1]++;
    }

    // Prefix sum
    for (int i = 1; i <= mat.rows; i++) {
        mat.row_ptr[i] += mat.row_ptr[i - 1];
    }

    // Fill CSR arrays
    int* local_count = (int*)calloc(mat.rows, sizeof(int));
    for (int i = 0; i < mat.nnz; i++) {
        int r = coo_rows[i];
        int dst = mat.row_ptr[r] + local_count[r]++;
        mat.col_idx[dst] = coo_cols[i];
        mat.values[dst] = coo_vals[i];
    }

    free(coo_rows);
    free(coo_cols);
    free(coo_vals);
    free(local_count);

    return mat;
}

struct RunResult {
    double time_ms;
    int iterations;
    bool converged;
};

RunResult run_amgx_solve(AMGX_solver_handle solver, AMGX_vector_handle b, AMGX_vector_handle x,
                         double* d_x, int rows, bool verbose) {
    // Reset solution vector
    double* h_x_zero = (double*)calloc(rows, sizeof(double));
    CUDA_CHECK(cudaMemcpy(d_x, h_x_zero, rows * sizeof(double), cudaMemcpyHostToDevice));
    AMGX_CHECK(AMGX_vector_upload(x, rows, 1, d_x));
    free(h_x_zero);

    if (verbose) {
        printf("Starting AMGX CG solver...\n");
        printf("========================================\n");
    }

    auto start = std::chrono::high_resolution_clock::now();
    AMGX_CHECK(AMGX_solver_solve(solver, b, x));
    auto end = std::chrono::high_resolution_clock::now();

    // Download solution from AmgX to d_x
    AMGX_CHECK(AMGX_vector_download(x, d_x));

    if (verbose) {
        printf("========================================\n");
    }

    RunResult result;
    result.time_ms = std::chrono::duration<double, std::milli>(end - start).count();

    AMGX_SOLVE_STATUS status;
    AMGX_CHECK(AMGX_solver_get_status(solver, &status));
    result.converged = (status == AMGX_SOLVE_SUCCESS);

    AMGX_CHECK(AMGX_solver_get_iterations_number(solver, &result.iterations));

    return result;
}

double calculate_median(std::vector<double>& times) {
    std::sort(times.begin(), times.end());
    size_t n = times.size();
    if (n % 2 == 0) {
        return (times[n / 2 - 1] + times[n / 2]) / 2.0;
    }
    return times[n / 2];
}

double calculate_mean(const std::vector<double>& times) {
    double sum = 0.0;
    for (double t : times)
        sum += t;
    return sum / times.size();
}

double calculate_std_dev(const std::vector<double>& times, double mean) {
    double sum_sq = 0.0;
    for (double t : times) {
        double diff = t - mean;
        sum_sq += diff * diff;
    }
    return sqrt(sum_sq / times.size());
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        fprintf(stderr,
                "Usage: %s <matrix.mtx> [--tol=1e-6] [--max-iters=1000] [--runs=10] "
                "[--json=<file>] [--csv=<file>]\n",
                argv[0]);
        fprintf(stderr,
                "Example: %s matrix/stencil_5000x5000.mtx --runs=10 --json=results/amgx.json\n",
                argv[0]);
        return 1;
    }

    const char* matrix_file = argv[1];
    double tolerance = 1e-6;
    int max_iters = 5000;
    int num_runs = 10;
    const char* json_file = nullptr;
    const char* csv_file = nullptr;

    // Parse arguments
    for (int i = 2; i < argc; i++) {
        if (strncmp(argv[i], "--tol=", 6) == 0) {
            tolerance = atof(argv[i] + 6);
        } else if (strncmp(argv[i], "--max-iters=", 12) == 0) {
            max_iters = atoi(argv[i] + 12);
        } else if (strncmp(argv[i], "--runs=", 7) == 0) {
            num_runs = atoi(argv[i] + 7);
        } else if (strncmp(argv[i], "--json=", 7) == 0) {
            json_file = argv[i] + 7;
        } else if (strncmp(argv[i], "--csv=", 6) == 0) {
            csv_file = argv[i] + 6;
        }
    }

    printf("========================================\n");
    printf("AMGX CG Solver (NVIDIA Reference)\n");
    printf("========================================\n");
    printf("Matrix: %s\n", matrix_file);
    printf("Tolerance: %.0e\n", tolerance);
    printf("Max iterations: %d\n", max_iters);
    printf("Benchmark runs: %d\n\n", num_runs);

    // Load matrix
    MatrixMarket mat = read_matrix_market(matrix_file);
    int grid_size = (int)sqrt(mat.rows);
    printf("Matrix loaded: %dx%d, %d nonzeros\n", mat.rows, mat.cols, mat.nnz);
    printf("Grid: %dx%d\n\n", grid_size, grid_size);

    // Initialize AMGX
    AMGX_CHECK(AMGX_initialize());
    AMGX_CHECK(AMGX_initialize_plugins());

    // Create config for PCG solver (no preconditioning)
    char config_string[512];
    snprintf(config_string, sizeof(config_string),
             "config_version=2, "
             "solver=PCG, "
             "preconditioner=NOSOLVER, "
             "max_iters=%d, "
             "convergence=RELATIVE_INI, "
             "tolerance=%.15e, "
             "norm=L2, "
             "print_solve_stats=0, "
             "monitor_residual=1, "  // Required for convergence check
             "obtain_timings=0",
             max_iters, tolerance);

    AMGX_config_handle cfg;
    AMGX_CHECK(AMGX_config_create(&cfg, config_string));

    // Create resources
    AMGX_resources_handle rsrc;
    AMGX_CHECK(AMGX_resources_create_simple(&rsrc, cfg));

    // Create matrix, vectors, and solver
    AMGX_matrix_handle A;
    AMGX_vector_handle b, x;
    AMGX_solver_handle solver;

    AMGX_CHECK(AMGX_matrix_create(&A, rsrc, AMGX_mode_dDDI));
    AMGX_CHECK(AMGX_vector_create(&b, rsrc, AMGX_mode_dDDI));
    AMGX_CHECK(AMGX_vector_create(&x, rsrc, AMGX_mode_dDDI));
    AMGX_CHECK(AMGX_solver_create(&solver, rsrc, AMGX_mode_dDDI, cfg));

    // Upload matrix
    AMGX_CHECK(AMGX_matrix_upload_all(A, mat.rows, mat.nnz, 1, 1, mat.row_ptr, mat.col_idx,
                                      mat.values, nullptr));

    // Create RHS: b = ones
    printf("RHS: b = ones, Initial guess: x0 = 0\n\n");
    double* h_b = (double*)malloc(mat.rows * sizeof(double));
    for (int i = 0; i < mat.rows; i++)
        h_b[i] = 1.0;

    double *d_b, *d_x;
    CUDA_CHECK(cudaMalloc(&d_b, mat.rows * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_x, mat.rows * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, mat.rows * sizeof(double), cudaMemcpyHostToDevice));

    AMGX_CHECK(AMGX_vector_upload(b, mat.rows, 1, d_b));

    // Setup solver
    AMGX_CHECK(AMGX_solver_setup(solver, A));

    // Warmup
    printf("Warmup (3 runs)...\n");
    for (int i = 0; i < 3; i++) {
        run_amgx_solve(solver, b, x, d_x, mat.rows, false);
    }

    // Benchmark
    printf("Running benchmark (%d runs)...\n", num_runs);
    std::vector<RunResult> results;
    for (int i = 0; i < num_runs; i++) {
        if (i == 5)
            nvtxRangePush("AmgX_Solve");  // Mark run 5 for profiling
        results.push_back(run_amgx_solve(solver, b, x, d_x, mat.rows, false));
        if (i == 5)
            nvtxRangePop();
    }

    // Verify solution with checksum (download x and compute sum + L2 norm)
    double* h_x = (double*)malloc(mat.rows * sizeof(double));
    CUDA_CHECK(cudaMemcpy(h_x, d_x, mat.rows * sizeof(double), cudaMemcpyDeviceToHost));

    double sum = 0.0;
    double norm2 = 0.0;
    for (int i = 0; i < mat.rows; i++) {
        sum += h_x[i];
        norm2 += h_x[i] * h_x[i];
    }
    double norm = sqrt(norm2);

    free(h_x);

    // Extract times
    std::vector<double> times;
    for (const auto& r : results) {
        times.push_back(r.time_ms);
    }

    // Calculate statistics
    double mean = calculate_mean(times);
    double std_dev = calculate_std_dev(times, mean);

    // Remove outliers (>2 std devs)
    std::vector<double> filtered_times;
    for (double t : times) {
        if (fabs(t - mean) <= 2.0 * std_dev) {
            filtered_times.push_back(t);
        }
    }

    double median = calculate_median(filtered_times);
    double final_mean = calculate_mean(filtered_times);
    double final_std = calculate_std_dev(filtered_times, final_mean);

    std::sort(filtered_times.begin(), filtered_times.end());
    double min_time = filtered_times.front();
    double max_time = filtered_times.back();

    int outliers_removed = times.size() - filtered_times.size();

    printf("Completed: %zu valid runs, %d outliers removed\n\n", filtered_times.size(),
           outliers_removed);

    // Print results
    printf("========================================\n");
    printf("Results\n");
    printf("========================================\n");
    printf("Converged: %s in %d iterations\n", results[0].converged ? "YES" : "NO",
           results[0].iterations);
    printf("Time (median): %.3f ms\n", median);
    printf("Stats: min=%.3f ms, max=%.3f ms, std=%.3f ms\n", min_time, max_time, final_std);

    double gflops = (2.0 * mat.nnz * results[0].iterations) / (median * 1e6);
    printf("GFLOPS: %.3f\n", gflops);

    printf("\n=== Output Checksum ===\n");
    printf("Sum(x):    %.16e\n", sum);
    printf("Norm2(x):  %.16e\n", norm);
    printf("========================================\n");

    // Export results
    if (json_file || csv_file) {
        MatrixInfo mat_info = {mat.rows, mat.cols, mat.nnz, grid_size};
        BenchmarkResults bench_results = {results[0].converged,
                                          results[0].iterations,
                                          median,  // time_total_ms (reuse for consistency)
                                          median,
                                          final_mean,
                                          min_time,
                                          max_time,
                                          final_std,
                                          (int)filtered_times.size(),
                                          outliers_removed};

        if (json_file) {
            export_amgx_json(json_file, "single-gpu", &mat_info, &bench_results);
        }
        if (csv_file) {
            export_amgx_csv(csv_file, "single-gpu", &mat_info, &bench_results, true);
        }
    }

    // Cleanup
    free(h_b);
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_x));

    AMGX_solver_destroy(solver);
    AMGX_vector_destroy(x);
    AMGX_vector_destroy(b);
    AMGX_matrix_destroy(A);
    AMGX_resources_destroy(rsrc);
    AMGX_config_destroy(cfg);

    AMGX_finalize_plugins();
    AMGX_finalize();

    free(mat.row_ptr);
    free(mat.col_idx);
    free(mat.values);

    return 0;
}
