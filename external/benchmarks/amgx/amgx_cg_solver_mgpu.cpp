/**
 * @file amgx_cg_solver_mgpu.cpp
 * @brief Multi-GPU CG solver using NVIDIA AmgX with MPI
 *
 * AmgX handles all internal communications automatically when using MPI.
 * Simplified approach: use upload_all() with local data, AmgX detects MPI context.
 *
 * Launch: mpirun -np N ./amgx_cg_solver_mgpu matrix.mtx [options]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <cuda_runtime.h>
#include <amgx_c.h>
#include <nvtx3/nvToolsExt.h>
#include <chrono>
#include <vector>
#include <algorithm>
#include "amgx_benchmark.h"

static int g_rank = 0;  // Global rank for callbacks

#define CUDA_CHECK(call)                                                                       \
    do {                                                                                       \
        cudaError_t err = call;                                                                \
        if (err != cudaSuccess) {                                                              \
            fprintf(stderr, "[Rank %d] CUDA error at %s:%d: %s\n", g_rank, __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                                  \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                      \
        }                                                                                      \
    } while (0)

#define AMGX_CHECK(call)                                                                       \
    do {                                                                                       \
        AMGX_RC err = call;                                                                    \
        if (err != AMGX_RC_OK) {                                                               \
            char str[4096];                                                                    \
            AMGX_get_error_string(err, str, 4096);                                             \
            fprintf(stderr, "[Rank %d] AMGX error at %s:%d: %s\n", g_rank, __FILE__, __LINE__, \
                    str);                                                                      \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                      \
        }                                                                                      \
    } while (0)

// Print callback - only rank 0 prints
void print_callback(const char* msg, int length) {
    if (g_rank == 0) {
        printf("%s", msg);
    }
}

struct MatrixMarket {
    int rows, cols, nnz;
    int* row_ptr;
    int* col_idx;
    double* values;
};

MatrixMarket read_matrix_market(const char* filename, int rank) {
    FILE* f = fopen(filename, "r");
    if (!f) {
        fprintf(stderr, "[Rank %d] Failed to open %s\n", rank, filename);
        MPI_Abort(MPI_COMM_WORLD, 1);
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
            fprintf(stderr, "[Rank %d] Error reading matrix line %d\n", rank, i);
            MPI_Abort(MPI_COMM_WORLD, 1);
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

struct DetailedTimers {
    double vector_upload_ms;
    double solve_ms;
    double vector_download_ms;
};

RunResult run_amgx_solve_mgpu(AMGX_solver_handle solver, AMGX_vector_handle b, AMGX_vector_handle x,
                              double* d_x, int n_local, bool enable_timers, int rank,
                              DetailedTimers* timers = nullptr) {
    // Create CUDA events for GPU-accurate timing
    cudaEvent_t start, stop, timer_start, timer_stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    if (enable_timers) {
        CUDA_CHECK(cudaEventCreate(&timer_start));
        CUDA_CHECK(cudaEventCreate(&timer_stop));
    }

    // Reset solution vector
    if (enable_timers)
        CUDA_CHECK(cudaEventRecord(timer_start));

    double* h_x_zero = (double*)calloc(n_local, sizeof(double));
    CUDA_CHECK(cudaMemcpy(d_x, h_x_zero, n_local * sizeof(double), cudaMemcpyHostToDevice));
    AMGX_CHECK(AMGX_vector_upload(x, n_local, 1, d_x));
    free(h_x_zero);

    if (enable_timers) {
        CUDA_CHECK(cudaEventRecord(timer_stop));
        CUDA_CHECK(cudaEventSynchronize(timer_stop));
        float elapsed_ms;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, timer_start, timer_stop));
        if (timers)
            timers->vector_upload_ms = elapsed_ms;
    }

    MPI_Barrier(MPI_COMM_WORLD);  // Sync all ranks before timing

    // Start overall timing
    CUDA_CHECK(cudaEventRecord(start));

    // AmgX solve (main computation)
    if (enable_timers)
        CUDA_CHECK(cudaEventRecord(timer_start));

    AMGX_CHECK(AMGX_solver_solve(solver, b, x));

    if (enable_timers) {
        CUDA_CHECK(cudaEventRecord(timer_stop));
        CUDA_CHECK(cudaEventSynchronize(timer_stop));
        float elapsed_ms;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, timer_start, timer_stop));
        if (timers)
            timers->solve_ms = elapsed_ms;
    }

    // Download solution from AmgX to d_x
    if (enable_timers)
        CUDA_CHECK(cudaEventRecord(timer_start));

    AMGX_CHECK(AMGX_vector_download(x, d_x));

    if (enable_timers) {
        CUDA_CHECK(cudaEventRecord(timer_stop));
        CUDA_CHECK(cudaEventSynchronize(timer_stop));
        float elapsed_ms;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, timer_start, timer_stop));
        if (timers)
            timers->vector_download_ms = elapsed_ms;
    }

    // Stop overall timing
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    RunResult result;
    float time_ms;
    CUDA_CHECK(cudaEventElapsedTime(&time_ms, start, stop));
    result.time_ms = time_ms;

    AMGX_SOLVE_STATUS status;
    AMGX_CHECK(AMGX_solver_get_status(solver, &status));
    result.converged = (status == AMGX_SOLVE_SUCCESS);

    AMGX_CHECK(AMGX_solver_get_iterations_number(solver, &result.iterations));

    // Cleanup events
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    if (enable_timers) {
        CUDA_CHECK(cudaEventDestroy(timer_start));
        CUDA_CHECK(cudaEventDestroy(timer_stop));
    }

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
    // Initialize MPI
    MPI_Init(&argc, &argv);

    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    g_rank = rank;  // Set global rank for callbacks

    if (argc < 2) {
        if (rank == 0) {
            fprintf(stderr, "Usage: mpirun -np <N> %s <matrix.mtx> [options]\n", argv[0]);
            fprintf(stderr, "Options:\n");
            fprintf(stderr, "  --tol=1e-6         Convergence tolerance (default: 1e-6)\n");
            fprintf(stderr, "  --max-iters=1000   Maximum CG iterations (default: 1000)\n");
            fprintf(stderr, "  --runs=10          Number of benchmark runs (default: 10)\n");
            fprintf(stderr, "  --timers           Enable detailed timing breakdown\n");
            fprintf(stderr, "  --json=<file>      Export results to JSON\n");
            fprintf(stderr, "  --csv=<file>       Export results to CSV\n");
            fprintf(stderr,
                    "\nExample: mpirun -np 4 %s matrix/stencil_5000x5000.mtx --runs=10 --timers\n",
                    argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    const char* matrix_file = argv[1];
    double tolerance = 1e-6;
    int max_iters = 5000;
    int num_runs = 10;
    bool enable_timers = false;
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
        } else if (strcmp(argv[i], "--timers") == 0) {
            enable_timers = true;
        } else if (strncmp(argv[i], "--json=", 7) == 0) {
            json_file = argv[i] + 7;
        } else if (strncmp(argv[i], "--csv=", 6) == 0) {
            csv_file = argv[i] + 6;
        }
    }

    if (rank == 0) {
        printf("========================================\n");
        printf("AmgX Multi-GPU CG Solver\n");
        printf("========================================\n");
        printf("Matrix: %s\n", matrix_file);
        printf("MPI ranks: %d\n", world_size);
        printf("Tolerance: %.0e\n", tolerance);
        printf("Max iterations: %d\n", max_iters);
        printf("Benchmark runs: %d\n\n", num_runs);
    }

    // Set GPU device (one GPU per MPI rank)
    int num_devices;
    CUDA_CHECK(cudaGetDeviceCount(&num_devices));
    int device_id = rank % num_devices;
    CUDA_CHECK(cudaSetDevice(device_id));

    // Verify device assignment for ALL ranks
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
    printf("[Rank %d] GPU assignment: device %d (%s), PCI %04x:%02x:%02x.0\n", rank, device_id,
           prop.name, prop.pciDomainID, prop.pciBusID, prop.pciDeviceID);

    MPI_Barrier(MPI_COMM_WORLD);  // Sync prints
    if (rank == 0)
        printf("\n");

    // Load full matrix (each rank independently)
    MatrixMarket mat = read_matrix_market(matrix_file, rank);
    int grid_size = (int)sqrt(mat.rows);

    if (rank == 0) {
        printf("Matrix loaded: %dx%d, %d nonzeros\n", mat.rows, mat.cols, mat.nnz);
        printf("Grid: %dx%d\n\n", grid_size, grid_size);
    }

    // Build consistent global partition vector FIRST (all ranks must agree)
    int* partition_vector_tmp = (int*)malloc((world_size + 1) * sizeof(int));
    partition_vector_tmp[0] = 0;
    for (int i = 0; i < world_size; i++) {
        int rows_for_rank = mat.rows / world_size;
        if (i == world_size - 1) {
            rows_for_rank = mat.rows - partition_vector_tmp[i];
        }
        partition_vector_tmp[i + 1] = partition_vector_tmp[i] + rows_for_rank;
    }

    // Use partition vector to define local range
    int row_offset = partition_vector_tmp[rank];
    int n_local = partition_vector_tmp[rank + 1] - partition_vector_tmp[rank];

    free(partition_vector_tmp);

    // Create local partition (rows [row_offset : row_offset + n_local))
    int* local_row_ptr = (int*)malloc((n_local + 1) * sizeof(int));
    local_row_ptr[0] = 0;

    // Count nnz in local partition
    int local_nnz = 0;
    for (int i = 0; i < n_local; i++) {
        int global_row = row_offset + i;
        int row_nnz = mat.row_ptr[global_row + 1] - mat.row_ptr[global_row];
        local_nnz += row_nnz;
        local_row_ptr[i + 1] = local_nnz;
    }

    // Extract local CSR partition with GLOBAL column indices (int64_t as per NVIDIA example)
    int64_t* local_col_idx = (int64_t*)malloc(local_nnz * sizeof(int64_t));
    double* local_values = (double*)malloc(local_nnz * sizeof(double));

    for (int i = 0; i < n_local; i++) {
        int global_row = row_offset + i;
        int src_start = mat.row_ptr[global_row];
        int row_nnz = mat.row_ptr[global_row + 1] - src_start;
        int dst_start = local_row_ptr[i];

        // Copy with GLOBAL column indices (int64_t for AmgX upload_all_global)
        for (int j = 0; j < row_nnz; j++) {
            local_col_idx[dst_start + j] = (int64_t)mat.col_idx[src_start + j];
        }
        memcpy(&local_values[dst_start], &mat.values[src_start], row_nnz * sizeof(double));
    }

    printf("[Rank %d] Partition: rows [%d:%d), %d nnz\n", rank, row_offset, row_offset + n_local,
           local_nnz);

    // Validate local CSR data before passing to AmgX
    bool validation_error = false;

    // Check row_ptr is monotonic and ends at local_nnz
    if (local_row_ptr[0] != 0) {
        fprintf(stderr, "[Rank %d] ERROR: local_row_ptr[0]=%d (should be 0)\n", rank,
                local_row_ptr[0]);
        validation_error = true;
    }
    if (local_row_ptr[n_local] != local_nnz) {
        fprintf(stderr, "[Rank %d] ERROR: local_row_ptr[%d]=%d but local_nnz=%d\n", rank, n_local,
                local_row_ptr[n_local], local_nnz);
        validation_error = true;
    }
    for (int i = 0; i < n_local; i++) {
        if (local_row_ptr[i] > local_row_ptr[i + 1]) {
            fprintf(stderr, "[Rank %d] ERROR: row_ptr not monotonic at i=%d: %d > %d\n", rank, i,
                    local_row_ptr[i], local_row_ptr[i + 1]);
            validation_error = true;
            break;
        }
    }

    // Check column indices are in valid range [0, mat.rows)
    int64_t min_col = mat.rows, max_col = -1;
    for (int i = 0; i < local_nnz; i++) {
        int64_t col = local_col_idx[i];
        if (col < 0 || col >= mat.rows) {
            fprintf(stderr, "[Rank %d] ERROR: col_idx[%d]=%ld out of range [0,%d)\n", rank, i,
                    (long)col, mat.rows);
            validation_error = true;
            break;
        }
        if (col < min_col)
            min_col = col;
        if (col > max_col)
            max_col = col;
    }

    printf("[Rank %d] Validation: row_ptr OK, col_idx range [%ld,%ld]\n", rank, (long)min_col,
           (long)max_col);
    fflush(stdout);

    if (validation_error) {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (rank == 0)
        printf("\n");

    // Initialize AmgX
    AMGX_SAFE_CALL(AMGX_initialize());
    AMGX_SAFE_CALL(AMGX_initialize_plugins());
    AMGX_SAFE_CALL(AMGX_register_print_callback(&print_callback));

    // Create config for CG solver (unpreconditioned)
    char config_string[512];
    snprintf(config_string, sizeof(config_string),
             "config_version=2, "
             "solver=CG, "
             "max_iters=%d, "
             "convergence=RELATIVE_INI, "
             "tolerance=%.15e, "
             "norm=L2, "
             "print_solve_stats=0, "
             "monitor_residual=1, "
             "obtain_timings=0",
             max_iters, tolerance);

    AMGX_config_handle cfg;
    AMGX_SAFE_CALL(AMGX_config_create(&cfg, config_string));

    // Mode: double precision, device (GPU), int indices, int pointers
    AMGX_Mode mode = AMGX_mode_dDDI;

    // Create resources with explicit MPI communicator (distributed API)
    // NOTE: OpenMPI defines MPI_Comm as pointer, so &MPI_COMM_WORLD gives MPI_Comm*
    // Cast to void* to match AmgX API signature
    AMGX_resources_handle rsrc;
    MPI_Comm mpi_comm = MPI_COMM_WORLD;
    int device_ids[1] = {device_id};
    if (rank == 0) {
        printf("Using distributed API with explicit MPI communicator\n");
        printf("sizeof(MPI_Comm) = %zu, mpi_comm value = %p\n", sizeof(MPI_Comm), (void*)mpi_comm);
    }
    // Pass &mpi_comm directly as void* (AmgX will cast internally to MPI_Comm*)
    AMGX_SAFE_CALL(AMGX_resources_create(&rsrc, cfg, (void*)&mpi_comm, 1, device_ids));

    // Create matrix, vectors, and solver BEFORE distribution (as per NVIDIA example order)
    AMGX_matrix_handle A;
    AMGX_vector_handle b, x;
    AMGX_solver_handle solver;

    AMGX_SAFE_CALL(AMGX_matrix_create(&A, rsrc, mode));
    AMGX_SAFE_CALL(AMGX_vector_create(&b, rsrc, mode));
    AMGX_SAFE_CALL(AMGX_vector_create(&x, rsrc, mode));
    AMGX_SAFE_CALL(AMGX_solver_create(&solver, rsrc, mode, cfg));

    // Get number of rings for halo communication (as per NVIDIA example)
    int nrings;
    AMGX_SAFE_CALL(AMGX_config_get_default_number_of_rings(cfg, &nrings));
    if (rank == 0) {
        printf("Using upload_all_global with nrings=%d\n", nrings);
        printf("Uploading local CSR (n_local=%d, nnz_local=%d, global col indices int64_t)\n\n",
               n_local, local_nnz);
    }

    // Upload matrix using upload_all_global (NVIDIA recommended approach)
    // This API automatically handles halo detection and communication setup
    AMGX_SAFE_CALL(AMGX_matrix_upload_all_global(A,
                                                 mat.rows,        // global number of rows
                                                 n_local,         // local number of rows
                                                 local_nnz,       // local number of nonzeros
                                                 1, 1,            // block dimensions
                                                 local_row_ptr,   // local row pointers
                                                 local_col_idx,   // global column indices (int64_t)
                                                 local_values,    // local values
                                                 NULL,            // no diagonal (CSR format)
                                                 nrings, nrings,  // halo rings
                                                 NULL));

    // Bind vectors to matrix (AmgX analyzes structure and determines halo sizes)
    AMGX_SAFE_CALL(AMGX_vector_bind(b, A));
    AMGX_SAFE_CALL(AMGX_vector_bind(x, A));

    // Create RHS: b = ones
    if (rank == 0) {
        printf("RHS: b = ones, Initial guess: x0 = 0\n\n");
    }

    double* h_b = (double*)malloc(n_local * sizeof(double));
    for (int i = 0; i < n_local; i++)
        h_b[i] = 1.0;

    double* h_x = (double*)calloc(n_local, sizeof(double));  // Initial guess x0 = 0

    double *d_b, *d_x;
    CUDA_CHECK(cudaMalloc(&d_b, n_local * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_x, n_local * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, n_local * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, n_local * sizeof(double), cudaMemcpyHostToDevice));

    // Upload vectors (AmgX will allocate space for halos internally based on binding)
    AMGX_SAFE_CALL(AMGX_vector_upload(b, n_local, 1, d_b));
    AMGX_SAFE_CALL(AMGX_vector_upload(x, n_local, 1, d_x));

    // Setup solver (builds internal structures + halos)
    AMGX_SAFE_CALL(AMGX_solver_setup(solver, A));

    // Warmup
    if (rank == 0)
        printf("Warmup (3 runs)...\n");
    for (int i = 0; i < 3; i++) {
        run_amgx_solve_mgpu(solver, b, x, d_x, n_local, false, rank, nullptr);
    }

    // Benchmark
    if (rank == 0)
        printf("Running benchmark (%d runs)...\n", num_runs);
    std::vector<RunResult> results;
    DetailedTimers cumulative_timers = {0.0, 0.0, 0.0};
    for (int i = 0; i < num_runs; i++) {
        if (i == 5)
            nvtxRangePush("AmgX_Solve");  // Mark run 5 for profiling
        DetailedTimers run_timers;
        results.push_back(
            run_amgx_solve_mgpu(solver, b, x, d_x, n_local, enable_timers, rank, &run_timers));
        if (enable_timers) {
            cumulative_timers.vector_upload_ms += run_timers.vector_upload_ms;
            cumulative_timers.solve_ms += run_timers.solve_ms;
            cumulative_timers.vector_download_ms += run_timers.vector_download_ms;
        }
        if (i == 5)
            nvtxRangePop();
    }

    // Verify solution with checksum (download x and compute sum + L2 norm)
    double* h_x_local = (double*)malloc(n_local * sizeof(double));
    CUDA_CHECK(cudaMemcpy(h_x_local, d_x, n_local * sizeof(double), cudaMemcpyDeviceToHost));

    double local_sum = 0.0;
    double local_norm2 = 0.0;
    for (int i = 0; i < n_local; i++) {
        local_sum += h_x_local[i];
        local_norm2 += h_x_local[i] * h_x_local[i];
    }

    double global_sum, global_norm2;
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_norm2, &global_norm2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    double global_norm = sqrt(global_norm2);

    free(h_x_local);

    // Extract times from all ranks
    std::vector<double> times;
    for (const auto& r : results) {
        times.push_back(r.time_ms);
    }

    // Gather timing data from all ranks (all ranks must participate)
    std::vector<double> all_upload_times, all_solve_times, all_download_times, all_total_times;
    if (enable_timers) {
        double avg_upload = cumulative_timers.vector_upload_ms / num_runs;
        double avg_solve = cumulative_timers.solve_ms / num_runs;
        double avg_download = cumulative_timers.vector_download_ms / num_runs;
        double median = calculate_median(times);

        if (rank == 0) {
            all_upload_times.resize(world_size);
            all_solve_times.resize(world_size);
            all_download_times.resize(world_size);
            all_total_times.resize(world_size);
        }

        MPI_Gather(&avg_upload, 1, MPI_DOUBLE, all_upload_times.data(), 1, MPI_DOUBLE, 0,
                   MPI_COMM_WORLD);
        MPI_Gather(&avg_solve, 1, MPI_DOUBLE, all_solve_times.data(), 1, MPI_DOUBLE, 0,
                   MPI_COMM_WORLD);
        MPI_Gather(&avg_download, 1, MPI_DOUBLE, all_download_times.data(), 1, MPI_DOUBLE, 0,
                   MPI_COMM_WORLD);
        MPI_Gather(&median, 1, MPI_DOUBLE, all_total_times.data(), 1, MPI_DOUBLE, 0,
                   MPI_COMM_WORLD);
    }

    // Calculate statistics (rank 0)
    if (rank == 0) {
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
        printf("Sum(x):    %.16e\n", global_sum);
        printf("Norm2(x):  %.16e\n", global_norm);
        printf("========================================\n");

        // Print detailed timers if enabled (data already gathered above)
        if (enable_timers) {
            printf("\n========================================\n");
            printf("Detailed Timing Breakdown (Per Rank)\n");
            printf("========================================\n");
            printf("Rank | Upload (ms) | Solve (ms) | Download (ms) | Total (ms) | Load Imbal\n");
            printf("-----|-------------|------------|---------------|------------|-----------\n");

            double min_total = *std::min_element(all_total_times.begin(), all_total_times.end());
            double max_total = *std::max_element(all_total_times.begin(), all_total_times.end());
            double load_imbalance = (max_total - min_total) / max_total * 100.0;

            for (int r = 0; r < world_size; r++) {
                double rank_imbalance =
                    (all_total_times[r] - min_total) / all_total_times[r] * 100.0;
                printf("%4d | %11.3f | %10.3f | %13.3f | %10.3f | %9.1f%%\n", r,
                       all_upload_times[r], all_solve_times[r], all_download_times[r],
                       all_total_times[r], rank_imbalance);
            }

            printf("-----|-------------|------------|---------------|------------|-----------\n");
            printf("Overall load imbalance: %.1f%%\n", load_imbalance);
            printf("========================================\n");
        }

        // Export results
        if (json_file || csv_file) {
            char mode_str[64];
            snprintf(mode_str, sizeof(mode_str), "multi-gpu-%d", world_size);

            MatrixInfo mat_info = {mat.rows, mat.cols, mat.nnz, grid_size};
            BenchmarkResults bench_results = {results[0].converged,
                                              results[0].iterations,
                                              median,
                                              median,
                                              final_mean,
                                              min_time,
                                              max_time,
                                              final_std,
                                              (int)filtered_times.size(),
                                              outliers_removed};

            if (json_file) {
                export_amgx_json(json_file, mode_str, &mat_info, &bench_results);
            }
            if (csv_file) {
                export_amgx_csv(csv_file, mode_str, &mat_info, &bench_results, true);
            }
        }
    }

    // Cleanup
    free(h_b);
    free(h_x);
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
    free(local_row_ptr);
    free(local_col_idx);
    free(local_values);

    MPI_Finalize();
    return 0;
}
