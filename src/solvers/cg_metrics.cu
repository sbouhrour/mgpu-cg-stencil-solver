/**
 * @file cg_metrics.cu
 * @brief CG solver metrics export (JSON/CSV)
 *
 * Author: Bouhrour Stephane
 * Date: 2025-11-24
 */

#include <stdio.h>
#include <string.h>
#include <time.h>
#include "solvers/cg_solver.h"
#include "solvers/cg_solver_mgpu.h"
#include "benchmark_stats.h"

extern "C" {

/**
 * @brief Export CG benchmark results to JSON format
 */
void export_cg_json(const char* filename, const char* mode, const MatrixData* mat,
                    const BenchmarkStats* bench_stats, const CGStats* cg_stats) {
    FILE* fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "Error: Could not open %s for writing\n", filename);
        return;
    }

    // Get timestamp
    time_t now = time(NULL);
    char timestamp[64];
    strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", localtime(&now));

    fprintf(fp, "{\n");
    fprintf(fp, "  \"timestamp\": \"%s\",\n", timestamp);
    fprintf(fp, "  \"solver\": \"CG\",\n");
    fprintf(fp, "  \"mode\": \"%s\",\n", mode);

    fprintf(fp, "  \"matrix\": {\n");
    fprintf(fp, "    \"rows\": %d,\n", mat->rows);
    fprintf(fp, "    \"cols\": %d,\n", mat->cols);
    fprintf(fp, "    \"nnz\": %lld,\n", mat->nnz);
    fprintf(fp, "    \"grid_size\": %d\n", mat->grid_size);
    fprintf(fp, "  },\n");

    fprintf(fp, "  \"convergence\": {\n");
    fprintf(fp, "    \"converged\": %s,\n", cg_stats->converged ? "true" : "false");
    fprintf(fp, "    \"iterations\": %d,\n", cg_stats->iterations);
    fprintf(fp, "    \"residual_norm\": %.15e\n", cg_stats->residual_norm);
    fprintf(fp, "  },\n");

    fprintf(fp, "  \"timing\": {\n");
    fprintf(fp, "    \"median_ms\": %.3f,\n", bench_stats->median_ms);
    fprintf(fp, "    \"mean_ms\": %.3f,\n", bench_stats->mean_ms);
    fprintf(fp, "    \"min_ms\": %.3f,\n", bench_stats->min_ms);
    fprintf(fp, "    \"max_ms\": %.3f,\n", bench_stats->max_ms);
    fprintf(fp, "    \"std_dev_ms\": %.3f,\n", bench_stats->std_dev_ms);
    fprintf(fp, "    \"spmv_ms\": %.3f,\n", cg_stats->time_spmv_ms);
    fprintf(fp, "    \"blas1_ms\": %.3f,\n", cg_stats->time_blas1_ms);
    fprintf(fp, "    \"reductions_ms\": %.3f\n", cg_stats->time_reductions_ms);
    fprintf(fp, "  },\n");

    fprintf(fp, "  \"statistics\": {\n");
    fprintf(fp, "    \"valid_runs\": %d,\n", bench_stats->valid_runs);
    fprintf(fp, "    \"outliers_removed\": %d\n", bench_stats->outliers_removed);
    fprintf(fp, "  },\n");

    fprintf(fp, "  \"performance\": {\n");
    double gflops = (2.0 * mat->nnz * cg_stats->iterations) / (cg_stats->time_spmv_ms * 1e6);
    fprintf(fp, "    \"gflops_spmv\": %.3f\n", gflops);
    fprintf(fp, "  },\n");

    fprintf(fp, "  \"validation\": {\n");
    fprintf(fp, "    \"solution_sum\": %.16e,\n", cg_stats->solution_sum);
    fprintf(fp, "    \"solution_norm\": %.16e\n", cg_stats->solution_norm);
    fprintf(fp, "  }\n");

    fprintf(fp, "}\n");

    fclose(fp);
    printf("Results exported to: %s\n", filename);
}

/**
 * @brief Export CG multi-GPU benchmark results to JSON
 */
void export_cg_mgpu_json(const char* filename, const char* mode, const MatrixData* mat,
                         const BenchmarkStats* bench_stats, const CGStatsMultiGPU* cg_stats,
                         int num_gpus) {
    FILE* fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "Error: Could not open %s for writing\n", filename);
        return;
    }

    time_t now = time(NULL);
    char timestamp[64];
    strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", localtime(&now));

    fprintf(fp, "{\n");
    fprintf(fp, "  \"timestamp\": \"%s\",\n", timestamp);
    fprintf(fp, "  \"solver\": \"CG Multi-GPU\",\n");
    fprintf(fp, "  \"mode\": \"%s\",\n", mode);
    fprintf(fp, "  \"num_gpus\": %d,\n", num_gpus);

    fprintf(fp, "  \"matrix\": {\n");
    fprintf(fp, "    \"rows\": %d,\n", mat->rows);
    fprintf(fp, "    \"cols\": %d,\n", mat->cols);
    fprintf(fp, "    \"nnz\": %lld,\n", mat->nnz);
    fprintf(fp, "    \"grid_size\": %d\n", mat->grid_size);
    fprintf(fp, "  },\n");

    fprintf(fp, "  \"convergence\": {\n");
    fprintf(fp, "    \"converged\": %s,\n", cg_stats->converged ? "true" : "false");
    fprintf(fp, "    \"iterations\": %d,\n", cg_stats->iterations);
    fprintf(fp, "    \"residual_norm\": %.15e\n", cg_stats->residual_norm);
    fprintf(fp, "  },\n");

    fprintf(fp, "  \"timing\": {\n");
    fprintf(fp, "    \"median_ms\": %.3f,\n", bench_stats->median_ms);
    fprintf(fp, "    \"mean_ms\": %.3f,\n", bench_stats->mean_ms);
    fprintf(fp, "    \"min_ms\": %.3f,\n", bench_stats->min_ms);
    fprintf(fp, "    \"max_ms\": %.3f,\n", bench_stats->max_ms);
    fprintf(fp, "    \"std_dev_ms\": %.3f\n", bench_stats->std_dev_ms);
    fprintf(fp, "  },\n");

    fprintf(fp, "  \"statistics\": {\n");
    fprintf(fp, "    \"valid_runs\": %d,\n", bench_stats->valid_runs);
    fprintf(fp, "    \"outliers_removed\": %d\n", bench_stats->outliers_removed);
    fprintf(fp, "  },\n");

    fprintf(fp, "  \"performance\": {\n");
    double gflops =
        (2.0 * mat->nnz * (double)cg_stats->iterations) / (bench_stats->median_ms * 1e6);
    fprintf(fp, "    \"gflops_spmv_est\": %.3f\n", gflops);
    fprintf(fp, "  },\n");

    int is_overlap = (strstr(mode, "overlap") != NULL);
    fprintf(fp, "  \"overlap\": {\n");
    if (is_overlap) {
        fprintf(fp, "    \"enabled\": true,\n");
        fprintf(fp, "    \"comm_total_ms\": %.3f,\n", cg_stats->time_comm_total_ms);
        fprintf(fp, "    \"comm_hidden_ms\": %.3f,\n", cg_stats->time_comm_hidden_ms);
        fprintf(fp, "    \"comm_exposed_ms\": %.3f,\n", cg_stats->time_comm_exposed_ms);
        fprintf(fp, "    \"overlap_efficiency_pct\": %.1f\n", cg_stats->overlap_efficiency * 100.0);
    } else {
        fprintf(fp, "    \"enabled\": false\n");
    }
    fprintf(fp, "  },\n");

    fprintf(fp, "  \"validation\": {\n");
    fprintf(fp, "    \"solution_sum\": %.16e,\n", cg_stats->solution_sum);
    fprintf(fp, "    \"solution_norm\": %.16e\n", cg_stats->solution_norm);
    fprintf(fp, "  }\n");

    fprintf(fp, "}\n");

    fclose(fp);
    printf("Results exported to: %s\n", filename);
}

/**
 * @brief Export CG benchmark results to CSV format
 */
void export_cg_csv(const char* filename, const char* mode, const MatrixData* mat,
                   const BenchmarkStats* bench_stats, const CGStats* cg_stats, bool write_header) {
    FILE* fp = fopen(filename, write_header ? "w" : "a");
    if (!fp) {
        fprintf(stderr, "Error: Could not open %s for writing\n", filename);
        return;
    }

    if (write_header) {
        fprintf(fp, "mode,rows,cols,nnz,grid_size,converged,iterations,residual_norm,");
        fprintf(fp, "median_ms,mean_ms,min_ms,max_ms,std_dev_ms,");
        fprintf(fp, "spmv_ms,blas1_ms,reductions_ms,");
        fprintf(fp, "valid_runs,outliers_removed,gflops_spmv,solution_sum,solution_norm\n");
    }

    double gflops = (2.0 * mat->nnz * cg_stats->iterations) / (cg_stats->time_spmv_ms * 1e6);

    fprintf(fp, "%s,%d,%d,%lld,%d,%d,%d,%.15e,", mode, mat->rows, mat->cols, mat->nnz,
            mat->grid_size, cg_stats->converged, cg_stats->iterations, cg_stats->residual_norm);
    fprintf(fp, "%.3f,%.3f,%.3f,%.3f,%.3f,", bench_stats->median_ms, bench_stats->mean_ms,
            bench_stats->min_ms, bench_stats->max_ms, bench_stats->std_dev_ms);
    fprintf(fp, "%.3f,%.3f,%.3f,", cg_stats->time_spmv_ms, cg_stats->time_blas1_ms,
            cg_stats->time_reductions_ms);
    fprintf(fp, "%d,%d,%.3f,%.16e,%.16e\n", bench_stats->valid_runs, bench_stats->outliers_removed,
            gflops, cg_stats->solution_sum, cg_stats->solution_norm);

    fclose(fp);
    if (write_header) {
        printf("Results exported to: %s\n", filename);
    }
}

}  // extern "C"
