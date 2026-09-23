/**
 * @file amgx_benchmark.h
 * @brief Common structures and utilities for AmgX benchmarking
 */

#ifndef AMGX_BENCHMARK_H
#define AMGX_BENCHMARK_H

#include <stdio.h>
#include <time.h>

struct BenchmarkResults {
    bool converged;
    int iterations;
    double time_total_ms;
    double median_ms;
    double mean_ms;
    double min_ms;
    double max_ms;
    double std_dev_ms;
    int valid_runs;
    int outliers_removed;
};

struct MatrixInfo {
    int rows;
    int cols;
    long long nnz;  // a 512^3 27-point operator has 3.6e9 entries
    int grid_size;
};

/**
 * @brief Export AmgX benchmark results to JSON
 */
inline void export_amgx_json(const char* filename, const char* mode, const MatrixInfo* mat_info,
                             const BenchmarkResults* results, int num_gpus = 1,
                             double max_rank_time = 0.0, double min_rank_time = 0.0) {
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
    fprintf(fp, "  \"solver\": \"AmgX CG\",\n");
    fprintf(fp, "  \"mode\": \"%s\",\n", mode);
    if (num_gpus > 1) {
        fprintf(fp, "  \"num_gpus\": %d,\n", num_gpus);
    }

    fprintf(fp, "  \"matrix\": {\n");
    fprintf(fp, "    \"rows\": %d,\n", mat_info->rows);
    fprintf(fp, "    \"cols\": %d,\n", mat_info->cols);
    fprintf(fp, "    \"nnz\": %lld,\n", mat_info->nnz);
    fprintf(fp, "    \"grid_size\": %d\n", mat_info->grid_size);
    fprintf(fp, "  },\n");

    fprintf(fp, "  \"convergence\": {\n");
    fprintf(fp, "    \"converged\": %s,\n", results->converged ? "true" : "false");
    fprintf(fp, "    \"iterations\": %d\n", results->iterations);
    fprintf(fp, "  },\n");

    fprintf(fp, "  \"timing\": {\n");
    fprintf(fp, "    \"median_ms\": %.3f,\n", results->median_ms);
    fprintf(fp, "    \"mean_ms\": %.3f,\n", results->mean_ms);
    fprintf(fp, "    \"min_ms\": %.3f,\n", results->min_ms);
    fprintf(fp, "    \"max_ms\": %.3f,\n", results->max_ms);
    fprintf(fp, "    \"std_dev_ms\": %.3f", results->std_dev_ms);
    if (num_gpus > 1 && max_rank_time > 0.0) {
        fprintf(fp, ",\n");
        fprintf(fp, "    \"max_rank_time_ms\": %.3f,\n", max_rank_time);
        fprintf(fp, "    \"min_rank_time_ms\": %.3f,\n", min_rank_time);
        double imbalance = 100.0 * (max_rank_time - min_rank_time) / max_rank_time;
        fprintf(fp, "    \"load_imbalance_pct\": %.1f", imbalance);
    }
    fprintf(fp, "\n  },\n");

    fprintf(fp, "  \"statistics\": {\n");
    fprintf(fp, "    \"valid_runs\": %d,\n", results->valid_runs);
    fprintf(fp, "    \"outliers_removed\": %d\n", results->outliers_removed);
    fprintf(fp, "  },\n");

    fprintf(fp, "  \"performance\": {\n");
    double gflops = (2.0 * mat_info->nnz * results->iterations) / (results->median_ms * 1e6);
    fprintf(fp, "    \"gflops\": %.3f\n", gflops);
    fprintf(fp, "  }\n");

    fprintf(fp, "}\n");

    fclose(fp);
    printf("Results exported to: %s\n", filename);
}

/**
 * @brief Export AmgX benchmark results to CSV
 */
inline void export_amgx_csv(const char* filename, const char* mode, const MatrixInfo* mat_info,
                            const BenchmarkResults* results, bool write_header, int num_gpus = 1,
                            double max_rank_time = 0.0, double min_rank_time = 0.0) {
    FILE* fp = fopen(filename, write_header ? "w" : "a");
    if (!fp) {
        fprintf(stderr, "Error: Could not open %s for writing\n", filename);
        return;
    }

    if (write_header) {
        fprintf(fp, "solver,mode,num_gpus,rows,cols,nnz,grid_size,converged,iterations,");
        fprintf(fp, "median_ms,mean_ms,min_ms,max_ms,std_dev_ms,");
        fprintf(fp, "max_rank_time_ms,min_rank_time_ms,load_imbalance_pct,");
        fprintf(fp, "valid_runs,outliers_removed,gflops\n");
    }

    double gflops = (2.0 * mat_info->nnz * results->iterations) / (results->median_ms * 1e6);
    double imbalance = (num_gpus > 1 && max_rank_time > 0.0)
                           ? 100.0 * (max_rank_time - min_rank_time) / max_rank_time
                           : 0.0;

    fprintf(fp, "AmgX,%s,%d,%d,%d,%lld,%d,%d,%d,", mode, num_gpus, mat_info->rows, mat_info->cols,
            mat_info->nnz, mat_info->grid_size, results->converged, results->iterations);
    fprintf(fp, "%.3f,%.3f,%.3f,%.3f,%.3f,", results->median_ms, results->mean_ms, results->min_ms,
            results->max_ms, results->std_dev_ms);
    fprintf(fp, "%.3f,%.3f,%.1f,", max_rank_time, min_rank_time, imbalance);
    fprintf(fp, "%d,%d,%.3f\n", results->valid_runs, results->outliers_removed, gflops);

    fclose(fp);
    if (write_header) {
        printf("Results exported to: %s\n", filename);
    }
}

#endif  // AMGX_BENCHMARK_H
