/**
 * @file stencil_partition.h
 * @brief This rank's rows of a 3D stencil operator, as the local CSR block AmgX uploads
 *
 * Built from the same in-memory generators as the custom solver (src/io/io.cu), so both
 * solvers see the same operator and the same Z-slab row partition without reading a file:
 * a 512^3 27-point operator has 3.6e9 entries.
 */
#ifndef STENCIL_PARTITION_H
#define STENCIL_PARTITION_H

#include <stdint.h>
#include <stdlib.h>
#include "io.h"

struct LocalCsr {
    int global_rows;
    int row_offset;
    int n_local;
    int nnz;           // local; AmgX dDDI indexes rows with int
    int* row_ptr;      // n_local + 1, local
    int64_t* col_idx;  // global column indices
    double* values;
    long long global_nnz;  // filled by the caller (reduction over ranks)
};

/**
 * @return 0 on success. stencil is 7 or 27; rows are sorted by column index.
 */
static int build_stencil_local_csr(const char* stub, int stencil, int rank, int world_size,
                                   LocalCsr* out) {
    MatrixData m;
    int err = (stencil == 27) ? load_matrix_stencil27_3d_from_grid(stub, &m, rank, world_size)
                              : load_matrix_stencil7_3d_from_grid(stub, &m, rank, world_size);
    if (err)
        return err;
    if (m.nnz > 2147483647LL) {
        fprintf(stderr, "[Rank %d] %lld local entries exceed AmgX int indexing: use more ranks\n",
                rank, m.nnz);
        return 1;
    }
    const long long rows = m.rows;
    out->global_rows = m.rows;
    out->n_local = (int)(rows / world_size);
    out->row_offset = rank * out->n_local;
    if (rank == world_size - 1)
        out->n_local = (int)(rows - out->row_offset);
    out->nnz = (int)m.nnz;

    out->row_ptr = (int*)calloc((size_t)out->n_local + 1, sizeof(int));
    out->col_idx = (int64_t*)malloc((size_t)m.nnz * sizeof(int64_t));
    out->values = (double*)malloc((size_t)m.nnz * sizeof(double));
    for (long long e = 0; e < m.nnz; e++)
        out->row_ptr[m.entries[e].row - out->row_offset + 1]++;
    for (int i = 0; i < out->n_local; i++)
        out->row_ptr[i + 1] += out->row_ptr[i];
    int* fill = (int*)calloc((size_t)out->n_local, sizeof(int));
    for (long long e = 0; e < m.nnz; e++) {
        const int r = m.entries[e].row - out->row_offset;
        const int dst = out->row_ptr[r] + fill[r]++;
        out->col_idx[dst] = m.entries[e].col;
        out->values[dst] = m.entries[e].value;
    }
    free(fill);
    free(m.entries);

    // Insertion sort by column inside each row (7 or 27 entries)
    for (int r = 0; r < out->n_local; r++) {
        for (int a = out->row_ptr[r] + 1; a < out->row_ptr[r + 1]; a++) {
            const int64_t c = out->col_idx[a];
            const double v = out->values[a];
            int b = a - 1;
            while (b >= out->row_ptr[r] && out->col_idx[b] > c) {
                out->col_idx[b + 1] = out->col_idx[b];
                out->values[b + 1] = out->values[b];
                b--;
            }
            out->col_idx[b + 1] = c;
            out->values[b + 1] = v;
        }
    }
    return 0;
}

#endif  // STENCIL_PARTITION_H
