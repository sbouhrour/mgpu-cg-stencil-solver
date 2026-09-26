/**
 * @file spmv_ellpack.h
 * @brief Definitions and utilities for ELLPACK matrix format.
 *
 * @details
 * This header provides:
 *  - The ELLPACKMatrix structure for storing sparse matrices in ELLPACK format.
 *  - Functions to build an ELLPACK matrix from CSR data.
 *
 * Author: Bouhrour Stephane
 * Date: 2025-07-15
 */

#ifndef SPMV_ELLPACK_H
#define SPMV_ELLPACK_H

/** @brief Maximum width for ELLPACK row representation. */
#define MAX_WIDTH 1000

/**
 * @struct ELLPACKMatrix
 * @brief Represents a sparse matrix in ELLPACK format.
 *
 * @param ell_width Maximum number of non-zero elements per row.
 * @param indices Column indices for each element (row-major).
 * @param values Non-zero values stored in aligned fashion per row.
 */
struct ELLPACKMatrix {
    int nb_rows;
    int nb_cols;
    int ell_width;
    int grid_size;  // n for an n×n grid
    int* indices;
    int nb_nonzeros;
    double* values;
};

/** @name ELLPACK Matrix Functions
 *  Functions to create and manage ELLPACK matrices.
 *  @{ */

/**
 * @brief Build an ELLPACKMatrix from an existing CSRMatrix.
 *
 * @param csr_matrix Pointer to the input CSR matrix.
 * @param ellpack_matrix Pointer to the output ELLPACK matrix.
 * @param max_width Pointer to store the computed maximum row width.
 * @return 0 on success, non-zero on failure.
 */
int build_ellpack_from_csr_struct(const struct CSRMatrix* csr_matrix, ELLPACKMatrix* ellpack_matrix,
                                  int* max_width);

/** @} */

#endif  // SPMV_ELLPACK_H
