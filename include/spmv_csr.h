/**
 * @file spmv_csr.h
 * @brief Definitions and utilities for CSR (Compressed Sparse Row) matrix format.
 *
 * @details
 * This header provides:
 *  - The CSRMatrix structure for storing sparse matrices in CSR format.
 *  - Utility functions to build and manage CSR matrices from generic matrix data.
 *
 * Author: Bouhrour Stephane
 * Date: 2025-07-15
 */

#ifndef SPMV_CSR_H
#define SPMV_CSR_H

/**
 * @struct CSRMatrix
 * @brief Represents a sparse matrix in CSR format.
 *
 * @param nb_rows Number of rows in the matrix.
 * @param nb_cols Number of columns in the matrix.
 * @param nb_nonzeros Number of non-zero elements.
 * @param row_ptr Array of size (nb_rows + 1) storing the cumulative non-zero count per row.
 * @param col_indices Array of column indices corresponding to non-zero values.
 * @param values Array of non-zero values.
 */
struct CSRMatrix {
    int nb_rows;
    int nb_cols;
    long long nb_nonzeros;
    int* row_ptr;
    int* col_indices;
    double* values;
};

/** @name CSR Matrix Functions
 *  Functions to create and manipulate CSR matrices.
 *  @{ */

/**
 * @brief Build a CSRMatrix structure from generic MatrixData.
 *
 * @param mat Pointer to the input MatrixData (contains raw entries).
 * @return 0 on success, non-zero on failure.
 */
int build_csr_struct(struct MatrixData* mat);

/** @} */

#endif  // SPMV_CSR_H
