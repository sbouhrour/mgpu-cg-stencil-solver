/**
 * @file io.h
 * @brief Matrix I/O utilities for loading and converting sparse matrices.
 *
 * @details
 * Responsibilities:
 *  - Define the `Entry` and `MatrixData` structures for Matrix Market representation.
 *  - Provide functions to read, parse, and convert matrices from Matrix Market format.
 *  - Support conversion between CSR and ELLPACK formats.
 *
 * Key Components:
 *  - `Entry`: Represents a single non-zero entry in a sparse matrix.
 *  - `MatrixData`: Stores matrix dimensions and entries before format conversion.
 *  - Functions for reading general or symmetric matrices and writing stencil-based matrices.
 *
 * Author: Bouhrour Stephane
 * Date: 2025-07-15
 */

#ifndef SPMV_IO_H
#define SPMV_IO_H

#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================
   CONSTANTS & LIMITS
   ============================ */
#define MAX_LINE_LENGTH 1024  ///< Maximum buffer size for reading a line

/* ============================
   DATA STRUCTURES
   ============================ */

/**
 * @struct Entry
 * @brief Represents a single non-zero entry in the sparse matrix.
 */
typedef struct {
    int row;       ///< Row index (0-based after adjustment)
    int col;       ///< Column index (0-based after adjustment)
    double value;  ///< Value of the non-zero element
} Entry;

/**
 * @struct MatrixData
 * @brief Stores the raw matrix read from a Matrix Market file.
 */
typedef struct MatrixData {
    int rows;  ///< Number of rows in the matrix
    int cols;  ///< Number of columns in the matrix
    long long
        nnz;  ///< Number of non-zero elements (long long to handle >2B for large 27pt stencils)
    int grid_size;   ///< Original grid size n for n×n stencil (-1 if not stencil)
    Entry* entries;  ///< Dynamic array of non-zero entries
} MatrixData;

/* Forward declarations for structures from other headers */
struct CSRMatrix;
struct ELLPACKMatrix;

/* ============================
   FUNCTION DECLARATIONS
   ============================ */

/**
 * @brief Detect the type of matrix from a Matrix Market file.
 *
 * @param filename Path to the .mtx file
 * @return int Matrix type code (e.g., 1 = general, 2 = symmetric)
 */
int read_matrix_type(const char* filename);

/**
 * @brief Reads a general (non-symmetric) matrix from a Matrix Market file.
 *
 * @param mat Output pointer to the MatrixData structure
 * @param filename Path to the .mtx file
 * @param rows Pointer to store number of rows
 * @param cols Pointer to store number of columns
 * @param nnz Pointer to store number of non-zero elements
 * @param csr_rowptr Output CSR row pointer
 * @param csr_colind Output CSR column indices
 * @param csr_val Output CSR values
 */
void read_matrix_general(MatrixData* mat, const char* filename, int* rows, int* cols, int* nnz,
                         int** csr_rowptr, int** csr_colind, double** csr_val);

/**
 * @brief Reads a symmetric matrix and expands it into a general format.
 *
 * @param mat Output pointer to the MatrixData structure
 * @param filename Path to the .mtx file
 * @param rows Pointer to store number of rows
 * @param cols Pointer to store number of columns
 * @param nnz Pointer to store number of non-zero elements
 * @param csr_rowptr Output CSR row pointer
 * @param csr_colind Output CSR column indices
 * @param csr_val Output CSR values
 * @param nnz_general Pointer to store expanded nnz count
 */
void read_matrix_symtogen(MatrixData* mat, const char* filename, int* rows, int* cols, int* nnz,
                          int** csr_rowptr, int** csr_colind, double** csr_val, int* nnz_general);

/**
 * @brief Load a Matrix Market file into a MatrixData structure.
 *
 * @param filename Path to the .mtx file
 * @param mat Pointer to MatrixData to fill
 * @return int 0 if successful, non-zero otherwise
 */
int load_matrix_market(const char* filename, MatrixData* mat);
int load_matrix_stencil27_3d_from_grid(const char* matrix_path, MatrixData* mat, int rank,
                                       int world_size);

/**
 * @brief Convert a CSR matrix to ELLPACK format.
 *
 * @param csr_matrix Pointer to the CSR matrix
 * @param ellpack_matrix Pointer to the ELLPACK matrix
 * @param max_width Pointer to store the maximum row width
 */
void convert_csr_to_ellpack(const struct CSRMatrix* csr_matrix,
                            struct ELLPACKMatrix* ellpack_matrix, int* max_width);

/**
 * @brief Generate and write a 5-point stencil matrix in Matrix Market format.
 *
 * @param n Grid dimension (matrix size will be n*n)
 * @param filename Output file name
 * @return int 0 if successful, non-zero otherwise
 */
int write_matrix_market_stencil5(int n, const char* filename);

/**
 * @brief Generate a 3D 7-point stencil matrix in Matrix Market format
 *
 * Creates an NxNxN 3D Laplacian on an N³×N³ matrix. Each point is connected to
 * itself (6.0) and its 6 neighbors in ±x, ±y, ±z directions (-1.0 each).
 * Boundary conditions: Dirichlet (skip neighbors outside domain).
 *
 * @param n Grid dimension (matrix size will be n*n*n)
 * @param filename Output file name
 * @return int 0 if successful, non-zero otherwise
 */
int write_matrix_market_stencil7(int n, const char* filename);

/**
 * @brief Generate a 3D 27-point stencil matrix in Matrix Market format
 *
 * Creates an NxNxN 3D Laplacian on an N³×N³ matrix. Each point is connected to
 * itself (26.0) and its 26 neighbors (face, edge, corner adjacent) (-1.0 each).
 * Boundary conditions: Dirichlet (skip neighbors outside domain).
 *
 * @param n Grid dimension (matrix size will be n*n*n)
 * @param filename Output file name
 * @return int 0 if successful, non-zero otherwise
 */
int write_matrix_market_stencil27(int n, const char* filename);

#ifdef __cplusplus
}
#endif

#endif  // SPMV_IO_H
