/**
 * @file io.cu
 * @brief Handles I/O operations for Matrix Market files and memory allocation.
 *
 * @details
 * I/O functionality for sparse matrix operations:
 * - Reading sparse matrices from Matrix Market (.mtx) format
 * - Converting matrices to CSR (Compressed Sparse Row) representation
 * - Handling symmetry expansion for symmetric matrices
 * - Writing generated matrices to Matrix Market format
 * - Managing memory allocation and deallocation for matrix data structures
 *
 * The implementation supports both general and symmetric matrix formats,
 * automatically detecting the matrix type and applying appropriate conversion
 * strategies for optimal SpMV performance.
 *
 * Author: Bouhrour Stephane
 * Date: 2025-07-15
 */

#include "io.h"
#include "spmv.h"

/**
 * @brief Determines the matrix type from Matrix Market file header.
 * @details Parses the Matrix Market file header to determine if the matrix
 * is stored in general or symmetric format. This information is crucial
 * for choosing the appropriate reading strategy.
 * @param filename Path to the Matrix Market file
 * @return 1 if general matrix, 2 if symmetric matrix, -1 on error
 */
int read_matrix_type(const char* filename) {
    int ret = -1;
    FILE* file;

    // Determine if the matrix is general or symmetrical
    file = fopen(filename, "r");
    if (file == NULL) {
        fprintf(stderr, "Error opening file\n");
        return ret;
    }

    char line[MAX_LINE_LENGTH];
    while (fgets(line, sizeof(line), file) != NULL) {
        // Check if the line is commented
        if (!(line[0] == '%')) {
            fprintf(stderr, "Error opening file\n");
            // Return if symmetrical or general have not been found while parsing header
            return ret;
        }

        if (strstr(line, "general") != NULL) {
            return 1;
        }
        if (strstr(line, "symmetric") != NULL) {
            return 2;
        }
    }

    fclose(file);
    return ret;
}

/**
 * @brief Generates a 3D 27-point stencil MatrixData for this MPI rank's partition only.
 * @details Reads only the file header to get grid_size N, then generates COO entries
 * for the z-slab partition owned by this rank (matching the solver's partitioning:
 * n_local = N³/world_size, row_offset = rank * n_local). Each rank only allocates
 * its share of memory (~total/world_size), preventing OOM for large grids.
 * mat->rows is set to the global N³ so the solver's row_ptr indexing works correctly.
 * @param matrix_path Path to the Matrix Market file (only header is read)
 * @param mat Pointer to MatrixData structure to fill
 * @param rank MPI rank of this process
 * @param world_size Total number of MPI ranks
 * @return 0 on success, non-zero on error
 */
int load_matrix_stencil27_3d_from_grid(const char* matrix_path, MatrixData* mat, int rank,
                                       int world_size) {
    // Read only the header to extract grid_size N
    FILE* f = fopen(matrix_path, "r");
    if (!f) {
        fprintf(stderr, "Error opening file: %s\n", matrix_path);
        return 1;
    }

    int N = -1;
    char buffer[MAX_LINE_LENGTH];
    while (fgets(buffer, MAX_LINE_LENGTH, f) != NULL) {
        if (buffer[0] == '%') {
            if (strstr(buffer, "STENCIL_GRID_SIZE") != NULL)
                sscanf(buffer, "%% STENCIL_GRID_SIZE %d", &N);
        } else {
            break;  // reached dimension line, stop
        }
    }
    fclose(f);

    if (N <= 0) {
        fprintf(stderr, "Could not find STENCIL_GRID_SIZE in header of %s\n", matrix_path);
        return 1;
    }

    long long matrix_size = (long long)N * N * N;

    // Compute this rank's row partition — mirrors cg_solver_mgpu_partitioned_3d logic:
    //   n_local = N³ / world_size,  row_offset = rank * n_local
    long long n_local_rows = matrix_size / world_size;
    long long row_start = (long long)rank * n_local_rows;
    long long row_end = (rank == world_size - 1) ? matrix_size : row_start + n_local_rows;

    // Convert row boundaries to z-plane (i) boundaries (rows = i*N²+j*N+k)
    int i_start = (int)(row_start / ((long long)N * N));
    int i_end = (int)(row_end / ((long long)N * N));

    if (rank == 0) {
        printf("Generating 27pt stencil in memory for N=%d, partition [rank %d/%d, i=%d..%d]...\n",
               N, rank, world_size, i_start, i_end - 1);
        fflush(stdout);
    }

    // Count exact nnz for the local partition only
    long long nnz = 0;
    for (int i = i_start; i < i_end; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                nnz++;  // center
                if (i > 0)
                    nnz++;
                if (i < N - 1)
                    nnz++;
                if (j > 0)
                    nnz++;
                if (j < N - 1)
                    nnz++;
                if (k > 0)
                    nnz++;
                if (k < N - 1)
                    nnz++;
                if (i > 0 && j > 0)
                    nnz++;
                if (i > 0 && j < N - 1)
                    nnz++;
                if (i < N - 1 && j > 0)
                    nnz++;
                if (i < N - 1 && j < N - 1)
                    nnz++;
                if (i > 0 && k > 0)
                    nnz++;
                if (i > 0 && k < N - 1)
                    nnz++;
                if (i < N - 1 && k > 0)
                    nnz++;
                if (i < N - 1 && k < N - 1)
                    nnz++;
                if (j > 0 && k > 0)
                    nnz++;
                if (j > 0 && k < N - 1)
                    nnz++;
                if (j < N - 1 && k > 0)
                    nnz++;
                if (j < N - 1 && k < N - 1)
                    nnz++;
                if (i > 0 && j > 0 && k > 0)
                    nnz++;
                if (i > 0 && j > 0 && k < N - 1)
                    nnz++;
                if (i > 0 && j < N - 1 && k > 0)
                    nnz++;
                if (i > 0 && j < N - 1 && k < N - 1)
                    nnz++;
                if (i < N - 1 && j > 0 && k > 0)
                    nnz++;
                if (i < N - 1 && j > 0 && k < N - 1)
                    nnz++;
                if (i < N - 1 && j < N - 1 && k > 0)
                    nnz++;
                if (i < N - 1 && j < N - 1 && k < N - 1)
                    nnz++;
            }
        }
    }

    if (rank == 0) {
        printf("  local nnz = %lld, allocating %.1f GB for COO entries...\n", nnz,
               (double)nnz * sizeof(Entry) / 1e9);
        fflush(stdout);
    }

    Entry* entries = (Entry*)malloc(nnz * sizeof(Entry));
    if (!entries) {
        fprintf(stderr, "[Rank %d] malloc failed for %lld entries (%.1f GB)\n", rank, nnz,
                (double)nnz * sizeof(Entry) / 1e9);
        return 1;
    }

    // Fill entries for local partition only
    long long idx = 0;
    long long local_points = (long long)(i_end - i_start) * N * N;
    long long progress_step = local_points / 20;
    if (progress_step == 0)
        progress_step = 1;

    for (int i = i_start; i < i_end; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                long long global_idx = (long long)i * N * N + j * N + k;
                int row = (int)global_idx;

                long long local_idx = global_idx - row_start;
                if (rank == 0 && local_idx % progress_step == 0) {
                    printf("\r  Generating entries: %d%%", (int)(local_idx * 100 / local_points));
                    fflush(stdout);
                }

                // Center
                entries[idx++] = (Entry){row, row, 26.0};

// 6 face neighbors
#define ADD(ni)                  \
    entries[idx++] = (Entry) {   \
        row, (int)((ni)-1), -1.0 \
    }
                if (i > 0)
                    ADD((long long)(i - 1) * N * N + j * N + k + 1);
                if (i < N - 1)
                    ADD((long long)(i + 1) * N * N + j * N + k + 1);
                if (j > 0)
                    ADD((long long)i * N * N + (j - 1) * N + k + 1);
                if (j < N - 1)
                    ADD((long long)i * N * N + (j + 1) * N + k + 1);
                if (k > 0)
                    ADD((long long)i * N * N + j * N + (k - 1) + 1);
                if (k < N - 1)
                    ADD((long long)i * N * N + j * N + (k + 1) + 1);
                // 12 edge neighbors
                if (i > 0 && j > 0)
                    ADD((long long)(i - 1) * N * N + (j - 1) * N + k + 1);
                if (i > 0 && j < N - 1)
                    ADD((long long)(i - 1) * N * N + (j + 1) * N + k + 1);
                if (i < N - 1 && j > 0)
                    ADD((long long)(i + 1) * N * N + (j - 1) * N + k + 1);
                if (i < N - 1 && j < N - 1)
                    ADD((long long)(i + 1) * N * N + (j + 1) * N + k + 1);
                if (i > 0 && k > 0)
                    ADD((long long)(i - 1) * N * N + j * N + (k - 1) + 1);
                if (i > 0 && k < N - 1)
                    ADD((long long)(i - 1) * N * N + j * N + (k + 1) + 1);
                if (i < N - 1 && k > 0)
                    ADD((long long)(i + 1) * N * N + j * N + (k - 1) + 1);
                if (i < N - 1 && k < N - 1)
                    ADD((long long)(i + 1) * N * N + j * N + (k + 1) + 1);
                if (j > 0 && k > 0)
                    ADD((long long)i * N * N + (j - 1) * N + (k - 1) + 1);
                if (j > 0 && k < N - 1)
                    ADD((long long)i * N * N + (j - 1) * N + (k + 1) + 1);
                if (j < N - 1 && k > 0)
                    ADD((long long)i * N * N + (j + 1) * N + (k - 1) + 1);
                if (j < N - 1 && k < N - 1)
                    ADD((long long)i * N * N + (j + 1) * N + (k + 1) + 1);
                // 8 corner neighbors
                if (i > 0 && j > 0 && k > 0)
                    ADD((long long)(i - 1) * N * N + (j - 1) * N + (k - 1) + 1);
                if (i > 0 && j > 0 && k < N - 1)
                    ADD((long long)(i - 1) * N * N + (j - 1) * N + (k + 1) + 1);
                if (i > 0 && j < N - 1 && k > 0)
                    ADD((long long)(i - 1) * N * N + (j + 1) * N + (k - 1) + 1);
                if (i > 0 && j < N - 1 && k < N - 1)
                    ADD((long long)(i - 1) * N * N + (j + 1) * N + (k + 1) + 1);
                if (i < N - 1 && j > 0 && k > 0)
                    ADD((long long)(i + 1) * N * N + (j - 1) * N + (k - 1) + 1);
                if (i < N - 1 && j > 0 && k < N - 1)
                    ADD((long long)(i + 1) * N * N + (j - 1) * N + (k + 1) + 1);
                if (i < N - 1 && j < N - 1 && k > 0)
                    ADD((long long)(i + 1) * N * N + (j + 1) * N + (k - 1) + 1);
                if (i < N - 1 && j < N - 1 && k < N - 1)
                    ADD((long long)(i + 1) * N * N + (j + 1) * N + (k + 1) + 1);
#undef ADD
            }
        }
    }
    if (rank == 0)
        printf("\r  Generating entries: 100%%\n");

    // mat->rows is GLOBAL (N³) so the solver's row_ptr indexing works correctly.
    // mat->nnz is LOCAL (only this rank's entries).
    mat->rows = (int)matrix_size;
    mat->cols = (int)matrix_size;
    mat->nnz = nnz;
    mat->grid_size = N;
    mat->entries = entries;

    if (rank == 0)
        printf("  Done: global %d rows, local %lld nnz (rank %d/%d)\n", mat->rows, mat->nnz, rank,
               world_size);
    return 0;
}

/**
 * @brief Loads a matrix from Matrix Market format into MatrixData structure.
 * @details Main entry point for matrix loading. Automatically detects matrix type
 * (general or symmetric) and calls the appropriate reading function. For symmetric
 * matrices, performs expansion to full format for efficient SpMV operations.
 * @param matrix_path Path to the Matrix Market file
 * @param mat Pointer to MatrixData structure to fill
 * @return 0 on success, non-zero on error
 */
int load_matrix_market(const char* matrix_path, MatrixData* mat) {
    // Read the matrix, fill mat->rows, mat->cols, mat->nnz, mat->entries
    printf("Loading matrix: %s\n", matrix_path);
    int rows, cols, nnz;
    int *row_ptr, *col_indices;
    int nnz_general;
    double* values;

    int type = read_matrix_type(matrix_path);
    if (type == 2)  // If symmetric type
    {
        read_matrix_symtogen(mat, matrix_path, &rows, &cols, &nnz, &row_ptr, &col_indices, &values,
                             &nnz_general);
        nnz = nnz_general;
    } else  // If general type
    {
        read_matrix_general(mat, matrix_path, &rows, &cols, &nnz, &row_ptr, &col_indices, &values);
    }

    return 0;
}

/**
 * @brief Reads a general (non-symmetric) matrix from Matrix Market format.
 * @details Parses a Matrix Market file containing a general matrix and populates
 * the MatrixData structure with entries in coordinate format. Adjusts 1-based
 * Matrix Market indices to 0-based indexing used internally.
 * @param mat Pointer to MatrixData structure to populate
 * @param filename Path to the Matrix Market file
 * @param rows Pointer to store number of rows
 * @param cols Pointer to store number of columns
 * @param nnz Pointer to store number of non-zero elements
 * @param csr_rowptr Pointer to CSR row pointer array (unused in this function)
 * @param csr_colind Pointer to CSR column indices array (unused in this function)
 * @param csr_val Pointer to CSR values array (unused in this function)
 */
void read_matrix_general(MatrixData* mat, const char* filename, int* rows, int* cols, int* nnz,
                         int** csr_rowptr, int** csr_colind, double** csr_val) {
    FILE* file;
    int i;
    Entry* entries;

    file = fopen(filename, "r");
    if (file == NULL) {
        fprintf(stderr, "Error opening file\n");
        return;
    }

    char buffer[MAX_LINE_LENGTH];  // Buffer to store each line
    int grid_size = -1;            // Valeur par défaut si pas trouvé
    while (fgets(buffer, MAX_LINE_LENGTH, file) != NULL) {
        if (buffer[0] != '%') {
            // Process the line here (or skip it)
            sscanf(buffer, "%d %d %d", rows, cols, nnz);
            break;  // Stop when three digits are found
        } else {
            // Chercher le commentaire STENCIL_GRID_SIZE
            if (strstr(buffer, "STENCIL_GRID_SIZE") != NULL) {
                sscanf(buffer, "%% STENCIL_GRID_SIZE %d", &grid_size);
            }
        }
    }

    entries = (Entry*)malloc((*nnz) * sizeof(Entry));
    if (!entries) {
        fprintf(stderr, "Allocation failed at line %d\n", __LINE__);
        exit(1);
    }
    if (entries == NULL) {
        fprintf(stderr, "Memory allocation error\n");
        return;
    }

    mat->entries = entries;
    mat->rows = *rows;
    mat->cols = *cols;
    mat->nnz = *nnz;
    mat->grid_size = grid_size;  // Stocker le grid_size extrait

    // Read entries
    for (i = 0; i < *nnz; i++) {
        int items_read =
            fscanf(file, "%d %d %le", &entries[i].row, &entries[i].col, &entries[i].value);
        if (items_read != 3) {
            fprintf(stderr, "Error reading matrix entry %d (expected 3 items, got %d)\n", i,
                    items_read);
            free(entries);
            fclose(file);
            return;
        }
        // Adjust indices (if necessary, because indices start from 1 in Matrix Market format)
        entries[i].row--;  // Adjust for 0-based indexing
        entries[i].col--;  // Adjust for 0-based indexing
    }

    fclose(file);

    return;
}

/**
 * @brief Reads a symmetric matrix and expands it to general format with CSR conversion.
 * @details Processes a Matrix Market file containing a symmetric matrix (lower triangular
 * part only) and expands it to full general format. Simultaneously converts the data
 * to CSR format for efficient SpMV operations. Handles diagonal elements correctly
 * to avoid duplication.
 * @param mat Pointer to MatrixData structure to populate
 * @param filename Path to the Matrix Market file
 * @param rows Pointer to store number of rows
 * @param cols Pointer to store number of columns
 * @param nnz Pointer to store original number of non-zero elements
 * @param csr_rowptr Pointer to store CSR row pointer array
 * @param csr_colind Pointer to store CSR column indices array
 * @param csr_val Pointer to store CSR values array
 * @param nnz_general Pointer to store expanded number of non-zero elements
 */
void read_matrix_symtogen(MatrixData* mat, const char* filename, int* rows, int* cols, int* nnz,
                          int** csr_rowptr, int** csr_colind, double** csr_val, int* nnz_general) {
    FILE* file;
    int i;
    Entry* entries;
    int *row_ptr, *col_indices;
    ;
    double* values;

    // Open the file
    file = fopen(filename, "r");
    if (file == NULL) {
        fprintf(stderr, "Error opening file\n");
        return;
    }

    // Read the header (Matrix Market format)
    // Skip comment mtx format
    char buffer[MAX_LINE_LENGTH];  // Buffer to store each line
    while (fgets(buffer, MAX_LINE_LENGTH, file) != NULL) {
        if (buffer[0] != '%') {
            // Process the line here (or skip it)
            sscanf(buffer, "%d %d %d", rows, cols, nnz);
            break;  // Stop when three digits are found
        }
    }

    // fscanf(file, "%d %d %d", rows, cols, nnz); // Read matrix dimensions and number of non-zero
    // entries
    fprintf(stderr, "in %s\n", filename);
    fprintf(stderr, "rows %d cols %d nnz %d \n", *rows, *cols, *nnz);

    entries = (Entry*)malloc(*nnz * sizeof(Entry));
    if (entries == NULL) {
        fprintf(stderr, "Memory allocation error\n");
        return;
    }

    // Read entries
    int nb_value_on_diag = 0;
    for (i = 0; i < *nnz; i++) {
        int items_read =
            fscanf(file, "%d %d %le", &entries[i].row, &entries[i].col, &entries[i].value);
        if (items_read != 3) {
            fprintf(stderr, "Error reading symmetric matrix entry %d (expected 3 items, got %d)\n",
                    i, items_read);
            free(entries);
            fclose(file);
            return;
        }
        if (i < 1) {
            fprintf(stderr, "%d %d %le\n", entries[i].row, entries[i].col, entries[i].value);
        }
        // Store the lower part are on diag and immediately after the symmetric if not on diag
        if (entries[i].col == entries[i].row) {
            nb_value_on_diag++;
        }
        // Adjust indices (if necessary, because indices start from 1 in Matrix Market format)
        entries[i].row--;  // Adjust for 0-based indexing
        entries[i].col--;  // Adjust for 0-based indexing
    }

    // Close the file
    fclose(file);

    // Allocate memory for CSR format arrays
    *nnz_general = (2 * (*nnz) - nb_value_on_diag);  // (2*(*nnz) - nb_value_on_diag) for the full
                                                     // nnz not only diag and lower

    row_ptr = (int*)calloc(*rows + 1, sizeof(int));
    col_indices = (int*)malloc(*nnz_general * sizeof(int));
    values = (double*)malloc(*nnz_general * sizeof(double));

    // To simplify access to pointers
    *csr_rowptr = row_ptr;
    *csr_colind = col_indices;
    *csr_val = values;

    if (row_ptr == NULL || col_indices == NULL || values == NULL) {
        fprintf(stderr, "Memory allocation error\n");
        free(entries);
        return;
    }

    // Convert to CSR format
    for (i = 0; i < *nnz; i++) {
        row_ptr[entries[i].row + 1]++;  // [row +1] for accumulate row offset in tab per row on top
                                        // of accumulation prec row ind
        if (entries[i].col != entries[i].row) {
            row_ptr[entries[i].col +
                    1]++;  // To have a general matrix csr not only the lower and diag part
        }
    }

    for (i = 1; i < *rows + 1; i++) {
        row_ptr[i] += row_ptr[i - 1];
    }
    row_ptr[0] = 0;

    int* cpt_row_local_idx = (int*)calloc(*rows, sizeof(int));
    for (i = 0; i < *nnz; i++) {
        int row = entries[i].row;
        int index = row_ptr[row];
        col_indices[index + cpt_row_local_idx[row]] = entries[i].col;
        values[index + cpt_row_local_idx[row]] = entries[i].value;
        cpt_row_local_idx[row]++;
        // printf("value row %d found, value %le\n", entries[i].row, values[index +
        // cpt_row_local_idx[row]]);
        if (entries[i].row != entries[i].col)  // Then add the symmetric
        {
            int row = entries[i].col;
            int index = row_ptr[row];
            col_indices[index + cpt_row_local_idx[row]] = entries[i].row;
            values[index + cpt_row_local_idx[row]] = entries[i].value;
            // printf("value sym of row %d found, value %le\n", entries[i].row, values[index +
            // cpt_row_local_idx[row]]);
            cpt_row_local_idx[row]++;
        }
    }

    return;
}

/**
 * @brief Generates and writes a 5-point stencil matrix to Matrix Market format.
 * @details Creates a 5-point finite difference stencil matrix for a 2D grid of size n×n.
 * The resulting matrix has dimensions (n²)×(n²) and represents discretized Laplacian
 * operator. Each interior point connects to its 4 neighbors with value -1.0 and
 * has a center value of -4.0. Boundary conditions are handled naturally.
 * @param n Grid dimension (creates n×n grid, resulting in n²×n² matrix)
 * @param filename Output file path for Matrix Market format
 * @return 0 on success, non-zero on error
 */
int write_matrix_market_stencil5(int n, const char* filename) {

    int grid_size = n * n;

    // Calculate the exact number of non-zeros
    int nnz = 0;
    for (int row = 0; row < n; row++) {
        for (int col = 0; col < n; col++) {
            nnz++;  // Center
            if (col > 0)
                nnz++;  // Left
            if (col < n - 1)
                nnz++;  // Right
            if (row > 0)
                nnz++;  // Top
            if (row < n - 1)
                nnz++;  // Bottom
        }
    }

    FILE* f = fopen(filename, "w");
    if (!f) {
        perror("fopen");
        exit(1);
    }

    // Write Matrix Market header
    fprintf(f, "%%%%MatrixMarket matrix coordinate real general\n");
    fprintf(f, "%% STENCIL_GRID_SIZE %d\n", n);  // Commentaire avec n original
    fprintf(f, "%d %d %d\n", grid_size, grid_size, nnz);

    // Write values with progress indication
    long long total_points = (long long)n * n;
    long long progress_step = total_points / 100;  // Print every 1%
    if (progress_step == 0)
        progress_step = 1;  // Ensure progress for small matrices

    printf("Writing matrix entries: 0%%");
    fflush(stdout);

    for (int row = 0; row < n; row++) {
        for (int col = 0; col < n; col++) {
            int idx = row * n + col + 1;  // 1-based index
            long long current_point = (long long)row * n + col;

            // Progress indicator - update every 1% or every 1000 points (whichever is smaller)
            if (current_point % progress_step == 0 || current_point % 1000 == 0) {
                int percent = (int)((current_point * 100) / total_points);
                printf("\rWriting matrix entries: %d%%", percent);
                fflush(stdout);
            }

            // Center (SPD: Laplacian + mass term, well-conditioned for CG)
            fprintf(f, "%d %d 5.0\n", idx, idx);

            // Left
            if (col > 0)
                fprintf(f, "%d %d -1.0\n", idx, idx - 1);

            // Right
            if (col < n - 1)
                fprintf(f, "%d %d -1.0\n", idx, idx + 1);

            // Top
            if (row > 0)
                fprintf(f, "%d %d -1.0\n", idx, idx - n);

            // Bottom
            if (row < n - 1)
                fprintf(f, "%d %d -1.0\n", idx, idx + n);
        }
    }

    printf("\rWriting matrix entries: 100%%\n");
    fclose(f);
    printf("Matrix generated: %s (%dx%d, %d nnz)\n", filename, grid_size, grid_size, nnz);
    return 0;
}

/**
 * Generate 3D 7-point stencil matrix in Matrix Market format
 * Grid: NxNxN, row-major ordering: global_idx = i*N*N + j*N + k
 */
int write_matrix_market_stencil7(int N, const char* filename) {
    long long matrix_size = (long long)N * N * N;

    // Calculate the exact number of non-zeros
    long long nnz = 0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                nnz++;  // Center
                if (i > 0)
                    nnz++;  // -x neighbor
                if (i < N - 1)
                    nnz++;  // +x neighbor
                if (j > 0)
                    nnz++;  // -y neighbor
                if (j < N - 1)
                    nnz++;  // +y neighbor
                if (k > 0)
                    nnz++;  // -z neighbor
                if (k < N - 1)
                    nnz++;  // +z neighbor
            }
        }
    }

    FILE* f = fopen(filename, "w");
    if (!f) {
        perror("fopen");
        exit(1);
    }

    // Write Matrix Market header
    fprintf(f, "%%%%MatrixMarket matrix coordinate real general\n");
    fprintf(f, "%% STENCIL_GRID_SIZE %d\n", N);
    fprintf(f, "%lld %lld %lld\n", matrix_size, matrix_size, nnz);

    // Write values with progress indication
    long long total_points = matrix_size;
    long long progress_step = total_points / 100;
    if (progress_step == 0)
        progress_step = 1;

    printf("Writing 3D matrix entries: 0%%");
    fflush(stdout);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                long long global_idx = (long long)i * N * N + j * N + k;
                long long row_1based = global_idx + 1;

                // Progress indicator
                if (global_idx % progress_step == 0 || global_idx % 10000 == 0) {
                    int percent = (int)((global_idx * 100) / total_points);
                    printf("\rWriting 3D matrix entries: %d%%", percent);
                    fflush(stdout);
                }

                // Center (Laplacian with mass term: 6.0)
                fprintf(f, "%lld %lld 6.0\n", row_1based, row_1based);

                // -x neighbor (i-1, j, k)
                if (i > 0) {
                    long long neighbor_idx = (long long)(i - 1) * N * N + j * N + k + 1;
                    fprintf(f, "%lld %lld -1.0\n", row_1based, neighbor_idx);
                }

                // +x neighbor (i+1, j, k)
                if (i < N - 1) {
                    long long neighbor_idx = (long long)(i + 1) * N * N + j * N + k + 1;
                    fprintf(f, "%lld %lld -1.0\n", row_1based, neighbor_idx);
                }

                // -y neighbor (i, j-1, k)
                if (j > 0) {
                    long long neighbor_idx = (long long)i * N * N + (j - 1) * N + k + 1;
                    fprintf(f, "%lld %lld -1.0\n", row_1based, neighbor_idx);
                }

                // +y neighbor (i, j+1, k)
                if (j < N - 1) {
                    long long neighbor_idx = (long long)i * N * N + (j + 1) * N + k + 1;
                    fprintf(f, "%lld %lld -1.0\n", row_1based, neighbor_idx);
                }

                // -z neighbor (i, j, k-1)
                if (k > 0) {
                    long long neighbor_idx = (long long)i * N * N + j * N + (k - 1) + 1;
                    fprintf(f, "%lld %lld -1.0\n", row_1based, neighbor_idx);
                }

                // +z neighbor (i, j, k+1)
                if (k < N - 1) {
                    long long neighbor_idx = (long long)i * N * N + j * N + (k + 1) + 1;
                    fprintf(f, "%lld %lld -1.0\n", row_1based, neighbor_idx);
                }
            }
        }
    }

    printf("\rWriting 3D matrix entries: 100%%\n");
    fclose(f);
    printf("3D Matrix generated: %s (%lld×%lld, %lld nnz)\n", filename, matrix_size, matrix_size,
           nnz);
    return 0;
}

/**
 * Generate 3D 27-point stencil matrix in Matrix Market format
 * Grid: NxNxN, row-major ordering: global_idx = i*N*N + j*N + k
 * Stencil: center = 26.0, all 26 neighbors = -1.0
 * 26 neighbors = 6 face + 12 edge + 8 corner adjacent
 */
int write_matrix_market_stencil27(int N, const char* filename) {
    long long matrix_size = (long long)N * N * N;

    // Calculate exact number of non-zeros
    long long nnz = 0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                nnz++;  // Center
                // 6 face neighbors
                if (i > 0)
                    nnz++;
                if (i < N - 1)
                    nnz++;
                if (j > 0)
                    nnz++;
                if (j < N - 1)
                    nnz++;
                if (k > 0)
                    nnz++;
                if (k < N - 1)
                    nnz++;
                // 12 edge neighbors
                if (i > 0 && j > 0)
                    nnz++;
                if (i > 0 && j < N - 1)
                    nnz++;
                if (i < N - 1 && j > 0)
                    nnz++;
                if (i < N - 1 && j < N - 1)
                    nnz++;
                if (i > 0 && k > 0)
                    nnz++;
                if (i > 0 && k < N - 1)
                    nnz++;
                if (i < N - 1 && k > 0)
                    nnz++;
                if (i < N - 1 && k < N - 1)
                    nnz++;
                if (j > 0 && k > 0)
                    nnz++;
                if (j > 0 && k < N - 1)
                    nnz++;
                if (j < N - 1 && k > 0)
                    nnz++;
                if (j < N - 1 && k < N - 1)
                    nnz++;
                // 8 corner neighbors
                if (i > 0 && j > 0 && k > 0)
                    nnz++;
                if (i > 0 && j > 0 && k < N - 1)
                    nnz++;
                if (i > 0 && j < N - 1 && k > 0)
                    nnz++;
                if (i > 0 && j < N - 1 && k < N - 1)
                    nnz++;
                if (i < N - 1 && j > 0 && k > 0)
                    nnz++;
                if (i < N - 1 && j > 0 && k < N - 1)
                    nnz++;
                if (i < N - 1 && j < N - 1 && k > 0)
                    nnz++;
                if (i < N - 1 && j < N - 1 && k < N - 1)
                    nnz++;
            }
        }
    }

    FILE* f = fopen(filename, "w");
    if (!f) {
        perror("fopen");
        exit(1);
    }

    fprintf(f, "%%%%MatrixMarket matrix coordinate real general\n");
    fprintf(f, "%% STENCIL_GRID_SIZE %d\n", N);
    fprintf(f, "%lld %lld %lld\n", matrix_size, matrix_size, nnz);

    long long total_points = matrix_size;
    long long progress_step = total_points / 100;
    if (progress_step == 0)
        progress_step = 1;

    printf("Writing 3D 27-point matrix entries: 0%%");
    fflush(stdout);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                long long global_idx = (long long)i * N * N + j * N + k;
                long long row_1based = global_idx + 1;

                if (global_idx % progress_step == 0 || global_idx % 10000 == 0) {
                    int percent = (int)((global_idx * 100) / total_points);
                    printf("\rWriting 3D 27-point matrix entries: %d%%", percent);
                    fflush(stdout);
                }

                // Center (26.0)
                fprintf(f, "%lld %lld 26.0\n", row_1based, row_1based);

                // 6 face neighbors (-1.0 each)
                if (i > 0) {
                    long long ni = (long long)(i - 1) * N * N + j * N + k + 1;
                    fprintf(f, "%lld %lld -1.0\n", row_1based, ni);
                }
                if (i < N - 1) {
                    long long ni = (long long)(i + 1) * N * N + j * N + k + 1;
                    fprintf(f, "%lld %lld -1.0\n", row_1based, ni);
                }
                if (j > 0) {
                    long long ni = (long long)i * N * N + (j - 1) * N + k + 1;
                    fprintf(f, "%lld %lld -1.0\n", row_1based, ni);
                }
                if (j < N - 1) {
                    long long ni = (long long)i * N * N + (j + 1) * N + k + 1;
                    fprintf(f, "%lld %lld -1.0\n", row_1based, ni);
                }
                if (k > 0) {
                    long long ni = (long long)i * N * N + j * N + (k - 1) + 1;
                    fprintf(f, "%lld %lld -1.0\n", row_1based, ni);
                }
                if (k < N - 1) {
                    long long ni = (long long)i * N * N + j * N + (k + 1) + 1;
                    fprintf(f, "%lld %lld -1.0\n", row_1based, ni);
                }

                // 12 edge neighbors (-1.0 each)
                if (i > 0 && j > 0) {
                    long long ni = (long long)(i - 1) * N * N + (j - 1) * N + k + 1;
                    fprintf(f, "%lld %lld -1.0\n", row_1based, ni);
                }
                if (i > 0 && j < N - 1) {
                    long long ni = (long long)(i - 1) * N * N + (j + 1) * N + k + 1;
                    fprintf(f, "%lld %lld -1.0\n", row_1based, ni);
                }
                if (i < N - 1 && j > 0) {
                    long long ni = (long long)(i + 1) * N * N + (j - 1) * N + k + 1;
                    fprintf(f, "%lld %lld -1.0\n", row_1based, ni);
                }
                if (i < N - 1 && j < N - 1) {
                    long long ni = (long long)(i + 1) * N * N + (j + 1) * N + k + 1;
                    fprintf(f, "%lld %lld -1.0\n", row_1based, ni);
                }
                if (i > 0 && k > 0) {
                    long long ni = (long long)(i - 1) * N * N + j * N + (k - 1) + 1;
                    fprintf(f, "%lld %lld -1.0\n", row_1based, ni);
                }
                if (i > 0 && k < N - 1) {
                    long long ni = (long long)(i - 1) * N * N + j * N + (k + 1) + 1;
                    fprintf(f, "%lld %lld -1.0\n", row_1based, ni);
                }
                if (i < N - 1 && k > 0) {
                    long long ni = (long long)(i + 1) * N * N + j * N + (k - 1) + 1;
                    fprintf(f, "%lld %lld -1.0\n", row_1based, ni);
                }
                if (i < N - 1 && k < N - 1) {
                    long long ni = (long long)(i + 1) * N * N + j * N + (k + 1) + 1;
                    fprintf(f, "%lld %lld -1.0\n", row_1based, ni);
                }
                if (j > 0 && k > 0) {
                    long long ni = (long long)i * N * N + (j - 1) * N + (k - 1) + 1;
                    fprintf(f, "%lld %lld -1.0\n", row_1based, ni);
                }
                if (j > 0 && k < N - 1) {
                    long long ni = (long long)i * N * N + (j - 1) * N + (k + 1) + 1;
                    fprintf(f, "%lld %lld -1.0\n", row_1based, ni);
                }
                if (j < N - 1 && k > 0) {
                    long long ni = (long long)i * N * N + (j + 1) * N + (k - 1) + 1;
                    fprintf(f, "%lld %lld -1.0\n", row_1based, ni);
                }
                if (j < N - 1 && k < N - 1) {
                    long long ni = (long long)i * N * N + (j + 1) * N + (k + 1) + 1;
                    fprintf(f, "%lld %lld -1.0\n", row_1based, ni);
                }

                // 8 corner neighbors (-1.0 each)
                if (i > 0 && j > 0 && k > 0) {
                    long long ni = (long long)(i - 1) * N * N + (j - 1) * N + (k - 1) + 1;
                    fprintf(f, "%lld %lld -1.0\n", row_1based, ni);
                }
                if (i > 0 && j > 0 && k < N - 1) {
                    long long ni = (long long)(i - 1) * N * N + (j - 1) * N + (k + 1) + 1;
                    fprintf(f, "%lld %lld -1.0\n", row_1based, ni);
                }
                if (i > 0 && j < N - 1 && k > 0) {
                    long long ni = (long long)(i - 1) * N * N + (j + 1) * N + (k - 1) + 1;
                    fprintf(f, "%lld %lld -1.0\n", row_1based, ni);
                }
                if (i > 0 && j < N - 1 && k < N - 1) {
                    long long ni = (long long)(i - 1) * N * N + (j + 1) * N + (k + 1) + 1;
                    fprintf(f, "%lld %lld -1.0\n", row_1based, ni);
                }
                if (i < N - 1 && j > 0 && k > 0) {
                    long long ni = (long long)(i + 1) * N * N + (j - 1) * N + (k - 1) + 1;
                    fprintf(f, "%lld %lld -1.0\n", row_1based, ni);
                }
                if (i < N - 1 && j > 0 && k < N - 1) {
                    long long ni = (long long)(i + 1) * N * N + (j - 1) * N + (k + 1) + 1;
                    fprintf(f, "%lld %lld -1.0\n", row_1based, ni);
                }
                if (i < N - 1 && j < N - 1 && k > 0) {
                    long long ni = (long long)(i + 1) * N * N + (j + 1) * N + (k - 1) + 1;
                    fprintf(f, "%lld %lld -1.0\n", row_1based, ni);
                }
                if (i < N - 1 && j < N - 1 && k < N - 1) {
                    long long ni = (long long)(i + 1) * N * N + (j + 1) * N + (k + 1) + 1;
                    fprintf(f, "%lld %lld -1.0\n", row_1based, ni);
                }
            }
        }
    }

    printf("\rWriting 3D 27-point matrix entries: 100%%\n");
    fclose(f);
    printf("3D 27-point Matrix generated: %s (%lld×%lld, %lld nnz)\n", filename, matrix_size,
           matrix_size, nnz);
    return 0;
}
