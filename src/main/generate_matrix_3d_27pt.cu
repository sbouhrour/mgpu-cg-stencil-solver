/**
 * @file generate_matrix_3d_27pt.cu
 * @brief Generates 3D 27-point stencil matrices in Matrix Market format.
 *
 * Creates an N×N×N 3D Laplacian on an N³×N³ matrix with 27-point stencil.
 *
 * Usage:
 *  ./bin/generate_matrix_3d_27pt 64 matrix/stencil3d_27pt_64.mtx
 *
 * Author: Bouhrour Stephane
 */

#include <stdio.h>
#include <stdlib.h>
#include "io.h"

int main(int argc, char* argv[]) {
    if (argc < 3) {
        printf("Usage: %s <grid_dim> <output_filename>\n", argv[0]);
        printf("Example: %s 64 matrix/stencil3d_27pt_64.mtx\n", argv[0]);
        return 1;
    }

    int grid = atoi(argv[1]);
    const char* filename = argv[2];
    return write_matrix_market_stencil27(grid, filename);
}
