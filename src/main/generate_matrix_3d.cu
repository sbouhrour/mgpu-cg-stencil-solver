/**
 * @file generate_matrix_3d.cu
 * @brief Generates 3D 7-point stencil matrices in Matrix Market format.
 *
 * Creates an N×N×N 3D Laplacian on an N³×N³ matrix.
 *
 * Usage:
 *  ./bin/generate_matrix_3d 64 matrix/stencil3d_64.mtx
 *
 * Author: Bouhrour Stephane
 */

#include <stdio.h>
#include <stdlib.h>
#include "io.h"

int main(int argc, char* argv[]) {
    if (argc < 3) {
        printf("Usage: %s <grid_dim> <output_filename>\n", argv[0]);
        printf("Example: %s 64 matrix/stencil3d_64.mtx\n", argv[0]);
        return 1;
    }

    int grid = atoi(argv[1]);        // 3D grid size
    const char* filename = argv[2];  // output file
    return write_matrix_market_stencil7(grid, filename);
}
